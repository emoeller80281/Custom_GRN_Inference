import csv
import json
import logging
import os
import sys
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from config.settings_hpc import *
from multiomic_transformer.datasets.dataset import (
    MultiomicTransformerDataset, MultiChromosomeDataset,
    ChromBucketBatchSampler, DistributedBatchSampler
    )
from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.utils.files import unique_path
from multiomic_transformer.utils import ewc_utils, plotting

warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group`")

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    torch.backends.cuda.enable_flash_sdp(True)

    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)
    
def setup_logging(rank: int):
    # Remove existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    level = logging.INFO if rank == 0 else logging.ERROR
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.flush = sys.stdout.flush  # ensure flush

    logging.basicConfig(
        level=level,
        format=f"%(message)s",
        handlers=[handler],
        force=True
    )
    
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        patience: int = 20,
        min_delta: float = 1e-3,

    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        
        self.scaler = GradScaler()

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
        )
        
        # Early stopping
        self.best_val_loss = float("inf")
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0

    def _run_batch(self, batch):
        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
        atac_wins, tf_tensor, targets, bias = (
            atac_wins.to(self.gpu_id),
            tf_tensor.to(self.gpu_id),
            targets.to(self.gpu_id),
            bias.to(self.gpu_id),
        )
        tf_ids, tg_ids, motif_mask = (
            tf_ids.to(self.gpu_id),
            tg_ids.to(self.gpu_id),
            motif_mask.to(self.gpu_id),
        )
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            mask_arg = motif_mask if USE_MOTIF_MASK else None
            preds, _ = self.model(
                atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias, motif_mask=mask_arg
            )

            # Base MSE loss
            mse_loss = self.loss_fn(preds, targets)

            # Correlation term
            preds_flat, targets_flat = preds.reshape(-1), targets.reshape(-1)
            if torch.std(targets_flat) > 1e-8 and torch.std(preds_flat) > 1e-8:
                vx, vy = preds_flat - preds_flat.mean(), targets_flat - targets_flat.mean()
                corr_loss = -torch.sum(vx * vy) / (
                    torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8
                )
            else:
                corr_loss = torch.tensor(0.0, device=preds.device)

            total_loss = mse_loss + CORR_LOSS_WEIGHT * corr_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return total_loss.detach(), mse_loss.detach()

    def _validate(self):
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        preds_list, tgts_list = [], []
        with torch.no_grad():
            for atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask in self.val_data:
                atac_wins = atac_wins.to(self.gpu_id)
                tf_tensor = tf_tensor.to(self.gpu_id)
                targets   = targets.to(self.gpu_id)
                bias      = bias.to(self.gpu_id)
                tf_ids    = tf_ids.to(self.gpu_id)
                tg_ids    = tg_ids.to(self.gpu_id)
                motif_mask= motif_mask.to(self.gpu_id)

                mask_arg = motif_mask if USE_MOTIF_MASK else None
                preds, _ = self.model(
                    atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias, motif_mask=mask_arg
                )
                
                loss  = F.mse_loss(preds, targets)
                total_loss += loss.item(); n_batches += 1
                preds_list.append(preds); tgts_list.append(targets)

        # flatten and pad preds with different size (from different chromosomes)
        device = preds_list[0].device if preds_list else torch.device(f"cuda:{self.gpu_id}")

        # 0) flatten per-batch and concatenate locally into 1D vectors
        preds_1d = torch.cat([p.reshape(-1) for p in preds_list], dim=0) if preds_list else torch.tensor([], device=device)
        tgts_1d  = torch.cat([t.reshape(-1) for t in tgts_list],  dim=0) if tgts_list  else torch.tensor([], device=device)

        # 1) gather local lengths
        world_size = dist.get_world_size()
        n_local = torch.tensor([preds_1d.numel()], device=device, dtype=torch.int64)
        n_list = [torch.zeros(1, device=device, dtype=torch.int64) for _ in range(world_size)]
        dist.all_gather(n_list, n_local)
        sizes = [int(t.item()) for t in n_list]
        max_n = max(sizes) if sizes else 0

        # 2) pad 1D vectors to max_n
        def pad_1d(x, target_len):
            if x.numel() == target_len:
                return x
            pad = torch.zeros(target_len - x.numel(), device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        preds_pad_1d = pad_1d(preds_1d, max_n)
        tgts_pad_1d  = pad_1d(tgts_1d,  max_n)

        # 3) all_gather padded 1D tensors
        gathered_preds_1d = [torch.empty_like(preds_pad_1d) for _ in range(world_size)]
        gathered_tgts_1d  = [torch.empty_like(tgts_pad_1d)  for _ in range(world_size)]
        dist.all_gather(gathered_preds_1d, preds_pad_1d)
        dist.all_gather(gathered_tgts_1d,  tgts_pad_1d)

        # 4) trim back to true sizes and stack globally
        preds_all = torch.cat([gp[:sizes[r]] for r, gp in enumerate(gathered_preds_1d)], dim=0).cpu().numpy()
        tgts_all  = torch.cat([gt[:sizes[r]] for r, gt in enumerate(gathered_tgts_1d)],  dim=0).cpu().numpy()

        # Replace NaN/inf (safety)
        preds_all = np.nan_to_num(preds_all, nan=0.0, posinf=1e6, neginf=-1e6)
        tgts_all  = np.nan_to_num(tgts_all,  nan=0.0, posinf=1e6, neginf=-1e6)

        # Metrics
        if np.std(tgts_all) < 1e-8 or np.std(preds_all) < 1e-8:
            pearson_corr, spearman_corr = 0.0, 0.0
        else:
            try:
                pearson_corr, _ = pearsonr(preds_all.ravel(), tgts_all.ravel())
                spearman_corr, _ = spearmanr(preds_all.ravel(), tgts_all.ravel())
            except Exception as e:
                logging.warning(f"Correlation failed: {e}")
                pearson_corr, spearman_corr = 0.0, 0.0

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss, float(pearson_corr), float(spearman_corr)
    
    def _run_epoch(self, epoch):
        sampler = getattr(self.train_data, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        total_loss, total_mse, n_batches = 0.0, 0.0, 0

        for batch in self.train_data:
            total_loss_val, mse_loss_val = self._run_batch(batch)
            total_loss += float(total_loss_val)
            total_mse += float(mse_loss_val)
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)
        avg_train_mse = total_mse / max(1, n_batches)

        avg_val_loss, pearson_corr, spearman_corr = self._validate()

        self.scheduler.step(avg_val_loss)

        return avg_train_loss, avg_train_mse, avg_val_loss, pearson_corr, spearman_corr


    def _save_checkpoint(self, epoch, path):
        ckp = self.model.module.state_dict()
        torch.save(ckp, os.path.join(path, "checkpoint.pt"))
        if self.gpu_id == 0:
            logging.info(f"\tTraining checkpoint saved")
        
    def train(self, max_epochs: int, path: str):
        best_val_loss = float("inf")
        best_pearson = float(0)
        patience_counter = 0
        history = []  # store per-epoch logs

        for epoch in range(max_epochs):
            train_loss, train_mse, val_loss, pearson_corr, spearman_corr = self._run_epoch(epoch)

            if self.gpu_id == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logging.info(
                    f"Epoch {epoch+1} | Train Total Loss: {train_loss:.4f} | "
                    f"Train MSE: {train_mse:.4f} | "
                    f"Val MSE: {val_loss:.4f} | "
                    f"Pearson: {pearson_corr:.3f} | Spearman: {spearman_corr:.3f} | "
                    f"LR: {lr:.2e}"
                )

                history.append({
                    "Epoch": epoch+1,
                    "Train Total Loss": train_loss,
                    "Train MSE": train_mse,
                    "Val MSE": val_loss,
                    "Pearson": pearson_corr,
                    "Spearman": spearman_corr,
                    "LR": lr,
                })
                                
            # Checkpoint + CSV log
            if epoch % self.save_every == 0:
                if self.gpu_id == 0:
                    self._save_checkpoint(epoch, path)
                    self._write_log_csv(history, path)
                dist.barrier()

            # Checkpoint + CSV log
            stop_tensor = torch.tensor(0, device=self.gpu_id)

            # --- Early stopping check (only rank 0 sets flag) ---
            if self.gpu_id == 0:
                if (val_loss < best_val_loss - self.min_delta) or (pearson_corr > best_pearson + self.min_delta):
                    # If either val_loss improved OR pearson improved, reset patience
                    best_val_loss = val_loss
                    best_pearson = max(best_pearson, pearson_corr)
                    patience_counter = 0
                else:
                    # No improvement
                    patience_counter += 1

                    if patience_counter >= self.patience:
                        logging.info("Early stopping triggered.")
                        self._save_checkpoint(epoch, path)
                        self._write_log_csv(history, path)
                        stop_tensor.fill_(1)  # <-- mark stop

                    else:
                        logging.info(f"    Loss did not improve {patience_counter}/{self.patience}")

            # --- Broadcast stop flag from rank 0 to all ranks ---
            dist.broadcast(stop_tensor, src=0)

            # --- All ranks see the same value now ---
            if stop_tensor.item() == 1:
                if self.gpu_id == 0:
                    logging.info("All ranks stopping training.")
                break


        # Final save if not early stopped
        if self.gpu_id == 0 and patience_counter < self.patience:
            self._write_log_csv(history, path)
            logging.info("Training loop exited normally.")
    
    def _write_log_csv(self, history, path):
        fieldnames = ["Epoch", "Train Total Loss", "Train MSE", "Val MSE", "Pearson", "Spearman", "LR"]
        log_path = os.path.join(path, "training_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
            

def load_train_objs():
    # Load the dataset
    dataset = MultiChromosomeDataset(
        data_dir=SAMPLE_DATA_CACHE_DIR,
        chrom_ids=CHROM_IDS,                  # list of chromosomes
        tf_vocab_path=os.path.join(COMMON_DATA, "tf_vocab.json"),
        tg_vocab_path=os.path.join(COMMON_DATA, "tg_vocab.json"),
        max_cached=2,                         # small LRU of per-chrom datasets
        # optional global subsampling knobs – applied consistently to every chrom:
        max_tfs=SUBSAMPLE_MAX_TFS,
        max_tgs=SUBSAMPLE_MAX_TGS,
        max_windows_per_chrom=SUBSAMPLE_MAX_WINDOWS_PER_CHROM,
        subset_seed=SUBSAMPLE_SEED,
    )

    # model vocab sizes must match the dataset’s global sub-vocab
    tf_vocab_size = len(dataset.tf_name2id_sub)
    tg_vocab_size = len(dataset.tg_name2id_sub)

    # Initiallize the model
    model = MultiomicTransformer(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        tf_vocab_size=tf_vocab_size,
        tg_vocab_size=tg_vocab_size,
        bias_scale=ATTN_BIAS_SCALE,
        use_bias=USE_DISTANCE_BIAS,
        use_shortcut=USE_SHORTCUT,
        use_motif_mask=USE_MOTIF_MASK,
        lambda_l1=SHORTCUT_L1,
        lambda_l2=SHORTCUT_L2,
        topk=SHORTCUT_TOPK,
        shortcut_dropout=SHORTCUT_DROPOUT
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
    return dataset, model, optimizer

def prepare_dataloader(dataset, batch_size, world_size=1, rank=0,
                       num_workers=4, pin_memory=True, seed=42, drop_last=True):
    """
    Build train/val/test loaders.
    - Single-chrom: same as before (random_split + DistributedSampler).
    - Multi-chrom : split by chromosomes; each split uses ChromBucketBatchSampler
                    (optionally sharded across ranks via DistributedBatchSampler).
    """
    import random
    g = torch.Generator(); g.manual_seed(seed)

    # ---------- Multi-chromosome path ----------
    if isinstance(dataset, MultiChromosomeDataset):
        # deterministic split of chromosome IDs
        def _split_chroms(xs, seed=42):
            xs = xs[:]
            rnd = random.Random(seed); rnd.shuffle(xs)
            n = len(xs)
            if n == 0: return [], [], []
            if n == 1: return xs, [], []
            if n == 2: return [xs[0]], [xs[1]], []
            # n >= 3: ~70/15/15 with largest-remainder rounding; ensure val/test >= 1
            ratios = (0.70, 0.15, 0.15)
            targets = [r * n for r in ratios]; base = [int(t) for t in targets]
            rem = n - sum(base)
            remainders = [t - b for t, b in zip(targets, base)]
            for i in sorted(range(3), key=lambda i: remainders[i], reverse=True)[:rem]:
                base[i] += 1
            n_train, n_val, n_test = base
            if n_val == 0 and n_train > 1:  n_val,  n_train = 1, n_train - 1
            if n_test == 0 and n_train > 1: n_test, n_train = 1, n_train - 1
            train = xs[:n_train]; val = xs[n_train:n_train+n_val]; test = xs[n_train+n_val:]
            return train, val, test

        train_chrs, val_chrs, test_chrs = _split_chroms(dataset.chrom_ids, seed=seed)

        # re-instantiate per-split datasets with the SAME sub-vocab config
        ds_args = dict(
            data_dir=dataset.data_dir,
            tf_vocab_path=dataset.tf_vocab_path,
            tg_vocab_path=dataset.tg_vocab_path,
            fine_tuner=dataset.fine_tuner,
            sample_name=dataset.sample_name,
            max_cached=dataset.max_cached,
            max_tfs=dataset.max_tfs,
            max_tgs=dataset.max_tgs,
            max_windows_per_chrom=dataset.max_windows_per_chrom,
            subset_seed=dataset.subset_seed,
        )
        ds_train = MultiChromosomeDataset(chrom_ids=train_chrs, **ds_args)
        ds_val   = MultiChromosomeDataset(chrom_ids=val_chrs,   **ds_args)
        ds_test  = MultiChromosomeDataset(chrom_ids=test_chrs,  **ds_args)

        # bucket batches by chromosome (keeps shapes consistent within a batch)
        base_train_bs = ChromBucketBatchSampler(ds_train, batch_size=batch_size, shuffle=True,  seed=seed)
        base_val_bs   = ChromBucketBatchSampler(ds_val,   batch_size=batch_size, shuffle=False, seed=seed)
        base_test_bs  = ChromBucketBatchSampler(ds_test,  batch_size=batch_size, shuffle=False, seed=seed)

        # shard whole batches across ranks in DDP
        if world_size > 1:
            train_bs = DistributedBatchSampler(base_train_bs, world_size, rank, drop_last=True)
            val_bs   = DistributedBatchSampler(base_val_bs,   world_size, rank, drop_last=False)
            test_bs  = DistributedBatchSampler(base_test_bs,  world_size, rank, drop_last=False)
        else:
            train_bs, val_bs, test_bs = base_train_bs, base_val_bs, base_test_bs

        train_loader = DataLoader(ds_train, batch_sampler=train_bs,
                                  collate_fn=MultiChromosomeDataset.collate_fn,
                                  num_workers=num_workers, pin_memory=pin_memory)
        val_loader   = DataLoader(ds_val,   batch_sampler=val_bs,
                                  collate_fn=MultiChromosomeDataset.collate_fn,
                                  num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = DataLoader(ds_test,  batch_sampler=test_bs,
                                  collate_fn=MultiChromosomeDataset.collate_fn,
                                  num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, val_loader, test_loader

    # ---------- Single-chromosome path (unchanged) ----------
    n_total = len(dataset)
    n_train = int(n_total * 0.70)
    n_val   = int(n_total * 0.15)
    n_test  = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=g)

    train_sampler = val_sampler = test_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, drop_last=drop_last)
        val_sampler   = DistributedSampler(val_set,   num_replicas=world_size, rank=rank, drop_last=False)
        test_sampler  = DistributedSampler(test_set,  num_replicas=world_size, rank=rank, drop_last=False)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              collate_fn=MultiomicTransformerDataset.collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, sampler=val_sampler,
                              collate_fn=MultiomicTransformerDataset.collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, sampler=test_sampler,
                              collate_fn=MultiomicTransformerDataset.collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return train_loader, val_loader, test_loader



def write_run_parameters(dataset, out_dir):

    logging.info("\n===== MultiomicTransformerDataset Loaded =====")
    logging.info(f"Chromosome:          {CHROM_IDS}")
    logging.info(f"Genes:               {len(dataset.tg_ids)}")
    logging.info(f"Windows (RE):        {dataset.num_windows}")
    logging.info(f"TFs:                 {len(dataset.tf_ids)}")
    logging.info(f"Metacells:           {len(dataset)}")
    logging.info(f"Epochs:              {TOTAL_EPOCHS}")
    logging.info(f"Batch Size:          {BATCH_SIZE}")
    logging.info(f"Model Dimension:     {D_MODEL}")
    logging.info(f"Attention Heads:     {NUM_HEADS}")
    logging.info(f"Attention Layers:    {NUM_LAYERS}")
    logging.info(f"Feedforward Layers:  {D_FF}")
    logging.info(f"Dropout:             {DROPOUT}")
    logging.info(f"TF-TG Shortcut?:     {USE_SHORTCUT}")
    logging.info(f"Dist bias?:          {USE_DISTANCE_BIAS}")
    logging.info(f"Motif Mask?:         {USE_MOTIF_MASK}")
    logging.info(f"Shortcut L1:         {SHORTCUT_L1}")
    logging.info(f"Shortcut L2:         {SHORTCUT_L2}")
    logging.info(f"Shortcut Dropout:    {SHORTCUT_DROPOUT}")
    logging.info(f"Shortcut Top K:      {SHORTCUT_TOPK}")
    logging.info("================================================")
    
    run_params = {
        "Genes": len(dataset.tg_ids),
        "Windows": dataset.num_windows,
        "TFs": len(dataset.tf_ids),
        "Metacells": len(dataset),  # store count, not huge list
        "Epochs": TOTAL_EPOCHS,
        "Batch Size": BATCH_SIZE,
        "d_model": D_MODEL,
        "corr_loss_weight": CORR_LOSS_WEIGHT,
        "Attention Heads": NUM_HEADS,
        "Model Layers": NUM_LAYERS,
        "d_feedforward": D_FF,
        "Dropout": DROPOUT,
        "tf_tg_shortcut": USE_SHORTCUT,
        "Distance Bias":USE_DISTANCE_BIAS,
        "Distance Bias Scale": ATTN_BIAS_SCALE,
        "Motif Mask": USE_MOTIF_MASK,
        "Shortcut L1": SHORTCUT_L1,
        "Shortcut L2": SHORTCUT_L2,
        "Shortcut Dropout": SHORTCUT_DROPOUT,
        "Shortcut Top K": SHORTCUT_TOPK
    }

    path = os.path.join(out_dir, "run_parameters.json")
    with open(path, "w") as f:
        json.dump(run_params, f, indent=4)  # indent=4 for readability
    logging.info(f"Run parameters written to {path}")

@torch.no_grad()
def get_mean_tf_tg_attention(model, dataloader, device="cuda"):
    model.eval()
    attn_accum = None
    n = 0
    for atac_wins, tf_tensor, _, bias, tf_ids, tg_ids, motif_mask in dataloader:
        atac_wins, tf_tensor, bias = atac_wins.to(device), tf_tensor.to(device), bias.to(device)
        tf_ids, tg_ids = tf_ids.to(device), tg_ids.to(device)
        motif_mask = motif_mask.to(device)

        # Forward pass
        _ = model(atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids,
                  bias=bias, motif_mask=motif_mask)

        # grab attn from shortcut layer each batch
        if hasattr(model.shortcut_layer, "attn"):
            attn_batch = model.shortcut_layer.attn.detach().cpu()
            attn_accum = attn_batch if attn_accum is None else attn_accum + attn_batch
            n += 1
    return attn_accum / max(n, 1)

def _mapping_to_ordered_list(name2id: dict):
    # convert {name: id} → [names] in id order
    return [k for k, _ in sorted(name2id.items(), key=lambda kv: kv[1])]

def write_experiment_settings_and_objects(training_output_dir: Path, dataset):
    """
    Works for both MultiChromosomeDataset and single-chrom MultiomicTransformerDataset.
    Writes tf/tg vocab mappings and run parameters. Skips scaler unless present.
    """
    os.makedirs(training_output_dir, exist_ok=True)

    # Pick the right mappings depending on dataset type
    tf_map = getattr(dataset, "tf_name2id_sub", None)
    tg_map = getattr(dataset, "tg_name2id_sub", None)

    if tf_map is None or tg_map is None:
        raise RuntimeError("Dataset is missing TF/TG name→id mappings.")

    # Persist mappings (name→id)
    with open(os.path.join(training_output_dir, "tf_vocab.json"), "w") as f:
        json.dump(tf_map, f)
    with open(os.path.join(training_output_dir, "tg_vocab.json"), "w") as f:
        json.dump(tg_map, f)

    # Also persist ordered name lists (useful for plotting/inspection)
    tf_names_ordered = _mapping_to_ordered_list(tf_map)
    tg_names_ordered = _mapping_to_ordered_list(tg_map)
    with open(os.path.join(training_output_dir, "tf_names_ordered.json"), "w") as f:
        json.dump(tf_names_ordered, f)
    with open(os.path.join(training_output_dir, "tg_names_ordered.json"), "w") as f:
        json.dump(tg_names_ordered, f)

    # Your existing run-parameter writer is fine to call here if it doesn’t assume single-chrom only
    write_run_parameters(dataset, training_output_dir)
    logging.info("Wrote experiment settings and objects to training output directory")
    
def save_tf_tg_embeddings_from_model(model, out_dir, vocab_dir):
    model = getattr(model, "module", model)  # unwrap DDP if needed
    torch.save(
        {
            "tf_emb":     model.tf_emb_table.weight.detach().cpu(),   # [T, D]
            "tg_emb":     model.tg_emb_table.weight.detach().cpu(),   # [G, D]
            "tg_dec_emb": model.tg_decoder_table.weight.detach().cpu()
        },
        os.path.join(out_dir, "tf_tg_embeddings.pt")
    )

    # Prefer the copies we already wrote into the training_output_dir
    with open(os.path.join(vocab_dir, "tf_vocab.json")) as f:
        tf_vocab_obj = json.load(f)
    with open(os.path.join(vocab_dir, "tg_vocab.json")) as f:
        tg_vocab_obj = json.load(f)

    tf_name2id = tf_vocab_obj.get("name_to_id", tf_vocab_obj)
    tg_name2id = tg_vocab_obj.get("name_to_id", tg_vocab_obj)

    tf_id2name = [None] * len(tf_name2id)
    for name, idx in tf_name2id.items():
        tf_id2name[idx] = name
    tg_id2name = [None] * len(tg_name2id)
    for name, idx in tg_name2id.items():
        tg_id2name[idx] = name

    torch.save(
        {"tf_id2name": tf_id2name, "tg_id2name": tg_id2name},
        os.path.join(out_dir, "tf_tg_vocab_id2name.pt")
    )
    
def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    
    # Early check to make sure the model dimension can be split evenly among the number of heads
    assert D_MODEL % NUM_HEADS == 0, f"{D_MODEL} not divisible by {NUM_HEADS}"
    
    ddp_setup(rank, world_size)
    setup_logging(rank)
    
    try:
        training_file_iter_format = "model_training_{:03d}"
        training_output_dir = unique_path(OUTPUT_DIR / CHROM_ID, training_file_iter_format)
        
        logging.info(f"\n =========== EXPERIMENT {training_output_dir.name.upper()} ===========")
                
        os.makedirs(training_output_dir, exist_ok=True)
            
        dataset, model, optimizer = load_train_objs()

        if rank == 0:
            write_experiment_settings_and_objects(training_output_dir, dataset)
            logging.info("Wrote experiment settings and objects to training output directory")
            logging.info("Preparing dataloader")

        train_loader, val_loader, test_loader = prepare_dataloader(dataset, batch_size, world_size, rank)
        
        if rank == 0:
            logging.info("Creating Trainer")
        
        loss_fn = nn.MSELoss()    
        
        trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, gpu_id=rank, save_every=save_every, patience=PATIENCE)
        
        if rank == 0:
            logging.info("\n ----- TRAINING STARTED -----")
        trainer.train(max_epochs=TOTAL_EPOCHS, path=training_output_dir)
        
        if rank == 0:
            model_for_save = getattr(trainer.model, "module", trainer.model)
            model_for_save.eval()

            # save embeddings directly (no reload)
            save_tf_tg_embeddings_from_model(
                model_for_save,
                out_dir=training_output_dir,
                vocab_dir=training_output_dir,
            )
            # Save model checkpoint
            torch.save(model_for_save.state_dict(),
                    os.path.join(training_output_dir, "trained_model.pt"))
            logging.info("Saved final trained model")
            
            torch.save(model_for_save.state_dict(),
                    os.path.join(training_output_dir, "trained_model.pt"))
        
            ewc_bundle_path = training_output_dir / "ewc_bundle.pth"
            fisher = ewc_utils.compute_fisher_diag(model_for_save, train_loader,
                                    device=(f"cuda:{rank}" if torch.cuda.is_available() else "cpu"),
                                    n_batches=100)
            ewc_utils.save_ewc_bundle(ewc_bundle_path, model_for_save, fisher)
            
            # Save TF→TG attention weights from the shortcut
            if hasattr(model_for_save, "shortcut_layer") and hasattr(model_for_save.shortcut_layer, "attn"):
                mean_attn = get_mean_tf_tg_attention(model_for_save, test_loader, device=f"cuda:{rank}")
                
            # Build ordered name lists from vocab files (IDs -> names)
            with open(os.path.join(training_output_dir, "tf_vocab.json")) as f:
                tf_vocab_obj = json.load(f)
            with open(os.path.join(training_output_dir, "tg_vocab.json")) as f:
                tg_vocab_obj = json.load(f)

            tf_name2id = tf_vocab_obj.get("name_to_id", tf_vocab_obj)
            tg_name2id = tg_vocab_obj.get("name_to_id", tg_vocab_obj)

            tf_id2name = [None] * len(tf_name2id)
            for name, idx in tf_name2id.items():
                tf_id2name[idx] = name

            tg_id2name = [None] * len(tg_name2id)
            for name, idx in tg_name2id.items():
                tg_id2name[idx] = name

            # When saving attention (rows=tg, cols=tf as in your code)
            mean_attn_df = pd.DataFrame(
                mean_attn.numpy(),
                index=tg_id2name,
                columns=tf_id2name
            )
            
            mean_attn_df.to_csv(os.path.join(training_output_dir, "tf_tg_mean_attention.csv"))
            logging.info(f"Saved TF→TG attention weights to 'tf_tg_mean_attention.csv'")
            # Training figures
            log_path = os.path.join(training_output_dir, "training_log.csv")
            log_df = pd.read_csv(log_path, header=0)
            
            pearson_corr_plt = plotting.plot_pearson_corr_across_epochs(
                df=log_df,
                dataset_name=DATASET_NAME,
                chrom_id=CHROM_ID
            )
            pearson_corr_plt.savefig(os.path.join(training_output_dir, "pearson_training.png"), dpi=300)
            
            train_val_loss_plt = plotting.plot_train_val_loss(
                df=log_df,
                dataset_name=DATASET_NAME,
                chrom_id=CHROM_ID
            )
            train_val_loss_plt.savefig(os.path.join(training_output_dir, "train_val_loss.png"), dpi=300)
            
            per_gene_corr_scatter_plt = plotting.plot_per_gene_correlation_scatterplot(
                model=model,
                dataloader=test_loader,
                use_mask=USE_MOTIF_MASK,
                gpu_id=0
            )
            per_gene_corr_scatter_plt.savefig(os.path.join(training_output_dir, "per_gene_corr_scatter.png"), dpi=300)
                
        if rank == 0:
            logging.info("\nIterations complete")
    
    finally:
        if dist.is_initialized():
            dist.barrier()
            if rank == 0:
                logging.info("\nDestroying process group")
            dist.destroy_process_group()
    
if __name__ == "__main__":
    
    main(rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        save_every=5,
        total_epochs=TOTAL_EPOCHS,
        batch_size=BATCH_SIZE)