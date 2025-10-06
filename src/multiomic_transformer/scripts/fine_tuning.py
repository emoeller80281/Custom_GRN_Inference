import csv
import json
import logging
import os
import sys
import glob
import warnings
import random

import joblib
import itertools
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
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from config.settings import *
from multiomic_transformer.data.linger_pseudobulk import pseudo_bulk
from multiomic_transformer.datasets.dataset import MultiomicTransformerDataset
from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.utils.files import unique_path
from multiomic_transformer.utils import ewc_utils
from multiomic_transformer.utils.sampler import DistributedWeightedSampler

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
    def __init__(self, model, train_data, val_data, loss_fn, optimizer,
                 gpu_id, save_every, patience=20, min_delta=1e-3,
                 ref_params=None, fisher_diag=None, lambda_ewc=0.0):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.scaler = GradScaler()
        
        self.log_train_breakdown_every = 5  # epochs

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        self.best_val_loss = float("inf")
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0

        # --- EWC state ---
        self.ref_params = ref_params
        self.fisher_diag = fisher_diag
        self.lambda_ewc = lambda_ewc


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
            # Run the forward pass to get the model output
            
            mask_arg = motif_mask if USE_MOTIF_MASK else None
            preds, _ = self.model(
                atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias, motif_mask=mask_arg
            )

            # Calculate loss
            loss = self.loss_fn(preds, targets)

            # Correlation bonus
            preds_flat, targets_flat = preds.reshape(-1), targets.reshape(-1)
            if torch.std(targets_flat) > 1e-8 and torch.std(preds_flat) > 1e-8:
                vx, vy = preds_flat - preds_flat.mean(), targets_flat - targets_flat.mean()
                corr_loss = -torch.sum(vx * vy) / (
                    torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-8
                )
            else:
                corr_loss = torch.tensor(0.0, device=preds.device)
            loss = loss + CORR_LOSS_WEIGHT * corr_loss
            
            # Add EWC penalty if available
            if self.ref_params is not None and self.fisher_diag is not None:
                loss_ewc = ewc_utils.ewc_penalty(
                    self.model.module,  # unwrap DDP
                    self.fisher_diag,
                    self.ref_params,
                    lambda_ewc=self.lambda_ewc
                )
                loss = loss + loss_ewc
            
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss

    def _validate(self):
        self.model.eval()
        per_dataset_metrics = {}  # {name: (val_loss, pearson, spearman)}

        with torch.no_grad():
            for name, loader in self.val_data.items():
                preds_list, tgts_list = [], []
                total_loss, n_batches = 0.0, 0
                for batch in loader:
                    atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = [
                        x.to(self.gpu_id) for x in batch
                    ]
                    preds, _ = self.model(atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias)
                    loss = F.mse_loss(preds, targets)
                    total_loss += loss.item(); n_batches += 1
                    preds_list.append(preds.cpu()); tgts_list.append(targets.cpu())

                preds = torch.cat(preds_list).numpy()
                tgts = torch.cat(tgts_list).numpy()

                val_loss = total_loss / max(1, n_batches)
                if np.std(tgts) < 1e-8 or np.std(preds) < 1e-8:
                    pearson_corr, spearman_corr = 0.0, 0.0
                else:
                    pearson_corr, _ = pearsonr(preds.ravel(), tgts.ravel())
                    spearman_corr, _ = spearmanr(preds.ravel(), tgts.ravel())

                per_dataset_metrics[name] = (val_loss, pearson_corr, spearman_corr)

        # compute global averages if needed
        avg_vals = np.mean([m[0] for m in per_dataset_metrics.values()])
        avg_pearson = np.mean([m[1] for m in per_dataset_metrics.values()])
        avg_spearman = np.mean([m[2] for m in per_dataset_metrics.values()])

        return avg_vals, avg_pearson, avg_spearman, per_dataset_metrics


    def _evaluate_loader(self, loader):
        """Run validation on a single DataLoader, return (loss, pearson, spearman)."""
        total_loss, n_batches = 0.0, 0
        preds_list, tgts_list = [], []

        with torch.no_grad():
            for atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask in loader:
                atac_wins  = atac_wins.to(self.gpu_id, non_blocking=True)
                tf_tensor  = tf_tensor.to(self.gpu_id, non_blocking=True)
                targets    = targets.to(self.gpu_id, non_blocking=True)
                bias       = bias.to(self.gpu_id, non_blocking=True)
                tf_ids     = tf_ids.to(self.gpu_id, non_blocking=True)
                tg_ids     = tg_ids.to(self.gpu_id, non_blocking=True)
                motif_mask = motif_mask.to(self.gpu_id, non_blocking=True)

                mask_arg = motif_mask if USE_MOTIF_MASK else None
                preds, _ = self.model(
                    atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids,
                    bias=bias, motif_mask=mask_arg
                )

                loss = F.mse_loss(preds, targets)
                total_loss += loss.item(); n_batches += 1
                preds_list.append(preds); tgts_list.append(targets)

        # Stack local tensors
        preds = torch.cat(preds_list, dim=0)
        tgts  = torch.cat(tgts_list, dim=0)

        # All-gather across ranks
        world_size = dist.get_world_size()
        gathered_preds = [torch.zeros_like(preds) for _ in range(world_size)]
        gathered_tgts  = [torch.zeros_like(tgts) for _ in range(world_size)]
        dist.all_gather(gathered_preds, preds)
        dist.all_gather(gathered_tgts, tgts)

        preds = torch.cat(gathered_preds, dim=0).cpu().numpy()
        tgts  = torch.cat(gathered_tgts, dim=0).cpu().numpy()

        # Safe conversion
        preds = np.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
        tgts  = np.nan_to_num(tgts, nan=0.0, posinf=1e6, neginf=-1e6)

        # Correlations
        if np.std(tgts) < 1e-8 or np.std(preds) < 1e-8:
            pearson_corr, spearman_corr = 0.0, 0.0
        else:
            try:
                pearson_corr, _ = pearsonr(preds.ravel(), tgts.ravel())
                spearman_corr, _ = spearmanr(preds.ravel(), tgts.ravel())
            except Exception as e:
                logging.warning(f"Correlation failed: {e}")
                pearson_corr, spearman_corr = 0.0, 0.0

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss, pearson_corr, spearman_corr

    def _run_epoch(self, epoch):
        total_loss, n_batches = 0.0, 0
        per_dataset_losses = {name: [0.0, 0] for name in self.train_data} if isinstance(self.train_data, dict) else None

        if isinstance(self.train_data, dict):
            batch_iter = balanced_round_robin(self.train_data)
        else:
            batch_iter = ((batch, None) for batch in self.train_data)

        for batch, dataset_name in batch_iter:
            loss_val = self._run_batch(batch)
            if loss_val is None:
                continue
            total_loss += loss_val
            n_batches += 1

            if per_dataset_losses is not None and dataset_name is not None:
                per_dataset_losses[dataset_name][0] += loss_val
                per_dataset_losses[dataset_name][1] += 1

        avg_train_loss = total_loss / max(1, n_batches)

        avg_val_loss, pearson_corr, spearman_corr, per_dataset_val_metrics = self._validate()
        self.scheduler.step(avg_val_loss)
        return avg_train_loss, avg_val_loss, pearson_corr, spearman_corr, per_dataset_losses, per_dataset_val_metrics


    def _save_checkpoint(self, epoch, path):
        os.makedirs(path, exist_ok=True)
        ckp = self.model.module.state_dict()
        ckpt_path = os.path.join(path, f"checkpoint_epoch{epoch}.pt")
        torch.save(ckp, ckpt_path)

        # Keep only the last 3 checkpoints
        checkpoints = sorted(glob.glob(os.path.join(path, "checkpoint_epoch*.pt")))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                try:
                    os.remove(old_ckpt)
                except OSError as e:
                    logging.warning(f"Could not delete {old_ckpt}: {e}")

        if self.gpu_id == 0:
            logging.info(f"\tTraining checkpoint saved: {ckpt_path}")

    def _log_train_breakdown(self, epoch, per_dataset_losses, per_dataset_val_metrics, path):
        if self.gpu_id != 0:
            return  # only log once in multi-GPU

        csv_path = os.path.join(path, "train_dataset_breakdown.csv")
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["epoch", "dataset", "train_loss", "val_loss", "pearson", "spearman"])

            for name in per_dataset_losses.keys():
                train_loss = per_dataset_losses[name][0] / max(1, per_dataset_losses[name][1])
                val_loss, pearson, spearman = per_dataset_val_metrics.get(name, (None, None, None))
                writer.writerow([epoch, name, train_loss, val_loss, pearson, spearman])

        logging.info(f"Per-dataset metrics logged for epoch {epoch}")
        
    def train(self, max_epochs: int, path: str):
        best_val_loss = float("inf")
        best_pearson = float(0)
        patience_counter = 0
        history = []  # store per-epoch logs

        for epoch in range(max_epochs):
            train_loss, val_loss, pearson_corr, spearman_corr, per_dataset_losses, per_dataset_val_metrics = self._run_epoch(epoch)

            if self.gpu_id == 0:

                logging.info(
                    f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Pearson: {pearson_corr:.3f} | Spearman: {spearman_corr:.3f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

                lr = self.optimizer.param_groups[0]['lr']
                history.append({
                    "Epoch": epoch,
                    "Train Loss": train_loss.detach().item() if torch.is_tensor(train_loss) else float(train_loss),
                    "Val Loss": val_loss.detach().item() if torch.is_tensor(val_loss) else float(val_loss),
                    "Pearson": float(pearson_corr),
                    "Spearman": float(spearman_corr),
                    "LR": float(lr),
                })
                                
            # Checkpoint + CSV log
            if epoch % self.save_every == 0:
                if self.gpu_id == 0:
                    self._save_checkpoint(epoch, path)
                    self._write_log_csv(history, path)
                dist.barrier()
                
            # Log per-dataset breakdown every N epochs
            if train_loss is not None and epoch % self.log_train_breakdown_every == 0:
                if self.gpu_id == 0:
                    self._log_train_breakdown(epoch, per_dataset_losses, per_dataset_val_metrics, path)

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
        fieldnames = ["Epoch", "Train Loss", "Val Loss", "Pearson", "Spearman", "LR"]
        log_path = os.path.join(path, "training_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
            

def load_train_objs():
    # --- Step 1: Load pseudobulk dataset (for Fisher / EWC) ---
    pseudobulk_dataset = MultiomicTransformerDataset(
        data_dir=SAMPLE_DATA_CACHE_DIR,
        chrom_id=CHROM_ID,
        tf_vocab_path=os.path.join(COMMON_DATA, "tf_vocab.json"),
        tg_vocab_path=os.path.join(COMMON_DATA, "tg_vocab.json"),
    )

    # --- Step 2: Load all single-cell datasets for vocab union ---
    single_cell_datasets = [
        MultiomicTransformerDataset(
            data_dir=SAMPLE_DATA_CACHE_DIR,
            chrom_id=CHROM_ID,
            tf_vocab_path=os.path.join(COMMON_DATA, "tf_vocab.json"),
            tg_vocab_path=os.path.join(COMMON_DATA, "tg_vocab.json"),
            fine_tuner=True,
            sample_name=sn
        )
        for sn in FINE_TUNING_DATASETS
    ]

    # --- Step 3: Build union vocab sizes ---
    all_tf_ids = set()
    all_tg_ids = set()
    for ds in single_cell_datasets:
        assert ds.tf_name2id is not None
        assert ds.tg_name2id is not None
        all_tf_ids.update(ds.tf_name2id.keys())
        all_tg_ids.update(ds.tg_name2id.keys())

    tf_vocab_size = len(all_tf_ids)
    tg_vocab_size = len(all_tg_ids)

    # --- Step 4: Initialize model ---
    model = MultiomicTransformer(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        tf_vocab_size=tf_vocab_size,
        tg_vocab_size=tg_vocab_size,
        bias_scale=ATTN_BIAS_SCALE,
        use_shortcut=USE_SHORTCUT,
        use_motif_mask=USE_MOTIF_MASK,
        lambda_l1=SHORTCUT_L1,
        lambda_l2=SHORTCUT_L2,
        topk=SHORTCUT_TOPK,
        shortcut_dropout=SHORTCUT_DROPOUT
    )

    # --- Step 5: Load pretrained weights if available ---
    pretrained_model = FINE_TUNING_DIR / "trained_model.pt"
    if pretrained_model.exists():
        logging.info(f"Loading pretrained weights from {pretrained_model}")
        state_dict = torch.load(pretrained_model, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    # --- Step 6: Fine-tune optimizer ---
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR
    )

    return pseudobulk_dataset, single_cell_datasets, model, optimizer




def prepare_dataloader(dataset, batch_size, world_size=1, rank=0,
                       num_workers=4, pin_memory=True, seed=42, drop_last=True):
    """
    Build train/val/test loaders with the dataset's collate_fn.
    Uses DistributedSampler only when world_size > 1.
    """
    # --- deterministic split
    g = torch.Generator()
    g.manual_seed(seed)

    n_total = len(dataset)
    n_train = int(n_total * 0.70)
    n_val   = int(n_total * 0.15)
    n_test  = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=g)

    # Initialize the DDP samplers
    if world_size > 1:
        train_sampler:  DistributedSampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, drop_last=drop_last)
        val_sampler:    DistributedSampler = DistributedSampler(val_set,   num_replicas=world_size, rank=rank, drop_last=False)
        test_sampler:   DistributedSampler = DistributedSampler(test_set,  num_replicas=world_size, rank=rank, drop_last=False)
        shuffle = False
    else:
        shuffle = True

    # Create separate dataloaders for train / val / split datasets
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=MultiomicTransformerDataset.collate_fn,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        sampler=val_sampler,
        collate_fn=MultiomicTransformerDataset.collate_fn,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        sampler=test_sampler,
        collate_fn=MultiomicTransformerDataset.collate_fn,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )

    return train_loader, val_loader, test_loader


def write_run_parameters(dataset, out_dir):
    has_dist_bias = "No"
    if dataset.dist_bias_tensor is not None:
        has_dist_bias = "Yes"
    
    has_motif_mask = "Yes"
    if USE_MOTIF_MASK == False or USE_SHORTCUT == False:
        has_motif_mask = "No"

    logging.info("\n===== MultiomicTransformerDataset Loaded =====")
    logging.info(f"Chromosome:          {CHROM_ID}")
    logging.info(f"Genes:               {len(dataset.tg_ids)}")
    logging.info(f"Windows (RE):        {dataset.num_windows}")
    logging.info(f"TFs:                 {len(dataset.tf_ids)}")
    logging.info(f"Metacells:           {len(dataset.metacell_names)}")
    logging.info(f"Epochs:              {TOTAL_EPOCHS}")
    logging.info(f"Batch Size:          {BATCH_SIZE}")
    logging.info(f"Model Dimension:     {D_MODEL}")
    logging.info(f"Attention Heads:     {NUM_HEADS}")
    logging.info(f"Attention Layers:    {NUM_LAYERS}")
    logging.info(f"Feedforward Layers:  {D_FF}")
    logging.info(f"Dropout:             {DROPOUT}")
    logging.info(f"Dist bias?:          {has_dist_bias}")
    logging.info(f"Motif Mask?:         {has_motif_mask}")
    logging.info(f"Shortcut L1:         {SHORTCUT_L1}")
    logging.info(f"Shortcut L2:         {SHORTCUT_L2}")
    logging.info(f"Shortcut Dropout:    {SHORTCUT_DROPOUT}")
    logging.info(f"Shortcut Top K:      {SHORTCUT_TOPK}")
    logging.info("================================================")
    
    run_params = {
        "Genes": dataset.tg_tensor_all.shape[0],
        "Windows": dataset.num_windows,
        "TFs": dataset.tf_tensor_all.shape[0],
        "Metacells": len(dataset.metacell_names),  # store count, not huge list
        "Epochs": TOTAL_EPOCHS,
        "Batch Size": BATCH_SIZE,
        "d_model": D_MODEL,
        "Attention Heads": NUM_HEADS,
        "Model Layers": NUM_LAYERS,
        "d_feedforward": D_FF,
        "Dropout": DROPOUT,
        "Distance Bias":has_dist_bias,
        "Motif Mask": has_motif_mask,
        "Shortcut L1": SHORTCUT_L1,
        "Shortcut L2": SHORTCUT_L2,
        "Shortcut Dropout": SHORTCUT_DROPOUT,
        "Shortcut Top K": SHORTCUT_TOPK
    }

    path = os.path.join(out_dir, "run_parameters.json")
    with open(path, "w") as f:
        json.dump(run_params, f, indent=4)  # indent=4 for readability
    logging.info(f"Run parameters written to {path}")
    
def balanced_round_robin(loaders, seed=42):
    """
    Cycle through multiple DataLoaders until the largest is exhausted.
    Smaller datasets are repeated to balance contribution.
    """
    max_len = max(len(loader) for loader in loaders.values())
    keys = list(loaders.keys())

    # Keep loaders intact, track iterators separately
    iters = {name: iter(loader) for name, loader in loaders.items()}

    for i in range(max_len * len(keys)):
        name = keys[i % len(keys)]
        try:
            batch = next(iters[name])
        except StopIteration:
            # Restart smaller dataset iterator
            iters[name] = iter(loaders[name])
            batch = next(iters[name])
        yield batch, name


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    setup_logging(rank)

    try:
        # --- Step 0: Load everything (pseudobulk, all single-cell, model, optimizer) ---
        pseudobulk_dataset, single_cell_datasets, model, optimizer = load_train_objs()
        device = f"cuda:{rank}"
        model = model.to(device)

        # --- Step 1: Compute Fisher on pseudobulk dataset ---
        pseudobulk_loader, _, _ = prepare_dataloader(
            pseudobulk_dataset, batch_size, world_size, rank
        )

        fisher_bundle_path = FINE_TUNING_DIR / "ewc_bundle.pt"
        if fisher_bundle_path.exists():
            ref_params, fisher_diag = ewc_utils.load_ewc_bundle(fisher_bundle_path, device=device)
            total_size = len(pseudobulk_dataset)
            if rank == 0:
                logging.info(f"Loaded existing Fisher/EWC bundle: {fisher_bundle_path}")
        else:
            if rank == 0:
                logging.info("Computing Fisher matrix on pseudobulk dataset...")
            fisher_diag = ewc_utils.compute_fisher_diag(
                model, pseudobulk_loader, device=device, n_batches=100
            )
            ref_params = {n: p.detach().clone().to(device) for n, p in model.named_parameters()}
            ewc_utils.save_ewc_bundle(fisher_bundle_path, model, fisher_diag)
            total_size = len(pseudobulk_dataset)
            if rank == 0:
                logging.info(f"Saved Fisher/EWC bundle to {fisher_bundle_path}")

        # --- Step 2: Build round-robin loaders ---
        train_loaders = {
            sn: prepare_dataloader(ds, batch_size, world_size, rank)[0]
            for sn, ds in zip(FINE_TUNING_DATASETS, single_cell_datasets)
        }
        val_loaders = {
            sn: prepare_dataloader(ds, batch_size, world_size, rank)[1]
            for sn, ds in zip(FINE_TUNING_DATASETS, single_cell_datasets)
        }
        test_loaders = {
            sn: prepare_dataloader(ds, batch_size, world_size, rank)[2]
            for sn, ds in zip(FINE_TUNING_DATASETS, single_cell_datasets)
        }

        # --- Step 3: Train with round-robin sampler ---
        trainer = Trainer(
            model, train_loaders, val_loaders, nn.MSELoss(), optimizer,
            gpu_id=rank, save_every=save_every, patience=FINETUNE_PATIENCE,
            ref_params=ref_params, fisher_diag=fisher_diag, lambda_ewc=EWC_LAMBDA
        )
        
        if rank == 0:
            logging.info("----- ROUND-ROBIN TRAINING STARTED -----")

        out_dir = FINE_TUNING_DIR / "fine_tuning"
        os.makedirs(out_dir, exist_ok=True)

        trainer.train(max_epochs=total_epochs, path=str(out_dir))

        # --- Step 4: Recompute Fisher on all fine-tuned data ---
        all_train_sets = torch.utils.data.ConcatDataset(single_cell_datasets)
        train_loader, _, _ = prepare_dataloader(all_train_sets, batch_size, world_size, rank)
        
        new_fisher = ewc_utils.compute_fisher_diag(model, train_loader, device=device, n_batches=100)
        new_ref_params = {n: p.detach().clone().to(device) for n, p in model.named_parameters()}
        len_all = sum(len(ds) for ds in single_cell_datasets)
        fisher_diag = ewc_utils.merge_fishers(fisher_diag, new_fisher, total_size, len_all)
        ref_params = new_ref_params

        if rank == 0:
            logging.info("\nFine-tuning complete on all datasets (round-robin).")

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