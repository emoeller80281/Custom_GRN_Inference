import os
import sys
import csv
import torch
import joblib
import json
import pandas as pd
import numpy as np
import logging
import pickle
import math
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformer import MultiomicTransformer
from transformer_dataset import MultiomicTransformerDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
import torch.distributed as dist

import matplotlib.pyplot as plt
import seaborn as sns

from eval import (
    per_gene_correlation
)

import warnings
warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group`")

CHROM_ID = "chr19"
SAMPLE_NAME = "mESC"

TOTAL_EPOCHS=200
BATCH_SIZE=8
GRAD_ACCUM_STEPS=8
PATIENCE=15
CORR_LOSS_WEIGHT=0.05

D_MODEL = 246
NUM_HEADS = 6
NUM_LAYERS = 3
D_FF = 768
DROPOUT = 0.1

ATTN_BIAS_SCALE = 1.0

# TF to TG shortcut parameters
SHORTCUT_L1 = 0
SHORTCUT_L2 = 0
SHORTCUT_TOPK = None
SHORTCUT_DROPOUT = 0

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
DATA_DIR = os.path.join(PROJECT_DIR, f"dev/transformer/transformer_data/{SAMPLE_NAME}")
OUTPUT_DIR = os.path.join(PROJECT_DIR, f"output/transformer_testing_output/{SAMPLE_NAME}/{CHROM_ID}")
CHIP_GROUND_TRUTH_FILE = os.path.join(PROJECT_DIR, "ground_truth_files/combined_ground_truth.csv")

def ddp_setup(rank, world_size):
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
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        gpu_id: int,
        save_every: int,
        patience: int = 10,
        min_delta: float = 1e-3,
        grad_accum_steps: int = 1
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.scaler = GradScaler(device="cuda")
        self.grad_accum_steps = grad_accum_steps
        self.step_count = 0

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=7
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

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            preds_out = self.model(
                atac_wins,
                tf_tensor,
                tf_ids=tf_ids,
                tg_ids=tg_ids,
                bias=bias,
                motif_mask=motif_mask,
            )
            preds = preds_out[0] if isinstance(preds_out, tuple) else preds_out

            # Base regression loss
            loss = self.criterion(preds, targets)

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

        # Gradient accumulation
        loss = loss / self.grad_accum_steps
        self.scaler.scale(loss).backward()
        self.step_count += 1

        if self.step_count % self.grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        return loss.detach() * self.grad_accum_steps



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

                preds_out = self.model(
                    atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids,
                    bias=bias, motif_mask=motif_mask
                )
                
                if isinstance(preds_out, tuple):
                    preds, _ = preds_out
                else:
                    preds = preds_out
                
                loss  = F.mse_loss(preds, targets)
                total_loss += loss.item(); n_batches += 1
                preds_list.append(preds); tgts_list.append(targets)


        # Stack local tensors
        preds = torch.cat(preds_list, dim=0)
        tgts = torch.cat(tgts_list, dim=0)

        # All-gather across ranks
        world_size = dist.get_world_size()
        gathered_preds = [torch.zeros_like(preds) for _ in range(world_size)]
        gathered_tgts  = [torch.zeros_like(tgts) for _ in range(world_size)]
        dist.all_gather(gathered_preds, preds)
        dist.all_gather(gathered_tgts, tgts)

        # Concatenate global arrays
        preds = torch.cat(gathered_preds, dim=0).cpu().numpy()
        tgts  = torch.cat(gathered_tgts, dim=0).cpu().numpy()

        # Replace NaN/inf with safe values
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
    
    def _run_epoch(self, epoch, chip_edges=None):
        sampler = getattr(self.train_data, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        total_loss, n_batches = 0.0, 0

        for atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask in self.train_data:
            loss_val = self._run_batch((atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask))
            if loss_val is None: 
                continue
            total_loss += loss_val
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)
        avg_val_loss, pearson_corr, spearman_corr = self._validate()

        # update LR schedule
        self.scheduler.step(avg_val_loss)

        return avg_train_loss, avg_val_loss, pearson_corr, spearman_corr

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
            train_loss, val_loss, pearson_corr, spearman_corr = self._run_epoch(epoch)


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
                    "Train Loss": float(train_loss),
                    "Val Loss": float(val_loss),
                    "Pearson": float(pearson_corr),
                    "Spearman": float(spearman_corr),
                    "LR": float(lr),
                })
                
            # Checkpoint + CSV log
            if epoch % self.save_every == 0:
                if self.gpu_id == 0:
                    self._save_checkpoint(epoch, path)
                    self._write_log_csv(history, path, epoch)
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
                        self._write_log_csv(history, path, epoch, final=True)
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
            self._write_log_csv(history, path, epoch, final=True)
            logging.info("Training loop exited normally.")
    
    def _write_log_csv(self, history, path, epoch, final=False):
        fieldnames = ["Epoch", "Train Loss", "Val Loss", "Pearson", "Spearman", "LR"]
        log_path = os.path.join(path, "training_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
            

def load_train_objs(DATA_DIR, CHROM_ID,
                    D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT,
                    lr=1e-3):

    COMMON_DIR = os.path.join(
        os.path.dirname(DATA_DIR),  # .../transformer_data/
        "common"
    )

    dataset = MultiomicTransformerDataset(
        data_dir=DATA_DIR,
        chrom_id=CHROM_ID,
        tf_vocab_path=os.path.join(COMMON_DIR, "tf_vocab.json"),
        tg_vocab_path=os.path.join(COMMON_DIR, "tg_vocab.json"),
    )

    # global vocab sizes (from common vocab)
    tf_vocab_size=len(dataset.tf_name2id)
    tg_vocab_size=len(dataset.tg_name2id)

    model = MultiomicTransformer(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        tf_vocab_size=tf_vocab_size,
        tg_vocab_size=tg_vocab_size,
        bias_scale=ATTN_BIAS_SCALE,
        use_shortcut=False,
        use_motif_mask=False,
        lambda_l1=SHORTCUT_L1,
        lambda_l2=SHORTCUT_L2,
        topk=SHORTCUT_TOPK,
        shortcut_dropout=SHORTCUT_DROPOUT
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return dataset, model, optimizer

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

    # --- samplers (DDP only)
    if world_size > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, drop_last=drop_last)
        val_sampler   = DistributedSampler(val_set,   num_replicas=world_size, rank=rank, drop_last=False)
        test_sampler  = DistributedSampler(test_set,  num_replicas=world_size, rank=rank, drop_last=False)
        shuffle = False
    else:
        train_sampler = val_sampler = test_sampler = None
        shuffle = True  # only for single-GPU/CPU

    # --- loaders (always use the dataset's collate_fn)
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


def write_run_parameters(dataset, out_dir, has_dist_bias, has_motif_mask):
    run_params = {
        "Genes": dataset.tg_tensor_all.shape[0],
        "Windows": dataset.num_windows,
        "TFs": dataset.tf_tensor_all.shape[0],
        "Metacells": len(dataset.metacell_names),  # store count, not huge list
        "Epochs": TOTAL_EPOCHS,
        "Batch Size": BATCH_SIZE,
        "Grad accumulation steps": GRAD_ACCUM_STEPS,
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

def dist_broadcast_list(obj, src=0):
    device = f"cuda:{torch.cuda.current_device()}"
    if dist.get_rank() == src:
        data = pickle.dumps(obj)
        tensor = torch.as_tensor(list(data), dtype=torch.uint8, device=device)
        size = torch.tensor([tensor.numel()], device=device)
    else:
        tensor = torch.tensor([], dtype=torch.uint8, device=device)
        size   = torch.tensor([0], device=device)

    dist.broadcast(size, src=src)
    if dist.get_rank() != src:
        tensor = torch.empty(size.item(), dtype=torch.uint8, device=device)
    dist.broadcast(tensor, src=src)
    return pickle.loads(bytes(tensor.tolist()))


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



def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    
    assert D_MODEL % NUM_HEADS == 0, f"{D_MODEL} not divisible by {NUM_HEADS}"
    
    ddp_setup(rank, world_size)
    setup_logging(rank)
    
    try:
        time_now = datetime.now().strftime("%d_%m_%H_%M_%S")
        training_output_dir = os.path.join(OUTPUT_DIR, f"model_training_{time_now}")
        os.makedirs(training_output_dir, exist_ok=True)
        
        if rank == 0:
            chip = pd.read_csv(CHIP_GROUND_TRUTH_FILE)
            chip_edges = {(t.capitalize(), g.capitalize()) for t, g in zip(chip["TF"], chip["TG"])}
        else:
            chip_edges = None
        chip_edges = dist_broadcast_list(chip_edges, src=0)
            
        dataset, model, optimizer = load_train_objs(
            DATA_DIR, CHROM_ID,
            D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT, lr=1e-3
        )
        
        if rank == 0:
            with open(os.path.join(training_output_dir, "tf_vocab.json"), "w") as f:
                json.dump(dataset.tf_name2id, f)
            with open(os.path.join(training_output_dir, "tg_vocab.json"), "w") as f:
                json.dump(dataset.tg_name2id, f)
                
            if getattr(dataset, "scaler", None) is not None:
                scaler_out = os.path.join(training_output_dir, "tg_scaler.pkl")
                joblib.dump(dataset.scaler, scaler_out)
                logging.info(f"Saved TG scaler to {scaler_out}")

        has_dist_bias = "No"
        if dataset.dist_bias_tensor is not None:
            has_dist_bias = "Yes"
            
        has_motif_mask = "No"
        if dataset.motif_mask_tensor is not None:
            has_motif_mask = "Yes"
        
        if rank == 0:
            logging.info("\n===== MultiomicTransformerDataset Loaded =====")
            logging.info(f"Chromosome:          {CHROM_ID}")
            logging.info(f"Genes:               {len(dataset.tg_ids)}")
            logging.info(f"Windows (RE):        {dataset.num_windows}")
            logging.info(f"TFs:                 {len(dataset.tf_ids)}")
            logging.info(f"Metacells:           {len(dataset.metacell_names)}")
            logging.info(f"Epochs:              {TOTAL_EPOCHS}")
            logging.info(f"Batch Size:          {BATCH_SIZE}")
            logging.info(f"Grad steps:          {GRAD_ACCUM_STEPS}")
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
            write_run_parameters(dataset, training_output_dir, has_dist_bias, has_motif_mask)
        
        if rank == 0:
            logging.info("Preparing dataloader")
        train_loader, val_loader, test_loader = prepare_dataloader(dataset, batch_size, world_size, rank)
        criterion = nn.MSELoss()
        
        if rank == 0:
            logging.info("Creating Trainer")
        trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, gpu_id=rank, save_every=save_every, patience=PATIENCE, grad_accum_steps=GRAD_ACCUM_STEPS)
        
        if rank == 0:
            logging.info("\n ----- TRAINING STARTED -----")
        trainer.train(max_epochs=TOTAL_EPOCHS, path=training_output_dir)
        
        if rank == 0:
            # Save model checkpoint
            torch.save(trainer.model.module.state_dict(),
                    os.path.join(training_output_dir, "trained_model.pt"))
            logging.info("Saved final trained model")
            
            # Save TF→TG attention weights from the shortcut
            if hasattr(trainer.model.module, "shortcut_layer") and hasattr(trainer.model.module.shortcut_layer, "attn"):
                mean_attn = get_mean_tf_tg_attention(trainer.model.module, test_loader, device=f"cuda:{rank}")
                mean_attn_df = pd.DataFrame(
                    mean_attn.numpy(),
                    index=dataset.tg_names,
                    columns=dataset.tf_names
                )
                mean_attn_df.to_csv(os.path.join(training_output_dir, "tf_tg_mean_attention.csv"))
                logging.info(f"Saved TF→TG attention weights to 'tf_tg_mean_attention.csv'")
                
                plt.figure(figsize=(12,8))
                sns.heatmap(mean_attn_df.iloc[:50, :50], cmap="viridis")
                plt.title("TF→TG attention weights (subset)")
                plt.tight_layout()
                plt.savefig(os.path.join(training_output_dir, "tf_tg_attention_heatmap.png"))                
                
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