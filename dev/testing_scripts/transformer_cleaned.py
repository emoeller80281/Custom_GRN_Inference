
import os
import glob
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
import numpy as np
import pandas as pd
import pybedtools
from grn_inference import utils
from typing import Optional, Tuple, Any, Dict
from scipy import sparse
from contextlib import nullcontext
from collections import OrderedDict
import logging
import time
from contextlib import nullcontext  
import multiprocessing as mp
import warnings
import math
from datetime import datetime
from transformer_dataset import WindowsWithTargets, make_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel as DDP

from sc_multi_transformer import MultiomicTransformer
from transformer_data_preparation import preprocess_training_data


# ------ PyTorch Configurations ------
warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group` or `barrier `")
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(1)
np.random.seed(42)

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TORCH_ALLOW_TF32"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

USE_AMP = False
AMP_DTYPE = torch.float16  # ignored
scaler = torch.amp.GradScaler(enabled=False)

world = int(os.environ.get("WORLD_SIZE", "1"))
per = max(1, mp.cpu_count() // world)

for var in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[var] = str(per)

torch.set_num_threads(per)
torch.set_num_interop_threads(1)

#=================================== USER SETTINGS ===================================
# ----- User Settings -----
load_model = False
window_size = 1000
num_cells = 500
chrom_id = "chr19"
force_recalculate = True

rna_data_filename = "mESC_filtered_L2_E7.5_rep1_RNA_processed.parquet"
atac_data_filename = "mESC_filtered_L2_E7.5_rep1_ATAC_processed.parquet"

# ----- Model Configurations -----
# Logging
log_every_n_steps = 5

# Model Size
d_model = 196
tf_channels = 64
kernel_and_stride_size = 4

# Encoder Settings
encoder_nhead = 7
encoder_dim_feedforward = 1024
encoder_num_layers = 3
dropout = 0.1

# Train-test split
validation_fraction = 0.15

# Training configurations
epochs = 75
batch_size = 5
effective_batch_size = 16
warmup_steps = 100
learning_rate = 3e-4
patience = 5
epochs_before_patience = 3

#=====================================================================================

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
MM10_GENOME_DIR = os.path.join(PROJECT_DIR, "data/reference_genome/mm10")
MM10_GENE_TSS_FILE = os.path.join(PROJECT_DIR, "data/genome_annotation/mm10/mm10_TSS.bed")
GROUND_TRUTH_DIR = os.path.join(PROJECT_DIR, "ground_truth_files")
SAMPLE_INPUT_DIR = os.path.join(PROJECT_DIR, "input/mESC/filtered_L2_E7.5_rep1")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output/transformer_testing_output")
DEBUG_FILE = os.path.join(PROJECT_DIR, "LOGS/transformer_training.debug")

MM10_FASTA_FILE = os.path.join(MM10_GENOME_DIR, f"{chrom_id}.fa")
MM10_CHROM_SIZES_FILE = os.path.join(MM10_GENOME_DIR, "chrom.sizes")

time_now = datetime.now().strftime("%d%m%y%H%M%S")

TRAINING_DIR=os.path.join(PROJECT_DIR, f"training_stats_{time_now}/")

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, f"checkpoints_{time_now}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "training_stats"), exist_ok=True)

def init_ddp():
    # torchrun sets these
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)
    return rank, world_size, local_rank

def setup_logging():
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    handlers = []
    if rank == 0:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.ERROR,
                        handlers=handlers, format="%(message)s")

def _rank0() -> bool:
    return (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized()

def _atomic_save(obj: Dict[str, Any], path: str) -> None:
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_checkpoint(model, optimizer, scheduler, epoch: int, loss: float,
                    best_val: float, fname: str) -> str:
    """Always safe to call from any rank; only rank0 writes."""
    if not _rank0():
        if dist.is_initialized(): dist.barrier()
        return fname

    logging.debug(f"Saving checkpoint -> {fname}")
    # unwrap DDP
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "epoch": epoch,
        "loss": loss,
        "best_val": best_val,
        "state_dict": raw_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "torch_version": torch.__version__,
    }
    _atomic_save(state, fname)
    if dist.is_initialized(): dist.barrier()
    return fname

def save_regular(model, optimizer, scheduler, epoch: int, loss: float, best_val: float) -> str:
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:04d}.pth.tar")
    return save_checkpoint(model, optimizer, scheduler, epoch, loss, best_val, path)

def unwrap(m):
    return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

def main():
    # ----- DDP init -----
    multi_gpu = torch.cuda.device_count() > 1 and os.environ.get("WORLD_SIZE", "1") != "1"
    if multi_gpu:
        rank, world, local = init_ddp()
    else:
        rank, world, local = 0, 1, 0

    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    setup_logging()
    
    
    

    # ----- Paths to precomputed dataset -----
    base = Path(OUTPUT_DIR) / "precomputed_dataset" / f"{chrom_id}_{window_size//1000}kb"
    manifest_path = base / "meta" / "manifest.parquet"
    genes_npy_path = base / "meta" / "genes.npy"
    tfs_npy_path   = base / "meta" / "tfs.npy"
    rna_parquet_path = os.path.join(SAMPLE_INPUT_DIR, rna_data_filename)
    
    if not os.path.exists(manifest_path):
        preprocess_training_data(window_size, num_cells, chrom_id, force_recalculate)

    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
    assert genes_npy_path.exists(), f"Missing genes.npy: {genes_npy_path}"
    assert tfs_npy_path.exists(),   f"Missing tfs.npy: {tfs_npy_path}"

    # ----- Load meta dims -----
    if rank == 0:
        logging.info("Loading dataset metadata")
    genes = np.load(genes_npy_path, allow_pickle=True)
    n_genes = int(len(genes))
    tfs = np.load(tfs_npy_path, allow_pickle=True)
    TF = int(len(tfs))

    # ----- Dataset & split -----
    if rank == 0:
        logging.info("Creating WindowsWithTargets")
    full_ds = WindowsWithTargets(str(manifest_path), str(rna_parquet_path), str(genes_npy_path), drop_empty=True)

    N = len(full_ds)
    rng = np.random.default_rng(0)
    indices = np.arange(N)
    rng.shuffle(indices)
    n_val = max(1, int(N * validation_fraction))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_ds = torch.utils.data.Subset(full_ds, train_idx)
    val_ds   = torch.utils.data.Subset(full_ds, val_idx)

    train_sampler: DistributedSampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) if multi_gpu else None
    val_sampler: DistributedSampler   = DistributedSampler(val_ds,   num_replicas=world, rank=rank, shuffle=False) if multi_gpu else None

    collate = make_collate(pad_to_max=True)

    if rank == 0:
        logging.info("Creating Training DataLoader")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True, collate_fn=collate, drop_last=False
    )
    
    if rank == 0:
        logging.info("Creating Validation DataLoader")
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate, drop_last=False
    )
    
    def build_stats_from_parquet(rna_parquet_path, genes, barcodes, device):
        df = pd.read_parquet(rna_parquet_path).set_index("gene_id")
        df = df.reindex(genes)           # << critical: identical order
        df = df[barcodes]                # train barcodes only

        mask = df.notna()
        arr  = df.fillna(0.0).astype("float32").to_numpy(copy=False)
        m    = np.where(mask.to_numpy(), arr, 0.0).sum(axis=1)
        c    = mask.to_numpy().sum(axis=1)
        mu_np = m / np.clip(c, 1, None)

        m2   = np.where(mask.to_numpy(), arr*arr, 0.0).sum(axis=1)
        var  = m2 / np.clip(c, 1, None) - mu_np**2
        sd_np = np.sqrt(np.clip(var, 0.0, None))

        mu  = torch.tensor(mu_np, device=device)
        sd  = torch.tensor(sd_np, device=device).clamp_(1e-2)  # avoid div-by-0
        seen = torch.tensor(c > 0, device=device)
        return mu, sd, seen

    train_barcodes = full_ds.manifest.iloc[train_idx]["barcode"].tolist()
    mu, sd, seen_genes_mask = build_stats_from_parquet(rna_parquet_path, genes, train_barcodes, device)


    # ----- Logging -----
    if rank == 0:
        nonempty = full_ds.manifest["wlen"].gt(0).sum()
        logging.info("Data Size:")
        logging.info(f"  - Window Size: {window_size} bp")
        logging.info(f"  - Samples (cells): {N:,} | non-empty: {int(nonempty):,}")
        logging.info(f"  - Num TFs = {TF:,}")
        logging.info(f"  - Num TGs = {n_genes:,}")

        logging.info("\nModel Parameters:")
        logging.info(f"  - Model Dimensions = {d_model}")
        logging.info(f"  - TF Linear Projection = {TF:,} -> {tf_channels}")
        logging.info(f"  - Window Data Pooling: kernel={kernel_and_stride_size}, stride={kernel_and_stride_size}")

        logging.info("\nEncoder Settings")
        logging.info(f"  - Encoder Layers = {encoder_num_layers}")
        logging.info(f"  - Number of Heads = {encoder_nhead}")
        logging.info(f"  - Feedforward Layer Neurons = {encoder_dim_feedforward}")

    # better filename (use counts, not list)
    if window_size > 1000:
        training_stats_filename = f"{chrom_id}_training_loss_window_{window_size // 1000}kb_kernel_{kernel_and_stride_size}_{N}_cells.csv"
    else:
        training_stats_filename = f"{chrom_id}_training_loss_window_{window_size}bp_kernel_{kernel_and_stride_size}_{N}_cells.csv"

    # ----- Model -----
    assert d_model % encoder_nhead == 0, "d_model must be divisible by nhead"

    model = MultiomicTransformer(
        n_genes=n_genes,
        tf_in_dim=TF,
        d_model=d_model,
        nhead=encoder_nhead,
        dff=encoder_dim_feedforward,
        dropout=dropout,
        n_layers=encoder_num_layers,
        kernel_stride_size=kernel_and_stride_size,
    ).to(device)

    if multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], output_device=local, find_unused_parameters=True
        )

    inner = unwrap(model)
    gene_ids = torch.arange(n_genes, device=device)

    # ----- Compute mu/sd on TRAIN only (from dataset’s aligned RNA/mask) -----
    # full_ds.rna_df/mask_df are aligned: index=genes, columns=barcodes in manifest order
    train_barcodes = full_ds.manifest.iloc[train_idx]["barcode"].tolist()
    mu, sd, seen_genes_mask = build_stats_from_parquet(rna_parquet_path, genes, train_barcodes, device)

    if dist.is_available() and dist.is_initialized():
        for t in (mu, sd, seen_genes_mask):
            dist.broadcast(t, src=0)
    
    rna_train  = full_ds.rna_df[train_barcodes].to_numpy(dtype=np.float32, copy=False)    # [G, Ntrain]
    mask_train = full_ds.mask_df[train_barcodes].to_numpy(dtype=np.float32, copy=False)   # [G, Ntrain]

    sum_y  = (rna_train * mask_train).sum(axis=1)                     # [G]
    sum_y2 = ((rna_train**2) * mask_train).sum(axis=1)                # [G]
    count  = mask_train.sum(axis=1)                                   # [G]

    mu_np = sum_y / np.maximum(count, 1.0)
    var_np = sum_y2 / np.maximum(count, 1.0) - mu_np**2
    sd_np = np.sqrt(np.clip(var_np, 0.0, None))
    # handle never-seen genes
    never = (count == 0)
    mu_np[never] = 0.0
    sd_np[never] = 1.0

    mu = torch.tensor(mu_np, dtype=torch.float32, device=device)
    sd = torch.tensor(sd_np, dtype=torch.float32, device=device).clamp_(min=1e-2)
    seen_genes_mask = torch.tensor(count > 0, dtype=torch.bool, device=device)  # [G]

    # ----- Optimizer + warmup→cosine -----
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    steps_per_epoch = len(train_loader)
    total_optimizer_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(0.05 * total_optimizer_steps))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))  # 0→1
        progress = float(current_step - warmup_steps) / float(max(1, total_optimizer_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))             # cosine 1→0

    sched = LambdaLR(opt, lr_lambda=lr_lambda, last_epoch=-1)

    # ----- Training loop -----
    loss_by_epoch = {"epoch": [], "train_loss": [], "val_loss": [], "epoch_sec": []}
    best_val, pat = float("inf"), 0
    start_epoch = 1

    if rank == 0:
        logging.info("\n ----- Model Training -----")

    for epoch in range(start_epoch, epochs + 1):
        if rank == 0:
            logging.info(f"\nEpoch ({epoch}/{epochs}) ")
            logging.info("  - Running Training ")

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start_time = time.time()
        model.train()
        inner = unwrap(model)

        running_loss_sum = 0.0
        running_valid_count = 0
        skipped = 0
        for step, batch in enumerate(train_loader, start=1):
            tf_pad, bias_pad, kpm, y, ymask, tf_expr = batch   # tf_pad:[B,W,TF], bias_pad:[B,W,G], kpm:[B,W], y:[B,G], ymask:[B,G]
            tf_pad   = tf_pad.to(device, non_blocking=True)
            bias_pad = bias_pad.to(device, non_blocking=True)
            kpm      = kpm.to(device, non_blocking=True)
            y        = y.to(device, non_blocking=True)
            ymask    = ymask.bool().to(device, non_blocking=True)
            tf_expr  = tf_expr.to(device, non_blocking=True)

            window_tokens = inner.window_from_tf(tf_pad)  # [B, W, d_model]
            tf_expr_norm = (tf_expr - tf_expr.mean(1, keepdim=True)) / (tf_expr.std(1, keepdim=True) + 1e-6)
            tf_expr_norm = torch.clamp(tf_expr_norm, -3.0, 3.0)
            tf_tokens = tf_expr_norm.unsqueeze(-1) * inner.tf_id_embed.weight.unsqueeze(0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                pred, _ = inner.forward(
                    windows=window_tokens,
                    tf_tokens=tf_tokens,
                    gene_ids=gene_ids,
                    key_padding_mask_win=kpm,
                    attn_bias_win=bias_pad,
                )

                y_norm = ((y - mu) / sd).clamp_(-8.0, 8.0)
                valid_mask = ymask & seen_genes_mask.unsqueeze(0) & torch.isfinite(y_norm)

                limit = 10.0
                pred_for_loss = limit * torch.tanh(pred / limit)

                if valid_mask.any():
                    batch_loss_sum = F.huber_loss(
                        pred_for_loss[valid_mask].float(),
                        y_norm[valid_mask].float(),
                        delta=1.0,
                        reduction="sum",
                    )
                    loss = batch_loss_sum
                    valid_elems = int(valid_mask.sum().item())
                if not valid_mask.any():
                    skipped += 1
                    continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(inner.parameters(), 1.0)
            opt.step()
            sched.step()

            running_loss_sum    += float(batch_loss_sum.item())
            running_valid_count += valid_elems

            if rank == 0 and (step % log_every_n_steps == 0):
                logging.info(f"       {step}/{len(train_loader)}")

        torch.cuda.synchronize(device)
        
        # ----- Validation -----
        model.eval()
        inner = unwrap(model)
        val_loss_sum, val_valid_count = 0.0, 0

        with torch.no_grad():
            if rank == 0:
                logging.info("  - Running Validation:")

            for vi, batch in enumerate(val_loader, start=1):
                tf_pad, bias_pad, kpm, y, ymask, tf_expr = batch   # tf_pad:[B,W,TF], bias_pad:[B,W,G], kpm:[B,W], y:[B,G], ymask:[B,G]
                tf_pad   = tf_pad.to(device, non_blocking=True)
                bias_pad = bias_pad.to(device, non_blocking=True)
                kpm      = kpm.to(device, non_blocking=True)
                y        = y.to(device, non_blocking=True)
                ymask    = ymask.bool().to(device, non_blocking=True)
                tf_expr  = tf_expr.to(device, non_blocking=True)

                window_tokens = inner.window_from_tf(tf_pad)  # [B, W, d_model]
                tf_expr_norm = (tf_expr - tf_expr.mean(1, keepdim=True)) / (tf_expr.std(1, keepdim=True) + 1e-6)
                tf_expr_norm = torch.clamp(tf_expr_norm, -3.0, 3.0)
                tf_tokens = tf_expr_norm.unsqueeze(-1) * inner.tf_id_embed.weight.unsqueeze(0)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    pred, _ = inner.forward(
                        windows=window_tokens,
                        tf_tokens=tf_tokens,
                        gene_ids=gene_ids,
                        key_padding_mask_win=kpm,
                        attn_bias_win=bias_pad,
                    )

                    y_norm = ((y - mu) / sd).clamp(-8.0, 8.0)
                    mask_eff = ymask & seen_genes_mask.unsqueeze(0) & torch.isfinite(y_norm)
                    if not mask_eff.any():
                        skipped += 1
                        continue

                    limit = 10.0
                    pred_for_loss = limit * torch.tanh(pred / limit)

                    val_batch_sum = F.huber_loss(
                        pred_for_loss[mask_eff].float(),
                        y_norm[mask_eff].float(),
                        delta=1.0,
                        reduction="sum",
                    )
                    val_loss_sum    += float(val_batch_sum.item())
                    val_valid_count += int(mask_eff.sum().item())

                if rank == 0 and (vi % log_every_n_steps == 0):
                    logging.info(f"       {vi}/{len(val_loader)}")

        torch.cuda.synchronize(device)
        
        global_cnt = torch.tensor([running_valid_count], device=device)
        if dist.is_initialized(): dist.all_reduce(global_cnt, op=dist.ReduceOp.SUM)
        if global_cnt.item() == 0:
            if rank == 0:
                logging.error("No valid supervised elements this epoch. Check gene alignment / masks.")
            break

        # ----- Epoch metrics (average per supervised element) -----
        sum_t = torch.tensor([running_loss_sum],    device=device)
        cnt_t = torch.tensor([running_valid_count], device=device)
        sum_v = torch.tensor([val_loss_sum],        device=device)
        cnt_v = torch.tensor([val_valid_count],     device=device)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_v, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_v, op=dist.ReduceOp.SUM)

        train_loss_avg = (sum_t / cnt_t.clamp(min=1)).item()
        val_loss_avg   = (sum_v / cnt_v.clamp(min=1)).item()

        if rank == 0:
            print(f"[Epoch {epoch:02d}] train_loss={train_loss_avg:.4f}  val_loss={val_loss_avg:.4f}")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # ----- Early stopping & checkpointing -----
        is_best = val_loss_avg < best_val - 1e-5
        if epoch > epochs_before_patience:
            if is_best:
                best_val = val_loss_avg
                pat = 0
            else:
                pat += 1
                logging.info(f"No improvement ({pat}/{patience}) — val {val_loss_avg:.4f} ≥ best {best_val:.4f}")
        stop = pat >= patience

        if rank == 0 and epoch % 5 == 0:
            save_regular(model, opt, sched, epoch, loss=train_loss_avg, best_val=best_val)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # ----- Log epoch -----
        if rank == 0:
            time_sec = time.time() - epoch_start_time
            loss_by_epoch["epoch"].append(epoch)
            loss_by_epoch["train_loss"].append(train_loss_avg)
            loss_by_epoch["val_loss"].append(val_loss_avg)
            loss_by_epoch["epoch_sec"].append(time_sec)

        if stop:
            if rank == 0:
                logging.info(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
                training_stats = pd.DataFrame(loss_by_epoch).set_index("epoch")
                training_stats.to_csv(os.path.join(OUTPUT_DIR, f"training_stats/{training_stats_filename}"), header=True, index=True)
            break

    # ----- Final save -----
    if rank == 0 and len(loss_by_epoch["epoch"]) > 0:
        training_stats = pd.DataFrame(loss_by_epoch).set_index("epoch")
        training_stats.to_csv(os.path.join(OUTPUT_DIR, f"training_stats/{training_stats_filename}"), header=True, index=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
        
if __name__ == "__main__":
    main()
