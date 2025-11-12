import csv
import json
import logging
import os
import sys
import time
import warnings
import pickle
from pathlib import Path

import sys
sys.path.append(Path(__file__).resolve().parent.parent.parent)

import random
import signal
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.distributed as dist
from typing import Union
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, random_split, Sampler
from torch.utils.data.distributed import DistributedSampler

from config.settings_hpc import *
from multiomic_transformer.datasets.dataset import (
    MultiomicTransformerDataset, MultiChromosomeDataset,
    ChromSubsetBatchSampler, DistributedBatchSampler, fit_simple_scalers,
    SimpleScaler, IndexedChromBucketBatchSampler
    )
from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.utils.files import unique_path
from multiomic_transformer.utils import ewc_utils, plotting

warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group`")

STOP_REQUESTED = False

def _signal_handler(signum, frame):
    # Mark that we should shut down gracefully.
    global STOP_REQUESTED
    STOP_REQUESTED = True

# Register for both Ctrl+C and elastic's terminate
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

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

def load_run_params_from_json(run_dir: Path) -> dict:
    """
    Load run_parameters.json from an existing training directory, if present.
    Returns {} if not found.
    """
    param_path = run_dir / "run_parameters.json"
    if not param_path.is_file():
        logging.warning(f"No run_parameters.json found in {run_dir}, using current config.")
        return {}
    with open(param_path, "r") as f:
        params = json.load(f)
    logging.info(f"Loaded run parameters from {param_path}")
    return params

    
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
        grad_accum_steps: int = 1,
        use_grad_accumulation: bool = False,

    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.corr_sq_warmup_epochs = 3 
        self.stop_requested = False
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_grad_accumulation = use_grad_accumulation
        
        self.scaler = GradScaler(init_scale=1024, growth_factor=1.5, backoff_factor=0.5, growth_interval=200)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode=MODE, 
            factor=SCHEDULER_FACTOR, 
            patience=SCHEDULER_PATIENCE,
            threshold=THRESHOLD,
            threshold_mode=THRESHOLD_MODE,
            cooldown=COOLDOWN,
            min_lr=MIN_LR,
        )
        
        # Early stopping
        self.best_val_loss = float("inf")
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
    
    def _should_stop(self):
        global STOP_REQUESTED
        return self.stop_requested or STOP_REQUESTED
    
    def _save_trained_model(self, path: str):
        if self.gpu_id != 0:
            return

        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model

        model.eval()
        ckpt = {
            "model_state_dict": model.state_dict(),
        }

        if hasattr(self, "tf_scaler") and self.tf_scaler is not None:
            ckpt["tf_scaler_mean"] = self.tf_scaler.mean.detach().cpu()
            ckpt["tf_scaler_std"]  = self.tf_scaler.std.detach().cpu()
        if hasattr(self, "tg_scaler") and self.tg_scaler is not None:
            ckpt["tg_scaler_mean"] = self.tg_scaler.mean.detach().cpu()
            ckpt["tg_scaler_std"]  = self.tg_scaler.std.detach().cpu()

        out_path = os.path.join(path, "trained_model.pt")
        torch.save(ckpt, out_path)
        logging.info(f"Saved trained model to {out_path}")

    def _handle_abort(self, epoch, path, history, reason: str):
        # Only rank 0 writes, but all ranks should sync.
        if self.gpu_id == 0:
            logging.info(f"{reason}: saving checkpoint and logs before exit.")
            last_epoch = max(0, epoch)
            self._save_checkpoint(last_epoch, path)
            self._write_log_csv(history, path)
            self._save_trained_model(path)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _run_batch(self, batch):
        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
        atac_wins = atac_wins.to(self.gpu_id)
        tf_tensor = tf_tensor.to(self.gpu_id)
        targets   = targets.to(self.gpu_id)
        bias      = bias.to(self.gpu_id)
        tf_ids    = tf_ids.to(self.gpu_id)
        tg_ids    = tg_ids.to(self.gpu_id)
        motif_mask= motif_mask.to(self.gpu_id)

        # Optional feature scaling (id-aware)
        if getattr(self, "tf_scaler", None) is not None:
            tf_tensor = self.tf_scaler.transform(tf_tensor, tf_ids)
        if getattr(self, "tg_scaler", None) is not None:
            targets  = self.tg_scaler.transform(targets, tg_ids)
            
        tf_tensor  = torch.nan_to_num(tf_tensor,  nan=0.0, posinf=1e6, neginf=-1e6)
        atac_wins  = torch.nan_to_num(atac_wins,  nan=0.0, posinf=1e6, neginf=-1e6)
        bias       = torch.nan_to_num(bias,       nan=0.0, posinf=5.0, neginf=-5.0)
        motif_mask = torch.nan_to_num(motif_mask, nan=0.0)

        for name, t in {
            "atac_wins": atac_wins, "tf_tensor": tf_tensor, "targets": targets,
            "bias": bias, "motif_mask": motif_mask
        }.items():
            if not torch.isfinite(t).all():
                bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5]
                raise RuntimeError(f"{name} has non-finite values; examples idx={bad}")
        
        with autocast(device_type="cuda"):
            mask_arg = motif_mask if USE_MOTIF_MASK else None
            preds, _ = self.model(
                atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias, motif_mask=mask_arg
            )

        # ---- loss & penalty in fp32 for stability ----
        preds32   = torch.nan_to_num(preds.float(),   nan=0.0, posinf=1e6, neginf=-1e6)
        targets32 = torch.nan_to_num(targets.float(), nan=0.0, posinf=1e6, neginf=-1e6)

        # Mean MSE in scaled space
        mse_loss = self.loss_fn(preds32, targets32)
        
        # ----- Unscaled MSE for logging (no grad) -----
        if getattr(self, "tg_scaler", None) is not None:
            with torch.no_grad():
                targets_u = self.tg_scaler.inverse_transform(targets32, tg_ids)
                preds_u   = self.tg_scaler.inverse_transform(preds32,   tg_ids)

                targets_u = torch.nan_to_num(targets_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)
                preds_u   = torch.nan_to_num(preds_u.float(),   nan=0.0, posinf=1e6, neginf=-1e6)

                mse_loss_unscaled = F.mse_loss(preds_u, targets_u).detach()
        else:
            mse_loss_unscaled = mse_loss.detach()

        # ---------- Correlation penalty via per-gene Pearson r^2 ----------
        # shapes: [B, G]
        x = preds32
        y = targets32
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        eps = 1e-8
        x_ss = (x * x).sum(dim=0)     # per gene
        y_ss = (y * y).sum(dim=0)     # per gene
        denom = (x_ss * y_ss).clamp_min(eps).sqrt()
        # valid genes need variance in both preds and targets
        valid = (x_ss > eps) & (y_ss > eps)

        corr = torch.zeros_like(denom)
        corr[valid] = (x[ :, valid] * y[ :, valid]).sum(dim=0) / denom[valid]
        corr = corr.clamp_(-1.0, 1.0)
        r2_g = corr * corr

        if valid.any():
            mean_r2 = r2_g[valid].mean()
        else:
            mean_r2 = torch.tensor(0.0, device=preds32.device, dtype=preds32.dtype)

        # Warmup the penalty weight over first few epochs
        warm = max(1, getattr(self, "corr_sq_warmup_epochs", 3))
        cur_epoch = getattr(self, "epoch", 0)  # set by your loop before calling _run_batch
        anneal = float(min(1.0, (cur_epoch + 1) / warm))

        r2_penalty = (CORR_LOSS_WEIGHT * anneal) * (1.0 - mean_r2)  # bounded ~[0, 2*lambda]

        total_loss = mse_loss + r2_penalty

        # Safety: if something went numerically wrong, skip the step
        if not torch.isfinite(total_loss):
            return (torch.tensor(0.0, device=self.gpu_id),
                    torch.tensor(0.0, device=self.gpu_id),
                    torch.tensor(0.0, device=self.gpu_id),
                    torch.tensor(0.0, device=self.gpu_id))

        # For logging: return components
        return total_loss, mse_loss.detach(), mse_loss_unscaled, mean_r2.detach(), r2_penalty.detach()

    def _validate(self):
        self.model.eval()

        total_loss_scaled = 0.0
        total_loss_unscaled = 0.0
        n_batches = 0

        # global accumulators (scaled space)
        sse_s   = torch.zeros(1, device=self.gpu_id)
        sumy_s  = torch.zeros(1, device=self.gpu_id)
        sumy2_s = torch.zeros(1, device=self.gpu_id)
        n_s     = torch.zeros(1, device=self.gpu_id)

        # global accumulators (UNSCALED space)
        sse_u   = torch.zeros(1, device=self.gpu_id)
        sumy_u  = torch.zeros(1, device=self.gpu_id)
        sumy2_u = torch.zeros(1, device=self.gpu_id)
        n_u     = torch.zeros(1, device=self.gpu_id)

        with torch.no_grad():
            for batch in self.val_data:
                if self._should_stop():
                    raise KeyboardInterrupt()
                (atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask) = batch

                atac_wins  = atac_wins.to(self.gpu_id, non_blocking=True)
                tf_tensor  = tf_tensor.to(self.gpu_id, non_blocking=True)
                targets    = targets.to(self.gpu_id, non_blocking=True)
                bias       = bias.to(self.gpu_id, non_blocking=True)
                tf_ids     = tf_ids.to(self.gpu_id, non_blocking=True)
                tg_ids     = tg_ids.to(self.gpu_id, non_blocking=True)
                motif_mask = motif_mask.to(self.gpu_id, non_blocking=True)

                # scale inputs/targets as in training (scaled space)
                if getattr(self, "tf_scaler", None) is not None:
                    tf_tensor = self.tf_scaler.transform(tf_tensor, tf_ids)
                if getattr(self, "tg_scaler", None) is not None:
                    targets_s = self.tg_scaler.transform(targets, tg_ids)
                else:
                    targets_s = targets

                mask_arg = motif_mask if USE_MOTIF_MASK else None

                preds, _ = self.model(
                    atac_wins, tf_tensor,
                    tf_ids=tf_ids, tg_ids=tg_ids,
                    bias=bias, motif_mask=mask_arg,
                )

                # numeric safety before metrics
                preds_s   = torch.nan_to_num(preds.float(),   nan=0.0, posinf=1e6, neginf=-1e6)
                targets_s = torch.nan_to_num(targets_s.float(), nan=0.0, posinf=1e6, neginf=-1e6)

                # --- MSE in scaled space (status quo) ---
                loss_s = F.mse_loss(preds_s, targets_s)
                total_loss_scaled += float(loss_s.item())
                n_batches += 1

                # accumulate for scaled R²
                y_s = targets_s.reshape(-1)
                p_s = preds_s.reshape(-1)
                sse_s   += torch.sum((y_s - p_s) ** 2)
                sumy_s  += torch.sum(y_s)
                sumy2_s += torch.sum(y_s ** 2)
                n_s     += y_s.numel()
            

                # ---------- Unscaled metrics ----------
                if getattr(self, "tg_scaler", None) is not None:
                    targets_u = self.tg_scaler.inverse_transform(targets_s, tg_ids)
                    preds_u   = self.tg_scaler.inverse_transform(preds_s,   tg_ids)
                else:
                    targets_u, preds_u = targets_s, preds_s

                targets_u = torch.nan_to_num(targets_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)
                preds_u   = torch.nan_to_num(preds_u.float(),   nan=0.0, posinf=1e6, neginf=-1e6)

                loss_u = F.mse_loss(preds_u, targets_u)
                total_loss_unscaled += float(loss_u.item())
                
                y_u = targets_u.reshape(-1)
                p_u = preds_u.reshape(-1)
                sse_u   += torch.sum((y_u - p_u) ** 2)
                sumy_u  += torch.sum(y_u)
                sumy2_u += torch.sum(y_u ** 2)
                n_u     += y_u.numel()

        if n_batches == 0 or n_s.item() == 0:
            return 0.0, 0.0, 0.0, 0.0

        # DDP all-reduce
        if dist.is_available() and dist.is_initialized():
            for t in (sse_s, sumy_s, sumy2_s, n_s, sse_u, sumy_u, sumy2_u, n_u):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        eps = 1e-12

        # scaled R²
        ybar_s = sumy_s / torch.clamp(n_s, min=1.0)
        sst_s  = sumy2_s - n_s * (ybar_s ** 2)
        r2_s   = torch.where(sst_s <= eps, torch.zeros_like(sst_s), 1.0 - sse_s / torch.clamp(sst_s, min=eps))

        # unscaled R²
        ybar_u = sumy_u / torch.clamp(n_u, min=1.0)
        sst_u  = sumy2_u - n_u * (ybar_u ** 2)
        r2_u   = torch.where(sst_u <= eps, torch.zeros_like(sst_u), 1.0 - sse_u / torch.clamp(sst_u, min=eps))

        avg_loss_scaled = total_loss_scaled / max(1, n_batches)
        avg_loss_unscaled = total_loss_unscaled / max(1, n_batches)


        # Return both: (scaled MSE, scaled R2, unscaled R2)
        return float(avg_loss_scaled), float(avg_loss_unscaled), float(r2_s.item()), float(r2_u.item())

    
    def _run_epoch(self, epoch):
        sampler = getattr(self.train_data, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        bs = getattr(self.train_data, "batch_sampler", None)
        if hasattr(bs, "set_epoch"):
            bs.set_epoch(epoch)
            
        total_loss_sum = 0.0
        total_mse_scaled_sum = 0.0
        total_mse_unscaled_sum = 0.0
        n_batches = 0
        self.epoch = epoch
        
        self.optimizer.zero_grad(set_to_none=True)
        progress_marks = [25, 50, 75]
        next_mark_idx = 0
        
        for iteration, batch in enumerate(self.train_data):
            if self._should_stop():
                raise KeyboardInterrupt()

            out = self._run_batch(batch)
            if out[0] is None:
                self.optimizer.zero_grad(set_to_none=True)
                continue

            (total_loss_val,
             mse_scaled,
             mse_unscaled,
             mean_corr,
             corr_weight) = out

            if not total_loss_val.requires_grad:
                raise RuntimeError("Bug: total_loss_val has no grad_fn")

            loss_for_backprop = total_loss_val / self.grad_accum_steps
            self.scaler.scale(loss_for_backprop).backward()

            if ((iteration + 1) % self.grad_accum_steps == 0
                or (iteration + 1) == len(self.train_data)):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss_sum          += float(total_loss_val.detach())
            total_mse_scaled_sum    += float(mse_scaled)
            total_mse_unscaled_sum  += float(mse_unscaled)
            n_batches += 1
            
            
            # ----- sparse progress logging -----
            if (
                self.gpu_id == 0
                and len(self.train_data) > 0
                and next_mark_idx < len(progress_marks)
            ):
                pct = int(100 * (iteration + 1) / len(self.train_data))
                # log when we *cross* the next mark
                if pct >= progress_marks[next_mark_idx]:
                    logging.info(
                        f"    [{progress_marks[next_mark_idx]}%] Iter {iteration}"
                    )
                    next_mark_idx += 1
                            # f"MSE + pearson r2 penalty: {total_loss_val:.4f} | "
                            # f"MSE Loss: {mse_loss_scaled:.4f} | "
                            # f"pearson r2 penalty: {corr_weight:.4f} | "
                            # f"Mean pearson r2: {mean_corr:.2f}"     

        avg_train_loss          = total_loss_sum / max(1, n_batches)
        avg_train_mse_scaled    = total_mse_scaled_sum / max(1, n_batches)
        avg_train_mse_unscaled  = total_mse_unscaled_sum / max(1, n_batches)

        avg_val_mse_scaled, avg_val_mse_unscaled, r2_s, r2_u = self._validate()
        self.scheduler.step(avg_val_mse_unscaled)

        return (
            avg_train_loss,
            avg_train_mse_scaled,
            avg_train_mse_unscaled,
            avg_val_mse_scaled,
            avg_val_mse_unscaled,
            r2_s,
            r2_u,
        )


    def _save_checkpoint(self, epoch: int, path: str):
        # Only rank 0 writes to disk
        """
        Saves the model, optimizer, and scheduler states to a checkpoint file.

        Only rank 0 writes to disk.

        Args:
            epoch (int): The current epoch number.
            path (str): The path to save the checkpoint file to.

        Notes:
            If the model is a DDP model, we save the wrapped model's state dict.
            If the model is not a DDP model, we save the model's state dict directly.
            If the optimizer or scheduler is not None, we save their state dicts to the checkpoint file.
            If the tf_scaler or tg_scaler is not None, we save their mean and std to the checkpoint file.
        """
        if self.gpu_id != 0:
            return

        # Handle DDP vs non-DDP models
        if hasattr(self.model, "module"):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_state,
        }

        # Optional: save optimizer/scheduler too
        if hasattr(self, "optimizer") and self.optimizer is not None:
            ckpt["optimizer_state_dict"] = self.optimizer.state_dict()
        if hasattr(self, "scheduler") and self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        # ---- Save scalers if present ----
        # Ensure they’re on CPU so checkpoint is portable
        if hasattr(self, "tf_scaler") and self.tf_scaler is not None:
            ckpt["tf_scaler_mean"] = self.tf_scaler.mean.detach().cpu()
            ckpt["tf_scaler_std"]  = self.tf_scaler.std.detach().cpu()

        if hasattr(self, "tg_scaler") and self.tg_scaler is not None:
            ckpt["tg_scaler_mean"] = self.tg_scaler.mean.detach().cpu()
            ckpt["tg_scaler_std"]  = self.tg_scaler.std.detach().cpu()
            


        os.makedirs(path, exist_ok=True)
        out_path = os.path.join(path, f"checkpoint_{epoch}.pt")
        torch.save(ckpt, out_path)
        logging.info(f"\tTraining checkpoint saved to {out_path}")

        
    def train(self, max_epochs: int, path: str, start_epoch: int = 0):
        best_val_loss = float("inf")
        best_r2 = float(0)
        patience_counter = 0
        history = []  # store per-epoch logs
        
        try:
            total_train_start_time = time.time()
            for epoch in range(start_epoch, max_epochs):
                epoch_start_time = time.time()
                if self._should_stop():
                    raise KeyboardInterrupt()
                
                (avg_train_loss,
                 avg_train_mse_scaled, 
                 avg_train_mse_unscaled, 
                 avg_val_mse_scaled, 
                 avg_val_mse_unscaled, 
                 r2_s, 
                 r2_u) = self._run_epoch(epoch)
                epoch_end_time = time.time()
                
                epoch_dur_sec = epoch_end_time - epoch_start_time
                
                if self._should_stop():
                    raise KeyboardInterrupt()

                if self.gpu_id == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    logging.info(
                        f"Epoch {epoch+1} | Train Total Loss: {avg_train_loss:.4f} | "
                        f"Train MSE: {avg_train_mse_unscaled:.4f} | "
                        f"Val MSE: {avg_val_mse_unscaled:.4f} | "
                        f"R2 (Unscaled): {r2_u:.3f} | "
                        f"R2 (Scaled): {r2_s:.3f} | "
                        f"LR: {lr:.2e} | "
                        f"Time: {epoch_dur_sec:.0f}s" 
                    )

                    history.append({
                        "Epoch": epoch+1,
                        "Train Total Loss": avg_train_loss,
                        "Train MSE": avg_train_mse_unscaled,
                        "Val MSE": avg_val_mse_unscaled,
                        "R2_u": r2_u,
                        "R2_s": r2_s,
                        "LR": lr,
                        "Time": round(epoch_dur_sec,0)
                    })
                                    
                # Checkpoint + CSV log
                if epoch % self.save_every == 0:
                    if self.gpu_id == 0:
                        self._save_checkpoint(epoch, path)
                        self._write_log_csv(history, path)
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()

                # Checkpoint + CSV log
                stop_tensor = torch.tensor(0, device=self.gpu_id)

                # --- Early stopping check (only rank 0 sets flag) ---
                if self.gpu_id == 0:
                    if (avg_val_mse_unscaled < best_val_loss - self.min_delta) or (r2_s > best_r2 + self.min_delta):
                        # If either val_loss improved OR r2_s improved, reset patience
                        best_val_loss = avg_val_mse_unscaled
                        best_r2 = max(best_r2, r2_s)
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
            
            total_train_end_time = time.time()
            
            total_training_time_min = total_train_end_time - total_train_start_time
            
            # Final save if not early stopped
            if self.gpu_id == 0 and patience_counter < self.patience:
                self._write_log_csv(history, path)
                logging.info("Training loop exited normally.")
                
                # Convert elapsed_seconds into hours, minutes, and seconds
                hours, remainder = divmod(total_training_time_min, 3600)  # 3600 seconds in an hour
                minutes, seconds = divmod(remainder, 60)         # 60 seconds in a minute
                logging.info(f"Total Training Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

        except KeyboardInterrupt:
            # graceful Ctrl+C
            epoch = locals().get("epoch", start_epoch)
            self._handle_abort(epoch, path, history, "KeyboardInterrupt")
            raise

        except RuntimeError as e:
            # catch CUDA OOM and save before dying
            if "out of memory" in str(e).lower():
                epoch = locals().get("epoch", start_epoch)
                self._handle_abort(epoch, path, history, "CUDA OOM")
            raise
        
    def _write_log_csv(self, history, path):
        fieldnames = ["Epoch", "Train Total Loss", "Train MSE", "Val MSE", "R2_u", "R2_s", "LR", "Time"]
        log_path = os.path.join(path, "training_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

def load_checkpoint(checkpoint_path, trainer, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = trainer.model
    state_dict = ckpt["model_state_dict"]

    # If model is wrapped in DDP, load into .module
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    # Optimizer / scaler if present
    if "optimizer_state_dict" in ckpt and hasattr(trainer, "optimizer"):
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if "scaler_state_dict" in ckpt and hasattr(trainer, "scaler"):
        try:
            trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception:
            # If scaler config changed, you can safely re-init; no hard fail
            pass

    # Next epoch index (we usually store the last finished epoch)
    start_epoch = ckpt.get("epoch", 0) + 1
    return start_epoch



def load_train_objs(run_cfg):
    # Dataset does not depend on d_model etc., but we keep it here
    dataset = MultiChromosomeDataset(
        data_dir=SAMPLE_DATA_CACHE_DIR,
        chrom_ids=CHROM_IDS,
        tf_vocab_path=os.path.join(COMMON_DATA, "tf_vocab.json"),
        tg_vocab_path=os.path.join(COMMON_DATA, "tg_vocab.json"),
        max_cached=2,
        max_tfs=SUBSAMPLE_MAX_TFS,
        max_tgs=SUBSAMPLE_MAX_TGS,
        max_windows_per_chrom=SUBSAMPLE_MAX_WINDOWS_PER_CHROM,
        max_cells=SUBSAMPLE_MAX_CELLS,
        subset_seed=SUBSAMPLE_SEED,
    )

    tf_vocab_size = len(dataset.tf_name2id_sub)
    tg_vocab_size = len(dataset.tg_name2id_sub)

    model = MultiomicTransformer(
        d_model=run_cfg["d_model"],
        num_heads=run_cfg["num_heads"],
        num_layers=run_cfg["num_layers"],
        d_ff=run_cfg["d_ff"],
        dropout=run_cfg["dropout"],
        tf_vocab_size=tf_vocab_size,
        tg_vocab_size=tg_vocab_size,
        bias_scale=ATTN_BIAS_SCALE,
        use_bias=run_cfg["use_dist_bias"],
        use_shortcut=run_cfg["use_shortcut"],
        use_motif_mask=run_cfg["use_motif_mask"],
        motif_mask_threshold=run_cfg["motif_mask_threshold"],
        motif_prior_scale=run_cfg["motif_prior_scale"],
        lambda_l1=run_cfg["shortcut_l1"],
        lambda_l2=run_cfg["shortcut_l2"],
        topk=run_cfg["shortcut_topk"],
        shortcut_dropout=run_cfg["shortcut_dropout"],
        use_gradient_checkpointing=run_cfg["use_grad_ckpt"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=run_cfg["lr"],
    )

    return dataset, model, optimizer

def prepare_dataloader(dataset, batch_size, world_size=1, rank=0,
                       num_workers=4, pin_memory=True, seed=42, drop_last=True):
    """
    Build train/val/test loaders.

    For MultiChromosomeDataset:
      - Use ONE shared dataset instance.
      - For EACH chromosome:
          * split its indices into train/val/test subsets
      - For EACH split:
          * use an IndexedChromBucketBatchSampler over its per-chrom index subsets
          * -> every split sees all chromosomes (by indices),
             but each batch is still single-chromosome (shape-safe).

    For other datasets:
      - Fallback to legacy random_split + DistributedSampler.
    """
    import random
    g = torch.Generator()
    g.manual_seed(seed)

    # ---------- Multi-chromosome path ----------
    if isinstance(dataset, MultiChromosomeDataset):
        # 1) Build per-chrom index ranges from dataset._offsets
        chrom_to_indices = {}
        for i, chrom in enumerate(dataset.chrom_ids):
            start = dataset._offsets[i]
            end = dataset._offsets[i + 1] if i + 1 < len(dataset._offsets) else len(dataset)
            if end > start:
                chrom_to_indices[chrom] = list(range(start, end))

        # 2) For each chrom, split its indices into train/val/test
        train_map = {}
        val_map = {}
        test_map = {}

        for chrom, idxs in chrom_to_indices.items():
            n = len(idxs)
            if n == 0:
                continue

            # deterministic per-chrom shuffle
            rnd = random.Random(seed + hash(chrom) % 10_000_000)
            idxs_shuf = idxs[:]
            rnd.shuffle(idxs_shuf)

            n_train = int(0.70 * n)
            n_val   = int(0.15 * n)
            n_test  = n - n_train - n_val

            # ensure we don't drop everything for tiny chromosomes
            if n_val == 0 and n_train > 1:
                n_val += 1
                n_train -= 1
            if n_test == 0 and n_train > 1:
                n_test += 1
                n_train -= 1

            train_idx = idxs_shuf[:n_train]
            val_idx   = idxs_shuf[n_train:n_train + n_val]
            test_idx  = idxs_shuf[n_train + n_val:]

            if train_idx:
                train_map[chrom] = train_idx
            if val_idx:
                val_map[chrom] = val_idx
            if test_idx:
                test_map[chrom] = test_idx

        # 3) Build base batch samplers (chrom-homogeneous, but all chroms per split)
        base_train_bs = IndexedChromBucketBatchSampler(
            train_map, batch_size=batch_size, shuffle=True, seed=seed
        )
        base_val_bs = IndexedChromBucketBatchSampler(
            val_map, batch_size=batch_size, shuffle=False, seed=seed
        )
        base_test_bs = IndexedChromBucketBatchSampler(
            test_map, batch_size=batch_size, shuffle=False, seed=seed
        )

        # 4) Optionally shard batches across ranks for DDP
        if world_size > 1:
            train_bs = DistributedBatchSampler(base_train_bs, world_size, rank, drop_last=drop_last)
            val_bs   = DistributedBatchSampler(base_val_bs,   world_size, rank, drop_last=False)
            test_bs  = DistributedBatchSampler(base_test_bs,  world_size, rank, drop_last=False)
        else:
            train_bs, val_bs, test_bs = base_train_bs, base_val_bs, base_test_bs

        # 5) Single shared dataset; samplers decide which indices belong to which split
        train_loader = DataLoader(
            dataset,
            batch_sampler=train_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            dataset,
            batch_sampler=val_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            dataset,
            batch_sampler=test_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader

    # ---------- Single-chromosome / legacy path (unchanged) ----------
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

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=MultiomicTransformerDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=MultiomicTransformerDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=MultiomicTransformerDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader

def write_run_parameters(dataset, out_dir):

    logging.info("\n===== MultiomicTransformerDataset Loaded =====")
    logging.info(f"Chromosome:          {CHROM_IDS}")
    logging.info(f"Genes:               {len(dataset.tg_ids)}")
    logging.info(f"Windows (RE):        {dataset.num_windows}")
    logging.info(f"TFs:                 {len(dataset.tf_ids)}")
    logging.info(f"Metacells:           {len(dataset.metacell_names)}")
    logging.info(f"Epochs:              {TOTAL_EPOCHS}")
    logging.info(f"Batch Size:          {BATCH_SIZE}")
    logging.info(f"Grad Accum Steps:    {GRAD_ACCUM_STEPS}")   
    if USE_GRAD_ACCUMULATION:
        logging.info(f"Effctve Batch Size:  {BATCH_SIZE * GRAD_ACCUM_STEPS}")   
    else:
        logging.info(f"Effctve Batch Size:  {BATCH_SIZE}")   
    logging.info(f"Use Grad Accum?:     {USE_GRAD_ACCUMULATION}")   
    logging.info(f"Use Grad Chkpt?:     {USE_GRAD_CHECKPOINTING}") 
    logging.info(f"Model Dimension:     {D_MODEL}")
    logging.info(f"Attention Heads:     {NUM_HEADS}")
    logging.info(f"Attention Layers:    {NUM_LAYERS}")
    logging.info(f"Feedforward Layers:  {D_FF}")
    logging.info(f"Dropout:             {DROPOUT}")
    logging.info(f"TF-TG Shortcut?:     {USE_SHORTCUT}")
    logging.info(f"Dist bias?:          {USE_DISTANCE_BIAS}")
    logging.info(f"Motif Mask?:         {USE_MOTIF_MASK}")
    logging.info(f"Mask Thresh:         {MOTIF_MASK_THRESH}")
    logging.info(f"Mask Soft Scale:     {MOTIF_PRIOR_SCALE}")
    logging.info(f"Shortcut L1:         {SHORTCUT_L1}")
    logging.info(f"Shortcut L2:         {SHORTCUT_L2}")
    logging.info(f"Shortcut Dropout:    {SHORTCUT_DROPOUT}")
    logging.info(f"Shortcut Top K:      {SHORTCUT_TOPK}")
    logging.info("================================================")
    
    run_params = {
        "epochs": TOTAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "use_grad_accum": USE_GRAD_ACCUMULATION,
        "use_grad_ckpt": USE_GRAD_CHECKPOINTING,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "d_ff": D_FF,
        "dropout": DROPOUT,
        "use_shortcut": USE_SHORTCUT,
        "use_dist_bias": USE_DISTANCE_BIAS,
        "use_motif_mask": USE_MOTIF_MASK,
        "motif_mask_threshold": MOTIF_MASK_THRESH,
        "motif_prior_scale": MOTIF_PRIOR_SCALE,
        "shortcut_l1": SHORTCUT_L1,
        "shortcut_l2": SHORTCUT_L2,
        "shortcut_dropout": SHORTCUT_DROPOUT,
        "shortcut_topk": SHORTCUT_TOPK,
        "lr": INITIAL_LEARNING_RATE,
        "genes": len(dataset.tg_ids),
        "windows": dataset.num_windows,
        "tfs": len(dataset.tf_ids),
        "metacells": len(dataset.metacell_names),
    }

    path = os.path.join(out_dir, "run_parameters.json")
    with open(path, "w") as f:
        json.dump(run_params, f, indent=4)  # indent=4 for readability
    logging.info(f"Run parameters written to {path}")

import torch
import torch.distributed as dist
from torch.utils._contextlib import _DecoratorContextManager  # if needed elsewhere


@torch.no_grad()
def get_mean_tf_tg_attention_ddp(model,
                                 dataloader,
                                 num_tfs: int,
                                 num_tgs: int,
                                 device: Union[str, torch.device]):
    """
    Compute global mean TF–TG attention over a (possibly DDP) dataloader.

    Assumes:
      - model.shortcut_layer.attn is set on each forward.
      - attn shape is one of:
          [B, H, G, T], [B, G, T], or [G, T]
        where G = #TG in batch, T = #TF in batch.
      - tf_ids and tg_ids in the batch give TF/TG indices in [0,num_tfs) / [0,num_tgs),
        either shape [B, T] (same across batch) or [T]; similarly for TG.

    Returns:
      mean_attn: [num_tgs, num_tfs] tensor on current device
                 (after all_reduce: identical on all ranks).
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()

    # global accumulators on this rank
    attn_sum = torch.zeros(num_tgs, num_tfs, device=device)
    attn_cnt = torch.zeros(num_tgs, num_tfs, device=device)

    for batch in dataloader:
        atac_wins, tf_tensor, _, bias, tf_ids, tg_ids, motif_mask = batch

        atac_wins  = atac_wins.to(device, non_blocking=True)
        tf_tensor  = tf_tensor.to(device, non_blocking=True)
        bias       = bias.to(device, non_blocking=True)
        tf_ids     = tf_ids.to(device, non_blocking=True)
        tg_ids     = tg_ids.to(device, non_blocking=True)
        motif_mask = motif_mask.to(device, non_blocking=True)

        # forward pass (no grad)
        _ = model(
            atac_wins,
            tf_tensor,
            tf_ids=tf_ids,
            tg_ids=tg_ids,
            bias=bias,
            motif_mask=motif_mask,
        )

        if not hasattr(model, "shortcut_layer"):
            raise RuntimeError("Model has no shortcut_layer; cannot read TF–TG attention.")

        if not hasattr(model.shortcut_layer, "attn"):
            # no attention recorded for this batch; skip
            continue

        attn = model.shortcut_layer.attn
        if attn is None:
            continue
        attn = attn.detach()

        # ---- Collapse attn to [G, T] ----
        if attn.dim() == 4:
            # [B, H, G, T] -> average over batch + heads
            attn_mat = attn.mean(dim=(0, 1))
        elif attn.dim() == 3:
            # [B, G, T] -> average over batch
            attn_mat = attn.mean(dim=0)
        elif attn.dim() == 2:
            # [G, T]
            attn_mat = attn
        else:
            raise RuntimeError(f"Unexpected attn dim {attn.shape}")

        G_b, T_b = attn_mat.shape

        # ---- Get TF/TG ids for this batch ----
        # If [B, T] with same row, take first; if [T], use as is.
        if tf_ids.dim() == 2:
            tf_ids_b = tf_ids[0]
        else:
            tf_ids_b = tf_ids
        if tg_ids.dim() == 2:
            tg_ids_b = tg_ids[0]
        else:
            tg_ids_b = tg_ids

        if tf_ids_b.numel() != T_b or tg_ids_b.numel() != G_b:
            raise RuntimeError(
                f"Mismatch: attn_mat {attn_mat.shape}, "
                f"tf_ids {tf_ids_b.shape}, tg_ids {tg_ids_b.shape}"
            )

        # ---- Scatter into global [num_tgs, num_tfs] ----
        # Build all (tg_idx, tf_idx) pairs for this block
        tf_idx = tf_ids_b.view(1, -1).expand(G_b, -1).reshape(-1)   # [G_b*T_b]
        tg_idx = tg_ids_b.view(-1, 1).expand(-1, T_b).reshape(-1)   # [G_b*T_b]
        vals   = attn_mat.reshape(-1)

        # Safety: keep only in-range indices
        valid = (tf_idx >= 0) & (tf_idx < num_tfs) & (tg_idx >= 0) & (tg_idx < num_tgs)
        if not valid.all():
            tf_idx = tf_idx[valid]
            tg_idx = tg_idx[valid]
            vals   = vals[valid]

        one = torch.ones_like(vals, device=device)

        attn_sum.index_put_((tg_idx, tf_idx), vals, accumulate=True)
        attn_cnt.index_put_((tg_idx, tf_idx), one,  accumulate=True)

    # ---- All-reduce across ranks ----
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(attn_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(attn_cnt, op=dist.ReduceOp.SUM)

    mean_attn = attn_sum / attn_cnt.clamp_min(1e-8)
    return mean_attn


def _mapping_to_ordered_list(name2id: dict):
    # convert {name: id} → [names] in id order
    return [k for k, _ in sorted(name2id.items(), key=lambda kv: kv[1])]

def write_experiment_settings_and_objects(training_output_dir: Path, dataset, test_loader):
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
    
    # Persist test loader
    torch.save(test_loader, os.path.join(training_output_dir, "test_loader.pt"))

    # Your existing run-parameter writer is fine to call here if it doesn’t assume single-chrom only
    write_run_parameters(dataset, training_output_dir)
    logging.info("Wrote experiment settings and objects to training output directory")
    
def save_tf_tg_embeddings_from_model(model, out_dir, vocab_dir):
    model = getattr(model, "module", model)  # unwrap DDP if needed
    torch.save(
        {
            "tf_emb":     model.tf_identity_emb.weight.detach().cpu(),   # [T, D]
            "tg_query_emb":     model.tg_query_emb.weight.detach().cpu(),   # [G, D]
            "tg_emb": model.tg_identity_emb.weight.detach().cpu()
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
    
    resume_ckpt = RESUME_CHECKPOINT_PATH
    
    try:
        if resume_ckpt and os.path.isfile(resume_ckpt):
            resume_ckpt = Path(resume_ckpt)
            training_output_dir = resume_ckpt.parent
            logging.info(f"\n =========== RESUMING FROM {resume_ckpt} ===========")

            # Load previous run parameters (if present)
            prev_params = load_run_params_from_json(training_output_dir)

            def g(key, default):
                return prev_params.get(key, default) if prev_params else default

            run_cfg = {
                "epochs":               g("epochs", TOTAL_EPOCHS),
                "batch_size":           g("batch_size", BATCH_SIZE),
                "grad_accum_steps":     g("grad_accum_steps", GRAD_ACCUM_STEPS),
                "d_model":              g("d_model", D_MODEL),
                "num_layers":           g("num_layers", NUM_LAYERS),
                "num_heads":            g("num_heads", NUM_HEADS),
                "d_ff":                 g("d_ff", D_FF),
                "dropout":              g("dropout", DROPOUT),
                "use_grad_ckpt":        g("use_grad_ckpt", USE_GRAD_CHECKPOINTING),
                "use_shortcut":         g("use_shortcut", USE_SHORTCUT),
                "use_dist_bias":        g("use_dist_bias", USE_DISTANCE_BIAS),
                "use_motif_mask":       g("use_motif_mask", USE_MOTIF_MASK),
                "motif_mask_threshold": g("motif_mask_threshold", MOTIF_MASK_THRESH),
                "motif_prior_scale":    g("motif_prior_scale", MOTIF_PRIOR_SCALE),
                "shortcut_l1":          g("shortcut_l1", SHORTCUT_L1),
                "shortcut_l2":          g("shortcut_l2", SHORTCUT_L2),
                "shortcut_topk":        g("shortcut_topk", SHORTCUT_TOPK),
                "shortcut_dropout":     g("shortcut_dropout", SHORTCUT_DROPOUT),
                "lr":                   g("lr", INITIAL_LEARNING_RATE),
            }

        else:
            # New experiment
            training_file_iter_format = "model_training_{:03d}"
            training_output_dir = unique_path(OUTPUT_DIR / CHROM_ID, training_file_iter_format)
            logging.info(f"\n =========== EXPERIMENT {training_output_dir.name.upper()} ===========")

            run_cfg = {
                "epochs":               TOTAL_EPOCHS,
                "batch_size":           BATCH_SIZE,
                "grad_accum_steps":     GRAD_ACCUM_STEPS,
                "d_model":              D_MODEL,
                "num_layers":           NUM_LAYERS,
                "num_heads":            NUM_HEADS,
                "d_ff":                 D_FF,
                "dropout":              DROPOUT,
                "use_grad_ckpt":        USE_GRAD_CHECKPOINTING,
                "use_shortcut":         USE_SHORTCUT,
                "use_dist_bias":        USE_DISTANCE_BIAS,
                "use_motif_mask":       USE_MOTIF_MASK,
                "motif_mask_threshold": MOTIF_MASK_THRESH,
                "motif_prior_scale":    MOTIF_PRIOR_SCALE,
                "shortcut_l1":          SHORTCUT_L1,
                "shortcut_l2":          SHORTCUT_L2,
                "shortcut_topk":        SHORTCUT_TOPK,
                "shortcut_dropout":     SHORTCUT_DROPOUT,
                "lr":                   INITIAL_LEARNING_RATE,
            }

        os.makedirs(training_output_dir, exist_ok=True)

        dataset, model, optimizer = load_train_objs(run_cfg)

        if rank == 0:
            logging.info("Preparing dataloader")

        train_loader, val_loader, test_loader = prepare_dataloader(
            dataset,
            batch_size=run_cfg["batch_size"],
            world_size=world_size,
            rank=rank,
        )
        
        if rank == 0 and not (resume_ckpt and os.path.isfile(resume_ckpt)):
            write_experiment_settings_and_objects(training_output_dir, dataset, test_loader)
            logging.info("Wrote experiment settings and objects to training output directory")

        if rank == 0:
            logging.info("Creating Trainer")

        loss_fn = nn.MSELoss()
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            gpu_id=rank,
            save_every=SAVE_EVERY_N_EPOCHS,
            patience=PATIENCE,
            grad_accum_steps=run_cfg["grad_accum_steps"],
            use_grad_accumulation=USE_GRAD_ACCUMULATION,
        )

        rank_device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        # ---- Resume or new scalers ----
        if resume_ckpt and os.path.isfile(resume_ckpt):
            start_epoch = load_checkpoint(str(resume_ckpt), trainer, rank_device)
            logging.info(f"Resuming training from epoch {start_epoch}")
        else:
            T = len(dataset.tf_name2id_sub)
            G = len(dataset.tg_name2id_sub)
            use_ddp_reduce = torch.distributed.is_initialized()

            tf_s, tg_s = fit_simple_scalers(
                train_loader,
                T_expected=T,
                G_expected=G,
                device_for_reduce=rank_device,
                use_ddp_reduce=use_ddp_reduce,
            )
            trainer.tf_scaler = SimpleScaler(tf_s.mean.to(rank_device), tf_s.std.to(rank_device))
            trainer.tg_scaler = SimpleScaler(tg_s.mean.to(rank_device), tg_s.std.to(rank_device))
            start_epoch = 0

        if rank == 0:
            logging.info("\n ----- TRAINING STARTED -----")

        trainer.train(
            max_epochs=TOTAL_EPOCHS,
            path=training_output_dir,
            start_epoch=start_epoch,
        )

        # ---------- Post-training: unwrap, save, etc. ----------
        model_for_eval = getattr(trainer.model, "module", trainer.model).to(rank_device)

        if rank == 0:
            model_for_eval.eval()
            # embeddings
            save_tf_tg_embeddings_from_model(
                model_for_eval,
                out_dir=training_output_dir,
                vocab_dir=training_output_dir,
            )
            # final checkpoint
            torch.save(
                {
                    "epoch": TOTAL_EPOCHS - 1,
                    "model_state_dict": model_for_eval.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "scheduler_state_dict": trainer.scheduler.state_dict(),
                    "best_val_loss": trainer.best_val_loss,
                    "tf_scaler_mean": trainer.tf_scaler.mean,
                    "tf_scaler_std": trainer.tf_scaler.std,
                    "tg_scaler_mean": trainer.tg_scaler.mean,
                    "tg_scaler_std": trainer.tg_scaler.std,
                },
                training_output_dir / "trained_model.pt",
            )
            logging.info("Saved final trained model")

        # ----- EWC fisher (rank 0 only) -----
        if rank == 0:
            ewc_bundle_path = training_output_dir / "ewc_bundle.pth"
            fisher = ewc_utils.compute_fisher_diag(
                model_for_eval, train_loader,
                device=rank_device,
                n_batches=100,
            )
            ewc_utils.save_ewc_bundle(ewc_bundle_path, model_for_eval, fisher)

        # ----- Global mean TF–TG attention via DDP all-reduce -----
        mean_attn = None
        if hasattr(model_for_eval, "shortcut_layer"):
            mean_attn = get_mean_tf_tg_attention_ddp(
                model_for_eval,
                test_loader,
                num_tfs=T,
                num_tgs=G,
                device=rank_device,
            )

            if rank == 0:
                mean_attn_cpu = mean_attn.detach().cpu()
                torch.save(
                    mean_attn_cpu,
                    training_output_dir / "mean_tf_tg_attention.pt"
                )
                logging.info("Saved global mean TF–TG attention matrix")

        # ----- Rank-0: save attention CSV + plots -----
        if rank == 0:
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

            if mean_attn is not None:
                mean_attn_df = pd.DataFrame(
                    mean_attn.detach().cpu().numpy(),
                    index=tg_id2name,
                    columns=tf_id2name,
                )
                mean_attn_df.to_csv(os.path.join(training_output_dir, "tf_tg_mean_attention.csv"))
                logging.info("Saved TF→TG attention weights to 'tf_tg_mean_attention.csv'")

            # Training figures
            log_path = os.path.join(training_output_dir, "training_log.csv")

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
        save_every=SAVE_EVERY_N_EPOCHS,
        total_epochs=TOTAL_EPOCHS,
        batch_size=BATCH_SIZE)