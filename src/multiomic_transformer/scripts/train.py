import csv
import json
import logging
import os
import sys
import warnings
import random
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import re
from typing import Optional
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
    MultiomicTransformerDataset,
    MultiChromosomeDataset,
    ChromBucketBatchSampler,
    DistributedBatchSampler
)
from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.utils.files import unique_path
from multiomic_transformer.utils import ewc_utils, plotting
import signal
import threading

# global, process-local flag toggled by signal handler
STOP_REQUESTED = threading.Event()

def _signal_handler(signum, frame):
    STOP_REQUESTED.set()

warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group`")

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    
    local_rank = int(os.environ["LOCAL_RANK"])
    use_cuda = torch.cuda.is_available()
    
    if use_cuda:
        logging.info("Running training on GPU")
        torch.cuda.set_device(local_rank)
        # Guard Flash SDP so CPU-only runs don't crash
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except Exception as e:
            logging.warning(f"Flash SDP not available/enabled: {e}")
    else:
        logging.info("Running training on CPU")

    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend, init_method="env://",
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
    
def _debug_multichrom_stats(mcd):
    if not isinstance(mcd, MultiChromosomeDataset):
        logging.info("Not multi-chrom; skip debug.")
        return
    logging.info("=== MultiChrom Debug ===")
    total_windows = 0
    per_chr = []
    uniq_tf, uniq_tg, uniq_metacells = set(), set(), set()
    for cid in mcd.chrom_ids:
        ds = mcd._load_chrom(cid)
        per_chr.append({
            "chrom": cid,
            "tg_count": len(ds.tg_ids),
            "tf_count": len(ds.tf_ids),
            "metacells": len(getattr(ds, "metacell_names", [])),
            "windows": int(ds.num_windows),
            "tg_sample": list(map(str, list(ds.tg_ids)[:5])),
        })
        uniq_tf.update(list(ds.tf_ids))
        uniq_tg.update(list(ds.tg_ids))
        uniq_metacells.update(getattr(ds, "metacell_names", []))
        total_windows += int(ds.num_windows)
    logging.info(f"Per-chrom stats: {per_chr}")
    logging.info(f"Union TG={len(uniq_tg)}, TF={len(uniq_tf)}, Metacells={len(uniq_metacells)}, Windows sum={total_windows}")

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
    
def save_training_state(trainer, epoch, path, filename="checkpoint.pt"):
    model = getattr(trainer.model, "module", trainer.model)
    state = {
        "epoch": epoch + 1,  # resume from the *next* epoch
        "model": model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "history": trainer.history,
        "best_val_loss": trainer.best_val_loss,
        "best_pearson": trainer.best_pearson,
        "patience_counter": trainer.patience_counter,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(state, os.path.join(path, filename))
    
def load_training_state(trainer, path, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model = getattr(trainer.model, "module", trainer.model)

    # Backward-compat: if it’s just a raw state_dict (older runs)
    if isinstance(state, dict) and "model" not in state:
        model.load_state_dict(state, strict=False)
        return 0

    # Full state resume
    model.load_state_dict(state["model"], strict=False)
    trainer.optimizer.load_state_dict(state["optimizer"])
    if "scaler" in state and state["scaler"]:
        trainer.scaler.load_state_dict(state["scaler"])
    if "scheduler" in state and state["scheduler"]:
        trainer.scheduler.load_state_dict(state["scheduler"])

    trainer.history = state.get("history", [])
    trainer.best_val_loss = state.get("best_val_loss", float("inf"))
    trainer.best_pearson = state.get("best_pearson", 0.0)
    trainer.patience_counter = state.get("patience_counter", 0)

    # RNG (nice-to-have for exact reproducibility)
    rng = state.get("rng")
    if rng:
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda"):
            for dev_idx, dev_state in enumerate(rng["cuda"]):
                torch.cuda.set_rng_state(dev_state, device=dev_idx)
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    return int(state.get("epoch", 0))

def _pick_ckpt_in_dir(run_dir: Path) -> Optional[Path]:
    # preference order: best -> trained -> checkpoint
    for name in ("best_checkpoint.pt", "trained_model.pt", "checkpoint.pt"):
        p = run_dir / name
        if p.exists():
            return p
    return None

def _latest_run_dir(out_root: Path, prefix: str) -> Optional[Path]:
    # matches e.g. "model_training_003"
    rx = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    best = None
    best_n = -1
    if not out_root.exists():
        return None
    for p in out_root.iterdir():
        if not p.is_dir():
            continue
        m = rx.match(p.name)
        if m:
            n = int(m.group(1))
            if n > best_n:
                best_n, best = n, p
    return best

def resolve_run_and_resume(out_root: Path, name_pattern: str):
    """
    Returns (training_output_dir: Path, resume_path: Optional[Path], inplace: bool)
    - If RESUME_FROM points to a file: resume *in place* in its parent
    - If RESUME_FROM points to a dir: pick best checkpoint in it and resume *in place*
    - Else if RESUME=1: pick latest run dir under out_root and resume *in place*
    - Else: start a *new run dir* (warm start only if RESUME_FROM points to a file AND RESUME_INPLACE=0)
    """
    env_resume_from = os.getenv("RESUME_FROM")
    env_resume = os.getenv("RESUME", "0") == "1"
    env_inplace = os.getenv("RESUME_INPLACE", "1") == "1"  # default to in-place

    # Case A: explicit RESUME_FROM
    if env_resume_from:
        p = Path(env_resume_from)
        if p.is_file():
            ckpt = p
            run_dir = p.parent if env_inplace else unique_path(out_root, name_pattern)
            return run_dir, ckpt, env_inplace
        if p.is_dir():
            ckpt = _pick_ckpt_in_dir(p)
            if ckpt is None:
                return p, None, True
            return p, ckpt, True  # dir implies in-place

    # Case B: RESUME=1 without a path: use latest run dir under out_root
    if env_resume:
        latest = _latest_run_dir(out_root, name_pattern.split("{")[0])
        if latest:
            ckpt = _pick_ckpt_in_dir(latest)
            return latest, ckpt, True

    # Case C: fresh run (no resume)
    return unique_path(out_root, name_pattern), None, False

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
        self.history = []
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
        dev = self.gpu_id

        # Move + basic sanitization
        atac_wins = atac_wins.to(dev, non_blocking=True)
        tf_tensor = tf_tensor.to(dev, non_blocking=True)
        targets   = targets.to(dev, non_blocking=True)
        bias      = bias.to(dev, non_blocking=True)
        tf_ids    = tf_ids.to(dev, non_blocking=True)
        tg_ids    = tg_ids.to(dev, non_blocking=True)
        motif_mask= motif_mask.to(dev, non_blocking=True)

        # Prevent softmax pathologies from extreme negatives/NaNs in bias
        bias = torch.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0)
        # Optional: clamp too-negative bias to avoid exp overflow: bias = bias.clamp(min=-20.0)

        self.optimizer.zero_grad(set_to_none=True)

        from torch.amp import autocast
        with autocast(device_type="cuda", enabled=True):
            mask_arg = motif_mask if USE_MOTIF_MASK else None
            preds, _ = self.model(
                atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias, motif_mask=mask_arg
            )
            mse_loss = self.loss_fn(preds, targets)

        # --------- Cross-rank "step OK?" decision ---------
        # Check local finiteness
        local_ok = torch.isfinite(preds).all() & torch.isfinite(mse_loss)
        # Reduce across ranks: all must be OK to proceed
        if torch.distributed.is_initialized():
            ok_tensor = local_ok.to(torch.int32)
            torch.distributed.all_reduce(ok_tensor, op=torch.distributed.ReduceOp.MIN)
            step_ok = bool(ok_tensor.item())
        else:
            step_ok = bool(local_ok)

        if not step_ok:
            # Everyone agrees to skip this step: no backward, no optimizer step
            if self.gpu_id == 0:
                logging.warning("Skipping step due to non-finite forward on at least one rank.")
            # Return benign tensors so your outer MSE accumulator stays finite
            # (preds==targets => zero contribution to SSE)
            return torch.tensor(0.0, device=dev), targets.detach(), targets.detach()

        # -------- Pearson (sample-wise across genes) in fp32 WITH grad --------
        P = preds.float()   # keep graph
        T = targets.float() # constant

        Pm = P - P.mean(dim=1, keepdim=True)
        Tm = T - T.mean(dim=1, keepdim=True)
        denom = (Pm.pow(2).sum(dim=1).sqrt() * Tm.pow(2).sum(dim=1).sqrt()).clamp_min(1e-8)
        r = (Pm * Tm).sum(dim=1) / denom
        r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        pearson_loss = 1.0 - r.mean()  # in [0, 2]

        corr_term = CORR_LOSS_WEIGHT * pearson_loss
        total_loss = mse_loss + corr_term

        # Backward + step (identical on all ranks because step_ok is global)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return total_loss.detach(), preds, targets


    
    def _run_val_batch(self, batch):
        """
        Runs a single validation batch.
        Returns:
        loss  : scalar tensor (MSE for this batch)
        preds : [B, G] tensor on current device
        targets: [B, G] tensor on current device
        """
        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch

        device = next(self.model.parameters()).device
        atac_wins = atac_wins.to(device, non_blocking=True)
        tf_tensor = tf_tensor.to(device, non_blocking=True)
        targets   = targets.to(device, non_blocking=True)
        bias      = bias.to(device, non_blocking=True)
        tf_ids    = tf_ids.to(device, non_blocking=True)
        tg_ids    = tg_ids.to(device, non_blocking=True)
        motif_mask= motif_mask.to(device, non_blocking=True)

        mask_arg = motif_mask if USE_MOTIF_MASK else None

        # forward (no grad outside)
        preds, _ = self.model(
            atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias, motif_mask=mask_arg
        )

        loss = F.mse_loss(preds, targets)
        return loss, preds, targets

    def _validate(self):
        # empty/disabled loader
        if self.val_data is None or (hasattr(self.val_data, "__len__") and len(self.val_data) == 0):
            logging.warning("Validation loader is None/empty; skipping validation.")
            return float("nan"), float("nan"), float("nan"), float("nan")

        self.model.eval()
        device = next(self.model.parameters()).device

        loss_sum, n_batches = 0.0, 0
        sse_local = torch.tensor(0.0, device=device, dtype=torch.float64)
        n_local   = torch.tensor(0,   device=device, dtype=torch.long)

        # Per-sample Pearson and Spearman accumulators (macro-averages)
        pearson_sum_local  = torch.tensor(0.0, device=device, dtype=torch.float64)
        pearson_cnt_local  = torch.tensor(0,   device=device, dtype=torch.long)
        spearman_sum_local = torch.tensor(0.0, device=device, dtype=torch.float64)
        spearman_cnt_local = torch.tensor(0,   device=device, dtype=torch.long)

        with torch.no_grad():
            for batch in self.val_data:
                v_loss, v_preds, v_targets = self._run_val_batch(batch)
                loss_sum += float(v_loss); n_batches += 1

                diff = (v_preds - v_targets).to(device)
                sse_local += (diff * diff).sum(dtype=torch.float64)
                n_local   += diff.numel()

                # -------- Per-sample Pearson across genes (vectorized, GPU) --------
                P = v_preds.detach().to(torch.float32)
                T = v_targets.detach().to(torch.float32)
                Pm = P - P.mean(dim=1, keepdim=True)
                Tm = T - T.mean(dim=1, keepdim=True)
                denom = (Pm.pow(2).sum(dim=1).sqrt() * Tm.pow(2).sum(dim=1).sqrt()).clamp_min(1e-8)
                corr = (Pm * Tm).sum(dim=1) / denom  # [B]
                mask = torch.isfinite(corr)
                pearson_sum_local += corr[mask].to(torch.float64).sum()
                pearson_cnt_local += mask.sum()

                # -------- Per-sample Spearman across genes (CPU loop via scipy) --------
                # This is robust to variable G across batches/chromosomes.
                P_np = P.detach().cpu().numpy()
                T_np = T.detach().cpu().numpy()
                from scipy.stats import spearmanr
                s_sum = 0.0
                s_cnt = 0
                for b in range(P_np.shape[0]):
                    pv, tv = P_np[b], T_np[b]
                    # Skip degenerate rows (constant vectors => undefined correlation)
                    if np.std(pv) <= 1e-8 or np.std(tv) <= 1e-8:
                        continue
                    res = spearmanr(pv, tv)
                    sp = res.statistic if hasattr(res, "statistic") else res[0]
                    if np.isfinite(sp):
                        s_sum += float(sp); s_cnt += 1
                if s_cnt:
                    spearman_sum_local += torch.tensor(s_sum, device=device, dtype=torch.float64)
                    spearman_cnt_local += torch.tensor(s_cnt, device=device, dtype=torch.long)

        import torch.distributed as dist
        if dist.is_initialized():
            vec = torch.stack([
                sse_local, n_local.to(torch.float64),
                pearson_sum_local, pearson_cnt_local.to(torch.float64),
                spearman_sum_local, spearman_cnt_local.to(torch.float64),
            ])
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
            sse_g, n_g, p_sum, p_cnt, s_sum, s_cnt = vec.tolist()
            val_mse       = (sse_g / n_g) if n_g > 0 else float("nan")
            pearson_corr  = (p_sum / p_cnt) if p_cnt > 0 else float("nan")
            spearman_corr = (s_sum / s_cnt) if s_cnt > 0 else float("nan")
        else:
            sse_g, n_g = sse_local.item(), int(n_local.item())
            val_mse = (sse_g / n_g) if n_g > 0 else float("nan")
            p_sum, p_cnt = pearson_sum_local.item(), int(pearson_cnt_local.item())
            s_sum, s_cnt = spearman_sum_local.item(), int(spearman_cnt_local.item())
            pearson_corr  = (p_sum / p_cnt) if p_cnt > 0 else float("nan")
            spearman_corr = (s_sum / s_cnt) if s_cnt > 0 else float("nan")

        avg_loss = loss_sum / max(1, n_batches)
        return avg_loss, val_mse, pearson_corr, spearman_corr


    
    def _run_epoch(self, epoch: int):
        # Epoch-sync the batch sampler used by the loader
        bs = getattr(self.train_data, "batch_sampler", None)
        if bs is not None and hasattr(bs, "set_epoch"):
            bs.set_epoch(epoch)
        else:
            s = getattr(self.train_data, "sampler", None)
            if s is not None and hasattr(s, "set_epoch"):
                s.set_epoch(epoch)

        # Re-seed RNGs (CPU/NumPy/Python/CUDA) per epoch
        base_seed = 1234
        torch.manual_seed(base_seed + epoch)
        np.random.seed(base_seed + epoch)
        random.seed(base_seed + epoch)
        torch.cuda.manual_seed_all(base_seed + epoch)

        # ---------------- TRAIN ----------------
        self.model.train()

        # Accumulate true train MSE: sum of squared error + element count
        device = next(self.model.parameters()).device
        sse_local = torch.tensor(0.0, device=device)
        n_local   = torch.tensor(0,   device=device, dtype=torch.long)

        total_loss = 0.0
        n_batches  = 0

        for batch in self.train_data:
            if STOP_REQUESTED.is_set():
                break

            # _run_batch should return the *opt step loss* (with regs) and pure MSE for logging
            total_loss_val, preds, targets = self._run_batch(batch)  # change to return preds/targets

            # accumulate optimizer loss for logging
            total_loss += float(total_loss_val)
            n_batches  += 1

            # accumulate MSE consistently (per-element)
            with torch.no_grad():
                diff = preds.detach() - targets.detach()
                sse_local += (diff * diff).sum()
                n_local   += diff.numel()

        # DDP reduce to get global train MSE
        vec = torch.stack([sse_local, n_local.to(torch.float64)])
        torch.distributed.all_reduce(vec, op=torch.distributed.ReduceOp.SUM)
        train_mse = (vec[0] / vec[1]).item() if vec[1] > 0 else 0.0
        avg_train_loss = total_loss / max(1, n_batches)

        # ---------------- VALIDATION ----------------
        torch.distributed.barrier()  # ensure all ranks finished training loop

        # Compute validation loss and correlations
        avg_val_loss, val_mse, pearson_corr, spearman_corr = self._validate()

        # ---------------- LR SCHEDULER ----------------
        # Ensure all ranks step with the same value (broadcast from rank 0)
        rank = torch.distributed.get_rank()
        val_loss_tensor = torch.tensor([avg_val_loss], device=device, dtype=torch.float32)
        torch.distributed.broadcast(val_loss_tensor, src=0)
        self.scheduler.step(val_loss_tensor.item())

        return avg_train_loss, train_mse, val_mse, pearson_corr, spearman_corr

    def _save_checkpoint(self, epoch, path):
        save_training_state(self, epoch, path, filename="checkpoint.pt")
        if self.gpu_id == 0:
            logging.info(f"\tTraining checkpoint saved")
        
    def train(self, max_epochs: int, path: str, start_epoch: int = 0):
        self.best_val_loss = getattr(self, "best_val_loss", float("inf"))
        self.best_pearson = getattr(self, "best_pearson", 0.0)
        self.patience_counter = getattr(self, "patience_counter", 0)

        for epoch in range(start_epoch, max_epochs):
            # fast exit if a signal was already received before epoch starts
            if STOP_REQUESTED.is_set():
                if self.gpu_id == 0:
                    logging.info("Interrupt received before epoch starts. Saving and exiting...")
                    self._save_checkpoint(epoch, path)
                    self._write_log_csv(self.history, path)
                # broadcast stop to keep all ranks in sync
                if dist.is_initialized():
                    stop_tensor = torch.tensor(1, device=self.gpu_id)
                    dist.broadcast(stop_tensor, src=0)
                break

            try:
                train_loss, train_mse, val_loss, pearson_corr, spearman_corr = self._run_epoch(epoch)
            except KeyboardInterrupt:
                # local catch just in case; prefer the signal path but handle both
                if self.gpu_id == 0:
                    logging.info("KeyboardInterrupt caught during epoch. Saving and exiting...")
                    self._save_checkpoint(epoch, path)
                    self._write_log_csv(self.history, path)
                if dist.is_initialized():
                    stop_tensor = torch.tensor(1, device=self.gpu_id)
                    dist.broadcast(stop_tensor, src=0)
                break

            if self.gpu_id == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logging.info(
                    f"Epoch {epoch+1} | Train Total Loss: {train_loss:.4f} | "
                    f"Train MSE: {train_mse:.4f} | "
                    f"Val MSE: {val_loss:.4f} | "
                    f"Pearson: {pearson_corr:.3f} | Spearman: {spearman_corr:.3f} | "
                    f"LR: {lr:.2e}"
                )
                self.history.append({
                    "Epoch": epoch+1,
                    "Train Total Loss": train_loss,
                    "Train MSE": train_mse,
                    "Val MSE": val_loss,
                    "Pearson": pearson_corr,
                    "Spearman": spearman_corr,
                    "LR": lr,
                })

            if (epoch + 1) % self.save_every == 0:
                if self.gpu_id == 0:
                    self._save_checkpoint(epoch, path)
                    self._write_log_csv(self.history, path)
                if dist.is_initialized():
                    dist.barrier()

            # build stop flag once per epoch (early stop OR user interrupt)
            stop_tensor = torch.tensor(0, device=self.gpu_id)

            if self.gpu_id == 0:
                # Check if the user requested an interrupt
                if STOP_REQUESTED.is_set():
                    logging.info("Interrupt requested. Saving and stopping...")
                    self._save_checkpoint(epoch, path)
                    self._write_log_csv(self.history, path)
                    stop_tensor.fill_(1)
                else:
                    if epoch > 10: # Wait some time before checking if loss improves
                        # Early stopping logic
                        val_improved = (val_loss < self.best_val_loss - self.min_delta)
                        pearson_improved = (pearson_corr > self.best_pearson + self.min_delta)

                        if val_improved or pearson_improved:
                            if val_improved:
                                self.best_val_loss = val_loss
                            if pearson_improved:
                                self.best_pearson = pearson_corr
                            self.patience_counter = 0
                        else:
                            self.patience_counter += 1
                            if self.patience_counter >= self.patience:
                                logging.info("Early stopping triggered.")
                                self._save_checkpoint(epoch, path)
                                self._write_log_csv(self.history, path)
                                stop_tensor.fill_(1)
                            else:
                                logging.info(f"    Loss did not improve {self.patience_counter}/{self.patience}")

            if dist.is_initialized():
                dist.broadcast(stop_tensor, src=0)

            if stop_tensor.item() == 1:
                if self.gpu_id == 0:
                    logging.info("All ranks stopping training.")
                break

        # Final save if not early stopped or interrupted
        if self.gpu_id == 0 and not STOP_REQUESTED.is_set() and self.patience_counter < self.patience:
            self._write_log_csv(self.history, path)
            logging.info("Training loop exited normally.")
    
    def _write_log_csv(self, history, path):
        fieldnames = ["Epoch", "Train Total Loss", "Train MSE", "Val MSE", "Pearson", "Spearman", "LR"]
        log_path = os.path.join(path, "training_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
            

def load_train_objs():
    """
    Creates either a single-chromosome dataset (legacy) or a MultiChromosomeDataset.
    Also returns model and optimizer.
    """
    tf_vocab_path = os.path.join(COMMON_DATA, "tf_vocab.json")
    tg_vocab_path = os.path.join(COMMON_DATA, "tg_vocab.json")

    # Prefer a list of chromosomes if provided in settings; otherwise, fall back to CHROM_ID.
    chrom_ids = None
    try:
        chrom_ids = CHROM_IDS
        if isinstance(chrom_ids, str):
            chrom_ids = [chrom_ids]
    except NameError:
        pass

    if chrom_ids and len(chrom_ids) > 1:
        # --- Multi-chrom mode ---
        dataset = MultiChromosomeDataset(
            data_dir=SAMPLE_DATA_CACHE_DIR,
            chrom_ids=chrom_ids,
            tf_vocab_path=tf_vocab_path,
            tg_vocab_path=tg_vocab_path,
            fine_tuner=False,
            sample_name=None,
            max_cached=2,
            # Subsample knobs (optional)
            max_tfs=SUBSAMPLE_MAX_TFS,
            max_tgs=SUBSAMPLE_MAX_TGS,
            max_windows_per_chrom=SUBSAMPLE_MAX_WINDOWS_PER_CHROM,
            subset_seed=SUBSAMPLE_SEED,
        )
    else:
        # --- Single-chrom mode ---
        dataset = MultiomicTransformerDataset(
            data_dir=SAMPLE_DATA_CACHE_DIR,
            chrom_id=CHROM_ID,
            tf_vocab_path=tf_vocab_path,
            tg_vocab_path=tg_vocab_path,
            max_tfs=SUBSAMPLE_MAX_TFS,
            max_tgs=SUBSAMPLE_MAX_TGS,
            max_windows=SUBSAMPLE_MAX_WINDOWS_PER_CHROM,
            subset_seed=SUBSAMPLE_SEED,
        )
        assert dataset.tf_name2id_sub is not None
        assert dataset.tg_name2id_sub is not None
        
    # Use the dataset’s global sub-vocab (fixed across all chromosomes)
    tf_vocab_size = len(dataset.tf_name2id_sub) if dataset.tf_name2id_sub else 0
    tg_vocab_size = len(dataset.tg_name2id_sub) if dataset.tg_name2id_sub else 0

    # Safety: fall back to full vocab (only if you truly didn’t create a sub-vocab)
    if tf_vocab_size == 0 or tg_vocab_size == 0:
        with open(tf_vocab_path) as f:
            tf_vocab_obj = json.load(f)
        with open(tg_vocab_path) as f:
            tg_vocab_obj = json.load(f)
        tf_vocab_size = len(tf_vocab_obj.get("name_to_id", tf_vocab_obj))
        tg_vocab_size = len(tg_vocab_obj.get("name_to_id", tg_vocab_obj))

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
    - Single-chrom: identical to before (random_split + DistributedSampler).
    - Multi-chrom : split by chromosomes; each split uses ChromBucketBatchSampler
                    (optionally sharded across ranks via DistributedBatchSampler).
    """
    g = torch.Generator()
    g.manual_seed(seed)
    
    def _split_chroms(xs, seed=42):
        """
        Deterministic split of chromosome IDs into train/val/test with sensible minima.
        - For n==1: train=1, val=0, test=0
        - For n==2: train=1, val=1, test=0   (so validation is never empty)
        - For n>=3: ~70/15/15 using largest-remainder rounding, then ensure val/test >=1
        """
        xs = xs[:]  # copy
        rnd = random.Random(seed)
        rnd.shuffle(xs)
        n = len(xs)

        if n == 0:
            return [], [], []
        if n == 1:
            return xs, [], []
        if n == 2:
            return [xs[0]], [xs[1]], []

        # n >= 3: target counts via largest remainder method
        ratios = (0.70, 0.15, 0.15)
        targets = [r * n for r in ratios]
        base = [int(t) for t in targets]  # floors
        rem = n - sum(base)
        remainders = [t - b for t, b in zip(targets, base)]
        # give leftover items to the largest remainders
        for i in sorted(range(3), key=lambda i: remainders[i], reverse=True)[:rem]:
            base[i] += 1

        n_train, n_val, n_test = base

        # ensure val/test get at least 1 if possible
        if n_val == 0 and n_train > 1:
            n_val, n_train = 1, n_train - 1
        if n_test == 0 and n_train > 1:
            n_test, n_train = 1, n_train - 1

        train = xs[:n_train]
        val   = xs[n_train:n_train + n_val]
        test  = xs[n_train + n_val:]
        return train, val, test


    # ---------- Multi-chromosome path ----------
    if isinstance(dataset, MultiChromosomeDataset):
        train_chrs, val_chrs, test_chrs = _split_chroms(dataset.chrom_ids, seed=seed)

        ds_train = MultiChromosomeDataset(
            data_dir=dataset.data_dir,
            chrom_ids=train_chrs,
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

        ds_val = MultiChromosomeDataset(
            data_dir=dataset.data_dir,
            chrom_ids=val_chrs,
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

        ds_test = MultiChromosomeDataset(
            data_dir=dataset.data_dir,
            chrom_ids=test_chrs,
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

        # Bucket batches by chromosome (consistent shapes within batch)
        base_train_bs = ChromBucketBatchSampler(ds_train, batch_size=batch_size, shuffle=True)
        base_val_bs   = ChromBucketBatchSampler(ds_val,   batch_size=batch_size, shuffle=False)
        base_test_bs  = ChromBucketBatchSampler(ds_test,  batch_size=batch_size, shuffle=False)

        if world_size > 1:
            train_bs = DistributedBatchSampler(base_train_bs, world_size, rank)
            val_bs   = DistributedBatchSampler(base_val_bs,   world_size, rank)
            test_bs  = DistributedBatchSampler(base_test_bs,  world_size, rank)
        else:
            train_bs, val_bs, test_bs = base_train_bs, base_val_bs, base_test_bs

        train_loader = DataLoader(
            ds_train,
            batch_sampler=train_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            ds_val,
            batch_sampler=val_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            ds_test,
            batch_sampler=test_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader

    # ---------- Single-chromosome path (unchanged logic) ----------
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
    """Logs run settings and writes a run_parameters.json file."""
    is_multi = isinstance(dataset, MultiChromosomeDataset)

    # helpers: count TF/TG using sub-vocab if present, else fall back
    def _count_tf(ds):
        if hasattr(ds, "tf_name2id_sub") and ds.tf_name2id_sub:
            return len(ds.tf_name2id_sub)
        if hasattr(ds, "tf_names") and ds.tf_names:
            return len(ds.tf_names)
        if hasattr(ds, "tf_ids"):
            return int(ds.tf_ids.numel())
        return 0

    def _count_tg(ds):
        if hasattr(ds, "tg_name2id_sub") and ds.tg_name2id_sub:
            return len(ds.tg_name2id_sub)
        if hasattr(ds, "tg_names") and ds.tg_names:
            return len(ds.tg_names)
        if hasattr(ds, "tg_ids"):
            return int(ds.tg_ids.numel())
        return 0

    if not is_multi:
        tf_n = _count_tf(dataset)
        tg_n = _count_tg(dataset)
        win_n = getattr(dataset, "num_windows", 0)

        logging.info("\n===== MultiomicTransformerDataset Loaded =====")
        logging.info(f"Chromosome:          {CHROM_ID}")
        logging.info(f"Genes (TG):          {tg_n}")
        logging.info(f"Windows (RE):        {win_n}")
        logging.info(f"TFs:                 {tf_n}")
        logging.info(f"Metacells:           {len(getattr(dataset, 'metacell_names', []))}")
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
            "Chromosomes": [CHROM_ID],
            "Genes": tg_n,
            "Windows": win_n,
            "TFs": tf_n,
            "Metacells": len(getattr(dataset, "metacell_names", [])),
            "Epochs": TOTAL_EPOCHS,
            "Batch Size": BATCH_SIZE,
            "d_model": D_MODEL,
            "corr_loss_weight": CORR_LOSS_WEIGHT,
            "Attention Heads": NUM_HEADS,
            "Model Layers": NUM_LAYERS,
            "d_feedforward": D_FF,
            "Dropout": DROPOUT,
            "tf_tg_shortcut": USE_SHORTCUT,
            "Distance Bias": USE_DISTANCE_BIAS,
            "Distance Bias Scale": ATTN_BIAS_SCALE,
            "Motif Mask": USE_MOTIF_MASK,
            "Shortcut L1": SHORTCUT_L1,
            "Shortcut L2": SHORTCUT_L2,
            "Shortcut Dropout": SHORTCUT_DROPOUT,
            "Shortcut Top K": SHORTCUT_TOPK,
        }

    else:
        # global sub-vocab sizes (shared across all chromosomes)
        tf_n = len(getattr(dataset, "tf_name2id_sub", {}) or [])
        tg_n = len(getattr(dataset, "tg_name2id_sub", {}) or [])
        # total windows after per-chrom subsampling (sum child.num_windows)
        total_windows = 0
        for cid in dataset.chrom_ids:
            ds = dataset._load_chrom(cid)
            total_windows += int(getattr(ds, "num_windows", 0))

        logging.info("\n===== MultiChromosomeDataset Loaded =====")
        logging.info(f"Num Chromosomes:     {len(dataset.chrom_ids)}")
        logging.info(f"Chromosomes:         {list(dataset.chrom_ids)}")
        logging.info(f"Unique TGs (global): {tg_n}")
        logging.info(f"Total Windows:       {total_windows}")
        logging.info(f"Unique TFs (global): {tf_n}")
        logging.info(f"Metacells (unique):  {len(getattr(ds, 'metacell_names', []))}")
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
            "Chromosomes": list(dataset.chrom_ids),
            "Genes": tg_n,
            "Windows": total_windows,
            "TFs": tf_n,
            "Metacells": len(getattr(ds, "metacell_names", [])),
            "Epochs": TOTAL_EPOCHS,
            "Batch Size": BATCH_SIZE,
            "d_model": D_MODEL,
            "corr_loss_weight": CORR_LOSS_WEIGHT,
            "Attention Heads": NUM_HEADS,
            "Model Layers": NUM_LAYERS,
            "d_feedforward": D_FF,
            "Dropout": DROPOUT,
            "tf_tg_shortcut": USE_SHORTCUT,
            "Distance Bias": USE_DISTANCE_BIAS,
            "Distance Bias Scale": ATTN_BIAS_SCALE,
            "Motif Mask": USE_MOTIF_MASK,
            "Shortcut L1": SHORTCUT_L1,
            "Shortcut L2": SHORTCUT_L2,
            "Shortcut Dropout": SHORTCUT_DROPOUT,
            "Shortcut Top K": SHORTCUT_TOPK,
        }

    path = os.path.join(out_dir, "run_parameters.json")
    with open(path, "w") as f:
        json.dump(run_params, f, indent=4)
    logging.info(f"Run parameters written to {path}")


@torch.no_grad()
def get_mean_tf_tg_attention(model, dataloader, device="cuda"):
    """
    Returns mean TF→TG attention aligned to the GLOBAL vocab:
      shape [G_global, T_global], averaged over all occurrences of each TG.
    """
    model.eval()
    m = getattr(model, "module", model)

    G_global = m.tg_emb_table.num_embeddings
    T_global = m.tf_emb_table.num_embeddings

    attn_sum   = torch.zeros(G_global, T_global, device="cpu")
    tg_counts  = torch.zeros(G_global, dtype=torch.long, device="cpu")

    for atac_wins, tf_tensor, _, bias, tf_ids, tg_ids, motif_mask in dataloader:
        atac_wins = atac_wins.to(device)
        tf_tensor = tf_tensor.to(device)
        bias      = bias.to(device)
        tf_ids    = tf_ids.to(device)
        tg_ids    = tg_ids.to(device)
        motif_mask= motif_mask.to(device)

        _ = m(atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias,
              motif_mask=(motif_mask if getattr(m, "use_motif_mask", False) else None))

        if not hasattr(m.shortcut_layer, "attn"):
            continue
        # attn_batch: [G_b, T_b], tf_ids: [T_b], tg_ids: [G_b]
        attn_batch = m.shortcut_layer.attn.detach().float().cpu()
        tf_idx_global = tf_ids.detach().long().cpu()
        tg_idx_global = tg_ids.detach().long().cpu()

        # add into global slots
        attn_sum[tg_idx_global[:, None], tf_idx_global[None, :]] += attn_batch
        tg_counts[tg_idx_global] += 1

    # avoid div-by-zero
    tg_counts = tg_counts.clamp_min(1)
    mean_attn = attn_sum / tg_counts[:, None]

    return mean_attn


def write_experiment_settings_and_objects(training_output_dir: Path, dataset: MultiomicTransformerDataset):
    """
    Writes vocab files/scaler and calls write_run_parameters().
    In multi-chrom mode, uses a representative single-chrom dataset to access vocab/scaler.
    """
    # Representative dataset for artifacts that live per-chrom (vocab/scaler in your setup)
    rep = dataset._load_chrom(dataset.chrom_ids[0]) if isinstance(dataset, MultiChromosomeDataset) else dataset

    # Vocab files (same behavior as before, but robust to multi-chrom)
    with open(os.path.join(training_output_dir, "tf_vocab.json"), "w") as f:
        json.dump(getattr(rep, "tf_name2id", {}), f)
    with open(os.path.join(training_output_dir, "tg_vocab.json"), "w") as f:
        json.dump(getattr(rep, "tg_name2id", {}), f)

    # Optional scaler
    if getattr(rep, "scaler", None) is not None:
        scaler_out = os.path.join(training_output_dir, "tg_scaler.pkl")
        joblib.dump(rep.scaler, scaler_out)
        logging.info(f"Saved TG scaler to {scaler_out}")

    # Write the (now multi-chrom aware) parameters JSON + logs
    write_run_parameters(dataset, training_output_dir)

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    # --- Fast guards + signals first ---
    assert D_MODEL % NUM_HEADS == 0, f"{D_MODEL} not divisible by {NUM_HEADS}"
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # --- DDP init + logging ---
    ddp_setup(rank, world_size)
    setup_logging(rank)

    try:
        # --- Build data/model/optim + output dir ---
        dataset, model, optimizer = load_train_objs()

        if isinstance(dataset, MultiChromosomeDataset):
            out_root = OUTPUT_DIR / "MULTI"
            # _debug_multichrom_stats(dataset)
        else:
            out_root = OUTPUT_DIR / CHROM_ID

        name_pattern = "model_training_{:03d}"
        
        # --- Resolve training_output_dir and resume target BEFORE making the dir ---
        training_output_dir, resume_ckpt, resume_inplace = resolve_run_and_resume(out_root, name_pattern)

        # Only create the dir if we're starting fresh or warm-starting into a *new* dir
        if not training_output_dir.exists():
            os.makedirs(training_output_dir, exist_ok=True)

        if rank == 0:
            logging.info(f"\n =========== EXPERIMENT {training_output_dir.name.upper()} ===========")
        
        if rank == 0 and not resume_inplace:
            write_experiment_settings_and_objects(training_output_dir, dataset)
            logging.info("Wrote experiment settings and objects to training output directory")
            logging.info("Preparing dataloader")

        train_loader, val_loader, test_loader = prepare_dataloader(
            dataset, batch_size, world_size, rank
        )

        if rank == 0:
            logging.info("Creating Trainer")

        loss_fn = nn.MSELoss()
        trainer = Trainer(
            model, train_loader, val_loader, loss_fn, optimizer,
            gpu_id=rank, save_every=save_every, patience=PATIENCE
        )

        # --- Resume path handling ---
        start_epoch = 0
        if resume_ckpt and resume_ckpt.exists():
            if rank == 0:
                logging.info(f"Resuming from: {resume_ckpt}")
            if dist.is_initialized():
                dist.barrier()
            start_epoch = load_training_state(
                trainer,
                str(resume_ckpt),
                map_location=(f"cuda:{rank}" if torch.cuda.is_available() else "cpu"),
            )
            if rank == 0:
                logging.info(f"Resumed to start at epoch index: {start_epoch}")
        elif resume_ckpt:
            if rank == 0:
                logging.warning(f"Resume checkpoint not found: {resume_ckpt}")
                
        if rank == 0:
            logging.info("\n ----- CUDA DEVICE INFO -----")
            logging.info(f"  - CUDA available?: {torch.cuda.is_available()}")
            logging.info(f"  - CUDA device count: {torch.cuda.device_count()}")
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                try:
                    logging.info(f"  - CUDA device LOCAL_RANK: {local_rank}")
                    logging.info(f"  - CUDA device name: {torch.cuda.get_device_name(local_rank)}")
                except Exception as e:
                    logging.warning(f"Could not query device name: {e}")
            # Check model params device
            try:
                pdev = next(trainer.model.parameters()).device
                logging.info(f"  - Model first param device: {pdev}")
            except StopIteration:
                logging.info("Model has no parameters? (unexpected)")
                
                # id range sanity
            m = getattr(trainer.model, "module", trainer.model)
            tf_ids = trainer.train_data.dataset._load_chrom(trainer.train_data.dataset.chrom_ids[0]).tf_ids
            tg_ids = trainer.train_data.dataset._load_chrom(trainer.train_data.dataset.chrom_ids[0]).tg_ids
            logging.info(f"  - TF ids: min={int(tf_ids.min())}, max={int(tf_ids.max())}, table={m.tf_emb_table.num_embeddings}")
            logging.info(f"  - TG ids: min={int(tg_ids.min())}, max={int(tg_ids.max())}, table={m.tg_emb_table.num_embeddings}")
            assert tf_ids.max().item() < model.tf_emb_table.num_embeddings
            assert tg_ids.max().item() < model.tg_emb_table.num_embeddings

        # --- Train (Ctrl+C-safe via STOP_REQUESTED path inside Trainer) ---
        if rank == 0:
            logging.info("\n ----- TRAINING STARTED -----")
        trainer.train(max_epochs=TOTAL_EPOCHS, path=training_output_dir, start_epoch=start_epoch)

        if rank == 0:
            model_for_save = getattr(trainer.model, "module", trainer.model)
            model_for_save.eval()

            # save embeddings directly (no reload)
            save_tf_tg_embeddings_from_model(
                model_for_save,
                out_dir=training_output_dir,
                vocab_dir=training_output_dir,
            )

            # only run the heavy “extras” if not interrupted
            if not STOP_REQUESTED.is_set():
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
                    
                    plt.figure(figsize=(12,8))
                    sns.heatmap(mean_attn_df.iloc[:50, :50], cmap="viridis")
                    plt.title("TF→TG attention weights (subset)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(training_output_dir, "tf_tg_attention_heatmap.png"))                
            
                # Training figures
                log_path = os.path.join(training_output_dir, "training_log.csv")
                if os.path.exists(log_path):
                    log_df = pd.read_csv(log_path, header=0)
                
                    chrom_label = CHROM_ID if not isinstance(dataset, MultiChromosomeDataset) else "MULTI"
                    
                    pearson_corr_plt = plotting.plot_pearson_corr_across_epochs(
                        df=log_df, dataset_name=DATASET_NAME, chrom_id=chrom_label
                    )
                    pearson_corr_plt.savefig(os.path.join(training_output_dir, "pearson_training.png"), dpi=300)
                    
                    train_val_loss_plt = plotting.plot_train_val_loss(
                        df=log_df, dataset_name=DATASET_NAME, chrom_id=chrom_label
                    )
                    train_val_loss_plt.savefig(os.path.join(training_output_dir, "train_val_loss.png"), dpi=300)
                else:
                    logging.warning("No training_log.csv yet; skipping training plots.")
                
                per_gene_corr_scatter_plt = plotting.plot_per_gene_correlation_scatterplot(
                    model=model_for_save,
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
    
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "0"))
    world_size  = int(os.environ.get("WORLD_SIZE", "1"))
    
    main(rank=global_rank,
         world_size=world_size,
         save_every=5,
         total_epochs=TOTAL_EPOCHS,
         batch_size=BATCH_SIZE)