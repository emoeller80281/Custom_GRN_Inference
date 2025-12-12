import csv
import glob
import itertools
import math
import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path
import shutil
import signal

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
    MultiChromosomeDataset,
    MultiomicTransformerDataset,
    SimpleScaler,
    fit_simple_scalers,
    IndexedChromBucketBatchSampler,
    DistributedBatchSampler,
)
from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.utils import ewc_utils
from multiomic_transformer.utils.files import unique_path

warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group`")

STOP_REQUESTED = False


def _signal_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def ddp_setup(rank: int, world_size: int, local_rank: int):
    """
    Args:
        rank: Global identifier for this process (RANK)
        world_size: Total number of processes across all nodes
        local_rank: GPU index for this process on the current node
    """

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
    """Load run_parameters.json from a previous experiment directory."""
    param_path = run_dir / "run_parameters.json"
    if not param_path.is_file():
        logging.warning(f"No run_parameters.json found in {run_dir}, using current config.")
        return {}
    with open(param_path, "r") as f:
        params = json.load(f)
    logging.info(f"Loaded run parameters from {param_path}")
    return params


def save_tf_tg_embeddings_from_model(model, out_dir, vocab_dir, epoch=None):
    model = getattr(model, "module", model)  # unwrap DDP if needed

    if epoch is not None:
        emb_save_path = os.path.join(out_dir, f"tf_tg_embeddings_{epoch}.pt")
    else:
        emb_save_path = os.path.join(out_dir, f"tf_tg_embeddings_final.pt")
    torch.save(
        {
            "tf_emb": model.tf_identity_emb.weight.detach().cpu(),
            "tg_query_emb": model.tg_query_emb.weight.detach().cpu(),
            "tg_emb": model.tg_identity_emb.weight.detach().cpu(),
        },
        emb_save_path,
    )

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
        os.path.join(out_dir, "tf_tg_vocab_id2name.pt"),
    )
    
class Trainer:
    def __init__(
        self,
        model,
        train_data,
        val_data,
        loss_fn,
        optimizer,
        gpu_id,
        global_rank,
        save_every,
        patience=20,
        min_delta=1e-3,
        ref_params=None,
        fisher_diag=None,
        lambda_ewc=0.0,
        grad_accum_steps=1,
        use_grad_accumulation=False,
    ):
        self.gpu_id = gpu_id
        self.global_rank = global_rank
        self.is_main = (global_rank == 0)
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        self.train_data = train_data
        self.val_data = val_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_every = save_every
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_grad_accumulation = use_grad_accumulation
        self.corr_sq_warmup_epochs = 3
        self.stop_requested = False

        self.log_train_breakdown_every = 1  # epochs

        self.scaler = GradScaler(init_scale=1024, growth_factor=1.5, backoff_factor=0.5, growth_interval=200)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=MODE,
            factor=SCHEDULER_FACTOR,
            patience=FINETUNE_SCHEDULER_PATIENCE,
            threshold=THRESHOLD,
            threshold_mode=THRESHOLD_MODE,
            cooldown=COOLDOWN,
            min_lr=MIN_LR,
        )

        self.best_val_loss = float("inf")
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0

        # --- scaling + EWC state ---
        self.tf_scaler = None
        self.tg_scaler = None
        self.ref_params = ref_params
        self.fisher_diag = fisher_diag
        self.lambda_ewc = lambda_ewc

    def _should_stop(self):
        global STOP_REQUESTED
        return self.stop_requested or STOP_REQUESTED

    def _save_trained_model(self, path: str):
        if not self.is_main:
            return

        model = getattr(self.model, "module", self.model)
        model.eval()

        ckpt = {"model_state_dict": model.state_dict()}

        if getattr(self, "tf_scaler", None) is not None:
            ckpt["tf_scaler_mean"] = self.tf_scaler.mean.detach().cpu()
            ckpt["tf_scaler_std"] = self.tf_scaler.std.detach().cpu()
        if getattr(self, "tg_scaler", None) is not None:
            ckpt["tg_scaler_mean"] = self.tg_scaler.mean.detach().cpu()
            ckpt["tg_scaler_std"] = self.tg_scaler.std.detach().cpu()

        out_path = os.path.join(path, "trained_model.pt")
        torch.save(ckpt, out_path)
        logging.info(f"Saved trained model to {out_path}")

    def _run_batch(self, batch):
        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
        atac_wins = atac_wins.to(self.gpu_id)
        tf_tensor = tf_tensor.to(self.gpu_id)
        targets = targets.to(self.gpu_id)
        bias = bias.to(self.gpu_id)
        tf_ids = tf_ids.to(self.gpu_id)
        tg_ids = tg_ids.to(self.gpu_id)
        motif_mask = motif_mask.to(self.gpu_id)

        # Optional feature scaling (id-aware)
        if getattr(self, "tf_scaler", None) is not None:
            tf_tensor = self.tf_scaler.transform(tf_tensor, tf_ids)
        if getattr(self, "tg_scaler", None) is not None:
            targets = self.tg_scaler.transform(targets, tg_ids)

        tf_tensor = torch.nan_to_num(tf_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        atac_wins = torch.nan_to_num(atac_wins, nan=0.0, posinf=1e6, neginf=-1e6)
        bias = torch.nan_to_num(bias, nan=0.0, posinf=5.0, neginf=-5.0)
        motif_mask = torch.nan_to_num(motif_mask, nan=0.0)

        for name, t in {
            "atac_wins": atac_wins,
            "tf_tensor": tf_tensor,
            "targets": targets,
            "bias": bias,
            "motif_mask": motif_mask,
        }.items():
            if not torch.isfinite(t).all():
                bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5]
                raise RuntimeError(f"{name} has non-finite values; examples idx={bad}")

        model_for_reg = getattr(self.model, "module", self.model)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            mask_arg = motif_mask if USE_MOTIF_MASK else None
            preds, attn, shortcut_contrib, _ = self.model(
                atac_wins,
                tf_tensor,
                tf_ids=tf_ids,
                tg_ids=tg_ids,
                bias=bias,
                motif_mask=mask_arg,
                return_shortcut_contrib=False,
            )

        preds32 = torch.nan_to_num(preds.float(), nan=0.0, posinf=1e6, neginf=-1e6)
        targets32 = torch.nan_to_num(targets.float(), nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Calculate MSE, but don't weight zero's as heavily to account for sparsity
        residual = preds32 - targets32

        zero_mask = (targets32 <= ZERO_EPS)
        weights = torch.ones_like(targets32)

        weights[zero_mask] = ZERO_WEIGHT

        mse_loss = (weights * residual**2).sum() / weights.sum()

        # ----- Unscaled MSE for logging (no grad) -----
        if getattr(self, "tg_scaler", None) is not None:
            with torch.no_grad():
                targets_u = self.tg_scaler.inverse_transform(targets32, tg_ids)
                preds_u = self.tg_scaler.inverse_transform(preds32, tg_ids)

                targets_u = torch.nan_to_num(targets_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)
                preds_u = torch.nan_to_num(preds_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)

                mse_loss_unscaled = F.mse_loss(preds_u, targets_u).detach()
        else:
            mse_loss_unscaled = mse_loss.detach()

        # ---------- Correlation penalty via per-gene Pearson r^2 ----------
        x = preds32
        y = targets32
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        eps = 1e-8
        x_ss = (x * x).sum(dim=0)
        y_ss = (y * y).sum(dim=0)
        denom = (x_ss * y_ss).clamp_min(eps).sqrt()
        valid = (x_ss > eps) & (y_ss > eps)

        corr = torch.zeros_like(denom)
        corr[valid] = (x[:, valid] * y[:, valid]).sum(dim=0) / denom[valid]
        corr = corr.clamp_(-1.0, 1.0)
        r2_g = corr * corr

        if valid.any():
            mean_r2 = r2_g[valid].mean()
        else:
            mean_r2 = torch.tensor(0.0, device=preds32.device, dtype=preds32.dtype)

        corr_warm = max(1, getattr(self, "corr_sq_warmup_epochs", 3))
        cur_epoch = getattr(self, "epoch", 0)
        corr_anneal = float(min(1.0, (cur_epoch + 1) / corr_warm))
        r2_penalty = (FINETUNE_CORR_WEIGHT * corr_anneal) * (1.0 - mean_r2)

        shortcut_reg = torch.tensor(0.0, device=self.gpu_id, dtype=torch.float32)
        if getattr(model_for_reg, "use_shortcut", False) and hasattr(model_for_reg, "shortcut_layer"):
            reg_val = model_for_reg.shortcut_layer.regularization()
            shortcut_reg = reg_val.to(dtype=torch.float32, device=self.gpu_id) * FINETUNE_SHORTCUT_REG_WEIGHT

        loss_ewc = torch.tensor(0.0, device=self.gpu_id, dtype=torch.float32)
        if self.ref_params is not None and self.fisher_diag is not None:
            loss_ewc = ewc_utils.ewc_penalty(
                model_for_reg,
                self.fisher_diag,
                self.ref_params,
                lambda_ewc=self.lambda_ewc,
            )

        total_loss = mse_loss + r2_penalty + shortcut_reg + loss_ewc

        if not torch.isfinite(total_loss):
            logging.warning("Non-finite loss encountered; skipping batch.")
            return None

        return (
            total_loss,
            mse_loss.detach(),
            mse_loss_unscaled,
            mean_r2.detach(),
            r2_penalty.detach(),
            loss_ewc.detach(),
        )

    def _validate(self):
        self.model.eval()

        total_loss_scaled_t = torch.zeros(1, device=self.gpu_id)
        total_loss_unscaled_t = torch.zeros(1, device=self.gpu_id)
        n_batches = 0

        sse_s = torch.zeros(1, device=self.gpu_id)
        sumy_s = torch.zeros(1, device=self.gpu_id)
        sumy2_s = torch.zeros(1, device=self.gpu_id)
        n_s = torch.zeros(1, device=self.gpu_id)

        sse_u = torch.zeros(1, device=self.gpu_id)
        sumy_u = torch.zeros(1, device=self.gpu_id)
        sumy2_u = torch.zeros(1, device=self.gpu_id)
        n_u = torch.zeros(1, device=self.gpu_id)

        per_dataset_stats = {}
        for name in self.val_data.keys():
            per_dataset_stats[name] = {
                "sse_s": torch.zeros(1, device=self.gpu_id),
                "sumy_s": torch.zeros(1, device=self.gpu_id),
                "sumy2_s": torch.zeros(1, device=self.gpu_id),
                "n_s": torch.zeros(1, device=self.gpu_id),
                "sse_u": torch.zeros(1, device=self.gpu_id),
                "sumy_u": torch.zeros(1, device=self.gpu_id),
                "sumy2_u": torch.zeros(1, device=self.gpu_id),
                "n_u": torch.zeros(1, device=self.gpu_id),
                "loss_s": torch.zeros(1, device=self.gpu_id),
                "loss_u": torch.zeros(1, device=self.gpu_id),
                "batches": torch.zeros(1, device=self.gpu_id),
            }

        with torch.no_grad():
            for name, loader in self.val_data.items():
                for batch in loader:
                    if self._should_stop():
                        raise KeyboardInterrupt()

                    atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch

                    atac_wins = atac_wins.to(self.gpu_id, non_blocking=True)
                    tf_tensor = tf_tensor.to(self.gpu_id, non_blocking=True)
                    targets = targets.to(self.gpu_id, non_blocking=True)
                    bias = bias.to(self.gpu_id, non_blocking=True)
                    tf_ids = tf_ids.to(self.gpu_id, non_blocking=True)
                    tg_ids = tg_ids.to(self.gpu_id, non_blocking=True)
                    motif_mask = motif_mask.to(self.gpu_id, non_blocking=True)

                    if getattr(self, "tf_scaler", None) is not None:
                        tf_tensor = self.tf_scaler.transform(tf_tensor, tf_ids)
                    if getattr(self, "tg_scaler", None) is not None:
                        targets_s = self.tg_scaler.transform(targets, tg_ids)
                    else:
                        targets_s = targets

                    mask_arg = motif_mask if USE_MOTIF_MASK else None
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        preds, _, _, _ = self.model(
                            atac_wins,
                            tf_tensor,
                            tf_ids=tf_ids,
                            tg_ids=tg_ids,
                            bias=bias,
                            motif_mask=mask_arg,
                            return_shortcut_contrib=False,
                        )

                    preds_s = torch.nan_to_num(preds.float(), nan=0.0, posinf=1e6, neginf=-1e6)
                    targets_s = torch.nan_to_num(targets_s.float(), nan=0.0, posinf=1e6, neginf=-1e6)

                    loss_s = F.mse_loss(preds_s, targets_s)
                    total_loss_scaled_t += loss_s.detach()
                    per_dataset_stats[name]["loss_s"] += loss_s.detach()
                    per_dataset_stats[name]["batches"] += 1
                    n_batches += 1

                    y_s = targets_s.reshape(-1)
                    p_s = preds_s.reshape(-1)
                    sse_s += torch.sum((y_s - p_s) ** 2)
                    sumy_s += torch.sum(y_s)
                    sumy2_s += torch.sum(y_s ** 2)
                    n_s += y_s.numel()

                    per_dataset_stats[name]["sse_s"] += torch.sum((y_s - p_s) ** 2)
                    per_dataset_stats[name]["sumy_s"] += torch.sum(y_s)
                    per_dataset_stats[name]["sumy2_s"] += torch.sum(y_s ** 2)
                    per_dataset_stats[name]["n_s"] += y_s.numel()

                    if getattr(self, "tg_scaler", None) is not None:
                        targets_u = self.tg_scaler.inverse_transform(targets_s, tg_ids)
                        preds_u = self.tg_scaler.inverse_transform(preds_s, tg_ids)
                    else:
                        targets_u, preds_u = targets_s, preds_s

                    targets_u = torch.nan_to_num(targets_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)
                    preds_u   = torch.nan_to_num(preds_u.float(),   nan=0.0, posinf=1e6, neginf=-1e6)

                    # ---- weighted MSE to reduce impact of zeros ----
                    residual_u = preds_u - targets_u
                    zero_mask  = (targets_u <= ZERO_EPS)  # treat exact zeros (or near-zero) as "dropout candidates"
                    weights_u  = torch.ones_like(targets_u)
                    weights_u[zero_mask] = ZERO_WEIGHT

                    # batch-weighted MSE in unscaled space
                    loss_u = (weights_u * residual_u**2).sum() / weights_u.sum()

                    total_loss_unscaled_t += loss_u.detach()
                    per_dataset_stats[name]["loss_u"] += loss_u.detach()

                    y_u = targets_u.reshape(-1)
                    p_u = preds_u.reshape(-1)
                    sse_u += torch.sum((y_u - p_u) ** 2)
                    sumy_u += torch.sum(y_u)
                    sumy2_u += torch.sum(y_u ** 2)
                    n_u += y_u.numel()

                    per_dataset_stats[name]["sse_u"] += torch.sum((y_u - p_u) ** 2)
                    per_dataset_stats[name]["sumy_u"] += torch.sum(y_u)
                    per_dataset_stats[name]["sumy2_u"] += torch.sum(y_u ** 2)
                    per_dataset_stats[name]["n_u"] += y_u.numel()

        n_batches_t = torch.tensor(n_batches, device=self.gpu_id, dtype=torch.long)

        if dist.is_available() and dist.is_initialized():
            for t in (sse_s, sumy_s, sumy2_s, n_s, sse_u, sumy_u, sumy2_u, n_u,
                      total_loss_scaled_t, total_loss_unscaled_t):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_batches_t, op=dist.ReduceOp.SUM)

            for stats in per_dataset_stats.values():
                for key in stats:
                    dist.all_reduce(stats[key], op=dist.ReduceOp.SUM)

        global_n_batches = int(n_batches_t.item()) if dist.is_available() and dist.is_initialized() else int(n_batches)

        if global_n_batches == 0 or n_s.item() == 0:
            return 0.0, 0.0, 0.0, 0.0, {}

        eps = 1e-12

        ybar_s = sumy_s / torch.clamp(n_s, min=1.0)
        sst_s = sumy2_s - n_s * (ybar_s ** 2)
        r2_s = torch.where(sst_s <= eps, torch.zeros_like(sst_s), 1.0 - sse_s / torch.clamp(sst_s, min=eps))

        ybar_u = sumy_u / torch.clamp(n_u, min=1.0)
        sst_u = sumy2_u - n_u * (ybar_u ** 2)
        r2_u = torch.where(sst_u <= eps, torch.zeros_like(sst_u), 1.0 - sse_u / torch.clamp(sst_u, min=eps))

        avg_loss_scaled = float(total_loss_scaled_t.item()) / max(1, global_n_batches)
        avg_loss_unscaled = float(total_loss_unscaled_t.item()) / max(1, global_n_batches)

        per_dataset_metrics = {}
        for name, stats in per_dataset_stats.items():
            batches = max(1, int(stats["batches"].item()))
            n_s_ds = stats["n_s"]
            n_u_ds = stats["n_u"]

            if n_s_ds.item() == 0:
                per_dataset_metrics[name] = {
                    "val_mse_scaled": 0.0,
                    "val_mse_unscaled": 0.0,
                    "r2_s": 0.0,
                    "r2_u": 0.0,
                }
                continue

            ybar_s_ds = stats["sumy_s"] / torch.clamp(n_s_ds, min=1.0)
            sst_s_ds = stats["sumy2_s"] - n_s_ds * (ybar_s_ds ** 2)
            r2_s_ds = torch.where(
                sst_s_ds <= eps, torch.zeros_like(sst_s_ds), 1.0 - stats["sse_s"] / torch.clamp(sst_s_ds, min=eps)
            )

            ybar_u_ds = stats["sumy_u"] / torch.clamp(n_u_ds, min=1.0)
            sst_u_ds = stats["sumy2_u"] - n_u_ds * (ybar_u_ds ** 2)
            r2_u_ds = torch.where(
                sst_u_ds <= eps, torch.zeros_like(sst_u_ds), 1.0 - stats["sse_u"] / torch.clamp(sst_u_ds, min=eps)
            )

            per_dataset_metrics[name] = {
                "val_mse_scaled": float((stats["loss_s"] / batches).item()),
                "val_mse_unscaled": float((stats["loss_u"] / batches).item()),
                "r2_s": float(r2_s_ds.item()),
                "r2_u": float(r2_u_ds.item()),
            }

        return float(avg_loss_scaled), float(avg_loss_unscaled), float(r2_s.item()), float(r2_u.item()), per_dataset_metrics

    def _run_epoch(self, epoch):
        if isinstance(self.train_data, dict):
            for loader in self.train_data.values():
                sampler = getattr(loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                bs = getattr(loader, "batch_sampler", None)
                if hasattr(bs, "set_epoch"):
                    bs.set_epoch(epoch)
            batch_iter = balanced_round_robin(self.train_data, max_steps=MAX_STEPS, seed=42)
            per_dataset_losses = {name: [0.0, 0] for name in self.train_data}
        else:
            sampler = getattr(self.train_data, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            bs = getattr(self.train_data, "batch_sampler", None)
            if hasattr(bs, "set_epoch"):
                bs.set_epoch(epoch)
            batch_iter = ((batch, None) for batch in self.train_data)
            per_dataset_losses = None

        total_loss_sum = 0.0
        total_mse_scaled_sum = 0.0
        total_r2_penalty_sum = 0.0
        total_mse_unscaled_sum = 0.0
        total_ewc_loss_sum = 0.0
        n_batches = 0
        self.epoch = epoch

        self.optimizer.zero_grad(set_to_none=True)

        total_batches = None
        if isinstance(self.train_data, dict):
            total_batches = max(len(loader) for loader in self.train_data.values()) * len(self.train_data)
        elif hasattr(self.train_data, "__len__"):
            total_batches = len(self.train_data)
            
        total_batches = None
        if isinstance(self.train_data, dict):
            total_batches = max(len(loader) for loader in self.train_data.values()) * len(self.train_data)
        elif hasattr(self.train_data, "__len__"):
            total_batches = len(self.train_data)

        # Progress markers in *fraction of the epoch*
        progress_marks = [0.25, 0.50, 0.75]
        next_mark_idx = 0
        
        for iteration, (batch, dataset_name) in enumerate(batch_iter):
            if self._should_stop():
                raise KeyboardInterrupt()

            out = self._run_batch(batch)
            if out is None:
                self.optimizer.zero_grad(set_to_none=True)
                continue

            (total_loss_val, mse_scaled, mse_unscaled, mean_corr, corr_weight, loss_ewc) = out

            if not total_loss_val.requires_grad:
                raise RuntimeError("Bug: total_loss_val has no grad_fn")

            loss_for_backprop = total_loss_val / self.grad_accum_steps
            self.scaler.scale(loss_for_backprop).backward()

            if ((iteration + 1) % self.grad_accum_steps == 0) or (
                total_batches is not None and (iteration + 1) == total_batches
            ):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss_sum += float(total_loss_val.detach())
            total_mse_scaled_sum += float(mse_scaled)
            total_mse_unscaled_sum += float(mse_unscaled)
            total_r2_penalty_sum  += float(corr_weight)
            total_ewc_loss_sum += float(loss_ewc)
            n_batches += 1
            
            # --- Progress logging based on total_batches ---
            if self.is_main and total_batches is not None and next_mark_idx < len(progress_marks):
                frac = (iteration + 1) / total_batches
                if frac >= progress_marks[next_mark_idx]:
                    pct = int(progress_marks[next_mark_idx] * 100)
                    logging.info(f"    [{pct}%] Iter {iteration + 1}/{total_batches}")
                    next_mark_idx += 1

            if per_dataset_losses is not None and dataset_name is not None:
                per_dataset_losses[dataset_name][0] += float(mse_unscaled)
                per_dataset_losses[dataset_name][1] += 1

        avg_train_loss = total_loss_sum / max(1, n_batches)
        avg_train_r2_penalty_loss = total_r2_penalty_sum / max(1, n_batches)
        avg_train_mse_scaled = total_mse_scaled_sum / max(1, n_batches)
        avg_train_mse_unscaled = total_mse_unscaled_sum / max(1, n_batches)
        avg_train_ewc_loss = total_ewc_loss_sum / max(1, n_batches)
        

        avg_val_mse_scaled, avg_val_mse_unscaled, r2_s, r2_u, per_dataset_val_metrics = self._validate()
        self.scheduler.step(avg_val_mse_unscaled)

        return (
            avg_train_loss,
            avg_train_r2_penalty_loss,
            avg_train_mse_scaled,
            avg_train_mse_unscaled,
            avg_train_ewc_loss,
            avg_val_mse_scaled,
            avg_val_mse_unscaled,
            r2_s,
            r2_u,
            per_dataset_losses,
            per_dataset_val_metrics,
        )

    def _save_checkpoint(self, epoch: int, path: str):
        if not self.is_main:
            return

        os.makedirs(path, exist_ok=True)

        if hasattr(self.model, "module"):
            model_for_save = self.model.module
        else:
            model_for_save = self.model

        model_state = model_for_save.state_dict()

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_state,
        }

        if hasattr(self, "optimizer") and self.optimizer is not None:
            ckpt["optimizer_state_dict"] = self.optimizer.state_dict()
        if hasattr(self, "scheduler") and self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        if hasattr(self, "tf_scaler") and self.tf_scaler is not None:
            ckpt["tf_scaler_mean"] = self.tf_scaler.mean.detach().cpu()
            ckpt["tf_scaler_std"] = self.tf_scaler.std.detach().cpu()
        if hasattr(self, "tg_scaler") and self.tg_scaler is not None:
            ckpt["tg_scaler_mean"] = self.tg_scaler.mean.detach().cpu()
            ckpt["tg_scaler_std"] = self.tg_scaler.std.detach().cpu()

        save_tf_tg_embeddings_from_model(model_for_save, out_dir=path, vocab_dir=path)

        out_path = os.path.join(path, f"fine_tune_checkpoint_{epoch}.pt")
        torch.save(ckpt, out_path)
        logging.info(f"\tTraining checkpoint saved to {out_path}")

    def _log_train_breakdown(self, epoch, per_dataset_losses, per_dataset_val_metrics, path):
        if not self.is_main or per_dataset_losses is None:
            return

        csv_path = os.path.join(path, "train_dataset_breakdown.csv")
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["epoch", "dataset", "train_mse_unscaled", "val_mse_unscaled", "r2_s", "r2_u"])

            for name in per_dataset_losses.keys():
                train_mse = per_dataset_losses[name][0] / max(1, per_dataset_losses[name][1])
                val_metrics = per_dataset_val_metrics.get(
                    name, {"val_mse_unscaled": None, "r2_s": None, "r2_u": None}
                )
                writer.writerow(
                    [epoch, name, train_mse, val_metrics.get("val_mse_unscaled"), val_metrics.get("r2_s"), val_metrics.get("r2_u")]
                )


    def train(self, max_epochs: int, path: str):
        best_r2 = float("-inf")
        patience_counter = 0
        history = []

        try:
            total_train_start_time = time.time()
            for epoch in range(max_epochs):
                epoch_start_time = time.time()
                if self._should_stop():
                    raise KeyboardInterrupt()

                (
                    avg_train_loss,
                    avg_train_r2_penalty_loss,
                    avg_train_mse_scaled,
                    avg_train_mse_unscaled,
                    avg_train_ewc_loss,
                    avg_val_mse_scaled,
                    avg_val_mse_unscaled,
                    r2_s,
                    r2_u,
                    per_dataset_losses,
                    per_dataset_val_metrics,
                ) = self._run_epoch(epoch)
                epoch_end_time = time.time()
                epoch_dur_sec = epoch_end_time - epoch_start_time

                if self._should_stop():
                    raise KeyboardInterrupt()

                if self.is_main:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logging.info(
                        f"Epoch {epoch+1} | "
                        f"Train MSE: {avg_train_mse_unscaled:.4f} | "
                        f"Train R2 Loss: {avg_train_r2_penalty_loss:.4f} | "
                        f"Train EWC: {avg_train_ewc_loss:.4f} | "
                        f"Val MSE: {avg_val_mse_unscaled:.4f} | "
                        f"R2 (Unscaled): {r2_u:.3f} | "
                        f"R2 (Scaled): {r2_s:.3f} | "
                        f"LR: {lr:.2e} | "
                        f"Time: {epoch_dur_sec:.0f}s"
                    )

                    epoch_log = {
                        "Epoch": epoch+1,
                        "Train MSE": avg_train_mse_unscaled,
                        "Train R2 Loss": avg_train_r2_penalty_loss,
                        "Train EWC": avg_train_ewc_loss,
                        "Val MSE": avg_val_mse_unscaled,
                        "R2_u": r2_u,
                        "R2_s": r2_s,
                        "LR": lr,
                        "Time": round(epoch_dur_sec, 0),
                    }
                    history.append(epoch_log)

                    self._write_log_csv(epoch_log, path)

                if epoch % self.save_every == 0:
                    if self.is_main:
                        self._save_checkpoint(epoch, path)
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()

                if epoch % self.log_train_breakdown_every == 0:
                    self._log_train_breakdown(epoch, per_dataset_losses, per_dataset_val_metrics, path)

                stop_tensor = torch.tensor(0, device=self.gpu_id)

                if self.is_main:
                    improved = False
                    if avg_val_mse_unscaled < self.best_val_loss - self.min_delta:
                        self.best_val_loss = avg_val_mse_unscaled
                        improved = True
                    if r2_s > best_r2 + self.min_delta:
                        best_r2 = r2_s
                        improved = True

                    if improved:
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= self.patience:
                            logging.info("Early stopping triggered (no improvement).")
                            self._save_checkpoint(epoch, path)
                            stop_tensor.fill_(1)
                        else:
                            logging.info(f"    Loss did not improve {patience_counter}/{self.patience}")

                dist.broadcast(stop_tensor, src=0)

                if stop_tensor.item() == 1:
                    if self.is_main:
                        logging.info("All ranks stopping training.")
                    break

            total_train_end_time = time.time()
            total_training_time_min = total_train_end_time - total_train_start_time

            if self.is_main and patience_counter < self.patience:
                logging.info("Training loop exited normally.")
                hours, remainder = divmod(total_training_time_min, 3600)
                minutes, seconds = divmod(remainder, 60)
                logging.info(f"Total Training Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

        except KeyboardInterrupt:
            epoch = locals().get("epoch", 0)
            if self.is_main:
                logging.info("Keyboard interrupt received; saving before exit.")
                self._save_checkpoint(epoch, path)
                self._save_trained_model(path)
            raise

    def _write_log_csv(self, history, path):
        fieldnames = ["Epoch", "Train MSE", "Train R2 Loss", "Train EWC", "Val MSE", "R2_u", "R2_s", "LR", "Time"]
        log_path = os.path.join(path, "training_log.csv")

        file_exists = os.path.isfile(log_path)
        mode = "a" if file_exists else "w"

        with open(log_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            if isinstance(history, dict):
                writer.writerow(history)
            else:
                writer.writerows(history)
            

def load_train_objs(run_cfg):
    # --- Step 1: Load pseudobulk dataset (for Fisher / EWC) across all chromosomes ---
    pseudobulk_dataset = MultiChromosomeDataset(
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
        allowed_samples=ALLOWED_SAMPLES,
    )

    # --- Step 2: Load all single-cell datasets across all chromosomes ---
    single_cell_datasets = [
        MultiChromosomeDataset(
            data_dir=SAMPLE_DATA_CACHE_DIR,
            chrom_ids=CHROM_IDS,
            tf_vocab_path=os.path.join(COMMON_DATA, "tf_vocab.json"),
            tg_vocab_path=os.path.join(COMMON_DATA, "tg_vocab.json"),
            fine_tuner=True,
            sample_name=sn,
            max_cached=2,
            max_tfs=SUBSAMPLE_MAX_TFS,
            max_tgs=SUBSAMPLE_MAX_TGS,
            max_windows_per_chrom=SUBSAMPLE_MAX_WINDOWS_PER_CHROM,
            max_cells=SUBSAMPLE_MAX_CELLS_FINETUNE,
            subset_seed=SUBSAMPLE_SEED,
        )
        for sn in FINE_TUNING_DATASETS
    ]

    # --- Step 3: Build vocab sizes (global across chromosomes) ---
    tf_vocab_size = len(pseudobulk_dataset.tf_name2id_sub)
    tg_vocab_size = len(pseudobulk_dataset.tg_name2id_sub)

    # --- Step 4: Initialize model ---
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

    tf_scaler = None
    tg_scaler = None

    # --- Step 5: Load pretrained weights/scalers if available ---
    pretrained_model = FINE_TUNING_DIR / "trained_model.pt"
    if pretrained_model.exists():
        logging.info(f"Loading pretrained weights from {pretrained_model}")
        state_dict = torch.load(pretrained_model, map_location="cpu")
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"], strict=False)
            tf_mean = state_dict.get("tf_scaler_mean")
            tf_std = state_dict.get("tf_scaler_std")
            tg_mean = state_dict.get("tg_scaler_mean")
            tg_std = state_dict.get("tg_scaler_std")
            if tf_mean is not None and tf_std is not None:
                tf_scaler = SimpleScaler(tf_mean, tf_std)
            if tg_mean is not None and tg_std is not None:
                tg_scaler = SimpleScaler(tg_mean, tg_std)
        else:
            model.load_state_dict(state_dict, strict=False)

    # --- Step 6: Fine-tune optimizer ---
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FINETUNE_LR,
    )

    return pseudobulk_dataset, single_cell_datasets, model, optimizer, tf_scaler, tg_scaler


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
    import random as pyrandom
    g = torch.Generator()
    g.manual_seed(seed)

    # ---------- Multi-chromosome path ----------
    if isinstance(dataset, MultiChromosomeDataset):
        chrom_to_indices = {}
        for i, chrom in enumerate(dataset.chrom_ids):
            start = dataset._offsets[i]
            end = dataset._offsets[i + 1] if i + 1 < len(dataset._offsets) else len(dataset)
            if end > start:
                chrom_to_indices[chrom] = list(range(start, end))

        train_map = {}
        val_map = {}
        test_map = {}

        for chrom, idxs in chrom_to_indices.items():
            n = len(idxs)
            if n == 0:
                continue

            rnd = pyrandom.Random(seed + hash(chrom) % 10_000_000)
            idxs_shuf = idxs[:]
            rnd.shuffle(idxs_shuf)

            n_train = int(0.70 * n)
            n_val = int(0.15 * n)
            n_test = n - n_train - n_val

            if n_val == 0 and n_train > 1:
                n_val += 1
                n_train -= 1
            if n_test == 0 and n_train > 1:
                n_test += 1
                n_train -= 1

            train_idx = idxs_shuf[:n_train]
            val_idx = idxs_shuf[n_train:n_train + n_val]
            test_idx = idxs_shuf[n_train + n_val:]

            if train_idx:
                train_map[chrom] = train_idx
            if val_idx:
                val_map[chrom] = val_idx
            if test_idx:
                test_map[chrom] = test_idx

        base_train_bs = IndexedChromBucketBatchSampler(
            train_map, batch_size=batch_size, shuffle=True, seed=seed
        )
        base_val_bs = IndexedChromBucketBatchSampler(
            val_map, batch_size=batch_size, shuffle=False, seed=seed
        )
        base_test_bs = IndexedChromBucketBatchSampler(
            test_map, batch_size=batch_size, shuffle=False, seed=seed
        )

        if world_size > 1:
            train_bs = DistributedBatchSampler(base_train_bs, world_size, rank, drop_last=drop_last)
            val_bs = DistributedBatchSampler(base_val_bs, world_size, rank, drop_last=False)
            test_bs = DistributedBatchSampler(base_test_bs, world_size, rank, drop_last=False)
        else:
            train_bs, val_bs, test_bs = base_train_bs, base_val_bs, base_test_bs

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

    # ---------- Single-chromosome / legacy path ----------
    n_total = len(dataset)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=g)

    train_sampler = val_sampler = test_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, drop_last=drop_last)
        val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, drop_last=False)
        test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, drop_last=False)

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


def write_run_parameters(dataset, out_dir, world_size, run_cfg):    
    logging.info("\n===== Dataset Loaded for Fine-Tuning =====")
    logging.info(f"Chromosomes:         {CHROM_IDS}")
    logging.info(f"Genes:               {len(dataset.tg_ids)}")
    logging.info(f"Windows (RE):        {dataset.num_windows}")
    logging.info(f"TFs:                 {len(dataset.tf_ids)}")
    if hasattr(dataset, "metacell_names"):
        logging.info(f"Metacells:           {len(dataset.metacell_names)}")
    logging.info(f"Epochs:              {FINETUNE_EPOCHS}")
    logging.info(f"Batch Size:          {BATCH_SIZE}")
    logging.info(f"GPUs:                {world_size}")
    logging.info(f"Grad Accum Steps:    {GRAD_ACCUM_STEPS}")
    logging.info(f"Use Grad Accum?:     {USE_GRAD_ACCUMULATION}")
    logging.info(f"Use Grad Chkpt?:     {USE_GRAD_CHECKPOINTING}")
    logging.info(f"Model Dimension:     {run_cfg['d_model']}")
    logging.info(f"Attention Heads:     {run_cfg['num_heads']}")
    logging.info(f"Attention Layers:    {run_cfg['num_layers']}")
    logging.info(f"Feedforward Layers:  {run_cfg['d_ff']}")
    logging.info(f"Dropout:             {run_cfg['dropout']}")
    logging.info(f"TF-TG Shortcut?:     {run_cfg['use_shortcut']}")
    logging.info(f"Dist bias?:          {run_cfg['use_dist_bias']}")
    logging.info(f"Motif Mask?:         {run_cfg['use_motif_mask']}")
    logging.info(f"Mask Thresh:         {run_cfg['motif_mask_threshold']}")
    logging.info(f"Mask Soft Scale:     {run_cfg['motif_prior_scale']}")
    logging.info(f"Shortcut L1:         {run_cfg['shortcut_l1']}")
    logging.info(f"Shortcut L2:         {run_cfg['shortcut_l2']}")
    logging.info(f"Shortcut Dropout:    {run_cfg['shortcut_dropout']}")
    logging.info(f"Shortcut Top K:      {run_cfg['shortcut_topk']}")
    logging.info("================================================")

    run_params = {
        "allowed_samples": ALLOWED_SAMPLES,
        "epochs": FINETUNE_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "use_grad_accum": USE_GRAD_ACCUMULATION,
        "use_grad_ckpt": USE_GRAD_CHECKPOINTING,
        "d_model": run_cfg["d_model"],
        "num_heads": run_cfg["num_heads"],
        "num_layers": run_cfg["num_layers"],
        "d_ff": run_cfg["d_ff"],
        "dropout": run_cfg["dropout"],
        "use_shortcut": run_cfg["use_shortcut"],
        "use_dist_bias": run_cfg["use_dist_bias"],
        "use_motif_mask": run_cfg["use_motif_mask"],
        "motif_mask_threshold": run_cfg["motif_mask_threshold"],
        "motif_prior_scale": run_cfg["motif_prior_scale"],
        "shortcut_l1": run_cfg["shortcut_l1"],
        "shortcut_l2": run_cfg["shortcut_l2"],
        "shortcut_dropout": run_cfg["shortcut_dropout"],
        "shortcut_topk": run_cfg["shortcut_topk"],
        "lr": FINETUNE_LR,
        "genes": len(dataset.tg_ids),
        "windows": dataset.num_windows,
        "tfs": len(dataset.tf_ids),
        "metacells": len(getattr(dataset, "metacell_names", [])) if hasattr(dataset, "metacell_names") else None,
    }

    path = os.path.join(out_dir, "run_parameters.json")
    with open(path, "w") as f:
        json.dump(run_params, f, indent=4)
    logging.info(f"Run parameters written to {path}")
    
def balanced_round_robin(loaders, max_steps=None, seed=42):
    max_len = max(len(loader) for loader in loaders.values())
    keys = list(loaders.keys())
    iters = {name: iter(loader) for name, loader in loaders.items()}

    total = max_len * len(keys)
    if max_steps is not None:
        total = min(total, max_steps)

    for i in range(total):
        name = keys[i % len(keys)]
        try:
            batch = next(iters[name])
        except StopIteration:
            iters[name] = iter(loaders[name])
            batch = next(iters[name])
        yield batch, name

def _mapping_to_ordered_list(name2id: dict):
    # convert {name: id} → [names] in id order
    return [k for k, _ in sorted(name2id.items(), key=lambda kv: kv[1])]

def write_experiment_settings_and_objects(training_output_dir: Path, sample_name, dataset, test_loader, world_size: int, run_cfg):
    """
    Works for both MultiChromosomeDataset and single-chrom MultiomicTransformerDataset.
    Writes tf/tg vocab mappings and run parameters. Skips scaler unless present.
    """
    sample_output_dir = training_output_dir / sample_name
    os.makedirs(sample_output_dir, exist_ok=True)

    # Pick the right mappings depending on dataset type
    tf_map = getattr(dataset, "tf_name2id_sub", None)
    tg_map = getattr(dataset, "tg_name2id_sub", None)

    if tf_map is None or tg_map is None:
        raise RuntimeError("Dataset is missing TF/TG name→id mappings.")

    # Persist mappings (name→id)
    with open(os.path.join(sample_output_dir, "tf_vocab.json"), "w") as f:
        json.dump(tf_map, f)
    with open(os.path.join(sample_output_dir, "tg_vocab.json"), "w") as f:
        json.dump(tg_map, f)

    # Also persist ordered name lists (useful for plotting/inspection)
    tf_names_ordered = _mapping_to_ordered_list(tf_map)
    tg_names_ordered = _mapping_to_ordered_list(tg_map)
    with open(os.path.join(sample_output_dir, "tf_names_ordered.json"), "w") as f:
        json.dump(tf_names_ordered, f)
    with open(os.path.join(sample_output_dir, "tg_names_ordered.json"), "w") as f:
        json.dump(tg_names_ordered, f)
    
    # Persist test loader
    torch.save(test_loader, os.path.join(sample_output_dir, "test_loader.pt"))

    # Your existing run-parameter writer is fine to call here if it doesn’t assume single-chrom only
    write_run_parameters(dataset, sample_output_dir, world_size, run_cfg)
    logging.info("Wrote experiment settings and objects to training output directory")


def main(rank: int, local_rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    assert D_MODEL % NUM_HEADS == 0, f"{D_MODEL} not divisible by {NUM_HEADS}"

    ddp_setup(rank, world_size, local_rank)
    
    print(
        f"[HOST {os.environ.get('HOSTNAME','?')}] "
        f"RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}",
        flush=True,
    )
    
    
    setup_logging(rank)

    try:
        prev_params = load_run_params_from_json(FINE_TUNING_DIR)

        def g(key, default):
            return prev_params.get(key, default) if prev_params else default

        run_cfg = {
            "allowed_samples": g("allowed_samples", ALLOWED_SAMPLES),
            "epochs": g("epochs", FINETUNE_EPOCHS),
            "batch_size": g("batch_size", BATCH_SIZE),
            "grad_accum_steps": g("grad_accum_steps", GRAD_ACCUM_STEPS),
            "d_model": g("d_model", D_MODEL),
            "num_layers": g("num_layers", NUM_LAYERS),
            "num_heads": g("num_heads", NUM_HEADS),
            "d_ff": g("d_ff", D_FF),
            "dropout": g("dropout", DROPOUT),
            "use_grad_ckpt": g("use_grad_ckpt", USE_GRAD_CHECKPOINTING),
            "use_shortcut": g("use_shortcut", USE_SHORTCUT),
            "use_dist_bias": g("use_dist_bias", USE_DISTANCE_BIAS),
            "use_motif_mask": g("use_motif_mask", USE_MOTIF_MASK),
            "motif_mask_threshold": g("motif_mask_threshold", MOTIF_MASK_THRESH),
            "motif_prior_scale": g("motif_prior_scale", MOTIF_PRIOR_SCALE),
            "shortcut_l1": g("shortcut_l1", SHORTCUT_L1),
            "shortcut_l2": g("shortcut_l2", SHORTCUT_L2),
            "shortcut_topk": g("shortcut_topk", SHORTCUT_TOPK),
            "shortcut_dropout": g("shortcut_dropout", SHORTCUT_DROPOUT),
            "lr": FINETUNE_LR,
        }

        fine_tune_dir = FINE_TUNING_DIR / "fine_tuning"
        os.makedirs(fine_tune_dir, exist_ok=True)

        for vocab_name in ("tf_vocab.json", "tg_vocab.json"):
            dest = fine_tune_dir / vocab_name
            if dest.exists():
                continue
            for cand in (FINE_TUNING_DIR / vocab_name, COMMON_DATA / vocab_name):
                if cand.is_file():
                    shutil.copy(cand, dest)
                    break

        pseudobulk_dataset, single_cell_datasets, model, optimizer, tf_scaler_loaded, tg_scaler_loaded = load_train_objs(
            run_cfg
        )
        rank_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        model = model.to(rank_device)

        pseudobulk_loader, _, _ = prepare_dataloader(
            pseudobulk_dataset, batch_size, world_size, rank
        )

        fisher_bundle_candidates = [FINE_TUNING_DIR / "ewc_bundle.pth", FINE_TUNING_DIR / "ewc_bundle.pt"]
        fisher_bundle_path = next((p for p in fisher_bundle_candidates if p.exists()), fisher_bundle_candidates[0])
        if fisher_bundle_path.exists():
            ref_params, fisher_diag = ewc_utils.load_ewc_bundle(fisher_bundle_path, device=rank_device)
            total_size = len(pseudobulk_dataset)
            if rank == 0:
                logging.info(f"Loaded existing Fisher/EWC bundle: {fisher_bundle_path}")
                print("Example ref_params keys:", list(ref_params.keys())[:5])
                print("Example model keys:", [n for n, _ in list(model.named_parameters())[:5]])
        else:
            if rank == 0:
                logging.info("Computing Fisher matrix on pseudobulk dataset...")
            fisher_diag = ewc_utils.compute_fisher_diag(
                model, pseudobulk_loader, device=rank_device, n_batches=100
            )
            ref_params = {n: p.detach().clone().to(rank_device) for n, p in model.named_parameters()}
            total_size = len(pseudobulk_dataset)
            if rank == 0:
                ewc_utils.save_ewc_bundle(fisher_bundle_path, model, fisher_diag)
                logging.info(f"Saved Fisher/EWC bundle to {fisher_bundle_path}")

        fisher_diag = {k: v.to(rank_device) for k, v in fisher_diag.items()} if fisher_diag is not None else None
        ref_params = {k: v.to(rank_device) for k, v in ref_params.items()} if ref_params is not None else None

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
        
        
        # if rank == 0:
        #     for sample_name, test_loader in test_loaders.items():
        #         write_experiment_settings_and_objects(FINE_TUNING_DIR, sample_name, single_cell_datasets, test_loader, world_size, run_cfg)
        #         logging.info("Wrote experiment settings and objects to training output directory")

        if tf_scaler_loaded is not None and tg_scaler_loaded is not None:
            tf_scaler = SimpleScaler(tf_scaler_loaded.mean.to(rank_device), tf_scaler_loaded.std.to(rank_device))
            tg_scaler = SimpleScaler(tg_scaler_loaded.mean.to(rank_device), tg_scaler_loaded.std.to(rank_device))
        else:
            # Use GLOBAL TF/TG vocab sizes (match tf_name2id/tg_name2id mapping)
            if getattr(pseudobulk_dataset, "tf_name2id", None) is not None:
                T = len(pseudobulk_dataset.tf_name2id)
            else:
                # fallback: infer from max id
                T = int(pseudobulk_dataset.tf_ids.max().item() + 1)

            if getattr(pseudobulk_dataset, "tg_name2id", None) is not None:
                G = len(pseudobulk_dataset.tg_name2id)
            else:
                G = int(pseudobulk_dataset.tg_ids.max().item() + 1)

            combined_loader = itertools.chain.from_iterable(train_loaders.values())
            use_ddp_reduce = torch.distributed.is_initialized()
            tf_s, tg_s = fit_simple_scalers(
                combined_loader,
                T_expected=T,
                G_expected=G,
                device_for_reduce=rank_device,
                use_ddp_reduce=use_ddp_reduce,
            )
            tf_scaler = SimpleScaler(tf_s.mean.to(rank_device), tf_s.std.to(rank_device))
            tg_scaler = SimpleScaler(tg_s.mean.to(rank_device), tg_s.std.to(rank_device))

        trainer = Trainer(
            model,
            train_loaders,
            val_loaders,
            nn.MSELoss(),
            optimizer,
            gpu_id=local_rank,
            global_rank=rank,
            save_every=save_every,
            patience=FINETUNE_PATIENCE,
            grad_accum_steps=run_cfg["grad_accum_steps"],
            use_grad_accumulation=USE_GRAD_ACCUMULATION,
            ref_params=ref_params,
            fisher_diag=fisher_diag,
            lambda_ewc=EWC_LAMBDA,
        )
        trainer.tf_scaler = tf_scaler
        trainer.tg_scaler = tg_scaler

        if rank == 0:
            write_run_parameters(pseudobulk_dataset, fine_tune_dir, world_size, run_cfg)
            logging.info("Wrote experiment settings and objects to fine-tuning directory")
            logging.info("----- ROUND-ROBIN TRAINING STARTED -----")

        trainer.train(max_epochs=total_epochs, path=str(fine_tune_dir))

        model_for_eval = getattr(trainer.model, "module", trainer.model).to(rank_device)

        if rank == 0:
            torch.save(
                {
                    "epoch": total_epochs - 1,
                    "model_state_dict": model_for_eval.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "scheduler_state_dict": trainer.scheduler.state_dict(),
                    "best_val_loss": trainer.best_val_loss,
                    "tf_scaler_mean": trainer.tf_scaler.mean,
                    "tf_scaler_std": trainer.tf_scaler.std,
                    "tg_scaler_mean": trainer.tg_scaler.mean,
                    "tg_scaler_std": trainer.tg_scaler.std,
                },
                fine_tune_dir / "trained_model.pt",
            )
            save_tf_tg_embeddings_from_model(model_for_eval, out_dir=fine_tune_dir, vocab_dir=fine_tune_dir)
            logging.info("Saved final fine-tuned model and embeddings")

        len_all = sum(len(ds) for ds in single_cell_datasets)
        new_fisher_accum = None
        fisher_size_accum = 0
        model_for_fisher = getattr(model_for_eval, "module", model_for_eval)

        for ds, name in zip(single_cell_datasets, FINE_TUNING_DATASETS):
            train_loader_ds, _, _ = prepare_dataloader(ds, batch_size, world_size, rank)
            fisher_ds = ewc_utils.compute_fisher_diag(model_for_fisher, train_loader_ds, device=rank_device, n_batches=100)
            fisher_size = len(ds)
            if new_fisher_accum is None:
                new_fisher_accum = fisher_ds
                fisher_size_accum = fisher_size
            else:
                new_fisher_accum = ewc_utils.merge_fishers(
                    new_fisher_accum, fisher_ds, fisher_size_accum, fisher_size
                )
                fisher_size_accum += fisher_size

        if fisher_diag is None:
            fisher_diag = new_fisher_accum
        else:
            fisher_diag = ewc_utils.merge_fishers(fisher_diag, new_fisher_accum, total_size, len_all)
        ref_params = {n: p.detach().clone() for n, p in model_for_fisher.named_parameters()}

        if rank == 0:
            ewc_bundle_out = fine_tune_dir / "ewc_bundle.pth"
            ewc_utils.save_ewc_bundle(ewc_bundle_out, model_for_fisher, fisher_diag)
            logging.info(f"Saved fine-tuned Fisher/EWC bundle to {ewc_bundle_out}")
            logging.info("\nFine-tuning complete on all datasets (round-robin).")

    finally:
        if dist.is_initialized():
            dist.barrier()
            if rank == 0:
                logging.info("\nDestroying process group")
            dist.destroy_process_group()


if __name__ == "__main__":
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    main(
        rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        save_every=SAVE_EVERY_N_EPOCHS,
        total_epochs=FINETUNE_EPOCHS,
        batch_size=BATCH_SIZE,
    )
