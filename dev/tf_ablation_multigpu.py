from collections import defaultdict
# transformer_testing.py
import os, sys, json
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib.ticker import FuncFormatter, MultipleLocator
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

import torch.distributed as dist
from torch.amp import autocast

import sys
PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
SRC_DIR = str(Path(PROJECT_DIR) / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from datetime import datetime
from config.settings_hpc import *

logging.basicConfig(level=logging.INFO, format='%(message)s')

from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.datasets.dataset import MultiChromosomeDataset, SimpleScaler, fit_simple_scalers

def setup_distributed():
    """Initialize distributed env if launched with torchrun; otherwise run in single-process mode."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank, distributed

argparser = argparse.ArgumentParser(description="Compute gradient-based TF->TG attributions")
argparser.add_argument("--selected_experiment_dir", type=str, required=True,
                       help="Name of the experiment directory (under experiments/mESC_no_scale_linear/)")
argparser.add_argument("--disable_amp", action="store_true",
                       help="Disable mixed-precision inference (defaults to enabled on CUDA)")
args = argparser.parse_args()

SELECTED_EXPERIMENT_DIR = Path(args.selected_experiment_dir)

rank, world_size, local_rank, distributed = setup_distributed()

if torch.cuda.is_available():
    device = torch.device(f"cuda:{local_rank}")
else:
    device = torch.device("cpu")

use_amp = (device.type == "cuda" and not args.disable_amp)

# 1) Load test loader and checkpoint
test_loader = torch.load(SELECTED_EXPERIMENT_DIR / "test_loader.pt", weights_only=False)

ckpt_path = os.path.join(SELECTED_EXPERIMENT_DIR, "trained_model.pt")
state = torch.load(ckpt_path, map_location="cpu")

# 2) Recreate model EXACTLY as in training
model = MultiomicTransformer(
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
    tf_vocab_size=len(state["tf_scaler_mean"]),
    tg_vocab_size=len(state["tg_scaler_mean"]),
    use_bias=USE_DISTANCE_BIAS,
    use_shortcut=USE_SHORTCUT,
    use_motif_mask=USE_MOTIF_MASK,
    use_edge_head=True,
    edge_extra_dim=0,
    edge_hidden_dim=128,
)

if isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)

# 3) Rebuild scalers on the SAME DEVICE as inputs
tg_scaler = SimpleScaler(
    mean=torch.as_tensor(state["tg_scaler_mean"], device=device, dtype=torch.float32),
    std=torch.as_tensor(state["tg_scaler_std"],  device=device, dtype=torch.float32),
)
tf_scaler = SimpleScaler(
    mean=torch.as_tensor(state["tf_scaler_mean"], device=device, dtype=torch.float32),
    std=torch.as_tensor(state["tf_scaler_std"],  device=device, dtype=torch.float32),
)

T_total = len(state["tf_scaler_mean"])   # total TF vocab size
G_total = len(state["tg_scaler_mean"])   # total TG vocab size

# Accumulators for TF knockout effect (CPU)
effect_sum   = torch.zeros(T_total, G_total, dtype=torch.float64)
effect_count = torch.zeros_like(effect_sum)

model.to(device).eval()

max_batches = None  # still allowed

iterator = test_loader
if rank == 0:
    iterator = tqdm(test_loader, desc="TF knockout (full model, optimized)", unit="batches", total=max_batches)

with torch.no_grad():
    for b_idx, batch in enumerate(iterator):
        if (max_batches is not None) and (b_idx >= max_batches):
            break

        # Only process batches assigned to this rank
        if b_idx % world_size != rank:
            continue
        
        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
        atac_wins  = atac_wins.to(device, non_blocking=True)
        tf_tensor  = tf_tensor.to(device, non_blocking=True)   # unscaled
        bias       = bias.to(device, non_blocking=True)
        tf_ids     = tf_ids.to(device)
        tg_ids     = tg_ids.to(device)
        motif_mask = motif_mask.to(device)

        # Shape of TF input
        if tf_tensor.dim() == 2:
            B, T_eval = tf_tensor.shape
            F_dim = 1
        else:
            B, T_eval, F_dim = tf_tensor.shape

        # --------- 1) Scale TFs ONCE per batch ---------
        if tf_scaler is not None:
            tf_scaled_base = tf_scaler.transform(tf_tensor, tf_ids)  # [B, T_eval] or [B, T_eval, F_dim]
        else:
            tf_scaled_base = tf_tensor

        # Work internally as 3D: [B, T_eval, F_dim]
        if tf_tensor.dim() == 2:
            tf_scaled_base_3d = tf_scaled_base.unsqueeze(-1)  # [B, T_eval, 1]
        else:
            tf_scaled_base_3d = tf_scaled_base               # [B, T_eval, F_dim]

        # --------- 2) Scaled value for "expression = 0" per TF (per position) ---------
        if tf_scaler is not None:
            # Build a single zero-expression tensor and transform once
            zeros_expr_1 = torch.zeros_like(tf_tensor[:1])   # [1, T_eval] or [1, T_eval, F_dim]
            zeros_scaled_1 = tf_scaler.transform(zeros_expr_1, tf_ids)  # [1, T_eval] or [1, T_eval, F_dim]

            if tf_tensor.dim() == 2:
                # [1, T_eval] -> [T_eval, 1]
                zeros_scaled = zeros_scaled_1.squeeze(0).unsqueeze(-1)  # [T_eval, 1]
            else:
                # [1, T_eval, F_dim] -> [T_eval, F_dim]
                zeros_scaled = zeros_scaled_1.squeeze(0)                # [T_eval, F_dim]
        else:
            zeros_scaled = torch.zeros(
                (T_eval, F_dim), device=device, dtype=tf_tensor.dtype
            )  # KO really is 0 in model input space

        # --------- 3) Baseline predictions (once per batch) ---------
        with autocast(device_type="cuda", enabled=use_amp):
            # Use original scaled base (2D/3D) for the model
            tf_scaled_for_model = tf_scaled_base if tf_tensor.dim() == 2 else tf_scaled_base_3d

            preds_base_s, _, _, _ = model(
                atac_wins, tf_scaled_for_model,
                tf_ids=tf_ids, tg_ids=tg_ids,
                bias=bias, motif_mask=motif_mask,
                return_edge_logits=True, return_shortcut_contrib=False,
                edge_extra_features=None,
            )

            if tg_scaler is not None:
                preds_base_u = tg_scaler.inverse_transform(preds_base_s, tg_ids)
            else:
                preds_base_u = preds_base_s

        preds_base_u = torch.nan_to_num(
            preds_base_u.float(), nan=0.0, posinf=1e6, neginf=-1e6
        )  # [B, G_eval]
        B, G_eval = preds_base_u.shape

        # CPU IDs once per batch
        tf_ids_cpu = tf_ids.detach().cpu().long()
        tg_ids_cpu = tg_ids.detach().cpu().long()

        # --------- 4) Prepare a working scaled TF tensor (cloned ONCE) ---------
        # We will modify one TF position at a time, run the model, then restore.
        tf_scaled_work = tf_scaled_base_3d.clone()  # [B, T_eval, F_dim]
        
        zero_eps = 1e-8  # tweak if needed

        for t_pos in range(T_eval):
            # ----------------------------------
            # 5a) Skip positions that are ~zero
            # ----------------------------------
            if tf_tensor.dim() == 2:
                unscaled_slice = tf_tensor[:, t_pos]          # [B]
            else:
                # if F_dim>1, first feature is usually expression;
                # adjust if your layout is different
                unscaled_slice = tf_tensor[:, t_pos, 0]       # [B]

            if torch.all(unscaled_slice.abs() < zero_eps):
                # KO is identical to baseline -> no effect, no need to run the model
                continue

            # ----------------------------------
            # 5b) Apply KO in *scaled* space
            # ----------------------------------
            # zeros_scaled[t_pos]: [F_dim]
            ko_val = zeros_scaled[t_pos].unsqueeze(0).expand(B, F_dim)  # [B, F_dim]
            tf_scaled_work[:, t_pos, :] = ko_val

            # Match original dimensionality for the model input
            if tf_tensor.dim() == 2:
                tf_scaled_input = tf_scaled_work.squeeze(-1)   # [B, T_eval]
            else:
                tf_scaled_input = tf_scaled_work              # [B, T_eval, F_dim]

            # Run knockout forward pass
            with autocast(device_type="cuda", enabled=use_amp):
                preds_ko_s, _, _, _ = model(
                    atac_wins, tf_scaled_input,
                    tf_ids=tf_ids, tg_ids=tg_ids,
                    bias=bias, motif_mask=motif_mask,
                    return_edge_logits=True, return_shortcut_contrib=False,
                    edge_extra_features=None,
                )

                if tg_scaler is not None:
                    preds_ko_u = tg_scaler.inverse_transform(preds_ko_s, tg_ids)
                else:
                    preds_ko_u = preds_ko_s

            preds_ko_u = torch.nan_to_num(
                preds_ko_u.float(), nan=0.0, posinf=1e6, neginf=-1e6
            )  # [B, G_eval]

            # delta = baseline - knockout (positive: TF supports expression)
            delta = preds_base_u - preds_ko_u          # [B, G_eval]
            delta_mean = delta.mean(dim=0)             # [G_eval]

            tf_global = int(tf_ids_cpu[t_pos].item())
            effect_sum[tf_global, tg_ids_cpu]   += delta_mean.cpu().to(torch.float64)
            effect_count[tf_global, tg_ids_cpu] += 1

            # ----------------------------------
            # 5c) Restore baseline slice *without* cloning
            # ----------------------------------
            tf_scaled_work[:, t_pos, :] = tf_scaled_base_3d[:, t_pos, :]

# Final average TF→TG knockout effect (expression units)
# Convert this rank's accumulators to numpy
effect_sum_np = effect_sum.numpy()
effect_count_np = effect_count.numpy()

# Save per-rank partial results
rank_effect_path  = SELECTED_EXPERIMENT_DIR / f"tf_tg_fullmodel_knockout_rank{rank}.npy"
rank_count_path   = SELECTED_EXPERIMENT_DIR / f"tf_tg_fullmodel_knockout_count_rank{rank}.npy"

np.save(rank_effect_path, effect_sum_np)
np.save(rank_count_path, effect_count_np)

if distributed:
    dist.barrier()  # make sure all ranks finished writing

# Only rank 0 merges everything
if rank == 0:
    total_effect_sum = effect_sum_np.copy()
    total_effect_count = effect_count_np.copy()

    for r in range(1, world_size):
        es = np.load(SELECTED_EXPERIMENT_DIR / f"tf_tg_fullmodel_knockout_rank{r}.npy")
        ec = np.load(SELECTED_EXPERIMENT_DIR / f"tf_tg_fullmodel_knockout_count_rank{r}.npy")
        total_effect_sum   += es
        total_effect_count += ec

    tf_tg_effect = total_effect_sum / (total_effect_count + 1e-12)
    tf_tg_effect_np = tf_tg_effect

    print("TF–TG full-model knockout effect matrix shape:", tf_tg_effect_np.shape)
    np.save(SELECTED_EXPERIMENT_DIR / "tf_tg_fullmodel_knockout.npy", tf_tg_effect_np)
    np.save(
        SELECTED_EXPERIMENT_DIR / "tf_tg_fullmodel_knockout_count.npy",
        total_effect_count
    )

# Clean up distributed if needed
if distributed:
    dist.destroy_process_group()
