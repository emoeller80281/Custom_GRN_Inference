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

import sys
PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
SRC_DIR = str(Path(PROJECT_DIR) / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from datetime import datetime
from config.settings_hpc import *

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------


from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.datasets.dataset import MultiChromosomeDataset, SimpleScaler, fit_simple_scalers

argparser = argparse.ArgumentParser(description="Compute gradient-based TF->TG attributions")
argparser.add_argument("--selected_experiment_dir", type=str, required=True,
                       help="Name of the experiment directory (under experiments/mESC_no_scale_linear/)")
args = argparser.parse_args()

SELECTED_EXPERIMENT_DIR = Path(args.selected_experiment_dir)

GROUND_TRUTH_DIR = os.path.join(PROJECT_DIR, "data/ground_truth_files")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) Rebuild scalers on the SAME DEVICE as inputs
tg_scaler = SimpleScaler(
    mean=torch.as_tensor(state["tg_scaler_mean"], device=device, dtype=torch.float32),
    std=torch.as_tensor(state["tg_scaler_std"],  device=device, dtype=torch.float32),
)
tf_scaler = SimpleScaler(
    mean=torch.as_tensor(state["tf_scaler_mean"], device=device, dtype=torch.float32),
    std=torch.as_tensor(state["tf_scaler_std"],  device=device, dtype=torch.float32),
)

T_total = len(state["tf_scaler_mean"])
G_total = len(state["tg_scaler_mean"])

# Put accumulators on the same device as the model
grad_sum   = torch.zeros(T_total, G_total, dtype=torch.float32, device=device)
grad_count = torch.zeros_like(grad_sum)

model.to(device).eval()

max_batches = 50

for b_idx, batch in enumerate(tqdm(test_loader, desc="Gradient attributions", unit="batches", total=max_batches)):
    if b_idx >= max_batches:
        break

    atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
    atac_wins  = atac_wins.to(device)
    tf_tensor  = tf_tensor.to(device).requires_grad_(True)
    bias       = bias.to(device)
    tf_ids     = tf_ids.to(device)
    tg_ids     = tg_ids.to(device)
    motif_mask = motif_mask.to(device)

    # scale inputs / run model in *scaled* space
    tf_scaled = tf_scaler.transform(tf_tensor, tf_ids) if tf_scaler is not None else tf_tensor

    preds_s, _, _, _ = model(
        atac_wins, tf_scaled,
        tf_ids=tf_ids, tg_ids=tg_ids,
        bias=bias, motif_mask=motif_mask,
        return_edge_logits=True, return_shortcut_contrib=False,
        edge_extra_features=None,
    )

    preds_u = tg_scaler.inverse_transform(preds_s, tg_ids) if tg_scaler is not None else preds_s
    preds_u = torch.nan_to_num(preds_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)

    B, G_eval = preds_u.shape

    # Shape of TF input
    if tf_tensor.dim() == 2:
        B_tf, T_eval = tf_tensor.shape
        F_dim = 1
    else:
        B_tf, T_eval, F_dim = tf_tensor.shape
    assert B_tf == B

    # Precompute TF ids in a flat form for index_add
    if tf_ids.dim() == 1:
        # [T_eval] -> [B*T_eval] by repeating for each sample
        tf_ids_flat = tf_ids.view(1, T_eval).expand(B, T_eval).reshape(-1)
    else:
        # [B, T_eval] -> [B*T_eval]
        tf_ids_flat = tf_ids.reshape(-1)

    for j in range(G_eval):
        # Zero grads
        model.zero_grad(set_to_none=True)
        if tf_tensor.grad is not None:
            tf_tensor.grad.zero_()

        # scalar for this TG across batch
        y_col = preds_u[:, j].sum()
        y_col.backward(retain_graph=True)

        grads = tf_tensor.grad

        # Aggregate over TF feature dims → [B, T_eval]
        if grads.dim() == 3:
            grad_abs = grads.abs().sum(dim=-1)
        elif grads.dim() == 2:
            grad_abs = grads.abs()
        else:
            raise RuntimeError(f"Unexpected grads shape: {grads.shape}")

        # Flatten grad_abs to [B*T_eval]
        grad_flat = grad_abs.reshape(-1).to(grad_sum.dtype)

        # Global TG id for this column
        if tg_ids.dim() == 1:
            tg_global = int(tg_ids[j].item())
        else:
            # if different per sample, you *cannot* store one column per tg_global;
            # typical setup is 1D, so we keep this simple:
            tg_global = int(tg_ids[0, j].item())

        # Column views in the big matrices
        col_grad   = grad_sum[:, tg_global]   # [T_total]
        col_count  = grad_count[:, tg_global] # [T_total]

        # Accumulate gradients for this TG:
        # each tf_ids_flat[k] gives a row index, grad_flat[k] the value
        col_grad.index_add_(0, tf_ids_flat, grad_flat)

        # Each (batch, position) is one "observation"
        ones_flat = torch.ones_like(grad_flat)
        col_count.index_add_(0, tf_ids_flat, ones_flat)

# Final average gradient attribution matrix [T_total, G_total]
grad_attr = grad_sum / (grad_count + 1e-12)

grad_attr_np = grad_attr.detach().cpu().numpy()
print("Gradient attribution matrix shape:", grad_attr_np.shape)
np.save(SELECTED_EXPERIMENT_DIR / "tf_tg_grad_attribution.npy", grad_attr_np)
