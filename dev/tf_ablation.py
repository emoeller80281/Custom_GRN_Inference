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

T_total = len(state["tf_scaler_mean"])   # total TF vocab size
G_total = len(state["tg_scaler_mean"])   # total TG vocab size

# Accumulators for TF knockout effect
effect_sum   = torch.zeros(T_total, G_total, dtype=torch.float64)
effect_count = torch.zeros_like(effect_sum)

model.to(device).eval()

max_batches = None  # or set to e.g. 50 for a quick approximate matrix

with torch.no_grad():
    for b_idx, batch in enumerate(tqdm(test_loader, desc="TF knockout (full model)", unit="batches", total=max_batches)):
        if (max_batches is not None) and (b_idx >= max_batches):
            if b_idx >= max_batches:
                break

        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
        atac_wins  = atac_wins.to(device)
        tf_tensor  = tf_tensor.to(device)   # keep unscaled copy
        bias       = bias.to(device)
        tf_ids     = tf_ids.to(device)
        tg_ids     = tg_ids.to(device)
        motif_mask = motif_mask.to(device)

        # ----- baseline predictions -----
        if tf_scaler is not None:
            tf_scaled_base = tf_scaler.transform(tf_tensor, tf_ids)
        else:
            tf_scaled_base = tf_tensor

        preds_base_s, _, _, _ = model(
            atac_wins, tf_scaled_base,
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
        )

        B, G_eval = preds_base_u.shape

        # Shape of TF input
        if tf_tensor.dim() == 2:
            B_tf, T_eval = tf_tensor.shape
            F_dim = 1
        else:
            B_tf, T_eval, F_dim = tf_tensor.shape
        assert B_tf == B, "Batch size mismatch"

        # Move ids to CPU once per batch
        tf_ids_cpu = tf_ids.detach().cpu().long()
        tg_ids_cpu = tg_ids.detach().cpu().long()

        # ----- loop over TF positions: zero that TF, re-run model -----
        for t_pos in range(T_eval):
            # clone original TF tensor and zero this TF across the batch
            tf_ko = tf_tensor.clone()
            if tf_ko.dim() == 2:
                tf_ko[:, t_pos] = 0.0
            else:
                tf_ko[:, t_pos, :] = 0.0

            if tf_scaler is not None:
                tf_scaled_ko = tf_scaler.transform(tf_ko, tf_ids)
            else:
                tf_scaled_ko = tf_ko

            preds_ko_s, _, _, _ = model(
                atac_wins, tf_scaled_ko,
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
            )

            # delta = baseline - knockout  (positive: TF supports expression)
            delta = preds_base_u - preds_ko_u      # [B, G_eval]
            delta_mean = delta.mean(dim=0)         # [G_eval], average over batch

            tf_global = int(tf_ids_cpu[t_pos].item())
            # accumulate this TF's effect vector into global matrix row
            effect_sum[tf_global, tg_ids_cpu]   += delta_mean.cpu().to(torch.float64)
            effect_count[tf_global, tg_ids_cpu] += 1

# Final average TF→TG knockout effect (expression units)
tf_tg_effect = effect_sum / (effect_count + 1e-12)
tf_tg_effect_np = tf_tg_effect.cpu().numpy()

print("TF–TG full-model knockout effect matrix shape:", tf_tg_effect_np.shape)
np.save(SELECTED_EXPERIMENT_DIR / "tf_tg_fullmodel_knockout.npy", tf_tg_effect_np)
np.save(SELECTED_EXPERIMENT_DIR / "tf_tg_fullmodel_knockout_count.npy",
        effect_count.cpu().numpy())