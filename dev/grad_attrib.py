from collections import defaultdict
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

from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.datasets.dataset import MultiChromosomeDataset, SimpleScaler, fit_simple_scalers

argparser = argparse.ArgumentParser(description="Compute gradient-based TF->TG attributions (OPTIMIZED)")
argparser.add_argument("--selected_experiment_dir", type=str, required=True,
                       help="Path to experiment directory")
argparser.add_argument("--chunk_size", type=int, default=50,
                       help="Number of TGs to process at once (reduce if OOM)")
argparser.add_argument("--use_amp", action="store_true", default=True,
                       help="Use automatic mixed precision for forward pass (default: True)")
argparser.add_argument("--no_amp", action="store_false", dest="use_amp",
                       help="Disable mixed precision")
args = argparser.parse_args()

SELECTED_EXPERIMENT_DIR = Path(args.selected_experiment_dir)
GROUND_TRUTH_DIR = os.path.join(PROJECT_DIR, "data/ground_truth_files")

# Use consistent device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check GPU capability for mixed precision
use_amp = args.use_amp and device.type == 'cuda'
if use_amp:
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(device)
        if capability[0] < 7:
            logging.warning(f"GPU compute capability {capability[0]}.{capability[1]} < 7.0, disabling AMP")
            use_amp = False
        else:
            logging.info(f"Using AMP for forward pass (gradients computed in FP32)")
    else:
        use_amp = False

logging.info(f"Using device: {device}, Mixed Precision: {use_amp}")

# ---------------------------------------------------------------------
# Load checkpoint and model
# ---------------------------------------------------------------------
test_loader = torch.load(SELECTED_EXPERIMENT_DIR / "test_loader.pt", weights_only=False)
ckpt_path = SELECTED_EXPERIMENT_DIR / "trained_model.pt"
state = torch.load(ckpt_path, map_location="cpu")

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

# Rebuild scalers on device
tg_scaler = SimpleScaler(
    mean=torch.as_tensor(state["tg_scaler_mean"], device=device, dtype=torch.float32),
    std=torch.as_tensor(state["tg_scaler_std"], device=device, dtype=torch.float32),
)
tf_scaler = SimpleScaler(
    mean=torch.as_tensor(state["tf_scaler_mean"], device=device, dtype=torch.float32),
    std=torch.as_tensor(state["tf_scaler_std"], device=device, dtype=torch.float32),
)

T_total = len(state["tf_scaler_mean"])
G_total = len(state["tg_scaler_mean"])

# Accumulators on CPU to save GPU memory
grad_sum = torch.zeros(T_total, G_total, dtype=torch.float32)
grad_count = torch.zeros_like(grad_sum)

model.to(device).eval()

# ---------------------------------------------------------------------
# OPTIMIZED GRADIENT COMPUTATION
# ---------------------------------------------------------------------
max_batches = None
chunk_size = args.chunk_size  # Process this many TGs at once

for b_idx, batch in enumerate(tqdm(test_loader, desc="Gradient attributions", unit="batches", total=max_batches)):
    if max_batches is not None and b_idx >= max_batches:
        break

    atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
    atac_wins = atac_wins.to(device)
    tf_tensor = tf_tensor.to(device).requires_grad_(True)
    bias = bias.to(device)
    tf_ids = tf_ids.to(device)
    tg_ids = tg_ids.to(device)
    motif_mask = motif_mask.to(device)

    # Scale inputs and run forward pass with mixed precision
    # Note: Gradients are automatically computed in FP32 for stability
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
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

    # Get TF shape
    if tf_tensor.dim() == 2:
        B_tf, T_eval = tf_tensor.shape
        F_dim = 1
    else:
        B_tf, T_eval, F_dim = tf_tensor.shape
    assert B_tf == B

    # Precompute TF ids in flat form
    if tf_ids.dim() == 1:
        tf_ids_flat = tf_ids.view(1, T_eval).expand(B, T_eval).reshape(-1)
    else:
        tf_ids_flat = tf_ids.reshape(-1)

    # Move to CPU for indexing
    tf_ids_flat_cpu = tf_ids_flat.cpu()
    tg_ids_cpu = tg_ids.cpu()

    # ---------------------------------------------------------------------
    # OPTIMIZED: Compute gradients more efficiently
    # Compute gradient for each TG output separately
    # ---------------------------------------------------------------------
    
    for j in range(G_eval):
        # Create gradient output for this specific TG
        grad_output_j = torch.zeros_like(preds_u)
        grad_output_j[:, j] = 1.0
        
        # CRITICAL: Only retain graph until the last TG
        # This prevents memory accumulation
        retain = (j < G_eval - 1)
        
        # Compute gradient
        grads_tuple = torch.autograd.grad(
            outputs=preds_u,
            inputs=tf_tensor,
            grad_outputs=grad_output_j,
            retain_graph=retain,  # Free graph on last iteration
            create_graph=False,
            allow_unused=False,
        )
        
        grads = grads_tuple[0]
        
        # Aggregate over TF feature dimensions
        if grads.dim() == 3:
            grad_abs = grads.abs().sum(dim=-1)
        elif grads.dim() == 2:
            grad_abs = grads.abs()
        else:
            raise RuntimeError(f"Unexpected grads shape: {grads.shape}")
        
        # Flatten [B, T_eval] → [B*T_eval]
        grad_flat = grad_abs.reshape(-1).cpu()
        
        # Global TG id
        if tg_ids.dim() == 1:
            tg_global = int(tg_ids_cpu[j].item())
        else:
            tg_global = int(tg_ids_cpu[0, j].item())
        
        # Accumulate on CPU (more memory efficient)
        col_grad = grad_sum[:, tg_global]
        col_count = grad_count[:, tg_global]
        
        col_grad.index_add_(0, tf_ids_flat_cpu, grad_flat)
        col_count.index_add_(0, tf_ids_flat_cpu, torch.ones_like(grad_flat))
    
    # Force cleanup after each batch
    del preds_u, preds_s, tf_scaled, atac_wins, tf_tensor
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# Final average gradient attribution matrix
grad_attr = grad_sum / (grad_count + 1e-12)
grad_attr_np = grad_attr.numpy()

print("Gradient attribution matrix shape:", grad_attr_np.shape)
np.save(SELECTED_EXPERIMENT_DIR / "tf_tg_grad_attribution.npy", grad_attr_np)
logging.info(f"Saved gradient attribution matrix to {SELECTED_EXPERIMENT_DIR / 'tf_tg_grad_attribution.npy'}")