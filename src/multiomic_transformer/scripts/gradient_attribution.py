from collections import defaultdict
import os, sys, json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
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

def setup_distributed():
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
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=device,
        )
    return rank, world_size, local_rank, distributed


def load_model(selected_experiment_dir, checkpoint_file, device):
    params_path = selected_experiment_dir / "run_parameters.json"
    with open(params_path, "r") as f:
        params = json.load(f)

    # Pull out architecture hyperparameters
    d_model   = params.get("d_model")
    num_heads = params.get("num_heads")
    num_layers = params.get("num_layers")
    d_ff      = params.get("d_ff")
    dropout   = params.get("dropout", 0.0)
    use_shortcut   = params.get("use_shortcut", False)
    use_dist_bias  = params.get("use_dist_bias", False)
    use_motif_mask = params.get("use_motif_mask", False)

    # 1) Load test loader and checkpoint
    test_loader = torch.load(selected_experiment_dir / "test_loader.pt", weights_only=False)

    ckpt_path = os.path.join(selected_experiment_dir, checkpoint_file)
    state = torch.load(ckpt_path, map_location="cpu")
    
    # 2) Recreate model EXACTLY as in training
    model = MultiomicTransformer(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        tf_vocab_size=len(state["tf_scaler_mean"]),
        tg_vocab_size=len(state["tg_scaler_mean"]),
        use_bias=use_dist_bias,
        use_shortcut=use_shortcut,
        use_motif_mask=use_motif_mask,
        use_edge_head=False,
        edge_extra_dim=0,
        edge_hidden_dim=128,
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device).eval()

    # 3) Rebuild scalers on the SAME DEVICE as inputs
    tg_scaler = SimpleScaler(
        mean=torch.as_tensor(state["tg_scaler_mean"], device=device, dtype=torch.float32),
        std=torch.as_tensor(state["tg_scaler_std"],  device=device, dtype=torch.float32),
    )
    tf_scaler = SimpleScaler(
        mean=torch.as_tensor(state["tf_scaler_mean"], device=device, dtype=torch.float32),
        std=torch.as_tensor(state["tf_scaler_std"],  device=device, dtype=torch.float32),
    )

    return model, test_loader, tg_scaler, tf_scaler, state

def run_gradient_attribution(
    selected_experiment_dir,
    model,
    test_loader,
    tg_scaler,
    tf_scaler,
    state,
    device,
    use_amp,
    rank,
    world_size,
    distributed,
    method: str = "saliency",          # "saliency" | "smoothgrad" | "ig"
    smoothgrad_samples: int = 8,
    smoothgrad_noise_std: float = 0.05,
    ig_steps: int = 8,
    max_batches: int = None,
    use_dataloader: bool = False,
):

    assert method in {"saliency", "smoothgrad", "ig"}, f"Unknown method: {method}"

    T_total = len(state["tf_scaler_mean"])
    G_total = len(state["tg_scaler_mean"])

    grad_sum = torch.zeros(T_total, G_total, device=device, dtype=torch.float32)
    grad_count = torch.zeros_like(grad_sum)

    model.to(device).eval()

    iterator = test_loader
    if rank == 0:
        iterator = tqdm(
            test_loader,
            desc=f"Gradient attributions ({method})",
            unit="batches",
            total=max_batches,
        )

    for b_idx, batch in enumerate(iterator):
        if max_batches is not None and b_idx >= max_batches:
            break

        # manual sharding if not using a DistributedSampler
        if not use_dataloader and (b_idx % world_size != rank):
            continue

        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
        atac_wins = atac_wins.to(device)
        tf_tensor = tf_tensor.to(device)
        bias = bias.to(device)
        tf_ids = tf_ids.to(device)
        tg_ids = tg_ids.to(device)
        motif_mask = motif_mask.to(device)

        # Shapes
        if tf_tensor.dim() == 2:
            B, T_eval = tf_tensor.shape
            F_dim = 1
        else:
            B, T_eval, F_dim = tf_tensor.shape

        # Flatten TF IDs over batch for aggregation later
        if tf_ids.dim() == 1:  # [T_eval]
            tf_ids_flat = tf_ids.view(1, T_eval).expand(B, T_eval).reshape(-1)
        else:                  # [B, T_eval]
            tf_ids_flat = tf_ids.reshape(-1)

        # One base forward to get G_eval (number of TGs)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            tf_scaled_base = (
                tf_scaler.transform(tf_tensor, tf_ids)
                if tf_scaler is not None
                else tf_tensor
            )
            preds_s_base, _, _, _ = model(
                atac_wins,
                tf_scaled_base,
                tf_ids=tf_ids,
                tg_ids=tg_ids,
                bias=bias,
                motif_mask=motif_mask,
                return_edge_logits=True,
                return_shortcut_contrib=False,
                edge_extra_features=None,
            )
            preds_u_base = (
                tg_scaler.inverse_transform(preds_s_base, tg_ids)
                if tg_scaler is not None
                else preds_s_base
            )
            preds_u_base = torch.nan_to_num(
                preds_u_base.float(), nan=0.0, posinf=1e6, neginf=-1e6
            )

        _, G_eval = preds_u_base.shape

        # ---------- METHOD 1: plain saliency (grad * input) ----------
        if method == "saliency":
            # Need gradients w.r.t. tf_tensor
            tf_tensor = tf_tensor.detach().requires_grad_(True)

            # Re-run forward with grad-tracking on tf_tensor
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                tf_scaled = (
                    tf_scaler.transform(tf_tensor, tf_ids)
                    if tf_scaler is not None
                    else tf_tensor
                )
                preds_s, _, _, _ = model(
                    atac_wins,
                    tf_scaled,
                    tf_ids=tf_ids,
                    tg_ids=tg_ids,
                    bias=bias,
                    motif_mask=motif_mask,
                    return_edge_logits=True,
                    return_shortcut_contrib=False,
                    edge_extra_features=None,
                )
                preds_u = (
                    tg_scaler.inverse_transform(preds_s, tg_ids)
                    if tg_scaler is not None
                    else preds_s
                )
                preds_u = torch.nan_to_num(
                    preds_u.float(), nan=0.0, posinf=1e6, neginf=-1e6
                )

            for j in range(G_eval):
                grad_output_j = torch.zeros_like(preds_u)
                grad_output_j[:, j] = 1.0

                retain = (j < G_eval - 1)

                grads = torch.autograd.grad(
                    outputs=preds_u,
                    inputs=tf_tensor,
                    grad_outputs=grad_output_j,
                    retain_graph=retain,
                    create_graph=False,
                    allow_unused=False,
                )[0]

                # grad * input (expression channel)
                if grads.dim() == 3:
                    expr_grad = grads[..., 0]
                    expr_input = tf_tensor[..., 0]
                else:
                    expr_grad = grads
                    expr_input = tf_tensor

                saliency = expr_grad * expr_input  # optionally .abs()

                if saliency.dim() == 3:
                    grad_abs = saliency.sum(dim=-1)
                else:
                    grad_abs = saliency

                grad_flat = grad_abs.reshape(-1)
                tg_global = int(tg_ids[j].item())

                col_grad = grad_sum[:, tg_global]
                col_count = grad_count[:, tg_global]

                col_grad.index_add_(0, tf_ids_flat, grad_flat)
                col_count.index_add_(0, tf_ids_flat, torch.ones_like(grad_flat))

        # ---------- METHOD 2: SmoothGrad ----------
        elif method == "smoothgrad":
            S = max(smoothgrad_samples, 1)

            for j in range(G_eval):
                grad_accum = torch.zeros_like(tf_tensor, dtype=torch.float32)

                for s in range(S):
                    noise = smoothgrad_noise_std * torch.randn_like(tf_tensor)
                    tf_noisy = (tf_tensor + noise).detach().requires_grad_(True)

                    with torch.amp.autocast(
                        device_type=device.type, enabled=use_amp
                    ):
                        tf_scaled = (
                            tf_scaler.transform(tf_noisy, tf_ids)
                            if tf_scaler is not None
                            else tf_noisy
                        )
                        preds_s, _, _, _ = model(
                            atac_wins,
                            tf_scaled,
                            tf_ids=tf_ids,
                            tg_ids=tg_ids,
                            bias=bias,
                            motif_mask=motif_mask,
                            return_edge_logits=True,
                            return_shortcut_contrib=False,
                            edge_extra_features=None,
                        )
                        preds_u = (
                            tg_scaler.inverse_transform(preds_s, tg_ids)
                            if tg_scaler is not None
                            else preds_s
                        )
                        preds_u = torch.nan_to_num(
                            preds_u.float(),
                            nan=0.0,
                            posinf=1e6,
                            neginf=-1e6,
                        )

                    grad_output_j = torch.zeros_like(preds_u)
                    grad_output_j[:, j] = 1.0

                    grads = torch.autograd.grad(
                        outputs=preds_u,
                        inputs=tf_noisy,
                        grad_outputs=grad_output_j,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )[0]

                    if grads.dim() == 3:
                        expr_grad = grads[..., 0]
                        expr_input = tf_noisy[..., 0]
                    else:
                        expr_grad = grads
                        expr_input = tf_noisy

                    saliency = expr_grad * expr_input
                    grad_accum = grad_accum + saliency

                grad_mean = grad_accum / float(S)

                if grad_mean.dim() == 3:
                    grad_abs = grad_mean.sum(dim=-1)
                else:
                    grad_abs = grad_mean

                grad_flat = grad_abs.reshape(-1)
                tg_global = int(tg_ids[j].item())

                col_grad = grad_sum[:, tg_global]
                col_count = grad_count[:, tg_global]

                col_grad.index_add_(0, tf_ids_flat, grad_flat)
                col_count.index_add_(0, tf_ids_flat, torch.ones_like(grad_flat))

        # ---------- METHOD 3: Integrated Gradients ----------
        elif method == "ig":
            M = max(ig_steps, 1)
            baseline = torch.zeros_like(tf_tensor)
            diff = tf_tensor - baseline

            for j in range(G_eval):
                ig_accum = torch.zeros_like(tf_tensor, dtype=torch.float32)

                for m in range(1, M + 1):
                    alpha = float(m) / float(M)
                    tf_step = (baseline + alpha * diff).detach().requires_grad_(True)

                    with torch.amp.autocast(
                        device_type=device.type, enabled=use_amp
                    ):
                        tf_scaled = (
                            tf_scaler.transform(tf_step, tf_ids)
                            if tf_scaler is not None
                            else tf_step
                        )
                        preds_s, _, _, _ = model(
                            atac_wins,
                            tf_scaled,
                            tf_ids=tf_ids,
                            tg_ids=tg_ids,
                            bias=bias,
                            motif_mask=motif_mask,
                            return_edge_logits=True,
                            return_shortcut_contrib=False,
                            edge_extra_features=None,
                        )
                        preds_u = (
                            tg_scaler.inverse_transform(preds_s, tg_ids)
                            if tg_scaler is not None
                            else preds_s
                        )
                        preds_u = torch.nan_to_num(
                            preds_u.float(),
                            nan=0.0,
                            posinf=1e6,
                            neginf=-1e6,
                        )

                    grad_output_j = torch.zeros_like(preds_u)
                    grad_output_j[:, j] = 1.0

                    grads = torch.autograd.grad(
                        outputs=preds_u,
                        inputs=tf_step,
                        grad_outputs=grad_output_j,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=False,
                    )[0]

                    ig_accum = ig_accum + grads

                ig_attr = diff * (ig_accum / float(M))

                if ig_attr.dim() == 3:
                    grad_abs = ig_attr[..., 0]
                else:
                    grad_abs = ig_attr

                grad_flat = grad_abs.reshape(-1)
                tg_global = int(tg_ids[j].item())

                col_grad = grad_sum[:, tg_global]
                col_count = grad_count[:, tg_global]

                col_grad.index_add_(0, tf_ids_flat, grad_flat)
                col_count.index_add_(0, tf_ids_flat, torch.ones_like(grad_flat))

        # cleanup
        del (
            preds_u_base,
            preds_s_base,
            tf_scaled_base,
            atac_wins,
            tf_tensor,
            bias,
            tf_ids,
            tg_ids,
            motif_mask,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # distributed reduction
    if distributed:
        dist.barrier()
        dist.all_reduce(grad_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(grad_count, op=dist.ReduceOp.SUM)

    if rank == 0:
        grad_attr = grad_sum / (grad_count + 1e-12)
        grad_attr_np = grad_attr.detach().cpu().numpy()

        print("Gradient attribution matrix shape:", grad_attr_np.shape)
        out_path = selected_experiment_dir / f"tf_tg_grad_attribution_{method}.npy"
        np.save(out_path, grad_attr_np)
        logging.info(f"Saved gradient attribution matrix to {out_path}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Compute gradient-based TF->TG attributions"
    )
    argparser.add_argument(
        "--selected_experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory",
    )
    argparser.add_argument(
        "--max_batches",
        default=None,
        type=int,
        help="Maximum number of batches to process (for debugging, defaults to all)",
    )
    argparser.add_argument(
        "--model_file",
        default="trained_model.pt",
        type=str,
        help="File for model checkpoint (default: trained_model.pt)",
    )
    argparser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable mixed-precision inference (defaults to enabled on CUDA)",
    )
    argparser.add_argument(
        "--method",
        type=str,
        default="saliency",
        choices=["saliency", "smoothgrad", "ig"],
        help="Attribution method to use",
    )
    argparser.add_argument(
        "--smoothgrad_samples",
        type=int,
        default=8,
        help="Number of noise samples for SmoothGrad",
    )
    argparser.add_argument(
        "--smoothgrad_noise_std",
        type=float,
        default=0.05,
        help="Noise std for SmoothGrad",
    )
    argparser.add_argument(
        "--ig_steps",
        type=int,
        default=8,
        help="Number of steps for Integrated Gradients",
    )

    args = argparser.parse_args()

    selected_experiment_dir = Path(args.selected_experiment_dir)

    rank, world_size, local_rank, distributed = setup_distributed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = args.use_amp and device.type == "cuda"
    if use_amp and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(device)
        if capability[0] < 7:
            logging.warning(
                f"GPU compute capability {capability[0]}.{capability[1]} < 7.0, disabling AMP"
            )
            use_amp = False

    model, test_loader, tg_scaler, tf_scaler, state = load_model(
        selected_experiment_dir=selected_experiment_dir,
        checkpoint_file=args.model_file,
        device=device,
    )

    run_gradient_attribution(
        selected_experiment_dir=selected_experiment_dir,
        model=model,
        test_loader=test_loader,
        tg_scaler=tg_scaler,
        tf_scaler=tf_scaler,
        state=state,
        device=device,
        use_amp=use_amp,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        method=args.method,
        smoothgrad_samples=args.smoothgrad_samples,
        smoothgrad_noise_std=args.smoothgrad_noise_std,
        ig_steps=args.ig_steps,
        max_batches=args.max_batches,
        use_dataloader=False,
    )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()