import sys, json, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm
import random
import zlib
import logging
import argparse

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
SRC_DIR = str(Path(PROJECT_DIR) / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import multiomic_transformer.utils.experiment_loader as experiment_loader
import multiomic_transformer.datasets.dataset_refactor as dataset_refactor
import multiomic_transformer.models.model as model_module
from multiomic_transformer.datasets.dataset_refactor import DistributedBatchSampler, MultiChromosomeDataset
from multiomic_transformer.datasets.dataset_refactor import IndexedChromBucketBatchSampler

GROUND_TRUTH_DIR = Path("data/ground_truth_files")

logging.basicConfig(level=logging.INFO, format="%(message)s")

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

def load_ground_truth(ground_truth_file):
    if type(ground_truth_file) == str:
        ground_truth_file = Path(ground_truth_file)
        
    if ground_truth_file.suffix == ".csv":
        sep = ","
    elif ground_truth_file.suffix == ".tsv":
        sep="\t"
        
    ground_truth_df = pd.read_csv(ground_truth_file, sep=sep, on_bad_lines="skip", engine="python")
    
    if "chip" in ground_truth_file.name and "atlas" in ground_truth_file.name:
        ground_truth_df = ground_truth_df[["source_id", "target_id"]]

    if ground_truth_df.columns[0] != "Source" or ground_truth_df.columns[1] != "Target":
        ground_truth_df = ground_truth_df.rename(columns={ground_truth_df.columns[0]: "Source", ground_truth_df.columns[1]: "Target"})
    ground_truth_df["Source"] = ground_truth_df["Source"].astype(str).str.upper()
    ground_truth_df["Target"] = ground_truth_df["Target"].astype(str).str.upper()
    
    # Build TF, TG, and edge sets for quick lookup later
    gt = ground_truth_df[["Source", "Target"]].dropna()

    gt_tfs = set(gt["Source"].unique())
    gt_tgs = set(gt["Target"].unique())
    
    gt_pairs = (gt["Source"] + "\t" + gt["Target"]).drop_duplicates()
    
    gt_lookup = (gt_tfs, gt_tgs, set(gt_pairs))
        
    return ground_truth_df, gt_lookup
def prepare_dataloader(dataset, batch_size, world_size, rank, num_workers=4, seed=42):

    # ---------- Multi-chromosome path ----------
    if isinstance(dataset, MultiChromosomeDataset):
        # 1) Build per-chrom index ranges from dataset._offsets
        chrom_to_indices = {}
        for i, chrom in enumerate(dataset.chrom_ids):
            start = dataset._offsets[i]
            end = dataset._offsets[i + 1] if i + 1 < len(dataset._offsets) else len(dataset)
            if end > start:
                chrom_to_indices[chrom] = list(range(start, end))

        per_chrom_data_map = {}

        for chrom, idxs in chrom_to_indices.items():
            n = len(idxs)
            if n == 0:
                continue

            # deterministic per-chrom shuffle
            chrom_hash = zlib.crc32(str(chrom).encode("utf-8")) & 0xFFFFFFFF
            rnd = random.Random(seed + chrom_hash % 10_000_000)
            idxs_shuf = idxs[:]
            rnd.shuffle(idxs_shuf)

            per_chrom_data_map[chrom] = idxs_shuf

        batch_sampler = IndexedChromBucketBatchSampler(
            per_chrom_data_map, batch_size=batch_size, shuffle=False, seed=seed
        )
        
        full_data_bs = DistributedBatchSampler(batch_sampler, world_size, rank, drop_last=False)

        full_data_loader = DataLoader(
            dataset,
            batch_sampler=full_data_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

        return full_data_loader

def run_gradient_attribution(
    selected_experiment_dir,
    model,
    test_loader,
    tg_scaler,
    tf_scaler,
    tf_names,
    tg_names,
    device,
    use_amp,
    rank,
    world_size,
    distributed,
    max_batches: int = None,
    save_every_n_batches: int = 20,
    disable_bias: bool = False,
    disable_motif_mask: bool = False,
    disable_shortcut: bool = False,
    zero_tf_expr: bool = False,
    use_dataloader: bool = True,
):
    
    if max_batches is not None:
        max_batches = min(max_batches, len(test_loader))
    else:
        max_batches = len(test_loader)

    T_total = len(tf_names)
    G_total = len(tg_names)

    grad_sum = torch.zeros(T_total, G_total, device=device, dtype=torch.float32)
    grad_count = torch.zeros_like(grad_sum)

    model.to(device).eval()

    iterator = test_loader
    if rank == 0:
        iterator = tqdm(
            test_loader,
            desc=f"Gradient attributions",
            unit="batches",
            total=max_batches,
            ncols=100,
            miniters=10,
            mininterval=2,
        )

    batch_grad_dfs = {}
    for b_idx, batch in enumerate(iterator):
        if max_batches is not None and b_idx >= max_batches:
            break

        # Manual sharding if the dataloader is not already distributed.
        if not use_dataloader and (b_idx % world_size != rank):
            continue

        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
        
        if disable_bias:
            bias = None
            model.use_bias = False
        if disable_motif_mask:
            motif_mask = None
            model.use_motif_mask = False
            if hasattr(model, "shortcut_layer"):
                model.shortcut_layer.use_motif_mask = False
        if disable_shortcut:
            if hasattr(model, "shortcut_layer"):
                with torch.no_grad():
                    model.shortcut_layer.scale.zero_()
        
        atac_wins = atac_wins.to(device)
        tf_tensor = tf_tensor.to(device)
        bias = bias.to(device) if bias is not None else None
        tf_ids = tf_ids.to(device)
        tg_ids = tg_ids.to(device)
        motif_mask = motif_mask.to(device) if motif_mask is not None else None

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

        G_eval = tg_ids.shape[-1]

        # Assign TGs to this rank and optionally chunk them to control memory.
        owned_tg_indices = torch.arange(G_eval, device=device)
        if world_size > 1:
            owned_tg_indices = owned_tg_indices[owned_tg_indices % world_size == rank]

        if owned_tg_indices.numel() == 0:
            if rank == 0:
                print(
                    f"[rank {rank}] owns 0 TGs out of {G_eval}; skipping this batch",
                    flush=True,
                )
            continue

        # ---------- METHOD 1: plain saliency (grad * input) ----------
        total_owned = owned_tg_indices.numel()

        for chunk_start in range(0, total_owned, total_owned):
            tg_chunk = owned_tg_indices[chunk_start : chunk_start + total_owned]

            # Slice TG-specific inputs to shrink the attention graph per chunk
            if bias is not None:
                bias_idx = tg_chunk
                if bias.device != tg_chunk.device:
                    bias_idx = tg_chunk.to(bias.device)
                if bias.dim() == 3:
                    bias_chunk = bias[:, bias_idx, :]
                else:
                    bias_chunk = bias[:, :, bias_idx, :]
                bias_chunk = bias_chunk.to(device, non_blocking=True)
            else:
                bias_chunk = None

            if motif_mask is not None:
                mm_idx = tg_chunk
                if motif_mask.device != tg_chunk.device:
                    mm_idx = tg_chunk.to(motif_mask.device)
                motif_mask_chunk = motif_mask[mm_idx].to(device, non_blocking=True)
            else:
                motif_mask_chunk = None

            if tg_ids.dim() == 1:
                tg_ids_chunk = tg_ids[tg_chunk]
            else:
                tg_ids_chunk = tg_ids[:, tg_chunk]

            # Need gradients w.r.t. tf_tensor
            tf_tensor_chunk = tf_tensor.detach().requires_grad_(True)
            
            if zero_tf_expr:
                tf_tensor_chunk = tf_tensor.detach().clone()
                tf_tensor_chunk.requires_grad_(True)

                # if tf_tensor is [B, T] (expression only)
                tf_tensor_chunk = tf_tensor_chunk * 0.0
            
            # Forward only for this TG chunk
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                tf_scaled = (
                    tf_scaler.transform(tf_tensor_chunk, tf_ids)
                    if tf_scaler is not None
                    else tf_tensor_chunk
                )

                preds_s, _, _ = model(
                    atac_wins,
                    tf_scaled,
                    tf_ids=tf_ids,
                    tg_ids=tg_ids_chunk,
                    bias=bias_chunk,
                    motif_mask=motif_mask_chunk,
                    return_shortcut_contrib=False,
                )
                preds_u = (
                    tg_scaler.inverse_transform(preds_s, tg_ids_chunk)
                    if tg_scaler is not None
                    else preds_s
                )
                preds_u = torch.nan_to_num(
                    preds_u.float(), nan=0.0, posinf=1e6, neginf=-1e6
                )

            local_chunk = tg_chunk.numel()

            for offset in range(local_chunk):
                retain = offset < (local_chunk - 1)

                grad_output_j = torch.zeros_like(preds_u)
                grad_output_j[:, offset] = 1.0

                grads = torch.autograd.grad(
                    outputs=preds_u,
                    inputs=tf_tensor_chunk,
                    grad_outputs=grad_output_j,
                    retain_graph=retain,
                    create_graph=False,
                )[0]

                # grad * input (expression channel)
                if grads.dim() == 3:
                    expr_grad = grads[..., 0]
                else:
                    expr_grad = grads

                # saliency = expr_grad * expr_input  # optionally .abs()
                saliency = expr_grad.abs()  # optionally .abs()
                
                # Sum the gradient over all TFs for this chunk
                if saliency.dim() == 3:
                    grad_abs = saliency.sum(dim=-1)
                else:
                    grad_abs = saliency
                
                grad_flat = grad_abs.reshape(-1)
                if tg_ids_chunk.dim() == 1:
                    tg_global = int(tg_ids_chunk[offset].item())
                else:
                    tg_global = int(tg_ids_chunk[0, offset].item())

                # grad_sum and grad_count are indexed by global TG ID, but we need to aggregate contributions from all samples in the batch that correspond to this TG
                col_grad = grad_sum[:, tg_global]
                col_count = grad_count[:, tg_global]

                # Sum the gradient over all samples in the batch that correspond to this chunk
                col_grad.index_add_(0, tf_ids_flat, grad_flat)
                col_count.index_add_(0, tf_ids_flat, torch.ones_like(grad_flat))
                
            # cleanup per chunk
            del (
                preds_u,
                preds_s,
                tf_scaled,
                tf_tensor_chunk,
                bias_chunk,
                motif_mask_chunk,
                tg_ids_chunk,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
        if save_every_n_batches is not None and rank == 0:
            if b_idx % save_every_n_batches == 0:
                # Filter to genes seen SO FAR (up to this batch)
                seen_tf_mask = grad_count.sum(dim=1) > 0
                seen_tg_mask = grad_count.sum(dim=0) > 0
                
                seen_tf_ids_batch = torch.nonzero(seen_tf_mask, as_tuple=True)[0].cpu().numpy()
                seen_tg_ids_batch = torch.nonzero(seen_tg_mask, as_tuple=True)[0].cpu().numpy()
                
                grad_attr_partial = grad_sum / (grad_count + 1e-12)
                grad_attr_compact = grad_attr_partial[seen_tf_mask][:, seen_tg_mask]
                grad_attr_np = grad_attr_compact.detach().cpu().numpy()
                
                batch_df_wide = pd.DataFrame(
                    grad_attr_np, 
                    index=[tf_names[i] for i in seen_tf_ids_batch],
                    columns=[tg_names[i] for i in seen_tg_ids_batch]
                )                
                batch_grad_dfs[b_idx] = batch_df_wide
                
                out_dir = selected_experiment_dir / "grad_attribution_batches"
                out_dir.mkdir(exist_ok=True)
                
                out_path = out_dir / f"gradient_attribution_batch_{b_idx}.parquet"
                batch_df_wide.to_parquet(out_path)
                

        # cleanup
        del (
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
    
    # After the loop - final output
    # Calculate the final gradient attribution score
    grad_attr = grad_sum / (grad_count + 1e-12)

    # Filter to only TFs and TGs seen across ALL batches
    seen_tf_mask = grad_count.sum(dim=1) > 0
    seen_tg_mask = grad_count.sum(dim=0) > 0

    seen_tf_ids = torch.nonzero(seen_tf_mask, as_tuple=True)[0]
    seen_tg_ids = torch.nonzero(seen_tg_mask, as_tuple=True)[0]

    grad_attr_compact = grad_attr[seen_tf_mask][:, seen_tg_mask]
    grad_attr_np = grad_attr_compact.detach().cpu().numpy()

    seen_tf_ids_np = seen_tf_ids.cpu().numpy()
    seen_tg_ids_np = seen_tg_ids.cpu().numpy()
    
    df_wide = pd.DataFrame(
        grad_attr_np, 
        index=[tf_names[i] for i in seen_tf_ids_np],
        columns=[tg_names[i] for i in seen_tg_ids_np]
    )
    
    out_path = selected_experiment_dir / f"gradient_attribution_raw.parquet"
    df_wide.to_parquet(out_path)  # or .to_pickle() for exact dtype preservation

    print(f"Gradient attribution: kept {len(seen_tf_ids_np)}/{T_total} TFs, {len(seen_tg_ids_np)}/{G_total} TGs")

    return df_wide, batch_grad_dfs

@torch.no_grad()
def run_tf_knockout(
    selected_experiment_dir,
    model,
    test_loader,
    tg_scaler,
    tf_scaler,
    tf_names,
    tg_names,
    device,
    use_amp,
    rank,
    world_size,
    distributed,
    max_batches=None,
    save_every_n_batches=20,
    max_tgs_per_batch=None,
    disable_bias=False,
    disable_motif_mask=False,
    disable_shortcut=False,
    zero_tf_expr=False,
    use_dataloader: bool = True,

    # NEW: KO baseline controls
    ko_mode="raw_percentile",   # "raw_zero" | "raw_percentile" | "scaled_k_sigma"
    raw_percentile=0.01,        # used if ko_mode="raw_percentile"
    k_sigma=3.0,                # used if ko_mode="scaled_k_sigma"
    skip_if_near_ko=True,
    eps=1e-8,
):
    """
    Returns:
      tf_tg_effect_np: [T_total, G_total] float64 mean delta (baseline - KO), aggregated over observed contexts
      effect_count_np: [T_total, G_total] float64 counts (#times each TF,TG updated)
    """
    
    if max_batches is not None:
        max_batches = min(max_batches, len(test_loader))
    else:
        max_batches = len(test_loader)
    
    T_total = len(tf_names)
    G_total = len(tg_names)

    effect_sum   = torch.zeros((T_total, G_total), device=device, dtype=torch.float64)
    effect_count = torch.zeros((T_total, G_total), device=device, dtype=torch.float64)

    model.to(device).eval()

    iterator = test_loader
    if rank == 0:
        iterator = tqdm(
            test_loader, 
            desc="TF knockout", 
            unit="batches", 
            total=max_batches, 
            miniters=10,
            mininterval=2,
            ncols=100
            )

    batch_tf_ko_dfs = {}
    for b_idx, batch in enumerate(iterator):
        if (max_batches is not None) and (b_idx >= max_batches):
            break

        # Manual sharding if the dataloader is not already distributed.
        if not use_dataloader and (b_idx % world_size != rank):
            continue
        
        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
        
        if disable_bias:
            bias = None
            model.use_bias = False
        if disable_motif_mask:
            motif_mask = None
            model.use_motif_mask = False
            if hasattr(model, "shortcut_layer"):
                model.shortcut_layer.use_motif_mask = False
        if disable_shortcut:
            if hasattr(model, "shortcut_layer"):
                with torch.no_grad():
                    model.shortcut_layer.scale.zero_()

        atac_wins  = atac_wins.to(device, non_blocking=True)
        tf_tensor  = tf_tensor.to(device, non_blocking=True)   # unscaled (your convention)
        bias       = bias.to(device, non_blocking=True) if bias is not None else None
        tf_ids     = tf_ids.to(device)
        tg_ids     = tg_ids.to(device)
        motif_mask = motif_mask.to(device, non_blocking=True) if motif_mask is not None else None

        # Shapes
        if tf_tensor.dim() == 2:
            B, T_eval = tf_tensor.shape
            F_dim = 1
            tf_unscaled_expr = tf_tensor                          # [B, T_eval]
        else:
            B, T_eval, F_dim = tf_tensor.shape
            tf_unscaled_expr = tf_tensor[..., 0]                  # [B, T_eval] (assumes expr is channel 0)
            
        if zero_tf_expr:
            tf_tensor = tf_tensor.clone()
            if tf_tensor.dim() == 2:
                tf_tensor = tf_tensor.zero_()
            else:
                tf_tensor[..., 0] = tf_tensor[..., 0].zero_()

        # ---- Scale TF inputs once ----
        if tf_scaler is not None:
            tf_scaled_base = tf_scaler.transform(tf_tensor, tf_ids)
        else:
            tf_scaled_base = tf_tensor

        # Work internally as 3D [B, T_eval, F_dim]
        if tf_scaled_base.dim() == 2:
            tf_scaled_base_3d = tf_scaled_base.unsqueeze(-1)      # [B, T_eval, 1]
        else:
            tf_scaled_base_3d = tf_scaled_base                    # [B, T_eval, F_dim]

        # ---- Baseline predictions ----
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds_base_s, _, _ = model(
                atac_wins,
                tf_scaled_base if tf_scaled_base.dim() == 2 else tf_scaled_base_3d,
                tf_ids=tf_ids, tg_ids=tg_ids,
                bias=bias, motif_mask=motif_mask,
                return_shortcut_contrib=False,
            )
            preds_base_u = tg_scaler.inverse_transform(preds_base_s, tg_ids) if tg_scaler is not None else preds_base_s

        preds_base_u = torch.nan_to_num(preds_base_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)  # [B, G_eval]
        B, G_eval = preds_base_u.shape

        owned_tg_indices = torch.arange(G_eval, device=device)
        if world_size > 1:
            owned_tg_indices = owned_tg_indices[owned_tg_indices % world_size == rank]
        if owned_tg_indices.numel() == 0:
            if rank == 0:
                print(
                    f"[rank {rank}] owns 0 TGs out of {G_eval}; skipping this batch",
                    flush=True,
                )
            continue

        chunk_size = max_tgs_per_batch if max_tgs_per_batch is not None else owned_tg_indices.numel()

        # ---- Build KO target values ----
        # We will produce a KO value in *scaled* space per TF position (and feature dim),
        # then broadcast to [B, F_dim] when applying.
        if ko_mode == "raw_zero":
            # raw expr=0 -> scaled via scaler (if present)
            if tf_scaler is not None:
                zeros_expr_1 = torch.zeros_like(tf_tensor[:1])
                zeros_scaled_1 = tf_scaler.transform(zeros_expr_1, tf_ids)
                if zeros_scaled_1.dim() == 2:
                    zeros_scaled = zeros_scaled_1.squeeze(0).unsqueeze(-1)  # [T_eval, 1]
                else:
                    zeros_scaled = zeros_scaled_1.squeeze(0)                # [T_eval, F_dim]
            else:
                zeros_scaled = torch.zeros((T_eval, F_dim), device=device, dtype=tf_tensor.dtype)

            ko_scaled_per_pos = zeros_scaled  # [T_eval, F_dim]

        elif ko_mode == "raw_percentile":
            # compute per-position raw percentile across batch (on expression channel)
            # KO raw target per t_pos: q_t = quantile(tf_unscaled_expr[:, t_pos], raw_percentile)
            q = torch.quantile(tf_unscaled_expr.float(), q=raw_percentile, dim=0)  # [T_eval]
            # create a raw tf_tensor clone of shape [1,B?] just to transform;
            # easiest: build tf_ko_raw with same shape as tf_tensor but only fill expr channel for one sample.
            tf_ko_raw_1 = tf_tensor[:1].clone()
            if tf_ko_raw_1.dim() == 2:
                tf_ko_raw_1[0, :] = q.to(tf_ko_raw_1.dtype)
            else:
                tf_ko_raw_1[0, :, 0] = q.to(tf_ko_raw_1.dtype)
                # leave other channels unchanged (or zero them if you prefer)
            if tf_scaler is not None:
                tf_ko_scaled_1 = tf_scaler.transform(tf_ko_raw_1, tf_ids)
            else:
                tf_ko_scaled_1 = tf_ko_raw_1
            if tf_ko_scaled_1.dim() == 2:
                ko_scaled_per_pos = tf_ko_scaled_1.squeeze(0).unsqueeze(-1)  # [T_eval,1]
            else:
                ko_scaled_per_pos = tf_ko_scaled_1.squeeze(0)                # [T_eval,F_dim]

        elif ko_mode == "scaled_k_sigma":
            # KO in scaled space: reduce expression feature by k_sigma standard deviations
            # NOTE: this uses the fact that scaler.transform standardizes by (x-mean)/std (or similar).
            # If your transform is different, adjust accordingly.
            ko_scaled_per_pos = tf_scaled_base_3d[:1].clone().squeeze(0)      # [T_eval,F_dim]
            if ko_scaled_per_pos.dim() == 1:
                ko_scaled_per_pos = ko_scaled_per_pos.unsqueeze(-1)
            # subtract k sigma on expr channel only (assumed 0)
            ko_scaled_per_pos[:, 0] = ko_scaled_per_pos[:, 0] - float(k_sigma)
        else:
            raise ValueError(f"Unknown ko_mode: {ko_mode}")

        # ---- Working tensor to edit in-place ----
        tf_scaled_work = tf_scaled_base_3d.clone()  # [B, T_eval, F_dim]

        for t_pos in range(T_eval):
            # Optionally skip if TF already near KO baseline in raw space
            if skip_if_near_ko:
                if ko_mode in {"raw_zero", "raw_percentile"}:
                    # compare raw expr to target raw baseline (approx)
                    raw_now = tf_unscaled_expr[:, t_pos].abs().max().item()
                    if ko_mode == "raw_zero" and raw_now < eps:
                        continue
                else:
                    # scaled-k-sigma: skip if scaled expr already very low? not necessary; leave on.
                    pass

            # Apply KO in scaled space for this position
            ko_val = ko_scaled_per_pos[t_pos].view(1, -1).expand(B, F_dim)  # [B,F_dim]
            tf_scaled_work[:, t_pos, :] = ko_val

            # Match model input dimensionality
            tf_scaled_input = tf_scaled_work.squeeze(-1) if tf_tensor.dim() == 2 else tf_scaled_work

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds_ko_s, _, _ = model(
                    atac_wins, tf_scaled_input,
                    tf_ids=tf_ids, tg_ids=tg_ids,
                    bias=bias, motif_mask=motif_mask,
                    return_shortcut_contrib=False,
                )
                preds_ko_u = tg_scaler.inverse_transform(preds_ko_s, tg_ids) if tg_scaler is not None else preds_ko_s

            preds_ko_u = torch.nan_to_num(preds_ko_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)  # [B,G_eval]

            # delta = baseline - KO (positive means TF supports expression)
            delta_mean = (preds_base_u - preds_ko_u).mean(dim=0)  # [G_eval]

            tf_global = int(tf_ids[t_pos].item())
            
            if tf_global < 0 or tf_global >= T_total:
                raise RuntimeError(
                    f"tf_global out of range: {tf_global} (T_total={T_total}). "
                    f"tf_ids min/max: {int(tf_ids.min())}/{int(tf_ids.max())}"
                )

            # tg ids that this rank will touch
            tg_globals_all = tg_ids[owned_tg_indices].long()

            bad = (tg_globals_all < 0) | (tg_globals_all >= G_total)
            if bad.any():
                bad_vals = tg_globals_all[bad][:20].detach().cpu().tolist()
                raise RuntimeError(
                    f"tg_globals out of range (showing up to 20): {bad_vals} (G_total={G_total}). "
                    f"tg_ids min/max: {int(tg_ids.min())}/{int(tg_ids.max())}"
                )

            # IMPORTANT: tg_ids are global vocab ids; tg_chunk indexes within batch output
            for start in range(0, owned_tg_indices.numel(), chunk_size):
                tg_chunk = owned_tg_indices[start : start + chunk_size]
                tg_globals = tg_ids[tg_chunk].long()              # [chunk]
                delta_chunk = delta_mean[tg_chunk].double()       # [chunk]
                effect_sum[tf_global, tg_globals] += delta_chunk
                effect_count[tf_global, tg_globals] += 1.0

            # restore
            tf_scaled_work[:, t_pos, :] = tf_scaled_base_3d[:, t_pos, :]
            
        if save_every_n_batches is not None and rank == 0:
            if b_idx % save_every_n_batches == 0:
                # Filter to genes seen SO FAR (up to this batch)
                seen_tf_mask = effect_count.sum(dim=1) > 0
                seen_tg_mask = effect_count.sum(dim=0) > 0
                
                seen_tf_ids_batch = torch.nonzero(seen_tf_mask, as_tuple=True)[0].cpu().numpy()
                seen_tg_ids_batch = torch.nonzero(seen_tg_mask, as_tuple=True)[0].cpu().numpy()
                
                tf_knockout_partial = effect_sum / (effect_count + 1e-12)
                tf_knockout_compact = tf_knockout_partial[seen_tf_mask][:, seen_tg_mask]
                tf_knockout_np = tf_knockout_compact.detach().cpu().numpy()
                
                batch_df_wide = pd.DataFrame(
                    tf_knockout_np, 
                    index=[tf_names[i] for i in seen_tf_ids_batch],
                    columns=[tg_names[i] for i in seen_tg_ids_batch]
                )
                batch_tf_ko_dfs[b_idx] = batch_df_wide
                
                out_dir = selected_experiment_dir / "tf_knockout_batches"
                out_dir.mkdir(exist_ok=True)
                
                out_path = out_dir / f"tf_knockout_raw_batch_{b_idx}.parquet"
                batch_df_wide.to_parquet(out_path)

        # free some refs (optional)
        del preds_base_s, preds_base_u

    if distributed:
        dist.barrier()
        dist.all_reduce(effect_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(effect_count, op=dist.ReduceOp.SUM)

    # After the loop - final output
    # Calculate the final TF knockout effect
    tf_tg_effect = effect_sum / (effect_count + 1e-12)
    
    # Filter to only TFs and TGs seen across ALL batches
    seen_tf_mask = effect_count.sum(dim=1) > 0
    seen_tg_mask = effect_count.sum(dim=0) > 0
    
    seen_tf_ids = torch.nonzero(seen_tf_mask, as_tuple=True)[0]
    seen_tg_ids = torch.nonzero(seen_tg_mask, as_tuple=True)[0]
    
    tf_tg_effect_compact = tf_tg_effect[seen_tf_mask][:, seen_tg_mask]
    tf_tg_effect_np = tf_tg_effect_compact.detach().cpu().numpy()
    
    seen_tf_ids_np = seen_tf_ids.cpu().numpy()
    seen_tg_ids_np = seen_tg_ids.cpu().numpy()
    
    df_wide = pd.DataFrame(
        tf_tg_effect_np, 
        index=[tf_names[i] for i in seen_tf_ids_np],
        columns=[tg_names[i] for i in seen_tg_ids_np]
    )
    
    out_path = selected_experiment_dir / f"tf_knockout_raw.parquet"
    df_wide.to_parquet(out_path)
    
    print(f"TF knockout: kept {len(seen_tf_ids_np)}/{T_total} TFs, {len(seen_tg_ids_np)}/{G_total} TGs")
    
    return df_wide, batch_tf_ko_dfs

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
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")
        test_loader = torch.load(selected_experiment_dir / "test_loader.pt", weights_only=False)

    ckpt_path = os.path.join(selected_experiment_dir, checkpoint_file)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")
        state = torch.load(ckpt_path, map_location="cpu")
    
    # 2) Recreate model EXACTLY as in training
    model = model_module.MultiomicTransformer(
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
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device).eval()

    # 3) Rebuild scalers on the SAME DEVICE as inputs
    tg_scaler = dataset_refactor.SimpleScaler(
        mean=torch.as_tensor(state["tg_scaler_mean"], device=device, dtype=torch.float32),
        std=torch.as_tensor(state["tg_scaler_std"],  device=device, dtype=torch.float32),
    )
    tf_scaler = dataset_refactor.SimpleScaler(
        mean=torch.as_tensor(state["tf_scaler_mean"], device=device, dtype=torch.float32),
        std=torch.as_tensor(state["tf_scaler_std"],  device=device, dtype=torch.float32),
    )

    return model, test_loader, tg_scaler, tf_scaler, state

def argparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_num", type=str, default=1)
    parser.add_argument("--checkpoint_name", type=str, default="trained_model.pt")
    parser.add_argument("--save_every_n_batches", type=int, default=-1)
    parser.add_argument("--force_recalculate", type=str, default="false", help="Whether to force recalculation even if output files already exist. Set to 'true' to enable.")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    rank, world_size, local_rank, distributed = setup_distributed()
    
    args = argparse_args()
        
    if rank == 0:
        logging.info("Loading experiment...")
    
    experiment_dir = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/"
    experiment_name = args.experiment_name
    checkpoint_name = args.checkpoint_name
    
    force_recalculate = args.force_recalculate
    
    if force_recalculate.lower() == "true":
        force_recalculate = True
    else:
        force_recalculate = False
    
    model_training_dir = os.path.join(experiment_dir, experiment_name, args.model_num)
    
    if not os.path.isdir(model_training_dir):
        model_training_dir = os.path.join(experiment_dir, experiment_name, "chr19", args.model_num)

    assert os.path.isdir(model_training_dir), \
        f"Model training directory not found: {model_training_dir}"
        
    assert checkpoint_name in os.listdir(model_training_dir), \
        f"Checkpoint {checkpoint_name} not found in {model_training_dir}"

    if "gradient_attribution_raw.parquet" in os.listdir(model_training_dir) and "tf_knockout_raw.parquet" in os.listdir(model_training_dir) and not force_recalculate:
        logging.info(f"Gradient attribution and TF knockout results already exist in {model_training_dir}, skipping computation.")
        exit(0)
        
    else:
        model_num = int(args.model_num.split("_")[-1])
            
        exp = experiment_loader.ExperimentLoader(
            experiment_dir = experiment_dir,
            experiment_name=experiment_name,
            model_num=model_num,
        )
        
        checkpoint_name = checkpoint_name
        
        assert checkpoint_name in os.listdir(exp.model_training_dir), \
            f"Checkpoint {checkpoint_name} not found in {exp.model_training_dir}"
        
        save_every_n_batches = args.save_every_n_batches
        if save_every_n_batches < 1:
            save_every_n_batches = None
        
        SAMPLE_DATA_CACHE_DIR = Path(PROJECT_DIR) / "data/training_data_cache" / exp.experiment_name
        CHROM_IDS = exp.experiment_settings_df.set_index("parameter").loc["CHROM_IDS"].value.split(" ")
        COMMON_DATA = SAMPLE_DATA_CACHE_DIR / "common"

        if rank == 0:
            logging.info("Creating dataset and dataloader...")
        dataset = MultiChromosomeDataset(
            data_dir=SAMPLE_DATA_CACHE_DIR,
            chrom_ids=CHROM_IDS,
            tf_vocab_path=os.path.join(COMMON_DATA, "tf_vocab.json"),
            tg_vocab_path=os.path.join(COMMON_DATA, "tg_vocab.json"),
        )

        if rank == 0:
            logging.info("Creating dataloader...")
        full_data_loader = prepare_dataloader(
            dataset, batch_size=args.batch_size, world_size=world_size, rank=rank, num_workers=1, seed=42
        )
        
        sample_type = exp.experiment_name.split("_")[0]

        if rank == 0:
            logging.info(f"\nExperiment: {exp.experiment_name}, Sample type: {sample_type}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if rank == 0:
            logging.info(f"Using device: {device}\n")

        max_batches = None
        disable_bias = False
        disable_motif_mask = False
        disable_shortcut = False
        zero_tf_expr = False
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

        print(
            f"[rank {rank} local_rank {local_rank}] "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"device_count={torch.cuda.device_count()}",
            flush=True
        )


        print(
            f"[rank {rank}] current_device={torch.cuda.current_device()} ",
            flush=True
        )

        free, total = torch.cuda.mem_get_info(device)
        print(f"[rank {rank}] mem free={free/1e9:.2f} GB / total={total/1e9:.2f} GB", flush=True)

        use_amp = True and device.type == "cuda"
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(device)
            if capability[0] < 7:
                logging.warning(
                    f"GPU compute capability {capability[0]}.{capability[1]} < 7.0, disabling AMP"
                )
                use_amp = False
                    
        model, test_loader, tg_scaler, tf_scaler, state = load_model(
            selected_experiment_dir=exp.model_training_dir,
            checkpoint_file=checkpoint_name,
            device=device
        )
        
        if max_batches is not None:
            max_batches = min(max_batches, len(test_loader))
        else:
            max_batches = len(test_loader)
        
        if rank == 0:
            logging.info(f"Checkpoint {checkpoint_name} loaded. Max batches for attribution: {max_batches}")
        
        selected_experiment_dir = exp.model_training_dir
        selected_experiment_dir.mkdir(parents=True, exist_ok=True)
        
        if not "gradient_attribution_raw.parquet" in os.listdir(selected_experiment_dir) and not force_recalculate:
            start_time = time.time()
            grad_attr_df, batch_grad_df_dict = run_gradient_attribution(
                selected_experiment_dir=selected_experiment_dir,
                model=model,
                test_loader=test_loader,
                tg_scaler=tg_scaler,
                tf_scaler=tf_scaler,
                tf_names=exp.tf_names,
                tg_names=exp.tg_names,
                use_amp=use_amp,
                max_batches=max_batches,
                device=device,
                rank=rank,
                world_size=world_size,
                distributed=distributed,
                disable_bias=disable_bias,
                disable_motif_mask=disable_motif_mask,
                disable_shortcut=disable_shortcut,
                zero_tf_expr=zero_tf_expr,
                save_every_n_batches=save_every_n_batches,
                use_dataloader=True,
                
            )
            end_time = time.time()
            logging.info(f"  - Gradient attribution finished {max_batches} batches in {end_time - start_time:.2f} seconds.")
        else:
            logging.info(f"Gradient attribution results already exist in {selected_experiment_dir}, skipping computation.")
        
        if not "tf_knockout_raw.parquet" in os.listdir(selected_experiment_dir) and not force_recalculate:
            for ko_mode in ["scaled_k_sigma"]: # "raw_zero", "raw_percentile", 
                logging.info(f"  - Running TF knockout with mode: {ko_mode}")
                start_time = time.time()
                tfko_df, batch_tf_ko_df_dict = run_tf_knockout(
                    selected_experiment_dir=selected_experiment_dir,
                    model=model,
                    test_loader=test_loader,
                    tg_scaler=tg_scaler,
                    tf_scaler=tf_scaler,
                    tf_names=exp.tf_names,
                    tg_names=exp.tg_names,
                    device=device,
                    use_amp=use_amp,
                    rank=rank,
                    world_size=world_size,
                    distributed=distributed,
                    max_batches=max_batches,
                    save_every_n_batches=save_every_n_batches,
                    ko_mode=ko_mode,
                    raw_percentile=0.01,
                    disable_bias=disable_bias,
                    disable_motif_mask=disable_motif_mask,
                    disable_shortcut=disable_shortcut,
                    zero_tf_expr=zero_tf_expr,
                    use_dataloader=True,
                    
                )
                end_time = time.time()
                logging.info(f"  - TF knockout finished {max_batches} batches in {end_time - start_time:.2f} seconds.")
        else:
            logging.info(f"TF knockout results already exist in {selected_experiment_dir}, skipping computation.")