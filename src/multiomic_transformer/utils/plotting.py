# transformer_testing.py
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import random
from itertools import cycle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, r2_score
import argparse

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
SRC_DIR = str(Path(PROJECT_DIR) / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from multiomic_transformer.datasets.dataset import SimpleScaler
from multiomic_transformer.models.model import MultiomicTransformer

logging.basicConfig(level=logging.INFO, format='%(message)s')

def format_gpu_usage_file(gpu_log_file):
    gpu = pd.read_csv(gpu_log_file)
    gpu.columns = gpu.columns.str.strip()
    gpu["timestamp"] = pd.to_datetime(gpu["timestamp"], errors="coerce")
    gpu["tsec"] = gpu["timestamp"].dt.floor("s")

    gpu["memory_used_gib"]  = gpu["memory.used [MiB]"].astype(str).str.extract(r"(\d+)").astype(float) / 1024
    gpu["memory_total_gib"] = gpu["memory.total [MiB]"].astype(str).str.extract(r"(\d+)").astype(float) / 1024

    t0 = gpu["tsec"].min()
    gpu["elapsed_s"] = (gpu["tsec"] - t0).dt.total_seconds().astype(int)
    gpu["elapsed_min"] = gpu["elapsed_s"] / 60.0
    gpu["elapsed_hr"] = gpu["elapsed_s"] / 3600.0
    

    # mean per second, then carry minutes as a column
    mean_per_sec = (
        gpu.groupby("elapsed_s", as_index=False)["memory_used_gib"]
           .mean()
           .sort_values("elapsed_s")
    )
    mean_per_sec["elapsed_min"] = mean_per_sec["elapsed_s"] / 60.0
    mean_per_sec["elapsed_hr"] = mean_per_sec["elapsed_s"] / 3600.0

    total_gib = float(gpu["memory_total_gib"].iloc[0])
    return gpu, mean_per_sec, total_gib

def plot_gpu_usage(gpu_log_dict, align_to_common_duration=False, smooth=None):
    """
    align_to_common_duration: if True, truncate each run to the shortest duration so curves end together.
    smooth: optional int window (in seconds) for a centered rolling mean on memory (e.g., smooth=5).
    """
    fig, ax = plt.subplots(figsize=(7,4))

    totals = []
    max_elapsed_by_run = {}
    for label, (_, mean_per_min, total_gib) in gpu_log_dict.items():
        totals.append(total_gib)
        max_elapsed_by_run[label] = mean_per_min["elapsed_min"].max()

    # shortest duration across runs (so lines end at the same x)
    common_end = min(max_elapsed_by_run.values()) if align_to_common_duration else None

    for label, (_, mean_per_min, total_gib) in gpu_log_dict.items():
        m = mean_per_min.copy()
        if align_to_common_duration and common_end is not None:
            m = m[m["elapsed_min"] <= common_end]

        if smooth and smooth > 1:
            m["memory_used_gib"] = m["memory_used_gib"].rolling(smooth, center=True, min_periods=1).mean()

        ax.plot(m["elapsed_hr"], m["memory_used_gib"], label=f"{label}", linewidth=3)

    max_total = max(totals)
    ax.axhline(max_total, linestyle="--", label=f"Max RAM")
    ax.set_ylabel("GiB")
    ax.set_xlabel("Minutes since start")
    ax.set_ylim(0, max_total + 1)
    ax.xaxis.set_major_locator(MultipleLocator(1))  # tick every 1 hour
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
    ax.set_xlabel("Hours since start")

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
    ax.set_title(
        f"Average GPU Memory vs. Elapsed Time\n"
    )
    plt.tight_layout()
    plt.show()
    return fig

def plot_gpu_usage_boxplots(gpu_log_dict, use_mean_per_sec: bool = True, title_suffix: str = ""):
    """
    gpu_log_dict: {label: (gpu_df, mean_per_sec_df, total_gib)}
        - same structure returned/consumed by format_gpu_usage_file / plot_gpu_usage.
    use_mean_per_sec:
        - True: boxplots over per-second mean memory (smoother).
        - False: boxplots over all raw samples.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    data = []
    labels = []
    capacities = []

    for label, (gpu_df, mean_per_sec_df, total_gib) in gpu_log_dict.items():
        if use_mean_per_sec:
            series = mean_per_sec_df["memory_used_gib"]
        else:
            series = gpu_df["memory_used_gib"]

        # drop NaNs just in case
        series = series.dropna()
        if series.empty:
            continue

        data.append(series.values)
        labels.append(label)
        capacities.append(total_gib)

    if not data:
        raise ValueError("No GPU usage data available to plot.")

    # Boxplot: one box per run
    bp = ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=True,
        patch_artist=True,
    )

    # Optional: horizontal line for (max) GPU capacity across runs
    max_cap = max(capacities)
    ax.axhline(max_cap, linestyle="--", linewidth=1.0,
               label=f"Max capacity")

    ax.set_ylabel("GPU memory used (GiB)")
    base_title = "GPU Memory Usage Distribution per Run"
    if title_suffix:
        base_title += f" ({title_suffix})"
    ax.set_title(base_title)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )

    plt.tight_layout()
    return fig

def plot_R2_across_epochs(df):
    fig = plt.figure(figsize=(6, 5))
    plt.plot(df.index, df["R2_u"], linewidth=2, label=f"Best R2 (unscaled) = {df['R2_u'].max():.2f}")
    plt.plot(df.index, df["R2_s"], linewidth=2, label=f"Best R2 (scaled)     = {df['R2_s'].max():.2f}")

    plt.title(f"TG Expression R2 Across Training", fontsize=17)
    plt.ylim((0,1))
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlabel("Epoch", fontsize=17)
    plt.ylabel("R2", fontsize=17)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    return fig

def plot_train_val_loss(df):
    fig = plt.figure(figsize=(6, 5))
    df = df.copy().iloc[5:, :]
    plt.plot(df["Epoch"], df["Train MSE"], label="Train MSE", linewidth=2)
    plt.plot(df["Epoch"], df["Val MSE"], label="Val MSE", linewidth=2)
    # plt.plot(df["Epoch"], df["Train Total Loss"], label="Train Total Loss", linestyle="--", alpha=0.7)

    plt.title(f"Train Val Loss Curves", fontsize=17)
    plt.xlabel("Epoch", fontsize=17)
    plt.ylabel("Loss", fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.ylim([0, 1])
    # plt.xlim(left=2)
    plt.legend(fontsize=15)
    plt.tight_layout()
    return fig

def nanaware_per_gene_stats(y_true, y_pred, eps=1e-8):
    """Compute per-gene metrics accounting for NaN entries."""
    N, G = y_true.shape
    r2 = np.full(G, np.nan, dtype=np.float64)
    pearson = np.full(G, np.nan, dtype=np.float64)
    mae = np.full(G, np.nan, dtype=np.float64)
    rmse = np.full(G, np.nan, dtype=np.float64)
    n_obs = np.zeros(G, dtype=np.int32)

    for j in range(G):
        mask = np.isfinite(y_true[:, j]) & np.isfinite(y_pred[:, j])
        m = mask.sum()
        n_obs[j] = m
        
        if m < 2:
            continue
            
        yt = y_true[mask, j].astype(np.float64)
        yp = y_pred[mask, j].astype(np.float64)

        diff = yt - yp
        mae[j] = np.mean(np.abs(diff))
        rmse[j] = np.sqrt(np.mean(diff**2))

        yt_c = yt - yt.mean()
        yp_c = yp - yp.mean()
        yt_std = np.sqrt((yt_c**2).sum())
        yp_std = np.sqrt((yp_c**2).sum())
        if yt_std > 0 and yp_std > 0:
            pearson[j] = (yt_c @ yp_c) / (yt_std * yp_std)

        sst = ((yt - yt.mean())**2).sum()
        sse = (diff**2).sum()
        r2[j] = 1.0 - sse / (sst + eps)

    return {"r2": r2, "pearson": pearson, "mae": mae, "rmse": rmse, "n_obs": n_obs}

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
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        missing, unexpected = model.load_state_dict(
            state["model_state_dict"], strict=False
        )
        if len(missing) > 0:
            logging.info(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            logging.info(f"Unexpected keys: {unexpected}")
    elif isinstance(state, dict) and "model_state_dict" not in state:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0:
            logging.info(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            logging.info(f"Unexpected keys: {unexpected}")
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0:
            logging.info(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            logging.info(f"Unexpected keys: {unexpected}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def plot_model_tg_predictions(device, model, test_loader, tg_scaler, tf_scaler):
        
    G_total = tg_scaler.mean.shape[0]  # total number of genes

    # ---- global per-gene accumulators (unscaled space) ----
    sse_g   = torch.zeros(G_total, dtype=torch.float64)
    sumy_g  = torch.zeros(G_total, dtype=torch.float64)
    sumy2_g = torch.zeros(G_total, dtype=torch.float64)
    cnt_g   = torch.zeros(G_total, dtype=torch.float64)
    sumpred_g = torch.zeros(G_total, dtype=torch.float64)

    ### For overall R² and scatter:
    all_preds_for_plot = []
    all_tgts_for_plot  = []

    with torch.no_grad():
        for batch in test_loader:
            atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
            atac_wins  = atac_wins.to(device)
            tf_tensor  = tf_tensor.to(device)
            targets    = targets.to(device)
            bias       = bias.to(device)
            tf_ids     = tf_ids.to(device)
            tg_ids     = tg_ids.to(device)
            motif_mask = motif_mask.to(device)

            # scale / predict exactly like in validation
            if tf_scaler is not None:
                tf_tensor = tf_scaler.transform(tf_tensor, tf_ids)
            if tg_scaler is not None:
                targets_s = tg_scaler.transform(targets, tg_ids)
            else:
                targets_s = targets

            preds_s, _, _ = model(
                atac_wins, tf_tensor,
                tf_ids=tf_ids, tg_ids=tg_ids,
                bias=bias, motif_mask=motif_mask,
                return_edge_logits=True, return_shortcut_contrib=False,
                edge_extra_features=None,
            )

            preds_s   = torch.nan_to_num(preds_s.float(),   nan=0.0, posinf=1e6, neginf=-1e6)
            targets_s = torch.nan_to_num(targets_s.float(), nan=0.0, posinf=1e6, neginf=-1e6)

            # unscale + clamp
            if tg_scaler is not None:
                targets_u = tg_scaler.inverse_transform(targets_s, tg_ids)
                preds_u   = tg_scaler.inverse_transform(preds_s,   tg_ids)
            else:
                targets_u, preds_u = targets_s, preds_s

            targets_u = torch.nan_to_num(targets_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)
            preds_u   = torch.nan_to_num(preds_u.float(),   nan=0.0, posinf=1e6, neginf=-1e6)
            preds_u   = preds_u.clamp_min(0.0)
            
            sumpred_batch = preds_u.sum(dim=0)


            # ---- store for overall R² / scatter ----
            all_tgts_for_plot.append(targets_u.detach().cpu().numpy())
            all_preds_for_plot.append(preds_u.detach().cpu().numpy())

            # ---- per-gene accumulators (unscaled) ----
            # shapes: [B, G_eval]
            err2   = (targets_u - preds_u) ** 2
            B      = targets_u.shape[0]

            # reduce over batch
            sse_batch   = err2.sum(dim=0)              # [G_eval]
            sumy_batch  = targets_u.sum(dim=0)
            sumy2_batch = (targets_u ** 2).sum(dim=0)
            cnt_batch   = torch.full_like(sse_batch, B, dtype=torch.float64)

            # move ids to CPU, accumulate into global vectors
            ids_cpu = tg_ids.cpu()
            sse_g.index_add_(0, ids_cpu, sse_batch.cpu().to(torch.float64))
            sumy_g.index_add_(0, ids_cpu, sumy_batch.cpu().to(torch.float64))
            sumy2_g.index_add_(0, ids_cpu, sumy2_batch.cpu().to(torch.float64))
            cnt_g.index_add_(0, ids_cpu, cnt_batch.cpu().to(torch.float64))
            sumpred_g.index_add_(0, ids_cpu, sumpred_batch.cpu().to(torch.float64))

    # ============================
    # 4) Per-gene R² (global)
    # ============================
    eps = 1e-12
    mask = cnt_g > 0  # genes that appeared in the test set

    mean_g = sumy_g[mask] / cnt_g[mask]
    sst_g  = sumy2_g[mask] - cnt_g[mask] * (mean_g ** 2)

    valid = sst_g > eps  # genes with non-trivial variance

    r2_g = torch.full_like(sse_g, float("nan"), dtype=torch.float64)

    idx_all  = mask.nonzero(as_tuple=True)[0]   # indices of genes with any data
    idx_keep = idx_all[valid]                   # subset with non-zero variance

    r2_g_values = 1.0 - (sse_g[idx_keep] / torch.clamp(sst_g[valid], min=eps))
    r2_g[idx_keep] = r2_g_values

    r2_g_cpu = r2_g.cpu().numpy()
    median_r2_gene = np.nanmedian(r2_g_cpu)

    # ============================
    # 5) Global R² + scatter plot
    # ============================
    preds_flat = np.concatenate([p.reshape(-1) for p in all_preds_for_plot])
    tgts_flat  = np.concatenate([t.reshape(-1) for t in all_tgts_for_plot])

    # Remove NaNs / infs
    valid = np.isfinite(preds_flat) & np.isfinite(tgts_flat)
    preds_clean = preds_flat[valid]
    tgts_clean  = tgts_flat[valid]
    
    mean_true_g = torch.full_like(sumy_g, float("nan"))
    mean_pred_g = torch.full_like(sumpred_g, float("nan"))

    mask = cnt_g > 0
    mean_true_g[mask] = sumy_g[mask] / cnt_g[mask]
    mean_pred_g[mask] = sumpred_g[mask] / cnt_g[mask]

    mean_true = mean_true_g.cpu().numpy()
    mean_pred = mean_pred_g.cpu().numpy()
    
    var_g = (sumy2_g[mask] / cnt_g[mask]) - (mean_true_g[mask] ** 2)
    keep = var_g.cpu().numpy() > 1e-6

    mean_true = mean_true[keep]
    mean_pred = mean_pred[keep]

    # Overall R² across all points
    r2_overall = r2_score(tgts_clean, preds_clean)

    logging.info("\n" + "="*60)
    logging.info("SCATTER PLOT STATISTICS")
    logging.info("="*60)
    logging.info(f"Overall R² (from all points): {r2_overall:.4f}")
    logging.info(f"N samples (valid points): {len(preds_clean):,}")
    logging.info(f"Median per-gene R²: {median_r2_gene:.4f}")
    
    # ---- per-gene mean expression plot ----
    per_gene_mean_fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(mean_true, mean_pred, alpha=0.6, s=20)

    lims = [
        min(mean_true.min(), mean_pred.min()),
        max(mean_true.max(), mean_pred.max()),
    ]
    ax.plot(lims, lims, "k--", lw=2)

    r2_tg = r2_score(mean_true, mean_pred)

    ax.set_xlabel("Mean actual TG expression")
    ax.set_ylabel("Mean predicted TG expression")
    ax.set_title(
        "Gene-level agreement between predicted \nand observed target gene expression\n"
        f"Mean $R^2$ = {r2_tg:.4f}"
    )

    ax.grid(alpha=0.3)
    plt.tight_layout()

    # ---- scatter plot of all points ----
    all_point_fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(tgts_clean, preds_clean, alpha=0.3, s=10, color='steelblue')

    lims = [
        min(np.min(tgts_clean), np.min(preds_clean)),
        max(np.max(tgts_clean), np.max(preds_clean))
    ]
    ax.plot(lims, lims, 'k--', lw=2, label='Perfect prediction')

    ax.set_xlabel('Actual Expression', fontsize=12)
    ax.set_ylabel('Predicted Expression', fontsize=12)
    ax.set_title(
        "Predicted vs Actual TG Expression\n"
        f"$R^2 = {r2_overall:.4f}$",
        fontsize=14,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return per_gene_mean_fig, all_point_fig, tgts_clean, preds_clean, mean_true, mean_pred

# AUROC Testing Plots
def balance_pos_neg(df, label_col="is_gt", random_state=0):
    rng = np.random.default_rng(random_state)
    df = df.copy()
    pos_df = df[df[label_col]]
    neg_df = df[~df[label_col]]

    n_pos = len(pos_df)
    n_neg = len(neg_df)
    if n_pos == 0 or n_neg == 0:
        logging.info("No positives or negatives, skipping balance")
        return df

    if n_neg < n_pos:
        pos_idx = rng.choice(pos_df.index.to_numpy(), size=n_neg, replace=False)
        pos_sample = pos_df.loc[pos_idx]
        neg_sample = neg_df
    else:
        pos_sample = pos_df
        neg_idx = rng.choice(neg_df.index.to_numpy(), size=n_pos, replace=False)
        neg_sample = neg_df.loc[neg_idx]

    balanced = pd.concat([pos_sample, neg_sample], axis=0)
    return balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

def create_random_distribution(scores: pd.Series, seed: int = 42) -> np.ndarray:
    random.seed(seed)
    uniform_distribution = np.random.uniform(low = scores.min(), high = scores.max(), size = len(scores))
    resampled_scores = np.random.choice(uniform_distribution, size=len(scores), replace=True)
    return resampled_scores

def matrix_to_df(mat, tf_names, tg_names, colname):
    """Flatten [T, G] matrix into (tf, tg, score) df."""
    T, G = mat.shape
    tf_idx, tg_idx = np.meshgrid(np.arange(T), np.arange(G), indexing="ij")
    df = pd.DataFrame({
        "Source": np.array(tf_names, dtype=object)[tf_idx.ravel()],
        "Target": np.array(tg_names, dtype=object)[tg_idx.ravel()],
        colname: mat.ravel(),
    })
    df["Source"] = df["Source"].astype(str).str.upper()
    df["Target"] = df["Target"].astype(str).str.upper()
    return df

def make_label_df_universe(tf_names, tg_names, chip_tf_set, gt_set):
    T, G = len(tf_names), len(tg_names)
    tf_idx, tg_idx = np.meshgrid(np.arange(T), np.arange(G), indexing="ij")
    df = pd.DataFrame({
        "Source": np.array(tf_names, dtype=object)[tf_idx.ravel()],
        "Target": np.array(tg_names, dtype=object)[tg_idx.ravel()],
    })
    df["Source"] = df["Source"].astype(str).str.upper()
    df["Target"] = df["Target"].astype(str).str.upper()
    df["is_gt"] = list(map(gt_set.__contains__, zip(df["Source"], df["Target"])))
    # only ChIP TFs
    df = df[df["Source"].isin(chip_tf_set)].reset_index(drop=True)

    return df

def label_edges(df, gt_edges):
    df = df.copy()
    df["is_gt"] = [(s, t) in gt_edges for s, t in zip(df["Source"], df["Target"])]
    return df

def filter_df_to_gene_set(df, gt_tfs, gt_tgs):
    df = df.copy()
    mask = df["Source"].isin(gt_tfs) & df["Target"].isin(gt_tgs)
    df = df[mask].reset_index(drop=True)
    return df

def _compute_roc_auc(y_true, scores):
    """
    Minimal ROC/AUC for binary labels y_true in {0,1} and real-valued scores.
    Returns fpr, tpr, auc.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    # sort by descending score
    order = np.argsort(-scores)
    y = y_true[order]

    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return np.array([0, 1]), np.array([0, 1]), np.nan

    # cumulative TP/FP as we move threshold down
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)

    tpr = tp / P
    fpr = fp / N

    # prepend (0,0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # trapezoidal AUC
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc

def compute_curves(df, score_col, label_col="is_gt", balance=True, name=""):
    """Return AUROC, AUPRC, and PR/ROC curves for one method."""
    df = df.dropna(subset=[score_col]).copy()
    
    if balance:
        df = balance_pos_neg(df, label_col=label_col, random_state=0)
        
    if len(df) == 0 or df[label_col].nunique() < 2:
        logging.info(f"Skipping {name}: need at least one positive and one negative, got {df[label_col].value_counts().to_dict()}")
        return None
    
    y = df[label_col].astype(int).values
    s = df[score_col].values

    auroc = roc_auc_score(y, s)
    auprc = average_precision_score(y, s)

    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)

    return {
        "name": name,
        "auroc": auroc,
        "auprc": auprc,
        "prec": prec,
        "rec": rec,
        "fpr": fpr,
        "tpr": tpr,
    }

def plot_auroc_per_min_tg_r2(results_r2, min_r2_grid, baseline_macro, baseline_micro, experiment_dir):
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(results_r2["min_tg_r2"], results_r2["macro_auroc"], marker="o", label="macro")
    ax.plot(results_r2["min_tg_r2"], results_r2["micro_auroc"], marker="o", label="micro")
    ax.hlines(baseline_macro, min_r2_grid[0], min_r2_grid[-1],
            color="grey", linestyle="dashed", label="baseline macro")
    ax.hlines(baseline_micro, min_r2_grid[0], min_r2_grid[-1],
            color="lightgrey", linestyle="dashed", label="baseline micro")

    ax.set_title("Macro / Micro AUROC vs. min TG R²", fontsize=14)
    ax.set_xlabel("Minimum TG R²", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)

    # Major ticks every 0.2
    ax.xaxis.set_major_locator(MultipleLocator(0.2))

    # One minor tick between each major (every 0.1)
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

    ax.tick_params(axis="x", which="major", labelsize=11)
    ax.tick_params(axis="y", which="major", labelsize=11)
    ax.tick_params(axis="x", which="minor", length=4)  # just the small tick bars, no labels

    ax.legend(bbox_to_anchor=(1.175, 0.5), loc="center", fontsize=11)
    fig.tight_layout()
    plt.savefig(Path(experiment_dir) / "threshold_GA_by_r2.svg")
    plt.close()

def plot_auroc_auprc(df, name, score_col, label_col="is_gt"):
    """_summary_

    Args:
        name (str): Dataset name
        score_col (str): Name of the score column
        label_col (str): Name of the label column

    Returns:
        tuple(plt.Figure, float, float): AUROC/AUPRC Figure, AUROC, AUPRC
    """
    
    df["random_scores"] = create_random_distribution(df[score_col])
    
    curves = compute_curves(df, score_col, label_col=label_col, balance=True, name=name)
    rand_curves = compute_curves(df, "random_scores", label_col=label_col, balance=True, name=name)
    
    auroc = curves["auroc"]
    auprc = curves["auprc"]
    prec = curves["prec"]
    rec = curves["rec"]
    fpr = curves["fpr"]
    tpr = curves["tpr"]
    
    rand_auroc = rand_curves["auroc"]
    rand_auprc = rand_curves["auprc"]
    rand_prec = rand_curves["prec"]
    rand_rec = rand_curves["rec"]
    rand_fpr = rand_curves["fpr"]
    rand_tpr = rand_curves["tpr"]
    
    # ROC plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ax[0].plot(rand_fpr, rand_tpr, color="#7ab4e8", linestyle="--", lw=2)
    ax[0].plot(fpr, tpr, lw=2, color="#4195df", label=f"AUROC = {auroc:.3f}\nRandom = {rand_auroc:.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title(f"AUROC")
    ax[0].legend(
        bbox_to_anchor=(0.5, -0.28),
        loc="upper center",
        borderaxespad=0.0
    )
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    
    # PR plot
    ax[1].plot(rand_rec, rand_prec, color="#7ab4e8", linestyle="--", lw=2)
    ax[1].plot(rec, prec, lw=2, color="#4195df", label=f"AUPRC = {auprc:.3f}\nRandom = {rand_auprc:.3f}")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title(f"AUPRC")
    ax[1].legend(
        bbox_to_anchor=(0.5, -0.28),
        loc="upper center",
        borderaxespad=0.0
    )
    ax[1].set_ylim(0, 1.0)
    ax[1].set_xlim(0, 1.0)
    plt.suptitle(name, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig, auroc, auprc

def plot_method_curve_variability(df_results, out_dir):
    """
    For each method in df_results, plot all ROC and PR curves across
    ground truths and samples to visualize variability.

    Parameters
    ----------
    df_results : pd.DataFrame
        Must have columns:
        - 'name'   : method name
        - 'gt_name': ground truth identifier
        - 'sample' : sample identifier (or 'FEATURES_ONLY' for feature-only runs)
        - 'fpr', 'tpr', 'prec', 'rec' : arrays (ndarray/list)
    out_dir : Path or str
        Directory to save per-method figures.
    feature_list : list of str, optional
        If provided, you can choose to skip or style features differently.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Unique methods (including features unless you filter)
    method_names = sorted(df_results["name"].unique())

    for method in method_names:

        df_m = df_results[df_results["name"] == method].copy()
        if df_m.empty:
            continue

        # Build a label per curve: GT / sample
        labels = df_m.apply(
            lambda row: f"{row['gt_name']} / {row['sample']}", axis=1
        )

        # Figure + axes
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

        title_fontsize = 14
        axes_fontsize = 12
        tick_fontsize = 10

        # Color cycle (reusable)
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
        n_colors = len(color_cycle)

        # ================= ROC panel =================
        for (idx, row), label in zip(df_m.iterrows(), labels):
            fpr = row["fpr"]
            tpr = row["tpr"]
            if fpr is None or tpr is None:
                continue

            # choose a color deterministically based on label
            c_idx = hash(label) % n_colors
            color = color_cycle[c_idx]

            ax[0].plot(
                fpr, tpr,
                lw=1.5,
                alpha=0.4,
                color=color,
            )

        ax[0].plot([0, 1], [0, 1], "k--", lw=1)
        ax[0].set_xlabel("False Positive Rate", fontsize=axes_fontsize)
        ax[0].set_ylabel("True Positive Rate", fontsize=axes_fontsize)
        ax[0].set_title(f"ROC curves: {method}", fontsize=title_fontsize)
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(0, 1)
        ax[0].tick_params(axis="both", labelsize=tick_fontsize)

        # ================= PR panel =================
        for (idx, row), label in zip(df_m.iterrows(), labels):
            rec = row["rec"]
            prec = row["prec"]
            if rec is None or prec is None:
                continue

            c_idx = hash(label) % n_colors
            color = color_cycle[c_idx]

            ax[1].plot(
                rec, prec,
                lw=1.5,
                alpha=0.4,
                color=color,
                label=label,
            )

        ax[1].set_xlabel("Recall", fontsize=axes_fontsize)
        ax[1].set_ylabel("Precision", fontsize=axes_fontsize)
        ax[1].set_title(f"PR curves: {method}", fontsize=title_fontsize)
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[1].tick_params(axis="both", labelsize=tick_fontsize)

        plt.tight_layout()

        method_safe = (
            method.replace(" ", "_")
                  .replace("/", "_")
                  .replace("+", "plus")
                  .lower()
        )
        out_path = os.path.join(out_dir, f"{method_safe}_curve_variability.svg")
        fig.savefig(out_path)
        
        plt.close(fig)

def plot_method_gt_heatmap(df_pooled: pd.DataFrame, metric: str = "auroc", per_tf: bool = False) -> plt.Figure:
    """
    Plot a heatmap of METHOD (rows) x GROUND TRUTH (cols) for AUROC or AUPRC.

    Rows are sorted by the mean metric across all ground truth datasets.
    """
    metric = metric.lower()
    if metric not in ("auroc", "auprc"):
        raise ValueError(f"metric must be 'auroc' or 'auprc', got {metric}")

    metric_col = metric  # 'auroc' or 'auprc'

    # 1) Order methods by mean metric across all GTs (descending)
    method_order = (
        df_pooled.groupby("name")[metric_col]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # 2) Pivot to METHOD x GT matrix
    heat_df = (
        df_pooled
        .pivot_table(index="name", columns="gt_name", values=metric_col)
        .loc[method_order]  # apply sorted method order
    )

    # 3) Plot heatmap with better sizing
    n_methods = len(heat_df.index)
    n_gts = len(heat_df.columns)
    
    fig, ax = plt.subplots(
        figsize=(
            max(n_gts * 1.2, 4),      # Width: 1.5 inches per GT, min 6
            max(n_methods * 0.4, 3),  # Height: 0.5 inches per method, min 4
        )
    )
    
    sns.heatmap(
        heat_df,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0.3,
        vmax=0.7,
        cbar_kws={"label": metric.upper()},
        ax=ax,
    )

    ax.set_xlabel("Ground truth dataset", fontsize=11)
    ax.set_ylabel("Method", fontsize=11)
    if per_tf == True:
        ax.set_title(
            f"Average per-TF {metric.upper()} score × ground truth\n"
            f"(methods sorted by mean {metric.upper()} across GTs)",
            fontsize=12,
            pad=10,
        )
    else:
        ax.set_title(
            f"{metric.upper()} score × ground truth\n"
            f"(methods sorted by mean {metric.upper()} across GTs)",
            fontsize=12,
            pad=10,
        )
    
    # Improve tick label readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    fig.tight_layout()
    return fig

def plot_chiptf_metric_auroc_by_quantile_from_scores(
    df,
    score_col,
    metric_name="scores",
    quantile_step=0.02,
    cmap_name="viridis",
):
    """
    Plot ROC curves and AUROC vs score quantile for a given method,
    using a long-form DataFrame with an 'is_gt' column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least:
            - score_col : float scores
            - "is_gt"   : boolean, True for ChIP edge, False otherwise
        You should already have restricted df to the universe you care about
        (e.g., ChIP TFs only) before calling this function.
    score_col : str
        Name of the column in df containing the scores for the method.
    metric_name : str
        Name to use in logging.infoouts and plot titles.
    quantile_step : float
        Step size for quantiles (e.g. 0.02).
    cmap_name : str
        Matplotlib colormap name.
    """

    # Drop missing scores
    df_use = df.dropna(subset=[score_col]).copy()
    
    df_use = balance_pos_neg(df=df_use, label_col="is_gt")

    scores = df_use[score_col].to_numpy(dtype=float)
    y = df_use["is_gt"].astype(int).to_numpy()

    # Filter out any non-finite scores
    finite_mask = np.isfinite(scores)
    scores = scores[finite_mask]
    y = y[finite_mask]

    if len(scores) == 0:
        raise ValueError(f"[{metric_name}] No valid scores to evaluate.")

    overall_frac = y.mean()

    # ----- ROC curves by quantile subset -----
    quantiles = np.arange(0.96, 0.04, -quantile_step)
    cmap = plt.get_cmap(cmap_name)

    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))

    qs_used = []
    auc_scores = []
    random_auc_scores = []
    
    for i, q in enumerate(quantiles):
        thr = np.quantile(scores, q)
        mask = scores >= thr
        y_sub = y[mask]
        s_sub = scores[mask]
        
        if len(s_sub) < 100:
            continue
        
        # skip degenerate subsets
        if len(y_sub) == 0 or y_sub.sum() == 0 or y_sub.sum() == len(y_sub):
            continue

        fpr, tpr, auc = _compute_roc_auc(y_sub, s_sub)

        t = float(i) / max(1, len(quantiles) - 1)
        color = cmap(1.0 - t)
        ax_roc.plot(fpr, tpr, color=color, lw=1.5, alpha=0.7)

        qs_used.append(q)
        auc_scores.append(auc)

    qs_used = np.array(qs_used)
    auc_scores = np.array(auc_scores)

    best_idx = np.nanargmax(auc_scores)
    best_auc = float(auc_scores[best_idx])
    best_q   = float(qs_used[best_idx])
    logging.info(f"\t- {metric_name} Best AUROC {best_auc:.4f} above quantile {best_q:.3f}")

    # diagonal baseline
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title(f"ROC by score quantile ({metric_name})")

    ax_roc.text(
        1.02, 0.5,
        f"Best AUROC = {best_auc:.3f}\nQuantile ≥ {best_q:.2f}",
        transform=ax_roc.transAxes,
        va="center",
        ha="left",
        clip_on=False,
    )

    plt.tight_layout()

    # ----- BEST AUROC subset -----
    best_thr = np.quantile(scores, best_q)
    best_mask = scores >= best_thr
    y_best = y[best_mask]
    s_best = scores[best_mask]

    fig_best, auc_best, auprc_best = plot_auroc_auprc(df=df_use[best_mask], name="", score_col=score_col, label_col="is_gt")
    
    fpr_best, tpr_best, auc_best = _compute_roc_auc(y_best, s_best)

    fig_best.suptitle(
        f"Best Quantile ROC\n{metric_name}, q ≥ {best_q:.2f}, AUC={auc_best:.3f}"
    )
    plt.show()

    # ----- AUROC vs quantile figure -----
    fig_auc, ax_auc = plt.subplots(figsize=(5, 4))
    ax_auc.plot(qs_used, auc_scores, marker="o")
    ax_auc.axhline(0.5, linestyle="--", linewidth=1, color="gray", alpha=0.7)

    ax_auc.set_xlabel("Score quantile q (keep scores ≥ quantile(q))")
    ax_auc.set_ylabel("AUROC within subset")
    ax_auc.set_title(f"AUROC vs score quantile ({metric_name})")

    # highlight best point
    ax_auc.scatter([best_q], [best_auc], color="red", zorder=5, label="Best")
    ax_auc.legend()

    finite_auc = np.isfinite(auc_scores)
    if finite_auc.any():
        ymin = max(0.0, np.nanmin(auc_scores) - 0.05)
        ymax = min(1.0, np.nanmax(auc_scores) + 0.05)
        ax_auc.set_ylim(ymin, ymax)

    plt.tight_layout()

    return fig_roc, fig_auc, fig_best, best_auc, best_q

def all_feature_auroc_auprc(feature_dict):
    # -----------------------------
    # Compute metrics for each method
    # -----------------------------
    results = []
    rand_results = []
    for name, df in feature_dict.items():
        res = compute_curves(df, score_col="Score", name=name)
        
        if res is None:
            # no usable data for this feature
            continue
        
        results.append(res)
        
        rand_res = compute_curves(
            df.assign(**{"Score": create_random_distribution(df["Score"])}),
            score_col="Score",
            name=f"{name} (Randomized Scores)"
        )
        rand_results.append(rand_res)

    # -----------------------------
    # Plot PR curves on one figure
    # -----------------------------
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))

    # AUROC for each method
    for i, (res, color) in enumerate(zip(results, colors)):
        ax[0].plot(
            res["fpr"], res["tpr"],
            lw=2,
            color=color,
            label=f"{res['name']}: {res['auroc']:.2f}"
        )
        ax[0].plot(
            rand_results[i]["fpr"], rand_results[i]["tpr"],
            lw=1,
            color=color,
            linestyle="--",
        )
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("TF–TG methods vs ChIP (ROC)")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].legend(
        title="ROC Scores",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        fontsize=10
    )

    # AUPRC for each method
    for i, (res, color) in enumerate(zip(results, colors)):
        ax[1].plot(
            res["rec"], res["prec"],
            lw=2,
            color=color,
            label=f"{res['name']}: {res['auprc']:.2f}"
        )
        ax[1].plot(
            rand_results[i]["rec"], rand_results[i]["prec"],
            lw=1,
            color=color,
            linestyle="--",
        )

    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("TF–TG methods vs ChIP (Precision–Recall)")
    ax[1].set_ylim(0, 1.0)
    ax[1].set_xlim(0, 1.0)
    ax[1].legend(
        title="PRC Scores",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        fontsize=10
    )
    plt.suptitle("AUROC and AUPRC All Model Features")
    plt.tight_layout()
    
    return fig

def plot_all_method_auroc_auprc(method_dict, gt_name, target_method="MultiomicTransformer", label_col="is_gt"):
    results = []
    rand_results = []

    for name, df_m in method_dict.items():
        
        df = df_m.copy()
        
        if name != target_method:
            df["Score"] = df["Score"].abs()
        
        # Compute and plot metrics
        res = compute_curves(df, score_col="Score", name=name, balance=True, label_col=label_col)

        if res is None:
            logging.info(f"Skipping {name} on {gt_name} (no usable positives/negatives)")
            continue

        results.append(res)
        
        rand_res = compute_curves(
            df.assign(**{
                "Score": create_random_distribution(df["Score"])
            }),
            score_col="Score",
            label_col=label_col,
            name=f"{name} (Randomized Scores)",
            balance=False  # Already balanced
        )
        rand_results.append(rand_res)
        
    # If nothing usable, bail out gracefully
    if not results or not rand_results:
        logging.info(f"No valid methods for {gt_name}")
        fig, ax = plt.subplots()
        return fig, results

    # Make sure lengths match
    min_len = min(len(results), len(rand_results))
    results = results[:min_len]
    rand_results = rand_results[:min_len]

    # -----------------------------
    # Sort methods
    # -----------------------------
    paired = list(zip(results, rand_results))

    if not paired:
        logging.info(f"No valid methods for {gt_name}")
        fig, ax = plt.subplots()
        return fig, results  # results will be empty

    # Sorted for ROC: by AUROC (descending)
    paired_by_auroc = sorted(paired, key=lambda pair: pair[0]["auroc"], reverse=True)

    # Sorted for PR: by AUPRC (descending)
    paired_by_auprc = sorted(paired, key=lambda pair: pair[0]["auprc"], reverse=True)

    # -----------------------------
    # Build color/alpha map per method
    # -----------------------------
    base_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                   "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    # All base method names (no "(Randomized Scores)" suffix)
    base_method_names = {res["name"] for res in results}

    # Colors available for non-target methods (avoid explicit red)
    other_colors = [c for c in base_colors if c not in ("red", "tab:red")]
    random.shuffle(other_colors)
    other_color_cycle = cycle(other_colors)

    color_map = {}

    for m in base_method_names:
        if target_method in m:
            # Highlight MultiomicTransformer
            color_map[m] = ("red", 1.0)
        else:
            color_map[m] = (next(other_color_cycle), 0.6)

    # -----------------------------
    # Plot PR + ROC curves
    # -----------------------------
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

    title_fontsize = 14      # panel titles
    axes_fontsize = 12       # axis labels, legend text
    tick_fontsize = 10       # tick labels

    # ===== ROC panel: sorted by AUROC =====
    for res, rand_res in paired_by_auroc:
        base_name = res["name"]
        color, alpha = color_map.get(base_name, ("tab:gray", 0.6))

        ax[0].plot(
            res["fpr"], res["tpr"],
            lw=2,
            color=color,
            alpha=alpha,
            label=f"{res['name']}: {res['auroc']:.2f}"
        )
        ax[0].plot(
            rand_res["fpr"], rand_res["tpr"],
            lw=1,
            color=color,
            alpha=alpha,
            linestyle="--",
        )

    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel("False Positive Rate", fontsize=axes_fontsize)
    ax[0].set_ylabel("True Positive Rate", fontsize=axes_fontsize)
    ax[0].set_title("AUROC", fontsize=title_fontsize)
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].tick_params(axis="both", labelsize=tick_fontsize)

    leg = ax[0].legend(
        title="ROC Scores",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=axes_fontsize,
        title_fontsize=axes_fontsize,
    )
    leg.get_title().set_fontweight("bold")

    # ===== PR panel: sorted by AUPRC =====
    for res, rand_res in paired_by_auprc:
        base_name = res["name"]
        color, alpha = color_map.get(base_name, ("tab:gray", 0.6))

        ax[1].plot(
            res["rec"], res["prec"],
            lw=2,
            color=color,
            alpha=alpha,
            label=f"{res['name']}: {res['auprc']:.2f}"
        )
        ax[1].plot(
            rand_res["rec"], rand_res["prec"],
            lw=1,
            color=color,
            alpha=alpha,
            linestyle="--",
        )

    ax[1].set_xlabel("Recall", fontsize=axes_fontsize)
    ax[1].set_ylabel("Precision", fontsize=axes_fontsize)
    ax[1].set_title("AUPRC", fontsize=title_fontsize)
    ax[1].set_xlim(0, 1.0)
    ax[1].set_ylim(0, 1.0)
    ax[1].tick_params(axis="both", labelsize=tick_fontsize)

    leg = ax[1].legend(
        title="PRC Scores",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=axes_fontsize,
        title_fontsize=axes_fontsize,
    )
    leg.get_title().set_fontweight("bold")

    plt.suptitle(f"All Methods vs {gt_name}", fontsize=title_fontsize + 2)
    plt.tight_layout()
    
    return fig, results

def plot_all_results_auroc_boxplot(df, per_tf=False):
    # 1. Order methods by mean AUROC (highest → lowest)
    method_order = (
        df.groupby("name")["auroc"]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    # 2. Prepare data in that order
    data = [df.loc[df["name"] == m, "auroc"].values for m in method_order]

    feature_list = [
        "Gradient Attribution",
        "TF Knockout",
        "TF-TG Embedding Similarity",
        "Shortcut Attention"
    ]
    my_color = "#4195df"
    other_color = "#747474"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline random line
    ax.axhline(y=0.5, color="#2D2D2D", linestyle='--', linewidth=1)

    # --- Boxplot (existing styling) ---
    bp = ax.boxplot(
        data,
        tick_labels=method_order,
        patch_artist=True,
        showfliers=False
    )

    # Color boxes: light blue for your methods, grey for others
    for box, method in zip(bp["boxes"], method_order):
        if method in feature_list:
            box.set_facecolor(my_color)
        else:
            box.set_facecolor(other_color)

    # Medians in black
    for median in bp["medians"]:
        median.set_color("black")

    # --- NEW: overlay jittered points for each method ---
    for i, method in enumerate(method_order, start=1):
        y = df.loc[df["name"] == method, "auroc"].values
        if len(y) == 0:
            continue

        # Small horizontal jitter around the box center (position i)
        x = np.random.normal(loc=i, scale=0.06, size=len(y))

        # Match point color to box color
        point_color = my_color if method in feature_list else other_color

        ax.scatter(
            x, y,
            color=point_color,
            alpha=0.7,
            s=18,
            edgecolor="k",
            linewidth=0.3,
            zorder=3,
        )
        
        mean_val = y.mean()
        ax.scatter(
            i, mean_val,
            color="white",
            edgecolor="k",
            s=30,
            zorder=4,
        )

    ax.set_xlabel("Method")
    ax.set_ylabel("AUROC across ground truths")
    if per_tf == True:
        ax.set_title("per-TF AUROC Scores per method")
        ax.set_ylim((0.0, 1.0))
    else:
        ax.set_title("AUROC Scores per method")
        ax.set_ylim((0.2, 0.8))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    
    return fig

def plot_all_results_auprc_boxplot(df, per_tf=False):
    # 1. Order methods by mean AUPRC (highest → lowest)
    method_order = (
        df.groupby("name")["auprc"]
        .mean()
        .sort_values(ascending=False)
        .index
    )

    # 2. Prepare data in that order
    data = [df.loc[df["name"] == m, "auprc"].values for m in method_order]

    feature_list = [
        "Gradient Attribution",
        "TF Knockout",
        "TF-TG Embedding Similarity",
        "Shortcut Attention"
    ]
    my_color = "#4195df"
    other_color = "#747474"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline line (same style as before)
    ax.axhline(y=0.5, color="#2D2D2D", linestyle='--', linewidth=1)

    # --- Boxplot (existing styling) ---
    bp = ax.boxplot(
        data,
        tick_labels=method_order,
        patch_artist=True,
        showfliers=False
    )

    # Color boxes: light blue for your methods, grey for others
    for box, method in zip(bp["boxes"], method_order):
        if method in feature_list:
            box.set_facecolor(my_color)
        else:
            box.set_facecolor(other_color)

    # Medians in black
    for median in bp["medians"]:
        median.set_color("black")

    # --- overlay jittered points for each method ---
    for i, method in enumerate(method_order, start=1):
        y = df.loc[df["name"] == method, "auprc"].values
        if len(y) == 0:
            continue

        # Small horizontal jitter around the box center (position i)
        x = np.random.normal(loc=i, scale=0.06, size=len(y))

        # Match point color to box color
        point_color = my_color if method in feature_list else other_color

        ax.scatter(
            x, y,
            color=point_color,
            alpha=0.7,
            s=18,
            edgecolor="k",
            linewidth=0.3,
            zorder=3,
        )
        
        mean_val = y.mean()
        ax.scatter(
            i, mean_val,
            color="white",
            edgecolor="k",
            s=30,
            zorder=4,
        )

    ax.set_xlabel("Method")
    ax.set_ylabel("AUPRC across ground truths")
    if per_tf == True:
        ax.set_title("per-TF AUPRC Scores per method")
        ax.set_ylim((0.0, 1.0))
    else:
        ax.set_title("AUPRC Scores per method")
        ax.set_ylim((0.2, 0.8))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    
    return fig

def locate_last_checkpoint(experiment_dir):
    checkpoint_files = sorted(experiment_dir.glob("checkpoint_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {experiment_dir}")
    last_checkpoint = checkpoint_files[-1]
    return last_checkpoint.name

if __name__ == "__main__":    
    arg_parser = argparse.ArgumentParser(description="Run AUROC and AUPR testing for trained models")

    arg_parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment to test")
    arg_parser.add_argument("--training_num", type=str, required=False, default="model_training_001", help="Training number folder to test")
    arg_parser.add_argument("--experiment_dir", type=Path, required=True, help="Full path to the experiment directory to test")
    arg_parser.add_argument("--model_file", type=str, required=False, default="trained_model.pt", help="Name of the trained model file (default: trained_model.pt)")

    args = arg_parser.parse_args()

    experiment = args.experiment
    experiment_dir = Path(args.experiment_dir)
    training_num = args.training_num if args.training_num else "model_training_001"
    
    FIG_DIR = Path("/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/FIGURES")
    FIG_DATA = Path("/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/FIGURE_DATA")

    if "chr19" in [p.name for p in Path(experiment_dir / experiment).iterdir()] and experiment != "mESC_no_scale_linear":
        EXPERIMENT_DIR = experiment_dir / experiment / "chr19" / training_num
        
        exp_fig_dir = FIG_DIR / experiment 
        exp_fig_data_dir = FIG_DATA / experiment
    else:
        EXPERIMENT_DIR = experiment_dir / experiment / training_num
        
        exp_fig_dir = FIG_DIR / experiment
        exp_fig_data_dir = FIG_DATA / experiment

    logging.info(f"Selected experiment directory: {EXPERIMENT_DIR}")

    if not os.path.exists(exp_fig_data_dir):
        os.makedirs(exp_fig_data_dir, exist_ok=True)
        
    if not os.path.exists(exp_fig_dir):
        os.makedirs(exp_fig_dir, exist_ok=True)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH  = 32
    TG_CHUNK = 64
    
    # Plot GPU usage
    for file in EXPERIMENT_DIR.glob("gpu_usage*.csv"):
        print(f"Processing GPU log file: {file.name}")
        gpu_file = format_gpu_usage_file(file)
        gpu_fig = plot_gpu_usage({"GPU Usage": gpu_file})

        logging.info("Plotting GPU memory usage...")
        gpu_fig.savefig(EXPERIMENT_DIR / "gpu_memory_requirements.svg")
        gpu_fig.savefig(exp_fig_dir / "gpu_memory_requirements.svg")
    else:
        logging.info("No GPU usage log found, skipping GPU memory plotting.")
    
    if os.path.exists(EXPERIMENT_DIR / "training_log.csv"):
        # Load training log
        training_log_df = pd.read_csv(os.path.join(EXPERIMENT_DIR, "training_log.csv"), header=0)
        training_log_df.dropna(inplace=True)
        
        # Plot Training R2 across epochs
        logging.info("Plotting training R2 across epochs...")
        train_r2_fig = plot_R2_across_epochs(training_log_df)
        train_r2_fig.savefig(os.path.join(EXPERIMENT_DIR, "eval_results_pearson_corr.svg"))
        train_r2_fig.savefig(os.path.join(EXPERIMENT_DIR, "eval_results_pearson_corr.png"), dpi=200)
        train_r2_fig.savefig(exp_fig_dir / "eval_results_pearson_corr.svg")
        training_log_df.to_csv(exp_fig_data_dir / "training_log_df.csv")
        
        # Plot Train Val Loss curves
        logging.info("Plotting training and validation loss curves...")
        train_val_loss_fig = plot_train_val_loss(training_log_df)
        train_val_loss_fig.savefig(os.path.join(EXPERIMENT_DIR, "train_val_loss_curves.svg"))
        train_val_loss_fig.savefig(os.path.join(EXPERIMENT_DIR, "train_val_loss_curves.png"), dpi=200)
        train_val_loss_fig.savefig(exp_fig_dir / "train_val_loss_curves.svg")
    else:
        logging.info("No training log found, skipping training R2 and loss plotting.")
    
    if not os.path.exists(EXPERIMENT_DIR / args.model_file):
        # Try to load the last checkpoint if the specified model file does not exist
        logging.info(f"Model file {args.model_file} not found, attempting to load last checkpoint...")
        last_checkpoint_file = locate_last_checkpoint(EXPERIMENT_DIR)
        args.model_file = last_checkpoint_file
        logging.info(f"Using last checkpoint file: {args.model_file}")

    # Load model and test data
    logging.info("Loading model and test data...")
    model, test_loader, tg_scaler, tf_scaler, state = load_model(
        selected_experiment_dir=EXPERIMENT_DIR,
        checkpoint_file=args.model_file,
        device=DEVICE
    )
    
    # Plot TG predictions
    logging.info("Plotting TG predictions...")
    per_gene_mean_fig, all_point_fig, tgts_clean, preds_clean, mean_true, mean_pred = plot_model_tg_predictions(
        device=DEVICE,
        model=model,
        test_loader=test_loader,
        tg_scaler=tg_scaler,
        tf_scaler=tf_scaler,
    )
    
    per_gene_mean_fig.savefig(EXPERIMENT_DIR / "per_gene_mean_expression.png", dpi=200)
    per_gene_mean_fig.savefig(exp_fig_dir / "per_gene_mean_expression.svg")
    
    all_point_fig.savefig(EXPERIMENT_DIR / "test_set_r2_distribution.png", dpi=200)
    all_point_fig.savefig(exp_fig_dir / "test_set_r2_distribution.svg")
    
    # Save R2 prediction data
    logging.info("Saving R2 prediction data...")
    r2_acc_dir = exp_fig_data_dir / "tg_pred_acc_data"
    if not os.path.isdir(r2_acc_dir):
        os.makedirs(r2_acc_dir, exist_ok=True)
    np.save(r2_acc_dir / "tgts_clean.npy", tgts_clean)
    np.save(r2_acc_dir / "preds_clean.npy", preds_clean)
    
    # Saving per-gene mean expression data
    np.save(r2_acc_dir / "mean_true.npy", mean_true)
    np.save(r2_acc_dir / "mean_pred.npy", mean_pred)

    logging.info("All done!")
    
    # ---- AUROC Testing Plots ----
    df_results = pd.read_csv(EXPERIMENT_DIR / "auroc_auprc_results_detailed.csv", index_col=None)
    df_pooled = (
        df_results
        .groupby(["gt_name", "name"], as_index=False)
        .agg(
            auroc=("auroc", "mean"),
            auprc=("auprc", "mean"),
        )
    )

    # Optional: keep a 'sample' column so existing plotting
    # functions expecting it keep working
    df_pooled["sample"] = "POOLED"

    df_results.to_csv(
        exp_fig_data_dir / "auroc_auprc_pooled_heatmap_data.csv",
        index=False,
    )

    auroc_heat_fig = plot_method_gt_heatmap(df_results, metric="auroc")
    auroc_heat_fig.savefig(
        EXPERIMENT_DIR / "method_gt_auroc_heatmap_pooled.png",
        dpi=300,
    )
    auroc_heat_fig.savefig(
        exp_fig_dir / f"method_gt_auroc_heatmap_pooled.svg"
    )
    plt.close(auroc_heat_fig)

    auprc_heat_fig = plot_method_gt_heatmap(df_results, metric="auprc")
    auprc_heat_fig.savefig(
        EXPERIMENT_DIR / "method_gt_auprc_heatmap_pooled.png",
        dpi=300,
    )
    auprc_heat_fig.savefig(
        exp_fig_dir / f"method_gt_auprc_heatmap_pooled.svg"
    )
    plt.close(auprc_heat_fig)

    # ------------------------------------------------------------
    # 2) Method ranking on pooled AUROCs
    # ------------------------------------------------------------
    method_rank_pooled = (
        df_pooled.groupby("name")
        .agg(
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            mean_auprc=("auprc", "mean"),
            std_auprc=("auprc", "std"),
            n_gt=("gt_name", "nunique"),
        )
        .sort_values("mean_auroc", ascending=False)
    )

    logging.info("\n=== Method ranking by POOLED AUROC (all TFs together) ===")
    logging.info(method_rank_pooled)

    method_rank_pooled.to_csv(
        EXPERIMENT_DIR / "method_ranking_by_auroc_pooled.csv"
    )

    # ------------------------------------------------------------
    # 3) Per-GT table for boxplots (still one row per (gt_name, method))
    # ------------------------------------------------------------
    per_gt_rank = (
        df_pooled
        .sort_values(["gt_name", "auroc"], ascending=[True, False])
        [["gt_name", "sample", "name", "auroc", "auprc"]]
    )
    per_gt_rank.to_csv(
        EXPERIMENT_DIR / "per_gt_method_aucs_detailed.csv",
        index=False,
    )

    # Boxplots reflect pooled AUROC/AUPRC per method & GT
    all_auroc_boxplots = plot_all_results_auroc_boxplot(per_gt_rank)
    all_auprc_boxplots = plot_all_results_auprc_boxplot(per_gt_rank)

    all_auroc_boxplots.savefig(
        EXPERIMENT_DIR / "all_results_auroc_boxplot_pooled.png",
        dpi=300,
    )
    all_auprc_boxplots.savefig(
        EXPERIMENT_DIR / "all_results_auprc_boxplot_pooled.png",
        dpi=300,
    )
    all_auroc_boxplots.savefig(
        exp_fig_dir / f"all_results_auroc_boxplot_pooled.svg"
    )
    all_auprc_boxplots.savefig(
        exp_fig_dir / f"all_results_auprc_boxplot_pooled.svg"
    )
    plt.close(all_auroc_boxplots)
    plt.close(all_auprc_boxplots)


    # ------------------------------------------------------------
    # 4) Per-TF metrics, ranking, and boxplots
    # ------------------------------------------------------------
    per_tf_metrics = pd.read_csv(EXPERIMENT_DIR / "per_tf_auroc_auprc_detailed.csv", index_col=None)
    
    
    # ===== Method 1: Mean of Each TF per Ground Truth =====
    # 4a) Create per-TF method ranking
    # Average across TFs within each (method, GT)
    method_gt_avg = (
        per_tf_metrics
        .groupby(['method', 'gt_name'], as_index=False)
        .agg(
            auroc=('auroc', 'mean'),
            auprc=('auprc', 'mean'),
            n_tfs=('tf', 'nunique'),
        )
    )
    
    # Plots a boxplot of per-TF AUROC/AUPRC scores for each method, aggregated by GT
    per_tf_df_agg_by_gt = method_gt_avg.copy().rename(columns={'method': 'name'})
    per_tf_auroc_agg_by_gt_boxplot = plot_all_results_auroc_boxplot(per_tf_df_agg_by_gt, per_tf=True)
    per_tf_auprc_agg_by_gt_boxplot = plot_all_results_auprc_boxplot(per_tf_df_agg_by_gt, per_tf=True)
    
    per_tf_auroc_agg_by_gt_boxplot.savefig(EXPERIMENT_DIR / "per_tf_auroc_agg_by_gt.png",dpi=300)
    per_tf_auprc_agg_by_gt_boxplot.savefig(EXPERIMENT_DIR / "per_tf_auprc_agg_by_gt.png",dpi=300)
    
    per_tf_auroc_agg_by_gt_boxplot.savefig(exp_fig_dir / f"per_tf_auroc_agg_by_gt.svg")
    per_tf_auprc_agg_by_gt_boxplot.savefig(exp_fig_dir / f"per_tf_auprc_agg_by_gt.svg")
    plt.close(per_tf_auroc_agg_by_gt_boxplot)
    plt.close(per_tf_auprc_agg_by_gt_boxplot)
    
    
    per_tf_df_agg_by_gt.to_csv(
        EXPERIMENT_DIR / "per_tf_auroc_auprc_agg_by_gt.csv",
        index=False,
    )
    
    # ===== Method 2: Aggregate across GTs within each method to get overall per-TF ranking =====
    method_rank_per_tf = (
        per_tf_metrics
        .groupby(['method', 'tf'], as_index=False)
        .agg(
            auroc=('auroc', 'mean'),
            auprc=('auprc', 'mean'),
            n_gt=('gt_name', 'nunique'),
        )
    )
    
    logging.info("\n=== Method ranking by PER-TF AUROC (averaged across TFs) ===")
    logging.info(method_rank_per_tf.groupby("method").agg(mean_auroc=('auroc', 'mean')).sort_values("mean_auroc", ascending=False))
    
    method_rank_per_tf.to_csv(
        EXPERIMENT_DIR / "method_ranking_by_per_tf_auroc.csv"
    )
    
    # 4b) Create per-TF boxplots
    # Plots each individual (method, TF) pair
    per_tf_for_plot = method_rank_per_tf.rename(columns={'method': 'name'})
    
    per_tf_for_plot.to_csv(
        exp_fig_data_dir / "per_tf_auroc_auprc_data.csv",
        index=False,
    )
    
    logging.info("Creating per-TF AUROC and AUPRC boxplots...")
    per_tf_auroc_agg_by_tf_boxplot = plot_all_results_auroc_boxplot(per_tf_for_plot, per_tf=True)
    per_tf_auprc_agg_by_tf_boxplot = plot_all_results_auprc_boxplot(per_tf_for_plot, per_tf=True)
    
    per_tf_auroc_agg_by_tf_boxplot.savefig(EXPERIMENT_DIR / "per_tf_auroc_agg_by_tf_boxplot.png",dpi=300)
    per_tf_auprc_agg_by_tf_boxplot.savefig(EXPERIMENT_DIR / "per_tf_auprc_agg_by_tf_boxplot.png",dpi=300)
    
    per_tf_auroc_agg_by_tf_boxplot.savefig(exp_fig_dir / f"per_tf_auroc_agg_by_tf_boxplot.svg")
    per_tf_auprc_agg_by_tf_boxplot.savefig(exp_fig_dir / f"per_tf_auprc_agg_by_tf_boxplot.svg")
    plt.close(per_tf_auroc_agg_by_tf_boxplot)
    plt.close(per_tf_auprc_agg_by_tf_boxplot)
    
    # ===== Method 3: Calculate the Mean TF AUROC/AUPRC Across All GTs =====
    per_tf_mean_across_gt = (
        per_tf_metrics
        .groupby(['method', 'tf'], as_index=False)
        .agg(
            auroc=('auroc', 'mean'),
            auprc=('auprc', 'mean'),
            n_gt=('gt_name', 'nunique'),
        )
    )

    per_tf_mean_across_gt.rename(columns={'tf': 'name'}, inplace=True)
    
    per_tf_mean_across_gt.to_csv(
        EXPERIMENT_DIR / "per_tf_mean_auroc_auprc_across_gt.csv",
        index=False,
    )
     #Plot the boxplots and save data like above methods
    per_tf_mean_auroc_boxplot = plot_all_results_auroc_boxplot(per_tf_mean_across_gt, per_tf=True)
    per_tf_mean_auprc_boxplot = plot_all_results_auprc_boxplot(per_tf_mean_across_gt, per_tf=True)
    
    per_tf_mean_auroc_boxplot.savefig(EXPERIMENT_DIR / "per_tf_mean_auroc_boxplot.png",dpi=300)
    per_tf_mean_auprc_boxplot.savefig(EXPERIMENT_DIR / "per_tf_mean_auprc_boxplot.png",dpi=300)
    
    per_tf_mean_auroc_boxplot.savefig(exp_fig_dir / f"per_tf_mean_auroc_boxplot.svg")
    per_tf_mean_auprc_boxplot.savefig(exp_fig_dir / f"per_tf_mean_auprc_boxplot.svg")
    plt.close(per_tf_mean_auroc_boxplot)
    plt.close(per_tf_mean_auprc_boxplot)
        
    # Save pooled per-GT metrics
    per_gt_rank.to_csv(
        EXPERIMENT_DIR / "per_gt_method_aucs_pooled.csv",
        index=False,
    )
    
    # 4a) Create per-TF method ranking
    method_gt_avg = (
        per_tf_metrics
        .groupby(['method', 'gt_name'], as_index=False)
        .agg(
            auroc=('auroc', 'mean'),
            auprc=('auprc', 'mean'),
            n_tfs=('tf', 'nunique'),
        )
    )

    # For heatmap input, rename 'method' -> 'name'
    method_gt_avg_for_heatmap = method_gt_avg.rename(columns={'method': 'name'})
    
    method_gt_avg_for_heatmap.to_csv(
        EXPERIMENT_DIR / "per_tf_auroc_auprc_data.csv",
        index=False,
    )

    # Per-TF AUROC heatmap
    auroc_heat_fig = plot_method_gt_heatmap(
        method_gt_avg_for_heatmap,
        metric="auroc",
        per_tf=True,
    )
    auroc_heat_fig.savefig(EXPERIMENT_DIR / "per_tf_auroc_heatmap.png", dpi=300)
    auroc_heat_fig.savefig(exp_fig_dir / f"per_tf_auroc_heatmap.svg")
    plt.close(auroc_heat_fig)

    # Per-TF AUPRC heatmap
    auprc_heat_fig = plot_method_gt_heatmap(
        method_gt_avg_for_heatmap,
        metric="auprc",
        per_tf=True,
    )
    auprc_heat_fig.savefig(EXPERIMENT_DIR / "per_tf_auprc_heatmap.png", dpi=300)
    auprc_heat_fig.savefig(exp_fig_dir / f"per_tf_auprc_heatmap.svg")
    plt.close(auprc_heat_fig)
    
    

    
    
    
