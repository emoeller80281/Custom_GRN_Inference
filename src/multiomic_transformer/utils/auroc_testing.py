# transformer_testing.py
import os, sys, json, re
import joblib
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import random
from itertools import cycle

from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from scipy.stats import rankdata
from re import sub
from tqdm import tqdm
from sklearn.metrics import r2_score
import logging
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from typing import Optional

import argparse

import sys

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
SRC_DIR = str(Path(PROJECT_DIR) / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')

from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.datasets.dataset_refactor import MultiChromosomeDataset, SimpleScaler, fit_simple_scalers

GROUND_TRUTH_DIR = Path(PROJECT_DIR, "data/ground_truth_files")

def load_ground_truth(ground_truth_file):
    if ground_truth_file.suffix == ".csv":
        sep = ","
    elif ground_truth_file.suffix == ".tsv":
        sep="\t"
        
    ground_truth_df = pd.read_csv(ground_truth_file, sep=sep, on_bad_lines="skip", engine="python")
    
    if "chip" in ground_truth_file.name and "atlas" in ground_truth_file.name:
        ground_truth_df = ground_truth_df[["source_id", "target_id"]]

    ground_truth_df = ground_truth_df.rename(columns={ground_truth_df.columns[0]: "Source", ground_truth_df.columns[1]: "Target"})
    ground_truth_df["Source"] = ground_truth_df["Source"].astype(str).str.upper()
    ground_truth_df["Target"] = ground_truth_df["Target"].astype(str).str.upper()
        
    return ground_truth_df

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

def tfwise_rank_normalize(df, score_col, new_col):
    """
    Add a TF-wise rank-normalized version of `score_col` in [0,1].
    Does NOT drop any rows; zeros just get low ranks within their TF.
    """
    df = df.copy()

    def _rank_to_01(s):
        arr = s.to_numpy(dtype=float)
        if len(arr) == 1:
            # single edge for this TF → neutral rank
            return np.full_like(arr, 0.5, dtype=float)
        ranks = rankdata(arr, method="average")  # 1..n
        return (ranks - 1) / (len(arr) - 1)      # 0..1

    df[new_col] = df.groupby("tf")[score_col].transform(_rank_to_01)
    return df

def eval_method(df, score_col, label_col="is_gt", name="", balance=True):
    """Compute AUROC, AUPRC, PR and ROC curves for a given score column."""
    df = df.dropna(subset=[score_col]).copy()

    if balance:
        df_bal = balance_pos_neg(df, label_col=label_col, random_state=0)
    else:
        df_bal = df

    y = df_bal[label_col].astype(int).values
    s = df_bal[score_col].values
    
    fig, auroc, auprc = plot_auroc_auprc(df, name, score_col, label_col=label_col)
    
    return fig, auroc, auprc

def precision_at_k(scores, y, ks):
    order = np.argsort(-scores)
    y_sorted = y[order]
    return {k: y_sorted[:k].mean() for k in ks}

def load_vocab(selected_experiment_dir):
    id2name = torch.load(selected_experiment_dir / "tf_tg_vocab_id2name.pt", map_location="cpu")
    tf_names = list(id2name["tf_id2name"])
    tg_names = list(id2name["tg_id2name"])

    return tf_names, tg_names

def load_tf_tg_embedding_similarity(selected_experiment_dir, tf_names, tg_names):
    emb = torch.load(selected_experiment_dir / "tf_tg_embeddings_final.pt", map_location="cpu")
    tf_emb = emb["tf_emb"]   # [T, d]
    tg_emb = emb["tg_emb"]   # [G, d]

    tf_norm = F.normalize(tf_emb, p=2, dim=1)
    tg_norm = F.normalize(tg_emb, p=2, dim=1)
    sim = (tf_norm @ tg_norm.T).cpu().numpy()   # [T, G]

    # Optional: row-wise z-score per TF
    mu = sim.mean(axis=1, keepdims=True)
    sigma = sim.std(axis=1, keepdims=True) + 1e-6
    cosine_z = (sim - mu) / sigma

    df_cos = matrix_to_df(cosine_z, tf_names, tg_names, "Score")
    df_cos = df_cos.dropna(subset=["Score"])
    
    return df_cos

def load_shortcut_attention(tf_names, tg_names, model, device, motif_mask=None):
    shortcut_scores = None
    
    with torch.no_grad():
        T = model.tf_identity_emb.num_embeddings
        G = model.tg_identity_emb.num_embeddings

        tf_ids = torch.arange(T, device=device, dtype=torch.long)
        tg_ids = torch.arange(G, device=device, dtype=torch.long)

        tf_id_emb = model.tf_identity_emb(tf_ids)    # [T, d]
        tg_emb    = model.tg_identity_emb(tg_ids)    # [G, d]

        dummy_tf_expr = torch.ones(1, T, device=device)

        if motif_mask is not None:
            motif_mask = motif_mask.to(device)

        _, attn = model.shortcut_layer(
            tg_emb, tf_id_emb, dummy_tf_expr, motif_mask=motif_mask
        ) 

        shortcut_scores = attn.T.cpu().numpy()  # [T, G]
    
    shortcut_attn_df = matrix_to_df(shortcut_scores, tf_names, tg_names, "Score")
        
    shortcut_attn_df = shortcut_attn_df.dropna(subset=["Score"])
        
    return shortcut_attn_df

def load_gradient_attribution_matrix(selected_experiment_dir, tf_names, tg_names):
    # --- load gradient attribution matrix ---
    grad = np.load(selected_experiment_dir / "tf_tg_grad_attribution.npy")  # shape [T, G]
    assert grad.shape == (len(tf_names), len(tg_names))

    # Optional: handle NaNs
    grad = np.nan_to_num(grad, nan=0.0)

    # Use absolute gradient magnitude as importance
    grad_abs = np.abs(grad)

    # Row-wise z-score per TF (ignore NaNs if you keep them)
    row_mean = grad_abs.mean(axis=1, keepdims=True)
    row_std  = grad_abs.std(axis=1, keepdims=True) + 1e-6
    grad_z = (grad_abs - row_mean) / row_std   # [T, G]

    # Build long-form dataframe
    T, G = grad_z.shape
    tf_idx, tg_idx = np.meshgrid(np.arange(T), np.arange(G), indexing="ij")

    gradient_attrib_df = pd.DataFrame({
        "Source": np.array(tf_names, dtype=object)[tf_idx.ravel()],
        "Target": np.array(tg_names, dtype=object)[tg_idx.ravel()],
        "Score": grad_z.ravel(),
    })
    gradient_attrib_df["Source"] = gradient_attrib_df["Source"].astype(str).str.upper()
    gradient_attrib_df["Target"] = gradient_attrib_df["Target"].astype(str).str.upper()
    
    gradient_attrib_df["Score"] = (
        (gradient_attrib_df["Score"] - gradient_attrib_df["Score"].min()) / 
        (gradient_attrib_df["Score"].max() - gradient_attrib_df["Score"].min())
    )
        
    return gradient_attrib_df

def load_tf_knockout_scores(selected_experiment_dir, tf_names, tg_names):
    effect = np.load(selected_experiment_dir / "tf_tg_fullmodel_knockout.npy")        # [T, G]
    counts = np.load(selected_experiment_dir / "tf_tg_fullmodel_knockout_count.npy")  # [T, G]

    mask_observed = counts > 0
    effect[~mask_observed] = np.nan

    effect_pos = np.clip(effect, 0, None)  # positives only, NaN preserved
    row_mean = np.nanmean(effect_pos, axis=1, keepdims=True)
    row_std  = np.nanstd(effect_pos,  axis=1, keepdims=True) + 1e-6
    knockout_z = (effect_pos - row_mean) / row_std

    tf_ko_df = matrix_to_df(knockout_z, tf_names, tg_names, "Score")
    tf_ko_df = tf_ko_df.dropna(subset=["Score"]).reset_index(drop=True)
    
    return tf_ko_df

def load_and_standardize_method(name: str, info: dict) -> pd.DataFrame:
    """
    Load a GRN CSV and rename tf_col/target_col/score_col -> Source/Target/Score.
    Extra columns are preserved.
    """
    if info["path"].suffix == ".tsv":
        sep = "\t"
    elif info["path"].suffix == ".csv":
        sep = ","
    
    df = pd.read_csv(info["path"], sep=sep, header=0, index_col=None)

    tf_col     = info["tf_col"]
    target_col = info["target_col"]
    score_col  = info["score_col"]

    rename_map = {
        tf_col: "Source",
        target_col: "Target",
        score_col: "Score",
    }

    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing expected columns: {missing}. Got: {list(df.columns)}")

    df = df.rename(columns=rename_map)

    df = df[["Source", "Target", "Score"]]
    df["Source"] = df["Source"].astype(str).str.upper()
    df["Target"] = df["Target"].astype(str).str.upper()

    return df

def calculate_per_tg_r2(model, test_loader, tg_names, tg_scaler, tf_scaler, state, device, checkpoint_file):
    G_total = len(state["tg_scaler_mean"])

    model.to(device).eval()

    G_total = tg_scaler.mean.shape[0]  # total number of genes

    # ---- global per-gene accumulators (unscaled space) ----
    sse_g   = torch.zeros(G_total, dtype=torch.float64)
    sumy_g  = torch.zeros(G_total, dtype=torch.float64)
    sumy2_g = torch.zeros(G_total, dtype=torch.float64)
    cnt_g   = torch.zeros(G_total, dtype=torch.float64)

    ### For overall R² and scatter:
    all_preds_for_plot = []
    all_tgts_for_plot  = []

    with torch.no_grad():
        total = len(test_loader)

        for batch_idx, batch in enumerate(
            tqdm(
                test_loader,
                desc="Evaluating on test set",
                unit="batches",
                total=total,
                miniters=max(1, total // 10),
            )
        ):            
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
                return_shortcut_contrib=False,
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
    
    assert len(tg_names) == G_total

    # counts per TG on CPU
    cnt_cpu = cnt_g.cpu().numpy()

    # Build a per-TG R² dataframe
    per_tg_r2_df = pd.DataFrame({
        "tg_idx": np.arange(G_total),
        "tg": tg_names,
        "r2": r2_g_cpu,
        "n_samples": cnt_cpu,
    })

    # Keep only genes that actually appeared and had variance
    per_tg_r2_valid = per_tg_r2_df[~np.isnan(per_tg_r2_df["r2"])].copy()

    # Sort by R²
    per_tg_r2_valid = per_tg_r2_valid.sort_values("r2", ascending=False)
    
    return per_tg_r2_valid

def evaluate_min_tg_r2_filters(
    base_edges_df,
    ground_truth_df_dict,
    tf_names,
    tg_names,
    min_r2_grid=None,
    min_edges=50,
    min_pos=10,
):
    """
    base_edges_df: DataFrame with at least ['Source','Target','Score','tg_r2']
    Returns
    -------
    results_df : DataFrame with per-threshold metrics
    baseline_macro : float, macro AUROC with NO TG R² filtering
    """

    if min_r2_grid is None:
        min_r2_grid = [-0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    total_edges_all = len(base_edges_df)
    total_tgs_all   = base_edges_df["Target"].nunique()

    results = []

    # -------------------------------------------------------
    # Helper: compute macro/micro AUROC for a given edge DF
    # -------------------------------------------------------
    def compute_macro_micro(edges_df):
        method_dict_base = {"Gradient Attribution": edges_df}
        per_tf_all_results = []

        for gt_name, ground_truth_df in ground_truth_df_dict.items():
            # Use the ground-truth TF/TG universe only (do not restrict to model vocab)
            chip_valid = ground_truth_df
            gt_edges = set(zip(chip_valid["Source"], chip_valid["Target"]))
            gt_tfs   = set(chip_valid["Source"])
            gt_tgs   = set(chip_valid["Target"])

            for method_name, method_df in method_dict_base.items():
                filtered_df = filter_df_to_gene_set(
                    method_df.copy(), gt_tfs, gt_tgs
                )
                labeled_df = label_edges(filtered_df, gt_edges)

                per_tf_df = compute_per_tf_metrics(
                    labeled_df,
                    score_col="Score",
                    label_col="is_gt",
                    tf_col="Source",
                    min_edges=min_edges,
                    min_pos=min_pos,
                    balance_for_auprc=True,
                )

                if per_tf_df.empty:
                    # nothing for this GT
                    continue

                per_tf_df["method"]  = method_name
                per_tf_df["gt_name"] = gt_name
                per_tf_all_results.append(per_tf_df)

        if not per_tf_all_results:
            return np.nan, np.nan

        per_tf_metrics = pd.concat(per_tf_all_results, ignore_index=True)
        df = per_tf_metrics.query("method == 'Gradient Attribution'")

        macro_auroc = df.groupby("tf")["auroc"].mean().mean()
        micro_auroc = (
            df.groupby("tf")
              .apply(lambda g: g["auroc"].mean() * g["n_edges"].sum(), include_groups=False)
              .sum() / df["n_edges"].sum()
        )
        return macro_auroc, micro_auroc

    # ---------------------------
    # 1) Baseline (no R² filter)
    # ---------------------------
    baseline_macro, baseline_micro = compute_macro_micro(base_edges_df)
    print(
        f"Baseline (no TG R² filter) macro/micro AUROC: "
        f"{baseline_macro:.3f} / {baseline_micro:.3f}"
    )

    # ---------------------------
    # 2) R²-threshold sweeps
    # ---------------------------
    for min_r2 in min_r2_grid:
        print(f"\n=== Evaluating min_tg_r2 >= {min_r2:.2f} ===")

        # Filter by TG R²
        edges_filt = base_edges_df[base_edges_df["tg_r2"] >= min_r2].copy()
        n_edges_filt = len(edges_filt)
        n_tgs_filt   = edges_filt["Target"].nunique()

        if n_edges_filt == 0:
            print("  No edges left after filtering; skipping this threshold.")
            continue

        macro_auroc, micro_auroc = compute_macro_micro(edges_filt)
        print(f"  Macro / micro AUROC: {macro_auroc:.3f} / {micro_auroc:.3f}")

        results.append({
            "min_tg_r2": min_r2,
            "n_edges_kept": n_edges_filt,
            "frac_edges_kept": n_edges_filt / total_edges_all,
            "n_tgs_kept": n_tgs_filt,
            "frac_tgs_kept": n_tgs_filt / total_tgs_all,
            "macro_auroc": macro_auroc,
            "micro_auroc": micro_auroc,
            "baseline_macro": baseline_macro,
            "baseline_micro": baseline_micro,
        })

    results_df = pd.DataFrame(results).sort_values("min_tg_r2")
    return results_df, baseline_macro, baseline_micro

def filter_grad_attrib_by_tg_r2(
    grad_attrib_df: pd.DataFrame,
    per_tg_r2_valid: pd.DataFrame,
    r2_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Keep only Gradient Attribution edges whose *target gene* has R² >= r2_threshold.
    
    grad_attrib_df: DataFrame with at least ['Source', 'Target', 'Score']
    per_tg_r2_valid: DataFrame from calculate_per_tg_r2 with ['tg', 'r2', 'n_samples']
    r2_threshold: minimum per-TG R² required to keep edges.
    """

    # Map TG -> R²
    r2_map = per_tg_r2_valid[["tg", "r2"]].copy()

    # Attach R² to each GA edge (left join on Target)
    df = grad_attrib_df.merge(
        r2_map,
        how="left",
        left_on="Target",
        right_on="tg",
    )

    before_edges = len(df)
    before_tgs   = df["Target"].nunique()

    # Keep only edges whose TG has a defined R² and passes the threshold
    mask = df["r2"].notna() & (df["r2"] >= r2_threshold)
    df_filt = df.loc[mask].copy()

    after_edges = len(df_filt)
    after_tgs   = df_filt["Target"].nunique()

    print(f"Filtering GA edges by TG R² ≥ {r2_threshold:.2f}")
    print(f"  Edges kept: {after_edges} / {before_edges} "
          f"({after_edges / max(1,before_edges):.3%})")
    print(f"  TGs kept:   {after_tgs} / {before_tgs} "
          f"({after_tgs / max(1,before_tgs):.3%})")

    # Drop the duplicate 'tg' column; keep 'r2' so you can use it later if you like
    df_filt.drop(columns=["tg"], inplace=True)

    # Optional: reorder columns
    df_filt = df_filt[["Source", "Target", "Score", "r2"] + 
                      [c for c in df_filt.columns if c not in ("Source","Target","Score","r2")]]

    return df_filt

def compute_per_tf_metrics(
    df,
    score_col: str = "Score",
    label_col: str = "is_gt",
    tf_col: str = "Source",
    min_edges: int = 10,
    min_pos: int = 1,
    balance_for_auprc: bool = True,   # renamed intent
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Compute per-TF AUROC and AUPRC.

    - AUROC is computed on the ORIGINAL (unbalanced) outgoing edges per TF.
    - AUPRC is computed on a BALANCED subsample per TF (if balance_for_auprc=True).

    Returns a DataFrame with one row per TF:
        ['tf', 'auroc', 'auprc', 'n_edges', 'n_pos', 'n_neg', 'n_edges_bal']
    """
    df = df.dropna(subset=[score_col]).copy()

    records = []
    for tf, sub in df.groupby(tf_col):
        n_edges = len(sub)
        n_pos = int(sub[label_col].sum())
        n_neg = n_edges - n_pos

        # basic filters to avoid degenerate metrics
        if n_edges < min_edges:
            continue
        if n_pos < min_pos or n_neg == 0:
            continue

        # ---------- AUROC on unbalanced ----------
        y_full = sub[label_col].astype(int).values
        s_full = sub[score_col].values

        # need both classes present
        if len(np.unique(y_full)) < 2:
            continue

        try:
            auroc = roc_auc_score(y_full, s_full)
        except ValueError:
            continue

        # ---------- AUPRC on balanced (optional) ----------
        if balance_for_auprc:
            sub_pr = balance_pos_neg(sub, label_col=label_col, random_state=random_state)
        else:
            sub_pr = sub

        y_pr = sub_pr[label_col].astype(int).values
        s_pr = sub_pr[score_col].values

        # need both classes present after balancing too
        if len(np.unique(y_pr)) < 2:
            # If balancing collapses a TF to one class, skip AUPRC (or set NaN)
            auprc = np.nan
        else:
            try:
                auprc = average_precision_score(y_pr, s_pr)
            except ValueError:
                auprc = np.nan

        records.append(
            {
                "tf": tf,
                "auroc": float(auroc),
                "auprc": (float(auprc) if np.isfinite(auprc) else np.nan),
                "n_edges": int(n_edges),
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
                "n_edges_bal": int(len(sub_pr)),
            }
        )

    return pd.DataFrame.from_records(records)


def get_last_checkpoint(exp_dir: Path) -> Optional[str]:
    """
    Find the checkpoint_<N>.pt file with the largest N in exp_dir.
    Returns the filename (not full path), or None if no checkpoints are found.
    """
    ckpt_files = list(exp_dir.glob("checkpoint_*.pt"))
    if not ckpt_files:
        return None

    def extract_step(path: Path) -> int:
        m = re.search(r"checkpoint_(\d+)\.pt$", path.name)
        return int(m.group(1)) if m else -1

    last_ckpt = max(ckpt_files, key=extract_step)
    logging.info(f"Selected last checkpoint: {last_ckpt.name} from {exp_dir}")
    return last_ckpt.name

if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser(description="Run AUROC and AUPR testing for trained models")

    arg_parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment to test")
    arg_parser.add_argument("--training_num", type=str, required=False, default="model_training_001", help="Training number folder to test")
    arg_parser.add_argument("--experiment_dir", type=Path, required=True, help="Full path to the experiment directory to test")
    arg_parser.add_argument("--model_file", type=str, required=False, default="trained_model.pt", help="Name of the trained model file (default: trained_model.pt)")
    arg_parser.add_argument("--dataset_type", type=str, required=True, choices=["mESC", "macrophage", "k562"], help="Type of dataset: mESC, macrophage, or k562")
    arg_parser.add_argument("--sample_name_list", type=str, nargs='+', required=False, default=[], help="List of sample names to include in the evaluation (optional)")
    arg_parser.add_argument("--r2_threshold", type=float, required=False, default=None, help="R² threshold for filtering Gradient Attribution edges (optional)")

    args = arg_parser.parse_args()

    experiment = args.experiment
    training_num = args.training_num if args.training_num else "model_training_001" 
    experiment_dir = Path(args.experiment_dir)
    dataset_type = args.dataset_type
    
    sample_name_list = args.sample_name_list
    if "chr19" in [p.name for p in Path(experiment_dir / experiment).iterdir()]:
        EXPERIMENT_DIR = experiment_dir / experiment / "chr19" / training_num
    else:
        EXPERIMENT_DIR = experiment_dir / experiment / training_num

    if dataset_type.lower() == "macrophage":
        ground_truth_file_dict = {
            "ChIP-Atlas": GROUND_TRUTH_DIR / "chipatlas_macrophage.csv",
            "RN204": GROUND_TRUTH_DIR / "rn204_macrophage_human_chipseq.tsv",
        }
    
    elif dataset_type.lower() == "mesc":
        ground_truth_file_dict = {
            "ChIP-Atlas": GROUND_TRUTH_DIR / "chip_atlas_tf_peak_tg_dist.csv",
            # "RN111_RN112": GROUND_TRUTH_DIR / "filtered_RN111_and_RN112_mESC_E7.5_rep1.tsv",
            # "ORTI": GROUND_TRUTH_DIR / "ORTI_ground_truth_TF_TG.csv",
            "RN111": GROUND_TRUTH_DIR / "RN111.tsv",
            "RN112": GROUND_TRUTH_DIR / "RN112.tsv",
            "RN114": GROUND_TRUTH_DIR / "RN114.tsv",
            "RN116": GROUND_TRUTH_DIR / "RN116.tsv",
        }
    elif dataset_type.lower() == "k562":
        ground_truth_file_dict = {
            "ChIP-Atlas": GROUND_TRUTH_DIR / "chipatlas_K562.csv",
            "RN117": GROUND_TRUTH_DIR / "RN117.tsv",
            # "RN118": GROUND_TRUTH_DIR / "RN118.tsv",
            # "RN119": GROUND_TRUTH_DIR / "RN119.tsv",
        }
    elif dataset_type.lower() == "t_cell":
        ground_truth_file_dict = {
            "ChIP-Atlas": GROUND_TRUTH_DIR / "chipatlas_t_cell.csv",
        }
    elif dataset_type.lower() == "ipsc":
        ground_truth_file_dict = {
            "ChIP-Atlas": GROUND_TRUTH_DIR / "chipatlas_iPSC.csv",
            "RN108": GROUND_TRUTH_DIR / "RN108.tsv",
        }
    
    FIG_DIR = Path("/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/FIGURES")
    FIG_DATA = Path("/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/FIGURE_DATA")

    exp_fig_dir = FIG_DIR / experiment / training_num
    exp_fig_data_dir = FIG_DATA / experiment / training_num

    if not os.path.exists(exp_fig_data_dir):
        os.makedirs(exp_fig_data_dir)
        
    if not os.path.exists(exp_fig_dir):
        os.makedirs(exp_fig_dir)
    
    ground_truth_df_dict = {}

    # Loop through each ground truth dataset and load each file
    for i, (gt_name, ground_truth_file) in enumerate(ground_truth_file_dict.items(), start=1):
        print(f"Loading {gt_name} ({i}/{len(ground_truth_file_dict)})")

        # --- Ground truth & sets ---
        ground_truth_df = load_ground_truth(ground_truth_file)
        
        ground_truth_df_dict[gt_name] = ground_truth_df
        print(f"  - TFs: {ground_truth_df['Source'].nunique():,}, TGs: {ground_truth_df['Target'].nunique():,}, Edges: {len(ground_truth_df):,}")

    feature_to_plot = "Gradient Attribution"

    per_tf_all_results = []
    logging.info(f"\nProcessing {experiment}")
    checkpoint_file = args.model_file
    
    if not Path(EXPERIMENT_DIR / checkpoint_file).is_file():
        logging.warning(f"Trained model file {checkpoint_file} in {EXPERIMENT_DIR} does not exist. Trying last checkpoint.")
        last_checkpoint = get_last_checkpoint(EXPERIMENT_DIR)
        if last_checkpoint is None:
            raise ValueError(f"Could not find last checkpoint in {EXPERIMENT_DIR}")
        else:
            logging.info(f"Selected last checkpoint: {last_checkpoint}")
            checkpoint_file = last_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, test_loader, tg_scaler, tf_scaler, state = load_model(EXPERIMENT_DIR, checkpoint_file, device)
    model.eval()
    
    # Vocab
    tf_names, tg_names = load_vocab(EXPERIMENT_DIR)
    grad_attrib_df = load_gradient_attribution_matrix(EXPERIMENT_DIR, tf_names, tg_names)
    
    auroc_by_tg_r2_threshold_file = EXPERIMENT_DIR / "auroc_by_tg_r2_threshold.csv"
    if not auroc_by_tg_r2_threshold_file.is_file():
        per_tg_r2_valid = calculate_per_tg_r2(
            model, test_loader, tg_names, tg_scaler, tf_scaler, state, device, checkpoint_file
        )
    
        # Attach raw r2 to each GA edge
        grad_attrib_with_r2 = (
            grad_attrib_df
            .merge(
                per_tg_r2_valid[["tg", "r2"]],
                how="left",
                left_on="Target",
                right_on="tg",
            )
            .drop(columns=["tg"])  # we already have Target
            .rename(columns={"r2": "tg_r2"})
        )

        grad_attrib_with_r2 = grad_attrib_with_r2.dropna(subset=["tg_r2"])
        
        min_r2_grid = [-1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results_r2, baseline_macro, baseline_micro = evaluate_min_tg_r2_filters(
            base_edges_df=grad_attrib_with_r2,
            ground_truth_df_dict=ground_truth_df_dict,
            tf_names=tf_names,
            tg_names=tg_names,
            min_r2_grid=min_r2_grid,
            min_edges=10,
            min_pos=1,
        )
        
        results_r2.to_csv(EXPERIMENT_DIR / "auroc_by_tg_r2_threshold.csv")
        
        plot_auroc_per_min_tg_r2(results_r2, min_r2_grid, baseline_macro, baseline_micro, EXPERIMENT_DIR)
    else:
        logging.info(f"Skipping calculation of per-TG R² threshold file (exists): {auroc_by_tg_r2_threshold_file}")
    
    r2_threshold = args.r2_threshold if hasattr(args, 'r2_threshold') else None
    if r2_threshold is not None:
        logging.info(f"Filtering Gradient Attribution edges by per-TG R² > {r2_threshold}")
        per_tg_r2_valid = pd.read_csv(auroc_by_tg_r2_threshold_file)
        
        # gradient_attrib_df has ['Source','Target','Score', ...]
        grad_attrib_filtered = filter_grad_attrib_by_tg_r2(
            grad_attrib_df,
            per_tg_r2_valid,
            r2_threshold=r2_threshold,
        )
    else:
        logging.info("Not filtering Gradient Attribution edges by per-TG R²")
        grad_attrib_filtered = grad_attrib_df

    # ---------- LOAD FEATURES ONCE ----------
    logging.info("Loading feature files")
    base_feature_dict = {
        "TF Knockout":               load_tf_knockout_scores(EXPERIMENT_DIR, tf_names, tg_names),
        "Gradient Attribution":       grad_attrib_filtered,
    }
    
    
    # Save pre-filtered full feature scores
    prefiltered_edges_dir = EXPERIMENT_DIR / "score_grns"
    os.makedirs(prefiltered_edges_dir, exist_ok=True)
    
    for feature_name, df in base_feature_dict.items():
        csv_path = prefiltered_edges_dir / f"{feature_name.replace(' ', '_')}.csv"
        df_export = df[["Source", "Target", "Score"]].copy()
        df_export.to_csv(csv_path, index=False)
        logging.info(f"Saved pre-filtered {feature_name} edges to {csv_path}")

    # ---------- LOAD METHOD GRNs ONCE ----------
    DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/testing_bear_grn/INFERRED.GRNS")

    sample_method_dict = {}
    
    for sample_name in sample_name_list:
        
        if dataset_type.lower() == "mesc":
            cell_oracle_path  = DIR / f"{sample_name}/CellOracle/filtered_L2_{sample_name}_out_E7.5_rep1_final_GRN.csv"
            directnet_path    = DIR / f"{sample_name}/DIRECTNET/{sample_name}_all_cells_Network_links.csv"
            figr_path         = DIR / f"{sample_name}/FigR/{sample_name}_all_cells_filtered_network.csv"
            granie_path       = DIR / f"{sample_name}/GRaNIE/GRN_connections_filtered_sorted_sc{sample_name}_all_cells_selected_uniq.csv"
            linger_path       = DIR / f"{sample_name}/LINGER/filtered_L2_{sample_name}.csv"
            pando_path        = DIR / f"{sample_name}/Pando/{sample_name}_all_cells_raw_network.csv"
            scenic_plus_path  = DIR / f"{sample_name}/SCENIC+/scenic_plus_inferred_grn_mESC_filtered_L2_{sample_name}.tsv"
            tripod_path       = DIR / f"{sample_name}/TRIPOD/gene_TF_highest_abs_coef.csv"
        
        elif dataset_type.lower() == "macrophage":
            if sample_name == "buffer_2":
                cell_oracle_path  = DIR / f"Macrophage_S2/CellOracle/Macrophase_buffer2_filtered_out_E7.5_rep1_final_GRN.csv"
                directnet_path    = DIR / f"Macrophage_S2/DIRECTNET/Network_links.csv"
                figr_path         = DIR / f"Macrophage_S2/FigR/Buffer2_filtered_network.csv"
                granie_path       = DIR / f"Macrophage_S2/GRaNIE/GRN_connections_filtered_sorted_scBuffer2_uniq.csv"
                linger_path       = DIR / f"Macrophage_S2/LINGER/cell_type_TF_gene_buffer2.csv"
                pando_path        = DIR / f"Macrophage_S2/Pando/Macrophage_buffer2_filtered_network.csv"
                scenic_plus_path  = DIR / f"Macrophage_S2/SCENIC+/scenic_plus_inferred_grn_macrophage_macrophage_buffer2_filtered.tsv"
                tripod_path       = DIR / f"Macrophage_S2/TRIPOD/gene_TF_highest_abs_coef.csv"
            else: # Use buffer 1 as default
                cell_oracle_path  = DIR / f"Macrophage_S1/CellOracle/Macrophase_buffer1_filtered_out_E7.5_rep1_final_GRN.csv"
                directnet_path    = DIR / f"Macrophage_S1/DIRECTNET/Network_links.csv"
                figr_path         = DIR / f"Macrophage_S1/FigR/Buffer1_filtered_network.csv"
                granie_path       = DIR / f"Macrophage_S1/GRaNIE/GRN_connections_filtered_sorted_scBuffer1_uniq.csv"
                linger_path       = DIR / f"Macrophage_S1/LINGER/cell_type_TF_gene.csv"
                pando_path        = DIR / f"Macrophage_S1/Pando/Macrophage_buffer1_raw_network.csv"
                scenic_plus_path  = DIR / f"Macrophage_S1/SCENIC+/scenic_plus_inferred_grn_macrophage_macrophage_buffer1_filtered.tsv"
                tripod_path       = DIR / f"Macrophage_S1/TRIPOD/gene_TF_highest_abs_coef.csv"
        
        elif dataset_type.lower() == "k562":
            cell_oracle_path  = DIR / f"{sample_name}/CellOracle/K562_human_filtered_out_E7.5_rep1_final_GRN.csv"
            directnet_path    = DIR / f"{sample_name}/DIRECTNET/Network_links.csv"
            figr_path         = DIR / f"{sample_name}/FigR/K562_filtered_network.csv"
            granie_path       = DIR / f"{sample_name}/GRaNIE/GRN_connections_filtered_sorted_scK562_uniq.csv"
            linger_path       = DIR / f"{sample_name}/LINGER/K562_LINGER_GRN_long.tsv"
            pando_path        = DIR / f"{sample_name}/Pando/K562_raw_network.csv"
            scenic_plus_path  = DIR / f"{sample_name}/SCENIC+/scenic_plus_inferred_grn_K562_K562_human_filtered.tsv"
            tripod_path       = DIR / f"{sample_name}/TRIPOD/gene_TF_highest_abs_coef.csv"
            
        method_info = {
            "CellOracle": {"path": cell_oracle_path, "tf_col": "source",    "target_col": "target",    "score_col": "coef_mean"},
            "SCENIC+":    {"path": scenic_plus_path, "tf_col": "Source",    "target_col": "Target",    "score_col": "Score"},
            "Pando":      {"path": pando_path,       "tf_col": "tf",        "target_col": "target",    "score_col": "estimate"},
            "LINGER":     {"path": linger_path,      "tf_col": "Source",    "target_col": "Target",    "score_col": "Score"},
            "FigR":       {"path": figr_path,        "tf_col": "Motif",     "target_col": "DORC",      "score_col": "Score"},
            "TRIPOD":     {"path": tripod_path,      "tf_col": "TF",        "target_col": "gene",      "score_col": "abs_coef"},
            "GRaNIE":     {"path": granie_path,      "tf_col": "TF.name",   "target_col": "gene.name", "score_col": "TF_gene.r"},
            # "DirectNet":   {"path": directnet_path,   "tf_col": "TF",        "target_col": "Gene",    "score_col": "types"},

        }
        
        standardized_method_dict = {}

        for method_name, info in method_info.items():
            df_std = load_and_standardize_method(method_name, info)
            standardized_method_dict[method_name] = df_std
        
        sample_method_dict[sample_name] = standardized_method_dict  

    # ----- RUN ANALYSIS ON ALL METHODS FOR ALL GROUND TRUTHS -----
    feature_names = list(base_feature_dict.keys())  # e.g. ["TF Knockout", "Gradient Attribution", ...]

    all_method_results = []
        
    for i, (gt_name, ground_truth_df) in enumerate(ground_truth_df_dict.items(), start=1):
        logging.info(f"\n\nEvaluating Features and Methods Against {gt_name} ({i}/{len(ground_truth_df_dict.keys())})")

        gt_analysis_dir = EXPERIMENT_DIR / f"{gt_name}_analysis"
        os.makedirs(gt_analysis_dir, exist_ok=True)

        # chip_valid = ground_truth_df[
        #     ground_truth_df["Source"].isin(tf_names)
        #     & ground_truth_df["Target"].isin(tg_names)
        # ]
        gt_edges = set(zip(ground_truth_df["Source"], ground_truth_df["Target"]))
        gt_tfs   = set(ground_truth_df["Source"])
        gt_tgs   = set(ground_truth_df["Target"])

        # =========================================================
        # 1) FEATURES: filter + label ONCE per GT
        # =========================================================
        logging.info("  - Filtering feature dataframes to ground truth and creating label column (once per GT)")
        feature_dict = {}
        for name, df in base_feature_dict.items():
            filtered = filter_df_to_gene_set(df.copy(), gt_tfs, gt_tgs)
            feature_dict[name] = label_edges(filtered, gt_edges)

        # (a) Plot ALL features vs this GT (your existing function)
        feat_fig = all_feature_auroc_auprc(feature_dict)
        feat_fig.savefig(gt_analysis_dir / f"all_feature_{gt_name}_auroc_auprc.svg")
        plt.close(feat_fig) 

        # (b) Record feature metrics ONCE per GT
        for name, df in feature_dict.items():
            res = compute_curves(df, score_col="Score", name=name, balance=True)
            if res is None:
                continue
            res["gt_name"] = gt_name
            res["sample"] = "FEATURES_ONLY"
            all_method_results.append(res)

        logging.info("  - Pooling methods across samples using mean edge score")

        pooled_method_dict = {}

        # assume all samples have the same method names
        method_names = list(next(iter(sample_method_dict.values())).keys())

        for method_name in method_names:
            dfs = []
            for sample_name, method_dict_base in sample_method_dict.items():
                df_sample = method_dict_base[method_name]

                # filter to GT gene set for THIS ground truth
                df_filtered = filter_df_to_gene_set(df_sample.copy(), gt_tfs, gt_tgs)
                dfs.append(df_filtered)

            if not dfs:
                continue

            # concatenate all samples for this method
            df_concat = pd.concat(dfs, ignore_index=True)

            # mean edge score across samples for each (Source, Target)
            df_mean = (
                df_concat
                .groupby(["Source", "Target"], as_index=False)["Score"]
                .mean()
            )

            # add GT label column
            pooled_method_dict[method_name] = label_edges(df_mean, gt_edges)
            
        # --------------------------------------------------------
        # Per-TF metrics for pooled methods (per GT, per method)
        # --------------------------------------------------------
        combined_for_per_tf = {**pooled_method_dict, **feature_dict}
        
        for method_name, df_labeled in combined_for_per_tf.items():
            per_tf_df = compute_per_tf_metrics(
                df_labeled,
                score_col="Score",
                label_col="is_gt",
                tf_col="Source",
                min_edges=50,
                min_pos=10,
                balance_for_auprc=True,
            )
            if per_tf_df.empty:
                continue

            per_tf_df["method"] = method_name
            per_tf_df["gt_name"] = gt_name
            per_tf_df["experiment"] = experiment

            per_tf_all_results.append(per_tf_df)


        # Combine pooled methods + features for plotting
        combined_dict = {**pooled_method_dict, **feature_dict}

        logging.info("    - Plotting pooled-method AUROC and AUPRC with model features included")

        for feature_name in feature_dict.keys():
            fig, res_list = plot_all_method_auroc_auprc(
                combined_dict,
                gt_name,
                target_method=feature_name,
            )

            # record metrics: one row per (method, GT), sample="POOLED"
            for r in res_list:
                if r["name"] in feature_names:
                    # skip feature rows; those are already added above
                    continue
                r["gt_name"] = gt_name
                r["sample"] = "POOLED"
                all_method_results.append(r)

            feature_name_safe = feature_name.replace(" ", "_").lower()
            fig.savefig(
                gt_analysis_dir
                / f"all_method_pooled_{feature_name_safe}_{gt_name}_auroc_auprc.png",
                dpi=300,
            )
            fig.savefig(
                gt_analysis_dir
                / f"all_method_pooled_{feature_name_safe}_{gt_name}_auroc_auprc.svg",
            )
            plt.close(fig)

        logging.info(f"  Done with {gt_name}")

    if all_method_results:
        df_results = pd.DataFrame(all_method_results)
        df_results.to_csv(
            EXPERIMENT_DIR / "auroc_auprc_results_detailed.csv",
            index=False,
        )
        logging.info(f"Saved all method results ({len(df_results)} rows) to detailed CSV")

        if per_tf_all_results:
            # Concatenate all per-TF results ONCE
            per_tf_metrics = pd.concat(per_tf_all_results, ignore_index=True)
            
            # Save detailed per-TF metrics (one row per TF per method per GT)
            per_tf_metrics.to_csv(
                EXPERIMENT_DIR / "per_tf_auroc_auprc_detailed.csv",
                index=False,
            )
            logging.info(
                f"Saved detailed per-TF metrics ({len(per_tf_metrics)} rows)"
            )
            
        else:
            logging.info("No per-TF metrics computed (likely too few edges per TF).")
        
    else:
        logging.info("No method results collected — check filtering/labeling.")