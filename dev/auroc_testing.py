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
import seaborn as sns

import argparse

import sys

from grn_inference.create_homer_peak_file import parse_args
PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
SRC_DIR = str(Path(PROJECT_DIR) / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from datetime import datetime
from config.settings_hpc import *

logging.basicConfig(level=logging.INFO, format='%(message)s')

from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.datasets.dataset import MultiChromosomeDataset, SimpleScaler, fit_simple_scalers

GROUND_TRUTH_DIR = Path(PROJECT_DIR, "data/ground_truth_files")

def load_ground_truth(ground_truth_file):
    if ground_truth_file.suffix == ".csv":
        sep = ","
    elif ground_truth_file.suffix == ".tsv":
        sep="\t"
        
    ground_truth_df = pd.read_csv(ground_truth_file, sep=sep, on_bad_lines="skip", engine="python")
    
    if ground_truth_file.name == "chip_atlas_tf_peak_tg_dist.csv":
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
            logging.info("Missing keys:", missing)
        if len(unexpected) > 0:
            logging.info("Unexpected keys:", unexpected)
    elif isinstance(state, dict) and "model_state_dict" not in state:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0:
            logging.info("Missing keys:", missing)
        if len(unexpected) > 0:
            logging.info("Unexpected keys:", unexpected)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0:
            logging.info("Missing keys:", missing)
        if len(unexpected) > 0:
            logging.info("Unexpected keys:", unexpected)

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

    return model, test_loader, tg_scaler, tf_scaler

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

def standardize_method_df(df, tf_col, target_col, score_col):
    df = df[[tf_col, target_col, score_col]].copy()
    df.columns = ["Source", "Target", "Score"]
    df["Score"] = df["Score"].abs()
    df["Source"] = df["Source"].astype(str).str.upper()
    df["Target"] = df["Target"].astype(str).str.upper()
    # aggregate duplicate Source–Target pairs
    df = (
        df.groupby(["Source", "Target"], as_index=False)["Score"]
          .max()
    )
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
    auc = np.trapezoid(tpr, fpr)  # or np.trapz(tpr, fpr) if needed
    return fpr, tpr, auc

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

def plot_all_method_auroc_auprc(method_dict, gt_name, target_method="MultiomicTransformer"):
    results = []
    rand_results = []

    for name, df_m in method_dict.items():
        
        df = df_m.copy()
        
        if name != target_method:
            df["Score"] = df["Score"].abs()
        
        # Compute and plot metrics
        res = compute_curves(df, score_col="Score", name=name, balance=True)

        if res is None:
            logging.info(f"Skipping {name} on {gt_name} (no usable positives/negatives)")
            continue

        results.append(res)
        
        rand_res = compute_curves(
            df.assign(**{
                "Score": create_random_distribution(df["Score"])
            }),
            score_col="Score",
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

def plot_all_results_auroc_boxplot(df):
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
    ax.set_title("AUROC distribution per method")
    ax.set_ylim((0.2, 0.8))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    
    return fig

def plot_all_results_auprc_boxplot(df):
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
    ax.set_title("AUPRC distribution per method")
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
        for (idx, row), label in zip(df_m.iterrows()):
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
        out_path = os.path.join(out_dir, f"{method_safe}_curve_variability.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

def plot_method_gt_heatmap(df_pooled: pd.DataFrame, metric: str = "auroc") -> plt.Figure:
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

    # 3) Plot heatmap
    fig, ax = plt.subplots(
        figsize=(
            1.2 * max(len(heat_df.columns), 3),
            0.4 * max(len(heat_df.index), 3),
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

    ax.set_xlabel("Ground truth dataset")
    ax.set_ylabel("Method")
    ax.set_title(
        f"{metric.upper()} per method × ground truth\n"
        f"(methods sorted by mean {metric.upper()} across GTs)"
    )
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

def compute_per_tf_metrics(
    df,
    score_col: str = "Score",
    label_col: str = "is_gt",
    tf_col: str = "Source",
    min_edges: int = 10,
    min_pos: int = 1,
    balance: bool = True,
) -> pd.DataFrame:
    """
    Compute per-TF AUROC and AUPRC.

    For each TF, we treat its outgoing edges (TF -> all TGs) as a local
    binary classification problem and compute AUROC/AUPRC for that TF.

    Returns a DataFrame with one row per TF:
        ['tf', 'auroc', 'auprc', 'n_edges', 'n_pos', 'n_neg']
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

        if balance:
            sub_bal = balance_pos_neg(sub, label_col=label_col, random_state=0)
        else:
            sub_bal = sub

        y = sub_bal[label_col].astype(int).values
        s = sub_bal[score_col].values

        # need both classes present *after* balancing
        if len(np.unique(y)) < 2:
            continue

        try:
            auroc = roc_auc_score(y, s)
            auprc = average_precision_score(y, s)
        except ValueError:
            # safety net if something degenerate sneaks through
            continue

        records.append(
            {
                "tf": tf,
                "auroc": float(auroc),
                "auprc": float(auprc),
                "n_edges": int(n_edges),
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
            }
        )

    return pd.DataFrame.from_records(records)

def get_last_checkpoint(exp_dir: Path) -> str | None:
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
    
    def parse_args():
        parser = argparse.ArgumentParser()
        
        parser.add_argument(
            "--experiment_dir_list",
            required=True,
            nargs="+",
            help="Path to experiment directory"
        )
        return parser.parse_args()
        
    args = parse_args()
    
    ground_truth_file_dict = {
        "ChIP-Atlas": GROUND_TRUTH_DIR / "chip_atlas_tf_peak_tg_dist.csv",
        # "RN111_RN112": GROUND_TRUTH_DIR / "filtered_RN111_and_RN112_mESC_E7.5_rep1.tsv",
        # "ORTI": GROUND_TRUTH_DIR / "ORTI_ground_truth_TF_TG.csv",
        "RN111": GROUND_TRUTH_DIR / "RN111.tsv",
        "RN112": GROUND_TRUTH_DIR / "RN112.tsv",
        "RN114": GROUND_TRUTH_DIR / "RN114.tsv",
        "RN115": GROUND_TRUTH_DIR / "RN115.tsv",
        "RN116": GROUND_TRUTH_DIR / "RN116.tsv",
    }

    feature_to_plot = "Gradient Attribution"

    # experiments = ["no_classifier_head_192", "no_classifier_head_256", "no_classifier_head_256_180_epoch"]
    # experiments = ["model_training_131_20k_metacells"]
    experiments = args.experiment_dir_list
    per_tf_all_results = []
    for experiment_dir in experiments:
        selected_experiment_dir = Path(PROJECT_DIR) / "experiments" / "mESC_no_scale_linear" / experiment_dir
        checkpoint_file = "trained_model.pt"
        
        if not Path(selected_experiment_dir / checkpoint_file).is_file():
            logging.warning(f"Trained model file {checkpoint_file} in {selected_experiment_dir} does not exist. Trying last checkpoint.")
            last_checkpoint = get_last_checkpoint(selected_experiment_dir)
            if last_checkpoint is None:
                raise ValueError(f"Could not find last checkpoint in {selected_experiment_dir}")
            else:
                logging.info(f"Selected last checkpoint: {last_checkpoint}")
                checkpoint_file = last_checkpoint

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, test_loader, tg_scaler, tf_scaler = load_model(selected_experiment_dir, checkpoint_file, device)
        model.eval()

        # Vocab
        tf_names, tg_names = load_vocab(selected_experiment_dir)

        # ---------- LOAD FEATURES ONCE ----------
        logging.info("Loading feature files")
        base_feature_dict = {
            "TF Knockout":               load_tf_knockout_scores(selected_experiment_dir, tf_names, tg_names),
            "Gradient Attribution":       load_gradient_attribution_matrix(selected_experiment_dir, tf_names, tg_names),
        }

        # ---------- LOAD METHOD GRNs ONCE ----------
        DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/testing_bear_grn/INFERRED.GRNS")

        samples = ["E7.5_rep1", "E7.5_rep2", "E8.5_rep1", "E8.5_rep2"]

        sample_method_dict = {}
        
        for sample_name in samples:
            
            cell_oracle_path  = DIR / f"{sample_name}/CellOracle/filtered_L2_{sample_name}_out_E7.5_rep1_final_GRN.csv"
            directnet_path    = DIR / f"{sample_name}/DIRECTNET/{sample_name}_all_cells_Network_links.csv"
            figr_path         = DIR / f"{sample_name}/FigR/{sample_name}_all_cells_filtered_network.csv"
            granie_path       = DIR / f"{sample_name}/GRaNIE/GRN_connections_filtered_sorted_sc{sample_name}_all_cells_selected_uniq.csv"
            linger_path       = DIR / f"{sample_name}/LINGER/filtered_L2_{sample_name}.csv"
            pando_path        = DIR / f"{sample_name}/Pando/{sample_name}_all_cells_raw_network.csv"
            scenic_plus_path  = DIR / f"{sample_name}/SCENIC+/scenic_plus_inferred_grn_mESC_filtered_L2_{sample_name}.tsv"
            tripod_path       = DIR / f"{sample_name}/TRIPOD/gene_TF_highest_abs_coef.csv"

            method_info = {
                "CellOracle": {"path": cell_oracle_path, "tf_col": "source",    "target_col": "target",    "score_col": "coef_mean"},
                "SCENIC+":    {"path": scenic_plus_path, "tf_col": "Source",    "target_col": "Target",    "score_col": "Score"},
                "Pando":      {"path": pando_path,       "tf_col": "tf",        "target_col": "target",    "score_col": "estimate"},
                "LINGER":     {"path": linger_path,      "tf_col": "Source",    "target_col": "Target",    "score_col": "Score"},
                "FigR":       {"path": figr_path,        "tf_col": "Motif",     "target_col": "DORC",      "score_col": "Score"},
                "TRIPOD":     {"path": tripod_path,      "tf_col": "TF",        "target_col": "gene",      "score_col": "abs_coef"},
                "GRaNIE":     {"path": granie_path,      "tf_col": "TF.name",   "target_col": "gene.name", "score_col": "TF_gene.r"},
            }
            
            standardized_method_dict = {}

            for method_name, info in method_info.items():
                df_std = load_and_standardize_method(method_name, info)
                standardized_method_dict[method_name] = df_std
            
            sample_method_dict[sample_name] = standardized_method_dict  

        # ----- RUN ANALYSIS ON ALL METHODS FOR ALL GROUND TRUTHS -----
        feature_names = list(base_feature_dict.keys())  # e.g. ["TF Knockout", "Gradient Attribution", ...]

        all_method_results = []

        for i, (gt_name, ground_truth_file) in enumerate(ground_truth_file_dict.items(), start=1):
            logging.info(f"\n\nEvaluating Features and Methods Against {gt_name} ({i}/{len(ground_truth_file_dict)})")

            gt_analysis_dir = selected_experiment_dir / f"{gt_name}_analysis"
            os.makedirs(gt_analysis_dir, exist_ok=True)

            # --- Ground truth & sets ---
            ground_truth_df = load_ground_truth(ground_truth_file)

            chip_valid = ground_truth_df[
                ground_truth_df["Source"].isin(tf_names)
                & ground_truth_df["Target"].isin(tg_names)
            ]
            gt_edges = set(zip(chip_valid["Source"], chip_valid["Target"]))
            gt_tfs   = set(chip_valid["Source"])
            gt_tgs   = set(chip_valid["Target"])

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
            feat_fig.savefig(gt_analysis_dir / f"all_feature_{gt_name}_auroc_auprc.png", dpi=300)

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
            for method_name, df_labeled in pooled_method_dict.items():
                per_tf_df = compute_per_tf_metrics(
                    df_labeled,
                    score_col="Score",
                    label_col="is_gt",
                    tf_col="Source",
                    min_edges=10,
                    min_pos=1,
                    balance=True,
                )
                if per_tf_df.empty:
                    continue

                per_tf_df["method"] = method_name
                per_tf_df["gt_name"] = gt_name
                per_tf_df["experiment"] = experiment_dir

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


            logging.info(f"  Done with {gt_name}")

        if all_method_results:
            df_results = pd.DataFrame(all_method_results)
            # df_results has:
            #   - one row per (feature, gt_name) from FEATURES_ONLY
            #   - one row per (method, gt_name, sample) from each sample

            # ------------------------------------------------------------
            # 1) Pool across samples: one row per (gt_name, name)
            #    AUROC/AUPRC = mean across samples
            # ------------------------------------------------------------
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
            
            auroc_heat_fig = plot_method_gt_heatmap(df_pooled, metric="auroc")
            auroc_heat_fig.savefig(
                selected_experiment_dir / "method_gt_auroc_heatmap_pooled.png",
                dpi=300,
            )
            plt.close(auroc_heat_fig)

            auprc_heat_fig = plot_method_gt_heatmap(df_pooled, metric="auprc")
            auprc_heat_fig.savefig(
                selected_experiment_dir / "method_gt_auprc_heatmap_pooled.png",
                dpi=300,
            )
            plt.close(auprc_heat_fig)

            # ------------------------------------------------------------
            # 2) Method ranking on pooled AUROCs only
            # ------------------------------------------------------------
            method_rank_auroc = (
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

            logging.info("\n=== Method ranking by mean AUROC across all ground truths (POOLED samples) ===")
            logging.info(method_rank_auroc)

            method_rank_auroc.to_csv(
                selected_experiment_dir / "method_ranking_by_auroc_pooled.csv"
            )

            # ------------------------------------------------------------
            # 3) Per-GT table for boxplots (still one row per (gt_name, method))
            # ------------------------------------------------------------
            per_gt_rank = (
                df_pooled
                .sort_values(["gt_name", "auroc"], ascending=[True, False])
                [["gt_name", "sample", "name", "auroc", "auprc"]]
            )

            # Boxplots now reflect pooled AUROC/AUPRC per method & GT
            all_auroc_boxplots = plot_all_results_auroc_boxplot(per_gt_rank)
            all_auprc_boxplots = plot_all_results_auprc_boxplot(per_gt_rank)

            all_auroc_boxplots.savefig(
                selected_experiment_dir / "all_results_auroc_boxplot_pooled.png",
                dpi=300,
            )
            all_auprc_boxplots.savefig(
                selected_experiment_dir / "all_results_auprc_boxplot_pooled.png",
                dpi=300,
            )

            # Save pooled per-GT metrics instead of per-sample metrics
            per_gt_rank.to_csv(
                selected_experiment_dir / "per_gt_method_aucs_pooled.csv",
                index=False,
            )
                    # existing `if all_method_results:` block ...

            # ------------------------------------------------------------
            # 4) Save per-TF metrics across all methods & ground truths
            # ------------------------------------------------------------
            if per_tf_all_results:
                per_tf_metrics = pd.concat(per_tf_all_results, ignore_index=True)
                per_tf_metrics.to_csv(
                    selected_experiment_dir / "per_tf_auroc_auprc_pooled.csv",
                    index=False,
                )
                logging.info(
                    f"Saved per-TF AUROC/AUPRC metrics to {selected_experiment_dir / 'per_tf_auroc_auprc_pooled.csv'}"
                )
            else:
                logging.info("No per-TF metrics computed (likely too few edges per TF).")

            
        else:
            logging.info("No method results collected — check filtering/labeling.")