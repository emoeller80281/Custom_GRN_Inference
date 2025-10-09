#!/usr/bin/env python
import csv
import json
import logging
import os
from pathlib import Path
import pickle
from unicodedata import numeric

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier
import xgboost as xgb
import joblib
import argparse
import matplotlib.pyplot as plt


from config.settings import *
from multiomic_transformer.datasets.dataset import MultiomicTransformerDataset
from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.scripts import fine_tuning
from multiomic_transformer.utils import plotting

# ---------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def load_model_from_run_params(run_dir, tf_vocab_size, tg_vocab_size, device="cpu"):
    with open(os.path.join(run_dir, "run_parameters.json")) as f:
        run_params = json.load(f)
    model = MultiomicTransformer(
        d_model=run_params.get("d_model", 384),
        num_heads=run_params.get("num_heads", 6),
        num_layers=run_params.get("num_layers", 3),
        d_ff=run_params.get("d_feedforward", 768),
        dropout=run_params.get("dropout", 0.1),
        tf_vocab_size=tf_vocab_size,
        tg_vocab_size=tg_vocab_size,
        bias_scale=run_params.get("attn_bias_scale", 1.0),
        use_shortcut=run_params.get("use_shortcut", True),
        use_motif_mask=run_params.get("use_motif_mask", False),
        lambda_l1=run_params.get("shortcut_l1", 0.0),
        lambda_l2=run_params.get("shortcut_l2", 0.0),
        topk=run_params.get("shortcut_topk", None),
        shortcut_dropout=run_params.get("shortcut_dropout", 0.0),
    ).to(device)
    return model, run_params


# ---------------------------------------------------------------------
# Helpers for normalization / gradients
# ---------------------------------------------------------------------
@torch.no_grad()
def _global_minmax_(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)

def zscore_per_cell(x: torch.Tensor, eps=1e-6):
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True).clamp_min(eps)
    return (x - mu) / sd


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------
@torch.no_grad()
def extract_shortcut_matrix(model, dataset, device=DEVICE, normalize="global"):
    model.eval()
    B, W = 1, 1
    atac_windows = torch.zeros(B, W, 1, device=device)
    tf_expr = torch.zeros(B, len(dataset.tf_names), device=device)
    tf_ids = torch.arange(len(dataset.tf_names), device=device)
    tg_ids = torch.arange(len(dataset.tg_names), device=device)

    logits, attn = model(atac_windows, tf_expr, tf_ids=tf_ids, tg_ids=tg_ids,
                         bias=None, motif_mask=None)

    mat = attn.detach().cpu().T  # [TF, TG]

    if hasattr(model, "shortcut") and hasattr(model.shortcut, "scale"):
        mat = model.shortcut.scale.item() * mat

    if normalize == "global":
        mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-8)

    return pd.DataFrame(mat.numpy(), index=dataset.tf_names, columns=dataset.tg_names)


import torch
import pandas as pd
from tqdm import tqdm
import logging

def extract_edge_features(
    model, dataloader, tf_names, tg_names, tg_id_map,
    chip_edges=None, gradient_attrib_df=None,
    device="cuda", tg_chunk=None
):
    """
    Vectorized edge feature extraction for the MultiomicTransformer.
    Produces a long-format DataFrame with one row per (TF, TG).

    tg_id_map : array mapping raw tg_ids (from dataloader) -> [0 … TG-1]
    """
    model.eval()
    TF, TG = len(tf_names), len(tg_names)

    # --- Accumulators on CPU (to avoid GPU OOM) ---
    attn_acc   = torch.zeros(TG, TF, dtype=torch.float32)   # [TG, TF]
    motif_acc  = torch.zeros(TG, TF, dtype=torch.float32)   # [TG, TF]
    pred_mu    = torch.zeros(TG, dtype=torch.float32)       # [TG]
    pred_sd    = torch.zeros(TG, dtype=torch.float32)       # [TG]
    bias_mu    = torch.zeros(TG, dtype=torch.float32)       # [TG]
    counts     = torch.zeros(TG, dtype=torch.float32)       # [TG]

    for batch in tqdm(dataloader, desc="Extracting edge features"):
        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]

        # --- map raw tg_ids -> global indices ---
        tg_batch_global = torch.as_tensor(
            tg_id_map[tg_ids.cpu().numpy()], device="cpu"
        )

        if tg_batch_global.max() >= TG or tg_batch_global.min() < 0:
            raise ValueError(
                f"tg_batch_global out of bounds! "
                f"max={tg_batch_global.max().item()}, TG={TG}"
            )

        # --- forward pass ---
        with torch.no_grad():
            if tg_chunk is None:
                out = model(atac_wins, tf_tensor,
                            tf_ids=tf_ids, tg_ids=tg_ids,
                            bias=bias, motif_mask=motif_mask)
                preds, attn = out if isinstance(out, tuple) else (out, None)

                preds_cpu = preds.detach().cpu()
                attn_cpu  = attn.detach().cpu() if attn is not None else None
                bias_cpu  = bias.mean(dim=(0,2)).detach().cpu()

                # accumulate
                pred_mu[tg_batch_global] += preds_cpu.mean(dim=0)
                pred_sd[tg_batch_global] += preds_cpu.std(dim=0)
                bias_mu[tg_batch_global] += bias_cpu
                counts[tg_batch_global]  += 1

                if attn_cpu is not None:
                    attn_acc[tg_batch_global] += attn_cpu
                if motif_mask is not None:
                    motif_acc[tg_batch_global] += motif_mask.float().cpu()

            else:
                # --- chunk over TGs to reduce memory ---
                for j0 in range(0, tg_ids.shape[0], tg_chunk):
                    j1 = min(j0 + tg_chunk, tg_ids.shape[0])
                    tg_subset = tg_ids[j0:j1]
                    tg_global_subset = tg_batch_global[j0:j1]

                    out = model(atac_wins, tf_tensor,
                                tf_ids=tf_ids, tg_ids=tg_subset,
                                bias=bias, motif_mask=motif_mask)
                    preds, attn = out if isinstance(out, tuple) else (out, None)

                    preds_cpu = preds.detach().cpu()
                    attn_cpu  = attn.detach().cpu() if attn is not None else None
                    bias_cpu  = bias[:, j0:j1, :].mean(dim=(0,2)).detach().cpu()

                    pred_mu[tg_global_subset] += preds_cpu.mean(dim=0)
                    pred_sd[tg_global_subset] += preds_cpu.std(dim=0)
                    bias_mu[tg_global_subset] += bias_cpu
                    counts[tg_global_subset]  += 1

                    if attn_cpu is not None:
                        attn_acc[tg_global_subset] += attn_cpu
                    if motif_mask is not None:
                        motif_acc[tg_global_subset] += motif_mask[:, j0:j1].float().cpu()

        # free GPU memory
        del preds, attn, bias
        torch.cuda.empty_cache()

    # --- Average by counts ---
    denom_TG   = counts.clamp_min(1)
    denom_TG2D = denom_TG.view(-1, 1)

    pred_mu    = (pred_mu / denom_TG)
    pred_sd    = (pred_sd / denom_TG)
    bias_mu    = (bias_mu / denom_TG)
    attn_mean  = (attn_acc / denom_TG2D)
    motif_mean = (motif_acc / denom_TG2D)

    # --- gradient attribution (optional) ---
    if gradient_attrib_df is not None:
        gradient_attrib_df = gradient_attrib_df.reindex(index=tf_names, columns=tg_names, fill_value=0.0)
        grad_attr = torch.tensor(gradient_attrib_df.values.T, dtype=torch.float32)
    else:
        grad_attr = torch.zeros_like(attn_mean)

    # --- build edge DataFrame ---
    tg_index = pd.Index(tg_names, name="TG")
    tf_index = pd.Index(tf_names, name="TF")

    df_attn  = pd.DataFrame(attn_mean.numpy(),  index=tg_index, columns=tf_index)
    df_motif = pd.DataFrame(motif_mean.numpy(), index=tg_index, columns=tf_index)
    df_grad  = pd.DataFrame(grad_attr.numpy(),  index=tg_index, columns=tf_index)

    long_attn  = df_attn.stack().rename("attn").reset_index()
    long_motif = df_motif.stack().rename("motif_mask").reset_index()
    long_grad  = df_grad.stack().rename("grad_attr").reset_index()

    df_tg = pd.DataFrame({
        "TG": tg_names,
        "pred_mean": pred_mu.numpy(),
        "pred_std":  pred_sd.numpy(),
        "bias_mean": bias_mu.numpy(),
    })

    edges = long_attn.merge(long_motif, on=["TG", "TF"], how="left") \
                     .merge(long_grad,  on=["TG", "TF"], how="left") \
                     .merge(df_tg,      on="TG",         how="left")

    if chip_edges is not None:
        chip_set = set((t.upper(), g.upper()) for t, g in chip_edges)
        edges["label"] = [(tf.upper(), tg.upper()) in chip_set
                          for tf, tg in zip(edges["TF"], edges["TG"])]
        edges["label"] = edges["label"].astype(int)

    cols = ["TF", "TG", "attn", "pred_mean", "pred_std", "bias_mean", "grad_attr", "motif_mask"]
    if "label" in edges:
        cols.append("label")
    edges = edges[cols]

    return edges





def gradient_attribution_matrix(model, dataset, loader, tg_chunk=64, device=DEVICE, normalize="global"):
    TF, TG = len(dataset.tf_names), len(dataset.tg_names)
    acc = torch.zeros(TF, TG, dtype=torch.float32, device="cpu")  # accumulate on CPU

    for batch in tqdm(loader, desc="Calculating gradient attributions"):
        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]

        for j0 in range(0, TG, tg_chunk):
            j1 = min(j0 + tg_chunk, TG)

            # fresh input with grad enabled
            tf_tensor_ = tf_tensor.detach().clone().requires_grad_(True)
            tf_norm    = zscore_per_cell(tf_tensor_)

            # forward only this TG slice
            out = model(
                atac_wins,
                tf_norm,
                tf_ids=tf_ids,
                tg_ids=torch.arange(j0, j1, device=device),
                bias=bias,
                motif_mask=motif_mask
            )
            preds = out[0] if isinstance(out, tuple) else out  # [B, chunk]

            # scalar objective
            out_chunk = preds.mean(dim=0).sum()
            grads = torch.autograd.grad(out_chunk, tf_norm, retain_graph=False, create_graph=False)[0]  # [B, TF]

            # accumulate: collapse batch, broadcast to chunk
            grad_vec = grads.abs().sum(dim=0)  # [TF]
            acc[:, j0:j1] += grad_vec.unsqueeze(1).expand(-1, j1-j0).cpu()

    # average across batches
    acc = acc / max(1, len(loader))

    if normalize == "global":
        acc = _global_minmax_(acc)

    return pd.DataFrame(acc.numpy(), index=dataset.tf_names, columns=dataset.tg_names)

def rank_gauss(s):
    r = s.rank(method="average")
    r = (r - 0.5) / max(len(s), 1)
    from scipy.special import erfinv
    return np.sqrt(2) * erfinv(2*r - 1)




# ---------------------------------------------------------------------
# Training classifier
# ---------------------------------------------------------------------

def train_logistic_baseline(df):
    """
    Train a logistic regression baseline to test if features carry global signal.
    """
    features = ["attn", "pred_mean", "pred_std", "bias_mean", "grad_attr", "motif_mask", "shortcut_weight"]
    X = df[features].values
    y = df["label"].values

    # Train/validation split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build pipeline: scaling + logistic regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, class_weight="balanced"))
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, y_pred)
    auprc = average_precision_score(y_val, y_pred)

    print(f"Baseline Logistic Regression - AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}")
    return pipe

def train_edge_classifier(df):
    features = ["attn","pred_mean","pred_std","bias_mean","grad_attr","motif_mask","shortcut_weight"]
    X = df[features].values
    y = df["label"].values
    groups = df["TF"].values  # or df["TG"] if you want cross-TG generalization

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    

    # stronger regularization for cross-TF generalization
    params = dict(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=1.0,
        reg_alpha=1e-2,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
    )

    aucs, auprcs = [], []
    best_clf = None; best_auc = -np.inf

    for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups), 1):
        clf = XGBClassifier(**params)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[va])[:,1]
        auc = roc_auc_score(y[va], p)
        aupr = average_precision_score(y[va], p)
        aucs.append(auc); auprcs.append(aupr)
        if auc > best_auc:
            best_auc, best_clf = auc, clf
        print(f"[fold {fold}] AUROC={auc:.3f}  AUPRC={aupr:.3f}")

    print(f"Mean AUROC={np.mean(aucs):.3f}±{np.std(aucs):.3f} | Mean AUPRC={np.mean(auprcs):.3f}±{np.std(auprcs):.3f}")
    return best_clf

def aggregate_edge_features_across_chromosomes(chrom_list, experiment_id):
    """
    Collects edge feature CSVs from each chromosome and merges them into one DataFrame.
    """
    all_edges = []
    for chrom_id in chrom_list:
        edge_path = OUTPUT_DIR / chrom_id / experiment_id / "test_results" / "edge_features.csv"
        if edge_path.exists():
            df = pd.read_csv(edge_path)
            df["chrom_id"] = chrom_id
            all_edges.append(df)
            logging.info(f"Loaded {edge_path}  shape={df.shape}")
        else:
            logging.warning(f"Missing {edge_path}, skipping")

    if not all_edges:
        raise FileNotFoundError("No per-chromosome edge feature files found.")
    
    combined = pd.concat(all_edges, ignore_index=True)
    logging.info(f"Combined edge feature matrix: shape={combined.shape}")
    return combined


def compute_or_load_validation(model, loader, device, out_dir, fname="val_metrics.pkl"):
    """
    Compute validation loss and correlations, or load them from disk if available.
    Saves: val_loss, pearson_corr, spearman_corr, all_preds, all_tgts
    """
    path = os.path.join(out_dir, fname)

    if os.path.exists(path):
        logging.info(f"Loading cached validation metrics from {path}")
        with open(path, "rb") as f:
            results = pickle.load(f)
        return results

    # --- Compute if not cached ---
    logging.info("\nComputing validation loss and correlations")
    preds_list, tgts_list = [], []
    total_loss, n_batches = 0.0, 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="val_loss_and_corr"):
            atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = [
                x.to(device) if torch.is_tensor(x) else x for x in batch
            ]
            preds, _ = model(atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids,
                             bias=bias, motif_mask=motif_mask)
            total_loss += torch.nn.functional.mse_loss(preds, targets).item()
            n_batches += 1
            preds_list.append(preds.cpu().numpy())
            tgts_list.append(targets.cpu().numpy())

    val_loss = total_loss / max(1, n_batches)
    all_preds = np.concatenate(preds_list)
    all_tgts  = np.concatenate(tgts_list)

    pearson_corr, _  = pearsonr(all_preds.ravel(), all_tgts.ravel())
    spearman_corr, _ = spearmanr(all_preds.ravel(), all_tgts.ravel())

    results = {
        "val_loss": val_loss,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "all_preds": all_preds,
        "all_tgts": all_tgts,
    }

    with open(path, "wb") as f:
        pickle.dump(results, f)
    logging.info(f"Saved validation metrics to {path}")

    return results

def predict_with_unified_classifier(model, input_edge_features, output_path):
    """
    Load a trained classifier and apply it to new edge feature data.
    """
    import joblib
    import pandas as pd

    # Ensure required feature columns exist
    required_features = ["attn", "pred_mean", "pred_std", "bias_mean", "grad_attr", "motif_mask", "shortcut_weight"]
    missing = [f for f in required_features if f not in input_edge_features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = input_edge_features[required_features].fillna(0).clip(-5, 5).values

    logging.info(f"Predicting probabilities for {len(X):,} edges")
    preds = model.predict_proba(X)[:, 1]

    result_df = input_edge_features[["TF", "TG"]].copy()
    result_df["score"] = preds

    # Sort by score descending (strongest predicted interactions first)
    result_df = result_df.sort_values("pred_score", ascending=False)

    result_df = result_df.rename(columns={"TF":"Source", "TG":"Target"})
    result_df.to_csv(output_path, index=False)
    logging.info(f"Saved predictions: {output_path} (shape={result_df.shape})")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or predict with unified classifier across multiple chromosomes")
    parser.add_argument("--chrom_list", nargs="+", help="List of chromosomes (e.g., chr1 chr2 chr3)")
    parser.add_argument("--experiment_id", type=str, required=True, help="Experiment name (e.g., model_training_001)")
    parser.add_argument("--chip_file", type=str, help="Path to ground truth CSV (for training)")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--output_path", type=str, help="Where to save prediction CSV")
    args = parser.parse_args()

    # --- Run unified training ---
    chrom_list = args.chrom_list
    experiment_id = args.experiment_id
    chip_file = Path(args.chip_file)
    gpu_id = args.gpu_id

    combined_df = aggregate_edge_features_across_chromosomes(chrom_list, experiment_id)
    combined_df = pd.get_dummies(combined_df, columns=["chrom_id"], drop_first=True)
    
    numeric_cols = ["attn", "pred_mean", "pred_std", "bias_mean", "grad_attr", "motif_mask", "shortcut_weight"]
    combined_df[numeric_cols] = combined_df[numeric_cols].fillna(0).clip(-5, 5)
    combined_df = combined_df.dropna(subset=["label"])

    logging.info("\nTraining unified XGBoost classifier across all chromosomes")
    clf = train_edge_classifier(combined_df)
    log_clf = train_logistic_baseline(combined_df)

    out_dir = OUTPUT_DIR / "combined" / experiment_id / "classifier"
    os.makedirs(out_dir, exist_ok=True)

    combined_df.to_csv(out_dir / "edge_features.csv")
    joblib.dump(clf, out_dir / "edge_classifier_unified.pkl")
    joblib.dump(log_clf, out_dir / "logistic_baseline_unified.pkl")

    logging.info(f"Saved unified classifiers in {out_dir}")
    
    predict_with_unified_classifier(clf, combined_df, args.output_path)
    