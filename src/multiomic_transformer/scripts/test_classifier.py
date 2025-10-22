#!/usr/bin/env python3
import os, json, logging, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)

from torch_geometric.data import Data
from multiomic_transformer.models.tf_tg_classifier import GRN_GAT_Encoder

logging.basicConfig(level=logging.INFO, format="%(message)s")

# -----------------------------
# Model head (must match train)
# -----------------------------
class EdgeClassifier(nn.Module):
    def __init__(self, base_model, hidden_dim=128):
        super().__init__()
        self.encoder = base_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x, edge_index, edge_attr, pairs):
        h, _ = self.encoder(x, edge_index, edge_attr)
        tf_emb = h[pairs[:, 0]]; tg_emb = h[pairs[:, 1]]
        return self.classifier(torch.cat([tf_emb, tg_emb], dim=1)).squeeze(-1)

# -----------------------------
# Helpers
# -----------------------------
BASE_FEATS = [
    "reg_potential", "motif_density", "mean_tf_expr",
    "mean_tg_expr", "expr_product", "motif_present"
]

def choose_edge_features(df: pd.DataFrame):
    feats = BASE_FEATS.copy()
    if "log_reg_pot" in df.columns:
        feats.append("log_reg_pot")
    # extras from STRING/TRRUST/KEGG (case-insensitive)
    for c in df.columns:
        lc = c.lower()
        if (lc.startswith("string_") or lc.startswith("trrust_") or lc.startswith("kegg_")) and pd.api.types.is_numeric_dtype(df[c]):
            feats.append(c)
    # dedupe, preserve order
    seen, out = set(), []
    for f in feats:
        if f in df.columns and f not in seen:
            seen.add(f); out.append(f)
    if not out:
        raise ValueError("No edge features found after selection.")
    return out

def load_edges_table(path: str):
    p = Path(path)
    if p.suffix.lower() in {".pq", ".parquet"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    # Normalize names
    df["TF"] = df["TF"].astype(str).str.upper().str.strip()
    df["TG"] = df["TG"].astype(str).str.upper().str.strip()
    return df

def merge_ground_truth(edge_df: pd.DataFrame, gt_path: str, sep: str):
    gt = pd.read_csv(gt_path, sep=sep)
    tf_col, tg_col = gt.columns[:2]
    gt = gt.rename(columns={tf_col: "TF", tg_col: "TG"})
    gt["TF"] = gt["TF"].astype(str).str.upper().str.strip()
    gt["TG"] = gt["TG"].astype(str).str.upper().str.strip()
    if "label" not in gt.columns:
        logging.warning("No 'label' in ground truth — assuming all provided TF–TG are positives.")
        gt["label"] = 1.0
    gt["label"] = pd.to_numeric(gt["label"], errors="coerce").fillna(0).clip(0,1).astype(float)

    merged = edge_df.drop(columns=["label"], errors="ignore").merge(
        gt[["TF","TG","label"]], on=["TF","TG"], how="left"
    )
    if merged["label"].isna().any():
        n = int(merged["label"].isna().sum())
        logging.warning(f"{n:,} edges missing in ground truth → assigning label=0.")
        merged["label"] = merged["label"].fillna(0.0)
    merged["label"] = merged["label"].astype(float)
    return merged

def build_graph_and_scalers(df: pd.DataFrame, edge_features, scaler_path=None, allow_fit=True):
    # Coerce numerics & impute
    if "motif_present" in df.columns:
        df["motif_present"] = pd.to_numeric(df["motif_present"], errors="coerce").fillna(0.0).astype(float)
    for c in edge_features:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Scale edge features
    scaler = None
    if scaler_path and Path(scaler_path).exists():
        import joblib
        scaler = joblib.load(scaler_path)
        logging.info(f"Loaded scaler from {scaler_path}")
    elif allow_fit:
        scaler = StandardScaler().fit(df[edge_features].to_numpy())
        logging.warning("No scaler provided — fitting StandardScaler on evaluation edges (may shift scores slightly).")
    else:
        raise FileNotFoundError("Scaler not found and fitting is disabled.")

    edge_attr = torch.tensor(scaler.transform(df[edge_features].to_numpy()), dtype=torch.float32)

    # Encode nodes (exactly like training)
    tf_enc = LabelEncoder(); tg_enc = LabelEncoder()
    tf_ids = tf_enc.fit_transform(df["TF"])
    tg_ids = tg_enc.fit_transform(df["TG"]) + len(tf_enc.classes_)
    n_tfs, n_tgs = len(tf_enc.classes_), len(tg_enc.classes_)
    n_nodes = n_tfs + n_tgs

    # Edge index (directed TF→TG; encoder will mirror internally)
    edge_index_np = np.array([tf_ids, tg_ids], dtype=np.int64)
    edge_index = torch.from_numpy(edge_index_np)

    # Node features (expression priors)
    if not {"mean_tf_expr","mean_tg_expr"}.issubset(df.columns):
        raise ValueError("mean_tf_expr / mean_tg_expr must be present for node features.")
    tf_expr = df.groupby("TF")["mean_tf_expr"].mean()
    tg_expr = df.groupby("TG")["mean_tg_expr"].mean()
    node_features = torch.tensor(
        np.concatenate([
            tf_expr.reindex(tf_enc.classes_).fillna(0).to_numpy().reshape(-1,1),
            tg_expr.reindex(tg_enc.classes_).fillna(0).to_numpy().reshape(-1,1)
        ]),
        dtype=torch.float32
    )

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    pairs = torch.tensor(np.stack([tf_ids, tg_ids], axis=1), dtype=torch.long)

    return data, pairs, (tf_enc, tg_enc), scaler, n_tfs, n_tgs, n_nodes

def sigmoid(x): return 1/(1+np.exp(-x))

def precision_at_k_by_tf(df_scores, gold_map, ks=(10,20,50,100,200,500)):
    rows = []
    for k in ks:
        vals = []
        for tf, gset in gold_map.items():
            sub = df_scores[df_scores["TF"]==tf]
            if sub.empty: continue
            s = sub.sort_values("Score", ascending=False).head(k)
            vals.append(s["TG"].isin(gset).mean())
        rows.append((k, float(np.mean(vals)) if len(vals)>0 else np.nan))
    return pd.DataFrame(rows, columns=["k","Precision"])

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True, help="Edges file (CSV or Parquet) with TF,TG and edge features.")
    ap.add_argument("--ground_truth", required=True, help="Ground truth file (first two cols TF,TG; optional label).")
    ap.add_argument("--sep", default=",", help="Separator for ground truth file (default ',').")
    ap.add_argument("--ckpt", required=True, help="Path to fine_tuned_gat_classifier.pt")
    ap.add_argument("--scaler", default=None, help="Optional: path to edge_feature_scaler.pkl saved at training time.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    chip_dir = Path(args.out_dir) / "chip_eval"
    chip_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & prep edges
    raw_df = load_edges_table(args.edges)
    edge_features = choose_edge_features(raw_df)
    logging.info(f"Using {len(edge_features)} edge features: {edge_features}")

    # 2) Merge ground truth
    merged = merge_ground_truth(raw_df, args.ground_truth, args.sep)

    # 3) Build graph exactly like training
    data, pairs, (tf_enc, tg_enc), scaler, n_tfs, n_tgs, n_nodes = build_graph_and_scalers(
        merged, edge_features, scaler_path=args.scaler, allow_fit=True
    )

    device = torch.device(args.device)
    data = data.to(device)
    pairs = pairs.to(device)

    # 4) Recreate model & load checkpoint (must match training hyperparams)
    encoder = GRN_GAT_Encoder(
        in_node_feats=1,
        in_edge_feats=len(edge_features),
        hidden_dim=128,
        heads=4,
        dropout=0.3,
        edge_dropout_p=0.0  # eval → no edge dropout
    ).to(device)

    model = EdgeClassifier(encoder, hidden_dim=128).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        logging.warning(f"Missing keys: {missing} | Unexpected keys: {unexpected}")
    model.eval()
    logging.info(f"Loaded checkpoint from {args.ckpt}")

    # 5) Predict scores for ALL edges present
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr, pairs).detach().cpu().numpy().ravel()
    scores = sigmoid(logits)

    # 6) Evaluation vs ground truth labels (if present)
    out_tbl = merged[["TF","TG","label"]].copy()
    out_tbl["Score"] = scores
    out_tbl.to_csv(chip_dir / "scored_edges.csv", index=False)

    if merged["label"].nunique() > 1:
        y_true = merged["label"].astype(int).values
        y_pred = scores
        # Align length (defensive)
        n = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:n], y_pred[:n]

        auroc = roc_auc_score(y_true, y_pred)
        aupr  = average_precision_score(y_true, y_pred)
        logging.info(f"[Overall] AUROC={auroc:.3f} | AUPR={aupr:.3f}")

        # Curves
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        prec, rec, _ = precision_recall_curve(y_true, y_pred)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--',c='gray')
        plt.title(f"ROC (AUROC={auroc:.3f})"); plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.subplot(1,2,2); plt.plot(rec,prec)
        plt.title(f"PR (AUPR={aupr:.3f})"); plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.tight_layout(); plt.savefig(chip_dir / "overall_roc_pr.png", dpi=180); plt.close()

    else:
        logging.warning("Ground-truth has a single class — skipping overall AUROC/AUPR.")

    # 7) Per-TF metrics + Precision@k (treat GT rows as positives; rest as negatives for that TF)
    gold = (
        merged.loc[merged["label"] == 1, ["TF","TG"]]
        .groupby("TF")["TG"].apply(lambda s: set(s.astype(str)))
        .to_dict()
    )
    per_rows = []
    for tf, pos_set in gold.items():
        sub = out_tbl[out_tbl["TF"] == tf]
        if sub.empty: continue
        y_true = sub["TG"].isin(pos_set).astype(int).values
        y_pred = sub["Score"].values
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            auroc_tf, aupr_tf = np.nan, np.nan
        else:
            auroc_tf = roc_auc_score(y_true, y_pred)
            aupr_tf  = average_precision_score(y_true, y_pred)
        # P@50, P@100
        order = np.argsort(-y_pred)
        p50  = y_true[order[:50]].mean()  if len(y_true) >= 50  else y_true[order].mean()
        p100 = y_true[order[:100]].mean() if len(y_true) >= 100 else y_true[order].mean()
        per_rows.append({"TF": tf, "n": int(len(y_true)), "n_pos": int(y_true.sum()),
                         "AUROC": auroc_tf, "AUPR": aupr_tf, "P@50": p50, "P@100": p100})

    per_tf = pd.DataFrame(per_rows).sort_values(["AUPR","AUROC"], ascending=False)
    per_tf.to_csv(chip_dir / "per_TF_metrics.csv", index=False)
    logging.info(f"Saved per-TF metrics for {len(per_tf)} TFs → {chip_dir/'per_TF_metrics.csv'}")

    # Precision@k curve (averaged over TFs)
    def precision_at_k(df_scores, gold_map):
        ks = [10,20,50,100,200,500]
        rows = []
        for k in ks:
            vals = []
            for tf, pos_set in gold_map.items():
                sub = df_scores[df_scores["TF"]==tf]
                if sub.empty: continue
                s = sub.sort_values("Score", ascending=False).head(k)
                vals.append(s["TG"].isin(pos_set).mean())
            rows.append((k, float(np.mean(vals)) if vals else np.nan))
        return pd.DataFrame(rows, columns=["k","Precision"])

    p_at_k = precision_at_k(out_tbl, gold)
    p_at_k.to_csv(chip_dir / "precision_at_k.csv", index=False)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(p_at_k["k"], p_at_k["Precision"], marker="o")
    plt.xscale("log"); plt.xlabel("k"); plt.ylabel("Precision"); plt.title("Precision@k (avg over TFs)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(chip_dir / "precision_at_k.png", dpi=180); plt.close()

    logging.info("ChIP comparison complete.")

if __name__ == "__main__":
    main()
