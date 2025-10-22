#!/usr/bin/env python3
"""
Two-phase GAT training:
  1. Self-supervised Deep Graph Infomax (DGI)
  2. Semi-supervised fine-tuning on known PKN edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from multiomic_transformer.models.tf_tg_classifier import GRN_GAT_Encoder
from multiomic_transformer.utils.gene_canonicalizer import GeneCanonicalizer
import logging, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from config.settings import *

logging.basicConfig(level=logging.INFO, format="%(message)s")

CHIP_GT_PATH = os.getenv("CHIP_GROUND_TRUTH", "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/ground_truth_files/mESC_beeline_ChIP-seq.csv")  # <-- EDIT
CHIP_GT_SEP  = os.getenv("CHIP_GROUND_TRUTH_SEP", ",")

canon = GeneCanonicalizer()
canon.load_gtf("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/genome_data/genome_annotation/mm10/Mus_musculus.gene_info")
canon.load_ncbi_gene_info("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/genome_data/reference_genome/mm10/Mus_musculus.GRCm39.113.gtf", species_taxid="10090")
logging.info("Map sizes:", canon.coverage_report())

# ============================================================
# Load data
# ============================================================

sample_dfs = []
for sample in SAMPLE_NAMES:
    if os.path.exists(SAMPLE_PROCESSED_DATA_DIR / sample / "tf_tg_data.parquet"):
        logging.info(f"Loading data for {sample}")
        
        sample_df = pd.read_parquet(SAMPLE_PROCESSED_DATA_DIR / sample / "tf_tg_data.parquet")
        logging.info(f"  - Loaded {sample}: {len(sample_df)} edges | {sample_df['TF'].nunique()} TFs | {sample_df['TG'].nunique()} TGs")

        sample_dfs.append(sample_df)
    
df = pd.concat(sample_dfs, ignore_index=True)
logging.info(f"Total: {len(df)} edges | {df['TF'].nunique()} TFs | {df['TG'].nunique()} TGs")

# ---- Edge feature selection ----
base_feats = [
    "reg_potential", "motif_density", "mean_tf_expr", 
    "mean_tg_expr", "expr_product", "motif_present"
    ]

# Include log_reg_pot if present
if "log_reg_pot" in df.columns:
    base_feats.append("log_reg_pot")

# case-insensitive prefix match for extras
extra_numeric = [
    c for c in df.columns
    if (c.lower().startswith("string_") or c.lower().startswith("trrust_") or c.lower().startswith("kegg_"))
    and pd.api.types.is_numeric_dtype(df[c])
]

edge_features = base_feats + extra_numeric
edge_features = list(dict.fromkeys(edge_features))  # dedupe, preserve order
if not edge_features:
    raise ValueError("No edge features found.")

# Ensure we have labels
if "label" not in df.columns:
    if "in_pkn" in df.columns:
        df["label"] = (df["in_pkn"] == 1).astype(int)
        logging.info("Created 'label' from 'in_pkn'.")
    else:
        logging.warning("No 'label' or 'in_pkn' found — Phase 2 will be skipped.")

# Keep only what we need going forward
keep_cols = ["TF", "TG"] + (["label"] if "label" in df.columns else [])
df = df[edge_features + keep_cols]

# ============================================================
# Coerce features to numeric & scale (safe)
# ============================================================

def _flatten_features(feats):
    flat = []
    for x in feats:
        if isinstance(x, (list, tuple, np.ndarray, pd.Index)):
            flat.extend(list(x))
        else:
            flat.append(x)
    out, seen = [], set()
    for f in map(str, flat):
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

edge_features = _flatten_features(edge_features)

missing = [c for c in edge_features if c not in df.columns]
if missing:
    raise ValueError(f"Missing edge feature columns: {missing}")

# Make binary columns float for scaler
if "motif_present" in df.columns:
    df["motif_present"] = pd.to_numeric(df["motif_present"], errors="coerce").fillna(0.0).astype(float)

coerced_info = []
for c in edge_features:
    s = pd.to_numeric(df[c], errors="coerce")
    n_null = int(s.isna().sum())
    df[c] = s.fillna(0.0)
    if n_null:
        coerced_info.append(f"{c}({n_null} NaN→0)")
if coerced_info:
    logging.info("Coerced edge features: " + "; ".join(coerced_info))

scaler = StandardScaler()
edge_attr = torch.tensor(scaler.fit_transform(df[edge_features].to_numpy()), dtype=torch.float32)

assert edge_attr.shape[0] == len(df), "edge_attr rows must equal number of edges"
assert edge_attr.shape[1] == len(edge_features), "edge_attr cols must equal number of features"

# ============================================================
# Encode nodes
# ============================================================
tf_encoder = LabelEncoder()
tg_encoder = LabelEncoder()
tf_ids = tf_encoder.fit_transform(df["TF"])
tg_ids = tg_encoder.fit_transform(df["TG"]) + len(tf_encoder.classes_)
n_tfs, n_tgs = len(tf_encoder.classes_), len(tg_encoder.classes_)
n_nodes = n_tfs + n_tgs
logging.info(f"Total nodes: {n_nodes} ({n_tfs} TFs, {n_tgs} TGs)")

# ============================================================
# Build graph
# ============================================================
# Directed TF->TG edges; the encoder will mirror them internally
edge_index_np = np.array([tf_ids, tg_ids], dtype=np.int64)
edge_index = torch.from_numpy(edge_index_np)

# Node features: simple expression priors (shape: n_nodes x 1)
tf_expr = df.groupby("TF")["mean_tf_expr"].mean()
tg_expr = df.groupby("TG")["mean_tg_expr"].mean()
node_features = torch.tensor(
    np.concatenate([
        tf_expr.reindex(tf_encoder.classes_).fillna(0).to_numpy().reshape(-1, 1),
        tg_expr.reindex(tg_encoder.classes_).fillna(0).to_numpy().reshape(-1, 1),
    ]),
    dtype=torch.float32
)

# ============================================================
# Model & device setup
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr).to(device)

model = GRN_GAT_Encoder(
    in_node_feats=1,
    in_edge_feats=len(edge_features),
    hidden_dim=128,
    heads=4,
    dropout=0.3,
    edge_dropout_p=0.3
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Sanity-check devices
logging.info(
    f"Device check → x:{data.x.device}, edge_index:{data.edge_index.device}, "
    f"edge_attr:{data.edge_attr.device}, model:{next(model.parameters()).device}"
)

# ============================================================
# Utility functions for self-supervised DGI training
# ============================================================

def corruption(x: torch.Tensor) -> torch.Tensor:
    """
    Corrupt node features by shuffling rows.
    Used in Deep Graph Infomax to create negative samples.
    """
    idx = torch.randperm(x.size(0))
    return x[idx]


def infomax_loss(h_real: torch.Tensor,
                 h_fake: torch.Tensor,
                 g_real: torch.Tensor,
                 EPS: float = 1e-8) -> torch.Tensor:
    """
    Deep Graph Infomax mutual-information objective.
    Encourages agreement between node embeddings (h_real)
    and the global summary vector (g_real) while penalizing
    agreement with corrupted features (h_fake).
    """
    # Global discriminator scores
    score_real = torch.sum(h_real * g_real, dim=1)
    score_fake = torch.sum(h_fake * g_real, dim=1)

    # Binary cross-entropy losses
    loss_real = -torch.log(torch.sigmoid(score_real) + EPS).mean()
    loss_fake = -torch.log(1 - torch.sigmoid(score_fake) + EPS).mean()

    return loss_real + loss_fake


# ============================================================
# Phase 1 — Self-supervised Deep Graph Infomax (DGI)
# ============================================================
phase1_losses = []

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    h_real, g_real = model(data.x, data.edge_index, data.edge_attr)
    h_fake, _ = model(corruption(data.x), data.edge_index, data.edge_attr)
    loss = infomax_loss(h_real, h_fake, g_real)
    loss.backward()
    optimizer.step()
    phase1_losses.append(loss.item())
    
    if epoch % 10 == 0:
        logging.info(f"[DGI] Epoch {epoch:03d} | Loss={loss.item():.6f}")


torch.save(model.state_dict(), "outputs/self_supervised_gat.pt")
logging.info("Saved pretrained weights to outputs/self_supervised_gat.pt")

# ============================================================
# Phase 2 — Semi-supervised fine-tuning using label edges
# ============================================================
if "label" not in df.columns:
    logging.warning("No 'label' column found — skipping fine-tuning phase.")
    exit()

logging.info("=== Phase 2: Semi-supervised fine-tuning on PKN edges ===")

# Prepare labeled edges (1 if label == 1 else 0)
df["label"] = (df["label"] == 1).astype(int)
tf_tg_pairs = torch.tensor(np.stack([tf_ids, tg_ids], axis=1), dtype=torch.long)
labels = torch.tensor(df["label"].values, dtype=torch.float32)

pairs_train, pairs_test, y_train, y_test = train_test_split(
    tf_tg_pairs.numpy(), labels.numpy(), test_size=0.2, stratify=labels.numpy(), random_state=42
)
pairs_train, pairs_test = torch.tensor(pairs_train).to(device), torch.tensor(pairs_test).to(device)
y_train, y_test = torch.tensor(y_train).float().to(device), torch.tensor(y_test).float().to(device)

# Simple classifier head on top of frozen encoder
class EdgeClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x, edge_index, edge_attr, pairs):
        h, _ = self.encoder(x, edge_index, edge_attr)
        tf_emb = h[pairs[:,0]]; tg_emb = h[pairs[:,1]]
        return self.classifier(torch.cat([tf_emb, tg_emb], dim=1)).squeeze(-1)
    
finetune_model = EdgeClassifier(model).to(device)
# Unfreeze only the last GAT + projection head
for n, p in finetune_model.encoder.named_parameters():
    p.requires_grad = ("gat2" in n) or ("proj" in n)

# Different LRs: small for encoder, larger for head
param_groups = [
    {"params": [p for n,p in finetune_model.encoder.named_parameters() if p.requires_grad], "lr": 5e-5},
    {"params": finetune_model.classifier.parameters(), "lr": 1e-4},
]
optimizer_finetune = torch.optim.Adam(param_groups, weight_decay=1e-5)

criterion = nn.BCEWithLogitsLoss()
optimizer_finetune = torch.optim.Adam(filter(lambda p: p.requires_grad, finetune_model.parameters()), lr=1e-4)

phase2_train_losses, phase2_val_auroc, phase2_val_aupr = [], [], []

pretrained_state = {k: v.detach().clone() for k, v in finetune_model.encoder.state_dict().items()}
lambda_l2sp = 1e-3

# Add L2-SP regularization to penalize deviation from pretrained weights
def l2sp_loss(module, ref_state):
    reg = 0.0
    for (k, p) in module.state_dict().items():
        if p.dtype.is_floating_point and k in ref_state:
            reg = reg + (p - ref_state[k]).pow(2).sum()
    return reg

for epoch in range(1, 101):
    finetune_model.train()
    optimizer_finetune.zero_grad()
    logits = finetune_model(data.x, data.edge_index, data.edge_attr, pairs_train)
    loss = criterion(logits, y_train)
    loss = loss + lambda_l2sp * l2sp_loss(finetune_model.encoder, pretrained_state)

    loss.backward()
    optimizer_finetune.step()
    phase2_train_losses.append(loss.item())

    if epoch % 10 == 0:
        finetune_model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(finetune_model(data.x, data.edge_index, data.edge_attr, pairs_test))
            auc = roc_auc_score(y_test.cpu(), preds.cpu())
            aupr = average_precision_score(y_test.cpu(), preds.cpu())
            phase2_val_auroc.append(auc)
            phase2_val_aupr.append(aupr)
        logging.info(f"[FineTune] Epoch {epoch:03d} | Loss={loss.item():.4f} | AUROC={auc:.3f} | AUPR={aupr:.3f}")


torch.save(finetune_model.state_dict(), "outputs/fine_tuned_gat_classifier.pt")
logging.info("Saved fine-tuned model to outputs/fine_tuned_gat_classifier.pt")

# --- use the SAME encoders used for the graph ---
# make dicts from the fitted LabelEncoders
tf_to_local = {name: i for i, name in enumerate(tf_encoder.classes_)}
tg_to_local = {name: i for i, name in enumerate(tg_encoder.classes_)}

df_scores = df[["TF", "TG"]].copy()
df_scores["TF"] = df_scores["TF"].astype(str).str.upper().str.strip()
df_scores["TG"] = df_scores["TG"].astype(str).str.upper().str.strip()

# map to local node ids; TGs must be offset by +n_tfs
tf_local = df_scores["TF"].map(tf_to_local)
tg_local = df_scores["TG"].map(tg_to_local)

valid = tf_local.notna() & tg_local.notna()
pairs_local = np.stack(
    [tf_local[valid].astype(int).to_numpy(),
     tg_local[valid].astype(int).to_numpy() + n_tfs],  # <-- offset
    axis=1
)
pairs_local = torch.as_tensor(pairs_local, dtype=torch.long, device=device)

# sanity checks
assert pairs_local[:,0].min().item() >= 0 and pairs_local[:,0].max().item() < n_tfs
assert pairs_local[:,1].min().item() >= n_tfs and pairs_local[:,1].max().item() < (n_tfs + n_tgs)

# run batched inference; align back to rows
probs = np.full(len(df_scores), np.nan, dtype=np.float32)
finetune_model.eval()
with torch.no_grad():
    B = 131072
    outs = []
    for i in range(0, pairs_local.shape[0], B):
        sl = pairs_local[i:i+B]
        logits = finetune_model(data.x, data.edge_index, data.edge_attr, sl)
        outs.append(logits.detach().cpu().numpy())
    logits = np.concatenate(outs, axis=0)
    probs_valid = 1.0 / (1.0 + np.exp(-logits))
probs[valid.values] = probs_valid.ravel()

df_scores["Score"] = probs


# ============================================================
# ===  ChIP-seq comparison (TF–TG gold edges)              ===
# ============================================================
# Expect a file with first two columns = TF, TG (case-insensitive).


def _load_chip_gold(gt_path: str, sep: str) -> pd.DataFrame:
    gt = pd.read_csv(gt_path, sep=sep)
    gt = gt.rename(columns={gt.columns[0]: "TF", gt.columns[1]: "TG"})
    # normalize column names and values
    tf_col, tg_col = gt.columns[:2]
    gt = gt.rename(columns={tf_col: "TF", tg_col: "TG"})
    gt["TF"] = gt["TF"].astype(str).str.upper().str.strip()
    gt["TG"] = gt["TG"].astype(str).str.upper().str.strip()
    if "label" not in gt.columns:
        logging.warning("No 'label' column in ChIP file — assuming all listed edges are positives.")
        gt["label"] = 1.0
    gt["label"] = pd.to_numeric(gt["label"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    return gt[["TF", "TG", "label"]]

def _precision_at_k_over_tfs(score_df: pd.DataFrame, gold_map: dict, ks=(10,20,50,100,200,500)):
    rows = []
    for k in ks:
        vals = []
        for tf, pos_set in gold_map.items():
            sub = score_df[score_df["TF"] == tf]
            if sub.empty:
                continue
            s = sub.sort_values("Score", ascending=False).head(k)
            vals.append(s["TG"].isin(pos_set).mean())
        rows.append((k, float(np.mean(vals)) if vals else np.nan))
    return pd.DataFrame(rows, columns=["k","Precision"])

try:
    if CHIP_GT_PATH and os.path.exists(CHIP_GT_PATH):
        chip_dir = os.path.join("outputs", "chip_eval")
        os.makedirs(chip_dir, exist_ok=True)
        logging.info(f"\n=== ChIP-seq comparison using: {CHIP_GT_PATH} ===")
        chip = _load_chip_gold(CHIP_GT_PATH, CHIP_GT_SEP)
        
        df_scores = canon.standardize_df(df_scores, tf_col="TF", tg_col="TG")
        chip       = canon.standardize_df(chip,       tf_col="TF", tg_col="TG")

        # Merge labels onto the model scores
        scored = df_scores.rename(columns={"prob":"Score"}).copy()
        scored = scored.merge(chip, on=["TF","TG"], how="left")
        if scored["label"].isna().any():
            nmiss = int(scored["label"].isna().sum())
            logging.warning(f"{nmiss:,} TF–TG pairs had no ChIP label — treating as 0.")
            scored["label"] = scored["label"].fillna(0).astype(int)
            
        # === Coverage diagnostics ===
        tf_chip = set(chip["TF"].unique())
        tg_chip = set(chip["TG"].unique())
        tf_model = set(df_scores["TF"].unique())
        tg_model = set(df_scores["TG"].unique())

        logging.info(f"[ChIP] TFs in file: {len(tf_chip)}, TGs in file: {len(tg_chip)}")
        logging.info(f"[Model] TFs in graph: {len(tf_model)}, TGs in graph: {len(tg_model)}")
        logging.info(f"[Overlap] TFs: {len(tf_chip & tf_model)} | TGs: {len(tg_chip & tg_model)}")

        pos_pairs_total = (chip["label"] == 1).sum()
        pos_pairs_in_model = chip.merge(df_scores[["TF","TG"]], on=["TF","TG"], how="inner")
        pos_pairs_in_model = pos_pairs_in_model[pos_pairs_in_model["label"] == 1]
        logging.info(f"[Positives] Total in ChIP file: {pos_pairs_total} | Present in model edges: {len(pos_pairs_in_model)}")

        # Per-TF positive coverage
        tf_pos_counts = chip[chip["label"]==1].groupby("TF").size().rename("n_pos_chip")
        tf_pos_in_model = pos_pairs_in_model.groupby("TF").size().rename("n_pos_in_model")
        tf_cov = pd.concat([tf_pos_counts, tf_pos_in_model], axis=1).fillna(0).astype(int)
        tf_cov["coverage_pct"] = 100 * tf_cov["n_pos_in_model"] / tf_cov["n_pos_chip"].replace(0, np.nan)
        tf_cov.sort_values("coverage_pct", ascending=True).to_csv(os.path.join(chip_dir, "tf_positive_coverage.csv"))
        logging.info(f"Saved TF positive coverage → {os.path.join(chip_dir, 'tf_positive_coverage.csv')}")

        # Overall metrics (if both classes present)
        chip_dir = os.path.join("outputs", "chip_eval")
        os.makedirs(chip_dir, exist_ok=True)
        scored.to_csv(os.path.join(chip_dir, "scored_edges_with_chip.csv"), index=False)

        y_true = scored["label"].values
        y_pred = scored["Score"].values
        if np.unique(y_true).size > 1:
            auroc = roc_auc_score(y_true, y_pred)
            aupr  = average_precision_score(y_true, y_pred)
            logging.info(f"[ChIP Overall] AUROC={auroc:.3f} | AUPR={aupr:.3f}")

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            prec, rec, _ = precision_recall_curve(y_true, y_pred)

            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--',c='gray')
            plt.title(f"ChIP ROC (AUROC={auroc:.3f})"); plt.xlabel("FPR"); plt.ylabel("TPR")
            plt.subplot(1,2,2); plt.plot(rec,prec)
            plt.title(f"ChIP PR (AUPR={aupr:.3f})"); plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.tight_layout(); plt.savefig(os.path.join(chip_dir, "overall_roc_pr.png"), dpi=180); plt.close()
        else:
            logging.warning("ChIP labels contain a single class — skipping overall AUROC/AUPR.")

        # Per-TF metrics and P@k
        gold_map = (
            scored.loc[scored["label"] == 1, ["TF","TG"]]
            .groupby("TF")["TG"].apply(lambda s: set(s.astype(str))).to_dict()
        )

        per_rows = []
        for tf, pos_set in gold_map.items():
            sub = scored[scored["TF"] == tf]
            if sub.empty: 
                continue
            y = sub["TG"].isin(pos_set).astype(int).values
            s = sub["Score"].values
            if y.sum() == 0 or y.sum() == len(y):
                auc_tf, aupr_tf = np.nan, np.nan
            else:
                auc_tf = roc_auc_score(y, s)
                aupr_tf = average_precision_score(y, s)
            order = np.argsort(-s)
            p50  = y[order[:50]].mean()  if len(y) >= 50  else y[order].mean()
            p100 = y[order[:100]].mean() if len(y) >= 100 else y[order].mean()
            per_rows.append({"TF": tf, "n": int(len(y)), "n_pos": int(y.sum()),
                             "AUROC": auc_tf, "AUPR": aupr_tf, "P@50": p50, "P@100": p100})

        per_tf = pd.DataFrame(per_rows).sort_values(["AUPR","AUROC"], ascending=False)
        per_tf.to_csv(os.path.join(chip_dir, "per_TF_metrics.csv"), index=False)
        logging.info(f"Saved per-TF ChIP metrics for {len(per_tf)} TFs → {os.path.join(chip_dir,'per_TF_metrics.csv')}")

        # Precision@k curve across TFs
        p_at_k = _precision_at_k_over_tfs(scored.rename(columns={"prob":"Score"}) if "prob" in scored.columns else scored,
                                          gold_map)
        p_at_k.to_csv(os.path.join(chip_dir, "precision_at_k.csv"), index=False)
        plt.figure(figsize=(6,4))
        plt.plot(p_at_k["k"], p_at_k["Precision"], marker="o")
        plt.xscale("log"); plt.xlabel("k"); plt.ylabel("Precision"); plt.title("Precision@k (avg over TFs)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(chip_dir, "precision_at_k.png"), dpi=180); plt.close()
        
        def lift_at_k(y, s, k):
            order = np.argsort(-s)
            top = y[order[:k]]
            base = y.mean() + 1e-12
            return float(top.mean() / base)

        def recall_at_k(y, s, k):
            order = np.argsort(-s)
            top_pos = (y[order[:k]] == 1).sum()
            total_pos = int((y == 1).sum())
            return float(top_pos / max(total_pos, 1))

        rows = []
        for k in (10, 20, 50, 100, 200, 500):
            lifts, recalls = [], []
            for tf, sub in scored.groupby("TF"):
                if sub["label"].sum() == 0:
                    continue
                y = sub["label"].to_numpy()
                s = sub["Score"].to_numpy()
                lifts.append(lift_at_k(y, s, min(k, len(sub))))
                recalls.append(recall_at_k(y, s, min(k, len(sub))))
            rows.append({"k": k, "Lift@k": np.mean(lifts) if lifts else np.nan,
                            "Recall@k": np.mean(recalls) if recalls else np.nan})
        pd.DataFrame(rows).to_csv(os.path.join(chip_dir, "lift_recall_at_k.csv"), index=False)
    else:
        logging.warning("CHIP_GT_PATH not set or file not found — skipping ChIP comparison.")
except Exception as e:
    logging.exception(f"ChIP-seq comparison failed: {e}")
    
    # ============================================================
# === Balanced scoring: macro per-TF and balanced micro     ===
# ============================================================

# 1) Macro per-TF (each TF contributes equally via simple mean)
valid_rows = []
for tf, sub in scored.groupby("TF"):
    # Need both classes to compute AUCs
    if sub["label"].nunique() < 2:
        continue
    y = sub["label"].to_numpy()
    s = sub["Score"].to_numpy()
    try:
        auc_tf  = roc_auc_score(y, s)
        aupr_tf = average_precision_score(y, s)
    except Exception:
        auc_tf, aupr_tf = np.nan, np.nan
    valid_rows.append({"TF": tf, "AUROC": auc_tf, "AUPR": aupr_tf,
                       "n": int(len(sub)), "n_pos": int(sub["label"].sum())})

macro_df = pd.DataFrame(valid_rows)
macro_auroc = float(np.nanmean(macro_df["AUROC"])) if not macro_df.empty else np.nan
macro_aupr  = float(np.nanmean(macro_df["AUPR"]))  if not macro_df.empty else np.nan
logging.info(f"[ChIP Macro] Mean per-TF AUROC={macro_auroc:.3f} | AUPR={macro_aupr:.3f}")
macro_df.to_csv(os.path.join(chip_dir, "per_TF_macro_metrics.csv"), index=False)

# 2) Balanced micro using per-TF sample weights
#    Each TF contributes the same total weight; within a TF, positives and negatives
#    split weight 50/50 (and are spread equally over their counts).
def per_tf_balanced_weights(df_tf):
    n = len(df_tf)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    n_pos = int(df_tf["label"].sum())
    n_neg = n - n_pos
    w = np.zeros(n, dtype=np.float64)
    if n_pos == 0 and n_neg == 0:
        return w
    # Within-TF: allocate 0.5 weight mass to each class (if present)
    if n_pos > 0:
        w[df_tf["label"].values == 1] = 0.5 / n_pos
    if n_neg > 0:
        w[df_tf["label"].values == 0] = 0.5 / n_neg
    # Will rescale across TFs so each TF sums to 1/N_tf later
    return w

parts, weights = [], []
for tf, sub in scored.groupby("TF"):
    if sub.empty:
        continue
    w_tf = per_tf_balanced_weights(sub)
    if w_tf.sum() == 0:
        continue
    parts.append(sub[["label","Score"]].reset_index(drop=True))
    weights.append(w_tf)

if parts:
    blended = pd.concat(parts, axis=0, ignore_index=True)
    w = np.concatenate(weights, axis=0)
    # Give each TF the same total weight: divide by #TFs that contributed
    n_tfs_contributing = len(weights)
    w = w / n_tfs_contributing
    y_all = blended["label"].to_numpy()
    s_all = blended["Score"].to_numpy()
    try:
        bal_micro_auroc = roc_auc_score(y_all, s_all, sample_weight=w)
        bal_micro_aupr  = average_precision_score(y_all, s_all, sample_weight=w)
        logging.info(f"[ChIP Balanced-Micro] AUROC={bal_micro_auroc:.3f} | AUPR={bal_micro_aupr:.3f}  "
                     f"(TF-equal + class-balanced within TF)")
        with open(os.path.join(chip_dir, "balanced_micro_summary.txt"), "w") as f:
            f.write(f"Balanced-Micro AUROC={bal_micro_auroc:.6f}\n")
            f.write(f"Balanced-Micro AUPR={bal_micro_aupr:.6f}\n")
            f.write(f"TFs contributing: {n_tfs_contributing}\n")
            
        # y_all, s_all, w already defined above in the balanced-micro section
        fpr_w, tpr_w, _ = roc_curve(y_all, s_all, sample_weight=w)
        prec_w, rec_w, _ = precision_recall_curve(y_all, s_all, sample_weight=w)

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(fpr_w, tpr_w)
        plt.plot([0,1],[0,1],'--',c='gray',lw=1)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"Balanced-Micro ROC (AUROC={bal_micro_auroc:.3f})")

        plt.subplot(1,2,2)
        plt.plot(rec_w, prec_w)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Balanced-Micro PR (AUPR={bal_micro_aupr:.3f})")

        plt.tight_layout()
        plt.savefig(os.path.join(chip_dir, "balanced_micro_roc_pr.png"), dpi=180)
        plt.close()

    except Exception as e:
        logging.warning(f"Balanced-micro scoring failed: {e}")
else:
    logging.warning("No TF groups available for balanced-micro scoring.")

# 3) (Optional) Undersampled micro (equal #pos=#neg per TF) – sanity check
#    This avoids weighting but keeps strict balance per TF by subsampling.
try:
    us_parts = []
    rng = np.random.default_rng(42)
    for tf, sub in scored.groupby("TF"):
        pos = sub[sub["label"] == 1]
        neg = sub[sub["label"] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        k = min(len(pos), len(neg))
        pos_s = pos.sample(n=k, random_state=42) if len(pos) > k else pos
        neg_s = neg.sample(n=k, random_state=42) if len(neg) > k else neg
        us_parts.append(pd.concat([pos_s, neg_s], axis=0))
    if us_parts:
        us = pd.concat(us_parts, axis=0, ignore_index=True)
        y_us = us["label"].to_numpy()
        s_us = us["Score"].to_numpy()
        auroc_us = roc_auc_score(y_us, s_us) if np.unique(y_us).size > 1 else np.nan
        aupr_us  = average_precision_score(y_us, s_us) if np.unique(y_us).size > 1 else np.nan
        logging.info(f"[ChIP Undersampled-Micro] AUROC={auroc_us:.3f} | AUPR={aupr_us:.3f}  "
                     f"(per-TF pos=neg subsample)")
        us.loc[:, ["TF","TG","Score","label"]].to_csv(os.path.join(chip_dir, "undersampled_micro_eval_table.csv"), index=False)
    else:
        logging.warning("No TFs had both classes for undersampled-micro scoring.")
except Exception as e:
    logging.warning(f"Undersampled-micro scoring failed: {e}")

# Fair-scope overall: only TFs with ≥1 positive, and only edges for those TFs
sc_tfs = [tf for tf, grp in scored.groupby("TF") if (grp["label"]==1).any()]
scoped = scored[scored["TF"].isin(sc_tfs)].copy()
if scoped["label"].nunique() > 1:
    auroc_sc = roc_auc_score(scoped["label"].values, scoped["Score"].values)
    aupr_sc  = average_precision_score(scoped["label"].values, scoped["Score"].values)
    logging.info(f"[ChIP FairScope Overall] AUROC={auroc_sc:.3f} | AUPR={aupr_sc:.3f}")




# ============================================================
# ===  Plot Phase 1 and Phase 2 Loss Curves  ===
# ============================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

# ---- Phase 1 ----
plt.subplot(1,2,1)
plt.plot(range(1, len(phase1_losses)+1), phase1_losses, color='steelblue')
plt.title("Phase 1: Self-supervised (DGI) Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

# ---- Phase 2 ----
plt.subplot(1,2,2)
plt.plot(range(1, len(phase2_train_losses)+1), phase2_train_losses, color='darkorange', label="Train Loss")
if len(phase2_val_auroc) > 0:
    # Plot validation AUROC scaled for visual comparison
    val_epochs = [i*10 for i in range(1, len(phase2_val_auroc)+1)]
    plt.twinx()
    plt.plot(val_epochs, phase2_val_auroc, color='green', label="Val AUROC")
    plt.ylabel("Validation AUROC", color='green')
plt.title("Phase 2: Fine-tuning Loss / AUROC")
plt.xlabel("Epoch")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/training_loss_curves.png", dpi=300)
plt.close()
logging.info("Saved training loss curves to outputs/training_loss_curves.png")


# ============================================================
# ===  Evaluation & Feature Importance Analysis  ===
# ============================================================


finetune_model.eval()
with torch.no_grad():
    logits = finetune_model(data.x, data.edge_index, data.edge_attr, pairs_test)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = y_test.cpu().numpy()

# ---------- Core Metrics ----------
fpr, tpr, _ = roc_curve(y_true, probs)
precision, recall, _ = precision_recall_curve(y_true, probs)
auc_val = roc_auc_score(y_true, probs)
aupr_val = average_precision_score(y_true, probs)
logging.info(f"\n=== Evaluation Metrics ===\nAUROC = {auc_val:.3f} | AUPR = {aupr_val:.3f}")

# ---------- Confusion Matrix ----------
pred_binary = (probs >= 0.5).astype(int)
cm = confusion_matrix(y_true, pred_binary)
tn, fp, fn, tp = cm.ravel()
logging.info(f"Confusion Matrix:\n{cm}")
logging.info(f"Precision={tp/(tp+fp+1e-9):.3f}, Recall={tp/(tp+fn+1e-9):.3f}, F1={(2*tp)/(2*tp+fp+fn+1e-9):.3f}")

# ---------- ROC & PR Curves ----------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, label=f"AUROC={auc_val:.3f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()

plt.subplot(1,2,2)
plt.plot(recall, precision, label=f"AUPR={aupr_val:.3f}")
plt.title("Precision-Recall Curve"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
plt.tight_layout()
plt.savefig("outputs/fine_tune_eval_curves.png", dpi=200)
plt.close()

# ============================================================
# ===  Self-Supervised Embedding Quality (Phase 1) ===
# ============================================================
logging.info("\n=== Embedding Quality Check (self-supervised phase) ===")
with torch.no_grad():
    h_final, _ = model(data.x, data.edge_index, data.edge_attr)

# Cosine similarity between known positive vs. random pairs
def cosine_sim(a,b): return F.cosine_similarity(a,b).mean().item()
pos_pairs = pairs_test[y_true == 1]
neg_pairs = pairs_test[y_true == 0]
pos_sim = cosine_sim(h_final[pos_pairs[:,0]], h_final[pos_pairs[:,1]])
neg_sim = cosine_sim(h_final[neg_pairs[:,0]], h_final[neg_pairs[:,1]])
logging.info(f"Mean cosine similarity — Positives: {pos_sim:.4f}, Negatives: {neg_sim:.4f}")

# ============================================================
# ===  Feature Importances via Gradients  ===
# ============================================================
logging.info("\n=== Computing feature importances via gradients ===")

# Clone edge attributes and enable gradients on GPU
edge_attr = data.edge_attr.clone().detach().to(device).requires_grad_(True)
finetune_model.zero_grad()

criterion = nn.BCEWithLogitsLoss()
logits = finetune_model(
    data.x.to(device),
    data.edge_index.to(device),
    edge_attr,
    pairs_test.to(device)
)
loss = criterion(logits, y_test.to(device))
loss.backward()

# Compute mean absolute gradients per feature
importance = edge_attr.grad.abs().mean(dim=0).cpu().numpy()
feature_importance = pd.Series(importance, index=edge_features).sort_values(ascending=False)
logging.info("\nFeature importance (mean |∂L/∂feature|):\n" + str(feature_importance))

# Plot importance
plt.figure(figsize=(7,4))
sns.barplot(x=feature_importance.values, y=feature_importance.index, orient="h")
plt.title("Edge Feature Importance (Gradient Magnitude)")
plt.tight_layout()
plt.savefig("outputs/feature_importance_barplot.png", dpi=200)
plt.close()

# ============================================================
# ===  Sensitivity / Ablation Analysis  ===
# ============================================================
logging.info("\n=== Performing sensitivity test per feature ===")
delta = 0.1
results = {}
for feat in edge_features:
    idx = edge_features.index(feat)
    edge_attr_mod = edge_attr.clone()
    edge_attr_mod[:, idx] += delta
    with torch.no_grad():
        preds = torch.sigmoid(finetune_model(data.x, data.edge_index, edge_attr_mod, pairs_test))
        results[feat] = roc_auc_score(y_test.cpu(), preds.cpu())
sens_df = pd.DataFrame.from_dict(results, orient='index', columns=['AUC_after_perturb'])
sens_df['ΔAUC'] = sens_df['AUC_after_perturb'] - auc_val
logging.info("\nFeature sensitivity (ΔAUROC when feature +0.1):\n" + str(sens_df.sort_values('ΔAUC', ascending=False)))
sens_df.to_csv("outputs/feature_sensitivity.csv")

# ============================================================
# ===  Per-TF, Per-TG, and Per-Pathway Evaluation  ===
# ============================================================

logging.info("\n=== Evaluating per-TF, per-TG, and KEGG pathway performance ===")

# ----- Per-TF AUROC / AUPR -----
tf_scores = []
for tf_name, tf_id in zip(tf_encoder.classes_, range(n_tfs)):
    mask = pairs_test[:,0].cpu().numpy() == tf_id
    if mask.sum() < 5:  # skip if too few edges
        continue
    y_true_tf = y_true[mask]
    y_pred_tf = probs[mask]
    if len(np.unique(y_true_tf)) < 2:
        continue
    auc_tf = roc_auc_score(y_true_tf, y_pred_tf)
    aupr_tf = average_precision_score(y_true_tf, y_pred_tf)
    tf_scores.append((tf_name, auc_tf, aupr_tf))
tf_df = pd.DataFrame(tf_scores, columns=["TF", "AUROC", "AUPR"]).sort_values("AUROC", ascending=False)
tf_df.to_csv("outputs/per_TF_metrics.csv", index=False)
logging.info(f"Saved per-TF metrics for {len(tf_df)} TFs to outputs/per_TF_metrics.csv")

# ----- Per-TG AUROC -----
tg_scores = []
for tg_name, tg_id in zip(tg_encoder.classes_, range(n_tgs)):
    mask = pairs_test[:,1].cpu().numpy() == (tg_id + n_tfs)
    if mask.sum() < 5:
        continue
    y_true_tg = y_true[mask]
    y_pred_tg = probs[mask]
    if len(np.unique(y_true_tg)) < 2:
        continue
    auc_tg = roc_auc_score(y_true_tg, y_pred_tg)
    tg_scores.append((tg_name, auc_tg))
tg_df = pd.DataFrame(tg_scores, columns=["TG", "AUROC"]).sort_values("AUROC", ascending=False)
tg_df.to_csv("outputs/per_TG_metrics.csv", index=False)
logging.info(f"Saved per-TG metrics for {len(tg_df)} TGs to outputs/per_TG_metrics.csv")

# ----- Per-Pathway (KEGG) Evaluation -----
if "kegg_pathways" in df.columns:
    logging.info("Computing per-KEGG-pathway AUROC")
    df_path = df.loc[df["kegg_pathways"].notna()].copy()
    df_path["kegg_pathways"] = df_path["kegg_pathways"].astype(str)

    pathway_metrics = []
    for pathway, subset in df_path.groupby("kegg_pathways"):
        idx = df_path.index.isin(subset.index)
        # Align with test mask
        mask = torch.tensor(idx[df.index[pairs_test.cpu().numpy()[:,0]]], dtype=torch.bool)
        y_true_pw = y_true[mask.cpu().numpy()] if mask.sum() > 0 else []
        y_pred_pw = probs[mask.cpu().numpy()] if mask.sum() > 0 else []
        if len(y_true_pw) < 5 or len(np.unique(y_true_pw)) < 2:
            continue
        auc_pw = roc_auc_score(y_true_pw, y_pred_pw)
        aupr_pw = average_precision_score(y_true_pw, y_pred_pw)
        pathway_metrics.append((pathway, auc_pw, aupr_pw))
    if pathway_metrics:
        kegg_df = pd.DataFrame(pathway_metrics, columns=["KEGG_Pathway","AUROC","AUPR"]).sort_values("AUROC", ascending=False)
        kegg_df.to_csv("outputs/per_KEGG_metrics.csv", index=False)
        logging.info(f"Saved per-pathway metrics for {len(kegg_df)} KEGG pathways to outputs/per_KEGG_metrics.csv")
    else:
        logging.warning("No valid KEGG pathway entries found for evaluation.")
else:
    logging.warning("No 'kegg_pathways' column present — skipping pathway evaluation.")

# ----- Visualization (optional) -----
import matplotlib.pyplot as plt
import seaborn as sns

# Top 50 TFs by AUROC
plt.figure(figsize=(8,4))
sns.barplot(x="AUROC", y="TF", hue="TF", data=tf_df.head(50), legend=False, palette="viridis")
plt.title("Top 50 TFs by AUROC (Fine-tuned GAT)")
plt.tight_layout()
plt.savefig("outputs/top_TF_AUROC.png", dpi=200)
plt.close()

# Top 25 KEGG pathways
if "kegg_df" in locals() and not kegg_df.empty:
    plt.figure(figsize=(8,4))
    sns.barplot(x="AUROC", y="KEGG_Pathway", data=kegg_df.head(25), hue="AUROC", palette="viridis")
    plt.title("Top 25 KEGG Pathways by AUROC")
    plt.tight_layout()
    plt.savefig("outputs/top_KEGG_AUROC.png", dpi=200)
    plt.close()

# ============================================================
# ===  Integrated Gradients (IG) Feature Attribution  ===
# ============================================================
logging.info("\n=== Computing feature importances via Integrated Gradients (IG) ===")

def integrated_gradients(
    model, data, pairs_test, y_test, edge_features, baseline=None, steps=50
):
    """
    Integrated gradients for edge features.
    Integrates gradients between a baseline and the actual input.
    """
    device = next(model.parameters()).device
    model.eval()

    # Baseline: zeros or provided
    if baseline is None:
        baseline = torch.zeros_like(data.edge_attr).to(device)

    # Scale inputs between baseline and actual edge attributes
    scaled_inputs = [
        baseline + (float(i) / steps) * (data.edge_attr - baseline)
        for i in range(steps + 1)
    ]

    total_gradients = torch.zeros_like(data.edge_attr).to(device)
    criterion = nn.BCEWithLogitsLoss()

    for scaled_attr in scaled_inputs:
        scaled_attr.requires_grad_(True)
        model.zero_grad()

        logits = finetune_model(
            data.x.to(device),
            data.edge_index.to(device),
            scaled_attr,
            pairs_test.to(device)
        )
        loss = criterion(logits, y_test.to(device))
        loss.backward()

        total_gradients += scaled_attr.grad.detach()

    avg_gradients = total_gradients / (steps + 1)
    ig_attributions = (data.edge_attr - baseline) * avg_gradients
    return ig_attributions.mean(dim=0).cpu().numpy()

# Run IG attribution
ig_attr = integrated_gradients(
    finetune_model, data, pairs_test, y_test, edge_features, steps=50
)
ig_series = pd.Series(ig_attr, index=edge_features).sort_values(ascending=False)
logging.info("\nIntegrated Gradients feature importances:\n" + str(ig_series))

# Plot IG feature importances
plt.figure(figsize=(7, 4))
sns.barplot(x=ig_series.values, y=ig_series.index, orient="h")
plt.title("Integrated Gradients Feature Importance (Mean Attribution)")
plt.tight_layout()
plt.savefig("outputs/feature_importance_IG.png", dpi=200)
plt.close()

# ============================================================
# ===  TF-Specific Integrated Gradients Analysis  ===
# ============================================================
logging.info("\n=== Computing TF-specific Integrated Gradients ===")

tf_ig_records = []

for tf_name, tf_id in zip(tf_encoder.classes_, range(n_tfs)):
    mask = pairs_test[:,0].cpu().numpy() == tf_id
    if mask.sum() < 10:  # need enough edges
        continue

    sub_pairs = pairs_test[mask]
    sub_labels = y_test[mask]
    if len(np.unique(sub_labels.cpu().numpy())) < 2:
        continue

    ig_attr = integrated_gradients(
        finetune_model,
        data,
        sub_pairs,
        sub_labels,
        edge_features,
        steps=30
    )

    tf_ig_records.append({
        "TF": tf_name,
        **{feat: val for feat, val in zip(edge_features, ig_attr)}
    })

tf_ig_df = pd.DataFrame(tf_ig_records)
tf_ig_df.to_csv("outputs/tf_specific_integrated_gradients.csv", index=False)
logging.info(f"Saved TF-specific IG feature attributions for {len(tf_ig_df)} TFs")

# ------------------------------------------------------------
# Optional: visualize top features for top 5 TFs (safe version)
# ------------------------------------------------------------
top_tfs = tf_df.head(5)["TF"].tolist()
plt.figure(figsize=(10, 4))

plotted = 0
for tf in top_tfs:
    match = tf_ig_df[tf_ig_df["TF"] == tf]
    if match.empty:
        logging.warning(f"Skipping {tf} — no IG attribution data found.")
        continue

    vals = match.iloc[0, 1:]  # feature columns
    plt.subplot(1, min(5, len(tf_ig_df)), plotted + 1)
    sns.barplot(x=vals.values, y=vals.index, orient="h")
    plt.title(tf)
    plotted += 1

if plotted == 0:
    logging.warning("No TFs had IG attribution data for plotting.")
else:
    plt.tight_layout()
    plt.savefig("outputs/tf_specific_IG_barplots.png", dpi=200)
    plt.close()
    logging.info(f"Saved IG feature barplots for {plotted} TFs.")


# ============================================================
# ===  TF Embedding Visualization (UMAP / t-SNE) ===
# ============================================================
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logging.info("\n=== Visualizing TF embeddings ===")

# Extract final node embeddings from self-supervised encoder
with torch.no_grad():
    h_final, _ = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
h_final = h_final.cpu().numpy()

# Extract TF-only embeddings (first n_tfs entries)
tf_embeddings = h_final[:n_tfs]
tf_names = tf_encoder.classes_

# ------------------------------------------------------------
# Load per-TF metrics if available
# ------------------------------------------------------------
try:
    tf_metrics = pd.read_csv("outputs/per_TF_metrics.csv")
    tf_metrics = tf_metrics.set_index("TF")
    auroc_map = [tf_metrics.loc[tf, "AUROC"] if tf in tf_metrics.index else np.nan for tf in tf_names]
except FileNotFoundError:
    logging.warning("per_TF_metrics.csv not found; skipping AUROC coloring.")
    auroc_map = [np.nan] * len(tf_names)

# ------------------------------------------------------------
# UMAP projection
# ------------------------------------------------------------
scaler = StandardScaler()
tf_scaled = scaler.fit_transform(tf_embeddings)

umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
tf_umap = umap_model.fit_transform(tf_scaled)

# Optional: alternative t-SNE view (comment out if large)
# tsne_model = TSNE(n_components=2, perplexity=20, random_state=42, learning_rate="auto")
# tf_tsne = tsne_model.fit_transform(tf_scaled)

# ------------------------------------------------------------
# Combine into DataFrame for plotting
# ------------------------------------------------------------
tf_vis_df = pd.DataFrame({
    "TF": tf_names,
    "UMAP1": tf_umap[:,0],
    "UMAP2": tf_umap[:,1],
    "AUROC": auroc_map
})
tf_vis_df.to_csv("outputs/tf_embedding_umap.csv", index=False)
logging.info(f"Saved TF UMAP embeddings to outputs/tf_embedding_umap.csv")

# ------------------------------------------------------------
# Plot UMAP colored by AUROC
# ------------------------------------------------------------
plt.figure(figsize=(7,6))
sns.scatterplot(
    data=tf_vis_df,
    x="UMAP1", y="UMAP2",
    hue="AUROC", palette="viridis", s=60, edgecolor="none"
)
plt.title("TF Embedding Map (colored by AUROC)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("outputs/tf_embedding_umap_by_AUROC.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Optional: Cluster labeling
# ------------------------------------------------------------
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, random_state=42).fit(tf_scaled)
tf_vis_df["Cluster"] = kmeans.labels_

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=tf_vis_df,
    x="UMAP1", y="UMAP2",
    hue="Cluster", palette="tab10", s=60, edgecolor="none"
)
plt.title("TF Embedding Clusters (KMeans)")
plt.tight_layout()
plt.savefig("outputs/tf_embedding_clusters.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Optional: Link clusters to top features (if IG results exist)
# ------------------------------------------------------------
try:
    ig_df = pd.read_csv("outputs/tf_specific_integrated_gradients.csv")
    ig_summary = ig_df.groupby("TF")[edge_features].mean()
    ig_summary["Cluster"] = tf_vis_df.set_index("TF")["Cluster"]
    cluster_summary = ig_summary.groupby("Cluster").mean()
    cluster_summary.to_csv("outputs/tf_cluster_feature_summary.csv")
    logging.info("Saved cluster-wise mean feature attributions to outputs/tf_cluster_feature_summary.csv")
except FileNotFoundError:
    logging.warning("tf_specific_integrated_gradients.csv not found; skipping feature summary by cluster.")
