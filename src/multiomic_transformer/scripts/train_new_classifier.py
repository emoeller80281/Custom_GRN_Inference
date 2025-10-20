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
import logging, os

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ============================================================
# Load data
# ============================================================
df = pd.read_parquet("data/processed/PBMC/LINGER_PBMC_SC_DATA/tf_tg_merged_features.parquet")
logging.info(f"Loaded {len(df)} edges | {df['TF'].nunique()} TFs | {df['TG'].nunique()} TGs")

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
edge_index = torch.tensor([tf_ids, tg_ids], dtype=torch.long)

edge_features = [
    "reg_potential", "motif_density", "mean_tf_expr",
    "mean_tg_expr", "expr_product", "motif_present"
]
scaler = StandardScaler()
edge_attr = torch.tensor(scaler.fit_transform(df[edge_features]), dtype=torch.float32)

tf_expr = df.groupby("TF")["mean_tf_expr"].mean()
tg_expr = df.groupby("TG")["mean_tg_expr"].mean()
node_features = torch.tensor(
    np.concatenate([
        tf_expr.reindex(tf_encoder.classes_).fillna(0).to_numpy().reshape(-1,1),
        tg_expr.reindex(tg_encoder.classes_).fillna(0).to_numpy().reshape(-1,1)
    ]),
    dtype=torch.float32
)
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# ============================================================
# Model
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRN_GAT_Encoder(
    in_node_feats=1,
    in_edge_feats=len(edge_features),
    hidden_dim=128,
    heads=4,
    dropout=0.3,
    edge_dropout_p=0.3
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# ============================================================
# Phase 1 — Self-supervised Deep Graph Infomax
# ============================================================
def corruption(x): return x[torch.randperm(x.size(0))]

def infomax_loss(h_real, h_fake, g):
    pos = torch.sum(h_real * g, dim=1)
    neg = torch.sum(h_fake * g, dim=1)
    return -torch.mean(F.logsigmoid(pos) + F.logsigmoid(-neg))

data = data.to(device)
logging.info("=== Phase 1: Self-supervised pretraining ===")
for epoch in range(1, 201):
    model.train(); optimizer.zero_grad()
    h_real, g_real = model(data.x, data.edge_index, data.edge_attr)
    h_fake, _ = model(corruption(data.x), data.edge_index, data.edge_attr)
    loss = infomax_loss(h_real, h_fake, g_real)
    loss.backward(); optimizer.step()
    if epoch % 10 == 0:
        logging.info(f"[DGI] Epoch {epoch:03d} | Loss={loss.item():.6f}")

torch.save(model.state_dict(), "outputs/self_supervised_gat.pt")
logging.info("Saved pretrained weights to outputs/self_supervised_gat.pt")

# ============================================================
# Phase 2 — Semi-supervised fine-tuning using in_pkn edges
# ============================================================
if "in_pkn" not in df.columns:
    logging.warning("No 'in_pkn' column found — skipping fine-tuning phase.")
    exit()

logging.info("=== Phase 2: Semi-supervised fine-tuning on PKN edges ===")

# Prepare labeled edges (1 if in_pkn == 1 else 0)
df["label"] = (df["in_pkn"] == 1).astype(int)
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

# Fine-tune only classifier (encoder frozen by default)
encoder_frozen = True
finetune_model = EdgeClassifier(model).to(device)
if encoder_frozen:
    for p in finetune_model.encoder.parameters():
        p.requires_grad = False

criterion = nn.BCEWithLogitsLoss()
optimizer_finetune = torch.optim.Adam(filter(lambda p: p.requires_grad, finetune_model.parameters()), lr=1e-4)

for epoch in range(1, 101):
    finetune_model.train(); optimizer_finetune.zero_grad()
    logits = finetune_model(data.x, data.edge_index, data.edge_attr, pairs_train)
    loss = criterion(logits, y_train)
    loss.backward(); optimizer_finetune.step()

    if epoch % 10 == 0:
        finetune_model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(finetune_model(data.x, data.edge_index, data.edge_attr, pairs_test))
            auc = roc_auc_score(y_test.cpu(), preds.cpu())
            aupr = average_precision_score(y_test.cpu(), preds.cpu())
        logging.info(f"[FineTune] Epoch {epoch:03d} | Loss={loss.item():.4f} | AUROC={auc:.3f} | AUPR={aupr:.3f}")

torch.save(finetune_model.state_dict(), "outputs/fine_tuned_gat_classifier.pt")
logging.info("Saved fine-tuned model to outputs/fine_tuned_gat_classifier.pt")


# ============================================================
# ===  Evaluation & Feature Importance Analysis  ===
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

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
# ===  Feature Importance via Integrated Gradients  ===
# ============================================================
logging.info("\n=== Computing feature importances via gradients ===")

edge_attr.requires_grad_(True)
criterion = nn.BCEWithLogitsLoss()
finetune_model.zero_grad()
logits = finetune_model(data.x, data.edge_index, edge_attr, pairs_test)
loss = criterion(logits, y_test)
loss.backward()

# Aggregate absolute gradients across all edges
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

# ----- 1️⃣ Per-TF AUROC / AUPR -----
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

# Top 10 TFs by AUROC
plt.figure(figsize=(8,4))
sns.barplot(x="AUROC", y="TF", data=tf_df.head(10), palette="viridis")
plt.title("Top 10 TFs by AUROC (Fine-tuned GAT)")
plt.tight_layout()
plt.savefig("outputs/top_TF_AUROC.png", dpi=200)
plt.close()

# Top 10 KEGG pathways
if "kegg_df" in locals() and not kegg_df.empty:
    plt.figure(figsize=(8,4))
    sns.barplot(x="AUROC", y="KEGG_Pathway", data=kegg_df.head(10), palette="mako")
    plt.title("Top 10 KEGG Pathways by AUROC")
    plt.tight_layout()
    plt.savefig("outputs/top_KEGG_AUROC.png", dpi=200)
    plt.close()
