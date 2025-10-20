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
# Convert once to NumPy array → fast tensor creation, no warning
edge_index_np = np.array([tf_ids, tg_ids], dtype=np.int64)
edge_index = torch.from_numpy(edge_index_np)

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
        tf_expr.reindex(tf_encoder.classes_).fillna(0).to_numpy().reshape(-1, 1),
        tg_expr.reindex(tg_encoder.classes_).fillna(0).to_numpy().reshape(-1, 1)
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

phase2_train_losses, phase2_val_auroc, phase2_val_aupr = [], [], []

for epoch in range(1, 101):
    finetune_model.train()
    optimizer_finetune.zero_grad()
    logits = finetune_model(data.x, data.edge_index, data.edge_attr, pairs_train)
    loss = criterion(logits, y_train)
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

# Top 10 TFs by AUROC
plt.figure(figsize=(8,4))
sns.barplot(x="AUROC", y="TF", hue="TF", data=tf_df.head(10), legend=False, palette="viridis")
plt.title("Top 10 TFs by AUROC (Fine-tuned GAT)")
plt.tight_layout()
plt.savefig("outputs/top_TF_AUROC.png", dpi=200)
plt.close()

# Top 10 KEGG pathways
if "kegg_df" in locals() and not kegg_df.empty:
    plt.figure(figsize=(8,4))
    sns.barplot(x="AUROC", y="KEGG_Pathway", data=kegg_df.head(10), hue="AUROC", palette="viridis")
    plt.title("Top 10 KEGG Pathways by AUROC")
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
