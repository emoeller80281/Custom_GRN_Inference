#!/usr/bin/env python
import os
import json
import torch
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch_geometric.data import Data
from multiomic_transformer.models.model import MultiomicTransformer, TFGNNClassifier

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
#  Load embeddings
# ============================================================
def load_embeddings(model_ckpt, tf_vocab_size, tg_vocab_size, d_model, device="cpu"):
    """Load TF/TG embeddings from a trained MultiomicTransformer checkpoint."""
    model = MultiomicTransformer(
        d_model=d_model,
        num_heads=6,
        num_layers=3,
        d_ff=768,
        dropout=0.1,
        tf_vocab_size=tf_vocab_size,
        tg_vocab_size=tg_vocab_size,
    )
    state = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    tf_embeddings = model.tf_emb_table.weight.detach().cpu()
    tg_embeddings = model.tg_emb_table.weight.detach().cpu()
    return tf_embeddings, tg_embeddings


# ============================================================
#  Build Graph Data
# ============================================================
def build_tg_graph(edge_df, tf_embeddings, tg_embeddings, tf_name2id, tg_name2id, edge_attr_cols):
    """Build PyG Data object from TF–TG edge table."""
    tf_offset = 0
    tg_offset = len(tf_embeddings)

    # Node features: TF + TG embeddings concatenated
    x = torch.cat([tf_embeddings, tg_embeddings], dim=0)

    # Build edge indices (TF → TG)
    tf_idx = torch.from_numpy(edge_df["TF"].map(tf_name2id).values).long()
    tg_idx = torch.from_numpy(edge_df["TG"].map(tg_name2id).values).long() + tg_offset
    edge_index = torch.stack([tf_idx, tg_idx], dim=0)

    # Edge attributes (local + global features)
    edge_attr = torch.tensor(edge_df[edge_attr_cols].fillna(0).values, dtype=torch.float32)

    # Labels
    y = torch.tensor(edge_df["label"].values, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ============================================================
#  Train TF–TG GNN
# ============================================================
def train_tg_gnn(edge_csv, global_features_csv, tf_embeddings, tg_embeddings, tf_name2id, tg_name2id, out_dir, device="cuda:0"):
    logging.info("Loading edge features and merging with global TF–TG features")

    edge_df = pd.read_csv(edge_csv)
    global_df = pd.read_csv(global_features_csv)

    # Normalize name columns
    edge_df["TF"] = edge_df["TF"].str.upper()
    edge_df["TG"] = edge_df["TG"].str.upper()
    global_df["TF_name"] = global_df["TF_name"].str.upper()
    global_df["TG_name"] = global_df["TG_name"].str.upper()

    # Merge global features into edge dataframe
    merged_df = edge_df.merge(
        global_df,
        left_on=["TF", "TG"],
        right_on=["TF_name", "TG_name"],
        how="left",
    )

    # Drop redundant columns
    merged_df.drop(columns=["TF_name", "TG_name"], inplace=True, errors="ignore")

    # Fill missing global features
    for col in [
        "TF_TG_expr_corr",
        "TF_mean_expr",
        "TG_mean_expr",
        "motif_density",
        "neg_log_tss",
        "log_mean_score",
    ]:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0.0)

    # Normalize all numerical columns
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    merged_df[numeric_cols] = (merged_df[numeric_cols] - merged_df[numeric_cols].mean()) / (
        merged_df[numeric_cols].std() + 1e-8
    )

    # Combine edge feature columns
    local_cols = ["attn", "motif_mask", "grad_attr", "bias_mean", "pred_mean"]
    global_cols = [
        "TF_TG_expr_corr",
        "TF_mean_expr",
        "TG_mean_expr",
        "motif_density",
        "neg_log_tss",
        "log_mean_score",
    ]
    edge_attr_cols = [c for c in local_cols + global_cols if c in merged_df.columns]

    logging.info(f"Using {len(edge_attr_cols)} edge attributes: {edge_attr_cols}")

    # Build graph
    data = build_tg_graph(merged_df, tf_embeddings, tg_embeddings, tf_name2id, tg_name2id, edge_attr_cols).to(device)

    # Initialize model
    model = TFGNNClassifier(num_features=tf_embeddings.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # Balanced BCE loss
    n_pos = float((data.y == 1).sum().item())
    n_neg = float((data.y == 0).sum().item())
    if n_pos == 0:
        logging.warning("No positive edges found, using default BCE loss.")
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logging.info(f"Using BCEWithLogitsLoss(pos_weight={pos_weight.item():.2f})")

    logging.info(f"Training GNN on {data.num_edges:,} edges ({int(n_pos)} positives, {int(n_neg)} negatives)")

    # Training loop
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        out = torch.clamp(out, -8, 8)
        out = out / 3.0  # temperature scaling
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            pred = torch.sigmoid(out).detach().cpu().numpy()
            auc = roc_auc_score(data.y.cpu().numpy(), pred)
            logging.info(f"[Epoch {epoch:03d}] Loss={loss.item():.4f} AUROC={auc:.3f}")

    # Save ROC curve
    pred = torch.sigmoid(out).detach().cpu().numpy()
    fpr, tpr, _ = roc_curve(data.y.cpu(), pred)
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("TF–TG GNN ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gnn_roc_curve_combined.png"))
    plt.close()

    # Save outputs
    torch.save(model.state_dict(), os.path.join(out_dir, "tf_tg_gnn.pt"))
    np.save(os.path.join(out_dir, "gnn_pred.npy"), pred)
    logging.info("✅ GNN training completed and artifacts saved.")


# ============================================================
#  Main entry point
# ============================================================
if __name__ == "__main__":
    model_training_num = "model_training_001"
    chrom_id_list = ["chr1", "chr2", "chr3"]

    TF_VOCAB_JSON = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/training_data_cache/common/tf_vocab.json"
    TG_VOCAB_JSON = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/training_data_cache/common/tg_vocab.json"
    OUT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC/gnn_classifier/"
    EDGE_CSV = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC/combined/model_training_001/classifier/edge_features.csv"
    GLOBAL_FEATURES_CSV = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/training_data_cache/mESC/tf_tg_features_all_chr.csv"

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(TF_VOCAB_JSON) as f:
        tf_name2id = json.load(f)
    with open(TG_VOCAB_JSON) as f:
        tg_name2id = json.load(f)

    # Combine embeddings from chromosome models
    tf_all, tg_all = [], []
    for chrom_id in chrom_id_list:
        TRAINED_MODEL = f"/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC/{chrom_id}/{model_training_num}/trained_model.pt"
        tf_e, tg_e = load_embeddings(
            model_ckpt=TRAINED_MODEL,
            tf_vocab_size=len(tf_name2id),
            tg_vocab_size=len(tg_name2id),
            d_model=384,
            device="cuda:0",
        )
        tf_all.append(tf_e)
        tg_all.append(tg_e)

    total_tf_embeddings = torch.mean(torch.stack(tf_all), dim=0)
    total_tg_embeddings = torch.mean(torch.stack(tg_all), dim=0)

    torch.save(
        {"tf_embeddings": total_tf_embeddings, "tg_embeddings": total_tg_embeddings},
        os.path.join(OUT_DIR, "combined_embeddings.pt"),
    )
    logging.info("Saved combined embeddings to combined_embeddings.pt")

    # Train model with global feature integration
    train_tg_gnn(
        edge_csv=EDGE_CSV,
        global_features_csv=GLOBAL_FEATURES_CSV,
        tf_embeddings=total_tf_embeddings,
        tg_embeddings=total_tg_embeddings,
        tf_name2id=tf_name2id,
        tg_name2id=tg_name2id,
        out_dir=OUT_DIR,
        device="cuda:0",
    )
