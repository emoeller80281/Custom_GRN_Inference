#!/usr/bin/env python
import os
import json
import torch
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from torch_geometric.data import Data
from multiomic_transformer.models.model import TFGNNClassifier

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
#  Build Graph for Inference
# ============================================================
def build_inference_graph(edge_df, tf_embeddings, tg_embeddings, tf_name2id, tg_name2id, edge_attr_cols):
    """Build PyG Data object for inference (same as training)."""
    tf_offset = 0
    tg_offset = len(tf_embeddings)
    x = torch.cat([tf_embeddings, tg_embeddings], dim=0)

    tf_idx = torch.from_numpy(edge_df["TF"].map(tf_name2id).values).long()
    tg_idx = torch.from_numpy(edge_df["TG"].map(tg_name2id).values).long() + tg_offset
    edge_index = torch.stack([tf_idx, tg_idx], dim=0)

    edge_attr = torch.tensor(edge_df[edge_attr_cols].fillna(0).values, dtype=torch.float32)
    y = torch.tensor(edge_df["label"].values, dtype=torch.float32) if "label" in edge_df else None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ============================================================
#  Evaluate Trained GNN
# ============================================================
def evaluate_combined_gnn(edge_csv, global_features_csv, tf_embeddings, tg_embeddings,
                          tf_name2id, tg_name2id, gnn_ckpt, out_dir, device="cuda:0"):
    logging.info("Loading test edges and merging with global TF–TG features")

    edge_df = pd.read_csv(edge_csv)
    global_df = pd.read_csv(global_features_csv)

    # Normalize name columns
    edge_df["TF"] = edge_df["TF"].str.upper()
    edge_df["TG"] = edge_df["TG"].str.upper()
    global_df["TF_name"] = global_df["TF_name"].str.upper()
    global_df["TG_name"] = global_df["TG_name"].str.upper()

    # Merge global features
    merged_df = edge_df.merge(
        global_df,
        left_on=["TF", "TG"],
        right_on=["TF_name", "TG_name"],
        how="left",
    )

    # Drop redundant name columns
    merged_df.drop(columns=["TF_name", "TG_name"], inplace=True, errors="ignore")

    # Fill missing values
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

    # Normalize numerics
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

    # Build inference graph
    data = build_inference_graph(merged_df, tf_embeddings, tg_embeddings,
                                 tf_name2id, tg_name2id, edge_attr_cols).to(device)

    # Load model
    model = TFGNNClassifier(num_features=tf_embeddings.shape[1]).to(device)
    model.load_state_dict(torch.load(gnn_ckpt, map_location=device))
    model.eval()

    logging.info(f"Loaded trained GNN model from {gnn_ckpt}")
    logging.info(f"Generating predictions for {data.num_edges:,} TF–TG pairs")

    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
        logits = torch.clamp(logits, -8, 8)
        logits = logits / 3.0  # match training temperature
        scores = torch.sigmoid(logits).cpu().numpy().ravel()

    # Store results
    merged_df["pred_score"] = scores
    merged_df.to_csv(os.path.join(out_dir, "gnn_predictions.csv"), index=False)

    logging.info("Saved predictions to gnn_predictions.csv")

    if "label" in merged_df:
        y_true = merged_df["label"].values
        auroc = roc_auc_score(y_true, scores)
        auprc = average_precision_score(y_true, scores)
        logging.info(f"AUROC={auroc:.3f} | AUPRC={auprc:.3f}")

        # Plot histogram
        plt.figure(figsize=(8, 6))
        bins = np.linspace(0, 1, 50)
        plt.hist(scores[y_true == 1], bins=bins, alpha=0.5, color="steelblue", label="True edges")
        plt.hist(scores[y_true == 0], bins=bins, alpha=0.5, color="lightgray", label="False edges")
        plt.axvline(0.5, color="k", linestyle="--", label="Threshold = 0.5")
        plt.xlabel("Predicted TF–TG Score")
        plt.ylabel("Frequency")
        plt.title("Predicted Score Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "score_distribution_combined.png"))
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"GNN ROC (AUROC={auroc:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "gnn_roc_curve_combined_test.png"))
        plt.close()

    logging.info("GNN inference and evaluation completed.")


# ============================================================
#  Main Entry
# ============================================================
if __name__ == "__main__":
    OUT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC/gnn_classifier/"
    EDGE_CSV = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC/combined/model_training_001/classifier/edge_features.csv"
    GLOBAL_FEATURES_CSV = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/training_data_cache/mESC/tf_tg_features_all_chr.csv"
    GNN_CKPT = os.path.join(OUT_DIR, "tf_tg_gnn.pt")
    COMBINED_EMBEDDINGS = os.path.join(OUT_DIR, "combined_embeddings.pt")
    TF_VOCAB_JSON = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/training_data_cache/common/tf_vocab.json"
    TG_VOCAB_JSON = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/training_data_cache/common/tg_vocab.json"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open(TF_VOCAB_JSON) as f:
        tf_name2id = json.load(f)
    with open(TG_VOCAB_JSON) as f:
        tg_name2id = json.load(f)

    emb_data = torch.load(COMBINED_EMBEDDINGS, map_location=device)
    tf_embeddings = emb_data["tf_embeddings"]
    tg_embeddings = emb_data["tg_embeddings"]

    evaluate_combined_gnn(
        edge_csv=EDGE_CSV,
        global_features_csv=GLOBAL_FEATURES_CSV,
        tf_embeddings=tf_embeddings,
        tg_embeddings=tg_embeddings,
        tf_name2id=tf_name2id,
        tg_name2id=tg_name2id,
        gnn_ckpt=GNN_CKPT,
        out_dir=OUT_DIR,
        device=device,
    )
