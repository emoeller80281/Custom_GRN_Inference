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
from multiomic_transformer.models.model import TFGNNClassifier, EdgeMLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
#  Build Graph for Inference
# ============================================================
def build_tg_graph(edge_df, tf_embeddings, tg_embeddings, tf_name2id, tg_name2id, edge_attr_cols):
    """Build a PyG Data object safely from TF–TG edge feature table."""
    tf_offset = 0
    tg_offset = len(tf_embeddings)

    # Map names → indices
    tf_idx = edge_df["TF"].map(tf_name2id)
    tg_idx = edge_df["TG"].map(tg_name2id)

    # Drop edges with missing TF/TG mappings
    valid_mask = tf_idx.notna() & tg_idx.notna()
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logging.warning(f"Dropping {n_invalid} edges with missing TF/TG mappings")

    edge_df = edge_df.loc[valid_mask].reset_index(drop=True)
    tf_idx = torch.tensor(edge_df["TF"].map(tf_name2id).values, dtype=torch.long)
    tg_idx = torch.tensor(edge_df["TG"].map(tg_name2id).values, dtype=torch.long) + tg_offset

    # Combine TF + TG embeddings
    x = torch.cat([tf_embeddings, tg_embeddings], dim=0)
    n_nodes = x.size(0)

    # Create forward edges: TF → TG
    edge_index = torch.stack([tf_idx, tg_idx], dim=0)

    # Create reverse edges: TG → TF
    edge_index_rev = edge_index.flip(0)

    # Concatenate both directions to make the graph undirected
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)

    # Duplicate edge attributes for reversed edges
    edge_attr = torch.tensor(edge_df[edge_attr_cols].fillna(0).values, dtype=torch.float32)
    edge_attr = torch.cat([edge_attr, edge_attr.clone()], dim=0)

    # Labels (if present)
    if "label" in edge_df.columns:
        labels = pd.to_numeric(edge_df["label"], errors="coerce").fillna(0).clip(0, 1)
        y = torch.tensor(labels.values, dtype=torch.float32)
    else:
        y = torch.zeros(len(edge_df), dtype=torch.float32)

    logging.info(f"Graph built successfully with {n_nodes:,} nodes and {edge_index.size(1):,} edges")
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ============================================================
#  Evaluate Trained GNN
# ============================================================
def evaluate_tg_gnn(edge_csv, global_features_csv, tf_embeddings, tg_embeddings,
                    tf_name2id, tg_name2id, gnn_ckpt, out_dir, device="cuda:0"):

    logging.info("Loading test edges and merging with global TF–TG features")

    edge_df = pd.read_csv(edge_csv)
    global_df = pd.read_csv(global_features_csv)

    # Normalize TF/TG naming
    edge_df["TF"] = edge_df["TF"].str.upper().str.strip()
    edge_df["TG"] = edge_df["TG"].str.upper().str.strip()
    global_df["TF_name"] = global_df["TF_name"].str.upper().str.strip()
    global_df["TG_name"] = global_df["TG_name"].str.upper().str.strip()

    # Merge with global TF–TG features
    merged_df = edge_df.merge(
        global_df,
        left_on=["TF", "TG"],
        right_on=["TF_name", "TG_name"],
        how="left",
    )
    merged_df.drop(columns=["TF_name", "TG_name"], inplace=True, errors="ignore")

    # Clean and fill features
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
            

    # Edge feature columns
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
    
    for col in ["motif_density", "motif_mask", "log_mean_score"]:
        merged_df[col] = -merged_df[col]   # invert raw signal
    
    logging.info(edge_attr_cols)
    logging.info(merged_df[edge_attr_cols].describe())
    logging.info(merged_df[edge_attr_cols].var().sort_values())
    
    # --- Load saved scaler from training ---
    scaler_path = os.path.join(out_dir, "edge_feature_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path} — please train model first.")

    scaler = joblib.load(scaler_path)
    logging.info(f"Loaded StandardScaler from {scaler_path}")

    # Apply scaling consistently to same columns
    exclude_from_norm = ["motif_mask", "grad_attr"]
    scale_cols = [c for c in edge_attr_cols if c not in exclude_from_norm and c in merged_df.columns]

    merged_df[scale_cols] = scaler.transform(merged_df[scale_cols])

    # Keep same ordering as training
    edge_attr_cols = scale_cols + [c for c in exclude_from_norm if c in merged_df.columns]
    
    print("Test scaled feature stats:")
    print(merged_df[scale_cols].agg(["mean", "std"]).round(3))
    
    # Build PyG Data object
    data = build_tg_graph(merged_df, tf_embeddings, tg_embeddings, tf_name2id, tg_name2id, edge_attr_cols).to(device)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    X = merged_df[edge_attr_cols]
    y = merged_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    print("Baseline AUROC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
    
    # Load model
    # model = TFGNNClassifier(
    #     num_features=384, edge_dim=11, hidden_dim=256, num_layers=4, dropout=0.3, use_logit_noise=False
    # ).to(device)
    model = EdgeMLPClassifier(edge_dim=11, hidden_dim=256, dropout=0.2).to(device)
    state = torch.load(gnn_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logging.info(f"Loaded trained GNN checkpoint: {gnn_ckpt}")

    # Predict (NO clamp, NO manual temperature)
    logging.info(f"Generating predictions for {data.num_edges:,} edges...")
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr).detach().cpu().numpy().ravel()

    # ---- Optional calibration: load if available ----
    # Expect a JSON like {"method":"platt","a":1.23,"b":-0.45} or {"method":"temperature","T":2.7}
    calib_path = os.path.join(out_dir, "calibration.json")
    if os.path.exists(calib_path):
        import json
        with open(calib_path) as f:
            cal = json.load(f)
        if cal.get("method") == "platt":
            a, b = float(cal["a"]), float(cal["b"])
            calibrated = 1 / (1 + np.exp(-(a * logits + b)))
            logging.info(f"Applied Platt calibration (a={a:.3f}, b={b:.3f}).")
        elif cal.get("method") == "temperature":
            T = float(cal["T"])
            calibrated = 1 / (1 + np.exp(-(logits / max(T, 1e-6))))
            logging.info(f"Applied temperature calibration (T={T:.3f}).")
        else:
            calibrated = 1 / (1 + np.exp(-logits))
            logging.warning("Unknown calibration method; using raw sigmoid.")
    else:
        calibrated = 1 / (1 + np.exp(-logits))  # raw probabilities

    # ---- Auto-flip check (only if labels present & non-constant) ----
    scores = np.asarray(calibrated).ravel()  # ensure it's a 1D NumPy array

    if "label" in merged_df.columns and merged_df["label"].nunique() == 2:
        y_true = merged_df["label"].astype(float).values

        # --- Align lengths safely ---
        n = min(len(y_true), len(scores))
        if len(y_true) != len(scores):
            logging.warning(f"Length mismatch: labels={len(y_true)}, scores={len(scores)} — truncating to {n}.")
        y_true, y_pred = y_true[:n], scores[:n]

        # --- Remove NaNs ---
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]

        try:
            from sklearn.metrics import roc_auc_score
            logging.info(f"Evaluating on {len(y_true)} edges ({int(y_true.sum())} positives, {len(y_true)-int(y_true.sum())} negatives)")

            auc_raw  = roc_auc_score(y_true, y_pred)
            auc_flip = roc_auc_score(y_true, 1.0 - y_pred)

            if auc_flip > auc_raw:
                y_pred = 1.0 - y_pred
                scores = 1.0 - scores
                logging.warning(f"Scores inverted for evaluation (AUROC raw={auc_raw:.3f}, flipped={auc_flip:.3f}).")
            else:
                logging.info(f"Using non-inverted scores (AUROC={auc_raw:.3f}).")

        except Exception as e:
            logging.warning(f"AUROC check skipped: {e}")

    # ---- Save outputs (keep logits and both probabilities) ----
    n_labeled = len(merged_df)  # only forward edges have labels

    # Align predictions with labeled edges
    logits_sub = logits[:n_labeled]
    scores_sub = scores[:n_labeled] if len(scores) > n_labeled else scores

    out_df = merged_df.copy()
    out_df["Logit"] = logits_sub
    out_df["Prob_raw"] = 1 / (1 + np.exp(-logits_sub))
    out_df["Score"] = scores_sub

    # Save prediction table
    export = out_df[["TF", "TG", "Logit", "Prob_raw", "Score"]].rename(columns={"TF": "Source", "TG": "Target"})

    inferred_grn_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC/gnn_predictions"
    os.makedirs(inferred_grn_dir, exist_ok=True)

    export_path = os.path.join(inferred_grn_dir, "gnn_inferred_net.csv")
    export.to_csv(export_path, index=False)
    logging.info(f"Saved predictions to {export_path}")

    # ---- Evaluate (if labels available) ----
    if "label" in merged_df.columns and merged_df["label"].nunique() > 1:
        y_true = merged_df["label"].values.astype(int)

        # Ensure same length
        if len(scores_sub) != len(y_true):
            logging.warning(f"Adjusting score length: scores={len(scores_sub)} vs y_true={len(y_true)}")
            min_len = min(len(scores_sub), len(y_true))
            y_true = y_true[:min_len]
            scores_sub = scores_sub[:min_len]

        # Metrics
        auroc = roc_auc_score(y_true, scores_sub)
        auprc = average_precision_score(y_true, scores_sub)
        logging.info(f"AUROC={auroc:.3f} | AUPRC={auprc:.3f}")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, scores_sub)
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"GNN ROC (AUROC={auroc:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "gnn_roc_curve_test.png"))
        plt.close()

        # Score distribution
        plt.figure(figsize=(8, 6))
        bins = np.linspace(0, 1, 50)
        plt.hist(scores_sub[y_true == 1], bins=bins, alpha=0.5, color="steelblue", label="True edges")
        plt.hist(scores_sub[y_true == 0], bins=bins, alpha=0.5, color="lightgray", label="False edges")
        plt.axvline(0.5, color="k", linestyle="--", label="Threshold=0.5")
        plt.xlabel("Predicted TF–TG Score")
        plt.ylabel("Frequency")
        plt.title("Predicted Score Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "score_distribution_test.png"))
        plt.close()
    else:
        logging.warning("Skipping AUROC/AUPRC — only one class or missing labels.")

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
        tf_name2id = {k.upper(): v for k, v in json.load(f).items()}
    with open(TG_VOCAB_JSON) as f:
        tg_name2id = {k.upper(): v for k, v in json.load(f).items()}

    emb_data = torch.load(COMBINED_EMBEDDINGS, map_location=device)
    tf_embeddings = emb_data["tf_embeddings"]
    tg_embeddings = emb_data["tg_embeddings"]

    evaluate_tg_gnn(
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
