#!/usr/bin/env python
import os
import json
import joblib
import torch
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge
import seaborn as sns
from multiomic_transformer.models.model import MultiomicTransformer, TFGNNClassifier, EdgeMLPClassifier
from sklearn.preprocessing import StandardScaler

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

    # Verify that edge indices are within bounds
    if len(tf_idx) == 0 or len(tg_idx) == 0:
        raise ValueError("No valid TF–TG edges left after filtering. Check vocab alignment.")

    # Create forward edges: TF → TG
    edge_index = torch.stack([tf_idx, tg_idx], dim=0)

    # Create reverse edges: TG → TF
    edge_index_rev = edge_index.flip(0)

    # Concatenate both directions to make the graph undirected
    edge_index = torch.cat([edge_index, edge_index_rev], dim=1)
    
    # Duplicate edge attributes accordingly
    edge_attr = torch.tensor(edge_df[edge_attr_cols].fillna(0).values, dtype=torch.float32)
    edge_attr = torch.cat([edge_attr, edge_attr.clone()], dim=0)
    
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge_attr shape: {edge_attr.shape}")
    print(f"Unique TF nodes: {torch.unique(tf_idx).numel()} | Unique TG nodes: {torch.unique(tg_idx).numel()}")

    # Labels
    if "label" not in edge_df.columns:
        logging.warning("No 'label' column found — defaulting all edges to 0")
        y = torch.zeros(len(edge_df), dtype=torch.float32)
    else:
        labels = pd.to_numeric(edge_df["label"], errors="coerce").fillna(0).clip(0, 1)
        y = torch.tensor(labels.values, dtype=torch.float32)

    y = torch.cat([y, y.clone()], dim=0)
    
    logging.info(f"Graph built successfully with {n_nodes:,} nodes and {edge_index.size(1):,} edges")
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ============================================================
#  Train TF–TG GNN
# ============================================================
def train_tg_gnn(edge_csv, ground_truth_csv, sep, global_features_csv, tf_embeddings, tg_embeddings, tf_name2id, tg_name2id, out_dir, device="cuda:0"):
    logging.info("Loading edge features and merging with global TF–TG features")

    # --- Load edge and ground-truth data ---
    edge_df = pd.read_csv(edge_csv)
    global_df = pd.read_csv(global_features_csv)

    # --- Load and normalize external ground truth labels ---
    ground_truth_df = pd.read_csv(ground_truth_csv, sep=sep)

    # Automatically assume first two columns are TF and TG
    tf_col, tg_col = ground_truth_df.columns[:2]
    ground_truth_df = ground_truth_df.rename(columns={tf_col: "TF", tg_col: "TG"})

    # If there's no label column, assume all edges in this file are positives (label=1)
    if "label" not in ground_truth_df.columns:
        logging.warning("No 'label' column found in ground_truth_csv — assuming all entries are positive edges.")
        ground_truth_df["label"] = 1.0

    # Clean text formatting
    ground_truth_df["TF"] = ground_truth_df["TF"].astype(str).str.upper().str.strip()
    ground_truth_df["TG"] = ground_truth_df["TG"].astype(str).str.upper().str.strip()

    # Ensure label is numeric binary
    ground_truth_df["label"] = (
        pd.to_numeric(ground_truth_df["label"], errors="coerce")
        .fillna(0)
        .clip(0, 1)
        .astype(float)
    )

    # Normalize edge_df names before merging
    edge_df["TF"] = edge_df["TF"].astype(str).str.upper().str.strip()
    edge_df["TG"] = edge_df["TG"].astype(str).str.upper().str.strip()

    # Merge external labels into edge_df
    edge_df = edge_df.drop(columns=["label"], errors="ignore").merge(
        ground_truth_df[["TF", "TG", "label"]],
        on=["TF", "TG"],
        how="left",
    )

    # Replace missing labels with 0 (negative edges)
    missing_labels = edge_df["label"].isna().sum()
    if missing_labels > 0:
        logging.warning(f"{missing_labels:,} edges had no ground-truth label — assigning 0.")
        edge_df["label"] = edge_df["label"].fillna(0)

    # Convert final label column to binary numeric
    edge_df["label"] = (
        pd.to_numeric(edge_df["label"], errors="coerce")
        .fillna(0)
        .clip(0, 1)
        .astype(float)
    )

    logging.info(f"Ground-truth labels merged: {edge_df['label'].value_counts(dropna=False).to_dict()}")


    # Normalize name columns
    edge_df["TF"] = edge_df["TF"].str.upper()
    edge_df["TG"] = edge_df["TG"].str.upper()
    global_df["TF_name"] = global_df["TF_name"].str.upper()
    global_df["TG_name"] = global_df["TG_name"].str.upper()
    
    logging.info(edge_df["label"].value_counts(dropna=False))
    logging.info(f"label dtype: {edge_df['label'].dtype}")

    # Merge global features into edge dataframe
    merged_df = edge_df.merge(
        global_df,
        left_on=["TF", "TG"],
        right_on=["TF_name", "TG_name"],
        how="left",
    )
    
    pos_df = merged_df[merged_df["label"] == 1.0]
    neg_df = merged_df[merged_df["label"] == 0.0]

    # Undersample negatives to match positives
    pos_per_tf = pos_df["TF"].value_counts().to_dict()

    neg_sampled_df = (
        neg_df.groupby("TF", group_keys=False)
            .apply(lambda g: g.sample(
                n=min(len(g), pos_per_tf.get(g.name, 0)),
                random_state=42))
    )

    balanced_df = pd.concat([pos_df, neg_sampled_df], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

    logging.info(f"Balanced dataset: {len(pos_df)} positives, {len(neg_sampled_df)} negatives")
    merged_df = balanced_df
    
    # --- Clean label column ---
    if "label" not in merged_df.columns:
        logging.warning("No 'label' column found — defaulting to zeros.")
        merged_df["label"] = 0.0
    else:
        merged_df["label"] = (
            pd.to_numeric(merged_df["label"], errors="coerce")
            .fillna(0)                # replace NaNs with 0
            .clip(0, 1)               # enforce binary range
            .astype(float)
        )

    logging.info(merged_df["label"].value_counts(dropna=False))
    logging.info(f"After merge: positives={np.sum(merged_df['label'] == 1)}, "
             f"negatives={np.sum(merged_df['label'] == 0)}")

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
            
    from sklearn.metrics import roc_auc_score

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
    
    # --- Fit StandardScaler on training edge attributes ---
    exclude_from_norm = ["motif_mask", "grad_attr"]
    scale_cols = [c for c in edge_attr_cols if c not in exclude_from_norm]
    for col in ["motif_density", "motif_mask", "log_mean_score"]:
        if merged_df[col].corr(merged_df["label"]) < 0:
            merged_df[col] = -merged_df[col]
            logging.info(f"Inverted {col} to match positive correlation with label.")


    scaler = StandardScaler()
    merged_df[scale_cols] = scaler.fit_transform(merged_df[scale_cols])

    # Save fitted scaler for reuse during inference
    scaler_path = os.path.join(out_dir, "edge_feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logging.info(f"Saved StandardScaler to {scaler_path}")

    # Update edge_attr_cols to use only normalized columns
    edge_attr_cols = scale_cols + [c for c in exclude_from_norm if c in merged_df.columns]
    
    for col in edge_attr_cols:
        try:
            auc = roc_auc_score(merged_df["label"], merged_df[col])
            print(f"{col:20s} AUROC (raw): {auc:.3f}")
        except ValueError:
            pass
        

    
    print("Train scaled feature stats:")
    print(f"edge_attr_cols: {edge_attr_cols}")
    print(merged_df[edge_attr_cols].describe())
    print(merged_df[scale_cols].agg(["mean", "std"]).round(3))
    
    # logging.info("Plotting feature signal structure")
    # sig_cols = ["motif_mask", "motif_density", "log_mean_score", "neg_log_tss"]
    # sns.pairplot(merged_df, vars=sig_cols, hue="label", plot_kws={"alpha":0.3})
    # plt.suptitle("Signal structure among key features", y=1.02)
    # plt.savefig(os.path.join(out_dir, "feature_signal_structure.png"))
    
    from sklearn.linear_model import LogisticRegression
    X = merged_df[["motif_mask","motif_density","log_mean_score","neg_log_tss","TF_mean_expr"]]
    y = merged_df["label"]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    print("Logistic Regression Baseline AUROC:", roc_auc_score(y, clf.predict_proba(X)[:,1]))
    
    feature_names = ["motif_mask","motif_density","log_mean_score","neg_log_tss","TF_mean_expr","TF_TG_expr_corr"]
    coefs = clf.coef_.flatten()
    for name, coef in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
        print(f"{name:20s} {coef:+.3f}")


    logging.info(f"Using {len(edge_attr_cols)} edge attributes: {edge_attr_cols}")

    train_df, val_df = train_test_split(
        merged_df,
        stratify=merged_df["label"],
        test_size=0.2,
        random_state=42
    )

    logging.info(f"Train edges: {len(train_df):,} | Val edges: {len(val_df):,}")
    
    # Build graph
    train_data = build_tg_graph(
        train_df, tf_embeddings, tg_embeddings, tf_name2id, tg_name2id, edge_attr_cols
    ).to(device)
    
    val_data = build_tg_graph(
        val_df, tf_embeddings, tg_embeddings, tf_name2id, tg_name2id, edge_attr_cols
    ).to(device)

    logging.info(f"Train graph: {train_data.num_nodes:,} nodes, {train_data.num_edges:,} edges")
    logging.info(f"Val graph:   {val_data.num_nodes:,} nodes, {val_data.num_edges:,} edges")
    
    # ----- Initialize model/opt/loss -----
    model = TFGNNClassifier(
        num_features=384,
        edge_dim=11,
        hidden_dim=256,
        num_layers=4,
        dropout=0.3,
        use_logit_noise=False,
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    logging.info(f"Using BCEWithLogitsLoss()")
    

    n_pos = int((train_data.y == 1).sum().item())
    n_neg = int((train_data.y == 0).sum().item())

    logging.info(f"Training GNN on {train_data.num_edges:,} edges "
                f"({n_pos} positives, {n_neg} negatives)")
    logging.info(f"Node feature dim = {train_data.x.shape[1]}")
    logging.info(f"Edge feature dim = {train_data.edge_attr.shape[1]}")
    logging.info(f"Graph: {train_data.num_nodes:,} nodes, {train_data.num_edges:,} edges")

    # ============================================================
    # 4. Training loop
    # ============================================================
    best_val_auc = -np.inf
    best_state = None
    patience = 50
    epochs_since_improve = 0

    for epoch in range(2000):
        model.train()
        optimizer.zero_grad()

        # ---- Forward + backward ----
        logits = model(train_data.x, train_data.edge_index, train_data.edge_attr)
        loss = loss_fn(logits, train_data.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            val_logits = model(val_data.x, val_data.edge_index, val_data.edge_attr)
            val_probs = torch.sigmoid(val_logits).detach().cpu().numpy().ravel()
            val_labels = val_data.y.cpu().numpy().ravel()
            val_auc = roc_auc_score(val_labels, val_probs)

            train_probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            train_labels = train_data.y.cpu().numpy().ravel()
            train_auc = roc_auc_score(train_labels, train_probs)

        # ---- Logging ----
        lr = optimizer.param_groups[0]["lr"]
        if epoch % 25 == 0 or epoch == 0:
            logging.info(
                f"[Epoch {epoch:04d}] "
                f"Loss={loss.item():.4f} | "
                f"AUC(train)={train_auc:.3f} | AUC(val)={val_auc:.3f} | LR={lr:.2e}"
            )

        # ---- Scheduler + Early stopping ----
        scheduler.step(val_auc)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                logging.info(
                    f"Early stopping at epoch {epoch} "
                    f"(best val AUROC = {best_val_auc:.3f})"
                )
                break

    # ----- Restore best weights -----
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    logging.info(f"Training complete. Best validation AUROC = {best_val_auc:.3f}")

    # ============================================================
    # 6. Save results and calibration (optional)
    # ============================================================
    with torch.no_grad():
        val_logits = model(val_data.x, val_data.edge_index, val_data.edge_attr).cpu().numpy().ravel()
    val_labels = val_data.y.cpu().numpy().ravel()

    # ============================================================
    # 6. Temperature scaling (simple, single parameter)
    # ============================================================
    import scipy.optimize as opt

    model.eval()
    with torch.no_grad():
        train_logits = model(train_data.x, train_data.edge_index, train_data.edge_attr).cpu().numpy().ravel()
        val_logits   = model(val_data.x,   val_data.edge_index,   val_data.edge_attr).cpu().numpy().ravel()

    train_labels = train_data.y.cpu().numpy().ravel()
    val_labels   = val_data.y.cpu().numpy().ravel()

    def nll_for_T(T):
        T = max(T, 1e-6)
        p = 1 / (1 + np.exp(-(val_logits / T)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -np.mean(val_labels * np.log(p) + (1 - val_labels) * np.log(1 - p))

    res = opt.minimize_scalar(nll_for_T, bounds=(0.5, 5.0), method="bounded")
    T_opt = float(res.x) if res.success else 1.0
    with open(os.path.join(out_dir, "calibration.json"), "w") as f:
        json.dump({"method": "temperature", "T": T_opt}, f)
    logging.info(f"Saved calibration.json with T={T_opt:.3f}")


    # ============================================================
    # 8. Save artifacts
    # ============================================================
    torch.save(model.state_dict(), os.path.join(out_dir, "tf_tg_gnn.pt"))
    np.save(os.path.join(out_dir, "train_probs.npy"), train_probs)
    np.save(os.path.join(out_dir, "val_probs.npy"), val_probs)
    logging.info("GNN training completed and artifacts saved.")

# ============================================================
#  Main entry point
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--ground_truth_file",
        required=True,
        help="Path to a ground truth file where the first column has TF names and the second column has TG names."
    )
    
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--sep",
        required=True,
        help="Ground truth separator"
    )
    
    args = parser.parse_args()
    
    model_training_num = "model_training_001"
    chrom_id_list = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7"]

    TF_VOCAB_JSON = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/training_data_cache/common/tf_vocab.json"
    TG_VOCAB_JSON = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/training_data_cache/common/tg_vocab.json"
    OUT_DIR = args.output_dir
    EDGE_CSV = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC/combined/edge_features/edge_features.csv"
    GLOBAL_FEATURES_CSV = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/training_data_cache/mESC/tf_tg_features_all_chr.csv"
    GROUND_TRUTH = args.ground_truth_file

    os.makedirs(OUT_DIR, exist_ok=True)

    with open(TF_VOCAB_JSON) as f:
        tf_name2id = json.load(f)
    with open(TG_VOCAB_JSON) as f:
        tg_name2id = json.load(f)
        
    # Normalize vocab case to uppercase for consistency
    tf_name2id = {k.upper(): v for k, v in tf_name2id.items()}
    tg_name2id = {k.upper(): v for k, v in tg_name2id.items()}

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
        ground_truth_csv=GROUND_TRUTH,
        sep=args.sep,
        global_features_csv=GLOBAL_FEATURES_CSV,
        tf_embeddings=total_tf_embeddings,
        tg_embeddings=total_tg_embeddings,
        tf_name2id=tf_name2id,
        tg_name2id=tg_name2id,
        out_dir=OUT_DIR,
        device="cuda:0",
    )
