#!/usr/bin/env python3
"""
export_gat_data.py
Stage 6 — Prepare TF–TG edge feature matrices for GAT / GNN models.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from multiomic_transformer.pipeline.io_utils import ensure_dir, write_done_flag, checkpoint_exists, StageTimer

def run_export_gat_data(
    tf_tg_feature_file: Path,
    string_pkn_file: Path,
    trrust_pkn_file: Path,
    kegg_pkn_file: Path,
    output_dir: Path,
    balance: bool = True,
    force: bool = False
):
    """Convert merged TF–TG features to a GNN-ready edge dataset."""
    ensure_dir(output_dir)
    done_flag = output_dir / "gat_export"
    if checkpoint_exists(done_flag) and not force:
        logging.info(f"[SKIP] {done_flag} already exists")
        return

    with StageTimer("Export GAT data"):
        # -------------------------------------------------------------
        # 1. Load TF–TG features
        # -------------------------------------------------------------
        df = pd.read_parquet(tf_tg_feature_file)
        df["TF"] = df["TF"].str.upper()
        df["TG"] = df["TG"].str.upper()
        logging.info(f"Loaded {len(df):,} TF–TG feature rows")

        # -------------------------------------------------------------
        # 2. Load prior-knowledge networks (PKNs)
        # -------------------------------------------------------------
        def load_pkn(path: Path) -> pd.DataFrame:
            """Load a prior-knowledge network CSV and normalize TF/TG columns."""
            df = pd.read_csv(path)
            possible_tf_cols = ["TF", "source_id", "protein1", "gene1"]
            possible_tg_cols = ["TG", "target_id", "protein2", "gene2"]

            tf_col = next((c for c in possible_tf_cols if c in df.columns), None)
            tg_col = next((c for c in possible_tg_cols if c in df.columns), None)

            if tf_col is None or tg_col is None:
                raise ValueError(f"Cannot find TF/TG columns in {path} — found columns: {df.columns.tolist()}")

            df = df.rename(columns={tf_col: "TF", tg_col: "TG"})
            df["TF"] = df["TF"].astype(str).str.upper()
            df["TG"] = df["TG"].astype(str).str.upper()
            return df

        string_pkn = load_pkn(string_pkn_file)
        trrust_pkn = load_pkn(trrust_pkn_file)
        kegg_pkn   = load_pkn(kegg_pkn_file)

        pkn_edges = (
            set(zip(string_pkn["TF"], string_pkn["TG"])) |
            set(zip(trrust_pkn["TF"], trrust_pkn["TG"])) |
            set(zip(kegg_pkn["TF"], kegg_pkn["TG"]))
        )
        logging.info(f"Loaded {len(pkn_edges):,} unique PKN edges")

        # -------------------------------------------------------------
        # 3. Label edges: 1 = in PKN, 0 = not in PKN
        # -------------------------------------------------------------
        df["label"] = [
            1 if (tf, tg) in pkn_edges else 0
            for tf, tg in zip(df["TF"], df["TG"])
        ]
        positives = df["label"].sum()
        logging.info(f"Positive edges: {positives:,} / {len(df):,}")

        # -------------------------------------------------------------
        # 4. Balance positives / negatives (optional)
        # -------------------------------------------------------------
        if balance and positives > 0:
            pos_df = df[df["label"] == 1]
            neg_df = df[df["label"] == 0].sample(n=len(pos_df), replace=True, random_state=42)
            df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=42)
            logging.info(f"Balanced dataset: {len(df):,} edges ({len(pos_df):,} pos, {len(neg_df):,} neg)")

        # -------------------------------------------------------------
        # 5. Select numeric columns & scale
        # -------------------------------------------------------------
        numeric_cols = [
            c for c in df.columns
            if df[c].dtype != "object" and c not in ["label"]
        ]
        X = df[numeric_cols].fillna(0).values.astype(np.float32)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(df["label"].values, dtype=torch.float32)

        # -------------------------------------------------------------
        # 6. Save outputs
        # -------------------------------------------------------------
        torch.save({"X": X_tensor, "y": y_tensor, "features": numeric_cols}, output_dir / "gat_features.pt")
        df.to_csv(output_dir / "gat_edges.csv", index=False)
        write_done_flag(done_flag)
        logging.info(f"[DONE] Exported GAT data → {output_dir}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Export TF–TG feature data for GAT training.")
    parser.add_argument("--tf_tg_feature_file", required=True)
    parser.add_argument("--string_pkn_file", required=True)
    parser.add_argument("--trrust_pkn_file", required=True)
    parser.add_argument("--kegg_pkn_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_export_gat_data(
        tf_tg_feature_file=Path(args.tf_tg_feature_file),
        string_pkn_file=Path(args.string_pkn_file),
        trrust_pkn_file=Path(args.trrust_pkn_file),
        kegg_pkn_file=Path(args.kegg_pkn_file),
        output_dir=Path(args.output_dir),
        force=args.force,
    )
