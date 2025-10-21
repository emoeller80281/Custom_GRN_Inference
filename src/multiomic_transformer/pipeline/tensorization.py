#!/usr/bin/env python3
"""
tensorization.py
Stage 4 — Convert merged TF–TG feature tables into transformer-ready tensors.

This stage:
    • Loads TF–TG feature parquet from Stage 3.
    • Builds TF and TG vocabularies (JSON mappings).
    • Scales numeric features and saves as a single tensor.
    • Produces checkpoint flags for reproducibility.

Outputs:
    transformer_features.pt
    tf_vocab.json
    tg_vocab.json
    tensorization.done
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from multiomic_transformer.pipeline.io_utils import ensure_dir, checkpoint_exists, write_done_flag, StageTimer

def run_tensorization(tf_tg_features_file: Path, output_dir: Path, force: bool = False):
    """
    Stage 4: Convert merged TF–TG features into tensors and vocabularies
    for downstream transformer or GNN training.

    Parameters
    ----------
    tf_tg_features_file : Path
        Input Parquet file from Stage 3 (TF–TG feature construction).
    output_dir : Path
        Directory where tensor outputs and vocab files will be written.
    force : bool, optional
        If True, re-run even if checkpoint exists.
    """
    ensure_dir(output_dir)
    done_flag = output_dir / "tensorization"
    if checkpoint_exists(done_flag) and not force:
        logging.info(f"[SKIP] Tensorization already completed at {output_dir}")
        return

    with StageTimer("Tensorization"):
        # -------------------------------------------------------------
        # 1. Load merged TF–TG features
        # -------------------------------------------------------------
        logging.info(f"Loading TF–TG features: {tf_tg_features_file}")
        df = pd.read_parquet(tf_tg_features_file)
        if df.empty:
            raise ValueError(f"No data found in {tf_tg_features_file}")

        # -------------------------------------------------------------
        # 2. Build TF and TG vocabularies
        # -------------------------------------------------------------
        tfs = sorted(df["TF"].dropna().unique().tolist())
        tgs = sorted(df["TG"].dropna().unique().tolist())
        tf_vocab = {tf: i for i, tf in enumerate(tfs)}
        tg_vocab = {tg: i for i, tg in enumerate(tgs)}
        logging.info(f"Vocabularies: {len(tf_vocab)} TFs, {len(tg_vocab)} TGs")

        # -------------------------------------------------------------
        # 3. Select numeric feature columns
        # -------------------------------------------------------------
        numeric_cols = [c for c in df.columns if df[c].dtype != "object"]
        if not numeric_cols:
            raise ValueError("No numeric feature columns found in TF–TG dataframe.")
        
        # Coerce any numeric-like columns
        for c in df.columns:
            if c not in ["TF", "TG"]:
                df[c] = pd.to_numeric(df[c])
                
        X = df[numeric_cols].fillna(0).values.astype(np.float32)
        logging.info(f"Using {len(numeric_cols)} numeric feature columns")

        # -------------------------------------------------------------
        # 4. Feature scaling
        # -------------------------------------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        torch.save(X_tensor, output_dir / "transformer_features.pt")
        logging.info(f"Saved feature tensor → {output_dir / 'transformer_features.pt'}")

        # -------------------------------------------------------------
        # 5. Save vocabularies
        # -------------------------------------------------------------
        with open(output_dir / "tf_vocab.json", "w") as f:
            json.dump(tf_vocab, f, indent=2)
        with open(output_dir / "tg_vocab.json", "w") as f:
            json.dump(tg_vocab, f, indent=2)
        logging.info("Saved TF/TG vocabularies")

        # -------------------------------------------------------------
        # 6. Write checkpoint flag
        # -------------------------------------------------------------
        write_done_flag(done_flag)
        logging.info(f"[DONE] Tensorization complete → {output_dir}")


# ---------------------------------------------------------------------
# Optional utilities (for later extensions)
# ---------------------------------------------------------------------
def align_to_vocab(names, vocab):
    """
    Return a list of indices matching names to a vocabulary.
    Unmatched entries return -1.
    """
    return [vocab.get(n, -1) for n in names]


def load_tensorization_outputs(output_dir: Path):
    """
    Convenience loader for downstream training.
    """
    X = torch.load(output_dir / "transformer_features.pt")
    with open(output_dir / "tf_vocab.json") as f:
        tf_vocab = json.load(f)
    with open(output_dir / "tg_vocab.json") as f:
        tg_vocab = json.load(f)
    return X, tf_vocab, tg_vocab
