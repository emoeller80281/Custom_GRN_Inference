#!/usr/bin/env python3
"""
dataset_assembly.py
Stage 5 — Prepare transformer-ready tensors and vocabularies for training.

This stage:
    • Loads transformer_features.pt + tf/tg vocabularies from Stage 4.
    • Optionally aligns TF/TG pairs and constructs edge_index tensors.
    • Writes files in the format expected by MultiomicTransformerDataset.
    • Produces a checkpoint flag for reproducibility.

Outputs (under output_dir):
    ├── tg_tensor_all_<chrom_id>.pt
    ├── tf_vocab.json
    ├── tg_vocab.json
    ├── edge_index.pt               (optional placeholder)
    └── dataset_assembly.done
"""

import json
import logging
from pathlib import Path
import torch
import pandas as pd
from multiomic_transformer.pipeline.io_utils import (
    ensure_dir,
    checkpoint_exists,
    write_done_flag,
)
from multiomic_transformer.pipeline.io_utils import StageTimer


def run_dataset_assembly(
    tensor_dir: Path,
    tf_tg_features_file: Path,
    output_dir: Path,
    chrom_id: str = "all",
    force: bool = False,
):
    """
    Stage 5: Assemble tensorization outputs into a dataset directory.

    Parameters
    ----------
    tensor_dir : Path
        Directory from Stage 4 containing transformer_features.pt and vocab JSONs.
    tf_tg_features_file : Path
        Stage 3 TF–TG feature parquet (for building edge_index).
    output_dir : Path
        Output directory where assembled tensors/vocabs will be written.
    chrom_id : str
        Identifier for chromosome or dataset partition (default 'all').
    force : bool, optional
        If True, overwrite existing checkpoint.
    """
    ensure_dir(output_dir)
    done_flag = output_dir / "dataset_assembly"
    if checkpoint_exists(done_flag) and not force:
        logging.info(f"[SKIP] Dataset assembly already completed at {output_dir}")
        return

    with StageTimer("Dataset Assembly"):
        # ---------------------------------------------------------
        # 1. Load tensorization outputs
        # ---------------------------------------------------------
        tensor_path = tensor_dir / "transformer_features.pt"
        tf_vocab_path = tensor_dir / "tf_vocab.json"
        tg_vocab_path = tensor_dir / "tg_vocab.json"

        X = torch.load(tensor_path)
        tf_vocab = json.load(open(tf_vocab_path))
        tg_vocab = json.load(open(tg_vocab_path))
        logging.info(f"Loaded tensor: {X.shape}, {len(tf_vocab)} TFs, {len(tg_vocab)} TGs")

        # ---------------------------------------------------------
        # 2. Build edge_index from Stage 3 TF–TG pairs
        # ---------------------------------------------------------
        df = pd.read_parquet(tf_tg_features_file)
        if not {"TF", "TG"}.issubset(df.columns):
            raise ValueError("TF–TG feature file missing TF/TG columns.")
        tf_indices = [tf_vocab[tf] for tf in df["TF"] if tf in tf_vocab]
        tg_indices = [tg_vocab[tg] for tg in df["TG"] if tg in tg_vocab]
        edge_index = torch.tensor([tf_indices, tg_indices], dtype=torch.long)
        logging.info(f"Edge index shape: {edge_index.shape}")

        # ---------------------------------------------------------
        # 3. Write dataset artifacts
        # ---------------------------------------------------------
        chrom_dir = output_dir / chrom_id
        ensure_dir(chrom_dir)
        torch.save(X, chrom_dir / f"tg_tensor_all_{chrom_id}.pt")
        torch.save(edge_index, chrom_dir / f"edge_index_{chrom_id}.pt")

        with open(output_dir / "tf_vocab.json", "w") as f:
            json.dump(tf_vocab, f, indent=2)
        with open(output_dir / "tg_vocab.json", "w") as f:
            json.dump(tg_vocab, f, indent=2)

        # ---------------------------------------------------------
        # 4. Checkpoint flag
        # ---------------------------------------------------------
        write_done_flag(done_flag)
        logging.info(f"[DONE] Dataset assembly complete → {output_dir}")
