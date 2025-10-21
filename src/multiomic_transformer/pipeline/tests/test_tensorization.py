#!/usr/bin/env python3
"""
Unit and integration tests for Stage 4 — Tensorization.
Validates conversion of TF–TG features into transformer-ready tensors,
vocabularies, and checkpoint flags.
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from multiomic_transformer.pipeline.tensorization import (
    run_tensorization,
    checkpoint_exists,
    load_tensorization_outputs,
)


def make_mock_tf_tg_features(tmpdir: Path) -> Path:
    """Create a minimal synthetic TF–TG features Parquet file."""
    df = pd.DataFrame({
        "TF": ["TF1", "TF1", "TF2", "TF2"],
        "TG": ["TG1", "TG2", "TG1", "TG2"],
        "pearson_corr": [0.5, -0.1, 0.3, 0.2],
        "spearman_corr": [0.6, -0.2, 0.4, 0.1],
        "reg_potential": [0.2, 0.8, 0.5, 0.1],
        "motif_density": [3, 0, 5, 2],
        "neg_log_tss_dist": [1.5, 0.7, 2.0, 1.1],
    })
    numeric_casts = [
    "reg_potential", "expr_product", "log_reg_pot", "neg_log_tss_dist",
    "mean_tf_expr", "mean_tg_expr", "pearson_corr", "spearman_corr", "motif_density"
    ]
    for col in numeric_casts:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        
    parquet_path = tmpdir / "tf_tg_features.parquet"
    df.to_parquet(parquet_path)
    return parquet_path


# ----------------------------------------------------------------------
# Core test: run_tensorization end-to-end
# ----------------------------------------------------------------------
def test_run_tensorization_creates_expected_outputs(tmp_path):
    tf_tg_file = make_mock_tf_tg_features(tmp_path)
    out_dir = tmp_path / "tensor_output"

    run_tensorization(tf_tg_features_file=tf_tg_file, output_dir=out_dir, force=True)

    # --- checkpoint ---
    assert checkpoint_exists(out_dir / "tensorization"), "Missing tensorization.done flag"

    # --- vocabularies ---
    tf_vocab = json.load(open(out_dir / "tf_vocab.json"))
    tg_vocab = json.load(open(out_dir / "tg_vocab.json"))
    assert len(tf_vocab) == 2 and len(tg_vocab) == 2
    assert "TF1" in tf_vocab and "TG1" in tg_vocab

    # --- tensor output ---
    tensor_path = out_dir / "transformer_features.pt"
    assert tensor_path.exists(), "Missing feature tensor"
    X = torch.load(tensor_path)
    assert isinstance(X, torch.Tensor)
    assert X.shape[0] == 4  # 4 TF–TG pairs
    assert X.shape[1] == 5  # 5 numeric columns

    # --- scaling sanity check (mean ~ 0, std ~ 1) ---
    arr = X.numpy()
    np.testing.assert_allclose(arr.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(arr.std(axis=0), 1, atol=1e-5)


# ----------------------------------------------------------------------
# Smoke test: re-load utility
# ----------------------------------------------------------------------
def test_load_tensorization_outputs(tmp_path):
    tf_tg_file = make_mock_tf_tg_features(tmp_path)
    out_dir = tmp_path / "tensor_output"

    run_tensorization(tf_tg_features_file=tf_tg_file, output_dir=out_dir, force=True)

    X, tf_vocab, tg_vocab = load_tensorization_outputs(out_dir)

    assert isinstance(X, torch.Tensor)
    assert isinstance(tf_vocab, dict)
    assert isinstance(tg_vocab, dict)
    assert X.shape[0] > 0
    assert "TF1" in tf_vocab and "TG1" in tg_vocab
