#!/usr/bin/env python3
"""
test_export_gat_data.py
Unit tests for Stage 6 – Export GAT data (TF–TG edge feature preparation).
"""

import torch
import pandas as pd
from pathlib import Path
from multiomic_transformer.pipeline.export_gat_data import run_export_gat_data
from multiomic_transformer.pipeline.io_utils import checkpoint_exists, write_done_flag


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def make_mock_tf_tg_features(tmp_path: Path) -> Path:
    """Create a minimal TF–TG feature parquet file."""
    df = pd.DataFrame({
        "TF": ["TF1", "TF1", "TF2", "TF2"],
        "TG": ["TG1", "TG2", "TG1", "TG2"],
        "mean_tf_expr": [0.5, 0.6, 0.7, 0.8],
        "mean_tg_expr": [0.9, 0.8, 0.7, 0.6],
        "pearson_corr": [0.1, 0.2, 0.3, 0.4],
        "spearman_corr": [0.5, 0.6, 0.7, 0.8],
        "motif_density": [1, 0, 3, 4],
        "reg_potential": [0.2, 0.5, 0.3, 0.1],
        "expr_product": [0.45, 0.48, 0.49, 0.48],
        "log_reg_pot": [0.18, 0.41, 0.26, 0.09],
        "motif_present": [1, 0, 1, 1],
    })
    out = tmp_path / "tf_tg_features.parquet"
    df.to_parquet(out)
    return out


def make_mock_pkn_csv(tmp_path: Path, name: str, pos_pairs=None) -> Path:
    """Make a small mock prior-knowledge network CSV file."""
    if pos_pairs is None:
        pos_pairs = [("TF1", "TG1"), ("TF2", "TG2")]
    pkn_df = pd.DataFrame(pos_pairs, columns=["protein1", "protein2"])
    out = tmp_path / f"{name}.csv"
    pkn_df.to_csv(out, index=False)
    return out


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_run_export_gat_data_creates_expected_outputs(tmp_path):
    """Integration test for Stage 6 export."""
    tf_tg_feature_file = make_mock_tf_tg_features(tmp_path)
    string_pkn_file = make_mock_pkn_csv(tmp_path, "string")
    trrust_pkn_file = make_mock_pkn_csv(tmp_path, "trrust")
    kegg_pkn_file = make_mock_pkn_csv(tmp_path, "kegg")

    out_dir = tmp_path / "gat_output"
    run_export_gat_data(
        tf_tg_feature_file=tf_tg_feature_file,
        string_pkn_file=string_pkn_file,
        trrust_pkn_file=trrust_pkn_file,
        kegg_pkn_file=kegg_pkn_file,
        output_dir=out_dir,
        force=True,
    )

    # Verify outputs
    assert checkpoint_exists(out_dir / "gat_export"), "Missing .done checkpoint"
    csv_path = out_dir / "gat_edges.csv"
    pt_path = out_dir / "gat_features.pt"
    assert csv_path.exists(), "Missing CSV output"
    assert pt_path.exists(), "Missing PT output"

    # Check data integrity
    df = pd.read_csv(csv_path)
    assert "label" in df.columns
    data = torch.load(pt_path)
    X, y = data["X"], data["y"]
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape[0] == len(df)
    assert y.shape[0] == len(df)
    assert (y.unique() <= 1).all(), "Labels should be binary (0 or 1)"
    assert len(data["features"]) > 0


def test_checkpoint_reuse(tmp_path):
    """Ensure the export step respects existing checkpoints."""
    tf_tg_feature_file = make_mock_tf_tg_features(tmp_path)
    string_pkn_file = make_mock_pkn_csv(tmp_path, "string")
    trrust_pkn_file = make_mock_pkn_csv(tmp_path, "trrust")
    kegg_pkn_file = make_mock_pkn_csv(tmp_path, "kegg")

    out_dir = tmp_path / "gat_output"
    flag = out_dir / "gat_export"
    out_dir.mkdir()
    write_done_flag(flag)

    # Should skip because checkpoint exists
    run_export_gat_data(
        tf_tg_feature_file=tf_tg_feature_file,
        string_pkn_file=string_pkn_file,
        trrust_pkn_file=trrust_pkn_file,
        kegg_pkn_file=kegg_pkn_file,
        output_dir=out_dir,
        force=False,
    )
    assert checkpoint_exists(flag)
