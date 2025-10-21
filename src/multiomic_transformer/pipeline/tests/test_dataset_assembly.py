#!/usr/bin/env python3
"""
test_dataset_assembly.py
Unit tests for Stage 5 – Dataset assembly.
"""

import json
import torch
import pandas as pd
from pathlib import Path
from multiomic_transformer.pipeline.dataset_assembly import run_dataset_assembly
from multiomic_transformer.pipeline.io_utils import checkpoint_exists, write_done_flag


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def make_mock_stage4_outputs(tmp_path: Path):
    """Create minimal tensorization outputs (Stage 4 mock)."""
    tensor_dir = tmp_path / "tensors"
    tensor_dir.mkdir()

    # Small 4×5 feature matrix
    X = torch.rand((4, 5))
    torch.save(X, tensor_dir / "transformer_features.pt")

    tf_vocab = {"TF1": 0, "TF2": 1}
    tg_vocab = {"TG1": 0, "TG2": 1}
    json.dump(tf_vocab, open(tensor_dir / "tf_vocab.json", "w"))
    json.dump(tg_vocab, open(tensor_dir / "tg_vocab.json", "w"))

    return tensor_dir


def make_mock_stage3_features(tmp_path: Path):
    """Create minimal TF–TG feature parquet (Stage 3 mock)."""
    df = pd.DataFrame({
        "TF": ["TF1", "TF1", "TF2", "TF2"],
        "TG": ["TG1", "TG2", "TG1", "TG2"],
        "mean_tf_expr": [0.5, 0.6, 0.7, 0.8],
        "mean_tg_expr": [0.9, 0.8, 0.7, 0.6],
        "pearson_corr": [0.1, 0.2, 0.3, 0.4],
        "spearman_corr": [0.5, 0.6, 0.7, 0.8],
        "motif_density": [1, 2, 3, 4],
    })
    parquet_path = tmp_path / "tf_tg_features.parquet"
    df.to_parquet(parquet_path)
    return parquet_path


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_run_dataset_assembly_creates_expected_outputs(tmp_path):
    """Integration test verifying dataset assembly output files and tensors."""
    tensor_dir = make_mock_stage4_outputs(tmp_path)
    tf_tg_file = make_mock_stage3_features(tmp_path)
    out_dir = tmp_path / "assembled"

    run_dataset_assembly(
        tensor_dir=tensor_dir,
        tf_tg_features_file=tf_tg_file,
        output_dir=out_dir,
        chrom_id="chrAll",
        force=True,
    )

    # Checkpoint flag
    assert checkpoint_exists(out_dir / "dataset_assembly"), "Missing dataset_assembly.done flag"

    # Check vocabularies
    tf_vocab = json.load(open(out_dir / "tf_vocab.json"))
    tg_vocab = json.load(open(out_dir / "tg_vocab.json"))
    assert len(tf_vocab) == 2 and len(tg_vocab) == 2
    assert "TF1" in tf_vocab and "TG1" in tg_vocab

    # Check tensors
    chrom_dir = out_dir / "chrAll"
    tg_tensor_path = chrom_dir / "tg_tensor_all_chrAll.pt"
    edge_index_path = chrom_dir / "edge_index_chrAll.pt"

    assert tg_tensor_path.exists(), "Missing tg_tensor_all file"
    assert edge_index_path.exists(), "Missing edge_index file"

    X = torch.load(tg_tensor_path)
    edges = torch.load(edge_index_path)

    assert isinstance(X, torch.Tensor)
    assert isinstance(edges, torch.Tensor)
    assert X.shape[1] == 5  # same as mock Stage 4 features
    assert edges.shape[0] == 2
    assert edges.shape[1] == 4  # 4 TF–TG pairs

    # Verify index ranges
    assert edges.max().item() < len(tg_vocab)
    assert edges.min().item() >= 0


def test_checkpoint_reuse(tmp_path):
    """Ensure re-running dataset assembly respects checkpoint."""
    tensor_dir = make_mock_stage4_outputs(tmp_path)
    tf_tg_file = make_mock_stage3_features(tmp_path)
    out_dir = tmp_path / "assembled"
    flag = out_dir / "dataset_assembly"

    # Manually write checkpoint
    out_dir.mkdir()
    write_done_flag(flag)

    # Should skip since checkpoint exists
    run_dataset_assembly(tensor_dir, tf_tg_file, out_dir, force=False)
    # The checkpoint file should still exist and be non-empty
    assert checkpoint_exists(flag)

def test_dataset_stage5_minimal(tmp_path):
    from multiomic_transformer.pipeline.dataset_assembly import run_dataset_assembly

    tensor_dir = make_mock_stage4_outputs(tmp_path)
    tf_tg_file = make_mock_stage3_features(tmp_path)
    out_dir = tmp_path / "assembled"
    run_dataset_assembly(tensor_dir, tf_tg_file, out_dir, chrom_id="chrAll", force=True)

    from multiomic_transformer.datasets.dataset import MultiomicTransformerDataset
    ds = MultiomicTransformerDataset(out_dir, chrom_id="chrAll")
    assert hasattr(ds, "tg_tensor_all")
    assert ds.tg_tensor_all.shape[0] > 0
    assert hasattr(ds, "edge_index")
    assert ds.edge_index.shape[1] == 4