#!/usr/bin/env python3
"""
test_pipeline.py

Comprehensive test suite for the GRN preprocessing pipeline:
- config.py
- io_utils.py
- qc_and_pseudobulk.py

Run with:
    pytest -v test_pipeline.py
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from anndata import AnnData
import scanpy as sc

from multiomic_transformer.pipeline.config import get_paths
from multiomic_transformer.pipeline.io_utils import (
    ensure_dir,
    write_parquet_safe,
    checkpoint_exists,
    write_done_flag,
    parquet_exists,
    read_parquet_safely,
)
from multiomic_transformer.pipeline.qc_and_pseudobulk import filter_and_qc, pseudo_bulk, run_qc_and_pseudobulk
from multiomic_transformer.pipeline.peak_gene_mapping import run_peak_gene_mapping
from multiomic_transformer.pipeline.tf_tg_feature_construction import run_tf_tg_feature_construction

# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def tmpdir():
    """Temporary directory for test outputs."""
    d = tempfile.mkdtemp(prefix="grn_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def mock_anndata_pair():
    """Generate small mock scRNA and scATAC AnnData datasets."""
    n_cells, n_genes, n_peaks = 100, 600, 800
    X_rna = np.random.poisson(1, (n_cells, n_genes))
    X_atac = np.random.poisson(1, (n_cells, n_peaks))
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)])
    var_rna = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    var_atac = pd.DataFrame(index=[f"peak{i}" for i in range(n_peaks)])
    adata_RNA = AnnData(X=sp.csr_matrix(X_rna), obs=obs, var=var_rna)
    adata_ATAC = AnnData(X=sp.csr_matrix(X_atac), obs=obs.copy(), var=var_atac)
    return adata_RNA, adata_ATAC


# ---------------------------------------------------------------------
# config.py tests
# ---------------------------------------------------------------------
def test_get_paths_returns_valid_paths(tmpdir):
    paths = get_paths(base_dir=tmpdir)
    assert "processed_data" in paths
    for p in paths.values():
        assert Path(p).exists()


# ---------------------------------------------------------------------
# io_utils.py tests
# ---------------------------------------------------------------------
def test_ensure_dir_and_write_parquet(tmpdir):
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = tmpdir / "test.parquet"
    ensure_dir(tmpdir)
    write_parquet_safe(df, path)
    assert path.exists()
    reloaded = read_parquet_safely(path)
    assert reloaded.equals(df)


def test_checkpoint_write_and_read(tmpdir):
    flag = tmpdir / "stage"
    write_done_flag(flag)
    assert checkpoint_exists(flag)


def test_parquet_exists(tmpdir):
    df = pd.DataFrame({"x": [1, 2]})
    path = tmpdir / "df.parquet"
    write_parquet_safe(df, path)
    assert parquet_exists(path)


# ---------------------------------------------------------------------
# qc_and_pseudobulk.py tests
# ---------------------------------------------------------------------
def test_filter_and_qc(mock_anndata_pair):
    adata_RNA, adata_ATAC = mock_anndata_pair
    adata_RNA_qc, adata_ATAC_qc = filter_and_qc(adata_RNA, adata_ATAC)
    # should reduce features and retain cells
    assert adata_RNA_qc.n_obs <= adata_RNA.n_obs
    assert adata_ATAC_qc.n_obs <= adata_ATAC.n_obs
    assert "X_pca" in adata_RNA_qc.obsm
    assert "X_pca" in adata_ATAC_qc.obsm


def test_pseudo_bulk(mock_anndata_pair):
    adata_RNA, adata_ATAC = mock_anndata_pair
    adata_RNA_qc, adata_ATAC_qc = filter_and_qc(adata_RNA, adata_ATAC)
    df_rna, df_atac = pseudo_bulk(adata_RNA_qc, adata_ATAC_qc)
    assert isinstance(df_rna, pd.DataFrame)
    assert isinstance(df_atac, pd.DataFrame)
    assert df_rna.shape[1] == df_atac.shape[1]  # same pseudobulks


def test_full_run_qc_and_pseudobulk(tmpdir, mock_anndata_pair):
    """Integration test: runs full pipeline end-to-end."""
    adata_RNA, adata_ATAC = mock_anndata_pair
    rna_path = tmpdir / "rna_test.h5ad"
    atac_path = tmpdir / "atac_test.h5ad"
    adata_RNA.write_h5ad(rna_path)
    adata_ATAC.write_h5ad(atac_path)

    run_qc_and_pseudobulk(
        rna_path=str(rna_path),
        atac_path=str(atac_path),
        organism="mm10",
        outdir=tmpdir,
        keep_intermediate=True,
    )

    expected_outputs = [
        "scRNA_processed.parquet",
        "scATAC_processed.parquet",
        "pseudobulk_scRNA.parquet",
        "pseudobulk_scATAC.parquet",
        "pseudobulk_metadata.parquet",
    ]
    for f in expected_outputs:
        assert (tmpdir / f).exists(), f"Missing expected output: {f}"

    # Checkpoint flags
    assert checkpoint_exists(tmpdir / ".qc")
    assert checkpoint_exists(tmpdir / ".pseudobulk")

# ------------------------------------------------------------------------------------
# Test: Peak–gene mapping distance and scoring
# ------------------------------------------------------------------------------------

def test_run_peak_gene_mapping_distance_computation(tmpdir):
    """
    Verify that run_peak_gene_mapping correctly computes distances and exponential weights.
    """
    # ------------------------------------------------------------------
    # Create mock peaks and TSS files
    # ------------------------------------------------------------------
    peaks_path = Path(tmpdir) / "mock_peaks.bed"
    tss_path = Path(tmpdir) / "mock_tss.bed"

    # 3 peaks, two on chr1 near geneA/B and one on chr2 near geneC
    peaks_content = """chr1\t1000\t1100
chr1\t5000\t5100
chr2\t9000\t9100
"""
    tss_content = """chr1\t1200\t1201\tGeneA
chr1\t7000\t7001\tGeneB
chr2\t9500\t9501\tGeneC
"""
    peaks_path.write_text(peaks_content)
    tss_path.write_text(tss_content)

    # ------------------------------------------------------------------
    # Run mapping
    # ------------------------------------------------------------------
    df = run_peak_gene_mapping(
        peaks_file=str(peaks_path),
        tss_bed_file=str(tss_path),
        organism="mm10",
        outdir=tmpdir,
        max_distance=10000,
        force_recompute=True,
    )

    # ------------------------------------------------------------------
    # Validate output schema
    # ------------------------------------------------------------------
    expected_cols = {"peak_id", "TG", "TSS_dist", "TSS_dist_score"}
    assert expected_cols.issubset(df.columns), f"Missing expected columns: {df.columns}"

    # ------------------------------------------------------------------
    # Check distance correctness
    # ------------------------------------------------------------------
    # chr1 peak1 (1000–1100) → geneA (1200) ⇒ |1100 - 1200| = 100
    d1 = df.loc[df["TG"] == "GeneA", "TSS_dist"].min()
    assert d1 == 100, f"Expected 100 bp distance, got {d1}"

    # chr1 peak2 (5000–5100) → geneB (7000) ⇒ |5100 - 7000| = 1900
    d2 = df.loc[df["TG"] == "GeneB", "TSS_dist"].min()
    assert d2 == 1900, f"Expected 1900 bp distance, got {d2}"

    # chr2 peak3 (9000–9100) → geneC (9500) ⇒ |9100 - 9500| = 400
    d3 = df.loc[df["TG"] == "GeneC", "TSS_dist"].min()
    assert d3 == 400, f"Expected 400 bp distance, got {d3}"

    # ------------------------------------------------------------------
    # Check exponential distance weighting
    # ------------------------------------------------------------------
    s1 = df.loc[df["TG"] == "GeneA", "TSS_dist_score"].max()
    s2 = df.loc[df["TG"] == "GeneB", "TSS_dist_score"].max()
    s3 = df.loc[df["TG"] == "GeneC", "TSS_dist_score"].max()

    # Closer distances should have higher scores
    assert s1 > s3 > s2 or s1 > s2 > s3, \
        f"Expected smaller distances to yield higher scores (got s1={s1}, s2={s2}, s3={s3})"

    # Scores should be between 0 and 1
    assert all((0 < v <= 1) for v in [s1, s2, s3]), "Scores must be between 0 and 1"

    # ------------------------------------------------------------------
    # Check that output Parquet was written
    # ------------------------------------------------------------------
    parquet_path = Path(tmpdir) / "peak_gene_distance.parquet"
    assert parquet_path.exists(), f"Missing expected output file: {parquet_path}"

# ---------------------------------------------------------------------
# Stage 3: TF–TG Feature Construction
# ---------------------------------------------------------------------
def test_run_tf_tg_feature_construction(tmp_path):
    """
    Validate Stage 3 TF–TG feature integration on synthetic data.

    This test verifies:
      • Correct merging of expression, regulatory potential, and distance data
      • Creation of expected feature columns
      • Proper .done checkpoint writing
      • Correlation values within [-1, 1]
      • No NaN or infinite values in key numerical fields
    """


    # --------------------------------------------------------------
    # 1. Create minimal synthetic inputs (guaranteed TF–TG overlap)
    # --------------------------------------------------------------
    # Expression for both TFs and TGs (ensure non-constant variance)
    expr_df = pd.DataFrame(
        {
            "S1": [0.1, 0.5, 0.3, 0.7],
            "S2": [0.2, 0.6, 0.4, 0.8],
            "S3": [0.3, 0.7, 0.5, 0.9],
        },
        index=["TF1", "TF2", "TG1", "TG2"]
    )
    pseudobulk_file = tmp_path / "pseudobulk_expr.parquet"
    expr_df.to_parquet(pseudobulk_file)


    # Ensure TF/TG names overlap with expression index
    tf_tg_reg = pd.DataFrame({
        "TF": ["TF1", "TF1", "TF2", "TF2"],
        "TG": ["TG1", "TG2", "TG1", "TG2"],   # matches expr_df.index
        "reg_potential": [0.2, 0.5, 0.3, 0.1],
        "motif_density": [3, 0, 5, 2],
    })
    reg_potential_file = tmp_path / "tf_tg_regulatory_potential.parquet"
    tf_tg_reg.to_parquet(reg_potential_file)

    peak_gene_links = pd.DataFrame({
        "peak_id": ["p1", "p2", "p3", "p4"],
        "TG": ["TG1", "TG1", "TG2", "TG2"],
        "TSS_dist": [1000, 50000, 10000, 8000]
    })
    peak_gene_links_file = tmp_path / "peak_gene_links.parquet"
    peak_gene_links.to_parquet(peak_gene_links_file)

    output_file = tmp_path / "tf_tg_features.parquet"

    # --------------------------------------------------------------
    # 2. Run feature construction
    # --------------------------------------------------------------
    run_tf_tg_feature_construction(
        pseudobulk_file=pseudobulk_file,
        reg_potential_file=reg_potential_file,
        peak_gene_links_file=peak_gene_links_file,
        output_file=output_file,
        force=True,
    )

    # --------------------------------------------------------------
    # 3. Verify output structure and checkpoint
    # --------------------------------------------------------------
    assert output_file.exists(), "TF–TG features parquet was not created"
    assert os.path.getsize(output_file) > 8, f"Parquet file too small, likely empty: {output_file}"

    df_out = pd.read_parquet(output_file)
    assert not df_out.empty, "TF–TG features DataFrame is empty"
    expected_cols = {
        "TF", "TG",
        "mean_tf_expr", "mean_tg_expr",
        "pearson_corr", "spearman_corr",
        "reg_potential", "motif_density",
        "expr_product", "log_reg_pot", "motif_present",
    }
    missing = expected_cols - set(df_out.columns)
    assert not missing, f"Missing expected columns: {missing}"
    assert df_out.shape[0] > 0, "Output TF–TG table is empty"

    done_marker = output_file.with_suffix(output_file.suffix + ".done")
    assert done_marker.exists(), "Stage 3 .done marker was not written"

    # --------------------------------------------------------------
    # 4. Sanity checks on numeric features
    # --------------------------------------------------------------
    numeric_cols = [
        "mean_tf_expr", "mean_tg_expr",
        "pearson_corr", "spearman_corr",
        "expr_product", "log_reg_pot",
    ]
    for col in numeric_cols:
        assert col in df_out.columns, f"{col} missing from output"
        assert not df_out[col].isna().any(), f"NaN values detected in {col}"
        assert np.isfinite(df_out[col]).all(), f"Infinite values detected in {col}"

    # Correlation sanity
    assert (df_out["pearson_corr"].between(-1, 1, inclusive="both")).all(), \
        "Pearson correlations out of range [-1, 1]"
    assert (df_out["spearman_corr"].between(-1, 1, inclusive="both")).all(), \
        "Spearman correlations out of range [-1, 1]"

    # Motif presence consistency
    assert set(df_out["motif_present"].unique()).issubset({0, 1}), \
        "motif_present column contains invalid values"

    # Expression product consistency
    expected_expr_product = (
        df_out["mean_tf_expr"] * df_out["mean_tg_expr"]
    ).round(6)
    assert np.allclose(
        df_out["expr_product"].round(6), expected_expr_product
    ), "expr_product column is inconsistent with mean_tf_expr × mean_tg_expr"

# ---------------------------------------------------------------------
# Stage 3 → Stage 4 Integration Smoke Test
# ---------------------------------------------------------------------
def test_tf_tg_features_feed_tensorization(tmp_path):
    """
    Smoke test verifying that the TF–TG feature output from Stage 3
    is schema-compatible with the tensorization step.

    This ensures:
      • Expected columns are numeric and finite
      • TF/TG vocabularies can be built
      • Data can be converted into float32 tensors without error
    """
    import numpy as np
    import pandas as pd
    import torch
    from multiomic_transformer.pipeline.tf_tg_feature_construction import run_tf_tg_feature_construction

    # --------------------------------------------------------------
    # 1. Create small synthetic Stage 3 input data
    # --------------------------------------------------------------
    expr_df = pd.DataFrame(
        {
            "S1": [0.1, 0.5, 0.3, 0.7],
            "S2": [0.2, 0.6, 0.4, 0.8],
            "S3": [0.3, 0.7, 0.5, 0.9],
        },
        index=["TF1", "TF2", "TG1", "TG2"]
    )
    pseudobulk_file = tmp_path / "pseudobulk_expr.parquet"
    expr_df.to_parquet(pseudobulk_file)

    tf_tg_reg = pd.DataFrame({
        "TF": ["TF1", "TF1", "TF2", "TF2"],
        "TG": ["TG1", "TG2", "TG1", "TG2"],
        "reg_potential": [0.1, 0.4, 0.2, 0.3],
        "motif_density": [1, 0, 4, 2]
    })
    reg_potential_file = tmp_path / "tf_tg_regulatory_potential.parquet"
    tf_tg_reg.to_parquet(reg_potential_file)

    peak_gene_links = pd.DataFrame({
        "peak_id": ["p1", "p2", "p3", "p4"],
        "TG": ["TG1", "TG1", "TG2", "TG2"],
        "TSS_dist": [1000, 4000, 500, 20000]
    })
    peak_gene_links_file = tmp_path / "peak_gene_links.parquet"
    peak_gene_links.to_parquet(peak_gene_links_file)

    output_file = tmp_path / "tf_tg_features.parquet"
    
    # --------------------------------------------------------------
    # 2. Generate TF–TG features using Stage 3
    # --------------------------------------------------------------
    run_tf_tg_feature_construction(
        pseudobulk_file=pseudobulk_file,
        reg_potential_file=reg_potential_file,
        peak_gene_links_file=peak_gene_links_file,
        output_file=output_file,
        force=True,
    )

    df = pd.read_parquet(output_file)
    assert not df.empty, "Stage 3 output is empty"

    # --------------------------------------------------------------
    # 3. Build vocabularies (mock tensorization step)
    # --------------------------------------------------------------
    tf_vocab = {tf: i for i, tf in enumerate(sorted(df["TF"].unique()))}
    tg_vocab = {tg: i for i, tg in enumerate(sorted(df["TG"].unique()))}
    assert len(tf_vocab) > 0 and len(tg_vocab) > 0, "Vocabularies are empty"

    # Map to indices
    df["TF_idx"] = df["TF"].map(tf_vocab)
    df["TG_idx"] = df["TG"].map(tg_vocab)

    # --------------------------------------------------------------
    # 4. Prepare feature tensor
    # --------------------------------------------------------------
    feature_cols = [
        "mean_tf_expr", "mean_tg_expr",
        "pearson_corr", "spearman_corr",
        "reg_potential", "motif_density",
        "expr_product", "log_reg_pot", "motif_present",
    ]
    X = df[feature_cols].astype(np.float32).values
    edge_index = torch.tensor(df[["TF_idx", "TG_idx"]].values.T, dtype=torch.long)
    edge_attr = torch.tensor(X, dtype=torch.float32)

    # --------------------------------------------------------------
    # 5. Validation
    # --------------------------------------------------------------
    assert edge_index.shape[0] == 2, "edge_index must have shape [2, E]"
    assert edge_index.shape[1] == edge_attr.shape[0], \
        "edge_index and edge_attr row counts mismatch"
    assert torch.isfinite(edge_attr).all(), "Non-finite values in edge_attr"

    # Sample tensor statistics for sanity
    means = edge_attr.mean(dim=0)
    assert not torch.isnan(means).any(), "Mean of feature tensor contains NaN"
    assert (means >= 0).any(), "All feature means are negative — unexpected"

