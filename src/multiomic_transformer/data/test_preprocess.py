# test_preprocess.py
# pytest -q
# Run only fast tests:        pytest -q -k "not slow"
# Include optional/slow ones: pytest -q

import os
import io
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import scipy.sparse as sp

# Try the package import first (repo layout). If that fails, allow
# a direct path import via environment variable PREPROCESS_PATH.
try:
    from multiomic_transformer.data.preprocess import (
        build_peak_locs_from_index,
        select_pkn_edges_from_df,
        compute_minmax_expr_mean,
        merge_tf_tg_attributes_with_combinations,
        make_peak_to_window_map,
        align_to_vocab,
        build_motif_mask,
        precompute_input_tensors,
        calculate_peak_to_tg_distance_score,
        merge_tf_tg_data_with_pkn,
        process_or_load_rna_atac_data,
        pseudo_bulk,
        filter_and_qc,
    )
except Exception:
    import importlib.util
    P = os.environ.get("PREPROCESS_PATH")
    if not P:
        raise
    spec = importlib.util.spec_from_file_location("preprocess_mod", P)
    preprocess_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess_mod)
    build_peak_locs_from_index = preprocess_mod.build_peak_locs_from_index
    select_pkn_edges_from_df = preprocess_mod.select_pkn_edges_from_df
    compute_minmax_expr_mean = preprocess_mod.compute_minmax_expr_mean
    merge_tf_tg_attributes_with_combinations = preprocess_mod.merge_tf_tg_attributes_with_combinations
    make_peak_to_window_map = preprocess_mod.make_peak_to_window_map
    align_to_vocab = preprocess_mod.align_to_vocab
    build_motif_mask = preprocess_mod.build_motif_mask
    precompute_input_tensors = preprocess_mod.precompute_input_tensors
    calculate_peak_to_tg_distance_score = preprocess_mod.calculate_peak_to_tg_distance_score
    merge_tf_tg_data_with_pkn = preprocess_mod.merge_tf_tg_data_with_pkn
    process_or_load_rna_atac_data = preprocess_mod.process_or_load_rna_atac_data
    pseudo_bulk = preprocess_mod.pseudo_bulk
    filter_and_qc = preprocess_mod.filter_and_qc


# --------- Core, fast, no external tools ---------

def test_build_peak_locs_from_index_parses_and_filters():
    idx = pd.Index(["1:100-200", "chrX:5-10", "chrM:1-2", "bad"])
    df = build_peak_locs_from_index(idx)
    # Should drop malformed "bad" and non-standard "chrM"
    assert set(df.columns) == {"chrom", "start", "end", "peak_id"}
    assert "bad" not in df["peak_id"].values
    assert not (df["chrom"] == "chrM").any()
    # Coordinates are ints and start<=end
    assert pd.api.types.is_integer_dtype(df["start"])
    assert pd.api.types.is_integer_dtype(df["end"])
    assert (df["end"] >= df["start"]).all()


def test_select_pkn_edges_from_df_undirected():
    df = pd.DataFrame({"TF": ["A", "B", "Z"], "TG": ["X", "Z", "B"]})
    # PKN has only (B,Z); undirected means (Z,B) also matches.
    pkn = {("B", "Z")}
    in_df, out_df = select_pkn_edges_from_df(df.copy(), pkn)
    assert set(map(tuple, in_df[["TF", "TG"]].values)) == {("B", "Z"), ("Z", "B")}
    assert set(map(tuple, out_df[["TF", "TG"]].values)) == {("A", "X")}


def test_compute_minmax_expr_mean_shapes_and_values():
    # Rows = genes, cols = cells
    tf_df = pd.DataFrame([[0.0, 1.0, 2.0], [10.0, 10.0, 10.0]], index=["TF1", "TF2"])
    tg_df = pd.DataFrame([[2.0, 2.0, 2.0], [0.0, 5.0, 10.0]], index=["TG1", "TG2"])
    m_tf, m_tg = compute_minmax_expr_mean(tf_df, tg_df)
    # TF1 minmax across columns → [0, 0.5, 1.0] mean = 0.5; TF2 is all equal → [0,0,0]? scaler puts 0s (avoid div by 0)
    tf_means = dict(zip(m_tf["TF"], m_tf["mean_tf_expr"]))
    tg_means = dict(zip(m_tg["TG"], m_tg["mean_tg_expr"]))
    assert pytest.approx(tf_means["TF1"], rel=1e-6) == 0.5
    assert tf_means["TF2"] == 0.0
    # TG1 constant row
    assert tg_means["TG1"] == 0.0
    # TG2 scales to [0, 0.5, 1.0] → mean = 0.5
    assert pytest.approx(tg_means["TG2"], rel=1e-6) == 0.5


def test_merge_tf_tg_attributes_with_combinations(tmp_path: Path):
    # combos
    tf_tg_df = pd.DataFrame({"TF": ["A", "A"], "TG": ["X", "Y"]})
    # reg potential
    tf_tg_reg_pot = pd.DataFrame({
        "TF": ["A", "A"],
        "TG": ["X", "Y"],
        "reg_potential": [2.0, 0.0],
        "motif_density": [3.0, 0.0],
    })
    # means
    m_tf = pd.DataFrame({"TF": ["A"], "mean_tf_expr": [0.25]})
    m_tg = pd.DataFrame({"TG": ["X", "Y"], "mean_tg_expr": [0.5, 0.1]})
    out_path = tmp_path / "combos.parquet"
    out = merge_tf_tg_attributes_with_combinations(tf_tg_df, tf_tg_reg_pot, m_tf, m_tg, out_path)
    assert out_path.exists()
    # Check engineered features
    row_x = out.loc[(out["TF"] == "A") & (out["TG"] == "X")].iloc[0]
    assert pytest.approx(row_x["expr_product"], rel=1e-6) == 0.25 * 0.5
    assert pytest.approx(row_x["log_reg_pot"], rel=1e-6) == np.log1p(2.0)
    assert row_x["motif_present"] == 1
    row_y = out.loc[(out["TF"] == "A") & (out["TG"] == "Y")].iloc[0]
    assert row_y["motif_present"] == 0


@pytest.mark.skipif(pytest.importorskip("pybedtools", reason="pybedtools not installed") is None, reason="pybedtools missing")
def test_make_peak_to_window_map_basic():
    # Two peaks, two windows with partial overlaps
    peaks_bed = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [100, 400],
        "end":   [200, 550],
        "peak_id": ["chr1:100-200", "chr1:400-550"],
    })
    windows_bed = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [0, 300],
        "end":   [300, 600],
        "win_idx": [0, 1],
    })
    mapping = make_peak_to_window_map(peaks_bed, windows_bed)
    # peak1 overlaps window0 more; peak2 overlaps window1 more
    assert mapping["chr1:100-200"] == 0
    assert mapping["chr1:400-550"] == 1


def test_align_to_vocab_happy_path_and_errors():
    names = ["G1", "G2", "G3"]
    vocab = {"G1": 10, "G3": 12}
    tensor_all = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float32)
    aligned, kept_names, kept_ids = align_to_vocab(names, vocab, tensor_all, label="TG")
    assert aligned.shape == (2, 2)
    assert kept_names == ["G1", "G3"]
    assert kept_ids == [10, 12]
    # No matches → error
    with pytest.raises(ValueError):
        align_to_vocab(["X"], vocab, torch.randn(1, 2), label="TG")


def test_build_motif_mask_small():
    tf_names = ["TF1", "TF2"]
    tg_names = ["G1", "G2", "G3"]
    # sliding_window_df needs columns: ["TF","peak_id","sliding_window_score"]
    sliding = pd.DataFrame({
        "TF": ["TF1", "TF1", "TF2"],
        "peak_id": ["p1", "p2", "p1"],
        "sliding_window_score": [1.0, 2.0, 3.0],
    })
    # genes_near_peaks needs ["peak_id","target_id"]
    gnp = pd.DataFrame({
        "peak_id": ["p1", "p2", "p1"],
        "target_id": ["G1", "G1", "G2"],
    })
    mask = build_motif_mask(tf_names, tg_names, sliding, gnp)
    assert mask.shape == (len(tg_names), len(tf_names))
    # For TG=G1, TF1 has p1=1.0 and p2=2.0 → max = 2.0
    i_g1 = tg_names.index("G1")
    j_tf1 = tf_names.index("TF1")
    assert pytest.approx(mask[i_g1, j_tf1], rel=1e-6) == 2.0
    # For TG=G1, TF2 has p1=3.0
    j_tf2 = tf_names.index("TF2")
    assert pytest.approx(mask[i_g1, j_tf2], rel=1e-6) == 3.0


def test_precompute_input_tensors_shapes():
    # TF expression [num_TF, num_cells]
    tf_expr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    # TG scaled [num_TG_chr, num_cells]
    tg_scaled = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32)
    # RE pseudobulk: rows=peaks, cols=cells
    re_df = pd.DataFrame([[1, 0], [0, 1], [1, 1]], index=["p1", "p2", "p3"], columns=["c1", "c2"])
    # window map: p1->0, p2->1, p3->0 ; windows: 2 rows
    window_map = {"p1": 0, "p2": 1, "p3": 0}
    windows = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [0, 100], "end": [100, 200]})
    tf_t, tg_t, atac_t = precompute_input_tensors(
        output_dir=".", genome_wide_tf_expression=tf_expr, TG_scaled=tg_scaled,
        total_RE_pseudobulk_chr=re_df, window_map=window_map, windows=windows
    )
    assert tf_t.shape == (2, 2)
    assert tg_t.shape == (3, 2)
    assert atac_t.shape == (2, 2)  # 2 windows × 2 cells


# --------- Functions needing small on-disk CSV/Parquet but still fast ---------

def test_merge_tf_tg_data_with_pkn_balances_and_metadata(tmp_path: Path):
    # Small TF-TG candidate set for a single TF with positives & negatives
    df = pd.DataFrame({"TF": ["A", "A", "A", "B"], "TG": ["X", "Y", "Z", "Z"]})
    # Create PKN CSVs with metadata
    cols = ["TF", "TG", "string_experimental_score", "trrust_regulation", "kegg_n_pathways"]
    string = pd.DataFrame([["A","X", 900, None, None], ["Z","B", 800, None, None]], columns=cols)
    trrust = pd.DataFrame([["A","Y", None, "Activation", None]], columns=cols)
    kegg   = pd.DataFrame([["Q","R", None, None, 3]], columns=cols)

    string_csv = tmp_path / "string.csv"
    trrust_csv = tmp_path / "trrust.csv"
    kegg_csv   = tmp_path / "kegg.csv"
    string.to_csv(string_csv, index=False)
    trrust.to_csv(trrust_csv, index=False)
    kegg.to_csv(kegg_csv, index=False)

    out_with_meta, out_plain = merge_tf_tg_data_with_pkn(
        df, string_csv, trrust_csv, kegg_csv,
        upscale_percent=1.0, seed=42, add_pkn_scores=True
    )
    # Expect positives for A-X (direct), A-Y (direct), and B-Z via undirected (Z,B)
    assert set(map(tuple, out_plain[["TF","TG"]].drop_duplicates().values)).issubset({("A","X"), ("A","Y"), ("B","Z")})
    # Metadata should be present where available
    assert any(c.startswith("STRING_") for c in out_with_meta.columns)
    assert any(c.startswith("TRRUST_") for c in out_with_meta.columns)
    # n_sources flags present
    assert {"in_STRING","in_TRRUST","in_KEGG","n_sources"}.issubset(out_with_meta.columns)


def test_process_or_load_rna_atac_data_load_existing_and_return_pseudobulk(tmp_path: Path, monkeypatch):
    # Create minimal processed parquet files that the loader can read
    processed_rna = pd.DataFrame([[1,2],[3,4]], index=["G1","G2"], columns=["C1","C2"])
    processed_atac = pd.DataFrame([[5,6],[7,8]], index=["chr1:1-2","chr1:3-4"], columns=["C1","C2"])
    (tmp_path / "sample").mkdir()
    rna_p = tmp_path / "sample" / "scRNA_seq_processed.parquet"
    atac_p = tmp_path / "sample" / "scATAC_seq_processed.parquet"
    processed_rna.to_parquet(rna_p, engine="pyarrow")
    processed_atac.to_parquet(atac_p, engine="pyarrow")
    # Provide pseudobulk TSVs to avoid calling pseudo_bulk
    TG_p = tmp_path / "sample" / "TG_pseudobulk.tsv"
    RE_p = tmp_path / "sample" / "RE_pseudobulk.tsv"
    processed_rna.to_csv(TG_p, sep="\t")
    processed_atac.to_csv(RE_p, sep="\t")

    rna_df, atac_df, TG_df, RE_df = process_or_load_rna_atac_data(tmp_path / "sample")
    assert rna_df.equals(processed_rna)
    assert atac_df.equals(processed_atac)
    assert not TG_df.empty and not RE_df.empty


# --------- Optional / slow / tool-dependent tests ---------

@pytest.mark.skipif(pytest.importorskip("pybedtools", reason="pybedtools not installed") is None, reason="pybedtools missing")
def test_calculate_peak_to_tg_distance_score_minimal(tmp_path: Path):
    # Minimal peaks and TSSs
    peaks = pd.DataFrame({
        "chrom": ["chr1","chr1"],
        "start": [100, 1000],
        "end":   [200, 1100],
        "peak_id": ["p1","p2"],
    })
    tss = pd.DataFrame({
        "chrom": ["chr1","chr1"],
        "start": [150, 1050],
        "end":   [151, 1051],
        "name":  ["G1","G2"],
    })
    # Write BEDs
    peak_bed = tmp_path / "peaks.bed"
    tss_bed  = tmp_path / "tss.bed"
    peaks[["chrom","start","end","peak_id"]].to_csv(peak_bed, sep="\t", header=False, index=False)
    tss[["chrom","start","end","name"]].to_csv(tss_bed, sep="\t", header=False, index=False)

    out_parquet = tmp_path / "pg.parquet"
    # Function requires dataframes too (for type checks); just pass the same
    df = calculate_peak_to_tg_distance_score(
        peak_bed, tss_bed, out_parquet,
        mesc_atac_peak_loc_df=peaks.copy(),
        gene_tss_df=tss.copy(),
        max_peak_distance=1000, distance_factor_scale=25000
    )
    assert out_parquet.exists()
    assert {"peak_id","target_id","TSS_dist","TSS_dist_score"}.issubset(df.columns)


@pytest.mark.slow
@pytest.mark.skipif(
    pytest.importorskip("scanpy", reason="scanpy not installed") is None or
    pytest.importorskip("leidenalg", reason="leidenalg not installed") is None,
    reason="scanpy/leidenalg missing",
)
def test_pseudo_bulk_minimal_runs():
    import scanpy as sc
    from anndata import AnnData
    # Minimal matrices: 6 cells, 4 genes; ATAC with 3 peaks
    X_rna = np.array([[1,2,3,4,5,6],
                      [2,3,4,5,6,7],
                      [0,0,0,1,1,1],
                      [5,4,3,2,1,0]], dtype=float).T  # cells x genes
    X_atac = np.array([[0,1,0,1,0,1],
                       [1,0,1,0,1,0],
                       [2,2,2,2,2,2]], dtype=float).T  # cells x peaks
    ad_rna = AnnData(X=X_rna)
    ad_rna.var_names = ["G1","G2","G3","G4"]
    ad_rna.obs_names = [f"C{i}" for i in range(6)]
    ad_atac = AnnData(X=X_atac)
    ad_atac.var_names = ["chr1:1-10","chr1:11-20","chr1:21-30"]
    ad_atac.obs_names = ad_rna.obs_names.copy()

    tg_df, re_df = pseudo_bulk(ad_rna, ad_atac, use_single=True, neighbors_k=3, resolution=0.5, aggregate="mean", pca_components=5)
    assert tg_df.shape[0] == 4
    assert re_df.shape[0] == 3
    assert tg_df.shape[1] >= 1 and re_df.shape[1] >= 1
