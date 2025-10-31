# test_preprocess.py
# pytest -q
# Run only fast tests:        pytest -q -k "not slow"
# Include optional/slow ones: pytest -q
# To run: poetry run pytest -v src/multiomic_transformer/data/test_preprocess.py > outputs/pytest_preprocessing.log

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import scipy.sparse as sp

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
        create_tf_tg_combination_files,
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
    create_tf_tg_combination_files = getattr(preprocess_mod, "create_tf_tg_combination_files", None)


# ---------- small helpers ----------
def _all_upper_no_version(strings):
    """every entry uppercase and without trailing dot-version (e.g., .1)"""
    for s in strings:
        if s != s.upper():
            return False
        if isinstance(s, str) and any(ch.isdigit() for ch in s.split(".")[-1]):
            parts = s.split(".")
            if len(parts) > 1 and parts[-1].isdigit():
                return False
    return True


# --------- Core, fast, no external tools ---------

def test_build_peak_locs_from_index_parses_and_filters():
    idx = pd.Index(["1:100-200", "chrX:5-10", "chrM:1-2", "bad"])
    df = build_peak_locs_from_index(idx)
    assert set(df.columns) == {"chrom", "start", "end", "peak_id"}
    assert "bad" not in df["peak_id"].tolist()
    # Accept both with and without 'chr' prefix
    assert df["peak_id"].str.match(r"^(chr)?[^:]+:\d+-\d+$").all()


def test_select_pkn_edges_from_df_undirected():
    df = pd.DataFrame({"TF": ["A", "B", "Z"], "TG": ["X", "Z", "B"]})
    # PKN has only (B,Z); undirected means (Z,B) also matches.
    pkn = {("B", "Z")}
    in_df, out_df = select_pkn_edges_from_df(df.copy(), pkn)
    assert set(map(tuple, in_df[["TF", "TG"]].values)) == {("B", "Z"), ("Z", "B")}
    assert set(map(tuple, out_df[["TF", "TG"]].values)) == {("A", "X")}


def test_compute_minmax_expr_mean_shapes_and_values():
    # Rows = genes, cols = cells
    tf_df = pd.DataFrame(
        [[0.0, 1.0, 2.0],   # TF1 → scaled [0, 0.5, 1.0] → mean 0.5
         [10.0, 10.0, 10.0]],  # TF2 → constant → scaled zeros → mean 0.0
        index=["TF1", "TF2"]
    )
    tg_df = pd.DataFrame(
        [[2.0, 2.0, 2.0],   # TG1 → constant → 0.0
         [0.0, 5.0, 10.0]], # TG2 → [0, 0.5, 1.0] → 0.5
        index=["TG1", "TG2"]
    )
    m_tf, m_tg = compute_minmax_expr_mean(tf_df, tg_df)

    tf_means = dict(zip(m_tf["TF"], m_tf["mean_tf_expr"]))
    tg_means = dict(zip(m_tg["TG"], m_tg["mean_tg_expr"]))

    assert pytest.approx(tf_means["TF1"], rel=1e-6) == 0.5
    assert pytest.approx(tf_means["TF2"], rel=1e-6) == 0.0
    assert pytest.approx(tg_means["TG1"], rel=1e-6) == 0.0
    assert pytest.approx(tg_means["TG2"], rel=1e-6) == 0.5
    # Shapes
    assert list(m_tf.columns) == ["TF", "mean_tf_expr"]
    assert list(m_tg.columns) == ["TG", "mean_tg_expr"]


def test_make_peak_to_window_map_basic():
    peaks = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [100, 500],
        "end":   [200, 600],
        "peak_id": ["chr1:100-200", "chr1:500-600"],
    })
    windows_bed = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "start": [0, 300, 600],
        "end":   [300, 600, 900],
        "win_idx": [0, 1, 2],
    })
    m = make_peak_to_window_map(peaks, windows_bed)
    assert isinstance(m, dict)
    assert set(m.keys()) == {"chr1:100-200", "chr1:500-600"}
    assert all(isinstance(v, int) for v in m.values())
    # Containment checks (end-inclusive behavior in your impl places 500–600 in window [300,600])
    assert m["chr1:100-200"] == 0
    assert m["chr1:500-600"] == 1



def test_align_to_vocab_happy_path_and_errors():
    # Global vocab and a tensor aligned to it (rows correspond to vocab order)
    vocab = ["A", "B", "C", "D"]
    vocab_map = {n: i for i, n in enumerate(vocab)}
    tensor_all = np.arange(4 * 3).reshape(4, 3)  # shape = (len(vocab), C)
    # rows:
    # A -> [0,1,2]
    # B -> [3,4,5]
    # C -> [6,7,8]
    # D -> [9,10,11]

    # Names specific to a chromosome/partition (some present, some absent)
    names = ["C", "X", "A"]  # "X" is unknown
    aligned, kept_names, kept_ids = align_to_vocab(names, vocab_map, tensor_all, label="genes")

    # Expect to keep only present names, in the order they appear in `names`
    assert kept_names == ["C", "A"]
    assert kept_ids == [2, 0]
    # Rows should be stacked in that same order
    np.testing.assert_array_equal(aligned, np.vstack([tensor_all[2], tensor_all[0]]))

    # If tensor_all doesn't match vocab length on axis 0, a ValueError should be raised
    bad_tensor = np.arange(3 * 3).reshape(3, 3)  # len != len(vocab)
    with pytest.raises(ValueError):
        _ = align_to_vocab(names, vocab_map, bad_tensor, label="genes")

    # If none of the names are in vocab, we should get empty outputs
    none_present = ["X", "Y"]
    aligned2, kept_names2, kept_ids2 = align_to_vocab(none_present, vocab_map, tensor_all, label="genes")
    assert aligned2.shape == (0, tensor_all.shape[1])
    assert kept_names2 == []
    assert kept_ids2 == []

    # Duplicates in names should typically mirror duplicates in output
    dup_names = ["B", "B", "E"]
    aligned3, kept_names3, kept_ids3 = align_to_vocab(dup_names, vocab_map, tensor_all, label="genes")
    assert kept_names3 == ["B", "B"]
    assert kept_ids3 == [1, 1]
    np.testing.assert_array_equal(aligned3, np.vstack([tensor_all[1], tensor_all[1]]))



@pytest.mark.skipif(pytest.importorskip("pybedtools") is None, reason="pybedtools not installed")
def test_calculate_peak_to_tg_distance_score_minimal(tmp_path: Path):
    # minimal, close-by peaks and TSSs
    peaks_df = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [100, 900],
        "end":   [200, 1000],
        "peak_id": ["chr1:100-200", "chr1:900-1000"],
    })
    tss_df = pd.DataFrame({
        "chrom": ["chr1", "chr1"],
        "start": [150, 950],         # within 50 bp of each peak
        "end":   [151, 951],
        "name":  ["GENEA", "GENEB"],
    })

    peak_bed = tmp_path / "peaks.bed"
    tss_bed  = tmp_path / "tss.bed"
    out_path = tmp_path / "dist.parquet"

    df = calculate_peak_to_tg_distance_score(
        peak_bed_file=str(peak_bed),
        tss_bed_file=str(tss_bed),
        peak_gene_dist_file=str(out_path),
        mesc_atac_peak_loc_df=peaks_df,
        gene_tss_df=tss_df,
        max_peak_distance=1000,
        distance_factor_scale=25000,
        force_recalculate=True,
    )

    assert {"peak_id", "target_id", "TSS_dist", "TSS_dist_score"}.issubset(df.columns)
    # both should be very close → high score
    assert (df["TSS_dist_score"] > 0.9).all()



def test_merge_tf_tg_attributes_with_combinations(tmp_path: Path):
    # all combos
    tf_tg = pd.DataFrame({"TF": ["A", "A", "B"], "TG": ["X", "Y", "Z"]})
    # reg potential/motif density for some pairs
    reg = pd.DataFrame({"TF": ["A"], "TG": ["X"], "reg_potential": [5.0], "motif_density": [3.0]})
    # means
    tf_means = pd.DataFrame({"TF": ["A", "B"], "mean_tf_expr": [0.5, 0.2]})
    tg_means = pd.DataFrame({"TG": ["X", "Y", "Z"], "mean_tg_expr": [0.1, 0.2, 0.3]})
    out_f = tmp_path / "combo_attrs.parquet"
    out = merge_tf_tg_attributes_with_combinations(tf_tg, reg, tf_means, tg_means, out_f, tf_vocab={"A","B"})
    exp_cols = {"TF","TG","reg_potential","motif_density","mean_tf_expr","mean_tg_expr","expr_product","log_reg_pot","motif_present"}
    assert exp_cols.issubset(out.columns)
    # standardized TF/TG after merge
    assert set(out["TF"]) <= {"A", "B"}
    assert _all_upper_no_version(out["TF"].tolist() + out["TG"].tolist())


def test_merge_tf_tg_data_with_pkn_balances_and_metadata(tmp_path: Path):
    # Ensure each TF has at least one negative candidate so balancing keeps it
    # Candidate TF–TG pairs
    df = pd.DataFrame(
        {"TF": ["A", "A", "A", "B", "B"],
         "TG": ["X", "Y", "Z", "Z", "Q"]}  # B has a negative (B,Q)
    )

    # Create PKN CSVs with metadata
    cols = ["TF", "TG", "string_experimental_score", "trrust_regulation", "kegg_n_pathways"]
    string = pd.DataFrame([["A","X", 900, None, None], ["Z","B", 800, None, None]], columns=cols)  # includes reversed (Z,B)
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
        seed=42, add_pkn_scores=True
    )

    pos = out_plain[out_plain.get("label", 0) == 1]
    pos_pairs = set(map(tuple, pos[["TF","TG"]].drop_duplicates().values))
    assert ("A", "X") in pos_pairs
    assert ("A", "Y") in pos_pairs

    # Metadata flags present
    assert {"in_STRING","in_TRRUST","in_KEGG","n_sources"}.issubset(out_with_meta.columns)

# --------- TF/TG combo writer (normalization + union tests) ---------

@pytest.mark.skipif(create_tf_tg_combination_files is None, reason="create_tf_tg_combination_files not exposed")
def test_create_tf_tg_combination_files_normalizes_and_unions(tmp_path: Path):
    # Write CSV with comma to avoid sep autodetection quirks
    tf_ref = pd.DataFrame({"TF_Name": ["Tp53", "Gata6", "Sox2"]})
    tf_ref_path = tmp_path / "tf_reference.csv"
    tf_ref.to_csv(tf_ref_path, index=False)

    dataset_dir = tmp_path / "dataset"
    genes_round1 = ["tp53.1", "gata6", "Actb.2"]

    import inspect
    sig = inspect.signature(create_tf_tg_combination_files)
    tfs1, tgs1, combos1 = create_tf_tg_combination_files(
        genes_round1, tf_ref_path, dataset_dir, tf_name_col="TF_Name"
    )
    out_dir = dataset_dir / "tf_tg_combos"

    total_p = out_dir / "total_genes.csv"
    tf_p    = out_dir / "tf_list.csv"
    tg_p    = out_dir / "tg_list.csv"
    combo_p = out_dir / "tf_tg_combos.csv"
    assert total_p.exists() and tf_p.exists() and tg_p.exists() and combo_p.exists()

    total = pd.read_csv(total_p)
    tf_list = pd.read_csv(tf_p)
    tg_list = pd.read_csv(tg_p)
    combos = pd.read_csv(combo_p)

    assert set(tf_list["TF"]) >= {"TP53", "GATA6"}
    assert "ACTB" in set(tg_list["TG"])
    assert _all_upper_no_version(total["Gene"].tolist())
    assert _all_upper_no_version(tf_list["TF"].tolist())
    assert _all_upper_no_version(tg_list["TG"].tolist())
    assert _all_upper_no_version((combos["TF"].tolist() + combos["TG"].tolist()))

    # Add a new TF; ACTB already present
    genes_round2 = ["Sox2", "Actb"]
    tfs2, tgs2, combos2 = create_tf_tg_combination_files(
        genes_round2, tf_ref_path, dataset_dir, tf_name_col="TF_Name"
    )

    total2 = pd.read_csv(total_p)
    tf_list2 = pd.read_csv(tf_p)
    tg_list2 = pd.read_csv(tg_p)
    combos2 = pd.read_csv(combo_p)

    assert {"TP53","GATA6","SOX2"}.issubset(set(tf_list2["TF"]))
    assert "ACTB" in set(tg_list2["TG"])
    assert _all_upper_no_version(total2["Gene"].tolist())
    assert _all_upper_no_version(tf_list2["TF"].tolist())
    assert _all_upper_no_version(tg_list2["TG"].tolist())
    assert _all_upper_no_version((combos2["TF"].tolist() + combos2["TG"].tolist()))



# --------- Optional slow tests (require scanpy/leidenalg) ---------
@pytest.mark.slow
@pytest.mark.skipif(
    pytest.importorskip("scanpy", reason="scanpy not installed") is None or
    pytest.importorskip("leidenalg", reason="leidenalg not installed") is None,
    reason="scanpy/leidenalg missing",
)
