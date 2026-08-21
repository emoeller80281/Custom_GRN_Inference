import os
import sys
import gtfparse
from matplotlib.backend_bases import NonGuiException
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json

import torch
import argparse

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
sys.path.append(str(PROJECT_DIR))

import utils
from utils import prepare_tftg_lookup_tables, build_tftg_inputs
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _format_chroms(chroms: list[str]) -> str:
    """Render a chromosome list for logging.

    Only for log messages -- the splits themselves are made with isin() on the full list.
    min()/max() would compare these as strings, where "9" > "15", so ["1".."15"] printed
    as "1-15" came out as "1-9". Numeric labels are ordered numerically and collapsed to
    a range only when they are actually contiguous.
    """
    numeric = sorted((c for c in chroms if str(c).isdigit()), key=int)
    other = sorted(str(c) for c in chroms if not str(c).isdigit())

    parts = []
    if numeric:
        contiguous = int(numeric[-1]) - int(numeric[0]) == len(numeric) - 1
        parts.append(f"{numeric[0]}-{numeric[-1]}" if contiguous and len(numeric) > 1
                     else ", ".join(numeric))
    parts.extend(other)
    return ", ".join(parts)


def split_genes_by_chromosome(
    gene_reference_file: Path,
    train_chroms: list[str] = None,
    val_chroms: list[str] = None,
    test_chroms: list[str] = None
    ):
    logging.info(f"Splitting genes into train/val/test based on chromosome:")
    gene_ref_df = gtfparse.read_gtf(gene_reference_file, result_type="pandas")

    gene_chrom: pd.DataFrame = gene_ref_df[["seqname", "gene_name"]].rename(
        columns={"seqname": "chrom", "gene_name": "TG"}
    )
    
    gene_chrom["chrom"] = gene_chrom["chrom"].astype(str).str.replace("^chr", "", regex=True)
    gene_chrom["TG"] = gene_chrom["TG"].str.upper()
    
    train_genes = gene_chrom[gene_chrom["chrom"].isin(train_chroms)][
        "TG"
    ].unique()
    logging.info(f"  - Train set: {len(train_genes):,} genes (chroms {_format_chroms(train_chroms)})")

    val_genes = gene_chrom[gene_chrom["chrom"].isin(val_chroms)][
        "TG"
    ].unique()
    logging.info(f"  - Validation set: {len(val_genes):,} genes (chroms {_format_chroms(val_chroms)})")

    test_genes = gene_chrom[gene_chrom["chrom"].isin(test_chroms)]["TG"].unique()
    logging.info(f"  - Test set: {len(test_genes):,} genes (chroms {_format_chroms(test_chroms)})")

    return train_genes, val_genes, test_genes

def create_train_val_test_splits(
    ground_truth_df: pd.DataFrame,
    train_genes: np.ndarray,
    val_genes: np.ndarray,
    test_genes: np.ndarray,
):
    train_genes_set = set(train_genes)
    val_genes_set = set(val_genes)
    test_genes_set = set(test_genes)

    # Subset the ground truth to create train/val/test splits based on the target gene chromosome splits
    gt_train_df = ground_truth_df[ground_truth_df["Target"].isin(train_genes_set)].copy()
    gt_val_df = ground_truth_df[ground_truth_df["Target"].isin(val_genes_set)].copy()
    gt_test_df = ground_truth_df[ground_truth_df["Target"].isin(test_genes_set)].copy()
    
    if len(gt_train_df) == 0:
        logging.warning("No training interactions found for the selected train genes.")
        logging.info(f"Dataset genes: {list(train_genes_set)[:5]}")
        logging.info(f"Ground truth target genes: {ground_truth_df['Target'].unique()[:5]}")

    logging.info(f"Train interactions: {len(gt_train_df)}")
    logging.info(f"Validation interactions: {len(gt_val_df)}")
    logging.info(f"Test interactions: {len(gt_test_df)}")

    return gt_train_df, gt_val_df, gt_test_df

def load_nmp_trajectory_tfs(grn_coef_file: Path) -> set[str]:
    """TFs the paper's NMP-trajectory GRN considered as candidate regulators.

    `global_chip_GRN_coef.txt.gz` is the TF->gene regression table from
    Argelaguet et al. 2022, fitted on the metacells of the NMP -> {spinal cord,
    somitic mesoderm} trajectory. Every TF appearing in it survived in silico
    ChIP-seq binding, the 50 kb peak-to-gene window, and the variance filters, so
    it is the broadest defensible definition of "implicated in NMP differentiation".
    Taking all of them -- not just the ones with significant betas -- keeps any TF
    the paper evaluated on that trajectory out of training.
    """
    if not grn_coef_file.exists():
        raise FileNotFoundError(
            f"NMP GRN coefficient file not found: {grn_coef_file}. "
            "It ships with the Argelaguet et al. 2022 download under "
            "results/rna_atac/gene_regulatory_networks/metacells/trajectories/nmp/."
        )

    grn_df = pd.read_csv(grn_coef_file, sep="\t", usecols=["tf"])
    nmp_tfs = set(grn_df["tf"].astype(str).str.upper().unique())

    logging.info(f"Loaded {len(nmp_tfs)} NMP-trajectory TFs from {grn_coef_file.name}")
    return nmp_tfs


def split_ground_truth_by_tf(
    ground_truth_df: pd.DataFrame,
    nmp_tfs: set[str],
    val_frac: float = 0.15,
    seed: int = 123,
    min_positives_per_tf: int = 0,
):
    """Split edges by their TF, holding out the NMP-differentiation TFs for test.

    Unlike the chromosome split, which partitions on the *target gene* and lets every
    TF appear in all three splits, this partitions on the *TF*: a TF's edges land
    entirely in one split. Train and validation TFs are disjoint from the test TFs, so
    the model is scored on regulators whose target preferences it has never seen.

    Target genes are deliberately *not* held out -- the same TG can appear in train and
    test under different TFs. That is the intended generalisation question here
    ("does this unseen TF regulate this gene?"), and holding out genes as well would
    shrink the evaluation set without making the TF question any cleaner.
    """
    all_tfs = set(ground_truth_df["Source"].unique())

    test_tfs = sorted(all_tfs & nmp_tfs)
    train_pool = sorted(all_tfs - nmp_tfs)

    # Drop TFs with too few ground-truth edges to learn from or to score.
    #
    # Counted in GROUND-TRUTH edges, not cached positives, because the cache does not exist
    # yet at this point. The empirical conversion on this dataset is ~0.268 cached positives
    # per ground-truth edge (pct_true_edges sampling), so a threshold of 200 leaves roughly
    # 50 positives per TF.
    #
    # Why it matters: at the old setting the training pool contained BAZ2A and TAF1 with ZERO
    # cached positives, KDM5B with 2 and ETV2 with 3. Those contribute nothing learnable, and
    # under --per_tf_pos_weight they attract the largest weights of all (w = n_neg/n_pos, cap
    # 50). In validation they are worse than useless: an AUROC built on one positive is that
    # edge's percentile rank, yet it carries the same 1/N weight in the macro average as a TF
    # with 3,962 positives. ETV2 alone swung 0.833 -> 0.505 between two runs and accounted for
    # about half the apparent macro difference.
    #
    # Applied to the train/val pool only. The test TFs are fixed by biology -- they are the
    # NMP-trajectory regulators the whole experiment is about -- and dropping any of them
    # would silently change the benchmark and break comparison with runs already scored.
    # Thin test TFs are reported below so they can be excluded at REPORTING time instead,
    # which leaves the split itself stable.
    if min_positives_per_tf > 0:
        counts = ground_truth_df.groupby("Source").size()
        thin_pool = [tf for tf in train_pool if counts.get(tf, 0) < min_positives_per_tf]
        train_pool = [tf for tf in train_pool if counts.get(tf, 0) >= min_positives_per_tf]
        logging.info(
            f"--min_positives_per_tf {min_positives_per_tf}: dropped {len(thin_pool)} of "
            f"{len(thin_pool) + len(train_pool)} train/val-pool TFs "
            f"({', '.join(f'{tf}:{counts.get(tf, 0)}' for tf in sorted(thin_pool)[:12])}"
            f"{', ...' if len(thin_pool) > 12 else ''})"
        )
        thin_test = [tf for tf in test_tfs if counts.get(tf, 0) < min_positives_per_tf]
        if thin_test:
            logging.warning(
                f"{len(thin_test)} TEST TFs also fall below the threshold and were KEPT on "
                f"purpose (the test split is fixed by biology): "
                f"{', '.join(f'{tf}:{counts.get(tf, 0)}' for tf in sorted(thin_test))}. "
                "Consider excluding them when reporting macro metrics."
            )
        if len(train_pool) < 2:
            raise ValueError(
                f"min_positives_per_tf={min_positives_per_tf} left only {len(train_pool)} "
                "pool TFs; lower the threshold."
            )

    if not test_tfs:
        raise ValueError(
            "No ground-truth TFs overlap the NMP TF list. Check that TF names are "
            "upper-cased on both sides."
        )
    if len(train_pool) < 2:
        raise ValueError(
            f"Only {len(train_pool)} non-NMP TFs available; not enough to train and validate."
        )

    # Deterministic TF-level validation holdout carved out of the training TFs, so
    # validation also measures generalisation to unseen TFs rather than unseen edges.
    rng = np.random.default_rng(seed)
    n_val = max(1, round(len(train_pool) * val_frac))
    val_tfs = sorted(rng.choice(train_pool, size=n_val, replace=False).tolist())
    train_tfs = sorted(set(train_pool) - set(val_tfs))

    gt_train_df = ground_truth_df[ground_truth_df["Source"].isin(train_tfs)].copy()
    gt_val_df = ground_truth_df[ground_truth_df["Source"].isin(val_tfs)].copy()
    gt_test_df = ground_truth_df[ground_truth_df["Source"].isin(test_tfs)].copy()

    logging.info("Splitting ground truth by transcription factor:")
    logging.info(f"  - Train: {len(train_tfs):3d} TFs, {len(gt_train_df):,} interactions")
    logging.info(f"  - Val:   {len(val_tfs):3d} TFs, {len(gt_val_df):,} interactions")
    logging.info(f"  - Test:  {len(test_tfs):3d} TFs, {len(gt_test_df):,} interactions (NMP differentiation)")
    logging.info(f"  - Test TFs: {', '.join(test_tfs)}")

    # Belt and braces: the whole point of this split is that no TF crosses it.
    assert not (set(train_tfs) & set(test_tfs)), "Train/test TF leakage"
    assert not (set(val_tfs) & set(test_tfs)), "Val/test TF leakage"
    assert not (set(train_tfs) & set(val_tfs)), "Train/val TF leakage"

    tf_split = {"train": train_tfs, "val": val_tfs, "test": test_tfs}
    return gt_train_df, gt_val_df, gt_test_df, tf_split


def create_true_false_edges_from_full_universe(
    edge_df: pd.DataFrame,
    tf_col: str = "Source",
    item_col: str = "Target",
    pct_true_edges: float | None = 1.0,
    true_false_ratio: float = 1.0,
    seed: int = 123,
):
    df_all = edge_df[[tf_col, item_col]].copy()

    df_all = df_all.dropna(subset=[tf_col, item_col])

    df_all[tf_col] = df_all[tf_col].astype(str)
    df_all[item_col] = df_all[item_col].astype(str)

    df_all = df_all.drop_duplicates([tf_col, item_col]).reset_index(drop=True)

    if df_all.empty:
        raise ValueError(
            f"No edges remain after filtering by tf_names using columns "
            f"{tf_col!r} and {item_col!r}."
        )

    candidate_tfs = sorted(df_all[tf_col].unique())
    candidate_items = sorted(df_all[item_col].unique())

    gt_pairs = set(zip(df_all[tf_col], df_all[item_col]))

    full_universe = (
        pd.MultiIndex
        .from_product([candidate_tfs, candidate_items], names=[tf_col, item_col])
        .to_frame(index=False)
    )

    full_universe["_pair"] = list(zip(full_universe[tf_col], full_universe[item_col]))
    full_universe["_in_gt"] = full_universe["_pair"].isin(gt_pairs)

    true_df = full_universe[full_universe["_in_gt"]].copy()
    false_df = full_universe[~full_universe["_in_gt"]].copy()

    if pct_true_edges is not None:
        if not (0 < pct_true_edges <= 1):
            raise ValueError("pct_true_edges must be in (0, 1] or None.")

        true_df = true_df.sample(frac=pct_true_edges, random_state=seed)

    n_false = round(len(true_df) * true_false_ratio)

    if n_false > len(false_df):
        logging.warning(
            f"Requested {n_false:,} false edges, but only {len(false_df):,} are available. "
            "Using all available false edges."
        )
        n_false = len(false_df)

    false_df = false_df.sample(n=n_false, random_state=seed)

    true_edges = set(zip(true_df[tf_col], true_df[item_col]))
    false_edges = set(zip(false_df[tf_col], false_df[item_col]))

    return true_edges, false_edges

def create_labeled_tf_tg_dataset(
    true_interactions: set[tuple[str, str]],
    false_interactions: set[tuple[str, str]],
    tf_name_to_idx: dict[str, int],
    tg_id_to_idx: dict[str, int],
    drop_missing: bool = True,
) -> pd.DataFrame:
    # sorted(), not bare set iteration: set order over string tuples depends on
    # PYTHONHASHSEED, so without this the row order -- and therefore anything
    # downstream that indexes by position, e.g. df.sample(n=...) -- differs in every
    # process even with a fixed random_state.
    rows = []
    for tf, tg in sorted(true_interactions):
        rows.append((tf, tg, 1))
    for tf, tg in sorted(false_interactions):
        rows.append((tf, tg, 0))

    df = pd.DataFrame(rows, columns=["tf_name", "tg_id", "label"])
    df["tf_idx"] = df["tf_name"].map(tf_name_to_idx)
    df["tg_idx"] = df["tg_id"].map(tg_id_to_idx)

    missing_mask = df["tf_idx"].isna() | df["tg_idx"].isna()
    if missing_mask.any():
        n_missing = missing_mask.sum()
        if drop_missing:
            logging.info(f"Dropping {n_missing} interactions with missing TF or TG indices.")
            df = df.loc[~missing_mask].copy()
        else:
            missing_examples = df.loc[missing_mask].head()
            raise ValueError(
                f"{n_missing} interactions are missing TF or TG indices.\n"
                f"Examples:\n{missing_examples}"
            )

    df["tf_idx"] = df["tf_idx"].astype(np.int64)
    df["tg_idx"] = df["tg_idx"].astype(np.int64)
    df["label"] = df["label"].astype(np.float32)

    return df.sample(frac=1.0, random_state=123).reset_index(drop=True)

def _create_labeled_df(
    gt_df: pd.DataFrame,
    pct_true_edges: float = 0.15,
    true_false_ratio: float = 2.0,
    seed: int = 123,
    *,
    tf_name_to_idx,
    tg_id_to_idx,
):
    gt_df = gt_df[
        gt_df["Source"].isin(tf_name_to_idx.keys()) &
        gt_df["Target"].isin(tg_id_to_idx.keys())
    ].copy()
    
    true_edges, false_edges = create_true_false_edges_from_full_universe(
        edge_df=gt_df,
        tf_col="Source",
        item_col="Target",
        pct_true_edges=pct_true_edges,
        true_false_ratio=true_false_ratio,
        seed=seed,
    )

    return create_labeled_tf_tg_dataset(
        true_interactions=true_edges,
        false_interactions=false_edges,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
        drop_missing=False,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_pairs", type=int, default=None)
    parser.add_argument("--max_peaks_per_tg", type=int, default=None)
    parser.add_argument("--max_cells_per_pair", type=int, default=8)
    parser.add_argument("--pct_true_edges", type=float, default=0.15)
    parser.add_argument("--true_false_ratio", type=float, default=2.0)
    parser.add_argument("--peak_flank_size", type=int, default=64)
    parser.add_argument("--num_cpu", type=int, default=8)
    parser.add_argument(
        "--split_mode",
        choices=["chromosome", "tf"],
        default="chromosome",
        help=(
            "How to partition train/val/test. 'chromosome' (default) splits on the target "
            "gene's chromosome. 'tf' splits on the transcription factor, holding out the TFs "
            "implicated in NMP differentiation for test."
        ),
    )
    parser.add_argument(
        "--nmp_grn_coef_file",
        type=Path,
        default=(
            config.DATA_DIR / "dropbox_data" / "extracted" / "results" / "rna_atac"
            / "gene_regulatory_networks" / "metacells" / "trajectories" / "nmp"
            / "global_chip_GRN_coef.txt.gz"
        ),
        help="TF->gene coefficient table defining the NMP-trajectory TFs (--split_mode tf only)",
    )
    parser.add_argument(
        "--val_tf_frac",
        type=float,
        default=0.15,
        help="Fraction of non-NMP TFs held out for validation (--split_mode tf only)",
    )
    parser.add_argument(
        "--min_positives_per_tf",
        type=int,
        default=0,
        help=(
            "Drop train/val-pool TFs with fewer than this many ground-truth edges (0 = off). "
            "~0.268 cached positives per ground-truth edge on this dataset, so 200 leaves "
            "roughly 50 positives per TF. Test TFs are never dropped -- they are fixed by "
            "biology -- but thin ones are logged."
        ),
    )
    parser.add_argument("--force_reload", action="store_true")
    parser.add_argument(
        "--build_resample_matrices_only",
        action="store_true",
        help=(
            "Build and cache atac_mat.pt/rna_mat.pt (the full peak x cell and gene x cell "
            "pseudobulk matrices) and then exit, skipping ground-truth-edge-bag construction. "
            "These two files are only needed for --resample_cells_per_epoch in "
            "train_tf_to_tg_model.py. Use this to backfill them onto a cache that already has "
            "everything else, without repeating the slow one-hot peak encoding or edge-bag "
            "build. Ignored (superseded by the full build) if combined with --force_reload."
        ),
    )
    args = parser.parse_args()
    
    logging.info(f" === Species: {config.species}, Cell Type: {config.cell_type}, Sample: {config.sample_name} ===\n")
    
    max_peaks_per_tg = args.max_peaks_per_tg
    max_cells_per_pair = args.max_cells_per_pair
    pct_true_edges = args.pct_true_edges
    true_false_ratio = args.true_false_ratio
    peak_flank_size = args.peak_flank_size
    num_cpu = args.num_cpu
        
    gene_ref_file = config.gene_ref_file
    genome_fasta_path = config.genome_fasta_path
    chrom_sizes_path = config.chrom_sizes_path
    
    assert gene_ref_file.exists(), f"Gene reference file not found: {gene_ref_file}"
    assert genome_fasta_path.exists(), f"Genome FASTA file not found: {genome_fasta_path}"
    assert chrom_sizes_path.exists(), f"Chromosome sizes file not found: {chrom_sizes_path}"
    
    # Create the training cache directory if it doesn't exist
    input_data_dir = Path(config.sample_input_data_dir)
    
    assert input_data_dir.exists(), f"Input data directory does not exist: {input_data_dir}"
    
    tf_tg_input_cache_dir = config.tf_tg_input_cache_dir

    tf_tg_input_cache_dir.mkdir(parents=True, exist_ok=True)
    
    tf_name_to_idx_cache_path = config.tf_name_to_idx_cache_path
    tf_embedding_cache_path = config.tf_embedding_cache_path
    tf_mask_cache_path = config.tf_mask_cache_path
    merged_ground_truth_path = config.merged_ground_truth_cache_path
    
    atac_peak_onehot_cache_path = config.tf_tg_atac_peak_cache_path
    train_file = config.tf_tg_train_cache_path
    val_file = config.tf_tg_val_cache_path
    test_file = config.tf_tg_test_cache_path
    
    metadata_file = config.tf_tg_metadata_cache_path
    manifest_file = config.tf_tg_manifest_cache_path
    
    required_cache_files = [
        tf_name_to_idx_cache_path,
        tf_embedding_cache_path,
        tf_mask_cache_path,
        atac_peak_onehot_cache_path,
        train_file,
        val_file,
        test_file,
        metadata_file,
        manifest_file,
    ]
    
    if (
        all(f.exists() for f in required_cache_files)
        and not args.force_reload
        and not args.build_resample_matrices_only
    ):
        logging.info("All required cache files already exist. Skipping construction (use --force_reload to override).")
        return

    # Load the input data for the sample
    required_input_files = [
        "RE_pseudobulk.parquet",
        "peak_to_gene_dist.parquet",
        "TG_pseudobulk.parquet"
    ]
    
    for filename in required_input_files:
        file_path = input_data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Required input file not found: {file_path}")
    
    # Read in the ATAC and RNA pseudobulk data, and the peak-to-gene distance file
    atac_pseudobulk = pd.read_parquet(input_data_dir / "RE_pseudobulk.parquet")
    peak_to_gene_distance = pd.read_parquet(input_data_dir / "peak_to_gene_dist.parquet")
    rna_pseudobulk = pd.read_parquet(input_data_dir / "TG_pseudobulk.parquet")
    
    logging.info(f"ATAC peaks BEFORE peak-to-gene filtering: {atac_pseudobulk.shape[0]:,}")
    # Keep only ATAC peaks that are present in the peak-to-gene distance table
    valid_peak_ids = set(peak_to_gene_distance["peak_id"])

    atac_pseudobulk = atac_pseudobulk.loc[
        atac_pseudobulk.index.isin(valid_peak_ids)
    ].copy()
    logging.info(f"ATAC peaks AFTER peak-to-gene filtering: {atac_pseudobulk.shape[0]:,}")
    
    rna_pseudobulk_norm = rna_pseudobulk.copy()
    rna_pseudobulk_norm.index = rna_pseudobulk_norm.index.str.upper()

    common_cells = sorted(set(rna_pseudobulk_norm.columns) & set(atac_pseudobulk.columns))
    
    if len(common_cells) == 0:
        raise ValueError(
            "No common pseudobulk cell columns between RNA and ATAC matrices."
        )

    logging.info(f"Common RNA/ATAC pseudobulk columns: {len(common_cells):,}")
        
    peak_to_gene = peak_to_gene_distance.copy()
    peak_to_gene["target_id_norm"] = peak_to_gene["target_id"].str.upper()

    # Load and merge the ground truth files, or load from cache if already merged
    if not merged_ground_truth_path.exists() or args.force_reload:
        merged_ground_truth_df = utils.load_ground_truth_files(
            config.gt_by_dataset_dict[config.cell_type]
        )
    else:
        merged_ground_truth_df = pd.read_parquet(merged_ground_truth_path)

    merged_ground_truth_df["Source"] = merged_ground_truth_df["Source"].str.upper()
    merged_ground_truth_df["Target"] = merged_ground_truth_df["Target"].str.upper()

    if not merged_ground_truth_path.exists() or args.force_reload:
        merged_ground_truth_df.to_parquet(merged_ground_truth_path, index=False)
    
    gt_tfs_in_rna = set(merged_ground_truth_df["Source"]).intersection(rna_pseudobulk_norm.index)
    gt_tgs_in_rna = set(merged_ground_truth_df["Target"]).intersection(rna_pseudobulk_norm.index)
    logging.info(f"Ground truth TFs in RNA pseudobulk: {len(gt_tfs_in_rna)} (Example: {list(gt_tfs_in_rna)[:5]})")
    logging.info(f"Ground truth TGs in RNA pseudobulk: {len(gt_tgs_in_rna)} (Example: {list(gt_tgs_in_rna)[:5]})")
    
    n_before_rna_filter = len(merged_ground_truth_df)

    # Subset the ground truth to only TFs and TGs present in the rna_pseudobulk 
    merged_ground_truth_df = merged_ground_truth_df[
        merged_ground_truth_df["Source"].isin(gt_tfs_in_rna) &
        merged_ground_truth_df["Target"].isin(gt_tgs_in_rna)
    ].copy()
    
    logging.info(
        f"Ground truth edges after RNA TF/TG filtering: "
        f"{len(merged_ground_truth_df):,} / {n_before_rna_filter:,}"
    )

    # Get the map of TF name to index
    tf_name_to_idx = pd.read_csv(tf_name_to_idx_cache_path)
    tf_name_to_idx["tf_name"] = tf_name_to_idx["tf_name"].str.upper()
    tf_name_to_idx = tf_name_to_idx.set_index("tf_name")["tf_idx"].to_dict()
    
    # Only keep ground truth TFs that have embeddings (i.e. were present in the TF-DNA model training data)
    gt_tfs_in_embeddings = set(tf_name_to_idx.keys()).intersection(gt_tfs_in_rna)
    logging.info(f"Ground truth TFs with embeddings: {len(gt_tfs_in_embeddings)} (Example: {list(gt_tfs_in_embeddings)[:5]})")
    
    n_before_tf_embedding_filter = len(merged_ground_truth_df)

    merged_ground_truth_df = merged_ground_truth_df[
        merged_ground_truth_df["Source"].isin(gt_tfs_in_embeddings)
    ].copy()

    logging.info(
        f"Ground truth edges after filtering to TFs with embeddings: "
        f"{len(merged_ground_truth_df):,} / {n_before_tf_embedding_filter:,}"
    )

    # Create a map of TG name to index for TGs present in the ground truth (and RNA pseudobulk)
    tg_id_to_idx = {tg: idx for idx, tg in enumerate(merged_ground_truth_df["Target"].unique())}
        
    tf_split = None

    if args.split_mode == "tf":
        # Hold out the TFs the paper implicated in NMP -> {spinal cord, somitic mesoderm}
        # differentiation, and train on every other TF across all cell types. Splitting on
        # the TF rather than the target chromosome keeps the whole genome available for
        # training while still evaluating on regulators the model has never seen.
        nmp_tfs = load_nmp_trajectory_tfs(args.nmp_grn_coef_file)
        gt_train_df, gt_val_df, gt_test_df, tf_split = split_ground_truth_by_tf(
            merged_ground_truth_df,
            nmp_tfs=nmp_tfs,
            val_frac=args.val_tf_frac,
            min_positives_per_tf=args.min_positives_per_tf,
            seed=123,
        )
    else:
        if config.species == "mm10":
            train_chroms = [str(i) for i in range(1, 16)]
            val_chroms = [ str(i) for i in range(16, 18)]
            test_chroms = [str(i) for i in range(18, 20)]
        elif config.species == "hg38":
            train_chroms = [str(i) for i in range(1, 18)]
            val_chroms = [str(i) for i in range(18, 20)]
            test_chroms = [str(i) for i in range(20, 23)]

        # Split genes into train/val/test based on chromosome using the GTF reference file
        train_genes, val_genes, test_genes = split_genes_by_chromosome(
            gene_ref_file,
            train_chroms=train_chroms,
            val_chroms=val_chroms,
            test_chroms=test_chroms
            )

        # Subset the ground truth to create train/val/test splits based on the target gene chromosome splits
        # (Only keeps TFs and TGs present in the ground truth and RNA pseudobulk, and only keeps TFs with embeddings)
        gt_train_df, gt_val_df, gt_test_df = create_train_val_test_splits(
            merged_ground_truth_df, train_genes, val_genes, test_genes
        )
    logging.info(f"After subsetting to TFs with embeddings and TGs in RNA pseudobulk:")
    logging.info(f"  - Train interactions: {len(gt_train_df)} (TFs: {gt_train_df['Source'].nunique()}, TGs: {gt_train_df['Target'].nunique()})")
    logging.info(f"  - Val interactions: {len(gt_val_df)} (TFs: {gt_val_df['Source'].nunique()}, TGs: {gt_val_df['Target'].nunique()})")
    logging.info(f"  - Test interactions: {len(gt_test_df)} (TFs: {gt_test_df['Source'].nunique()}, TGs: {gt_test_df['Target'].nunique()})")

    # Create labeled TF-TG datasets for train/val/test splits
    # (samples true and false edges according to pct_true_edges and true_false_ratio)
    tf_tg_labeled_train_df = _create_labeled_df(
        gt_train_df,
        pct_true_edges,
        true_false_ratio,
        seed=123,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
    )
    tf_tg_labeled_val_df = _create_labeled_df(
        gt_val_df,
        pct_true_edges,
        true_false_ratio,
        seed=124,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
    )
    tf_tg_labeled_test_df = _create_labeled_df(
        gt_test_df,
        pct_true_edges,
        true_false_ratio,
        seed=125,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
    )

    # Create a map of ATAC peaks to indices in the pseudobulk matrix, filtering to valid chromosomes
    dataset_peaks = atac_pseudobulk.index.to_list()
    
    # Only use peaks from standard chromosomes (chr1-chr19 for mm10, chr1-chr22 for hg38) to avoid issues with 
    # non-standard chromosomes and contigs
    if config.species == "mm10":
        valid_chroms = {f"chr{i}" for i in range(1, 20)}
    elif config.species == "hg38":
        valid_chroms = {f"chr{i}" for i in range(1, 23)}
        
    dataset_peaks = [peak for peak in dataset_peaks if peak.split(":", 1)[0] in valid_chroms]
    atac_peak_map = {peak: idx for idx, peak in enumerate(dataset_peaks)}

    # Load cached TF embeddings and masks from TF-DNA model training
    tf_embeddings_tensor = torch.load(tf_embedding_cache_path, weights_only=True)
    tf_mask_tensor = torch.load(tf_mask_cache_path, weights_only=True)

    # Create or load cached one-hot encodings for ATAC peaks
    # One-hot encodings use ACGT order and uses 'flank_size' bp upstream and downstream of the peak center.    
    if os.path.exists(atac_peak_onehot_cache_path) and not args.force_reload:
        atac_peak_tensor = torch.load(atac_peak_onehot_cache_path, weights_only=True)
        
        expected_n_peaks = len(dataset_peaks)

        if atac_peak_tensor.shape[0] != expected_n_peaks:
            raise ValueError(
                f"ATAC one-hot tensor has {atac_peak_tensor.shape[0]:,} peaks, "
                f"but current dataset_peaks has {expected_n_peaks:,}. "
                "Delete the cached ATAC peak tensor or rerun with --force_reload."
            )
        
    else:
        logging.info("Creating centered peak one-hot encodings for ATAC peaks...")
        atac_peak_array = utils.create_centered_peak_onehot_array(
            peak_ids=dataset_peaks,
            genome_fasta=genome_fasta_path,
            chrom_sizes=utils.load_chrom_sizes(chrom_sizes_path),
            peak_id_to_idx=atac_peak_map,
            flank_size=peak_flank_size,
            dtype=np.uint8,
            pad_out_of_bounds=True,
            num_workers=num_cpu,
            show_progress=True,
            chunk_size=10000,
        )
        atac_peak_tensor = torch.as_tensor(atac_peak_array, dtype=torch.uint8)
        atac_peak_tensor = atac_peak_tensor.float()
        torch.save(atac_peak_tensor, atac_peak_onehot_cache_path)
        
    if atac_peak_tensor.dtype == torch.uint8:
        atac_peak_tensor = atac_peak_tensor.float()

    tg_to_peak_info, cell_to_idx, atac_mat, rna_mat, gene_to_rna_idx = prepare_tftg_lookup_tables(
        peak_to_gene=peak_to_gene,
        atac_peak_map=atac_peak_map,
        atac_pseudobulk=atac_pseudobulk,
        rna_pseudobulk_norm=rna_pseudobulk_norm,
        dataset_peaks=dataset_peaks,
        common_cells=common_cells,
        max_precompute_peaks=max_peaks_per_tg,
    )

    # Full [n_peaks, n_cells] / [n_genes, n_cells] matrices, only needed by
    # --resample_cells_per_epoch (train_tf_to_tg_model.py draws fresh cell columns from these
    # every epoch instead of reusing the frozen per-edge bag below). Column order matches
    # cell_to_idx / metadata["cell_to_idx"] exactly, since both come from this same
    # prepare_tftg_lookup_tables() call.
    atac_mat_cache_path = config.tf_tg_atac_mat_cache_path
    rna_mat_cache_path = config.tf_tg_rna_mat_cache_path

    if not atac_mat_cache_path.exists() or not rna_mat_cache_path.exists() or args.force_reload:
        logging.info(f"Saving full pseudobulk matrices for --resample_cells_per_epoch to {tf_tg_input_cache_dir}")
        torch.save(torch.as_tensor(atac_mat, dtype=torch.float32), atac_mat_cache_path)
        torch.save(torch.as_tensor(rna_mat, dtype=torch.float32), rna_mat_cache_path)

    if args.build_resample_matrices_only:
        logging.info(
            "--build_resample_matrices_only set: atac_mat.pt/rna_mat.pt written, "
            "skipping edge-bag construction."
        )
        return

    def _sample_df(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
        if n is None or len(df) <= n:
            return df
        return df.sample(n=n, random_state=seed)

    # Optionally sample a subset of TF-TG pairs for faster testing and debugging 
    if args.sample_pairs is not None:
        tf_tg_labeled_train_df = _sample_df(tf_tg_labeled_train_df, n=args.sample_pairs, seed=123)
        tf_tg_labeled_val_df = _sample_df(tf_tg_labeled_val_df, n=args.sample_pairs, seed=123)
        tf_tg_labeled_test_df = _sample_df(tf_tg_labeled_test_df, n=args.sample_pairs, seed=123)
    
    # Determine the maximum number of peaks to consider across all TGs in the dataset 
    # to ensure consistent tensor shapes
    tf_tg_df = pd.concat([tf_tg_labeled_train_df, tf_tg_labeled_val_df, tf_tg_labeled_test_df], ignore_index=True)
    
    if tf_tg_df.empty:
        raise ValueError(
            "No labeled TF-TG pairs were created across train/val/test. "
            "Check RNA filtering, TF embedding filtering, chromosome splits, and ground truth overlap."
        )
    
    max_peaks_real = max(
        len(tg_to_peak_info.get(tg_name, {}).get("peak_indices", []))
        for tg_name in tf_tg_df["tg_id"]
    )
    
    # Check that at least some TGs have peaks within 100kb, otherwise the model will have no signal to learn from
    n_tgs_with_peaks = sum(
        len(tg_to_peak_info.get(tg, {}).get("peak_indices", [])) > 0
        for tg in tf_tg_df["tg_id"].unique()
    )
    
    logging.info(f"TGs with at least one peak within 100kb: {n_tgs_with_peaks:,} / {tf_tg_df['tg_id'].nunique():,}")
    logging.info(f"Max peaks per TG after filtering/capping: {max_peaks_real:,}")

    if max_peaks_real == 0:
        raise ValueError(
            "No labeled TGs have peaks within 100kb. Check target_id_norm/tg_id matching, "
            "peak IDs, chromosome filtering, and TSS distance file."
        )
    
    common_build_kwargs = dict(
        max_peaks_per_tg=max_peaks_per_tg,
        max_cells_per_pair=max_cells_per_pair,
        tg_to_peak_info=tg_to_peak_info,
        cell_to_idx=cell_to_idx,
        atac_mat=atac_mat,
        rna_mat=rna_mat,
        gene_to_rna_idx=gene_to_rna_idx,
        common_cells=common_cells,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
        max_peaks_real=max_peaks_real,
    )
    
    if all(f.exists() for f in [train_file, val_file, test_file]) and not args.force_reload:
        logging.info("Cached input files already exist. Skipping (use --force_reload to override).")
        return
    
    # Build the compact TF-TG input datasets for train/val/test splits
    logging.info("\nBuilding training inputs")
    tftg_inputs_train = build_tftg_inputs(
        tf_tg_labeled_train_df,
        seed=123,
        **common_build_kwargs,
    )

    logging.info("\nBuilding validation inputs")
    tftg_inputs_val = build_tftg_inputs(
        tf_tg_labeled_val_df,
        seed=124,
        **common_build_kwargs,
    )

    logging.info("\nBuilding test inputs")
    tftg_inputs_test = build_tftg_inputs(
        tf_tg_labeled_test_df,
        seed=125,
        **common_build_kwargs,
    )

    # Save compact split inputs
    torch.save(tftg_inputs_train, train_file)
    torch.save(tftg_inputs_val, val_file)
    torch.save(tftg_inputs_test, test_file)

    # Save mapping dictionaries and metadata
    metadata = {
        "tf_name_to_idx": tf_name_to_idx,
        "tg_id_to_idx": tg_id_to_idx,
        "gene_to_rna_idx": gene_to_rna_idx,
        "cell_to_idx": cell_to_idx,
        "max_peaks_per_tg": max_peaks_per_tg,
        "max_cells_per_pair": max_cells_per_pair,
        "flank_size": peak_flank_size,
        "peak_dtype": "uint8",
        "max_peaks_real": max_peaks_real,
        "split_mode": args.split_mode,
    }
    if tf_split is not None:
        # Persist the exact TF partition -- evaluation needs to know which TFs were held
        # out, and it is not recoverable from the tensors alone.
        metadata["tf_split"] = tf_split
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    # Save a manifest to keep track of model settings and dataset versions
    manifest = {
        "split_mode": args.split_mode,
        "n_train_tfs": len(tf_split["train"]) if tf_split else None,
        "n_val_tfs": len(tf_split["val"]) if tf_split else None,
        "n_test_tfs": len(tf_split["test"]) if tf_split else None,
        "max_peaks_per_tg": max_peaks_per_tg,
        "max_cells_per_pair": max_cells_per_pair,
        "flank_size": peak_flank_size,
        "atac_peak_tensor_dtype": str(atac_peak_tensor.dtype),
        "atac_peak_tensor_shape": list(atac_peak_tensor.shape),
        "tf_embeddings_tensor_shape": list(tf_embeddings_tensor.shape),
        "tf_mask_tensor_shape": list(tf_mask_tensor.shape),
        "n_train_rows": int(len(tftg_inputs_train["label"])),
        "n_val_rows": int(len(tftg_inputs_val["label"])),
        "n_test_rows": int(len(tftg_inputs_test["label"])),
    }

    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    logging.info(f"Wrote training data and metadata to {tf_tg_input_cache_dir}")


if __name__ == "__main__":
    main()
