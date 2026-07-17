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

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
sys.path.append(str(PROJECT_DIR))

import utils
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
    logging.info(f"  - Train set: {len(train_genes):,} genes (chroms {min(train_chroms)}-{max(train_chroms)})")

    val_genes = gene_chrom[gene_chrom["chrom"].isin(val_chroms)][
        "TG"
    ].unique()
    logging.info(f"  - Validation set: {len(val_genes):,} genes (chroms {min(val_chroms)}-{max(val_chroms)})")

    test_genes = gene_chrom[gene_chrom["chrom"].isin(test_chroms)]["TG"].unique()
    logging.info(f"  - Test set: {len(test_genes):,} genes (chroms {min(test_chroms)}-{max(test_chroms)})")

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
    rows = []
    for tf, tg in true_interactions:
        rows.append((tf, tg, 1))
    for tf, tg in false_interactions:
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

def prepare_tftg_lookup_tables(
    peak_to_gene,
    atac_peak_map,
    atac_pseudobulk,
    rna_pseudobulk_norm,
    dataset_peaks,
    common_cells,
    max_precompute_peaks=None,
):
    valid_peak_set = set(atac_peak_map.keys())

    peak_to_gene_valid = peak_to_gene[
        peak_to_gene["peak_id"].isin(valid_peak_set)
    ].copy()

    peak_to_gene_valid["abs_dist"] = peak_to_gene_valid["TSS_dist"].abs()

    tg_to_peak_info = {}

    # Subset to only peaks within 100kb of the TG TSS and sort by distance
    for tg_norm, sub in peak_to_gene_valid.groupby("target_id_norm", sort=False):
        sub = sub[sub["abs_dist"] <= 100_000].sort_values("abs_dist")

        if sub.empty:
            continue

        # Optional cap to only use the closest N peaks per TG
        if max_precompute_peaks is not None:
            sub = sub.head(max_precompute_peaks)

        peak_ids = sub["peak_id"].tolist()
        peak_indices = np.asarray(
            [atac_peak_map[p] for p in peak_ids],
            dtype=np.int64,
        )
        peak_distances = sub["TSS_dist"].to_numpy(dtype=np.float32)

        tg_to_peak_info[tg_norm] = {
            "peak_ids": peak_ids,
            "peak_indices": peak_indices,
            "peak_distances": peak_distances,
        }

    cell_to_idx = {cell: i for i, cell in enumerate(common_cells)}

    atac_mat = (
        atac_pseudobulk
        .reindex(index=dataset_peaks, columns=common_cells)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    rna_mat = (
        rna_pseudobulk_norm
        .reindex(columns=common_cells)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    gene_to_rna_idx = {gene: i for i, gene in enumerate(rna_pseudobulk_norm.index)}

    return tg_to_peak_info, cell_to_idx, atac_mat, rna_mat, gene_to_rna_idx


def build_tftg_inputs(
    tf_tg_df,
    max_peaks_per_tg=None,
    max_cells_per_pair=8,
    seed=123,
    silence=False,
    *,
    tg_to_peak_info,
    cell_to_idx,
    atac_mat,
    rna_mat,
    gene_to_rna_idx,
    common_cells,
    tf_name_to_idx,
    tg_id_to_idx,
    max_peaks_real,
):
    """
    Build one compact item per TF-TG edge.

    Output shapes:
        label:              [E]
        tf_idx:             [E]
        tg_idx:             [E]
        peak_indices:       [E, P]
        peak_distance:      [E, P]
        peak_mask:          [E, P]
        peak_accessibility: [E, C, P]
        tf_expression:      [E, C]
        tg_expression:      [E, C]
    """

    rng = np.random.default_rng(seed)

    tf_names = []
    tg_names = []
    cell_ids_all = []
    labels = []

    tf_indices = []
    tg_indices = []
    peak_indices_all = []
    peak_access_all = []
    peak_dist_all = []
    peak_masks_all = []
    tf_expr_all = []
    tg_expr_all = []

    common_cells = list(common_cells)
    n_common_cells = len(common_cells)

    n_total = len(tf_tg_df)
    log_every = max(1, n_total // 50)

    for i, row in enumerate(tf_tg_df.itertuples(index=False), start=1):
        if silence == False:
            if i == 1 or i % log_every == 0 or i == n_total:
                logging.info(f"Building compact TF-TG edges: {100 * i / n_total:.1f}% ({i:,}/{n_total:,})")

        tf_name = row.tf_name
        tg_name = row.tg_id
        label = float(row.label)

        tf_idx = tf_name_to_idx.get(tf_name)
        tg_idx = tg_id_to_idx.get(tg_name)

        if tf_idx is None or tg_idx is None:
            continue

        peak_info = tg_to_peak_info.get(tg_name)
        if peak_info is None:
            continue

        peak_ids_real = list(peak_info["peak_ids"])
        peak_indices_real = list(peak_info["peak_indices"])
        peak_dst_real = list(peak_info["peak_distances"])

        n_peaks = len(peak_indices_real)
        if n_peaks == 0:
            continue

        peak_indices = np.asarray(peak_indices_real, dtype=np.int64)
        peak_dst = np.asarray(peak_dst_real, dtype=np.float32)
        peak_mask = np.ones(n_peaks, dtype=bool)

        if n_peaks < max_peaks_real:
            pad_len = max_peaks_real - n_peaks

            peak_indices = np.pad(
                peak_indices,
                (0, pad_len),
                constant_values=0,
            )

            peak_dst = np.pad(
                peak_dst,
                (0, pad_len),
                constant_values=0.0,
            )

            peak_mask = np.pad(
                peak_mask,
                (0, pad_len),
                constant_values=False,
            )

        # Sample cells
        if max_cells_per_pair is None or max_cells_per_pair >= n_common_cells:
            sampled_cells = common_cells
        else:
            sampled_cells = rng.choice(
                common_cells,
                size=max_cells_per_pair,
                replace=False,
            ).tolist()

        sampled_cell_indices = np.asarray(
            [cell_to_idx[c] for c in sampled_cells],
            dtype=np.int64,
        )

        C = len(sampled_cell_indices)
        P = max_peaks_real

        # ATAC accessibility: [C, P]
        peak_acc_matrix = np.zeros((C, P), dtype=np.float32)
        peak_acc_matrix[:, :n_peaks] = atac_mat[
            np.ix_(peak_indices_real, sampled_cell_indices)
        ].T

        # RNA expression: [C]
        tf_rna_idx = gene_to_rna_idx.get(tf_name)
        tg_rna_idx = gene_to_rna_idx.get(tg_name)

        if tf_rna_idx is None or tg_rna_idx is None:
            raise ValueError(
                f"TF or TG missing from RNA matrix after filtering: "
                f"tf_name={tf_name}, tg_name={tg_name}, "
                f"tf_rna_idx={tf_rna_idx}, tg_rna_idx={tg_rna_idx}"
            )

        tf_expr_vals = np.asarray(
            rna_mat[tf_rna_idx, sampled_cell_indices],
            dtype=np.float32,
        ).reshape(-1)

        tg_expr_vals = np.asarray(
            rna_mat[tg_rna_idx, sampled_cell_indices],
            dtype=np.float32,
        ).reshape(-1)

        # Append once per TF-TG edge
        tf_names.append(tf_name)
        tg_names.append(tg_name)
        cell_ids_all.append(sampled_cells)
        labels.append(label)

        tf_indices.append(tf_idx)
        tg_indices.append(tg_idx)
        peak_indices_all.append(peak_indices)
        peak_access_all.append(peak_acc_matrix)
        peak_dist_all.append(peak_dst)
        peak_masks_all.append(peak_mask)
        tf_expr_all.append(tf_expr_vals)
        tg_expr_all.append(tg_expr_vals)

    if len(labels) == 0:
        raise ValueError(
            "No TF-TG examples were created. Check TF/TG IDs, peak-to-gene mapping, "
            "and overlap with ATAC/RNA matrices."
        )

    return {
        "tf_name": tf_names,
        "tg_name": tg_names,
        "cell_ids": cell_ids_all,

        "label": torch.tensor(labels, dtype=torch.float32),

        "tf_idx": torch.tensor(tf_indices, dtype=torch.long),
        "tg_idx": torch.tensor(tg_indices, dtype=torch.long),

        "peak_indices": torch.tensor(np.stack(peak_indices_all), dtype=torch.long),
        "peak_accessibility": torch.tensor(np.stack(peak_access_all), dtype=torch.float32),
        "peak_mask": torch.tensor(np.stack(peak_masks_all), dtype=torch.bool),
        "peak_distance": torch.tensor(np.stack(peak_dist_all), dtype=torch.float32),

        "tf_expression": torch.tensor(np.stack(tf_expr_all), dtype=torch.float32),
        "tg_expression": torch.tensor(np.stack(tg_expr_all), dtype=torch.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_pairs", type=int, default=None)
    parser.add_argument("--max_peaks_per_tg", type=int, default=None)
    parser.add_argument("--max_cells_per_pair", type=int, default=8)
    parser.add_argument("--pct_true_edges", type=float, default=0.15)
    parser.add_argument("--true_false_ratio", type=float, default=2.0)
    parser.add_argument("--peak_flank_size", type=int, default=64)
    parser.add_argument("--num_cpu", type=int, default=8)
    parser.add_argument("--force_reload", action="store_true")
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
    
    if all(f.exists() for f in required_cache_files) and not args.force_reload:
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
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    # Save a manifest to keep track of model settings and dataset versions
    manifest = {
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
