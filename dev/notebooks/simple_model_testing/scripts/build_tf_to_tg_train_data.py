import os
import sys
import json
import gtfparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from tqdm import tqdm

import torch
import argparse

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
sys.path.append(str(PROJECT_DIR))

import utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_ground_truth(ground_truth_file: Path | str) -> pd.DataFrame:
    if isinstance(ground_truth_file, str):
        ground_truth_file = Path(ground_truth_file)

    logging.info(f"Loading ground truth file: {ground_truth_file.name}")

    if ground_truth_file.suffix == ".csv":
        sep = ","
    elif ground_truth_file.suffix == ".tsv":
        sep = "\t"

    ground_truth_df = pd.read_csv(ground_truth_file, sep=sep, on_bad_lines="skip", engine="python")

    if "chip" in ground_truth_file.name and "atlas" in ground_truth_file.name:
        ground_truth_df = ground_truth_df[["source_id", "target_id"]]

    if ground_truth_df.columns[0] != "Source" or ground_truth_df.columns[1] != "Target":
        ground_truth_df = ground_truth_df.rename(
            columns={ground_truth_df.columns[0]: "Source", ground_truth_df.columns[1]: "Target"}
        )
    ground_truth_df["Source"] = ground_truth_df["Source"].astype(str).str.capitalize()
    ground_truth_df["Target"] = ground_truth_df["Target"].astype(str).str.capitalize()

    return ground_truth_df[["Source", "Target"]].dropna()


def split_genes_by_chromosome(gene_reference_file: Path):
    gene_ref_df = gtfparse.read_gtf(gene_reference_file, result_type="pandas")

    gene_chrom = gene_ref_df[["seqname", "gene_name"]].rename(
        columns={"seqname": "chrom", "gene_name": "TG"}
    )

    train_genes = gene_chrom[gene_chrom["chrom"].isin([str(i) for i in range(1, 16)])][
        "TG"
    ].unique()
    logging.info(f"Train set: {len(train_genes)} genes")

    val_genes = gene_chrom[gene_chrom["chrom"].isin([str(i) for i in range(16, 19)])][
        "TG"
    ].unique()
    logging.info(f"Validation set: {len(val_genes)} genes")

    test_genes = gene_chrom[gene_chrom["chrom"].isin([str(19)])]["TG"].unique()
    logging.info(f"Test set: {len(test_genes)} genes")

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

    gt_train_df = ground_truth_df[ground_truth_df["Target"].isin(train_genes_set)].copy()
    gt_val_df = ground_truth_df[ground_truth_df["Target"].isin(val_genes_set)].copy()
    gt_test_df = ground_truth_df[ground_truth_df["Target"].isin(test_genes_set)].copy()

    logging.info(f"Train interactions: {len(gt_train_df)}")
    logging.info(f"Validation interactions: {len(gt_val_df)}")
    logging.info(f"Test interactions: {len(gt_test_df)}")

    return gt_train_df, gt_val_df, gt_test_df


def load_ground_truth_files(gt_path_list: list[Path]) -> pd.DataFrame:
    gt_dfs = [load_ground_truth(gt_path) for gt_path in gt_path_list]
    return pd.concat(gt_dfs, ignore_index=True)


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
    true_edges, false_edges = utils.create_true_false_edges(
        edge_df=gt_df,
        tf_names=tf_name_to_idx.keys(),
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
    max_precompute_peaks=64,
):
    valid_peak_set = set(atac_peak_map.keys())

    peak_to_gene_valid = peak_to_gene[peak_to_gene["peak_id"].isin(valid_peak_set)].copy()
    peak_to_gene_valid["abs_dist"] = peak_to_gene_valid["TSS_dist"].abs()

    tg_to_peak_info = {}
    for tg_norm, sub in peak_to_gene_valid.groupby("target_id_norm", sort=False):
        sub = sub.sort_values("abs_dist").head(max_precompute_peaks)

        peak_ids = sub["peak_id"].tolist()
        peak_indices = np.asarray([atac_peak_map[p] for p in peak_ids], dtype=np.int64)
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
    max_peaks_per_tg=64,
    max_cells_per_pair=8,
    seed=123,
    zero_fields=None,
    *,
    tg_to_peak_info,
    cell_to_idx,
    atac_mat,
    rna_mat,
    gene_to_rna_idx,
    common_cells,
    tf_name_to_idx,
    tg_id_to_idx,
):
    """
    Build compact TFTGRegulationModel inputs.

    This version stores indices and small numeric arrays only.
    Large lookup tensors are retrieved lazily in Dataset.__getitem__().
    """

    rng = np.random.default_rng(seed)

    tf_names = []
    tg_names = []
    cell_ids = []
    labels = []

    tf_indices = []
    tg_indices = []
    peak_indices_all = []

    peak_access = []
    peak_dist = []
    peak_masks = []
    tf_expr = []
    tg_expr = []
    peak_id_sets = []

    if zero_fields is None:
        zero_fields = set()
    else:
        zero_fields = set(zero_fields)

    ZEROABLE_FIELDS = {
        "tf_embedding",
        "peak_sequences",
        "peak_accessibility",
        "peak_distance",
        "tf_expression",
        "tg_expression",
        "tg_embedding",
    }

    invalid = zero_fields - ZEROABLE_FIELDS
    if invalid:
        raise ValueError(
            f"Invalid zero_fields: {sorted(invalid)}. "
            f"Valid options are: {sorted(ZEROABLE_FIELDS)}"
        )

    common_cells = list(common_cells)
    n_common_cells = len(common_cells)
    miniters = max(1, len(tf_tg_df) // 1000)

    for _, row in tqdm(
        tf_tg_df.iterrows(),
        total=len(tf_tg_df),
        ncols=100,
        desc="Building compact TF-TG inputs",
        miniters=miniters,
        
    ):
        tf_name = row["tf_name"]
        tg_name = row["tg_id"]
        label = float(row["label"])

        tf_norm = str(tf_name).upper()
        tg_norm = str(tg_name).upper()

        tf_idx = tf_name_to_idx.get(tf_name)
        tg_idx = tg_id_to_idx.get(tg_name)

        if tf_idx is None or tg_idx is None:
            continue

        peak_info = tg_to_peak_info.get(tg_norm)
        if peak_info is None:
            continue

        # -----------------------------
        # Peak index / distance setup
        # -----------------------------
        peak_ids_real = list(peak_info["peak_ids"][:max_peaks_per_tg])
        peak_indices_real = list(peak_info["peak_indices"][:max_peaks_per_tg])
        peak_dst_real = list(peak_info["peak_distances"][:max_peaks_per_tg])

        n_peaks = len(peak_indices_real)
        if n_peaks == 0:
            continue

        peak_indices = np.asarray(peak_indices_real, dtype=np.int64)
        peak_dst = np.asarray(peak_dst_real, dtype=np.float32)
        peak_mask = np.ones(n_peaks, dtype=bool)
        peak_ids = list(peak_ids_real)

        if n_peaks < max_peaks_per_tg:
            pad_len = max_peaks_per_tg - n_peaks

            # Use 0 as dummy padded peak index.
            # It will be masked out by peak_mask.
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

            peak_ids = peak_ids + [""] * pad_len

        # -----------------------------
        # Sample cells
        # -----------------------------
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

        # -----------------------------
        # Vectorized ATAC accessibility
        # -----------------------------
        peak_acc_matrix = np.zeros(
            (len(sampled_cell_indices), max_peaks_per_tg),
            dtype=np.float32,
        )

        peak_acc_matrix[:, :n_peaks] = atac_mat[
            np.ix_(peak_indices_real, sampled_cell_indices)
        ].T

        # -----------------------------
        # Vectorized RNA expression
        # -----------------------------
        tf_rna_idx = gene_to_rna_idx.get(tf_norm)
        tg_rna_idx = gene_to_rna_idx.get(tg_norm)

        if tf_rna_idx is None:
            tf_expr_vals = np.zeros(len(sampled_cell_indices), dtype=np.float32)
        else:
            tf_expr_vals = rna_mat[tf_rna_idx, sampled_cell_indices].astype(np.float32)

        if tg_rna_idx is None:
            tg_expr_vals = np.zeros(len(sampled_cell_indices), dtype=np.float32)
        else:
            tg_expr_vals = rna_mat[tg_rna_idx, sampled_cell_indices].astype(np.float32)

        # -----------------------------
        # Append one row per sampled cell
        # -----------------------------
        for j, cell_id in enumerate(sampled_cells):
            tf_indices.append(tf_idx)
            tg_indices.append(tg_idx)
            peak_indices_all.append(peak_indices)

            peak_access.append(peak_acc_matrix[j])
            peak_dist.append(peak_dst)
            peak_masks.append(peak_mask)

            tf_expr.append(float(tf_expr_vals[j]))
            tg_expr.append(float(tg_expr_vals[j]))

            peak_id_sets.append(peak_ids)
            tf_names.append(tf_name)
            tg_names.append(tg_name)
            cell_ids.append(cell_id)
            labels.append(label)

    if len(labels) == 0:
        raise ValueError(
            "No TF-TG examples were created. Check TF/TG IDs, peak-to-gene mapping, "
            "and overlap with ATAC/RNA matrices."
        )

    return {
        "tf_name": tf_names,
        "tg_name": tg_names,
        "cell_id": cell_ids,
        "peak_ids": peak_id_sets,

        "label": torch.tensor(labels, dtype=torch.float32),

        # Compact indices
        "tf_idx": torch.tensor(tf_indices, dtype=torch.long),
        "tg_idx": torch.tensor(tg_indices, dtype=torch.long),
        "peak_indices": torch.tensor(np.stack(peak_indices_all), dtype=torch.long),

        # Small per-cell/per-edge tensors
        "peak_accessibility": torch.tensor(np.stack(peak_access), dtype=torch.float32),
        "peak_mask": torch.tensor(np.stack(peak_masks), dtype=torch.bool),
        "peak_distance": torch.tensor(np.stack(peak_dist), dtype=torch.float32),
        "tf_expression": torch.tensor(tf_expr, dtype=torch.float32),
        "tg_expression": torch.tensor(tg_expr, dtype=torch.float32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_dir", type=str, required=False, help="Path to directory containing training data cache files (if not using default)")
    parser.add_argument("--sample_pairs", type=int, default=None)
    parser.add_argument("--max_peaks_per_tg", type=int, default=64)
    parser.add_argument("--max_cells_per_pair", type=int, default=8)
    parser.add_argument("--pct_true_edges", type=float, default=0.15)
    parser.add_argument("--true_false_ratio", type=float, default=2.0)
    parser.add_argument("--force_reload", action="store_true")
    args = parser.parse_args()
    
    # Create the training cache directory if it doesn't exist
    training_data_dir = args.training_data_dir

    if training_data_dir:
        tf_tg_input_cache_dir = Path(training_data_dir)
    else:
        tf_tg_input_cache_dir = PROJECT_DIR / "data" / "training_data_cache"
    tf_tg_input_cache_dir.mkdir(exist_ok=True, parents=True)

    gene_ref_file = DATA_DIR / "genome_data" / "genome_annotation" / "mm10" / "Mus_musculus.GRCm39.115.gtf.gz"
    genome_fasta_path = DATA_DIR / "genome_data" / "reference_genome" / "mm10" / "mm10.fa"
    chrom_sizes_path = DATA_DIR / "genome_data" / "reference_genome" / "mm10" / "mm10.chrom.sizes"

    atac_pseudobulk = pd.read_parquet(PROJECT_DIR / "data" / "ATAC_data" / "RE_pseudobulk.parquet")
    peak_to_gene_distance = pd.read_parquet(PROJECT_DIR / "data" / "ATAC_data" / "peak_to_gene_dist.parquet")
    rna_pseudobulk = pd.read_parquet(PROJECT_DIR / "data" / "RNA_data" / "TG_pseudobulk.parquet")

    mm10_chip_atlas_file = DATA_DIR / "ground_truth_files" / "chip_atlas_tf_peak_tg_dist.csv"
    rn111_file = DATA_DIR / "ground_truth_files" / "RN111.tsv"
    rn112_file = DATA_DIR / "ground_truth_files" / "RN112.tsv"
    rn114_file = DATA_DIR / "ground_truth_files" / "RN114.tsv"
    rn116_file = DATA_DIR / "ground_truth_files" / "RN116.tsv"

    merged_ground_truth_df = load_ground_truth_files([
        mm10_chip_atlas_file,
        rn111_file,
        rn112_file,
        rn114_file,
        rn116_file,
    ])

    tf_name_to_idx_cache_path = tf_tg_input_cache_dir / "tf_name_to_idx.csv"
    tf_name_to_idx = pd.read_csv(tf_name_to_idx_cache_path).set_index("tf_name")["tf_idx"].to_dict()
    tg_id_to_idx = {tg: idx for idx, tg in enumerate(merged_ground_truth_df["Target"].unique())}

    train_genes, val_genes, test_genes = split_genes_by_chromosome(gene_ref_file)
    gt_train_df, gt_val_df, gt_test_df = create_train_val_test_splits(
        merged_ground_truth_df, train_genes, val_genes, test_genes
    )

    tf_tg_labeled_train_df = _create_labeled_df(
        gt_train_df,
        args.pct_true_edges,
        args.true_false_ratio,
        seed=123,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
    )
    tf_tg_labeled_val_df = _create_labeled_df(
        gt_val_df,
        args.pct_true_edges,
        args.true_false_ratio,
        seed=123,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
    )
    tf_tg_labeled_test_df = _create_labeled_df(
        gt_test_df,
        args.pct_true_edges,
        args.true_false_ratio,
        seed=123,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
    )

    dataset_peaks = atac_pseudobulk.index.to_list()
    valid_chroms = {f"chr{i}" for i in range(1, 20)}
    dataset_peaks = [peak for peak in dataset_peaks if peak.split(":", 1)[0] in valid_chroms]
    atac_peak_map = {peak: idx for idx, peak in enumerate(dataset_peaks)}

    tf_embedding_cache_path = tf_tg_input_cache_dir / "tf_embeddings.pt"
    tf_mask_cache_path = tf_tg_input_cache_dir / "tf_masks.pt"
    atac_peak_onehot_cache_path = tf_tg_input_cache_dir / "atac_peak_onehot_array.pt"

    tf_embeddings_tensor = torch.load(tf_embedding_cache_path)
    tf_mask_tensor = torch.load(tf_mask_cache_path)

    dataset_peaks = list(atac_peak_map.keys())
    if os.path.exists(atac_peak_onehot_cache_path):
        atac_peak_tensor = torch.load(atac_peak_onehot_cache_path)
    else:
        logging.info("Creating centered peak one-hot encodings for ATAC peaks...")
        atac_peak_array = utils.create_centered_peak_onehot_array(
            peak_ids=dataset_peaks,
            genome_fasta=genome_fasta_path,
            chrom_sizes=utils.load_chrom_sizes(chrom_sizes_path),
            peak_id_to_idx=atac_peak_map,
            flank_size=64,
            dtype=np.uint8,
            pad_out_of_bounds=True,
            num_workers=16,
            show_progress=True,
            chunk_size=10000,
        )
        atac_peak_tensor = torch.as_tensor(atac_peak_array, dtype=torch.uint8)
        atac_peak_tensor = atac_peak_tensor.float()
        torch.save(atac_peak_tensor, atac_peak_onehot_cache_path)
        
    if atac_peak_tensor.dtype == torch.uint8:
        atac_peak_tensor = atac_peak_tensor.float()

    rna_pseudobulk_norm = rna_pseudobulk.copy()
    rna_pseudobulk_norm.index = rna_pseudobulk_norm.index.str.upper()

    common_cells = sorted(set(rna_pseudobulk_norm.columns) & set(atac_pseudobulk.columns))
    peak_to_gene = peak_to_gene_distance.copy()
    peak_to_gene["target_id_norm"] = peak_to_gene["target_id"].str.upper()

    tg_to_peak_info, cell_to_idx, atac_mat, rna_mat, gene_to_rna_idx = prepare_tftg_lookup_tables(
        peak_to_gene=peak_to_gene,
        atac_peak_map=atac_peak_map,
        atac_pseudobulk=atac_pseudobulk,
        rna_pseudobulk_norm=rna_pseudobulk_norm,
        dataset_peaks=dataset_peaks,
        common_cells=common_cells,
        max_precompute_peaks=64,
    )

    def _sample_df(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
        if n is None or len(df) <= n:
            return df
        return df.sample(n=n, random_state=seed)

    if args.sample_pairs is None:
        args.sample_pairs = len(tf_tg_labeled_train_df)

    tf_tg_train_subset = _sample_df(tf_tg_labeled_train_df, n=args.sample_pairs, seed=123)
    tf_tg_val_subset = _sample_df(tf_tg_labeled_val_df, n=args.sample_pairs // 2, seed=123)
    tf_tg_test_subset = _sample_df(tf_tg_labeled_test_df, n=args.sample_pairs // 4, seed=123)

    common_build_kwargs = dict(
        max_peaks_per_tg=8,
        max_cells_per_pair=16,
        zero_fields=None,
        tg_to_peak_info=tg_to_peak_info,
        cell_to_idx=cell_to_idx,
        atac_mat=atac_mat,
        rna_mat=rna_mat,
        gene_to_rna_idx=gene_to_rna_idx,
        common_cells=common_cells,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
    )

    train_file = tf_tg_input_cache_dir / "tf_to_tg_training_data" / "train.lmdb"
    val_file = tf_tg_input_cache_dir / "tf_to_tg_training_data" / "val.lmdb"
    test_file = tf_tg_input_cache_dir / "tf_to_tg_training_data" / "test.lmdb"
    if all(f.exists() for f in [train_file, val_file, test_file]) and not args.force_reload:
        logging.info("LMDB files already exist. Skipping LMDB creation.")
        return
    
    logging.info("Building training inputs")
    tftg_inputs_train = build_tftg_inputs(
        tf_tg_train_subset,
        seed=123,
        **common_build_kwargs,
    )

    logging.info("Building validation inputs")
    tftg_inputs_val = build_tftg_inputs(
        tf_tg_val_subset,
        seed=124,
        **common_build_kwargs,
    )

    logging.info("Building test inputs")
    tftg_inputs_test = build_tftg_inputs(
        tf_tg_test_subset,
        seed=125,
        **common_build_kwargs,
    )


    tf_tg_input_cache_dir = (
        tf_tg_input_cache_dir
        / "tf_tg_training_data_cache"
    )

    tf_tg_input_cache_dir.mkdir(parents=True, exist_ok=True)

    # Save compact split inputs
    torch.save(tftg_inputs_train, tf_tg_input_cache_dir / "tftg_inputs_train.pt")
    torch.save(tftg_inputs_val, tf_tg_input_cache_dir / "tftg_inputs_val.pt")
    torch.save(tftg_inputs_test, tf_tg_input_cache_dir / "tftg_inputs_test.pt")

    # Save lookup tensors
    torch.save(atac_peak_tensor, tf_tg_input_cache_dir / "atac_peak_tensor.pt")

    # Save mapping dictionaries and metadata
    metadata = {
        "tf_name_to_idx": tf_name_to_idx,
        "tg_id_to_idx": tg_id_to_idx,
        "gene_to_rna_idx": gene_to_rna_idx,
        "cell_to_idx": cell_to_idx,
        "max_peaks_per_tg": 8,
        "max_cells_per_pair": 16,
        "flank_size": 64,
        "peak_dtype": "uint8",
    }
    with open(tf_tg_input_cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # Save a manifest to keep track of model settings and dataset versions
    manifest = {
        "max_peaks_per_tg": 8,
        "max_cells_per_pair": 16,
        "flank_size": 64,
        "atac_peak_tensor_dtype": str(atac_peak_tensor.dtype),
        "atac_peak_tensor_shape": list(atac_peak_tensor.shape),
        "tf_embeddings_tensor_shape": list(tf_embeddings_tensor.shape),
        "tf_mask_tensor_shape": list(tf_mask_tensor.shape),
        "n_train_rows": int(len(tftg_inputs_train["label"])),
        "n_val_rows": int(len(tftg_inputs_val["label"])),
        "n_test_rows": int(len(tftg_inputs_test["label"])),
    }

    with open(tf_tg_input_cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logging.info(f"Wrote training data and metadata to {tf_tg_input_cache_dir}")


if __name__ == "__main__":
    main()
