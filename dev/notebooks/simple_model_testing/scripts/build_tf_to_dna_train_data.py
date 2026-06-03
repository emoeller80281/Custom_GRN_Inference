import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse

import torch
from torch.nn.utils.rnn import pad_sequence

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
sys.path.append(str(PROJECT_DIR))

import utils
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def create_labeled_tf_peak_dataset(
    true_interactions: set[tuple[str, str]],
    false_interactions: set[tuple[str, str]],
    tf_name_to_idx: dict[str, int],
    peak_id_to_idx: dict[str, int],
    drop_missing: bool = True,
) -> pd.DataFrame:
    rows = []
    for tf, peak in true_interactions:
        rows.append((tf, peak, 1))
    for tf, peak in false_interactions:
        rows.append((tf, peak, 0))

    df = pd.DataFrame(rows, columns=["tf_name", "peak_id", "label"])
    df["tf_idx"] = df["tf_name"].map(tf_name_to_idx)
    df["peak_idx"] = df["peak_id"].map(peak_id_to_idx)

    missing_mask = df["tf_idx"].isna() | df["peak_idx"].isna()
    if missing_mask.any():
        n_missing = missing_mask.sum()
        if drop_missing:
            logging.info(f"Dropping {n_missing} interactions with missing TF or peak indices.")
            df = df.loc[~missing_mask].copy()
        else:
            missing_examples = df.loc[missing_mask].head()
            raise ValueError(
                f"{n_missing} interactions are missing TF or peak indices.\n"
                f"Examples:\n{missing_examples}"
            )

    df["tf_idx"] = df["tf_idx"].astype(np.int64)
    df["peak_idx"] = df["peak_idx"].astype(np.int64)
    df["label"] = df["label"].astype(np.float32)

    return df.sample(frac=1.0, random_state=123).reset_index(drop=True)


def load_ordered_tf_embeddings(
    embedding_dir,
    tf_name_to_idx,
    suffix="_protein_embedding.pt",
    weights_only=True,
):
    embedding_dir = Path(embedding_dir)
    available_files = {}

    for path in embedding_dir.glob(f"*{suffix}"):
        tf_name = path.name.replace(suffix, "")
        available_files[tf_name] = path

    n_tfs = len(tf_name_to_idx)
    ordered_tf_names = [None] * n_tfs
    ordered_embeddings = [None] * n_tfs
    ordered_lengths = [0] * n_tfs

    missing_tfs = []
    for tf_name, tf_idx in tf_name_to_idx.items():
        ordered_tf_names[tf_idx] = tf_name

        if tf_name not in available_files:
            missing_tfs.append(tf_name)
            continue

        emb = torch.load(
            available_files[tf_name],
            weights_only=weights_only,
            map_location="cpu",
        )
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
        emb = emb.float()

        ordered_embeddings[tf_idx] = emb
        ordered_lengths[tf_idx] = emb.shape[0]

    if missing_tfs:
        raise FileNotFoundError(
            f"Missing embeddings for {len(missing_tfs)} TFs. "
            f"Examples: {missing_tfs[:20]}"
        )

    lengths = torch.tensor(ordered_lengths, dtype=torch.long)
    embeddings_padded = pad_sequence(
        ordered_embeddings,
        batch_first=True,
        padding_value=0.0,
    )

    max_len = embeddings_padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "tf_names": ordered_tf_names,
        "embeddings": embeddings_padded,
        "lengths": lengths,
        "mask": mask,
    }


def main():
    argparser = argparse.ArgumentParser(description="Build TF-to-DNA training data")
    argparser.add_argument("--pct_true_edges", type=float, default=0.25)
    argparser.add_argument("--true_false_ratio", type=float, default=0.25)
    argparser.add_argument("--force_reload", action="store_true")
    args = argparser.parse_args()
    
    species = config.species

    genome_fasta_path = config.genome_fasta_path
    chrom_sizes_path = config.chrom_sizes_path
    embedding_dir = config.embedding_dir
    
    if not config.tf_dna_input_cache_dir.exists():
        config.tf_dna_input_cache_dir.mkdir(parents=True, exist_ok=True)
        
    # Shared cache files for both TF-to-TG and TF-to-DNA training
    tf_name_to_idx_cache_path = config.tf_name_to_idx_cache_path
    tf_embedding_cache_path = config.tf_embedding_cache_path
    tf_mask_cache_path = config.tf_mask_cache_path
    
    # TF-DNA training specific cache files
    tf_dna_peak_id_to_idx_cache_path = config.tf_dna_peak_id_to_idx_cache_path
    tf_dna_edge_tf_idx_cache_path = config.tf_dna_edge_tf_idx_cache_path
    tf_dna_edge_peak_idx_cache_path = config.tf_dna_edge_peak_idx_cache_path
    tf_dna_edge_labels_cache_path = config.tf_dna_edge_labels_cache_path
    tf_dna_tf_lengths_cache_path = config.tf_dna_tf_lengths_cache_path
    tf_dna_peak_onehot_cache_path = config.tf_dna_peak_onehot_cache_path
    tf_dna_train_idx_cache_path = config.tf_dna_train_idx_cache_path
    tf_dna_val_idx_cache_path = config.tf_dna_val_idx_cache_path
    tf_dna_test_idx_cache_path = config.tf_dna_test_idx_cache_path

    tf_embedding_files = list(embedding_dir.glob("*_protein_embedding.pt"))
    embedded_tf_names = [f.stem.split("_protein_embedding")[0] for f in tf_embedding_files]
    logging.info(f"TFs with embeddings: {len(embedded_tf_names)}")
    logging.info(f"Example TFs with embeddings: {embedded_tf_names[:10]}")
 
    logging.info("Loading true edges...")
    true_edge_file = DATA_DIR / "ground_truth_files" / f"chip_atlas_{species}_all.parquet"
    true_edge_df = pd.read_parquet(true_edge_file)
    true_edge_df = true_edge_df[true_edge_df["source_id"].isin(embedded_tf_names)]
    logging.info(f"    Done. Loaded {len(true_edge_df):,} ChIP-Atlas edges")
    
    # Creates or loads the TF and peak name to index mapping files
    logging.info("Creating TF and peak name to index mappings...")
    map_cache_files = [tf_name_to_idx_cache_path, tf_dna_peak_id_to_idx_cache_path]
    if all(f.exists() for f in map_cache_files) and not args.force_reload:
        logging.info(f"    Loading TF and peak name to index mappings from cache...")
        tf_name_to_idx = pd.read_csv(tf_name_to_idx_cache_path).set_index("tf_name")[
            "tf_idx"
        ].to_dict()
        peak_id_to_idx = pd.read_csv(tf_dna_peak_id_to_idx_cache_path).set_index("peak_id")[
            "peak_idx"
        ].to_dict()
    else:
        tf_names = true_edge_df["source_id"].unique().tolist()
        peak_ids = true_edge_df["peak_id"].unique().tolist()
        
        tf_name_to_idx = {tf: idx for idx, tf in enumerate(tf_names)}
        peak_id_to_idx = {peak: idx for idx, peak in enumerate(peak_ids)}
        pd.DataFrame({"tf_name": list(tf_name_to_idx.keys()), "tf_idx": list(tf_name_to_idx.values())}).to_csv(
            tf_name_to_idx_cache_path,
            index=False,
        )
        pd.DataFrame({"peak_id": list(peak_id_to_idx.keys()), "peak_idx": list(peak_id_to_idx.values())}).to_csv(
            tf_dna_peak_id_to_idx_cache_path,
            index=False,
        )
        
    tf_names = list(tf_name_to_idx.keys())
    peak_ids = list(peak_id_to_idx.keys())
    logging.info(f"    Done. Found {len(tf_names)} unique TFs and {len(peak_ids)} unique peaks in the ChIP-Atlas dataset.")

    # Creates the labeled dataset of TF-peak interactions for training, and caches the edge indices and labels as tensors
    logging.info("Creating or loading labeled dataset of TF-peak interactions")
    edge_cache_files = [tf_dna_edge_tf_idx_cache_path, tf_dna_edge_peak_idx_cache_path, tf_dna_edge_labels_cache_path]
    if all(f.exists() for f in edge_cache_files) and not args.force_reload:
        logging.info(f"    Loading edge indices and labels from cache...")
        edge_tf_idx_tensor = torch.load(tf_dna_edge_tf_idx_cache_path, weights_only=True)
        edge_peak_idx_tensor = torch.load(tf_dna_edge_peak_idx_cache_path, weights_only=True)
        edge_labels_tensor = torch.load(tf_dna_edge_labels_cache_path, weights_only=True)
    else:
        logging.info("    Creating True/False edges")
        true_interactions, false_interactions = utils.create_true_false_edges(
            edge_df=true_edge_df,
            tf_names=tf_names,
            tf_col="source_id",
            item_col="peak_id",
            pct_true_edges=args.pct_true_edges,
            true_false_ratio=args.true_false_ratio,
        )

        logging.info(f"    Creating labeled dataset with {len(true_interactions)} true interactions and {len(false_interactions)} false interactions.")
        tf_peak_labeled_df = create_labeled_tf_peak_dataset(
            true_interactions=true_interactions,
            false_interactions=false_interactions,
            tf_name_to_idx=tf_name_to_idx,
            peak_id_to_idx=peak_id_to_idx,
            drop_missing=False,
        )

        edge_tf_idx = tf_peak_labeled_df["tf_idx"].to_numpy(dtype=np.int64)
        edge_peak_idx = tf_peak_labeled_df["peak_idx"].to_numpy(dtype=np.int64)
        edge_labels = tf_peak_labeled_df["label"].to_numpy(dtype=np.float32)

        edge_tf_idx_tensor = torch.as_tensor(edge_tf_idx, dtype=torch.long)
        edge_peak_idx_tensor = torch.as_tensor(edge_peak_idx, dtype=torch.long)
        edge_labels_tensor = torch.as_tensor(edge_labels, dtype=torch.float32)

        logging.info("    Saving edge indices and labels to cache...")
        torch.save(edge_tf_idx_tensor, tf_dna_edge_tf_idx_cache_path)
        torch.save(edge_peak_idx_tensor, tf_dna_edge_peak_idx_cache_path)
        torch.save(edge_labels_tensor, tf_dna_edge_labels_cache_path)
    logging.info(f"    Done. Created/Loaded dataset with {len(edge_labels_tensor):,} edges.")

    logging.info("Creating or loading ordered TF embeddings...")
    tf_cache_files = [tf_embedding_cache_path, tf_mask_cache_path, tf_dna_tf_lengths_cache_path]
    if all(f.exists() for f in tf_cache_files) and not args.force_reload:
        logging.info(f"    Loading ordered TF embeddings and masks from cache...")
        tf_embeddings_tensor = torch.load(tf_embedding_cache_path, weights_only=True)
        tf_mask_tensor = torch.load(tf_mask_cache_path, weights_only=True)
        tf_lengths_tensor = torch.load(tf_dna_tf_lengths_cache_path, weights_only=True)
    else:
        logging.info("    Ordering TF embeddings to match the TF name to index map...")
        tf_data = load_ordered_tf_embeddings(
            embedding_dir=embedding_dir,
            tf_name_to_idx=tf_name_to_idx,
        )

        tf_embeddings_tensor = tf_data["embeddings"]
        tf_mask_tensor = tf_data["mask"]
        tf_lengths_tensor = tf_data["lengths"]

        logging.info("    Saving TF embeddings and masks to cache...")
        torch.save(tf_embeddings_tensor, tf_embedding_cache_path)
        torch.save(tf_mask_tensor, tf_mask_cache_path)
        torch.save(tf_lengths_tensor, tf_dna_tf_lengths_cache_path)
    logging.info(f"    Done. Created/Loaded TF embeddings with shape {tf_embeddings_tensor.shape} and mask with shape {tf_mask_tensor.shape}.")
        
    logging.info("Creating or loading peak one-hot encodings...")
    peak_ids = list(peak_id_to_idx.keys())
    chrom_sizes = utils.load_chrom_sizes(chrom_sizes_path)

    if os.path.exists(tf_dna_peak_onehot_cache_path) and not args.force_reload:
        logging.info(f"    Loading peak one-hot encodings from cache...")
        peak_tensor = torch.load(tf_dna_peak_onehot_cache_path, weights_only=True)
    else:
        logging.info("    Creating centered peak one-hot encodings...")
        peak_onehot_array = utils.create_centered_peak_onehot_array(
            peak_ids=peak_ids,
            genome_fasta=genome_fasta_path,
            chrom_sizes=chrom_sizes,
            peak_id_to_idx=peak_id_to_idx,
            flank_size=64,
            dtype=np.uint8,
            pad_out_of_bounds=True,
            num_workers=12,
            show_progress=True,
            chunk_size=10000,
        )
        peak_tensor = torch.as_tensor(peak_onehot_array, dtype=torch.uint8)
        peak_tensor = peak_tensor.float()
        torch.save(peak_tensor, tf_dna_peak_onehot_cache_path)
        logging.info("    Done. Created/Loaded peak one-hot encodings.")

    logging.info("Creating or loading train/val/test splits...")
    if all(p.exists() for p in [tf_dna_train_idx_cache_path, tf_dna_val_idx_cache_path, tf_dna_test_idx_cache_path]) and not args.force_reload:
        logging.info(f"    Loading train/val/test splits from cache...")
        train_idx = torch.load(tf_dna_train_idx_cache_path, weights_only=True)
        val_idx = torch.load(tf_dna_val_idx_cache_path, weights_only=True)
        test_idx = torch.load(tf_dna_test_idx_cache_path, weights_only=True)
    else:
        logging.info("    Creating train/val/test splits based on chromosome...")
        peak_id_df = pd.read_csv(tf_dna_peak_id_to_idx_cache_path)
        peak_id_df = peak_id_df.sort_values("peak_idx")
        peak_ids_ordered = peak_id_df["peak_id"].tolist()

        edge_peak_ids = np.array([peak_ids_ordered[idx] for idx in edge_peak_idx_tensor.numpy()])

        def _get_chrom(peak_id: str) -> str:
            return peak_id.split(":", 1)[0]

        edge_chroms = np.array([_get_chrom(pid) for pid in edge_peak_ids])

        train_chroms = {f"chr{i}" for i in range(1, 16)}
        val_chroms = {f"chr{i}" for i in range(16, 18)}
        test_chroms = {f"chr{i}" for i in range(18, 20)}
        
        logging.info(f"    Train chroms: {train_chroms}")
        logging.info(f"    Val chroms: {val_chroms}")
        logging.info(f"    Test chroms: {test_chroms}")

        train_mask = np.isin(edge_chroms, list(train_chroms))
        val_mask = np.isin(edge_chroms, list(val_chroms))
        test_mask = np.isin(edge_chroms, list(test_chroms))

        train_idx = torch.as_tensor(np.where(train_mask)[0], dtype=torch.long)
        val_idx = torch.as_tensor(np.where(val_mask)[0], dtype=torch.long)
        test_idx = torch.as_tensor(np.where(test_mask)[0], dtype=torch.long)

        torch.save(train_idx, tf_dna_train_idx_cache_path)
        torch.save(val_idx, tf_dna_val_idx_cache_path)
        torch.save(test_idx, tf_dna_test_idx_cache_path)
    logging.info(f"    Done. Created/Loaded train/val/test splits with {len(train_idx)}, {len(val_idx)}, and {len(test_idx)} edges, respectively.")

    logging.info("\nFinished building TF-to-DNA training data")
    logging.info(f"  - Cached TF embeddings: {tf_embeddings_tensor.shape}")
    logging.info(f"  - Cached peak tensor: {peak_tensor.shape}")
    logging.info(f"  - Train/Val/Test edges: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")


if __name__ == "__main__":
    main()
