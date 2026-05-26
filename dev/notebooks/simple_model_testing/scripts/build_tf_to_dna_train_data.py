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
    weights_only=False,
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
    argparser.add_argument("--training_data_dir", type=str, required=False, help="Path to directory containing training data cache files (if not using default)")
    argparser.add_argument("--pct_true_edges", type=float, default=0.25)
    argparser.add_argument("--true_false_ratio", type=float, default=0.25)
    argparser.add_argument("--force_reload", action="store_true")
    args = argparser.parse_args()

    genome_fasta_path = DATA_DIR / "genome_data" / "reference_genome" / "mm10" / "mm10.fa"
    chrom_sizes_path = DATA_DIR / "genome_data" / "reference_genome" / "mm10" / "mm10.chrom.sizes"
    embedding_dir = PROJECT_DIR / "data" / "tf_data" / "tf_embeddings"
    
    training_data_dir = args.training_data_dir

    if training_data_dir:
        training_cache_dir = Path(training_data_dir)
    else:
        training_cache_dir = PROJECT_DIR / "data" / "training_data_cache"
    training_cache_dir.mkdir(exist_ok=True, parents=True)

    tf_name_to_idx_cache_path = training_cache_dir / "tf_name_to_idx.csv"
    peak_id_to_idx_cache_path = training_cache_dir / "peak_id_to_idx.csv"
    edge_tf_idx_cache_path = training_cache_dir / "edge_tf_idx.pt"
    edge_peak_idx_cache_path = training_cache_dir / "edge_peak_idx.pt"
    edge_labels_cache_path = training_cache_dir / "edge_labels.pt"
    tf_embedding_cache_path = training_cache_dir / "tf_embeddings.pt"
    tf_mask_cache_path = training_cache_dir / "tf_masks.pt"
    tf_lengths_cache_path = training_cache_dir / "tf_lengths.pt"
    peak_onehot_cache_path = training_cache_dir / "peak_onehot_array.pt"
    train_idx_cache_path = training_cache_dir / "train_idx.pt"
    val_idx_cache_path = training_cache_dir / "val_idx.pt"
    test_idx_cache_path = training_cache_dir / "test_idx.pt"

    tf_embedding_files = list(embedding_dir.glob("*_protein_embedding.pt"))
    embedded_tf_names = [f.stem.split("_protein_embedding")[0] for f in tf_embedding_files]
    logging.info(f"TFs with embeddings: {len(embedded_tf_names)}")
    logging.info(f"Example TFs with embeddings: {embedded_tf_names[:10]}")
 
    logging.info("Loading true edges...")
    true_edge_file = DATA_DIR / "ground_truth_files" / "chip_atlas_mm10_all_sample.csv"
    true_edge_df = pd.read_csv(true_edge_file)
    true_edge_df = true_edge_df[true_edge_df["source_id"].isin(embedded_tf_names)]
    logging.info("  - Done loading edges")
    
    tf_names = true_edge_df["source_id"].unique().tolist()
    peak_ids = true_edge_df["peak_id"].unique().tolist()

    map_cache_files = [tf_name_to_idx_cache_path, peak_id_to_idx_cache_path]
    if all(f.exists() for f in map_cache_files) and not args.force_reload:
        tf_name_to_idx = pd.read_csv(tf_name_to_idx_cache_path).set_index("tf_name")[
            "tf_idx"
        ].to_dict()
        peak_id_to_idx = pd.read_csv(peak_id_to_idx_cache_path).set_index("peak_id")[
            "peak_idx"
        ].to_dict()
    else:
        tf_name_to_idx = {tf: idx for idx, tf in enumerate(tf_names)}
        peak_id_to_idx = {peak: idx for idx, peak in enumerate(peak_ids)}
        pd.DataFrame({"tf_name": list(tf_name_to_idx.keys()), "tf_idx": list(tf_name_to_idx.values())}).to_csv(
            tf_name_to_idx_cache_path,
            index=False,
        )
        pd.DataFrame({"peak_id": list(peak_id_to_idx.keys()), "peak_idx": list(peak_id_to_idx.values())}).to_csv(
            peak_id_to_idx_cache_path,
            index=False,
        )

    edge_cache_files = [edge_tf_idx_cache_path, edge_peak_idx_cache_path, edge_labels_cache_path]
    if all(f.exists() for f in edge_cache_files) and not args.force_reload:
        edge_tf_idx_tensor = torch.load(edge_tf_idx_cache_path)
        edge_peak_idx_tensor = torch.load(edge_peak_idx_cache_path)
        edge_labels_tensor = torch.load(edge_labels_cache_path)
    else:
        true_interactions, false_interactions = utils.create_true_false_edges(
            edge_df=true_edge_df,
            tf_names=tf_names,
            tf_col="source_id",
            item_col="peak_id",
            pct_true_edges=args.pct_true_edges,
            true_false_ratio=args.true_false_ratio,
        )

        logging.info(f"Creating labeled dataset with {len(true_interactions)} true interactions and {len(false_interactions)} false interactions.")
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

        logging.info("Saving edge indices and labels to cache...")
        torch.save(edge_tf_idx_tensor, edge_tf_idx_cache_path)
        torch.save(edge_peak_idx_tensor, edge_peak_idx_cache_path)
        torch.save(edge_labels_tensor, edge_labels_cache_path)

    tf_cache_files = [tf_embedding_cache_path, tf_mask_cache_path, tf_lengths_cache_path]
    if all(f.exists() for f in tf_cache_files) and not args.force_reload:
        tf_embeddings_tensor = torch.load(tf_embedding_cache_path)
        tf_mask_tensor = torch.load(tf_mask_cache_path)
        tf_lengths_tensor = torch.load(tf_lengths_cache_path)
    else:
        tf_data = load_ordered_tf_embeddings(
            embedding_dir=embedding_dir,
            tf_name_to_idx=tf_name_to_idx,
        )

        tf_embeddings_tensor = tf_data["embeddings"]
        tf_mask_tensor = tf_data["mask"]
        tf_lengths_tensor = tf_data["lengths"]

        logging.info("Saving TF embeddings and masks to cache...")
        torch.save(tf_embeddings_tensor, tf_embedding_cache_path)
        torch.save(tf_mask_tensor, tf_mask_cache_path)
        torch.save(tf_lengths_tensor, tf_lengths_cache_path)

    peak_ids = list(peak_id_to_idx.keys())
    chrom_sizes = utils.load_chrom_sizes(chrom_sizes_path)

    if os.path.exists(peak_onehot_cache_path) and not args.force_reload:
        peak_tensor = torch.load(peak_onehot_cache_path)
    else:
        logging.info("Creating centered peak one-hot encodings...")
        peak_onehot_array = utils.create_centered_peak_onehot_array(
            peak_ids=peak_ids,
            genome_fasta=genome_fasta_path,
            chrom_sizes=chrom_sizes,
            peak_id_to_idx=peak_id_to_idx,
            flank_size=128,
            dtype=np.float32,
            pad_out_of_bounds=True,
            num_workers=12,
            show_progress=True,
        )
        peak_tensor = torch.as_tensor(peak_onehot_array, dtype=torch.float32)
        torch.save(peak_tensor, peak_onehot_cache_path)

    if all(p.exists() for p in [train_idx_cache_path, val_idx_cache_path, test_idx_cache_path]) and not args.force_reload:
        train_idx = torch.load(train_idx_cache_path)
        val_idx = torch.load(val_idx_cache_path)
        test_idx = torch.load(test_idx_cache_path)
    else:
        logging.info("Creating train/val/test splits based on chromosome...")
        peak_id_df = pd.read_csv(peak_id_to_idx_cache_path)
        peak_id_df = peak_id_df.sort_values("peak_idx")
        peak_ids_ordered = peak_id_df["peak_id"].tolist()

        edge_peak_ids = np.array([peak_ids_ordered[idx] for idx in edge_peak_idx_tensor.numpy()])

        def _get_chrom(peak_id: str) -> str:
            return peak_id.split(":", 1)[0]

        edge_chroms = np.array([_get_chrom(pid) for pid in edge_peak_ids])

        train_chroms = {f"chr{i}" for i in range(1, 16)}
        val_chroms = {f"chr{i}" for i in range(16, 18)}
        test_chroms = {"chr18", "chr19"}

        train_mask = np.isin(edge_chroms, list(train_chroms))
        val_mask = np.isin(edge_chroms, list(val_chroms))
        test_mask = np.isin(edge_chroms, list(test_chroms))

        train_idx = torch.as_tensor(np.where(train_mask)[0], dtype=torch.long)
        val_idx = torch.as_tensor(np.where(val_mask)[0], dtype=torch.long)
        test_idx = torch.as_tensor(np.where(test_mask)[0], dtype=torch.long)

        torch.save(train_idx, train_idx_cache_path)
        torch.save(val_idx, val_idx_cache_path)
        torch.save(test_idx, test_idx_cache_path)

    logging.info("\nFinished building TF-to-DNA training data")
    logging.info(f"  - Cached TF embeddings: {tf_embeddings_tensor.shape}")
    logging.info(f"  - Cached peak tensor: {peak_tensor.shape}")
    logging.info(f"  - Train/Val/Test edges: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")


if __name__ == "__main__":
    main()
