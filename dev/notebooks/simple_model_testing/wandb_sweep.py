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

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import torch
import argparse
import wandb

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
sys.path.append(str(PROJECT_DIR))

import utils
import config
import scripts.build_tf_to_tg_train_data as build_tf_to_tg_train_data
import scripts.train_tf_to_tg_model as train_tf_to_tg_model
import models.tf_to_dna as tf_to_dna_module
import models.tf_to_tg as tf_to_tg_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import hashlib

SWEEP_PARAMETER_NAMES = (
    "epochs",
    "d_model",
    "tf_peak_chunk_size",
    "batch_size",
    "num_gpus",
    "num_nodes",
    "max_peaks_per_tg",
    "max_cells_per_pair",
    "pct_true_edges",
    "true_false_ratio",
    "peak_flank_size",
)

def get_sweep_setting_hash(
    max_peaks_per_tg: int,
    max_cells_per_pair: int,
    pct_true_edges: float,
    true_false_ratio: float,
    peak_flank_size: int,
) -> str:
    sweep_setting_cache_string = (
        f"{max_peaks_per_tg}_{max_cells_per_pair}_"
        f"{pct_true_edges}_{true_false_ratio}_{peak_flank_size}"
    )
    return hashlib.md5(sweep_setting_cache_string.encode("utf-8")).hexdigest()


def _coerce_sweep_value(
    name: str,
    cli_value,
    config_value,
    cast,
):
    """Prefer W&B config values and tolerate unresolved ${...} CLI placeholders."""
    value = config_value if config_value is not None else cli_value

    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if value.startswith("${") and value.endswith("}"):
            return None

    try:
        return cast(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for {name}: {value}") from exc



def create_new_tf_tg_regulation_model(
    tf_bind_model_path: Path,
    tf_embeddings_tensor: torch.Tensor,
    tf_mask_tensor: torch.Tensor,
    checkpoint_path: Path | None = None,
    d_model: int = 128,
    tf_peak_chunk_size: int = 128,
    
) -> tf_to_tg_module.TFTGRegulationModel:

    # 1) Recreate the base TF→DNA model with the same hyperparameters
    base_model = tf_to_dna_module.TFPeakBindingModel(
        tf_embedding_dim=128,
        hidden_dim=128,
        dropout=0.3,
        num_layers=4,
        num_heads=4,
        dim_head=32,
    )

    # 2) Wrap in Lightning module and load checkpoint
    lit_model = tf_to_dna_module.LitTFPeakBindingModel.load_from_checkpoint(
        checkpoint_path=tf_bind_model_path,
        model=base_model,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
    )

    # 3) Get the trained base model and freeze it
    trained_tf_peak_model = lit_model.model

    trained_tf_peak_model.eval()

    for p in trained_tf_peak_model.parameters():
        p.requires_grad = False

    trained_tf_peak_model = torch.compile(
        trained_tf_peak_model,
        mode="reduce-overhead",
        fullgraph=False,
    )

    # 4) Inject into your TF→TG model
    tf_tg_model = tf_to_tg_module.TFTGRegulationModel(
        pretrained_tf_peak_model=trained_tf_peak_model,
        d_model=d_model,
        tf_peak_chunk_size=tf_peak_chunk_size,
    )

    # 5) Optionally load TF→TG checkpoint
    if checkpoint_path is not None:
        logging.info(f"Loading TF→TG model weights from checkpoint: {checkpoint_path}")

        tf_tg_ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        fixed = {}

        for key, value in tf_tg_ckpt["state_dict"].items():
            if key.startswith("model."):
                key = key[len("model."):]
            fixed[key] = value

        tf_tg_model.load_state_dict(fixed, strict=True)

    return tf_tg_model

def build_tf_tg_input_cache(
    sample_name: str,
    cell_type: str,
    species: str,
    max_peaks_per_tg: int,
    max_cells_per_pair: int,
    pct_true_edges: float,
    true_false_ratio: float,
    peak_flank_size: int,
    num_cpu: int,
    force_reload: bool,
    sample_pairs: int | None = 100_000,
):
    
    if species == "mm10":
        gene_ref_file = DATA_DIR / "genome_data" / "genome_annotation" / "mm10" / "Mus_musculus.GRCm39.115.gtf.gz"
    elif species == "hg38":
        gene_ref_file = DATA_DIR / "genome_data" / "genome_annotation" / "hg38" / "Homo_sapiens.GRCh38.113.gtf.gz"
        
    genome_fasta_path = DATA_DIR / "genome_data" / "reference_genome" / species / f"{species}.fa"
    chrom_sizes_path = DATA_DIR / "genome_data" / "reference_genome" / species / f"{species}.chrom.sizes"
        
    assert gene_ref_file.exists(), f"Gene reference file not found: {gene_ref_file}"
    assert genome_fasta_path.exists(), f"Genome FASTA file not found: {genome_fasta_path}"
    assert chrom_sizes_path.exists(), f"Chromosome sizes file not found: {chrom_sizes_path}"
    
    training_cache_dir = PROJECT_DIR / "data" / f"{cell_type}_cache"
    # Create the training cache directory if it doesn't exist
    input_data_dir = Path(PROJECT_DIR / "data" / "sample_input_data" / cell_type / sample_name)
    
    assert input_data_dir.exists(), f"Input data directory does not exist: {input_data_dir}"
    
    # Encode the sweep settings into a string and hash it to create a unique cache directory for this sweep configuration
    sweep_setting_hash = get_sweep_setting_hash(
        max_peaks_per_tg=max_peaks_per_tg,
        max_cells_per_pair=max_cells_per_pair,
        pct_true_edges=pct_true_edges,
        true_false_ratio=true_false_ratio,
        peak_flank_size=peak_flank_size,
    )
    
    tf_tg_input_cache_dir = training_cache_dir / "tf_tg_training_cache" / sample_name / "wandb_sweep" / f"tf_tg_sweep_{sweep_setting_hash}"
    tf_tg_input_cache_dir.mkdir(parents=True, exist_ok=True)

    tf_name_to_idx_cache_path = training_cache_dir / "tf_name_to_idx.csv"
    tf_embedding_cache_path = training_cache_dir / "tf_embeddings.pt"
    tf_mask_cache_path = training_cache_dir / "tf_masks.pt"
    merged_ground_truth_path = training_cache_dir / f"{cell_type}_merged_ground_truth.parquet"
    
    atac_peak_onehot_cache_path = tf_tg_input_cache_dir / "atac_peak_tensor.pt"
    train_file = tf_tg_input_cache_dir / "tftg_inputs_train.pt"
    val_file = tf_tg_input_cache_dir / "tftg_inputs_val.pt"
    # test_file = tf_tg_input_cache_dir / "tftg_inputs_test.pt"
    
    metadata_file = tf_tg_input_cache_dir / "metadata.json"
    manifest_file = tf_tg_input_cache_dir / "manifest.json"
    
    required_cache_files = [
        tf_name_to_idx_cache_path,
        tf_embedding_cache_path,
        tf_mask_cache_path,
        atac_peak_onehot_cache_path,
        train_file,
        val_file,
        # test_file,
        metadata_file,
        manifest_file,
    ]
    
    if all(f.exists() for f in required_cache_files) and not force_reload:
        logging.info("All required cache files already exist. Skipping construction (use --force_reload to override).")
        return sweep_setting_hash

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
    valid_peak_ids = set(peak_to_gene_distance["peak_id"])
    atac_pseudobulk = atac_pseudobulk.loc[atac_pseudobulk.index.isin(valid_peak_ids)].copy()
    logging.info(f"ATAC peaks AFTER peak-to-gene filtering: {atac_pseudobulk.shape[0]:,}")
    rna_pseudobulk_norm = rna_pseudobulk.copy()
    rna_pseudobulk_norm.index = rna_pseudobulk_norm.index.str.upper()

    # Load and merge the ground truth files, or load from cache if already merged
    if not merged_ground_truth_path.exists() or force_reload:

        merged_ground_truth_df: pd.DataFrame = utils.load_ground_truth_files(
            config.gt_by_dataset_dict[cell_type]
            )
        
        merged_ground_truth_df["Source"] = merged_ground_truth_df["Source"].str.upper()
        merged_ground_truth_df["Target"] = merged_ground_truth_df["Target"].str.upper()

        gt_tfs_in_rna = set(merged_ground_truth_df["Source"]).intersection(rna_pseudobulk_norm.index)
        gt_tgs_in_rna = set(merged_ground_truth_df["Target"]).intersection(rna_pseudobulk_norm.index)
        logging.info(f"Ground truth TFs in RNA pseudobulk: {len(gt_tfs_in_rna)}")
        logging.info(f"Ground truth TGs in RNA pseudobulk: {len(gt_tgs_in_rna)}")

        n_before_rna_filter = len(merged_ground_truth_df)
        merged_ground_truth_df = merged_ground_truth_df[
            merged_ground_truth_df["Source"].isin(gt_tfs_in_rna) &
            merged_ground_truth_df["Target"].isin(gt_tgs_in_rna)
        ].copy()
        logging.info(
            f"Ground truth edges after RNA TF/TG filtering: {len(merged_ground_truth_df):,} / {n_before_rna_filter:,}"
        )

    else:
        merged_ground_truth_df = pd.read_parquet(merged_ground_truth_path)

    merged_ground_truth_df["Source"] = merged_ground_truth_df["Source"].str.upper()
    merged_ground_truth_df["Target"] = merged_ground_truth_df["Target"].str.upper()

    gt_tfs_in_rna = set(merged_ground_truth_df["Source"]).intersection(rna_pseudobulk_norm.index)
    gt_tgs_in_rna = set(merged_ground_truth_df["Target"]).intersection(rna_pseudobulk_norm.index)
    logging.info(f"Ground truth TFs in RNA pseudobulk: {len(gt_tfs_in_rna)}")
    logging.info(f"Ground truth TGs in RNA pseudobulk: {len(gt_tgs_in_rna)}")

    n_before_rna_filter = len(merged_ground_truth_df)
    merged_ground_truth_df = merged_ground_truth_df[
        merged_ground_truth_df["Source"].isin(gt_tfs_in_rna) &
        merged_ground_truth_df["Target"].isin(gt_tgs_in_rna)
    ].copy()
    logging.info(
        f"Ground truth edges after RNA TF/TG filtering: {len(merged_ground_truth_df):,} / {n_before_rna_filter:,}"
    )

    merged_ground_truth_df.to_parquet(merged_ground_truth_path, index=False)
    
    # Get the map of TF name to index
    tf_name_to_idx = pd.read_csv(tf_name_to_idx_cache_path)
    tf_name_to_idx["tf_name"] = tf_name_to_idx["tf_name"].str.upper()
    tf_name_to_idx = tf_name_to_idx.set_index("tf_name")["tf_idx"].to_dict()

    gt_tfs_in_embeddings = set(tf_name_to_idx.keys()).intersection(merged_ground_truth_df["Source"])
    n_before_tf_embedding_filter = len(merged_ground_truth_df)
    merged_ground_truth_df = merged_ground_truth_df[
        merged_ground_truth_df["Source"].isin(gt_tfs_in_embeddings)
    ].copy()
    logging.info(
        f"Ground truth edges after filtering to TFs with embeddings: {len(merged_ground_truth_df):,} / {n_before_tf_embedding_filter:,}"
    )

    tg_id_to_idx = {tg: idx for idx, tg in enumerate(merged_ground_truth_df["Target"].unique())}
    
    if species == "mm10":
        train_chroms = [str(i) for i in range(1, 16)]
        val_chroms = [ str(i) for i in range(16, 18)]
        test_chroms = [str(i) for i in range(18, 20)]
    elif species == "hg38":
        train_chroms = [str(i) for i in range(1, 18)]
        val_chroms = [str(i) for i in range(18, 20)]
        test_chroms = [str(i) for i in range(20, 23)]

    # Split genes into train/val/test based on chromosome
    train_genes, val_genes, test_genes = build_tf_to_tg_train_data.split_genes_by_chromosome(
        gene_ref_file,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=test_chroms
        )
    gt_train_df, gt_val_df, _ = build_tf_to_tg_train_data.create_train_val_test_splits(
        merged_ground_truth_df, train_genes, val_genes, test_genes
    )

    # Create labeled TF-TG datasets for train/val/test splits
    # (samples true and false edges according to pct_true_edges and true_false_ratio)
    tf_tg_labeled_train_df = build_tf_to_tg_train_data._create_labeled_df(
        gt_train_df,
        pct_true_edges,
        true_false_ratio,
        seed=123,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
    )
    tf_tg_labeled_val_df = build_tf_to_tg_train_data._create_labeled_df(
        gt_val_df,
        pct_true_edges,
        true_false_ratio,
        seed=123,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
    )
    
    # tf_tg_labeled_test_df = build_tf_to_tg_train_data._create_labeled_df(
    #     gt_test_df,
    #     pct_true_edges,
    #     true_false_ratio,
    #     seed=123,
    #     tf_name_to_idx=tf_name_to_idx,
    #     tg_id_to_idx=tg_id_to_idx,
    # )

    # Create a map of ATAC peaks to indices in the pseudobulk matrix, filtering to valid chromosomes
    dataset_peaks = atac_pseudobulk.index.to_list()
    if species == "mm10":
        valid_chroms = {f"chr{i}" for i in range(1, 20)}
    else:
        valid_chroms = {f"chr{i}" for i in range(1, 23)}
    dataset_peaks = [peak for peak in dataset_peaks if peak.split(":", 1)[0] in valid_chroms]
    atac_peak_map = {peak: idx for idx, peak in enumerate(dataset_peaks)}

    # Load cached TF embeddings and masks from TF-DNA model training
    tf_embeddings_tensor = torch.load(tf_embedding_cache_path, weights_only=True)
    tf_mask_tensor = torch.load(tf_mask_cache_path, weights_only=True)

    # Create or load cached one-hot encodings for ATAC peaks
    # One-hot encodings use ACGT order and uses 'flank_size' bp upstream and downstream of the peak center.
    dataset_peaks = list(atac_peak_map.keys())
    if os.path.exists(atac_peak_onehot_cache_path):
        atac_peak_tensor = torch.load(atac_peak_onehot_cache_path, weights_only=True)
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

    common_cells = sorted(set(rna_pseudobulk_norm.columns) & set(atac_pseudobulk.columns))
    peak_to_gene = peak_to_gene_distance.copy()
    peak_to_gene["target_id_norm"] = peak_to_gene["target_id"].str.upper()

    tg_to_peak_info, cell_to_idx, atac_mat, rna_mat, gene_to_rna_idx = build_tf_to_tg_train_data.prepare_tftg_lookup_tables(
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
    
    if sample_pairs is not None:
        logging.info(f"Sampling {args.sample_pairs} TF-TG pairs from each of train/val/test splits")
        tf_tg_labeled_train_df = _sample_df(tf_tg_labeled_train_df, n=sample_pairs, seed=123)
        tf_tg_labeled_val_df = _sample_df(tf_tg_labeled_val_df, n=sample_pairs, seed=123)
        # tf_tg_labeled_test_df = _sample_df(tf_tg_labeled_test_df, n=sample_pairs, seed=123)
    
    tf_tg_df = pd.concat([tf_tg_labeled_train_df, tf_tg_labeled_val_df], ignore_index=True)
    if tf_tg_df.empty:
        raise ValueError(
            "No labeled TF-TG pairs were created across train/val/test. "
            "Check RNA filtering, TF embedding filtering, chromosome splits, and ground truth overlap."
        )

    max_peaks_real = max(len(tg_to_peak_info.get(tg_name, {}).get("peak_indices", [])) for tg_name in tf_tg_df["tg_id"])
    n_tgs_with_peaks = sum(len(tg_to_peak_info.get(tg, {}).get("peak_indices", [])) > 0 for tg in tf_tg_df["tg_id"].unique())
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
    
    if all(f.exists() for f in [train_file, val_file]) and not force_reload:
        logging.info("Cached input files already exist. Skipping (use --force_reload to override).")
        return sweep_setting_hash
    
    logging.info("\nBuilding training inputs")
    tftg_inputs_train = build_tf_to_tg_train_data.build_tftg_inputs(
        tf_tg_labeled_train_df,
        seed=123,
        **common_build_kwargs,
    )

    logging.info("\nBuilding validation inputs")
    tftg_inputs_val = build_tf_to_tg_train_data.build_tftg_inputs(
        tf_tg_labeled_val_df,
        seed=124,
        **common_build_kwargs,
    )
    
    # logging.info("\nBuilding test inputs")
    # tftg_inputs_test = build_tf_to_tg_train_data.build_tftg_inputs(
    #     tf_tg_labeled_test_df,
    #     seed=125,
    #     **common_build_kwargs,
    # )

    # Save compact split inputs
    torch.save(tftg_inputs_train, train_file)
    torch.save(tftg_inputs_val, val_file)
    # torch.save(tftg_inputs_test, test_file)

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
        # "n_test_rows": int(len(tftg_inputs_test["label"])),
    }

    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    logging.info(f"Wrote training data and metadata to {tf_tg_input_cache_dir}")
    
    return sweep_setting_hash

def make_dataloader(dataset, *, batch_size, shuffle, num_workers, prefetch_factor):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=train_tf_to_tg_model.collate_tftg_edge_bags,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)

def train_tf_tg_model(
    sample_name: str,
    cell_type: str,
    species: str,
    sweep_setting_hash: str,
    checkpoint_path: str | Path | None,
    max_peaks_per_tg: int,
    max_cells_per_pair: int,
    pct_true_edges: float,
    true_false_ratio: float,
    peak_flank_size: int,
    num_gpus: int,
    num_nodes: int,
    epochs: int = 50,
    batch_size: int = 128,
    wandb_config: dict[str, object] | None = None,
):
        
    output_dir = PROJECT_DIR / "checkpoints" / f"{cell_type}" / f"{sample_name}" / "wandb_sweep" / f"tf_tg_train_{sample_name}_{sweep_setting_hash}"
    
    run_name = f"tf_tg_{sample_name}_{sweep_setting_hash}"
    
    training_cache_dir = PROJECT_DIR / "data" / f"{cell_type}_cache"
    tf_tg_input_cache_dir = training_cache_dir / "tf_tg_training_cache" / sample_name / "wandb_sweep" / f"tf_tg_sweep_{sweep_setting_hash}"
    
    # Load the trained TF embedding and mask tensors from the TF→DNA model cache 
    # (these are needed for the TF→TG model since it uses the pretrained TF peak embedding module)
    tf_embeddings_tensor = torch.load(
        training_cache_dir / "tf_embeddings.pt",
        weights_only=True,
    )
    tf_mask_tensor = torch.load(
        training_cache_dir / "tf_masks.pt",
        weights_only=True,
    )
    
    # TF-TG training specific cache files
    tf_tg_atac_peak_cache_path = tf_tg_input_cache_dir / "atac_peak_tensor.pt"
    tf_tg_metadata_cache_path = tf_tg_input_cache_dir / "metadata.json"
    tf_tg_manifest_cache_path = tf_tg_input_cache_dir / "manifest.json"
    tf_tg_train_cache_path = tf_tg_input_cache_dir / "tftg_inputs_train.pt"
    tf_tg_val_cache_path = tf_tg_input_cache_dir / "tftg_inputs_val.pt"
    # tf_tg_test_cache_path = tf_tg_input_cache_dir / "tftg_inputs_test.pt"
    
    # Load the train/val/test splits of the compact TF-TG input tensors 
    # that were preprocessed and cached by the data preprocessing script
    tftg_inputs_train = torch.load(
        tf_tg_train_cache_path,
        weights_only=False,
    )
    tftg_inputs_val = torch.load(
        tf_tg_val_cache_path,
        weights_only=False,
    )
    # tftg_inputs_test = torch.load(
    #     tf_tg_test_cache_path,
    #     weights_only=False,
    # )

    atac_peak_tensor = torch.load(
        tf_tg_atac_peak_cache_path,
        weights_only=True,
    )
    
    # Re-create the datasets and dataloaders using the loaded compact inputs and lookup tensors
    train_dataset = train_tf_to_tg_model.TFTGEdgeBagDataset(
        tftg_inputs_train,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor
    )

    val_dataset = train_tf_to_tg_model.TFTGEdgeBagDataset(
        tftg_inputs_val,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor

    )

    # test_dataset = train_tf_to_tg_model.TFTGEdgeBagDataset(
    #     tftg_inputs_test,
    #     tf_embeddings_tensor=tf_embeddings_tensor,
    #     tf_mask_tensor=tf_mask_tensor,
    #     atac_peak_tensor=atac_peak_tensor
    # )

    # Create the DataLoaders with the tested batching path from the multigpu-safe script
    train_loader = make_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        prefetch_factor=4,
    )

    val_loader = make_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
    )

    # test_loader = make_dataloader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=2,
    #     prefetch_factor=2,
    # )

    train_tf_to_tg_model.log_once(f"Train/Val sizes: {len(train_dataset)}, {len(val_dataset)}")

    tf_bind_model_path = config.tf_dna_model_checkpoints[cell_type]
    
    tf_tg_model = create_new_tf_tg_regulation_model(
        tf_bind_model_path=Path(tf_bind_model_path),
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        d_model=args.d_model,
        tf_peak_chunk_size=args.tf_peak_chunk_size,
        )

    pooling_mode = "lse"
    pooling_temperature = 1.0

    train_tf_to_tg_model.log_once("\nStarting Lightning training...")

    lit_model = tf_to_tg_module.LitTFTGRegulationModel(
        model=tf_tg_model,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
        pooling_mode=pooling_mode,
        pooling_temperature=pooling_temperature,
        enable_timing_sync=True,
    )
    
    checkpoint_callback = train_tf_to_tg_model.ModelCheckpoint(
        dirpath=output_dir,
        filename="epoch={epoch:02d}-val_auroc={val/auroc:.4f}-val_loss={val/loss:.4f}",
        monitor="val/auroc",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    
    early_stopping_callback = train_tf_to_tg_model.EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=15,
    )

    lr_monitor = train_tf_to_tg_model.LearningRateMonitor(logging_interval="epoch")

    wandb_logger = train_tf_to_tg_model.WandbLogger(
        project="tf_tg_regulation_prediction",
        name=run_name,
        save_dir=output_dir,
        config=wandb_config,
    )

    wandb_logger.log_hyperparams({
        "sample_name": sample_name,
        "cell_type": cell_type,
        "species": species,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_batches": len(train_loader),
        "num_gpus": num_gpus,
        "num_nodes": num_nodes,
        "run_name": run_name,
        "max_peaks_per_tg": max_peaks_per_tg,
        "max_cells_per_pair": max_cells_per_pair,
        "pct_true_edges": pct_true_edges,
        "true_false_ratio": true_false_ratio,
        "pooling_mode": pooling_mode,
        "pooling_temperature": pooling_temperature,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "flank_size": peak_flank_size,
        "max_precompute_peaks": max_peaks_per_tg,
        "persistent_workers": True,
    })
    
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            os.environ.get("SLURM_NTASKS", "1"),
        )
    )

    use_ddp = world_size > 1
    
    train_tf_to_tg_model.log_once(f"Num GPUs: {world_size} | Batch size: {batch_size}")
    train_tf_to_tg_model.log_once(f"Num steps per epoch: {len(train_loader)}")
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_gpus,
        num_nodes=num_nodes,
        strategy="ddp" if use_ddp else "auto",
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            train_tf_to_tg_model.TQDMProgressBar(refresh_rate=50),
            checkpoint_callback,
            early_stopping_callback,
            lr_monitor,
        ],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
        default_root_dir=output_dir,
        enable_progress_bar=True,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )
    
    trainer.fit(
        lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sample_name", type=str, help="Sample name for training (e.g., 'E7.5_rep1')")
    parser.add_argument("--d_model", type=str, default="128", help="Dimension of the model (default: 128)")
    parser.add_argument("--tf_peak_chunk_size", type=str, default="64", help="Chunk size for TF peak embeddings (default: 64)")
    parser.add_argument("--epochs", type=str, default="25", help="Number of training epochs")
    parser.add_argument("--num_gpus", type=str, default="1", help="Number of GPU devices to use for training")
    parser.add_argument("--num_nodes", type=str, default="1", help="Number of nodes to use for training")
    parser.add_argument("--max_peaks_per_tg", type=str, default="64", help="Maximum number of peaks to consider per TG (default: 64)")
    parser.add_argument("--max_cells_per_pair", type=str, default="8", help="Maximum number of cells to sample per TF-TG pair (default: 8)")
    parser.add_argument("--batch_size", type=str, default="128", help="Batch size for training (default: 32)")
    parser.add_argument("--pct_true_edges", type=str, default="0.05", help="Percentage of true edges to include in the training set (default: 0.15)")
    parser.add_argument("--true_false_ratio", type=str, default="1.0", help="Ratio of true to false edges in the training set (default: 2.0)")
    parser.add_argument("--peak_flank_size", type=str, default="64", help="Size of the flank region around each peak (default: 64)")
    parser.add_argument("--num_cpu", type=int, default=8, help="Number of CPU workers to use for preprocessing")
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to a model checkpoint to resume training from")
    parser.add_argument("--force_reload", action="store_true", help="Whether to force reload cached data instead of using existing cache files")
    parser.add_argument("--sample_pairs", type=int, default=100_000, help="Number of TF-TG pairs to sample for training (default: 10,000)")
    args = parser.parse_args()

    sweep_run = wandb.init(
        project="tf_tg_regulation_prediction",
        config=vars(args),
        job_type="tf_tg_sweep",
    )
    wandb.define_metric("val/auroc.max", summary="max")
    run_config = dict(sweep_run.config)

    for parameter_name in SWEEP_PARAMETER_NAMES:
        if parameter_name in run_config and run_config[parameter_name] is not None:
            setattr(args, parameter_name, run_config[parameter_name])

    args.sample_name = _coerce_sweep_value("sample_name", args.sample_name, run_config.get("sample_name"), str)
    args.d_model = _coerce_sweep_value("d_model", args.d_model, run_config.get("d_model"), int)
    args.tf_peak_chunk_size = _coerce_sweep_value("tf_peak_chunk_size", args.tf_peak_chunk_size, run_config.get("tf_peak_chunk_size"), int)
    args.epochs = _coerce_sweep_value("epochs", args.epochs, run_config.get("epochs"), int)
    args.batch_size = _coerce_sweep_value("batch_size", args.batch_size, run_config.get("batch_size"), int)
    args.num_gpus = _coerce_sweep_value("num_gpus", args.num_gpus, run_config.get("num_gpus"), int)
    args.num_nodes = _coerce_sweep_value("num_nodes", args.num_nodes, run_config.get("num_nodes"), int)
    args.max_peaks_per_tg = _coerce_sweep_value("max_peaks_per_tg", args.max_peaks_per_tg, run_config.get("max_peaks_per_tg"), int)
    args.max_cells_per_pair = _coerce_sweep_value("max_cells_per_pair", args.max_cells_per_pair, run_config.get("max_cells_per_pair"), int)
    args.pct_true_edges = _coerce_sweep_value("pct_true_edges", args.pct_true_edges, run_config.get("pct_true_edges"), float)
    args.true_false_ratio = _coerce_sweep_value("true_false_ratio", args.true_false_ratio, run_config.get("true_false_ratio"), float)
    args.peak_flank_size = _coerce_sweep_value("peak_flank_size", args.peak_flank_size, run_config.get("peak_flank_size"), int)
    args.force_reload = args.force_reload or os.environ.get("FORCE_RELOAD", "").lower() in {"1", "true", "yes", "on"}

    max_peaks_per_tg = args.max_peaks_per_tg
    max_cells_per_pair = args.max_cells_per_pair
    pct_true_edges = args.pct_true_edges
    true_false_ratio = args.true_false_ratio
    peak_flank_size = int(args.peak_flank_size)
    
    sample_to_cell_type_species = {
        "E7.5_rep1": ("mESC", "mm10"),
        "E8.5_rep1": ("mESC", "mm10"),
        "hepatocytes_1": ("mouse_hepatocytes", "mm10"),
        "hepatocytes_3": ("mouse_hepatocytes", "mm10"),
        "buffer_1": ("Macrophage", "hg38"),
        "buffer_2": ("Macrophage", "hg38"),
        "sample_1": ("K562", "hg38"),
    }
    
    cell_type = sample_to_cell_type_species[args.sample_name][0]
    species = sample_to_cell_type_species[args.sample_name][1]
    
    sweep_setting_hash = build_tf_tg_input_cache(
        sample_name=args.sample_name,
        cell_type=cell_type,
        species=species,
        max_peaks_per_tg=max_peaks_per_tg,
        max_cells_per_pair=max_cells_per_pair,
        pct_true_edges=pct_true_edges,
        true_false_ratio=true_false_ratio,
        peak_flank_size=peak_flank_size,
        num_cpu=args.num_cpu,
        force_reload=args.force_reload,
        sample_pairs=args.sample_pairs,
    )

    try:
        train_tf_tg_model(
            sample_name=args.sample_name,
            cell_type=cell_type,
            species=species,
            sweep_setting_hash=sweep_setting_hash,
            checkpoint_path=Path(args.checkpoint_path) if args.checkpoint_path else None,
            max_peaks_per_tg=max_peaks_per_tg,
            max_cells_per_pair=max_cells_per_pair,
            pct_true_edges=pct_true_edges,
            true_false_ratio=true_false_ratio,
            peak_flank_size=peak_flank_size,
            num_gpus=args.num_gpus,
            num_nodes=args.num_nodes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            wandb_config=run_config,
        )
    finally:
        wandb.finish()