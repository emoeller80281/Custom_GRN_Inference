import os
from re import I
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

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
sys.path.append(str(PROJECT_DIR))

import utils
import config
import scripts.build_tf_to_tg_train_data as build_tf_to_tg_train_data
import scripts.train_tf_to_tg_model as train_tf_to_tg_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import hashlib

def build_tf_tg_input_cache(
    sample_pairs: int,
    max_peaks_per_tg: int,
    max_cells_per_pair: int,
    pct_true_edges: float,
    true_false_ratio: float,
    peak_flank_size: int,
    num_cpu: int,
    force_reload: bool,
):
        
    gene_ref_file = config.gene_ref_file
    genome_fasta_path = config.genome_fasta_path
    chrom_sizes_path = config.chrom_sizes_path
    
    assert gene_ref_file.exists(), f"Gene reference file not found: {gene_ref_file}"
    assert genome_fasta_path.exists(), f"Genome FASTA file not found: {genome_fasta_path}"
    assert chrom_sizes_path.exists(), f"Chromosome sizes file not found: {chrom_sizes_path}"
    
    # Create the training cache directory if it doesn't exist
    input_data_dir = Path(config.sample_input_data_dir)
    
    assert input_data_dir.exists(), f"Input data directory does not exist: {input_data_dir}"
    
    # Encode the sweep settings into a string and hash it to create a unique cache directory for this sweep configuration
    sweep_setting_cache_string = f"{sample_pairs}_{max_peaks_per_tg}_{max_cells_per_pair}_{pct_true_edges}_{true_false_ratio}_{peak_flank_size}"
    sweep_setting_hash = hashlib.md5(sweep_setting_cache_string.encode('utf-8')).hexdigest()
    
    tf_tg_input_cache_dir = config.tf_tg_input_cache_dir / "wandb_sweep" / f"tf_tg_sweep_{sweep_setting_hash}"

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
    
    if all(f.exists() for f in required_cache_files) and not force_reload:
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

    # Load and merge the ground truth files, or load from cache if already merged
    if not merged_ground_truth_path.exists() or force_reload:

        merged_ground_truth_df: pd.DataFrame = utils.load_ground_truth_files(
            config.gt_by_dataset_dict[config.cell_type]
            )
        
        if config.species == "mm10":
            merged_ground_truth_df["Source"] = merged_ground_truth_df["Source"].str.capitalize()
            merged_ground_truth_df["Target"] = merged_ground_truth_df["Target"].str.capitalize()
        elif config.species == "hg38":
            merged_ground_truth_df["Source"] = merged_ground_truth_df["Source"].str.upper()
            merged_ground_truth_df["Target"] = merged_ground_truth_df["Target"].str.upper()
            
        merged_ground_truth_df.to_parquet(merged_ground_truth_path, index=False)
    else:
        merged_ground_truth_df = pd.read_parquet(merged_ground_truth_path)
    
    # Get the map of TF name to index
    tf_name_to_idx = pd.read_csv(tf_name_to_idx_cache_path).set_index("tf_name")["tf_idx"].to_dict()
    tg_id_to_idx = {tg: idx for idx, tg in enumerate(merged_ground_truth_df["Target"].unique())}
    
    if config.species == "mm10":
        train_chroms = [str(i) for i in range(1, 16)]
        val_chroms = [ str(i) for i in range(16, 18)]
        test_chroms = [str(i) for i in range(18, 20)]
    elif config.species == "hg38":
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
    gt_train_df, gt_val_df, gt_test_df = build_tf_to_tg_train_data.create_train_val_test_splits(
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
    tf_tg_labeled_test_df = build_tf_to_tg_train_data._create_labeled_df(
        gt_test_df,
        pct_true_edges,
        true_false_ratio,
        seed=123,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
    )

    # Create a map of ATAC peaks to indices in the pseudobulk matrix, filtering to valid chromosomes
    dataset_peaks = atac_pseudobulk.index.to_list()
    valid_chroms = {f"chr{i}" for i in range(1, 20)}
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

    rna_pseudobulk_norm = rna_pseudobulk.copy()
    rna_pseudobulk_norm.index = rna_pseudobulk_norm.index.str.upper()

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

    if sample_pairs is None:
        sample_pairs = len(tf_tg_labeled_train_df)

    tf_tg_train_subset = _sample_df(tf_tg_labeled_train_df, n=sample_pairs, seed=123)
    tf_tg_val_subset = _sample_df(tf_tg_labeled_val_df, n=sample_pairs // 2, seed=123)
    tf_tg_test_subset = _sample_df(tf_tg_labeled_test_df, n=sample_pairs // 4, seed=123)

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
    )
    
    if all(f.exists() for f in [train_file, val_file, test_file]) and not force_reload:
        logging.info("Cached input files already exist. Skipping (use --force_reload to override).")
        return
    
    logging.info("\nBuilding training inputs")
    tftg_inputs_train = build_tf_to_tg_train_data.build_tftg_inputs(
        tf_tg_train_subset,
        seed=123,
        **common_build_kwargs,
    )

    logging.info("\nBuilding validation inputs")
    tftg_inputs_val = build_tf_to_tg_train_data.build_tftg_inputs(
        tf_tg_val_subset,
        seed=124,
        **common_build_kwargs,
    )

    logging.info("\nBuilding test inputs")
    tftg_inputs_test = build_tf_to_tg_train_data.build_tftg_inputs(
        tf_tg_test_subset,
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
    
    return sweep_setting_hash

def train_tf_tg_model(
    tf_embedding_cache_path: str | Path,
    sweep_setting_hash: str,
    tf_bind_model_path: str | Path,
    sample_pairs: int,
    max_peaks_per_tg: int,
    max_cells_per_pair: int,
    pct_true_edges: float,
    true_false_ratio: float,
    peak_flank_size: int,
    num_gpus: int,
    num_nodes: int,
    epochs: int = 50,
    batch_size: int = 128,
):
    
    sample_name = config.sample_name
    
    output_dir = PROJECT_DIR / "checkpoints" / f"{config.cell_type}" / f"{sample_name}" / "wandb_sweep" / f"tf_tg_train_{sample_name}_{sweep_setting_hash}"
    
    run_name = f"tf_tg_{sample_name}_{sweep_setting_hash}"
    
    tf_tg_input_cache_dir = config.tf_tg_input_cache_dir / "wandb_sweep" / f"tf_tg_sweep_{sweep_setting_hash}"
    
    # Load the trained TF embedding and mask tensors from the TF→DNA model cache 
    # (these are needed for the TF→TG model since it uses the pretrained TF peak embedding module)
    tf_embeddings_tensor = torch.load(
        config.tf_embedding_cache_path,
        weights_only=True,
    )
    tf_mask_tensor = torch.load(
        config.tf_mask_cache_path,
        weights_only=True,
    )
    
    # TF-TG training specific cache files
    tf_tg_atac_peak_cache_path = tf_tg_input_cache_dir / "atac_peak_tensor.pt"
    tf_tg_metadata_cache_path = tf_tg_input_cache_dir / "metadata.json"
    tf_tg_manifest_cache_path = tf_tg_input_cache_dir / "manifest.json"
    tf_tg_train_cache_path = tf_tg_input_cache_dir / "tftg_inputs_train.pt"
    tf_tg_val_cache_path = tf_tg_input_cache_dir / "tftg_inputs_val.pt"
    tf_tg_test_cache_path = tf_tg_input_cache_dir / "tftg_inputs_test.pt"
    
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
    tftg_inputs_test = torch.load(
        tf_tg_test_cache_path,
        weights_only=False,
    )

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

    test_dataset = train_tf_to_tg_model.TFTGEdgeBagDataset(
        tftg_inputs_test,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor
    )

    # Create the DataLoaders with the custom collate function for batching edge bags
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=train_tf_to_tg_model.collate_tftg_edge_bags,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=train_tf_to_tg_model.collate_tftg_edge_bags,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=train_tf_to_tg_model.collate_tftg_edge_bags,
        )

    train_tf_to_tg_model.log_once(f"Train/Val/Test sizes: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

    tf_tg_model = train_tf_to_tg_model.create_new_tf_tg_binding_model(tf_bind_model_path, tf_embeddings_tensor, tf_mask_tensor)

    criterion = torch.nn.BCEWithLogitsLoss()

    score_threshold = 0.5
    pooling_mode = "lse"
    pooling_temperature = 1.0

    epoch_rows = []

    def metrics_to_row(
        metrics,
        epoch,
        split,
        train_loss=np.nan,
    ):
        pos_rate = metrics["n_pos"] / max(metrics["n_edges"], 1)

        return {
            "epoch": epoch,
            "split": split,
            "train_loss": train_loss,
            "loss": metrics["loss"],
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "rand_auroc": metrics["rand_auroc"],
            "rand_auprc": metrics["rand_auprc"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "n_edges": metrics["n_edges"],
            "n_pos": metrics["n_pos"],
            "n_neg": metrics["n_neg"],
            "pos_rate": pos_rate,
            "score_threshold": metrics["score_threshold"],
            "pooling_mode": pooling_mode,
            "pooling_temperature": pooling_temperature,
        }

    train_tf_to_tg_model.log_once("\nStarting Lightning training...")

    lit_model = train_tf_to_tg_model.LitTFTGRegulationModel(
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
        save_top_k=100,
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
    )

    wandb_logger.log_hyperparams({
        "sample_name": sample_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_batches": len(train_loader),
        "num_gpus": num_gpus,
        "num_nodes": num_nodes,
        "run_name": run_name,
        "sample_pairs": sample_pairs,
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
        val_dataloaders=val_loader
    )