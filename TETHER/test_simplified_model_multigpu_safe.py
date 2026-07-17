import json
import os
import sys
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
DATA_DIR = PROJECT_DIR / "cached_data"
CHKPT_DIR = PROJECT_DIR / "checkpoints"
RESULT_DIR = PROJECT_DIR / "testing_results"

sys.path.append(str(PROJECT_DIR))

import models.tf_to_dna as tf_to_dna_module
import scripts.build_tf_to_tg_train_data as tf_tg_data_builder
import utils
import config
import warnings
import argparse
import time

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

tf_tg_input_cache_dir = DATA_DIR / "tf_tg_training_cache"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def create_tf_tg_index_to_name_mappings(tf_name_to_idx, tg_id_to_idx):
    tf_idx_to_name = {idx: name for name, idx in tf_name_to_idx.items()}
    tg_idx_to_name = {idx: name for name, idx in tg_id_to_idx.items()}
    return tf_idx_to_name, tg_idx_to_name

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", type=str, required=True, help="Species for training")
    parser.add_argument("--cell_type", type=str, required=True, help="Cell type for training")
    parser.add_argument("--sample_name", type=str, required=True, help="Sample name for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPU devices to use for training")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use for training")
    parser.add_argument("--job_id", type=str, required=True, help="SLURM job ID for this training run")
    parser.add_argument("--sample_pairs", type=int, default=None, help="Number of TF-TG pairs to sample for training (default: use all)")
    parser.add_argument("--max_peaks_per_tg", type=int, required=False, default=None, help="Maximum number of peaks to consider per TG")
    parser.add_argument("--max_cells_per_pair", type=int, default=8, help="Maximum number of cells to sample per TF-TG pair")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--pct_true_edges", type=float, default=0.15, help="Percentage of true edges to include in the training set")
    parser.add_argument("--true_false_ratio", type=float, default=2.0, help="Ratio of true to false edges in the training set")
    parser.add_argument("--peak_flank_size", type=int, default=128, help="Size of the flank region around peaks")
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to a TF-TG model checkpoint to resume weights from")
    parser.add_argument("--force_reload", action="store_true", help="Force rank 0 to rebuild cached data")
    parser.add_argument("--num_workers", type=int, default=6, help="DataLoader workers per rank")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch factor when num_workers > 0")
    parser.add_argument("--cache_wait_poll_seconds", type=int, default=30, help="Seconds between cache-ready checks on nonzero ranks")
    parser.add_argument("--model_variant", type=str, choices=["normal", "no_peak_tg_distance", "no_peak_info", "no_expr_info", "no_tf_dna_binding"], help="Variant of the simplified model to use")
    return parser


def build_paths(args):
    cell_type_cache_dir = DATA_DIR / f"{args.cell_type}_cache"
    cache_dir = cell_type_cache_dir / f"{args.sample_name}_simplified_model_cache"
    output_dir = CHKPT_DIR / "simplified_model" / f"{args.cell_type}_{args.sample_name}_simplified_model_test_{args.job_id}"
    return {
        "cell_type_cache_dir": cell_type_cache_dir,
        "cache_dir": cache_dir,
        "output_dir": output_dir,
        "atac_peak_tensor": cache_dir / "atac_peak_tensor.pt",
        "metadata": cache_dir / "metadata.json",
        "manifest": cache_dir / "manifest.json",
        "train": cache_dir / "tftg_inputs_train.pt",
        "val": cache_dir / "tftg_inputs_val.pt",
        "test": cache_dir / "tftg_inputs_test.pt",
        "ready": cache_dir / ".cache_ready",
        "failed": cache_dir / ".cache_failed",
    }


def build_and_save_training_cache(args, paths):
    """Run only on global rank 0. Creates all cached tensors/files atomically."""
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["ready"].unlink(missing_ok=True)
    paths["failed"].unlink(missing_ok=True)

    try:
        gene_ref_file, genome_fasta_path, chrom_sizes_path, train_chroms, val_chroms, test_chroms, valid_chroms = utils.get_reference_paths_and_chroms(args.species)
        sample_input_data_dir = PROJECT_DIR.parent / "data" / "sample_input_data" / args.cell_type / args.sample_name

        logging.info("Rank 0 reading ATAC/RNA pseudobulk and peak-to-gene files")
        atac_pseudobulk = pd.read_parquet(sample_input_data_dir / "RE_pseudobulk.parquet")
        peak_to_gene_distance = pd.read_parquet(sample_input_data_dir / "peak_to_gene_dist.parquet")
        rna_pseudobulk = pd.read_parquet(sample_input_data_dir / "TG_pseudobulk.parquet")

        logging.info(f"ATAC peaks BEFORE peak-to-gene filtering: {atac_pseudobulk.shape[0]:,}")
        valid_peak_ids = set(peak_to_gene_distance["peak_id"])
        atac_pseudobulk = atac_pseudobulk.loc[atac_pseudobulk.index.isin(valid_peak_ids)].copy()
        logging.info(f"ATAC peaks AFTER peak-to-gene filtering: {atac_pseudobulk.shape[0]:,}")

        rna_pseudobulk_norm = rna_pseudobulk.copy()
        rna_pseudobulk_norm.index = rna_pseudobulk_norm.index.str.upper()
        common_cells = sorted(set(rna_pseudobulk_norm.columns) & set(atac_pseudobulk.columns))
        if len(common_cells) == 0:
            raise ValueError("No common pseudobulk cell columns between RNA and ATAC matrices.")
        logging.info(f"Common RNA/ATAC pseudobulk columns: {len(common_cells):,}")

        peak_to_gene = peak_to_gene_distance.copy()
        peak_to_gene["target_id_norm"] = peak_to_gene["target_id"].str.upper()

        merged_ground_truth_path = sample_input_data_dir / "merged_ground_truth.parquet"
        if not merged_ground_truth_path.exists():
            merged_ground_truth_df = utils.load_ground_truth_files(config.gt_by_dataset_dict[args.cell_type])
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
        logging.info(f"Ground truth edges after RNA TF/TG filtering: {len(merged_ground_truth_df):,} / {n_before_rna_filter:,}")

        tf_embeddings_tensor, tf_mask_tensor, tf_name_to_idx = utils.load_tf_embedding_resources(paths)

        gt_tfs_in_embeddings = set(tf_name_to_idx.keys()).intersection(gt_tfs_in_rna)
        n_before_tf_embedding_filter = len(merged_ground_truth_df)
        merged_ground_truth_df = merged_ground_truth_df[merged_ground_truth_df["Source"].isin(gt_tfs_in_embeddings)].copy()
        logging.info(f"Ground truth edges after filtering to TFs with embeddings: {len(merged_ground_truth_df):,} / {n_before_tf_embedding_filter:,}")

        tg_id_to_idx = {tg: idx for idx, tg in enumerate(merged_ground_truth_df["Target"].unique())}
        train_genes, val_genes, test_genes = tf_tg_data_builder.split_genes_by_chromosome(
            gene_ref_file,
            train_chroms=train_chroms,
            val_chroms=val_chroms,
            test_chroms=test_chroms,
        )
        gt_train_df, gt_val_df, gt_test_df = tf_tg_data_builder.create_train_val_test_splits(
            merged_ground_truth_df, train_genes, val_genes, test_genes
        )
        logging.info(f"Train interactions: {len(gt_train_df):,}; Val: {len(gt_val_df):,}; Test: {len(gt_test_df):,}")

        tf_tg_labeled_train_df = tf_tg_data_builder._create_labeled_df(
            gt_train_df, args.pct_true_edges, args.true_false_ratio, seed=123,
            tf_name_to_idx=tf_name_to_idx, tg_id_to_idx=tg_id_to_idx,
        )
        tf_tg_labeled_val_df = tf_tg_data_builder._create_labeled_df(
            gt_val_df, args.pct_true_edges, args.true_false_ratio, seed=124,
            tf_name_to_idx=tf_name_to_idx, tg_id_to_idx=tg_id_to_idx,
        )
        tf_tg_labeled_test_df = tf_tg_data_builder._create_labeled_df(
            gt_test_df, args.pct_true_edges, args.true_false_ratio, seed=125,
            tf_name_to_idx=tf_name_to_idx, tg_id_to_idx=tg_id_to_idx,
        )

        dataset_peaks = [peak for peak in atac_pseudobulk.index.to_list() if peak.split(":", 1)[0] in valid_chroms]
        atac_peak_map = {peak: idx for idx, peak in enumerate(dataset_peaks)}

        logging.info("Creating centered one-hot encoded ATAC peak tensor on rank 0")
        atac_peak_array = utils.create_centered_peak_onehot_array(
            peak_ids=dataset_peaks,
            genome_fasta=genome_fasta_path,
            chrom_sizes=utils.load_chrom_sizes(chrom_sizes_path),
            peak_id_to_idx=atac_peak_map,
            flank_size=args.peak_flank_size,
            dtype=np.uint8,
            pad_out_of_bounds=True,
            num_workers=8,
            show_progress=False,
            chunk_size=10000,
        )
        atac_peak_tensor = torch.as_tensor(atac_peak_array, dtype=torch.uint8).float()
        logging.info(f"ATAC peak tensor shape: {tuple(atac_peak_tensor.shape)}")

        logging.info("Constructing TF-TG lookup tables on rank 0")
        tg_to_peak_info, cell_to_idx, atac_mat, rna_mat, gene_to_rna_idx = utils.prepare_tftg_lookup_tables(
            peak_to_gene=peak_to_gene,
            atac_peak_map=atac_peak_map,
            atac_pseudobulk=atac_pseudobulk,
            rna_pseudobulk_norm=rna_pseudobulk_norm,
            dataset_peaks=dataset_peaks,
            common_cells=common_cells,
            max_precompute_peaks=args.max_peaks_per_tg,
        )

        tf_tg_df = pd.concat([tf_tg_labeled_train_df, tf_tg_labeled_val_df, tf_tg_labeled_test_df], ignore_index=True)
        max_peaks_real = max(len(tg_to_peak_info.get(tg_name, {}).get("peak_indices", [])) for tg_name in tf_tg_df["tg_id"])
        n_tgs_with_peaks = sum(len(tg_to_peak_info.get(tg, {}).get("peak_indices", [])) > 0 for tg in tf_tg_df["tg_id"].unique())
        logging.info(f"TGs with at least one peak within 100kb: {n_tgs_with_peaks:,} / {tf_tg_df['tg_id'].nunique():,}")
        logging.info(f"Max peaks per TG after filtering/capping: {max_peaks_real:,}")

        common_build_kwargs = dict(
            max_cells_per_pair=args.max_cells_per_pair,
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

        logging.info("Building train TF-TG input dataset on rank 0")
        tftg_inputs_train = utils.build_tftg_inputs(tf_tg_labeled_train_df, seed=123, silence=False, **common_build_kwargs)
        logging.info("Building validation TF-TG input dataset on rank 0")
        tftg_inputs_val = utils.build_tftg_inputs(tf_tg_labeled_val_df, seed=124, silence=False, **common_build_kwargs)
        logging.info("Building test TF-TG input dataset on rank 0")
        tftg_inputs_test = utils.build_tftg_inputs(tf_tg_labeled_test_df, seed=125, silence=False, **common_build_kwargs)

        tf_embeddings_tensor = torch.load(paths["cell_type_cache_dir"] / "tf_embeddings.pt", map_location="cpu", weights_only=True)
        tf_mask_tensor = torch.load(paths["cell_type_cache_dir"] / "tf_masks.pt", map_location="cpu", weights_only=True)

        metadata = {
            "tf_name_to_idx": tf_name_to_idx,
            "tg_id_to_idx": tg_id_to_idx,
            "gene_to_rna_idx": gene_to_rna_idx,
            "cell_to_idx": cell_to_idx,
            "max_peaks_per_tg": args.max_peaks_per_tg,
            "max_cells_per_pair": args.max_cells_per_pair,
            "flank_size": args.peak_flank_size,
            "peak_dtype": "uint8",
            "max_peaks_real": max_peaks_real,
        }
        manifest = {
            "species": args.species,
            "cell_type": args.cell_type,
            "sample_name": args.sample_name,
            "max_peaks_per_tg": args.max_peaks_per_tg,
            "max_cells_per_pair": args.max_cells_per_pair,
            "flank_size": args.peak_flank_size,
            "atac_peak_tensor_dtype": str(atac_peak_tensor.dtype),
            "atac_peak_tensor_shape": list(atac_peak_tensor.shape),
            "tf_embeddings_tensor_shape": list(tf_embeddings_tensor.shape),
            "tf_mask_tensor_shape": list(tf_mask_tensor.shape),
            "n_train_rows": int(len(tftg_inputs_train["label"])),
            "n_val_rows": int(len(tftg_inputs_val["label"])),
            "n_test_rows": int(len(tftg_inputs_test["label"])),
        }

        logging.info(f"Saving rank-0-built cache to {paths['cache_dir']}")
        utils.atomic_torch_save(atac_peak_tensor, paths["atac_peak_tensor"])
        utils.atomic_torch_save(tftg_inputs_train, paths["train"])
        utils.atomic_torch_save(tftg_inputs_val, paths["val"])
        utils.atomic_torch_save(tftg_inputs_test, paths["test"])
        utils.atomic_json_dump(metadata, paths["metadata"], indent=4)
        utils.atomic_json_dump(manifest, paths["manifest"], indent=2)
        paths["ready"].write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        logging.info("Rank 0 finished cache construction")
    except Exception as exc:
        paths["failed"].write_text(repr(exc))
        raise

def make_dataloader(dataset, *, batch_size, shuffle, num_workers, prefetch_factor):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=utils.collate_tftg_edge_bags,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def main():
    args = make_parser().parse_args()
    global_rank, local_rank, node_rank, world_size = utils.configure_rank_logging()
    is_rank0 = global_rank == 0
    use_ddp = world_size > 1 or args.num_gpus * args.num_nodes > 1
    paths = build_paths(args)
    
    if args.model_variant == "normal":
        import models.tf_to_tg as tf_to_tg_module
    elif args.model_variant == "no_peak_tg_distance":
        import models.simplified_models.tf_to_tg_no_peak_tg_distance as tf_to_tg_module
    elif args.model_variant == "no_peak_info":
        import models.simplified_models.tf_to_tg_no_peak_info as tf_to_tg_module
    elif args.model_variant == "no_expr_info":
        import models.simplified_models.tf_to_tg_no_expr_info as tf_to_tg_module
    elif args.model_variant == "no_tf_dna_binding":
        import models.simplified_models.tf_to_tg_no_binding as tf_to_tg_module

    if is_rank0:
        if args.force_reload or not utils.cache_is_complete(paths):
            build_and_save_training_cache(args, paths)
        else:
            logging.info(f"Using existing complete cache in {paths['cache_dir']}")
    else:
        logging.warning(f"Waiting for rank 0 to finish or validate cache in {paths['cache_dir']}")
        utils.wait_for_cache(paths, poll_seconds=args.cache_wait_poll_seconds, timeout_seconds=None)

    tftg_inputs_train, tftg_inputs_val, tftg_inputs_test, atac_peak_tensor, tf_embeddings_tensor, tf_mask_tensor = utils.load_training_cache(paths)

    train_dataset = utils.TFTGEdgeBagDataset(
        tftg_inputs_train,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor,
    )
    val_dataset = utils.TFTGEdgeBagDataset(
        tftg_inputs_val,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor,
    )
    test_dataset = utils.TFTGEdgeBagDataset(
        tftg_inputs_test,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor,
    )

    train_loader = make_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = make_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )
    test_loader = make_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    if is_rank0:
        logging.info(f"Created DataLoaders with train={len(train_dataset):,}, val={len(val_dataset):,}, test={len(test_dataset):,}")
        logging.info(f"Per-rank batch size: {args.batch_size}; train batches per rank before DDP sampling: {len(train_loader):,}")

    tf_dna_model_chkpt = utils.tf_dna_checkpoint_for_cell_type(args.cell_type)
    
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
        checkpoint_path=tf_dna_model_chkpt,
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
        d_model=128,
        tf_peak_chunk_size=128,
    )

    pooling_mode = "lse"
    pooling_temperature = 1.0
    lit_model = tf_to_tg_module.LitTFTGRegulationModel(
        model=tf_tg_model,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
        pooling_mode=pooling_mode,
        pooling_temperature=pooling_temperature,
        enable_timing_sync=False,
    )

    run_name = f"{args.model_variant}_{args.sample_name}_{args.job_id}"
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=paths["output_dir"],
        filename="epoch={epoch:02d}-val_auroc={val/auroc:.4f}-val_loss={val/loss:.4f}",
        monitor="val/auroc",
        mode="max",
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_stopping_callback = EarlyStopping(monitor="val/loss", mode="min", patience=15)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    wandb_logger = None
    if is_rank0:
        wandb_logger = WandbLogger(
            project=f"tf_tg_feature_ablation",
            name=run_name,
            save_dir=paths["output_dir"],
        )
        wandb_logger.log_hyperparams({
            "species": args.species,
            "cell_type": args.cell_type,
            "sample_name": args.sample_name,
            "model_variant": args.model_variant,
            "epochs": args.epochs,
            "batch_size_per_rank": args.batch_size,
            "num_batches_per_rank": len(train_loader),
            "num_gpus_arg": args.num_gpus,
            "num_nodes_arg": args.num_nodes,
            "world_size_env": world_size,
            "job_id": args.job_id,
            "run_name": run_name,
            "sample_pairs": len(train_dataset),
            "max_peaks_per_tg": args.max_peaks_per_tg,
            "max_cells_per_pair": args.max_cells_per_pair,
            "pct_true_edges": args.pct_true_edges,
            "true_false_ratio": args.true_false_ratio,
            "pooling_mode": pooling_mode,
            "pooling_temperature": pooling_temperature,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "flank_size": args.peak_flank_size,
            "persistent_workers": args.num_workers > 0,
            "num_workers_per_rank": args.num_workers,
            "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
            "tf_bind_model_path": str(tf_dna_model_chkpt),
        })

    logging.info(f"World size: {world_size}; use_ddp={use_ddp}; local_rank={local_rank}")
    strategy = DDPStrategy(process_group_backend="nccl", find_unused_parameters=False) if use_ddp else "auto"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=25),
            checkpoint_callback,
            early_stopping_callback,
            lr_monitor,
        ],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
        default_root_dir=paths["output_dir"],
        enable_progress_bar=is_rank0,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        use_distributed_sampler=use_ddp,
    )

    trainer.fit(
        lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    main()

