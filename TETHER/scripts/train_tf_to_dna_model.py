
from ast import arg
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
sys.path.append(str(PROJECT_DIR))

import models.tf_to_dna as tf_to_dna_module
import config
import utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

class TFPeakEdgeDataset(Dataset):
    def __init__(
        self,
        edge_tf_idx,
        edge_peak_idx,
        edge_labels,
        peak_tensor,
    ):
        self.edge_tf_idx = edge_tf_idx.long()
        self.edge_peak_idx = edge_peak_idx.long()
        self.edge_labels = edge_labels.float()

        # Peak tensor kept on CPU due to large size
        self.peak_tensor = peak_tensor

    def __len__(self):
        return len(self.edge_labels)

    def __getitem__(self, idx):
        peak_idx = self.edge_peak_idx[idx]

        return {
            "tf_idx": self.edge_tf_idx[idx],
            "peak_idx": peak_idx,
            "peak_embedding": self.peak_tensor[peak_idx],
            "label": self.edge_labels[idx],
        }

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train TF-to-DNA binding model")
    argparser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    argparser.add_argument("--num_gpus", type=int, default=1, help="Number of GPU devices to use for training")
    argparser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use for training")
    argparser.add_argument("--model_dim", type=int, default=128, help="Dimension of the model")
    argparser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    argparser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the model")
    argparser.add_argument("--job_id", type=str, required=True, help="SLURM job ID for this training run")
    argparser.add_argument("--checkpoint_path", type=str, required=False, help="Path to a model checkpoint to resume training from")
    argparser.add_argument("--force_reload", action="store_true", help="Whether to force reload cached data instead of using existing cache files")
    args = argparser.parse_args()
    
    epochs = args.epochs
    num_gpus = args.num_gpus
    num_nodes = args.num_nodes
    model_dim = args.model_dim
    batch_size = args.batch_size
    num_layers = args.num_layers
    job_id = args.job_id
    checkpoint_path = args.checkpoint_path
    force_reload = args.force_reload
    
    output_dir = PROJECT_DIR / "checkpoints" / f"tf_dna_{config.species}_{job_id}"
    
    run_name = f"tf_dna_{config.species}_{job_id}"
        
    # Shared cache files for both TF-to-TG and TF-to-DNA training
    tf_name_to_idx_cache_path = config.tf_name_to_idx_cache_path
    tf_embedding_cache_path = config.tf_embedding_cache_path
    tf_mask_cache_path = config.tf_mask_cache_path
    
    # TF-DNA training specific cache files
    tf_dna_edge_tf_idx_cache_path = config.tf_dna_edge_tf_idx_cache_path
    tf_dna_edge_peak_idx_cache_path = config.tf_dna_edge_peak_idx_cache_path
    tf_dna_edge_labels_cache_path = config.tf_dna_edge_labels_cache_path
    tf_dna_tf_lengths_cache_path = config.tf_dna_tf_lengths_cache_path
    tf_dna_peak_onehot_cache_path = config.tf_dna_peak_onehot_cache_path
    tf_dna_train_idx_cache_path = config.tf_dna_train_idx_cache_path
    tf_dna_val_idx_cache_path = config.tf_dna_val_idx_cache_path
    tf_dna_test_idx_cache_path = config.tf_dna_test_idx_cache_path
    
    cache_files = [
        tf_name_to_idx_cache_path,
        tf_dna_edge_tf_idx_cache_path,
        tf_dna_edge_peak_idx_cache_path,
        tf_dna_edge_labels_cache_path,
        tf_embedding_cache_path,
        tf_mask_cache_path,
        tf_dna_peak_onehot_cache_path,
        tf_dna_train_idx_cache_path,
        tf_dna_val_idx_cache_path,
        tf_dna_test_idx_cache_path,
    ]
    missing = [str(path) for path in cache_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing TF-to-DNA cache files. Run build_tf_to_dna_train_data.py first.\n"
            + "\n".join(missing)
        )

    # Load cached data
    edge_tf_idx_tensor: torch.Tensor = torch.load(tf_dna_edge_tf_idx_cache_path, weights_only=True)
    edge_peak_idx_tensor: torch.Tensor = torch.load(tf_dna_edge_peak_idx_cache_path, weights_only=True)
    edge_labels_tensor: torch.Tensor = torch.load(tf_dna_edge_labels_cache_path, weights_only=True)
    tf_embeddings_tensor: torch.Tensor = torch.load(tf_embedding_cache_path, weights_only=True)
    tf_mask_tensor: torch.Tensor = torch.load(tf_mask_cache_path, weights_only=True)
    peak_tensor: torch.Tensor = torch.load(tf_dna_peak_onehot_cache_path, weights_only=True)
    
    # Load train/val/test splits
    train_idx: torch.Tensor = torch.load(tf_dna_train_idx_cache_path, weights_only=True)
    val_idx: torch.Tensor = torch.load(tf_dna_val_idx_cache_path, weights_only=True)
    test_idx: torch.Tensor = torch.load(tf_dna_test_idx_cache_path, weights_only=True)

    if peak_tensor.dtype == torch.uint8:
        peak_tensor = peak_tensor.float()

    edge_dataset = TFPeakEdgeDataset(
        edge_tf_idx=edge_tf_idx_tensor,
        edge_peak_idx=edge_peak_idx_tensor,
        edge_labels=edge_labels_tensor,
        peak_tensor=peak_tensor,
    )

    train_dataset = Subset(edge_dataset, train_idx.tolist())
    val_dataset = Subset(edge_dataset, val_idx.tolist())
    test_dataset = Subset(edge_dataset, test_idx.tolist())

    # Create dataloaders for each split
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    
    base_model = tf_to_dna_module.TFPeakBindingModel(
        tf_embedding_dim=model_dim,
        hidden_dim=model_dim,
        dropout=0.3,
        num_layers=num_layers,
        num_heads=4,
        dim_head=32,
    )

    # PyTorch Lightning wrapper for training
    lit_model = tf_to_dna_module.LitTFPeakBindingModel(
        model=base_model,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
        enable_timing_sync=True,
    )


    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="epoch={epoch:02d}-val_auroc={val/auroc:.4f}-val_loss={val/loss:.4f}",
        monitor="val/auroc",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=10,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    wandb_logger = WandbLogger(
        project="tf_peak_binding",
        name=run_name,
        save_dir=output_dir,
    )
    
    wandb_logger.log_hyperparams({
        "sample_name": config.sample_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_batches": len(train_loader),
        "model_dim": model_dim,
        "num_layers": num_layers,
        "num_gpus": num_gpus*num_nodes,
        "num_nodes": num_nodes,
        "job_id": job_id,
        "run_name": run_name,
    })
    
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            os.environ.get("SLURM_NTASKS", "1"),
        )
    )

    use_ddp = world_size > 1
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_gpus,
        num_nodes=num_nodes,
        strategy=DDPStrategy(broadcast_buffers=False) if use_ddp else "auto",
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=50),
            checkpoint_callback,
            early_stopping_callback,
            lr_monitor,
        ],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=50,
        default_root_dir=output_dir,
        enable_progress_bar=True,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )
    
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True
    
    # Debug to find NaNs in the loss
    torch.autograd.set_detect_anomaly(False)
    
    if checkpoint_path is not None:
        logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.fit(
            lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=checkpoint_path,
        )
    else:
        trainer.fit(
            lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
                    