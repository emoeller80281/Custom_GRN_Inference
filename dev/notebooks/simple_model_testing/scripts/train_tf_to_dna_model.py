
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
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
sys.path.append(str(PROJECT_DIR))

import models.tf_to_dna as tf_to_dna_module
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
    """
    Create a labeled TF-peak interaction dataset.

    Labels:
        true interactions  -> 1
        false interactions -> 0

    Returns
    -------
    pd.DataFrame with columns:
        tf_name, peak_id, tf_idx, peak_idx, label
    """

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
            print(f"Dropping {n_missing} interactions with missing TF or peak indices.")
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

    # Optional: shuffle rows
    df = df.sample(frac=1.0, random_state=123).reset_index(drop=True)

    return df

def load_ordered_tf_embeddings(
    embedding_dir,
    tf_name_to_idx,
    suffix="_protein_embedding.pt",
    weights_only=False,
):
    embedding_dir = Path(embedding_dir)

    # Map TF name -> file path
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

        # Convert [1, L, D] -> [L, D]
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)

        emb = emb.float()

        ordered_embeddings[tf_idx] = emb
        ordered_lengths[tf_idx] = emb.shape[0]

    if len(missing_tfs) > 0:
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
        "embeddings": embeddings_padded,  # [n_tfs, max_tf_len, embedding_dim]
        "lengths": lengths,              # [n_tfs]
        "mask": mask,                    # [n_tfs, max_tf_len]
    }

class TFPeakEdgeDataset(Dataset):
    def __init__(
        self,
        tf_embeddings,
        tf_mask,
        peak_embeddings,
        edge_tf_idx,
        edge_peak_idx,
        edge_labels,
    ):
        # Inputs
        self.tf_embeddings = tf_embeddings
        self.tf_mask = tf_mask
        self.peak_embeddings = peak_embeddings

        # Labels and indices for edges
        self.edge_tf_idx = edge_tf_idx
        self.edge_peak_idx = edge_peak_idx
        self.edge_labels = edge_labels

    def __len__(self):
        return len(self.edge_labels)

    def __getitem__(self, idx):
        tf_idx = self.edge_tf_idx[idx]
        peak_idx = self.edge_peak_idx[idx]

        return {
            "tf_embedding": self.tf_embeddings[tf_idx],      # [max_tf_len, 128]
            "tf_mask": self.tf_mask[tf_idx],                 # [max_tf_len]
            "peak_embedding": self.peak_embeddings[peak_idx],# [512, 4]
            "label": self.edge_labels[idx],                  # scalar
            "tf_idx": tf_idx,
            "peak_idx": peak_idx,
        }

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train TF-to-DNA binding model")
    argparser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    argparser.add_argument("--num_gpus", type=int, default=1, help="Number of GPU devices to use for training")
    argparser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use for training")
    argparser.add_argument("--model_dim", type=int, default=128, help="Dimension of the model")
    argparser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    argparser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the model")
    argparser.add_argument("--run_name", type=str, required=True, default="tfbind_train", help="Name of the training run")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to save model checkpoints")
    argparser.add_argument("--training_data_dir", type=str, required=False, help="Path to directory containing training data cache files (if not using default)")
    argparser.add_argument("--checkpoint_path", type=str, required=False, help="Path to a model checkpoint to resume training from")
    argparser.add_argument("--force_reload", action="store_true", help="Whether to force reload cached data instead of using existing cache files")
    args = argparser.parse_args()
    
    epochs = args.epochs
    num_gpus = args.num_gpus
    num_nodes = args.num_nodes
    model_dim = args.model_dim
    batch_size = args.batch_size
    num_layers = args.num_layers
    run_name = args.run_name
    output_dir = args.output_dir
    checkpoint_path = args.checkpoint_path
    force_reload = args.force_reload
    training_data_dir = args.training_data_dir

    if training_data_dir:
        training_cache_dir = Path(training_data_dir)
    else:
        training_cache_dir = PROJECT_DIR / "data" / "training_data_cache"
    training_cache_dir.mkdir(exist_ok=True, parents=True)
    
    # Define paths for cached data files
    tf_name_to_idx_cache_path = training_cache_dir / "tf_name_to_idx.csv"
    peak_id_to_idx_cache_path = training_cache_dir / "peak_id_to_idx.csv"
    edge_tf_idx_cache_path = training_cache_dir / "edge_tf_idx.pt"
    edge_peak_idx_cache_path = training_cache_dir / "edge_peak_idx.pt"
    edge_labels_cache_path = training_cache_dir / "edge_labels.pt"
    tf_embedding_cache_path = training_cache_dir / "tf_embeddings.pt"
    tf_mask_cache_path = training_cache_dir / "tf_masks.pt"
    tf_lengths_cache_path = training_cache_dir / "tf_lengths.pt"
    peak_onehot_cache_path = training_cache_dir / "peak_onehot_array.pt"
    
    cache_files = [
        tf_name_to_idx_cache_path,
        peak_id_to_idx_cache_path,
        edge_tf_idx_cache_path,
        edge_peak_idx_cache_path,
        edge_labels_cache_path,
        tf_embedding_cache_path,
        tf_mask_cache_path,
        peak_onehot_cache_path,
        training_cache_dir / "train_idx.pt",
        training_cache_dir / "val_idx.pt",
        training_cache_dir / "test_idx.pt",
    ]
    missing = [str(path) for path in cache_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing TF-to-DNA cache files. Run build_tf_to_dna_train_data.py first.\n"
            + "\n".join(missing)
        )

    edge_tf_idx_tensor = torch.load(edge_tf_idx_cache_path)
    edge_peak_idx_tensor = torch.load(edge_peak_idx_cache_path)
    edge_labels_tensor = torch.load(edge_labels_cache_path)
    tf_embeddings_tensor = torch.load(tf_embedding_cache_path)
    tf_mask_tensor = torch.load(tf_mask_cache_path)
    peak_tensor = torch.load(peak_onehot_cache_path)
    train_idx = torch.load(training_cache_dir / "train_idx.pt")
    val_idx = torch.load(training_cache_dir / "val_idx.pt")
    test_idx = torch.load(training_cache_dir / "test_idx.pt")

    edge_dataset = TFPeakEdgeDataset(
        tf_embeddings=tf_embeddings_tensor,
        tf_mask=tf_mask_tensor,
        peak_embeddings=peak_tensor,
        edge_tf_idx=edge_tf_idx_tensor,
        edge_peak_idx=edge_peak_idx_tensor,
        edge_labels=edge_labels_tensor,
    )

    train_dataset = Subset(edge_dataset, train_idx.tolist())
    val_dataset = Subset(edge_dataset, val_idx.tolist())
    test_dataset = Subset(edge_dataset, test_idx.tolist())

    # Create dataloaders for each split
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
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
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
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
        strategy="ddp" if use_ddp else "auto",
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=100),
            checkpoint_callback,
            early_stopping_callback,
            lr_monitor,
        ],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=100,
        default_root_dir=output_dir,
        enable_progress_bar=True,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )
    
    torch.set_float32_matmul_precision('medium')
    
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
                    