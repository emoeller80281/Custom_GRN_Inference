
import os
import sys
import gtfparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import warnings
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
)

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies import DDPStrategy

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
sys.path.append(str(PROJECT_DIR))

import models.tf_to_tg as tf_to_tg_module
import models.tf_to_dna as tf_to_dna_module
import config
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

warnings.filterwarnings(
    "ignore",
    message="This DataLoader will create .* worker processes in total\.",
    category=UserWarning,
    module="torch.utils.data.dataloader",
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

@rank_zero_only
def log_once(msg: str) -> None:
    logging.info(msg)

def create_new_tf_tg_binding_model(
    tf_bind_model_path: Path,
    tf_embeddings_tensor: torch.Tensor,
    tf_mask_tensor: torch.Tensor,
    checkpoint_path: Path | None = None,
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
        d_model=128,
        tf_peak_chunk_size=128,
    )

    # 5) Optionally load TF→TG checkpoint
    if checkpoint_path is not None:
        log_once(f"Loading TF→TG model weights from checkpoint: {checkpoint_path}")

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


class TFTGEdgeBagDataset(Dataset):
    def __init__(
        self,
        inputs,
        *,
        tf_embeddings_tensor,
        tf_mask_tensor,
        atac_peak_tensor,
    ):
        self.inputs = inputs
        self.tf_embeddings_tensor = tf_embeddings_tensor
        self.tf_mask_tensor = tf_mask_tensor
        self.atac_peak_tensor = atac_peak_tensor

    def __len__(self):
        return len(self.inputs["label"])

    def __getitem__(self, idx):
        tf_idx = self.inputs["tf_idx"][idx]
        tg_idx = self.inputs["tg_idx"][idx]

        peak_indices = self.inputs["peak_indices"][idx]          # [P]
        peak_sequences = self.atac_peak_tensor[peak_indices]     # [P, L, 4]

        tf_embedding = self.tf_embeddings_tensor[tf_idx]         # [T, D]
        tf_mask = self.tf_mask_tensor[tf_idx]                    # [T]

        return {
            "label": self.inputs["label"][idx],
            "tf_name": self.inputs["tf_name"][idx],
            "tg_name": self.inputs["tg_name"][idx],
            "cell_ids": self.inputs["cell_ids"][idx],
            "tf_idx": tf_idx,
            "tg_idx": tg_idx,
            "tf_embedding": tf_embedding.float(),
            "tf_mask": tf_mask.bool(),
            "peak_indices": peak_indices,
            "peak_sequences": peak_sequences,
            "peak_distance": self.inputs["peak_distance"][idx].float(),
            "peak_mask": self.inputs["peak_mask"][idx].bool(),
            "peak_accessibility": self.inputs["peak_accessibility"][idx].float(),
            "tf_expression": self.inputs["tf_expression"][idx].float(),
            "tg_expression": self.inputs["tg_expression"][idx].float(),
        }
        
def collate_tftg_edge_bags(batch):
    output = {
        "label": torch.stack([b["label"] for b in batch]).float(),

        "tf_idx": torch.stack([b["tf_idx"] for b in batch]).long(),
        "tg_idx": torch.stack([b["tg_idx"] for b in batch]).long(),

        "tf_embedding": torch.stack([b["tf_embedding"] for b in batch]),
        "tf_mask": torch.stack([b["tf_mask"] for b in batch]),

        "peak_indices": torch.stack([b["peak_indices"] for b in batch]),
        "peak_sequences": torch.stack([b["peak_sequences"] for b in batch]),
        "peak_distance": torch.stack([b["peak_distance"] for b in batch]),
        "peak_mask": torch.stack([b["peak_mask"] for b in batch]),

        "peak_accessibility": torch.stack([b["peak_accessibility"] for b in batch]),
        "tf_expression": torch.stack([b["tf_expression"] for b in batch]),
        "tg_expression": torch.stack([b["tg_expression"] for b in batch]),

        "tf_name": [b["tf_name"] for b in batch],
        "tg_name": [b["tg_name"] for b in batch],
        "cell_ids": [b["cell_ids"] for b in batch],
    }

    E, C = output["tf_expression"].shape
    output["cell_mask"] = torch.ones(E, C, dtype=torch.bool)

    return output
        
        
@torch.no_grad()
def _move_batch_to_device(batch, device):
    moved = {
        "tf_embedding": batch["tf_embedding"].to(device, non_blocking=True),
        "tf_mask": batch["tf_mask"].to(device, non_blocking=True),
        "peak_sequences": batch["peak_sequences"].to(device, non_blocking=True),
        "peak_accessibility": batch["peak_accessibility"].to(device, non_blocking=True),
        "peak_distance": batch["peak_distance"].to(device, non_blocking=True),
        "tf_expression": batch["tf_expression"].to(device, non_blocking=True),
        "tg_expression": batch["tg_expression"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
    }

    if "cell_mask" in batch:
        moved["cell_mask"] = batch["cell_mask"].to(device, non_blocking=True)

    if "peak_mask" in batch:
        moved["peak_mask"] = batch["peak_mask"].to(device, non_blocking=True)

    return moved


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    pooling_mode: str = "lse",
    pooling_temperature: float = 1.0,
):
    model.eval()
    total_loss = 0.0
    n_edges = 0
    for batch in loader:
        batch = _move_batch_to_device(batch, device)

        labels = batch["label"]
        cell_mask = batch["cell_mask"]
        E, C = cell_mask.shape

        edge_logits, _ = model.forward(
            tf_embedding=batch["tf_embedding"],
            tf_mask=batch["tf_mask"],
            peak_sequences=batch["peak_sequences"],
            peak_accessibility=batch["peak_accessibility"],
            peak_distance=batch["peak_distance"],
            tf_expression=batch["tf_expression"],
            tg_expression=batch["tg_expression"],
            peak_mask=batch.get("peak_mask", None),
            cell_mask=cell_mask,
            pooling_mode=pooling_mode,
            pooling_temperature=pooling_temperature,
        )
        loss = criterion(edge_logits, labels)
        total_loss += loss.item() * E
        n_edges += E
    return total_loss / max(n_edges, 1)

def compute_binary_classification_metrics(
    labels,
    scores,
    score_threshold: float = 0.5,
    random_state: int = 42,
):
    """
    labels: array-like of 0/1 labels
    scores: array-like of predicted probabilities after sigmoid
    """

    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores).astype(float).ravel()

    preds = (scores >= score_threshold).astype(int)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)

    if len(np.unique(labels)) < 2:
        auroc = np.nan
        auprc = np.nan
        rand_auroc = np.nan
        rand_auprc = np.nan
    else:
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)

        rng = np.random.default_rng(random_state)
        rand_scores = rng.permutation(scores)

        rand_auroc = roc_auc_score(labels, rand_scores)
        rand_auprc = average_precision_score(labels, rand_scores)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "rand_auroc": rand_auroc,
        "rand_auprc": rand_auprc,
        "accuracy": accuracy,
        "precision": precision,
        "n_edges": len(labels),
        "n_pos": int(labels.sum()),
        "n_neg": int((labels == 0).sum()),
        "score_threshold": score_threshold,
    }
    
@torch.no_grad()
def evaluate_with_metrics(
    model,
    loader,
    criterion,
    device,
    score_threshold: float = 0.5,
    random_state: int = 42,
    pooling_mode: str = "lse",
    pooling_temperature: float = 1.0,
):
    model.eval()

    total_loss = 0.0
    n_edges = 0

    all_scores = []
    all_labels = []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)

        labels = batch["label"]
        cell_mask = batch["cell_mask"]
        E, C = cell_mask.shape

        edge_logits, _ = model.forward(
            tf_embedding=batch["tf_embedding"],
            tf_mask=batch["tf_mask"],
            peak_sequences=batch["peak_sequences"],
            peak_accessibility=batch["peak_accessibility"],
            peak_distance=batch["peak_distance"],
            tf_expression=batch["tf_expression"],
            tg_expression=batch["tg_expression"],
            peak_mask=batch.get("peak_mask", None),
            cell_mask=cell_mask,
            pooling_mode=pooling_mode,
            pooling_temperature=pooling_temperature,
        )

        loss = criterion(edge_logits, labels)

        total_loss += loss.item() * E
        n_edges += E

        scores = torch.sigmoid(edge_logits)

        all_scores.append(scores.detach().cpu().numpy().ravel())
        all_labels.append(labels.detach().cpu().numpy().ravel())

    mean_loss = total_loss / max(n_edges, 1)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    metrics = compute_binary_classification_metrics(
        labels=all_labels,
        scores=all_scores,
        score_threshold=score_threshold,
        random_state=random_state,
    )

    metrics["loss"] = mean_loss
    metrics["score_min"] = float(all_scores.min())
    metrics["score_max"] = float(all_scores.max())
    metrics["score_mean"] = float(all_scores.mean())
    metrics["score_std"] = float(all_scores.std())
    metrics["n_pred_pos"] = int((all_scores >= score_threshold).sum())

    return metrics


if __name__ == "__main__":
    
    """
    Need arguments for:
    
    sample_pairs: int | None
    max_peaks_per_tg: int | None
    max_cells_per_pair: int | None
    batch_size: int
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPU devices to use for training")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use for training")
    parser.add_argument("--job_id", type=str, help="SLURM job ID for this training run")
    parser.add_argument("--sample_pairs", type=int, default=None, help="Number of TF-TG pairs to sample for training (default: use all)")
    parser.add_argument("--max_peaks_per_tg", type=int, required=False, default=None, help="Maximum number of peaks to consider per TG (default: 64)")
    parser.add_argument("--max_cells_per_pair", type=int, default=8, help="Maximum number of cells to sample per TF-TG pair (default: 8)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--pct_true_edges", type=float, default=0.15, help="Percentage of true edges to include in the training set (default: 0.15)")
    parser.add_argument("--true_false_ratio", type=float, default=2.0, help="Ratio of true to false edges in the training set (default: 2.0)")
    parser.add_argument("--peak_flank_size", type=int, default=128, help="Size of the flank region around peaks (default: 128)")
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to a model checkpoint to resume training from")
    parser.add_argument("--force_reload", action="store_true", help="Whether to force reload cached data instead of using existing cache files")
    args = parser.parse_args()

    gene_ref_file = DATA_DIR / "genome_data" / "genome_annotation" / "mm10" / "Mus_musculus.GRCm39.115.gtf.gz"
    genome_fasta_path = DATA_DIR / "genome_data" / "reference_genome" / "mm10" / "mm10.fa"
    chrom_sizes_path = DATA_DIR / "genome_data" / "reference_genome" / "mm10" / "mm10.chrom.sizes"
    
    epochs = args.epochs
    num_gpus = args.num_gpus
    num_nodes = args.num_nodes
    job_id = args.job_id
    checkpoint_path = args.checkpoint_path
    force_reload = args.force_reload
    batch_size = args.batch_size
    sample_pairs = args.sample_pairs
    max_peaks_per_tg = args.max_peaks_per_tg
    max_cells_per_pair = args.max_cells_per_pair
    pct_true_edges = args.pct_true_edges
    true_false_ratio = args.true_false_ratio
    peak_flank_size = args.peak_flank_size
    
    assert config.cell_type in config.tf_dna_model_checkpoints, \
        f"Cell type '{config.cell_type}' not found in TF→DNA model checkpoints."
    
    tf_bind_model_path = config.tf_dna_model_checkpoints[config.cell_type] 

    sample_name = config.sample_name
    
    output_dir = PROJECT_DIR / "checkpoints" / f"{config.cell_type}" / f"{sample_name}" / f"tf_tg_train_{sample_name}_{job_id}"
    
    run_name = f"tf_tg_{sample_name}_{job_id}"
    
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
    
    # Load the train/val/test splits of the compact TF-TG input tensors 
    # that were preprocessed and cached by the data preprocessing script
    tftg_inputs_train = torch.load(
        config.tf_tg_train_cache_path,
        weights_only=False,
    )
    tftg_inputs_val = torch.load(
        config.tf_tg_val_cache_path,
        weights_only=False,
    )
    tftg_inputs_test = torch.load(
        config.tf_tg_test_cache_path,
        weights_only=False,
    )

    atac_peak_tensor = torch.load(
        config.tf_tg_atac_peak_cache_path,
        weights_only=True,
    )

    # Load the metadata
    with open(config.tf_tg_metadata_cache_path, "r") as f:
        metadata = json.load(f)
        
    tf_name_to_idx = metadata["tf_name_to_idx"]
    tg_id_to_idx = metadata["tg_id_to_idx"]

    # Load the manifest and verify tensor shapes and dtypes match expectations
    with open(config.tf_tg_manifest_cache_path) as f:
        manifest = json.load(f)
    
    log_once(json.dumps(manifest, indent=2))

    assert tuple(manifest["atac_peak_tensor_shape"]) == tuple(atac_peak_tensor.shape)
    assert manifest["atac_peak_tensor_dtype"] == str(atac_peak_tensor.dtype)
    
    # Re-create the datasets and dataloaders using the loaded compact inputs and lookup tensors
    train_dataset = TFTGEdgeBagDataset(
        tftg_inputs_train,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor
    )

    val_dataset = TFTGEdgeBagDataset(
        tftg_inputs_val,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor

    )

    test_dataset = TFTGEdgeBagDataset(
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
        collate_fn=collate_tftg_edge_bags,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_tftg_edge_bags,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate_tftg_edge_bags,
        )

    log_once(f"Train/Val/Test sizes: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

    tf_tg_model = create_new_tf_tg_binding_model(
        tf_bind_model_path, 
        tf_embeddings_tensor, 
        tf_mask_tensor,
        checkpoint_path=checkpoint_path
        )

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

    log_once("\nStarting Lightning training...")

    lit_model = tf_to_tg_module.LitTFTGRegulationModel(
        model=tf_tg_model,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
        pooling_mode=pooling_mode,
        pooling_temperature=pooling_temperature,
        enable_timing_sync=False,
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="epoch={epoch:02d}-val_auroc={val/auroc:.4f}-val_loss={val/loss:.4f}",
        monitor="val/auroc",
        mode="max",
        save_top_k=500,
        save_last=True,
        auto_insert_metric_name=False,
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=15,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    wandb_logger = WandbLogger(
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
        "job_id": job_id,
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
        "tf_bind_model_path": str(tf_bind_model_path),
    })
    
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            os.environ.get("SLURM_NTASKS", "1"),
        )
    )

    use_ddp = world_size > 1
    
    log_once(f"Num GPUs: {world_size} | Batch size: {batch_size}")
    log_once(f"Num steps per epoch: {len(train_loader)}")
    
    strategy=DDPStrategy(
        process_group_backend="nccl",
        find_unused_parameters=False,
    ) if use_ddp else "auto"
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_gpus,
        num_nodes=num_nodes,
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
        default_root_dir=output_dir,
        enable_progress_bar=True,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )
    
    if checkpoint_path is not None:
        log_once(f"Resuming training from checkpoint: {checkpoint_path}")

    trainer.fit(
        lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )