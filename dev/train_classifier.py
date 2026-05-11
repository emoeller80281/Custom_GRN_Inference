import os
import json
import torch
import pandas as pd
import logging
import importlib
from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image as PILImage
from IPython.display import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_info

import sys
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
SRC_DIR = str(PROJECT_DIR / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import multiomic_transformer.datasets.classifier_dataset as classifier_dataset
import multiomic_transformer.models.model_classifier as model_classifier
from multiomic_transformer.models import model_classifier_lightning
import multiomic_transformer.utils.classifier_data_formatter as data_formatter
import multiomic_transformer.utils.experiment_classifier_handler as experiment_handler
importlib.reload(classifier_dataset)
importlib.reload(model_classifier)
importlib.reload(experiment_handler)
importlib.reload(model_classifier_lightning)
MultiomicTransformer = model_classifier.MultiomicTransformer

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)

torch.set_float32_matmul_precision('medium')

GROUND_TRUTH_DIR = PROJECT_DIR / "data" / "ground_truth_files"
DATA_DIR = Path("/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/")

cell_type="mESC"
sample_name="E7.5_rep1"
experiment_name=f"{cell_type}_{sample_name}_tutorial"
organism_code="mm10"
model_num=1

n_epochs = 20
batch_size = 24
dataloader_num_workers = 4

tdf = data_formatter.load_tdf(
    settings_path = DATA_DIR / "PROCESSED_DATA" / experiment_name / "settings.json"
)

# Verify that the data cache files exist. If not, this method will create them.
tdf.create_or_load_data_cache(sample_name=tdf.sample_names[0], force_recalculate=False)

# The ExperimentHandler is a higher level class that handles the training and evaluation of the model.
# It takes in a TrainingDataFormatter object to handle file paths, data loading, and caching.
logging.info("Initializing ExperimentHandler...")
exp = experiment_handler.ExperimentHandler(
    training_data_formatter=tdf,
    experiment_dir=DATA_DIR / "EXPERIMENTS",
    model_num=model_num,
    silence_warnings=False,
)

# Full-grid classifier training settings.
# Each forward pass materializes logits/features over [B, n_tfs, n_tgs], so keep B conservative.
exp.epochs = n_epochs
exp.batch_size = batch_size

logging.info("Creating dataset...")
exp.create_multichrom_dataset(max_cached=50)

# Prepares the Train/Val/Test dataloaders, being careful to balance the number of 
# batches from each chromosome in each set.
logging.info("Preparing DataLoader...")
train_loader, val_loader, test_loader = exp.prepare_dataloader(
    batch_size=exp.batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    persistent_workers=True,
)

# Creates scalers for the RNA and ATAC data based on a small sample of the training split.
logging.info("Creating scalers...")
exp.create_scalers(train_loader, max_batches=100)

# Verify the new full-grid batch contract before constructing the Lightning module.
atac_wins, tf_expr, tg_expr, labels, bias, tf_ids, tg_ids, motif_mask = next(iter(train_loader))
assert tf_expr.ndim == 2, f"tf_expr must be [B, n_tfs], got {tuple(tf_expr.shape)}"
assert tg_expr.ndim == 2, f"tg_expr must be [B, n_tgs], got {tuple(tg_expr.shape)}"
assert tf_ids.ndim == 1 and tf_ids.numel() == tf_expr.shape[1], (tuple(tf_ids.shape), tuple(tf_expr.shape))
assert tg_ids.ndim == 1 and tg_ids.numel() == tg_expr.shape[1], (tuple(tg_ids.shape), tuple(tg_expr.shape))
assert bias.ndim == 3 and bias.shape[:2] == (tf_expr.shape[0], tg_expr.shape[1]), (tuple(bias.shape), tuple(tg_expr.shape))
assert labels.shape == (tf_expr.shape[0], tf_expr.shape[1], tg_expr.shape[1]), (tuple(labels.shape), tuple(tf_expr.shape), tuple(tg_expr.shape))
logging.info(
    "Full-grid batch shapes: atac=%s tf_expr=%s tg_expr=%s labels=%s bias=%s tf_ids=%s tg_ids=%s",
    tuple(atac_wins.shape),
    tuple(tf_expr.shape),
    tuple(tg_expr.shape),
    tuple(labels.shape),
    tuple(bias.shape),
    tuple(tf_ids.shape),
    tuple(tg_ids.shape),
)
del atac_wins, tf_expr, tg_expr, labels, bias, tf_ids, tg_ids, motif_mask

tf_vocab_size = int(exp.dataset.tf_ids.numel())
tg_vocab_size = int(exp.dataset.tg_ids.numel())

model = MultiomicTransformer(
    d_model=exp.d_model,
    num_heads=exp.num_heads,
    num_layers=exp.num_layers,
    d_ff=exp.d_ff,
    dropout=exp.dropout,
    tf_vocab_size=tf_vocab_size,
    tg_vocab_size=tg_vocab_size,
    use_bias=exp.use_dist_bias,
    bias_scale=exp.bias_scale,
    window_pool_size=exp.kernel_size,
)

output_dir = exp.model_training_dir / "lightning_logs"
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# Callbacks
# -------------------------------------------------------
callbacks = [
    ModelCheckpoint(
        dirpath=output_dir,
        filename="{epoch:02d}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    ),
    EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=5,
    ),
    LearningRateMonitor(logging_interval="step"),
]

# -------------------------------------------------------
# W&B logger
# -------------------------------------------------------
rank_zero_info("Setting up Weights and Biases logger")
wandb_logger = None
wandb_logger_dir = exp.model_training_dir / "wandb_logs"
wandb_logger_dir.mkdir(parents=True, exist_ok=True)

wandb_logger = WandbLogger(
    project="multiomic_transformer_classifier",
    name=exp.experiment_name,
    save_dir=wandb_logger_dir,
    log_model=True,
)

wandb_logger.experiment.config.update({
    "model": "MultiomicTransformerClassifier",
    "d_model": exp.d_model,
    "num_heads": exp.num_heads,
    "num_layers": exp.num_layers,
    "d_ff": exp.d_ff,
    "dropout": exp.dropout,
    "use_dist_bias": exp.use_dist_bias,
    "bias_scale": exp.bias_scale,
    "kernel_size": exp.kernel_size,
    "batch_format": "full_grid",
    "logit_shape": "[B, n_tfs, n_tgs]",
})

# Hide metrics from auto-generated W&B charts
wandb_logger.experiment.define_metric("trainer/global_step", hidden=True)
wandb_logger.experiment.define_metric("epoch", hidden=True)
wandb_logger.experiment.define_metric("lr-AdamW", hidden=True)

world_size = int(
    os.environ.get(
        "WORLD_SIZE",
        os.environ.get("SLURM_NTASKS", "1"),
    )
)

use_ddp = world_size > 1


rank_zero_info("Setting up PyTorch Lightning Trainer...")
trainer = pl.Trainer(
    max_epochs=exp.epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=world_size if torch.cuda.is_available() else None,
    strategy="ddp_find_unused_parameters_true" if use_ddp else "auto",
    logger=wandb_logger,
    callbacks=[TQDMProgressBar(refresh_rate=200)] + callbacks,
    gradient_clip_val=1.0,
    deterministic=True,
    default_root_dir=output_dir,
    enable_progress_bar=True,
    enable_checkpointing=True,
    check_val_every_n_epoch=1,
    # limit_train_batches=1024,
    # limit_val_batches=256,
)

chipatlas_edge_df = exp.load_ground_truth(GROUND_TRUTH_DIR / "chip_atlas_tf_peak_tg_dist.csv")[0]

importlib.reload(model_classifier_lightning)
lit_model = model_classifier_lightning.LitMultiomicTransformer(
    exp=exp,
    model=model,
    ground_truth_edges=chipatlas_edge_df,
    ground_truth_name="ChIP-Atlas mESC",
    ground_truth_negative_ratio=10.0,
    max_ground_truth_pairs=10000
)

trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
wandb.finish()