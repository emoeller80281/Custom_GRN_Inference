from html import parser
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

from multiomic_transformer.models import model_lightning
import multiomic_transformer.utils.data_formatter as data_formatter
import multiomic_transformer.utils.experiment_handler as experiment_handler
from multiomic_transformer.models.model_simplified import MultiomicTransformer

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)

torch.set_float32_matmul_precision('medium')

import argparse

parser = argparse.ArgumentParser(description="Train MultiomicTransformer with Weights and Biases logging")
parser.add_argument("--model", type=str, help="Model name for W&B logging")
parser.add_argument("--bias_scale", type=float, default=0.1, help="Scale of the distance bias")
parser.add_argument("--d_ff", type=int, default=2048, help="Dimension of the feed-forward layer")
parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--kernel_size", type=int, default=16, help="Kernel size for window pooling")
parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
parser.add_argument("--use_dist_bias", type=str, default="false", help="Whether to use distance bias")

args = parser.parse_args()

GROUND_TRUTH_DIR = PROJECT_DIR / "data" / "ground_truth_files"
DATA_DIR = Path("/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/PROCESSED_DATA")

# %%
DATA_DIR = Path("/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/")

cell_type="mESC"
sample_name="E7.5_rep1"
experiment_name=f"{cell_type}_{sample_name}_tutorial"
organism_code="mm10"

tdf = data_formatter.load_tdf(
    settings_path = DATA_DIR / "PROCESSED_DATA" / experiment_name / "settings.json"
)

# Verify that the data cache files exist. If not, this method will create them.
tdf.create_or_load_data_cache(sample_name=tdf.sample_names[0], force_recalculate=False)

logging.info("Initializing ExperimentHandler...")
exp = experiment_handler.ExperimentHandler(
    training_data_formatter=tdf,
    experiment_dir=DATA_DIR / "EXPERIMENTS",
    model_num=1,
    silence_warnings=False,
)

logging.info("Creating dataset...")
exp.create_multichrom_dataset(max_cached=100)

logging.info("Preparing DataLoader...")
train_loader, val_loader, test_loader = exp.prepare_dataloader(
    batch_size=64,
    num_workers=8
)

# Creates scalers for the RNA and ATAC data based on the training split.
logging.info("Creating scalers...")
exp.create_scalers(train_loader)

exp.d_model = args.d_model
exp.num_heads = args.num_heads
exp.num_layers = args.num_layers
exp.d_ff = args.d_ff
exp.dropout = args.dropout
exp.use_dist_bias = True if args.use_dist_bias == "true" else False
exp.bias_scale = args.bias_scale
exp.kernel_size = args.kernel_size
exp.print_model_settings()

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
        filename="{epoch:02d}-{val/roc_auc:.4f}-{val/loss:.4f}",
        monitor="val/mse_unscaled",
        mode="min",
        save_top_k=3,
        save_last=True,
    ),
    EarlyStopping(
        monitor="val/mse_unscaled",
        mode="min",
        patience=5,
    ),
    LearningRateMonitor(logging_interval="epoch"),
]

rank_zero_info("Setting up Weights and Biases logger")
wandb_logger = None
wandb_logger_dir = exp.model_training_dir / "wandb_logs"
wandb_logger_dir.mkdir(parents=True, exist_ok=True)

wandb_logger = WandbLogger(
    project="multiomic_transformer",
    name=exp.experiment_name,
    save_dir=wandb_logger_dir,
    log_model=True,
)

wandb_logger.experiment.config.update({
    "model": "MultiomicTransformer",
    "d_model": exp.d_model,
    "num_heads": exp.num_heads,
    "num_layers": exp.num_layers,
    "d_ff": exp.d_ff,
    "dropout": exp.dropout,
    "use_dist_bias": exp.use_dist_bias,
    "bias_scale": exp.bias_scale,
    "kernel_size": exp.kernel_size,
})

# Hide metrics from auto-generated W&B charts
wandb_logger.experiment.define_metric("trainer/global_step", hidden=True)
wandb_logger.experiment.define_metric("epoch", hidden=True)
wandb_logger.experiment.define_metric("lr-AdamW", hidden=True)

rank_zero_info("Setting up PyTorch Lightning Trainer...")
trainer = pl.Trainer(
    max_epochs=exp.epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    strategy="auto",
    logger=wandb_logger,
    callbacks=[TQDMProgressBar(refresh_rate=200)],
    gradient_clip_val=1.0,
    deterministic=True,
    default_root_dir=output_dir,
    enable_progress_bar=True,
    enable_checkpointing=True,
    check_val_every_n_epoch=1,
)

importlib.reload(model_lightning)
lit_model = model_lightning.LitMultiomicTransformer(
    exp=exp,
    model=model
)

trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
wandb.finish()

exp.model = lit_model.model

# Runs gradient attribution to calculate the gradients between each TF input and each TG output.
logging.info("\nRunning Gradient Attribution")
atac_grn_df, batch_atac_grn_df = exp.run_atac_gradient_attribution(
    test_loader,
    chunk_size=1,
    max_batches=None,
    show_tqdm=True,
    save_every_n_batches=100,
    )

# Runs gradient attribution to calculate the gradients between each TF input and each TG output.
logging.info("\nRunning Gradient Attribution")
grn_df, batch_grn_df = exp.run_gradient_attribution(
    test_loader,
    chunk_size=1,
    max_batches=None,
    max_tgs_per_batch=None,
    show_tqdm=True,
    save_every_n_batches=100,
    )

# Loads a ground truth file with columns "Source" and "Target" for TF-TG interactions.
logging.info("Loading ground truth datasets...")
GROUND_TRUTH_DIR = Path(PROJECT_DIR) / "data" / "ground_truth_files"
gt_by_dataset_dict = {
    "ChIP-Atlas mESC": exp.load_ground_truth(GROUND_TRUTH_DIR / "chip_atlas_tf_peak_tg_dist.csv"),
    "RN111": exp.load_ground_truth(GROUND_TRUTH_DIR / "RN111.tsv"),
    "RN112": exp.load_ground_truth(GROUND_TRUTH_DIR / "RN112.tsv"),
    "RN114": exp.load_ground_truth(GROUND_TRUTH_DIR / "RN114.tsv"),
    "RN116": exp.load_ground_truth(GROUND_TRUTH_DIR / "RN116.tsv"),        
}

# Calculates the AUROC of the predicted GRN against multiple ground truth datasets.
logging.info("\nCalculating AUROC")
auroc_df = exp.calculate_auroc_all_sample_gts(exp.grn, gt_by_dataset_dict)     
logging.info(f"Pooled Median AUROC: {auroc_df['pooled_median_auroc'].iloc[0]:.3f}")       
logging.info(f"Per-TF Median AUROC: {auroc_df['per_tf_median_auroc'].iloc[0]:.3f}")

exp.save_handler()


