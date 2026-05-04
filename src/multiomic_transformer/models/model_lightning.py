import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import wandb
import time

from torchmetrics.regression import R2Score

from pathlib import Path
import sys
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
SRC_DIR = str(PROJECT_DIR / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from multiomic_transformer.models.model_simplified import MultiomicTransformer
from multiomic_transformer.utils.experiment_handler import ExperimentHandler

# ================================================================
# STEP-BASED WARMUP + COSINE LR
# ================================================================
class LitMultiomicTransformer(pl.LightningModule):
    def __init__(
        self, 
        exp: ExperimentHandler,
        model: MultiomicTransformer = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["exp", "model"])

        self.model = model
        self.exp = exp
        self.learning_rate = exp.initial_lr
        self.scheduler_reduction_factor = exp.scheduler_reduction_factor
        self.scheduler_patience_epochs = exp.scheduler_patience_epochs
        self.scheduler_cooldown_epochs = exp.scheduler_cooldown_epochs

        self.tf_scaler = exp.tf_scaler
        self.tg_scaler = exp.tg_scaler
        
        self.epoch_step_times = []

        self.val_r2_scaled = R2Score()
        self.val_r2_unscaled = R2Score()

    def forward(self, atac_wins, tf_tensor, tf_ids=None, tg_ids=None, bias=None):
        return self.model(
            atac_wins,
            tf_tensor,
            tf_ids=tf_ids,
            tg_ids=tg_ids,
            bias=bias,
        )

    @staticmethod
    def _match_pred_target_shape(preds, targets):
        if preds.shape != targets.shape:
            if preds.ndim == targets.ndim + 1 and preds.shape[-1] == 1:
                preds = preds.squeeze(-1)
        return preds

    def training_step(self, batch, batch_idx):
        atac_wins, tf_tensor, targets_unscaled, bias, tf_ids, tg_ids, _ = batch
        
        tf_tensor = self.tf_scaler.transform(tf_tensor, tf_ids)
        targets_scaled = self.tg_scaler.transform(targets_unscaled, tg_ids)
        
        preds_scaled = self.model(
            atac_wins,
            tf_tensor,
            tf_ids=tf_ids,
            tg_ids=tg_ids,
            bias=bias,
        )

        preds_scaled = self._match_pred_target_shape(preds_scaled, targets_scaled)

        train_mse_scaled = F.mse_loss(
            preds_scaled,
            targets_scaled.float(),
        )

        self.log(
            "train/mse_scaled",
            train_mse_scaled,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return train_mse_scaled

    def validation_step(self, batch, batch_idx):
        atac_wins, tf_tensor, targets_unscaled, bias, tf_ids, tg_ids, _ = batch
        
        tf_tensor = self.tf_scaler.transform(tf_tensor, tf_ids)
        targets_scaled = self.tg_scaler.transform(targets_unscaled, tg_ids)

        preds_scaled = self.model(
            atac_wins,
            tf_tensor,
            tf_ids=tf_ids,
            tg_ids=tg_ids,
            bias=bias,
        )

        preds_scaled = self._match_pred_target_shape(preds_scaled, targets_scaled)

        preds_unscaled = self.tg_scaler.inverse_transform(preds_scaled, tg_ids)

        val_mse_scaled = F.mse_loss(
            preds_scaled,
            targets_scaled.float(),
        )

        val_mse_unscaled = F.mse_loss(
            preds_unscaled,
            targets_unscaled.float(),
        )

        self.val_r2_scaled.update(
            preds_scaled.detach().float().reshape(-1),
            targets_scaled.detach().float().reshape(-1),
        )

        self.val_r2_unscaled.update(
            preds_unscaled.detach().float().reshape(-1),
            targets_unscaled.detach().float().reshape(-1),
        )

        self.log(
            "val/mse_scaled",
            val_mse_scaled,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        self.log(
            "val/mse_unscaled",
            val_mse_unscaled,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/r2_scaled",
            self.val_r2_scaled,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/r2_unscaled",
            self.val_r2_unscaled,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self, "_batch_start_time"):
            step_time = time.time() - self._batch_start_time

            self.epoch_step_times.append(step_time)

            self.log(
                "train/step_time_sec",
                step_time,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )

    def on_train_epoch_end(self):
        if len(self.epoch_step_times) > 0:
            avg_step_time = float(np.mean(self.epoch_step_times))
            total_step_time = float(np.sum(self.epoch_step_times))

            self.log(
                "train/avg_step_time_sec",
                avg_step_time,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            self.log(
                "train/total_train_time_sec",
                total_step_time,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            self.epoch_step_times = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_reduction_factor,
            patience=self.scheduler_patience_epochs,
            cooldown=self.scheduler_cooldown_epochs,
            threshold=1e-4,
            min_lr=1e-7,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/mse_unscaled",   # or "val/mse_scaled"
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
                "name": "lr_plateau",
            },
        }