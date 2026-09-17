"""Cell-type edge-bag model from test_build_celltype_tf_tg_data.ipynb."""
import time

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

class SimpleTFTGRegulationModel(nn.Module):
    def __init__(
        self,
        pretrained_tf_peak_model=None,
        d_model=128,
        num_heads=4,
        dropout=0.1,
        tf_expression_mean=0.0,
        tf_expression_std=1.0,
        tg_expression_mean=0.0,
        tg_expression_std=1.0,
        peak_accessibility_mean=0.0,
        peak_accessibility_std=1.0,
    ):
        super().__init__()

        self.tf_peak_model = pretrained_tf_peak_model
        if self.tf_peak_model is not None:
            self.tf_peak_model.requires_grad_(False)
            self.tf_peak_model.eval()

        # Values in the input matrices are already depth-normalized and log1p
        # transformed. These shared statistics are fitted from training edges only.
        # They are reconstructed from Lightning hyperparameters when loading a
        # checkpoint, so non-persistent buffers preserve old-checkpoint compatibility.
        scaler_values = {
            "tf_expression_mean": tf_expression_mean,
            "tf_expression_std": tf_expression_std,
            "tg_expression_mean": tg_expression_mean,
            "tg_expression_std": tg_expression_std,
            "peak_accessibility_mean": peak_accessibility_mean,
            "peak_accessibility_std": peak_accessibility_std,
        }
        for name in (
            "tf_expression_std", "tg_expression_std", "peak_accessibility_std"
        ):
            if float(scaler_values[name]) <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in scaler_values.items():
            self.register_buffer(name, torch.tensor(float(value)), persistent=False)

        # Binding, accessibility, scaled distance, distance weight.
        self.peak_feature_proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.tf_expr_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.tg_expr_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.tg_query_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.peak_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.tf_peak_model is not None:
            self.tf_peak_model.eval()
        return self

    def forward(self, batch, pooling_temperature=1.0):
        if pooling_temperature <= 0:
            raise ValueError("pooling_temperature must be positive")
        if not batch["cell_mask"].any(dim=1).all():
            raise ValueError("Every edge must contain at least one real cell")
        accessibility = batch["peak_accessibility"].float()
        distance = batch["peak_distance"].float()
        peak_mask = batch["peak_mask"].bool()
        cell_mask = batch["cell_mask"].bool()

        B, C, P = accessibility.shape

        # Standardize real entries and restore padding to zero. Biological zeros are
        # real measurements and remain part of the train-fitted distribution.
        tf_expression = (
            batch["tf_expression"].float() - self.tf_expression_mean
        ) / self.tf_expression_std
        tg_expression = (
            batch["tg_expression"].float() - self.tg_expression_mean
        ) / self.tg_expression_std
        accessibility = (
            accessibility - self.peak_accessibility_mean
        ) / self.peak_accessibility_std
        tf_expression = tf_expression.masked_fill(~cell_mask, 0)
        tg_expression = tg_expression.masked_fill(~cell_mask, 0)
        accessibility = accessibility.masked_fill(
            ~(cell_mask[:, :, None] & peak_mask[:, None, :]), 0
        )

        if "binding_score" in batch:
            binding = batch["binding_score"]
        else:
            if self.tf_peak_model is None:
                raise ValueError("Supply precomputed binding_score or a TF-DNA model")
            sequences = batch["peak_sequences"].float()
            L = sequences.shape[2]

            with torch.no_grad():
                binding = self.tf_peak_model(
                    tf_embedding=batch["tf_embedding"].repeat_interleave(P, dim=0),
                    tf_mask=batch["tf_mask"].repeat_interleave(P, dim=0),
                    peak_embedding=sequences.reshape(B * P, L, 4),
                ).reshape(B, P).sigmoid()

        # 2. Same distance transformations as your original model.
        abs_distance = distance.abs()
        distance_scaled = (abs_distance / 250_000.0).clamp(0, 1)
        distance_weight = torch.exp(-abs_distance / 50_000.0)

        binding = binding.masked_fill(~peak_mask, 0)
        distance_scaled = distance_scaled.masked_fill(~peak_mask, 0)
        distance_weight = distance_weight.masked_fill(~peak_mask, 0)

        accessibility = accessibility.masked_fill(
            ~peak_mask[:, None, :], 0
        )

        # 3. Build one token per peak for each sampled cell.
        peak_features = torch.stack(
            [
                binding[:, None, :].expand(B, C, P),
                accessibility,
                distance_scaled[:, None, :].expand(B, C, P),
                distance_weight[:, None, :].expand(B, C, P),
            ],
            dim=-1,
        )                                               # [B, C, P, 4]

        peak_tokens = self.peak_feature_proj(
            peak_features.reshape(B * C, P, 4)
        )                                               # [B*C, P, d_model]

        # 4. Expression-conditioned query.
        tf_token = self.tf_expr_proj(
            tf_expression.reshape(B * C, 1)
        )
        tg_token = self.tg_expr_proj(
            tg_expression.reshape(B * C, 1)
        )

        query = self.tg_query_proj(
            tf_token + tg_token
        ).unsqueeze(1)                                  # [B*C, 1, d_model]

        # 5. Attend to candidate peaks.
        valid_peaks = (
            peak_mask[:, None, :]
            .expand(B, C, P)
            .reshape(B * C, P)
        )

        # Empty bags receive zero peak context.
        # Only nonempty bags enter attention.
        has_peaks = valid_peaks.any(dim=1)
        peak_context = torch.zeros_like(tf_token)

        rows = has_peaks.nonzero(as_tuple=True)[0]
        for selected in rows.split(32768):
            if selected.numel() == 0:
                continue
            attended, _ = self.peak_attention(
                query=query[selected], key=peak_tokens[selected],
                value=peak_tokens[selected],
                key_padding_mask=~valid_peaks[selected], need_weights=False,
            )
            peak_context[selected] = self.norm(attended.squeeze(1))

        # 6. One logit per sampled cell.
        features = torch.cat(
            [peak_context, tf_token, tg_token],
            dim=-1,
        )

        cell_logits = self.classifier(features).reshape(B, C)

        # 7. Normalized log-sum-exp over real cells.
        # Each example must contain at least one real cell.
        temperature = pooling_temperature
        masked_logits = cell_logits.masked_fill(
            ~cell_mask, float("-inf")
        )

        edge_logits = temperature * (
            torch.logsumexp(masked_logits / temperature, dim=1)
            - cell_mask.sum(dim=1).float().log()
        )

        return edge_logits, cell_logits

class LitTFTGRegulationModel(pl.LightningModule):
    """Lightning training for precomputed binding scores and cell-type bags.

    Metrics are exact over the epoch on the single device used by the launcher.
    Checkpoints contain the downstream model; TF-DNA provenance is in run_config.json.
    """

    def __init__(self, d_model=128, num_heads=4, dropout=0.1, lr=1e-4,
                 weight_decay=1e-4, pooling_temperature=1.0, pos_weight=1.0,
                 plateau_patience=4, tf_expression_mean=0.0,
                 tf_expression_std=1.0, tg_expression_mean=0.0,
                 tg_expression_std=1.0, peak_accessibility_mean=0.0,
                 peak_accessibility_std=1.0, enable_timing_sync=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleTFTGRegulationModel(
            d_model=d_model, num_heads=num_heads, dropout=dropout,
            tf_expression_mean=tf_expression_mean,
            tf_expression_std=tf_expression_std,
            tg_expression_mean=tg_expression_mean,
            tg_expression_std=tg_expression_std,
            peak_accessibility_mean=peak_accessibility_mean,
            peak_accessibility_std=peak_accessibility_std)
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self._predictions = {"val": [], "test": []}
        self._celltype_metric_history = {"val": {}, "test": {}}
        self._hidden_wandb_metrics = set()
        self._prev_batch_end_time = None
        self._epoch_start_time = None
        self._step_start_time = None
        self._backward_start_time = None
        self._timing_window_size = 50
        self._timing_windows = {
            "load": [],
            "h2d": [],
            "forward": [],
            "backward": [],
            "step": [],
        }
        self._latest_timing_avgs = {}

    def _sync_if_cuda(self, device=None):
        if self.hparams.enable_timing_sync and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _record_timing(self, name, value):
        window = self._timing_windows[name]
        window.append(value)
        if len(window) > self._timing_window_size:
            window.pop(0)
        self._latest_timing_avgs[name] = sum(window) / len(window)

    def forward(self, batch):
        return self.model(batch, self.hparams.pooling_temperature)

    def _step(self, batch, stage):
        forward_start = None
        if stage == "train":
            self._sync_if_cuda()
            forward_start = time.perf_counter()

        logits, _ = self(batch)

        if forward_start is not None:
            self._sync_if_cuda()
            self._record_timing("forward", time.perf_counter() - forward_start)

        labels = batch["label"].float()
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits.float(), labels, pos_weight=self.pos_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {stage} loss")
        self.log(f"{stage}/loss", loss, on_step=stage == "train",
                 on_epoch=True, prog_bar=True, batch_size=len(labels))
        probs = logits.detach().float().sigmoid()
        self.log(f"{stage}/acc", ((probs >= .5) == labels.bool()).float().mean(),
                 on_step=False, on_epoch=True, batch_size=len(labels))
        if stage != "train":
            self._predictions[stage].append((
                probs.cpu(), labels.int().cpu(),
                list(batch["sample_id"]), list(batch["cell_type"])))
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not self.training:
            return batch
        if self._prev_batch_end_time is not None:
            self._record_timing(
                "load", time.perf_counter() - self._prev_batch_end_time
            )
        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.training:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        start_time = time.perf_counter()
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        self._sync_if_cuda(device)
        self._record_timing("h2d", time.perf_counter() - start_time)
        return batch

    def on_train_epoch_start(self):
        for window in self._timing_windows.values():
            window.clear()
        self._latest_timing_avgs.clear()
        self._prev_batch_end_time = None
        self._epoch_start_time = time.perf_counter()

    def on_train_batch_start(self, batch, batch_idx):
        self._sync_if_cuda()
        self._step_start_time = time.perf_counter()

    def on_before_backward(self, loss):
        self._sync_if_cuda()
        self._backward_start_time = time.perf_counter()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        start_time = self._backward_start_time or time.perf_counter()
        result = super().optimizer_step(
            epoch, batch_idx, optimizer, optimizer_closure
        )
        self._sync_if_cuda()
        self._record_timing("backward", time.perf_counter() - start_time)
        self._backward_start_time = None
        return result

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._step_start_time is None:
            return
        self._sync_if_cuda()
        self._record_timing("step", time.perf_counter() - self._step_start_time)
        self._step_start_time = None
        self._prev_batch_end_time = time.perf_counter()

        if batch_idx % 50 == 0:
            for name, avg_value in self._latest_timing_avgs.items():
                self.log(
                    f"train/{name}_time_avg", avg_value,
                    on_step=True, on_epoch=False, prog_bar=False,
                    logger=True, sync_dist=False,
                )

    def on_train_epoch_end(self):
        if self._epoch_start_time is None:
            return
        self._sync_if_cuda()
        epoch_time_mins = (
            time.perf_counter() - self._epoch_start_time
        ) / 60.0
        self._epoch_start_time = None
        self.log(
            "train/epoch_time_min", epoch_time_mins,
            on_step=False, on_epoch=True, prog_bar=True,
            logger=True, sync_dist=False,
        )

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def on_validation_epoch_start(self):
        self._predictions["val"].clear()

    def on_test_epoch_start(self):
        self._predictions["test"].clear()

    def _record_celltype_metric(self, stage, key, value):
        # The explicit step-zero validation is epoch 0. Validation after the first
        # training epoch is epoch 1, although Lightning's current_epoch is still 0.
        epoch = int(self.current_epoch)
        if stage == "val" and self.global_step > 0:
            epoch += 1

        series = self._celltype_metric_history[stage].setdefault(
            key.removeprefix(f"{stage}/"), {"epochs": [], "values": []}
        )
        if series["epochs"] and series["epochs"][-1] == epoch:
            series["values"][-1] = float(value)
        else:
            series["epochs"].append(epoch)
            series["values"].append(float(value))

        experiment = getattr(self.logger, "experiment", None)
        if (key not in self._hidden_wandb_metrics
                and callable(getattr(experiment, "define_metric", None))):
            # Retain the scalar history and summary without creating one automatic
            # W&B panel for every sample/cell-type/metric combination.
            experiment.define_metric(key, hidden=True)
            self._hidden_wandb_metrics.add(key)

    def _log_combined_celltype_metrics(self, stage):
        history = self._celltype_metric_history[stage]
        experiment = getattr(self.logger, "experiment", None)
        if (not history or self.global_rank != 0
                or not callable(getattr(experiment, "define_metric", None))):
            return

        import wandb

        ordered = sorted(history.items())
        chart = wandb.plot.line_series(
            xs=[series["epochs"] for _, series in ordered],
            ys=[series["values"] for _, series in ordered],
            keys=[name for name, _ in ordered],
            title=f"{stage.capitalize()} cell-type AUROC and AUPRC",
            xname="Epoch",
        )
        self.logger.log_metrics(
            {f"{stage}/celltype_auroc_auprc": chart},
            step=self.global_step,
        )

    def _epoch_metrics(self, stage):
        batches = self._predictions[stage]
        if not batches:
            return
        probs = torch.cat([b[0] for b in batches]).numpy()
        labels = torch.cat([b[1] for b in batches]).numpy()
        samples = np.asarray([s for b in batches for s in b[2]])
        slices = np.asarray([s for b in batches for s in b[3]])
        batches.clear()
        unique_samples = np.unique(samples)
        groups = [(None, None, np.ones(len(labels), dtype=bool))]
        if len(unique_samples) == 1:
            groups.extend(("celltype", s, slices == s) for s in np.unique(slices))
        else:
            for sample in unique_samples:
                sample_mask = samples == sample
                groups.append(("sample", sample, sample_mask))
                for cell_type in np.unique(slices[sample_mask]):
                    groups.append((
                        f"sample/{sample}/celltype", cell_type,
                        sample_mask & (slices == cell_type),
                    ))
        macro = {"auroc": [], "auprc": []}
        for group, name, mask in groups:
            if len(np.unique(labels[mask])) < 2:
                continue
            values = {"auroc": roc_auc_score(labels[mask], probs[mask]),
                      "auprc": average_precision_score(labels[mask], probs[mask])}
            for metric, value in values.items():
                key = f"{stage}/{metric}" if name is None else f"{stage}/{group}/{name}/{metric}"
                is_celltype = group == "celltype" or (
                    group and group.endswith("/celltype")
                )
                if is_celltype:
                    self._record_celltype_metric(stage, key, value)
                self.log(key, float(value), prog_bar=name is None)
                if is_celltype:
                    macro[metric].append(value)
        for metric, values in macro.items():
            if values:
                self.log(f"{stage}/macro_celltype_{metric}", float(np.mean(values)))
        self._log_combined_celltype_metrics(stage)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["celltype_metric_history"] = self._celltype_metric_history

    def on_load_checkpoint(self, checkpoint):
        history = checkpoint.get("celltype_metric_history")
        if history is not None:
            self._celltype_metric_history = history

    def on_validation_epoch_end(self):
        self._epoch_metrics("val")

    def on_test_epoch_end(self):
        self._epoch_metrics("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=.5, patience=self.hparams.plateau_patience)
        return {"optimizer": optimizer, "lr_scheduler": {
            "scheduler": scheduler, "monitor": "val/loss"}}
