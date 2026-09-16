"""Cell-type edge-bag model from test_build_celltype_tf_tg_data.ipynb."""
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
                 peak_accessibility_std=1.0):
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

    def forward(self, batch):
        return self.model(batch, self.hparams.pooling_temperature)

    def _step(self, batch, stage):
        logits, _ = self(batch)
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

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def on_validation_epoch_start(self):
        self._predictions["val"].clear()

    def on_test_epoch_start(self):
        self._predictions["test"].clear()

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
                self.log(key, float(value), prog_bar=name is None)
                if group == "celltype" or (group and group.endswith("/celltype")):
                    macro[metric].append(value)
        for metric, values in macro.items():
            if values:
                self.log(f"{stage}/macro_celltype_{metric}", float(np.mean(values)))
        self.log(f"{stage}/n_scorable_celltypes", float(len(macro["auroc"])))

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
