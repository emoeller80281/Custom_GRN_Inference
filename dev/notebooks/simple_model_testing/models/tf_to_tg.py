import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
)
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import wandb
import time

class TFTGRegulationModel(nn.Module):
    def __init__(
        self,
        pretrained_tf_peak_model,
        d_model,
        num_heads=4,
        dropout=0.1,
        tf_peak_chunk_size=256,
    ):
        super().__init__()

        self.tf_peak_model = pretrained_tf_peak_model
        self.tf_peak_chunk_size = tf_peak_chunk_size

        self.peak_feature_proj = nn.Sequential(
            nn.Linear(4, d_model),  # binding, accessibility, distance_scaled, distance_weight
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

        # peak_context + tf_expr + tg_expr
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def pool_cell_logits(
        cell_logits,
        cell_mask=None,
        mode="lse",
        temperature=1.0,
    ):
        """
        cell_logits: [E, C]
        cell_mask: [E, C], True for real cells, False for padded cells
        """

        if cell_mask is None:
            cell_mask = torch.ones_like(cell_logits, dtype=torch.bool)

        if mode == "mean":
            masked_logits = cell_logits.masked_fill(~cell_mask, 0.0)
            denom = cell_mask.sum(dim=1).clamp_min(1)
            return masked_logits.sum(dim=1) / denom

        elif mode == "max":
            masked_logits = cell_logits.masked_fill(~cell_mask, float("-inf"))
            return masked_logits.max(dim=1).values

        elif mode == "lse":
            masked_logits = cell_logits.masked_fill(~cell_mask, float("-inf"))
            n_cells = cell_mask.sum(dim=1).clamp_min(1)

            return temperature * (
                torch.logsumexp(masked_logits / temperature, dim=1)
                - torch.log(n_cells.float())
            )

        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

    def forward(
        self,
        tf_embedding,
        tf_mask,
        peak_sequences,
        peak_accessibility,
        peak_distance,
        tf_expression,
        tg_expression,
        cell_mask,
        peak_mask=None,
        pooling_mode: str = "lse",
        pooling_temperature: float = 1.0,
    ):
        """
        Bag-level forward pass.

        This computes TF-DNA binding once per TF-TG edge and peak,
        then reuses those binding scores across sampled cells.

        Parameters
        ----------
        tf_embedding : [E T, D]
        tf_mask : [E, T]
        peak_sequences : [E, P, L, 4]
        peak_accessibility : [E, C, P]
        peak_distance : [E, P]
        tf_expression : [E, C]
        tg_expression : [E, C]
        cell_mask : [E, C]
        peak_mask : [E, P], optional

        Returns
        -------
        edge_logits : [E]
        cell_logits : [E, C]
        """

        if not torch.is_floating_point(peak_sequences):
            peak_sequences = peak_sequences.float()

        E, C = cell_mask.shape
        _, P, L, nuc_dim = peak_sequences.shape
        EC = E * C

        # ------------------------------------------------------------
        # 1. Cell-invariant edge-level tensors
        # ------------------------------------------------------------
        # These are repeated across cells in your current dataloader.
        # Use only the first cell to avoid C-fold redundant TF-DNA inference.
        tf_embedding_edge = tf_embedding         # [E, T, D]
        tf_mask_edge = tf_mask                  # [E, T]
        peak_sequences_edge = peak_sequences    # [E, P, L, 4]
        peak_distance_edge = peak_distance        # [E, P]

        if peak_mask is not None:
            peak_mask_edge = peak_mask            # [E, P]
        else:
            peak_mask_edge = None

        # ------------------------------------------------------------
        # 2a. Frozen TF-DNA binding model: [E, P]
        # ------------------------------------------------------------

        # Flatten the peaks into a single batch dimension of ExP
        peak_seq_flat = peak_sequences_edge.reshape(E * P, L, nuc_dim)

        chunk_size = self.tf_peak_chunk_size
        if chunk_size is None or chunk_size <= 0:
            chunk_size = E * P

        with torch.no_grad():
            binding_logits_flat = torch.empty(
                E * P,
                device=peak_sequences_edge.device,
                dtype=peak_sequences_edge.dtype,
            )

            for start in range(0, E * P, chunk_size):
                end = min(start + chunk_size, E * P)

                flat_idx = torch.arange(start, end, device=peak_sequences_edge.device)
                edge_idx = flat_idx // P

                tf_embedding_chunk = tf_embedding_edge[edge_idx]
                tf_mask_chunk = tf_mask_edge[edge_idx]
                peak_seq_chunk = peak_seq_flat[start:end]

                logits_chunk = self.tf_peak_model(
                    tf_embedding=tf_embedding_chunk,
                    tf_mask=tf_mask_chunk,
                    peak_embedding=peak_seq_chunk,
                )

                # Copy values out before next compiled-model invocation
                binding_logits_flat[start:end].copy_(logits_chunk)

        binding_logits = binding_logits_flat.reshape(E, P)
        
        # ------------------------------------------------------------
        # 2b. Mask and expand TF-peak binding scores across cells
        # ------------------------------------------------------------
        # Sigmoid to convert logits to probabilities
        binding_score = torch.sigmoid(binding_logits)  # [E, P]

        # If a peak mask is provided, set binding scores of masked peaks to 0
        if peak_mask_edge is not None:
            binding_score = binding_score.masked_fill(~peak_mask_edge, 0.0)

        # Reuse TF-peak binding score across cells
        binding_score = binding_score[:, None, :].expand(E, C, P)  # [E, C, P]

        # ------------------------------------------------------------
        # 3. Distance features
        # ------------------------------------------------------------
        abs_distance = peak_distance_edge.abs()
        distance_scaled = torch.clamp(abs_distance / 250_000.0, 0.0, 1.0)   # [E, P]
        distance_weight = torch.exp(-abs_distance / 50_000.0)               # [E, P]

        if peak_mask_edge is not None:
            distance_scaled = distance_scaled.masked_fill(~peak_mask_edge, 0.0)
            distance_weight = distance_weight.masked_fill(~peak_mask_edge, 0.0)

        distance_scaled = distance_scaled[:, None, :].expand(E, C, P) # [E, C, P]
        distance_weight = distance_weight[:, None, :].expand(E, C, P) # [E, C, P]

        # ------------------------------------------------------------
        # 4. Cell-specific peak features
        # ------------------------------------------------------------
        if peak_mask_edge is not None:
            peak_accessibility = peak_accessibility.masked_fill(
                ~peak_mask_edge[:, None, :],
                0.0,
            )
            
        assert binding_score.shape == peak_accessibility.shape, (
            f"binding_score {binding_score.shape} != peak_accessibility {peak_accessibility.shape}"
        )
        assert distance_scaled.shape == peak_accessibility.shape, (
            f"distance_scaled {distance_scaled.shape} != peak_accessibility {peak_accessibility.shape}"
        )
        assert distance_weight.shape == peak_accessibility.shape, (
            f"distance_weight {distance_weight.shape} != peak_accessibility {peak_accessibility.shape}"
        )

        peak_features = torch.stack(
            [
                binding_score,
                peak_accessibility,
                distance_scaled,
                distance_weight,
            ],
            dim=-1,
        )  # [E, C, P, 4]

        peak_features = peak_features.reshape(EC, P, 4)  # [E*C, P, 4]
        peak_tokens = self.peak_feature_proj(peak_features)  # [E*C, P, d_model]

        # ------------------------------------------------------------
        # 5. Expression tokens
        # ------------------------------------------------------------
        tf_expr_token = self.tf_expr_proj(
            tf_expression.reshape(EC, 1)
        )  # [E*C, d_model]

        tg_expr_token = self.tg_expr_proj(
            tg_expression.reshape(EC, 1)
        )  # [E*C, d_model]

        tg_query_input = tf_expr_token + tg_expr_token

        tg_query = self.tg_query_proj(tg_query_input).unsqueeze(1)  # [E*C, 1, d_model]

        # ------------------------------------------------------------
        # 6. TG query attends to linked peak tokens
        # ------------------------------------------------------------
        key_padding_mask = None

        if peak_mask_edge is not None:
            key_padding_mask = peak_mask_edge[:, None, :].expand(E, C, P)
            key_padding_mask = ~key_padding_mask.reshape(EC, P)  # True = ignore

        peak_context, _ = self.peak_attention(
            query=tg_query,
            key=peak_tokens,
            value=peak_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        peak_context = self.norm(peak_context.squeeze(1))  # [E*C, d_model]

        # ------------------------------------------------------------
        # 7. Cell-level logits
        # ------------------------------------------------------------
        final = torch.cat(
            [
                peak_context,
                tf_expr_token,
                tg_expr_token,
            ],
            dim=-1,
        )  # [E*C, d_model * 3]

        cell_logits = self.classifier(final).squeeze(-1)  # [E*C]
        cell_logits = cell_logits.reshape(E, C)           # [E, C]

        # ------------------------------------------------------------
        # 8. Pool cell logits into edge logits
        # ------------------------------------------------------------
        edge_logits = self.pool_cell_logits(
            cell_logits,
            cell_mask=cell_mask,
            mode=pooling_mode,
            temperature=pooling_temperature,
        )  # [E]

        return edge_logits, cell_logits
    
class LitTFTGRegulationModel(pl.LightningModule):
    def __init__(
        self,
        model: TFTGRegulationModel,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: float | None = None,
        pooling_mode: str = "lse",
        pooling_temperature: float = 1.0,
        logit_clamp: float | None = 20.0,
        enable_timing_sync: bool = False,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.pooling_mode = pooling_mode
        self.pooling_temperature = pooling_temperature
        self.logit_clamp = logit_clamp
        self.enable_timing_sync = enable_timing_sync

        self.save_hyperparameters(ignore=["model"])

        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight_tensor)
        else:
            self.pos_weight = None

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

        self.val_probs = []
        self.val_targets = []
        self._prev_batch_end_time = None
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

    def _sync_if_cuda(self, device=None) -> None:
        if self.enable_timing_sync and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _record_timing(self, name: str, value: float) -> None:
        window = self._timing_windows[name]
        window.append(value)
        if len(window) > self._timing_window_size:
            window.pop(0)

        self._latest_timing_avgs[name] = sum(window) / len(window)

    def forward(self, batch):
        
        return self.model(
            tf_embedding=batch["tf_embedding"],
            tf_mask=batch["tf_mask"],
            peak_sequences=batch["peak_sequences"],
            peak_accessibility=batch["peak_accessibility"],
            peak_distance=batch["peak_distance"],
            tf_expression=batch["tf_expression"],
            tg_expression=batch["tg_expression"],
            cell_mask=batch["cell_mask"],
            peak_mask=batch.get("peak_mask", None),
            pooling_mode=self.pooling_mode,
            pooling_temperature=self.pooling_temperature,
        )

    def _loss(self, logits, labels):
        if self.pos_weight is not None:
            return nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=self.pos_weight,
            )

        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
        )

    def _shared_step(self, batch, stage: str):
        labels = batch["label"].float()

        forward_start = None
        if stage == "train":
            self._sync_if_cuda()
            forward_start = time.perf_counter()

        edge_logits, _ = self.forward(batch)

        if forward_start is not None:
            self._sync_if_cuda()
            forward_time = time.perf_counter() - forward_start
            self._record_timing("forward", forward_time)

        if self.logit_clamp is not None:
            edge_logits = edge_logits.clamp(min=-self.logit_clamp, max=self.logit_clamp)

        loss = self._loss(edge_logits, labels)
        probs = torch.sigmoid(edge_logits)

        if stage == "train":
            acc = self.train_acc(probs, labels.int())
        elif stage == "val":
            acc = self.val_acc(probs, labels.int())

            self.val_probs.append(probs.detach().float().cpu())
            self.val_targets.append(labels.detach().int().cpu())
        else:
            raise ValueError(f"Unknown stage: {stage}")

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=(stage != "train"),
        )

        self.log(
            f"{stage}/acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=(stage != "train"),
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not self.training:
            return batch

        if self._prev_batch_end_time is not None:
            load_time = time.perf_counter() - self._prev_batch_end_time
            self._record_timing("load", load_time)

        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.training:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        start_time = time.perf_counter()
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        self._sync_if_cuda(device)
        h2d_time = time.perf_counter() - start_time
        self._record_timing("h2d", h2d_time)
        return batch
    
    def on_train_epoch_start(self):
        for k in self._timing_windows:
            self._timing_windows[k].clear()
        self._latest_timing_avgs.clear()
        self._prev_batch_end_time = None

    def on_train_batch_start(self, batch, batch_idx):
        self._sync_if_cuda()
        self._step_start_time = time.perf_counter()

    def on_before_backward(self, loss):
        self._sync_if_cuda()
        self._backward_start_time = time.perf_counter()

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        start_time = self._backward_start_time or time.perf_counter()
        result = super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure,
        )
        self._sync_if_cuda()
        backward_opt_time = time.perf_counter() - start_time
        self._backward_start_time = None
        self._record_timing("backward", backward_opt_time)
        return result

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._step_start_time is None:
            return

        self._sync_if_cuda()
        step_time = time.perf_counter() - self._step_start_time
        self._step_start_time = None
        self._record_timing("step", step_time)
        self._prev_batch_end_time = time.perf_counter()

        if batch_idx % 50 == 0:
            for name, avg_value in self._latest_timing_avgs.items():
                self.log(
                    f"train/{name}_time_avg",
                    avg_value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def on_validation_epoch_start(self):
        self.val_probs.clear()
        self.val_targets.clear()

    def on_validation_epoch_end(self):
        if not self.val_probs:
            return

        probs = torch.cat(self.val_probs, dim=0).view(-1)
        targets = torch.cat(self.val_targets, dim=0).view(-1).int()

        self.val_probs.clear()
        self.val_targets.clear()

        if len(np.unique(targets)) >= 2:
            auroc = roc_auc_score(targets, probs)
            auprc = average_precision_score(targets, probs)
        else:
            auroc = np.nan
            auprc = np.nan

        self.log("val/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/auprc", auprc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


        if not getattr(self.logger, "experiment", None):
            return

        if not self.trainer.is_global_zero:
            return

        try:
            p_np = probs.detach().cpu().numpy()
            y_np = targets.detach().cpu().numpy()

            pre, rec, _ = precision_recall_curve(y_np, p_np)
            fpr, tpr, _ = roc_curve(y_np, p_np)

            def _sample_curve(x, y, n=100):
                if len(x) <= n:
                    return x, y
                idx = np.linspace(0, len(x) - 1, n).astype(int)
                return x[idx], y[idx]

            rec_s, pre_s = _sample_curve(rec, pre, n=100)
            fpr_s, tpr_s = _sample_curve(fpr, tpr, n=100)

            self.logger.experiment.log({
                "val/pr_curve": wandb.plot.line_series(
                    [rec_s],
                    [pre_s],
                    keys=["precision"],
                    xname="Recall",
                ),
                "val/roc_curve": wandb.plot.line_series(
                    [fpr_s],
                    [tpr_s],
                    keys=["TPR"],
                    xname="FPR",
                ),
            })
        except Exception as e:
            print("[WARN] PR/ROC curve error:", e)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=5, 
            threshold=1e-4, 
            threshold_mode='rel', 
            cooldown=3, 
            min_lr=1e-7, 
            eps=1e-08
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",      # Adjust LR per 'epoch' or 'step'
                "frequency": 1,           # How often to step the scheduler
                "monitor": "val/loss",    # Metric to track for ReduceLROnPlateau
            },
        }

# ----- Utility Functions -----
@torch.no_grad()
def move_batch_to_device(batch, device):
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

