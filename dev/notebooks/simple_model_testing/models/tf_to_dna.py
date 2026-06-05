import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy

from bidirectional_cross_attention import BidirectionalCrossAttentionTransformer

from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score, accuracy_score
)
import wandb
import time

class TFPeakBindingModel(nn.Module):
    def __init__(
        self,
        tf_embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_head: int = 32,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # TF protein sequence encoder:
        # [B, tf_len, 128] -> [B, tf_len, hidden_dim]
        self.tf_encoder = nn.Sequential(
            nn.Linear(tf_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Peak DNA sequence encoder:
        # [B, 4, 512] -> [B, 32, hidden_dim]
        self.peak_encoder = nn.Sequential(
            # First Conv layer to capture local motifs 
            # (sets of 15 nucleotides, roughly the size of a TF binding motif)
            # [B, 4, 512] -> [B, 64, 128]
            nn.Conv1d(4, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(4),  # 512 -> 128

            # Second Conv layer to reduce dimensionality
            # [B, 64, 128] -> [B, 128, 32]
            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(4),  # 128 -> 32

            # Third Conv layer to get to final hidden_dim
            # [B, 128, 32] -> [B, hidden_dim, 32]
            nn.Conv1d(128, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Bidirectional cross-attention transformer layers to allow TF and peak representations to attend to each other
        self.cross_attn = BidirectionalCrossAttentionTransformer(
            dim=hidden_dim,
            depth=num_layers,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            final_norms=True,
        )

        # Final classifier summarizes the attention output to estimate a binding probability
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def masked_mean_pool(self, x, mask):
        """
        x:    [B, seq_len, hidden_dim]
        mask: [B, seq_len], True for real tokens
        """
        mask = mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, tf_embedding, tf_mask, peak_embedding):
        """
        tf_embedding:   [B, max_tf_len, 128]
        tf_mask:        [B, max_tf_len]
        peak_embedding: [B, 512, 4]
        """

        # Encode TF tokens
        tf_tokens = self.tf_encoder(tf_embedding)
        # [B, max_tf_len, hidden_dim]

        # Encode peak tokens
        peak_x = peak_embedding.transpose(1, 2)
        # [B, 512, 4] -> [B, 4, 512]

        peak_tokens = self.peak_encoder(peak_x)
        # [B, hidden_dim, 32]

        peak_tokens = peak_tokens.transpose(1, 2)
        # [B, 32, hidden_dim]

        # Peak tokens are not padded, so all positions are valid
        peak_mask = torch.ones(
            peak_tokens.shape[:2],
            dtype=torch.bool,
            device=peak_tokens.device,
        )

        # Bidirectional cross-attention:
        # TF attends to peak, and peak attends to TF
        tf_tokens, peak_tokens = self.cross_attn(
            x=tf_tokens,
            context=peak_tokens,
            mask=tf_mask,
            context_mask=peak_mask,
        )

        # Pool sequence outputs
        tf_pooled = self.masked_mean_pool(tf_tokens, tf_mask)
        peak_pooled = peak_tokens.mean(dim=1)

        # Classify TF-peak pair
        joint = torch.cat([tf_pooled, peak_pooled], dim=-1)
        logits = self.classifier(joint).squeeze(-1)

        return logits
    

class LitTFPeakBindingModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        tf_embeddings_tensor: torch.Tensor,
        tf_mask_tensor: torch.Tensor,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: float | None = None,
        logit_clamp: float | None = 20.0,
        enable_timing_sync: bool = False,
    ):
        super().__init__()

        self.model = model
        
        self.register_buffer(
            "tf_embeddings_tensor",
            tf_embeddings_tensor.float(),
            persistent=False,
        )

        self.register_buffer(
            "tf_mask_tensor",
            tf_mask_tensor.bool(),
            persistent=False,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.logit_clamp = logit_clamp
        self.enable_timing_sync = enable_timing_sync

        # Save hyperparameters except the raw nn.Module object
        self.save_hyperparameters(
            ignore=[
                "model",
                "tf_embeddings_tensor",
                "tf_mask_tensor",
            ]
        )

        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight_tensor)
        else:
            self.pos_weight = None

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

    def forward(self, tf_embedding, tf_mask, peak_embedding):
        return self.model(
            tf_embedding=tf_embedding,
            tf_mask=tf_mask,
            peak_embedding=peak_embedding,
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
        tf_idx = batch["tf_idx"].long()
        labels = batch["label"].float()

        tf_embedding = self.tf_embeddings_tensor[tf_idx]
        tf_mask = self.tf_mask_tensor[tf_idx]
        peak_embedding = batch["peak_embedding"].float()

        forward_start = None
        if stage == "train":
            self._sync_if_cuda()
            forward_start = time.perf_counter()

        logits = self(
            tf_embedding=tf_embedding,
            tf_mask=tf_mask,
            peak_embedding=peak_embedding,
        )

        if forward_start is not None:
            self._sync_if_cuda()
            forward_time = time.perf_counter() - forward_start
            self._record_timing("forward", forward_time)

        if self.logit_clamp is not None:
            logits = logits.clamp(min=-self.logit_clamp, max=self.logit_clamp)

        loss = self._loss(logits, labels)

        if stage == "val":
            probs = torch.sigmoid(logits).detach().cpu()
            targets = labels.detach().cpu().int()

            self.val_probs.append(probs)
            self.val_targets.append(targets)

            self.log(
                "val/loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        else:
            self.log(
                "train/loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=False,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

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

    def on_after_backward(self):
        if self._backward_start_time is None:
            return

        self._sync_if_cuda()
        backward_time = time.perf_counter() - self._backward_start_time
        self._backward_start_time = None
        self._record_timing("backward", backward_time)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._step_start_time is None:
            return

        self._sync_if_cuda()
        step_time = time.perf_counter() - self._step_start_time
        self._step_start_time = None
        self._record_timing("step", step_time)
        self._prev_batch_end_time = time.perf_counter()

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
        
        val_auroc = roc_auc_score(targets, probs)
        val_auprc = average_precision_score(targets, probs)
        val_acc = accuracy_score(targets, probs >= 0.5)

        self.log("val/auroc", val_auroc, prog_bar=True, logger=True, sync_dist=False)
        self.log("val/auprc", val_auprc, prog_bar=True, logger=True, sync_dist=False)
        self.log("val/acc", val_acc, prog_bar=False, logger=True, sync_dist=False)

        if self.trainer.is_global_zero:
            try:
                p_np = probs.detach().cpu().numpy()
                y_np = targets.detach().cpu().numpy()

                pre, rec, _ = precision_recall_curve(y_np, p_np)
                fpr, tpr, _ = roc_curve(y_np, p_np)

                self.logger.experiment.log({
                    "val/pr_curve": wandb.plot.line_series([rec], [pre], keys=["precision"], xname="Recall"),
                    "val/roc_curve": wandb.plot.line_series([fpr], [tpr], keys=["TPR"], xname="FPR"),
                })
            except Exception as e:
                print("[WARN] PR/ROC curve error:", e)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        return optimizer
