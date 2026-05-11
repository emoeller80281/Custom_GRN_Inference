import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy

from bidirectional_cross_attention import BidirectionalCrossAttentionTransformer

class TFPeakBindingModel(nn.Module):
    def __init__(
        self,
        tf_embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
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
            # [B, 4, 512] -> [B, 64, 512]
            nn.Conv1d(4, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(4),  # 512 -> 128

            # Second Conv layer to reduce dimensionality
            # [B, 64, 128] -> [B, hidden_dim, 128]
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
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: float | None = None,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

        # Save hyperparameters except the raw nn.Module object
        self.save_hyperparameters(ignore=["model"])

        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight_tensor)
        else:
            self.pos_weight = None

        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()

        self.train_auprc = BinaryAveragePrecision()
        self.val_auprc = BinaryAveragePrecision()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

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
        tf_embedding = batch["tf_embedding"]
        tf_mask = batch["tf_mask"]
        peak_embedding = batch["peak_embedding"]
        labels = batch["label"].float()

        logits = self(
            tf_embedding=tf_embedding,
            tf_mask=tf_mask,
            peak_embedding=peak_embedding,
        )

        loss = self._loss(logits, labels)
        probs = torch.sigmoid(logits)

        if stage == "train":
            auroc = self.train_auroc(probs, labels.int())
            auprc = self.train_auprc(probs, labels.int())
            acc = self.train_acc(probs, labels.int())
        elif stage == "val":
            auroc = self.val_auroc(probs, labels.int())
            auprc = self.val_auprc(probs, labels.int())
            acc = self.val_acc(probs, labels.int())
        else:
            raise ValueError(f"Unknown stage: {stage}")

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            f"{stage}/auroc",
            auroc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            f"{stage}/auprc",
            auprc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            f"{stage}/acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        return optimizer