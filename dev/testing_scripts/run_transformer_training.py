import os
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sc_multi_transformer import MultiomicTransformer
from transformer_dataset import WindowsWithTargets, make_collate

#=================================== USER SETTINGS ===================================
# ----- User Settings -----
load_model = False
window_size = 800
num_cells = 1000
chrom_id = "chr19"

atac_data_filename = "mESC_filtered_L2_E7.5_rep1_ATAC_processed.parquet"
rna_data_filename = "mESC_filtered_L2_E7.5_rep1_RNA_processed.parquet"

# ----- Model Configurations -----
# Logging
log_every_n_steps = 20

# Model Size
d_model = 256
tf_channels = 64
kernel_and_stride_size = 1

# Encoder Settings
encoder_nhead = 8
encoder_dim_feedforward = 1024
encoder_num_layers = 6
dropout = 0.1

# Train-test split
validation_fraction = 0.15

# Training configurations
epochs = 20
batch_size = 10
effective_batch_size = 16
warmup_steps = 100
learning_rate = 3e-4
weight_decay = 1e-4
patience = 5
epochs_before_patience = 3

#=====================================================================================

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
SAMPLE_INPUT_DIR = os.path.join(PROJECT_DIR, "input/mESC/filtered_L2_E7.5_rep1")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output/transformer_testing_output")
DEBUG_FILE = os.path.join(PROJECT_DIR, "LOGS/transformer_training.debug")

time_now = datetime.now().strftime("%d%m%y%H%M%S")
TRAINING_DIR = os.path.join(PROJECT_DIR, f"training_stats_{time_now}/")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, f"checkpoints_{time_now}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "training_stats"), exist_ok=True)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        gene_ids: torch.Tensor,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.gene_ids = gene_ids.to(gpu_id)  # [G] long
        self.log_every_n_steps = log_every_n_steps
        self.mu = None
        self.sd = None

        
    def _run_batch(self, tf_pad, bias_pad, kpm, targets, tmask):
        self.optimizer.zero_grad(set_to_none=True)

        # move to device
        tf_pad   = tf_pad.to(self.gpu_id, non_blocking=True)
        bias_pad = bias_pad.to(self.gpu_id, non_blocking=True)
        kpm      = kpm.to(self.gpu_id, non_blocking=True)
        targets  = targets.to(self.gpu_id, non_blocking=True)
        tmask    = tmask.to(self.gpu_id, non_blocking=True)  # float32 {0,1}

        # sanitize inputs (defensive)
        tf_pad   = torch.nan_to_num(tf_pad,   nan=0.0, posinf=0.0, neginf=0.0)
        bias_pad = torch.nan_to_num(bias_pad, nan=0.0, posinf=0.0, neginf=0.0)
        targets  = torch.nan_to_num(targets,  nan=0.0, posinf=0.0, neginf=0.0)

        tokens = self.model.window_from_tf(tf_pad)
        preds, _ = self.model(tokens, self.gene_ids, key_padding_mask=kpm,
                            attn_bias=bias_pad, already_pooled=True)

        # masked Huber on z-space if μ/σ are fitted
        if (self.mu is not None) and (self.sd is not None):
            preds_z   = (preds   - self.mu) / self.sd
            targets_z = (targets - self.mu) / self.sd
            per_elem = F.huber_loss(preds_z, targets_z, delta=1.0, reduction="none")
        else:
            per_elem = F.huber_loss(preds, targets, delta=1.0, reduction="none")

        num = (tmask > 0).sum().clamp(min=1)
        loss = (per_elem * tmask).sum() / num
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.detach().cpu())
        
    def _run_epoch(self, epoch):
        # print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.train_data)}")
        running = 0.0
        first_bsz_printed = False

        for step, batch in enumerate(self.train_data, start=1):
            tf_pad, bias_pad, kpm, targets, tmask = batch
            if not first_bsz_printed:
                # print(f"  batch size: {tf_pad.size(0)}")
                first_bsz_printed = True

            loss_val = self._run_batch(tf_pad, bias_pad, kpm, targets, tmask)
            running += loss_val

            # if step % self.log_every_n_steps == 0:
            #     print(f"  step {step} | loss {loss_val:.6f}")
                
    def _save_checkpoint(self, epoch):
        path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch{epoch}.pt")
        torch.save(self.model.state_dict(), path)
        # print(f"Epoch {epoch} | Training checkpoint saved at {path}")
        
    def train(self, max_epochs: int):
        for epoch in range(1, max_epochs+1):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        mae_sum = 0.0; mse_sum = 0.0; n = 0.0
        corr_sum = 0.0; corr_batches = 0
        for tf_pad, bias_pad, kpm, targets, tmask in loader:
            tf_pad   = tf_pad.to(self.gpu_id, non_blocking=True)
            bias_pad = bias_pad.to(self.gpu_id, non_blocking=True)
            kpm      = kpm.to(self.gpu_id, non_blocking=True)
            targets  = targets.to(self.gpu_id, non_blocking=True)
            tmask    = tmask.to(self.gpu_id, non_blocking=True)

            tf_pad   = torch.nan_to_num(tf_pad)
            bias_pad = torch.nan_to_num(bias_pad)
            targets  = torch.nan_to_num(targets)

            tokens = self.model.window_from_tf(tf_pad)
            preds, _ = self.model(tokens, self.gene_ids, key_padding_mask=kpm,
                                attn_bias=bias_pad, already_pooled=True)

            diff = (preds - targets)
            mae_sum += (tmask * diff.abs()).sum().item()
            mse_sum += (tmask * diff.pow(2)).sum().item()
            n += tmask.sum().item()

            # correlation
            corr = Trainer.masked_pearson(preds, targets, tmask).item()
            corr_sum += corr; corr_batches += 1

        mae = mae_sum / max(1.0, n)
        rmse = (mse_sum / max(1.0, n)) ** 0.5
        corr = corr_sum / max(1, corr_batches)
        self.model.train()
        return mae, rmse, corr
    
    @torch.no_grad()
    def fit_target_norm(self, loader):
        self.model.eval()
        sum_y = sum_y2 = sum_m = None
        for _, _, _, targets, tmask in loader:
            targets = targets.to(self.gpu_id)
            tmask   = tmask.to(self.gpu_id)
            if sum_y is None:
                G = targets.size(1)
                sum_y  = torch.zeros(G, device=self.gpu_id)
                sum_y2 = torch.zeros(G, device=self.gpu_id)
                sum_m  = torch.zeros(G, device=self.gpu_id)
            sum_y  += (targets * tmask).sum(dim=0)
            sum_y2 += (targets * targets * tmask).sum(dim=0)
            sum_m  += tmask.sum(dim=0)
        mu = sum_y / sum_m.clamp(min=1)
        var = sum_y2 / sum_m.clamp(min=1) - mu * mu
        sd = var.clamp_min(1e-6).sqrt()
        self.mu = mu
        self.sd = sd
        self.model.train()
    
    @staticmethod
    @torch.no_grad()
    def masked_baseline(val_loader):
        # predict per-gene train mean (or 0s if you don’t cache train mean yet)
        mae_sum = mse_sum = n = 0.0
        for tf_pad, bias_pad, kpm, targets, tmask in val_loader:
            diff = targets  # baseline=0
            mae_sum += (tmask * diff.abs()).sum().item()
            mse_sum += (tmask * diff.pow(2)).sum().item()
            n += tmask.sum().item()
        mae = mae_sum / max(1.0, n)
        rmse = (mse_sum / max(1.0, n)) ** 0.5
        print(f"[baseline] mae {mae:.4f} | rmse {rmse:.4f}")
    
    @staticmethod
    @torch.no_grad()
    def masked_pearson(preds, targets, mask, eps=1e-8):
        # preds, targets: [B,G], mask: [B,G] (0/1)
        m = mask > 0
        x = preds[m]; y = targets[m]
        x = x - x.mean(); y = y - y.mean()
        denom = (x.norm() * y.norm()).clamp_min(eps)
        return (x * y).sum() / denom

def load_train_objs():
    # paths
    meta_dir = Path(OUTPUT_DIR) / "precomputed_dataset" / f"{chrom_id}_{window_size//1000}kb" / "meta"
    manifest_path = meta_dir / "manifest.parquet"
    genes_path = meta_dir / "genes.npy"
    rna_path = Path(SAMPLE_INPUT_DIR) / rna_data_filename

    # dataset (features + targets)
    dataset = WindowsWithTargets(str(manifest_path), str(rna_path), str(genes_path), drop_empty=True)

    # infer dims from first non-empty sample
    sample = dataset[0]
    tf_in_dim = int(sample["tf_windows"].shape[1])
    n_genes = int(sample["target"].shape[0])

    # model
    model = MultiomicTransformer(
        n_genes=n_genes,
        tf_in_dim=tf_in_dim,
        d_model=d_model,
        nhead=encoder_nhead,
        dff=encoder_dim_feedforward,
        dropout=dropout,
        n_layers=encoder_num_layers,
        kernel_stride_size=kernel_and_stride_size,
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # gene ids tensor (used every forward)
    gene_ids = torch.arange(n_genes, dtype=torch.long)

    # collate for ragged windows
    collate_fn = make_collate(pad_to_max=True)

    return dataset, model, optimizer, gene_ids, collate_fn


def split_dataset(ds, val_frac=0.15, seed=42):
    idx = np.arange(len(ds))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    return Subset(ds, idx[n_val:]), Subset(ds, idx[:n_val])

def prepare_loader(ds, batch_size, collate_fn, shuffle, workers=8):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
        num_workers=workers, persistent_workers=True, prefetch_factor=2,
        collate_fn=collate_fn,
    )
    
def main(device, total_epochs, save_every):
    dataset, model, optimizer, gene_ids, collate_fn = load_train_objs()

    train_ds, val_ds = split_dataset(dataset, validation_fraction)
    train_loader = prepare_loader(train_ds, batch_size, collate_fn, shuffle=True)
    val_loader   = prepare_loader(val_ds,  batch_size, collate_fn, shuffle=False)
    
    trainer = Trainer(model, train_loader, optimizer, device, save_every, gene_ids.to(device))
    trainer.masked_baseline(val_loader)
    trainer.fit_target_norm(train_loader) 

    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    for epoch in range(1, epochs+1):
        trainer._run_epoch(epoch)
        val_mae, val_rmse, val_corr = trainer.evaluate(val_loader)
        print(f"[val] epoch {epoch} | mae {val_mae:.4f} | rmse {val_rmse:.4f} | r {val_corr:.3f}")
        scheduler.step(val_rmse)
        if epoch % save_every == 0:
            trainer._save_checkpoint(epoch)
    
    
if __name__ == "__main__":
    device = 0 if torch.cuda.is_available() else "cpu"
    total_epochs = epochs
    save_every = 10
    main(device, total_epochs, save_every)
    