import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformer_2 import MultiomicTransformer
from transformer_2_dataset import MultiomicTransformerDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_rank, is_initialized, is_available

import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
DATA_DIR = os.path.join(PROJECT_DIR, "dev/testing_scripts/transformer_data")

D_MODEL = 128
NUM_HEADS = 8
D_FF = 256
DROPOUT = 0.1

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)
    
def setup_logging(rank: int):
    # Remove existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    level = logging.INFO if rank == 0 else logging.ERROR
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.flush = sys.stdout.flush  # ensure flush

    logging.basicConfig(
        level=level,
        format=f"%(message)s",
        handlers=[handler],
        force=True
    )
    
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        patience: int = 5,
        min_delta: float = 1e-4,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        
        # Early stopping
        self.best_val_loss = float("inf")
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0

    def _run_batch(self, batch, targets):
        atac_wins, tf_tensor = batch
        self.optimizer.zero_grad()
        output = self.model(atac_wins, tf_tensor)   # << your model signature
        loss = F.mse_loss(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss
        
    def _validate(self):
        self.model.eval()
        total_val_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for atac_wins, tf_tensor, targets in self.val_data:
                atac_wins, tf_tensor, targets = (
                    atac_wins.to(self.gpu_id),
                    tf_tensor.to(self.gpu_id),
                    targets.to(self.gpu_id),
                )
                output = self.model(atac_wins, tf_tensor)

                loss = F.mse_loss(output, targets)
                total_val_loss += loss.item()

                # store predictions & targets on CPU for correlation
                all_preds.append(output.detach().cpu())
                all_targets.append(targets.detach().cpu())

        self.model.train()
        avg_val_loss = total_val_loss / len(self.val_data)

        # flatten tensors to 1D for correlation
        preds = torch.cat(all_preds).numpy().ravel()
        tgts = torch.cat(all_targets).numpy().ravel()

        pearson_corr, _ = pearsonr(preds, tgts)
        spearman_corr, _ = spearmanr(preds, tgts)

        return avg_val_loss, pearson_corr, spearman_corr

    def _run_epoch(self, epoch):
        b_sz = self.train_data.batch_size
        self.train_data.sampler.set_epoch(epoch)
        total_loss, n_batches = 0.0, 0
        for atac_wins, tf_tensor, targets in self.train_data:
            atac_wins, tf_tensor, targets = atac_wins.to(self.gpu_id), tf_tensor.to(self.gpu_id), targets.to(self.gpu_id)
            loss_val = self._run_batch((atac_wins, tf_tensor), targets)
            total_loss += loss_val
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)
        avg_val_loss, pearson_corr, spearman_corr = self._validate()
        
        # Scheduler step
        self.scheduler.step(avg_val_loss)

        if self.gpu_id == 0:
            logging.info(
                f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Pearson: {pearson_corr:.3f} | Spearman: {spearman_corr:.3f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
        # Early stopping check
        if avg_val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                return False
        return True

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        if self.gpu_id == 0:
            logging.info(f"\tTraining checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            train_loss = self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        if self.gpu_id == 0:
            logging.info("Training loop exited normally.")
            

def load_train_objs():
    dataset = MultiomicTransformerDataset(
        data_dir=DATA_DIR,
        chrom_id="chr19"
    ) 
    model = MultiomicTransformer(
        D_MODEL, NUM_HEADS, D_FF, DROPOUT, 
        dataset.num_tf, dataset.num_windows, dataset.num_tg
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return dataset, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int, world_size: int, rank: int):
    train_frac, val_frac, test_frac = 0.7, 0.15, 0.15
    n_total = len(dataset)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=DistributedSampler(train_set, num_replicas=world_size, rank=rank))
    val_loader   = DataLoader(val_set, batch_size=batch_size, sampler=DistributedSampler(val_set, num_replicas=world_size, rank=rank))
    test_loader  = DataLoader(test_set, batch_size=batch_size, sampler=DistributedSampler(test_set, num_replicas=world_size, rank=rank))

    return train_loader, val_loader, test_loader


def evaluate_and_plot(model, dataloader, gpu_id=0, outpath=None):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for atac_wins, tf_tensor, targets in dataloader:
            atac_wins, tf_tensor = atac_wins.to(gpu_id), tf_tensor.to(gpu_id)
            output = model(atac_wins, tf_tensor)
            preds.append(output.cpu())
            tgts.append(targets.cpu())
    preds = torch.cat(preds).numpy().ravel()
    tgts = torch.cat(tgts).numpy().ravel()

    corr, _ = pearsonr(preds, tgts)
    logging.info(f"Test Pearson correlation: {corr:.3f}")

    plt.figure(figsize=(6,6))
    plt.scatter(tgts, preds, alpha=0.5, s=5)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(f"Predicted vs True (r={corr:.3f})")
    plt.plot([tgts.min(), tgts.max()], [tgts.min(), tgts.max()], 'r--')
    if outpath:
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()
        
def per_gene_correlation(model, dataloader, gpu_id=0, gene_names=None):
    """
    Compute Pearson & Spearman correlation per gene across the test set.

    Args:
        model       : trained PyTorch model
        dataloader  : DataLoader over test set
        gpu_id      : device id
        gene_names  : list of gene names for annotation (optional)

    Returns:
        DataFrame with [gene, pearson, spearman]
    """
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for atac_wins, tf_tensor, targets in dataloader:
            atac_wins, tf_tensor = atac_wins.to(gpu_id), tf_tensor.to(gpu_id)
            output = model(atac_wins, tf_tensor)
            preds.append(output.cpu().numpy())
            tgts.append(targets.cpu().numpy())

    preds = np.concatenate(preds, axis=0)   # [samples, num_genes]
    tgts  = np.concatenate(tgts, axis=0)

    results = []
    for i in range(tgts.shape[1]):  # loop over genes
        if np.std(tgts[:, i]) < 1e-8:   # avoid constant targets
            pear, spear = np.nan, np.nan
        else:
            pear, _ = pearsonr(preds[:, i], tgts[:, i])
            spear, _ = spearmanr(preds[:, i], tgts[:, i])
        results.append((pear, spear))

    df = pd.DataFrame(results, columns=["pearson", "spearman"])
    if gene_names is not None:
        df.insert(0, "gene", gene_names)
    return df

def plot_gene_correlation_distribution(corr_df, out_prefix):
    """
    Plot distributions of per-gene Pearson & Spearman correlations.
    """
    plt.figure(figsize=(6,4))
    sns.histplot(corr_df['pearson'].dropna(), bins=30, kde=True, color="steelblue")
    plt.xlabel("Pearson correlation")
    plt.ylabel("Number of genes")
    plt.title("Per-gene Pearson correlation")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_pearson_hist.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6,4))
    sns.histplot(corr_df['spearman'].dropna(), bins=30, kde=True, color="darkorange")
    plt.xlabel("Spearman correlation")
    plt.ylabel("Number of genes")
    plt.title("Per-gene Spearman correlation")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_spearman_hist.png", dpi=300)
    plt.close()

    # Optional: violin for side-by-side view
    plt.figure(figsize=(5,4))
    sns.violinplot(data=corr_df[['pearson','spearman']], inner="quartile", palette="Set2")
    plt.ylabel("Correlation")
    plt.title("Distribution of per-gene correlations")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_violin.png", dpi=300)
    plt.close()

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    setup_logging(rank)
    
    if rank == 0:
        logging.info("Loading Training Objectives")
    dataset, model, optimizer = load_train_objs()
    if rank == 0:
        logging.info("Preparing DataLoader")
    train_loader, val_loader, test_loader = prepare_dataloader(dataset, batch_size, world_size, rank)
    if rank == 0:
        logging.info("Initializing Trainer")
    trainer = Trainer(model, train_loader, val_loader, optimizer, rank, save_every)
    if rank == 0:
        logging.info("\n ----- Training -----")
    trainer.train(total_epochs)
    
    if rank == 0:
        evaluate_and_plot(
            trainer.model.module, 
            test_loader, 
            gpu_id=0,
            outpath=os.path.join(PROJECT_DIR, "dev/testing_scripts/tmp/transformer_training.png")
        )
        
    if rank == 0:
        logging.info("Evaluating per-gene correlations on test set...")
        corr_df = per_gene_correlation(
            model, test_loader, gpu_id=rank,
            gene_names=dataset.TG_pseudobulk.index.tolist()
        )
        out_prefix = os.path.join(PROJECT_DIR, "dev/testing_scripts/tmp/per_gene_correlations")
        corr_df.to_csv(out_prefix + ".csv", index=False)
        logging.info(f"Saved per-gene correlations to {out_prefix}.csv")

        # Quick summary
        logging.info(f"Median Pearson: {np.nanmedian(corr_df['pearson']):.3f}, "
                    f"Top 5 genes: {corr_df.sort_values('pearson', ascending=False).head()['gene'].tolist()}")

        # Plots
        plot_gene_correlation_distribution(corr_df, out_prefix)
        logging.info(f"Saved per-gene correlation plots to {out_prefix}_*.png")
    
    if rank == 0:
        logging.info("Main finished, destroying process group")

        
    destroy_process_group()
    


if __name__ == "__main__":
    main(rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        save_every=5,
        total_epochs=20,
        batch_size=2)