import os
import sys
import csv
import torch
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformer_2 import MultiomicTransformer
from transformer_2_dataset import MultiomicTransformerDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group`")

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
DATA_DIR = os.path.join(PROJECT_DIR, "dev/testing_scripts/transformer_data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output/transformer_testing_output")

TOTAL_EPOCHS=20
BATCH_SIZE=32

D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 6
D_FF = 1024
DROPOUT = 0.1

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", init_method="env://",
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
        criterion: torch.nn.Module,
        gpu_id: int,
        save_every: int,
        patience: int = 10,
        min_delta: float = 1e-3,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        
        # Early stopping
        self.best_val_loss = float("inf")
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0

    def _run_batch(self, batch, targets):
        atac_wins, tf_tensor = batch
        self.optimizer.zero_grad()
        
        # Calculate predictions
        preds = self.model(atac_wins, tf_tensor)
                
        # Calculate loss
        loss = self.criterion(preds, targets)
        
        # Add a correlation-based loss to maximize Pearson correlation
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        if torch.std(targets_flat) > 1e-8 and torch.std(preds_flat) > 1e-8:
            vx = preds_flat - torch.mean(preds_flat)
            vy = targets_flat - torch.mean(targets_flat)
            corr_loss = -torch.sum(vx * vy) / (
                torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8
            )
        else:
            corr_loss = torch.tensor(0.0, device=preds.device)
            
        loss = loss + 0.1 * corr_loss
        
        # L1 penalty for sparsity on the TF->TG weights at the end of the model
        if hasattr(self.model, "tf_tg_weights"):
            l1_penalty = 1e-4 * torch.norm(self.model.tf_tg_weights, p=1)
            loss = loss + l1_penalty
        
        # Safety check to prevent GPUs from hanging on infinite loss values
        if not torch.isfinite(loss):
            # replace with safe scalar, skip backprop
            logging.warning(f"Rank {self.gpu_id}: skipping batch due to non-finite loss")
            loss = torch.tensor(0.0, device=self.gpu_id, requires_grad=True)
        
        # Backprop
        loss.backward()
            
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        return loss
        
    def _validate(self):
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        preds_list, tgts_list = [], []

        with torch.no_grad():
            for atac_wins, tf_tensor, targets in self.val_data:
                atac_wins, tf_tensor, targets = (
                    atac_wins.to(self.gpu_id),
                    tf_tensor.to(self.gpu_id),
                    targets.to(self.gpu_id),
                )
                output = self.model(atac_wins, tf_tensor)
                loss = F.mse_loss(output, targets)
                total_loss += loss.item()
                n_batches += 1
                preds_list.append(output)
                tgts_list.append(targets)

        # Stack local tensors
        preds = torch.cat(preds_list, dim=0)
        tgts = torch.cat(tgts_list, dim=0)

        # All-gather across ranks
        world_size = dist.get_world_size()
        gathered_preds = [torch.zeros_like(preds) for _ in range(world_size)]
        gathered_tgts  = [torch.zeros_like(tgts) for _ in range(world_size)]
        dist.all_gather(gathered_preds, preds)
        dist.all_gather(gathered_tgts, tgts)

        # Concatenate global arrays
        preds = torch.cat(gathered_preds, dim=0).cpu().numpy()
        tgts  = torch.cat(gathered_tgts, dim=0).cpu().numpy()

        # Replace NaN/inf with safe values
        preds = np.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
        tgts  = np.nan_to_num(tgts, nan=0.0, posinf=1e6, neginf=-1e6)

        # Correlations
        if np.std(tgts) < 1e-8 or np.std(preds) < 1e-8:
            pearson_corr, spearman_corr = 0.0, 0.0
        else:
            try:
                pearson_corr, _ = pearsonr(preds.ravel(), tgts.ravel())
                spearman_corr, _ = spearmanr(preds.ravel(), tgts.ravel())
            except Exception as e:
                logging.warning(f"Correlation failed: {e}")
                pearson_corr, spearman_corr = 0.0, 0.0

        avg_loss = total_loss / max(1, n_batches)
        return avg_loss, pearson_corr, spearman_corr
    
    def _run_epoch(self, epoch):
        b_sz = self.train_data.batch_size
        self.train_data.sampler.set_epoch(epoch)
        total_loss, n_batches = 0.0, 0

        for atac_wins, tf_tensor, targets in self.train_data:
            atac_wins, tf_tensor, targets = atac_wins.to(self.gpu_id), tf_tensor.to(self.gpu_id), targets.to(self.gpu_id)
            loss_val = self._run_batch((atac_wins, tf_tensor), targets)
            if loss_val is None:
                continue
            total_loss += loss_val
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)
        avg_val_loss, pearson_corr, spearman_corr = self._validate()

        return avg_train_loss, avg_val_loss, pearson_corr, spearman_corr

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        if self.gpu_id == 0:
            logging.info(f"\tTraining checkpoint saved at {PATH}")

                        
        

    def train(self, max_epochs: int, log_path: str = "training_log.csv"):
        best_val_loss = float("inf")
        patience_counter = 0
        history = []  # store per-epoch logs

        for epoch in range(max_epochs):
            train_loss, val_loss, pearson_corr, spearman_corr = self._run_epoch(epoch)

            # Save stats to history
            if self.gpu_id == 0:
                logging.info(
                    f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Pearson: {pearson_corr:.3f} | Spearman: {spearman_corr:.3f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
                
                lr = self.optimizer.param_groups[0]['lr']
                history.append({
                    "Epoch": epoch,
                    "Train Loss": train_loss.detach().cpu().item() if torch.is_tensor(train_loss) else float(train_loss),
                    "Val Loss": val_loss.detach().cpu().item() if torch.is_tensor(val_loss) else float(val_loss),
                    "Pearson": float(pearson_corr),
                    "Spearman": float(spearman_corr),
                    "LR": float(lr),
                })

            # Checkpoint + CSV log
            if epoch % self.save_every == 0:
                if self.gpu_id == 0:
                    self._save_checkpoint(epoch)
                    self._write_log_csv(history, log_path, epoch)
                dist.barrier()
                
            stop_tensor = torch.tensor(0, device=self.gpu_id)

            # Early stopping check
            if val_loss < best_val_loss - self.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
                if patience_counter >= self.patience and self.gpu_id == 0:
                    logging.info("Early stopping triggered.")
                    self._save_checkpoint(epoch)
                    self._write_log_csv(history, log_path, epoch, final=True)
                    stop_tensor.fill_(1)  # rank 0 sets the stop flag

            # broadcast stop flag from rank 0 → all ranks
            dist.broadcast(stop_tensor, src=0)

            if stop_tensor.item() == 1:
                break
            else:
                if self.gpu_id == 0 and patience_counter > 0:
                    logging.info(f"    Loss did not improve {patience_counter}/{self.patience}")

        # Final save if not early stopped
        if self.gpu_id == 0 and patience_counter < self.patience:
            self._write_log_csv(history, log_path, epoch, final=True)
            logging.info("Training loop exited normally.")
    
    def _write_log_csv(self, history, log_path, epoch, final=False):
        fieldnames = ["Epoch", "Train Loss", "Val Loss", "Pearson", "Spearman", "LR"]
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        tag = "final" if final else "intermediate"
        logging.info(f"    {tag.capitalize()} training log written at epoch {epoch}")
            

def load_train_objs():
    dataset = MultiomicTransformerDataset(
        data_dir=DATA_DIR,
        chrom_id="chr19"
    ) 
    model = MultiomicTransformer(
        D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT, 
        dataset.num_tf, dataset.num_tg
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

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=DistributedSampler(train_set, num_replicas=world_size, rank=rank, drop_last=True))
    val_loader   = DataLoader(val_set, batch_size=batch_size, sampler=DistributedSampler(val_set, num_replicas=world_size, rank=rank, drop_last=True))
    test_loader  = DataLoader(test_set, batch_size=batch_size, sampler=DistributedSampler(test_set, num_replicas=world_size, rank=rank, drop_last=True))

    return train_loader, val_loader, test_loader

def plot_per_gene_correlation_scatterplot(model, dataloader, scaler, gpu_id=0, outpath=None):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for atac_wins, tf_tensor, targets in dataloader:
            atac_wins, tf_tensor = atac_wins.to(gpu_id), tf_tensor.to(gpu_id)
            output = model(atac_wins, tf_tensor)
            preds.append(output.cpu().numpy())
            tgts.append(targets.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    tgts  = np.concatenate(tgts, axis=0)

    # inverse-transform
    preds_rescaled = scaler.inverse_transform(preds)
    tgts_rescaled  = scaler.inverse_transform(tgts)

    corr, _ = pearsonr(preds_rescaled.ravel(), tgts_rescaled.ravel())
    logging.info(f"Test Pearson correlation: {corr:.3f}")

    plt.figure(figsize=(6,6))
    plt.scatter(tgts_rescaled, preds_rescaled, alpha=0.5, s=5)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(f"Predicted vs True (r={corr:.3f})")
    plt.plot([tgts_rescaled.min(), tgts_rescaled.max()],
             [tgts_rescaled.min(), tgts_rescaled.max()], 'r--')
    if outpath:
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()
        
def per_gene_correlation(model, dataloader, scaler, gpu_id=0, gene_names=None):
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
    
    # inverse-transform
    preds_rescaled = scaler.inverse_transform(preds)
    tgts_rescaled  = scaler.inverse_transform(tgts)

    results = []
    for i in range(tgts_rescaled.shape[1]):  # loop over genes
        if np.std(tgts_rescaled[:, i]) < 1e-8:   # avoid constant targets
            pear, spear = np.nan, np.nan
        else:
            pear, _ = pearsonr(preds_rescaled[:, i], tgts_rescaled[:, i])
            spear, _ = spearmanr(preds_rescaled[:, i], tgts_rescaled[:, i])
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

def write_run_parameters(dataset, out_dir):
    run_params = {
        "Genes": dataset.num_tg,
        "Windows": dataset.num_windows,
        "TFs": dataset.num_tf,
        "Metacells": len(dataset.metacell_names),  # store count, not huge list
        "Epochs": TOTAL_EPOCHS,
        "Batch Size": BATCH_SIZE,
        "d_model": D_MODEL,
        "Attention Heads": NUM_HEADS,
        "Model Layers": NUM_LAYERS,
        "d_feedforward": D_FF,
        "Dropout": DROPOUT,
    }

    path = os.path.join(out_dir, "run_parameters.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for key, value in run_params.items():
            writer.writerow([key, value])
    logging.info(f"Run parameters written to {path}")

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    setup_logging(rank)
    
    time_now = datetime.now().strftime("%d_%m_%H_%M_%S")
    training_output_dir = os.path.join(OUTPUT_DIR, f"model_training_{time_now}")
    
    os.makedirs(training_output_dir, exist_ok=True)
    
    if rank == 0:
        logging.info("Loading Training Objectives")
    dataset, model, optimizer = load_train_objs()

    
    if rank == 0:
        logging.info("===== MultiomicTransformerDataset Loaded =====")
        logging.info(f"Genes:            {dataset.num_tg}")
        logging.info(f"Windows (RE):     {dataset.num_windows}")
        logging.info(f"TFs:              {dataset.num_tf}")
        logging.info(f"Metacells:        {len(dataset.metacell_names)}")
        logging.info("================================================")
        write_run_parameters(dataset, training_output_dir)
    
    if rank == 0:
        logging.info("Preparing DataLoader")
    train_loader, val_loader, test_loader = prepare_dataloader(dataset, batch_size, world_size, rank)
    if rank == 0:
        logging.info("Initializing Trainer")
    criterion = nn.MSELoss()
    trainer = Trainer(
        model=model, 
        train_data=train_loader, 
        val_data=val_loader, 
        optimizer=optimizer, 
        criterion=criterion, 
        gpu_id=rank, 
        save_every=save_every
        )
    if rank == 0:
        logging.info("\n ----- Training -----")
    trainer.train(
        max_epochs=total_epochs, 
        log_path=os.path.join(training_output_dir, "training_log.csv")
        )
    
    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            logging.info("Training Complete! Synchronizing ranks")
    
    if rank == 0:
        tf_tg_weights = trainer.model.module.tf_tg_weights.detach().cpu().numpy()
        tf_names = dataset.tf_names  # assuming you have these
        tg_names = dataset.TG_pseudobulk.index.tolist()

        tf_tg_df = pd.DataFrame(tf_tg_weights, index=tf_names, columns=tg_names)
        tf_tg_df.to_csv(os.path.join(training_output_dir, "tf_tg_weights.csv"))
        logging.info("Saved TF→TG weights to CSV")

    
    if rank == 0:
        scaler = joblib.load(os.path.join(DATA_DIR, "tg_scaler.pkl"))
        
        out_prefix = os.path.join(training_output_dir, "per_gene_correlation")
        
        plot_per_gene_correlation_scatterplot(
            trainer.model.module, 
            test_loader, 
            scaler,
            gpu_id=0,
            outpath=os.path.join(training_output_dir, "per_gene_correlation_scatterplot.png")
        )
        
        logging.info("Evaluating per-gene correlations on test set...")
        corr_df = per_gene_correlation(
            model, test_loader, scaler, gpu_id=rank,
            gene_names=dataset.TG_pseudobulk.index.tolist()
        )
        corr_df.to_csv(out_prefix + ".csv", index=False)
        logging.info(f"Saved per-gene correlations to {out_prefix}.csv")

        # Quick summary
        logging.info(f"Median Pearson: {np.nanmedian(corr_df['pearson']):.3f}, "
                    f"Top 5 genes: {corr_df.sort_values('pearson', ascending=False).head()['gene'].tolist()}")

        # Plots
        plot_gene_correlation_distribution(corr_df, out_prefix)
        logging.info(f"Saved per-gene correlation plots to {out_prefix}_*.png")
    

    if dist.is_initialized():
        dist.barrier()
        if rank == 0:
            logging.info("Destroying process group")
        dist.destroy_process_group()
    
if __name__ == "__main__":
    main(rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        save_every=5,
        total_epochs=TOTAL_EPOCHS,
        batch_size=BATCH_SIZE)