import os
import sys
import csv
import torch
import joblib
import json
import pandas as pd
import numpy as np
import logging
import pickle
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, skew, kurtosis
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformer import MultiomicTransformer
from transformer_dataset import MultiomicTransformerDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import matplotlib.pyplot as plt
import seaborn as sns

from eval import (
    per_gene_correlation
)

import warnings
warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group`")

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
DATA_DIR = os.path.join(PROJECT_DIR, "dev/transformer/transformer_data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output/transformer_testing_output")
GROUND_TRUTH_FILE = os.path.join(PROJECT_DIR, "ground_truth_files/mESC_beeline_ChIP-seq.csv")

CHROM_ID = "chr19"

TOTAL_EPOCHS=500
BATCH_SIZE=32
TRAINING_ITERATIONS = 30
PERCENT_DROP=0
PATIENCE=15

D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 3
D_FF = 768
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

    def _run_batch(self, batch, targets, bias=None):
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
            
        loss = loss + 0.05 * corr_loss
        
        # # L1 penalty for sparsity on the TF->TG weights at the end of the model
        # if hasattr(self.model, "tf_tg_weights"):
        #     l1_penalty = 1e-4 * torch.norm(self.model.tf_tg_weights, p=1)
        #     loss = loss + l1_penalty
        
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
            for atac_wins, tf_tensor, targets, bias in self.val_data:
                atac_wins, tf_tensor, targets, bias = (
                    atac_wins.to(self.gpu_id),
                    tf_tensor.to(self.gpu_id),
                    targets.to(self.gpu_id),
                    bias.to(self.gpu_id)
                )
                preds = self.model(atac_wins, tf_tensor, bias=bias)
                loss = F.mse_loss(preds, targets)
                total_loss += loss.item()
                n_batches += 1
                preds_list.append(preds)
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

        for atac_wins, tf_tensor, targets, bias in self.train_data:
            atac_wins, tf_tensor, targets, bias = (
                atac_wins.to(self.gpu_id),
                tf_tensor.to(self.gpu_id),
                targets.to(self.gpu_id),
                bias.to(self.gpu_id),
            )
            loss_val = self._run_batch((atac_wins, tf_tensor), targets, bias=bias)
            if loss_val is None:
                continue
            total_loss += loss_val
            n_batches += 1

        avg_train_loss = total_loss / max(1, n_batches)
        avg_val_loss, pearson_corr, spearman_corr = self._validate()
        
        self.scheduler.step(avg_val_loss)

        return avg_train_loss, avg_val_loss, pearson_corr, spearman_corr

    def _save_checkpoint(self, epoch, path):
        ckp = self.model.module.state_dict()
        torch.save(ckp, os.path.join(path, "checkpoint.pt"))
        if self.gpu_id == 0:
            logging.info(f"\tTraining checkpoint saved")
        
    def train(self, max_epochs: int, path: str):
        best_val_loss = float("inf")
        best_pearson = float(0)
        patience_counter = 0
        history = []  # store per-epoch logs

        for epoch in range(max_epochs):
            train_loss, val_loss, pearson_corr, spearman_corr = self._run_epoch(epoch)

            # Save stats to history
            if self.gpu_id == 0:
                logging.info(
                    f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | "
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
                    self._save_checkpoint(epoch, path)
                    self._write_log_csv(history, path, epoch)
                dist.barrier()
                
            stop_tensor = torch.tensor(0, device=self.gpu_id)

            # Early stopping check
            # Track best values
            if val_loss < best_val_loss - self.min_delta or pearson_corr > best_pearson + self.min_delta:
                # If either val_loss improved OR pearson improved, reset patience
                best_val_loss = min(best_val_loss, val_loss)
                best_pearson = max(best_pearson, pearson_corr)
                patience_counter = 0
            else:
                # No improvement in *both* metrics
                patience_counter += 1
                
                if patience_counter >= self.patience and self.gpu_id == 0:
                    logging.info("Early stopping triggered.")
                    self._save_checkpoint(epoch, path)
                    self._write_log_csv(history, path, epoch, final=True)
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
            self._write_log_csv(history, path, epoch, final=True)
            logging.info("Training loop exited normally.")
    
    def _write_log_csv(self, history, path, epoch, final=False):
        fieldnames = ["Epoch", "Train Loss", "Val Loss", "Pearson", "Spearman", "LR"]
        log_path = os.path.join(path, "training_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        tag = "final" if final else "intermediate"
        # logging.info(f"    {tag.capitalize()} training log written at epoch {epoch}")
            

def load_train_objs(subset_genes=None):
    dataset = MultiomicTransformerDataset(
        data_dir=DATA_DIR,
        chrom_id=CHROM_ID
    ) 
    if subset_genes is not None:
        dataset.filter_genes(subset_genes)
    
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


def write_run_parameters(dataset, out_dir, has_dist_bias):
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
        "Distance Bias":has_dist_bias
    }

    path = os.path.join(out_dir, "run_parameters.json")
    with open(path, "w") as f:
        json.dump(run_params, f, indent=4)  # indent=4 for readability
    logging.info(f"Run parameters written to {path}")
    
def nclb_refinement(model, dataset, training_output_dir, min_corr, iter_id):
    _, _, test_loader = prepare_dataloader(dataset, BATCH_SIZE, world_size=1, rank=0)
    corr_df = per_gene_correlation(
        model, test_loader, dataset.scaler, gpu_id=0, gene_names=dataset.tg_names
    )
    
    avg_corr = corr_df["pearson"].mean()
    
    if avg_corr < min_corr:
        # Remove bottom 10% of genes by Pearson correlation
        cutoff = np.percentile(corr_df["pearson"], PERCENT_DROP)
        predicted_subset = corr_df.loc[corr_df["pearson"] > cutoff, "gene"]
    else:
        predicted_subset = corr_df["gene"]

    subset_path = os.path.join(training_output_dir, f"learnable_genes_iter{iter_id}.csv")
    predicted_subset.to_csv(subset_path, index=False)

    return avg_corr, subset_path

def compute_tf_tg_distance_features(tf_tg_weights, tf_names, tg_names, dist_df, window_map, atac_window_tensor):
    """
    For each TF–TG pair, compute a distance-weighted accessibility score.
    
    Calculates the average peak accessibility for each TF-peak-TG edge
    to get a TF-TG distance score.
    """

    # Map peak -> dist_score -> gene
    tg_to_scores = {tg: {} for tg in tg_names}
    for _, row in dist_df.iterrows():
        peak_id = row["peak_id"]
        tg = row["target_id"]
        dist_score = row["TSS_dist_score"]
        if tg in tg_to_scores:
            tg_to_scores[tg][peak_id] = dist_score

    features = []
    for i, tf in enumerate(tf_names):
        for j, tg in enumerate(tg_names):
            weight = tf_tg_weights[i, j]

            # collect peaks mapped to this TG
            peak_scores = tg_to_scores.get(tg, {})
            if not peak_scores:
                dist_weighted_accessibility = 0.0
            else:
                # Get accessibility of peaks' windows
                scores = []
                for peak_id, dist_score in peak_scores.items():
                    if peak_id in window_map:
                        win_idx = window_map[peak_id]
                        acc = atac_window_tensor[win_idx].mean().item()  # avg across metacells
                        scores.append(acc * dist_score)
                dist_weighted_accessibility = sum(scores) / (len(scores) + 1e-8)

            features.append([weight, dist_weighted_accessibility])

    return np.array(features)


def build_edge_features(tf_tg_weights, tf_names, tg_names, chip_edges, dist_df, window_map, atac_window_tensor):
    features = []
    labels = []
    edge_list = []

    # Precompute gene -> max distance score from peaks
    # dist_df = dist_df.copy()
    # dist_df["target_id"] = dist_df["target_id"].str.upper()
    dist_features = compute_tf_tg_distance_features(tf_tg_weights, tf_names, tg_names, dist_df, window_map, atac_window_tensor)

    idx = 0
    for i, tf in enumerate(tf_names):
        for j, tg in enumerate(tg_names):
            edge = (tf.upper(), tg.upper())
            features.append(dist_features[idx])  # [weight, dist-weighted accessibility]
            labels.append(1 if edge in chip_edges else 0)
            edge_list.append(edge)
            idx += 1

    features = np.array(features)
    labels = np.array(labels)
    return features, labels, edge_list

def train_tf_tg_classifier(tf_tg_weights, dataset, ground_truth_file, out_path):
    chip_df = pd.read_csv(ground_truth_file)
    chip_edges = set((g1.upper(), g2.upper()) for g1, g2 in zip(chip_df["Gene1"], chip_df["Gene2"]))
    
    dist_df = pd.read_parquet(os.path.join(DATA_DIR, f"genes_near_peaks_{CHROM_ID}.parquet"))
    
    X, y, edge_list = build_edge_features(
        tf_tg_weights, dataset.tf_names, dataset.tg_names, chip_edges, dist_df, dataset.window_map, dataset.atac_window_tensor_all
    )
    
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=stratify, random_state=42
    )
    
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from scipy.stats import randint

    # Define distributions instead of full grids
    param_dist = {
        "n_estimators": randint(100, 1000),       # sample between 100 and 999 trees
        "max_depth": [None] + list(range(10, 60, 10)),
        "max_features": ["sqrt", "log2", None],
        "min_samples_split": randint(2, 20)       # sample split thresholds
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced", random_state=42),
        param_distributions=param_dist,
        n_iter=50,                # number of random combos to try
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    print("Best params:", random_search.best_params_)
    print("Best AUROC:", random_search.best_score_)

    y_score = random_search.predict_proba(X_test)[:, 1]
    y_pred = random_search.predict(X_test)

    print("\tTF-TG classification report:")
    print(classification_report(y_test, y_pred))
    logging.info(f"TF-TG AUROC: {roc_auc_score(y_test, y_score):.4f}")

    # --- Save all edges with predicted scores ---
    all_probs = random_search.predict_proba(X)[:, 1]
    edge_df = pd.DataFrame({
        "TF": [e[0] for e in edge_list],
        "TG": [e[1] for e in edge_list],
        "probability": all_probs,
        "label": y
    })
    edge_df.to_csv(out_path, index=False)
    logging.info(f"Saved edge predictions to {out_path}")

    return random_search, edge_df

def dist_broadcast_list(obj, src=0):
    """Broadcast a Python list or object from src rank to all ranks."""
    if dist.get_rank() == src:
        data = pickle.dumps(obj)
        tensor = torch.ByteTensor(list(data)).to("cuda")
        size = torch.tensor([tensor.size(0)], device="cuda")
    else:
        tensor = torch.ByteTensor().to("cuda")
        size = torch.tensor([0], device="cuda")

    # Broadcast size first
    dist.broadcast(size, src=src)

    # Resize and broadcast actual data
    if dist.get_rank() != src:
        tensor = torch.empty(size.item(), dtype=torch.uint8, device="cuda")
    dist.broadcast(tensor, src=src)

    data = bytes(tensor.tolist())
    return pickle.loads(data)

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    setup_logging(rank)
    
    try:
        time_now = datetime.now().strftime("%d_%m_%H_%M_%S")
        training_output_dir = os.path.join(OUTPUT_DIR, f"model_training_{time_now}")
        os.makedirs(training_output_dir, exist_ok=True)
        
        i = 1
        stop_flag = False
        subset_genes = None

        MIN_CORR = 0.5 # Filter low confidence genes
        TRANSFORMER_CORR_THRESH = 0.75
        
        while i <= TRAINING_ITERATIONS and not stop_flag:
            # if rank == 0:
            #     logging.info("Loading Training Objectives")
            iter_dir = os.path.join(training_output_dir, f"iter{i}")
            os.makedirs(iter_dir, exist_ok=True)
            
            if rank == 0:
                logging.info(f"\n ----- Training Iteration {i} -----")
                
            dataset, model, optimizer = load_train_objs(subset_genes=subset_genes)
            
            has_dist_bias = "No"
            if dataset.dist_bias_tensor is not None:
                has_dist_bias = "Yes"
            
            if rank == 0:
                logging.info("\n===== MultiomicTransformerDataset Loaded =====")
                logging.info(f"Genes:               {dataset.num_tg}")
                logging.info(f"Windows (RE):        {dataset.num_windows}")
                logging.info(f"TFs:                 {dataset.num_tf}")
                logging.info(f"Metacells:           {len(dataset.metacell_names)}")
                logging.info(f"Epochs:              {TOTAL_EPOCHS}")
                logging.info(f"Iterations:          {TRAINING_ITERATIONS}")
                logging.info(f"Batch Size:          {BATCH_SIZE}")
                logging.info(f"Model Dimension:     {D_MODEL}")
                logging.info(f"Attention Heads:     {NUM_HEADS}")
                logging.info(f"Attention Layers:    {NUM_LAYERS}")
                logging.info(f"Feedforward Layers:  {D_FF}")
                logging.info(f"Dropout:             {DROPOUT}")
                logging.info(f"Dist bias?:          {has_dist_bias}")
                logging.info("================================================")
                write_run_parameters(dataset, iter_dir, has_dist_bias)
            
            train_loader, val_loader, test_loader = prepare_dataloader(dataset, batch_size, world_size, rank)
            criterion = nn.MSELoss()
            trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, gpu_id=rank, save_every=save_every)
        

            trainer.train(max_epochs=TOTAL_EPOCHS, path=iter_dir)

            if rank == 0:
                tf_tg_weights = trainer.model.module.tf_tg_weights.detach().cpu().numpy()
                out_file = os.path.join(training_output_dir, "tf_tg_classifier_predictions.csv")
                clf, edge_df = train_tf_tg_classifier(tf_tg_weights, dataset, GROUND_TRUTH_FILE, out_file)
                
                avg_corr, subset_path = nclb_refinement(
                    model, dataset, iter_dir, MIN_CORR, i
                )
                logging.info(f"Iteration {i} | Avg Pearson = {avg_corr:.3f}")
                
                if avg_corr >= TRANSFORMER_CORR_THRESH:
                    logging.info(f"Stopping refinement: Pearson correlation threshold of {avg_corr} reached.")
                    stop_flag = True
                else:
                    stop_flag = False
                    subset_genes = pd.read_csv(subset_path, header=None)[0].tolist()
            else:
                stop_flag = False
                subset_genes = None

            # Broadcast stop flag
            stop_tensor = torch.tensor(int(stop_flag), device="cuda")
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                break

            # Broadcast subset genes
            subset_genes = dist_broadcast_list(subset_genes, src=0)
        
            if dist.is_initialized():
                dist.barrier()
                if rank == 0:
                    logging.info(f"Training Complete for iteration {i}! Synchronizing ranks")
            
            if rank == 0:
                # Save model checkpoint
                torch.save(trainer.model.module.state_dict(),
                        os.path.join(iter_dir, "trained_model.pt"))
                # logging.info("Saved final trained model")
                
                # Save supporting objects
                joblib.dump(dataset.scaler, os.path.join(iter_dir, "tg_scaler.pkl"))
                pd.DataFrame(trainer.model.module.tf_tg_weights.detach().cpu().numpy(),
                            index=dataset.tf_names,
                            columns=dataset.tg_names
                ).to_csv(os.path.join(iter_dir, "tf_tg_weights.csv"))
                
                tf_tg_weights = trainer.model.module.tf_tg_weights.detach().cpu().numpy()
                tg_names = dataset.tg_names

                tf_tg_df = pd.DataFrame(tf_tg_weights, index=dataset.tf_names, columns=tg_names)
                tf_tg_df.to_csv(os.path.join(iter_dir, "tf_tg_weights.csv"))
                # logging.info("Saved TF→TG weights to CSV")
                
            i += 1
                
                
        if rank == 0:
            logging.info("\nIterations complete")
    
    finally:
        if dist.is_initialized():
            dist.barrier()
            if rank == 0:
                logging.info("\nDestroying process group")
            dist.destroy_process_group()
    
if __name__ == "__main__":
    main(rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        save_every=5,
        total_epochs=TOTAL_EPOCHS,
        batch_size=BATCH_SIZE)