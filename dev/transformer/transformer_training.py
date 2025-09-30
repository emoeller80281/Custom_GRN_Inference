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
import math
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
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

CHROM_ID = "chr1"
SAMPLE_NAME = "mESC"

TOTAL_EPOCHS=500
BATCH_SIZE=32
TRAINING_ITERATIONS = 30
PERCENT_DROP=0
PATIENCE=20

D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 3
D_FF = 768
DROPOUT = 0.1

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
DATA_DIR = os.path.join(PROJECT_DIR, f"dev/transformer/transformer_data/{SAMPLE_NAME}")
OUTPUT_DIR = os.path.join(PROJECT_DIR, f"output/transformer_testing_output/{SAMPLE_NAME}/{CHROM_ID}")
GROUND_TRUTH_FILE = os.path.join(PROJECT_DIR, "ground_truth_files/mESC_beeline_ChIP-seq.csv")

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
            self.optimizer, mode="min", factor=0.1, patience=7
        )
        
        # Early stopping
        self.best_val_loss = float("inf")
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0

    def _run_batch(self, batch):
        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids = batch
        atac_wins = atac_wins.to(self.gpu_id)
        tf_tensor = tf_tensor.to(self.gpu_id)
        targets   = targets.to(self.gpu_id)
        bias      = bias.to(self.gpu_id)
        tf_ids    = tf_ids.to(self.gpu_id)
        tg_ids    = tg_ids.to(self.gpu_id)

        self.optimizer.zero_grad()
        preds = self.model(atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias)

        loss = self.criterion(preds, targets)

        # correlation bonus (unchanged)
        preds_flat, targets_flat = preds.reshape(-1), targets.reshape(-1)
        if torch.std(targets_flat) > 1e-8 and torch.std(preds_flat) > 1e-8:
            vx = preds_flat - torch.mean(preds_flat)
            vy = targets_flat - torch.mean(targets_flat)
            corr_loss = -torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2))*torch.sqrt(torch.sum(vy**2)) + 1e-8)
        else:
            corr_loss = torch.tensor(0.0, device=preds.device)
        loss = loss + 0.05 * corr_loss

        if not torch.isfinite(loss):
            logging.warning(f"Rank {self.gpu_id}: skipping batch due to non-finite loss")
            return None

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss

    def _validate(self):
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        preds_list, tgts_list = [], []
        with torch.no_grad():
            for atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids in self.val_data:
                atac_wins = atac_wins.to(self.gpu_id)
                tf_tensor = tf_tensor.to(self.gpu_id)
                targets   = targets.to(self.gpu_id)
                bias      = bias.to(self.gpu_id)
                tf_ids    = tf_ids.to(self.gpu_id)
                tg_ids    = tg_ids.to(self.gpu_id)

                preds = self.model(atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias)
                loss  = F.mse_loss(preds, targets)
                total_loss += loss.item(); n_batches += 1
                preds_list.append(preds); tgts_list.append(targets)


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
        sampler = getattr(self.train_data, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        total_loss, n_batches = 0.0, 0
        for atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids in self.train_data:
            loss_val = self._run_batch((atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids))
            if loss_val is None: continue
            total_loss += loss_val; n_batches += 1

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

            # Checkpoint + CSV log
            stop_tensor = torch.tensor(0, device=self.gpu_id)

            # --- Early stopping check (only rank 0 sets flag) ---
            if self.gpu_id == 0:
                if val_loss < best_val_loss - self.min_delta or pearson_corr > best_pearson + self.min_delta:
                    # If either val_loss improved OR pearson improved, reset patience
                    best_val_loss = min(best_val_loss, val_loss)
                    best_pearson = max(best_pearson, pearson_corr)
                    patience_counter = 0
                else:
                    # No improvement
                    patience_counter += 1

                    if patience_counter >= self.patience:
                        logging.info("Early stopping triggered.")
                        self._save_checkpoint(epoch, path)
                        self._write_log_csv(history, path, epoch, final=True)
                        stop_tensor.fill_(1)  # <-- mark stop

                    else:
                        logging.info(f"    Loss did not improve {patience_counter}/{self.patience}")

            # --- Broadcast stop flag from rank 0 to all ranks ---
            dist.broadcast(stop_tensor, src=0)

            # --- All ranks see the same value now ---
            if stop_tensor.item() == 1:
                if self.gpu_id == 0:
                    logging.info("All ranks stopping training.")
                break


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
            

def load_train_objs(DATA_DIR, CHROM_ID,
                    D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT,
                    subset_genes=None, lr=1e-3):

    COMMON_DIR = os.path.join(
        os.path.dirname(DATA_DIR),  # .../transformer_data/
        "common"
    )

    dataset = MultiomicTransformerDataset(
        data_dir=DATA_DIR,
        chrom_id=CHROM_ID,
        tf_vocab_path=os.path.join(COMMON_DIR, "tf_vocab.json"),
        tg_vocab_path=os.path.join(COMMON_DIR, "tg_vocab.json"),
    )
    if subset_genes is not None:
        dataset.filter_genes(subset_genes)

    # global vocab sizes (from common vocab, loaded into dataset.*_name2id)
    tf_vocab_size = len(dataset.tf_name2id)
    tg_vocab_size = len(dataset.tg_name2id)

    model = MultiomicTransformer(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        tf_vocab_size=tf_vocab_size,
        tg_vocab_size=tg_vocab_size,
        use_shortcut=False
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return dataset, model, optimizer


def prepare_dataloader(dataset, batch_size, world_size=1, rank=0,
                       num_workers=4, pin_memory=True, seed=42, drop_last=True):
    """
    Build train/val/test loaders with the dataset's collate_fn.
    Uses DistributedSampler only when world_size > 1.
    """
    # --- deterministic split
    g = torch.Generator()
    g.manual_seed(seed)

    n_total = len(dataset)
    n_train = int(n_total * 0.70)
    n_val   = int(n_total * 0.15)
    n_test  = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=g)

    # --- samplers (DDP only)
    if world_size > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, drop_last=drop_last)
        val_sampler   = DistributedSampler(val_set,   num_replicas=world_size, rank=rank, drop_last=False)
        test_sampler  = DistributedSampler(test_set,  num_replicas=world_size, rank=rank, drop_last=False)
        shuffle = False
    else:
        train_sampler = val_sampler = test_sampler = None
        shuffle = True  # only for single-GPU/CPU

    # --- loaders (always use the dataset's collate_fn)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle,
        sampler=train_sampler,
        collate_fn=MultiomicTransformerDataset.collate_fn,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        sampler=val_sampler,
        collate_fn=MultiomicTransformerDataset.collate_fn,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        sampler=test_sampler,
        collate_fn=MultiomicTransformerDataset.collate_fn,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )

    return train_loader, val_loader, test_loader


def write_run_parameters(dataset, out_dir, has_dist_bias):
    run_params = {
        "Genes": dataset.tg_tensor_all.shape[0],
        "Windows": dataset.num_windows,
        "TFs": dataset.tf_tensor_all.shape[0],
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

# helpers_tf_tg.py
import math, torch

@torch.no_grad()
def get_tf_tg_affinity(model, tf_ids: torch.LongTensor, tg_ids: torch.LongTensor, device="cuda"):
    """Static TF↔TG similarity used by the shortcut: [T_eval, G_eval]."""
    model.eval()
    tf_ids, tg_ids = tf_ids.to(device), tg_ids.to(device)
    tf_base = model.tf_emb_table(tf_ids)          # [T_eval, D]
    tg_dec  = model.tg_decoder_table(tg_ids)      # [G_eval, D]
    sim = (tg_dec @ tf_base.T) / math.sqrt(model.d_model)   # [G_eval, T_eval]
    return sim.T.contiguous()                                   # [T_eval, G_eval]

@torch.no_grad()
def get_mean_attention(model, dataloader, device="cuda", weight_by_tf_expr=False):
    """
    Average TG→TF attention across the dataset. Optionally weight by TF expression per batch.
    Returns:
      mean_attn: [T_eval, G_eval]
      tf_ids, tg_ids: LongTensors used (assumed constant across the dataset)
    """
    model.eval()
    acc = None
    n   = 0
    tf_ids_ref = tg_ids_ref = None

    for atac_wins, tf_tensor, _, bias, tf_ids, tg_ids in dataloader:
        atac_wins, tf_tensor, bias = atac_wins.to(device), tf_tensor.to(device), bias.to(device)
        tf_ids, tg_ids = tf_ids.to(device), tg_ids.to(device)

        tf_base = model.tf_emb_table(tf_ids)           # [T_eval,D]
        tg_dec  = model.tg_decoder_table(tg_ids)       # [G_eval,D]
        sim = (tg_dec @ tf_base.T) / math.sqrt(model.d_model)   # [G_eval,T_eval]
        attn = torch.softmax(sim, dim=-1).T                        # [T_eval,G_eval]

        if weight_by_tf_expr:
            # normalize per-batch TF expr to [0,1] and average over cells
            tf_min = tf_tensor.min(dim=1, keepdim=True).values
            tf_max = tf_tensor.max(dim=1, keepdim=True).values
            tf_norm = (tf_tensor - tf_min) / (tf_max - tf_min + 1e-6)    # [B,T_eval]
            # weight attention by mean TF activity across batch
            w = tf_norm.mean(dim=0, keepdim=True).T                      # [T_eval,1]
            attn = attn * w                                              # [T_eval,G_eval]

        acc = attn if acc is None else acc + attn
        n += 1
        tf_ids_ref, tg_ids_ref = tf_ids.detach().cpu(), tg_ids.detach().cpu()

    mean_attn = (acc / max(n,1)).detach().cpu()
    return mean_attn, tf_ids_ref, tg_ids_ref



from collections import defaultdict

def compute_tf_tg_distance_features(sim_mat, attn_mat, tf_names, tg_names,
                                    dist_df, window_map, atac_window_tensor):
    """
    Build [n_tf*n_tg, 2] features: [similarity/weight, distance-weighted accessibility]
    sim_mat:   [n_tg, n_tf]   (TG x TF similarity or weights)
    attn_mat:  [n_tg, n_tf]   (optional extra feature; if unused, ignore)
    """
    tg_set = set(tg_names)
    peak_set = set(window_map.keys())

    # Keep only rows that map to our TGs and peaks
    df = dist_df[dist_df["target_id"].isin(tg_set) & dist_df["peak_id"].isin(peak_set)].copy()

    # Map TG -> dict(peak_id -> dist_score)
    tg_to_scores = defaultdict(dict)
    for _, row in df.iterrows():
        tg = row["target_id"]
        peak_id = row["peak_id"]
        tg_to_scores[tg][peak_id] = float(row["TSS_dist_score"])

    # Precompute peak accessibility per window (mean across metacells)
    # atac_window_tensor shape: [W, C]
    atac_mean = atac_window_tensor.mean(dim=1).cpu().numpy()  # [W]

    features = []
    for j, tg in enumerate(tg_names):
        peak_scores = tg_to_scores.get(tg, {})
        # distance-weighted accessibility for this TG
        if peak_scores:
            acc_scores = []
            for peak_id, dist_score in peak_scores.items():
                win_idx = window_map[peak_id]          # safe because filtered to peak_set
                acc = atac_mean[win_idx]
                acc_scores.append(acc * dist_score)
            dist_weighted_accessibility = float(sum(acc_scores) / (len(acc_scores) + 1e-8))
        else:
            dist_weighted_accessibility = 0.0

        for i, tf in enumerate(tf_names):
            weight = float(sim_mat[j, i])  # or whichever score you want to use
            features.append([weight, dist_weighted_accessibility])

    return np.array(features)



def build_edge_features(model, dataset, chip_edges, dist_df, window_map):
    tf_names = dataset.tf_names
    tg_names = dataset.tg_names

    # filter dist table to model’s TGs & mapped peaks
    tg_set = set(tg_names)
    peak_set = set(window_map.keys())
    dist_df = dist_df[dist_df["target_id"].isin(tg_set) & dist_df["peak_id"].isin(peak_set)].copy()

    # get similarity/attention matrices from the model
    # e.g., sim_mat = model.export_tg_tf_similarity(tf_names, tg_names)  -> [G,T]
    sim_mat = model.export_tg_tf_similarity(tf_names, tg_names)  # implement this
    attn_mat = None  # or another optional matrix/feature

    X = compute_tf_tg_distance_features(sim_mat, attn_mat, tf_names, tg_names, dist_df, dataset.window_map, dataset.atac_window_tensor_all)

    # labels
    edge_list, labels = [], []
    for i, tf in enumerate(tf_names):
        for j, tg in enumerate(tg_names):
            edge = (tf.upper(), tg.upper())
            edge_list.append(edge)
            labels.append(1 if edge in chip_edges else 0)

    y = np.asarray(labels, dtype=np.int64)
    return X, y, edge_list




def train_tf_tg_classifier(model, dataset, ground_truth_file, out_path,
                                device="cuda", weight_attn_by_tf=False):
    # ground truth
    chip_df = pd.read_csv(ground_truth_file)
    chip_edges = set((a.upper(), b.upper()) for a,b in zip(chip_df["Gene1"], chip_df["Gene2"]))

    # distance table
    dist_df = pd.read_parquet(os.path.join(dataset.data_dir,
                                           f"genes_near_peaks_{dataset.chrom_id}.parquet"))

    # dataloader for mean attention (same one you already have)
    _, _, test_loader = prepare_dataloader(dataset, batch_size=64, world_size=1, rank=0)

    # signals
    sim_mat = get_tf_tg_affinity(model, dataset.tf_ids, dataset.tg_ids, device=device).cpu().numpy()
    mean_attn, _, _ = get_mean_attention(model, test_loader, device=device,
                                         weight_by_tf_expr=weight_attn_by_tf)     # [T,G]
    attn_mat = mean_attn.numpy()

    # features/labels
    X, y, edge_list = build_edge_features(
        sim_mat, attn_mat,
        dataset.tf_names, dataset.tg_names,
        chip_edges, dist_df,
        dataset.window_map, dataset.atac_window_tensor_all
    )

    # optional normalization (helps RF a bit when scales differ a lot)
    # from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # RF as before
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y if len(np.unique(y))>1 else None, random_state=42
    )

    param_dist = {
        "n_estimators": randint(200, 1200),
        "max_depth": [None] + list(range(10, 70, 10)),
        "max_features": ["sqrt", "log2", None],
        "min_samples_split": randint(2, 20),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        param_distributions=param_dist, n_iter=50,
        scoring="roc_auc", cv=cv, n_jobs=-1, verbose=2, random_state=42
    )
    rs.fit(X_tr, y_tr)

    print("Best params:", rs.best_params_)
    print("Best AUROC (CV):", rs.best_score_)
    y_score = rs.predict_proba(X_te)[:, 1]
    y_pred  = rs.predict(X_te)
    print("TF–TG classification report:")
    print(classification_report(y_te, y_pred))
    print("Test AUROC:", roc_auc_score(y_te, y_score))

    # save edge scores
    all_probs = rs.predict_proba(X)[:, 1]
    edge_df = pd.DataFrame({
        "TF": [e[0] for e in edge_list],
        "TG": [e[1] for e in edge_list],
        "probability": all_probs,
        "label": y
    })
    edge_df.to_csv(out_path, index=False)
    return rs, edge_df



def dist_broadcast_list(obj, src=0):
    device = f"cuda:{torch.cuda.current_device()}"
    if dist.get_rank() == src:
        data = pickle.dumps(obj)
        tensor = torch.as_tensor(list(data), dtype=torch.uint8, device=device)
        size = torch.tensor([tensor.numel()], device=device)
    else:
        tensor = torch.tensor([], dtype=torch.uint8, device=device)
        size   = torch.tensor([0], device=device)

    dist.broadcast(size, src=src)
    if dist.get_rank() != src:
        tensor = torch.empty(size.item(), dtype=torch.uint8, device=device)
    dist.broadcast(tensor, src=src)
    return pickle.loads(bytes(tensor.tolist()))


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
                
            dataset, model, optimizer = load_train_objs(
                DATA_DIR, CHROM_ID,
                D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT,
                subset_genes=subset_genes, lr=1e-3
            )
            
            if rank == 0:
                with open(os.path.join(iter_dir, "tf_vocab.json"), "w") as f:
                    json.dump(dataset.tf_name2id, f)
                with open(os.path.join(iter_dir, "tg_vocab.json"), "w") as f:
                    json.dump(dataset.tg_name2id, f)

            has_dist_bias = "No"
            if dataset.dist_bias_tensor is not None:
                has_dist_bias = "Yes"
            
            if rank == 0:
                logging.info("\n===== MultiomicTransformerDataset Loaded =====")
                logging.info(f"Chromosome:          {CHROM_ID}")
                logging.info(f"Genes:               {len(dataset.tg_ids)}")
                logging.info(f"Windows (RE):        {dataset.num_windows}")
                logging.info(f"TFs:                 {len(dataset.tf_ids)}")
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
            trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, gpu_id=rank, save_every=save_every, patience=PATIENCE)
        

            trainer.train(max_epochs=TOTAL_EPOCHS, path=iter_dir)

            if rank == 0:
                out_file = os.path.join(training_output_dir, "tf_tg_classifier_predictions.csv")
                clf, edge_df = train_tf_tg_classifier(
                    trainer.model.module, 
                    dataset, 
                    GROUND_TRUTH_FILE, 
                    out_file,
                    device=f"cuda:{rank}",
                    weight_attn_by_tf=False
                    )
                
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
                
                W = trainer.model.module.tf_tg_weights.detach().cpu().numpy()   # [tf_vocab, tg_vocab]
                tf_idx = [dataset.tf_name2id[n] for n in dataset.tf_names]
                tg_idx = [dataset.tg_name2id[n] for n in dataset.tg_names]
                W_sub = W[np.ix_(tf_idx, tg_idx)]
                pd.DataFrame(W_sub, index=dataset.tf_names, columns=dataset.tg_names)\
                .to_csv(os.path.join(iter_dir, "tf_tg_weights.csv"))
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