
from ast import arg
import os
import sys
import gtfparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
)

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
sys.path.append(str(PROJECT_DIR))

import models.tf_to_tg as tf_to_tg_module
import models.tf_to_dna as tf_to_dna_module
import utils

logging.basicConfig(level=logging.INFO)

def create_new_tf_peak_binding_model(ckpt_path: Path) -> tf_to_tg_module.TFTGRegulationModel:
    # 1) Recreate the base TF→DNA model with the same hyperparameters
    base_model = tf_to_dna_module.TFPeakBindingModel(
        tf_embedding_dim=128,
        hidden_dim=128,
        dropout=0.3,
        num_layers=4,
        num_heads=4,
        dim_head=32,
    )

    # 2) Wrap in Lightning module and load checkpoint
    lit_model = tf_to_dna_module.LitTFPeakBindingModel.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        model=base_model,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
    )

    state = torch.load(ckpt_path, map_location="cpu")
    lit_model.load_state_dict(state["state_dict"], strict=True)

    # 3) Get the trained base model and freeze it
    trained_tf_peak_model = lit_model.model
    trained_tf_peak_model.eval()
    for p in trained_tf_peak_model.parameters():
        p.requires_grad = False

    # 4) Inject into your TF→TG model
    tf_tg_model = tf_to_tg_module.TFTGRegulationModel(
        pretrained_tf_peak_model=trained_tf_peak_model,
        d_model=128,
    )
    
    return tf_tg_model

def load_ground_truth(ground_truth_file: Path | str) -> pd.DataFrame:
    if type(ground_truth_file) == str:
        ground_truth_file = Path(ground_truth_file)
        
    logging.info(f"Loading ground truth file: {ground_truth_file.name}")

    if ground_truth_file.suffix == ".csv":
        sep = ","
    elif ground_truth_file.suffix == ".tsv":
        sep="\t"
        
    ground_truth_df = pd.read_csv(ground_truth_file, sep=sep, on_bad_lines="skip", engine="python")
    
    if "chip" in ground_truth_file.name and "atlas" in ground_truth_file.name:
        ground_truth_df = ground_truth_df[["source_id", "target_id"]]

    if ground_truth_df.columns[0] != "Source" or ground_truth_df.columns[1] != "Target":
        ground_truth_df = ground_truth_df.rename(columns={ground_truth_df.columns[0]: "Source", ground_truth_df.columns[1]: "Target"})
    ground_truth_df["Source"] = ground_truth_df["Source"].astype(str).str.capitalize()
    ground_truth_df["Target"] = ground_truth_df["Target"].astype(str).str.capitalize()
    
    # Build TF, TG, and edge sets for quick lookup later
    gt = ground_truth_df[["Source", "Target"]].dropna()

    return gt

def split_genes_by_chromosome(gene_reference_file: Path):
    gene_ref_df = gtfparse.read_gtf(
        gene_reference_file,
        result_type="pandas"
        )
        
    gene_chrom = gene_ref_df[["seqname", "gene_name"]].rename(columns={"seqname": "chrom", "gene_name": "TG"})
    gene_chrom["chrom"] = gene_chrom["chrom"]
    gene_chrom["TG"] = gene_chrom["TG"]
    # gene_chrom = gene_chrom[gene_chrom["TG"].str.upper().isin(dataset_tgs)]

    # Train set: genes on chromosomes 1 - 15
    train_genes = gene_chrom[gene_chrom["chrom"].isin([str(i) for i in range(1, 16)])]["TG"].unique()
    logging.info(f"Train set: {len(train_genes)} genes")

    # Validation set: genes on chromosomes 16 - 18
    val_genes = gene_chrom[gene_chrom["chrom"].isin([str(i) for i in range(16, 19)])]["TG"].unique()
    logging.info(f"Validation set: {len(val_genes)} genes")

    # Test set: genes on chromosome 19
    test_genes = gene_chrom[gene_chrom["chrom"].isin([str(19)])]["TG"].unique()
    logging.info(f"Test set: {len(test_genes)} genes")
    
    return train_genes, val_genes, test_genes

def create_train_val_test_splits(ground_truth_df: pd.DataFrame, train_genes: np.ndarray, val_genes: np.ndarray, test_genes: np.ndarray):
    # Create train/val/test splits of the ground truth based on the TG's chromosome
    train_genes_set = set(train_genes)
    val_genes_set = set(val_genes)
    test_genes_set = set(test_genes)

    gt_train_df = ground_truth_df[ground_truth_df["Target"].isin(train_genes_set)].copy()
    gt_val_df = ground_truth_df[ground_truth_df["Target"].isin(val_genes_set)].copy()
    gt_test_df = ground_truth_df[ground_truth_df["Target"].isin(test_genes_set)].copy()

    logging.info(f"Train interactions: {len(gt_train_df)}")
    logging.info(f"Validation interactions: {len(gt_val_df)}")
    logging.info(f"Test interactions: {len(gt_test_df)}")

    return gt_train_df, gt_val_df, gt_test_df

def load_ground_truth_files(gt_path_list: list[Path]) -> pd.DataFrame:
    gt_dfs = []
    for gt_path in gt_path_list:
        gt_df = load_ground_truth(gt_path)
        gt_dfs.append(gt_df)
    merged_gt_df = pd.concat(gt_dfs, ignore_index=True)
    return merged_gt_df

def create_labeled_tf_tg_dataset(
    true_interactions: set[tuple[str, str]],
    false_interactions: set[tuple[str, str]],
    tf_name_to_idx: dict[str, int],
    tg_id_to_idx: dict[str, int],
    drop_missing: bool = True,
) -> pd.DataFrame:
    """
    Create a labeled TF-TG interaction dataset.

    Labels:
        true interactions  -> 1
        false interactions -> 0

    Returns
    -------
    pd.DataFrame with columns:
        tf_name, tg_id, tf_idx, tg_idx, label
    """

    rows = []

    # Create rows for true interactions with label 1
    for tf, tg in true_interactions:
        rows.append((tf, tg, 1))

    # Create rows for false interactions with label 0
    for tf, tg in false_interactions:
        rows.append((tf, tg, 0))

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=["tf_name", "tg_id", "label"])

    # Map TF names and TG IDs to their respective indices
    df["tf_idx"] = df["tf_name"].map(tf_name_to_idx)
    df["tg_idx"] = df["tg_id"].map(tg_id_to_idx)

    # Check for any missing mappings and handle them
    missing_mask = df["tf_idx"].isna() | df["tg_idx"].isna()

    if missing_mask.any():
        n_missing = missing_mask.sum()

        if drop_missing:
            logging.info(f"Dropping {n_missing} interactions with missing TF or TG indices.")
            df = df.loc[~missing_mask].copy()
        else:
            missing_examples = df.loc[missing_mask].head()
            raise ValueError(
                f"{n_missing} interactions are missing TF or TG indices.\n"
                f"Examples:\n{missing_examples}"
            )

    # Convert indices and labels to appropriate data types
    df["tf_idx"] = df["tf_idx"].astype(np.int64)
    df["tg_idx"] = df["tg_idx"].astype(np.int64)
    df["label"] = df["label"].astype(np.float32)

    # Optional: shuffle rows
    df = df.sample(frac=1.0, random_state=123).reset_index(drop=True)

    return df


def _create_labeled_df(gt_df: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
    true_edges, false_edges = utils.create_true_false_edges(
        edge_df=gt_df,
        tf_names=tf_name_to_idx.keys(),
        tf_col="Source",
        item_col="Target",
        pct_true_edges=0.15,
        true_false_ratio=2.0,
        seed=seed,
    )
    return create_labeled_tf_tg_dataset(
        true_interactions=true_edges,
        false_interactions=false_edges,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
        drop_missing=False,
    )
    

def build_tftg_inputs(tf_tg_df, max_peaks_per_tg=64, max_cells_per_pair=8, seed=123, zero_fields=None):
    rng = np.random.default_rng(seed)
    tf_names = []
    tg_names = []
    cell_ids = []
    labels = []
    tf_embeddings = []
    tf_masks = []
    peak_seqs = []
    peak_access = []
    peak_dist = []
    tf_expr = []
    tg_expr = []
    tg_embed = []
    peak_id_sets = []
    
    if zero_fields is None:
        zero_fields = set()
    else:
        zero_fields = set(zero_fields)
        
    ZEROABLE_FIELDS = {
        "tf_embedding",
        "peak_sequences",
        "peak_accessibility",
        "peak_distance",
        "tf_expression",
        "tg_expression",
        "tg_embedding",
    }

    def validate_zero_fields(zero_fields):
        zero_fields = set(zero_fields or [])
        invalid = zero_fields - ZEROABLE_FIELDS
        if invalid:
            raise ValueError(
                f"Invalid zero_fields: {sorted(invalid)}. "
                f"Valid options are: {sorted(ZEROABLE_FIELDS)}"
            )
        return zero_fields
    
    zero_fields = validate_zero_fields(zero_fields)

    # Iterate over TF-TG pairs in the labeled dataset and build inputs for the model
    for _, row in tf_tg_df.iterrows():
        tf_name = row["tf_name"]
        tg_name = row["tg_id"]
        label = float(row["label"])
        tg_norm = str(tg_name).upper()
        tf_norm = str(tf_name).upper()

        # Peaks linked to this TG and present in ATAC data
        tg_peaks = peak_to_gene[peak_to_gene["target_id_norm"] == tg_norm]
        tg_peaks = tg_peaks[tg_peaks["peak_id"].isin(atac_peak_map.keys())]
        if tg_peaks.empty:
            continue

        # Take max_peak_per_tg closest peaks by absolute distance
        tg_peaks = tg_peaks.assign(abs_dist=tg_peaks["TSS_dist"].abs())
        tg_peaks = tg_peaks.sort_values("abs_dist").head(max_peaks_per_tg)

        # Get the one-hot encoded sequences and distances for these peaks
        peak_ids = tg_peaks["peak_id"].tolist()
        peak_indices = [atac_peak_map[p] for p in peak_ids]
        peak_seq = atac_peak_tensor[peak_indices]
        peak_seq = peak_seq[:max_peaks_per_tg]
        peak_dst = tg_peaks["TSS_dist"].to_numpy(dtype=np.float32)

        # Pad to max_peaks_per_tg if there are fewer peaks than the maximum
        # Creates a tensor of shape [max_peaks_per_tg, peak_len, num_nucleotides] where excess peaks are zero-padded
        n_peaks = peak_seq.shape[0]
        if n_peaks < max_peaks_per_tg:
            pad_len = max_peaks_per_tg - n_peaks
            peak_seq = torch.nn.functional.pad(peak_seq, (0, 0, 0, 0, 0, pad_len))
            peak_dst = np.pad(peak_dst, (0, pad_len), constant_values=0.0)
            peak_ids = peak_ids + [""] * pad_len

        tf_idx = tf_name_to_idx.get(tf_name)
        tg_idx = tg_id_to_idx.get(tg_name)
        if tf_idx is None or tg_idx is None:
            continue

        # Randomly sample cells to size max_cells_per_pair keep the batch size reasonable
        if max_cells_per_pair is None or max_cells_per_pair >= len(common_cells):
            sampled_cells = common_cells
        else:
            sampled_cells = rng.choice(common_cells, size=max_cells_per_pair, replace=False).tolist()

        # Create inputs for each sampled cell and append to the lists
        for cell_id in sampled_cells:
            # ATAC accessibility for this cell and these peaks
            peak_acc = atac_pseudobulk.reindex(peak_ids)[cell_id].fillna(0.0).to_numpy(dtype=np.float32)
            if peak_acc.shape[0] < max_peaks_per_tg:
                peak_acc = np.pad(peak_acc, (0, max_peaks_per_tg - peak_acc.shape[0]), constant_values=0.0)

            tf_expr_val = float(rna_pseudobulk_norm.loc[tf_norm, cell_id]) if tf_norm in rna_pseudobulk_norm.index else 0.0
            tg_expr_val = float(rna_pseudobulk_norm.loc[tg_norm, cell_id]) if tg_norm in rna_pseudobulk_norm.index else 0.0

            tf_embeddings.append(tf_embeddings_tensor[tf_idx])
            tf_masks.append(tf_mask_tensor[tf_idx])
            peak_seqs.append(peak_seq)
            peak_access.append(peak_acc)
            peak_dist.append(peak_dst)
            tf_expr.append(tf_expr_val)
            tg_expr.append(tg_expr_val)
            tg_embed.append(tg_embedding_table(torch.tensor([tg_idx])).squeeze(0).detach())
            peak_id_sets.append(peak_ids)
            tf_names.append(tf_name)
            tg_names.append(tg_name)
            cell_ids.append(cell_id)
            labels.append(label)
    
    # Combine the inputs for each TF-TG pair into tensors
    peak_distance_tensor = torch.tensor(np.stack(peak_dist), dtype=torch.float32)
    tf_embedding = torch.stack(tf_embeddings)
    tf_mask = torch.stack(tf_masks)
    peak_sequences = torch.stack(peak_seqs)
    peak_accessibility = torch.tensor(np.stack(peak_access), dtype=torch.float32)
    tf_expr_tensor = torch.tensor(tf_expr, dtype=torch.float32)
    tg_expr_tensor = torch.tensor(tg_expr, dtype=torch.float32)
    tg_embed_tensor = torch.stack(tg_embed)

    def maybe_zero(name, tensor, zero_fields):
        return torch.zeros_like(tensor) if name in zero_fields else tensor

    return {
        "tf_name": tf_names,
        "tg_name": tg_names,
        "cell_id": cell_ids,
        "peak_ids": peak_id_sets,
        "label": torch.tensor(labels, dtype=torch.float32),

        "tf_embedding": maybe_zero("tf_embedding", tf_embedding, zero_fields),
        "tf_mask": tf_mask,

        "peak_sequences": maybe_zero("peak_sequences", peak_sequences, zero_fields),

        "peak_accessibility": maybe_zero("peak_accessibility", peak_accessibility, zero_fields),
        "peak_distance": maybe_zero("peak_distance", peak_distance_tensor, zero_fields),
        "tf_expression": maybe_zero("tf_expression", tf_expr_tensor, zero_fields),
        "tg_expression": maybe_zero("tg_expression", tg_expr_tensor, zero_fields),
        "tg_embedding": maybe_zero("tg_embedding", tg_embed_tensor, zero_fields),
    }

class TFTGInputsDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
        self.length = inputs["label"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "tf_embedding": self.inputs["tf_embedding"][idx],
            "tf_mask": self.inputs["tf_mask"][idx],
            "peak_sequences": self.inputs["peak_sequences"][idx],
            "peak_accessibility": self.inputs["peak_accessibility"][idx],
            "peak_distance": self.inputs["peak_distance"][idx],
            "tf_expression": self.inputs["tf_expression"][idx],
            "tg_expression": self.inputs["tg_expression"][idx],
            "tg_embedding": self.inputs["tg_embedding"][idx],
            "label": self.inputs["label"][idx],
            "tf_name": self.inputs["tf_name"][idx],
            "tg_name": self.inputs["tg_name"][idx],
            "cell_id": self.inputs["cell_id"][idx],
            "peak_ids": self.inputs["peak_ids"][idx],
        }
        
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        logits = model(
            tf_embedding=batch["tf_embedding"],
            
            tf_mask=batch["tf_mask"],
            peak_sequences=batch["peak_sequences"],
            peak_accessibility=batch["peak_accessibility"],
            peak_distance=batch["peak_distance"],
            tf_expression=batch["tf_expression"],
            tg_expression=batch["tg_expression"],
            tg_embedding=batch["tg_embedding"],
        )
        loss = criterion(logits, batch["label"])
        total_loss += loss.item() * batch["label"].shape[0]
    return total_loss / max(len(loader.dataset), 1)

def _move_batch_to_device(batch, device):
    return {
        "tf_embedding": batch["tf_embedding"].to(device),
        "tf_mask": batch["tf_mask"].to(device),
        "peak_sequences": batch["peak_sequences"].to(device),
        "peak_accessibility": batch["peak_accessibility"].to(device),
        "peak_distance": batch["peak_distance"].to(device),
        "tf_expression": batch["tf_expression"].to(device),
        "tg_expression": batch["tg_expression"].to(device),
        "tg_embedding": batch["tg_embedding"].to(device),
        "label": batch["label"].to(device),
    }

def compute_binary_classification_metrics(
    labels,
    scores,
    score_threshold: float = 0.5,
    random_state: int = 42,
):
    """
    labels: array-like of 0/1 labels
    scores: array-like of predicted probabilities after sigmoid
    """

    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores).astype(float).ravel()

    preds = (scores >= score_threshold).astype(int)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)

    if len(np.unique(labels)) < 2:
        auroc = np.nan
        auprc = np.nan
        rand_auroc = np.nan
        rand_auprc = np.nan
    else:
        auroc = roc_auc_score(labels, scores)
        auprc = average_precision_score(labels, scores)

        rng = np.random.default_rng(random_state)
        rand_scores = rng.permutation(scores)

        rand_auroc = roc_auc_score(labels, rand_scores)
        rand_auprc = average_precision_score(labels, rand_scores)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "rand_auroc": rand_auroc,
        "rand_auprc": rand_auprc,
        "accuracy": accuracy,
        "precision": precision,
        "n_edges": len(labels),
        "n_pos": int(labels.sum()),
        "n_neg": int((labels == 0).sum()),
        "score_threshold": score_threshold,
    }
    
@torch.no_grad()
def evaluate_with_metrics(
    model,
    loader,
    criterion,
    device,
    score_threshold: float = 0.5,
    random_state: int = 42,
):
    model.eval()

    total_loss = 0.0
    n_examples = 0

    all_scores = []
    all_labels = []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)

        logits = model(
            tf_embedding=batch["tf_embedding"],
            tf_mask=batch["tf_mask"],
            peak_sequences=batch["peak_sequences"],
            peak_accessibility=batch["peak_accessibility"],
            peak_distance=batch["peak_distance"],
            tf_expression=batch["tf_expression"],
            tg_expression=batch["tg_expression"],
            tg_embedding=batch["tg_embedding"],
        )

        labels = batch["label"]

        loss = criterion(logits, labels)

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        n_examples += batch_size

        scores = torch.sigmoid(logits)

        all_scores.append(scores.detach().cpu().numpy().ravel())
        all_labels.append(labels.detach().cpu().numpy().ravel())

    mean_loss = total_loss / max(n_examples, 1)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    metrics = compute_binary_classification_metrics(
        labels=all_labels,
        scores=all_scores,
        score_threshold=score_threshold,
        random_state=random_state,
    )

    metrics["loss"] = mean_loss

    return metrics

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            tf_embedding=batch["tf_embedding"],
            tf_mask=batch["tf_mask"],
            peak_sequences=batch["peak_sequences"],
            peak_accessibility=batch["peak_accessibility"],
            peak_distance=batch["peak_distance"],
            tf_expression=batch["tf_expression"],
            tg_expression=batch["tg_expression"],
            tg_embedding=batch["tg_embedding"],
        )
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch["label"].shape[0]
    return total_loss / max(len(loader.dataset), 1)

def train_one_epoch_track_test_metrics(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    epoch: int,
    global_step_start: int = 0,
    max_batches: int = 50,
    score_threshold: float = 0.5,
    random_state: int = 42,
    eval_every_n_batches: int = 1,
):
    """
    Trains for one epoch.

    After every eval_every_n_batches train batches, evaluates the full test set.

    Returns
    -------
    mean_train_loss : float
    metrics_df : pd.DataFrame
        One row per test-set evaluation.
    global_step : int
        Updated global step after this epoch.
    """

    model.train()

    total_train_loss = 0.0
    n_train_examples = 0

    metric_rows = []
    global_step = global_step_start

    for batch_num, batch in enumerate(train_loader):
        if batch_num >= max_batches:
            break
        
        model.train()

        batch = _move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            tf_embedding=batch["tf_embedding"],
            tf_mask=batch["tf_mask"],
            peak_sequences=batch["peak_sequences"],
            peak_accessibility=batch["peak_accessibility"],
            peak_distance=batch["peak_distance"],
            tf_expression=batch["tf_expression"],
            tg_expression=batch["tg_expression"],
            tg_embedding=batch["tg_embedding"],
        )

        labels = batch["label"]
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        train_batch_loss = loss.item()

        total_train_loss += train_batch_loss * batch_size
        n_train_examples += batch_size

        global_step += 1

        should_eval = (
            eval_every_n_batches is not None
            and eval_every_n_batches > 0
            and ((batch_num + 1) % eval_every_n_batches == 0)
        )

        if should_eval:
            test_seed = random_state + global_step

            test_metrics = evaluate_with_metrics(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                score_threshold=score_threshold,
                random_state=test_seed,
            )

            row = {
                "epoch": epoch,
                "batch_num": batch_num,
                "global_step": global_step,
                "train_batch_loss": train_batch_loss,
                "test_loss": test_metrics["loss"],
                "auroc": test_metrics["auroc"],
                "auprc": test_metrics["auprc"],
                "rand_auroc": test_metrics["rand_auroc"],
                "rand_auprc": test_metrics["rand_auprc"],
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "n_edges": test_metrics["n_edges"],
                "n_pos": test_metrics["n_pos"],
                "n_neg": test_metrics["n_neg"],
                "score_threshold": test_metrics["score_threshold"],
            }

            metric_rows.append(row)

            logging.info(
                f"Epoch {epoch:02d} | "
                f"batch {batch_num:04d} | "
                f"step {global_step:05d} | "
                f"train_batch_loss={train_batch_loss:.4f} | "
                f"test_loss={test_metrics['loss']:.4f} | "
                f"AUROC={test_metrics['auroc']:.4f} | "
                f"AUPRC={test_metrics['auprc']:.4f} | "
                f"rand_AUROC={test_metrics['rand_auroc']:.4f} | "
                f"rand_AUPRC={test_metrics['rand_auprc']:.4f} | "
                f"pos_rate={test_metrics['n_pos'] / max(test_metrics['n_edges'], 1):.4f}"
            )

    mean_train_loss = total_train_loss / max(n_train_examples, 1)
    metrics_df = pd.DataFrame(metric_rows)

    return mean_train_loss, metrics_df, global_step

if __name__ == "__main__":
    ckpt_path = PROJECT_DIR / "checkpoints" / "tfbind_train_3668735" / "epoch=03-val_auroc=0.9271-val_loss=0.2623.ckpt"

    tf_name_file = DATA_DIR / "databases" / "motif_information" / "mm10" / "TF_Information_all_motifs.txt"
    gene_ref_file = DATA_DIR / "genome_data" / "genome_annotation" / "mm10" / "Mus_musculus.GRCm39.115.gtf.gz"
    genome_fasta_path = DATA_DIR / "genome_data" / "reference_genome" / "mm10" / "mm10.fa"
    chrom_sizes_path = DATA_DIR / "genome_data" / "reference_genome" / "mm10" / "mm10.chrom.sizes"

    atac_pseudobulk = pd.read_parquet(PROJECT_DIR / "data" / "ATAC_data" / "RE_pseudobulk.parquet")
    peak_to_gene_distance = pd.read_parquet(PROJECT_DIR / "data" / "ATAC_data" / "peak_to_gene_dist.parquet")

    rna_pseudobulk = pd.read_parquet(PROJECT_DIR / "data" / "RNA_data" / "TG_pseudobulk.parquet")
    
    mm10_chip_atlas_file = DATA_DIR / "ground_truth_files" / "chip_atlas_tf_peak_tg_dist.csv"
    rn111_file = DATA_DIR / "ground_truth_files" / "RN111.tsv"
    rn112_file = DATA_DIR / "ground_truth_files" / "RN112.tsv"
    rn114_file = DATA_DIR / "ground_truth_files" / "RN114.tsv"
    rn116_file = DATA_DIR / "ground_truth_files" / "RN116.tsv"
    
    merged_ground_truth_df = load_ground_truth_files([
        mm10_chip_atlas_file,
        rn111_file,
        rn112_file,
        rn114_file,
        rn116_file,
    ])
    
    training_cache_dir = PROJECT_DIR / "data" / "training_data_cache"
    tf_name_to_idx_cache_path = training_cache_dir / "tf_name_to_idx.csv"
    tf_name_to_idx = pd.read_csv(tf_name_to_idx_cache_path).set_index("tf_name")["tf_idx"].to_dict()

    # Create a mapping from TG name to index across all ground-truth TGs
    tg_id_to_idx = {tg: idx for idx, tg in enumerate(merged_ground_truth_df["Target"].unique())}

    train_genes, val_genes, test_genes = split_genes_by_chromosome(gene_ref_file)
    
    gt_train_df, gt_val_df, gt_test_df = create_train_val_test_splits(merged_ground_truth_df, train_genes, val_genes, test_genes)
    
    logging.info(f"Train genes in ground truth: {gt_train_df['Target'].nunique()}")
    logging.info(f"Validation genes in ground truth: {gt_val_df['Target'].nunique()}")
    logging.info(f"Test genes in ground truth: {gt_test_df['Target'].nunique()}")
    
    # Create labeled datasets, split by train/val/test chromosomes
    tf_tg_labeled_train_df = _create_labeled_df(gt_train_df)
    tf_tg_labeled_val_df = _create_labeled_df(gt_val_df)
    tf_tg_labeled_test_df = _create_labeled_df(gt_test_df)

    logging.info(f"Train labeled edges: {len(tf_tg_labeled_train_df)}")
    logging.info(f"Val labeled edges: {len(tf_tg_labeled_val_df)}")
    logging.info(f"Test labeled edges: {len(tf_tg_labeled_test_df)}")
    
    dataset_peaks = atac_pseudobulk.index.to_list()
    valid_chroms = {f"chr{i}" for i in range(1, 20)}
    dataset_peaks = [peak for peak in dataset_peaks if peak.split(":", 1)[0] in valid_chroms]
    atac_peak_map = {peak: idx for idx, peak in enumerate(dataset_peaks)}
    logging.info(f"Peaks in dataset: {dataset_peaks[:2]}...{dataset_peaks[-2:]} (total: {len(dataset_peaks)})")

    # Build TFTGRegulationModel inputs from TF-TG ground truth + ATAC peaks
    training_cache_dir = PROJECT_DIR / "data" / "training_data_cache"
    tf_embedding_cache_path = training_cache_dir / "tf_embeddings.pt"
    tf_mask_cache_path = training_cache_dir / "tf_masks.pt"
    atac_peak_onehot_cache_path = training_cache_dir / "atac_peak_onehot_array.pt"

    tf_embeddings_tensor = torch.load(tf_embedding_cache_path)
    tf_mask_tensor = torch.load(tf_mask_cache_path)

    # Create or load the one-hot encoded peak sequences for the ATAC peaks in the dataset
    dataset_peaks = list(atac_peak_map.keys())
    if os.path.exists(atac_peak_onehot_cache_path):
        atac_peak_tensor = torch.load(atac_peak_onehot_cache_path)
    else:
        atac_peak_array = utils.create_centered_peak_onehot_array(
            peak_ids=dataset_peaks,
            genome_fasta=genome_fasta_path,
            chrom_sizes=utils.load_chrom_sizes(chrom_sizes_path),
            peak_id_to_idx=atac_peak_map,
            flank_size=128,
            dtype=np.float32,
            pad_out_of_bounds=True,
            num_workers=1,
        )
        atac_peak_tensor = torch.as_tensor(atac_peak_array, dtype=torch.float32)
        torch.save(atac_peak_tensor, atac_peak_onehot_cache_path)

    # Align cell IDs across RNA and ATAC pseudobulk matrices
    rna_pseudobulk_norm = rna_pseudobulk.copy()
    rna_pseudobulk_norm.index = rna_pseudobulk_norm.index.str.upper()

    common_cells = sorted(set(rna_pseudobulk_norm.columns) & set(atac_pseudobulk.columns))
    logging.info(f"Common cells: {len(common_cells)}")

    peak_to_gene = peak_to_gene_distance.copy()
    peak_to_gene["target_id_norm"] = peak_to_gene["target_id"].str.upper()

    tg_embedding_table = torch.nn.Embedding(len(tg_id_to_idx), 128)
        
    def _sample_df(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
        if n is None or len(df) <= n:
            return df
        return df.sample(n=n, random_state=seed)

    # Sample a subset of the TF-TG pairs for faster training during development
    tf_tg_train_subset = _sample_df(tf_tg_labeled_train_df, n=128, seed=123)
    tf_tg_val_subset = _sample_df(tf_tg_labeled_val_df, n=128, seed=123)
    tf_tg_test_subset = _sample_df(tf_tg_labeled_test_df, n=128, seed=123)

    zero_fields = None

    # Build the TFTGRegulationModel inputs for the sampled TF-TG pairs for the train/val/test sets
    tftg_inputs_train = build_tftg_inputs(tf_tg_train_subset, max_peaks_per_tg=8, max_cells_per_pair=16, zero_fields=zero_fields)
    tftg_inputs_val = build_tftg_inputs(tf_tg_val_subset, max_peaks_per_tg=8, max_cells_per_pair=16, zero_fields=zero_fields)
    tftg_inputs_test = build_tftg_inputs(tf_tg_test_subset, max_peaks_per_tg=8, max_cells_per_pair=16, zero_fields=zero_fields)

    logging.info({"train": len(tftg_inputs_train["label"]), "val": len(tftg_inputs_val["label"]), "test": len(tftg_inputs_test["label"])})

    # train_dataset = TFTGInputsDataset(shuffled_inputs_train)
    train_dataset = TFTGInputsDataset(tftg_inputs_train)
    val_dataset = TFTGInputsDataset(tftg_inputs_val)
    test_dataset = TFTGInputsDataset(tftg_inputs_test)

    batch_size = 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        )

    logging.info(f"Train/Val/Test sizes: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

    tf_tg_model = create_new_tf_peak_binding_model(ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf_tg_model = tf_tg_model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(tf_tg_model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 1
    global_step = 0

    all_metric_dfs = []

    initial_metrics = evaluate_with_metrics(
        model=tf_tg_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        score_threshold=0.5,
        random_state=42,
    )

    initial_row = {
        "epoch": 0,
        "batch_num": -1,
        "global_step": 0,
        "train_batch_loss": np.nan,
        "test_loss": initial_metrics["loss"],
        "auroc": initial_metrics["auroc"],
        "auprc": initial_metrics["auprc"],
        "rand_auroc": initial_metrics["rand_auroc"],
        "rand_auprc": initial_metrics["rand_auprc"],
        "accuracy": initial_metrics["accuracy"],
        "precision": initial_metrics["precision"],
        "n_edges": initial_metrics["n_edges"],
        "n_pos": initial_metrics["n_pos"],
        "n_neg": initial_metrics["n_neg"],
        "score_threshold": initial_metrics["score_threshold"],
    }

    logging.info("\nInitial evaluation before training:")
    logging.info(initial_row)

    logging.info("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        train_loss, epoch_metrics_df, global_step = train_one_epoch_track_test_metrics(
            model=tf_tg_model,
            train_loader=train_loader,
            test_loader=train_loader,  # evaluate on train set for debugging; switch to test_loader for real eval
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            global_step_start=global_step,
            max_batches=25,
            score_threshold=0.5,
            random_state=42,
            eval_every_n_batches=1,  # evaluate test set after every train batch
        )

        val_loss = evaluate(tf_tg_model, val_loader, criterion, device)

        all_metric_dfs.append(epoch_metrics_df)

        logging.info(
            f"Epoch {epoch:02d} complete | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

    batch_metric_df = pd.concat(all_metric_dfs, ignore_index=True)

    batch_metric_path = PROJECT_DIR / "testing_results" / "test_metrics_per_train_batch.csv"
    batch_metric_path.parent.mkdir(parents=True, exist_ok=True)

    batch_metric_df.to_csv(batch_metric_path, index=False)

    batch_metric_df.head()