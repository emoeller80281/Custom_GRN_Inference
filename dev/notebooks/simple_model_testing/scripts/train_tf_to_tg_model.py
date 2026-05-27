
import os
import sys
import gtfparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
)

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
sys.path.append(str(PROJECT_DIR))

import models.tf_to_tg as tf_to_tg_module
import models.tf_to_dna as tf_to_dna_module
import utils
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_new_tf_tg_binding_model(tf_bind_model_path: Path) -> tf_to_tg_module.TFTGRegulationModel:
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
        checkpoint_path=tf_bind_model_path,
        model=base_model,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
    )

    state = torch.load(tf_bind_model_path, map_location="cpu")
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
        tf_peak_chunk_size=2048,
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


def prepare_tftg_lookup_tables(
    peak_to_gene,
    atac_peak_map,
    atac_pseudobulk,
    rna_pseudobulk_norm,
    dataset_peaks,
    common_cells,
    max_precompute_peaks=64,
):
    valid_peak_set = set(atac_peak_map.keys())

    peak_to_gene_valid = peak_to_gene[
        peak_to_gene["peak_id"].isin(valid_peak_set)
    ].copy()

    peak_to_gene_valid["abs_dist"] = peak_to_gene_valid["TSS_dist"].abs()

    tg_to_peak_info = {}

    for tg_norm, sub in peak_to_gene_valid.groupby("target_id_norm", sort=False):
        sub = sub.sort_values("abs_dist").head(max_precompute_peaks)

        peak_ids = sub["peak_id"].tolist()
        peak_indices = np.asarray([atac_peak_map[p] for p in peak_ids], dtype=np.int64)
        peak_distances = sub["TSS_dist"].to_numpy(dtype=np.float32)

        tg_to_peak_info[tg_norm] = {
            "peak_ids": peak_ids,
            "peak_indices": peak_indices,
            "peak_distances": peak_distances,
        }

    cell_to_idx = {cell: i for i, cell in enumerate(common_cells)}

    atac_mat = (
        atac_pseudobulk
        .reindex(index=dataset_peaks, columns=common_cells)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    rna_mat = (
        rna_pseudobulk_norm
        .reindex(columns=common_cells)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    gene_to_rna_idx = {
        gene: i for i, gene in enumerate(rna_pseudobulk_norm.index)
    }

    return tg_to_peak_info, cell_to_idx, atac_mat, rna_mat, gene_to_rna_idx

class TFTGEdgeBagDataset(Dataset):
    def __init__(
        self,
        inputs,
        *,
        tf_embeddings_tensor,
        tf_mask_tensor,
        atac_peak_tensor,
        max_cells_per_pair=None,
        zero_fields=None,
        strict=True,
    ):
        """
        Groups rows from build_tftg_inputs() by (tf_name, tg_name).

        Each item is one TF-TG edge bag containing multiple sampled cells.
        Large tensors are gathered lazily using compact indices.
        """

        self.inputs = inputs
        self.tf_embeddings_tensor = tf_embeddings_tensor
        self.tf_mask_tensor = tf_mask_tensor
        self.atac_peak_tensor = atac_peak_tensor
        self.max_cells_per_pair = max_cells_per_pair

        if zero_fields is None:
            zero_fields = set()
        else:
            zero_fields = set(zero_fields)

        self.zero_fields = zero_fields

        required_list_keys = [
            "tf_name",
            "tg_name",
            "cell_id",
        ]

        required_tensor_keys = [
            "label",
            "tf_idx",
            "tg_idx",
            "peak_indices",
            "peak_accessibility",
            "peak_distance",
            "peak_mask",
            "tf_expression",
            "tg_expression",
        ]

        lengths = {}

        for key in required_list_keys:
            lengths[key] = len(inputs[key])

        for key in required_tensor_keys:
            lengths[key] = inputs[key].shape[0]

        unique_lengths = sorted(set(lengths.values()))

        if len(unique_lengths) != 1:
            msg = "\n".join(
                [f"{key:20s}: {length}" for key, length in lengths.items()]
            )

            if strict:
                raise ValueError(
                    "Input fields have inconsistent first-dimension lengths:\n"
                    f"{msg}\n\n"
                    "This usually means one of the lists was appended without "
                    "a matching tensor row, or tensors were filtered after metadata "
                    "was created."
                )
            else:
                self.n_rows = min(unique_lengths)
                print(
                    "WARNING: Input fields have inconsistent lengths. "
                    f"Using first {self.n_rows} rows only.\n{msg}"
                )
        else:
            self.n_rows = unique_lengths[0]

        groups = defaultdict(list)

        for i in range(self.n_rows):
            tf = inputs["tf_name"][i]
            tg = inputs["tg_name"][i]
            groups[(tf, tg)].append(i)

        self.edge_keys = list(groups.keys())
        self.groups = [
            torch.tensor(v, dtype=torch.long)
            for v in groups.values()
        ]

    def __len__(self):
        return len(self.groups)

    def _maybe_zero(self, name, tensor):
        if name in self.zero_fields:
            return torch.zeros_like(tensor)
        return tensor

    def __getitem__(self, idx):
        row_idx = self.groups[idx]

        if self.max_cells_per_pair is not None:
            row_idx = row_idx[: self.max_cells_per_pair]

        row_idx_list = row_idx.tolist()

        label = self.inputs["label"][row_idx[0]]

        # These should be constant within a TF-TG edge bag
        tf_idx = self.inputs["tf_idx"][row_idx[0]]
        tg_idx = self.inputs["tg_idx"][row_idx[0]]

        # Gather large static lookup tensors once per edge bag
        tf_embedding = self.tf_embeddings_tensor[tf_idx]
        tf_mask = self.tf_mask_tensor[tf_idx]

        # Small per-cell tensors
        peak_indices = self.inputs["peak_indices"][row_idx]          # [C, P]
        peak_accessibility = self.inputs["peak_accessibility"][row_idx]  # [C, P]
        peak_distance = self.inputs["peak_distance"][row_idx]        # [C, P]
        peak_mask = self.inputs["peak_mask"][row_idx]                # [C, P]
        tf_expression = self.inputs["tf_expression"][row_idx]        # [C]
        tg_expression = self.inputs["tg_expression"][row_idx]        # [C]

        # Gather peak sequences lazily
        # atac_peak_tensor is probably uint8: [n_peaks, seq_len, 4]
        # peak_sequences: [C, P, seq_len, 4]
        peak_sequences = self.atac_peak_tensor[peak_indices]

        # Repeat static TF/TG features across cells because your current collate/model
        # expect a cell dimension on every field.
        n_cells = row_idx.shape[0]

        tf_embedding = tf_embedding.unsqueeze(0).expand(n_cells, -1, -1)
        tf_mask = tf_mask.unsqueeze(0).expand(n_cells, -1)

        item = {
            "label": label,
            "tf_name": self.edge_keys[idx][0],
            "tg_name": self.edge_keys[idx][1],
            "cell_ids": [
                self.inputs["cell_id"][i]
                for i in row_idx_list
            ],

            "tf_idx": tf_idx,
            "tg_idx": tg_idx,
            "peak_indices": peak_indices,

            "tf_embedding": self._maybe_zero(
                "tf_embedding",
                tf_embedding.float(),
            ),
            "tf_mask": tf_mask.bool(),

            "peak_sequences": self._maybe_zero(
                "peak_sequences",
                peak_sequences.float(),
            ),
            "peak_accessibility": self._maybe_zero(
                "peak_accessibility",
                peak_accessibility.float(),
            ),
            "peak_distance": self._maybe_zero(
                "peak_distance",
                peak_distance.float(),
            ),
            "peak_mask": peak_mask.bool(),

            "tf_expression": self._maybe_zero(
                "tf_expression",
                tf_expression.float(),
            ),
            "tg_expression": self._maybe_zero(
                "tg_expression",
                tg_expression.float(),
            )
        }

        return item
        
def collate_tftg_edge_bags(batch):
    """
    Pads each TF-TG edge bag to the max number of cells in the batch.
    """

    labels = torch.stack([b["label"] for b in batch]).float()  # [E]

    max_cells = max(b["tf_expression"].shape[0] for b in batch)
    batch_size = len(batch)

    cell_mask = torch.zeros(batch_size, max_cells, dtype=torch.bool)

    output = {
        "label": labels,
        "cell_mask": cell_mask,
        "tf_name": [b["tf_name"] for b in batch],
        "tg_name": [b["tg_name"] for b in batch],
        "cell_ids": [b["cell_ids"] for b in batch],
        "tf_idx": torch.stack([b["tf_idx"] for b in batch]).long(),
        "tg_idx": torch.stack([b["tg_idx"] for b in batch]).long(),
    }

    tensor_keys = [
        "tf_embedding",
        "tf_mask",
        "peak_sequences",
        "peak_accessibility",
        "peak_distance",
        "peak_mask",
        "tf_expression",
        "tg_expression",
        "peak_indices",
    ]

    for key in tensor_keys:
        example = batch[0][key]
        padded_shape = (batch_size, max_cells) + tuple(example.shape[1:])
        padded = example.new_zeros(padded_shape)

        for i, b in enumerate(batch):
            n_cells = b[key].shape[0]
            padded[i, :n_cells] = b[key]

        output[key] = padded

    for i, b in enumerate(batch):
        n_cells = b["tf_expression"].shape[0]
        cell_mask[i, :n_cells] = True

    return output
        
        
@torch.no_grad()
def _move_batch_to_device(batch, device):
    moved = {
        "tf_embedding": batch["tf_embedding"].to(device),
        "tf_mask": batch["tf_mask"].to(device),
        "peak_sequences": batch["peak_sequences"].to(device),
        "peak_accessibility": batch["peak_accessibility"].to(device),
        "peak_distance": batch["peak_distance"].to(device),
        "tf_expression": batch["tf_expression"].to(device),
        "tg_expression": batch["tg_expression"].to(device),
        "label": batch["label"].to(device),
    }

    if "cell_mask" in batch:
        moved["cell_mask"] = batch["cell_mask"].to(device)

    if "peak_mask" in batch:
        moved["peak_mask"] = batch["peak_mask"].to(device)

    return moved


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    pooling_mode: str = "lse",
    pooling_temperature: float = 1.0,
):
    model.eval()
    total_loss = 0.0
    n_edges = 0
    for batch in loader:
        batch = _move_batch_to_device(batch, device)

        labels = batch["label"]
        cell_mask = batch["cell_mask"]
        E, C = cell_mask.shape

        edge_logits, _ = model.forward(
            tf_embedding=batch["tf_embedding"],
            tf_mask=batch["tf_mask"],
            peak_sequences=batch["peak_sequences"],
            peak_accessibility=batch["peak_accessibility"],
            peak_distance=batch["peak_distance"],
            tf_expression=batch["tf_expression"],
            tg_expression=batch["tg_expression"],
            peak_mask=batch.get("peak_mask", None),
            cell_mask=cell_mask,
            pooling_mode=pooling_mode,
            pooling_temperature=pooling_temperature,
        )
        loss = criterion(edge_logits, labels)
        total_loss += loss.item() * E
        n_edges += E
    return total_loss / max(n_edges, 1)

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
    pooling_mode: str = "lse",
    pooling_temperature: float = 1.0,
):
    model.eval()

    total_loss = 0.0
    n_edges = 0

    all_scores = []
    all_labels = []

    for batch in loader:
        batch = _move_batch_to_device(batch, device)

        labels = batch["label"]
        cell_mask = batch["cell_mask"]
        E, C = cell_mask.shape

        edge_logits, _ = model.forward(
            tf_embedding=batch["tf_embedding"],
            tf_mask=batch["tf_mask"],
            peak_sequences=batch["peak_sequences"],
            peak_accessibility=batch["peak_accessibility"],
            peak_distance=batch["peak_distance"],
            tf_expression=batch["tf_expression"],
            tg_expression=batch["tg_expression"],
            peak_mask=batch.get("peak_mask", None),
            cell_mask=cell_mask,
            pooling_mode=pooling_mode,
            pooling_temperature=pooling_temperature,
        )

        loss = criterion(edge_logits, labels)

        total_loss += loss.item() * E
        n_edges += E

        scores = torch.sigmoid(edge_logits)

        all_scores.append(scores.detach().cpu().numpy().ravel())
        all_labels.append(labels.detach().cpu().numpy().ravel())

    mean_loss = total_loss / max(n_edges, 1)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    metrics = compute_binary_classification_metrics(
        labels=all_labels,
        scores=all_scores,
        score_threshold=score_threshold,
        random_state=random_state,
    )

    metrics["loss"] = mean_loss
    metrics["score_min"] = float(all_scores.min())
    metrics["score_max"] = float(all_scores.max())
    metrics["score_mean"] = float(all_scores.mean())
    metrics["score_std"] = float(all_scores.std())
    metrics["n_pred_pos"] = int((all_scores >= score_threshold).sum())

    return metrics

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    pooling_mode: str = "lse",
    pooling_temperature: float = 1.0,
):
    model.train()
    total_loss = 0.0
    n_edges = 0
    pbar = tqdm(loader, desc="Training", ncols=100)

    for batch in pbar:
        batch = _move_batch_to_device(batch, device)

        labels = batch["label"]
        cell_mask = batch["cell_mask"]
        E, C = cell_mask.shape

        optimizer.zero_grad(set_to_none=True)
        edge_logits, _ = model.forward(
            tf_embedding=batch["tf_embedding"],
            tf_mask=batch["tf_mask"],
            peak_sequences=batch["peak_sequences"],
            peak_accessibility=batch["peak_accessibility"],
            peak_distance=batch["peak_distance"],
            tf_expression=batch["tf_expression"],
            tg_expression=batch["tg_expression"],
            peak_mask=batch.get("peak_mask", None),
            cell_mask=cell_mask,
            pooling_mode=pooling_mode,
            pooling_temperature=pooling_temperature,
        )
        loss = criterion(edge_logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * E
        n_edges += E
    return total_loss / max(n_edges, 1)

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
    pooling_mode: str = "lse",
    pooling_temperature: float = 1.0,
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
    n_train_edges = 0

    metric_rows = []
    global_step = global_step_start

    for batch_num, batch in enumerate(train_loader):
        if batch_num >= max_batches:
            break
        
        model.train()

        batch = _move_batch_to_device(batch, device)

        labels = batch["label"]
        cell_mask = batch["cell_mask"]
        E, C = cell_mask.shape

        optimizer.zero_grad(set_to_none=True)

        edge_logits, _ = model(
            tf_embedding=batch["tf_embedding"],
            tf_mask=batch["tf_mask"],
            peak_sequences=batch["peak_sequences"],
            peak_accessibility=batch["peak_accessibility"],
            peak_distance=batch["peak_distance"],
            tf_expression=batch["tf_expression"],
            tg_expression=batch["tg_expression"],
            cell_mask=cell_mask,
            peak_mask=batch.get("peak_mask", None),
            pooling_mode=pooling_mode,
            pooling_temperature=pooling_temperature,
        )

        loss = criterion(edge_logits, labels)

        loss.backward()
        optimizer.step()

        train_batch_loss = loss.item()

        total_train_loss += train_batch_loss * E
        n_train_edges += E

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
                "E": E,
                "C": C,
                "pooling_mode": pooling_mode,
                "pooling_temperature": pooling_temperature,
            }

            metric_rows.append(row)

            logging.info(
                f"Epoch {epoch:02d} | "
                f"batch {batch_num:04d} | "
                f"step {global_step:05d} | "
                f"E={E} | C={C} | "
                f"train_batch_loss={train_batch_loss:.4f} | "
                f"test_loss={test_metrics['loss']:.4f} | "
                f"AUROC={test_metrics['auroc']:.4f} | "
                f"AUPRC={test_metrics['auprc']:.4f} | "
                f"rand_AUROC={test_metrics['rand_auroc']:.4f} | "
                f"rand_AUPRC={test_metrics['rand_auprc']:.4f} | "
                f"pos_rate={test_metrics['n_pos'] / max(test_metrics['n_edges'], 1):.4f}"
            )

    mean_train_loss = total_train_loss / max(n_train_edges, 1)
    metrics_df = pd.DataFrame(metric_rows)

    return mean_train_loss, metrics_df, global_step

if __name__ == "__main__":
    
    """
    Need arguments for:
    
    sample_pairs: int | None
    max_peaks_per_tg: int
    max_cells_per_pair: int | None
    batch_size: int
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPU devices to use for training")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use for training")
    parser.add_argument("--run_name", type=str, help="Name of the training run (for logging/checkpointing)")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--sample_pairs", type=int, default=None, help="Number of TF-TG pairs to sample for training (default: use all)")
    parser.add_argument("--max_peaks_per_tg", type=int, default=64, help="Maximum number of peaks to consider per TG (default: 64)")
    parser.add_argument("--max_cells_per_pair", type=int, default=8, help="Maximum number of cells to sample per TF-TG pair (default: 8)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--pct_true_edges", type=float, default=0.15, help="Percentage of true edges to include in the training set (default: 0.15)")
    parser.add_argument("--true_false_ratio", type=float, default=2.0, help="Ratio of true to false edges in the training set (default: 2.0)")
    parser.add_argument("--peak_flank_size", type=int, default=64)
    parser.add_argument("--tf_bind_model_path", type=str, required=False, help="Path to the TF→DNA model checkpoint to initialize from (if not using default)")
    parser.add_argument("--training_data_dir", type=str, required=False, help="Path to directory containing training data cache files (if not using default)")
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to a model checkpoint to resume training from")
    parser.add_argument("--force_reload", action="store_true", help="Whether to force reload cached data instead of using existing cache files")
    args = parser.parse_args()

    gene_ref_file = DATA_DIR / "genome_data" / "genome_annotation" / "mm10" / "Mus_musculus.GRCm39.115.gtf.gz"
    genome_fasta_path = DATA_DIR / "genome_data" / "reference_genome" / "mm10" / "mm10.fa"
    chrom_sizes_path = DATA_DIR / "genome_data" / "reference_genome" / "mm10" / "mm10.chrom.sizes"
    
    epochs = args.epochs
    num_gpus = args.num_gpus
    num_nodes = args.num_nodes
    output_dir = args.output_dir
    run_name = args.run_name
    checkpoint_path = args.checkpoint_path
    force_reload = args.force_reload
    batch_size = args.batch_size
    sample_pairs = args.sample_pairs
    max_peaks_per_tg = args.max_peaks_per_tg
    max_cells_per_pair = args.max_cells_per_pair
    pct_true_edges = args.pct_true_edges
    true_false_ratio = args.true_false_ratio
    training_data_dir = args.training_data_dir
    tf_bind_model_path = Path(args.tf_bind_model_path)
    peak_flank_size = args.peak_flank_size
    
    prefetch_factor = 4
    num_workers_per_dataloader = 6

    if training_data_dir:
        training_cache_dir = Path(training_data_dir)
    else:
        training_cache_dir = PROJECT_DIR / "data" / "training_data_cache"
    training_cache_dir.mkdir(exist_ok=True, parents=True)
    
    tf_tg_input_cache_dir = training_cache_dir / "tf_tg_training_data_cache"
    
    # Load the compact split inputs
    tftg_inputs_train = torch.load(tf_tg_input_cache_dir / "tftg_inputs_train.pt")
    tftg_inputs_val = torch.load(tf_tg_input_cache_dir / "tftg_inputs_val.pt")
    tftg_inputs_test = torch.load(tf_tg_input_cache_dir / "tftg_inputs_test.pt")

    # Load the lookup tensors
    tf_embeddings_tensor = torch.load(training_cache_dir / "tf_embeddings.pt")
    tf_mask_tensor = torch.load(training_cache_dir / "tf_masks.pt")
    atac_peak_tensor = torch.load(tf_tg_input_cache_dir / "atac_peak_tensor.pt")

    # Load the metadata
    with open(tf_tg_input_cache_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
        
    tf_name_to_idx = metadata["tf_name_to_idx"]
    tg_id_to_idx = metadata["tg_id_to_idx"]

    # Load the manifest and verify tensor shapes and dtypes match expectations
    with open(tf_tg_input_cache_dir / "manifest.json") as f:
        manifest = json.load(f)

    print(json.dumps(manifest, indent=2))

    assert tuple(manifest["atac_peak_tensor_shape"]) == tuple(atac_peak_tensor.shape)
    assert manifest["atac_peak_tensor_dtype"] == str(atac_peak_tensor.dtype)
    
    # Re-create the datasets and dataloaders using the loaded compact inputs and lookup tensors
    train_dataset = TFTGEdgeBagDataset(
        tftg_inputs_train,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor,
        zero_fields=None,
        strict=True,
    )

    val_dataset = TFTGEdgeBagDataset(
        tftg_inputs_val,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor,
        zero_fields=None,
        strict=True,
    )

    test_dataset = TFTGEdgeBagDataset(
        tftg_inputs_test,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor,
        zero_fields=None,
        strict=True,
    )

    if atac_peak_tensor.dtype == torch.uint8:
        atac_peak_tensor = atac_peak_tensor.float()

    # Create the DataLoaders with the custom collate function for batching edge bags
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_per_dataloader,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_tftg_edge_bags,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_per_dataloader,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_tftg_edge_bags,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_per_dataloader,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_tftg_edge_bags,
        )

    logging.info(f"Train/Val/Test sizes: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

    tf_tg_model = create_new_tf_tg_binding_model(tf_bind_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf_tg_model = tf_tg_model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    score_threshold = 0.5
    pooling_mode = "lse"
    pooling_temperature = 1.0

    epoch_rows = []

    def metrics_to_row(
        metrics,
        epoch,
        split,
        train_loss=np.nan,
    ):
        pos_rate = metrics["n_pos"] / max(metrics["n_edges"], 1)

        return {
            "epoch": epoch,
            "split": split,
            "train_loss": train_loss,
            "loss": metrics["loss"],
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "rand_auroc": metrics["rand_auroc"],
            "rand_auprc": metrics["rand_auprc"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "n_edges": metrics["n_edges"],
            "n_pos": metrics["n_pos"],
            "n_neg": metrics["n_neg"],
            "pos_rate": pos_rate,
            "score_threshold": metrics["score_threshold"],
            "pooling_mode": pooling_mode,
            "pooling_temperature": pooling_temperature,
        }

    logging.info("\nStarting Lightning training...")

    lit_model = tf_to_tg_module.LitTFPeakBindingModel(
        model=tf_tg_model,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
        pooling_mode=pooling_mode,
        pooling_temperature=pooling_temperature,
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="epoch={epoch:02d}-val_auroc={val/auroc:.4f}-val_loss={val/loss:.4f}",
        monitor="val/auroc",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=10,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    wandb_logger = WandbLogger(
        project="tf_tg_regulation_prediction",
        name=run_name,
        save_dir=output_dir,
    )

    wandb_logger.log_hyperparams({
        "epochs": epochs,
        "batch_size": batch_size,
        "sample_pairs": sample_pairs,
        "max_peaks_per_tg": max_peaks_per_tg,
        "max_cells_per_pair": max_cells_per_pair,
        "pct_true_edges": pct_true_edges,
        "true_false_ratio": true_false_ratio,
        "pooling_mode": pooling_mode,
        "pooling_temperature": pooling_temperature,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "flank_size": peak_flank_size,
        "max_precompute_peaks": max_peaks_per_tg,
        "num_workers": num_workers_per_dataloader,
        "prefetch_factor": prefetch_factor,
        "persistent_workers": True,
    })
    
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            os.environ.get("SLURM_NTASKS", "1"),
        )
    )

    use_ddp = world_size > 1
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_gpus,
        num_nodes=num_nodes,
        strategy="ddp" if use_ddp else "auto",
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=100),
            checkpoint_callback,
            early_stopping_callback,
            lr_monitor,
        ],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
        default_root_dir=output_dir,
        enable_progress_bar=True,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )
    
    torch.set_float32_matmul_precision('medium')

    if checkpoint_path is not None:
        logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.fit(
            lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            tf_bind_model_path=checkpoint_path,
        )
    else:
        trainer.fit(
            lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

    tf_tg_model = lit_model.model
    tf_tg_model = tf_tg_model.to(device)

    final_train_metrics = evaluate_with_metrics(
        model=tf_tg_model,
        loader=train_loader,
        criterion=criterion,
        device=device,
        score_threshold=score_threshold,
        random_state=42 + epochs,
        pooling_mode=pooling_mode,
        pooling_temperature=pooling_temperature,
    )

    final_val_metrics = evaluate_with_metrics(
        model=tf_tg_model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        score_threshold=score_threshold,
        random_state=10_000 + epochs,
        pooling_mode=pooling_mode,
        pooling_temperature=pooling_temperature,
    )

    final_test_metrics = evaluate_with_metrics(
        model=tf_tg_model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        score_threshold=score_threshold,
        random_state=20_000 + epochs,
        pooling_mode=pooling_mode,
        pooling_temperature=pooling_temperature,
    )

    epoch_rows.append(
        metrics_to_row(
            metrics=final_train_metrics,
            epoch=epochs,
            split="train",
            train_loss=np.nan,
        )
    )

    epoch_rows.append(
        metrics_to_row(
            metrics=final_val_metrics,
            epoch=epochs,
            split="val",
            train_loss=np.nan,
        )
    )

    epoch_rows.append(
        metrics_to_row(
            metrics=final_test_metrics,
            epoch=epochs,
            split="test",
            train_loss=np.nan,
        )
    )

    epoch_metric_df = pd.DataFrame(epoch_rows)

    epoch_metric_path = PROJECT_DIR / "testing_results" / "metrics_per_epoch.csv"
    epoch_metric_path.parent.mkdir(parents=True, exist_ok=True)

    epoch_metric_df.to_csv(epoch_metric_path, index=False)
    
