import itertools
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score
)
import torch
from tqdm import tqdm

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
sys.path.append(str(PROJECT_DIR))

import models.tf_to_tg as tf_to_tg_module


def grn_edge_set(df, source_col: str = "Source", target_col: str = "Target"):
    """Unique upper-cased (source, target) edges in a GRN, ignoring scores."""
    return set(zip(df[source_col].str.upper(), df[target_col].str.upper()))


def calculate_universe_jaccard(df_x, df_y):
    """Jaccard of two GRNs' full edge sets, with no score cut applied.

    This is the ceiling diagnostic for any top-k Jaccard computed on the same pair.
    A value of 1.0 means both runs scored exactly the same TF-TG pairs, so a top-k
    Jaccard between them reflects ranking disagreement alone. Anything below 1.0
    means the two runs scored partly different edges, and edges present in only one
    run can never intersect -- so part of the top-k non-overlap is missing edges
    rather than unstable rankings.
    """
    edges_x, edges_y = grn_edge_set(df_x), grn_edge_set(df_y)
    union = edges_x | edges_y

    return len(edges_x & edges_y) / len(union) if union else np.nan


def summarize_universe_overlap(score_dfs_by_subsample, method_name, sample_name=None):
    """Collapse one group's per-subsample GRNs into a single edge-universe summary row.

    `score_dfs_by_subsample` maps subsample_num -> labeled score df, i.e. one value
    of the dicts built by the stability GRN loaders.
    """
    edge_sets = {num: grn_edge_set(df) for num, df in score_dfs_by_subsample.items()}

    if not edge_sets:
        return None

    pairwise = [
        calculate_universe_jaccard(score_dfs_by_subsample[num_x], score_dfs_by_subsample[num_y])
        for num_x, num_y in itertools.combinations(sorted(edge_sets), 2)
    ]

    return {
        "method_name": method_name,
        "sample_name": sample_name,
        "n_subsamples": len(edge_sets),
        "mean_universe_jaccard": float(np.mean(pairwise)) if pairwise else np.nan,
        "min_universe_jaccard": float(np.min(pairwise)) if pairwise else np.nan,
        "min_edges": min(len(edges) for edges in edge_sets.values()),
        "max_edges": max(len(edges) for edges in edge_sets.values()),
        "core_edges_all_subsamples": len(set.intersection(*edge_sets.values())),
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
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    preds_sorted_indices = np.argsort(scores)[::-1]
    preds_sorted = preds[preds_sorted_indices]
    labels_sorted = labels[preds_sorted_indices]
    
    early_precision = precision_score(labels_sorted[:10_000], preds_sorted[:10_000], zero_division=0)

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
        "early_precision": early_precision,
        "recall": recall,
        "f1": f1,
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

    for batch in tqdm(loader, desc="Evaluating"):
        batch = tf_to_tg_module.move_batch_to_device(batch, device)

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