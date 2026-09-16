#!/usr/bin/env python
"""Train cell-type-specific TF->TG edge bags on one GPU (or CPU).

Defaults follow the liver notebook, using every eligible edge. TF-DNA binding
scores are cached per split and invalidated when their exact inputs change.
Example: python scripts/train_tf_to_tg_celltype_model.py --wandb_mode offline
"""
import argparse
import hashlib
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from models.tf_to_tg_celltype import LitTFTGRegulationModel


class TwoPercentProgressBar(TQDMProgressBar):
    """Refresh each phase at most 50 times, plus its final completion update."""

    def _should_update(self, current, total):
        if not self.is_enabled:
            return False
        if not math.isfinite(total):
            return False
        interval = max(1, math.ceil(total / 50))
        return current % interval == 0 or current == total

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        bar = self.train_progress_bar
        current = batch_idx + 1
        if bar is not None and self._should_update(current, bar.total):
            # Lightning normally refreshes once for the count and again for metrics.
            bar.set_postfix(self.get_metrics(trainer, pl_module), refresh=False)
            bar.n = current
            bar.refresh()

def _format_chroms(chroms: list[str]) -> str:
    """Render a chromosome list for logging.

    Only for log messages -- the splits themselves are made with isin() on the full list.
    min()/max() would compare these as strings, where "9" > "15", so ["1".."15"] printed
    as "1-15" came out as "1-9". Numeric labels are ordered numerically and collapsed to
    a range only when they are actually contiguous.
    """
    numeric = sorted((c for c in chroms if str(c).isdigit()), key=int)
    other = sorted(str(c) for c in chroms if not str(c).isdigit())

    parts = []
    if numeric:
        contiguous = int(numeric[-1]) - int(numeric[0]) == len(numeric) - 1
        parts.append(f"{numeric[0]}-{numeric[-1]}" if contiguous and len(numeric) > 1
                     else ", ".join(numeric))
    parts.extend(other)
    return ", ".join(parts)

def split_genes_by_chromosome(
    gene_reference_file: Path,
    train_chroms: list[str] = None,
    val_chroms: list[str] = None,
    test_chroms: list[str] = None
    ):
    logging.info(f"Splitting genes into train/val/test based on chromosome:")
    import gtfparse
    gene_ref_df = gtfparse.read_gtf(gene_reference_file, result_type="pandas")

    gene_chrom: pd.DataFrame = gene_ref_df[["seqname", "gene_name"]].rename(
        columns={"seqname": "chrom", "gene_name": "TG"}
    )
    
    gene_chrom["chrom"] = gene_chrom["chrom"].astype(str).str.replace("^chr", "", regex=True)
    gene_chrom["TG"] = gene_chrom["TG"].str.upper()
    
    train_genes = gene_chrom[gene_chrom["chrom"].isin(train_chroms)][
        "TG"
    ].unique()
    logging.info(f"  - Train set: {len(train_genes):,} genes (chroms {_format_chroms(train_chroms)})")

    val_genes = gene_chrom[gene_chrom["chrom"].isin(val_chroms)][
        "TG"
    ].unique()
    logging.info(f"  - Validation set: {len(val_genes):,} genes (chroms {_format_chroms(val_chroms)})")

    test_genes = gene_chrom[gene_chrom["chrom"].isin(test_chroms)]["TG"].unique()
    logging.info(f"  - Test set: {len(test_genes):,} genes (chroms {_format_chroms(test_chroms)})")

    return train_genes, val_genes, test_genes

def create_axis_matched_edges(edge_df, tf_col="Source", item_col="Target",
                              axis="tg", true_false_ratio=1.0, seed=123):
    """Negatives drawn inside each level of one axis, matching that marginal exactly.

    axis="tg": for each target gene, negatives are TFs not linked to it. Every
        TG-level feature -- peak bag, TSS distance, target expression -- then carries
        zero information about the label.
    axis="tf": the mirror, for each TF.

    It cannot do both. Matching one marginal SHARPENS the imbalance on the other: a
    TF linked to 87% of genes is almost never drawable as a negative under axis="tg",
    so its per-TF label rate goes to 0.98. Use create_degree_matched_edges when both
    axes must be balanced.
    """
    rng = np.random.default_rng(seed)

    df = (edge_df[[tf_col, item_col]].dropna().astype(str)
          .drop_duplicates().reset_index(drop=True))

    group_col, other_col = (item_col, tf_col) if axis == "tg" else (tf_col, item_col)
    as_pair = (lambda k, o: (o, k)) if axis == "tg" else (lambda k, o: (k, o))

    all_others = np.array(sorted(df[other_col].unique()))
    true_edges, false_edges, short = set(), set(), 0

    for key, group in df.groupby(group_col):
        positives = set(group[other_col])
        true_edges |= {as_pair(key, o) for o in positives}

        available = all_others[~np.isin(all_others, list(positives))]
        wanted = round(len(positives) * true_false_ratio)
        n_drawn = min(wanted, len(available))
        short += n_drawn < wanted
        if n_drawn:
            drawn = rng.choice(available, size=n_drawn, replace=False)
            false_edges |= {as_pair(key, o) for o in drawn}

    logging.info(
        f"{axis.upper()}-matched: {len(true_edges):,} positive, {len(false_edges):,} "
        f"negative (ratio {len(false_edges) / max(len(true_edges), 1):.2f}, "
        f"requested {true_false_ratio}); {short:,} / {df[group_col].nunique():,} "
        f"levels could not reach it"
    )
    return true_edges, false_edges

def create_balanced_edges(edge_df, *, balance_tf, balance_tg,
                          tf_col="Source", item_col="Target",
                          pct_true_edges=1.0, true_false_ratio=1.0, seed=123):
    """Dispatch to the sampler matching the requested balance.

    both  -> rectangles: exact on TF and TG, at the cost of using a subset of the GT
    tg    -> per-TG negatives; TF marginal is left skewed and usually worse
    tf    -> per-TF negatives; TG marginal is left skewed
    none  -> uniform over the TF x TG box; both identities predict the label
    """
    if balance_tf and balance_tg:
        return create_degree_matched_edges(edge_df, tf_col, item_col, seed=seed)
    if balance_tf or balance_tg:
        return create_axis_matched_edges(
            edge_df, tf_col, item_col,
            axis="tf" if balance_tf else "tg",
            true_false_ratio=true_false_ratio, seed=seed,
        )
    return create_true_false_edges_from_full_universe(
        edge_df, tf_col, item_col,
        pct_true_edges=pct_true_edges,
        true_false_ratio=true_false_ratio, seed=seed,
    )

def create_degree_matched_edges(edge_df, tf_col="Source", item_col="Target",
                                seed=123, max_partners=32):
    """Negatives that preserve BOTH the TF out-degree and the TG in-degree.

    Pairs of GT edges (t1,g1),(t2,g2) whose crossed pairs (t1,g2),(t2,g1) are both
    non-edges contribute two positives and two negatives. Each of t1, t2, g1, g2 then
    gains exactly one positive and one negative, so per-TF and per-TG label rates are
    0.5 by construction and neither identity predicts the label.

    A rectangle is all-or-nothing. Accepting one whose positives were already used
    would add two negatives and fewer than two positives, which is what breaks the
    balance -- so every one of the four pairs must be unused.

    Each GT edge is consumed at most once, so this walks a shuffled edge list and
    looks for a partner, rather than sampling random pairs and rejecting: as the
    unused pool shrinks, random pairing almost always collides.
    """
    rng = np.random.default_rng(seed)

    df = (edge_df[[tf_col, item_col]].dropna().astype(str)
          .drop_duplicates().reset_index(drop=True))
    gt = set(zip(df[tf_col], df[item_col]))

    pool = list(gt)
    rng.shuffle(pool)
    available = set(pool)

    true_edges, false_edges = set(), set()

    for edge in pool:
        if edge not in available:
            continue
        t1, g1 = edge

        partners = rng.choice(len(pool), size=min(max_partners, len(pool)), replace=False)
        for p in partners:
            other = pool[p]
            if other not in available or other == edge:
                continue
            t2, g2 = other
            if t1 == t2 or g1 == g2:
                continue
            if (t1, g2) in gt or (t2, g1) in gt:
                continue
            if (t1, g2) in false_edges or (t2, g1) in false_edges:
                continue

            true_edges |= {edge, other}
            false_edges |= {(t1, g2), (t2, g1)}
            available.discard(edge)
            available.discard(other)
            break

    logging.info(
        f"Degree-matched sampling: {len(true_edges):,} positive, {len(false_edges):,} "
        f"negative ({len(true_edges) / max(len(gt), 1):.1%} of the GT used)"
    )
    return true_edges, false_edges

def create_true_false_edges_from_full_universe(
    edge_df: pd.DataFrame,
    tf_col: str = "Source",
    item_col: str = "Target",
    pct_true_edges: float | None = 1.0,
    true_false_ratio: float = 1.0,
    seed: int = 123,
):
    df_all = edge_df[[tf_col, item_col]].copy()

    df_all = df_all.dropna(subset=[tf_col, item_col])

    df_all[tf_col] = df_all[tf_col].astype(str)
    df_all[item_col] = df_all[item_col].astype(str)

    df_all = df_all.drop_duplicates([tf_col, item_col]).reset_index(drop=True)

    if df_all.empty:
        raise ValueError(
            f"No edges remain after filtering by tf_names using columns "
            f"{tf_col!r} and {item_col!r}."
        )

    candidate_tfs = sorted(df_all[tf_col].unique())
    candidate_items = sorted(df_all[item_col].unique())

    gt_pairs = set(zip(df_all[tf_col], df_all[item_col]))

    full_universe = (
        pd.MultiIndex
        .from_product([candidate_tfs, candidate_items], names=[tf_col, item_col])
        .to_frame(index=False)
    )

    full_universe["_pair"] = list(zip(full_universe[tf_col], full_universe[item_col]))
    full_universe["_in_gt"] = full_universe["_pair"].isin(gt_pairs)

    true_df = full_universe[full_universe["_in_gt"]].copy()
    false_df = full_universe[~full_universe["_in_gt"]].copy()

    if pct_true_edges is not None:
        if not (0 < pct_true_edges <= 1):
            raise ValueError("pct_true_edges must be in (0, 1] or None.")

        true_df = true_df.sample(frac=pct_true_edges, random_state=seed)

    n_false = round(len(true_df) * true_false_ratio)

    if n_false > len(false_df):
        logging.warning(
            f"Requested {n_false:,} false edges, but only {len(false_df):,} are available. "
            "Using all available false edges."
        )
        n_false = len(false_df)

    false_df = false_df.sample(n=n_false, random_state=seed)

    true_edges = set(zip(true_df[tf_col], true_df[item_col]))
    false_edges = set(zip(false_df[tf_col], false_df[item_col]))

    return true_edges, false_edges

def create_labeled_tf_tg_dataset(
    true_interactions: set[tuple[str, str]],
    false_interactions: set[tuple[str, str]],
    tf_name_to_idx: dict[str, int],
    drop_missing: bool = True,
) -> pd.DataFrame:
    # sorted(), not bare set iteration: set order over string tuples depends on
    # PYTHONHASHSEED, so without this the col order -- and therefore anything
    # downstream that indexes by position, e.g. df.sample(n=...) -- differs in every
    # process even with a fixed random_state.
    rows = []
    for tf, tg in sorted(true_interactions):
        rows.append((tf, tg, 1))
    for tf, tg in sorted(false_interactions):
        rows.append((tf, tg, 0))

    df = pd.DataFrame(rows, columns=["tf_name", "tg_id", "label"])
    df["tf_embedding_idx"] = df["tf_name"].map(tf_name_to_idx)

    missing_mask = df["tf_embedding_idx"].isna()
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

    df["tf_embedding_idx"] = df["tf_embedding_idx"].astype(np.int64)
    df["label"] = df["label"].astype(np.bool_)

    return df.sample(frac=1.0, random_state=123).reset_index(drop=True)

def _create_labeled_df(
    gt_df: pd.DataFrame,
    seed: int = 123,
    *,
    tf_name_to_idx,
    tg_id_to_idx,
    balance_tf: bool = True,
    balance_tg: bool = True,
    pct_true_edges: float = 1.0,
    true_false_ratio: float = 1.0,
):
    gt_df = gt_df[
        gt_df["Source"].isin(tf_name_to_idx.keys()) &
        gt_df["Target"].isin(tg_id_to_idx.keys())
    ].copy()

    true_edges, false_edges = create_balanced_edges(
        gt_df, balance_tf=balance_tf, balance_tg=balance_tg,
        pct_true_edges=pct_true_edges, true_false_ratio=true_false_ratio, seed=seed,
    )

    return create_labeled_tf_tg_dataset(
        true_interactions=true_edges,
        false_interactions=false_edges,
        tf_name_to_idx=tf_name_to_idx,
        drop_missing=False,
    )

class TFTGEdgeBagDataset(Dataset):
    def __init__(
        self,
        inputs,  # Labeled edge DataFrame
        *,
        tf_embeddings_tensor,
        tf_mask_tensor,
        atac_peak_tensor,
        atac_mat,
        rna_mat,
        cell_pools,
        max_peaks_per_tg=25,
        resample_max_cells_per_pair=64,
        resample_cells=True,
        binding_scores=None,   # [len(inputs), P], precomputed; skips peak_sequences
        seed=42,
    ):
        self.inputs = inputs.reset_index(drop=True).copy()

        self.tf_embeddings_tensor = tf_embeddings_tensor
        self.tf_mask_tensor = tf_mask_tensor
        self.atac_peak_tensor = atac_peak_tensor

        # Keep sparse matrices sparse
        self.atac_mat = atac_mat.tocsr() if sparse.issparse(atac_mat) else atac_mat
        self.rna_mat = rna_mat.tocsr() if sparse.issparse(rna_mat) else rna_mat

        self.cell_pools = {k: np.asarray(v, dtype=np.int64) for k, v in cell_pools.items()}
        self.C = int(resample_max_cells_per_pair)
        self.P = int(max_peaks_per_tg)
        self.resample_cells = resample_cells
        self.seed = int(seed)
        self._rng = None
        self.binding_scores = binding_scores
        
        if binding_scores is not None and len(binding_scores) != len(self.inputs):
            raise ValueError(
                f"binding_scores has {len(binding_scores)} rows but inputs has "
                f"{len(self.inputs)}. These must be built from the same frame."
            )

        if self.C <= 0 or self.P <= 0:
            raise ValueError("Cell and peak limits must be positive.")

        if self.rna_mat.shape[0] != self.atac_mat.shape[0]:
            raise ValueError("RNA and ATAC must have aligned cell rows.")
        
        for key, pool in self.cell_pools.items():
            if pool.ndim != 1 or len(pool) == 0:
                raise ValueError(f"cell pool {key} must be a nonempty 1D array.")
            if len(np.unique(pool)) != len(pool):
                raise ValueError(f"cell pool {key} contains duplicate positions.")
            if pool.min() < 0 or pool.max() >= self.rna_mat.shape[0]:
                raise ValueError(f"cell pool {key} has out-of-range positions.")

        missing = {
            (s, c) for s, c in zip(self.inputs.sample_id, self.inputs.cell_type)
        } - set(self.cell_pools)
        if missing:
            raise ValueError(f"No cell pool for {sorted(missing)[:5]}")

        if self.atac_peak_tensor.shape[0] != self.atac_mat.shape[1]:
            raise ValueError(
                "Peak sequences and ATAC columns must use the same peak order."
            )

    def __len__(self):
        return len(self.inputs)

    @staticmethod
    def _gather(matrix, rows, cols):
        """Gather a cells × features slice as a float tensor."""
        if torch.is_tensor(matrix):
            rows = torch.as_tensor(rows, dtype=torch.long, device=matrix.device)
            cols = torch.as_tensor(cols, dtype=torch.long, device=matrix.device)
            return matrix.index_select(0, rows).index_select(1, cols).float()

        block = matrix[rows][:, cols]
        if sparse.issparse(block):
            block = block.toarray()

        return torch.as_tensor(np.asarray(block), dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.inputs.iloc[idx]
        tf_idx = int(row.tf_embedding_idx)

        # These lists must already be sorted/capped together during preparation.
        peak_cols = np.asarray(row.peak_atac_cols, dtype=np.int64)
        distances = np.asarray(row.peak_distances, dtype=np.float32)

        n_peaks = len(peak_cols)
        if len(distances) != n_peaks or not 0 <= n_peaks <= self.P:
            raise ValueError(f"Invalid peak bag for edge {idx}.")

        if np.any(peak_cols < 0) or np.any(peak_cols >= self.atac_mat.shape[1]):
            raise ValueError(f"Invalid ATAC column for edge {idx}.")

        # Training: fresh draws. Validation/test: fixed draw for each edge.
        if self.resample_cells:
            if self._rng is None:
                self._rng = np.random.default_rng(
                    np.random.SeedSequence([self.seed, torch.initial_seed()])
                )
            rng = self._rng
        else:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, int(idx)])
            )

        pool = self.cell_pools[(row.sample_id, row.cell_type)]
        n_cells = min(self.C, len(pool))
        cell_rows = rng.choice(pool, size=n_cells, replace=False)

        # Fixed-size cell bag.
        cell_indices = torch.full((self.C,), -1, dtype=torch.long)
        cell_indices[:n_cells] = torch.from_numpy(cell_rows)

        cell_mask = torch.zeros(self.C, dtype=torch.bool)
        cell_mask[:n_cells] = True

        # Fixed-size peak bag.
        peak_indices = torch.full((self.P,), -1, dtype=torch.long)
        peak_indices[:n_peaks] = torch.from_numpy(peak_cols)

        peak_mask = torch.zeros(self.P, dtype=torch.bool)
        peak_mask[:n_peaks] = True

        peak_distance = torch.zeros(self.P, dtype=torch.float32)
        peak_distance[:n_peaks] = torch.from_numpy(distances)

        # Only gather real sequences; never index padding with -1.
        # These remain zero-filled when no candidate peaks exist.
        peak_sequences = None if self.binding_scores is not None else torch.zeros(
            (self.P, *self.atac_peak_tensor.shape[1:]),
            dtype=torch.float32,
        )
        peak_accessibility = torch.zeros(self.C, self.P)

        if n_peaks > 0:
            if peak_sequences is not None:
                peak_sequences[:n_peaks] = self.atac_peak_tensor[
                    torch.from_numpy(peak_cols)
                ].float()

            peak_accessibility[:n_cells, :n_peaks] = self._gather(
                self.atac_mat, cell_rows, peak_cols
            )

        tf_expression = torch.zeros(self.C)
        tf_expression[:n_cells] = self._gather(
            self.rna_mat, cell_rows, [int(row.tf_rna_col)]
        ).flatten()

        tg_expression = torch.zeros(self.C)
        tg_expression[:n_cells] = self._gather(
            self.rna_mat, cell_rows, [int(row.tg_rna_col)]
        ).flatten()

        item = {
            "sample_id": row.sample_id,
            "cell_type": row.cell_type,
            "tf_name": row.tf_name,
            "tg_id": row.tg_id,
            "label": torch.tensor(float(row.label)),
            "tf_idx": torch.tensor(tf_idx, dtype=torch.long),
            "cell_indices": cell_indices,
            "cell_mask": cell_mask,
            "peak_indices": peak_indices,
            "peak_distance": peak_distance,
            "peak_mask": peak_mask,
            "peak_accessibility": peak_accessibility,
            "tf_expression": tf_expression,
            "tg_expression": tg_expression,
        }
        
        if self.binding_scores is not None:
            item["binding_score"] = self.binding_scores[idx]
        else:
            item["peak_sequences"] = peak_sequences
            item["tf_embedding"] = self.tf_embeddings_tensor[tf_idx].float()
            item["tf_mask"] = self.tf_mask_tensor[tf_idx].bool()

        return item

@torch.no_grad()
def precompute_binding_scores(
    labeled_df,
    peak_model,
    *,
    tf_embeddings_tensor,
    tf_mask_tensor,
    atac_peak_tensor,
    max_peaks_per_tg,
    device,
    chunk_size=1024,
):
    """Binding score for every (edge, peak slot) in labeled_df, as [n_edges, P] float32.

    Rows align with labeled_df's positional order, which is what the dataset uses
    after reset_index(drop=True). Distinct (tf_idx, peak_col) pairs are computed
    once and scattered, so a pair shared by several edges costs one forward pass.
    """
    n_edges = len(labeled_df)
    out = torch.zeros(n_edges, max_peaks_per_tg, dtype=torch.float32)

    # Flatten every occupied slot into parallel arrays.
    edge_rows, slots, tf_ids, peak_cols = [], [], [], []
    for row_position, (_, row) in enumerate(labeled_df.iterrows()):
        cols = row.peak_atac_cols
        tf_id = int(row.tf_embedding_idx)
        for slot, peak_col in enumerate(cols[:max_peaks_per_tg]):
            edge_rows.append(row_position)
            slots.append(slot)
            tf_ids.append(tf_id)
            peak_cols.append(int(peak_col))

    if not edge_rows:
        return out

    edge_rows = np.asarray(edge_rows)
    slots = np.asarray(slots)
    pairs = np.stack([np.asarray(tf_ids), np.asarray(peak_cols)], axis=1)

    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
    unique_scores = torch.zeros(len(unique_pairs), dtype=torch.float32)

    was_training = peak_model.training
    peak_model.eval()
    try:
        chunks = range(0, len(unique_pairs), chunk_size)
        for start in tqdm(chunks, desc="Binding scores",
                          miniters=max(1, math.ceil(len(chunks) / 50)),
                          maxinterval=float("inf")):
            chunk = unique_pairs[start:start + chunk_size]
            tf_chunk = torch.from_numpy(chunk[:, 0]).long()
            peak_chunk = torch.from_numpy(chunk[:, 1]).long()

            logits = peak_model(
                tf_embedding=tf_embeddings_tensor[tf_chunk].float().to(device),
                tf_mask=tf_mask_tensor[tf_chunk].bool().to(device),
                peak_embedding=atac_peak_tensor[peak_chunk].float().to(device),
            )
            unique_scores[start:start + len(chunk)] = logits.reshape(-1).sigmoid().cpu()
    finally:
        peak_model.train(was_training)

    out[edge_rows, slots] = unique_scores[inverse]

    print(f"{len(unique_pairs):,} distinct (TF, peak) pairs for "
          f"{len(edge_rows):,} slots across {n_edges:,} edges "
          f"({out.numel() * 4 / 1e6:.1f} MB)")
    return out


def binding_score_cache_path(
    prepared_dir,
    split_name,
    labeled_df,
    *,
    tf_dna_checkpoint,
    max_peaks_per_tg,
):
    """Return a cache path tied to every input that selects a binding score.

    Binding depends on the TF-DNA checkpoint and, for each positional row, its TF
    embedding index and capped peak-column list. Including row order is essential:
    TFTGEdgeBagDataset indexes the score tensor by DataFrame position.
    """
    digest = hashlib.sha1()
    checkpoint = Path(tf_dna_checkpoint).resolve()
    checkpoint_stat = checkpoint.stat()
    digest.update(str(checkpoint).encode())
    digest.update(f"{checkpoint_stat.st_size}:{checkpoint_stat.st_mtime_ns}".encode())
    digest.update(f"P={max_peaks_per_tg};N={len(labeled_df)}".encode())

    for row in labeled_df[["tf_embedding_idx", "peak_atac_cols"]].itertuples(index=False):
        peak_cols = np.asarray(row.peak_atac_cols[:max_peaks_per_tg], dtype=np.int64)
        header = np.asarray(
            [int(row.tf_embedding_idx), len(peak_cols)], dtype=np.int64
        )
        digest.update(header.tobytes())
        digest.update(peak_cols.tobytes())

    cache_key = digest.hexdigest()[:16]
    return Path(prepared_dir) / f"binding_scores_{split_name}_{cache_key}.pt"


def load_or_precompute_binding_scores(
    split_name,
    labeled_df,
    peak_model,
    *,
    prepared_dir,
    tf_dna_checkpoint,
    tf_embeddings_tensor,
    tf_mask_tensor,
    atac_peak_tensor,
    max_peaks_per_tg,
    device,
    chunk_size,
):
    """Load a valid split cache or compute and atomically save it."""
    expected_shape = (len(labeled_df), max_peaks_per_tg)
    cache_path = binding_score_cache_path(
        prepared_dir,
        split_name,
        labeled_df,
        tf_dna_checkpoint=tf_dna_checkpoint,
        max_peaks_per_tg=max_peaks_per_tg,
    )

    if cache_path.exists():
        try:
            scores = torch.load(cache_path, map_location="cpu", weights_only=True)
            cache_is_valid = (
                torch.is_tensor(scores)
                and tuple(scores.shape) == expected_shape
                and scores.dtype == torch.float32
                and bool(torch.isfinite(scores).all())
            )
            if cache_is_valid:
                logging.info(
                    "%s: loaded cached binding scores %s from %s",
                    split_name, tuple(scores.shape), cache_path,
                )
                return scores, cache_path
            logging.warning(
                "%s: ignoring binding cache with shape %s; expected %s",
                split_name, getattr(scores, "shape", None), expected_shape,
            )
        except Exception as error:
            logging.warning(
                "%s: could not load binding cache %s (%s); rebuilding",
                split_name, cache_path, error,
            )

    scores = precompute_binding_scores(
        labeled_df,
        peak_model,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor,
        max_peaks_per_tg=max_peaks_per_tg,
        device=device,
        chunk_size=chunk_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp-{os.getpid()}")
    torch.save(scores, temporary_path)
    os.replace(temporary_path, cache_path)
    logging.info(
        "%s: cached binding scores %s at %s",
        split_name, tuple(scores.shape), cache_path,
    )
    return scores, cache_path


def prepare_data(args):
    import muon as mu
    import utils
    DATA_DIR = args.data_dir
    species = args.species
    tissue = args.tissue
    sample_name = args.sample_name
    max_peaks_per_tg = args.max_peaks_per_tg
    max_cells_per_pair = args.max_cells_per_pair
    true_false_ratio = args.true_false_ratio
    peak_flank_size = args.peak_flank_size
    num_cpu = args.num_workers or 1
    if species == "mm10":
        valid_chroms = {f"chr{i}" for i in range(1, 20)}
    
        train_chroms = [str(i) for i in range(1, 16)]
        val_chroms = [ str(i) for i in range(16, 18)]
        test_chroms = [str(i) for i in range(18, 20)]
    
        gene_ref_file = DATA_DIR / "genome_data" / "genome_annotation" / "mm10" / "Mus_musculus.GRCm39.115.gtf.gz"
    
    
    elif species == "hg38":
        valid_chroms = {f"chr{i}" for i in range(1, 23)}
    
        train_chroms = [str(i) for i in range(1, 18)]
        val_chroms = [str(i) for i in range(18, 20)]
        test_chroms = [str(i) for i in range(20, 23)]
    
        gene_ref_file = DATA_DIR / "genome_data" / "genome_annotation" / "hg38" / "Homo_sapiens.GRCh38.113.gtf.gz"


    if args.gene_ref_file is not None:
        gene_ref_file = args.gene_ref_file
    # Split genes into train/val/test based on chromosome using the GTF reference file
    train_genes, val_genes, test_genes = split_genes_by_chromosome(
        gene_ref_file,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=test_chroms
        )
    logging.info(f" Building for Sample: {sample_name}, Tissue: {tissue}\n")
    
    genome_fasta_path = DATA_DIR / "genome_data" / "reference_genome" / species / f"{species}.fa"
    chrom_sizes_path = DATA_DIR / "genome_data" / "reference_genome" / species / f"{species}.chrom.sizes"

    assert gene_ref_file.exists(), f"Gene reference file not found: {gene_ref_file}"
    assert genome_fasta_path.exists(), f"Genome FASTA file not found: {genome_fasta_path}"
    assert chrom_sizes_path.exists(), f"Chromosome sizes file not found: {chrom_sizes_path}"

    input_data_dir = DATA_DIR / "processed" / tissue / sample_name
    assert input_data_dir.exists(), f"Input data directory does not exist: {input_data_dir}"

    training_cache_dir = args.output_dir / "prepared"
    tf_dna_input_cache_dir = PROJECT_DIR / "cached_data" / species / "tf_dna_cache"
    tf_tg_input_cache_dir = training_cache_dir / sample_name

    tf_tg_input_cache_dir.mkdir(parents=True, exist_ok=True)

    tf_name_to_idx_cache_path = tf_dna_input_cache_dir / "tf_name_to_idx.csv"
    tf_embedding_cache_path = tf_dna_input_cache_dir / "tf_embeddings.pt"
    tf_mask_cache_path = tf_dna_input_cache_dir / "tf_masks.pt"
        
    atac_peak_onehot_cache_path = tf_tg_input_cache_dir / "atac_peak_tensor.pt"


    cell_type_specific_gt_dir = DATA_DIR / "ground_truth_files" / "cell_type_specific"

    # -----------------------------------
    # DATA LOADING
    # -----------------------------------

    # Load the processed Muon object
    mdata = mu.read(input_data_dir / "multiome_processed.h5mu")
    logging.info(f"Loaded MuData object:")
    logging.info(f"  {mdata.n_obs:,} cells")
    logging.info(f"  {mdata.mod['rna'].n_vars:,} genes.")
    logging.info(f"  {mdata.mod['atac'].n_vars:,} peaks.")

    # --------------------------------------------------
    # Prepare the FULL SAMPLE before caching
    # --------------------------------------------------
    rna_sample = mdata.mod["rna"]
    atac_sample = mdata.mod["atac"]

    assert rna_sample.obs_names.is_unique
    assert atac_sample.obs_names.is_unique
    assert atac_sample.var_names.is_unique

    # Align both modalities to exactly the same cell order.
    shared_cells = rna_sample.obs_names[
        rna_sample.obs_names.isin(atac_sample.obs_names)
    ]

    rna_sample = rna_sample[shared_cells].copy()
    atac_sample = atac_sample[shared_cells].copy()

    assert rna_sample.obs_names.equals(atac_sample.obs_names)

    # Filter the actual ATAC matrix, not just the sequence list.
    keep_peaks = np.array([
        peak.split(":", 1)[0] in valid_chroms
        for peak in atac_sample.var_names
    ])

    atac_sample = atac_sample[:, keep_peaks].copy()

    # Normalize identifiers before creating downstream objects.
    rna_sample.var_names = rna_sample.var_names.str.upper()
    atac_sample.var["nearest_gene"] = (
        atac_sample.var["nearest_gene"].str.upper()
    )

    assert rna_sample.var_names.is_unique, (
        "Uppercasing produced duplicate RNA gene names."
    )

    # One peak order for accessibility, sequences, and edge annotations.
    dataset_peaks = atac_sample.var_names.tolist()
    atac_peak_map = {
        peak: idx for idx, peak in enumerate(dataset_peaks)
    }

    # Matrices remain cells × features.
    atac_mat = atac_sample.layers["counts"]
    rna_mat = rna_sample.layers["counts"]


    # --------------------------------------------------
    # Sample-level maps. None of these depend on cell type.
    # --------------------------------------------------
    tf_rna_col_map = {gene: idx for idx, gene in enumerate(rna_sample.var_names)}
    tg_rna_col_map = tf_rna_col_map.copy()

    # Cell indices always refer to the full aligned sample, so every cell pool is a
    # set of positions into these same matrices.
    cell_row_map = {cell: idx for idx, cell in enumerate(rna_sample.obs_names)}

    print("RNA:", rna_mat.shape)
    print("ATAC:", atac_mat.shape)
    print("Sequence peaks:", len(dataset_peaks))
    print("Annotated cell types:", rna_sample.obs["celltype"].nunique())

    # -----------------------------------
    # SAMPLE-LEVEL TF TABLES
    # -----------------------------------
    tf_name_to_idx = pd.read_csv(tf_name_to_idx_cache_path)
    tf_name_to_idx["tf_name"] = tf_name_to_idx["tf_name"].str.upper()
    tf_name_to_idx = tf_name_to_idx.set_index("tf_name")["tf_idx"].to_dict()

    # The TF universe is a property of the sample, not of any one cell type: a TF is
    # usable if it has an embedding and is measured in this RNA matrix.
    tfs_with_embeddings = set(tf_name_to_idx) & set(rna_sample.var_names)
    rna_genes = set(rna_sample.var_names)
    logging.info(f"TFs with embeddings and RNA: {len(tfs_with_embeddings)}")

    # -----------------------------------
    # SLICE INVENTORY
    # -----------------------------------
    # mESC_label_map.tsv maps the paper's annotated cell types onto the ground-truth
    # slices. Several annotated types pool into one slice, and the pooled cell count
    # is what makes most slices usable at all.
    MIN_CELLS_PER_SLICE = args.min_cells_per_slice or 2 * max_cells_per_pair   # a 64-cell bag must be a real subsample
    MIN_TFS_PER_SLICE   = args.min_tfs_per_slice    # fewer TFs cannot support a within-TG comparison
    MAX_GT_DENSITY      = args.max_gt_density # at density 1.0 the GT is the whole box: no negatives exist

    celltype_counts = rna_sample.obs["celltype"].value_counts()

    # A label map only exists where several annotated cell types pool into one ground
    # truth slice (mESC groups 37 types into 21). Where it is absent, as for liver, the
    # mapping is the identity: each annotated cell type is its own slice.
    label_map_path = cell_type_specific_gt_dir / f"{tissue}_label_map.tsv"
    if label_map_path.exists():
        label_map = pd.read_csv(label_map_path, sep="\t", comment="#")
        slice_members = label_map.groupby("slice")["paper_celltype"].apply(list).to_dict()
        logging.info(f"{label_map_path.name}: {len(slice_members)} slices from "
                     f"{label_map['paper_celltype'].nunique()} annotated types")
    else:
        slice_members = {c: [c] for c in celltype_counts.index}
        logging.info(f"No {label_map_path.name}; one slice per annotated cell type "
                     f"({len(slice_members)})")


    def gt_path_for(slice_name):
        """build_celltype_ground_truth.py writes
        f"{cell_type.replace(' ', '_')}_ground_truth.parquet", so the annotation's
        "B cells" is B_cells_ground_truth.parquet on disk. Without this substitution
        the liver B and T cell slices are silently skipped as missing.
        """
        return (cell_type_specific_gt_dir / species / tissue /
                f"{slice_name.replace(' ', '_')}_ground_truth.parquet")


    slice_info = {}
    for slice_name, members in slice_members.items():
        present = [c for c in members if c in celltype_counts.index]
        n_cells = int(celltype_counts.reindex(present).fillna(0).sum())
        gt_path = gt_path_for(slice_name)
        if n_cells == 0 or not gt_path.exists():
            continue

        gt = pd.read_parquet(gt_path)
        gt["Source"] = gt["Source"].str.upper()
        gt["Target"] = gt["Target"].str.upper()
        gt = gt[gt["Source"].isin(tfs_with_embeddings) & gt["Target"].isin(rna_genes)]
        gt = gt.drop_duplicates(["Source", "Target"])

        n_tfs, n_tgs = gt["Source"].nunique(), gt["Target"].nunique()
        density = len(gt) / max(n_tfs * n_tgs, 1)

        if n_cells < MIN_CELLS_PER_SLICE:
            reason = "too few cells"
        elif n_tfs < MIN_TFS_PER_SLICE:
            reason = "too few TFs"
        elif density > MAX_GT_DENSITY:
            reason = f"density {density:.2f}"
        else:
            reason = None

        slice_info[slice_name] = dict(members=present, n_cells=n_cells, gt=gt,
                                      n_tfs=n_tfs, n_tgs=n_tgs, density=density,
                                      skip_reason=reason)

    if not slice_info:
        raise ValueError("No ground-truth slices overlap the sample annotations and RNA genes")

    report = pd.DataFrame([
        {"slice": s, "cells": v["n_cells"], "types": len(v["members"]),
         "edges": len(v["gt"]), "TFs": v["n_tfs"], "TGs": v["n_tgs"],
         "density": round(v["density"], 3), "use": v["skip_reason"] or "yes"}
        for s, v in slice_info.items()
    ]).sort_values("cells", ascending=False)
    print(report.to_string(index=False))

    usable = [s for s, v in slice_info.items() if v["skip_reason"] is None]
    print(f"\n{len(usable)} usable slices, "
          f"{sum(slice_info[s]['n_cells'] for s in usable):,} cells")

    if not usable:
        raise ValueError("No slices pass the cell count, TF count, and density filters")

    # Rebuild against the current peak order and flank size. Keep uint8 on CPU.
    logging.info("Creating centered peak one-hot encodings for ATAC peaks...")
    atac_peak_array = utils.create_centered_peak_onehot_array(
        peak_ids=dataset_peaks,
        genome_fasta=genome_fasta_path,
        chrom_sizes=utils.load_chrom_sizes(chrom_sizes_path),
        peak_id_to_idx=atac_peak_map,
        flank_size=peak_flank_size,
        dtype=np.uint8,
        pad_out_of_bounds=True,
        num_workers=num_cpu,
        show_progress=True,
        progress_miniters=max(1, math.ceil(len(dataset_peaks) / 50)),
        chunk_size=10000,
    )
    atac_peak_tensor = torch.as_tensor(atac_peak_array, dtype=torch.uint8)
    torch.save(atac_peak_tensor, atac_peak_onehot_cache_path)

    assert atac_mat.shape[1] == len(dataset_peaks)
    assert atac_peak_tensor.shape[0] == len(dataset_peaks)
    assert atac_sample.var_names.tolist() == dataset_peaks
    def build_tg_peak_lookup(atac_adata, atac_peak_map, link_peak_names=None):
        """Unpack .uns['peak_gene_links'] into a per-gene slice function.

        Three peak orders are in play:
          1. what preprocessing wrote      -- what links["peak_idx"] indexes
          2. atac_adata.var_names          -- (1) after the valid_chroms filter
          3. atac_peak_map                 -- the order atac_mat's columns use

        Only (1) resolves peak_idx, and only (3) may reach the dataset. Resolving
        against (2) reads the wrong peak, or runs off the end as it did here.
        """
        links = atac_adata.uns["peak_gene_links"]

        if link_peak_names is None:
            if "peaks" not in links:
                raise KeyError(
                    "peak_gene_links has no 'peaks' array. Re-run the preprocessing "
                    "cell, or pass link_peak_names=mdata.mod['atac'].var_names."
                )
            link_peak_names = links["peaks"]

        peak_names = np.asarray(link_peak_names).astype(str)
        genes = np.char.upper(np.asarray(links["genes"]).astype(str))
        gene_ptr = np.asarray(links["gene_ptr"])
        peak_idx = np.asarray(links["peak_idx"])
        tss_dist = np.asarray(links["TSS_dist"])

        if peak_idx.size and peak_idx.max() >= len(peak_names):
            raise ValueError(
                f"Link table references peak {peak_idx.max():,} but only "
                f"{len(peak_names):,} peak names were given. These are different "
                "peak universes -- the .uns table is stale, or the wrong var_names "
                "were passed."
            )

        gene_row = {gene: i for i, gene in enumerate(genes)}
        empty = {"peak_ids": [], "peak_distances": [], "peak_col_idxs": []}

        def get_peak_info_for_tg(tg, max_peaks_per_tg):
            row = gene_row.get(tg)
            if row is None:
                return dict(empty)

            start, stop = gene_ptr[row], gene_ptr[row + 1]
            names = peak_names[peak_idx[start:stop]]
            distances = tss_dist[start:stop]

            peak_ids, peak_distances, peak_col_idxs = [], [], []
            for name, distance in zip(names, distances):
                col = atac_peak_map.get(name)
                if col is None:
                    continue          # dropped by the valid_chroms filter
                peak_ids.append(name)
                peak_distances.append(float(distance))
                peak_col_idxs.append(int(col))
                if len(peak_ids) == max_peaks_per_tg:
                    break             # links are distance-sorted within a gene

            return {
                "peak_ids": peak_ids,
                "peak_distances": peak_distances,
                "peak_col_idxs": peak_col_idxs,
            }

        return get_peak_info_for_tg

    balance_tf = args.balance_tf    # negatives preserve each TF's out-degree
    balance_tg = args.balance_tg    # negatives preserve each TG's in-degree

    # Peaks depend on the target gene only, so one lookup serves every slice.
    # atac_sample, not atac_adata: the peak side is independent of which cells you took.
    get_peak_info_for_tg = build_tg_peak_lookup(
        atac_sample, atac_peak_map, link_peak_names=mdata.mod["atac"].var_names
    )

    cell_pools = {}
    labeled_frames = {"train": [], "val": [], "test": []}

    for slice_name in usable:
        info = slice_info[slice_name]

        # The cell pool is the union of every annotated type mapped to this slice.
        cell_pools[(sample_name, slice_name)] = np.flatnonzero(
            rna_sample.obs["celltype"].isin(info["members"]).fillna(False).to_numpy()
        )

        # Split by chromosome FIRST, then balance inside each split. A rectangle spans
        # two target genes; built before the split it gets cut across train and val and
        # neither side stays degree-matched.
        for split_name, genes in [("train", train_genes), ("val", val_genes), ("test", test_genes)]:
            split_gt = info["gt"][info["gt"]["Target"].isin(set(genes))]
            if split_gt.empty:
                continue

            frame = _create_labeled_df(
                split_gt, seed=args.seed,
                tf_name_to_idx=tf_name_to_idx,
                tg_id_to_idx=tg_rna_col_map,
                balance_tf=balance_tf,
                balance_tg=balance_tg,
                true_false_ratio=true_false_ratio,
            )

            if frame.empty:
                continue

            frame["sample_id"] = sample_name
            frame["cell_type"] = slice_name
            labeled_frames[split_name].append(frame)

        logging.info(
            f"{slice_name}: {info['n_cells']:,} cells from {len(info['members'])} "
            f"annotated types, {len(info['gt']):,} GT edges, {info['n_tfs']} TFs"
        )

    # One peak bag per target gene, over the union across slices.
    if not labeled_frames["train"] or not labeled_frames["val"]:
        raise ValueError("Both train and validation must contain eligible edges")

    all_tgs = pd.concat(
        [f["tg_id"] for frames in labeled_frames.values() for f in frames]
    ).dropna().unique()
    tg_to_peak_info = {tg: get_peak_info_for_tg(tg, max_peaks_per_tg) for tg in all_tgs}

    col_order = ["sample_id", "cell_type", "tf_name", "tg_id", "peak_ids", "peak_distances",
                 "label", "tf_embedding_idx", "peak_atac_cols", "tf_rna_col", "tg_rna_col"]

    splits = {}
    for split_name, frames in labeled_frames.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df["tf_rna_col"] = df["tf_name"].map(tf_rna_col_map)
        df["tg_rna_col"] = df["tg_id"].map(tg_rna_col_map)
        df["peak_ids"]       = df["tg_id"].map(lambda t: tg_to_peak_info.get(t, {}).get("peak_ids", []))
        df["peak_atac_cols"] = df["tg_id"].map(lambda t: tg_to_peak_info.get(t, {}).get("peak_col_idxs", []))
        df["peak_distances"] = df["tg_id"].map(lambda t: tg_to_peak_info.get(t, {}).get("peak_distances", []))
        splits[split_name] = df[col_order].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    gt_train_df = splits["train"]


    if gt_train_df["peak_atac_cols"].map(len).max() == 0:
        raise ValueError("No training TGs have candidate peaks")
    for name, frame in splits.items():
        frame.to_parquet(args.output_dir / f"edges_{name}.parquet", index=False)
    return splits, cell_pools, atac_mat, rna_mat, atac_peak_tensor, tf_embedding_cache_path, tf_mask_cache_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=["mm10", "hg38"], default="mm10")
    parser.add_argument("--tissue", default="mouse_liver")
    parser.add_argument("--sample_name", default="liver_sample")
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR.parent / "data")
    parser.add_argument("--gene_ref_file", type=Path,
                        help="Override the notebook's species-specific gene annotation")
    parser.add_argument("--tf_dna_checkpoint", type=Path,
                        help="Required for hg38; defaults to the notebook checkpoint for mm10")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--resume_from_checkpoint", type=Path)
    parser.add_argument("--job_id", default=os.environ.get("SLURM_JOB_ID", "local"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_cells_per_pair", type=int, default=64)
    parser.add_argument("--max_peaks_per_tg", type=int, default=25)
    parser.add_argument("--peak_flank_size", type=int, default=128)
    parser.add_argument("--binding_chunk_size", type=int, default=128)
    parser.add_argument("--true_false_ratio", type=float, default=2.)
    parser.add_argument("--balance_tf", action="store_true")
    parser.add_argument("--balance_tg", action="store_true")
    parser.add_argument("--min_cells_per_slice", type=int)
    parser.add_argument("--min_tfs_per_slice", type=int, default=5)
    parser.add_argument("--max_gt_density", type=float, default=.60)
    parser.add_argument("--resample_cells", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pooling_temperature", type=float, default=1.)
    parser.add_argument("--pos_weight", type=float, default=1.)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--plateau_patience", type=int, default=4)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.)
    parser.add_argument("--precision", choices=["32-true", "16-mixed", "bf16-mixed"], default="32-true")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--wandb_project", default="TETHER-celltype-TF-TG")
    parser.add_argument("--wandb_entity")
    parser.add_argument("--wandb_run_id", help="Reuse with --resume_from_checkpoint to resume W&B")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--run_name")
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run one train/validation batch after full data and binding preparation")
    args = parser.parse_args()
    for key in ("epochs", "batch_size", "max_cells_per_pair", "max_peaks_per_tg",
                "peak_flank_size", "binding_chunk_size", "num_heads", "d_model",
                "accumulate_grad_batches"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive")
    if args.num_workers < 0 or args.d_model % args.num_heads or args.d_model < 2:
        parser.error("num_workers must be nonnegative; d_model >= 2 and divisible by num_heads")
    if args.lr <= 0 or args.pooling_temperature <= 0 or args.pos_weight <= 0:
        parser.error("lr, pooling_temperature, and pos_weight must be positive")
    if args.true_false_ratio < 0 or not 0 < args.max_gt_density <= 1:
        parser.error("true_false_ratio must be nonnegative and max_gt_density in (0, 1]")
    if not 0 <= args.dropout < 1 or args.weight_decay < 0 or args.gradient_clip_val < 0:
        parser.error("dropout must be in [0, 1); weight decay and gradient clipping must be nonnegative")
    if args.min_tfs_per_slice <= 0 or (args.min_cells_per_slice is not None and args.min_cells_per_slice <= 0):
        parser.error("Slice cell and TF thresholds must be positive")
    if args.tf_dna_checkpoint is None:
        if args.species != "mm10":
            parser.error("--tf_dna_checkpoint is required for hg38")
        args.tf_dna_checkpoint = PROJECT_DIR / "checkpoints/new_tf_dna_models/mm10_3831017_epoch_12.ckpt"
    if not args.tf_dna_checkpoint.is_file():
        parser.error(f"TF-DNA checkpoint not found: {args.tf_dna_checkpoint}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    args.run_name = args.run_name or f"celltype_{args.sample_name}_{args.job_id}_{stamp}"
    args.output_dir = args.output_dir or PROJECT_DIR / "checkpoints/celltype_tf_tg" / args.run_name
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        raise ValueError("This launcher supports one process and one GPU per run")
    pl.seed_everything(args.seed, workers=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    splits, pools, atac, rna, peaks, embedding_path, mask_path = prepare_data(args)
    embeddings = torch.load(embedding_path, map_location="cpu", weights_only=True)
    masks = torch.load(mask_path, map_location="cpu", weights_only=True)
    from models.tf_to_dna import TFPeakBindingModel, LitTFPeakBindingModel
    base = TFPeakBindingModel(tf_embedding_dim=128, hidden_dim=128, dropout=.3,
                              num_layers=4, num_heads=4, dim_head=32)
    binding_model = LitTFPeakBindingModel.load_from_checkpoint(
        str(args.tf_dna_checkpoint), map_location="cpu", model=base,
        tf_embeddings_tensor=embeddings, tf_mask_tensor=masks,
        lr=1e-4, weight_decay=1e-4, pos_weight=None).model
    use_cuda = args.accelerator != "cpu" and torch.cuda.is_available()
    if args.accelerator == "gpu" and not use_cuda:
        raise RuntimeError("GPU requested but CUDA is unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    binding_model.requires_grad_(False).eval().to(device)
    datasets = {}
    for name, frame in splits.items():
        if frame.empty:
            continue
        scores, binding_cache_file = load_or_precompute_binding_scores(
            name,
            frame,
            binding_model,
            prepared_dir=args.output_dir / "prepared",
            tf_dna_checkpoint=args.tf_dna_checkpoint,
            tf_embeddings_tensor=embeddings,
            tf_mask_tensor=masks,
            atac_peak_tensor=peaks,
            max_peaks_per_tg=args.max_peaks_per_tg,
            device=device,
            chunk_size=args.binding_chunk_size,
        )
        datasets[name] = TFTGEdgeBagDataset(
            frame, tf_embeddings_tensor=None, tf_mask_tensor=None,
            atac_peak_tensor=peaks, atac_mat=atac, rna_mat=rna, cell_pools=pools,
            max_peaks_per_tg=args.max_peaks_per_tg,
            resample_max_cells_per_pair=args.max_cells_per_pair,
            resample_cells=name == "train" and args.resample_cells,
            binding_scores=scores, seed=args.seed)
        config[f"{name}_edges"] = len(frame)
        config[f"{name}_positive_fraction"] = float(frame.label.mean())
        config[f"{name}_celltypes"] = sorted(frame.cell_type.unique().tolist())
        config[f"{name}_binding_cache"] = str(binding_cache_file)
    del binding_model, base, embeddings, masks
    if use_cuda:
        torch.cuda.empty_cache()
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    loaders = {name: DataLoader(
        dataset, batch_size=args.batch_size, shuffle=name == "train",
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=use_cuda, drop_last=False,
    ) for name, dataset in datasets.items()}
    module = LitTFTGRegulationModel(**{key: getattr(args, key) for key in (
        "d_model", "num_heads", "dropout", "lr", "weight_decay",
        "pooling_temperature", "pos_weight", "plateau_patience")})
    if args.wandb_mode == "disabled":
        logger = CSVLogger(str(args.output_dir), name="metrics")
    else:
        logger = WandbLogger(
            project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
            save_dir=str(args.output_dir), offline=args.wandb_mode == "offline",
            id=args.wandb_run_id, resume="allow" if args.wandb_run_id else None,
            log_model=False)
    logger.log_hyperparams(config)
    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints", filename="epoch-{epoch:03d}",
        auto_insert_metric_name=False, monitor="val/loss", mode="min",
        save_top_k=1, save_last=True)
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=1, max_epochs=args.epochs,
        precision=args.precision, logger=logger, default_root_dir=str(args.output_dir),
        callbacks=[checkpoint, TwoPercentProgressBar(), LearningRateMonitor(logging_interval="epoch"),
                   EarlyStopping(monitor="val/loss", mode="min",
                                 patience=args.early_stopping_patience, check_finite=True)],
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val, log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run)
    try:
        trainer.fit(module, loaders["train"], loaders["val"],
                    ckpt_path=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
        if "test" in loaders and not args.fast_dev_run:
            metrics = trainer.test(module, loaders["test"], ckpt_path="best")
            (args.output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        logging.info("Best checkpoint: %s", checkpoint.best_model_path)
    finally:
        if isinstance(logger, WandbLogger):
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
