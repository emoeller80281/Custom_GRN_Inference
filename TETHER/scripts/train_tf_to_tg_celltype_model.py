#!/usr/bin/env python
"""Train cell-type-specific TF->TG edge bags on one GPU (or CPU).

Defaults follow the liver notebook, using every eligible edge. Prepared inputs,
TF-DNA binding scores, and input scaling are reused across matching new runs and
invalidated when their data or preprocessing configuration changes.
Example: python scripts/train_tf_to_tg_celltype_model.py --wandb_mode offline

Training-only holdouts retain their chromosome-held-out validation/test edges:
  --holdout_sample mESC:E7.5_rep1 --holdout_celltype "Definitive endoderm"
"""
import argparse
import copy
import hashlib
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
import time

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.6"

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
)
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from models.tf_to_tg_celltype import LitTFTGRegulationModel


PREPARED_CACHE_VERSION = 1


def _file_identity(path):
    """Return the inexpensive identity used to invalidate prepared-data caches."""
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _optional_file_identity(path):
    path = Path(path)
    return _file_identity(path) if path.is_file() else None


def _atomic_write_json(path, value):
    path = Path(path)
    temporary_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary_path, path)


def _atomic_torch_save(value, path):
    path = Path(path)
    temporary_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def _cache_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()[:16]


def build_run_cache_manifest(args, source_specs, holdout_samples, holdout_celltypes):
    """Describe every input that can change prepared edges, binding, or scaling."""
    data_dir = args.data_dir
    if args.species == "mm10":
        default_gene_ref = (
            data_dir / "genome_data/genome_annotation/mm10" /
            "Mus_musculus.GRCm39.115.gtf.gz"
        )
    else:
        default_gene_ref = (
            data_dir / "genome_data/genome_annotation/hg38" /
            "Homo_sapiens.GRCh38.113.gtf.gz"
        )
    gene_ref = args.gene_ref_file or default_gene_ref
    reference_dir = data_dir / "genome_data/reference_genome" / args.species
    ground_truth_root = data_dir / "ground_truth_files/cell_type_specific"
    tf_dna_cache = PROJECT_DIR / "cached_data" / args.species / "tf_dna_cache"

    sources = []
    for tissue, sample_name in source_specs:
        ground_truth_dir = ground_truth_root / args.species / tissue
        ground_truth_files = sorted(ground_truth_dir.glob("*_ground_truth.parquet"))
        if not ground_truth_files:
            raise FileNotFoundError(
                f"No ground-truth parquet files found in {ground_truth_dir}"
            )
        sources.append({
            "tissue": tissue,
            "sample_name": sample_name,
            "multiome": _file_identity(
                data_dir / "processed" / tissue / sample_name / "multiome_processed.h5mu"
            ),
            "label_map": _optional_file_identity(
                ground_truth_root / f"{tissue}_label_map.tsv"
            ),
            "ground_truth": [_file_identity(path) for path in ground_truth_files],
        })

    return {
        "cache_version": PREPARED_CACHE_VERSION,
        "species": args.species,
        "datasets": [f"{tissue}:{sample}" for tissue, sample in source_specs],
        "holdout_samples": sorted(":".join(spec) for spec in holdout_samples),
        "holdout_celltypes": sorted(holdout_celltypes),
        "settings": {
            "seed": args.seed,
            "max_cells_per_pair": args.max_cells_per_pair,
            "max_peaks_per_tg": args.max_peaks_per_tg,
            "peak_flank_size": args.peak_flank_size,
            "true_false_ratio": args.true_false_ratio,
            "balance_tf": args.balance_tf,
            "balance_tg": args.balance_tg,
            "min_cells_per_slice": args.min_cells_per_slice,
            "min_tfs_per_slice": args.min_tfs_per_slice,
            "max_gt_density": args.max_gt_density,
            "scaler_edges_per_sample": (
                min(args.scaler_edges_per_sample, args.batch_size)
                if args.fast_dev_run else args.scaler_edges_per_sample
            ),
            "fast_dev_run": args.fast_dev_run,
        },
        "references": {
            "gene_reference": _file_identity(gene_ref),
            "genome_fasta": _file_identity(reference_dir / f"{args.species}.fa"),
            "chromosome_sizes": _file_identity(
                reference_dir / f"{args.species}.chrom.sizes"
            ),
            "tf_name_to_idx": _file_identity(tf_dna_cache / "tf_name_to_idx.csv"),
            "tf_embeddings": _file_identity(tf_dna_cache / "tf_embeddings.pt"),
            "tf_masks": _file_identity(tf_dna_cache / "tf_masks.pt"),
            "tf_dna_checkpoint": _file_identity(args.tf_dna_checkpoint),
        },
        "sources": sources,
    }


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
        # PyArrow-backed parquet columns can expose read-only NumPy views. PyTorch
        # warns when torch.from_numpy shares those buffers, even though these tensors
        # are only copied into padded outputs below. Own writable arrays explicitly.
        peak_cols = np.array(row.peak_atac_cols, dtype=np.int64, copy=True)
        distances = np.array(row.peak_distances, dtype=np.float32, copy=True)

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


def make_balanced_scaler_dataset(train_datasets, edges_per_sample, seed):
    """Draw the same number of unique, deterministic train edges from each sample."""
    if not train_datasets:
        raise ValueError("At least one training dataset is required to fit scaling")
    if edges_per_sample <= 0:
        raise ValueError("edges_per_sample must be positive")

    sample_size = min(edges_per_sample, *(len(dataset) for dataset in train_datasets))
    logging.info(
        "Fitting shared scaling with %s unique training edges from each of %d samples",
        f"{sample_size:,}",
        len(train_datasets),
    )
    rng = np.random.default_rng(seed)
    subsets = []
    for dataset in train_datasets:
        if len(dataset) == 0:
            raise ValueError("Cannot fit scaling from an empty training dataset")
        indices = rng.choice(
            len(dataset),
            size=sample_size,
            replace=False,
        )
        subsets.append(Subset(dataset, indices.tolist()))
    return ConcatDataset(subsets)


@torch.no_grad()
def fit_shared_input_scaler(loader):
    """Fit global moments from real entries in depth-normalized/log1p train inputs."""
    totals = {
        name: {"sum": 0.0, "sum_sq": 0.0, "count": 0}
        for name in ("tf_expression", "tg_expression", "peak_accessibility")
    }

    def update(name, values):
        values = values.double()
        totals[name]["sum"] += values.sum().item()
        totals[name]["sum_sq"] += values.square().sum().item()
        totals[name]["count"] += values.numel()

    for batch in tqdm(loader, desc="Fitting shared input scaling"):
        cell_mask = batch["cell_mask"].bool()
        peak_mask = batch["peak_mask"].bool()
        update("tf_expression", batch["tf_expression"][cell_mask])
        update("tg_expression", batch["tg_expression"][cell_mask])
        accessibility_mask = cell_mask[:, :, None] & peak_mask[:, None, :]
        update(
            "peak_accessibility",
            batch["peak_accessibility"][accessibility_mask],
        )

    scaler = {}
    for name, values in totals.items():
        if values["count"] == 0:
            raise ValueError(f"No real values were available for {name} scaling")
        mean = values["sum"] / values["count"]
        variance = max(values["sum_sq"] / values["count"] - mean ** 2, 1e-12)
        scaler[name] = {
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
            "count": int(values["count"]),
        }
    return scaler


def validate_input_scaler(scaler):
    expected = {"tf_expression", "tg_expression", "peak_accessibility"}
    if set(scaler) != expected:
        raise ValueError(f"Input scaler keys are {set(scaler)}; expected {expected}")
    for name, values in scaler.items():
        if not {"mean", "std", "count"}.issubset(values):
            raise ValueError(f"Input scaler entry {name} is incomplete")
        if (not math.isfinite(float(values["mean"]))
                or not math.isfinite(float(values["std"]))
                or float(values["std"]) <= 0
                or int(values["count"]) <= 0):
            raise ValueError(f"Input scaler entry {name} is invalid: {values}")
    return scaler


def make_sample_balanced_sampler(datasets, seed):
    """Give every source sample equal expected probability during joint training."""
    if len(datasets) < 2:
        return None
    weights = torch.cat([
        torch.full((len(dataset),), 1.0 / len(dataset), dtype=torch.double)
        for dataset in datasets
    ])
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=sum(len(dataset) for dataset in datasets),
        replacement=True,
        generator=generator,
    )

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
    full_tqdm=False,
):
    """Binding score for every (edge, peak slot) in labeled_df, as [n_edges, P] float32."""
    n_edges = len(labeled_df)
    out = torch.zeros(n_edges, max_peaks_per_tg, dtype=torch.float32)

    if n_edges == 0:
        return out, {}

    # Vectorized extraction from dataframe
    tf_ids_base = labeled_df['tf_embedding_idx'].to_numpy(dtype=int)
    peak_cols_list = labeled_df['peak_atac_cols'].to_list()
    
    edge_rows, slots, tf_ids, peak_cols = [], [], [], []
    for row_position, cols in enumerate(peak_cols_list):
        valid_cols = cols[:max_peaks_per_tg]
        n_slots = len(valid_cols)
        
        edge_rows.extend([row_position] * n_slots)
        slots.extend(range(n_slots))
        tf_ids.extend([tf_ids_base[row_position]] * n_slots)
        peak_cols.extend(map(int, valid_cols))

    if not edge_rows:
        return out, {}

    edge_rows = np.asarray(edge_rows)
    slots = np.asarray(slots)
    pairs = np.stack([np.asarray(tf_ids), np.asarray(peak_cols)], axis=1)

    unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)

    was_training = peak_model.training
    peak_model.eval()
    
    torch.cuda.reset_peak_memory_stats()
    total_pairs = len(unique_pairs)
    
    # Cast peak tensor to float globally once to keep the inner loop fast
    # (Since it was passed as uint8 in your script)
    atac_peak_tensor_float = atac_peak_tensor.to(dtype=torch.float32)
    
    # Move the unique pairs indices to the GPU once
    unique_pairs_gpu = torch.from_numpy(unique_pairs).to(device=device, dtype=torch.long)
    unique_scores_gpu = torch.zeros(total_pairs, device=device, dtype=torch.float16)

    gpu_usage = {}
    miniters = 1 if full_tqdm else max(1, math.ceil(total_pairs / chunk_size / 50))
    
    global_start_time = time.time()

    try:
        chunks = range(0, total_pairs, chunk_size)
        pbar = tqdm(chunks, desc="Binding scores", miniters=miniters, maxinterval=float("inf"))
        
        with torch.inference_mode():
            for start in pbar:
                chunk = unique_pairs_gpu[start:start + chunk_size]
                current_end = start + len(chunk)
                
                tf_chunk = chunk[:, 0]
                peak_chunk = chunk[:, 1]
                
                # --- FIX: Direct indexing natively handles N-Dimensional tensors ---
                tf_emb = tf_embeddings_tensor[tf_chunk]
                tf_msk = tf_mask_tensor[tf_chunk]
                pk_emb = atac_peak_tensor_float[peak_chunk]
                
                gpu_mem_reserved = torch.cuda.memory_reserved(device=device) / 1e9
                gpu_mem_allocated = torch.cuda.memory_allocated(device=device) / 1e9

                with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                    logits = peak_model(
                        tf_embedding=tf_emb,
                        tf_mask=tf_msk,
                        peak_embedding=pk_emb,
                    )
                    scores = logits.reshape(-1).sigmoid()
                
                unique_scores_gpu[start:current_end] = scores
                
                step_time = time.time() - global_start_time
                
                pbar.set_postfix({
                    "pairs_processed": f"{current_end}/{total_pairs}",
                    "gpu_res": f"{gpu_mem_reserved:.2f}GB",
                    "gpu_alloc": f"{gpu_mem_allocated:.2f}GB",
                })
                
                gpu_usage[start] = {
                    "time": step_time,
                    "reserved": gpu_mem_reserved,
                    "allocated": gpu_mem_allocated,
                }
    finally:
        peak_model.train(was_training) 

    # Bring calculated GPU values down to the CPU before structural mapping
    unique_scores = unique_scores_gpu.cpu().float()

    out[edge_rows, slots] = unique_scores[inverse]

    print(f"{len(unique_pairs):,} distinct (TF, peak) pairs for "
          f"{len(edge_rows):,} slots across {n_edges:,} edges "
          f"({out.numel() * 4 / 1e6:.1f} MB)")
          
    return out, gpu_usage


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
    full_tqdm=False,
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
                return scores, cache_path, None
            logging.warning(
                "%s: ignoring binding cache with shape %s; expected %s",
                split_name, getattr(scores, "shape", None), expected_shape,
            )
        except Exception as error:
            logging.warning(
                "%s: could not load binding cache %s (%s); rebuilding",
                split_name, cache_path, error,
            )

    scores, gpu_usage = precompute_binding_scores(
        labeled_df,
        peak_model,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor,
        max_peaks_per_tg=max_peaks_per_tg,
        device=device,
        chunk_size=chunk_size,
        full_tqdm=full_tqdm
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp-{os.getpid()}")
    torch.save(scores, temporary_path)
    os.replace(temporary_path, cache_path)
    logging.info(
        "%s: cached binding scores %s at %s",
        split_name, tuple(scores.shape), cache_path,
    )
    return scores, cache_path, gpu_usage


def save_prepared_cache(
    output_dir,
    sample_name,
    splits,
    cell_pools,
    atac_peak_tensor,
    input_manifest,
):
    """Publish a complete prepared source, with the manifest written last."""
    output_dir = Path(output_dir)
    for name, frame in splits.items():
        edge_path = output_dir / f"edges_{name}.parquet"
        temporary_path = edge_path.with_name(
            f"{edge_path.name}.tmp-{os.getpid()}.parquet"
        )
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, edge_path)

    pool_metadata = []
    pool_arrays = {}
    for index, ((pool_sample, cell_type), cell_indices) in enumerate(
        sorted(cell_pools.items())
    ):
        array_key = f"pool_{index}"
        pool_arrays[array_key] = np.asarray(cell_indices, dtype=np.int64)
        pool_metadata.append({
            "array_key": array_key,
            "sample_id": pool_sample,
            "cell_type": cell_type,
        })
    cell_pools_path = output_dir / "cell_pools.npz"
    temporary_pool_path = cell_pools_path.with_name(
        f"{cell_pools_path.name}.tmp-{os.getpid()}.npz"
    )
    np.savez_compressed(temporary_pool_path, **pool_arrays)
    os.replace(temporary_pool_path, cell_pools_path)

    peak_path = output_dir / "prepared" / sample_name / "atac_peak_tensor.pt"
    _atomic_torch_save(atac_peak_tensor, peak_path)
    _atomic_write_json(output_dir / "prepared_manifest.json", {
        "inputs": input_manifest,
        "outputs": {
            "splits": list(splits),
            "cell_pools": pool_metadata,
            "atac_peak_tensor_shape": list(atac_peak_tensor.shape),
            "atac_peak_tensor_dtype": str(atac_peak_tensor.dtype),
        },
    })


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
    prepared_manifest_path = args.output_dir / "prepared_manifest.json"
    cell_pools_cache_path = args.output_dir / "cell_pools.npz"


    cell_type_specific_gt_dir = DATA_DIR / "ground_truth_files" / "cell_type_specific"

    # -----------------------------------
    # DATA LOADING
    # -----------------------------------
    print(f"\n{species} {tissue} {sample_name}")
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
    atac_mat = atac_sample.X
    rna_mat = rna_sample.X


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

    if prepared_manifest_path.is_file():
        try:
            cached_manifest = json.loads(prepared_manifest_path.read_text())
            if cached_manifest.get("inputs") == args.prepared_cache_manifest:
                split_names = cached_manifest["outputs"]["splits"]
                if not {"train", "val"}.issubset(split_names):
                    raise ValueError("cached splits do not include train and validation")
                if not set(split_names).issubset({"train", "val", "test"}):
                    raise ValueError(f"unknown cached split names: {split_names}")
                splits = {
                    name: pd.read_parquet(args.output_dir / f"edges_{name}.parquet")
                    for name in split_names
                }
                atac_peak_tensor = torch.load(
                    atac_peak_onehot_cache_path,
                    map_location="cpu",
                    weights_only=True,
                )
                expected_peak_shape = (len(dataset_peaks), 2 * peak_flank_size, 4)
                if (tuple(atac_peak_tensor.shape) != expected_peak_shape
                        or atac_peak_tensor.dtype != torch.uint8):
                    raise ValueError(
                        f"cached ATAC peak tensor is {tuple(atac_peak_tensor.shape)} "
                        f"{atac_peak_tensor.dtype}; expected {expected_peak_shape} torch.uint8"
                    )

                cell_pools = {}
                with np.load(cell_pools_cache_path, allow_pickle=False) as pool_archive:
                    for pool in cached_manifest["outputs"]["cell_pools"]:
                        indices = pool_archive[pool["array_key"]].astype(
                            np.int64, copy=False
                        )
                        if indices.size and (
                            indices.min() < 0 or indices.max() >= rna_mat.shape[0]
                        ):
                            raise ValueError(
                                f"cached cell pool {pool['cell_type']} is out of bounds"
                            )
                        cell_pools[(pool["sample_id"], pool["cell_type"])] = indices

                cached_celltypes = {
                    (sample_name, cell_type)
                    for frame in splits.values()
                    for cell_type in frame["cell_type"].unique()
                }
                if not cached_celltypes.issubset(cell_pools):
                    raise ValueError("cached edges reference a missing cell pool")

                logging.info(
                    "%s:%s loaded prepared inputs from %s",
                    tissue, sample_name, args.output_dir,
                )
                return (
                    splits, cell_pools, atac_mat, rna_mat, atac_peak_tensor,
                    tf_embedding_cache_path, tf_mask_cache_path,
                )
            logging.info(
                "%s:%s prepared cache manifest changed; rebuilding",
                tissue, sample_name,
            )
        except Exception as error:
            logging.warning(
                "%s:%s could not load prepared cache from %s (%s); rebuilding",
                tissue, sample_name, args.output_dir, error,
            )

    # The GTF and ground-truth files are only needed on a prepared-cache miss.
    train_genes, val_genes, test_genes = split_genes_by_chromosome(
        gene_ref_file,
        train_chroms=train_chroms,
        val_chroms=val_chroms,
        test_chroms=test_chroms,
    )

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
        slice_members = label_map.groupby("celltype_group")["celltype"].apply(list).to_dict()
        logging.info(f"{label_map_path.name}: {len(slice_members)} slices from "
                     f"{label_map['celltype'].nunique()} annotated types")
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

    save_prepared_cache(
        args.output_dir,
        sample_name,
        splits,
        cell_pools,
        atac_peak_tensor,
        args.prepared_cache_manifest,
    )
    logging.info(
        "%s:%s saved prepared inputs to %s",
        tissue, sample_name, args.output_dir,
    )
    return splits, cell_pools, atac_mat, rna_mat, atac_peak_tensor, tf_embedding_cache_path, tf_mask_cache_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", choices=["mm10", "hg38"], default="mm10")
    parser.add_argument("--tissue", default="mouse_liver")
    parser.add_argument("--sample_name", default="liver_sample")
    parser.add_argument(
        "--dataset", action="append", metavar="TISSUE:SAMPLE",
        help=("Repeat to train jointly across samples, for example "
              "--dataset mouse_liver:liver_sample --dataset mESC:E7.5_rep1. "
              "When omitted, --tissue and --sample_name define one dataset."),
    )
    parser.add_argument(
        "--holdout_sample", action="append", default=[], metavar="TISSUE:SAMPLE",
        help=("Repeat to exclude a complete source sample from training while retaining "
              "its chromosome-held-out validation and test edges."),
    )
    parser.add_argument(
        "--holdout_celltype", action="append", default=[], metavar="CELLTYPE",
        help=("Repeat to exclude a pooled cell-type slice from training in every sample "
              "while retaining its validation and test edges. Matching is exact."),
    )
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR.parent / "data")
    parser.add_argument("--gene_ref_file", type=Path,
                        help="Override the notebook's species-specific gene annotation")
    parser.add_argument("--tf_dna_checkpoint", type=Path,
                        help="Required for hg38; defaults to the notebook checkpoint for mm10")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help=("Exact directory for reusable prepared inputs, binding scores, and "
              "input scaling. By default, a stable directory is selected from "
              "the data and preprocessing configuration."),
    )
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
    parser.add_argument(
        "--balance_samples", action=argparse.BooleanOptionalAction, default=True,
        help="Give every source sample equal expected training probability",
    )
    parser.add_argument(
        "--scaler_edges_per_sample", type=int, default=100000,
        help="Equal number of training edges per sample used to fit shared scaling",
    )
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
    if (args.num_workers < 0 or args.scaler_edges_per_sample <= 0
            or args.d_model % args.num_heads or args.d_model < 2):
        parser.error(
            "num_workers must be nonnegative; scaler_edges_per_sample must be "
            "positive; d_model >= 2 and divisible by num_heads"
        )
    if args.dataset:
        malformed = [value for value in args.dataset
                     if value.count(":") != 1 or not all(value.split(":"))]
        if malformed:
            parser.error(f"--dataset values must be TISSUE:SAMPLE; invalid: {malformed}")
    malformed_holdouts = [
        value for value in args.holdout_sample
        if value.count(":") != 1 or not all(value.split(":"))
    ]
    if malformed_holdouts:
        parser.error(
            "--holdout_sample values must be TISSUE:SAMPLE; invalid: "
            f"{malformed_holdouts}"
        )
    if any(not value.strip() for value in args.holdout_celltype):
        parser.error("--holdout_celltype values must be nonempty")
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
    if args.dataset:
        default_sample_label = (
            "joint" if len(args.dataset) > 1 else args.dataset[0].split(":", 1)[1]
        )
    else:
        default_sample_label = args.sample_name
    args.run_name = args.run_name or f"celltype_{default_sample_label}_{args.job_id}_{stamp}"
    args.output_dir = args.output_dir or PROJECT_DIR / "checkpoints/celltype_tf_tg" / args.run_name
    return args


def main():
    args = parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        raise ValueError("This launcher supports one process and one GPU per run")
    
    pl.seed_everything(args.seed, workers=True)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_values = args.dataset or [f"{args.tissue}:{args.sample_name}"]
    
    source_specs = [tuple(value.split(":", 1)) for value in dataset_values]
    if len(set(source_specs)) != len(source_specs):
        raise ValueError(f"Duplicate --dataset entries: {source_specs}")
    
    holdout_samples = {
        tuple(value.split(":", 1)) for value in args.holdout_sample
    }
    
    unknown_holdout_samples = holdout_samples - set(source_specs)
    if unknown_holdout_samples:
        raise ValueError(
            "Holdout samples must match a configured dataset: "
            f"{sorted(unknown_holdout_samples)}"
        )
        
    holdout_celltypes = set(args.holdout_celltype)

    cache_manifest = build_run_cache_manifest(
        args, source_specs, holdout_samples, holdout_celltypes,
    )
    cache_key = _cache_digest(cache_manifest)
    if args.cache_dir is None:
        args.cache_dir = (
            PROJECT_DIR / "cached_data" / args.species /
            "celltype_tf_tg" / cache_key
        )
        
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    run_cache_manifest_path = args.cache_dir / "cache_manifest.json"
    if run_cache_manifest_path.is_file():
        existing_cache_manifest = json.loads(run_cache_manifest_path.read_text())
        if existing_cache_manifest != cache_manifest:
            raise ValueError(
                f"Cache directory belongs to a different data configuration: "
                f"{args.cache_dir}. Choose another --cache_dir or remove the override."
            )
    else:
        _atomic_write_json(run_cache_manifest_path, cache_manifest)
    logging.info("Prepared-data cache: %s (key %s)", args.cache_dir, cache_key)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["cache_key"] = cache_key
    _atomic_write_json(args.output_dir / "run_config.json", config)

    prepared_sources = []
    for tissue, sample_name in source_specs:
        source_args = copy.copy(args)
        source_args.tissue = tissue
        source_args.sample_name = sample_name
        source_args.output_dir = (
            args.cache_dir / "prepared_sources" / f"{tissue}__{sample_name}"
        )
        source_args.prepared_cache_manifest = cache_manifest
        source_args.output_dir.mkdir(parents=True, exist_ok=True)
        prepared = prepare_data(source_args)
        prepared_sources.append({
            "tissue": tissue,
            "sample_name": sample_name,
            "output_dir": source_args.output_dir,
            "splits": prepared[0],
            "pools": prepared[1],
            "atac": prepared[2],
            "rna": prepared[3],
            "peaks": prepared[4],
            "embedding_path": prepared[5],
            "mask_path": prepared[6],
        })

    available_train_celltypes = {
        cell_type
        for source in prepared_sources
        for cell_type in source["splits"]["train"]["cell_type"].unique()
    }
    unknown_holdout_celltypes = holdout_celltypes - available_train_celltypes
    if unknown_holdout_celltypes:
        raise ValueError(
            "Holdout cell types do not match any training cell-type slice: "
            f"{sorted(unknown_holdout_celltypes)}"
        )

    embedding_paths = {str(source["embedding_path"].resolve()) for source in prepared_sources}
    mask_paths = {str(source["mask_path"].resolve()) for source in prepared_sources}
    
    if len(embedding_paths) != 1 or len(mask_paths) != 1:
        raise ValueError("Joint datasets must share the same TF embeddings and masks")
    
    embedding_path = prepared_sources[0]["embedding_path"]
    mask_path = prepared_sources[0]["mask_path"]
    
    embeddings = torch.load(embedding_path, map_location="cpu", weights_only=True)
    masks = torch.load(mask_path, map_location="cpu", weights_only=True)
    
    from models.tf_to_dna import TFPeakBindingModel, LitTFPeakBindingModel
    
    base = TFPeakBindingModel(tf_embedding_dim=128, hidden_dim=128, dropout=.3,
                              num_layers=4, num_heads=4, dim_head=32)
    
    binding_model = LitTFPeakBindingModel.load_from_checkpoint(
        str(args.tf_dna_checkpoint), 
        map_location="cpu", 
        model=base,
        tf_embeddings_tensor=embeddings, 
        tf_mask_tensor=masks,
        lr=1e-4, 
        weight_decay=1e-4, 
        pos_weight=None
        ).model
    
    use_cuda = args.accelerator != "cpu" and torch.cuda.is_available()
    if args.accelerator == "gpu" and not use_cuda:
        raise RuntimeError("GPU requested but CUDA is unavailable")
    device = torch.device("cuda" if use_cuda else "cpu")
    
    binding_model.requires_grad_(False).eval().to(device)
    embeddings = embeddings.to(device=device, dtype=torch.float32)
    masks = masks.to(device=device, dtype=torch.bool)
    logging.info(
        "Keeping TF embeddings %s and masks %s resident on %s for binding-score calculation",
        tuple(embeddings.shape), tuple(masks.shape), device,
    )
    
    dataset_parts = {"train": [], "val": [], "test": []}
    scaler_train_parts = []
    scaler_train_sources = []
    for source in prepared_sources:
        source_key = f"{source['tissue']}__{source['sample_name']}"
        binding_peak_tensor = source["peaks"].to(device=device, dtype=torch.uint8)
        logging.info(
            "%s: keeping ATAC peak tensor %s resident on %s for binding-score calculation",
            source_key, tuple(binding_peak_tensor.shape), device,
        )
        for split_name, frame in source["splits"].items():
            if frame.empty:
                continue
            original_edge_count = len(frame)
            if split_name == "train":
                if (source["tissue"], source["sample_name"]) in holdout_samples:
                    frame = frame.iloc[0:0].copy()
                elif holdout_celltypes:
                    frame = frame.loc[
                        ~frame["cell_type"].isin(holdout_celltypes)
                    ].reset_index(drop=True)
                removed_edge_count = original_edge_count - len(frame)
                if removed_edge_count:
                    logging.info(
                        "%s: removed %s of %s training edges for holdout selection",
                        source_key,
                        f"{removed_edge_count:,}",
                        f"{original_edge_count:,}",
                    )
                config[f"{source_key}_train_holdout_edges"] = removed_edge_count
                config[f"{source_key}_train_original_edges"] = original_edge_count
                config[f"{source_key}_train_edges"] = len(frame)
                if frame.empty:
                    continue
                
            scores, binding_cache_file, gpu_usage = load_or_precompute_binding_scores(
                split_name,
                frame,
                binding_model,
                prepared_dir=source["output_dir"] / "prepared",
                tf_dna_checkpoint=args.tf_dna_checkpoint,
                tf_embeddings_tensor=embeddings,
                tf_mask_tensor=masks,
                atac_peak_tensor=binding_peak_tensor,
                max_peaks_per_tg=args.max_peaks_per_tg,
                device=device,
                chunk_size=args.binding_chunk_size,
            )
            
            dataset_kwargs = dict(
                tf_embeddings_tensor=None,
                tf_mask_tensor=None,
                atac_peak_tensor=source["peaks"],
                atac_mat=source["atac"],
                rna_mat=source["rna"],
                cell_pools=source["pools"],
                max_peaks_per_tg=args.max_peaks_per_tg,
                resample_max_cells_per_pair=args.max_cells_per_pair,
                binding_scores=scores,
                seed=args.seed,
            )
            
            dataset_parts[split_name].append(TFTGEdgeBagDataset(
                frame,
                resample_cells=split_name == "train" and args.resample_cells,
                **dataset_kwargs,
            ))
            
            if split_name == "train":
                scaler_train_parts.append(TFTGEdgeBagDataset(
                    frame, resample_cells=False, **dataset_kwargs,
                ))
                scaler_train_sources.append(
                    f"{source['tissue']}:{source['sample_name']}"
                )
                
            config[f"{source_key}_{split_name}_edges"] = len(frame)
            config[f"{source_key}_{split_name}_positive_fraction"] = float(frame.label.mean())
            config[f"{source_key}_{split_name}_celltypes"] = sorted(
                frame.cell_type.unique().tolist()
            )
            config[f"{source_key}_{split_name}_binding_cache"] = str(binding_cache_file)

        del binding_peak_tensor
        if use_cuda:
            torch.cuda.empty_cache()
        logging.info("%s: released ATAC peak tensor from %s", source_key, device)

    del binding_model, base, embeddings, masks
    if use_cuda:
        torch.cuda.empty_cache()
    logging.info("Released binding model, TF embeddings, and TF masks from %s", device)

    if not dataset_parts["train"] or not dataset_parts["val"]:
        raise ValueError("Joint training requires train and validation data")
    datasets = {
        split_name: (parts[0] if len(parts) == 1 else ConcatDataset(parts))
        for split_name, parts in dataset_parts.items() if parts
    }

    scaler_path = args.cache_dir / "input_scaler.json"
    scaler_manifest_path = args.cache_dir / "input_scaler_manifest.json"
    scaler_manifest = {
        "datasets": [f"{tissue}:{sample}" for tissue, sample in source_specs],
        "scaler_train_datasets": scaler_train_sources,
        "holdout_samples": sorted(":".join(spec) for spec in holdout_samples),
        "holdout_celltypes": sorted(holdout_celltypes),
        "train_edges": [len(dataset) for dataset in scaler_train_parts],
        "edges_per_sample": (
            min(args.scaler_edges_per_sample, args.batch_size)
            if args.fast_dev_run else args.scaler_edges_per_sample
        ),
        "seed": args.seed,
        "cache_key": cache_key,
        "max_cells_per_pair": args.max_cells_per_pair,
        "input_space": "per_sample_depth_normalized_log1p_X",
    }
    cached_manifest = (
        json.loads(scaler_manifest_path.read_text())
        if scaler_manifest_path.exists() else None
    )
    if scaler_path.exists() and cached_manifest == scaler_manifest:
        input_scaler = validate_input_scaler(json.loads(scaler_path.read_text()))
        logging.info("Loaded shared input scaling from %s", scaler_path)
    else:
        scaler_dataset = make_balanced_scaler_dataset(
            scaler_train_parts, scaler_manifest["edges_per_sample"], args.seed,
        )
        scaler_loader = DataLoader(
            scaler_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=False,
            drop_last=False,
        )
        input_scaler = validate_input_scaler(fit_shared_input_scaler(scaler_loader))
        _atomic_write_json(scaler_path, input_scaler)
        _atomic_write_json(scaler_manifest_path, scaler_manifest)
        logging.info("Saved shared input scaling to %s", scaler_path)
    config["input_scaler"] = input_scaler
    config["input_scaler_path"] = str(scaler_path)

    _atomic_write_json(args.output_dir / "run_config.json", config)
    train_sampler = (
        make_sample_balanced_sampler(dataset_parts["train"], args.seed)
        if args.balance_samples else None
    )
    loaders = {}
    for split_name, dataset in datasets.items():
        sampler = train_sampler if split_name == "train" else None
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=split_name == "train" and sampler is None,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=use_cuda,
            prefetch_factor=4 if args.num_workers > 0 else None,
            drop_last=False,
        )

    scaler_hparams = {
        f"{name}_{stat}": input_scaler[name][stat]
        for name in ("tf_expression", "tg_expression", "peak_accessibility")
        for stat in ("mean", "std")
    }
    
    module = LitTFTGRegulationModel(**{key: getattr(args, key) for key in (
        "d_model", "num_heads", "dropout", "lr", "weight_decay",
        "pooling_temperature", "pos_weight", "plateau_patience")},
        **scaler_hparams)
    
    if args.wandb_mode == "disabled":
        logger = CSVLogger(str(args.output_dir), name="metrics")
    else:
        import wandb
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            save_dir=str(args.output_dir),
            offline=args.wandb_mode == "offline",
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            log_model=False,
            save_code=True,
        )

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
        # A full step-zero validation below replaces Lightning's two-batch sanity check.
        num_sanity_val_steps=0 if args.resume_from_checkpoint is None else 2,
        fast_dev_run=args.fast_dev_run
    )
    
    try:
        if args.resume_from_checkpoint is None:
            logging.info("Evaluating the untrained model on the validation set at step 0")
            untrained_results = trainer.validate(
                module,
                dataloaders=loaders["val"],
                verbose=True,
            )
            untrained_metrics = {
                key: float(value)
                for key, value in (untrained_results[0] if untrained_results else {}).items()
            }
            (args.output_dir / "untrained_val_metrics.json").write_text(
                json.dumps(untrained_metrics, indent=2)
            )

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
