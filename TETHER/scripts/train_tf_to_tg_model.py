
import math
import os
import sys
import gtfparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
import warnings
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
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies import DDPStrategy

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
sys.path.append(str(PROJECT_DIR))

# The frozen TF-DNA submodule is compiled (see create_new_tf_tg_regulation_model), and
# Dynamo caches one entry per distinct input shape for it. Torch's default limit is 8.
#
# The TF crop ladder alone produces 4 distinct widths per species, so the default leaves
# almost no headroom, and nothing announces itself when it runs out: the graphs evict each
# other, every batch recompiles, and throughput just oscillates instead of erroring. Two
# ordinary config changes would cross the line on their own -- raising
# --tf_peak_chunk_size above CHUNK_QUANTUM (256), which unpins the chunk width, or dropping
# --keep_tf_dna_in_eval, which switches to the ragged-final-chunk path and adds a second
# shape per crop.
#
# 128 matches generate_all_predictions.py, so training and inference have the same budget.
# Raising the limit only permits more cached graphs; it does not create them.
torch._dynamo.config.cache_size_limit = 128

import models.tf_to_tg_testing as tf_to_tg_module
import models.tf_to_dna as tf_to_dna_module
import config
# Same sampler the TF-DNA trainer uses, applied to peak counts instead of protein
# lengths. It is generic over `lengths`, so it is imported rather than duplicated.
from scripts.batch_samplers import LengthGroupedBatchSampler, dataloader_worker_init

# NOTE: utils is deliberately NOT imported at module level. utils.py does
#     from scripts.train_tf_to_tg_model import TFTGEdgeBagDataset, collate_tftg_edge_bags
# so importing it from here is circular and breaks every consumer of this module
# (generate_all_predictions.py, plot_auprc_all_methods.py, ...). The same loop is why
# LengthGroupedBatchSampler lives in scripts/batch_samplers.py rather than in the TF-DNA
# trainer, which imports utils itself. get_rank_info is imported locally where used.
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

warnings.filterwarnings(
    "ignore",
    message="This DataLoader will create .* worker processes in total\.",
    category=UserWarning,
    module="torch.utils.data.dataloader",
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

@rank_zero_only
def log_once(msg: str) -> None:
    logging.info(msg)


def create_new_tf_tg_regulation_model(
    tf_bind_model_path: Path,
    tf_embeddings_tensor: torch.Tensor,
    tf_mask_tensor: torch.Tensor,
    checkpoint_path: Path | None = None,
    d_model: int = 128,
    tf_peak_chunk_size: int = 128,
    keep_tf_peak_model_in_eval: bool = False,
) -> tf_to_tg_module.TFTGRegulationModel:

    # 1) Recreate the base TF→DNA model with the same hyperparameters
    tf_dna_hidden_dim = 128  # matches base_model's hidden_dim below
    base_model = tf_to_dna_module.TFPeakBindingModel(
        tf_embedding_dim=128,
        hidden_dim=tf_dna_hidden_dim,
        dropout=0.3,
        num_layers=4,
        num_heads=4,
        dim_head=32,
    )

    # 2) Wrap in Lightning module and load checkpoint
    lit_model = tf_to_dna_module.LitTFPeakBindingModel.load_from_checkpoint(
        checkpoint_path=tf_bind_model_path,
        model=base_model,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
    )

    # 3) Get the trained base model and freeze it
    trained_tf_peak_model = lit_model.model

    trained_tf_peak_model.eval()

    for p in trained_tf_peak_model.parameters():
        p.requires_grad = False

    # Plain torch.compile, matching utils.load_tf_{dna,tg}_model on the inference path.
    #
    # No mode="reduce-overhead". That enables CUDA graphs, which re-record whenever an
    # input shape reappears after eviction. TFTGRegulationModel deliberately produces
    # several shapes (one per TF crop width x chunk count), and measured against plain
    # compile on TF-major batches the median was 1.9x worse while p90 was 14.6x worse
    # (2653 ms vs 181 ms) -- the tail, not the median, is what a full run pays. Default
    # mode: 94 ms median / 181 ms p90.
    #
    # Inference was fixed in f440ef2 ("Remove CUDA graphs from inference and fix the
    # recompile storm they masked"); that commit edited this file but not this call, so
    # training kept re-recording graphs for another two weeks. The two paths compile the
    # same submodule over the same shapes, so they should compile it the same way.
    trained_tf_peak_model = torch.compile(trained_tf_peak_model)

    # 4) Inject into your TF→TG model
    tf_tg_model = tf_to_tg_module.TFTGRegulationModel(
        pretrained_tf_peak_model=trained_tf_peak_model,
        d_model=d_model,
        tf_peak_chunk_size=tf_peak_chunk_size,
        tf_binding_hidden_dim=tf_dna_hidden_dim // 2,
        keep_tf_peak_model_in_eval=keep_tf_peak_model_in_eval,
    )
    logging.info(
        "Frozen TF-DNA submodule will run in "
        + ("EVAL mode during training (running BatchNorm stats, fast path enabled)"
           if keep_tf_peak_model_in_eval else
           "TRAIN mode during training (batch BatchNorm stats -- the historical default)")
    )

    # 5) Optionally load TF→TG checkpoint
    if checkpoint_path is not None:
        logging.info(f"Loading TF→TG model weights from checkpoint: {checkpoint_path}")

        tf_tg_ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        fixed = {}

        for key, value in tf_tg_ckpt["state_dict"].items():
            if key.startswith("model."):
                key = key[len("model."):]
            fixed[key] = value

        tf_tg_model.load_state_dict(fixed, strict=True)

    return tf_tg_model


# Keys in the cached dict indexed by TG (gather via tg_idx), not by edge -- row-filtering
# an edge subset must never touch these; every kept edge's tg_idx still points at the same,
# unchanged TG row. Listed explicitly rather than relying only on the length check in
# _take_rows below, which would happen to also skip them (len(v) == n_tg != n_edges) but
# silently, with no signal if a future reshape ever made that coincidentally untrue.
TG_LEVEL_KEYS = {"tg_peak_indices", "tg_peak_distance", "tg_peak_mask"}


def _take_rows(inputs, keep_idx, n_edges):
    """Select `keep_idx` rows from every per-edge entry in `inputs`.

    Shared by stratified_train_subsample and filter_low_positive_tfs below -- both filter
    the same dict of per-edge tensors/lists at the same point in the pipeline (after load,
    before TFTGEdgeBagDataset construction). TG-level tables (TG_LEVEL_KEYS) pass through
    untouched -- every kept edge's tg_idx still indexes the same TG rows.
    """
    keep_list = keep_idx.tolist() if torch.is_tensor(keep_idx) else list(keep_idx)
    keep_idx_t = keep_idx if torch.is_tensor(keep_idx) else torch.as_tensor(keep_idx)

    def _take(k, v):
        if k in TG_LEVEL_KEYS:
            return v
        # Every remaining entry in this cache is per-edge, but three of them (tf_name,
        # tg_name, cell_ids) are plain lists and do not accept fancy indexing.
        if not hasattr(v, "__len__") or len(v) != n_edges:
            return v
        if isinstance(v, list):
            return [v[i] for i in keep_list]
        return v[keep_idx_t]

    return {k: _take(k, v) for k, v in inputs.items()}


def stratified_train_subsample(train_inputs, frac, seed=0):
    """Indices for a per-(TF, label) stratified subsample of the training edges.

    Stratifying on the TF ALONE is not enough, and the failure mode is specific: at frac=0.25
    a TF-only draw left two of the 82 TFs with zero positive edges, so --per_tf_pos_weight
    fell back to w=1.0 for them, and the median weight moved 19.29 -> 15.01 (weight
    correlation against full data only 0.919). Those are precisely the rare-positive TFs the
    weighting exists to correct, so a cap sweep run on such a subsample would be comparing
    the wrong thing.

    Drawing the same fraction of each TF's positives and negatives separately keeps every
    TF's positive COUNT proportional, so w_t = n_neg_t / n_pos_t is preserved to within
    rounding and no TF can lose its positives entirely.

    Returned indices are SORTED. The edge universe is TF-major and the model's TF-crop
    ladder re-records a CUDA graph whenever the crop width changes, so a shuffled subset
    changes crop almost every batch and costs 15-38 s/batch.
    """
    tf_idx = train_inputs["tf_idx"].reshape(-1).numpy()
    labels = train_inputs["label"].reshape(-1).numpy()
    rng = np.random.default_rng(seed)

    keep = []
    for t in np.unique(tf_idx):
        for lab in (0, 1):
            pos = np.flatnonzero((tf_idx == t) & (labels == lab))
            if len(pos) == 0:
                continue
            n = max(1, int(round(len(pos) * frac)))
            keep.append(rng.choice(pos, size=n, replace=False))
    keep = np.sort(np.concatenate(keep))

    log_once(
        f"--train_subsample_frac {frac:g}: {len(keep):,} of {len(tf_idx):,} training edges "
        f"({len(keep)/len(tf_idx):.1%}) across {len(np.unique(tf_idx))} TFs, stratified by "
        f"(TF, label). Positive rate {labels.mean():.4f} -> {labels[keep].mean():.4f}."
    )
    return keep


def compute_per_tf_pos_weight(train_inputs, n_tf, max_weight=10.0):
    """Per-TF positive class weight w_t = n_neg_t / n_pos_t over the whole training split.

    Global counts, deliberately, not per-batch counts. A weight derived from the batch
    would make an edge's contribution depend on which other edges happened to land beside
    it, so the same edge carries a different weight in different batches and the gradient
    picks up a bias (the expectation of a ratio is not the ratio of the expectations).
    These counts are a fixed property of the split, computed once here.

    TFs absent from the training split, or with no positives, or with no negatives, get
    weight 1.0 -- there is nothing to balance and w would be 0 or undefined.

    max_weight caps the correction. Positive rates run down to ~4e-4, which would otherwise
    ask for w ~= 2500 and let a handful of positive edges dominate the gradient. At the
    median rate (~6%) the uncapped weight is ~16, so even a cap of 10 leaves most of the
    distribution under-corrected relative to the uncapped ratio -- deliberately: a TF whose
    positive rate is this extreme is better handled by dropping it via
    --min_tf_positive_count (see filter_low_positive_tfs below) than by asking a single
    gradient multiplier to both balance its classes AND dominate the batch. 50 was measured
    to let a handful of near-empty-positive TFs swamp the gradient early in training; 10
    still fully corrects any TF down to a 10% positive rate.
    """
    labels = train_inputs["label"].reshape(-1).float()
    tf_idx = train_inputs["tf_idx"].reshape(-1).long()

    n_pos = torch.zeros(n_tf, dtype=torch.float64).index_add_(0, tf_idx, labels.double())
    n_all = torch.zeros(n_tf, dtype=torch.float64).index_add_(
        0, tf_idx, torch.ones_like(labels, dtype=torch.float64)
    )
    n_neg = n_all - n_pos

    weights = torch.ones(n_tf, dtype=torch.float32)
    ok = (n_pos > 0) & (n_neg > 0)
    weights[ok] = (n_neg[ok] / n_pos[ok]).clamp(max=max_weight).float()

    present = n_all > 0
    capped = ok & ((n_neg / n_pos.clamp(min=1)) > max_weight)
    rates = (n_pos[present] / n_all[present]).numpy()
    log_once(
        f"--per_tf_pos_weight: {int(present.sum())} TFs in the training split, "
        f"{int(ok.sum())} balanceable, {int(capped.sum())} capped at {max_weight:g}. "
        f"Positive rate min={rates.min():.4f} median={np.median(rates):.4f} "
        f"max={rates.max():.4f}; weight median={weights[ok].median():.2f} "
        f"max={weights[ok].max():.2f}."
    )
    return weights


def filter_low_positive_tfs(train_inputs, n_tf, min_positive_count):
    """Drop TRAINING rows for any TF with fewer than min_positive_count positives.

    A TF with too few positives to estimate a stable pos_weight also has too few to learn
    a reliable per-TF boundary from -- row-exclusion is no worse than down-weighting it,
    and it keeps compute_per_tf_pos_weight's counts drawn from the split actually trained
    on. Only the train split is filtered; val/test keep every row so a dropped TF's
    validation AUROC still surfaces rather than being hidden.
    """
    tf_idx = train_inputs["tf_idx"].reshape(-1).long()
    labels = train_inputs["label"].reshape(-1).float()

    n_pos = torch.zeros(n_tf, dtype=torch.float64).index_add_(0, tf_idx, labels.double())
    keep_tfs = (n_pos >= min_positive_count).nonzero(as_tuple=True)[0]
    keep_idx = torch.isin(tf_idx, keep_tfs).nonzero(as_tuple=True)[0]

    n_dropped_tfs = int((n_pos < min_positive_count).sum())
    log_once(
        f"--min_tf_positive_count {min_positive_count}: dropping {n_dropped_tfs} TF(s) "
        f"with fewer than {min_positive_count} positive training edges, removing "
        f"{len(tf_idx) - len(keep_idx):,} of {len(tf_idx):,} training rows "
        f"({(len(tf_idx) - len(keep_idx)) / max(1, len(tf_idx)):.1%})."
    )
    return keep_idx


class TFTGEdgeBagDataset(Dataset):
    """
    One item per TF-TG edge bag.

    `return_tf_indices` controls how the TF protein embedding reaches the model:

      False (default, unchanged): the full [T, D] embedding is gathered here and
        travels through the collate function and the host-to-device copy for every
        edge. With T ~= 4000 and a batch of 512 that is ~1 GB of pinned transfer per
        batch, nearly all of it duplicate -- a batch references only a few hundred
        distinct TFs, and the whole table is under 2 GB.

      True: only `tf_idx` is returned, and the model gathers from an embedding table
        held resident on the device (see TFTGRegulationModel.set_tf_embedding_table).
        Mathematically identical, but the gather happens in device memory instead of
        across PCIe.

    peak_indices/peak_distance/peak_mask are stored once per TG (`inputs["tg_peak_*"]`,
    gathered here via tg_idx), and peak_accessibility/tf_expression/tg_expression are never
    materialized in the cache at all -- only `inputs["cell_indices"]` (the C sampled cell
    columns for that edge) is, and the actual values are gathered here from atac_mat/rna_mat.
    Both are therefore required unconditionally now, not just for resample_cells=True.

    `resample_cells` controls WHICH cell columns are used:

      False (default, unchanged): use the fixed `inputs["cell_indices"][idx]` columns
        chosen once at build time by build_tf_to_tg_train_data.py, i.e. the same cells for
        a given edge on every epoch of every run. This is what every existing checkpoint
        was trained under, and produces bit-identical accessibility/expression values to
        the old fully-materialized cache -- only the storage changed.

      True: redraw `resample_max_cells_per_pair` cell columns from the full atac_mat/rna_mat
        pseudobulk matrices on every __getitem__ call, so a given edge sees a different cell
        subset epoch to epoch (and even within an epoch, across workers). Peaks/labels/TF
        embedding are untouched -- only which cells represent the edge changes. Clamps
        resample_max_cells_per_pair to the available cell pool so every item has the same C
        (no per-item cell padding needed).
    """

    def __init__(
        self,
        inputs,
        *,
        tf_embeddings_tensor,
        tf_mask_tensor,
        atac_peak_tensor,
        atac_mat,
        rna_mat,
        gene_to_rna_idx,
        return_tf_indices=False,
        resample_cells=False,
        idx_to_cell=None,
        resample_max_cells_per_pair=None,
    ):
        self.inputs = inputs
        self.tf_embeddings_tensor = tf_embeddings_tensor
        self.tf_mask_tensor = tf_mask_tensor
        self.atac_peak_tensor = atac_peak_tensor
        self.return_tf_indices = return_tf_indices

        # torch.as_tensor, not a bare assignment: some callers hand this a numpy array
        # fresh out of prepare_tftg_lookup_tables (a cache-miss build), others a tensor
        # already round-tripped through torch.load (a cache hit / the training cache).
        # Indexing a numpy array with a length-1 torch LongTensor silently collapses to a
        # scalar select (numpy treats it as __index__-able) instead of preserving the row
        # as a [1, n_cells] slice the way indexing a torch tensor does -- and edges with
        # exactly one real peak are common (this build's own log reported "min 1 real
        # peaks per edge"). Normalizing here means _gather_cell_features never has to care
        # which kind of array-like it was handed.
        self.atac_mat = torch.as_tensor(atac_mat, dtype=torch.float32)
        self.rna_mat = torch.as_tensor(rna_mat, dtype=torch.float32)
        self.gene_to_rna_idx = gene_to_rna_idx
        self.idx_to_cell = idx_to_cell

        self.resample_cells = resample_cells
        if resample_cells:
            n_pool = self.atac_mat.shape[1]
            requested = resample_max_cells_per_pair or n_pool
            self.max_cells_per_pair = min(requested, n_pool)
            if requested > n_pool:
                logging.warning(
                    f"resample_cells_per_epoch requested {requested} cells/edge but only "
                    f"{n_pool} are in the pool; clamping to {n_pool}."
                )
            # Lazily created per-process so forked DataLoader workers each get an
            # independently-seeded stream (numpy pulls fresh OS entropy with no seed arg)
            # instead of all workers replaying the same draws.
            self._rng = None

    def __len__(self):
        return len(self.inputs["label"])

    def _gather_cell_features(self, peak_indices, peak_mask, cell_cols, tf_name, tg_name):
        """peak_accessibility/tf_expression/tg_expression for one edge, from atac_mat/rna_mat.

        Shared by the default (fixed inputs["cell_indices"]) and resample_cells (freshly
        redrawn every call) paths -- identical computation, only the source of cell_cols
        differs.
        """
        real_peak_rows = peak_indices[peak_mask]   # [n_real], long

        C = cell_cols.shape[0]
        P = peak_indices.shape[0]
        peak_accessibility = torch.zeros(C, P, dtype=torch.float32)
        acc_real = self.atac_mat[real_peak_rows][:, cell_cols]   # [n_real, C]
        peak_accessibility[:, : acc_real.shape[0]] = acc_real.T

        tf_rna_idx = self.gene_to_rna_idx[tf_name]
        tg_rna_idx = self.gene_to_rna_idx[tg_name]
        tf_expression = self.rna_mat[tf_rna_idx, cell_cols]      # [C]
        tg_expression = self.rna_mat[tg_rna_idx, cell_cols]      # [C]

        return peak_accessibility, tf_expression, tg_expression

    def _resample_cell_features(self, idx, peak_indices, peak_mask):
        if self._rng is None:
            self._rng = np.random.default_rng()

        n_pool = self.atac_mat.shape[1]
        C = self.max_cells_per_pair
        sampled_cols = self._rng.choice(n_pool, size=C, replace=False)
        cell_cols = torch.from_numpy(sampled_cols).long()

        peak_accessibility, tf_expression, tg_expression = self._gather_cell_features(
            peak_indices, peak_mask, cell_cols,
            self.inputs["tf_name"][idx], self.inputs["tg_name"][idx],
        )

        cell_ids = (
            [self.idx_to_cell[c] for c in sampled_cols.tolist()]
            if self.idx_to_cell is not None else None
        )

        return peak_accessibility, tf_expression, tg_expression, cell_ids

    def __getitem__(self, idx):
        tf_idx = self.inputs["tf_idx"][idx]
        tg_idx = self.inputs["tg_idx"][idx]

        peak_indices = self.inputs["tg_peak_indices"][tg_idx]    # [P]
        peak_mask = self.inputs["tg_peak_mask"][tg_idx]          # [P]
        peak_sequences = self.atac_peak_tensor[peak_indices]     # [P, L, 4]

        if self.resample_cells:
            peak_accessibility, tf_expression, tg_expression, cell_ids = (
                self._resample_cell_features(idx, peak_indices, peak_mask)
            )
        else:
            cell_cols = self.inputs["cell_indices"][idx]         # [C]
            peak_accessibility, tf_expression, tg_expression = self._gather_cell_features(
                peak_indices, peak_mask, cell_cols,
                self.inputs["tf_name"][idx], self.inputs["tg_name"][idx],
            )
            cell_ids = self.inputs["cell_ids"][idx]

        item = {
            "label": self.inputs["label"][idx],
            "tf_name": self.inputs["tf_name"][idx],
            "tg_name": self.inputs["tg_name"][idx],
            "cell_ids": cell_ids,
            "tf_idx": tf_idx,
            "tg_idx": tg_idx,
            "peak_indices": peak_indices,
            "peak_sequences": peak_sequences,
            "peak_distance": self.inputs["tg_peak_distance"][tg_idx].float(),
            "peak_mask": peak_mask.bool(),
            "peak_accessibility": peak_accessibility.float(),
            "tf_expression": tf_expression.float(),
            "tg_expression": tg_expression.float(),
        }

        if not self.return_tf_indices:
            item["tf_embedding"] = self.tf_embeddings_tensor[tf_idx].float()   # [T, D]
            item["tf_mask"] = self.tf_mask_tensor[tf_idx].bool()               # [T]

        return item
        
class ResidentTFEmbeddingTable(Callback):
    """Pin the TF embedding table to each rank's GPU for the tf_idx gather path.

    Registered from a callback rather than before `trainer.fit` because
    `set_tf_embedding_table` places the table on `next(model.parameters()).device`, and
    the model is still on CPU until the strategy sets it up. Hooking the *_start events
    runs after device placement and after the DDP wrap, so each rank pins its own copy.

    The table is a non-persistent buffer that is None at wrap time, so it stays out of
    DDP's buffer list and is never broadcast -- it is identical, read-only, and derived
    from the same cache file on every rank, so there is nothing to synchronise.
    """

    def __init__(self, tf_embeddings_tensor, tf_mask_tensor):
        self.tf_embeddings_tensor = tf_embeddings_tensor
        self.tf_mask_tensor = tf_mask_tensor

    def _register(self, pl_module):
        if getattr(pl_module.model, "tf_embedding_table", None) is None:
            pl_module.model.set_tf_embedding_table(
                self.tf_embeddings_tensor, self.tf_mask_tensor
            )

    def on_fit_start(self, trainer, pl_module):
        self._register(pl_module)

    def on_validation_start(self, trainer, pl_module):
        self._register(pl_module)

    def on_test_start(self, trainer, pl_module):
        self._register(pl_module)

    def on_predict_start(self, trainer, pl_module):
        self._register(pl_module)


# Crop ladder for the peak axis, the same idea as models/tf_to_dna.TF_CROP_LADDER applied
# to a different axis. Bags are padded to the widest TG in the split (P = 90-100 on the
# mESC reps) while the median edge has 3 real peaks, so most of every bag is padding.
#
# Cropping to the exact batch maximum saves the most, but produces a new tensor shape per
# batch: measured over one epoch of E7.5_rep1 at batch 256 it gave 22 distinct widths
# (25 for E8.5_rep1). That is the condition both TF_CROP_LADDERs exist to avoid -- each new
# shape costs an Inductor compile and a CUDA-graph recording, and the peak width multiplies
# against (crop, n_chunks) rather than adding to it.
#
# These rungs cut that to at most 6 shapes for mean width 9.0 against 6.4 exact -- roughly
# 40% of the trim's saving given back, still ~10x narrower than the padded width. Geometric
# rather than even because the distribution is heavily skewed (median 3, max 90).
PEAK_CROP_LADDER = (4, 8, 16, 32, 64)




def collate_tftg_edge_bags(batch):
    output = {
        "label": torch.stack([b["label"] for b in batch]).float(),

        "tf_idx": torch.stack([b["tf_idx"] for b in batch]).long(),
        "tg_idx": torch.stack([b["tg_idx"] for b in batch]).long(),

        "peak_indices": torch.stack([b["peak_indices"] for b in batch]),
        "peak_sequences": torch.stack([b["peak_sequences"] for b in batch]),
        "peak_distance": torch.stack([b["peak_distance"] for b in batch]),
        "peak_mask": torch.stack([b["peak_mask"] for b in batch]),

        "peak_accessibility": torch.stack([b["peak_accessibility"] for b in batch]),
        "tf_expression": torch.stack([b["tf_expression"] for b in batch]),
        "tg_expression": torch.stack([b["tg_expression"] for b in batch]),

        "tf_name": [b["tf_name"] for b in batch],
        "tg_name": [b["tg_name"] for b in batch],
        "cell_ids": [b["cell_ids"] for b in batch],
    }

    # Absent when the dataset was built with return_tf_indices=True, in which case the
    # model gathers the embeddings itself from its resident table using tf_idx.
    if "tf_embedding" in batch[0]:
        output["tf_embedding"] = torch.stack([b["tf_embedding"] for b in batch])
        output["tf_mask"] = torch.stack([b["tf_mask"] for b in batch])

    E, C = output["tf_expression"].shape
    output["cell_mask"] = torch.ones(E, C, dtype=torch.bool)

    # Trim the padded peak axis to what this batch actually uses.
    #
    # Bags are padded to the widest TG in the whole split (P = 90-100 on the mESC reps)
    # while the median edge has 3 real peaks, so ~97% of every peak slot is padding.
    # key_padding_mask makes the attention ignore those slots but does not make them
    # cheap: peak_feature_proj and MultiheadAttention still materialise [E*C, P, d_model],
    # and the padded peak_sequences are still shipped host->device.
    #
    # Cutting to a rung at or above the last column any edge actually uses is exact -- the
    # dropped columns are padding for every row, so nothing that could have been attended
    # to is removed. This is a no-op unless batches are peak-homogeneous, which is what
    # LengthGroupedBatchSampler below arranges; under uniform shuffling one wide edge
    # drags the whole batch back to full width.
    used_columns = output["peak_mask"].any(dim=0)
    if bool(used_columns.any()):
        used = int(torch.nonzero(used_columns)[-1]) + 1
        padded = output["peak_mask"].shape[1]
        # Round the used width UP to a ladder rung, never down -- a rung below `used`
        # would drop a real peak.
        width = min(next((rung for rung in PEAK_CROP_LADDER if rung >= used), padded), padded)
        if width < padded:
            for key in ("peak_indices", "peak_sequences", "peak_distance", "peak_mask"):
                output[key] = output[key][:, :width].contiguous()
            # [E, C, P] -- the peak axis is last here.
            output["peak_accessibility"] = output["peak_accessibility"][:, :, :width].contiguous()

    return output

if __name__ == "__main__":
    
    """
    Need arguments for:
    
    sample_pairs: int | None
    max_peaks_per_tg: int | None
    max_cells_per_pair: int | None
    batch_size: int
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPU devices to use for training")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use for training")
    parser.add_argument("--job_id", type=str, help="SLURM job ID for this training run")
    parser.add_argument("--sample_pairs", type=int, default=None, help="Number of TF-TG pairs to sample for training (default: use all)")
    parser.add_argument(
        "--keep_tf_dna_in_eval",
        action="store_true",
        help=(
            "Keep the frozen TF-DNA submodule in eval mode for the whole run, so its "
            "BatchNorm layers use their running statistics instead of per-batch ones and "
            "stop mutating those statistics. This is how a frozen feature extractor is "
            "normally used and it removes a train/inference mismatch, but it changes the "
            "binding scores the TF-TG model trains against (mean 1.14 logits, 2.1%% of "
            "pairs crossing p=0.5), so checkpoints trained with it are NOT comparable to "
            "existing ones. Also enables the padding-skip/crop fast path: 1822 -> 311 "
            "ms/step measured at max_peaks_per_tg=100, max_cells_per_pair=24, batch 32."
        ),
    )
    parser.add_argument("--max_peaks_per_tg", type=int, required=False, default=None, help="Maximum number of peaks to consider per TG (default: 64)")
    parser.add_argument("--max_cells_per_pair", type=int, default=8, help="Maximum number of cells to sample per TF-TG pair (default: 8)")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Base learning rate. Left unset, it is scaled from the reference point this "
            "model was tuned at (1e-4 at an effective batch of 32, i.e. batch_size 8 on 4 "
            "GPUs) according to --lr_scale_rule. Pass a value to pin it."
        ),
    )
    parser.add_argument(
        "--lr_scale_rule",
        choices=["sqrt", "linear", "none"],
        default="none",
        help=(
            "How to scale the reference LR by effective batch (batch_size * gpus * nodes). "
            "Defaults to 'none' so callers that do not pass it keep the historical fixed "
            "1e-4 -- 03b_train_tf_to_tg_model.sh, wandb_sweep.py and the stability scripts "
            "all share this entry point, and a silent LR change would not be obvious there. "
            "'sqrt' multiplies by sqrt(ratio) -- the usual choice for Adam-family "
            "optimizers, where gradient noise falls with sqrt(batch). 'linear' multiplies "
            "by the ratio, which is the SGD result and tends to overshoot with AdamW. "
            "'none' keeps the reference LR. Ignored when --lr is given."
        ),
    )
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=0.0,
        help=(
            "Epochs spent linearly warming the LR from ~0 to its base value. 0 disables "
            "warmup (historical behaviour). Worth turning on whenever the LR is scaled up: "
            "ReduceLROnPlateau only reacts after val/loss stalls, so it cannot protect the "
            "early steps where a large batch at a high LR diverges. Counted in epochs "
            "rather than a fraction of the run because EarlyStopping usually ends training "
            "well before --epochs, which would make a percentage-of-total warmup arbitrary. "
            "1.0 is a reasonable starting point."
        ),
    )
    parser.add_argument(
        "--precision",
        default="16-mixed",
        help=(
            "Lightning precision. Default '16-mixed' (fp16) is unchanged. On Ampere or "
            "newer (A100), prefer 'bf16-mixed': fp16 saturates at ~65504, and GradScaler "
            "only rescues gradient overflow, not forward activations -- an overflow there "
            "becomes inf then NaN, which is how run 3788646 died at epoch 7. bf16 carries "
            "fp32's exponent range at the same speed, removing that failure mode. Not "
            "supported on V100."
        ),
    )
    parser.add_argument(
        "--selection_metric",
        default="val/macro_auroc",
        help=(
            "Metric ModelCheckpoint and EarlyStopping both use (mode=max). Keep this equal "
            "to --plateau_monitor: run 3809742 stopped on pooled val/auroc while its LR "
            "schedule tracked macro, and died at epoch 21 with macro still improving."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for LengthGroupedBatchSampler's per-epoch shuffle. Every rank must use "
             "the same value: ranks build the identical batch list and take a strided "
             "slice of it, so a differing seed would give them overlapping edges.",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument(
        "--train_subsample_frac",
        type=float,
        default=1.0,
        help=(
            "Train on this fraction of the training edges (1.0 = all, the default). "
            "Stratified per TF so each TF keeps the same share of its edges and its "
            "positive rate -- and therefore its --per_tf_pos_weight value -- is preserved. "
            "Validation and test are never subsampled, so metrics stay comparable to full "
            "runs. Indices are kept sorted: the edge universe is TF-major and the TF-crop "
            "ladder re-records a CUDA graph whenever the crop width changes, so a scrambled "
            "order costs 15-38 s/batch. Intended for hyperparameter screening -- at 0.25 an "
            "epoch costs ~12.6 min instead of ~50.5. Always include a known configuration "
            "as a control arm: a subsampled run is a proxy, and it is only trustworthy for "
            "RANKING configurations if the control lands near its full-data trajectory."
        ),
    )
    parser.add_argument(
        "--plateau_monitor",
        default="val/auroc",
        help=(
            "Metric ReduceLROnPlateau watches. Default val/macro_auroc. Do NOT set this to "
            "val/loss: on run 3801811 val/loss bottomed at epoch 0 and never recovered, so "
            "the schedule cut the LR 10x at epoch 6 and again at 15 while macro AUROC was "
            "still climbing, flattening the run for its last twelve epochs."
        ),
    )
    parser.add_argument(
        "--plateau_mode", default="max", choices=["min", "max"],
        help="'max' for AUROC-like monitors, 'min' for loss-like ones.",
    )
    parser.add_argument(
        "--plateau_factor", type=float, default=0.5,
        help="LR multiplier on plateau. Was 0.1; a 10x cut is a decision, not an adjustment.",
    )
    parser.add_argument("--plateau_patience", type=int, default=4,
        help="Was 8; on a metric as noisy as macro AUROC the counter was reset by "
             "chance upticks before it could ever reach the threshold.")
    parser.add_argument("--plateau_cooldown", type=int, default=2)
    parser.add_argument(
        "--per_tf_pos_weight",
        action="store_true",
        help=(
            "Weight the positive class per TF (w_t = n_neg_t / n_pos_t over the training "
            "split) instead of leaving BCE unweighted. Equalises every TF's effective "
            "positive rate to 0.5, which removes the incentive to encode a per-TF logit "
            "offset -- the between-TF shortcut that gained run 3799581 +0.052 pooled AUROC "
            "on held-out TFs while macro moved only +0.004. Off by default; changes the "
            "training objective, so val/loss is not comparable to unflagged runs."
        ),
    )
    parser.add_argument(
        "--per_tf_pos_weight_max",
        type=float,
        default=10.0,
        help=(
            "Cap on the per-TF positive weight (--per_tf_pos_weight only). Positive rates "
            "reach ~4e-4, which uncapped would ask for a weight of ~2500 and let a few "
            "edges dominate the gradient. A TF whose positive rate is that extreme is "
            "better excluded via --min_tf_positive_count than corrected by raising this cap."
        ),
    )
    parser.add_argument(
        "--min_tf_positive_count",
        type=int,
        default=0,
        help=(
            "Drop training rows for any TF with fewer than this many positive training "
            "edges (0 disables). Complements --per_tf_pos_weight_max: a lower cap "
            "under-corrects TFs whose positive rate is too extreme to balance usefully, "
            "so drop them instead of asking the loss to compensate."
        ),
    )
    parser.add_argument(
        "--eval_in_training_precision",
        action="store_true",
        help=(
            "Score val/test under the same autocast as training instead of fp32. Only "
            "for reproducing a pre-fix run's numbers. bf16 costs two mantissa bits vs "
            "fp16, and the resulting tied logits corrupt AUROC by an amount that GROWS "
            "with training (0.009 at epoch 0, 0.042 by epoch 5 on run 3793729) -- which "
            "made an improving model log a falling curve and made ModelCheckpoint keep "
            "epoch 0 over a better epoch 5. Leave this off unless you need the old "
            "numbers back."
        ),
    )
    parser.add_argument(
        "--tf_embedding_on_device",
        action="store_true",
        help=(
            "Keep the TF protein embedding table resident on each GPU and gather from it by "
            "tf_idx, instead of the dataloader shipping a full [T, D] embedding with every "
            "edge. Mathematically identical (same gather, different place), but the per-edge "
            "copy is 2.86 MB against ~102 KB for the rest of the item, so it dominates "
            "collate and host-to-device transfer -- and it scales with batch size, which "
            "otherwise makes a larger --batch_size counterproductive. Costs ~1.3 GB of GPU "
            "memory per rank."
        ),
    )
    parser.add_argument(
        "--resample_cells_per_epoch",
        action="store_true",
        help=(
            "Instead of reusing the fixed per-edge cell bag baked into the cache (the "
            "historical default, used by every existing checkpoint), redraw "
            "--max_cells_per_pair cells for each training edge from the full pseudobulk "
            "pool on every access -- a different subset each epoch. Only affects the "
            "training set; validation/test keep the frozen cached bags so val metrics stay "
            "comparable across epochs. Requires atac_mat.pt/rna_mat.pt, built by "
            "build_tf_to_tg_train_data.py --build_resample_matrices_only."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--pct_true_edges", type=float, default=0.15, help="Percentage of true edges to include in the training set (default: 0.15)")
    parser.add_argument("--true_false_ratio", type=float, default=2.0, help="Ratio of true to false edges in the training set (default: 2.0)")
    parser.add_argument("--peak_flank_size", type=int, default=128, help="Size of the flank region around peaks (default: 128)")
    parser.add_argument("--checkpoint_path", type=str, required=False,
        help="Path to a checkpoint to WARM-START model weights from. Loads weights only -- "
             "optimizer state, epoch counter, LR schedule and callback state all start "
             "fresh, and warmup re-runs. For continuing an interrupted run use "
             "--resume_from_checkpoint instead.")
    parser.add_argument("--resume_from_checkpoint", type=str, required=False,
        help="Path to a checkpoint to genuinely RESUME from (passed to trainer.fit(ckpt_path=...)). "
             "Restores optimizer moments, epoch counter, LR scheduler and callback state, so "
             "training continues as though it had never stopped. This is what you want after a "
             "run dies partway -- warm-starting weights instead would reset the optimizer and "
             "re-run warmup, which is not comparable to a run that trained continuously.")
    parser.add_argument("--force_reload", action="store_true", help="Whether to force reload cached data instead of using existing cache files")
    args = parser.parse_args()

    epochs = args.epochs
    num_gpus = args.num_gpus
    num_nodes = args.num_nodes
    job_id = args.job_id
    checkpoint_path = args.checkpoint_path
    force_reload = args.force_reload
    batch_size = args.batch_size
    sample_pairs = args.sample_pairs
    max_peaks_per_tg = args.max_peaks_per_tg
    max_cells_per_pair = args.max_cells_per_pair
    pct_true_edges = args.pct_true_edges
    true_false_ratio = args.true_false_ratio
    peak_flank_size = args.peak_flank_size
    
    assert config.cell_type in config.tf_dna_model_checkpoints, \
        f"Cell type '{config.cell_type}' not found in TF→DNA model checkpoints."
    
    tf_bind_model_path = config.tf_dna_model_checkpoints[config.cell_type] 

    sample_name = config.sample_name
    
    output_dir = PROJECT_DIR / "checkpoints" / f"{config.cell_type}" / f"{sample_name}" / f"tf_tg_train_{sample_name}_{job_id}"
    
    run_name = f"tf_tg_{sample_name}_{job_id}"
    
    # Load the trained TF embedding and mask tensors from the TF→DNA model cache 
    # (these are needed for the TF→TG model since it uses the pretrained TF peak embedding module)
    tf_embeddings_tensor = torch.load(
        config.tf_embedding_cache_path,
        weights_only=True,
    )
    tf_mask_tensor = torch.load(
        config.tf_mask_cache_path,
        weights_only=True,
    )
    
    # Load the train/val/test splits of the compact TF-TG input tensors 
    # that were preprocessed and cached by the data preprocessing script
    tftg_inputs_train = torch.load(
        config.tf_tg_train_cache_path,
        weights_only=False,
    )
    tftg_inputs_val = torch.load(
        config.tf_tg_val_cache_path,
        weights_only=False,
    )
    # The test split is deliberately NOT loaded here -- see build_test_loader() below.
    # trainer.fit() never touches it, and at 64 cells/edge it is 7.4 GB per rank of file
    # read and resident RAM that sits idle for the whole run.

    atac_peak_tensor = torch.load(
        config.tf_tg_atac_peak_cache_path,
        weights_only=True,
    )

    # Load the metadata
    with open(config.tf_tg_metadata_cache_path, "r") as f:
        metadata = json.load(f)
        
    tf_name_to_idx = metadata["tf_name_to_idx"]
    tg_id_to_idx = metadata["tg_id_to_idx"]

    # atac_mat/rna_mat/gene_to_rna_idx are always needed now: TFTGEdgeBagDataset gathers
    # peak_accessibility/tf_expression/tg_expression from them via inputs["cell_indices"],
    # regardless of --resample_cells_per_epoch (that flag only controls whether the cell
    # columns used are the fixed ones baked into the cache, or freshly redrawn every call).
    for p in (config.tf_tg_atac_mat_cache_path, config.tf_tg_rna_mat_cache_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} is required -- peak_accessibility/tf_expression/tg_expression are "
                "gathered from it at read time, not stored in the edge-bag cache. Build it "
                "with: python3 scripts/build_tf_to_tg_train_data.py "
                "--build_resample_matrices_only (plus the same --max_peaks_per_tg etc as "
                "the existing cache), or rebuild the whole cache."
            )

    atac_mat = torch.load(config.tf_tg_atac_mat_cache_path, weights_only=True)
    rna_mat = torch.load(config.tf_tg_rna_mat_cache_path, weights_only=True)
    gene_to_rna_idx = metadata["gene_to_rna_idx"]
    dataset_kwargs = dict(atac_mat=atac_mat, rna_mat=rna_mat, gene_to_rna_idx=gene_to_rna_idx)

    resample_kwargs = {}
    if args.resample_cells_per_epoch:
        cell_to_idx = metadata["cell_to_idx"]
        idx_to_cell = [None] * len(cell_to_idx)
        for cell_name, i in cell_to_idx.items():
            idx_to_cell[i] = cell_name

        log_once(
            f"--resample_cells_per_epoch enabled: redrawing {max_cells_per_pair} cells/edge "
            f"from a pool of {atac_mat.shape[1]} for training only "
            f"(atac_mat {tuple(atac_mat.shape)}, rna_mat {tuple(rna_mat.shape)})."
        )

        resample_kwargs = dict(
            resample_cells=True,
            idx_to_cell=idx_to_cell,
            resample_max_cells_per_pair=max_cells_per_pair,
        )

    # Load the manifest and verify tensor shapes and dtypes match expectations
    with open(config.tf_tg_manifest_cache_path) as f:
        manifest = json.load(f)

    log_once(json.dumps(manifest, indent=2))

    assert tuple(manifest["atac_peak_tensor_shape"]) == tuple(atac_peak_tensor.shape)
    assert manifest["atac_peak_tensor_dtype"] == str(atac_peak_tensor.dtype)
    assert manifest.get("tftg_format_version") == 2, (
        f"{config.tf_tg_manifest_cache_path} predates the compact edge-bag format "
        "(peak_accessibility/tf_expression/tg_expression are now gathered from atac_mat/"
        "rna_mat instead of stored per-edge). Rebuild the cache with "
        "scripts/build_tf_to_tg_train_data.py."
    )

    # Re-create the datasets and dataloaders using the loaded compact inputs and lookup tensors
    # Applies to all three splits: the choice is only about where the gather happens, so
    # train/val/test must agree or the model would be fed tf_embedding in one loop and
    # tf_idx in another.
    tf_source_kwargs = dict(return_tf_indices=args.tf_embedding_on_device)

    # Subsample BEFORE the dataset is built, so per-TF weights below are computed from the
    # data actually trained on rather than from the full split.
    if args.train_subsample_frac < 1.0:
        keep_idx = stratified_train_subsample(tftg_inputs_train, args.train_subsample_frac)
        tftg_inputs_train = _take_rows(
            tftg_inputs_train, keep_idx, len(tftg_inputs_train["label"])
        )

    # Applied after subsampling (not before) so a TF pushed below threshold by the
    # subsample draw is still caught, and before compute_per_tf_pos_weight below so its
    # counts and --per_tf_pos_weight_max's cap only ever see the TFs actually trained on.
    if args.min_tf_positive_count > 0:
        keep_idx = filter_low_positive_tfs(
            tftg_inputs_train,
            n_tf=tf_embeddings_tensor.shape[0],
            min_positive_count=args.min_tf_positive_count,
        )
        tftg_inputs_train = _take_rows(
            tftg_inputs_train, keep_idx, len(tftg_inputs_train["label"])
        )

    train_dataset = TFTGEdgeBagDataset(
        tftg_inputs_train,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor,
        **dataset_kwargs,
        **tf_source_kwargs,
        **resample_kwargs,
    )

    val_dataset = TFTGEdgeBagDataset(
        tftg_inputs_val,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor,
        **dataset_kwargs,
        **tf_source_kwargs,
    )

    # Group edges by how many real peaks they carry, so collate_tftg_edge_bags can trim the
    # peak axis. The trim is worthless on its own: with uniform shuffling a batch of 256
    # almost always contains one 90-peak edge, so the batch stays at full width. Grouping is
    # what makes the median batch narrow.
    #
    # Val and test are grouped without shuffling -- the metrics are order-invariant there, so
    # it is free speed.
    #
    # Two consequences worth knowing. Batch composition now correlates with peak count, so
    # gradient noise is no longer i.i.d. across batches, and checkpoints are not strictly
    # comparable to runs trained under uniform shuffling. And the sampler truncates each
    # split to a whole multiple of world_size batches (an uneven count deadlocks DDP), so up
    # to world_size-1 batches are dropped per split per epoch.
    from utils import get_rank_info  # local: see the note beside the imports

    global_rank, _, _, sampler_world_size = get_rank_info()
    ddp_kwargs = dict(num_replicas=sampler_world_size, rank=global_rank) if sampler_world_size > 1 else {}

    def peak_counts(inputs):
        """Real peaks per edge, in the split's own order."""
        return inputs["tg_peak_mask"][inputs["tg_idx"]].sum(dim=1).numpy()

    train_sampler = LengthGroupedBatchSampler(
        peak_counts(tftg_inputs_train),
        batch_size=batch_size, shuffle=True, seed=args.seed, **ddp_kwargs,
    )
    val_sampler = LengthGroupedBatchSampler(
        peak_counts(tftg_inputs_val),
        batch_size=batch_size, shuffle=False, **ddp_kwargs,
    )

    # Create the DataLoaders with the custom collate function for batching edge bags
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=dataloader_worker_init,
        collate_fn=collate_tftg_edge_bags,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        worker_init_fn=dataloader_worker_init,
        collate_fn=collate_tftg_edge_bags,
        )

    def build_test_loader():
        """Load the test split and wrap it in a DataLoader, on demand.

        Kept out of the eager path because fit() only ever reads train and val: loading
        the test tensors up front cost every rank a 7.4 GB read and the RAM to hold it
        for the duration of training, for nothing. Call this immediately before
        trainer.test(), so the cost lands once and only if the split is actually scored.

        Test always reads the frozen cached bags -- never the resampled cells -- so a
        score is reproducible against a fixed set of cells.
        """
        log_once(f"Loading test split from {config.tf_tg_test_cache_path} ...")
        tftg_inputs_test = torch.load(
            config.tf_tg_test_cache_path,
            weights_only=False,
        )
        test_dataset = TFTGEdgeBagDataset(
            tftg_inputs_test,
            tf_embeddings_tensor=tf_embeddings_tensor,
            tf_mask_tensor=tf_mask_tensor,
            atac_peak_tensor=atac_peak_tensor,
            **dataset_kwargs,
            **tf_source_kwargs,
        )
        log_once(f"Test split loaded: {len(test_dataset):,} edges")
        test_sampler = LengthGroupedBatchSampler(
            peak_counts(tftg_inputs_test),
            batch_size=batch_size, shuffle=False, **ddp_kwargs,
        )
        return DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            worker_init_fn=dataloader_worker_init,
            collate_fn=collate_tftg_edge_bags,
            )

    # Test size comes from the manifest so reporting it does not pull in the tensors.
    log_once(
        f"Train/Val/Test sizes: {len(train_dataset)}, {len(val_dataset)}, "
        f"{manifest['n_test_rows']} (test not loaded until scored)"
    )

    tf_tg_model = create_new_tf_tg_regulation_model(
        tf_bind_model_path,
        tf_embeddings_tensor,
        tf_mask_tensor,
        checkpoint_path=checkpoint_path,
        keep_tf_peak_model_in_eval=args.keep_tf_dna_in_eval,
        )

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

    log_once("\nStarting Lightning training...")

    # ---- Learning rate and warmup ----------------------------------------------------
    # The reference point is the configuration this model was tuned at: lr 1e-4 with
    # batch_size 8 across 4 GPUs, i.e. an effective batch of 32.
    REFERENCE_LR = 1e-4
    REFERENCE_EFFECTIVE_BATCH = 32

    effective_batch = batch_size * max(1, num_gpus) * max(1, num_nodes)
    batch_ratio = effective_batch / REFERENCE_EFFECTIVE_BATCH

    if args.lr is not None:
        learning_rate = args.lr
        lr_origin = "pinned via --lr"
    elif args.lr_scale_rule == "linear":
        learning_rate = REFERENCE_LR * batch_ratio
        lr_origin = f"linear scaling of {REFERENCE_LR:g} by {batch_ratio:.2f}x"
    elif args.lr_scale_rule == "sqrt":
        learning_rate = REFERENCE_LR * math.sqrt(batch_ratio)
        lr_origin = f"sqrt scaling of {REFERENCE_LR:g} by sqrt({batch_ratio:.2f})"
    else:
        learning_rate = REFERENCE_LR
        lr_origin = "unscaled reference"

    # trainer.global_step counts optimizer steps on a single rank, and DDP shards the
    # sampler across ranks -- but len(train_loader) here is still the unsharded length,
    # because Lightning does not inject the DistributedSampler until fit() starts. Divide
    # it out, or warmup would be world_size times longer than intended.
    world_size_for_steps = max(1, num_gpus) * max(1, num_nodes)
    steps_per_epoch = math.ceil(len(train_loader) / world_size_for_steps)
    warmup_steps = int(round(args.warmup_epochs * steps_per_epoch))

    log_once(
        f"Learning rate {learning_rate:.3e} ({lr_origin}); effective batch "
        f"{effective_batch} = {batch_size} x {num_gpus} GPUs x {num_nodes} nodes "
        f"({batch_ratio:.2f}x the {REFERENCE_EFFECTIVE_BATCH} reference)."
    )
    if warmup_steps > 0:
        log_once(
            f"Linear warmup over {warmup_steps:,} steps "
            f"({args.warmup_epochs:g} epochs at {steps_per_epoch:,} steps/epoch/rank), "
            f"then ReduceLROnPlateau on val/loss takes over."
        )
    else:
        log_once("No LR warmup (--warmup_epochs 0); ReduceLROnPlateau on val/loss only.")

    per_tf_pos_weight = None
    if args.per_tf_pos_weight:
        per_tf_pos_weight = compute_per_tf_pos_weight(
            tftg_inputs_train,
            n_tf=tf_embeddings_tensor.shape[0],
            max_weight=args.per_tf_pos_weight_max,
        )

    lit_model = tf_to_tg_module.LitTFTGRegulationModel(
        model=tf_tg_model,
        lr=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=1e-4,
        pos_weight=None,
        per_tf_pos_weight=per_tf_pos_weight,
        plateau_monitor=args.plateau_monitor,
        plateau_mode=args.plateau_mode,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        plateau_cooldown=args.plateau_cooldown,
        pooling_mode=pooling_mode,
        pooling_temperature=pooling_temperature,
        enable_timing_sync=False,
        fp32_eval=not args.eval_in_training_precision,
    )
    
    # All three callbacks -- selection, stopping, LR -- now agree on val/macro_auroc.
    # Run 3809742 is why. Its LR scheduler watched val/macro_auroc while these two still
    # watched pooled val/auroc, so the run was killed at epoch 21 by EarlyStopping counting
    # 15 non-improving epochs of POOLED AUROC (best 0.6949 at epoch 6) while macro was still
    # setting new highs as late as epoch 13 (0.6773). The LR scheduler never fired at all:
    # macro's new bests at epochs 8/11/13 kept resetting num_bad_epochs, which reached 8 at
    # the final epoch -- one short of the 9 that patience=8 requires. Three callbacks
    # steering on two different metrics is the same fault as the original val/loss bug.
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="epoch={epoch:02d}-macro={val/macro_auroc:.4f}-val_auroc={val/auroc:.4f}-val_loss={val/loss:.4f}",
        monitor=args.selection_metric,
        mode="max",
        save_top_k=500,
        save_last=True,
        auto_insert_metric_name=False,
    )
    
    # Track val/auroc, not val/loss. Run 3799581 showed why: val/loss peaked at epoch 3
    # and drifted up, so EarlyStopping counted down and killed the run at epoch 18 -- which
    # turned out to be its BEST epoch by val/auroc (0.7364, a new high on the final epoch).
    # The run was terminated while still improving on the metric that is actually used to
    # select the checkpoint. Loss and ranking quality diverge here because loss is dominated
    # by confidence/calibration: on run 3793729 train loss fell 0.363 -> 0.19 while train
    # AUROC moved +0.004. Stopping on the metric ModelCheckpoint selects on keeps the two
    # callbacks consistent.
    early_stopping_callback = EarlyStopping(
        monitor=args.selection_metric,
        mode="max",
        patience=args.early_stopping_patience,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    extra_callbacks = []
    if args.tf_embedding_on_device:
        extra_callbacks.append(
            ResidentTFEmbeddingTable(tf_embeddings_tensor, tf_mask_tensor)
        )
        log_once(
            f"--tf_embedding_on_device enabled: pinning the "
            f"{tuple(tf_embeddings_tensor.shape)} TF embedding table "
            f"({tf_embeddings_tensor.numel() * 4 / 1e9:.2f} GB) to each rank and gathering "
            f"by tf_idx, instead of shipping "
            f"{tf_embeddings_tensor.shape[1] * tf_embeddings_tensor.shape[2] * 4 / 1e6:.2f} MB "
            f"per edge ({tf_embeddings_tensor.shape[1] * tf_embeddings_tensor.shape[2] * 4 * batch_size / 1e6:.0f} MB "
            f"per batch of {batch_size}) through the dataloader."
        )

    wandb_logger = WandbLogger(
        project="tf_tg_regulation_prediction",
        name=run_name,
        save_dir=output_dir,
    )

    wandb_logger.log_hyperparams({
        "sample_name": sample_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_batches": len(train_loader),
        "num_gpus": num_gpus,
        "num_nodes": num_nodes,
        "job_id": job_id,
        "run_name": run_name,
        "sample_pairs": sample_pairs,
        "max_peaks_per_tg": max_peaks_per_tg,
        "max_cells_per_pair": max_cells_per_pair,
        "resample_cells_per_epoch": args.resample_cells_per_epoch,
        "pct_true_edges": pct_true_edges,
        "true_false_ratio": true_false_ratio,
        "pooling_mode": pooling_mode,
        "pooling_temperature": pooling_temperature,
        "lr": learning_rate,
        "lr_scale_rule": args.lr_scale_rule if args.lr is None else "pinned",
        "effective_batch": effective_batch,
        "warmup_steps": warmup_steps,
        "warmup_epochs": args.warmup_epochs,
        "tf_embedding_on_device": args.tf_embedding_on_device,
        "precision": args.precision,
        "weight_decay": 1e-4,
        "flank_size": peak_flank_size,
        "max_precompute_peaks": max_peaks_per_tg,
        "persistent_workers": True,
        "tf_bind_model_path": str(tf_bind_model_path),
    })
    
    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            os.environ.get("SLURM_NTASKS", "1"),
        )
    )

    use_ddp = world_size > 1
    
    log_once(
        f"Validation/test scored in fp32 (autocast disabled), training in {args.precision}."
        if not args.eval_in_training_precision else
        f"Validation/test scored in {args.precision} (--eval_in_training_precision): "
        "metrics are NOT comparable across precisions and understate a bf16 run."
    )
    log_once(f"Num GPUs: {world_size} | Batch size: {batch_size}")
    log_once(f"Num steps per epoch: {len(train_loader)}")
    
    strategy=DDPStrategy(
        process_group_backend="nccl",
        find_unused_parameters=False,
    ) if use_ddp else "auto"
    
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_gpus,
        num_nodes=num_nodes,
        strategy=strategy,
        # The batch samplers above already shard by rank. Left at the default, Lightning
        # would wrap them in a DistributedSampler as well and each rank would see a
        # shard of a shard.
        use_distributed_sampler=False,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=25),
            checkpoint_callback,
            early_stopping_callback,
            lr_monitor,
            *extra_callbacks,
        ],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
        default_root_dir=output_dir,
        enable_progress_bar=True,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )
    
    if checkpoint_path is not None:
        log_once(
            f"Warm-starting model weights from: {checkpoint_path} "
            "(optimizer/epoch/scheduler state NOT restored)"
        )
    if args.resume_from_checkpoint is not None:
        log_once(f"Resuming full training state from: {args.resume_from_checkpoint}")

    trainer.fit(
        lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_checkpoint,
    )

    # No trainer.test() here, deliberately: LitTFTGRegulationModel has no test_step, and
    # _shared_step accepts only "train"/"val". Test scoring on the held-out TFs happens
    # against a saved checkpoint in plot_auprc_all_methods.py / generate_all_predictions.py.
    # build_test_loader() above is the hook to use if that ever moves in-process.
    log_once(f"Best checkpoint by val/auroc: {checkpoint_callback.best_model_path or '(none)'}")