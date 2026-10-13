#!/usr/bin/env python
"""Train cell-type-specific TF->TG edge bags from per-sample caches.

Build each sample's cache first with scripts/build_tf_to_tg_celltype_data.py
(bash_scripts/03c_build_celltype_data.sh). This script never computes binding
scores. It stops before loading any data if a sample has no complete cache for the
given preprocessing settings, so those settings must match the build job.

Holdouts are applied here, so one set of sample caches serves every holdout choice:
  --holdout_sample mESC:E7.5_rep1 --holdout_celltype "Definitive endoderm"
By default, every cell type of a holdout sample that also appears in another
sample's training split is held out as well (--no-auto_holdout_celltypes disables).

The shared input scaler is fit from the training edges of this run and cached in
the run-level cache directory, which also links every sample cache under
prepared_sources/ for the evaluation notebooks.
"""
import argparse
import json
import logging
import os
import math
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.6"

import numpy as np
from scipy import sparse
import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler, Subset, Dataset
from tqdm import tqdm
from scripts.batch_samplers import SampleBalancedSampler

from scripts.build_tf_to_tg_celltype_data import (  # noqa: E402
    BINDING_MANIFEST,
    SAMPLE_CACHE_VERSION,
    add_data_args,
    finalize_data_args,
    is_sample_cache_complete,
    load_cached_source,
    sample_cache_location,
    _atomic_torch_save,
    _atomic_write_json,
    _cache_digest,
    binding_score_cache_path,
)

import pytorch_lightning as pl  # noqa: E402
from pytorch_lightning.callbacks import (  # noqa: E402
    EarlyStopping, LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger  # noqa: E402
from models.tf_to_tg_celltype import LitTFTGRegulationModel  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_data_args(parser)
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
    parser.add_argument(
        "--auto_holdout_celltypes", action=argparse.BooleanOptionalAction, default=True,
        help=("Also hold out every cell type of a --holdout_sample that appears in the "
              "training split of another sample. Matching is exact."),
    )
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument(
        "--cache_dir", type=Path,
        help=("Run-level directory for the input scaler and links to the sample caches. "
              "By default, a stable directory is selected from the run configuration."),
    )
    parser.add_argument("--resume_from_checkpoint", type=Path)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--slurm_id", default=os.environ.get("SLURM_JOB_ID", "local"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--resample_cells", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--balance_samples", action=argparse.BooleanOptionalAction, default=True,
        help="Give every source sample equal expected training probability",
    )
    parser.add_argument(
        "--train_edges_per_sample", type=int, default=None,
        help="With --balance_samples, edges drawn from each source sample per "
             "epoch (default: size of the smallest training sample)",
    )
    parser.add_argument(
        "--scaler_edges_per_sample", type=int, default=100000,
        help="Equal number of training edges per sample used to fit shared scaling",
    )
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_cross_attn_layers", type=int, default=1,
                        help="Cross-attention layers from the cell query to the peaks")
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
    parser.add_argument("--fast_dev_run", action="store_true",
                        help="Run one train/validation batch after loading the cached data")
    args = parser.parse_args(argv)
    source_specs = finalize_data_args(parser, args)

    for key in ("epochs", "batch_size", "num_heads", "d_model", "accumulate_grad_batches",
                "num_cross_attn_layers"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive")
    if args.scaler_edges_per_sample <= 0 or args.d_model % args.num_heads or args.d_model < 2:
        parser.error(
            "scaler_edges_per_sample must be positive; d_model >= 2 and divisible by num_heads"
        )
    malformed_holdouts = [
        value for value in args.holdout_sample
        if value.count(":") != 1 or not all(value.split(":"))
    ]
    if malformed_holdouts:
        parser.error(
            f"--holdout_sample values must be TISSUE:SAMPLE; invalid: {malformed_holdouts}"
        )
    if any(not value.strip() for value in args.holdout_celltype):
        parser.error("--holdout_celltype values must be nonempty")
    if args.lr <= 0 or args.pooling_temperature <= 0 or args.pos_weight <= 0:
        parser.error("lr, pooling_temperature, and pos_weight must be positive")
    if not 0 <= args.dropout < 1 or args.weight_decay < 0 or args.gradient_clip_val < 0:
        parser.error("dropout must be in [0, 1); weight decay and gradient clipping must be nonnegative")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = args.run_name if args.run_name is not None else "joint" if len(source_specs) > 1 else source_specs[0][1]
    run_prefix = f"celltype_{run_name}"
    if args.output_dir is None and args.run_name is None and args.slurm_id != "local":
        # A SLURM requeue reruns this script under the same job ID. Lightning saves
        # hpc_ckpt_N.ckpt in the previous attempt's output directory and resumes only
        # when default_root_dir points there, so reuse it instead of a new timestamp.
        previous = sorted(
            (PROJECT_DIR / "checkpoints/celltype_tf_tg").glob(f"{run_prefix}_*"),
            key=lambda path: path.stat().st_mtime,
        )
        if previous:
            args.output_dir = previous[-1]
            args.run_name = args.output_dir.name
    
    run_prefix_full = f"{run_prefix}_{args.slurm_id}_{stamp}"
    args.output_dir = args.output_dir or PROJECT_DIR / "checkpoints/celltype_tf_tg" / run_prefix_full
    
    if args.wandb_run_id is None:
        # Continue the same W&B run after a requeue. Run directories end in "-<run id>".
        latest_wandb_run = args.output_dir / "wandb" / "latest-run"
        if latest_wandb_run.exists():
            args.wandb_run_id = latest_wandb_run.resolve().name.rsplit("-", 1)[-1]
    return args, source_specs


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

        if sparse.issparse(matrix):
            block = matrix[rows][:, cols].toarray()
        else:
            # One cells x features gather. matrix[rows][:, cols] first copies every
            # column of the selected rows: on a dense 147k-peak ATAC matrix that is
            # ~75 MB per edge (float64) to keep 25 values.
            block = matrix[np.ix_(rows, cols)]

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
def fit_shared_input_scaler(loader, full_tqdm=False):
    """Fit global moments from real entries in depth-normalized/log1p train inputs."""
    miniters = 1 if full_tqdm else max(1, math.ceil(len(loader) / 50))
    totals = {
        name: {"sum": 0.0, "sum_sq": 0.0, "count": 0}
        for name in ("tf_expression", "tg_expression", "peak_accessibility")
    }

    def update(name, values):
        values = values.double()
        totals[name]["sum"] += values.sum().item()
        totals[name]["sum_sq"] += values.square().sum().item()
        totals[name]["count"] += values.numel()

    progress = tqdm(
        loader,
        desc="Fitting shared input scaling",
        miniters=miniters,
        maxinterval=float("inf"),
    )
    for batch in progress:
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


def make_sample_balanced_sampler(datasets, seed, edges_per_sample=None):
    """Give every source sample equal expected probability during joint training."""
    if len(datasets) < 2:
        return None
    return SampleBalancedSampler(datasets, seed, edges_per_sample)


def link_sample_cache(cache_dir, tissue, sample_name, sample_dir):
    """Expose a sample cache at the path the evaluation notebooks read."""
    link = cache_dir / "prepared_sources" / f"{tissue}__{sample_name}"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == sample_dir.resolve():
        return
    temporary_link = link.with_name(f"{link.name}.tmp-{os.getpid()}")
    temporary_link.symlink_to(sample_dir.resolve(), target_is_directory=True)
    os.replace(temporary_link, link)


def main(argv=None, extra_callbacks=()):
    """argv and extra_callbacks let wandb_sweep.py run this in-process."""
    args, source_specs = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        raise ValueError("This launcher supports one process and one GPU per run")

    pl.seed_everything(args.seed, workers=True)

    holdout_samples = {tuple(value.split(":", 1)) for value in args.holdout_sample}
    unknown_holdout_samples = holdout_samples - set(source_specs)
    if unknown_holdout_samples:
        raise ValueError(
            "Holdout samples must match a configured dataset: "
            f"{sorted(unknown_holdout_samples)}"
        )
    holdout_celltypes = set(args.holdout_celltype)

    # Check every sample cache before loading any multiome data.
    locations = {
        spec: sample_cache_location(args, *spec) for spec in source_specs
    }
    missing = {
        f"{tissue}:{sample_name}": str(directory)
        for (tissue, sample_name), (directory, _) in locations.items()
        if not is_sample_cache_complete(directory)
    }
    if missing:
        raise FileNotFoundError(
            "No complete cache for these samples with the current preprocessing "
            "settings. Build them with bash_scripts/03c_build_celltype_data.sh, "
            "using the same settings as this run:\n"
            + "\n".join(f"  {name}: expected {path}" for name, path in missing.items())
        )

    if args.auto_holdout_celltypes and holdout_samples:
        # The build step records each split's cell types, so this needs no data loading.
        split_celltypes = {
            spec: {
                split_name: set(split["celltypes"])
                for split_name, split in json.loads(
                    (directory / BINDING_MANIFEST).read_text()
                )["splits"].items()
            }
            for spec, (directory, _) in locations.items()
        }
        training_celltypes = set().union(*(
            split_celltypes[spec].get("train", set())
            for spec in source_specs if spec not in holdout_samples
        ))
        holdout_sample_celltypes = set().union(*(
            celltypes
            for spec in holdout_samples
            for celltypes in split_celltypes[spec].values()
        ))
        shared_celltypes = (training_celltypes & holdout_sample_celltypes) - holdout_celltypes
        logging.info(
            "Holding out %d cell types shared by the holdout and training samples: %s",
            len(shared_celltypes), sorted(shared_celltypes),
        )
        holdout_celltypes |= shared_celltypes
    # run_config.json records the full set; the notebooks read it from "holdout_celltype".
    args.holdout_celltype = sorted(holdout_celltypes)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scaler_edges_per_sample = (
        min(args.scaler_edges_per_sample, args.batch_size)
        if args.fast_dev_run else args.scaler_edges_per_sample
    )
    sample_cache_dirs = {
        f"{tissue}:{sample_name}": str(directory)
        for (tissue, sample_name), (directory, _) in locations.items()
    }
    cache_manifest = {
        "cache_version": SAMPLE_CACHE_VERSION,
        "species": args.species,
        "datasets": [f"{tissue}:{sample}" for tissue, sample in source_specs],
        "holdout_samples": sorted(":".join(spec) for spec in holdout_samples),
        "holdout_celltypes": sorted(holdout_celltypes),
        "sample_caches": sample_cache_dirs,
        "scaler": {
            "edges_per_sample": scaler_edges_per_sample,
            "max_cells_per_pair": args.max_cells_per_pair,
            "seed": args.seed,
        },
    }
    cache_key = _cache_digest(cache_manifest)
    if args.cache_dir is None:
        args.cache_dir = (
            PROJECT_DIR / "cached_data" / args.species / "celltype_tf_tg" / "runs" / cache_key
        )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    run_cache_manifest_path = args.cache_dir / "cache_manifest.json"
    if run_cache_manifest_path.is_file():
        if json.loads(run_cache_manifest_path.read_text()) != cache_manifest:
            raise ValueError(
                f"Cache directory belongs to a different run configuration: "
                f"{args.cache_dir}. Choose another --cache_dir or remove the override."
            )
    else:
        _atomic_write_json(run_cache_manifest_path, cache_manifest)
    for (tissue, sample_name), (directory, _) in locations.items():
        link_sample_cache(args.cache_dir, tissue, sample_name, directory)
    logging.info("Run cache: %s (key %s)", args.cache_dir, cache_key)

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["slurm_id"] = os.environ.get("SLURM_JOB_ID", "local")
    config["cache_key"] = cache_key
    config["sample_cache_dirs"] = sample_cache_dirs
    _atomic_write_json(args.output_dir / "run_config.json", config)

    prepared_sources = [
        load_cached_source(args, tissue, sample_name, directory, manifest)
        for (tissue, sample_name), (directory, manifest) in locations.items()
    ]

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

    use_cuda = args.accelerator != "cpu" and torch.cuda.is_available()
    if args.accelerator == "gpu" and not use_cuda:
        raise RuntimeError("GPU requested but CUDA is unavailable")

    dataset_parts = {"train": [], "val": [], "test": []}
    scaler_train_parts = []
    scaler_train_sources = []
    for source in prepared_sources:
        source_key = f"{source['tissue']}__{source['sample_name']}"
        for split_name, frame in source["splits"].items():
            if frame.empty:
                continue
            scores, binding_cache_file = source["binding_scores"][split_name]
            if split_name == "train":
                original_edge_count = len(frame)
                keep = np.ones(original_edge_count, dtype=bool)
                if (source["tissue"], source["sample_name"]) in holdout_samples:
                    keep[:] = False
                elif holdout_celltypes:
                    keep = ~frame["cell_type"].isin(holdout_celltypes).to_numpy()
                removed_edge_count = int((~keep).sum())
                if removed_edge_count:
                    # Binding rows are positional, so filter both with the same mask.
                    frame = frame.loc[keep].reset_index(drop=True)
                    scores = scores[torch.from_numpy(keep)]
                    logging.info(
                        "%s: removed %s of %s training edges for holdout selection",
                        source_key, f"{removed_edge_count:,}", f"{original_edge_count:,}",
                    )
                    if not frame.empty:
                        # The notebooks find training binding scores by the content
                        # hash of the holdout-filtered frame.
                        binding_cache_file = binding_score_cache_path(
                            source["output_dir"] / "prepared", 
                            "train", 
                            frame,
                            tf_dna_checkpoint=args.tf_dna_checkpoint,
                            max_peaks_per_tg=args.max_peaks_per_tg,
                        )
                        if not binding_cache_file.exists():
                            _atomic_torch_save(scores, binding_cache_file)
                config[f"{source_key}_train_holdout_edges"] = removed_edge_count
                config[f"{source_key}_train_original_edges"] = original_edge_count
                config[f"{source_key}_train_edges"] = len(frame)
                if frame.empty:
                    continue

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
                scaler_train_sources.append(f"{source['tissue']}:{source['sample_name']}")

            config[f"{source_key}_{split_name}_edges"] = len(frame)
            config[f"{source_key}_{split_name}_positive_fraction"] = float(frame.label.mean())
            config[f"{source_key}_{split_name}_celltypes"] = sorted(
                frame.cell_type.unique().tolist()
            )
            config[f"{source_key}_{split_name}_binding_cache"] = str(binding_cache_file)

    if not dataset_parts["train"] or not dataset_parts["val"]:
        raise ValueError("Joint training requires train and validation data")
    datasets = {
        split_name: (parts[0] if len(parts) == 1 else ConcatDataset(parts))
        for split_name, parts in dataset_parts.items() if parts
    }

    scaler_path = args.cache_dir / "input_scaler.json"
    scaler_manifest_path = args.cache_dir / "input_scaler_manifest.json"
    scaler_manifest = {
        "datasets": cache_manifest["datasets"],
        "scaler_train_datasets": scaler_train_sources,
        "holdout_samples": cache_manifest["holdout_samples"],
        "holdout_celltypes": cache_manifest["holdout_celltypes"],
        "train_edges": [len(dataset) for dataset in scaler_train_parts],
        "edges_per_sample": scaler_edges_per_sample,
        "seed": args.seed,
        "cache_key": cache_key,
        "max_cells_per_pair": args.max_cells_per_pair,
        "input_space": "per_sample_depth_normalized_log1p_X",
    }
    cached_scaler_manifest = (
        json.loads(scaler_manifest_path.read_text())
        if scaler_manifest_path.exists() else None
    )
    if scaler_path.exists() and cached_scaler_manifest == scaler_manifest:
        input_scaler = validate_input_scaler(json.loads(scaler_path.read_text()))
        logging.info("Loaded shared input scaling from %s", scaler_path)
    else:
        scaler_dataset = make_balanced_scaler_dataset(
            scaler_train_parts, scaler_edges_per_sample, args.seed,
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
        make_sample_balanced_sampler(
            dataset_parts["train"], args.seed, args.train_edges_per_sample
        )
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
        "d_model", "num_heads", "num_cross_attn_layers", "dropout", "lr", "weight_decay",
        "pooling_temperature", "pos_weight", "plateau_patience")},
        **scaler_hparams)

    if args.wandb_mode == "disabled":
        logger = CSVLogger(str(args.output_dir), name="metrics")
    else:
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

    # With ckpt_path=None under SLURM, Lightning loads the newest hpc_ckpt_N.ckpt in
    # default_root_dir, which the auto-requeue handler wrote before the time limit.
    resuming = (args.resume_from_checkpoint is not None
                or any(args.output_dir.glob("hpc_ckpt_*.ckpt")))
    if resuming:
        logging.info("Resuming training in %s", args.output_dir)

    checkpoint = ModelCheckpoint(
        dirpath=args.output_dir / "checkpoints", filename="epoch-{epoch:03d}",
        auto_insert_metric_name=False, monitor="val/loss", mode="min",
        save_top_k=3, save_last=True)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.epochs,
        precision=args.precision,
        logger=logger,
        default_root_dir=str(args.output_dir),
        callbacks=[
            checkpoint, 
            TwoPercentProgressBar(), 
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(
                monitor="val/loss", 
                mode="min",
                patience=args.early_stopping_patience,
                check_finite=True
                ),
            *extra_callbacks,
            ],
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val, log_every_n_steps=50,
        # A full step-zero validation below replaces Lightning's two-batch sanity check.
        num_sanity_val_steps=1 if resuming else 0,
        fast_dev_run=args.fast_dev_run,
    )

    try:
        if not resuming:
            logging.info("Evaluating the untrained model on the validation set at step 0")
            untrained_results = trainer.validate(module, dataloaders=loaders["val"], verbose=True)
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
