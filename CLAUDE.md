# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TETHER is a deep-learning pipeline for inferring gene regulatory networks (GRNs) from
single-cell multiome data (paired scRNA-seq + scATAC-seq). It predicts transcription-factor
→ target-gene (TF→TG) regulatory edges and benchmarks them against external GRN methods
(SCENIC+, LINGER, CellOracle, Pando, FigR, GRaNIE).

Nearly all real code lives in the `TETHER/` subdirectory. The repo root mostly holds `data/`,
`LOGS/`, and cluster/Jupyter launch scripts.

## Environment & execution model

- **This is a SLURM HPC codebase.** Nearly every workflow is a `#SBATCH` batch script submitted
  with `sbatch <script>.sh` (or `sbatch --array=...` for the array jobs). There is no local
  "build". GPU work targets `dense` partition (a100/v100); CPU cache-building targets `compute`.
- **Conda env `my_env`** is activated (`source activate my_env`) by every batch script and is the
  environment for all Python here. The TF-embedding step (`01_generate_tf_embeddings.sh`) switches
  to a second env `tfbindformer` for the Foldseek/ProstT5 3Di step. The repo-root `.venv/` is not
  what the batch scripts use.
- **Absolute paths are hardcoded** to `/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER`
  throughout (`config.py`, every `.sh`, `PROJECT_DIR`/`DATA_DIR` constants in scripts). Moving the
  repo or running from another root requires editing these. Scripts `sys.path.append(PROJECT_DIR)`
  so `import config`, `import utils`, `import models...` resolve.
- Multi-GPU training uses DDP launched via `srun ... torchrun`-style env setup — the batch scripts
  dynamically discover the `10.90.29.*` NIC for `NCCL_SOCKET_IFNAME` and set `MASTER_ADDR/PORT`.
  Preserve that networking preamble when copying a training script.

## The two-stage model (the core architecture)

The GRN model is two stacked models; understanding this split is essential:

1. **TF–DNA binding model** (`TETHER/models/tf_to_dna.py`, `TFPeakBindingModel`): given a TF protein
   embedding and a peak's one-hot DNA sequence, predicts whether that TF binds that peak. TF proteins
   are encoded from ProstT5/Foldseek **3Di structural embeddings** (not raw sequence); peaks are
   encoded with a 1D-conv stack, and the two are fused with `BidirectionalCrossAttentionTransformer`.
   Trained first, standalone, to high AUROC.

2. **TF–TG regulation model** (`TETHER/models/tf_to_tg.py`, `TFTGRegulationModel`): the actual GRN
   model. It **freezes** a pretrained TF–DNA model as a submodule, uses it to score (TF, peak) binding,
   then combines binding + peak accessibility + peak→TSS distance + TF/TG expression. It attends over
   nearby peaks per (TF, TG, cell) and **pools across sampled cells** (log-sum-exp by default,
   `pooling_temperature`) to emit one logit per TF→TG edge. Works on compact per-edge "edge bags"
   (up to N nearby peaks × M sampled cells).

Ablation variants live in `TETHER/models/simplified_models/` (`no_binding`, `no_expr_info`,
`no_peak_info`, `no_peak_tg_distance`) and are trained/tested by `test_simplified_model_multigpu_safe.py`
(despite the name, this is a training+eval script, not a pytest test).

## End-to-end pipeline (run in order)

Numbered batch scripts in `TETHER/bash_scripts/` define the canonical order:

| Step | Script | What it does |
|---|---|---|
| Preprocess | `run_muon_preprocessing.sh` → `muon_preprocessing.py` | raw 10x fragments/counts → RNA/ATAC pseudobulk parquet + peak→gene distances (muon/MuData) |
| 01 | `01_generate_tf_embeddings.sh` | download TF protein seqs (ChIP-Atlas), Foldseek/ProstT5 3Di tokens, `extract_tf_embeddings.py` → per-TF embedding tensors |
| 02a | `02a_build_tf_to_dna_cache.sh` → `scripts/build_tf_to_dna_train_data.py` | build & cache TF–DNA training edges |
| 02b | `02b_train_tf_to_dna_model.sh` → `scripts/train_tf_to_dna_model.py` | train TF–DNA binding model (produces the checkpoints frozen in stage 2) |
| 03a | `03a_build_tf_to_tg_cache.sh` → `scripts/build_tf_to_tg_train_data.py` | build & cache TF–TG edge bags |
| 03b | `03b_train_tf_to_tg_model.sh` → `scripts/train_tf_to_tg_model.py` | train TF–TG regulation model |

Cache-build steps take `--force_reload` to ignore existing caches; without it they reuse whatever is
in `TETHER/cached_data/{cell_type}_cache/`. Caching is keyed only by cell_type/sample_name, so
upstream data changes are **not** auto-detected — pass `--force_reload` or delete the cache.

## Evaluation & analysis scripts (in `TETHER/`)

- `plot_auprc_all_methods.py` (via `run_plot_auprc_all_methods.sh`): the main benchmark. Scores the
  TF–TG model (own test set + cross-trained) against external methods with PR curves / AUPRC.
  **Fully documented in `TETHER/docs/auprc_calculation_method.md` — read that before touching it.**
- `analyze_stability.py` (via `run_analyze_stability.sh`): runs a trained model against its own test
  set across subsampled runs to measure edge-ranking stability. Array job (`TASK_ID / NUM_SUBSAMPLES`).
- `model_generalizability.py` (via `run_model_generalizability.sh`): cross-model/cross-sample eval;
  a `curated` vs full all-pairs `EXPERIMENT_MODE` selects which (model, test) combos run.
- `stability_model_training.py`, `wandb_sweep.py` / `run_wandb_sweep.sh` / `wandb_sweep.yaml`:
  stability-oriented training and W&B hyperparameter sweeps.
- `test_tf_tg_predictions.ipynb` is the main figure/analysis notebook — most statistics and plots
  live here. **How each statistic is computed and which figures it produces is documented in
  `TETHER/docs/notebook_statistics_and_plots.md`.** `combine_figures.ipynb` + `figure_code.py`
  assemble publication figures. Launch Jupyter on a GPU node with `jupyter_dense_a100.sh` /
  `jupyter_dense_v100.sh` (URLs land in `jupyter_urls/`).

Shared logic lives in `TETHER/utils.py` (large — dataset building, one-hot encoding, model loading,
lookup tables), `plotting_utils.py`, `stat_utils.py`.

## config.py — the central dataset switch

`TETHER/config.py` is imported by the analysis scripts/notebooks and defines the "current" dataset by
**module-level variables you edit**: `species` (`mm10`/`hg38`), `cell_type`, `sample_name` (blocks of
these are commented out — uncomment one). It then derives every cache/data/genome/checkpoint path and
asserts valid cell types (`Macrophage`, `mESC`, `K562`, `iPSC`, `mouse_liver`, `mouse_hepatocytes`).
It also holds `tf_dna_model_checkpoints` (per-species frozen TF–DNA checkpoint) and the ground-truth
file lists per cell type. Note: the CLI eval scripts take `--species/--cell_type/--sample_name` args
instead of reading these module globals — check whether a given entry point uses argparse or `config.`.

## Data & evaluation conventions

- **Train/val/test split is by chromosome** (assigned via the target gene's chromosome), not random:
  mm10 test = chr 18-19, hg38 test = chr 20-22 (see `split_genes_by_chromosome`). Held-out chromosomes
  are how leakage is avoided.
- All TF/TG/method-column name matching is done on **upper-cased** strings everywhere to avoid join
  failures across data sources.
- `data/` is almost entirely git-ignored (see `.gitignore`); only a small `DS012_mESC` sample, kept
  motif-information files, and `.gitkeep` placeholders are tracked. Large binaries (`.pt`, `.h5ad`,
  `.parquet`, `.png`, `.pkl`, `.pybiomart.sqlite`, `wandb/`, `checkpoints/`, `plots/`) are ignored.

## Homer setup (from README)

Some peak-annotation work needs Homer: install into a `Homer/` dir via `perl ./configureHomer.pl
-install`, add `<Homer>/.//bin/` to `PATH`, and install the genome (`-install hg38` or `-install mm10`).
