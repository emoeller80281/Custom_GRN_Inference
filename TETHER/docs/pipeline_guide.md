# Pipeline guide: TF-DNA cache to method-comparison figures (mm10)

This guide lists every step from building the TF-DNA training cache to the metrics and figures
that compare the TF-TG models with other GRN inference methods. Each step is one SLURM launcher
in `TETHER/bash_scripts/`.

## Rules for every step

- Never run Python on the gateway node (hostname contains `hgtw`). Submit every step with `sbatch`.
- Submit from the repository root, because the launchers write logs to `LOGS/<step>/`:
  ```bash
  REPO=/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER
  cd $REPO
  ```
- Extra arguments after the launcher name go to its Python script. If an argument is also set
  inside the launcher, the last value wins, so you can override a launcher setting without
  editing it.
- To start a step when the previous one succeeds, use `sbatch --dependency=afterok:<job id> ...`.

## Overview

| Step | Launcher | Resources | Output | Time |
|---|---|---|---|---|
| 1. TF-DNA cache | `02a_build_tf_to_dna_cache.sh` | compute, 64 CPUs, 370 GB | `TETHER/cached_data/mm10/<cache>/` | 20 min (last run) |
| 2. Train TF-DNA model | `02b_train_tf_to_dna_model.sh` | dense, 4 × A100 | `TETHER/checkpoints/tf_dna_mm10_<job>/` | 4 h for 25 epochs (last run) |
| 3. Choose a TF-DNA checkpoint | manual `cp` | none | `TETHER/checkpoints/new_tf_dna_models/` | none |
| 4. Evaluate TF-DNA model (optional) | `run_tf_dna_landscape.sh`, `run_tf_dna_cutrun_evaluation.sh` | 1 GPU | `TETHER/new_plots/...` | limit 2 h |
| 5. TF-TG sample caches | `03c_build_celltype_data.sh` | array of 17 tasks, 1 V100 each | binding scores per sample | limit 24 h |
| 6. Holdout TF-TG models | `03e_train_holdout_celltype_tf_to_tg_model.sh` | six jobs, 1 A100 each | `TETHER/checkpoints/celltype_tf_tg/celltype_joint_<job>_*/` | limit 12 h |
| 7. GRNs | `run_generate_grns_from_model.sh` | array of 6 tasks, 1 V100 each | `test_set_grn.csv`, `<sample>_cross_grn.csv` in each run folder | limit 24 h |
| 8. Compare with other methods | `run_compare_grn_methods.sh` | compute, 16 CPUs, 192 GB | tables in `TETHER/new_testing_results/`, plots in `--plot_dir` | 25 min with relabeling, 54 GB (last run) |
| 9. Stability (optional) | `scripts/stability_analysis/run_generate_stability_subsample_grns.sh` | array of 6 tasks, 1 GPU each | `TETHER/testing_results/stability_results/<folder>/` | limit 12 h |

### Inputs that must already exist

- Processed samples: `data/processed/<tissue>/<sample>/multiome_processed.h5mu`.
- TF protein embeddings: `data/tf_data/mm10/...`. Rerun `01_generate_tf_embeddings.sh` only when
  the TF proteins change. After that, every later step must run again.
- ChIP-Atlas peaks: `data/ground_truth_files/chip_atlas_mm10_all.parquet`.
- Cell-type ground truth: `data/ground_truth_files/cell_type_specific/mm10/<dataset>/`
  (`run_build_gt.sh`, one dataset per run; set `CELL_TYPE` in the script).
- Other-method GRNs: `/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/formatted_GRNs_celltype/<method>/`.

## Names to update when a model changes

The holdout run names come from SLURM job IDs (`celltype_joint_<job>_<start time>`), so these
places must name the new runs or checkpoint:

| What | Where | How |
|---|---|---|
| TF-DNA checkpoint for the TF-TG caches | step 5 | pass `--tf_dna_checkpoint`; the default is the old `mm10_3831017_epoch_12` |
| TF-DNA checkpoint for the holdout models | step 6 | pass `--tf_dna_checkpoint`, or edit `TF_DNA_CHECKPOINT` in 03e |
| Holdout runs for the GRNs | step 7 | edit `EXPERIMENT_LIST` in `run_generate_grns_from_model.sh` |
| Holdout runs for the comparison | step 8 | pass `--runs name=celltype_joint_<job>_*`, or edit `DEFAULT_MODEL_RUNS` |
| Holdout runs for stability | step 9 | edit `MODEL_RUNS` and `OUTPUT_DIR` in `generate_stability_subsample_grns.py` |
| Notebook views | `celltype_model_evaluation.ipynb` | edit `model_run_dict` and `tf_dna_model_path` in cell 0 |
| Default TF-DNA checkpoint of scripts | `TETHER/config.py` | `mm10_tf_dna_path` (used when a script gets no `--checkpoint`) |

## Step 1: Build the TF-DNA cache

```bash
sbatch --mem=370G TETHER/bash_scripts/02a_build_tf_to_dna_cache.sh \
    --site_merge_distance 64 --false_overlap_flank 128 \
    --cache_dir $REPO/TETHER/cached_data/mm10/tf_dna_cache_<name>
```

- `--site_merge_distance 64` merges each TF's replicate ChIP-Atlas peaks (centres chained within
  64 bp) into one site. Each site gives one positive.
- `--false_overlap_flank 128` rejects a negative when its 256 bp window overlaps any ChIP-Atlas
  peak of the same TF. Without it, about one third of the negatives are bound sites.
- Both options default to 0 (off), which rebuilds the 2026-10-22 cache.
- `--cache_dir` is required, and its path must contain the job's species. The species is `mm10`
  unless the job sets it (`sbatch --export=ALL,species=hg38 ...`). The TF-TG cache build (step 5)
  reads its TF table and TF embeddings from `cached_data/<species>/tf_dna_cache`, so do not overwrite
  that folder unless the TF table stays the same.
- The launcher always passes `--force_reload`. The last build peaked at 214 GB, so request 370 GB.

Check the log, `LOGS/build_tf_dna_cache/build_tf_dna_cache_<job>.err`:
- `Merged ... TF-peak pairs into ... sites` and `none within 128 bp of the TF's own peaks`.
- `Train/Val/Test edges` at the end. Train is chr1–15, val is chr16–17, test is chr18–19.

The TF table of the new cache must be the same as the one in `tf_dna_cache`:
```bash
cmp TETHER/cached_data/mm10/tf_dna_cache/tf_name_to_idx.csv TETHER/cached_data/mm10/tf_dna_cache_<name>/tf_name_to_idx.csv
```

## Step 2: Train the TF-DNA model

```bash
# Same training as run 3925194, on the new cache
sbatch --dependency=afterok:<step 1 job> --job-name=tf_dna_<tag> \
    TETHER/bash_scripts/02b_train_tf_to_dna_model.sh --cache_dir <cache>

# With regularization: reverse-complement augmentation, weight decay 1e-2, warmup + cosine LR
sbatch --dependency=afterok:<step 1 job> --job-name=tf_dna_<tag> \
    TETHER/bash_scripts/02b_train_tf_to_dna_model.sh --cache_dir <cache> \
    --rc_augment --weight_decay 1e-2 --lr_schedule cosine --warmup_epochs 1 \
    --epochs 20 --early_stopping_patience 20
```

- Launcher defaults: 50 epochs, batch 512 per GPU, LR 3e-4, ReduceLROnPlateau (×0.1),
  weight decay 1e-4, early stopping on val loss with patience 10.
- Outputs in `TETHER/checkpoints/tf_dna_mm10_<job>/`: `epoch_<N>.ckpt` for every epoch, the top 3
  epochs by `val/per_tf_auroc`, and `last.ckpt`.
- W&B: project `tf_peak_binding`, run `tf_dna_mm10_<job>`. The log is
  `LOGS/tf_dna_model/<job name>_<job>.log`.
- Val metrics use all val edges, gathered from the 4 GPUs. Runs before 2026-10-26 logged AUROC on
  GPU rank 0's quarter only.

## Step 3: Choose and copy a checkpoint

Pick the epoch with the highest `val/per_tf_auroc` in W&B. Val loss also measures calibration, so
it can rise while the ranking still improves. Copy the checkpoint under a name that records the run
and the epoch:

```bash
cp TETHER/checkpoints/tf_dna_mm10_<job>/epoch_<N>.ckpt \
   TETHER/checkpoints/new_tf_dna_models/mm10_<job>_epoch_<N>.ckpt
```

- `utils.load_tf_dna_model` rebuilds the model with fixed sizes (128-d, 4 layers, 4 heads). A
  model trained with other `--model_dim` or `--num_layers` values does not load.
- A model trained with `--rc_augment` averages both strands during validation only. The TF-TG
  binding scores (step 5) and the landscape script score one strand.

## Step 4: Evaluate the TF-DNA model (optional)

```bash
# Motif-insertion landscape: new_plots/prediction_landscapes/
# --cache-dir: the cache the checkpoint was trained on (reference peaks); default config.py cache
sbatch TETHER/bash_scripts/run_tf_dna_landscape.sh --species mm10 --tfs all \
    --checkpoint <checkpoint> --cache-dir <cache dir> --output <name>_all_tfs.html

# CUT&RUN peaks (NANOG, CTCF): first set tf_dna_model_path in tf_dna_vs_cut&run.ipynb
sbatch TETHER/bash_scripts/run_tf_dna_cutrun_evaluation.sh
```

## Step 5: Build the TF-TG sample caches

```bash
sbatch TETHER/bash_scripts/03c_build_celltype_data.sh \
    --max_cells_per_pair 25 --max_peaks_per_tg 25 \
    --tf_dna_checkpoint $REPO/TETHER/checkpoints/new_tf_dna_models/mm10_<job>_epoch_<N>.ckpt
```

- One array task per sample (17 samples, `--array=0-16%9`). Rebuild some samples only with
  `sbatch --array=3,7 ...`.
- `--max_cells_per_pair` and `--max_peaks_per_tg` must match step 6 (25 and 25 for the holdout
  models). Otherwise training cannot find the caches.
- Always pass `--tf_dna_checkpoint`. The default is the old model `mm10_3831017_epoch_12`.
- The binding-score files carry a key made from the checkpoint, so caches for other checkpoints
  stay in place. A finished sample has a `binding_manifest.json`.

## Step 6: Train the six holdout TF-TG models

Each model holds out one tissue. Run the six submissions together:

```bash
S=TETHER/bash_scripts/03e_train_holdout_celltype_tf_to_tg_model.sh
CK="--tf_dna_checkpoint $REPO/TETHER/checkpoints/new_tf_dna_models/mm10_<job>_epoch_<N>.ckpt"
D="--dependency=afterok:<step 5 array job>"

sbatch $D $S $CK --holdout_sample mouse_liver:liver_sample                                   # no_liver
sbatch $D $S $CK --holdout_sample mESC:E7.5_rep1 --holdout_sample mESC:E7.5_rep2 \
                 --holdout_sample mESC:E8.0_rep1 --holdout_sample mESC:E8.0_rep2 \
                 --holdout_sample mESC:E8.5_rep1 --holdout_sample mESC:E8.5_rep2             # no_mesc
sbatch $D $S $CK --holdout_sample GSE140203_shareseq:shareseq_brain \
                 --holdout_sample 10x_E18_mouse_brain:E18_brain                              # no_brain
sbatch $D $S $CK --holdout_sample GSE140203_shareseq:skin_late_anagen                        # no_skin
sbatch $D $S $CK --holdout_sample kidney:Ctrl_4weeks_1 --holdout_sample kidney:Ctrl_4weeks_2 \
                 --holdout_sample kidney:Ctrl_6months_1                                      # no_kidney
sbatch $D $S $CK --holdout_sample GSE246464_HSC:young_rep1 --holdout_sample GSE246464_HSC:young_rep2 \
                 --holdout_sample GSE246464_HSC:old_rep1 --holdout_sample GSE246464_HSC:old_rep2  # no_hsc
```

- Write down the six job IDs. Each run folder is
  `TETHER/checkpoints/celltype_tf_tg/celltype_joint_<job>_<start time>/`, with `run_config.json`
  and `checkpoints/last.ckpt`.
- Cell types of a held-out sample that also occur in training samples are held out of training
  automatically.
- W&B project: `celltype-TF-TG`.

## Step 7: Generate the GRNs

Edit `EXPERIMENT_LIST` in `TETHER/bash_scripts/run_generate_grns_from_model.sh` to the six new
runs, in the order liver, mESC, brain, skin, kidney, HSC:

```bash
EXPERIMENT_LIST=(
    "celltype_joint_<liver job>_*"
    "celltype_joint_<mesc job>_*"
    ...
)
```

```bash
sbatch --dependency=afterok:<six step 6 jobs, colon-separated> \
    TETHER/bash_scripts/run_generate_grns_from_model.sh
```

Each run folder then has `test_set_grn.csv` (training samples, test-chromosome targets) and one
`<sample>_cross_grn.csv` per held-out sample. Existing GRNs are skipped; pass `--overwrite` to
replace them.

## Step 8: Compare with the other methods

```bash
sbatch --dependency=afterok:<step 7 array job> TETHER/bash_scripts/run_compare_grn_methods.sh \
    --runs no_liver=celltype_joint_<j1>_* no_mesc=celltype_joint_<j2>_* \
           no_brain=celltype_joint_<j3>_* no_skin=celltype_joint_<j4>_* \
           no_kidney=celltype_joint_<j5>_* no_hsc=celltype_joint_<j6>_* \
    --plot_dir TETHER/new_plots/method_comparison_<tag>
```

Create `LOGS/compare_grn_methods/` once before the first submission.

The script (`scripts/model_evaluation/compare_grn_methods.py`) is the "Cross-trained model vs
other inference methods" section of `celltype_model_evaluation.ipynb` (cells 17–30):

1. It labels every method's GRN against the cell-type ground truth for each (sample, cell type).
2. It calculates AUROC, AUPRC and the early metrics on the top 10% of each method's edges.
3. It calculates random baselines with shuffled scores (10 shuffles).
4. It saves the boxplots, the heatmaps and the average-rank plots.

The TF-TG model in the combined comparison is the mean score over the six models. Each model is
also plotted alone.

Outputs (`<hash>` is made from the samples, cell types, methods and run names):

| File | Content |
|---|---|
| `new_testing_results/all_method_grn_dict_<hash>.pkl` | labeled GRNs (cache, about 4.6 GB) |
| `new_testing_results/celltype_metrics_<hash>.tsv` | mean-score model and other methods |
| `new_testing_results/celltype_metrics_per_model_<hash>.tsv` | each model separately |
| `new_testing_results/celltype_random_baselines_<hash>.tsv` | shuffled-score baselines |
| `new_testing_results/celltype_average_method_rank_<hash>.tsv` | average rank per method |
| `new_testing_results/method_comparison_<hash>.json` | runs and settings of this comparison |
| `<plot_dir>/model_vs_other_method_boxplots/` | boxplots, all models and one per model |
| `<plot_dir>/rank_plots/rank_heatmaps/`, `rank_lollipop_plots/` | per-cell-type AUROC/AUPRC, ranks |

- The default `--plot_dir` is `TETHER/new_plots`, which replaces the current figures. Use a new
  folder for each model set.
- The cache is reused when the hash matches. After regenerating GRNs for runs with the same names,
  pass `--force_relabel`.
- Without `--runs`, the script uses the 2026-10-22 holdout models.

## Step 9: Stability (optional)

Edit `MODEL_RUNS` and `OUTPUT_DIR` in `scripts/stability_analysis/generate_stability_subsample_grns.py`,
then:

```bash
sbatch TETHER/scripts/stability_analysis/run_generate_stability_subsample_grns.sh
```

Then point `stability.ipynb` at the new output folder.

## Full chain example

```bash
cd $REPO
C=$REPO/TETHER/cached_data/mm10/tf_dna_cache_<name>
J1=$(sbatch --parsable --mem=370G TETHER/bash_scripts/02a_build_tf_to_dna_cache.sh \
        --site_merge_distance 64 --false_overlap_flank 128 --cache_dir $C)
J2=$(sbatch --parsable --dependency=afterok:$J1 TETHER/bash_scripts/02b_train_tf_to_dna_model.sh --cache_dir $C)
# Wait for J2, choose the epoch in W&B, copy it (step 3), then:
J5=$(sbatch --parsable TETHER/bash_scripts/03c_build_celltype_data.sh \
        --max_cells_per_pair 25 --max_peaks_per_tg 25 --tf_dna_checkpoint <checkpoint>)
# Step 6: six sbatch calls with --dependency=afterok:$J5, keep their job IDs.
# Step 7: edit EXPERIMENT_LIST, then J7=$(sbatch --parsable --dependency=afterok:<six IDs> ...)
# Step 8: sbatch --dependency=afterok:$J7 TETHER/bash_scripts/run_compare_grn_methods.sh --runs ...
```

The chain stops twice for decisions: after step 2 (choose the epoch) and after step 6 (put the
new run names into the GRN launcher).
