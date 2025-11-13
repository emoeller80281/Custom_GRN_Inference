#!/bin/bash
#SBATCH --job-name=run_tf_binding_model
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=128G


set -euo pipefail

# Ensure log directory exists (Slurm won't create it)
mkdir -p LOGS/transformer_logs/04_testing

# (Recommended) Make sure conda is available in this non-interactive shell
# If your cluster sets this up differently, adjust these 2 lines accordingly.
source activate my_env
# Optional: echo some context
echo "Job: ${SLURM_JOB_NAME:-aggregate_features}  ID: ${SLURM_JOB_ID:-N/A}"
echo "Running on: $(hostname)  at: $(date)"

# Launch
srun -n 1 python dev/tf_binding_model_pipeline.py \
  --ground-truth \
    data/ground_truth_files/chipatlas_mESC.csv \
    data/ground_truth_files/mESC_beeline_ChIP-seq.csv \
    data/ground_truth_files/chip_atlas_tf_peak_tg_dist.csv \
    data/ground_truth_files/chipatlas_beeline_mESC_shared_edges.csv \
    /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/testing_bear_grn/GROUND.TRUTHS/filtered_RN111_and_RN112_mESC_E7.5_rep1.tsv \
  --peak-data "data/raw/mESC_no_scale_linear/E7.5_rep1/scATAC_seq_processed.parquet" \
  --gene-data "data/raw/mESC_no_scale_linear/E7.5_rep1/scRNA_seq_processed.parquet" \
  --split-strategy edge \
  --cv-strategy leave_one_out \
  --model-type hgb \
  --neg-pos-ratio 1