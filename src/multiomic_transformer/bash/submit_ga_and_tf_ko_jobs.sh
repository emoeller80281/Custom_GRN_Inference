#!/bin/bash -l
#SBATCH --job-name=submit_ga_and_tf_ko
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem=8G

set -euo pipefail

cd "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"

# ------------------------------------------------------------
# Define experiment directory
# ------------------------------------------------------------
EXPERIMENT_DIR=/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC_no_scale_linear
SELECTED_EXPERIMENT_DIR=$EXPERIMENT_DIR/model_training_128_10k_metacells
MODEL_FILE=trained_model.pt

COMMON_EXPORT="ALL,EXPERIMENT_DIR=$EXPERIMENT_DIR,SELECTED_EXPERIMENT_DIR=$SELECTED_EXPERIMENT_DIR,MODEL_FILE=$MODEL_FILE"

# ------------------------------------------------------------
# Submit TF knockout job
# ------------------------------------------------------------
JOB1=$(sbatch --export="$COMMON_EXPORT" -J tf_knockout src/multiomic_transformer/bash/run_attn_template.sh tf_knockout | awk '{print $4}')
echo "Submitted TF knockout job: $JOB1"

# ------------------------------------------------------------
# Submit gradient attribution job
# ------------------------------------------------------------
JOB2=$(sbatch --export="$COMMON_EXPORT" -J grad_attrib src/multiomic_transformer/bash/run_attn_template.sh grad_attrib | awk '{print $4}')
echo "Submitted gradient attribution job: $JOB2"

# ------------------------------------------------------------
# Wait for both jobs to finish
# ------------------------------------------------------------
JOB3=$(sbatch \
  --dependency=afterok:${JOB1}:${JOB2} \
  --export="$COMMON_EXPORT" \
  -J auroc_testing \
  dev/run_auroc_testing.sh | awk '{print $4}')

echo "All jobs finished successfully!"