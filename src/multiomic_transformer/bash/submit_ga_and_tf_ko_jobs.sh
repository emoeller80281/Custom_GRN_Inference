#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------
# Define experiment directory
# ------------------------------------------------------------
EXPERIMENT_DIR=/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC_no_scale_linear
SELECTED_EXPERIMENT_DIR=$EXPERIMENT_DIR/model_training_192_10k_metacells
MODEL_FILE=checkpoint_195.pt

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
