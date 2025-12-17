#!/bin/bash -l
#SBATCH --job-name=auroc_testing
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%A/%x_%A_%a.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%A/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --array=2

set -euo pipefail

cd "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"

source .venv/bin/activate

EXPERIMENT_DIR=${EXPERIMENT_DIR:-/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments}

# EXPERIMENT_DIR_LIST=(
#     $EXPERIMENT_DIR/mESC_no_scale_linear/model_training_192_10k_metacells
#     $EXPERIMENT_DIR/mESC_large_neighborhood/chr19/model_training_001
#     $EXPERIMENT_DIR/mESC_small_neighborhood/chr19/model_training_001
#     $EXPERIMENT_DIR/mESC_small_neighborhood_high_self_weight/chr19/model_training_001
#     $EXPERIMENT_DIR/mESC_max_dist_bias/chr19/model_training_002
#     $EXPERIMENT_DIR/mESC_slow_decay_max_dist/chr19/model_training_001
#     $EXPERIMENT_DIR/mESC_filter_lowest_ten_pct/chr19/model_training_003
# )

EXPERIMENT_DIR_LIST=(
    $EXPERIMENT_DIR/mESC_lower_peak_threshold/chr19/model_training_001
    $EXPERIMENT_DIR/mESC_no_filter_to_nearest_gene/chr19/model_training_001
    $EXPERIMENT_DIR/mESC_lower_max_peak_dist/chr19/model_training_001
)

# EXPERIMENT_DIR_LIST=(
#     $EXPERIMENT_DIR/mESC_filter_lowest_ten_pct/chr19/model_training_003/fine_tuning
# )

# Select the experiment for this array task
SELECTED_EXPERIMENT_DIR=${EXPERIMENT_DIR_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Running auroc testing on experiment directory: $SELECTED_EXPERIMENT_DIR"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

poetry run python ./dev/auroc_testing.py \
    --experiment_dir_list "$SELECTED_EXPERIMENT_DIR"

echo "finished"