#!/bin/bash -l
#SBATCH --job-name=generate_plots
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%A/%x_%A_%a.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%A/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --array=0%4

set -euo pipefail

cd "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"

source .venv/bin/activate

EXPERIMENT_DIR=${EXPERIMENT_DIR:-/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments}

PLOTTING_EXPERIMENT_LIST=(
    "mESC_lower_peak_threshold|model_training_001|checkpoint_60.pt"
    # "mESC_no_filter_to_nearest_gene|model_training_001|trained_model.pt"
    # "mESC_smaller_window_size|model_training_001|trained_model.pt"
    # "mESC_larger_window_size|model_training_001|trained_model.pt"
    # "mESC_lower_max_peak_dist|model_training_001|trained_model.pt"
    # "mESC_higher_max_peak_dist|model_training_001|trained_model.pt"
)

# ==========================================
#        EXPERIMENT SELECTION
# ==========================================
# Get the current experiment based on SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ ${TASK_ID} -ge ${#PLOTTING_EXPERIMENT_LIST[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${TASK_ID}) exceeds number of experiments (${#PLOTTING_EXPERIMENT_LIST[@]})"
    exit 1
fi

EXPERIMENT_CONFIG="${PLOTTING_EXPERIMENT_LIST[$TASK_ID]}"

# Parse experiment configuration
IFS='|' read -r EXPERIMENT_NAME TRAINING_NUM MODEL_FILE <<< "$EXPERIMENT_CONFIG"

echo ""
echo "=========================================="
echo "  EXPERIMENT: ${EXPERIMENT_NAME}"
echo "  TRAINING_NUM: ${TRAINING_NUM}"
echo "  MODEL_FILE ID: ${MODEL_FILE}"
echo "  TASK ID: ${TASK_ID}"
echo "=========================================="
echo ""

poetry run python ./src/multiomic_transformer/utils/plotting.py \
    --experiment "$EXPERIMENT_NAME" \
    --training_num "$TRAINING_NUM" \
    --experiment_dir "$EXPERIMENT_DIR" \
    --model_file "$MODEL_FILE"
echo "finished"