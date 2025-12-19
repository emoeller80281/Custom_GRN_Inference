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
#SBATCH --array=0-1%6

set -euo pipefail

cd "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"

source .venv/bin/activate

EXPERIMENT_DIR=${EXPERIMENT_DIR:-/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments}

EXPERIMENT_LIST=(
    # "mESC_no_scale_linear|model_training_128_10k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_1k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_5k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_10k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_10k_metacells_6_layers|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_10k_metacells_8_heads|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_15k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_50k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_256_15k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_256_20k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_320_10k_metacells|trained_model.pt"
    # "mESC_large_neighborhood_count_filter|model_training_001|trained_model.pt"
    # "mESC_large_neighborhood|model_training_001|trained_model.pt"
    # "mESC_small_neighborhood|model_training_001|trained_model.pt"
    # "mESC_small_neighborhood_high_self_weight|model_training_001|trained_model.pt"
    # "mESC_slower_dist_decay|model_training_001|trained_model.pt"
    # "mESC_max_dist_bias|model_training_002|trained_model.pt"
    # "mESC_slow_decay_max_dist|model_training_001|trained_model.pt"
    # "mESC_filter_lowest_ten_pct|model_training_003|trained_model.pt"
    # "mESC_lower_peak_threshold|model_training_001|trained_model.pt"
    "mESC_no_filter_to_nearest_gene|model_training_001|trained_model.pt"
    # "mESC_smaller_window_size|model_training_001|trained_model.pt"
    # "mESC_larger_window_size|model_training_001|trained_model.pt"
    "mESC_lower_max_peak_dist|model_training_001|trained_model.pt"
    # "mESC_higher_max_peak_dist|model_training_001|trained_model.pt"
    # "mESC_test_new_pipeline|model_training_001|trained_model.pt"
    # "mESC_slow_decay_filter_ten_pct|model_training_001|trained_model.pt"
)

# ==========================================
#        EXPERIMENT SELECTION
# ==========================================
# Get the current experiment based on SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ ${TASK_ID} -ge ${#EXPERIMENT_LIST[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${TASK_ID}) exceeds number of experiments (${#EXPERIMENT_LIST[@]})"
    exit 1
fi

EXPERIMENT_CONFIG="${EXPERIMENT_LIST[$TASK_ID]}"

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

echo "Plotting Training Figures"
poetry run python ./src/multiomic_transformer/utils/plotting.py \
    --experiment "$EXPERIMENT_NAME" \
    --training_num "$TRAINING_NUM" \
    --experiment_dir "$EXPERIMENT_DIR" \
    --model_file "$MODEL_FILE"

echo "Running AUROC Testing"
poetry run python ./src/multiomic_transformer/utils/auroc_testing.py \
    --experiment "$EXPERIMENT_NAME" \
    --training_num "$TRAINING_NUM" \
    --experiment_dir "$EXPERIMENT_DIR" \
    --model_file "$MODEL_FILE"

echo "finished"