#!/bin/bash -l
#SBATCH --job-name=model_generalizability
#SBATCH --output=LOGS/model_performance/model_generalizability_%A/%x_%A_%a.log
#SBATCH --error=LOGS/model_performance/model_generalizability_%A/%x_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --array=0-13%3

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate my_env

EXPERIMENT_COMBO_ITEMS=(
    "mESC|E7.5_rep1"
    "mESC|E8.5_rep1"
    "mouse_hepatocytes|hepatocytes_1"
    "mouse_hepatocytes|hepatocytes_3"
    "Macrophage|buffer_1"
    "Macrophage|buffer_2"
    "K562|sample_1"
    "iPSC|WT_D13_rep1"
)

# All model-training sample × test-set sample combinations
ALL_COMBINATIONS=()
for model_combo in "${EXPERIMENT_COMBO_ITEMS[@]}"; do
    for test_combo in "${EXPERIMENT_COMBO_ITEMS[@]}"; do
        ALL_COMBINATIONS+=("${model_combo}|${test_combo}")
    done
done

CURATED_EXPERIMENT_LIST=(
    # === mESC Evaluations ====
    # Same cell-type, same sample evaluations with own sample test sets
    "mESC|E7.5_rep1|mESC|E7.5_rep1"
    "mESC|E8.5_rep1|mESC|E8.5_rep1"

    # # # Same cell-type, different sample evaluations with mouse hepatocyte test sets
    # "mESC|E7.5_rep1|mESC|E8.5_rep1"
    # "mESC|E8.5_rep1|mESC|E7.5_rep1"

    # # # Cross cell-type, same organism evaluations with mESC test sets
    "mESC|E7.5_rep1|mouse_hepatocytes|hepatocytes_1"
    # "mESC|E7.5_rep1|mouse_hepatocytes|hepatocytes_3"
    "mESC|E8.5_rep1|mouse_hepatocytes|hepatocytes_1"
    # "mESC|E8.5_rep1|mouse_hepatocytes|hepatocytes_3"

    # # # Cross cell-type, different organism evaluations with Macrophage test sets
    # "mESC|E7.5_rep1|Macrophage|buffer_1"
    # "mESC|E7.5_rep1|Macrophage|buffer_2"
    # "mESC|E8.5_rep1|Macrophage|buffer_1"
    # "mESC|E8.5_rep1|Macrophage|buffer_2"
    
    # # # ==== Hepatocyte Evaluations ====
    # # # Same cell-type, same sample evaluations with own sample test sets
    "mouse_hepatocytes|hepatocytes_1|mouse_hepatocytes|hepatocytes_1"
    "mouse_hepatocytes|hepatocytes_3|mouse_hepatocytes|hepatocytes_3"
    
    # # # Same cell-type, different sample evaluations with mouse hepatocyte test sets
    # "mouse_hepatocytes|hepatocytes_1|mouse_hepatocytes|hepatocytes_3"
    # "mouse_hepatocytes|hepatocytes_3|mouse_hepatocytes|hepatocytes_1"
    
    # # # Cross cell-type, same organism evaluations with mESC test sets
    "mouse_hepatocytes|hepatocytes_1|mESC|E7.5_rep1"
    # "mouse_hepatocytes|hepatocytes_1|mESC|E8.5_rep1"
    "mouse_hepatocytes|hepatocytes_3|mESC|E7.5_rep1"
    # "mouse_hepatocytes|hepatocytes_3|mESC|E8.5_rep1"
    
    # # # Cross cell-type, different organism evaluations with Macrophage test sets
    # "mouse_hepatocytes|hepatocytes_1|Macrophage|buffer_1"
    # "mouse_hepatocytes|hepatocytes_1|Macrophage|buffer_2"
    # "mouse_hepatocytes|hepatocytes_3|Macrophage|buffer_1"
    # "mouse_hepatocytes|hepatocytes_3|Macrophage|buffer_2"
    
    # # # === Macrophage Evaluations ====
    # # # Same cell-type, same sample evaluations with own sample test sets
    "Macrophage|buffer_1|Macrophage|buffer_1"
    "Macrophage|buffer_2|Macrophage|buffer_2"
    
    # # # Same cell-type, different sample evaluations with Macrophage test sets
    # "Macrophage|buffer_1|Macrophage|buffer_2"
    # "Macrophage|buffer_2|Macrophage|buffer_1"

    # # Different cell-type, same organism evaluations with K562 test sets
    "Macrophage|buffer_1|K562|sample_1"
    "Macrophage|buffer_2|K562|sample_1"

    # # # Cross cell-type, different organism evaluations with mESC test sets
    # "Macrophage|buffer_1|mESC|E7.5_rep1"
    # "Macrophage|buffer_1|mESC|E8.5_rep1"
    # "Macrophage|buffer_2|mESC|E7.5_rep1"
    # "Macrophage|buffer_2|mESC|E8.5_rep1"
    
    # # # Cross-cell type, different organism evaluations with mouse hepatocyte test sets
    # "Macrophage|buffer_1|mouse_hepatocytes|hepatocytes_1"
    # "Macrophage|buffer_1|mouse_hepatocytes|hepatocytes_3"
    # "Macrophage|buffer_2|mouse_hepatocytes|hepatocytes_1"
    # "Macrophage|buffer_2|mouse_hepatocytes|hepatocytes_3"
    
    "K562|sample_1|K562|sample_1"
    "K562|sample_1|Macrophage|buffer_1"
    # "K562|sample_1|Macrophage|buffer_2"
)

# --- Memory + math ---
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

# --- Threading ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

# ==========================================
#        EXPERIMENT SELECTION
# ==========================================
EXPERIMENT_MODE=curated

case "$EXPERIMENT_MODE" in
    curated)
        EXPERIMENT_LIST=("${CURATED_EXPERIMENT_LIST[@]}")
        ;;
    all)
        EXPERIMENT_LIST=("${ALL_COMBINATIONS[@]}")
        ;;
    *)
        echo "ERROR: Unknown EXPERIMENT_MODE='$EXPERIMENT_MODE'"
        echo "Valid options are: curated, all"
        exit 1
        ;;
esac

echo "[INFO] EXPERIMENT_MODE=${EXPERIMENT_MODE}"
echo "[INFO] Number of experiments: ${#EXPERIMENT_LIST[@]}"

# Get the current experiment based on SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ "$TASK_ID" -ge "${#EXPERIMENT_LIST[@]}" ]; then
    echo "[INFO] SLURM_ARRAY_TASK_ID (${TASK_ID}) exceeds number of experiments (${#EXPERIMENT_LIST[@]}). Skipping."
    exit 0
fi

EXPERIMENT_CONFIG="${EXPERIMENT_LIST[$TASK_ID]}"

# Parse experiment configuration
IFS='|' read -r model_cell_type model_training_sample test_set_cell_type evaluation_sample <<< "$EXPERIMENT_CONFIG"

echo "[INFO] Running experiment with:"
echo "  model_cell_type=$model_cell_type"
echo "  model_training_sample=$model_training_sample"
echo "  test_set_cell_type=$test_set_cell_type"
echo "  evaluation_sample=$evaluation_sample"


echo "[INFO] Running model generalizability test..."
srun python3 ${PROJECT_DIR}/model_generalizability.py \
    --model_cell_type "$model_cell_type" \
    --model_training_sample "$model_training_sample" \
    --test_set_cell_type "$test_set_cell_type" \
    --evaluation_sample "$evaluation_sample" \
    --batch_size 256
