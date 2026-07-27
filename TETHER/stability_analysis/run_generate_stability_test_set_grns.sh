#!/bin/bash -l
#SBATCH --job-name=generate_model_stability_grns
#SBATCH --output=LOGS/stability_grns/stability_grns_%A/%x_%A_%a.log
#SBATCH --error=LOGS/stability_grns/stability_grns_%A/%x_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --array=0-139%10

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate my_env

EXPERIMENT_LIST=(
    # Own-model runs (model trained and tested on same sample)
    "mESC|E7.5_rep1|mESC|E7.5_rep1"
    "mESC|E8.5_rep1|mESC|E8.5_rep1"
    "Macrophage|buffer_1|Macrophage|buffer_1"
    "Macrophage|buffer_2|Macrophage|buffer_2"
    "K562|sample_1|K562|sample_1"
    "mouse_hepatocytes|hepatocytes_1|mouse_hepatocytes|hepatocytes_1"
    "mouse_hepatocytes|hepatocytes_3|mouse_hepatocytes|hepatocytes_3"

    # # Cross-trained model runs (from samples_to_run)
    "mouse_hepatocytes|hepatocytes_1|mESC|E7.5_rep1"
    "mouse_hepatocytes|hepatocytes_1|mESC|E8.5_rep1"
    "K562|sample_1|Macrophage|buffer_1"
    "K562|sample_1|Macrophage|buffer_2"
    "Macrophage|buffer_1|K562|sample_1"
    "mESC|E7.5_rep1|mouse_hepatocytes|hepatocytes_1"
    "mESC|E7.5_rep1|mouse_hepatocytes|hepatocytes_3"
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

NUM_SUBSAMPLES=10

# Get the current experiment and subsample based on SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
EXPERIMENT_IDX=$((TASK_ID / NUM_SUBSAMPLES))
SUBSAMPLE_NUMBER=$((TASK_ID % NUM_SUBSAMPLES))

if [ ${EXPERIMENT_IDX} -ge ${#EXPERIMENT_LIST[@]} ]; then
    echo "Error: Experiment index out of bounds"
    exit 1;
fi

EXPERIMENT_CONFIG="${EXPERIMENT_LIST[$EXPERIMENT_IDX]}"

# Parse experiment configuration
IFS='|' read -r model_cell_type model_training_sample test_set_cell_type evaluation_sample <<< "$EXPERIMENT_CONFIG"

echo "[INFO] Running experiment with:"
echo "  model_cell_type=$model_cell_type"
echo "  model_training_sample=$model_training_sample"
echo "  test_set_cell_type=$test_set_cell_type"
echo "  evaluation_sample=$evaluation_sample"
echo "  stability_number=$SUBSAMPLE_NUMBER"

echo "[INFO] Analyzing stability..."
srun python3 ${PROJECT_DIR}/stability_analysis/generate_stability_test_set_grns.py \
    --model_cell_type "$model_cell_type" \
    --model_training_sample "$model_training_sample" \
    --test_set_cell_type "$test_set_cell_type" \
    --evaluation_sample "$evaluation_sample" \
    --stability_number "$SUBSAMPLE_NUMBER" \
    --batch_size 256 \
    --force_reload