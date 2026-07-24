#!/bin/bash -l
#SBATCH --job-name=analyze_stability
#SBATCH --output=LOGS/stability_analysis/analyze_stability_%A/%x_%A_%a.log
#SBATCH --error=LOGS/stability_analysis/analyze_stability_%A/%x_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --array=0-79%10

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate my_env

EXPERIMENT_LIST=(
    "mESC|E7.5_rep1"
    "mESC|E8.5_rep1"
    "mouse_hepatocytes|hepatocytes_1"
    "mouse_hepatocytes|hepatocytes_3"
    "Macrophage|buffer_1"
    "Macrophage|buffer_2"
    "K562|sample_1"
    "iPSC|WT_D13_rep1"
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

EXPERIMENT_CONFIG="${EXPERIMENT_LIST[$EXPERIMENT_IDX]}"

if [ ${EXPERIMENT_IDX} -ge ${#EXPERIMENT_LIST[@]} ]; then 
    echo "Error: Experiment index out of bounds"
    exit 1; 
fi

# Parse experiment configuration
IFS='|' read -r cell_type sample_name <<< "$EXPERIMENT_CONFIG"

echo "[INFO] Running experiment with cell_type=$cell_type, sample_name=$sample_name, subsample_number=$SUBSAMPLE_NUMBER"

echo "[INFO] Running experiment with:"
echo "  cell_type=$cell_type"
echo "  sample_name=$sample_name"
echo "  subsample_number=$SUBSAMPLE_NUMBER"

echo "[INFO] Analyzing stability..."
srun python3 ${PROJECT_DIR}/stability_analysis/analyze_stability.py \
    --cell_type "$cell_type" \
    --sample_name "$sample_name" \
    --subsample_num "$SUBSAMPLE_NUMBER" \
    --batch_size 256
