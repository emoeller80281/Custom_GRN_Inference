#!/bin/bash -l
#SBATCH --job-name=full_pipeline
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%A/%x_%A_%a.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%A/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 12
#SBATCH --mem=64G

set -eo pipefail

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER

source activate my_env

RAW_DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/RAW_DATA"
PROCESSED_DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/PROCESSED_DATA"
EXPERIMENT_OUTPUT_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments"


EXPERIMENT_LIST=(
    "mESC_E7.5_rep1_full_pipeline|E7.5_rep1|mm10|mESC"
)

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32  # optional but ok
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export BLIS_NUM_THREADS=12
export KMP_AFFINITY=granularity=fine,compact,1,0

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
IFS='|' read -r EXPERIMENT_NAME SAMPLE_NAME ORGANISM_CODE SAMPLE_TYPE <<< "$EXPERIMENT_CONFIG"


# ------------------------------------------------------------
# Run full pipeline
# ------------------------------------------------------------
torchrun --standalone --nnodes=1 --nproc_per_node=1 dev/full_pipeline.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --sample_name "$SAMPLE_NAME" \
    --organism_code "$ORGANISM_CODE" \
    --sample_type "$SAMPLE_TYPE" \
    --raw_data_dir "$RAW_DATA_DIR" \
    --processed_data_dir "$PROCESSED_DATA_DIR" \
    --experiment_output_dir "$EXPERIMENT_OUTPUT_DIR"

echo "finished successfully!"