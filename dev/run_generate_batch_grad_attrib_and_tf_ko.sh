#!/bin/bash -l
#SBATCH --job-name=generate_batch_grad
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%A_%a.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --array=0%4

set -eo pipefail

cd "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"

source activate my_env

EXPERIMENT_DIR=${EXPERIMENT_DIR:-/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments}

EXPERIMENT_LIST=(
    "iPSC_hvg_filter_disp_0.2|model_training_001|iPSC|iPSC_sample"
)

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export BLIS_NUM_THREADS=8
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
IFS='|' read -r EXPERIMENT_NAME TRAINING_NUM DATASET_TYPE SAMPLE_NAMES <<< "$EXPERIMENT_CONFIG"

echo ""
echo "=========================================="
echo "  EXPERIMENT: ${EXPERIMENT_NAME}"
echo "  TRAINING_NUM: ${TRAINING_NUM}"
echo "  TASK ID: ${TASK_ID}"
echo "=========================================="
echo ""

echo "Running generate_batch_grad_attrib_and_tf_ko.py"
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    dev/generate_batch_grad_attrib_and_tf_ko.py \
        --experiment_name "${EXPERIMENT_NAME}" \
        --model_num "${TRAINING_NUM}" \
        --checkpoint_name "trained_model.pt" \
        --batch_size 16 \
        --save_every_n_batches 0

echo "finished successfully!"