#!/bin/bash -l
#SBATCH --job-name=hyperparameter_tuning
#SBATCH --output=LOGS/transformer_logs/hyperparameter_tuning/%x_%A/%x_%A_%a.log
#SBATCH --error=LOGS/transformer_logs/hyperparameter_tuning/%x_%A/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH -c 48
#SBATCH --mem=128G
#SBATCH --array=0%4

set -eo pipefail

PROJECT_SCRIPT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"

cd "$PROJECT_SCRIPT_DIR"

source activate my_env

RAW_DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/RAW_DATA"
PROCESSED_DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/PROCESSED_DATA/HYPERPARAMETER_TUNING"
EXPERIMENT_OUTPUT_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/EXPERIMENTS/HYPERPARAMETER_TUNING"
TRAINING_DATA_CACHE_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TRAINING_DATA_CACHE/HYPERPARAMETER_TUNING"

mkdir -p "$PROCESSED_DATA_DIR"
mkdir -p "$EXPERIMENT_OUTPUT_DIR"
mkdir -p "$TRAINING_DATA_CACHE_DIR"

EXPERIMENT_NAME="auroc_by_kernel_size"
MM10_N_CHROMS=19
HG38_N_CHROMS=21

# sample_type|sample_name|experiment_header|n_chroms|organism_code
EXPERIMENT_LIST=(
    # "mESC|E7.5_rep1|${EXPERIMENT_NAME}|${MM10_N_CHROMS}|mm10"
    # "mESC|E7.5_rep2|${EXPERIMENT_NAME}|${MM10_N_CHROMS}|mm10"
    # "mESC|E8.5_rep1|${EXPERIMENT_NAME}|${MM10_N_CHROMS}|mm10"
    # "mESC|E8.5_rep2|${EXPERIMENT_NAME}|${MM10_N_CHROMS}|mm10"

    "Macrophage|buffer_1|${EXPERIMENT_NAME}|${HG38_N_CHROMS}|hg38"
    "Macrophage|buffer_2|${EXPERIMENT_NAME}|${HG38_N_CHROMS}|hg38"
    "Macrophage|buffer_3|${EXPERIMENT_NAME}|${HG38_N_CHROMS}|hg38"
    "Macrophage|buffer_4|${EXPERIMENT_NAME}|${HG38_N_CHROMS}|hg38"

    "iPSC|WT_D13_rep1|${EXPERIMENT_NAME}|${HG38_N_CHROMS}|hg38"

    "K562|sample_1|${EXPERIMENT_NAME}|${HG38_N_CHROMS}|hg38"
)

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32  # optional but ok
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=48
export BLIS_NUM_THREADS=48
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
IFS='|' read -r SAMPLE_TYPE SAMPLE_NAME EXPERIMENT_HEADER N_CHROMS ORGANISM_CODE <<< "$EXPERIMENT_CONFIG"

# Match the SBATCH %x_%A/%x_%A_%a.log expansion at runtime.
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SLURM_LOG_FILE="${PROJECT_SCRIPT_DIR}/LOGS/transformer_logs/hyperparameter_tuning/${SLURM_JOB_NAME}_${ARRAY_JOB_ID}/${SLURM_JOB_NAME}_${ARRAY_JOB_ID}_${ARRAY_TASK_ID}.log"
SLURM_LOG_DIR="$(dirname "$SLURM_LOG_FILE")"

# Keep model-training logs colocated with the SLURM task logs, but isolated per sample.
MODEL_LOG_DIR="${SLURM_LOG_DIR}/${EXPERIMENT_HEADER}_${SAMPLE_NAME}"
mkdir -p "$MODEL_LOG_DIR"

echo "SLURM resolved log file: $SLURM_LOG_FILE"
echo "Model log dir: $MODEL_LOG_DIR"

# ------------------------------------------------------------
# Run hyperparameter tuning
# ------------------------------------------------------------
python ./dev/hyperparameter_tuning_new_pipeline.py \
    --sample_type "$SAMPLE_TYPE" \
    --sample_name "$SAMPLE_NAME" \
    --experiment_header "$EXPERIMENT_HEADER" \
    --n_chroms "$N_CHROMS" \
    --organism_code "$ORGANISM_CODE" \
    --raw_data_dir "$RAW_DATA_DIR" \
    --processed_data_dir "$PROCESSED_DATA_DIR" \
    --experiment_dir "$EXPERIMENT_OUTPUT_DIR" \
    --training_cache_dir "$TRAINING_DATA_CACHE_DIR" \
    --log_dir "$MODEL_LOG_DIR"

echo "finished successfully!"