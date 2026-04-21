#!/bin/bash -l
#SBATCH --job-name=stability_full_pipeline
#SBATCH --output=LOGS/transformer_logs/pipeline_logs/%x_%A/%x_%A_%a.log
#SBATCH --error=LOGS/transformer_logs/pipeline_logs/%x_%A/%x_%A_%a.err
#SBATCH --time=18:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=92G
#SBATCH --array=0%4

set -eo pipefail

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER

source activate my_env

RAW_DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/RAW_DATA"
STABILITY_RAW_DATASETS_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/STABILITY_RAW_DATASETS"
PROCESSED_DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/PROCESSED_DATA/STABILITY"
EXPERIMENT_OUTPUT_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/EXPERIMENTS/STABILITY"
TRAINING_DATA_CACHE_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TRAINING_DATA_CACHE"

EXPERIMENT_LIST=(
    # "mESC_E7.5_rep1_full_pipeline|E7.5_rep1|mm10|mESC"
    # "mESC_E7.5_rep2_full_pipeline|E7.5_rep2|mm10|mESC"
    # "mESC_E8.5_rep1_full_pipeline|E8.5_rep1|mm10|mESC"
    # "mESC_E8.5_rep2_full_pipeline|E8.5_rep2|mm10|mESC"

    # "Macrophage_buffer_1_full_pipeline|buffer_1|hg38|Macrophage"
    # "Macrophage_buffer_2_full_pipeline|buffer_2|hg38|Macrophage"
    # "Macrophage_buffer_3_full_pipeline|buffer_3|hg38|Macrophage"
    # "Macrophage_buffer_4_full_pipeline|buffer_4|hg38|Macrophage"

    # "iPSC_WT_D13_rep1_full_pipeline|WT_D13_rep1|hg38|iPSC"

    "K562_sample_1_full_pipeline|sample_1|hg38|K562"
)

NUM_SUBSAMPLES=10
NUM_EXPERIMENTS=${#EXPERIMENT_LIST[@]}
TOTAL_TASKS=$((NUM_EXPERIMENTS * NUM_SUBSAMPLES))

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-unset}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export NUMEXPR_NUM_THREADS=12
export BLIS_NUM_THREADS=12
export KMP_AFFINITY=granularity=fine,compact,1,0

# ==========================================
#        TASK SELECTION
# ==========================================
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ "${TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${TASK_ID}) exceeds total number of tasks (${TOTAL_TASKS})"
    exit 1
fi

EXPERIMENT_IDX=$((TASK_ID / NUM_SUBSAMPLES))
SUBSAMPLE_IDX=$((TASK_ID % NUM_SUBSAMPLES))
SUBSAMPLE_NUM=$((SUBSAMPLE_IDX + 1))

EXPERIMENT_CONFIG="${EXPERIMENT_LIST[$EXPERIMENT_IDX]}"

# Parse experiment configuration
IFS='|' read -r BASE_EXPERIMENT_NAME SAMPLE_NAME ORGANISM_CODE SAMPLE_TYPE <<< "$EXPERIMENT_CONFIG"

# Create unique experiment name per subsample
EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_70pct_subsample_${SUBSAMPLE_NUM}"

# Build raw h5mu path
RAW_H5_DATA_FILE="${STABILITY_RAW_DATASETS_DIR}/${SAMPLE_TYPE}_10x_raw/${SAMPLE_NAME}/70pct_subsample_${SUBSAMPLE_NUM}.h5mu"

echo "----------------------------------------"
echo "Experiment index: ${EXPERIMENT_IDX}"
echo "Subsample index: ${SUBSAMPLE_IDX}"
echo "Subsample number: ${SUBSAMPLE_NUM}"
echo "Base experiment name: ${BASE_EXPERIMENT_NAME}"
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Sample name: ${SAMPLE_NAME}"
echo "Sample type: ${SAMPLE_TYPE}"
echo "Organism code: ${ORGANISM_CODE}"
echo "Raw h5mu file: ${RAW_H5_DATA_FILE}"
echo "----------------------------------------"

if [ ! -f "${RAW_H5_DATA_FILE}" ]; then
    echo "ERROR: raw subsample file not found:"
    echo "  ${RAW_H5_DATA_FILE}"
    exit 1
fi

# Optional: stagger starts slightly to reduce simultaneous filesystem hits
sleep $((SUBSAMPLE_IDX * 5))

# ------------------------------------------------------------
# Run full pipeline
# ------------------------------------------------------------
torchrun --standalone --nnodes=1 --nproc_per_node=1 dev/full_pipeline.py \
    --experiment_name "${EXPERIMENT_NAME}" \
    --sample_name "${SAMPLE_NAME}" \
    --raw_data_dir "${RAW_DATA_DIR}" \
    --organism_code "${ORGANISM_CODE}" \
    --sample_type "${SAMPLE_TYPE}" \
    --processed_data_dir "${PROCESSED_DATA_DIR}" \
    --experiment_output_dir "${EXPERIMENT_OUTPUT_DIR}" \
    --training_data_cache_dir "${TRAINING_DATA_CACHE_DIR}" \
    --raw_h5_data_file "${RAW_H5_DATA_FILE}"

echo "finished successfully!"