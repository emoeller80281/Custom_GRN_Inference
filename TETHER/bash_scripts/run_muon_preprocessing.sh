#!/bin/bash -l
#SBATCH --job-name=muon_preprocessing
#SBATCH --output=LOGS/muon_preprocessing/%x_%A_%a.log
#SBATCH --error=LOGS/muon_preprocessing/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --array=0-9%3

set -eo pipefail

source activate my_env

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"

# These samples' raw data lives on the DATA mount (already renamed to the plain
# barcodes/features/matrix/fragments names muon_preprocessing.py expects), not
# under PROJECT_DIR/data/raw like the older mESC_10x_data samples.
RAW_DATA_ROOT="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/RAW_DATA"

# Each entry: dataset_name|cell_type|organism_code|sample_name
EXPERIMENT_LIST=(
    "mESC_10x_raw|mESC|mm10|E7.5_rep2"
    "mESC_10x_raw|mESC|mm10|E7.75_rep1"
    "mESC_10x_raw|mESC|mm10|E8.0_rep1"
    "mESC_10x_raw|mESC|mm10|E8.0_rep2"
    "mESC_10x_raw|mESC|mm10|E8.5_rep1"
    "mESC_10x_raw|mESC|mm10|E8.5_rep2"
    "mESC_10x_raw|mESC|mm10|E8.5_CRISPR_T_KO"
    "mESC_10x_raw|mESC|mm10|E8.5_CRISPR_T_WT"
    "mESC_10x_raw|mESC|mm10|E8.75_rep1"
    "mESC_10x_raw|mESC|mm10|E8.75_rep2"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ ${TASK_ID} -ge ${#EXPERIMENT_LIST[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${TASK_ID}) exceeds number of experiments (${#EXPERIMENT_LIST[@]})"
    exit 1
fi

EXPERIMENT_CONFIG="${EXPERIMENT_LIST[$TASK_ID]}"
IFS='|' read -r DATASET_NAME CELL_TYPE ORGANISM_CODE SAMPLE_NAME <<< "$EXPERIMENT_CONFIG"

echo "[INFO] Running muon preprocessing for:"
echo "  dataset_name=$DATASET_NAME"
echo "  cell_type=$CELL_TYPE"
echo "  organism_code=$ORGANISM_CODE"
echo "  sample_name=$SAMPLE_NAME"

TSS_PATH="${PROJECT_DIR}/data/genome_data/genome_annotation/${ORGANISM_CODE}/gene_tss.bed"
TF_LIST_FILE=""

RAW_DATA_DIR="${RAW_DATA_ROOT}/${DATASET_NAME}/"
PROCESSED_DATA_DIR="${PROJECT_DIR}/data/sample_input_data/${CELL_TYPE}"

FRAG_PATH="${RAW_DATA_DIR}/${SAMPLE_NAME}/fragments.tsv.gz"

# Optional inputs for alternate loading modes.
RNA_COUNT_FILE=""
ATAC_COUNT_FILE=""
RAW_H5_FILE=""

python $PROJECT_DIR/TETHER/muon_preprocessing.py \
    --project-dir "${PROJECT_DIR}" \
    --tss-path "${TSS_PATH}" \
    --raw-data-dir "${RAW_DATA_DIR}" \
    --processed-data-dir "${PROCESSED_DATA_DIR}" \
    --sample-name "${SAMPLE_NAME}" \
    --rna-count-file "${RNA_COUNT_FILE}" \
    --atac-count-file "${ATAC_COUNT_FILE}" \
    --raw-h5-file "${RAW_H5_FILE}" \
    --tf-list-file "${TF_LIST_FILE}" \
    --frag-path "${FRAG_PATH}"
