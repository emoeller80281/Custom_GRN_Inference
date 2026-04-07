#!/bin/bash
#SBATCH --job-name=generate_batch_grad_Attrib
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=128G

set -euo pipefail

source activate my_env

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/"

DATASET_NAME="iPSC_10x_raw"
SAMPLE_NAME="WT_D25_rep1"
PROCESSED_DATA_NAME="iPSC_${SAMPLE_NAME}_muon_preprocessing"
ORGANISM_CODE="hg38"

TSS_PATH="${PROJECT_DIR}/data/genome_data/genome_annotation/${ORGANISM_CODE}/gene_tss.bed"
TF_LIST_FILE=""

RAW_DATA_DIR="${PROJECT_DIR}/data/raw/${DATASET_NAME}/"
PROCESSED_DATA_DIR="${PROJECT_DIR}/data/processed/${PROCESSED_DATA_NAME}"

FRAG_PATH="${RAW_DATA_DIR}/${SAMPLE_NAME}/fragments.tsv.gz"

# Optional inputs for alternate loading modes.
RNA_COUNT_FILE=""
ATAC_COUNT_FILE=""
RAW_H5_FILE=""

python /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/muon_preprocessing.py \
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
