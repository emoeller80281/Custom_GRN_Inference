#!/bin/bash
#SBATCH --job-name=scrna_annot_mESC
#SBATCH --output=LOGS/scrna_annotation/%x_%A_%a.log
#SBATCH --error=LOGS/scrna_annotation/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --array=0-10

# scRNA QC / clustering / cell-type annotation across the mESC (mouse gastrulation)
# 10x Multiome timecourse.
#
# Thresholds come from data/qc_filtering_settings.tsv, which now carries a row per
# sample chosen from that sample's own pre-filter QC distributions (see
# TETHER/mouse_preprocessing_scripts/01_qc_scan/qc_scan.py and data/qc_scan/). MAD alone was abandoned: it is a
# relative rule, so on a globally high-mitochondrial sample it set the cap above
# the bulk of the data and filtered almost nothing.

set -eo pipefail

# `set -u` must come *after* activation: the env's MKL activate.d hook reads
# $MKL_INTERFACE_LAYER unguarded and aborts under nounset.
source activate my_env
set -u

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
RAW_DIR="${DATA_DIR}/RAW_DATA/mESC_10x_raw"
OUT_ROOT="${PROJECT_DIR}/data/processed/mESC"

SAMPLES=(
    E7.5_rep1
    E7.5_rep2
    E7.75_rep1
    E8.0_rep1
    E8.0_rep2
    E8.5_rep1
    E8.5_rep2
    E8.5_CRISPR_T_WT
    E8.5_CRISPR_T_KO
    E8.75_rep1
    E8.75_rep2
)

SAMPLE_NAME="${SAMPLES[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="${OUT_ROOT}/${SAMPLE_NAME}"

echo "=== ${SAMPLE_NAME} (array task ${SLURM_ARRAY_TASK_ID}) on $(hostname) ==="
echo "input:  ${RAW_DIR}/${SAMPLE_NAME}"
echo "output: ${OUT_DIR}"
date

mkdir -p "${OUT_DIR}"

python "${PROJECT_DIR}/TETHER/mouse_preprocessing_scripts/02_per_sample_annotation/annotate_scrna_celltypes.py" \
    --input_dir "${RAW_DIR}/${SAMPLE_NAME}" \
    --sample_name "${SAMPLE_NAME}" \
    --out_dir "${OUT_DIR}" \
    --marker_panel mouse_gastrulation \
    --seed 0

echo "=== ${SAMPLE_NAME} done ==="
date

# The mtx read cache is only useful for a rerun of the same sample and is large
# (~1-2 GB); drop it so the timecourse does not add tens of GB of scratch.
rm -rf "${OUT_DIR}/cache"
