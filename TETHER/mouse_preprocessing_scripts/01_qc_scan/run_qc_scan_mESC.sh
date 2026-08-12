#!/bin/bash
#SBATCH --job-name=qc_scan_mESC
#SBATCH --output=LOGS/scrna_annotation/%x_%A_%a.log
#SBATCH --error=LOGS/scrna_annotation/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH --array=0-10

set -eo pipefail
source activate my_env
set -u

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
RAW_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/RAW_DATA/mESC_10x_raw"

SAMPLES=(E7.5_rep1 E7.5_rep2 E7.75_rep1 E8.0_rep1 E8.0_rep2 E8.5_rep1 E8.5_rep2 \
         E8.5_CRISPR_T_WT E8.5_CRISPR_T_KO E8.75_rep1 E8.75_rep2)
S="${SAMPLES[$SLURM_ARRAY_TASK_ID]}"

python "${PROJECT_DIR}/TETHER/mouse_preprocessing_scripts/01_qc_scan/qc_scan.py" \
    --input_dir "${RAW_DIR}/${S}" --sample_name "${S}" \
    --out_dir "${PROJECT_DIR}/data/qc_scan"
