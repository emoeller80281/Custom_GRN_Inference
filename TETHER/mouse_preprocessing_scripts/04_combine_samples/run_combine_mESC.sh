#!/bin/bash
#SBATCH --job-name=combine_mESC
#SBATCH --output=LOGS/scrna_annotation/%x_%j.log
#SBATCH --error=LOGS/scrna_annotation/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p memory
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=750G

# Combine the wild-type mESC gastrulation timecourse into one Harmony-integrated
# MuData (RNA + consensus-peak ATAC). The E8.5 CRISPR T-KO/T-WT pair is excluded.

set -eo pipefail
source activate my_env
set -u

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
cd "${PROJECT_DIR}"

python "${PROJECT_DIR}/TETHER/mouse_preprocessing_scripts/04_combine_samples/combine_mesc_samples.py" \
    --in_root "${PROJECT_DIR}/data/processed/mESC" \
    --out_dir "${PROJECT_DIR}/data/processed/mESC/combined" \
    --seed 0
