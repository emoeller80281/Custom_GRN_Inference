#!/bin/bash
#SBATCH --job-name=centroid_annot
#SBATCH --output=LOGS/scrna_annotation/%x_%j.log
#SBATCH --error=LOGS/scrna_annotation/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH -p memory
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=700G

set -eo pipefail
source activate my_env
set -u
PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
cd "${PROJECT_DIR}"
python "${PROJECT_DIR}/TETHER/mouse_preprocessing_scripts/05_atlas_annotation/centroid_label_transfer.py" \
    --h5mu "${PROJECT_DIR}/data/processed/mESC/combined/mESC_combined.h5mu"
