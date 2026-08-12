#!/bin/bash
#SBATCH --job-name=inspect_disputed
#SBATCH --output=LOGS/scrna_annotation/%x_%j.log
#SBATCH --error=LOGS/scrna_annotation/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=180G

set -eo pipefail
source activate my_env
set -u
PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
cd "${PROJECT_DIR}"
python "${PROJECT_DIR}/TETHER/mouse_preprocessing_scripts/06_label_validation/inspect_unresolved_clusters.py" \
    --h5mu "${PROJECT_DIR}/data/processed/mESC/combined/mESC_combined.h5mu"
