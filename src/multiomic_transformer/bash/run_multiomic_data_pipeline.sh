#!/bin/bash -l
#SBATCH --job-name=multiomic_pipeline
#SBATCH --output=LOGS/pipeline_logs/%x_%A.log
#SBATCH --error=LOGS/pipeline_logs/%x_%A.err
#SBATCH --time=4:00:00
#SBATCH -p memory
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH --mem=256G

set -euo pipefail

# -------------------------------------------------------------------------
#  Environment setup
# -------------------------------------------------------------------------
cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER

# Load poetry environment
# Note: you use `source activate` due to conda incompatibilities.
source activate my_env

# -------------------------------------------------------------------------
#  Arguments
# -------------------------------------------------------------------------
DATASET=${1:-"mESC"}
ORGANISM=${2:-"mm10"}
OUTDIR=${3:-"/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/outputs/${DATASET}"}
START_STAGE=${4:-1}
STOP_STAGE=${5:-6}
FORCE_FLAG=${6:-""}

RNA_PATH="data/processed/mESC/E7.5_rep1/E7.5_rep1_RNA_qc.h5ad"
ATAC_PATH="data/processed/mESC/E7.5_rep1/E7.5_rep1_ATAC_qc.h5ad"

# Example:
# sbatch run_multiomic_data_pipeline.sh mESC mm10 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/outputs/mESC 1 6 --force

# -------------------------------------------------------------------------
#  Logging setup
# -------------------------------------------------------------------------
echo "==========================================================="
echo " Running Multiomic GRN Pipeline "
echo " Dataset : ${DATASET}"
echo " Organism: ${ORGANISM}"
echo " Outdir  : ${OUTDIR}"
echo " Stages  : ${START_STAGE} → ${STOP_STAGE}"
echo " Force   : ${FORCE_FLAG}"
echo "==========================================================="

# -------------------------------------------------------------------------
#  Execute the pipeline
# -------------------------------------------------------------------------
poetry run python ./src/multiomic_transformer/pipeline/build_multiomic_data_pipeline.py \
    --dataset "${DATASET}" \
    --organism "${ORGANISM}" \
    --outdir "${OUTDIR}" \
    --start "${START_STAGE}" \
    --stop "${STOP_STAGE}" \
    --rna "${RNA_PATH}" \
    --atac "${ATAC_PATH}" \
    ${FORCE_FLAG}

# -------------------------------------------------------------------------
#  Completion message
# -------------------------------------------------------------------------
echo "==========================================================="
echo "Pipeline completed successfully at: $(date)"
echo "Output directory: ${OUTDIR}"
echo "==========================================================="
