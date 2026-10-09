#!/bin/bash -l
#SBATCH --job-name=stability_subsample_grns
#SBATCH --output=LOGS/stability_subsample_grns/%x_%A_%a.log
#SBATCH --error=LOGS/stability_subsample_grns/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=dense
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --array=0-5

# Predicts the GRN of each stability subsample with the model that held out its tissue.
# GRNs that already exist are skipped.
# Submit from the repository root (which contains LOGS):
#   sbatch TETHER/stability_analysis/run_generate_stability_subsample_grns.sh [extra args]
# Extra arguments go to generate_stability_subsample_grns.py, for example:
#   --checkpoint epoch-009.ckpt  --overwrite
set -eo pipefail

PROJECT_DIR=/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER

conda activate my_env

TISSUE_LIST=(Liver Embryo Brain Skin Kidney HSC)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [ ${TASK_ID} -ge ${#TISSUE_LIST[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${TASK_ID}) exceeds number of tissues (${#TISSUE_LIST[@]})"
    exit 1
fi
TISSUE="${TISSUE_LIST[$TASK_ID]}"

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Tissue: ${TISSUE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python "${PROJECT_DIR}/stability_analysis/generate_stability_subsample_grns.py" \
    --tissue "${TISSUE}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"
