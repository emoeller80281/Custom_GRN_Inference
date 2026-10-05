#!/bin/bash -l
#SBATCH --job-name=generate_grns
#SBATCH --output=LOGS/generate_grns_%j.log
#SBATCH --error=LOGS/generate_grns_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=dense
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G

# Writes test_set_grn.csv and one <sample>_cross_grn.csv per holdout sample into
# the model run directory. GRNs that already exist are skipped.
# Submit from the repository root (which contains LOGS):
#   sbatch TETHER/scripts/run_generate_grns_from_model.sh <run name or run dir> [extra args]
# Extra arguments go to generate_grns_from_model.py, for example:
#   --checkpoint epoch-006.ckpt  --holdout_celltypes_only  --overwrite  --num_workers 8
set -eo pipefail

RUN_DIR="celltype_joint_3899211_20261002_143015_907548"

PROJECT_DIR=/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER

conda activate my_env

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Run: ${RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python "${PROJECT_DIR}/scripts/generate_grns_from_model.py" \
    --run_dir "${RUN_DIR}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"
