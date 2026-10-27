#!/bin/bash -l
#SBATCH --job-name=generate_grns
#SBATCH --output=LOGS/generate_grns/generate_grns_%A_%a.log
#SBATCH --error=LOGS/generate_grns/generate_grns_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=dense
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --array=0-5

# Writes test_set_grn.csv and one <sample>_cross_grn.csv per holdout sample into
# the model run directory. GRNs that already exist are skipped.
# Submit from the repository root (which contains LOGS):
#   sbatch TETHER/scripts/run_generate_grns_from_model.sh [extra args]
# Extra arguments go to generate_grns_from_model.py, for example:
#   --checkpoint epoch-006.ckpt  --holdout_celltypes_only  --overwrite  --num_workers 8
set -eo pipefail

PROJECT_DIR=/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER

cd "$PROJECT_DIR"

conda activate my_env

# No liver, mESC, brain, skin, kidney, HSC. Entries can be glob patterns, because a run
# name ends in its start time. Retrained 2026-10-27 with TF-DNA model mm10_3928135_epoch_19
# (overlap-filtered negatives, merged positives, RC augmentation, cosine LR).
EXPERIMENT_LIST=(
    "celltype_joint_3928252_*"
    "celltype_joint_3928253_*"
    "celltype_joint_3928254_*"
    "celltype_joint_3928255_*"
    "celltype_joint_3928256_*"
    "celltype_joint_3928257_*"
)
# 2026-10-22 models (TF-DNA model mm10_3925194_epoch_21):
#   celltype_joint_3926071_* ... celltype_joint_3926076_* (same order)
# 2026-10-02 models (TF-DNA model mm10_3831017_epoch_12):
#   celltype_joint_3899206_20261002_125300_917425
#   celltype_joint_3899207_20261002_125401_578199
#   celltype_joint_3899208_20261002_125401_578203
#   celltype_joint_3899209_20261002_125401_578193
#   celltype_joint_3899210_20261002_125501_478329
#   celltype_joint_3899211_20261002_143015_907548

# ==========================================
#        EXPERIMENT SELECTION
# ==========================================
# Get the current experiment based on SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ ${TASK_ID} -ge ${#EXPERIMENT_LIST[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${TASK_ID}) exceeds number of experiments (${#EXPERIMENT_LIST[@]})"
    exit 1
fi

RUN_DIR="${EXPERIMENT_LIST[$TASK_ID]}"
MATCHES=( "${PROJECT_DIR}"/checkpoints/celltype_tf_tg/${RUN_DIR} )
if [ ${#MATCHES[@]} -ne 1 ] || [ ! -d "${MATCHES[0]}" ]; then
    echo "ERROR: ${RUN_DIR} matches ${#MATCHES[@]} entries, expected one run directory: ${MATCHES[*]}"
    exit 1
fi
RUN_DIR=$(basename "${MATCHES[0]}")

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Run: ${RUN_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python "${PROJECT_DIR}/scripts/generate_grns_from_model.py" \
    --run_dir "${RUN_DIR}" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"
