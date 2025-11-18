#!/bin/bash -l
#SBATCH --job-name=grad_attrib
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%A_%a.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64G

set -euo pipefail

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32  # optional but ok
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export BLIS_NUM_THREADS=8
export KMP_AFFINITY=granularity=fine,compact,1,0


# ------------------------------------------------------------
# Run classifier for this chromosome
# ------------------------------------------------------------
SELECTED_EXPERIMENT_DIR=/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC_no_scale_linear/no_classifier_head
poetry run python ./dev/grad_attrib.py \
    --selected_experiment_dir "$SELECTED_EXPERIMENT_DIR"


echo "finished successfully!"