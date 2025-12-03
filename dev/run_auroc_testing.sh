#!/bin/bash -l
#SBATCH --job-name=auroc_testing
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=64G

set -euo pipefail

cd "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
source .venv/bin/activate

SELECTED_EXPERIMENT_DIR=(
    # "model_training_192_1k_metacells"
    # "model_training_192_5k_metacells"
    "model_training_192_10k_metacells"
    )

poetry run python ./dev/auroc_testing.py \
    --experiment_dir_list "${SELECTED_EXPERIMENT_DIR[@]}"

echo "finished"