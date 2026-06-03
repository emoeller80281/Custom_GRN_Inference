#!/bin/bash -l
#SBATCH --job-name=tf_dna_model
#SBATCH --output=LOGS/tf_dna_model/%x_%j.log
#SBATCH --error=LOGS/tf_dna_model/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=256G

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate my_env

echo "[INFO] Building TF-to-DNA datasets..."
python3 ${PROJECT_DIR}/scripts/build_tf_to_dna_train_data.py \
    --pct_true_edges 0.05 \
    --true_false_ratio 0.25 \
    --force_reload
