#!/bin/bash -l
#SBATCH --job-name=build_tf_dna_cache
#SBATCH --output=LOGS/build_tf_dna_cache/%x_%j.log
#SBATCH --error=LOGS/build_tf_dna_cache/%x_%j.err
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
    --pct_true_edges 1.0 \
    --true_false_ratio 10.00 \
    --force_reload
