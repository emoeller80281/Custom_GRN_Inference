#!/bin/bash -l
#SBATCH --job-name=GAT_data_preprocessing
#SBATCH --output=LOGS/transformer_logs/02_prepare_transformer_data/%x_%A.log
#SBATCH --error=LOGS/transformer_logs/02_prepare_transformer_data/%x_%A.err
#SBATCH --time=4:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH --mem=256G

set -euo pipefail

module load bedtools
source activate /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/.venv/bin/activate

poetry run python ./src/multiomic_transformer/data/preprocess.py --num_cpu 64