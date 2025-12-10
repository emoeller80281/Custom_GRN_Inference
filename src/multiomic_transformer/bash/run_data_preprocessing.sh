#!/bin/bash -l
#SBATCH --job-name=data_preprocessing
#SBATCH --output=LOGS/transformer_logs/02_prepare_transformer_data/%x_%A.log
#SBATCH --error=LOGS/transformer_logs/02_prepare_transformer_data/%x_%A.err
#SBATCH --time=4:00:00
#SBATCH -p memory
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=512G

set -euo pipefail

module load bedtools
source .venv/bin/activate

poetry run python ./src/multiomic_transformer/data/preprocess.py --num_cpu 8