#!/bin/bash
#SBATCH --job-name=aggregate_features
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=128G

set -euo pipefail

source activate my_env

python /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/agg_data.py
