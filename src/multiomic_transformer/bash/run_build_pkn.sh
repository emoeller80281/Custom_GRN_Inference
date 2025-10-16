#!/bin/bash -l
#SBATCH --job-name=build_pkn
#SBATCH --output=LOGS/transformer_logs/00_raw_data_processing/%x_%A.log
#SBATCH --error=LOGS/transformer_logs/00_raw_data_processing/%x_%A.err
#SBATCH --time=16:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 21
#SBATCH --mem=128G

set -euo pipefail

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER

python ./src/multiomic_transformer/data/build_pkn.py