#!/bin/bash -l
#SBATCH --job-name=linger_pseudobulk
#SBATCH --output=LOGS/transformer_logs/01_pseudobulk/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/01_pseudobulk/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 10
#SBATCH --mem=160G

set -euo pipefail

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER

source activate my_env

python src/multiomic_transformer/data/linger_pseudobulk.py