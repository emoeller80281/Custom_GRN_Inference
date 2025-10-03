#!/bin/bash -l
#SBATCH --job-name=prepare_transformer_data
#SBATCH --output=LOGS/transformer_logs/prepare_transformer_data/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/prepare_transformer_data/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 50
#SBATCH --mem=160G

set -euo pipefail

source activate my_env

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER

export NUM_CPUS=${SLURM_CPUS_PER_TASK:-1}

python -m dev.transformer.prepare_transformer_data