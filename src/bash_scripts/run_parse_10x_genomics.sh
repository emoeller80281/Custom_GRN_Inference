#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 1
#SBATCH --mem 256G
#SBATCH -o LOGS/parse_10x_genomics.log
#SBATCH -e LOGS/parse_10x_genomics.err
#srun source /gpfs/Home/esm5360/miniconda3/envs/my_env

echo "Running 'parse_10x_genomics.py'"
BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")

/usr/bin/time -v python3 "${BASE_DIR}/src/testing_scripts/parse_10x_genomics.py" \