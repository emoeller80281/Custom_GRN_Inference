#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem 64G
#SBATCH -o LOGS/combine_sample_score_datasets.log
#SBATCH -e LOGS/combine_sample_score_datasets.log
#srun source /gpfs/Home/esm5360/miniconda3/envs/my_env

BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")

/usr/bin/time -v python3 "${BASE_DIR}/src/testing_scripts/combine_sample_score_datasets.py" \
