#!/bin/bash -l
#SBATCH --partition compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task 1
#SBATCH --mem 126G
#SBATCH -o LOGS/compare_score_dist.log
#SBATCH -e LOGS/compare_score_dist.err

conda activate my_env

python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/src/testing_scripts/compare_score_distributions.py
