#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 2
#SBATCH --mem-per-cpu=32G
#SBATCH -o LOGS/combine_dataframe.log
#SBATCH -e LOGS/combine_dataframes.err
#srun source /gpfs/Home/esm5360/miniconda3/envs/my_env

/usr/bin/time -v python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/src/testing_scripts/combine_dataframes.py