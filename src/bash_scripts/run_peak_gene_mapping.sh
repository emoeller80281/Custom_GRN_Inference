#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem-per-cpu=16G
#SBATCH -o LOGS/peak_gene_mapping.log
#SBATCH -e LOGS/peak_gene_mapping.err
#srun source /gpfs/Home/esm5360/miniconda3/envs/my_env

/usr/bin/time -v python3 /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/src/testing_scripts/peak_gene_mapping_bedtools.py