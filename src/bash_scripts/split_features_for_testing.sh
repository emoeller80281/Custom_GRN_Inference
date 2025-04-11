#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 2
#SBATCH --mem-per-cpu=32G
#SBATCH -o LOGS/feature_testing.log
#SBATCH -e LOGS/feature_testing.log
#srun source /gpfs/Home/esm5360/miniconda3/envs/my_env

BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
INFERRED_GRN_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/K562/K562_human_filtered/inferred_grns"

/usr/bin/time -v python3 "${BASE_DIR}/src/testing_scripts/split_features_for_testing.py" \
    --inferred_grn_dir "${INFERRED_GRN_DIR}"