#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem 64G
#SBATCH -o LOGS/combine_sample_score_datasets.log
#SBATCH -e LOGS/combine_sample_score_datasets.log
#srun source /gpfs/Home/esm5360/miniconda3/envs/my_env

BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")

FEATURE_SET_FILENAME="inferred_network_enrich_feat_w_string.parquet"
OUTPUT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output"
COMBIND_DATAFRAME_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/combined_inferred_dfs"

/usr/bin/time -v python3 "${BASE_DIR}/src/testing_scripts/combine_sample_score_datasets.py" \
    --feature_set_filename "${FEATURE_SET_FILENAME}" \
    --sample_output_dir "${OUTPUT_DIR}" \
    --combined_dataframe_dir "${OUTPUT_DIR}/combined_inferred_dfs"