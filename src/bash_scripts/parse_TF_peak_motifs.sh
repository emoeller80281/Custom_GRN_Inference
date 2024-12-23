#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem-per-cpu=4G
#SBATCH -o LOGS/parse_TF_peak_motifs.log
#SBATCH -e LOGS/parse_TF_peak_motifs.err
#srun source /gpfs/Home/esm5360/miniconda3/envs/my_env

source /gpfs/Home/esm5360/miniconda3/bin/activate my_env

INPUT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/homer_tf_motif_scores"
OUTPUT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/total_motif_regulatory_scores.tsv"

python3 src/python_scripts/parse_TF_peak_motifs.py \
    --input_dir "${INPUT_DIR}" \
    --output_file "${OUTPUT_DIR}" \
    --cpu_count 32