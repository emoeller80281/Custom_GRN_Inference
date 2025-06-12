#!/bin/bash -l
#SBATCH --partition=memory
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem=124G
#SBATCH -o /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/chipseq_sliding_window.log
#SBATCH -e /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/chipseq_sliding_window.err

set -euo pipefail

source activate my_env

cd "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"

poetry run python "src/grn_inference/pipeline/sliding_window_tf_peak_motifs.py" \
    --tf_names_file "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/motif_information/mm10/TF_Information_all_motifs.txt" \
    --meme_dir "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/motif_information/mm10/mm10_motif_meme_files" \
    --reference_genome_dir '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/reference_genome/mm10' \
    --atac_data_file "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC/DS011_mESC_sample1/DS011_mESC_ATAC_processed.parquet" \
    --rna_data_file "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC/DS011_mESC_sample1/DS011_mESC_RNA_processed.parquet" \
    --output_dir "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC" \
    --species "mm10" \
    --num_cpu "32" \
    --fig_dir "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC"