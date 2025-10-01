#!/bin/bash -l
#SBATCH --job-name=moods_tf_to_peak
#SBATCH --output=LOGS/transformer_logs/motif_mask/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/motif_mask/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=160G

set -euo pipefail

source activate my_env

MM10_MOTIF_MEME_FILE_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/motif_information/mm10/mm10_motif_meme_files"
PEAKS_BED_FILE="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/transformer_testing_output/peak_tmp.bed"
CHR19_MM10_FASTA_FILE="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/reference_genome/mm10/chr19.fa"
JASPAR_PFM_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/motif_information/JASPAR/pfm_files"
JASPAR_MEME_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/motif_information/JASPAR/meme_files"

MOODS_OUT_PATH="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/transformer/mESC/chr19/chr19_moods_sites.tsv"

python /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/transformer/transformer_data/build_motif_mask/moods_scan.py \
  --peaks $PEAKS_BED_FILE \
  --fasta $CHR19_MM10_FASTA_FILE \
  --motifs $JASPAR_PFM_DIR/*.pfm \
  --out $MOODS_OUT_PATH \
  --threshold 6.0