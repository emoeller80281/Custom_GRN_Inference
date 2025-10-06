#!/bin/bash
#SBATCH --job-name=peak_calling
#SBATCH --output=LOGS/transformer_logs/00_raw_data_processing/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/00_raw_data_processing/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=128G

set -euo pipefail

RAW_DATA_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/raw/GSE218576"
PROCESSED_DATA_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/processed/GSE218576"
CHROM_SIZE_FILE="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/genome_data/reference_genome/mm10/mm10.chrom.sizes"

sample_fragment_file="B6E12-5-1_atac_fragments.tsv.gz"

# Ensure output directory exists
mkdir -p "$PROCESSED_DATA_DIR"

# -----------------------
# Step 1. Convert fragments to BED
# -----------------------
echo "[INFO] Converting fragments file to BED..."
zcat "${RAW_DATA_DIR}/${sample_fragment_file}" \
    | awk '{print $1"\t"$2"\t"$3"\t"$4"\t"$5}' \
    > "${PROCESSED_DATA_DIR}/atac_with_bc.bed"

# -----------------------
# Step 2. Sort BED file
# -----------------------
echo "[INFO] Sorting BED file..."
sort -k1,1 -k2,2n "${PROCESSED_DATA_DIR}/atac_with_bc.bed" -o "${PROCESSED_DATA_DIR}/atac_with_bc.bed"

# -----------------------
# Step 3. Call peaks with MACS2
# -----------------------
echo "[INFO] Calling peaks with MACS2..."
macs2 callpeak \
    -t "${PROCESSED_DATA_DIR}/atac_with_bc.bed" \
    -f BED \
    -g mm \
    --nomodel \
    --shift -100 \
    --extsize 200 \
    -n atac_peaks \
    --outdir "$PROCESSED_DATA_DIR"

PEAK_FILE="${PROCESSED_DATA_DIR}/atac_peaks_peaks.narrowPeak"

# -----------------------
# Step 4. Build peak × cell matrix
# -----------------------
echo "[INFO] Building peak × cell matrix..."
PEAK_BED="${PROCESSED_DATA_DIR}/atac_peaks.bed"
cut -f1-3 "$PEAK_FILE" > "$PEAK_BED"

# Intersect fragments with peaks to assign counts per cell barcode
# Each line: peak region + cell barcode
bedtools intersect -a "$PEAK_BED" -b "${PROCESSED_DATA_DIR}/atac_with_bc.bed" -wa -wb \
    | awk '{print $1":"$2"-"$3"\t"$7}' \
    > "${PROCESSED_DATA_DIR}/peak_cell_hits.tsv"