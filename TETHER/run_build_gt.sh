#!/bin/bash -l
#SBATCH --job-name=build_celltype_gt
#SBATCH --output=LOGS/%x_%j.log
#SBATCH --error=LOGS/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64G
#
# Peaks -> per-slice TF-TG edges for one dataset's cell type map, then the provenance check.
# Everything is written under data/ground_truth_files/cell_type_specific/ (GT_DIR):
#   chipatlas/, remap/                  peak sets, shared by every dataset's map
#   mm10/<CELL_TYPE>/                   <slice>_ground_truth.parquet
#   reports/gt_breadth_mm10_<CELL_TYPE>.tsv
# CELL_TYPE is the dataset prefix of <CELL_TYPE>_cell_type_map.tsv in GT_DIR.

set -eo pipefail

CELL_TYPE="GSE209610_kidney_controls"

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
HERE="${PROJECT_DIR}/TETHER"
GT_DIR="${PROJECT_DIR}/data/ground_truth_files/cell_type_specific"
MAP="${GT_DIR}/${CELL_TYPE}_cell_type_map.tsv"

cd "$HERE"

source activate my_env

export PYTHONPATH="$HERE:${PYTHONPATH:-}"
# sort spills to TMPDIR; node-local /tmp is too small for multi-million-peak BEDs.
export TMPDIR="$HERE/.sorttmp"

mkdir -p "$TMPDIR"

python3 -u "$HERE/download_ground_truth.py" --organism mm10 \
    --map "$MAP" \
    --out_dir "$GT_DIR"

python3 -u "$HERE/scripts/build_celltype_ground_truth.py" --organism mm10 \
    --map "$MAP" \
    --gt_dir "$GT_DIR" \
    --out_dir "$GT_DIR" \
    --dataset "$CELL_TYPE" \
    --reports "${GT_DIR}/reports" \
    --report_tag "mm10_${CELL_TYPE}" \
    --max_tss_dist 100000

python3 -u "$HERE/verify_ground_truth.py" --organism mm10 \
    --map "$MAP" \
    --gt_dir "${GT_DIR}/mm10/${CELL_TYPE}"
