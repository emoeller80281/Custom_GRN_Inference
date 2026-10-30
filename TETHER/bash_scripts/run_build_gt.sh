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
#   chipatlas/<ORGANISM>/, remap/<ORGANISM>/  peak sets, shared by the organism's dataset maps
#   <ORGANISM>/<CELL_TYPE>/             <slice>_ground_truth.parquet
#   reports/gt_breadth_<ORGANISM>_<CELL_TYPE>.tsv
# CELL_TYPE is the dataset prefix of <CELL_TYPE>_cell_type_map.tsv in GT_DIR.

set -eo pipefail

ORGANISM="mm10"
CELL_TYPE="GSE246464_HSC"

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
GT_DIR="${PROJECT_DIR}/data/ground_truth_files/cell_type_specific"
MAP="${GT_DIR}/${CELL_TYPE}_cell_type_map.tsv"

cd "$PROJECT_DIR/TETHER"

source activate my_env

export PYTHONPATH="$PROJECT_DIR/TETHER:${PYTHONPATH:-}"
# sort spills to TMPDIR; node-local /tmp is too small for multi-million-peak BEDs.
export TMPDIR="$PROJECT_DIR/.sorttmp"

mkdir -p "$TMPDIR"

python3 -u "$PROJECT_DIR/TETHER/scripts/generate_celltype_ground_truth/download_ground_truth.py" --organism "$ORGANISM" \
    --map "$MAP" \
    --out_dir "$GT_DIR"

python3 -u "$PROJECT_DIR/TETHER/scripts/generate_celltype_ground_truth/build_celltype_ground_truth.py" --organism "$ORGANISM" \
    --map "$MAP" \
    --gt_dir "$GT_DIR" \
    --out_dir "$GT_DIR" \
    --dataset "$CELL_TYPE" \
    --reports "${GT_DIR}/reports" \
    --report_tag "${ORGANISM}_${CELL_TYPE}" \
    --max_tss_dist 100000

python3 -u "$PROJECT_DIR/TETHER/scripts/generate_celltype_ground_truth/verify_ground_truth.py" --organism "$ORGANISM" \
    --map "$MAP" \
    --gt_dir "${GT_DIR}/${ORGANISM}/${CELL_TYPE}"
