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
# Peaks -> per-slice TF-TG edges for the mESC slices, then the provenance check.
# Run after: python3 download_ground_truth.py --organism mm10 --map cell_type_map_mesc.tsv

set -eo pipefail

CELL_TYPE="mESC"

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/"
HERE="${PROJECT_DIR}/TETHER"
GT_DIR="${PROJECT_DIR}/data/ground_truth_files/cell_type_specific"

cd "$HERE"

source activate my_env

export PYTHONPATH="$HERE:${PYTHONPATH:-}"
# sort spills to TMPDIR; node-local /tmp is too small for multi-million-peak BEDs.
export TMPDIR="$HERE/.sorttmp"

mkdir -p "$TMPDIR"

python3 -u "$HERE/scripts/build_celltype_ground_truth.py" --organism mm10 \
    --map "$HERE/${CELL_TYPE}_cell_type_map.tsv" \
    --gt_dir "$GT_DIR" \
    --out_dir "$GT_DIR" \
    --reports "${GT_DIR}/reports" \
    --report_tag mm10_${CELL_TYPE} \
    --max_tss_dist 100000 \

python3 -u "$HERE/scripts/verify_ground_truth.py" --organism mm10 \
    --map "$HERE/cell_type_map_mesc.tsv" \
    --gt_dir "$HERE/ground_truth/mesc/mm10"
