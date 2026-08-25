#!/bin/bash -l
#SBATCH --job-name=build_tf_dna_cache
#SBATCH --output=LOGS/build_tf_dna_cache/%x_%j.log
#SBATCH --error=LOGS/build_tf_dna_cache/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=256G

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate my_env

echo "[INFO] Building TF-to-DNA datasets..."
# --pct_true_edges: 1.0 is no longer viable. The ChIP-Atlas parquet holds 268M unique
# (TF, peak) pairs and nearly every peak appears in exactly one edge, so the peak one-hot
# scales with the edge count -- at 1.0 with ratio 5 that is 1.6B edges and a 271 GB one-hot.
# 0.01 gives 2.7M true / 13.4M false = 16.1M edges, ~16.5 GB: about the size of the cache
# this replaces, and 4 DDP ranks of it fit 02b's --mem=128G.
#
# Keep every flag on its own line with no comments between them. A comment line inside a
# backslash continuation ends the command: the remaining flags become part of the comment
# and the script silently runs on argparse defaults.
python3 ${PROJECT_DIR}/scripts/build_tf_to_dna_train_data.py \
    --pct_true_edges 0.01 \
    --true_false_ratio 5.00 \
    --peak_flank_size 128 \
    --force_reload
