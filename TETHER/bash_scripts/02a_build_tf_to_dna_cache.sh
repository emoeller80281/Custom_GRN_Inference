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

# Extra arguments to sbatch are passed on. --cache_dir is required, and its path must contain the
# species. The species is mm10 unless the job sets it (hg38 needs more memory than this default):
#   sbatch 02a_build_tf_to_dna_cache.sh --site_merge_distance 64 --false_overlap_flank 128 --cache_dir <dir>
#   sbatch -p memory --mem=600G --export=ALL,species=hg38 02a_build_tf_to_dna_cache.sh --cache_dir <dir>
python3 ${PROJECT_DIR}/scripts/build_tf_to_dna_train_data.py \
    --min_true_edges_per_tf 500 \
    --max_true_edges_per_tf 20000 \
    --true_false_ratio 1.0 \
    --false_edge_sampling by_source_tf \
    --max_tf_length 2048 \
    --peak_flank_size 128 \
    --force_reload \
    "$@"
