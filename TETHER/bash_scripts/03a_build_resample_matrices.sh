#!/bin/bash -l
#SBATCH --job-name=build_resample_mats
#SBATCH --output=LOGS/build_tf_tg_cache/%x_%j.log
#SBATCH --error=LOGS/build_tf_tg_cache/%x_%j.err
#SBATCH --time=8:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=192G

# Backfill atac_mat.pt / rna_mat.pt onto an existing TF-TG cache so training can run with
# --resample_cells_per_epoch. Builds nothing else -- the edge bags, one-hot peaks, ground
# truth and splits already in the cache are reused untouched.
#
# Must be given the SAME split/peak settings as the cache it is backfilling, or the
# lookup tables it derives them from won't line up.

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR

mkdir -p LOGS/build_tf_tg_cache

source activate my_env

python3 ${PROJECT_DIR}/scripts/build_tf_to_tg_train_data.py \
    --build_resample_matrices_only \
    --split_mode tf \
    --val_tf_frac 0.25 \
    --max_cells_per_pair 24 \
    --max_peaks_per_tg 25 \
    --pct_true_edges 0.3 \
    --true_false_ratio 10.0 \
    --peak_flank_size 128 \
    --num_cpu $SLURM_CPUS_PER_TASK
