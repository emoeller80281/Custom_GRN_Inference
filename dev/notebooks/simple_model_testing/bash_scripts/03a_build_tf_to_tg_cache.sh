#!/bin/bash -l
#SBATCH --job-name=build_tf_tg_cache
#SBATCH --output=LOGS/build_tf_tg_cache/%x_%j.log
#SBATCH --error=LOGS/build_tf_tg_cache/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=256G

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate my_env

max_cells_per_pair=32
# max_peaks_per_tg=4
peak_flank_size=128
pct_true_edges=0.15
true_false_ratio=2.0

echo "[INFO] Building and Caching Training Data..."
python3 ${PROJECT_DIR}/scripts/build_tf_to_tg_train_data.py \
    --max_cells_per_pair $max_cells_per_pair \
    --pct_true_edges $pct_true_edges \
    --true_false_ratio $true_false_ratio \
    --peak_flank_size $peak_flank_size \
    --num_cpu $SLURM_CPUS_PER_TASK \
    --force_reload

# --max_peaks_per_tg $max_peaks_per_tg \