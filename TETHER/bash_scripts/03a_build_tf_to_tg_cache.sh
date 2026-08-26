#!/bin/bash -l
#SBATCH --job-name=build_tf_tg_cache
#SBATCH --output=LOGS/build_tf_tg_cache/%x_%A_%a.log
#SBATCH --error=LOGS/build_tf_tg_cache/%x_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH --array=0-8%9

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate my_env

# ==========================================
#        DATASET SELECTION
# ==========================================
# One entry per array task, as "species|cell_type|sample_name". Keep #SBATCH --array
# above in sync with the length of this list (a task past the end exits with an error
# rather than silently running the wrong dataset).
#
# These are exported as TETHER_* and picked up by config.py, so config.py itself never
# needs editing to run a batch -- every array task reads the same file but resolves its
# own dataset from its own environment.
EXPERIMENT_LIST=(
    "mm10|mESC|E7.5_rep1"
    "mm10|mESC|E7.5_rep2"
    "mm10|mESC|E8.5_rep1"
    "mm10|mESC|E8.5_rep2"
    "mm10|mouse_hepatocytes|hepatocytes_1"
    "mm10|mouse_hepatocytes|hepatocytes_3"
    "hg38|Macrophage|buffer_1"
    "hg38|Macrophage|buffer_2"
    "hg38|K562|sample_1"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ ${TASK_ID} -ge ${#EXPERIMENT_LIST[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${TASK_ID}) exceeds number of experiments (${#EXPERIMENT_LIST[@]})"
    exit 1
fi

EXPERIMENT_CONFIG="${EXPERIMENT_LIST[$TASK_ID]}"

# Parse experiment configuration
IFS='|' read -r species cell_type sample_name <<< "$EXPERIMENT_CONFIG"

# config.py reads these three from the environment. species is exported explicitly so an
# inconsistent species/cell_type pair in the list above trips config.py's assert rather
# than being silently corrected.
export species cell_type sample_name

echo "[INFO] Array task ${TASK_ID}: ${species} / ${cell_type} / ${sample_name}"
python3 -c "import config; print('[INFO] config.py resolved:', config.describe_dataset())"

max_cells_per_pair=64
max_peaks_per_tg=25
peak_flank_size=128
pct_true_edges=0.3
true_false_ratio=5.0

echo "[INFO] Building and Caching Training Data..."
python3 ${PROJECT_DIR}/scripts/build_tf_to_tg_train_data.py \
    --max_cells_per_pair $max_cells_per_pair \
    --max_peaks_per_tg $max_peaks_per_tg \
    --pct_true_edges $pct_true_edges \
    --true_false_ratio $true_false_ratio \
    --peak_flank_size $peak_flank_size \
    --num_cpu $SLURM_CPUS_PER_TASK \
    --force_reload
