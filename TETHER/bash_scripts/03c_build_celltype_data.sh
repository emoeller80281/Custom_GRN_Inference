#!/bin/bash -l
#SBATCH --job-name=celltype_data
#SBATCH --output=LOGS/celltype_data_%A_%a.log
#SBATCH --error=LOGS/celltype_data_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=dense
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --array=0-16%8

# Builds one sample cache (prepared edges + TF-DNA binding scores) per array task.
# Submit from the repository root (which contains LOGS):
#   sbatch TETHER/bash_scripts/03c_build_celltype_data.sh
# Rebuild only some samples by index, for example: sbatch --array=3,7 ...
# Keep --array in sync with the length of DATASETS.
#
# The preprocessing settings below select the cache. They must match the ones in
# 03d_train_cached_celltype_tf_to_tg_model.sh, or training cannot find the caches.
set -eo pipefail
PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd "$PROJECT_DIR"
source activate my_env
set -u
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DATASETS=(
    "mESC:E7.5_rep1"
    "mESC:E7.5_rep2"
    "mESC:E8.0_rep1"
    "mESC:E8.0_rep2"
    "mESC:E8.5_rep1"
    "mESC:E8.5_rep2"
    "kidney:Ctrl_4weeks_1"
    "kidney:Ctrl_4weeks_2"
    "kidney:Ctrl_6months_1"
    "GSE140203_shareseq:skin_late_anagen"
    "GSE140203_shareseq:brain"
    "10x_E18_mouse_brain:brain"
    "GSE246464_HSC:young_rep1"
    "GSE246464_HSC:young_rep2"
    "GSE246464_HSC:old_rep1"
    "GSE246464_HSC:old_rep2"
    "mouse_liver:liver_sample"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [ "$TASK_ID" -ge "${#DATASETS[@]}" ]; then
    echo "ERROR: array task ${TASK_ID} exceeds the ${#DATASETS[@]} configured datasets" >&2
    exit 1
fi
DATASET="${DATASETS[$TASK_ID]}"
echo "[INFO] Array task ${TASK_ID}: ${DATASET} on $(hostname)"

srun python -u scripts/build_tf_to_tg_celltype_data.py \
    --species mm10 \
    --dataset "$DATASET" \
    --max_cells_per_pair 25 \
    --max_peaks_per_tg 25 \
    --binding_chunk_size 512 \
    --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
    "$@"
