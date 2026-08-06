#!/bin/bash -l
#SBATCH --job-name=generate_all_predictions
#SBATCH --output=LOGS/full_grn/generate_all_predictions_%A/%x_%A_%a.log
#SBATCH --error=LOGS/full_grn/generate_all_predictions_%A/%x_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --array=1-6%7

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate my_env

EXPERIMENT_LIST=(
    "mm10|mESC|E7.5_rep1|mouse_hepatocytes|hepatocytes_1"
    "mm10|mESC|E8.5_rep1|mouse_hepatocytes|hepatocytes_1"
    "hg38|Macrophage|buffer_1|K562|sample_1"
    "hg38|Macrophage|buffer_2|K562|sample_1"
    "hg38|K562|sample_1|Macrophage|buffer_1"
    "mm10|mouse_hepatocytes|hepatocytes_1|mESC|E7.5_rep1"
    "mm10|mouse_hepatocytes|hepatocytes_3|mESC|E7.5_rep1"
)

# --- Memory + math ---
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 was inherited from the DDP training
# script and removed here. It stops the caching allocator splitting blocks over 32 MB,
# which with ~1 GB per-batch tensors left it unable to reuse large free blocks: a run
# on Macrophage/buffer_1 reserved 31,650 MB while only 4,455 MB was live.
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

# --- Threading ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

# No NCCL / rendezvous setup here on purpose. generate_all_predictions.py is a plain
# single-process, single-GPU inference script -- it never calls init_process_group and
# never reads RANK/LOCAL_RANK. The DDP preamble copied from the training scripts brought
# two problems with it: the 10.90.29.* interface check would hard-exit an otherwise
# healthy job, and launching under `torchrun --standalone` pins the c10d rendezvous to
# localhost:29400, so any two of the seven concurrent array tasks landing on the same
# node would collide with "Address already in use".

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist        = " $SLURM_JOB_NODELIST
echo "Host            = " $(hostname)
echo "GPUs visible    = " $(nvidia-smi -L | wc -l)
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo ""

export PYTHONFAULTHANDLER=1

# ==========================================
#        EXPERIMENT SELECTION
# ==========================================
# Get the current experiment based on SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ ${TASK_ID} -ge ${#EXPERIMENT_LIST[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${TASK_ID}) exceeds number of experiments (${#EXPERIMENT_LIST[@]})"
    exit 1
fi

EXPERIMENT_CONFIG="${EXPERIMENT_LIST[$TASK_ID]}"

# Parse experiment configuration
IFS='|' read -r species cell_type sample_name cross_model_cell_type cross_model_sample_name <<< "$EXPERIMENT_CONFIG"

echo "[INFO] Generating full test-set predictions for:"
echo "  species=$species"
echo "  cell_type=$cell_type"
echo "  sample_name=$sample_name"
echo "  cross_model_cell_type=$cross_model_cell_type"
echo "  cross_model_sample_name=$cross_model_sample_name"

echo "[INFO] Starting inference..."
python ${PROJECT_DIR}/generate_all_predictions.py \
    --species "$species" \
    --cell_type "$cell_type" \
    --sample_name "$sample_name" \
    --cross_model_cell_type "$cross_model_cell_type" \
    --cross_model_sample_name "$cross_model_sample_name" \
    --max_peaks_per_tg 8 \
    --max_cells_per_pair 25 \
    --batch_size 512 \
    --tf_peak_chunk_size 256 \
    --force_reload
