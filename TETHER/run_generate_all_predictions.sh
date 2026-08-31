#!/bin/bash -l
#SBATCH --job-name=generate_all_predictions
#SBATCH --output=LOGS/full_grn/generate_all_predictions_%A/%x_%A_%a.log
#SBATCH --error=LOGS/full_grn/generate_all_predictions_%A/%x_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=192G
#SBATCH --array=0-7

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate my_env

EXPERIMENT_LIST=(
    # "mm10|mESC|E8.5_CRISPR_T_KO|mouse_hepatocytes|hepatocytes_1"
    # "mm10|mESC|E8.5_CRISPR_T_WT|mouse_hepatocytes|hepatocytes_1"
    "mm10|mESC|E7.5_rep1|mouse_hepatocytes|hepatocytes_1"
    "mm10|mESC|E7.5_rep2|mouse_hepatocytes|hepatocytes_1"
    "mm10|mESC|E8.5_rep1|mouse_hepatocytes|hepatocytes_1"
    "hg38|Macrophage|buffer_1|K562|sample_1"
    "hg38|Macrophage|buffer_2|K562|sample_1"
    "hg38|K562|sample_1|Macrophage|buffer_1"
    "mm10|mouse_hepatocytes|hepatocytes_1|mESC|E7.5_rep1"
    "mm10|mouse_hepatocytes|hepatocytes_3|mESC|E7.5_rep1"


)


MAX_PEAKS_PER_TG=25
MAX_CELLS_PER_PAIR=100

export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Threading ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

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

echo "[INFO] TF embedding tables      : $TF_DNA_CACHE"
echo "[INFO] Bag geometry             : ${MAX_PEAKS_PER_TG} peaks x ${MAX_CELLS_PER_PAIR} cells"

FULL_GRN_DIR="${PROJECT_DIR}/new_testing_results/full_test_grns"
SAMPLE_FULL_GRN_FILE="${FULL_GRN_DIR}/${sample_name}_model_vs_${sample_name}_full_grn.tsv"
CROSS_TF_TG_FILE="${FULL_GRN_DIR}/${cross_model_sample_name}_model_vs_${sample_name}_full_grn.tsv"
if [ -f "$SAMPLE_FULL_GRN_FILE" ] && [ -f "$CROSS_TF_TG_FILE" ]; then
    echo "[INFO] $SAMPLE_FULL_GRN_FILE and $CROSS_TF_TG_FILE already exist -- full GRN already completed. Skipping."
    echo "[INFO] Delete the relevant file (or add --force_reload below) to force a rerun."
    exit 0
fi

# NOTE: keep comments ABOVE this command. A comment line between backslash-continued
# argument lines silently comments out every remaining flag -- `bash -n` still passes,
# and the run proceeds on argparse defaults.
echo "[INFO] Starting inference..."
python ${PROJECT_DIR}/generate_all_predictions.py \
    --species "$species" \
    --cell_type "$cell_type" \
    --sample_name "$sample_name" \
    --cross_model_cell_type "$cross_model_cell_type" \
    --cross_model_sample_name "$cross_model_sample_name" \
    --tf_dna_cache_dir "$TF_DNA_CACHE" \
    --max_peaks_per_tg $MAX_PEAKS_PER_TG \
    --max_cells_per_pair $MAX_CELLS_PER_PAIR \
    --batch_size 256 \
    --tf_peak_chunk_size 1024 \
    --no_compile \
    --all_chromosomes
    
    # --force_reload
