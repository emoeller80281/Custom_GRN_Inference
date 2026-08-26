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
#SBATCH --mem=192G
#SBATCH --array=0-1%2

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate my_env

EXPERIMENT_LIST=(
    "mm10|mESC|E8.5_CRISPR_T_KO|mouse_hepatocytes|hepatocytes_1"
    "mm10|mESC|E8.5_CRISPR_T_WT|mouse_hepatocytes|hepatocytes_1"
    # "mm10|mESC|E7.5_rep1|mouse_hepatocytes|hepatocytes_1"
    # "mm10|mESC|E7.5_rep2|mouse_hepatocytes|hepatocytes_1"
    # "mm10|mESC|E8.5_rep1|mouse_hepatocytes|hepatocytes_1"
    # "hg38|Macrophage|buffer_1|K562|sample_1"
    # "hg38|Macrophage|buffer_2|K562|sample_1"
    # "hg38|K562|sample_1|Macrophage|buffer_1"
    # "mm10|mouse_hepatocytes|hepatocytes_1|mESC|E7.5_rep1"
    # "mm10|mouse_hepatocytes|hepatocytes_3|mESC|E7.5_rep1"


)

# ==========================================
#   PAIRING FOR THE hepatocytes_1 MODEL
# ==========================================
# A TF-TG checkpoint is only meaningful alongside the frozen TF-DNA model it was trained
# against AND the TF embedding table that model was trained on. All three come from the
# hepatocytes_1 run (job 3709466), whose wandb config.yaml records the TF-DNA path.
#
# config.py currently points mm10 at tf_dna_mm10_3831017, a *different, later* model, so
# the default would silently mispair them.
TF_DNA_CKPT="checkpoints/tf_dna_mm10_3697823/epoch=07-val_auroc=0.9743-val_loss=0.1661.ckpt"

# hepatocytes_1 predates the 2026-08-24 embedding fix, so it expects the per-TF
# random-basis embeddings. The current cache holds the shared-PCA ones; the shapes are
# identical (443, 5588, 128), so a mismatch produces meaningless scores without erroring.
# cached_data_old is NOT a fallback -- its copy was rebuilt and matches the new one.
TF_DNA_CACHE="cached_data/mm10/tf_dna_cache_prefix_embeddings"

# Bag geometry from the training run itself, not the script defaults:
#   wandb argv       --max_cells_per_pair 24, --max_peaks_per_tg not passed (uncapped)
#   training cache   max_peaks_real = 59, max_cells_per_pair = 24
#
# 59 (the widest bag the model ever saw) was tried first and OOMed both array tasks of
# job 3855147 after ~45 min, in exactly the inductor-autotuning clone described below:
# 10.91 GiB requested on top of 28.80 GiB resident, on a 31.73 GiB V100. Peak count sets
# the widest shape bucket, so it is the effective lever; 25 cuts that dimension ~2.4x.
#
# Cost of 25 vs 59, measured on these samples' peak_to_gene_dist tables:
#   KO  429 of 50,093 TGs truncated (0.9%),   4,642 peak-links dropped
#   WT  506 of 49,617 TGs truncated (1.0%),   5,688 peak-links dropped
# Peaks are ordered by |TSS_dist|, so what is dropped is each TG's most distant peaks.
MAX_PEAKS_PER_TG=25
MAX_CELLS_PER_PAIR=24

export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

# 3811170_0 hit a CUDA OOM inside torch._inductor's autotuning benchmark clone
# (needed ~11GiB on top of 28.39GiB already resident) when --all_chromosomes hit
# a shape bucket wider than usual. expandable_segments reduces the fragmentation
# that left only 264MiB free despite 2.71GiB being reserved-but-unallocated.
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

echo "[INFO] Frozen TF-DNA checkpoint : $TF_DNA_CKPT"
echo "[INFO] TF embedding tables      : $TF_DNA_CACHE"
echo "[INFO] Bag geometry             : ${MAX_PEAKS_PER_TG} peaks x ${MAX_CELLS_PER_PAIR} cells"

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
    --tf_dna_checkpoint "$TF_DNA_CKPT" \
    --tf_dna_cache_dir "$TF_DNA_CACHE" \
    --skip_own_model \
    --max_peaks_per_tg $MAX_PEAKS_PER_TG \
    --max_cells_per_pair $MAX_CELLS_PER_PAIR \
    --batch_size 128 \
    --tf_peak_chunk_size 1024 \
    --force_reload \
    --all_chromosomes
