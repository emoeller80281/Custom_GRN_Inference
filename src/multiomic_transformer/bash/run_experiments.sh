#!/bin/bash
#SBATCH --job-name=grn_experiments
#SBATCH --output=LOGS/transformer_logs/experiments/%x_%A/%x_%A_%a.log
#SBATCH --error=LOGS/transformer_logs/experiments/%x_%A/%x_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH --array=0-1%2

set -euo pipefail

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER
module load bedtools
source .venv/bin/activate

# ==========================================
#        EXPERIMENT CONFIGURATION
# ==========================================

# Default/Initial Settings (baseline for all experiments)
# Preprocessing parameters
DEFAULT_MIN_GENES_PER_CELL=200
DEFAULT_MIN_PEAKS_PER_CELL=200
DEFAULT_FILTER_TYPE="count"
DEFAULT_FILTER_OUT_LOWEST_COUNTS_GENES=3
DEFAULT_FILTER_OUT_LOWEST_COUNTS_PEAKS=3
DEFAULT_FILTER_OUT_LOWEST_PCT_GENES=0.1
DEFAULT_FILTER_OUT_LOWEST_PCT_PEAKS=0.1
DEFAULT_NEIGHBORS_K=20
DEFAULT_PCA_COMPONENTS=25
DEFAULT_HOPS=1
DEFAULT_SELF_WEIGHT=1.0
DEFAULT_WINDOW_SIZE=1000
DEFAULT_DISTANCE_SCALE_FACTOR=20000
DEFAULT_MAX_PEAK_DISTANCE=100000
DEFAULT_DIST_BIAS_MODE="logsumexp"
DEFAULT_FILTER_TO_NEAREST_GENE=true
DEFAULT_PROMOTER_BP=""

# Model training parameters
DEFAULT_TOTAL_EPOCHS=250
DEFAULT_BATCH_SIZE=16
DEFAULT_PATIENCE=10
DEFAULT_SAVE_EVERY_N_EPOCHS=5
DEFAULT_CORR_LOSS_WEIGHT=1.0
DEFAULT_EDGE_LOSS_WEIGHT=0.0
DEFAULT_COS_WEIGHT=0.0
DEFAULT_SHORTCUT_REG_WEIGHT=0.0
DEFAULT_GRAD_ACCUM_STEPS=1
DEFAULT_USE_GRAD_ACCUMULATION=true
DEFAULT_USE_GRAD_CHECKPOINTING=true
DEFAULT_MODE="min"
DEFAULT_INITIAL_LEARNING_RATE=2.5e-4
DEFAULT_SCHEDULER_FACTOR=0.25
DEFAULT_SCHEDULER_PATIENCE=5
DEFAULT_THRESHOLD=1e-3
DEFAULT_THRESHOLD_MODE="rel"
DEFAULT_COOLDOWN=4
DEFAULT_MIN_LR=2.5e-6
DEFAULT_D_MODEL=192
DEFAULT_NUM_HEADS=4
DEFAULT_NUM_LAYERS=3
DEFAULT_D_FF=768
DEFAULT_DROPOUT=0.10
DEFAULT_USE_DISTANCE_BIAS=true
DEFAULT_USE_SHORTCUT=true
DEFAULT_USE_MOTIF_MASK=true
DEFAULT_MOTIF_MASK_THRESH=0.0
DEFAULT_MOTIF_PRIOR_SCALE=0.0
DEFAULT_ATTN_BIAS_SCALE=0.0
DEFAULT_SHORTCUT_L1=0.0
DEFAULT_SHORTCUT_L2=0.0
DEFAULT_SHORTCUT_TOPK=""
DEFAULT_SHORTCUT_DROPOUT=0.0
DEFAULT_SUBSAMPLE_MAX_TFS=""
DEFAULT_SUBSAMPLE_MAX_TGS=""
DEFAULT_SUBSAMPLE_MAX_WINDOWS_PER_CHROM=""
DEFAULT_SUBSAMPLE_MAX_CELLS=10000
DEFAULT_SUBSAMPLE_SEED=42
DEFAULT_ALLOWED_SAMPLES=""
DEFAULT_RESUME_CHECKPOINT_PATH=""

# Define experiments as arrays
# Format: "EXPERIMENT_NAME|DATASET_NAME|PARAMETER_OVERRIDES"
# PARAMETER_OVERRIDES format: "PARAM1=VALUE1;PARAM2=VALUE2;..."

# "test_new_pipeline|mESC_test_new_pipeline|MIN_GENES_PER_CELL=100;MIN_PEAKS_PER_CELL=100;HOPS=0;FILTER_TYPE=pct;FILTER_OUT_LOWEST_PCT_GENES=0.1;FILTER_OUT_LOWEST_PCT_PEAKS=0.1"
# "slow_decay_filter_ten_pct|mESC_slow_decay_filter_ten_pct|MIN_GENES_PER_CELL=100;MIN_PEAKS_PER_CELL=100;HOPS=0;FILTER_TYPE=pct;FILTER_OUT_LOWEST_PCT_GENES=0.1;FILTER_OUT_LOWEST_PCT_PEAKS=0.1;DISTANCE_SCALE_FACTOR=40000"

EXPERIMENTS=(
    # "no_filter_to_nearest_gene|mESC_no_filter_to_nearest_gene|FILTER_TO_NEAREST_GENE=false;HOPS=0"
    # "smaller_window_size|mESC_smaller_window_size|WINDOW_SIZE=500;HOPS=0"
    # "larger_window_size|mESC_larger_window_size|WINDOW_SIZE=1500;HOPS=0"
    # "lower_max_peak_dist|mESC_lower_max_peak_dist|MAX_PEAK_DISTANCE=50000;HOPS=0"
    # "higher_max_peak_dist|mESC_higher_max_peak_dist|MAX_PEAK_DISTANCE=150000;HOPS=0"
    # "fast_decay_large_window|mESC_fast_decay_large_window|WINDOW_SIZE=1500;DISTANCE_SCALE_FACTOR=10000;HOPS=0"
    # "slow_decay_small_window|mESC_slow_decay_small_window|WINDOW_SIZE=500;DISTANCE_SCALE_FACTOR=40000;HOPS=0"

    # "fewer_pca_components|mESC_fewer_pca_components|PCA_COMPONENTS=15;HOPS=0"
    # "more_pca_components|mESC_more_pca_components|PCA_COMPONENTS=50;HOPS=0"

    # "one_hop_diffusion|mESC_one_hop_diffusion|HOPS=1;NEIGHBORS_K=20"
    # "two_hop_diffusion|mESC_two_hop_diffusion|HOPS=2;NEIGHBORS_K=20"
    # "one_hop_large_neighborhood|mESC_one_hop_large_neighborhood|HOPS=1;NEIGHBORS_K=30"

    # "strict_genes_lenient_peaks|mESC_strict_genes_lenient_peaks|FILTER_TYPE=pct;FILTER_OUT_LOWEST_PCT_GENES=0.2;FILTER_OUT_LOWEST_PCT_PEAKS=0.05;HOPS=0"
    # "lenient_genes_strict_peaks|mESC_lenient_genes_strict_peaks|FILTER_TYPE=pct;FILTER_OUT_LOWEST_PCT_GENES=0.05;FILTER_OUT_LOWEST_PCT_PEAKS=0.2;HOPS=0"
    # "strict_filter_twenty_pct|mESC_strict_filter_twenty_pct|FILTER_TYPE=pct;FILTER_OUT_LOWEST_PCT_GENES=0.2;FILTER_OUT_LOWEST_PCT_PEAKS=0.2;HOPS=0"

    # "promoter_2kb|mESC_promoter_2kb|PROMOTER_BP=2000;MAX_PEAK_DISTANCE=50000;HOPS=0"
    # "promoter_5kb|mESC_promoter_5kb|PROMOTER_BP=5000;MAX_PEAK_DISTANCE=50000;HOPS=0"
    
    # "very_short_range|mESC_very_short_range|MAX_PEAK_DISTANCE=25000;DISTANCE_SCALE_FACTOR=10000;HOPS=0"
    # "long_range_enhancers|mESC_long_range_enhancers|MAX_PEAK_DISTANCE=250000;DISTANCE_SCALE_FACTOR=50000;HOPS=0"

    "slow_decay_long_range_two_hop|mESC_slow_decay_long_range_two_hop|DISTANCE_SCALE_FACTOR=40000;MAX_PEAK_DISTANCE=150000;HOPS=2;NEIGHBORS_K=20"
    # "slow_decay_long_range_zero_hops|mESC_slow_decay_long_range_zero_hops|DISTANCE_SCALE_FACTOR=40000;MAX_PEAK_DISTANCE=150000;HOPS=0;NEIGHBORS_K=20"

    # "promoter_only_10kb_two_hop|mESC_promoter_only_10kb_two_hop|PROMOTER_BP=10000;HOPS=2;NEIGHBORS_K=20"
    # "promoter_only_5kb_two_hop|mESC_promoter_only_5kb_two_hop|PROMOTER_BP=5000;HOPS=2;NEIGHBORS_K=20"
    # "promoter_only_2kb_two_hop|mESC_promoter_only_2kb_two_hop|PROMOTER_BP=2000;HOPS=2;NEIGHBORS_K=20"
    
    # "decay_30k_long_range_two_hop|mESC_decay_30k_long_range_two_hop|DISTANCE_SCALE_FACTOR=30000;MAX_PEAK_DISTANCE=150000;HOPS=2;NEIGHBORS_K=20"
    "decay_50k_long_range_two_hop|mESC_decay_50k_long_range_two_hop|DISTANCE_SCALE_FACTOR=50000;MAX_PEAK_DISTANCE=150000;HOPS=2;NEIGHBORS_K=20"
    # "decay_75k_long_range_two_hop|mESC_decay_75k_long_range_two_hop|DISTANCE_SCALE_FACTOR=75000;MAX_PEAK_DISTANCE=150000;HOPS=2;NEIGHBORS_K=20"


)


# ==========================================
#        EXPERIMENT SELECTION
# ==========================================

# Get the current experiment based on SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ ${TASK_ID} -ge ${#EXPERIMENTS[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${TASK_ID}) exceeds number of experiments (${#EXPERIMENTS[@]})"
    exit 1
fi

EXPERIMENT_CONFIG="${EXPERIMENTS[$TASK_ID]}"

# Parse experiment configuration
IFS='|' read -r EXPERIMENT_NAME DATASET_NAME PARAM_OVERRIDES <<< "$EXPERIMENT_CONFIG"

echo ""
echo "=========================================="
echo "  EXPERIMENT: ${EXPERIMENT_NAME}"
echo "  DATASET: ${DATASET_NAME}"
echo "  TASK ID: ${TASK_ID}"
echo "=========================================="
echo ""

# ==========================================
#        PARAMETER INITIALIZATION
# ==========================================

# Initialize all parameters with defaults
# Preprocessing parameters
MIN_GENES_PER_CELL=${DEFAULT_MIN_GENES_PER_CELL}
MIN_PEAKS_PER_CELL=${DEFAULT_MIN_PEAKS_PER_CELL}
FILTER_TYPE=${DEFAULT_FILTER_TYPE}
FILTER_OUT_LOWEST_COUNTS_GENES=${DEFAULT_FILTER_OUT_LOWEST_COUNTS_GENES}
FILTER_OUT_LOWEST_COUNTS_PEAKS=${DEFAULT_FILTER_OUT_LOWEST_COUNTS_PEAKS}
FILTER_OUT_LOWEST_PCT_GENES=${DEFAULT_FILTER_OUT_LOWEST_PCT_GENES}
FILTER_OUT_LOWEST_PCT_PEAKS=${DEFAULT_FILTER_OUT_LOWEST_PCT_PEAKS}
NEIGHBORS_K=${DEFAULT_NEIGHBORS_K}
PCA_COMPONENTS=${DEFAULT_PCA_COMPONENTS}
HOPS=${DEFAULT_HOPS}
SELF_WEIGHT=${DEFAULT_SELF_WEIGHT}
WINDOW_SIZE=${DEFAULT_WINDOW_SIZE}
DISTANCE_SCALE_FACTOR=${DEFAULT_DISTANCE_SCALE_FACTOR}
MAX_PEAK_DISTANCE=${DEFAULT_MAX_PEAK_DISTANCE}
DIST_BIAS_MODE=${DEFAULT_DIST_BIAS_MODE}
FILTER_TO_NEAREST_GENE=${DEFAULT_FILTER_TO_NEAREST_GENE}
PROMOTER_BP=${DEFAULT_PROMOTER_BP}

# Model training parameters
TOTAL_EPOCHS=${DEFAULT_TOTAL_EPOCHS}
BATCH_SIZE=${DEFAULT_BATCH_SIZE}
PATIENCE=${DEFAULT_PATIENCE}
SAVE_EVERY_N_EPOCHS=${DEFAULT_SAVE_EVERY_N_EPOCHS}
CORR_LOSS_WEIGHT=${DEFAULT_CORR_LOSS_WEIGHT}
EDGE_LOSS_WEIGHT=${DEFAULT_EDGE_LOSS_WEIGHT}
COS_WEIGHT=${DEFAULT_COS_WEIGHT}
SHORTCUT_REG_WEIGHT=${DEFAULT_SHORTCUT_REG_WEIGHT}
GRAD_ACCUM_STEPS=${DEFAULT_GRAD_ACCUM_STEPS}
USE_GRAD_ACCUMULATION=${DEFAULT_USE_GRAD_ACCUMULATION}
USE_GRAD_CHECKPOINTING=${DEFAULT_USE_GRAD_CHECKPOINTING}
MODE=${DEFAULT_MODE}
INITIAL_LEARNING_RATE=${DEFAULT_INITIAL_LEARNING_RATE}
SCHEDULER_FACTOR=${DEFAULT_SCHEDULER_FACTOR}
SCHEDULER_PATIENCE=${DEFAULT_SCHEDULER_PATIENCE}
THRESHOLD=${DEFAULT_THRESHOLD}
THRESHOLD_MODE=${DEFAULT_THRESHOLD_MODE}
COOLDOWN=${DEFAULT_COOLDOWN}
MIN_LR=${DEFAULT_MIN_LR}
D_MODEL=${DEFAULT_D_MODEL}
NUM_HEADS=${DEFAULT_NUM_HEADS}
NUM_LAYERS=${DEFAULT_NUM_LAYERS}
D_FF=${DEFAULT_D_FF}
DROPOUT=${DEFAULT_DROPOUT}
USE_DISTANCE_BIAS=${DEFAULT_USE_DISTANCE_BIAS}
USE_SHORTCUT=${DEFAULT_USE_SHORTCUT}
USE_MOTIF_MASK=${DEFAULT_USE_MOTIF_MASK}
MOTIF_MASK_THRESH=${DEFAULT_MOTIF_MASK_THRESH}
MOTIF_PRIOR_SCALE=${DEFAULT_MOTIF_PRIOR_SCALE}
ATTN_BIAS_SCALE=${DEFAULT_ATTN_BIAS_SCALE}
SHORTCUT_L1=${DEFAULT_SHORTCUT_L1}
SHORTCUT_L2=${DEFAULT_SHORTCUT_L2}
SHORTCUT_TOPK=${DEFAULT_SHORTCUT_TOPK}
SHORTCUT_DROPOUT=${DEFAULT_SHORTCUT_DROPOUT}
SUBSAMPLE_MAX_TFS=${DEFAULT_SUBSAMPLE_MAX_TFS}
SUBSAMPLE_MAX_TGS=${DEFAULT_SUBSAMPLE_MAX_TGS}
SUBSAMPLE_MAX_WINDOWS_PER_CHROM=${DEFAULT_SUBSAMPLE_MAX_WINDOWS_PER_CHROM}
SUBSAMPLE_MAX_CELLS=${DEFAULT_SUBSAMPLE_MAX_CELLS}
SUBSAMPLE_SEED=${DEFAULT_SUBSAMPLE_SEED}
ALLOWED_SAMPLES=${DEFAULT_ALLOWED_SAMPLES}
RESUME_CHECKPOINT_PATH=${DEFAULT_RESUME_CHECKPOINT_PATH}

# Apply parameter overrides
if [ -n "${PARAM_OVERRIDES}" ]; then
    echo "Applying parameter overrides:"
    IFS=';' read -ra OVERRIDES <<< "$PARAM_OVERRIDES"
    for override in "${OVERRIDES[@]}"; do
        if [ -n "$override" ]; then
            IFS='=' read -r param_name param_value <<< "$override"
            echo "  - ${param_name} = ${param_value}"
            
            # Apply the override by setting the variable dynamically
            case "$param_name" in
                # Preprocessing parameters
                MIN_GENES_PER_CELL) MIN_GENES_PER_CELL=$param_value ;;
                MIN_PEAKS_PER_CELL) MIN_PEAKS_PER_CELL=$param_value ;;
                FILTER_TYPE) FILTER_TYPE=$param_value ;;
                FILTER_OUT_LOWEST_COUNTS_GENES) FILTER_OUT_LOWEST_COUNTS_GENES=$param_value ;;
                FILTER_OUT_LOWEST_COUNTS_PEAKS) FILTER_OUT_LOWEST_COUNTS_PEAKS=$param_value ;;
                FILTER_OUT_LOWEST_PCT_GENES) FILTER_OUT_LOWEST_PCT_GENES=$param_value ;;
                FILTER_OUT_LOWEST_PCT_PEAKS) FILTER_OUT_LOWEST_PCT_PEAKS=$param_value ;;
                NEIGHBORS_K) NEIGHBORS_K=$param_value ;;
                PCA_COMPONENTS) PCA_COMPONENTS=$param_value ;;
                HOPS) HOPS=$param_value ;;
                SELF_WEIGHT) SELF_WEIGHT=$param_value ;;
                WINDOW_SIZE) WINDOW_SIZE=$param_value ;;
                DISTANCE_SCALE_FACTOR) DISTANCE_SCALE_FACTOR=$param_value ;;
                MAX_PEAK_DISTANCE) MAX_PEAK_DISTANCE=$param_value ;;
                DIST_BIAS_MODE) DIST_BIAS_MODE=$param_value ;;
                FILTER_TO_NEAREST_GENE) FILTER_TO_NEAREST_GENE=$param_value ;;
                PROMOTER_BP) PROMOTER_BP=$param_value ;;
                # Model training parameters
                TOTAL_EPOCHS) TOTAL_EPOCHS=$param_value ;;
                BATCH_SIZE) BATCH_SIZE=$param_value ;;
                PATIENCE) PATIENCE=$param_value ;;
                SAVE_EVERY_N_EPOCHS) SAVE_EVERY_N_EPOCHS=$param_value ;;
                CORR_LOSS_WEIGHT) CORR_LOSS_WEIGHT=$param_value ;;
                EDGE_LOSS_WEIGHT) EDGE_LOSS_WEIGHT=$param_value ;;
                COS_WEIGHT) COS_WEIGHT=$param_value ;;
                SHORTCUT_REG_WEIGHT) SHORTCUT_REG_WEIGHT=$param_value ;;
                GRAD_ACCUM_STEPS) GRAD_ACCUM_STEPS=$param_value ;;
                USE_GRAD_ACCUMULATION) USE_GRAD_ACCUMULATION=$param_value ;;
                USE_GRAD_CHECKPOINTING) USE_GRAD_CHECKPOINTING=$param_value ;;
                MODE) MODE=$param_value ;;
                INITIAL_LEARNING_RATE) INITIAL_LEARNING_RATE=$param_value ;;
                SCHEDULER_FACTOR) SCHEDULER_FACTOR=$param_value ;;
                SCHEDULER_PATIENCE) SCHEDULER_PATIENCE=$param_value ;;
                THRESHOLD) THRESHOLD=$param_value ;;
                THRESHOLD_MODE) THRESHOLD_MODE=$param_value ;;
                COOLDOWN) COOLDOWN=$param_value ;;
                MIN_LR) MIN_LR=$param_value ;;
                D_MODEL) D_MODEL=$param_value ;;
                NUM_HEADS) NUM_HEADS=$param_value ;;
                NUM_LAYERS) NUM_LAYERS=$param_value ;;
                D_FF) D_FF=$param_value ;;
                DROPOUT) DROPOUT=$param_value ;;
                USE_DISTANCE_BIAS) USE_DISTANCE_BIAS=$param_value ;;
                USE_SHORTCUT) USE_SHORTCUT=$param_value ;;
                USE_MOTIF_MASK) USE_MOTIF_MASK=$param_value ;;
                MOTIF_MASK_THRESH) MOTIF_MASK_THRESH=$param_value ;;
                MOTIF_PRIOR_SCALE) MOTIF_PRIOR_SCALE=$param_value ;;
                ATTN_BIAS_SCALE) ATTN_BIAS_SCALE=$param_value ;;
                SHORTCUT_L1) SHORTCUT_L1=$param_value ;;
                SHORTCUT_L2) SHORTCUT_L2=$param_value ;;
                SHORTCUT_TOPK) SHORTCUT_TOPK=$param_value ;;
                SHORTCUT_DROPOUT) SHORTCUT_DROPOUT=$param_value ;;
                SUBSAMPLE_MAX_TFS) SUBSAMPLE_MAX_TFS=$param_value ;;
                SUBSAMPLE_MAX_TGS) SUBSAMPLE_MAX_TGS=$param_value ;;
                SUBSAMPLE_MAX_WINDOWS_PER_CHROM) SUBSAMPLE_MAX_WINDOWS_PER_CHROM=$param_value ;;
                SUBSAMPLE_MAX_CELLS) SUBSAMPLE_MAX_CELLS=$param_value ;;
                SUBSAMPLE_SEED) SUBSAMPLE_SEED=$param_value ;;
                ALLOWED_SAMPLES) ALLOWED_SAMPLES=$param_value ;;
                RESUME_CHECKPOINT_PATH) RESUME_CHECKPOINT_PATH=$param_value ;;
                *) echo "WARNING: Unknown parameter: $param_name" ;;
            esac
        fi
    done
    echo ""
fi

# ==========================================
#        FIXED CONFIGURATION
# ==========================================

# Basic paths - these remain constant across experiments
ROOT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
PROJECT_DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
PROJECT_RESULT_DIR="/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
RAW_SINGLE_CELL_DATA="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS"
RAW_10X_RNA_DATA_DIR="${RAW_SINGLE_CELL_DATA}/DS014_DOI496239_MOUSE_ESC_RAW_FILES"
RAW_ATAC_PEAK_MATRIX_FILE="${RAW_SINGLE_CELL_DATA}/DS014_DOI496239_MOUSE_ESCDAYS7AND8/scATAC_PeakMatrix.txt"

# Dataset configuration
ORGANISM_CODE="mm10"

# Derived paths (based on DATASET_NAME)
DATABASE_DIR="${ROOT_DIR}/data"
PROCESSED_DATA="${DATABASE_DIR}/processed"
TRAINING_DATA_CACHE="${DATABASE_DIR}/training_data_cache"
EXPERIMENT_DIR="${PROJECT_DATA_DIR}/experiments"
OUTPUT_DIR="${EXPERIMENT_DIR}/${DATASET_NAME}"
SAMPLE_PROCESSED_DATA_DIR="${PROCESSED_DATA}/${DATASET_NAME}"
SAMPLE_DATA_CACHE_DIR="${TRAINING_DATA_CACHE}/${DATASET_NAME}"
COMMON_DATA="${SAMPLE_DATA_CACHE_DIR}/common"

# Sample information
SAMPLE_NAMES="E7.5_rep1 E7.5_rep2 E7.75_rep1 E8.0_rep2 E8.5_rep2 E8.75_rep2 E8.0_rep1 E8.5_rep1"
VALIDATION_DATASETS="E8.75_rep1"

# Chromosomes to process
CHROM_ID="chr19"
CHROM_IDS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19"

# Force recalculate (set to true if you want to reprocess data)
FORCE_RECALCULATE=false

# ==========================================
#        CPU DETECTION
# ==========================================

determine_num_cpus() {
    echo ""
    echo "[INFO] Checking the number of CPUs available for parallel processing"
    if [ -z "${SLURM_CPUS_PER_TASK:-}" ]; then
        if command -v nproc &> /dev/null; then
            TOTAL_CPUS=$(nproc --all)
            case $TOTAL_CPUS in
                [1-15]) IGNORED_CPUS=1 ;;
                [16-31]) IGNORED_CPUS=2 ;;
                *) IGNORED_CPUS=4 ;;
            esac
            NUM_CPU=$((TOTAL_CPUS - IGNORED_CPUS))
            echo "    - Running locally. Detected $TOTAL_CPUS CPUs, reserving $IGNORED_CPUS for system tasks. Using $NUM_CPU CPUs."
        else
            NUM_CPU=1
            echo "    - Running locally. Unable to detect CPUs, defaulting to $NUM_CPU CPU."
        fi
    else
        NUM_CPU=${SLURM_CPUS_PER_TASK}
        echo "    - Running on SLURM. Number of CPUs allocated: ${NUM_CPU}"
    fi
}
determine_num_cpus

# ==========================================
#             PREPROCESSING
# ==========================================
echo ""
echo "=========================================="
echo "         STARTING PREPROCESSING"
echo "=========================================="
echo ""

# Build preprocessing command
PREPROCESS_CMD="python src/multiomic_transformer/data/preprocess_argparse.py \
    --root_dir ${ROOT_DIR} \
    --num_cpu ${NUM_CPU} \
    --project_data_dir ${PROJECT_DATA_DIR} \
    --project_result_dir ${PROJECT_RESULT_DIR} \
    --organism_code ${ORGANISM_CODE} \
    --dataset_name ${DATASET_NAME} \
    --sample_names ${SAMPLE_NAMES} \
    --chrom_id ${CHROM_ID} \
    --chrom_ids ${CHROM_IDS} \
    --raw_single_cell_data ${RAW_SINGLE_CELL_DATA} \
    --raw_10x_rna_data_dir ${RAW_10X_RNA_DATA_DIR} \
    --raw_atac_peak_matrix_file ${RAW_ATAC_PEAK_MATRIX_FILE} \
    --min_genes_per_cell ${MIN_GENES_PER_CELL} \
    --min_peaks_per_cell ${MIN_PEAKS_PER_CELL} \
    --filter_type ${FILTER_TYPE} \
    --filter_out_lowest_counts_genes ${FILTER_OUT_LOWEST_COUNTS_GENES} \
    --filter_out_lowest_counts_peaks ${FILTER_OUT_LOWEST_COUNTS_PEAKS} \
    --filter_out_lowest_pct_genes ${FILTER_OUT_LOWEST_PCT_GENES} \
    --filter_out_lowest_pct_peaks ${FILTER_OUT_LOWEST_PCT_PEAKS} \
    --neighbors_k ${NEIGHBORS_K} \
    --pca_components ${PCA_COMPONENTS} \
    --hops ${HOPS} \
    --self_weight ${SELF_WEIGHT} \
    --window_size ${WINDOW_SIZE} \
    --distance_scale_factor ${DISTANCE_SCALE_FACTOR} \
    --max_peak_distance ${MAX_PEAK_DISTANCE} \
    --dist_bias_mode ${DIST_BIAS_MODE}"

# Add optional validation datasets
if [ -n "${VALIDATION_DATASETS}" ]; then
    PREPROCESS_CMD="${PREPROCESS_CMD} --validation_datasets ${VALIDATION_DATASETS}"
fi

# Add optional flags
if [ "${FORCE_RECALCULATE}" = "true" ]; then
    PREPROCESS_CMD="${PREPROCESS_CMD} --force_recalculate"
fi

if [ "${FILTER_TO_NEAREST_GENE}" = "true" ]; then
    PREPROCESS_CMD="${PREPROCESS_CMD} --filter_to_nearest_gene"
fi

# Add optional promoter_bp if set
if [ -n "${PROMOTER_BP}" ]; then
    PREPROCESS_CMD="${PREPROCESS_CMD} --promoter_bp ${PROMOTER_BP}"
fi

# Execute preprocessing
eval ${PREPROCESS_CMD}

echo ""
echo "Preprocessing completed successfully!"
echo ""

# ==========================================
#              MODEL TRAINING
# ==========================================

# Check if we're on a GPU-enabled partition before training
if [[ "${SLURM_JOB_PARTITION:-}" == "dense" ]] || [[ "${SLURM_JOB_PARTITION:-}" == "gpu" ]]; then    
    echo ""
    echo "=========================================="
    echo "         STARTING MODEL TRAINING"
    echo "=========================================="
    echo ""
    echo "[INFO] Detected compute partition, progressing with model training"

    # Build training command
    TRAIN_CMD="src/multiomic_transformer/scripts/multinode_train_argparse.py \
        --sample_data_cache_dir ${SAMPLE_DATA_CACHE_DIR} \
        --common_data ${COMMON_DATA} \
        --output_dir ${OUTPUT_DIR} \
        --chrom_id ${CHROM_ID} \
        --chrom_ids ${CHROM_IDS} \
        --total_epochs ${TOTAL_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --patience ${PATIENCE} \
        --save_every_n_epochs ${SAVE_EVERY_N_EPOCHS} \
        --corr_loss_weight ${CORR_LOSS_WEIGHT} \
        --edge_loss_weight ${EDGE_LOSS_WEIGHT} \
        --cos_weight ${COS_WEIGHT} \
        --shortcut_reg_weight ${SHORTCUT_REG_WEIGHT} \
        --grad_accum_steps ${GRAD_ACCUM_STEPS} \
        --mode ${MODE} \
        --initial_learning_rate ${INITIAL_LEARNING_RATE} \
        --scheduler_factor ${SCHEDULER_FACTOR} \
        --scheduler_patience ${SCHEDULER_PATIENCE} \
        --threshold ${THRESHOLD} \
        --threshold_mode ${THRESHOLD_MODE} \
        --cooldown ${COOLDOWN} \
        --min_lr ${MIN_LR} \
        --d_model ${D_MODEL} \
        --num_heads ${NUM_HEADS} \
        --num_layers ${NUM_LAYERS} \
        --d_ff ${D_FF} \
        --dropout ${DROPOUT} \
        --motif_mask_thresh ${MOTIF_MASK_THRESH} \
        --motif_prior_scale ${MOTIF_PRIOR_SCALE} \
        --attn_bias_scale ${ATTN_BIAS_SCALE} \
        --shortcut_l1 ${SHORTCUT_L1} \
        --shortcut_l2 ${SHORTCUT_L2} \
        --shortcut_dropout ${SHORTCUT_DROPOUT} \
        --subsample_max_cells ${SUBSAMPLE_MAX_CELLS} \
        --subsample_seed ${SUBSAMPLE_SEED} \
        --use_torch_compile \
        --use_profiler \
        --profiler_start_step 5 \
        --profiler_active_steps 3"

    # Add boolean flags
    if [ "${USE_GRAD_ACCUMULATION}" = "true" ]; then
        TRAIN_CMD="${TRAIN_CMD} --use_grad_accumulation"
    fi

    if [ "${USE_GRAD_CHECKPOINTING}" = "true" ]; then
        TRAIN_CMD="${TRAIN_CMD} --use_grad_checkpointing"
    fi

    if [ "${USE_DISTANCE_BIAS}" = "true" ]; then
        TRAIN_CMD="${TRAIN_CMD} --use_distance_bias"
    fi

    if [ "${USE_SHORTCUT}" = "true" ]; then
        TRAIN_CMD="${TRAIN_CMD} --use_shortcut"
    fi

    if [ "${USE_MOTIF_MASK}" = "true" ]; then
        TRAIN_CMD="${TRAIN_CMD} --use_motif_mask"
    fi

    # Add optional integer/string parameters
    if [ -n "${SHORTCUT_TOPK}" ]; then
        TRAIN_CMD="${TRAIN_CMD} --shortcut_topk ${SHORTCUT_TOPK}"
    fi

    if [ -n "${SUBSAMPLE_MAX_TFS}" ]; then
        TRAIN_CMD="${TRAIN_CMD} --subsample_max_tfs ${SUBSAMPLE_MAX_TFS}"
    fi

    if [ -n "${SUBSAMPLE_MAX_TGS}" ]; then
        TRAIN_CMD="${TRAIN_CMD} --subsample_max_tgs ${SUBSAMPLE_MAX_TGS}"
    fi

    if [ -n "${SUBSAMPLE_MAX_WINDOWS_PER_CHROM}" ]; then
        TRAIN_CMD="${TRAIN_CMD} --subsample_max_windows_per_chrom ${SUBSAMPLE_MAX_WINDOWS_PER_CHROM}"
    fi

    if [ -n "${ALLOWED_SAMPLES}" ]; then
        TRAIN_CMD="${TRAIN_CMD} --allowed_samples ${ALLOWED_SAMPLES}"
    fi

    if [ -n "${RESUME_CHECKPOINT_PATH}" ]; then
        TRAIN_CMD="${TRAIN_CMD} --resume_checkpoint_path ${RESUME_CHECKPOINT_PATH}"
    fi

    # --- Memory + math ---
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
    export TORCH_ALLOW_TF32=1
    export NVIDIA_TF32_OVERRIDE=1

    # --- Threading: set to match SLURM CPUs per task ---
    THREADS=${SLURM_CPUS_PER_TASK:-1}
    export OMP_NUM_THREADS=$THREADS
    export MKL_NUM_THREADS=$THREADS
    export OPENBLAS_NUM_THREADS=$THREADS
    export NUMEXPR_NUM_THREADS=$THREADS
    export BLIS_NUM_THREADS=$THREADS
    export KMP_AFFINITY=granularity=fine,compact,1,0

    # --- NCCL / networking overrides ---
    # Dynamically find the interface with 10.90.29.* network
    export IFACE=$(ip -o -4 addr show | grep "10.90.29." | awk '{print $2}')

    if [ -z "$IFACE" ]; then
        echo "[ERROR] Could not find interface with 10.90.29.* network on $(hostname)"
        ip -o -4 addr show  # Show all interfaces for debugging
        exit 1
    fi

    echo "[INFO] Using IFACE=$IFACE on host $(hostname)"
    ip -o -4 addr show "$IFACE"

    export NCCL_SOCKET_IFNAME="$IFACE"
    export GLOO_SOCKET_IFNAME="$IFACE"

    # (keep InfiniBand disabled if IB isn’t properly configured)
    export NCCL_IB_DISABLE=1

    export TORCH_DISTRIBUTED_DEBUG=DETAIL

    ##### Number of total processes
    echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
    echo "Nodelist        = " $SLURM_JOB_NODELIST
    echo "Number of nodes = " $SLURM_JOB_NUM_NODES
    echo "Ntasks per node = " $SLURM_NTASKS_PER_NODE
    echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
    echo ""

    # ---------- Logging ----------
    LOGDIR="$PWD/LOGS/transformer_logs/experiments/gpu_usage"
    mkdir -p "$LOGDIR"
    JOB=transformer_training
    ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$PPID}}
    ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
    TS=$(date +%Y%m%d_%H%M%S)
    GPULOG="$LOGDIR/gpu_usage_${ARRAY_JOB_ID}_${ARRAY_TASK_ID}.csv"

    # ---------- GPU sampler (runs only on the batch node) ----------
    trap 'pkill -P $$ || true' EXIT
    nvidia-smi -L
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
    --format=csv -l 30 > "$GPULOG" &

    # ---------- torchrun multi-node launch ----------
    # Pick the first node as rendezvous/master
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_PORT=29500
    export MASTER_ADDR MASTER_PORT

    echo "[INFO] MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

    # ---------- Optional network diagnostics ----------
    DEBUG_NET=${DEBUG_NET:-1}   # set to 0 to skip tests once things work

    NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
    MASTER_NODE=${NODES[0]}

    echo "[NET] Nodes in this job: ${NODES[*]}"
    echo "[NET] MASTER_NODE=${MASTER_NODE}, IFACE=${IFACE:-<unset>}"

    NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
    echo "[INFO] Using nproc_per_node=$NPROC_PER_NODE based on GPUs per node"

    # Execute training with torchrun
    srun bash -c "torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=\$SLURM_NODEID \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        ${TRAIN_CMD}"

    echo ""
    echo "=========================================="
    echo "  EXPERIMENT COMPLETED: ${EXPERIMENT_NAME}"
    echo "  DATASET: ${DATASET_NAME}"
    echo "=========================================="
    echo ""
else
    echo ""
    echo "=========================================="
    echo "   SKIPPING MODEL TRAINING"
    echo "=========================================="
    echo ""
    echo "[INFO] Running on ${SLURM_JOB_PARTITION:-unknown} partition (not 'dense'), skipping model training"
    echo "[INFO] Submit another job on the 'dense' partition to run training"
fi

