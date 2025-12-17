#!/bin/bash -l
#SBATCH --job-name=transformer_training
#SBATCH --output=LOGS/transformer_logs/03_training/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/03_training/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH -p dense
#SBATCH -N 2
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=128G

set -euo pipefail

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER
source .venv/bin/activate

# ==========================================
#             USER VARIABLES
# ==========================================

# ----- Basic Paths -----
ROOT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
PROJECT_DATA_DIR="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
PROJECT_RESULT_DIR="/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
RAW_SINGLE_CELL_DATA="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS"
RAW_10X_RNA_DATA_DIR="${RAW_SINGLE_CELL_DATA}/DS014_DOI496239_MOUSE_ESC_RAW_FILES"
RAW_ATAC_PEAK_MATRIX_FILE="${RAW_SINGLE_CELL_DATA}/DS014_DOI496239_MOUSE_ESCDAYS7AND8/scATAC_PeakMatrix.txt"

# ----- Dataset Configuration -----
ORGANISM_CODE="mm10"
DATASET_NAME="mESC_higher_max_peak_dist"

# ----- PATH SETUP -----
# Database and genome paths
DATABASE_DIR="${ROOT_DIR}/data"
GENOME_DIR="${DATABASE_DIR}/genome_data/reference_genome/${ORGANISM_CODE}"
CHROM_SIZES_FILE="${GENOME_DIR}/${ORGANISM_CODE}.chrom.sizes"
GTF_FILE_DIR="${DATABASE_DIR}/genome_data/genome_annotation/${ORGANISM_CODE}"
NCBI_FILE_DIR="${DATABASE_DIR}/genome_data/genome_annotation/${ORGANISM_CODE}"
GENE_TSS_FILE="${DATABASE_DIR}/genome_data/genome_annotation/${ORGANISM_CODE}/gene_tss.bed"
TF_FILE="${DATABASE_DIR}/databases/motif_information/${ORGANISM_CODE}/TF_Information_all_motifs.txt"
JASPAR_PFM_DIR="${DATABASE_DIR}/databases/motif_information/JASPAR/pfm_files"
MOTIF_DIR="${DATABASE_DIR}/databases/motif_information/${ORGANISM_CODE}/pwms_all_motifs"

# Ground truth data
CHIP_GROUND_TRUTH="${DATABASE_DIR}/ground_truth_files/mESC_beeline_ChIP-seq.csv"
CHIP_GROUND_TRUTH_SEP=","

# Data directories
PROCESSED_DATA="${DATABASE_DIR}/processed"
TRAINING_DATA_CACHE="${DATABASE_DIR}/training_data_cache"
RAW_DATA="${DATABASE_DIR}/raw"
PKN_DIR="${DATABASE_DIR}/prior_knowledge_network_data/${ORGANISM_CODE}"

# PKN files
STRING_DIR="${DATABASE_DIR}/prior_knowledge_network_data/${ORGANISM_CODE}/STRING"
TRRUST_DIR="${DATABASE_DIR}/prior_knowledge_network_data/${ORGANISM_CODE}/TRRUST"
KEGG_DIR="${DATABASE_DIR}/prior_knowledge_network_data/${ORGANISM_CODE}/KEGG"

# Experiment paths
EXPERIMENT_DIR="${PROJECT_DATA_DIR}/experiments"
OUTPUT_DIR="${EXPERIMENT_DIR}/${DATASET_NAME}"

# Sample-specific paths
SAMPLE_PROCESSED_DATA_DIR="${PROCESSED_DATA}/${DATASET_NAME}"
SAMPLE_DATA_CACHE_DIR="${TRAINING_DATA_CACHE}/${DATASET_NAME}"
COMMON_DATA="${SAMPLE_DATA_CACHE_DIR}/common"

# Fine-tuning (optional - set FINE_TUNING_TRAINED_MODEL if needed)
FINE_TUNING_TRAINED_MODEL=""  # e.g., "chr19/model_training_003"
FINE_TUNING_DIR="${OUTPUT_DIR}/${FINE_TUNING_TRAINED_MODEL}"

# Inferred network output
INFERRED_NETWORK_OUTPUT_DIR="${ROOT_DIR}/output/transformer_testing_output/chrom_inferred_grn_orti_chipatlas_rn117_unique"

# ----- Sample Information -----
SAMPLE_NAMES="E7.5_rep1 E7.5_rep2 E7.75_rep1 E8.0_rep2 E8.5_rep2 E8.75_rep2 E8.0_rep1 E8.5_rep1"
VALIDATION_DATASETS="E8.75_rep1"
FINE_TUNING_DATASETS="E7.5_rep1 E7.5_rep2 E7.75_rep1 E8.0_rep2 E8.5_rep2 E8.75_rep2 E8.0_rep1 E8.5_rep1"

# Chromosomes to process
CHROM_ID="chr19"
CHROM_IDS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19"

# ----- QC Filtering Parameters -----
MIN_GENES_PER_CELL=200
MIN_PEAKS_PER_CELL=200
FILTER_TYPE="count"
FILTER_OUT_LOWEST_COUNTS_GENES=3
FILTER_OUT_LOWEST_COUNTS_PEAKS=3
FILTER_OUT_LOWEST_PCT_GENES=0.1
FILTER_OUT_LOWEST_PCT_PEAKS=0.01

# ----- Pseudobulk Parameters -----
NEIGHBORS_K=20
PCA_COMPONENTS=25
HOPS=0
SELF_WEIGHT=1.0

# ----- Data Preprocessing and Caching -----
FORCE_RECALCULATE=false
WINDOW_SIZE=1000
DISTANCE_SCALE_FACTOR=20000
MAX_PEAK_DISTANCE=150000
DIST_BIAS_MODE="logsumexp"
FILTER_TO_NEAREST_GENE=true
PROMOTER_BP=""  # Leave empty for None

# ----- Model Training Parameters -----
TOTAL_EPOCHS=250
BATCH_SIZE=16
PATIENCE=10
SAVE_EVERY_N_EPOCHS=5

# Loss weights
CORR_LOSS_WEIGHT=1.0
EDGE_LOSS_WEIGHT=0.0
COS_WEIGHT=0.0
SHORTCUT_REG_WEIGHT=0.0

# Optimization
GRAD_ACCUM_STEPS=1
USE_GRAD_ACCUMULATION=true
USE_GRAD_CHECKPOINTING=true

# Learning rate schedule
MODE="min"
INITIAL_LEARNING_RATE=2.5e-4
SCHEDULER_FACTOR=0.25
SCHEDULER_PATIENCE=5
THRESHOLD=1e-3
THRESHOLD_MODE="rel"
COOLDOWN=4
MIN_LR=2.5e-6

# Model architecture
D_MODEL=192
NUM_HEADS=4
NUM_LAYERS=3
D_FF=$((D_MODEL * 4))  # 768
DROPOUT=0.10

# Model features
USE_DISTANCE_BIAS=true
USE_SHORTCUT=true
USE_MOTIF_MASK=true
MOTIF_MASK_THRESH=0.0
MOTIF_PRIOR_SCALE=0.0
ATTN_BIAS_SCALE=0.0

# Shortcut parameters
SHORTCUT_L1=0.0
SHORTCUT_L2=0.0
SHORTCUT_TOPK=""  # Leave empty for None
SHORTCUT_DROPOUT=0.0

# Data subsampling
SUBSAMPLE_MAX_TFS=""  # Leave empty for None
SUBSAMPLE_MAX_TGS=""  # Leave empty for None
SUBSAMPLE_MAX_WINDOWS_PER_CHROM=""  # Leave empty for None
SUBSAMPLE_MAX_CELLS=10000
SUBSAMPLE_SEED=42
ALLOWED_SAMPLES=""  # Leave empty for None

# Checkpoint resumption
RESUME_CHECKPOINT_PATH=""  # Leave empty for None


determine_num_cpus() {
    echo ""
    echo "[INFO] Checking the number of CPUs available for parallel processing"
    if [ -z "${SLURM_CPUS_PER_TASK:-}" ]; then
        if command -v nproc &> /dev/null; then
            TOTAL_CPUS=$(nproc --all)
            case $TOTAL_CPUS in
                [1-15]) IGNORED_CPUS=1 ;;  # Reserve 1 CPU for <=15 cores
                [16-31]) IGNORED_CPUS=2 ;; # Reserve 2 CPUs for <=31 cores
                *) IGNORED_CPUS=4 ;;       # Reserve 4 CPUs for >=32 cores
            esac
            NUM_CPU=$((TOTAL_CPUS - IGNORED_CPUS))
            echo "    - Running locally. Detected $TOTAL_CPUS CPUs, reserving $IGNORED_CPUS for system tasks. Using $NUM_CPU CPUs."
        else
            NUM_CPU=1  # Fallback
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
echo ""
echo "=========================================="
echo "         STARTING MODEL TRAINING"
echo "=========================================="
echo ""

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
    --subsample_seed ${SUBSAMPLE_SEED}"

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

# Execute training with torchrun
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(hostname):29500 \
    ${TRAIN_CMD}

echo ""
echo "Model training completed successfully!"
echo ""