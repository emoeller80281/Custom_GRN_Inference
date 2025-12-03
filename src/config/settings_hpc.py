from pathlib import Path
import numpy as np
import itertools

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

chr_nums = [f"chr{i}" for i in range(1, 20)]
chrom_list = chr_nums #+ ["chrX", "chrY"]

# ----- SAMPLE INFORMATION -----
# Sample information
ORGANISM_CODE = "mm10"
DATASET_NAME = "mESC_no_scale_linear"
CHROM_ID_LIST = chrom_list
CHROM_ID = "chr19"
CHROM_IDS = chr_nums
# , "E7.75_rep1", "E8.0_rep2", "E8.5_rep2", "E8.75_rep2", "E7.5_rep2", "E8.0_rep1", "E8.5_rep1"
SAMPLE_NAMES = ["E7.5_rep1", "E7.5_rep2", "E7.75_rep1", "E8.0_rep2", "E8.5_rep2", "E8.75_rep2", "E8.0_rep1", "E8.5_rep1"]
FINE_TUNING_DATASETS = ["E7.5_rep1"]

# Paths to the raw scRNA-seq and scATAC-seq data
RAW_SINGLE_CELL_DATA = Path("/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS")
RAW_10X_RNA_DATA_DIR = RAW_SINGLE_CELL_DATA / "DS014_DOI496239_MOUSE_ESC_RAW_FILES"
RAW_ATAC_PEAK_MATRIX_FILE = RAW_SINGLE_CELL_DATA / "DS014_DOI496239_MOUSE_ESCDAYS7AND8" / "scATAC_PeakMatrix.txt"

RAW_GSE218576_DIR= ROOT_DIR / "data/raw/GSE218576"
PROCESSED_GSE218576_DIR = ROOT_DIR / "data/processed/GSE218576"

# ----- DATA PREPARATION -----
# Pseudobulk and Data Preprocessing
# Change these files to suit what you want the raw or processed files to look like.
# Please respect the file extensions here
PROCESSED_RNA_FILENAME = "scRNA_seq_processed.parquet"
PROCESSED_ATAC_FILENAME = "scATAC_seq_processed.parquet"
RAW_RNA_FILE = "scRNA_seq_raw.parquet"
RAW_ATAC_FILE = "scATAC_seq_raw.parquet"
ADATA_RNA_FILE = "adata_RNA.h5ad"
ADATA_ATAC_FILE = "adata_ATAC.h5ad"
PSEUDOBULK_TG_FILE = "TG_pseudobulk.tsv"
PSEUDOBULK_RE_FILE = "RE_pseudobulk.tsv"

NEIGHBORS_K = 20        # Number of nearest neighpers per cell in the KNN graph
PCA_COMPONENTS = 25     # Number of PCs per modality, controls variation
HOPS = 1                # Number of neighbors-of-neighbors to use (smooths across community)
SELF_WEIGHT = 1.0       # How much to weight the cells own gene expression. Higher = les blending with neighbors

# Data Preprocessing and Caching
VALIDATION_DATASETS = ["E8.75_rep1"]
FORCE_RECALCULATE = True                # Recomputes genomic windows, peak-TG distance, and re-runs sliding window TF-peak scan
WINDOW_SIZE = 1_000                     # Aggregates peaks within WINDOW_SIZE bp genomic tiles
DISTANCE_SCALE_FACTOR = 20_000           # Weights the peak-gene TSS distance score. Lower numbers = faster dropoff
MAX_PEAK_DISTANCE = 100_000              # Masks out peaks further than this distance from the gene TSS
DIST_BIAS_MODE = "logsumexp"            # Method for calcuting window -> gene TSS distance. Options: "max" | "sum" | "mean" | "logsumexp"
FILTER_TO_NEAREST_GENE = True           # Associate peaks to the nearest gene
PROMOTER_BP = None #10_000

# ----- MODEL TRAINING PARAMETERS -----
TOTAL_EPOCHS=200
BATCH_SIZE=128
PATIENCE=15
CORR_LOSS_WEIGHT=1.0    
ALLOWED_SAMPLES=None #["E7.5_REP1"]        

D_MODEL = 192
NUM_HEADS = 4
NUM_LAYERS = 3
D_FF = D_MODEL * 4
DROPOUT = 0.10
EDGE_LOSS_WEIGHT=0.0            # Weight for edge loss contribution
COS_WEIGHT=0.0                  # Weight for cosine contrastive loss contribution   

SAVE_EVERY_N_EPOCHS=5           # Chooses how many epochs to run before saving a checkpoint
RESUME_CHECKPOINT_PATH=None     #"/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC_no_scale_linear/chr19/model_training_049/checkpoint_120.pt"

GRAD_ACCUM_STEPS=1
USE_GRAD_ACCUMULATION=True
USE_GRAD_CHECKPOINTING=True

# Training scheduler settings
MODE="min"                      # min = improvement means a lower number; max = improvement means a higher number
INITIAL_LEARNING_RATE = 5e-4    # Initial learning rate for the model
SCHEDULER_FACTOR=0.5            # How much to reduce the learning rate on a plateau
SCHEDULER_PATIENCE=5            # How long to wait with no improvement without dropping the learning rate
THRESHOLD=1e-3                  # Defines how much better the next epoch has to be to count as being "better"
THRESHOLD_MODE="rel"            # rel helps filter noise for datasets with different loss scales. new best = previous best * (1 - threshold) difference
COOLDOWN=3                      # How many epochs to pause after a drop before testing for improvement, lets the training stabilize a bit
MIN_LR=1e-5                     # Wont drop the learning rate below this, prevents learning from stalling due to tiny lr

# TF to TG shortcut parameters
USE_DISTANCE_BIAS = True
USE_SHORTCUT = True
USE_MOTIF_MASK = True
MOTIF_MASK_THRESH = 0.0         # Only allow TF-TG edges with p <= -log10(MOTIF_MASK_THRESH)
MOTIF_PRIOR_SCALE = 0.0         # Allows adding scaled motif scores on edges that are not filtered
SHORTCUT_L1 = 0                 # Encourages sparsity between TF-TG edges in the TFtoTG shortcut
SHORTCUT_L2 = 0 #1e-3
SHORTCUT_TOPK = None
SHORTCUT_DROPOUT = 0.0 #0.1
SHORTCUT_REG_WEIGHT = 1.0

# peak-TG distance bias
ATTN_BIAS_SCALE = 1.0 

# Sampling (optional)
SUBSAMPLE_MAX_TFS = None #400 # 150
SUBSAMPLE_MAX_TGS = None #2000 #None
SUBSAMPLE_MAX_WINDOWS_PER_CHROM = None #1000
SUBSAMPLE_MAX_CELLS= None
SUBSAMPLE_SEED = 42

# ----- FINE TUNING ON SINGLE-CELL DATA -----
FINE_TUNING_TRAINED_MODEL = "model_training_011"
FINETUNE_PATIENCE = 20
FINETUNE_LR = 1e-4       # smaller LR for refinement
EWC_LAMBDA = 10.0

# ----- GAT CLASSIFIER MODEL -----
PRETRAINED_EMB_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC_no_scale_linear/chr19/model_training_035/"

# Model
HIDDEN_DIM = 384
GAT_HEADS = 6
GAT_DROPOUT = 0.30
EDGE_DROPOUT = 0.30

# Phase 1 (DGI)
DGI_EPOCHS = 80

# Phase 2 (fine-tune)
FINETUNE_EPOCHS = 80
TEST_SIZE = 0.20
LR_ENCODER = 1e-5       # Learning rate for the (partially) unfrozen encoder
LR_HEAD = 1e-4          # Learning rate for the classifier head
WEIGHT_DECAY = 1e-4
L2SP_LAMBDA = 0      # Strength of L2-SP regularization

# ChIP eval
PRECISION_AT_K = (10, 20, 50, 100, 200, 500)
RUN_OPTUNA = False

# ----- PATH SETUP -----
# Fixed Paths
DATA_DIR = ROOT_DIR / "data"
GENOME_DIR = DATA_DIR / "genome_data" / "reference_genome" / ORGANISM_CODE
CHROM_SIZES_FILE = GENOME_DIR / f"{ORGANISM_CODE}.chrom.sizes"
GTF_FILE_DIR = DATA_DIR / "genome_data" / "genome_annotation" / ORGANISM_CODE
NCBI_FILE_DIR = DATA_DIR / "genome_data" / "genome_annotation" / ORGANISM_CODE 
GENE_TSS_FILE = DATA_DIR / "genome_data" / "genome_annotation" / ORGANISM_CODE / "gene_tss.bed"
TF_FILE = DATA_DIR / "databases" / "motif_information" / ORGANISM_CODE / "TF_Information_all_motifs.txt"
JASPAR_PFM_DIR = DATA_DIR / "databases" / "motif_information" / "JASPAR" / "pfm_files"
MOTIF_DIR = DATA_DIR / "databases" / "motif_information" / ORGANISM_CODE / "pwms_all_motifs"

CHIP_GROUND_TRUTH = DATA_DIR / "ground_truth_files" / "mESC_beeline_ChIP-seq.csv"
CHIP_GROUND_TRUTH_SEP = ","

PROCESSED_DATA = DATA_DIR / "processed"
TRAINING_DATA_CACHE = DATA_DIR / "training_data_cache"
RAW_DATA = DATA_DIR / "raw"
PKN_DIR = DATA_DIR / "prior_knowledge_network_data" / ORGANISM_CODE

# PKN files
STRING_DIR = DATA_DIR / "prior_knowledge_network_data" / ORGANISM_CODE / "STRING" 
TRRUST_DIR = DATA_DIR / "prior_knowledge_network_data" / ORGANISM_CODE / "TRRUST" 
KEGG_DIR = DATA_DIR / "prior_knowledge_network_data" / ORGANISM_CODE / "KEGG" 

COMMON_DATA = TRAINING_DATA_CACHE / "common"

EXPERIMENT_DIR = ROOT_DIR / "experiments"
OUTPUT_DIR = EXPERIMENT_DIR / DATASET_NAME

# Sample-specific paths
SAMPLE_PROCESSED_DATA_DIR = PROCESSED_DATA / DATASET_NAME
SAMPLE_DATA_CACHE_DIR = TRAINING_DATA_CACHE / DATASET_NAME

FINE_TUNING_DIR = OUTPUT_DIR / FINE_TUNING_TRAINED_MODEL

INFERRED_NETWORK_OUTPUT_DIR = ROOT_DIR / "output" / "transformer_testing_output" / "chrom_inferred_grn_orti_chipatlas_rn117_unique"





