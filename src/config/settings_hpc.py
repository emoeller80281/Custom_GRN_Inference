from pathlib import Path
import numpy as np
import itertools

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# HARDCODED PATHS -- CHANGE
PROJECT_DATA_DIR = Path("/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
PROJECT_RESULT_DIR = Path("/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")

chr_nums = [f"chr{i}" for i in range(1, 20)]
chrom_list = chr_nums #+ ["chrX", "chrY"]

# ----- SAMPLE INFORMATION -----
# Sample information
ORGANISM_CODE = "mm10"
DATASET_NAME = "mESC_filter_lowest_ten_pct"
CHROM_ID_LIST = chrom_list
CHROM_ID = "chr19"
CHROM_IDS = chr_nums
# , "E7.75_rep1", "E8.0_rep2", "E8.5_rep2", "E8.75_rep2", "E7.5_rep2", "E8.0_rep1", "E8.5_rep1"
SAMPLE_NAMES = ["E7.5_rep1", "E7.5_rep2", "E7.75_rep1", "E8.0_rep2", "E8.5_rep2", "E8.75_rep2", "E8.0_rep1", "E8.5_rep1"]
FINE_TUNING_DATASETS = ["E7.5_rep1", "E7.5_rep2", "E7.75_rep1", "E8.0_rep2", "E8.5_rep2", "E8.75_rep2", "E8.0_rep1", "E8.5_rep1"]

# Paths to the raw scRNA-seq and scATAC-seq data
RAW_SINGLE_CELL_DATA = Path("/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS")
RAW_10X_RNA_DATA_DIR = RAW_SINGLE_CELL_DATA / "DS014_DOI496239_MOUSE_ESC_RAW_FILES"
RAW_ATAC_PEAK_MATRIX_FILE = RAW_SINGLE_CELL_DATA / "DS014_DOI496239_MOUSE_ESCDAYS7AND8" / "scATAC_PeakMatrix.txt"

RAW_GSE218576_DIR= ROOT_DIR / "data/raw/GSE218576"
PROCESSED_GSE218576_DIR = ROOT_DIR / "data/processed/GSE218576"

# ----- DATA PREPARATION -----
# QC Filtering
MIN_GENES_PER_CELL = 200         # Minimum number of genes expressed per cell
MIN_PEAKS_PER_CELL = 200         # Minimum number of peaks expressed per cell

FILTER_TYPE = "count"              # Choose whether to filter cells by percent of cells expressing each gene or peak. Options are "pct" or "count"

FILTER_OUT_LOWEST_COUNTS_GENES = 3
FILTER_OUT_LOWEST_COUNTS_PEAKS = 3

FILTER_OUT_LOWEST_PCT_GENES = 0.1
FILTER_OUT_LOWEST_PCT_PEAKS = 0.1

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

NEIGHBORS_K = 30        # Number of nearest neighpers per cell in the KNN graph
PCA_COMPONENTS = 25     # Number of PCs per modality, controls variation
HOPS = 0                # Number of neighbors-of-neighbors to use (smooths across community)
SELF_WEIGHT = 1.0       # How much to weight the cells own gene expression. Higher = less blending with neighbors

# Data Preprocessing and Caching
VALIDATION_DATASETS = ["E8.75_rep1"]
FORCE_RECALCULATE = False                # Recomputes genomic windows, peak-TG distance, and re-runs sliding window TF-peak scan
WINDOW_SIZE = 1_000                     # Aggregates peaks within WINDOW_SIZE bp genomic tiles
DISTANCE_SCALE_FACTOR = 20_000           # Weights the peak-gene TSS distance score. Lower numbers = faster dropoff
MAX_PEAK_DISTANCE = 100_000              # Masks out peaks further than this distance from the gene TSS
DIST_BIAS_MODE = "logsumexp"            # Method for calcuting window -> gene TSS distance. Options: "max" | "sum" | "mean" | "logsumexp"
FILTER_TO_NEAREST_GENE = True           # Associate peaks to the nearest gene
PROMOTER_BP = None #10_000

# ----- MODEL TRAINING PARAMETERS -----
TOTAL_EPOCHS=250
BATCH_SIZE=16
PATIENCE=10
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
RESUME_CHECKPOINT_PATH=None #"/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC_small_neighborhood/chr19/model_training_001/checkpoint_14.pt"

GRAD_ACCUM_STEPS=2
USE_GRAD_ACCUMULATION=True
USE_GRAD_CHECKPOINTING=True

# Training scheduler settings
MODE="min"                      # min = improvement means a lower number; max = improvement means a higher number
INITIAL_LEARNING_RATE = 2.50e-4    # Initial learning rate for the model
SCHEDULER_FACTOR=0.25            # How much to reduce the learning rate on a plateau
SCHEDULER_PATIENCE=5            # How long to wait with no improvement without dropping the learning rate
THRESHOLD=1e-3                  # Defines how much better the next epoch has to be to count as being "better"
THRESHOLD_MODE="rel"            # rel helps filter noise for datasets with different loss scales. new best = previous best * (1 - threshold) difference
COOLDOWN=4                      # How many epochs to pause after a drop before testing for improvement, lets the training stabilize a bit
MIN_LR=2.5e-6                     # Wont drop the learning rate below this, prevents learning from stalling due to tiny lr

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
SHORTCUT_REG_WEIGHT = 0.0

# peak-TG distance bias
ATTN_BIAS_SCALE = 0.0 

# Sampling (optional)
SUBSAMPLE_MAX_TFS = None #400 # 150
SUBSAMPLE_MAX_TGS = None #2000 #None
SUBSAMPLE_MAX_WINDOWS_PER_CHROM = None #1000
SUBSAMPLE_MAX_CELLS = 10_000
SUBSAMPLE_SEED = 42

# ----- FINE TUNING ON SINGLE-CELL DATA -----
FINE_TUNING_TRAINED_MODEL = "chr19/model_training_003"
FINETUNE_EPOCHS = 250
FINETUNE_PATIENCE = 20
FINETUNE_SCHEDULER_PATIENCE = 10

FINETUNE_LR = 3e-04                 # smaller LR for refinement
FINETUNE_BATCH_SIZE = 320
FINETUNE_GRAD_ACCUM_STEPS = 1
FINETUNE_USE_GRAD_ACCUMULATION = True
FINETUNE_USE_GRAD_CHECKPOINTING = True
EWC_LAMBDA = 1e-5                       # Reduced from 2.5e-4 to allow model to adapt to single-cell data
MAX_STEPS = None                     # Maximum number of batches to process
FINETUNE_CORR_WEIGHT=0.2                # Increased from 0.05 to make R² penalty stronger
FINETUNE_EDGE_WEIGHT=0.0
FINETUNE_SHORTCUT_REG_WEIGHT=0.0
ZERO_EPS = 1e-6                     # Consider a value as zero if it is less than this
ZERO_WEIGHT = 0.5                   # Increased from 0.1 - don't ignore sparse data as heavily

PRESENCE_EPS = 0.0          # threshold for "non-zero" expression
BCE_WEIGHT   = 1.0
REG_WEIGHT   = 1.0

# Sample the max number of cells per sample
SUBSAMPLE_MAX_CELLS_FINETUNE = None


# ----- PATH SETUP -----
# Paths for specific data
DATABASE_DIR = ROOT_DIR / "data"
GENOME_DIR = DATABASE_DIR / "genome_data" / "reference_genome" / ORGANISM_CODE
CHROM_SIZES_FILE = GENOME_DIR / f"{ORGANISM_CODE}.chrom.sizes"
GTF_FILE_DIR = DATABASE_DIR / "genome_data" / "genome_annotation" / ORGANISM_CODE
NCBI_FILE_DIR = DATABASE_DIR / "genome_data" / "genome_annotation" / ORGANISM_CODE 
GENE_TSS_FILE = DATABASE_DIR / "genome_data" / "genome_annotation" / ORGANISM_CODE / "gene_tss.bed"
TF_FILE = DATABASE_DIR / "databases" / "motif_information" / ORGANISM_CODE / "TF_Information_all_motifs.txt"
JASPAR_PFM_DIR = DATABASE_DIR / "databases" / "motif_information" / "JASPAR" / "pfm_files"
MOTIF_DIR = DATABASE_DIR / "databases" / "motif_information" / ORGANISM_CODE / "pwms_all_motifs"

CHIP_GROUND_TRUTH = DATABASE_DIR / "ground_truth_files" / "mESC_beeline_ChIP-seq.csv"
CHIP_GROUND_TRUTH_SEP = ","

PROCESSED_DATA = DATABASE_DIR / "processed"
TRAINING_DATA_CACHE = DATABASE_DIR / "training_data_cache"
RAW_DATA = DATABASE_DIR / "raw"
PKN_DIR = DATABASE_DIR / "prior_knowledge_network_data" / ORGANISM_CODE

# PKN files
STRING_DIR = DATABASE_DIR / "prior_knowledge_network_data" / ORGANISM_CODE / "STRING" 
TRRUST_DIR = DATABASE_DIR / "prior_knowledge_network_data" / ORGANISM_CODE / "TRRUST" 
KEGG_DIR = DATABASE_DIR / "prior_knowledge_network_data" / ORGANISM_CODE / "KEGG" 

COMMON_DATA = TRAINING_DATA_CACHE / "common"

EXPERIMENT_DIR = PROJECT_DATA_DIR / "experiments"
OUTPUT_DIR = EXPERIMENT_DIR / DATASET_NAME

# Sample-specific paths
SAMPLE_PROCESSED_DATA_DIR = PROCESSED_DATA / DATASET_NAME
SAMPLE_DATA_CACHE_DIR = TRAINING_DATA_CACHE / DATASET_NAME

FINE_TUNING_DIR = OUTPUT_DIR / FINE_TUNING_TRAINED_MODEL

INFERRED_NETWORK_OUTPUT_DIR = ROOT_DIR / "output" / "transformer_testing_output" / "chrom_inferred_grn_orti_chipatlas_rn117_unique"





