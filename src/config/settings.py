from pathlib import Path
import numpy as np
import itertools

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

chr_nums = [f"chr{i}" for i in range(1, 20)]
chrom_list = chr_nums + ["chrX", "chrY"]

# ----- SAMPLE INFORMATION -----
# Sample information
ORGANISM_CODE = "mm10"
DATASET_NAME = "DS012_mESC"
CHROM_ID_LIST = chrom_list
CHROM_ID = "chr1"
# , "E7.75_rep1", "E8.0_rep2", "E8.5_rep2", "E8.75_rep2", "E7.5_rep2", "E8.0_rep1", "E8.5_rep1"
SAMPLE_NAMES = ["DS012_sample"]
FINE_TUNING_DATASETS = ["DS012_sample"]

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

NEIGHBORS_K = 20
LEIDEN_RESOLUTION = 1.0
AGGREGATION_METHOD = "mean" # "sum" or "mean"

# Data Preprocessing and Caching
VALIDATION_DATASETS = ["E8.75_rep1"]
FORCE_RECALCULATE = True                # Recomputes genomic windows, peak-TG distance, and re-runs MOODS TF-peak scan
WINDOW_SIZE = 25_000                    # Aggregates peaks within WINDOW_SIZE bp genomic tiles
DISTANCE_SCALE_FACTOR = 250_000         # Weights the peak-gene TSS distance score. Lower numbers = faster dropoff
MAX_PEAK_DISTANCE = 1_000_000           # Masks out peaks further than this distance from the gene TSS
DIST_BIAS_MODE = "mean"                 # Method for calcuting window -> gene TSS distance. Options: "max" | "sum" | "mean" | "logsumexp"

# ----- MODEL TRAINING PARAMETERS -----
TOTAL_EPOCHS=500
BATCH_SIZE=32
PATIENCE=15
CORR_LOSS_WEIGHT=0.5

D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 3
D_FF = 768
DROPOUT = 0.1

INITIAL_LEARNING_RATE = 1e-3
SCHEDULER_FACTOR=0.5
SCHEDULER_PATIENCE=10

# TF to TG shortcut parameters
USE_DISTANCE_BIAS = True
USE_SHORTCUT = True
USE_MOTIF_MASK = False
SHORTCUT_L1 = 0
SHORTCUT_L2 = 0
SHORTCUT_TOPK = None
SHORTCUT_DROPOUT = 0

# peak-TG distance bias
ATTN_BIAS_SCALE = 1.0 

# ----- FINE TUNING ON SINGLE-CELL DATA -----
FINE_TUNING_TRAINED_MODEL = "model_training_014"
FINETUNE_PATIENCE = 20
FINETUNE_LR = 1e-4       # smaller LR for refinement
EWC_LAMBDA = 10.0

# ----- PATH SETUP -----
# Fixed Paths
DATA_DIR = ROOT_DIR / "data"
GENOME_DIR = DATA_DIR / "genome_data" / "reference_genome" / ORGANISM_CODE
CHROM_SIZES_FILE = GENOME_DIR / "chrom.sizes"
GTF_FILE_DIR = DATA_DIR / "genome_data" / "genome_annotation" / ORGANISM_CODE
NCBI_FILE_DIR = DATA_DIR / "genome_data" / "genome_annotation" / ORGANISM_CODE 
GENE_TSS_FILE = DATA_DIR / "genome_data" / "genome_annotation" / ORGANISM_CODE / "gene_tss.bed"
TF_FILE = DATA_DIR / "databases" / "motif_information" / ORGANISM_CODE / "TF_Information_all_motifs.txt"
JASPAR_PFM_DIR = DATA_DIR / "databases" / "motif_information" / "JASPAR" / "pfm_files"
MOTIF_DIR = DATA_DIR / "databases" / "motif_information" / ORGANISM_CODE / "pwms_all_motifs"

CHIP_GT_PATH = DATA_DIR / "ground_truth_files" / "mESC_beeline_ChIP-seq.csv"
CHIP_GT_SEP = ","

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



