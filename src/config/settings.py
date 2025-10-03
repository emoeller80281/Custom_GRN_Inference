from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# ----- SAMPLE INFORMATION -----
# Sample information
ORGANISM_CODE = "mm10"
SAMPLE_NAME = "mESC"
CHROM_ID = "chr19"
SAMPLE_NAMES = ["E7.5_rep1", "E7.75_rep1", "E8.0_rep2", "E8.5_rep2", "E8.75_rep2", "E7.5_rep2", "E8.0_rep1", "E8.5_rep1"]
RAW_SINGLE_CELL_DATA = Path("/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS")

# ----- DATA PREPARATION -----
# Pseudobulk and Data Preprocessing
NEIGHBORS_K = 20
LEIDEN_RESOLUTION = 1.0
AGGREGATION_METHOD = "mean" # "sum" or "mean"
RAW_10X_RNA_DATA_DIR = RAW_SINGLE_CELL_DATA / "DS014_DOI496239_MOUSE_ESC_RAW_FILES"
RAW_ATAC_PEAK_MATRIX_FILE = RAW_SINGLE_CELL_DATA / "DS014_DOI496239_MOUSE_ESCDAYS7AND8" / "scATAC_PeakMatrix.txt"

# Data Preprocessing and Caching
VALIDATION_DATASETS = ["E8.75_rep1"]
FORCE_RECALCULATE = True                # Recomputes genomic windows, peak-TG distance, and re-runs MOODS TF-peak scan
WINDOW_SIZE = 25_000                    # Aggregates peaks within WINDOW_SIZE bp genomic tiles
DISTANCE_SCALE_FACTOR = 25_000          # Weights the peak-gene TSS distance score. Lower numbers = faster dropoff
MAX_PEAK_DISTANCE = 100_000_000         # Masks out peaks further than this distance from the gene TSS
DIST_BIAS_MODE = "max"                  # Method for calcuting window -> gene TSS distance. Options: "max" | "sum" | "mean" | "logsumexp"
MOODS_PVAL_THRESHOLD = 1e-3             # MOODS binding significance threshold for associating a TF to a peak




# TF-peak binding calculation parameters
MOODS_PVAL_THRESHOLD=1e-3



# ----- EXPERIMENT VARIABLES -----




# ----- PATH SETUP -----


# Fixed Paths
DATA_DIR = ROOT_DIR / "data"
GENOME_DIR = DATA_DIR / "genome_data" / "reference_genome" / ORGANISM_CODE
CHROM_SIZES_FILE = GENOME_DIR / "chrom.sizes"
GENE_TSS_FILE = DATA_DIR / "genome_data" / "genome_annotation" / ORGANISM_CODE / "gene_tss.bed"
JASPAR_PFM_DIR = DATA_DIR / "motif_information" / "Jaspar" / "pfm_files"

PROCESSED_DATA = DATA_DIR / "processed"
TRAINING_DATA_CACHE = DATA_DIR / "training_data_cache"
RAW_DATA = DATA_DIR / "raw"

COMMON_DATA = TRAINING_DATA_CACHE / "common"

# Sample-specific paths
SAMPLE_PROCESSED_DATA_DIR = PROCESSED_DATA / SAMPLE_NAME
SAMPLE_DATA_CACHE_DIR = TRAINING_DATA_CACHE / SAMPLE_NAME
SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR = SAMPLE_DATA_CACHE_DIR / CHROM_ID
