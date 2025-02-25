#!/bin/bash -l

#SBATCH --job-name custom_grn_method
#SBATCH --partition compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task 10
#SBATCH --mem 64G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

# =============================================
# SELECT WHICH PROCESSES TO RUN
# =============================================
STEP010_CICERO_MAP_PEAKS_TO_TG=true
STEP020_CICERO_PEAK_TO_TG_SCORE=true
STEP030_TF_TO_PEAK_SCORE=true
STEP040_TF_TO_TG_SCORE=true
STEP050_TRAIN_RANDOM_FOREST=true

# =============================================
# USER PATH VARIABLES
# =============================================
SAMPLE_NAME="mESC_full_test"
ORGANISM="mm10"
CONDA_ENV_NAME="my_env"

BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")

# Input file paths
INPUT_DIR="$BASE_DIR/input/mESC"
ATAC_DATA_FILE="$INPUT_DIR/mESC_filtered_L2_E7.5_rep2_ATAC.csv"
RNA_DATA_FILE="$INPUT_DIR/mESC_filtered_L2_E7.5_rep2_RNA.csv"
GROUND_TRUTH_FILE="$INPUT_DIR/RN111.tsv"

# Other paths
PYTHON_SCRIPT_DIR="$BASE_DIR/src/python_scripts"
R_SCRIPT_DIR="$BASE_DIR/src/r_scripts"
OUTPUT_DIR="$BASE_DIR/output/$SAMPLE_NAME"
REFERENCE_GENOME_DIR="$BASE_DIR/reference_genome/$ORGANISM"

TF_NAMES_FILE="$BASE_DIR/motif_information/$ORGANISM/TF_Information_all_motifs.txt"
MEME_DIR="$BASE_DIR/motif_information/$ORGANISM/${ORGANISM}_motif_meme_files"

# Sample-specific paths
CICERO_OUTPUT_FILE="$OUTPUT_DIR/peak_gene_associations.csv"

LOG_DIR="$BASE_DIR/LOGS/${SAMPLE_NAME}/"
FIG_DIR="$BASE_DIR/figures/$SAMPLE_NAME"

mkdir -p "${LOG_DIR}"

# Set output and error files dynamically
exec > "${LOG_DIR}/main_pipeline.log" 2> "${LOG_DIR}/main_pipeline.err"

# =============================================
# FUNCTIONS
# =============================================

# -------------- VALIDATION FUNCTIONS ----------------------

check_for_running_jobs() {
    echo "[INFO] Checking for running jobs with the same name..."
    if [ -z "${SLURM_JOB_NAME:-}" ]; then
        echo "    Not running in a SLURM environment, not checking for running tasks"
        return 0
    fi

    # Use the SLURM job name for comparison
    JOB_NAME="${SLURM_JOB_NAME:-custom_grn_method}"  # Dynamically retrieve the job name from SLURM

    # Check for running jobs with the same name, excluding the current job
    RUNNING_COUNT=$(squeue --name="$JOB_NAME" --noheader | wc -l)

    # If other jobs with the same name are running, exit
    if [ "$RUNNING_COUNT" -gt 1 ]; then
        echo "[WARNING] A job with the name '"$JOB_NAME"' is already running:"
        echo "    Exiting to avoid conflicts."
        exit 1
    
    # If no other jobs are running, pass
    else
        echo "    No other jobs with the name '"$JOB_NAME"'"
    fi
}

validate_critical_variables() {
    # Make sure that all of the required user variables are set
    local critical_vars=(
        SAMPLE_NAME \
        ORGANISM \
        BASE_DIR \
        RNA_DATA_FILE \
        ATAC_DATA_FILE \
        OUTPUT_DIR \
        LOG_DIR \
        TF_NAMES_FILE \
        CONDA_ENV_NAME
        )
    for var in "${critical_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            echo "[ERROR] Required variable $var is not set."
            exit 1
        fi
    done
}

validate_critical_variables

# Function to check if at least one process is selected
check_pipeline_steps() {
    if ! $STEP010_CICERO_MAP_PEAKS_TO_TG \
    && ! $STEP020_CICERO_PEAK_TO_TG_SCORE \
    && ! $STEP030_TF_TO_PEAK_SCORE \
    && ! $STEP040_TF_TO_TG_SCORE \
    && ! $STEP050_TRAIN_RANDOM_FOREST; then \
        echo "Error: At least one process must be enabled to run the pipeline."
        exit 1
    fi
}

# Function to validate required tools
check_tools() {
    local required_tools=(python3 conda)

    echo "[INFO] Validating required tools."
    for tool in "${required_tools[@]}"; do
        if ! command -v $tool &> /dev/null; then
            echo "[ERROR] $tool is not installed or not in the PATH."
            exit 1
        else
            echo "[INFO] $tool is available."
        fi
    done

    # Handle GNU parallel
    if ! command -v parallel &> /dev/null; then
        echo "[INFO] GNU parallel not found in PATH. Attempting to load module..."
        if command -v module &> /dev/null; then
            module load parallel || {
                echo "[ERROR] Failed to load GNU parallel using module."
                exit 1
            }
            echo "[INFO] GNU parallel module loaded successfully."
        else
            echo "[ERROR] Module command not available. GNU parallel is required."
            exit 1
        fi
    else
        echo "[INFO] GNU parallel is available."
    fi
}

determine_num_cpus() {
    if [ -z "${SLURM_CPUS_PER_TASK:-}" ]; then
        if command -v nproc &> /dev/null; then
            TOTAL_CPUS=$(nproc --all)
            case $TOTAL_CPUS in
                [1-15]) IGNORED_CPUS=1 ;;  # Reserve 1 CPU for <=15 cores
                [16-31]) IGNORED_CPUS=2 ;; # Reserve 2 CPUs for <=31 cores
                *) IGNORED_CPUS=4 ;;       # Reserve 4 CPUs for >=32 cores
            esac
            NUM_CPU=$((TOTAL_CPUS - IGNORED_CPUS))
            echo "[INFO] Running locally. Detected $TOTAL_CPUS CPUs, reserving $IGNORED_CPUS for system tasks. Using $NUM_CPU CPUs."
        else
            NUM_CPU=1  # Fallback
            echo "[INFO] Running locally. Unable to detect CPUs, defaulting to $NUM_CPU CPU."
        fi
    else
        NUM_CPU=${SLURM_CPUS_PER_TASK}
        echo "[INFO] Running on SLURM. Number of CPUs allocated: ${NUM_CPU}"
    fi
}

# Function to validate input files
check_input_files() {
    echo "[INFO] Validating input files."
    local files=("$ATAC_DATA_FILE" "$RNA_DATA_FILE")
    for file in "${files[@]}"; do
        if [ ! -f "$file" ]; then
            echo "[ERROR] File not found: $file"
            exit 1
        elif [ ! -r "$file" ]; then
            echo "[ERROR] File is not readable: $file"
            exit 1
        fi
    done
    echo "[INFO] Input files validated successfully."
}

# Function to activate Conda environment
activate_conda_env() {
    CONDA_BASE=$(conda info --base)
    if [ -z "$CONDA_BASE" ]; then
        echo "Error: Conda base could not be determined. Is Conda installed and in your PATH?"
        exit 1
    fi

    source "$CONDA_BASE/bin/activate"
    if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
        echo "Error: Conda environment '$CONDA_ENV_NAME' does not exist."
        exit 1
    fi

    conda activate "$CONDA_ENV_NAME" || { echo "Error: Failed to activate Conda environment '$CONDA_ENV_NAME'."; exit 1; }
    echo "Activated Conda environment: $CONDA_ENV_NAME"
}

# Function to ensure required directories exist
setup_directories() {
    echo "[INFO] Ensuring required directories exist."
    local dirs=( 
        "$INPUT_DIR" \
        "$OUTPUT_DIR" \
        "$LOG_DIR" \
        "$FIG_DIR" \
        )

    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
    echo "    Required directories created."
}

check_r_environment() {
    REQUIRED_R_VERSION="4.3.2"  # Replace with your required version
    echo "    Checking R environment..."

    # Check if the 'module' command exists
    if ! command -v module &> /dev/null; then
        echo "        [ERROR] 'module' command is not available. Ensure the environment module system is installed."
        exit 1
    fi

    # Check if the 'rstudio' module is available
    if ! module avail rstudio &> /dev/null; then
        echo "        [ERROR] 'rstudio' module is not available. Check your module system."
        exit 1
    fi

    # Load the 'rstudio' module
    module load rstudio
    if [ $? -ne 0 ]; then
        echo "        [ERROR] Failed to load 'rstudio' module."
        exit 1
    else
        echo "        [INFO] Successfully loaded 'rstudio' module."
    fi

    # Check R version
    if ! command -v R &> /dev/null; then
        echo "        [ERROR] R is not installed. Please install R version $REQUIRED_R_VERSION or later."
        exit 1
    fi

    # Check if the installed version of R is different
    INSTALLED_R_VERSION=$(R --version | grep -oP "(?<=R version )\d+\.\d+\.\d+" | head -1)
    if [[ "$(printf '%s\n' "$REQUIRED_R_VERSION" "$INSTALLED_R_VERSION" | sort -V | head -1)" != "$REQUIRED_R_VERSION" ]]; then
        echo "        [ERROR] Installed R version ($INSTALLED_R_VERSION) is older than required ($REQUIRED_R_VERSION). Please update R."
        exit 1
    fi
    echo "        R version $INSTALLED_R_VERSION is installed."

    # Check for required R packages
    Rscript $R_SCRIPT_DIR/check_dependencies.r
    echo ""
}

download_file_if_missing() {
    local file_path=$1
    local file_url=$2
    local file_description=$3

    if [ ! -f "$file_path" ]; then
        echo "    $file_description not found, downloading..."
        curl -s -o "$file_path" "$file_url"

        if [ $? -ne 0 ] || [ ! -s "$file_path" ]; then
            echo "[ERROR] Failed to download or validate $file_description from $file_url."
            exit 1
        else
            echo "        Successfully downloaded $file_description"
        fi
    else
        echo "        Using existing $file_description"
    fi
}

check_cicero_genome_files_exist() {

    if [ "$ORGANISM" == "mm10" ]; then
        echo "    $ORGANISM detected, using mouse genome"

        CHROM_SIZES="$INPUT_DIR/mm10.chrom.sizes"
        CHROM_SIZES_URL="https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes"

        GENE_ANNOT="$INPUT_DIR/Mus_musculus.GRCm39.113.gtf.gz"
        GENE_ANNOT_URL="https://ftp.ensembl.org/pub/release-113/gtf/mus_musculus/Mus_musculus.GRCm39.113.gtf.gz"
    
    elif [ "$ORGANISM" == "hg38" ]; then
        echo "    $ORGANISM detected, using human genome"

        CHROM_SIZES="$INPUT_DIR/hg38.chrom.sizes"
        CHROM_SIZES_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes"

        GENE_ANNOT="$INPUT_DIR/Homo_sapiens.GRCh38.113.gtf.gz"
        GENE_ANNOT_URL="https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/Homo_sapiens.GRCh38.113.gtf.gz"

    else
        echo "    [ERROR] Unsupported ORGANISM: $ORGANISM"
        exit 1
    fi

    # Check chromosome sizes and gene annotation files
    download_file_if_missing "$CHROM_SIZES" "$CHROM_SIZES_URL" "$ORGANISM chromosome sizes file"
    download_file_if_missing "$GENE_ANNOT" "$GENE_ANNOT_URL" "$ORGANISM gene annotation file"
}

# -------------- MAIN PIPELINE FUNCTIONS --------------
run_cicero() {
    echo ""
    echo "Cicero: Mapping scATACseq peaks to target genes"

    # Validate variables
    if [[ -z "$R_SCRIPT_DIR" || -z "$ATAC_DATA_FILE" || -z "$OUTPUT_DIR" || -z "$LOG_DIR" ]]; then
        echo "[ERROR] One or more required variables (R_SCRIPT_DIR, ATAC_DATA_FILE, OUTPUT_DIR, LOG_DIR) are not set."
        exit 1
    fi

    # Ensure log directory exists
    mkdir -p "$LOG_DIR"

    # Check R environment
    check_r_environment

    # Check for the chomosome size and gene annotation files in INPUT_DIR
    check_cicero_genome_files_exist

    # Load R module (optional, for HPC systems)
    if command -v module &> /dev/null; then
        module load rstudio
    fi

    echo "    Checks complete, running Cicero"

    /usr/bin/time -v \
    Rscript "$R_SCRIPT_DIR/Step010.run_cicero.r" \
        "$ATAC_DATA_FILE" \
        "$OUTPUT_DIR" \
        "$CHROM_SIZES" \
        "$GENE_ANNOT" 
    
} 2> "$LOG_DIR/Step010.run_cicero.log"

run_cicero_peak_to_tg_score() {
    echo ""
    echo "Python: Parsing Cicero peak to TG scores"
    /usr/bin/time -v \
    python3 "$PYTHON_SCRIPT_DIR/Step020.cicero_peak_to_tg_score.py" \
        --fig_dir "$FIG_DIR" \
        --output_dir "$OUTPUT_DIR" 
    
} 2> "$LOG_DIR/Step020.cicero_peak_to_tg_score.log"

run_tf_to_peak_score() {
    echo ""
    echo "Python: Calculating TF to peak scores"
    /usr/bin/time -v \
    python3 "$PYTHON_SCRIPT_DIR/Step030.tf_to_peak_score.py" \
        --tf_names_file "$TF_NAMES_FILE"\
        --meme_dir "$MEME_DIR"\
        --reference_genome_dir "$REFERENCE_GENOME_DIR"\
        --atac_data_file "$ATAC_DATA_FILE" \
        --rna_data_file "$RNA_DATA_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --num_cpu "$NUM_CPU" 
    
} 2> "$LOG_DIR/Step030.tf_to_peak_score.log"

run_tf_to_tg_score() {
    echo ""
    echo "Python: Calculating TF to TG scores"
    /usr/bin/time -v \
    python3 "$PYTHON_SCRIPT_DIR/Step040.tf_to_tg_score.py" \
        --rna_data_file "$RNA_DATA_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --fig_dir "$FIG_DIR" 
    
} 2> "$LOG_DIR/Step040.tf_to_tg_score.log"

run_random_forest_training() {
    echo ""
    echo "Python: Training Random Forest"
    /usr/bin/time -v \
    python3 "$PYTHON_SCRIPT_DIR/Step050.train_random_forest.py" \
        --ground_truth_file "$GROUND_TRUTH_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --fig_dir "$FIG_DIR" 

} 2> "$LOG_DIR/Step050.train_random_forest.log"


# =============================================
# MAIN PIPELINE
# =============================================

# Help option
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash main_pipeline.sh"
    echo "This script executes a single-cell GRN inference pipeline."
    echo "Modify the flags at the top of the script to enable/disable steps."
    exit 0
fi

# Perform validation
check_for_running_jobs
check_pipeline_steps
check_tools
determine_num_cpus
check_input_files
activate_conda_env
setup_directories

# Execute selected pipeline steps
if [ "$STEP010_CICERO_MAP_PEAKS_TO_TG" = true ]; then run_cicero; fi
if [ "$STEP020_CICERO_PEAK_TO_TG_SCORE" = true ]; then run_cicero_peak_to_tg_score; fi
if [ "$STEP030_TF_TO_PEAK_SCORE" = true ]; then run_tf_to_peak_score; fi
if [ "$STEP040_TF_TO_TG_SCORE" = true ]; then run_tf_to_tg_score; fi
if [ "$STEP050_TRAIN_RANDOM_FOREST" = true ]; then run_random_forest_training; fi