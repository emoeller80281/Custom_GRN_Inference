#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem-per-cpu=16G
#SBATCH -o LOGS/find_tf_motifs.log
#SBATCH -e LOGS/find_tf_motifs.err

set -euo pipefail  # Strict error handling

# =========== SELECT WHICH PROCESSES TO RUN ================
CICERO_MAP_PEAKS_TO_TG=true
CREATE_HOMER_PEAK_FILE=false
HOMER_FIND_MOTIFS_GENOME=false
HOMER_ANNOTATE_PEAKS=false
PROCESS_MOTIF_FILES=false
PARSE_TF_PEAK_MOTIFS=false
CALCULATE_TF_REGULATION_SCORE=false

# ================= USER PATH VARIABLES ====================
BASE_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
INPUT_DIR="$BASE_DIR/input"
ATAC_DATA_FILE="$INPUT_DIR/macrophage_buffer1_filtered_ATAC.csv"
RNA_DATA_FILE="$INPUT_DIR/macrophage_buffer1_filtered_RNA.csv"
RDS_DATA_FILE="$INPUT_DIR/Macrophase_buffer1_filtered.rds"
CONDA_ENV_NAME="my_env"
PARALLEL_JOBS=32

# Other paths
PYTHON_SCRIPT_DIR="$BASE_DIR/src/python_scripts"
R_SCRIPT_DIR="$BASE_DIR/src/r_scripts"
OUTPUT_DIR="$BASE_DIR/output"
HOMER_DIR="$BASE_DIR/Homer/bin"
MOTIF_DIR="$OUTPUT_DIR/knownResults"
HOMER_PEAK_FILE="$INPUT_DIR/Homer_peaks.txt"
TF_MOTIF_BINDING_SCORE_FILE="$OUTPUT_DIR/total_motif_regulatory_scores.tsv"
PROCESSED_MOTIF_DIR="$OUTPUT_DIR/homer_tf_motif_scores"

# =================== FUNCTIONS ============================
# Function to check if at least one process is selected
check_pipeline_steps() {
    if ! $CICERO_MAP_PEAKS_TO_TG && ! $CREATE_HOMER_PEAK_FILE && ! $HOMER_FIND_MOTIFS_GENOME && \
       ! $HOMER_ANNOTATE_PEAKS && ! $PROCESS_MOTIF_FILES && ! $PARSE_TF_PEAK_MOTIFS && ! $CALCULATE_TF_REGULATION_SCORE; then
        echo "Error: At least one process must be enabled to run the pipeline."
        exit 1
    fi
}

# Function to validate required tools
check_tools() {
    local required_tools=(Rscript perl parallel python3 conda)
    for tool in "${required_tools[@]}"; do
        if ! command -v $tool &> /dev/null; then
            echo "Error: $tool is not installed or not in the PATH."
            exit 1
        fi
    done
}

# Function to validate input files
check_input_files() {
    if [ ! -f "$ATAC_DATA_FILE" ]; then echo "Error: ATAC data file not found at $ATAC_DATA_FILE"; exit 1; fi
    if [ ! -f "$RNA_DATA_FILE" ]; then echo "Error: RNA data file not found at $RNA_DATA_FILE"; exit 1; fi
    if [ ! -f "$RDS_DATA_FILE" ]; then echo "Error: RDS data file not found at $RDS_DATA_FILE"; exit 1; fi
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
    mkdir -p "$PROCESSED_MOTIF_DIR" "$OUTPUT_DIR"
    touch "$TF_MOTIF_BINDING_SCORE_FILE"
}

# ================= MAIN PIPELINE ==========================
# Perform validation
check_pipeline_steps
check_tools
check_input_files
activate_conda_env
setup_directories

# Execute pipeline steps
if [ "$CICERO_MAP_PEAKS_TO_TG" = true ]; then
    module load rstudio
    echo "Cicero: Mapping scATACseq peaks to target genes"
    Rscript "$R_SCRIPT_DIR/cicero.r" "$RDS_DATA_FILE" "$OUTPUT_DIR"
fi

if [ "$CREATE_HOMER_PEAK_FILE" = true ]; then
    echo "Python: Creating Homer peak file"
    python3 "$PYTHON_SCRIPT_DIR/Step010.create_homer_peak_file.py" --atac_data_file "$ATAC_DATA_FILE"
fi

if [ "$HOMER_FIND_MOTIFS_GENOME" = true ]; then
    echo "Homer: Running findMotifsGenome.pl"
    perl "$HOMER_DIR/findMotifsGenome.pl" "$HOMER_PEAK_FILE" "hg38" "$OUTPUT_DIR" -size 200
fi

if [ "$HOMER_ANNOTATE_PEAKS" = true ]; then
    echo "Homer: Running annotatePeaks.pl"
    perl "$HOMER_DIR/annotatePeaks.pl" "$HOMER_PEAK_FILE" "hg38" -m "$MOTIF_DIR/known1.motif" > "$OUTPUT_DIR/known_motif_1_motif_to_peak.txt"
fi

if [ "$PROCESS_MOTIF_FILES" = true ]; then
    echo "Python: Processing motif files in parallel"
    module load parallel
    find "$MOTIF_DIR" -name "*.motif" | parallel -j "$PARALLEL_JOBS" \
        "perl $HOMER_DIR/annotatePeaks.pl {} 'hg38' -m {} > $PROCESSED_MOTIF_DIR/{/.}_tf_motifs.txt"
fi

if [ "$PARSE_TF_PEAK_MOTIFS" = true ]; then
    echo "Python: Parsing TF binding motif results from Homer"
    python3 "$PYTHON_SCRIPT_DIR/Step020.parse_TF_peak_motifs.py" \
        --input_dir "$PROCESSED_MOTIF_DIR" --output_file "$TF_MOTIF_BINDING_SCORE_FILE" --cpu_count "$PARALLEL_JOBS"
fi

if [ "$CALCULATE_TF_REGULATION_SCORE" = true ]; then
    echo "Python: Calculating TF-TG regulatory potential"
    python3 "$PYTHON_SCRIPT_DIR/Step030.find_overlapping_TFs.py" \
        --rna_data_file "$RNA_DATA_FILE" --tf_motif_binding_score_file "$TF_MOTIF_BINDING_SCORE_FILE"
fi
