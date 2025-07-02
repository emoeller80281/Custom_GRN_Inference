#!/bin/bash -l

#SBATCH --partition compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

# =============================================
#        SELECT PIPELINE STEPS TO RUN
# =============================================
# Run the peak to TG regulatory potential calculation methods
CICERO_MAP_PEAKS_TO_TG=false
CICERO_PEAK_TO_TG_SCORE=false

PEAK_TO_TG_CORRELATION=false

# Run the TF to peak binding score calculation methods
SLIDING_WINDOW_TF_TO_PEAK_SCORE=false
HOMER_TF_TO_PEAK_SCORE=true

# Combine the score DataFrames
COMBINE_DATAFRAMES=false
SUBSAMPLE_PERCENT=50 # Percent of rows of the combined dataframe to subsample

# Train a predictive model to infer the GRN
TRAIN_XGBOOST_CLASSIFIER=false

# =============================================
#              USER PATH VARIABLES
# =============================================
CONDA_ENV_NAME="my_env"

BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")

# Input file paths
INPUT_DIR="${BASE_DIR}/input/${CELL_TYPE}/${SAMPLE_NAME}"
RNA_FILE_NAME="$INPUT_DIR/$RNA_FILE_NAME"
ATAC_FILE_NAME="$INPUT_DIR/$ATAC_FILE_NAME"

# Other paths
PYTHON_SCRIPT_DIR="$BASE_DIR/src/grn_inference"
R_SCRIPT_DIR="$BASE_DIR/pipeline_scripts/r_scripts"
OUTPUT_DIR="$BASE_DIR/output/$CELL_TYPE/$SAMPLE_NAME"
REFERENCE_GENOME_DIR="$BASE_DIR/data/reference_genome/$SPECIES"

INFERRED_GRN_DIR="$OUTPUT_DIR/inferred_grns"
TRAINED_MODEL_DIR="$OUTPUT_DIR/trained_models"

# Name of the inferrred network file with STRING PPI interaction columns
INFERRED_NET_FILE="$INFERRED_GRN_DIR/inferred_score_df.parquet"

# ----- Resource / Database files -----
STRING_DB_DIR="$BASE_DIR/data/string_database/$SPECIES/"
# ENHANCERDB_FILE="$BASE_DIR/enhancer_db/enhancer" # Deprecated
TF_NAMES_FILE="$BASE_DIR/data/motif_information/$SPECIES/TF_Information_all_motifs.txt"
MEME_DIR="$BASE_DIR/data/motif_information/$SPECIES/${SPECIES}_motif_meme_files"

# LOG_DIR="$BASE_DIR/LOGS/${SAMPLE_NAME}/"
FIG_DIR="$BASE_DIR/figures/$SPECIES/$SAMPLE_NAME"
mkdir -p "${FIG_DIR}"

LOG_DIR="${BASE_DIR}/LOGS/${CELL_TYPE}_logs/${SAMPLE_NAME}_logs/"

if [ "$SPECIES" == "human" ]; then
    SPECIES="hg38"
elif [ "$SPECIES" == "mouse" ]; then
    SPECIES="mm10"
fi

echo "[INFO] Input files:"
echo "    - RNA Data File: $RNA_FILE_NAME"
echo "    - ATAC Data File: $ATAC_FILE_NAME"
echo "    - Cell Type: $CELL_TYPE"
echo "    - Sample: $SAMPLE_NAME"
echo "    - Species: $SPECIES"
echo ""

# =============================================
#                   FUNCTIONS
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
        echo "    - Exiting to avoid conflicts."
        exit 1
    
    # If no other jobs are running, pass
    else
        echo "    - No other jobs with the name '"$JOB_NAME"'"
    fi
}

validate_critical_variables() {
    # Make sure that all of the required user variables are set
    local critical_vars=(
        SAMPLE_NAME \
        SPECIES \
        BASE_DIR \
        RNA_FILE_NAME \
        ATAC_FILE_NAME \
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

check_pipeline_steps() {
    echo ""
    steps=(
        "CICERO_MAP_PEAKS_TO_TG"
        "CICERO_PEAK_TO_TG_SCORE"
        "PEAK_TO_TG_CORRELATION"
        "SLIDING_WINDOW_TF_TO_PEAK_SCORE"
        "HOMER_TF_TO_PEAK_SCORE"
        "COMBINE_DATAFRAMES"
        "TRAIN_XGBOOST_CLASSIFIER"
    )

    enabled=0
    echo "[INFO] Enabled pipeline steps:"

    # Echo which steps are enabled
    for step in "${steps[@]}"; do
        if ${!step}; then
            echo "    - $step"
            enabled=1
        fi
    done

    # If no steps are enabled, output an error and exit.
    if [ $enabled -eq 0 ]; then
        echo ""
        echo "[ERROR] At least one process must be enabled to run the pipeline."
        exit 1
    fi
    echo ""
}

check_tools() {
    local required_tools=(python3 conda)

    echo "[INFO] Validating required tools."
    for tool in "${required_tools[@]}"; do
        if ! command -v $tool &> /dev/null; then
            echo "[ERROR] $tool is not installed or not in the PATH."
            exit 1
        else
            echo "    - $tool is available."
        fi
    done
}

load_parallel() {
    # Handle GNU parallel
    echo ""
    echo "[INFO] Checking for the GNU parallel module for running commands in parallel"
    if ! command -v parallel &> /dev/null; then
        echo "    - GNU parallel not found in PATH. Attempting to load module..."
        if command -v module &> /dev/null; then
            module load parallel || {
                echo "[ERROR] Failed to load GNU parallel using module."
                exit 1
            }
            echo "    - GNU parallel module loaded successfully."
        else
            echo "[ERROR] Module command not available. GNU parallel is required."
            exit 1
        fi
    else
        echo "    - GNU parallel is available."
    fi
}

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

check_input_files() {
    echo ""
    echo "[INFO] Validating input files exist..."
    local files=("$ATAC_FILE_NAME" "$RNA_FILE_NAME")
    for file in "${files[@]}"; do
        if [ ! -f "$file" ]; then
            echo "[ERROR] File not found: $file"
            exit 1
        else
            echo "    - $file exists"
        fi
        if [ ! -r "$file" ]; then
            echo "[ERROR] File is not readable: $file"
            exit 1
        fi
    done
}

activate_conda_env() {
    echo ""
    echo "[INFO] Attempting to load the specified Conda module"
    CONDA_BASE=$(conda info --base)
    if [ -z "$CONDA_BASE" ]; then
        echo ""
        echo "[ERROR] Conda base could not be determined. Is Conda installed and in your PATH?"
        exit 1
    fi

    source "$CONDA_BASE/bin/activate"
    if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
        echo ""
        echo "[ERROR] Conda environment '$CONDA_ENV_NAME' does not exist."
        exit 1
    fi

    conda activate "$CONDA_ENV_NAME" || { echo "Error: Failed to activate Conda environment '$CONDA_ENV_NAME'."; exit 1; }
    echo "    - Successfully activated Conda environment: $CONDA_ENV_NAME"
}

setup_directories() {
    echo ""
    echo "[INFO] Ensuring required directories exist."
    local dirs=( 
        "$INPUT_DIR" \
        "$OUTPUT_DIR" \
        "$LOG_DIR" \
        "$FIG_DIR" \
        "$INFERRED_GRN_DIR" \
        "$TRAINED_MODEL_DIR"
        )

    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        echo "    - $dir"
    done
}

check_r_environment() {
    echo "    [INFO] Checking R environment..."

    # Check R version
    if ! command -v R &> /dev/null; then
        echo "        [ERROR] R is not installed. Please install R version $REQUIRED_R_VERSION or later."
        exit 1
    fi

    # Check if the installed version of R is different
    INSTALLED_R_VERSION=$(R --version | grep -oP "(?<=R version )\d+\.\d+\.\d+" | head -1)

    echo "        R version $INSTALLED_R_VERSION is installed."

}

download_file_if_missing() {
    local file_path="$1"
    local file_url="$2"
    local file_description="$3"

    if [ ! -f "$file_path" ]; then
        echo "    - $file_description not found, downloading..."
        curl -s -o "$file_path" "$file_url"

        if [ $? -ne 0 ] || [ ! -s "$file_path" ]; then
            echo "[ERROR] Failed to download or validate $file_description from $file_url."
            exit 1
        else
            echo "        - Successfully downloaded $file_description"
        fi
    else
        echo "         - Using existing $file_description"
    fi
}

check_cicero_genome_files_exist() {

    if [ "$SPECIES" == "mm10" ]; then
        echo "    - $SPECIES detected, using mouse genome"

        CHROM_SIZES="$REFERENCE_GENOME_DIR/mm10.chrom.sizes"
        CHROM_SIZES_URL="https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes"

        GENE_ANNOT="$REFERENCE_GENOME_DIR/Mus_musculus.GRCm39.113.gtf.gz"
        GENE_ANNOT_URL="https://ftp.ensembl.org/pub/release-113/gtf/mus_musculus/Mus_musculus.GRCm39.113.gtf.gz"
    
    elif [ "$SPECIES" == "hg38" ]; then
        echo "    - $SPECIES detected, using human genome"

        CHROM_SIZES="$REFERENCE_GENOME_DIR/hg38.chrom.sizes"
        CHROM_SIZES_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes"

        GENE_ANNOT="$REFERENCE_GENOME_DIR/Homo_sapiens.GRCh38.113.gtf.gz"
        GENE_ANNOT_URL="https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/Homo_sapiens.GRCh38.113.gtf.gz"

    else
        echo ""
        echo "[ERROR] Unsupported SPECIES: $SPECIES"
        exit 1
    fi

    # Check chromosome sizes and gene annotation files
    download_file_if_missing "$CHROM_SIZES" "$CHROM_SIZES_URL" "$SPECIES chromosome sizes file"
    download_file_if_missing "$GENE_ANNOT" "$GENE_ANNOT_URL" "$SPECIES gene annotation file"
}

run_dataset_preprocessing() {
    echo ""
    echo "Python: Checking if the scATAC-seq and scRNA-seq data is normalized or raw counts"
    /usr/bin/time -v \
    poetry run python "$PYTHON_SCRIPT_DIR/pipeline/preprocess_datasets.py" \
        --atac_data_file "$ATAC_FILE_NAME" \
        --rna_data_file "$RNA_FILE_NAME" \
        --species "$SPECIES" \
        --output_dir "$OUTPUT_DIR"
    
    # After preprocessing, update the file names to the new processed files.
    # This assumes that the processed file name is constructed by replacing
    # the original file extension with _processed.tsv.
    new_atac_file="$(dirname "$ATAC_FILE_NAME")/$(basename "$ATAC_FILE_NAME" | sed 's/\.[^.]*$/_processed.parquet/')"
    new_rna_file="$(dirname "$RNA_FILE_NAME")/$(basename "$RNA_FILE_NAME" | sed 's/\.[^.]*$/_processed.parquet/')"
    
    ATAC_FILE_NAME="$new_atac_file"
    RNA_FILE_NAME="$new_rna_file"
    
} 2> "$LOG_DIR/dataset_preprocessing.log"

check_processed_files() {
    echo ""

    # Derive the expected processed filenames from the original file names.
    new_atac_file="$(dirname "$ATAC_FILE_NAME")/$(basename "$ATAC_FILE_NAME" | sed 's/\.[^.]*$/_processed.parquet/')"
    new_rna_file="$(dirname "$RNA_FILE_NAME")/$(basename "$RNA_FILE_NAME" | sed 's/\.[^.]*$/_processed.parquet/')"

    if [ -f "$new_atac_file" ] && [ -f "$new_rna_file" ]; then
        echo "[INFO] Processed files found:"
        echo "    - ATAC: $new_atac_file"
        echo "    - RNA:  $new_rna_file"
        # Update global variables so downstream steps use the processed files
        ATAC_FILE_NAME="$new_atac_file"
        RNA_FILE_NAME="$new_rna_file"
    else
        echo "[INFO] Processed files not found. Running dataset preprocessing..."
        run_dataset_preprocessing
    fi
}

check_string_db_files() {
    echo ""
    echo "[INFO] Checking for STRING database protein_info and protein_links_detailed files for $SPECIES"

    if [ ! -d "$STRING_DB_DIR" ]; then
        echo "[WARNING] STRING database directory missing for $SPECIES, creating and downloading files..."
        mkdir -p "$STRING_DB_DIR"
    fi

    if [ ! -f "$STRING_DB_DIR/protein_info.txt" ]; then
        if [ "$SPECIES" == "mm10" ]; then
            STRING_ORG_CODE="10090"
        elif [ "$SPECIES" == "hg38" ]; then
            STRING_ORG_CODE="9606"
        else
            echo "[ERROR] $SPECIES is not valid, please specify either mm10 or hg38"
            exit 1
        fi

        PROTEIN_INFO_FILE_PATH="$STRING_DB_DIR/${STRING_ORG_CODE}.protein.info.v12.0.txt.gz"
        PROTEIN_INFO_FILE_URL="https://stringdb-downloads.org/download/protein.info.v12.0/${STRING_ORG_CODE}.protein.info.v12.0.txt.gz"
        PROTEIN_INFO_DESCRIPTION="$SPECIES STRING database info file"

        download_file_if_missing "$PROTEIN_INFO_FILE_PATH" "$PROTEIN_INFO_FILE_URL" "$PROTEIN_INFO_DESCRIPTION"

        echo "    - Downloaded! Unzipping gunzip file"
        gunzip -f "$PROTEIN_INFO_FILE_PATH"

        echo "    - Renaming file to protein_info.txt"
        mv "$STRING_DB_DIR/${STRING_ORG_CODE}.protein.info.v12.0.txt" "$STRING_DB_DIR/protein_info.txt"
    else
        echo "    - STRING protein_info.txt file found"
    fi

    if [ ! -f "$STRING_DB_DIR/protein_links_detailed.txt" ]; then
        if [ "$SPECIES" == "mm10" ]; then
            STRING_ORG_CODE="10090"
        elif [ "$SPECIES" == "hg38" ]; then
            STRING_ORG_CODE="9606"
        else
            echo "[ERROR] $SPECIES is not valid, please specify either mm10 or hg38"
            exit 1
        fi

        PROTEIN_LINKS_FILE_PATH="$STRING_DB_DIR/${STRING_ORG_CODE}.protein.links.detailed.v12.0.txt.gz"
        PROTEIN_LINKS_FILE_URL="https://stringdb-downloads.org/download/protein.links.detailed.v12.0/${STRING_ORG_CODE}.protein.links.detailed.v12.0.txt.gz"
        PROTEIN_LINKS_DESCRIPTION="$SPECIES STRING database detailed protein-protein link file"

        download_file_if_missing "$PROTEIN_LINKS_FILE_PATH" "$PROTEIN_LINKS_FILE_URL" "$PROTEIN_LINKS_DESCRIPTION"

        echo "    - Downloaded! Unzipping gunzip file"
        gunzip -f "$PROTEIN_LINKS_FILE_PATH"

        echo "    - Renaming file to protein_links_detailed.txt"
        mv "$STRING_DB_DIR/${STRING_ORG_CODE}.protein.links.detailed.v12.0.txt" "$STRING_DB_DIR/protein_links_detailed.txt"
    else
        echo "    - STRING protein_links_detailed.txt file found"
    fi
}


# -------------- HOMER FUNCTIONS --------------
install_homer() {
    mkdir -p "$BASE_DIR/data/homer"
    wget "http://homer.ucsd.edu/homer/configureHomer.pl" -P "$BASE_DIR/data/homer"
    perl "$BASE_DIR/data/homer/configureHomer.pl" -install
} 2> "$LOG_DIR/Homer_logs/01.install_homer.log"

install_homer_species_genome() {
    perl "$BASE_DIR/data/homer/configureHomer.pl" -install "$SPECIES"
} 2> "$LOG_DIR/Homer_logs/02.install_homer_species.log"

create_homer_peak_file() {
    /usr/bin/time -v \
    poetry run python "$PYTHON_SCRIPT_DIR/create_homer_peak_file.py" \
        --output_dir "$OUTPUT_DIR"
} 2> "$LOG_DIR/Homer_logs/03.create_homer_peak_file.log"

homer_find_motifs() {
    mkdir -p "$OUTPUT_DIR/homer_results"
    perl "$BASE_DIR/data/homer/bin/findMotifsGenome.pl" "$OUTPUT_DIR/tmp/homer_peaks.txt" "$SPECIES" "$OUTPUT_DIR/homer_results/" -size 200 -p $NUM_CPU
    echo "    Done!"
} 2> "$LOG_DIR/Homer_logs/04.homer_findMotifsGenome.log"

homer_process_motif_files() {
    echo ""
    echo "----- Homer annotatePeaks.pl -----"
    echo "[INFO] Starting motif file processing"

    load_parallel

    # Look through the knownResults dir of the Homer findMotifsGenome.pl output
    MOTIF_DIR="$OUTPUT_DIR/homer_results/knownResults"
    # Detect files to process
    motif_files=$(find "$MOTIF_DIR" -name "*.motif")
    if [ -z "$motif_files" ]; then
        echo "[ERROR] No motif files found in $MOTIF_DIR."
        exit 1
    fi

    # Log number of files to process
    file_count=$(echo "$motif_files" | wc -l)
    echo "[INFO] Found $file_count motif files to process."

    
    # Create output directory if it doesn't exist
    PROCESSED_MOTIF_DIR="$OUTPUT_DIR/homer_results/homer_tf_motif_scores"
    mkdir -p "$PROCESSED_MOTIF_DIR"

    # Process files in parallel
    if [ "$use_parallel" = true ]; then
        echo "$motif_files" | /usr/bin/time -v parallel -j "$NUM_CPU" \
            "perl $BASE_DIR/data/homer/bin/annotatePeaks.pl $OUTPUT_DIR/homer_peaks.txt '$SPECIES' -m {} > $PROCESSED_MOTIF_DIR/{/}_tf_motifs.txt"

        module unload parallel

    # Process files sequentially
    else
        for file in $motif_files; do
            local output_file="$PROCESSED_MOTIF_DIR/$(basename "$file" .motif)_tf_motifs.txt"
            /usr/bin/time -v \
            "perl $BASE_DIR/data/homer/bin/annotatePeaks.pl $OUTPUT_DIR/homer_peaks.txt '$SPECIES' -m $file > $output_file" \

            if [ $? -ne 0 ]; then
                echo "[ERROR] Failed to process motif file: $file" >> "$LOG_DIR/step05_sequential.err"
            else
                echo "[INFO] Successfully processed: $file"
            fi
        done
    fi

    # Check for errors
    if [ $? -ne 0 ]; then
        echo "[ERROR] Motif file processing failed. Check logs in $LOG_DIR for details."
        exit 1
    fi

    echo "[INFO] Motif file processing completed successfully."
    echo ""

} 2> "$LOG_DIR/Homer_logs/05.homer_annotatePeaks.log"

# -------------- MAIN PIPELINE FUNCTIONS --------------
run_cicero() {
    echo ""
    echo "Cicero: Mapping scATACseq peaks to target genes"

    # Validate variables
    if [[ -z "$R_SCRIPT_DIR" || -z "$ATAC_FILE_NAME" || -z "$OUTPUT_DIR" || -z "$LOG_DIR" ]]; then
        echo "[ERROR] One or more required variables (R_SCRIPT_DIR, ATAC_FILE_NAME, OUTPUT_DIR, LOG_DIR) are not set."
        exit 1
    fi

    if [ ! -f "$OUTPUT_DIR/cicero_atac_input.txt" ]; then
        {
            /usr/bin/time -v \
                poetry run python "$PYTHON_SCRIPT_DIR/convert_atac_to_sparse_for_cicero.py" \
                    --atac_data_file "$ATAC_FILE_NAME" \
                    --output_dir "$OUTPUT_DIR"
        } &> "$LOG_DIR/convert_atac_to_sparse_for_cicero.log"
    fi

    # Ensure log directory exists
    mkdir -p "$LOG_DIR"

    # Check R environment
    check_r_environment

    # Check for the chomosome size and gene annotation files in INPUT_DIR
    check_cicero_genome_files_exist

    echo "    Checks complete, running Cicero"
    conda deactivate
    conda activate monocle3_env

    export HDF5_DIR=$CONDA_PREFIX
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    export CFLAGS="-std=gnu99"
    export CXXFLAGS="-std=gnu++14"

    # Run your R script and pass PYTHON_PATH to the environment
    /usr/bin/time -v \
        Rscript "$R_SCRIPT_DIR/run_cicero.r" \
            "$OUTPUT_DIR/cicero_atac_input.txt" \
            "$OUTPUT_DIR" \
            "$CHROM_SIZES" \
            "$GENE_ANNOT" \
            &> "$LOG_DIR/Step010.run_cicero.log"

    
    # Unload the rstudio module and re-activate the conda environment after running the first step
    # module unload rstudio
    activate_conda_env

}

run_cicero_peak_to_tg_score() {
    echo ""
    echo "Python: Parsing Cicero peak to TG scores"
    /usr/bin/time -v \
    poetry run python "$PYTHON_SCRIPT_DIR/pipeline/cicero_peak_to_tg_score.py" \
        --output_dir "$OUTPUT_DIR" 
    
} 2> "$LOG_DIR/Step015.cicero_peak_to_tg_score.log"

run_correlation_peak_to_tg_score() {
    echo ""
    echo "Python: Calculating correlation peak to TG score"
    /usr/bin/time -v \
    poetry run python "$PYTHON_SCRIPT_DIR/pipeline/peak_gene_correlation.py" \
        --atac_data_file "$ATAC_FILE_NAME" \
        --rna_data_file "$RNA_FILE_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --num_cpu "$NUM_CPU" \
        --fig_dir "$FIG_DIR"

} 2> "$LOG_DIR/Step020.peak_gene_correlation.log"

run_sliding_window_tf_to_peak_score() {
    echo ""
    echo "Python: Calculating sliding window TF to peak scores"
    /usr/bin/time -v \
    poetry run python "$PYTHON_SCRIPT_DIR/pipeline/sliding_window_tf_peak_motifs.py" \
        --tf_names_file "$TF_NAMES_FILE"\
        --meme_dir "$MEME_DIR"\
        --reference_genome_dir "$REFERENCE_GENOME_DIR"\
        --output_dir "$OUTPUT_DIR" \
        --species "$SPECIES" \
        --num_cpu "$NUM_CPU" \
        --fig_dir "$FIG_DIR"
    
} 2> "$LOG_DIR/Step040.sliding_window_tf_peak_motifs.log"

run_homer() {
    echo ""
    echo "===== Running Homer ====="
    echo "[HINT] Homer log files are found under Homer_logs in the sample log directory"
    mkdir -p "$LOG_DIR/Homer_logs/"

    echo "Searching for '${BASE_DIR}/data/homer' directory..."
    # Check to make sure Homer is installed, else install it
    if [ ! -d "$BASE_DIR/data/homer" ]; then
        echo ""
        echo "    Homer installation not found, installing..."
        install_homer
        echo "    Done!"
    else
        echo "    - Found existing installation of homer"
    fi

    # Make sure that the homer directory is part of the path
    echo "Adding the 'homer/bin' directory to PATH"
    export PATH="$BASE_DIR/data/homer/bin:$PATH"
    export PERL5LIB="$BASE_DIR/data/homer/bin:${PERL5LIB:-}"
    
    echo "Checking for the Homer ${SPECIES} genome file..."
    # Check if the species Homer genome is installed
    if [ -d "$BASE_DIR/data/homer/data/genomes/${SPECIES}" ]; then
        echo "    - ${SPECIES} genome is installed"
    else
        echo "    - ${SPECIES} genome is not installed, installing..."
        install_homer_species_genome
        echo "    Done!"
    fi


    echo "Checking for existing Homer peak file (created from the ATACseq dataset)"
    if [ ! -f "$OUTPUT_DIR/homer_peaks.txt" ]; then
        echo "    Homer peak file not found, creating..."
        create_homer_peak_file
        echo "    Done!"
    else
        echo "    - Existing Homer peak file found"
    fi

    # If the homer_results directory doesn't exist for the sample, run findMotifsGenome
    if [ ! -d "$OUTPUT_DIR/homer_results/knownResults" ] || [ -z "$(ls -A "$OUTPUT_DIR/homer_results/knownResults" 2>/dev/null)" ]; then
        echo "[INFO] Running HOMER: knownResults/ missing or empty."
        homer_find_motifs
    else
        echo "[INFO] HOMER knownResults/ already exists and is populated."
    fi


    if [ ! -d "$OUTPUT_DIR/homer_results/homer_tf_motif_scores" ] || [ -z "$(ls -A "$OUTPUT_DIR/homer_results/homer_tf_motif_scores" 2>/dev/null)" ]; then
        echo "[INFO] Running HOMER annotatePeaks."
        homer_process_motif_files
    else
        echo "[INFO] annotatePeaks results already exist."
    fi
    echo "Finished running Homer"

    if [ -d "$OUTPUT_DIR/homer_results" ] && [ -z "$(ls -A "$OUTPUT_DIR/homer_results")" ]; then
        echo "[WARN] HOMER directory exists but is empty — removing..."
        rm -rf "$OUTPUT_DIR/homer_results"
    fi
}

run_homer_tf_to_peak_score() {
    echo ""
    echo "Python: Calculating homer TF to peak scores"
    /usr/bin/time -v \
    poetry run python "$PYTHON_SCRIPT_DIR/pipeline/homer_tf_peak_motifs.py" \
        --input_dir "${OUTPUT_DIR}/homer_results/homer_tf_motif_scores" \
        --output_dir "$OUTPUT_DIR" \
        --cpu_count $NUM_CPU

} 2> "$LOG_DIR/Step050.homer_tf_to_peak_motifs.log"

run_combine_dataframes() {
    echo ""
    echo "Python: Creating TF to TG score DataFrame"
    /usr/bin/time -v \
    poetry run python "$PYTHON_SCRIPT_DIR/pipeline/combine_dataframes.py" \
        --rna_data_file "$RNA_FILE_NAME" \
        --atac_data_file "$ATAC_FILE_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --inferred_grn_dir "$INFERRED_GRN_DIR" \
        --string_dir "$STRING_DB_DIR" 
    
} 2> "$LOG_DIR/Step060.combine_dataframes.log"

run_classifier_training() {
    echo ""
    echo "Python: Training XGBoost Classifier"
    /usr/bin/time -v \
    poetry run python "$PYTHON_SCRIPT_DIR/pipeline/train_xgboost.py" \
        --ground_truth_file "$GROUND_TRUTH_FILE" \
        --inferred_network_file "$INFERRED_NET_FILE" \
        --trained_model_dir "$TRAINED_MODEL_DIR" \
        --fig_dir "$FIG_DIR" \
        --model_save_name "xgb_full_network_model"


} 2> "$LOG_DIR/Step070.train_xgboost.log"

run_homer_and_score() {
    run_homer
    run_homer_tf_to_peak_score
}

# =============================================
#               MAIN PIPELINE
# =============================================

# Help option
if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash main_pipeline.sh"
    echo "This script executes a single-cell GRN inference pipeline."
    echo "Modify the flags at the top of the script to enable/disable steps."
    exit 0
fi

# ----- Perform validation of pipeline requirements -----
validate_critical_variables
check_for_running_jobs
check_pipeline_steps
check_tools
determine_num_cpus
check_processed_files
check_input_files
activate_conda_env
setup_directories


# ----- Execute selected pipeline steps -----
if [ "$CICERO_MAP_PEAKS_TO_TG" = true ]; then run_cicero; fi

# Map array index to function name
parallel_jobs=()

[ "$CICERO_PEAK_TO_TG_SCORE" = true ] && parallel_jobs+=("run_cicero_peak_to_tg_score")
[ "$PEAK_TO_TG_CORRELATION" = true ] && parallel_jobs+=("run_correlation_peak_to_tg_score")
[ "$SLIDING_WINDOW_TF_TO_PEAK_SCORE" = true ] && parallel_jobs+=("run_sliding_window_tf_to_peak_score")
[ "$HOMER_TF_TO_PEAK_SCORE" = true ] && parallel_jobs+=("run_homer_and_score")

# Run all selected steps in background
for job in "${parallel_jobs[@]}"; do
    echo "[INFO] Launching $job in background..."
    $job &
done

# Wait for all to finish
if [ ${#parallel_jobs[@]} -gt 0 ]; then
    echo "[INFO] Waiting for parallel scoring steps to finish..."
    wait
    echo "[INFO] All parallel scoring steps completed."
fi

if [ "$COMBINE_DATAFRAMES" = true ]; then check_string_db_files; run_combine_dataframes; fi
if [ "$TRAIN_XGBOOST_CLASSIFIER" = true ]; then run_classifier_training; fi