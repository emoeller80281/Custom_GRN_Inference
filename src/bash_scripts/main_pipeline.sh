#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem-per-cpu=16G
#SBATCH -o LOGS/find_tf_motifs.log
#SBATCH -e LOGS/find_tf_motifs.err

# Activate conda environment
source /gpfs/Home/esm5360/miniconda3/bin/activate my_env

set -e  # Exit script if any command fails

# =========== SELECT WHICH PROCESSES TO RUN ================
CREATE_HOMER_PEAK_FILE=false
HOMER_FIND_MOTIFS_GENOME=false
HOMER_ANNOTATE_PEAKS=false
PROCESS_MOTIF_FILES=false
PARSE_TF_PEAK_MOTIFS=false
CALCULATE_TF_REGULATION_SCORE=true


# ================= USER PATH VARIABLES ====================
# Base directory
BASE_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"

SRC_DIR="$BASE_DIR/src/python_scripts"
INPUT_DIR="$BASE_DIR/input"
OUTPUT_DIR="$BASE_DIR/output"
HOMER_DIR="$BASE_DIR/Homer/bin"
MOTIF_DIR="$OUTPUT_DIR/knownResults"

# Paths to input files
ATAC_DATA_FILE="$INPUT_DIR/macrophage_buffer1_filtered_ATAC.csv"
RNA_DATA_FILE="$INPUT_DIR/macrophage_buffer1_filtered_RNA.csv"
HOMER_PEAK_FILE="$INPUT_DIR/Homer_peaks.txt"

# Paths to output files
TF_MOTIF_BINDING_SCORE_FILE="$OUTPUT_DIR/total_motif_regulatory_scores.tsv"
PROCESSED_MOTIF_DIR="$OUTPUT_DIR/homer_tf_motif_scores"


# ===================== PIPELINE ============================
# Export necessary paths
export PATH="$HOMER_DIR:$PATH"

# Ensure required directories exist
mkdir -p "$PROCESSED_MOTIF_DIR"
mkdir -p "$OUTPUT_DIR"
touch "$TF_MOTIF_BINDING_SCORE_FILE"

cd "$HOMER_DIR"

if [ "$CREATE_HOMER_PEAK_FILE" = true ]; then
    echo "Creating Homer peak file"
    python3 "$SRC_DIR/Step010.create_homer_peak_file.py" \
        --atac_data_file "$ATAC_DATA_FILE"
fi

if [ "$HOMER_FIND_MOTIFS_GENOME" = true ]; then
    echo "Running Homer findMotifsGenome.pl"
    touch "$HOMER_PEAK_FILE"
    perl findMotifsGenome.pl "$HOMER_PEAK_FILE" "hg38" "$OUTPUT_DIR" -size 200
fi

if [ "$HOMER_ANNOTATE_PEAKS" = true ]; then
    echo "Running Homer annotatePeaks.pl"
    perl annotatePeaks.pl "$HOMER_PEAK_FILE" "hg38" \
        -m "$MOTIF_DIR/known1.motif" \
        > "$OUTPUT_DIR/known_motif_1_motif_to_peak.txt"
fi

if [ "$PROCESS_MOTIF_FILES" = true ]; then
    echo "Processing motif files in parallel"

    process_motif_file() {
        motif_file=$1
        input_file=$2
        genome=$3
        output_dir=$4

        motif_basename=$(basename "$motif_file" .motif)
        output_file="$output_dir/${motif_basename}_tf_motifs.txt"

        echo "Processing $motif_basename"
        annotatePeaks.pl "$input_file" "$genome" -m "$motif_file" > "$output_file"
    }

    export -f process_motif_file
    export HOMER_PEAK_FILE
    export MOTIF_DIR
    export PROCESSED_MOTIF_DIR

    find "$MOTIF_DIR" -name "*.motif" | parallel -j 32 process_motif_file {} "$HOMER_PEAK_FILE" "hg38" "$PROCESSED_MOTIF_DIR"

    echo "All motifs processed in parallel. Results saved in $PROCESSED_MOTIF_DIR"
fi

if [ "$PARSE_TF_PEAK_MOTIFS" = true ]; then
    python3 "$SRC_DIR/Step020.parse_TF_peak_motifs.py" \
        --input_dir "$PROCESSED_MOTIF_DIR" \
        --output_file "$TF_MOTIF_BINDING_SCORE_FILE" \
        --cpu_count 32
fi

if [ "$CALCULATE_TF_REGULATION_SCORE" = true ]; then
    python3 "$SRC_DIR/Step030.find_overlapping_TFs.py" \
        --rna_data_file "$RNA_DATA_FILE" \
        --tf_motif_binding_score_file "$TF_MOTIF_BINDING_SCORE_FILE"
fi
