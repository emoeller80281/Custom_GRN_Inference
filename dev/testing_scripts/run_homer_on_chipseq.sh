#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem=256G
#SBATCH -o /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/chipseq_homer.log
#SBATCH -e /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/chipseq_homer.err

set -euo pipefail

source activate my_env

BASE_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
OUTPUT_DIR="$BASE_DIR/output/chipseq_homer"
SPECIES="mm10"
NUM_CPU=32

cd "$OUTPUT_DIR"

# Make sure that the homer directory is part of the path
echo "Adding the 'homer/bin' directory to PATH"
export PATH="$BASE_DIR/data/homer/bin:$PATH"
# export PERL5LIB="$BASE_DIR/data/homer/bin:$PERL5LIB"

echo "Running findMotifsGenome.pl"
mkdir -p "$OUTPUT_DIR/homer_results"
perl "$BASE_DIR/data/homer/bin/findMotifsGenome.pl" \
    "$OUTPUT_DIR/tmp/homer_peaks.txt" \
    "$SPECIES" "$OUTPUT_DIR/homer_results/" \
    -size 200 \
    -p $NUM_CPU \
    -redundant 0.5
echo "    Done!"

echo ""
echo "----- Homer annotatePeaks.pl -----"
echo "[INFO] Starting motif file processing"

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
echo "Running annotatePeaks.pl"
echo "$motif_files" | /usr/bin/time -v parallel -j "$NUM_CPU" \
    "perl $BASE_DIR/data/homer/bin/annotatePeaks.pl $OUTPUT_DIR/tmp/homer_peaks.txt '$SPECIES' -m {} > $PROCESSED_MOTIF_DIR/{/}_tf_motifs.txt"

module unload parallel

PYTHON_SCRIPT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/src/grn_inference"

echo ""
echo "Python: Calculating homer TF to peak scores"
/usr/bin/time -v \
poetry run python "$PYTHON_SCRIPT_DIR/pipeline/homer_tf_peak_motifs.py" \
    --input_dir "${OUTPUT_DIR}/homer_results/homer_tf_motif_scores" \
    --output_dir "$OUTPUT_DIR" \
    --cpu_count $NUM_CPU