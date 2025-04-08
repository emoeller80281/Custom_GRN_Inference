#!/bin/bash -l
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 2
#SBATCH --mem-per-cpu=32G
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --array=0-4

# Set base directories and files
BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
OUTPUT_DIR="$BASE_DIR/output/K562/K562_human_filtered"
GROUND_TRUTH_FILE="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/ground_truth_files/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"
PYTHON_SCRIPT_DIR="$BASE_DIR/src/python_scripts"
LOG_DIR="$BASE_DIR/LOGS"

# Define array of sample names and corresponding feature files
SAMPLES=( 
    all_features_raw 
    all_features_w_string 
    all_method_combos_summed 
    all_method_combos_raw 
    string_score_only
)

FILES=( 
    "$OUTPUT_DIR/full_network_feature_files/inferred_network_raw.csv" 
    "$OUTPUT_DIR/full_network_feature_files/inferred_network_w_string.csv" 
    "$OUTPUT_DIR/full_network_feature_files/full_inferred_network_agg_method_combo.csv" 
    "$OUTPUT_DIR/full_network_feature_files/full_inferred_network_each_method_combo.csv" 
    "$OUTPUT_DIR/full_network_feature_files/string_score_only.csv" 
)

# Use the SLURM_ARRAY_TASK_ID to select the corresponding sample and feature file.
SAMPLE=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
FEATURE_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

# Create a directory for figures for this sample
FIG_DIR="$BASE_DIR/figures/hg38/K562_human_filtered/$SAMPLE"
mkdir -p "$FIG_DIR"

# Function to run classifier training
run_classifier_training() {
    echo ""
    echo "Python: Training XGBoost Classifier for sample: ${SAMPLE}"
    /usr/bin/time -v \
    python3 "$PYTHON_SCRIPT_DIR/Step080.train_xgboost.py" \
            --ground_truth_file "$GROUND_TRUTH_FILE" \
            --inferred_network_file "$FEATURE_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --fig_dir "$FIG_DIR" \
            --model_save_name "xgb_${SAMPLE}_model"
}

# Run the training function and redirect any stderr output to a specific log file for this sample
run_classifier_training 2> "$LOG_DIR/xgboost_training_${SAMPLE}.log"
