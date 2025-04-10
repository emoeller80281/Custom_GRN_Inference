#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=LOGS/apply_trained_xgboost_%a.log
#SBATCH --error=LOGS/apply_trained_xgboost_%a.err
#SBATCH --array=0

source activate my_env

BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")

# Define output directories and ground truth files (if needed)
K562_OUTPUT="$PROJECT_DIR/output/K562/K562_human_filtered"
K562_GROUND_TRUTH="$PROJECT_DIR/ground_truth_files/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"

# Specify the trained model files
MODEL_FILES=( \
    "$K562_OUTPUT/xgb_all_features_raw_model.pkl" \ 
    "$K562_OUTPUT/xgb_all_features_w_string_model.pkl" \ 
    "$K562_OUTPUT/xgb_all_method_combos_raw_model.pkl" \
    "$K562_OUTPUT/xgb_all_method_combos_summed_model.pkl" \
    "$K562_OUTPUT/xgb_string_scores_only_model.pkl" \
)

# Specify the path to the target score file to pass into the model for inferring edge scores
TARGET_FILES=( \
    "$K562_OUTPUT/inferred_network_raw.csv" 
    "$K562_OUTPUT/inferred_network_w_string.csv" 
    "$K562_OUTPUT/inferred_network_method_combos_summed.csv" 
    "$K562_OUTPUT/inferred_network_method_combos_raw.csv" 
    "$K562_OUTPUT/inferred_network_string_scores_only.csv" 
)

# Specify the save name
SAVE_NAMES=( \
    "inferred_network_raw" \
    "inferred_network_w_string"
    "inferred_network_method_combos_summed" \
    "inferred_network_method_combos_raw" \
    "inferred_network_string_scores_only"
)

# Use SLURM_ARRAY_TASK_ID to index into each array
INDEX=${SLURM_ARRAY_TASK_ID}

MODEL=${MODEL_FILES[$INDEX]}
TARGET=${TARGET_FILES[$INDEX]}
SAVE_NAME=${SAVE_NAMES[$INDEX]}

echo "[INFO] Running task $INDEX:"
echo "        Model: $MODEL"
echo "        Target: $TARGET"
echo "        Save Name: $SAVE_NAME"

# Run the python script for the selected model
python3 "$BASE_DIR/src/python_scripts/Step090.apply_trained_xgboost.py" \
    --output_dir "$K562_OUTPUT" \
    --model "$MODEL" \
    --target "$TARGET" \
    --save_name "${SAVE_NAME}_xgb_inferred_grn.tsv"