#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=LOGS/apply_trained_xgboost_%a.log
#SBATCH --error=LOGS/apply_trained_xgboost_%a.err
#SBATCH --array=0

source activate my_env

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/"

# Define output directories and ground truth files (if needed)
K562_OUTPUT="$PROJECT_DIR/output/K562/K562_human_filtered"
K562_GROUND_TRUTH="$PROJECT_DIR/ground_truth_files/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"

# Arrays for model file, target file and save_name for each task.
# Adjust the array elements as appropriate for your case.


MODEL_FILES=( \
    #"$K562_OUTPUT/xgb_all_features_raw_model.pkl" \
    #"$K562_OUTPUT/xgb_all_method_combos_summed_model.pkl" \
    "$K562_OUTPUT/xgb_all_method_combos_raw_model.pkl" \

)



TARGET_FILES=( \
    #"$K562_OUTPUT/full_network_feature_files/inferred_network_raw.csv" \
    #"$K562_OUTPUT/full_network_feature_files/full_inferred_network_agg_method_combo.csv" \
    "$K562_OUTPUT/full_network_feature_files/full_inferred_network_each_method_combo.csv" \

)

# 
SAVE_NAMES=( \
    #"full_network_all_features_raw" \
    #"full_network_all_method_combos_summed" \
    "full_network_all_method_combos_raw" \
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
python3 "$PROJECT_DIR/src/python_scripts/Step090.apply_trained_xgboost.py" \
    --output_dir "$K562_OUTPUT" \
    --model "$MODEL" \
    --target "$TARGET" \
    --save_name "${SAVE_NAME}_xgb_inferred_grn.tsv"