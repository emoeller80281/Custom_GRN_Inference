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


TRAINED_MODEL_DIR="$K562_OUTPUT/trained_models"

MODEL_PREDICTION_DIR="$K562_OUTPUT/model_predictions"
K562_GROUND_TRUTH="$PROJECT_DIR/ground_truth_files/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"


# Run the python script for the selected model
python3 "$BASE_DIR/src/python_scripts/Step090.apply_trained_xgboost.py" \
    --output_dir "$MODEL_PREDICTION_DIR" \
    --model "$MODEL" \
    --target "$TARGET" \
    --save_name "${SAVE_NAME}_xgb_inferred_grn.tsv"