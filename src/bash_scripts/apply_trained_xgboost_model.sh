#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=LOGS/apply_trained_xgboost.log
#SBATCH --error=LOGS/apply_trained_xgboost.log

source activate my_env

BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")

MODEL_PREDICTION_DIR="$BASE_DIR/output/combined_inferred_dfs/model_predictions"

MODEL="$BASE_DIR/output/combined_inferred_dfs/xgb_trained_models/xgb_mESC_combined_model.json"

TARGET="$BASE_DIR/output/mESC/filtered_L2_E7.5_rep1/inferred_grns/inferred_score_df.parquet"

SAVE_NAME="combined_mESC_vs_E7.5_rep1_xgb_pred.tsv"

# Run the python script for the selected model
python3 "$BASE_DIR/src/python_scripts/Step090.apply_trained_xgboost.py" \
    --output_dir "$MODEL_PREDICTION_DIR" \
    --model "$MODEL" \
    --target "$TARGET" \
    --save_name "$SAVE_NAME"