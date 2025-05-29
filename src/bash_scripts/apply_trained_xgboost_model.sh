#!/bin/bash -l
#SBATCH --partition=memory
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=LOGS/apply_trained_xgboost.log
#SBATCH --error=LOGS/apply_trained_xgboost.log

source activate my_env

BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")

COMBINED_MODEL_PREDICTION_DIR="$BASE_DIR/output/combined_inferred_dfs/model_predictions"

MESC_COMBINED_MODEL="$BASE_DIR/output/combined_inferred_dfs/xgb_trained_models/xgb_mESC_combined_model.json"

KIDNEY_TARGET="$BASE_DIR/output/mouse_kidney/mouse_kidney_sample1/inferred_grns/inferred_score_df.parquet"
DS011_TARGET="$BASE_DIR/output/DS011_mESC/DS011_mESC_sample1/inferred_grns/inferred_score_df.parquet"

KIDNEY_SAVE_NAME="combined_mESC_vs_mouse_kidney_xgb_pred.tsv"
DS011_SAVE_NAME="combined_mESC_vs_DS011_mESC_xgb_pred.tsv"

# Run the python script for the selected model
python3 "$BASE_DIR/src/python_scripts/Step090.apply_trained_xgboost.py" \
    --output_dir "$COMBINED_MODEL_PREDICTION_DIR" \
    --model "$MESC_COMBINED_MODEL" \
    --target "$DS011_TARGET" \
    --save_name "$DS011_SAVE_NAME"

echo "DONE!"