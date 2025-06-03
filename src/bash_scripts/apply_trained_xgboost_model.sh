#!/bin/bash -l
#SBATCH --partition=memory
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=LOGS/apply_trained_xgboost.log
#SBATCH --error=LOGS/apply_trained_xgboost.log

source activate my_env

BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")

KIDNEY_TARGET="$BASE_DIR/output/mouse_kidney/mouse_kidney_sample1/inferred_grns/inferred_score_df.parquet"
DS011_TARGET="$BASE_DIR/output/DS011_mESC/DS011_mESC_sample1/inferred_grns/inferred_score_df.parquet"

# Running predictions for DS011 using the combined sample mESC model
COMBINED_MODEL_PREDICTION_DIR="$BASE_DIR/output/combined_inferred_dfs/model_predictions"
MESC_COMBINED_MODEL="$BASE_DIR/output/combined_inferred_dfs/xgb_trained_models/xgb_mESC_combined_model_no_peak_agg.json"
COMBINED_VS_DS011_SAVE_NAME="combined_mESC_vs_DS011_mESC_xgb_pred.tsv"

# # Run the python script for the selected model
# python3 "$BASE_DIR/src/python_scripts/Step090.apply_trained_xgboost.py" \
#     --output_dir "$COMBINED_MODEL_PREDICTION_DIR" \
#     --model "$MESC_COMBINED_MODEL" \
#     --target "$DS011_TARGET" \
#     --save_name "$COMBINED_VS_DS011_SAVE_NAME"

# echo "DONE!"

# # Running predictions for DS011 using filtered_L2_E7.5_rep2 model
# MESC_MODEL_PREDICTION_DIR="$BASE_DIR/output/mESC/filtered_L2_E7.5_rep2/model_predictions"
# MESC_MODEL="$BASE_DIR/output/mESC/filtered_L2_E7.5_rep2/trained_models/xgb_mESC_filtered_L2_E7.5_rep2_no_peak_agg.json"
# MESC_VS_DS011_SAVE_NAME="filtered_L2_E7.5_vs_DS011_xgb_pred_no_peak_agg.tsv"

# # Run the python script for the selected model
# python3 "$BASE_DIR/src/python_scripts/Step090.apply_trained_xgboost.py" \
#     --output_dir "$MESC_MODEL_PREDICTION_DIR" \
#     --model "$MESC_MODEL" \
#     --target "$DS011_TARGET" \
#     --save_name "$MESC_VS_DS011_SAVE_NAME"

# echo "DONE!"

DS011_PREDICTION_DIR="$BASE_DIR/output/DS011_mESC/DS011_mESC_sample1/model_predictions"
DS011_MODEL="$BASE_DIR/output/DS011_mESC/DS011_mESC_sample1/trained_models/xgb_mESC_DS011_model.json"
MESC_TARGET="$BASE_DIR/output/mESC/filtered_L2_E7.5_rep2/inferred_grns/inferred_score_df.parquet"
DS011_VS_MESC_SAVE_NAME="DS011_vs_filtered_L2_E7.5_xgb_pred.tsv"

# Running predictions for filtered_L2_E7.5_rep2 using DS011 model
python3 "$BASE_DIR/src/python_scripts/Step090.apply_trained_xgboost.py" \
    --output_dir "$DS011_PREDICTION_DIR" \
    --model "$DS011_MODEL" \
    --target "$MESC_TARGET" \
    --save_name "$DS011_VS_MESC_SAVE_NAME"

echo "DONE!"