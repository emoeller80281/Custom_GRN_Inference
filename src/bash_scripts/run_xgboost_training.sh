#!/bin/bash -l
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -o "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/xgboost_training.log"
#SBATCH -e "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/xgboost_training.err"

# Set base directories and files
BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
PYTHON_SCRIPT_DIR="$BASE_DIR/src/python_scripts"

COMBINED_GROUND_TRUTH="$BASE_DIR/ground_truth_files/combined_mESC_ground_truth.tsv"
COMBINED_INFERRED_NET="$BASE_DIR/output/combined_inferred_dfs/mESC_combined_inferred_score_df.parquet"
OUTPUT_DIR="$BASE_DIR/output/combined_inferred_dfs/xgb_trained_models"
FIG_DIR="$BASE_DIR/figures/mm10/combined_samples"

mkdir -p "$FIG_DIR"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Python: Training XGBoost Classifier"
/usr/bin/time -v python3 "$PYTHON_SCRIPT_DIR/Step070.train_xgboost.py" \
        --ground_truth_file "$COMBINED_GROUND_TRUTH" \
        --inferred_network_file "$COMBINED_INFERRED_NET" \
        --trained_model_dir "$OUTPUT_DIR" \
        --fig_dir "$FIG_DIR" \
        --model_save_name "xgb_mESC_combined_model"

