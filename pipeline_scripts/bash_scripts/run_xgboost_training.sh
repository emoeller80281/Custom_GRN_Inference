#!/bin/bash -l
#SBATCH -p memory
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -o "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/xgboost_training.log"
#SBATCH -e "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/xgboost_training.err"

determine_num_cpus() {
    echo ""
    echo "[INFO] Checking the number of CPUs available for parallel processing"
    if [ -z "${SLURM_CPUS_PER_TASK:-}" ]; then
        if command -v nproc &> /dev/null; then
            TOTAL_CPUS=$(nproc --all)
            case $TOTAL_CPUS in
                [1-15]) IGNORED_CPUS=1 ;;  # Reserve 1 CPU for <=15 cores
                [16-31]) IGNORED_CPUS=2 ;; # Reserve 2 CPUs for <=31 cores
                *) IGNORED_CPUS=4 ;;       # Reserve 4 CPUs for >=32 cores
            esac
            NUM_CPU=$((TOTAL_CPUS - IGNORED_CPUS))
            echo "    - Running locally. Detected $TOTAL_CPUS CPUs, reserving $IGNORED_CPUS for system tasks. Using $NUM_CPU CPUs."
        else
            NUM_CPU=1  # Fallback
            echo "    - Running locally. Unable to detect CPUs, defaulting to $NUM_CPU CPU."
        fi
    else
        NUM_CPU=${SLURM_CPUS_PER_TASK}
        echo "    - Running on SLURM. Number of CPUs allocated: ${NUM_CPU}"
    fi
}

determine_num_cpus

conda activate my_env

# Set base directories and files
BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
PYTHON_SCRIPT_DIR="$BASE_DIR/src/grn_inference/pipeline"

# GROUND_TRUTH="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"
GROUND_TRUTH="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN115_LOGOF_ESCAPE_Mouse_ESC.tsv"

# COMBINED_INFERRED_NET="$BASE_DIR/output/combined_inferred_dfs/mESC_combined_inferred_score_df.parquet"
# COMBINED_OUTPUT_DIR="$BASE_DIR/output/combined_inferred_dfs/xgb_trained_models"
# COMBINED_FIG_DIR="$BASE_DIR/figures/mm10/combined_samples_no_peak_agg"

# mkdir -p "$COMBINED_FIG_DIR"
# mkdir -p "$COMBINED_OUTPUT_DIR"

# echo ""
# echo "Python: Training XGBoost Classifier"
# /usr/bin/time -v python3 "$PYTHON_SCRIPT_DIR/Step070.train_xgboost.py" \
#         --ground_truth_file "$GROUND_TRUTH" \
#         --inferred_network_file "$COMBINED_INFERRED_NET" \
#         --trained_model_dir "$COMBINED_OUTPUT_DIR" \
#         --fig_dir "$COMBINED_FIG_DIR" \
#         --model_save_name "xgb_mESC_combined_model_no_peak_agg" \
#         --num_cpu "$NUM_CPU"

# MESC_SAMPLE_INFERRED_NET="$BASE_DIR/output/mESC/filtered_L2_E7.5_rep2/inferred_grns/inferred_score_df.parquet"
# MESC_SAMPLE_OUTPUT_DIR="$BASE_DIR/output/mESC/filtered_L2_E7.5_rep2/trained_models"
# MESC_SAMPLE_FIG_DIR="$BASE_DIR/figures/mm10/filtered_L2_E7.5_rep2_no_peak_agg"

# mkdir -p "$MESC_SAMPLE_FIG_DIR"
# mkdir -p "$MESC_SAMPLE_OUTPUT_DIR"

# echo ""
# echo "Python: Training XGBoost Classifier"
# /usr/bin/time -v python3 "$PYTHON_SCRIPT_DIR/Step070.train_xgboost.py" \
#         --ground_truth_file "$GROUND_TRUTH" \
#         --inferred_network_file "$MESC_SAMPLE_INFERRED_NET" \
#         --trained_model_dir "$MESC_SAMPLE_OUTPUT_DIR" \
#         --fig_dir "$MESC_SAMPLE_FIG_DIR" \
#         --model_save_name "xgb_mESC_filtered_L2_E7.5_rep2_no_peak_agg" \
#         --num_cpu "$NUM_CPU"


DS011_INFERRED_NET="$BASE_DIR/output/DS011_mESC/DS011_mESC_sample1/inferred_grns/inferred_score_df.parquet"
DS011_OUTPUT_DIR="$BASE_DIR/output/DS011_mESC/DS011_mESC_sample1/trained_models"
DS011_FIG_DIR="$BASE_DIR/figures/mm10/DS011"

mkdir -p "$DS011_FIG_DIR"
mkdir -p "$DS011_OUTPUT_DIR"

echo ""
echo "Python: Training XGBoost Classifier"
/usr/bin/time -v python3 "$PYTHON_SCRIPT_DIR/train_xgboost.py" \
        --ground_truth_file "$GROUND_TRUTH" \
        --inferred_network_file "$DS011_INFERRED_NET" \
        --trained_model_dir "$DS011_OUTPUT_DIR" \
        --fig_dir "$DS011_FIG_DIR" \
        --model_save_name "xgb_mESC_DS011_model" \
        --num_cpu "$NUM_CPU"

