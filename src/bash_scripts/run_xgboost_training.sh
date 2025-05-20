#!/bin/bash -l
#SBATCH -p compute
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
        --model_save_name "xgb_mESC_combined_model" \
        --num_cpu "$NUM_CPU"

