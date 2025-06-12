#!/bin/bash -l
#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -o "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/train_test_evaluate.log"
#SBATCH -e "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/train_test_evaluate.err"

set -euo pipefail

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

source /gpfs/Home/esm5360/miniconda3/etc/profile.d/conda.sh

# ----- MODIFY THE FOLLOWING SETTINGS -----
COMPARE_FEATURE_SCORES=false
RUN_TRAINING=false
RUN_PREDICTION=true
RUN_STATS_ANALYSIS=true

GROUND_TRUTH_NAME="RN111_ChIPSeq"

MODEL_TRAINING_CELL_TYPE="DS011_mESC"
MODEL_TRAINING_SAMPLE="DS011_mESC_sample1_old"

PREDICTION_TARGET_CELL_TYPE="mESC"
PREDICTION_TARGET_SAMPLE="filtered_L2_E7.5_rep2_old"
# ------------------------------------------

PROJECT_BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
STATS_BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/STATISTICAL_ANALYSIS")
REF_NET_DIR=$(readlink -f "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS")
STATS_OUTPUT_DIR="$PROJECT_BASE_DIR/output/"

PROJECT_PYTHON_SCRIPT_DIR="$PROJECT_BASE_DIR/src/grn_inference/pipeline"

MODEL_TRAINING_GROUND_TRUTH="$REF_NET_DIR/RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"
STATS_ANALYSIS_GROUND_TRUTH="$REF_NET_DIR/RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"

MODEL_PREDICTION_DIR="$PROJECT_BASE_DIR/output/$MODEL_TRAINING_CELL_TYPE/$MODEL_TRAINING_SAMPLE/model_predictions"
MODEL_TRAINING_INFERRED_NET="$PROJECT_BASE_DIR/output/$MODEL_TRAINING_CELL_TYPE/$MODEL_TRAINING_SAMPLE/inferred_grns/inferred_score_df.parquet"
MODEL_SAVE_DIR="$PROJECT_BASE_DIR/output/$MODEL_TRAINING_CELL_TYPE/$MODEL_TRAINING_SAMPLE/trained_models"
MODEL_FIG_DIR="$PROJECT_BASE_DIR/figures/$MODEL_TRAINING_CELL_TYPE/$MODEL_TRAINING_SAMPLE"
MODEL_SAVE_NAME="xgb_${MODEL_TRAINING_SAMPLE}_model"

PREDICTION_TARGET_INFERRED_NET="$PROJECT_BASE_DIR/output/$PREDICTION_TARGET_CELL_TYPE/$PREDICTION_TARGET_SAMPLE/inferred_grns/inferred_score_df.parquet"
PREDICTION_SAVE_NAME="${MODEL_TRAINING_SAMPLE}_vs_${PREDICTION_TARGET_SAMPLE}_xgb_pred.tsv"

MODEL_TRAINING_GROUND_TRUTH_FILE="${MODEL_TRAINING_GROUND_TRUTH##*/}"
PREDICTION_GROUND_TRUTH_FILE="${STATS_ANALYSIS_GROUND_TRUTH##*/}"

mkdir -p "$MODEL_SAVE_DIR"
mkdir -p "$MODEL_FIG_DIR"
mkdir -p "$MODEL_PREDICTION_DIR"
mkdir -p "$PROJECT_BASE_DIR/output"

create_overlapping_feature_score_histogram() {
    conda activate my_env

    echo ""
    echo "Plotting overlapping feature score histogram between the model training dataset and the target dataset"
    /usr/bin/time -v poetry run python "$PROJECT_PYTHON_SCRIPT_DIR/compare_score_distributions.py" \
        --model_training_inferred_net "$MODEL_TRAINING_INFERRED_NET" \
        --prediction_target_inferred_net "$PREDICTION_TARGET_INFERRED_NET" \
        --model_training_sample_name "$MODEL_TRAINING_SAMPLE" \
        --prediction_target_sample_name "$PREDICTION_TARGET_SAMPLE" \
        --fig_dir "$PROJECT_BASE_DIR/output/prediction_accuracy_results/${MODEL_TRAINING_SAMPLE}_model/${PREDICTION_TARGET_SAMPLE}_target"
    echo "    DONE!"
}

run_xgboost_training() {
    conda activate my_env

    echo ""
    echo "  ----- Training XGBoost Classifier -----"
    echo "    Ground Truth = ${MODEL_TRAINING_GROUND_TRUTH_FILE}"
    echo "    Training Set = ${MODEL_TRAINING_SAMPLE}"

    /usr/bin/time -v poetry run python "$PROJECT_PYTHON_SCRIPT_DIR/train_xgboost.py" \
            --ground_truth_file "$MODEL_TRAINING_GROUND_TRUTH" \
            --inferred_network_file "$MODEL_TRAINING_INFERRED_NET" \
            --trained_model_dir "$MODEL_SAVE_DIR" \
            --fig_dir "$MODEL_FIG_DIR" \
            --model_save_name "$MODEL_SAVE_NAME" \
            --num_cpu "$NUM_CPU"
    echo "    DONE!"
}

run_model_predictions() {
    conda activate my_env

    echo ""
    echo "  ----- Applying XGBoost Classifier -----"
    echo "    Model Trained on = ${MODEL_TRAINING_SAMPLE}"
    echo "    Trained Model = ${MODEL_SAVE_NAME}"
    echo "    Target = ${PREDICTION_TARGET_SAMPLE}"

    /usr/bin/time -v poetry run python "$PROJECT_PYTHON_SCRIPT_DIR/apply_trained_xgboost.py" \
        --output_dir "${MODEL_PREDICTION_DIR}" \
        --model "${MODEL_SAVE_DIR}/${MODEL_SAVE_NAME}.json" \
        --target "${PREDICTION_TARGET_INFERRED_NET}" \
        --save_name "${PREDICTION_SAVE_NAME}"
    echo "    DONE!"
}

run_stats_analysis() {
    echo ""
    echo "  ----- Running Statistics for XGBoost Predictions -----"
    echo "    Trained Model = ${MODEL_TRAINING_SAMPLE}"
    echo "    Target = ${PREDICTION_TARGET_SAMPLE}"
    echo "    Ground Truth = ${PREDICTION_GROUND_TRUTH_FILE}"

    conda activate grn_analysis

    /usr/bin/time -v python3 "$STATS_BASE_DIR/Analyze_Inferred_GRN.py" \
        --inferred_net_filename "$PREDICTION_SAVE_NAME" \
        --method_name "prediction_accuracy_results" \
        --batch_name "${MODEL_TRAINING_SAMPLE}_model/${PREDICTION_TARGET_SAMPLE}_target/${GROUND_TRUTH_NAME}_ground_truth" \
        --method_input_path "$MODEL_PREDICTION_DIR" \
        --ground_truth_path "$STATS_ANALYSIS_GROUND_TRUTH" \
        --output_dir "$PROJECT_BASE_DIR/output/"
    echo "    DONE!"
}

if [ "$COMPARE_FEATURE_SCORES" = true ]; then
    create_overlapping_feature_score_histogram
fi

if [ "$RUN_TRAINING" = true ]; then
    run_xgboost_training
fi

if [ "$RUN_PREDICTION" = true ]; then
    run_model_predictions
fi

if [ "$RUN_STATS_ANALYSIS" = true ]; then
    run_stats_analysis
fi


