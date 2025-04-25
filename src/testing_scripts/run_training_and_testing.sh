#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -o /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/split_train_apply.log
#SBATCH -e /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/split_train_apply.err

set -euo pipefail

CONDA_ENV_NAME="my_env"
activate_conda_env() {
    CONDA_BASE=$(conda info --base)
    if [ -z "$CONDA_BASE" ]; then
        echo "[ERROR] Conda base could not be determined. Is Conda installed and in your PATH?"
        exit 1
    fi

    source "$CONDA_BASE/bin/activate"
    if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
        echo "[ERROR] Conda environment '$CONDA_ENV_NAME' does not exist."
        exit 1
    fi

    conda activate "$CONDA_ENV_NAME" || { echo "Error: Failed to activate Conda environment '$CONDA_ENV_NAME'."; exit 1; }
    echo "[INFO] Activated Conda environment: $CONDA_ENV_NAME"
}

# Handles splitting features, training XGBoost models from those features, and making predictions with the models
run_split_train_test() {
    local CELL_TYPE="$1"
    local MODEL_NAME="$2"
    local GROUND_TRUTH_FILE="$3"
    local TARGET_NAME="$4"
    local SPECIES="$5"

    # The next two parameters are the names of the arrays
    local SAMPLE_NAMES_ARRAY_NAME="$6"
    local TARGET_DIR_ARRAY_NAME="$7"

    # Create namerefs for the arrays to pass in the entire list of targets and samples
    declare -n SAMPLE_NAMES="$SAMPLE_NAMES_ARRAY_NAME"
    declare -n TARGET_DIR="$TARGET_DIR_ARRAY_NAME"

    local STRING_DB_DIR="$BASE_DIR"/string_database/$SPECIES/

    # Run for each selected sample
    for SAMPLE_NAME in "${SAMPLE_NAMES[@]}"; do
        echo ""
        echo "===== ADDING STRING SCORES FOR ${CELL_TYPE} - Sample: ${SAMPLE_NAME} ====="

        # Set the output and log directory for the sample
        local OUTPUT_DIR="$BASE_DIR/output/${CELL_TYPE}/${SAMPLE_NAME}"

        local INFERRED_GRN_DIR="$OUTPUT_DIR/inferred_grns"
        local TRAINED_MODEL_DIR="$OUTPUT_DIR/trained_models"
        local MODEL_PREDICTION_DIR="$OUTPUT_DIR/model_predictions"
        local LOG_DIR="${BASE_DIR}/LOGS/${CELL_TYPE}_logs/${SAMPLE_NAME}_logs"

        mkdir -p "${MODEL_PREDICTION_DIR}"
        mkdir -p "${LOG_DIR}"

        # # Check to see if any of the feature set data file exists 
        # local FEATURE_SET_FILES_EXIST=true
        for FEATURE_SET in "${FEATURE_SET_NAMES[@]}"; do
            echo ""
            echo "===== ${CELL_TYPE} — Sample ${SAMPLE_NAME} — Feature set: ${FEATURE_SET} ====="

            # prefer the “_w_string.parquet” if it exists
            if [[ -f "${INFERRED_GRN_DIR}/${FEATURE_SET}_w_string.parquet" ]]; then
                FEATURE_FILE="${INFERRED_GRN_DIR}/${FEATURE_SET}_w_string.parquet"
                echo "    Found string-augmented file: ${FEATURE_SET}_w_string.parquet"
            else
                FEATURE_FILE="${INFERRED_GRN_DIR}/${FEATURE_SET}.parquet"
                echo "    Using original file: ${FEATURE_SET}.parquet"

                python3 "$PYTHON_SCRIPT_DIR/Step070.find_edges_in_string_db.py" \
                    --inferred_net_file "$FEATURE_FILE" \
                    --string_dir       "$STRING_DB_DIR" \
                    --output_dir       "$INFERRED_GRN_DIR"
            fi
        done

        echo ""
        echo "===== XGBOOST MODEL TRAINING FOR ${CELL_TYPE} - Sample: ${SAMPLE_NAME} ====="
        for FEATURE_SET in "${FEATURE_SET_NAMES[@]}"; do

            # Train the XGBoost classifier for the current feature set if a trained model doesn't exist
            if [ ! -f "${TRAINED_MODEL_DIR}/xgb_${FEATURE_SET}_model.pkl" ]; then
                local FEATURE_FILE="${INFERRED_GRN_DIR}/${FEATURE_SET}_w_string.parquet"
                local FIG_DIR="${BASE_DIR}/figures/hg38/${SAMPLE_NAME}/${FEATURE_SET}"
                mkdir -p "$FIG_DIR"

                
                echo "    Python: Training XGBoost Classifier for feature set '${FEATURE_SET}'"
                python3 "$PYTHON_SCRIPT_DIR/Step080.train_xgboost.py" \
                        --ground_truth_file "$GROUND_TRUTH_FILE" \
                        --inferred_network_file "$FEATURE_FILE" \
                        --trained_model_dir "$TRAINED_MODEL_DIR" \
                        --fig_dir "$FIG_DIR" \
                        --model_save_name "xgb_${FEATURE_SET}_model"
                echo "        Done!"

            else
                echo "    Trained XGBoost model 'xgb_${FEATURE_SET}_model.pkl' already exists for ${SAMPLE_NAME}, skipping..."
            fi
        done

        echo ""
        echo "===== RUNNING XGBOOST MODEL APPLICATION ====="
        # Apply the trained XGBoost classifier model for each feature set to the corresponding feature set in the target directory
        for FEATURE_SET in "${FEATURE_SET_NAMES[@]}"; do

            local MODEL_FILE="${TRAINED_MODEL_DIR}/xgb_${FEATURE_SET}_model.pkl"

            for TARGET in "${TARGET_DIR[@]}"; do

                # Skip if the prediction file already exists
                if [ ! -f "${MODEL_PREDICTION_DIR}/${CELL_TYPE}_vs_${TARGET_NAME}_${FEATURE_SET}_xgb_pred.parquet" ]; then
                    # Check to make sure the target feature set dataframe file exists, otherwise skip
                    if [ -f "${TARGET}/${FEATURE_SET}_w_string.parquet" ]; then
                        local TARGET_FILE="${TARGET}/${FEATURE_SET}_w_string.parquet"
                        echo "    Python: Applying trained XGBoost classifier for ${FEATURE_SET} to target: ${TARGET_NAME}"
                        python3 "$BASE_DIR/src/python_scripts/Step090.apply_trained_xgboost.py" \
                            --output_dir "${MODEL_PREDICTION_DIR}" \
                            --model "$MODEL_FILE" \
                            --target "$TARGET_FILE" \
                            --save_name "${MODEL_NAME}_vs_${TARGET_NAME}_${FEATURE_SET}_xgb_pred.tsv"
                        echo "        Done!"
                    else
                        echo "    Feature set ${FEATURE_SET}_w_string.parquet does not exist for ${TARGET}"
                    fi

                else
                    echo "    Prediction file for target ${TARGET} exists for sample ${SAMPLE_NAME} feature set ${FEATURE_SET}, skipping..."
                fi

            done
        done
    done
}

# Activate the Conda environment
activate_conda_env

# Global variables
BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
PYTHON_SCRIPT_DIR="$BASE_DIR/src/python_scripts"

K562_INFERRED_NET_DIR="$BASE_DIR/output/K562/K562_human_filtered/inferred_grns"
MACROPHAGE_INFERRED_NET_DIR="$BASE_DIR/output/macrophage/macrophage_buffer1_filtered/inferred_grns"
MESC_INFERRED_NET_DIR="$BASE_DIR/output/mESC/filtered_L2_E7.5_rep1/inferred_grns"

# Core names of the different feature files to build off of
FEATURE_SET_NAMES=( \
    "inferred_network" \
    "inferred_network_50pct" \
    "inferred_network_enrich_feat" \
)

# Define arrays for each cell type
SAMPLE_NAMES_K562=( "K562_human_filtered" )
TARGET_DIR_K562=( "$K562_INFERRED_NET_DIR" )
TARGET_NAME_K562="K562" # Set the target for making predictions with the cell type's trained models (test model on same vs different cell type / sample)
GROUND_TRUTH_FILE_K562="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"

SAMPLE_NAMES_MACROPHAGE=( "macrophage_buffer1_filtered" )
TARGET_DIR_MACROPHAGE=( "$MACROPHAGE_INFERRED_NET_DIR" )
TARGET_NAME_MACROPHAGE="macrophage"
GROUND_TRUTH_FILE_MACROPHAGE="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN204_ChIPSeq_ChIPAtlas_Human_Macrophages.tsv"


SAMPLE_NAMES_MESC=( \
        # "filtered_L2_E7.5_rep1"
        # "filtered_L2_E7.5_rep2"
        # "filtered_L2_E7.75_rep1"
        # "filtered_L2_E8.0_rep1"
        # "filtered_L2_E8.0_rep2"
        "filtered_L2_E8.5_rep1"
        # "filtered_L2_E8.5_rep2"
        # "filtered_L2_E8.75_rep1"
        # "filtered_L2_E8.75_rep2"
    )
TARGET_DIR_MESC=( "$MESC_INFERRED_NET_DIR" )
TARGET_NAME_MESC="mESC_E7.5_rep1"
GROUND_TRUTH_FILE_MESC="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"


# # Run for K562
# run_split_train_test "K562" "$GROUND_TRUTH_FILE_K562" "$TARGET_NAME_K562" "hg38" SAMPLE_NAMES_K562 TARGET_DIR_K562 hg38

# # Run for macrophage
# run_split_train_test "macrophage" "$GROUND_TRUTH_FILE_MACROPHAGE" "$TARGET_NAME_MACROPHAGE" "hg38" SAMPLE_NAMES_MACROPHAGE TARGET_DIR_MACROPHAGE 

# # Run for mESC
run_split_train_test "mESC" "mESC_E8.5_rep1" "$GROUND_TRUTH_FILE_MESC" "$TARGET_NAME_MESC" "mm10" SAMPLE_NAMES_MESC TARGET_DIR_MESC 