#!/bin/bash -l

submit_job() {
    local SAMPLE_NAME=$1
    local CELL_TYPE=$2
    local SPECIES=$3
    local RNA_FILE_NAME=$4
    local ATAC_FILE_NAME=$5
    local GROUND_TRUTH_FILE=$6

    # Ensure the log directory exists
    mkdir -p "LOGS/${CELL_TYPE}_logs/${SAMPLE_NAME}_logs"

    # Submit the job
    sbatch \
        --export=ALL,SAMPLE_NAME="$SAMPLE_NAME",CELL_TYPE="$CELL_TYPE",SPECIES="$SPECIES",RNA_FILE_NAME="$RNA_FILE_NAME",ATAC_FILE_NAME="$ATAC_FILE_NAME",GROUND_TRUTH_FILE="$GROUND_TRUTH_FILE" \
        --output="LOGS/${CELL_TYPE}_logs/${SAMPLE_NAME}_logs/custom_grn_${CELL_TYPE}_${SAMPLE_NAME}.out" \
        --error="LOGS/${CELL_TYPE}_logs/${SAMPLE_NAME}_logs/custom_grn_${CELL_TYPE}_${SAMPLE_NAME}.err" \
        --job-name="custom_grn_method_${CELL_TYPE}_${SAMPLE_NAME}" \
        /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/src/bash_scripts/main_pipeline.sh
}

run_macrophage() {
    local CELL_TYPE="macrophage"
    local SAMPLE_NAMES=(
        # "macrophage_buffer1_filtered"
        "macrophage_buffer2_filtered"
        "macrophage_buffer3_filtered"
        "macrophage_buffer4_filtered"
        )
    local SPECIES="hg38"

    for SAMPLE_NAME in "${SAMPLE_NAMES[@]}"; do
        local RNA_FILE_NAME="${SAMPLE_NAME}_RNA.csv"
        local ATAC_FILE_NAME="${SAMPLE_NAME}_ATAC.csv"
        local GROUND_TRUTH_FILE="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN204_ChIPSeq_ChIPAtlas_Human_Macrophages.tsv"
        
        # Submit the job for each sample
        submit_job \
            "$SAMPLE_NAME" \
            "$CELL_TYPE" \
            "$SPECIES" \
            "$RNA_FILE_NAME" \
            "$ATAC_FILE_NAME" \
            "$GROUND_TRUTH_FILE"
    done
}

run_mESC(){
    local CELL_TYPE="mESC"
    local SAMPLE_NAMES=(
        # "filtered_L2_E7.5_rep1"
        # "filtered_L2_E7.5_rep2"
        # "filtered_L2_E7.75_rep1"
        "filtered_L2_E8.0_rep1"
        "filtered_L2_E8.0_rep2"
        "filtered_L2_E8.5_CRISPR_T_KO"
        "filtered_L2_E8.5_CRISPR_T_WT"
        "filtered_L2_E8.5_rep1"
        "filtered_L2_E8.5_rep2"
        "filtered_L2_E8.75_rep1"
        "filtered_L2_E8.75_rep2"
    )
    local SPECIES="mm10"

    # Submit each SAMPLE_NAME as a separate job
    for SAMPLE_NAME in "${SAMPLE_NAMES[@]}"; do
        local RNA_FILE_NAME="mESC_${SAMPLE_NAME}_RNA.csv"
        local ATAC_FILE_NAME="mESC_${SAMPLE_NAME}_ATAC.csv"
        local GROUND_TRUTH_FILE="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"

        # Submit the job for each sample
        submit_job \
            "$SAMPLE_NAME" \
            "$CELL_TYPE" \
            "$SPECIES" \
            "$RNA_FILE_NAME" \
            "$ATAC_FILE_NAME" \
            "$GROUND_TRUTH_FILE"

    done
}

run_K562(){
    local CELL_TYPE="K562"
    local SAMPLE_NAMES=(
        "K562_human_filtered"
    )
    local SPECIES="hg38"

    # Submit each SAMPLE_NAME as a separate job
    for SAMPLE_NAME in "${SAMPLE_NAMES[@]}"; do
        local RNA_FILE_NAME="${SAMPLE_NAME}_RNA.csv"
        local ATAC_FILE_NAME="${SAMPLE_NAME}_ATAC.csv"
        local GROUND_TRUTH_FILE="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"

        # Submit the job for each sample
        submit_job \
            "$SAMPLE_NAME" \
            "$CELL_TYPE" \
            "$SPECIES" \
            "$RNA_FILE_NAME" \
            "$ATAC_FILE_NAME" \
            "$GROUND_TRUTH_FILE"
    done
}

# run_mESC
# run_K562
run_macrophage