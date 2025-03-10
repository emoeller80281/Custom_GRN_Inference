#!/bin/bash -l

#SBATCH --partition compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --mem 128G
#SBATCH --output=LOGS/apply_sc_trained_rf_model.log
#SBATCH --error=LOGS/apply_sc_trained_rf_model.err

source activate my_env

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/"

MESC_OUTPUT="$PROJECT_DIR/output/mESC/filtered_L2_E7.5_rep1/"
MESC_GROUND_TRUTH="$PROJECT_DIR/ground_truth_files/RN111.tsv"

MACROPHAGE_OUTPUT="$PROJECT_DIR/output/macrophage/macrophage_buffer1_filtered"
MACROPHAGE_GROUND_TRUTH="$PROJECT_DIR/ground_truth_files/RN204_macrophage_ground_truth.tsv"

K562_OUTPUT="$PROJECT_DIR/output/K562/K562_human_filtered"
K562_GROUND_TRUTH="$PROJECT_DIR/ground_truth_files/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"


determine_num_cpus() {
    if [ -z "${SLURM_CPUS_PER_TASK:-}" ]; then
        if command -v nproc &> /dev/null; then
            TOTAL_CPUS=$(nproc --all)
            case $TOTAL_CPUS in
                [1-15]) IGNORED_CPUS=1 ;;  # Reserve 1 CPU for <=15 cores
                [16-31]) IGNORED_CPUS=2 ;; # Reserve 2 CPUs for <=31 cores
                *) IGNORED_CPUS=4 ;;       # Reserve 4 CPUs for >=32 cores
            esac
            NUM_CPU=$((TOTAL_CPUS - IGNORED_CPUS))
            echo "[INFO] Running locally. Detected $TOTAL_CPUS CPUs, reserving $IGNORED_CPUS for system tasks. Using $NUM_CPU CPUs."
        else
            NUM_CPU=1  # Fallback
            echo "[INFO] Running locally. Unable to detect CPUs, defaulting to $NUM_CPU CPU."
        fi
    else
        NUM_CPU=${SLURM_CPUS_PER_TASK}
        echo "[INFO] Running on SLURM. Number of CPUs allocated: ${NUM_CPU}"
    fi
}

determine_num_cpus

# ----- K562 BULK MODEL -----
# STEP 1: Create a bulk inferred network using averaged TF, TG, and peak expression
echo "Running Step040.tf_to_tg_score.py to create bulk model scores dataframe for mESC"
python3 "$PROJECT_DIR/src/python_scripts/Step040.tf_to_tg_score.py" \
    --rna_data_file "$PROJECT_DIR/input/K562/K562_human_filtered/K562_human_filtered_RNA.csv" \
    --atac_data_file "$PROJECT_DIR/input/K562/K562_human_filtered/K562_human_filtered_ATAC.csv" \
    --output_dir "$K562_OUTPUT" \
    --fig_dir "$PROJECT_DIR/figures/K562/K562_human_filtered" \
    --bulk_or_cell "bulk"

# STEP 2: Train a random forest on the bulk inferred network
echo "Training the random forest on the bulk model"
python3 "$PROJECT_DIR/src/python_scripts/Step050.train_random_forest.py" \
        --ground_truth_file "$K562_GROUND_TRUTH" \
        --output_dir "$K562_OUTPUT" \
        --fig_dir "$PROJECT_DIR/figures/K562/K562_human_filtered"

# ----- MACROPHAGE BULK MODEL -----
# STEP 1: Create a bulk inferred network using averaged TF, TG, and peak expression
echo "Running Step040.tf_to_tg_score.py to create bulk model scores dataframe for mESC"
python3 "$PROJECT_DIR/src/python_scripts/Step040.tf_to_tg_score.py" \
    --rna_data_file "$PROJECT_DIR/input/macrophage/macrophage_buffer1_filtered/macrophage_buffer1_filtered_RNA.csv" \
    --atac_data_file "$PROJECT_DIR/input/macrophage/macrophage_buffer1_filtered/macrophage_buffer1_filtered_ATAC.csv" \
    --output_dir "$MACROPHAGE_OUTPUT" \
    --fig_dir "$PROJECT_DIR/figures/hg38/macrophage_buffer1_filtered" \
    --bulk_or_cell "bulk"

# STEP 2: Train a random forest on the bulk inferred network
echo "Training the random forest on the bulk model"
python3 "$PROJECT_DIR/src/python_scripts/Step050.train_random_forest.py" \
        --ground_truth_file "$MACROPHAGE_GROUND_TRUTH" \
        --output_dir "$MACROPHAGE_OUTPUT" \
        --fig_dir "$PROJECT_DIR/figures/hg38/macrophage_buffer1_filtered"

# STEP 1: Create a bulk inferred network using averaged TF, TG, and peak expression
# echo "Running Step040.tf_to_tg_score.py to create bulk model scores dataframe for mESC"
# python3 "$PROJECT_DIR/src/python_scripts/Step040.tf_to_tg_score.py" \
#     --rna_data_file "$PROJECT_DIR/input/mESC/filtered_L2_E7.5_rep1/mESC_filtered_L2_E7.5_rep1_RNA.csv" \
#     --atac_data_file "$PROJECT_DIR/input/mESC/filtered_L2_E7.5_rep1/mESC_filtered_L2_E7.5_rep1_ATAC.csv" \
#     --output_dir "$MESC_OUTPUT" \
#     --fig_dir "$PROJECT_DIR/figures/mm10/filtered_L2_E7.5_rep1" \
#     --bulk_or_cell "bulk"

# STEP 2: Train a random forest on the bulk inferred network
# echo "Training the random forest on the bulk model"
# python3 "$PROJECT_DIR/src/python_scripts/Step050.train_random_forest.py" \
#         --ground_truth_file "$MESC_GROUND_TRUTH" \
#         --output_dir "$MESC_OUTPUT" \
#         --fig_dir "$PROJECT_DIR/figures/mm10/filtered_L2_E7.5_rep1"

# STEP 3: Create cell-level scores dataframes using cell-level TF, TG and peak expression values
# echo "Running Step040.tf_to_tg_score.py to create cell-level score dataframes for mESC"
# python3 "$PROJECT_DIR/src/python_scripts/Step040.tf_to_tg_score.py" \
#     --rna_data_file "$PROJECT_DIR/input/mESC/filtered_L2_E7.5_rep1/mESC_filtered_L2_E7.5_rep1_RNA.csv" \
#     --atac_data_file "$PROJECT_DIR/input/mESC/filtered_L2_E7.5_rep1/mESC_filtered_L2_E7.5_rep1_ATAC.csv" \
#     --output_dir "$MESC_OUTPUT/" \
#     --cell_level_net_dir "$MESC_OUTPUT/cell_networks_raw" \
#     --fig_dir "$PROJECT_DIR/figures/mm10/filtered_L2_E7.5_rep1" \
#     --num_cpu "$NUM_CPU" \
#     --num_cells 25 \
#     --bulk_or_cell "cell"

# STEP 4: Make TF to TG predictions on the cell-level scores using the random forest model
# echo "Applying trained random forest model from mESC to cell-level score dataframes for mESC"
# python3 "$PROJECT_DIR/src/testing_scripts/apply_sc_trained_rf_model.py" \
#     --output_dir "$MESC_OUTPUT/" \
#     --model "$MESC_OUTPUT/trained_random_forest_model.pkl" \
#     --cell_level_net_dir  "$MESC_OUTPUT/cell_networks_raw" \
#     --num_cpu "$NUM_CPU"

# STEP 5: Use the cell-level predictions to determine how many of the cells agreed on if an edge should exist.
#         Then, re-train a random forest model using the increased context gained from the consensus
# echo "Re-training the random forest model with single-cell random forest score consensus"
# python3 "$PROJECT_DIR/src/testing_scripts/consensus_scoring.py" \
#     --output_dir "$MESC_OUTPUT/" \
#     --ground_truth_file "$MESC_GROUND_TRUTH" \
#     --fig_dir "$PROJECT_DIR/figures/mm10/filtered_L2_E7.5_rep1" \
#     --cell_rf_net_dir  "$MESC_OUTPUT/cell_networks_rf" \
#     --num_cpu "$NUM_CPU"

# STEP 6: Create new cell-level scores dataframes using cell-level TF, TG and peak expression values
# echo "Running Step040.tf_to_tg_score.py to create cell-level score dataframes for mESC"
# python3 "$PROJECT_DIR/src/python_scripts/Step040.tf_to_tg_score.py" \
#     --rna_data_file "$PROJECT_DIR/input/mESC/filtered_L2_E7.5_rep1/mESC_filtered_L2_E7.5_rep1_RNA.csv" \
#     --atac_data_file "$PROJECT_DIR/input/mESC/filtered_L2_E7.5_rep1/mESC_filtered_L2_E7.5_rep1_ATAC.csv" \
#     --output_dir "$MESC_OUTPUT/cell_networks_raw_round_two" \
#     --fig_dir "$PROJECT_DIR/figures/mm10/filtered_L2_E7.5_rep1" \
#     --num_cpu "$NUM_CPU" \
#     --num_cells 25 \
#     --bulk_or_cell "cell"

# STEP 7: Make a second round of TF to TG prediciton on new cell-level scores using the better random forest model
# echo "Running Step040.tf_to_tg_score.py to create new cell-level scores using the refined random forest"
# python3 "$PROJECT_DIR/src/testing_scripts/apply_sc_trained_rf_model.py" \
#     --output_dir "$MESC_OUTPUT/cell_networks_rf_refined" \
#     --model "$MESC_OUTPUT/trained_random_forest_model.pkl" \
#     --cell_level_net_dir  "$MESC_OUTPUT/cell_networks_raw_round_two" \
#     --num_cpu "$NUM_CPU"








