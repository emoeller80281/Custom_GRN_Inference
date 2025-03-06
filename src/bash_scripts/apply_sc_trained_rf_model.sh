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
MACROPHAGE_OUTPUT="$PROJECT_DIR/output/macrophage"
K562_OUTPUT="$PROJECT_DIR/output/K562/K562_human_filtered"

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

# echo "Running Step040.tf_to_tg_score.py to create cell-level score dataframes for mESC"
# python3 "$PROJECT_DIR/src/python_scripts/Step040.tf_to_tg_score.py" \
#     --rna_data_file "$PROJECT_DIR/input/mESC/filtered_L2_E7.5_rep1/mESC_filtered_L2_E7.5_rep1_RNA.csv" \
#     --output_dir "$MESC_OUTPUT" \
#     --fig_dir "$PROJECT_DIR/figures/mm10/filtered_L2_E7.5_rep1" \
#     --num_cpu "$NUM_CPU" \
#     --num_cells 50

echo "Applying trained random forest model from mESC to cell-level score dataframes for mESC"
python3 "$PROJECT_DIR/src/testing_scripts/apply_sc_trained_rf_model.py" \
    --output_dir "$MESC_OUTPUT/" \
    --model "$MESC_OUTPUT/trained_random_forest_model.pkl" \
    --cell_level_net_dir  "$MESC_OUTPUT/cell_networks_raw" \
    --num_cpu "$NUM_CPU"






