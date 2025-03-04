#!/bin/bash -l

#SBATCH --partition compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task 2
#SBATCH --mem 32G
#SBATCH --output=LOGS/apply_trained_rf_model.log
#SBATCH --error=LOGS/apply_trained_rf_model.err

source activate my_env

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/"

MESC_OUTPUT="$PROJECT_DIR/output/mESC/filtered_L2_E7.5_rep1"
MACROPHAGE_OUTPUT="$PROJECT_DIR/output/macrophage"
K562_OUTPUT="$PROJECT_DIR/output/K562/K562_human_filtered"

# # mESC vs macrophage
# python3 "$PROJECT_DIR/src/testing_scripts/apply_trained_rf_model.py" \
#     --output_dir "$MESC_OUTPUT/filtered_L2_E7.5_rep1" \
#     --model "$MESC_OUTPUT/filtered_L2_E7.5_rep1/trained_random_forest_model.pkl" \
#     --target "$MACROPHAGE_OUTPUT/macrophage_buffer1_filtered/inferred_network_raw.pkl" \
#     --save_name "mESC_vs_macrophage"

# # mESC vs K562
# python3 "$PROJECT_DIR/src/testing_scripts/apply_trained_rf_model.py" \
#     --output_dir "$MESC_OUTPUT/filtered_L2_E7.5_rep1" \
#     --model "$MESC_OUTPUT/filtered_L2_E7.5_rep1/trained_random_forest_model.pkl" \
#     --target "$K562_OUTPUT/inferred_network_raw.pkl" \
#     --save_name "mESC_vs_K562"

# # Macrophage vs K562
# python3 "$PROJECT_DIR/src/testing_scripts/apply_trained_rf_model.py" \
#     --output_dir "$MACROPHAGE_OUTPUT/macrophage_buffer1_filtered" \
#     --model "$MACROPHAGE_OUTPUT/macrophage_buffer1_filtered/trained_random_forest_model.pkl" \
#     --target "$K562_OUTPUT/inferred_network_raw.pkl" \
#     --save_name "macrophage_vs_K562"

# # Macrophage vs mESC
# python3 "$PROJECT_DIR/src/testing_scripts/apply_trained_rf_model.py" \
#     --output_dir "$MACROPHAGE_OUTPUT/macrophage_buffer1_filtered" \
#     --model "$MACROPHAGE_OUTPUT/macrophage_buffer1_filtered/trained_random_forest_model.pkl" \
#     --target "$MESC_OUTPUT/filtered_L2_E7.5_rep1/inferred_network_raw.pkl" \
#     --save_name "macrophage_vs_mESC"

# # K562 vs macrophage
# python3 "$PROJECT_DIR/src/testing_scripts/apply_trained_rf_model.py" \
#     --output_dir "$K562_OUTPUT" \
#     --model "$K562_OUTPUT/trained_random_forest_model.pkl" \
#     --target "$MACROPHAGE_OUTPUT/macrophage_buffer1_filtered/inferred_network_raw.pkl" \
#     --save_name "K562_vs_macrophage"

# K562 vs mESC
# python3 "$PROJECT_DIR/src/testing_scripts/apply_trained_rf_model.py" \
#     --output_dir "$K562_OUTPUT" \
#     --model "$K562_OUTPUT/trained_random_forest_model.pkl" \
#     --target "$MESC_OUTPUT/filtered_L2_E7.5_rep1/inferred_network_raw.pkl" \
#     --save_name "K562_vs_mESC"

# mESC sample vs sample
python3 "$PROJECT_DIR/src/testing_scripts/apply_trained_rf_model.py" \
    --output_dir "$MESC_OUTPUT/filtered_L2_E7.5_rep1" \
    --model "$MESC_OUTPUT/filtered_L2_E7.5_rep1/trained_random_forest_model.pkl" \
    --target  "$MESC_OUTPUT/filtered_L2_E7.5_rep2/inferred_network_raw.pkl" \
    --save_name "mESC1_vs_mESC2"

# Macrophage sample vs sample
python3 "$PROJECT_DIR/src/testing_scripts/apply_trained_rf_model.py" \
    --output_dir "$MACROPHAGE_OUTPUT/macrophage_buffer1_filtered" \
    --model "$MACROPHAGE_OUTPUT/macrophage_buffer1_filtered/trained_random_forest_model.pkl" \
    --target  "$MACROPHAGE_OUTPUT/macrophage_buffer2_filtered/inferred_network_raw.pkl" \
    --save_name "macrophage1_vs_macrophage2"




