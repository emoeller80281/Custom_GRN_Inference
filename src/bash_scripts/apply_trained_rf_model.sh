#!/bin/bash -l

source activate my_env

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/"

python3 "$PROJECT_DIR/src/testing_scripts/apply_trained_rf_model.py" \
    --output_dir "$PROJECT_DIR/output/mESC/filtered_L2_E7.5_rep1" \
    --model "$PROJECT_DIR/output/mESC/filtered_L2_E7.5_rep1/trained_random_forest_model.pkl" \
    --target "$PROJECT_DIR/output/macrophage/macrophage_buffer1_filtered/inferred_network_raw.pkl" \
    --save_name "mESC_vs_macrophage"

