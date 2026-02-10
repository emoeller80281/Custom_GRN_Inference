#!/bin/bash -l
#SBATCH --job-name=auroc_testing
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%A/%x_%A_%a.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%A/%x_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --array=0-1%4

set -euo pipefail

cd "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"

source .venv/bin/activate

EXPERIMENT_DIR=${EXPERIMENT_DIR:-/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments}

EXPERIMENT_LIST=(
    # "mESC_no_scale_linear|model_training_128_10k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_1k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_5k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_10k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_10k_metacells_6_layers|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_10k_metacells_8_heads|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_15k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_192_50k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_256_15k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_256_20k_metacells|trained_model.pt"
    # "mESC_no_scale_linear|model_training_320_10k_metacells|trained_model.pt"
    # "mESC_large_neighborhood_count_filter|model_training_001|trained_model.pt"
    # "mESC_large_neighborhood|model_training_001|trained_model.pt"
    # "mESC_small_neighborhood|model_training_001|trained_model.pt"
    # "mESC_small_neighborhood_high_self_weight|model_training_001|trained_model.pt"
    # "mESC_slower_dist_decay|model_training_001|trained_model.pt"
    # "mESC_max_dist_bias|model_training_002|trained_model.pt"
    # "mESC_slow_decay_max_dist|model_training_001|trained_model.pt"
    # "mESC_filter_lowest_ten_pct|model_training_003|trained_model.pt"
    # "mESC_lower_peak_threshold|model_training_001|trained_model.pt"
    # "mESC_no_filter_to_nearest_gene|model_training_003|trained_model.pt"
    # "mESC_smaller_window_size|model_training_004|trained_model.pt"
    # "mESC_larger_window_size|model_training_002|trained_model.pt"
    # "mESC_lower_max_peak_dist|model_training_002|trained_model.pt"
    # "mESC_higher_max_peak_dist|model_training_002|trained_model.pt"
    # "mESC_test_new_pipeline|model_training_002|trained_model.pt"
    # "mESC_slow_decay_filter_ten_pct|model_training_001|trained_model.pt"
    # "mESC_fast_decay_large_window|model_training_001|trained_model.pt"
    # "mESC_slow_decay_small_window|model_training_001|trained_model.pt"
    # "mESC_fewer_pca_components|model_training_001|trained_model.pt"
    # "mESC_more_pca_components|model_training_001|trained_model.pt"
    # "mESC_one_hop_diffusion|model_training_001|trained_model.pt"
    # "mESC_two_hop_diffusion|model_training_001|trained_model.pt"
    # "mESC_one_hop_large_neighborhood|model_training_001|trained_model.pt"
    # "mESC_strict_genes_lenient_peaks|model_training_001|trained_model.pt"
    # "mESC_lenient_genes_strict_peaks|model_training_001|trained_model.pt"
    # "mESC_strict_filter_twenty_pct|model_training_001|trained_model.pt"
    # "mESC_promoter_2kb|model_training_001|trained_model.pt"
    # "mESC_promoter_5kb|model_training_001|trained_model.pt"
    # "mESC_very_short_range|model_training_001|trained_model.pt"
    # "mESC_long_range_enhancers|model_training_001|trained_model.pt"
    # "mESC_slow_decay_long_range_two_hop|model_training_001|trained_model.pt"
    # "mESC_slow_decay_long_range_zero_hops|model_training_001|trained_model.pt"
    # "mESC_decay_30k_long_range_two_hop|model_training_001|trained_model.pt"
    # "mESC_decay_50k_long_range_two_hop|model_training_001|trained_model.pt"
    # "mESC_decay_75k_long_range_two_hop|model_training_001|trained_model.pt"
    # "mESC_promoter_only_5kb_two_hop|model_training_001|trained_model.pt"
    # "mESC_promoter_only_10kb_two_hop|model_training_001|trained_model.pt"
    # "mESC_promoter_only_2kb_two_hop|model_training_001|trained_model.pt"
    # "mESC_two_hop_hvg_small|model_training_001|trained_model.pt"
    # "mESC_two_hop_no_hvg_small|model_training_002|trained_model.pt"

    # "Macrophage_base_settings|model_training_006|trained_model.pt"
    # "Macrophage_model_d_128_ff_512|model_training_001|trained_model.pt"
    # "Macrophage_small_batch_size|model_training_002|trained_model.pt"
    # "Macrophage_loose_1_pct_filtering|model_training_001|trained_model.pt"
    # "Macrophage_strict_10_pct_filtering|model_training_001|trained_model.pt"
    # "Macrophage_40k_distance_scale_factor|model_training_001|trained_model.pt"
    # "Macrophage_10k_distance_scale_factor|model_training_002|trained_model.pt"
    # "Macrophage_150k_max_peak_dist|model_training_001|trained_model.pt"
    # "Macrophage_50k_max_peak_dist|model_training_001|trained_model.pt"
    # "Macrophage_slow_decay_long_range_two_hop|model_training_001|trained_model.pt"
    # "Macrophage_slow_decay_long_range|model_training_001|trained_model.pt"
    # "Macrophage_zero_hops|model_training_001|trained_model.pt"
    # "Macrophage_two_hops|model_training_001|trained_model.pt"
    # "Macrophage_loose_1_pct_filter_50_min_per_cell|model_training_001|trained_model.pt"
    # "Macrophage_small_model_loose_1_pct_filtering|model_training_001|trained_model.pt"
    # "Macrophage_two_hops_small_batch|model_training_001|trained_model.pt"
    # "Macrophage_two_hops_150k_max_peak_dist|model_training_001|trained_model.pt"
    # "Macrophage_two_hops_slow_decay_long_range|model_training_001|trained_model.pt"
    # "Macrophage_two_hops_slow_decay_long_range_small_batch|model_training_001|trained_model.pt"
    # "Macrophage_three_hops_small_batch|model_training_001|trained_model.pt"
    # "Macrophage_two_hops_50k_max_peak_dist|model_training_001|trained_model.pt"
    # "Macrophage_two_hops_10k_distance_scale_factor|model_training_001|trained_model.pt"
    # "Macrophage_two_hops_40k_distance_scale_factor|model_training_001|trained_model.pt"
    # "Macrophage_two_hops_loose_1_pct_filtering|model_training_001|trained_model.pt"
    # "Macrophage_two_hops_moderate_5_pct_filtering_small_batch|model_training_001|trained_model.pt"
    # "Macrophage_small_model_two_hops_long_range_small_batch|model_training_002|trained_model.pt"
    # "Macrophage_best_filter_long_range_2_hop_small|model_training_001|trained_model.pt"
    # "Macrophage_best_filter_long_range_3_hop_small|model_training_001|trained_model.pt"
    # "Macrophage_best_filter_long_range_2_hop_small_max_bias|model_training_001|trained_model.pt"
    # "Macrophage_best_filter_long_range_2_hop_small_500bp_window|model_training_001|trained_model.pt"
    # "Macrophage_best_filter_long_range_2_hop_small_1500bp_window|model_training_001|trained_model.pt"
    # "Macrophage_best_filter_long_range_2_hop_small_fewer_neighbors|model_training_001|trained_model.pt"
    # "Macrophage_best_filter_long_range_2_hop_tiny|model_training_001|trained_model.pt"


    # "K562_base_settings|model_training_001|trained_model.pt"
    # "K562_model_d_128_ff_512|model_training_001|trained_model.pt"
    # "K562_small_batch_size|model_training_001|trained_model.pt"
    # "K562_loose_1_pct_filtering|model_training_001|trained_model.pt"
    # "K562_strict_10_pct_filtering|model_training_001|trained_model.pt"
    # "K562_40k_distance_scale_factor|model_training_001|trained_model.pt"
    # "K562_10k_distance_scale_factor|model_training_001|trained_model.pt"
    # "K562_150k_max_peak_dist|model_training_002|trained_model.pt"
    # "K562_50k_max_peak_dist|model_training_001|trained_model.pt"
    # "K562_slow_decay_long_range_two_hop|model_training_001|trained_model.pt"
    # "K562_slow_decay_long_range|model_training_001|trained_model.pt"
    # "K562_zero_hops|model_training_001|trained_model.pt"
    # "K562_two_hops|model_training_001|trained_model.pt"
    # "K562_loose_1_pct_filter_50_min_per_cell|model_training_001|trained_model.pt"
    # "K562_small_model_loose_1_pct_filtering|model_training_001|trained_model.pt"
    # "K562_two_hops_small_batch|model_training_002|trained_model.pt"
    # "K562_two_hops_150k_max_peak_dist|model_training_001|trained_model.pt"
    # "K562_two_hops_slow_decay_long_range|model_training_002|trained_model.pt"
    # "K562_two_hops_slow_decay_long_range_small_batch|model_training_001|trained_model.pt"
    # "K562_three_hops_small_batch|model_training_001|trained_model.pt"
    # "K562_two_hops_50k_max_peak_dist|model_training_001|trained_model.pt"
    # "K562_two_hops_10k_distance_scale_factor|model_training_001|trained_model.pt"
    # "K562_two_hops_40k_distance_scale_factor|model_training_001|trained_model.pt"
    # "K562_two_hops_loose_1_pct_filtering|model_training_001|trained_model.pt"
    # "K562_two_hops_moderate_5_pct_filtering_small_batch|model_training_001|trained_model.pt"
    # "K562_small_model_two_hops_long_range_small_batch|model_training_001|trained_model.pt"
    # "K562_stability_test_01|model_training_001|trained_model.pt"
    # "K562_stability_test_02|model_training_001|trained_model.pt"
    # "K562_stability_test_03|model_training_001|trained_model.pt"
    # "K562_stability_test_04|model_training_001|trained_model.pt"
    # "K562_stability_test_05|model_training_001|trained_model.pt"
    # "K562_two_hops_slow_decay_long_range_no_hvg|model_training_001|trained_model.pt"
    # "K562_hvg_filter_none|model_training_001|trained_model.pt"

    # "mESC_E7.5_rep1_hvg_filter_only_rna|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.6|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.5|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.4|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.3|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.2|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.1|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.05|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.01|model_training_001|trained_model.pt"

    # "mESC_E7.5_rep2_hvg_filter_only_rna|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.6|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.5|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.4|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.3|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.2|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.1|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.05|model_training_001|trained_model.pt"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.01|model_training_001|trained_model.pt"

    # "mESC_E8.5_rep1_hvg_filter_only_rna|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.6|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.5|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.4|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.3|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.2|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.1|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.05|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.01|model_training_001|trained_model.pt"

    # "mESC_E8.5_rep2_hvg_filter_only_rna|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.6|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.5|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.4|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.3|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.2|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.1|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.05|model_training_001|trained_model.pt"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.01|model_training_001|trained_model.pt"

    # "mESC_1_sample_hvg_filter_disp_0.01|model_training_001|trained_model.pt"
    # "mESC_2_sample_hvg_filter_disp_0.01|model_training_001|trained_model.pt"
    # "mESC_3_sample_hvg_filter_disp_0.01|model_training_001|trained_model.pt"
    # "mESC_4_sample_hvg_filter_disp_0.01|model_training_001|trained_model.pt"

    # "Macrophage_buffer_1_hvg_filter_only_rna|model_training_001|trained_model.pt"
    # "Macrophage_buffer_1_hvg_filter_disp_0.6|model_training_001|trained_model.pt"
    # "Macrophage_buffer_1_hvg_filter_disp_0.5|model_training_001|trained_model.pt"
    # "Macrophage_buffer_1_hvg_filter_disp_0.4|model_training_001|trained_model.pt"
    # "Macrophage_buffer_1_hvg_filter_disp_0.3|model_training_001|trained_model.pt"
    # "Macrophage_buffer_1_hvg_filter_disp_0.2|model_training_001|trained_model.pt"
    # "Macrophage_buffer_1_hvg_filter_disp_0.1|model_training_001|trained_model.pt"
    # "Macrophage_buffer_1_hvg_filter_disp_0.05|model_training_001|trained_model.pt"
    # "Macrophage_buffer_1_hvg_filter_disp_0.01|model_training_001|trained_model.pt"

    # "Macrophage_buffer_2_hvg_filter_only_rna|model_training_001|trained_model.pt"
    # "Macrophage_buffer_2_hvg_filter_disp_0.6|model_training_001|trained_model.pt"
    # "Macrophage_buffer_2_hvg_filter_disp_0.5|model_training_001|trained_model.pt"
    # "Macrophage_buffer_2_hvg_filter_disp_0.4|model_training_001|trained_model.pt"
    # "Macrophage_buffer_2_hvg_filter_disp_0.3|model_training_001|trained_model.pt"
    # "Macrophage_buffer_2_hvg_filter_disp_0.2|model_training_001|trained_model.pt"
    # "Macrophage_buffer_2_hvg_filter_disp_0.1|model_training_001|trained_model.pt"
    # "Macrophage_buffer_2_hvg_filter_disp_0.05|model_training_001|trained_model.pt"
    # "Macrophage_buffer_2_hvg_filter_disp_0.01|model_training_001|trained_model.pt"

    # "Macrophage_buffer_3_hvg_filter_none|model_training_001|trained_model.pt"
    # "Macrophage_buffer_3_hvg_filter_only_rna|model_training_001|trained_model.pt"
    # "Macrophage_buffer_3_hvg_filter_disp_0.6|model_training_001|trained_model.pt"
    # "Macrophage_buffer_3_hvg_filter_disp_0.5|model_training_001|trained_model.pt"
    # "Macrophage_buffer_3_hvg_filter_disp_0.4|model_training_001|trained_model.pt"
    # "Macrophage_buffer_3_hvg_filter_disp_0.3|model_training_001|trained_model.pt"
    # "Macrophage_buffer_3_hvg_filter_disp_0.2|model_training_001|trained_model.pt"
    # "Macrophage_buffer_3_hvg_filter_disp_0.1|model_training_001|trained_model.pt"
    # "Macrophage_buffer_3_hvg_filter_disp_0.05|model_training_001|trained_model.pt"
    # "Macrophage_buffer_3_hvg_filter_disp_0.01|model_training_001|trained_model.pt"

    # "Macrophage_buffer_4_hvg_filter_none|model_training_001|trained_model.pt"
    # "Macrophage_buffer_4_hvg_filter_only_rna|model_training_001|trained_model.pt"
    # "Macrophage_buffer_4_hvg_filter_disp_0.6|model_training_001|trained_model.pt"
    # "Macrophage_buffer_4_hvg_filter_disp_0.5|model_training_001|trained_model.pt"
    # "Macrophage_buffer_4_hvg_filter_disp_0.4|model_training_001|trained_model.pt"
    # "Macrophage_buffer_4_hvg_filter_disp_0.3|model_training_001|trained_model.pt"
    # "Macrophage_buffer_4_hvg_filter_disp_0.2|model_training_001|trained_model.pt"
    # "Macrophage_buffer_4_hvg_filter_disp_0.1|model_training_001|trained_model.pt"
    # "Macrophage_buffer_4_hvg_filter_disp_0.05|model_training_001|trained_model.pt"
    # "Macrophage_buffer_4_hvg_filter_disp_0.01|model_training_001|trained_model.pt"

    # "Macrophage_buffer_12_hvg_filter_disp_0.1|model_training_001|trained_model.pt"
    # "Macrophage_buffer_123_hvg_filter_disp_0.1|model_training_001|trained_model.pt"
    # "Macrophage_buffer_1234_hvg_filter_disp_0.1|model_training_001|trained_model.pt"



)

# DATASET_TYPE="mESC"
# SAMPLE_NAMES="E8.5_rep2" # E7.5_rep2 E8.5_rep1 E8.5_rep2

DATASET_TYPE="macrophage"
SAMPLE_NAMES="buffer_1"

# DATASET_TYPE="k562"
# SAMPLE_NAMES="K562"


# ==========================================
#        EXPERIMENT SELECTION
# ==========================================
# Get the current experiment based on SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ ${TASK_ID} -ge ${#EXPERIMENT_LIST[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${TASK_ID}) exceeds number of experiments (${#EXPERIMENT_LIST[@]})"
    exit 1
fi

EXPERIMENT_CONFIG="${EXPERIMENT_LIST[$TASK_ID]}"

# Parse experiment configuration
IFS='|' read -r EXPERIMENT_NAME TRAINING_NUM MODEL_FILE <<< "$EXPERIMENT_CONFIG"

echo ""
echo "=========================================="
echo "  EXPERIMENT: ${EXPERIMENT_NAME}"
echo "  TRAINING_NUM: ${TRAINING_NUM}"
echo "  MODEL_FILE ID: ${MODEL_FILE}"
echo "  TASK ID: ${TASK_ID}"
echo "=========================================="
echo ""

echo "Running AUROC Testing"
poetry run python ./src/multiomic_transformer/utils/auroc_testing.py \
    --experiment "$EXPERIMENT_NAME" \
    --training_num "$TRAINING_NUM" \
    --experiment_dir "$EXPERIMENT_DIR" \
    --model_file "$MODEL_FILE" \
    --dataset_type "$DATASET_TYPE" \
    --sample_name_list $SAMPLE_NAMES

echo "Plotting Training Figures"
poetry run python ./src/multiomic_transformer/utils/plotting.py \
    --experiment "$EXPERIMENT_NAME" \
    --training_num "$TRAINING_NUM" \
    --experiment_dir "$EXPERIMENT_DIR" \
    --model_file "$MODEL_FILE"

echo "finished"