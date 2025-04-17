#!/bin/bash -l
#SBATCH --partition compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task 10
#SBATCH --mem 64G
#SBATCH -o /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/parameter_testing.log
#SBATCH -e /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/LOGS/parameter_testing.err

set -euo pipefail
conda activate my_env

BASE_DIR=$(readlink -f "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
NUM_CPU=10

# define one entry per sample
declare -a SAMPLES=(K562) # macrophage mESC
declare -a INFERRED_NETS=(
  "$BASE_DIR/output/K562/K562_human_filtered/inferred_grns/inferred_network_w_string.csv"
#   "$BASE_DIR/output/macrophage/macrophage_buffer1_filtered/inferred_grns/inferred_network_w_string.csv"
#   "$BASE_DIR/output/mESC/filtered_L2_E7.5_rep1/inferred_grns/inferred_network_w_string.csv"
)
declare -a GROUND_TRUTHS=(
  "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN117_ChIPSeq_PMID37486787_Human_K562.tsv"
#   "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN204_ChIPSeq_ChIPAtlas_Human_Macrophages.tsv"
#   "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"
)
declare -a FIG_DIRS=(
  "$BASE_DIR/figures/hg38/K562_human_filtered"
#   "$BASE_DIR/figures/hg38/macrophage_buffer1_filtered"
#   "$BASE_DIR/figures/mm10/filtered_L2_E7.5_rep1"
)

for i in "${!SAMPLES[@]}"; do
  SAMPLE=${SAMPLES[$i]}
  INFERRED_NET=${INFERRED_NETS[$i]}
  GROUND_TRUTH=${GROUND_TRUTHS[$i]}
  FIG_DIR=${FIG_DIRS[$i]}

  echo "=== Running parameter search for $SAMPLE ==="
  mkdir -p "$FIG_DIR/parameter_search"

  python "${BASE_DIR}/src/testing_scripts/xgboost_parameter_testing.py" \
    --ground_truth_file  "$GROUND_TRUTH" \
    --inferred_network_file  "$INFERRED_NET" \
    --fig_dir  "$FIG_DIR/parameter_search" \
    --cpu_count  $NUM_CPU

  echo "=== Done $SAMPLE ==="
done
