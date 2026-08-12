#!/bin/bash
#SBATCH --job-name=scrna_report_mESC
#SBATCH --output=LOGS/scrna_annotation/%x_%A_%a.log
#SBATCH --error=LOGS/scrna_annotation/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --array=0-10

# Build the per-sample HTML report from an annotate_scrna_celltypes.py output dir.
# Run after run_scrna_annotation_mESC.sh has finished.

set -eo pipefail
source activate my_env
set -u

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
OUT_ROOT="${PROJECT_DIR}/data/processed/mESC"

SAMPLES=(
    E7.5_rep1
    E7.5_rep2
    E7.75_rep1
    E8.0_rep1
    E8.0_rep2
    E8.5_rep1
    E8.5_rep2
    E8.5_CRISPR_T_WT
    E8.5_CRISPR_T_KO
    E8.75_rep1
    E8.75_rep2
)

SAMPLE_NAME="${SAMPLES[$SLURM_ARRAY_TASK_ID]}"

# Human-facing titles: the CRISPR pair reads better with the genotype up front.
case "${SAMPLE_NAME}" in
    E8.5_CRISPR_T_WT) TITLE="T-WT E8.5 Lineage Atlas" ;;
    E8.5_CRISPR_T_KO) TITLE="T-KO E8.5 Lineage Atlas" ;;
    *)                TITLE="${SAMPLE_NAME//_/ } Lineage Atlas" ;;
esac

echo "=== report: ${SAMPLE_NAME} -> ${TITLE} ==="
python "${PROJECT_DIR}/TETHER/scripts/build_scrna_report.py" \
    --sample_dir "${OUT_ROOT}/${SAMPLE_NAME}" \
    --sample_name "${SAMPLE_NAME}" \
    --title "${TITLE}"
echo "=== done ${SAMPLE_NAME} ==="
