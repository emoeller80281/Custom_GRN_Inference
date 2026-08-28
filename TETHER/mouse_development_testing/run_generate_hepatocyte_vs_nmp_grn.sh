#!/bin/bash -l
#SBATCH --job-name=hepatocyte_vs_nmp_grn
#SBATCH --output=LOGS/mouse_development_testing/hepatocyte_vs_nmp_grn_%j/%x_%j.log
#SBATCH --error=LOGS/mouse_development_testing/hepatocyte_vs_nmp_grn_%j/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=192G

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
cd "$PROJECT_DIR"

echo "Activating conda environment..."
source activate my_env

export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

# --- Threading ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

export PYTHONFAULTHANDLER=1

echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist        = " $SLURM_JOB_NODELIST
echo "Host            = " $(hostname)
echo "GPUs visible    = " $(nvidia-smi -L | wc -l)
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo ""

SAMPLE_NAME="E8.5_CRISPR_T_WT"
METACELL_FILE="E8.5_rep1_NMP_metacells_T_WT.txt"
PREDICTION_OUTPUT_FILE="hepatocyte_model_vs_T_WT_NMP_metacell_GRN.csv"

echo "[INFO] Generating hepatocyte-model-vs-NMP-metacell GRN predictions..."
echo "[INFO] Sample name: $SAMPLE_NAME"
echo "[INFO] Metacell file: $METACELL_FILE"
echo "[INFO] Prediction output file: $PREDICTION_OUTPUT_FILE"

python "${PROJECT_DIR}/TETHER/mouse_development_testing/generate_hepatocyte_vs_nmp_grn.py" \
    --sample_name "$SAMPLE_NAME" \
    --metacell_file "$METACELL_FILE" \
    --prediction_output_file "$PREDICTION_OUTPUT_FILE"
