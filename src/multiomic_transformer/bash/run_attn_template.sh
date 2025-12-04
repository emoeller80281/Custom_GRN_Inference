#!/bin/bash -l
#SBATCH --job-name=attn_job
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 8
#SBATCH --mem=64G

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <mode>"
  echo "  mode = tf_knockout | grad_attrib"
  exit 1
fi

MODE="$1"

case "$MODE" in
  tf_knockout)
    PY_SCRIPT="./src/multiomic_transformer/scripts/tf_knockout_attn_only.py"
    ;;
  grad_attrib)
    PY_SCRIPT="./src/multiomic_transformer/scripts/gradient_attribution_attn_only.py"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Expected: tf_knockout or grad_attrib"
    exit 1
    ;;
esac

# ------------------------------------------------------------
# Shared experiment paths (come from env, or use defaults)
# ------------------------------------------------------------
EXPERIMENT_DIR=${EXPERIMENT_DIR:-/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC_no_scale_linear}
SELECTED_EXPERIMENT_DIR=${SELECTED_EXPERIMENT_DIR:-$EXPERIMENT_DIR/model_training_192_10k_metacells}
MODEL_FILE=${MODEL_FILE:-checkpoint_195.pt}

echo "Mode: $MODE"
echo "Python script: $PY_SCRIPT"
echo "EXPERIMENT_DIR: $EXPERIMENT_DIR"
echo "SELECTED_EXPERIMENT_DIR: $SELECTED_EXPERIMENT_DIR"
echo "MODEL_FILE: $MODEL_FILE"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export BLIS_NUM_THREADS=8
export KMP_AFFINITY=granularity=fine,compact,1,0

# ------------------------------------------------------------
# Run the selected Python script
# ------------------------------------------------------------
torchrun --standalone --nnodes=1 --nproc_per_node=4 "$PY_SCRIPT" \
    --selected_experiment_dir "$SELECTED_EXPERIMENT_DIR" \
    --model_file "$MODEL_FILE" \
    --use_amp

echo "[$MODE] finished successfully!"
