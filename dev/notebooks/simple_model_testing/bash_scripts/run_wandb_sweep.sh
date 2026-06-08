#!/bin/bash -l
#SBATCH --job-name=tf_tg_sweep
#SBATCH --output=LOGS/tf_tg_sweep/%x_%j.log
#SBATCH --error=LOGS/tf_tg_sweep/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --signal=SIGUSR1@90

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing"
SWEEP_CONFIG="${PROJECT_DIR}/wandb_sweep.yaml"

if [ $# -lt 1 ]; then
    echo "Usage: sbatch run_wandb_sweep.sh <entity>/<project>/<sweep-id>"
    echo "   or: sbatch run_wandb_sweep.sh wandb agent <entity>/<project>/<sweep-id>"
    exit 1
fi

if [ "$1" = "wandb" ] && [ "${2:-}" = "agent" ] && [ $# -ge 3 ]; then
    SWEEP_ID="$3"
elif [ "$1" = "agent" ] && [ $# -ge 2 ]; then
    SWEEP_ID="$2"
else
    SWEEP_ID="$1"
fi

if ! echo "$SWEEP_ID" | grep -Eq '^[^/]+/[^/]+/[^/]+$'; then
    echo "[ERROR] Invalid sweep path: '$SWEEP_ID'"
    echo "Expected format: <entity>/<project>/<sweep-id>"
    exit 1
fi

cd "$PROJECT_DIR"
mkdir -p LOGS/tf_tg_sweep

echo "Activating conda environment and starting W&B sweep agent..."
source activate my_env

# --- Memory + math ---
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

# --- Threading ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

# --- NCCL / networking overrides ---
# Dynamically find the interface with 10.90.29.* network
export IFACE=$(ip -o -4 addr show | grep "10.90.29." | awk '{print $2}')

if [ -z "$IFACE" ]; then
    echo "[ERROR] Could not find interface with 10.90.29.* network on $(hostname)"
    ip -o -4 addr show
    exit 1
fi

echo "[INFO] Using IFACE=$IFACE on host $(hostname)"
ip -o -4 addr show "$IFACE"

export NCCL_SOCKET_IFNAME="$IFACE"
export GLOO_SOCKET_IFNAME="$IFACE"
export NCCL_IB_DISABLE=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "[INFO] Sweep config: $SWEEP_CONFIG"
echo "[INFO] Sweep path: $SWEEP_ID"
echo "[INFO] W&B project: tf_tg_sweep"

echo "[INFO] Starting W&B agent on a 1-node / 4-GPU allocation..."
srun --ntasks=1 wandb agent "$SWEEP_ID"
