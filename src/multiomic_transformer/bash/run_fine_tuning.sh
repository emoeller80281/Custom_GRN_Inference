#!/bin/bash -l
#SBATCH --job-name=model_fine_tuning
#SBATCH --output=LOGS/transformer_logs/04_fine_tuning/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/04_fine_tuning/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH -p gpu
#SBATCH -N 3
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=128G

set -euo pipefail

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER

source .venv/bin/activate

# --- Memory + math ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

# --- Threading: set to match SLURM CPUs per task ---
THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export BLIS_NUM_THREADS=$THREADS
export KMP_AFFINITY=granularity=fine,compact,1,0

# --- NCCL / networking overrides ---
# Dynamically find the interface with 10.90.29.* network
export IFACE=$(ip -o -4 addr show | grep "10.90.29." | awk '{print $2}')

if [ -z "$IFACE" ]; then
    echo "[ERROR] Could not find interface with 10.90.29.* network on $(hostname)"
    ip -o -4 addr show  # Show all interfaces for debugging
    exit 1
fi

echo "[INFO] Using IFACE=$IFACE on host $(hostname)"
ip -o -4 addr show "$IFACE"

export NCCL_SOCKET_IFNAME="$IFACE"
export GLOO_SOCKET_IFNAME="$IFACE"

# (keep InfiniBand disabled if IB isn’t properly configured)
export NCCL_IB_DISABLE=1

export TORCH_DISTRIBUTED_DEBUG=DETAIL


##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist        = " $SLURM_JOB_NODELIST
echo "Number of nodes = " $SLURM_JOB_NUM_NODES
echo "Ntasks per node = " $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo ""

# ---------- Logging ----------
LOGDIR="$PWD/LOGS/transformer_logs/04_fine_tuning"
mkdir -p "$LOGDIR"
JOB=model_fine_tuning
ID=${SLURM_JOB_ID:-$PPID}
TS=$(date +%Y%m%d_%H%M%S)
GPULOG="$LOGDIR/gpu_usage_${ID}.csv"

# ---------- GPU sampler (runs only on the batch node) ----------
trap 'pkill -P $$ || true' EXIT
nvidia-smi -L
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
  --format=csv -l 30 > "$GPULOG" &

# ---------- torchrun multi-node launch ----------
# Pick the first node as rendezvous/master
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

echo "[INFO] MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# ---------- Optional network diagnostics ----------
DEBUG_NET=${DEBUG_NET:-1}   # set to 0 to skip tests once things work

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_NODE=${NODES[0]}

echo "[NET] Nodes in this job: ${NODES[*]}"
echo "[NET] MASTER_NODE=${MASTER_NODE}, IFACE=${IFACE:-<unset>}"

NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo "[INFO] Using nproc_per_node=$NPROC_PER_NODE based on GPUs per node"

# Launch torchrun on ALL nodes / tasks via srun
srun bash -c "torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=\$SLURM_NODEID \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  src/multiomic_transformer/scripts/fine_tuning.py"