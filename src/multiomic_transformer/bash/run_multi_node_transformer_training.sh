#!/bin/bash -l
#SBATCH --job-name=transformer_training
#SBATCH --output=LOGS/transformer_logs/03_training/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/03_training/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH -p dense
#SBATCH -N 3
#SBATCH --gres=gpu:v100:4
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
# Pick the NIC that has 10.90.29.* (eno1 from your tests)
export IFACE=eno1

# Tell NCCL & GLOO to use this NIC
export NCCL_SOCKET_IFNAME="$IFACE"
export GLOO_SOCKET_IFNAME="$IFACE"   # sometimes helps c10d too

echo "[INFO] Using IFACE=$IFACE on host $(hostname)"
ip -o -4 addr show "$IFACE"

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
LOGDIR="$PWD/LOGS/transformer_logs/03_training"
mkdir -p "$LOGDIR"
JOB=transformer_training
ID=${SLURM_JOB_ID:-$PPID}
TS=$(date +%Y%m%d_%H%M%S)
OUT="$LOGDIR/${JOB}_${ID}.${TS}.log"
ERR="${OUT%.log}.err"
GPULOG="$LOGDIR/gpu_usage_${JOB}_${ID}.${TS}.csv"

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

# Launch torchrun on ALL nodes / tasks via srun
srun bash -c "torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --node_rank=\$SLURM_NODEID \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  src/multiomic_transformer/scripts/multinode_train.py"
