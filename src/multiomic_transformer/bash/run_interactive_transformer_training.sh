# ------- interactive_torchrun.sh -------
#!/usr/bin/env bash
set -euo pipefail

# Project root (adjust if needed)
cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER

source .venv/bin/activate

# ---------- Logging ----------
LOGDIR="$PWD/LOGS/transformer_logs/03_training"
mkdir -p "$LOGDIR"
JOB=transformer_training
ID=${SLURM_JOB_ID:-$PPID}                 # fall back to shell PID if no SLURM_JOB_ID
TS=$(date +%Y%m%d_%H%M%S)
OUT="$LOGDIR/${JOB}_${ID}.${TS}.log"
ERR="${OUT%.log}.err"
GPULOG="$LOGDIR/gpu_usage_${JOB}_${ID}.${TS}.csv"

echo "LOGDIR = $LOGDIR"
echo "LOG    = $OUT"
echo "ERR    = $ERR"

# ---------- Memory / math ----------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

# ---------- Threading ----------
THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export NUMEXPR_NUM_THREADS=$THREADS
export BLIS_NUM_THREADS=$THREADS
export KMP_AFFINITY=granularity=fine,compact,1,0

# ---------- NCCL / DDP ----------
export TORCH_NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Optional: sometimes helps avoid weird fabrics on single node
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

# ---------- World / ranks ----------
# Use the allocation hint if present; else count local GPUs
NPROC=${SLURM_NTASKS_PER_NODE:-$(nvidia-smi -L | wc -l | tr -d ' ')}
export WORLD_SIZE=$NPROC
echo "WORLD_SIZE = $WORLD_SIZE  (nproc_per_node=$NPROC)"

# ---------- GPU sampler ----------
trap 'pkill -P $$ || true' EXIT
nvidia-smi -L
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
  --format=csv -l 30 > "$GPULOG" &

# ---------- Pick a master port that’s unlikely to conflict ----------
MASTER_PORT=$(( 10000 + (RANDOM % 50000) ))

# ---------- Launch ----------
# stdbuf keeps logs line-buffered so they appear in files promptly.
stdbuf -oL -eL torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  src/multiomic_transformer/scripts/train.py \
  >"$OUT" 2>"$ERR" &
PID=$!

echo "[info] torchrun PID: $PID"
echo "[info] Training logs:"
echo "  stdout: $OUT"
echo "  stderr: $ERR"
echo "  gpu:    $GPULOG"

wait $PID
