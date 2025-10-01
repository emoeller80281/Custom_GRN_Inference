#!/bin/bash -l
#SBATCH --job-name=transformer_training
#SBATCH --output=LOGS/transformer_logs/training/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/training/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p dense
#SBATCH -N 2
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 4
#SBATCH --mem=128G

set -euo pipefail

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER

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

# --- NCCL / DDP diagnostics ---
export TORCH_NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Torchrun multi-node
NNODES=$SLURM_NNODES
NPROC_PER_NODE=$SLURM_NTASKS_PER_NODE
NODE_RANK=$SLURM_NODEID
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=$((12000 + RANDOM % 20000))

# Create logs & start GPU sampler
mkdir -p LOGS
trap 'pkill -P $$ || true' EXIT
nvidia-smi -L
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
  --format=csv -l 30 > LOGS/gpu_usage_transformer_training.log &

# torchrun --standalone --nproc_per_node=$SLURM_NTASKS_PER_NODE ./dev/transformer/transformer_training.py

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  ./dev/transformer/transformer_training.py