#!/bin/bash -l
#SBATCH --job-name=transformer_training
#SBATCH --output=LOGS/transformer_logs/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
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

# --- Threading: 8 threads per rank (since -c 8) ---
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export BLIS_NUM_THREADS=8
export KMP_AFFINITY=granularity=fine,compact,1,0

# --- NCCL / DDP diagnostics ---
export TORCH_NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=INFO

# Create logs & start GPU sampler
mkdir -p LOGS
trap 'pkill -P $$ || true' EXIT
nvidia-smi -L
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total \
  --format=csv -l 30 > LOGS/gpu_usage_transformer_training.log &

# Launch: 4 ranks on this node, bind ranks to cores
torchrun --standalone --nproc_per_node=1 ./dev/testing_scripts/transformer_2_training.py

# Launch: 2 ranks on this node, bind ranks to cores
# torchrun --standalone --nproc_per_node=4 ./dev/testing_scripts/transformer.py
