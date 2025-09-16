#!/bin/bash -l
#SBATCH --job-name=transformer_training
#SBATCH --output=LOGS/%x.log
#SBATCH --error=LOGS/%x.err
#SBATCH -p dense
#SBATCH -N 1                      # Number of nodes
#SBATCH --gres=gpu:1             # GPUs per node
#SBATCH -c 16                    # CPU cores
#SBATCH --mem=128G

set -euo pipefail

echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_PROCID: $SLURM_PROCID"

# Go to project root
cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER

# Limit GPU memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

# Keep track of the resource requirements needed by the GPUs every 30 seconds in a csv file
trap 'pkill -P $$ || true' EXIT
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total --format=csv -l 30 > LOGS/gpu_usage_transformer_training.log &

# Ensure logs/ directory exists
mkdir -p logs

# Optional: activate Poetry env manually (usually not needed if using `poetry run`)
# source $(poetry env info --path)/bin/activate

# # Run DDP training using 1 GPUs on 1 node
# poetry run torchrun \
#   --standalone \
#   --nproc_per_node=1 \
#   ./dev/testing_scripts/transformer.py

# Single GPU Training
srun --unbuffered \
  torchrun --standalone --nproc_per_node=1 ./dev/testing_scripts/transformer.py

# # Multiple GPU Training
# srun --unbuffered \
#   torchrun --nproc_per_node=4 --nnodes=1 ./dev/testing_scripts/transformer.py