#!/bin/bash -l
#SBATCH --job-name=gnn_classifier
#SBATCH --output=LOGS/transformer_logs/03_training/%x_%j.log
#SBATCH --error=LOGS/transformer_logs/03_training/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=128G

set -euo pipefail

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

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist        = " $SLURM_JOB_NODELIST
echo "Number of nodes = " $SLURM_JOB_NUM_NODES
echo "Ntasks per node = " $SLURM_NTASKS_PER_NODE
echo "WORLD_SIZE      = " $WORLD_SIZE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo ""

# Training
torchrun --standalone --nproc_per_node=$SLURM_NTASKS_PER_NODE src/multiomic_transformer/scripts/train_classifier.py \
    --ground_truth_file "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/ground_truth_files/combined_ground_truth_no_rn111_or_rn112_edges.csv" \
    --output_dir "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC/train_chip_gnn_classifier/" \
    --sep ","

# torchrun --standalone --nproc_per_node=$SLURM_NTASKS_PER_NODE src/multiomic_transformer/scripts/train_classifier.py \
#     --ground_truth_file "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/testing_bear_grn/GROUND.TRUTHS/filtered_RN111_and_RN112_mESC_E7.5_rep1.tsv" \
#     --output_dir "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC/test_chip_gnn_classifier/" \
#     --sep "\t"

# # Testing
# torchrun --standalone --nproc_per_node=$SLURM_NTASKS_PER_NODE src/multiomic_transformer/scripts/test_classifier.py \
#     --ground_truth_file "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/ground_truth_files/combined_ground_truth_no_rn111_or_rn112_edges.csv" \
#     --output_dir "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC/train_chip_gnn_classifier/" \
#     --sep ","

torchrun --standalone --nproc_per_node=$SLURM_NTASKS_PER_NODE src/multiomic_transformer/scripts/test_classifier.py \
    --ground_truth_file "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/testing_bear_grn/GROUND.TRUTHS/filtered_RN111_and_RN112_mESC_E7.5_rep1.tsv" \
    --output_dir "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC/train_chip_gnn_classifier/" \
    --sep "\t"
