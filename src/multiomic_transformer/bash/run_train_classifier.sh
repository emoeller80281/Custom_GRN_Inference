#!/bin/bash -l
#SBATCH --job-name=classifier_training
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%A_%a.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:p100:2
#SBATCH --ntasks-per-node=2
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --array=0-7

set -euo pipefail

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export BLIS_NUM_THREADS=8
export KMP_AFFINITY=granularity=fine,compact,1,0
export TORCH_NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ------------------------------------------------------------
# Define chromosome list
# ------------------------------------------------------------
CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr13)
CHROM_ID=${CHROMS[$SLURM_ARRAY_TASK_ID]}

echo "Processing chromosome: ${CHROM_ID}"

# ------------------------------------------------------------
# Run classifier for this chromosome
# ------------------------------------------------------------
torchrun --standalone --nproc_per_node=1 ./src/multiomic_transformer/classifier.py \
    --chrom_id "${CHROM_ID}" \
    --experiment_id "model_training_001" \
    --gpu_id 0 \
    --chip_file "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/ground_truth_files/combined_ground_truth_no_rn111_or_rn112_edges.csv" \
    --inferred_network_outdir "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/chrom_inferred_grn_orti_chipatlas_rn117_unique"

echo "Chromosome ${CHROM_ID} finished successfully!"
