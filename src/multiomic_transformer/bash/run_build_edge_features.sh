#!/bin/bash -l
#SBATCH --job-name=build_edge_features
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%A.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%A.err
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 21
#SBATCH --mem=64G

set -euo pipefail

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

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

#,chr2,chr3,chr4,chr5,chr6,chr7,chr13

python ./src/multiomic_transformer/data/build_edge_features.py \
  --cache_dir /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/training_data_cache/mESC \
  --model_ckpt /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC/chr1/model_training_001/trained_model.pt \
  --chip_file "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/ground_truth_files/combined_ground_truth_no_rn111_or_rn112_edges.csv" \
  --out_csv /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/mESC/combined/model_training_001/classifier/edge_features.csv \
  --chrom_list "chr13" \
  --num_cpu 16

  echo "Finished successfully!"