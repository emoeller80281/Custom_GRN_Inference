#!/bin/bash -l
#SBATCH --job-name=celltype_tf_tg
#SBATCH --output=LOGS/celltype_tf_tg_%j.log
#SBATCH --error=LOGS/celltype_tf_tg_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=dense
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --signal=SIGUSR1@90

# Submit from the repository root (which contains LOGS):
# sbatch TETHER/bash_scripts/03b_train_celltype_tf_to_tg_model.sh
# Extra CLI arguments override the defaults below.
# Matching submissions reuse the stable prepared-data cache automatically.
set -eo pipefail
PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd "$PROJECT_DIR"
source activate my_env
set -u
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --dataset mouse_liver:liver_sample \
# --dataset mESC:E7.5_rep1 \
# --dataset mESC:E7.5_rep2 \
# --dataset mESC:E8.0_rep1 \
# --dataset mESC:E8.0_rep2 \
# --dataset mESC:E8.5_rep1 \
# --dataset mESC:E8.5_rep2 \

srun python -u scripts/train_tf_to_tg_celltype_model.py \
    --species mm10 \
    --dataset mouse_liver:liver_sample \
    --dataset mESC:E7.5_rep1 \
    --holdout_sample mESC:E7.5_rep1 \
    --epochs 250 \
    --accelerator gpu \
    --batch_size 512 \
    --max_cells_per_pair 64 \
    --max_peaks_per_tg 25 \
    --binding_chunk_size 1024 \
    --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
    --job_id "${SLURM_JOB_ID:-local}" \
    --precision 32-true \
    --wandb_project celltype-TF-TG \
    --resume_from_checkpoint /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER/checkpoints/celltype_tf_tg/celltype_joint_3884482_20260922_182318_756745/checkpoints/epoch-013.ckpt
    "$@"
