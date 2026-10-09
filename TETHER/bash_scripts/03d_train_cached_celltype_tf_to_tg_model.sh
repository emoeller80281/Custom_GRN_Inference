#!/bin/bash -l
#SBATCH --job-name=celltype_tf_tg
#SBATCH --output=LOGS/celltype_tf_tg/celltype_tf_tg_%j.log
#SBATCH --error=LOGS/celltype_tf_tg/celltype_tf_tg_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=dense
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --signal=SIGUSR1@90
# A requeued job reuses the same log files; append so the earlier attempt is kept.
#SBATCH --open-mode=append

# Trains from the per-sample caches built by 03c_build_celltype_data.sh.
# Submit from the repository root (which contains LOGS):
#   sbatch TETHER/bash_scripts/03d_train_cached_celltype_tf_to_tg_model.sh
# To start as soon as every cache job succeeds:
#   sbatch --dependency=afterok:<03c array job ID> TETHER/bash_scripts/03d_train_cached_celltype_tf_to_tg_model.sh
# Extra CLI arguments override the defaults below.
#
# --species, --max_cells_per_pair, --max_peaks_per_tg and the other preprocessing
# settings must match 03c. The job stops early and lists the expected cache paths
# if any sample has no complete cache.
#
# Cell types of a --holdout_sample that also appear in another sample's training
# split are held out of training automatically. Add --holdout_celltype for more,
# or --no-auto_holdout_celltypes to turn this off.
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

srun python -u scripts/train_cached_tf_to_tg_celltype_model.py \
    --species mm10 \
    --dataset mESC:E7.5_rep1 \
    --dataset mESC:E7.5_rep2 \
    --dataset mESC:E8.0_rep1 \
    --dataset mESC:E8.0_rep2 \
    --dataset mESC:E8.5_rep1 \
    --dataset mESC:E8.5_rep2 \
    --dataset kidney:Ctrl_4weeks_1 \
    --dataset kidney:Ctrl_4weeks_2 \
    --dataset kidney:Ctrl_6months_1 \
    --dataset GSE140203_shareseq:skin_late_anagen \
    --dataset GSE140203_shareseq:shareseq_brain \
    --dataset 10x_E18_mouse_brain:E18_brain \
    --dataset GSE246464_HSC:young_rep1 \
    --dataset GSE246464_HSC:young_rep2 \
    --dataset GSE246464_HSC:old_rep1 \
    --dataset GSE246464_HSC:old_rep2 \
    --dataset mouse_liver:liver_sample \
    --max_cells_per_pair 50 \
    --max_peaks_per_tg 50 \
    --epochs 250 \
    --early_stopping_patience 250 \
    --plateau_patience 15 \
    --lr 6.67e-4 \
    --dropout 0.01 \
    --num_heads 8 \
    --d_model 64 \
    --num_cross_attn_layers 2 \
    --accelerator gpu \
    --batch_size 128 \
    --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
    --run_name "long_training" \
    --precision 32-true \
    --wandb_project celltype-TF-TG \
    "$@"
