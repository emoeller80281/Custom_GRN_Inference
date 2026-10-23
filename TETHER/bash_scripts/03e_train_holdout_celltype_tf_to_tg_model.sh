#!/bin/bash -l
#SBATCH --job-name=celltype_tf_tg
#SBATCH --output=LOGS/celltype_tf_tg/celltype_tf_tg_%j.log
#SBATCH --error=LOGS/celltype_tf_tg/celltype_tf_tg_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=dense
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --signal=SIGUSR1@90
#SBATCH --open-mode=append

# Leave-one-tissue-out TF-TG models with the settings of the 2026-10-02 runs
# (jobs 3899206-3899211): 25 cells and 25 peaks per edge, 10 epochs, batch 512, and the
# training script defaults for the model. Only the TF-DNA checkpoint (binding scores) and
# the renamed brain samples differ. Build the caches first with
#   sbatch TETHER/bash_scripts/03c_build_celltype_data.sh --max_cells_per_pair 25 \
#       --max_peaks_per_tg 25 --tf_dna_checkpoint <same TF_DNA_CHECKPOINT as below>
# Submit from the repository root (which contains LOGS), one job per held-out tissue:
#   sbatch TETHER/bash_scripts/03e_train_holdout_celltype_tf_to_tg_model.sh \
#       --holdout_sample mouse_liver:liver_sample
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

TF_DNA_CHECKPOINT="$PROJECT_DIR/checkpoints/new_tf_dna_models/mm10_3925194_epoch_21.ckpt"

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
    --tf_dna_checkpoint "$TF_DNA_CHECKPOINT" \
    --max_cells_per_pair 25 \
    --max_peaks_per_tg 25 \
    --epochs 10 \
    --accelerator gpu \
    --batch_size 512 \
    --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
    --precision 32-true \
    --wandb_project celltype-TF-TG \
    "$@"
