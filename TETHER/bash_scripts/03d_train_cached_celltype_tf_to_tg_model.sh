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
# SPECIES selects the datasets (default mm10), and must match the 03c run that built the caches:
#   sbatch --export=ALL,SPECIES=hg38 TETHER/bash_scripts/03d_train_cached_celltype_tf_to_tg_model.sh
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

SPECIES="${SPECIES:-mm10}"
EXTRA_ARGS=()
if [ "$SPECIES" = "hg38" ]; then
DATASETS=(
    "ENCODE_adrenal:adrenal_adult_ENCDO221IPH"
    "ENCODE_bile_duct:bile_duct_adult_ENCDO575WHY"
    "ENCODE_colon:colon_adult_ENCDO856ZOJ"
    "ENCODE_fallopian_tube:fallopian_tube_adult_ENCDO411EVD"
    "ENCODE_heart:heart_lv_adult_ENCDO575WHY"
    "ENCODE_lung:lung_adult_ENCDO528BHB"
    "ENCODE_pancreas:pancreas_adult_ENCDO221IPH"
    "GSE194122_bone_marrow:s1d1"
    "GSE207335_epidermis:S1"
    "GSE240061_muscle_pre_exercise:Ex-1_Pre_tr1"
    # GSE277326_retina_macula_rpe:Retina_donor1 is left out (2026-11-10): no slice passes the cache
    # filters (rods: density 0.66 > --max_gt_density 0.60), and the cache build then stops.
    "GSE319044_lung_spleen_immune:Spleen-SMO-04"
)
# Must match 03c: the TF-DNA checkpoint is part of the sample cache key.
EXTRA_ARGS=(--tf_dna_checkpoint "$PROJECT_DIR/checkpoints/tf_dna_hg38_3930720/epoch_18.ckpt")
elif [ "$SPECIES" = "mm10" ]; then
DATASETS=(
    "mESC:E7.5_rep1"
    "mESC:E7.5_rep2"
    "mESC:E8.0_rep1"
    "mESC:E8.0_rep2"
    "mESC:E8.5_rep1"
    "mESC:E8.5_rep2"
    "kidney:Ctrl_4weeks_1"
    "kidney:Ctrl_4weeks_2"
    "kidney:Ctrl_6months_1"
    "GSE140203_shareseq:skin_late_anagen"
    "GSE140203_shareseq:shareseq_brain"
    "10x_E18_mouse_brain:E18_brain"
    "GSE246464_HSC:young_rep1"
    "GSE246464_HSC:young_rep2"
    "GSE246464_HSC:old_rep1"
    "GSE246464_HSC:old_rep2"
    "mouse_liver:liver_sample"
)
else
    echo "ERROR: SPECIES must be mm10 or hg38, not ${SPECIES}" >&2
    exit 1
fi
# DATASETS_OVERRIDE (space-separated "dataset:sample" entries) replaces the list above, e.g. for a
# control run on a subset. Set it in the submitting shell and keep ALL in --export:
#   DATASETS_OVERRIDE="ENCODE_adrenal:adrenal_adult_ENCDO221IPH ENCODE_colon:colon_adult_ENCDO856ZOJ" \
#       sbatch --export=ALL,SPECIES=hg38 TETHER/bash_scripts/03d_train_cached_celltype_tf_to_tg_model.sh
if [ -n "${DATASETS_OVERRIDE:-}" ]; then
    read -r -a DATASETS <<< "$DATASETS_OVERRIDE"
    echo "DATASETS_OVERRIDE: ${DATASETS[*]}"
fi
DATASET_ARGS=()
for dataset in "${DATASETS[@]}"; do
    DATASET_ARGS+=(--dataset "$dataset")
done

srun python -u scripts/train_cached_tf_to_tg_celltype_model.py \
    --species "$SPECIES" \
    "${DATASET_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --max_cells_per_pair 50 \
    --max_peaks_per_tg 50 \
    --epochs 100 \
    --early_stopping_patience 100 \
    --plateau_patience 15 \
    --lr 6.67e-5 \
    --dropout 0.01 \
    --num_heads 8 \
    --d_model 64 \
    --num_cross_attn_layers 2 \
    --accelerator gpu \
    --batch_size 256 \
    --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
    --run_name "long_training" \
    --precision 32-true \
    --wandb_project celltype-TF-TG \
    "$@"
