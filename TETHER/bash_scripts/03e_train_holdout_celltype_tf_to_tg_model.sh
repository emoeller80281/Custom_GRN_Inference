#!/bin/bash -l
#SBATCH --job-name=celltype_tf_tg
#SBATCH --output=LOGS/celltype_tf_tg/celltype_tf_tg_%j.log
#SBATCH --error=LOGS/celltype_tf_tg/celltype_tf_tg_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=dense
#SBATCH --nodes=1
# One A100. Each rank of the 4-GPU run 3992446 used about 103 GB.
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --signal=SIGUSR1@90
#SBATCH --open-mode=append

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

# Name the run after the held-out samples, e.g. holdout_heart_lv_adult_ENCDO575WHY.
HOLDOUT_NAMES=()
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --holdout_sample) HOLDOUT_NAMES+=("${args[i + 1]#*:}") ;;
        --holdout_sample=*) value="${args[i]#*=}"; HOLDOUT_NAMES+=("${value#*:}") ;;
    esac
done
if [ "${#HOLDOUT_NAMES[@]}" -eq 0 ]; then
    echo "ERROR: give at least one --holdout_sample TISSUE:SAMPLE (use 03d for a run without a holdout)" >&2
    exit 1
fi
RUN_NAME="holdout_$(IFS=_; echo "${HOLDOUT_NAMES[*]}")"

SPECIES="${SPECIES:-mm10}"
EXTRA_ARGS=()
if [ "$SPECIES" = "hg38" ]; then
DATASETS=(
    "ENCODE_adrenal:adrenal_adult_ENCDO221IPH"
    "ENCODE_adrenal:adrenal_adult_ENCDO856ZOJ"
    "ENCODE_bile_duct:bile_duct_adult_ENCDO575WHY"
    "ENCODE_colon:colon_adult_ENCDO856ZOJ"
    "ENCODE_colon:colon_adult_ENCDO186XRB"
    "ENCODE_fallopian_tube:fallopian_tube_adult_ENCDO411EVD"
    "ENCODE_fallopian_tube:fallopian_tube_adult_ENCDO856ZOJ"
    "ENCODE_heart:heart_lv_adult_ENCDO575WHY"
    "ENCODE_heart:heart_lv_adult_ENCDO411EVD"
    "ENCODE_lung:lung_adult_ENCDO528BHB"
    "ENCODE_lung:lung_child_ENCDO575EGL"
    "ENCODE_pancreas:pancreas_adult_ENCDO221IPH"
    "ENCODE_pancreas:pancreas_child_ENCDO575EGL"
    "ENCODE_vena_cava:vena_cava_adult_ENCDO221IPH"
    "GSE194122_bone_marrow:s1d1"
    "GSE194122_bone_marrow:s1d2"
    "GSE207335_epidermis:S1"
    "GSE207335_epidermis:S2"
    "GSE240061_muscle_pre_exercise:Ex-1_Pre_tr1"
    "GSE240061_muscle_pre_exercise:Ex-1_Pre_tr2"
    "GSE319044_lung_spleen_immune:Spleen-SMO-04"
    "GSE319044_lung_spleen_immune:Spleen-SMO-1"
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
DATASET_ARGS=()
for dataset in "${DATASETS[@]}"; do
    DATASET_ARGS+=(--dataset "$dataset")
done

export TETHER_RUN_STAMP="${TETHER_RUN_STAMP:-$(date +%Y%m%d_%H%M%S_%6N)}"
srun python -u scripts/train_cached_tf_to_tg_celltype_model.py \
    --devices 1 \
    --species "$SPECIES" \
    "${DATASET_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --max_cells_per_pair 50 \
    --max_peaks_per_tg 50 \
    --epochs 100 \
    --early_stopping_patience 100 \
    --plateau_patience 15 \
    --lr 3.335e-5 \
    --dropout 0.01 \
    --num_heads 4 \
    --d_model 128 \
    --num_cross_attn_layers 2 \
    --accelerator gpu \
    --batch_size 256 \
    --num_workers "${SLURM_CPUS_PER_TASK:-4}" \
    --run_name "$RUN_NAME" \
    --precision 32-true \
    --wandb_project celltype-TF-TG \
    "$@"
