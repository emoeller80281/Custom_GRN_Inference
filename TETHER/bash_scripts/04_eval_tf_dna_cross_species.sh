#!/bin/bash -l
#SBATCH --job-name=tf_dna_cross
#SBATCH --output=LOGS/eval_cross_species/%x_%j.log
#SBATCH --error=LOGS/eval_cross_species/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR
source activate my_env

export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

# Point these at the checkpoints to compare. Use the best-val_auroc file rather than
# last.ckpt once the runs have finished.
MM10_CKPT="checkpoints/tf_dna_mm10_3831017/last.ckpt"
HG38_CKPT="checkpoints/tf_dna_hg38_3831693/last.ckpt"
N_EDGES=200000
OUT_DIR="results/cross_species"
mkdir -p "$OUT_DIR"

# Full transfer matrix: each model against both species' test edges. The diagonal is the
# within-species baseline the off-diagonal has to be read against -- a cross-species AUROC
# means nothing without the number the same model gets at home.
#
# The shuffled runs are the control that makes the rest interpretable: they permute which
# embedding each TF receives, so if the scores hold up the model is reading DNA alone.
for SPEC in \
    "mm10 $MM10_CKPT mm10 real" \
    "mm10 $MM10_CKPT hg38 real" \
    "hg38 $HG38_CKPT hg38 real" \
    "hg38 $HG38_CKPT mm10 real" \
    "mm10 $MM10_CKPT hg38 shuffled" \
    "hg38 $HG38_CKPT mm10 shuffled" \
    "mm10 $MM10_CKPT mm10 shuffled" \
    "hg38 $HG38_CKPT hg38 shuffled"
do
    set -- $SPEC
    MODEL_SPECIES=$1; CKPT=$2; EVAL_SPECIES=$3; MODE=$4
    echo "=============================================================="
    echo "$MODEL_SPECIES model -> $EVAL_SPECIES test, $MODE embeddings"
    echo "=============================================================="
    python3 -u scripts/eval_tf_dna_cross_species.py \
        --model_ckpt "$CKPT" \
        --model_species "$MODEL_SPECIES" \
        --eval_species "$EVAL_SPECIES" \
        --tf_embedding_mode "$MODE" \
        --n_edges $N_EDGES \
        --batch_size 512 \
        --out "$OUT_DIR/${MODEL_SPECIES}_to_${EVAL_SPECIES}_${MODE}.parquet"
done

echo "Done. Per-edge scores in $OUT_DIR"
