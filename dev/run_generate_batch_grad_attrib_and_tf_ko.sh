#!/bin/bash -l
#SBATCH --job-name=generate_batch_grad
#SBATCH --output=LOGS/transformer_logs/04_testing/%x_%A/%x_%A_%a.log
#SBATCH --error=LOGS/transformer_logs/04_testing/%x_%A/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --array=0-9%3

set -eo pipefail

cd "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"

source activate my_env

EXPERIMENT_DIR=${EXPERIMENT_DIR:-/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments}

EXPERIMENT_LIST=(
    # "iPSC_hvg_filter_disp_0.05|model_training_001|iPSC|iPSC_sample"
    # "iPSC_hvg_filter_disp_0.1|model_training_001|iPSC|iPSC_sample"
    # "iPSC_hvg_filter_disp_0.2|model_training_001|iPSC|iPSC_sample"
    # "iPSC_hvg_filter_disp_0.3|model_training_001|iPSC|iPSC_sample"
    # "iPSC_hvg_filter_disp_0.4|model_training_001|iPSC|iPSC_sample"
    # "iPSC_hvg_filter_disp_0.5|model_training_001|iPSC|iPSC_sample"
    # "iPSC_hvg_filter_disp_0.6|model_training_001|iPSC|iPSC_sample"

    # "iPSC_3kb_max_dist|model_training_001|iPSC|iPSC_sample"
    # "iPSC_6kb_max_dist|model_training_001|iPSC|iPSC_sample"

    # "K562_hvg_filter_only_rna|model_training_001|k562|K562"
    # "K562_hvg_filter_none|model_training_001|k562|K562"
    # "K562_hvg_filter_disp_0.01|model_training_001|k562|K562"
    # "K562_hvg_filter_disp_0.05|model_training_001|k562|K562"
    # "K562_hvg_filter_disp_0.1|model_training_001|k562|K562"
    # "K562_hvg_filter_disp_0.2|model_training_001|k562|K562"
    # "K562_hvg_filter_disp_0.3|model_training_001|k562|K562"
    # "K562_hvg_filter_disp_0.4|model_training_001|k562|K562"
    # "K562_hvg_filter_disp_0.5|model_training_001|k562|K562"
    # "K562_hvg_filter_disp_0.6|model_training_001|k562|K562"

    # "mESC_1_sample_hvg_filter_disp_0.01|model_training_001|mESC|E7.5_rep1"
    # "mESC_2_sample_hvg_filter_disp_0.01|model_training_001|mESC|E7.5_rep1 E7.5_rep2"
    # "mESC_3_sample_hvg_filter_disp_0.01|model_training_001|mESC|E7.5_rep1 E7.5_rep2 E8.5_rep1"
    # "mESC_4_sample_hvg_filter_disp_0.01|model_training_001|mESC|E7.5_rep1 E7.5_rep2 E8.5_rep1 E8.5_rep2"
    # "mESC_5_sample_hvg_filter_disp_0.01|model_training_001|mESC|E7.5_rep1 E7.5_rep2 E8.5_rep1 E8.5_rep2"
    # "mESC_6_sample_hvg_filter_disp_0.01|model_training_001|mESC|E7.5_rep1 E7.5_rep2 E8.5_rep1 E8.5_rep2"
    # "mESC_7_sample_hvg_filter_disp_0.01|model_training_001|mESC|E7.5_rep1 E7.5_rep2 E8.5_rep1 E8.5_rep2"

    # "Macrophage_buffer_1_hvg_filter_only_rna|model_training_002|macrophage|buffer_1"
    # "Macrophage_buffer_1_hvg_filter_none|model_training_002|macrophage|buffer_1"
    # "Macrophage_buffer_1_hvg_filter_disp_0.6|model_training_002|macrophage|buffer_1"
    # "Macrophage_buffer_1_hvg_filter_disp_0.5|model_training_002|macrophage|buffer_1"
    # "Macrophage_buffer_1_hvg_filter_disp_0.4|model_training_002|macrophage|buffer_1"
    # "Macrophage_buffer_1_hvg_filter_disp_0.3|model_training_002|macrophage|buffer_1"
    # "Macrophage_buffer_1_hvg_filter_disp_0.2|model_training_002|macrophage|buffer_1"
    # "Macrophage_buffer_1_hvg_filter_disp_0.1|model_training_002|macrophage|buffer_1"
    # "Macrophage_buffer_1_hvg_filter_disp_0.05|model_training_002|macrophage|buffer_1"
    # "Macrophage_buffer_1_hvg_filter_disp_0.01|model_training_002|macrophage|buffer_1"

    # "Macrophage_buffer_2_hvg_filter_only_rna|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_2_hvg_filter_none|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_2_hvg_filter_disp_0.6|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_2_hvg_filter_disp_0.5|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_2_hvg_filter_disp_0.4|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_2_hvg_filter_disp_0.3|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_2_hvg_filter_disp_0.2|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_2_hvg_filter_disp_0.1|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_2_hvg_filter_disp_0.05|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_2_hvg_filter_disp_0.01|model_training_001|macrophage|buffer_1"

    # "Macrophage_buffer_3_hvg_filter_none|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_3_hvg_filter_only_rna|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_3_hvg_filter_disp_0.6|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_3_hvg_filter_disp_0.5|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_3_hvg_filter_disp_0.4|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_3_hvg_filter_disp_0.3|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_3_hvg_filter_disp_0.2|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_3_hvg_filter_disp_0.1|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_3_hvg_filter_disp_0.05|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_3_hvg_filter_disp_0.01|model_training_001|macrophage|buffer_1"

    # "Macrophage_buffer_4_hvg_filter_none|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_4_hvg_filter_only_rna|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_4_hvg_filter_disp_0.6|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_4_hvg_filter_disp_0.5|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_4_hvg_filter_disp_0.4|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_4_hvg_filter_disp_0.3|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_4_hvg_filter_disp_0.2|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_4_hvg_filter_disp_0.1|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_4_hvg_filter_disp_0.05|model_training_001|macrophage|buffer_1"
    # "Macrophage_buffer_4_hvg_filter_disp_0.01|model_training_001|macrophage|buffer_1"

    # "mESC_E7.5_rep1_hvg_filter_only_rna|model_training_001|mESC|E7.5_rep1"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.6|model_training_001|mESC|E7.5_rep1"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.5|model_training_001|mESC|E7.5_rep1"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.4|model_training_001|mESC|E7.5_rep1"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.3|model_training_001|mESC|E7.5_rep1"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.2|model_training_001|mESC|E7.5_rep1"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.1|model_training_001|mESC|E7.5_rep1"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.05|model_training_001|mESC|E7.5_rep1"
    # "mESC_E7.5_rep1_hvg_filter_disp_0.01|model_training_001|mESC|E7.5_rep1"

    # "mESC_E7.5_rep2_hvg_filter_only_rna|model_training_001|mESC|E7.5_rep2"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.6|model_training_001|mESC|E7.5_rep2"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.5|model_training_001|mESC|E7.5_rep2"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.4|model_training_001|mESC|E7.5_rep2"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.3|model_training_001|mESC|E7.5_rep2"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.2|model_training_001|mESC|E7.5_rep2"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.1|model_training_001|mESC|E7.5_rep2"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.05|model_training_001|mESC|E7.5_rep2"
    # "mESC_E7.5_rep2_hvg_filter_disp_0.01|model_training_001|mESC|E7.5_rep2"

    # "mESC_E7.75_rep1_hvg_filter_only_rna|model_training_001|mESC|E7.75_rep1"
    # "mESC_E7.75_rep1_hvg_filter_disp_0.6|model_training_001|mESC|E7.75_rep1"
    # "mESC_E7.75_rep1_hvg_filter_disp_0.5|model_training_001|mESC|E7.75_rep1"
    # "mESC_E7.75_rep1_hvg_filter_disp_0.4|model_training_001|mESC|E7.75_rep1"
    # "mESC_E7.75_rep1_hvg_filter_disp_0.3|model_training_001|mESC|E7.75_rep1"
    # "mESC_E7.75_rep1_hvg_filter_disp_0.2|model_training_001|mESC|E7.75_rep1"
    # "mESC_E7.75_rep1_hvg_filter_disp_0.1|model_training_001|mESC|E7.75_rep1"
    # "mESC_E7.75_rep1_hvg_filter_disp_0.05|model_training_001|mESC|E7.75_rep1"
    # "mESC_E7.75_rep1_hvg_filter_disp_0.01|model_training_001|mESC|E7.75_rep1"

    # "mESC_E8.0_rep1_hvg_filter_only_rna|model_training_001|mESC|E8.0_rep1"
    # "mESC_E8.0_rep1_hvg_filter_disp_0.6|model_training_001|mESC|E8.0_rep1"
    # "mESC_E8.0_rep1_hvg_filter_disp_0.5|model_training_001|mESC|E8.0_rep1"
    # "mESC_E8.0_rep1_hvg_filter_disp_0.4|model_training_001|mESC|E8.0_rep1"
    # "mESC_E8.0_rep1_hvg_filter_disp_0.3|model_training_001|mESC|E8.0_rep1"
    # "mESC_E8.0_rep1_hvg_filter_disp_0.2|model_training_001|mESC|E8.0_rep1"
    "mESC_E8.0_rep1_hvg_filter_disp_0.1|model_training_001|mESC|E8.0_rep1"
    "mESC_E8.0_rep1_hvg_filter_disp_0.05|model_training_001|mESC|E8.0_rep1"
    "mESC_E8.0_rep1_hvg_filter_disp_0.01|model_training_001|mESC|E8.0_rep1"

    "mESC_E8.0_rep2_hvg_filter_only_rna|model_training_001|mESC|E8.0_rep2"
    "mESC_E8.0_rep2_hvg_filter_disp_0.6|model_training_001|mESC|E8.0_rep2"
    "mESC_E8.0_rep2_hvg_filter_disp_0.5|model_training_001|mESC|E8.0_rep2"
    "mESC_E8.0_rep2_hvg_filter_disp_0.4|model_training_001|mESC|E8.0_rep2"
    "mESC_E8.0_rep2_hvg_filter_disp_0.3|model_training_001|mESC|E8.0_rep2"
    "mESC_E8.0_rep2_hvg_filter_disp_0.2|model_training_001|mESC|E8.0_rep2"
    "mESC_E8.0_rep2_hvg_filter_disp_0.1|model_training_001|mESC|E8.0_rep2"
    # "mESC_E8.0_rep2_hvg_filter_disp_0.05|model_training_001|mESC|E8.0_rep2"
    # "mESC_E8.0_rep2_hvg_filter_disp_0.01|model_training_001|mESC|E8.0_rep2"

    # "mESC_E8.5_rep1_hvg_filter_only_rna|model_training_001|mESC|E8.5_rep1"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.6|model_training_001|mESC|E8.5_rep1"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.5|model_training_001|mESC|E8.5_rep1"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.4|model_training_001|mESC|E8.5_rep1"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.3|model_training_001|mESC|E8.5_rep1"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.2|model_training_001|mESC|E8.5_rep1"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.1|model_training_001|mESC|E8.5_rep1"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.05|model_training_001|mESC|E8.5_rep1"
    # "mESC_E8.5_rep1_hvg_filter_disp_0.01|model_training_001|mESC|E8.5_rep1"

    # "mESC_E8.5_rep2_hvg_filter_only_rna|model_training_001|mESC|E8.5_rep2"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.6|model_training_001|mESC|E8.5_rep2"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.5|model_training_001|mESC|E8.5_rep2"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.4|model_training_001|mESC|E8.5_rep2"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.3|model_training_001|mESC|E8.5_rep2"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.2|model_training_001|mESC|E8.5_rep2"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.1|model_training_001|mESC|E8.5_rep2"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.05|model_training_001|mESC|E8.5_rep2"
    # "mESC_E8.5_rep2_hvg_filter_disp_0.01|model_training_001|mESC|E8.5_rep2"

    # "Macrophage_buffer_2_small_hvg_filter_disp_0.2|model_training_001|macrophage|buffer_2"

    # "mESC_E7.5_rep1_disp_0.2_128d|model_training_001|mESC|E7.5_rep1"
    # "mESC_E7.5_rep2_disp_0.2_128d|model_training_001|mESC|E7.5_rep2"
    # "mESC_E8.5_rep1_disp_0.2_128d|model_training_001|mESC|E8.5_rep1"
    # "mESC_E8.5_rep2_disp_0.2_128d|model_training_001|mESC|E8.5_rep2"

    # "mESC_E7.5_rep1_disp_0.2_192d|model_training_001|mESC|E7.5_rep1"
    # "mESC_E7.5_rep2_disp_0.2_192d|model_training_001|mESC|E7.5_rep2"
    # "mESC_E8.5_rep1_disp_0.2_192d|model_training_001|mESC|E8.5_rep1"
    # "mESC_E8.5_rep2_disp_0.2_192d|model_training_001|mESC|E8.5_rep2"

)

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export BLIS_NUM_THREADS=8
export KMP_AFFINITY=granularity=fine,compact,1,0

# ==========================================
#        EXPERIMENT SELECTION
# ==========================================
# Get the current experiment based on SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ ${TASK_ID} -ge ${#EXPERIMENT_LIST[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${TASK_ID}) exceeds number of experiments (${#EXPERIMENT_LIST[@]})"
    exit 1
fi

EXPERIMENT_CONFIG="${EXPERIMENT_LIST[$TASK_ID]}"

# Parse experiment configuration
IFS='|' read -r EXPERIMENT_NAME TRAINING_NUM DATASET_TYPE SAMPLE_NAMES <<< "$EXPERIMENT_CONFIG"

echo ""
echo "=========================================="
echo "  EXPERIMENT: ${EXPERIMENT_NAME}"
echo "  TRAINING_NUM: ${TRAINING_NUM}"
echo "  TASK ID: ${TASK_ID}"
echo "=========================================="
echo ""

echo "Running generate_batch_grad_attrib_and_tf_ko.py"
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    dev/generate_batch_grad_attrib_and_tf_ko.py \
        --experiment_name "${EXPERIMENT_NAME}" \
        --model_num "${TRAINING_NUM}" \
        --checkpoint_name "trained_model.pt" \
        --batch_size 64 \
        --save_every_n_batches 40

echo "finished successfully!"