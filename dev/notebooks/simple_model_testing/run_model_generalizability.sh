#!/bin/bash -l
#SBATCH --job-name=model_generalizability
#SBATCH --output=LOGS/model_performance/model_generalizability_%A/%x_%A_%a.log
#SBATCH --error=LOGS/model_performance/model_generalizability_%A/%x_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --array=0-35%8

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate my_env

EXPERIMENT_LIST=(

    # === mESC Evaluations ====
    # Same cell-type, same sample evaluations with own sample test sets
    "mESC|E7.5_rep1|mESC|E7.5_rep1"
    "mESC|E8.5_rep1|mESC|E8.5_rep1"

    # Same cell-type, different sample evaluations with mouse hepatocyte test sets
    "mESC|E7.5_rep1|mESC|E8.5_rep1"
    "mESC|E8.5_rep1|mESC|E7.5_rep1"

    # Cross cell-type, same organism evaluations with mESC test sets
    "mESC|E7.5_rep1|mouse_hepatocytes|hepatocytes_1"
    "mESC|E7.5_rep1|mouse_hepatocytes|hepatocytes_3"
    "mESC|E8.5_rep1|mouse_hepatocytes|hepatocytes_1"
    "mESC|E8.5_rep1|mouse_hepatocytes|hepatocytes_3"

    # Cross cell-type, different organism evaluations with Macrophage test sets
    "mESC|E7.5_rep1|Macrophage|buffer_1"
    "mESC|E7.5_rep1|Macrophage|buffer_2"
    "mESC|E8.5_rep1|Macrophage|buffer_1"
    "mESC|E8.5_rep1|Macrophage|buffer_2"
    
    # ==== Hepatocyte Evaluations ====
    # Same cell-type, same sample evaluations with own sample test sets
    "mouse_hepatocytes|hepatocytes_1|mouse_hepatocytes|hepatocytes_1"
    "mouse_hepatocytes|hepatocytes_3|mouse_hepatocytes|hepatocytes_3"
    
    # Same cell-type, different sample evaluations with mouse hepatocyte test sets
    "mouse_hepatocytes|hepatocytes_1|mouse_hepatocytes|hepatocytes_3"
    "mouse_hepatocytes|hepatocytes_3|mouse_hepatocytes|hepatocytes_1"
    
    # Cross cell-type, same organism evaluations with mESC test sets
    "mouse_hepatocytes|hepatocytes_1|mESC|E7.5_rep1"
    "mouse_hepatocytes|hepatocytes_1|mESC|E8.5_rep1"
    "mouse_hepatocytes|hepatocytes_3|mESC|E7.5_rep1"
    "mouse_hepatocytes|hepatocytes_3|mESC|E8.5_rep1"
    
    # Cross cell-type, different organism evaluations with Macrophage test sets
    "mouse_hepatocytes|hepatocytes_1|Macrophage|buffer_1"
    "mouse_hepatocytes|hepatocytes_1|Macrophage|buffer_2"
    "mouse_hepatocytes|hepatocytes_3|Macrophage|buffer_1"
    "mouse_hepatocytes|hepatocytes_3|Macrophage|buffer_2"
    
    # === Macrophage Evaluations ====
    # Same cell-type, same sample evaluations with own sample test sets
    "Macrophage|buffer_1|Macrophage|buffer_1"
    "Macrophage|buffer_2|Macrophage|buffer_2"
    
    # Same cell-type, different sample evaluations with Macrophage test sets
    "Macrophage|buffer_1|Macrophage|buffer_2"
    "Macrophage|buffer_2|Macrophage|buffer_1"

    # Cross cell-type, different organism evaluations with mESC test sets
    "Macrophage|buffer_1|mESC|E7.5_rep1"
    "Macrophage|buffer_1|mESC|E8.5_rep1"
    "Macrophage|buffer_2|mESC|E7.5_rep1"
    "Macrophage|buffer_2|mESC|E8.5_rep1"
    
    # Cross-cell type, different organism evaluations with mouse hepatocyte test sets
    "Macrophage|buffer_1|mouse_hepatocytes|hepatocytes_1"
    "Macrophage|buffer_1|mouse_hepatocytes|hepatocytes_3"
    "Macrophage|buffer_2|mouse_hepatocytes|hepatocytes_1"
    "Macrophage|buffer_2|mouse_hepatocytes|hepatocytes_3"
    
)

# --- Memory + math ---
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

# --- Threading ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

# --- NCCL / networking overrides ---
# Dynamically find the interface with 10.90.29.* network
export IFACE=$(ip -o -4 addr show | grep "10.90.29." | awk '{print $2}')

if [ -z "$IFACE" ]; then
    echo "[ERROR] Could not find interface with 10.90.29.* network on $(hostname)"
    ip -o -4 addr show  # Show all interfaces for debugging
    exit 1
fi

echo "[INFO] Using IFACE=$IFACE on host $(hostname)"
ip -o -4 addr show "$IFACE"

export NCCL_SOCKET_IFNAME="$IFACE"
export GLOO_SOCKET_IFNAME="$IFACE"

export NCCL_IB_DISABLE=0

export TORCH_DISTRIBUTED_DEBUG=DETAIL

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist        = " $SLURM_JOB_NODELIST
echo "Number of nodes = " $SLURM_JOB_NUM_NODES
echo "Ntasks per node = " $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo ""

# ---------- torchrun multi-node launch ----------
# Pick the first node as rendezvous/master
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT

echo "[INFO] MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# ---------- Optional network diagnostics ----------
DEBUG_NET=${DEBUG_NET:-1}   # set to 0 to skip tests once things work

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_NODE=${NODES[0]}

echo "[NET] Nodes in this job: ${NODES[*]}"
echo "[NET] MASTER_NODE=${MASTER_NODE}, IFACE=${IFACE:-<unset>}"

NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo "[INFO] Using nproc_per_node=$NPROC_PER_NODE based on GPUs per node"

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

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
IFS='|' read -r model_cell_type model_training_sample test_set_cell_type evaluation_sample <<< "$EXPERIMENT_CONFIG"

echo "[INFO] Running experiment with:"
echo "  model_cell_type=$model_cell_type"
echo "  model_training_sample=$model_training_sample"
echo "  test_set_cell_type=$test_set_cell_type"
echo "  evaluation_sample=$evaluation_sample"


echo "[INFO] Running model generalizability test..."
srun python3 ${PROJECT_DIR}/model_generalizability.py \
    --model_cell_type "$model_cell_type" \
    --model_training_sample "$model_training_sample" \
    --test_set_cell_type "$test_set_cell_type" \
    --evaluation_sample "$evaluation_sample" \
    --subset_size 10000 \
    --batch_size 256
