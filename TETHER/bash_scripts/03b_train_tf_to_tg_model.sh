#!/bin/bash -l
#SBATCH --job-name=tf_tg_model
#SBATCH --output=LOGS/tf_tg_model/%x_%A_%a.log
#SBATCH --error=LOGS/tf_tg_model/%x_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --signal=SIGUSR1@90
#SBATCH --array=0-8%4

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate my_env

# ==========================================
#        DATASET SELECTION
# ==========================================
# One entry per array task, as "species|cell_type|sample_name". Keep #SBATCH --array
# above in sync with the length of this list (a task past the end exits with an error
# rather than silently running the wrong dataset).
#
# These are exported as TETHER_* and picked up by config.py, so config.py itself never
# needs editing to run a batch -- every array task reads the same file but resolves its
# own dataset from its own environment.
EXPERIMENT_LIST=(
    "mm10|mESC|E7.5_rep1"
    "mm10|mESC|E7.5_rep2"
    "mm10|mESC|E8.5_rep1"
    "mm10|mESC|E8.5_rep2"
    "mm10|mouse_hepatocytes|hepatocytes_1"
    "mm10|mouse_hepatocytes|hepatocytes_3"
    "hg38|Macrophage|buffer_1"
    "hg38|Macrophage|buffer_2"
    "hg38|K562|sample_1"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ ${TASK_ID} -ge ${#EXPERIMENT_LIST[@]} ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID (${TASK_ID}) exceeds number of experiments (${#EXPERIMENT_LIST[@]})"
    exit 1
fi

EXPERIMENT_CONFIG="${EXPERIMENT_LIST[$TASK_ID]}"

# Parse experiment configuration
IFS='|' read -r species cell_type sample_name <<< "$EXPERIMENT_CONFIG"

# config.py reads these three from the environment. species is exported explicitly so an
# inconsistent species/cell_type pair in the list above trips config.py's assert rather
# than being silently corrected.
export species cell_type sample_name

echo "[INFO] Array task ${TASK_ID}: ${species} / ${cell_type} / ${sample_name}"
python3 -c "import config; print('[INFO] config.py resolved:', config.describe_dataset())"

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

max_cells_per_pair=64
max_peaks_per_tg=25
peak_flank_size=128
pct_true_edges=0.3
true_false_ratio=5.0

# echo "[INFO] Building and Caching Training Data..."
# python3 ${PROJECT_DIR}/scripts/build_tf_to_tg_train_data.py \
#     --max_cells_per_pair $max_cells_per_pair \
#     --pct_true_edges $pct_true_edges \
#     --true_false_ratio $true_false_ratio \
#     --peak_flank_size $peak_flank_size \
#     --num_cpu $SLURM_CPUS_PER_TASK \
#     --force_reload

echo "[INFO] Starting training..."
srun python3 ${PROJECT_DIR}/scripts/train_tf_to_tg_model.py \
    --epochs 250 \
    --num_gpus $NPROC_PER_NODE \
    --num_nodes $SLURM_JOB_NUM_NODES \
    --job_id ${SLURM_JOB_ID} \
    --max_cells_per_pair $max_cells_per_pair \
    --max_peaks_per_tg $max_peaks_per_tg \
    --peak_flank_size $peak_flank_size \
    --pct_true_edges $pct_true_edges \
    --true_false_ratio $true_false_ratio \
    --batch_size 256 \
    --keep_tf_dna_in_eval \
    --tf_embedding_on_device \
    --resample_cells_per_epoch \
    --lr 2.828e-4 \
    --warmup_epochs 1.0 \
    --per_tf_pos_weight \
    --precision 32-true