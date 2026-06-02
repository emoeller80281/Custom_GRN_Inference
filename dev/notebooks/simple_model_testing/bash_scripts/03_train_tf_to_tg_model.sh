#!/bin/bash -l
#SBATCH --job-name=tf_tg_model
#SBATCH --output=LOGS/tf_tg_model/%x_%j.log
#SBATCH --error=LOGS/tf_tg_model/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=2
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --signal=SIGUSR1@90

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing"
cd $PROJECT_DIR

echo "Activating conda environment and starting training..."
source activate my_env

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
MASTER_PORT=29500
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

tf_bind_model_path="${PROJECT_DIR}/checkpoints/tfbind_train_3671604/epoch=06-val_auroc=0.9186-val_loss=0.2750.ckpt"

max_cells_per_pair=16
max_peaks_per_tg=8
peak_flank_size=128
pct_true_edges=1.0
true_false_ratio=2.0

echo "[INFO] Building and Caching Training Data..."
python3 ${PROJECT_DIR}/scripts/build_tf_to_tg_train_data.py \
    --max_peaks_per_tg $max_peaks_per_tg \
    --max_cells_per_pair $max_cells_per_pair \
    --pct_true_edges $pct_true_edges \
    --true_false_ratio $true_false_ratio \
    --peak_flank_size $peak_flank_size \
    --num_cpu $SLURM_CPUS_PER_TASK

echo "[INFO] Starting training..."
srun python3 ${PROJECT_DIR}/scripts/train_tf_to_tg_model.py \
    --epochs 250 \
    --sample_name $sample_name \
    --num_gpus $NPROC_PER_NODE \
    --num_nodes $SLURM_JOB_NUM_NODES \
    --job_id ${SLURM_JOB_ID} \
    --tf_bind_model_path $tf_bind_model_path \
    --max_peaks_per_tg $max_peaks_per_tg \
    --max_cells_per_pair $max_cells_per_pair \
    --peak_flank_size $peak_flank_size \
    --batch_size 1024