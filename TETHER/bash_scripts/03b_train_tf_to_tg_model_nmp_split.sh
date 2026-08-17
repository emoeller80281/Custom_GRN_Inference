#!/bin/bash -l
#SBATCH --job-name=tf_tg_model_nmp
#SBATCH --output=LOGS/tf_tg_model/%x_%j.log
#SBATCH --error=LOGS/tf_tg_model/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH -c 8
#SBATCH --mem=384G
#SBATCH --signal=SIGUSR1@90

# Train the TF-TG regulation model on the Argelaguet et al. 2022 organogenesis metacells
# with a TRANSCRIPTION-FACTOR split: trained on TFs not implicated in NMP differentiation,
# tested on the 52 TFs the paper's NMP-trajectory GRN did implicate (T, SOX2, CDX2, LEF1,
# TCF7L1/2, OLIG2, PAX6, ZIC3, ...).
#
# Requires 03a_build_tf_to_tg_cache_nmp_split.sh to have run first, and config.py set to
# species=mm10, cell_type=mESC, sample_name=WT_timecourse_metacells.
#
# The frozen TF-DNA submodule is checkpoints/tf_dna_mm10_3697823/epoch=07-...ckpt, chosen
# by config.tf_dna_model_checkpoints["mESC"], and --keep_tf_dna_in_eval pins it to eval
# mode (running BatchNorm stats, fast path) for the whole run.
#
# Throughput settings come from the single-GPU sweep in probe_tf_tg_throughput.sh
# (job 3788640, V100-32GB, eval mode):
#
#   per-edge emb   batch  8  cells 24    188.8 edges/s   <- previous settings
#   device-table   batch 64  cells 24    421.0 edges/s   2.2x
#   device-table   batch128  cells 24    461.6 edges/s   2.4x
#   device-table   batch128  cells 64    442.2 edges/s   2.3x
#
# --tf_embedding_on_device is what makes the larger batch pay off: without it the loader
# ships a 2.86 MB TF embedding per edge, so the waste grows with batch size (183 MB/batch
# at 64) and batch 64 only reaches 268 edges/s. Batch 64 rather than the marginally faster
# 128 because lr is hardcoded at 1e-4 and 64 already raises the effective batch 4x64=256
# from 4x8=32; revisit together with lr if you want the last ~10%.
#
# --resample_cells_per_epoch redraws each edge's cells from the full 1,896-metacell pool
# every epoch (training only; val/test stay on the frozen cached bags, which are now built
# at the same 64 cells so train and eval agree).
#
# Batch is 64, NOT larger, and this is a measured decision -- do not raise it without
# re-tuning lr. Run 3793551 tried batch 256 (effective 1024) at this same lr and overfit:
#
#   epoch  train loss   val loss   val AUROC     vs 3788646 @ batch 64
#     0      0.363       0.3000     0.6737         0.7032
#     1      0.268       0.2938     0.6941  <-best 0.7297
#     2      0.249       0.3018     0.6921         0.7444  <- 3788646 best
#     3      0.185       0.3162     0.6773         0.7306
#
# Train loss fell FASTER than at batch 64 while val loss rose for three straight epochs.
# That is the large-batch generalization gap, not slow convergence: gradient noise scale
# goes as lr/batch, so 4x the batch at fixed lr is 4x less noise, which finds sharper
# minima. Batch 256 was 2.8x faster per epoch (25 min vs 72) and still the wrong trade --
# it never came within 0.05 AUROC of what batch 64 reached in 3 epochs.
#
# If batch >64 is retried, raise lr WITH it: the linear rule (1.13e-3 at batch 256) is what
# restores the noise scale; sqrt (5.66e-4) only half-corrects it.
#
# lr 2.828e-4 is what sqrt scaling gave at batch 64, and it trained cleanly for 7 epochs
# (val AUROC 0.70 -> 0.744) before dying on an fp16 NaN that bf16 now removes.
#
# Precision: bf16-mixed, which needs Ampere and is why this now targets a100. fp16 saturates
# at ~65504 and GradScaler only rescues gradient overflow, not forward activations -- run
# 3788646 hit NaN logits at epoch 7 under 16-mixed. bf16 has fp32's exponent range at the
# same speed, which removes that failure mode rather than papering over it.
#
# Schedule: ReduceLROnPlateau on val/loss was already in configure_optimizers and still
# owns the decay. --warmup_epochs 1.0 adds the piece a scaled-up LR actually needs, since
# plateau only reacts after val/loss has already stalled and cannot protect the first
# steps. Warmup is counted in epochs, not a fraction of --epochs, because EarlyStopping
# (patience 15) normally ends the run well before 250.

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
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
true_false_ratio=10.0


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
    --batch_size 64 \
    --keep_tf_dna_in_eval \
    --tf_embedding_on_device \
    --resample_cells_per_epoch \
    --lr 2.828e-4 \
    --warmup_epochs 1.0 \
    --precision bf16-mixed

#     --max_peaks_per_tg $max_peaks_per_tg \
