#!/bin/bash -l
#SBATCH --job-name=tf_tg_model_nmp
#SBATCH --output=LOGS/tf_tg_model/%x_%j.log
#SBATCH --error=LOGS/tf_tg_model/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:4
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
# FIXED 2026-08-18: val/test are now scored in fp32 regardless of --precision
# (LitTFTGRegulationModel.fp32_eval, default True). The metrics this script logs are
# trustworthy again. What follows is why that mattered -- measured on run 3793729's own
# checkpoints (jobs 3794653/3797681/3799232, identical weights and data):
#
#                       epoch 0   epoch 5     delta
#   val pooled  fp32     0.6833    0.6843     +0.001
#   val macro   fp32     0.6371    0.6668     +0.030   <- model genuinely improving
#   val pooled  bf16     0.6740    0.6421     -0.032   <- what this script logged
#   train macro fp32     0.6461    0.6500     +0.004
#
# The bf16 penalty was smallest at epoch 0 (0.0094) and 0.018-0.042 thereafter, so bf16
# systematically favoured the EARLIEST checkpoint. Re-scoring all 16 in fp32 (job 3799232)
# found the best is epoch 3 (macro 0.6817); the run had kept epoch 0, which ranks 16th of
# 16 on macro. Rank correlation between the logged metric and fp32 macro: rho = -0.09.
#
# Batch is now 256. The earlier claim here that batch 256 "overfit" was WRONG: it compared
# 3793551 (bf16) against 3788646 (fp16), i.e. across precisions. Like-for-like, both bf16,
# batch 256 was BETTER at every comparable epoch (ep1 0.6941 vs 0.6648, ep2 0.6921 vs
# 0.6715, ep3 0.6773 vs 0.6696) and 2.8x faster per epoch (25 min vs 45). With fp32 eval
# in place, this run finally selects checkpoints on a trustworthy signal.
#
# lr stays 2.828e-4: that is the value batch 256 actually ran at when it beat batch 64, so
# it is the one with evidence behind it. Do NOT also change lr here -- changing batch and
# lr together, with the metric only just repaired, would make the result uninterpretable.
#
# Also note: train loss fell 0.363 -> 0.19 while train macro AUROC moved only +0.004, so
# that loss drop is the model growing more confident, not better at ranking. Do not read
# falling train loss + rising val loss here as overfitting -- the per-TF diagnostic found
# NO TF memorisation (held-out TFs improved 7x more than trained-on TFs).
#
# lr 2.828e-4 is what sqrt scaling gave at batch 64. ReduceLROnPlateau's 10x cut fired
# after epoch 7 and did NOT help (epoch 8 was slightly worse), which independently rules
# out lr as the cause of the apparent decline.
#
# Hardware: V100, not A100 (the a100 nodes are tied up by multi-day jobs). That forces the
# precision choice, because all three options behave differently on Volta (sm_70):
#
#   16-mixed  native and fast on V100 -- and exactly what produced NaN logits at epoch 7
#             in run 3788646, on this same hardware. fp16 saturates at ~65504 and GradScaler
#             rescues gradient overflow only, not forward activations. Not worth repeating.
#   bf16      torch.cuda.is_bf16_supported() returns True on V100 but there are no bf16
#             tensor cores before Ampere, and the autocast makes Inductor skip compilation
#             entirely. Measured in the benchmark on V100: uncompiled bf16 ~2.8 s/it against
#             warm compiled fp32 ~0.9 s/it. bf16 is the SLOWER option here, not the faster.
#   32-true   no overflow mode at all, and compiles properly. Volta has no TF32, so this is
#             genuine fp32 throughout. <- chosen
#
# The model is only 207K trainable parameters and is latency-bound rather than compute-bound
# (GPU util sat at 30-70% even at batch 256), so fp32 costs far less than its FLOP ratio
# suggests. Note this also makes fp32_eval a no-op: the run is fp32 everywhere.
#
# Memory is the one risk: V100 is 32 GB against the A100's 80 GB, and batch 256 measured
# ~11 GB under bf16 on the A100, so fp32 should land near 20-24 GB. If it OOMs it will do so
# on the first batch, not hours in -- drop to --batch_size 128 and note it here.
#
# Schedule: ReduceLROnPlateau on val/loss was already in configure_optimizers and still
# owns the decay. --warmup_epochs 1.0 adds the piece a scaled-up LR actually needs, since
# plateau only reacts after val/loss has already stalled and cannot protect the first
# steps. Warmup is counted in epochs, not a fraction of --epochs, because EarlyStopping
# (patience 15) normally ends the run well before 250.
#
# NEW 2026-08-19 -- --per_tf_pos_weight. Weights the positive class per TF
# (w_t = n_neg_t / n_pos_t over the training split, capped at 50) instead of leaving BCE
# unweighted. The problem it targets: plain BCE is minimised partly by getting each TF's
# absolute score level right, and across this split the optimal constant logit runs from
# -0.41 at a 40%-positive TF to -3.89 at a 2%-positive one. That 3.5-logit spread is a
# gradient signal with nothing to do with ranking targets inside a TF, and it looks like
# where run 3799581's extra capacity went: against run 3793729 on the 52 held-out NMP TFs
# it gained +0.052 pooled AUROC but only +0.004 macro, median per-TF AUROC was flat
# (0.6039 -> 0.6034), and just 29 of 52 TFs improved -- a coin flip. Weighting each TF to
# an effective 50% positive rate makes the optimal offset 0 for every TF, so encoding
# "how active is this TF" stops paying.
#
# The signature of success is unusual and worth stating up front: macro AUROC should rise
# while pooled AUROC FALLS. Pooled is inflated by exactly the between-TF separation this
# removes. Do not read a pooled drop here as a regression.
#
# val/loss is no longer comparable to any earlier run -- the objective changed. val/auroc
# and val/macro_auroc still are. ReduceLROnPlateau monitors the weighted val/loss, which
# is intentional: the schedule should track the objective actually being optimised.
#
# Cache: this run REQUIRES the rebuild from 03a_build_tf_to_tg_cache_nmp_split.sh with
# --val_tf_frac now 0.25 (was 0.15). The old 16-TF validation split had only 12 scorable
# TFs -- BAZ2A/GCM1 had zero positives, ETV2 one, KDM5B two -- and ETV2's single positive
# alone accounted for about half the apparent macro gap between the two previous runs.
# Because train/val TF membership changes, this run's val numbers are NOT comparable to
# 3793729 or 3799581. The 52-TF test split is unchanged, so test numbers still are.

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
    --batch_size 256 \
    --keep_tf_dna_in_eval \
    --tf_embedding_on_device \
    --resample_cells_per_epoch \
    --lr 2.828e-4 \
    --warmup_epochs 1.0 \
    --per_tf_pos_weight \
    --precision 32-true

#     --max_peaks_per_tg $max_peaks_per_tg \
