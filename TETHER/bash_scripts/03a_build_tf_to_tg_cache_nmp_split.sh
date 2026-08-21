#!/bin/bash -l
#SBATCH --job-name=build_tf_tg_cache_nmp
#SBATCH --output=LOGS/build_tf_tg_cache/%x_%j.log
#SBATCH --error=LOGS/build_tf_tg_cache/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=384G

# Build the TF-TG training cache for the Argelaguet et al. 2022 organogenesis metacells
# using a TRANSCRIPTION-FACTOR split instead of the usual chromosome split.
#
#   test  = TFs the paper's NMP-trajectory GRN implicated in NMP -> {spinal cord,
#           somitic mesoderm} differentiation
#   train = every other TF, across all cell types and all chromosomes
#   val   = a TF-disjoint 25% holdout carved out of the training TFs (was 15%; see below)
#
# config.py must be set to: species=mm10, cell_type=mESC,
# sample_name=WT_timecourse_metacells

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR

mkdir -p LOGS/build_tf_tg_cache

echo "Activating conda environment..."
source activate my_env

# The TF split keeps every chromosome in training, so this cache is built from ~878k
# ground-truth edges -- roughly 16x what the per-library mESC samples had. At
# pct_true_edges=1.0 the edge bags would run to ~65 GB, so the true edges are subsampled
# to 30%. That lands the row counts (~1.8M train) in the same range as the existing
# E7.5_rep2 / E8.5_rep2 caches, and holds true_false_ratio at the 10.0 those runs used so
# AUPRC stays comparable across samples.
#
# max_cells_per_pair is 64, not the earlier 24: the throughput probe (job 3788640) showed
# that raising cells/edge costs only ~4% throughput, because the expensive frozen TF-DNA
# path scales with edges x peaks and not with cells. Training resamples its cells from the
# full 1,896-metacell pool, but val/test read the frozen bags built here, so building them
# at 64 keeps evaluation on the same cell count the model trains at.
# true_false_ratio 10.0 -> 5.0 on 2026-08-20: negatives per positive, so the sampled
# positive rate goes from ~0.097 to ~0.17. Note this does NOT create more positives -- the
# positive count per TF is set by pct_true_edges and is unchanged; only the negatives shrink.
# Two consequences to carry forward: chance AUPRC moves with it, so AUPRC is not comparable
# to any earlier run, and the per-TF weights w_t = n_neg_t/n_pos_t roughly halve, which
# shifts what --per_tf_pos_weight_max actually binds on (median weight was 19.29 at 10.0).
#
# --min_positives_per_tf 200 drops train/val-pool TFs with fewer than 200 ground-truth
# edges, about 50 cached positives at this dataset's 0.268 positives-per-GT-edge rate.
# It removes 8 train and 4 val TFs, among them BAZ2A and TAF1 (ZERO cached positives),
# KDM5B (2) and ETV2 (3). Test TFs are deliberately exempt: they are the NMP regulators the
# experiment exists to measure, and dropping any would break comparison with scored runs.
max_cells_per_pair=64
max_peaks_per_tg=25
peak_flank_size=128
pct_true_edges=0.3
true_false_ratio=5.0

# val_tf_frac raised 0.15 -> 0.25 on 2026-08-19. At 0.15 the split produced 16 validation
# TFs of which only 12 were scorable: BAZ2A and GCM1 had ZERO positive edges, ETV2 had 1 and
# KDM5B had 2 out of ~16,740 each. An AUROC built on a single positive is that one edge's
# percentile rank, yet it carried the same 1/14 weight in the macro average as a TF with
# 3,962 positives -- ETV2 alone swung 0.833 -> 0.505 between runs 3793729 and 3799581 and
# accounted for roughly half the entire apparent macro decline. Macro across run 3799581's
# 19 checkpoints was 0.6239 +/- 0.0117 with a +0.18 epoch correlation: noise, not signal.
# 0.25 takes the validation set to ~27 TFs (~22 scorable) at a cost of ~11 training TFs.
# That trade is worth taking here because the per-TF diagnostic found NO memorisation of
# training TFs (held-out TFs improved 7x more), so the marginal training TF is currently
# worth less than the marginal validation TF.
#
# NOTE this does not fix the root cause: split_ground_truth_by_tf draws validation TFs
# uniformly at random from the training pool with no minimum positive-edge count
# (build_tf_to_tg_train_data.py:162), so ~18% of any draw is unusable. Widening raises the
# absolute count of scorable TFs without changing that fraction. A minimum-positives
# constraint on the draw is the actual fix and is not implemented.
echo "[INFO] Building and caching TF-TG training data (TF split)..."
python3 ${PROJECT_DIR}/scripts/build_tf_to_tg_train_data.py \
    --split_mode tf \
    --val_tf_frac 0.25 \
    --min_positives_per_tf 200 \
    --max_cells_per_pair $max_cells_per_pair \
    --max_peaks_per_tg $max_peaks_per_tg \
    --pct_true_edges $pct_true_edges \
    --true_false_ratio $true_false_ratio \
    --peak_flank_size $peak_flank_size \
    --num_cpu $SLURM_CPUS_PER_TASK \
    --force_reload
