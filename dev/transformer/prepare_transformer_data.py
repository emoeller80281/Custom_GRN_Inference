import os
import torch
import pandas as pd
import logging
import pybedtools
import json
import pickle
import random
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from grn_inference import utils

logging.basicConfig(level=logging.INFO, format="%(message)s")

WINDOW_SIZE = 25000
SAMPLE_NAME = "DS011"
CHROM_ID = "chr19"

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"

MM10_GENOME_DIR = os.path.join(PROJECT_DIR, "data/reference_genome/mm10")
MM10_CHROM_SIZES_FILE = os.path.join(MM10_GENOME_DIR, "chrom.sizes")
MM10_GENE_TSS_FILE = os.path.join(PROJECT_DIR, "data/genome_annotation/mm10/mm10_TSS.bed")
SAMPLE_INPUT_DIR = os.path.join(PROJECT_DIR, f"input/transformer_input/{SAMPLE_NAME}/")
OUTPUT_DIR = os.path.join(PROJECT_DIR, f"output/transformer_testing_output/{SAMPLE_NAME}/{CHROM_ID}")

TRANSFORMER_DATA_DIR = os.path.join(PROJECT_DIR, f"dev/transformer/transformer_data/{SAMPLE_NAME}_{CHROM_ID}")
os.makedirs(TRANSFORMER_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_or_load_genomic_windows(force_recalculate=False):
    genome_window_file = os.path.join(MM10_GENOME_DIR, f"mm10_{CHROM_ID}_windows_{WINDOW_SIZE // 1000}kb.bed")
    if not os.path.exists(genome_window_file) or force_recalculate:
        
        logging.info("\nCreating genomic windows")
        mm10_genome_windows = pybedtools.bedtool.BedTool().window_maker(g=MM10_CHROM_SIZES_FILE, w=WINDOW_SIZE)
        mm10_windows = (
            mm10_genome_windows
            .filter(lambda x: x.chrom == CHROM_ID)  # TEMPORARY Restrict to one chromosome for testing
            .saveas(genome_window_file)
            .to_dataframe()
        )
    else:
        
        logging.info("\nLoading existing genomic windows")
        mm10_windows = pybedtools.BedTool(genome_window_file).to_dataframe()
        
    return mm10_windows

def make_peak_to_window_map(peaks_bed: pd.DataFrame, windows_bed: pd.DataFrame) -> dict[str, int]:
    """
    Map each peak to the window it overlaps the most.
    If a peak ties across multiple windows, assign randomly.
    
    Parameters
    ----------
    peaks_bed : DataFrame
        Must have ['chrom','start','end','peak_id'] columns
    windows_bed : DataFrame
        Must have ['chrom','start','end','win_idx'] columns
    
    Returns
    -------
    mapping : dict[str, int]
        peak_id -> window index
    """
    bedtool_peaks = pybedtools.BedTool.from_dataframe(peaks_bed)
    bedtool_windows = pybedtools.BedTool.from_dataframe(windows_bed)

    overlaps = {}
    for interval in bedtool_peaks.intersect(bedtool_windows, wa=True, wb=True):
        peak_id = interval.name
        win_idx = int(interval.fields[-1])  # window index
        peak_start, peak_end = int(interval.start), int(interval.end)
        win_start, win_end = int(interval.fields[-3]), int(interval.fields[-2])

        # Compute overlap length
        overlap_len = min(peak_end, win_end) - max(peak_start, win_start)

        # Track best overlap for each peak
        if peak_id not in overlaps:
            overlaps[peak_id] = []
        overlaps[peak_id].append((overlap_len, win_idx))

    # Resolve ties by max overlap, then random choice
    mapping = {}
    for peak_id, ov_list in overlaps.items():
        max_overlap = max(ov_list, key=lambda x: x[0])[0]
        candidates = [win_idx for ol, win_idx in ov_list if ol == max_overlap]
        mapping[peak_id] = random.choice(candidates)  # pick randomly if tie

    return mapping

def calculate_peak_to_tg_distance_score(mesc_atac_peak_loc_df, gene_tss_df, force_recalculate=False):
    if not os.path.isfile(os.path.join(OUTPUT_DIR, "genes_near_peaks.parquet")) or force_recalculate:
        if "peak_tmp.bed" not in os.listdir(OUTPUT_DIR) or "tss_tmp.bed" not in os.listdir(OUTPUT_DIR) or force_recalculate:
        
            logging.info("Calculating peak to TG distance score")
            peak_bed = pybedtools.BedTool.from_dataframe(
                mesc_atac_peak_loc_df[["chrom", "start", "end", "peak_id"]]
                ).saveas(os.path.join(OUTPUT_DIR, "peak_tmp.bed"))

            tss_bed = pybedtools.BedTool.from_dataframe(
                gene_tss_df[["chrom", "start", "end", "name"]]
                ).saveas(os.path.join(OUTPUT_DIR, "tss_tmp.bed"))
            
        peak_bed = pybedtools.BedTool(os.path.join(OUTPUT_DIR, "peak_tmp.bed"))
        tss_bed = pybedtools.BedTool(os.path.join(OUTPUT_DIR, "tss_tmp.bed"))
    

        genes_near_peaks = utils.find_genes_near_peaks(peak_bed, tss_bed)

        # Restrict to peaks within 1 Mb of a gene TSS
        genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= 1e6]

        # Scale the TSS distance score by the exponential scaling factor
        genes_near_peaks = genes_near_peaks.copy()
        genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / 250000)

        genes_near_peaks.to_parquet(os.path.join(OUTPUT_DIR, "genes_near_peaks.parquet"), compression="snappy", engine="pyarrow")
    else:
        genes_near_peaks = pd.read_parquet(os.path.join(OUTPUT_DIR, "genes_near_peaks.parquet"), engine="pyarrow")
    
    return genes_near_peaks

# sample_name_list = ["E7.5_rep1", "E7.5_rep1", "E7.75_rep1", "E8.0_rep2", "E8.5_rep2",
#                     "E8.75_rep2", "E7.5_rep2", "E8.0_rep1", "E8.5_rep1"]
holdout = ["E8.75_rep1"]

sample_name_list = ["DS011_sample1"]

mm10_gene_tss_bed = pybedtools.BedTool(MM10_GENE_TSS_FILE)
gene_tss_df = (
    mm10_gene_tss_bed
    .filter(lambda x: x.chrom == CHROM_ID)
    .saveas(os.path.join(MM10_GENOME_DIR, f"mm10_{CHROM_ID}_gene_tss.bed"))
    .to_dataframe()
    .sort_values(by="start", ascending=True)
    )

with open(os.path.join(PROJECT_DIR, f"dev/transformer/mesc_homer_tfs.pkl"), 'rb') as f:
    tf_names: list = pickle.load(f)
    
logging.info(f"\nHomer TFs: \t{tf_names[:5]}\n\tTotal {len(tf_names)} TFs")

TG_pseudobulk_global = []
TG_pseudobulk_samples = []
RE_pseudobulk_samples = []
peaks_df_samples = []

for sample_name in sample_name_list:
    sample_data_dir = os.path.join(SAMPLE_INPUT_DIR, sample_name)
    if not os.path.exists(sample_data_dir):
        logging.warning(f"Skipping {sample_name}: directory not found")
        continue
    if len(os.listdir(sample_data_dir)) != 5:
        logging.warning(f"Skipping {sample_name}: expected 5 files, found {len(os.listdir(sample_data_dir))}")
        continue
    if sample_name in holdout:
        logging.warning(f"Skipping {sample_name}: in holdout list")
        continue
    else:
        logging.info(f"\nLoading Pseudobulk data for {sample_name}")
        TG_pseudobulk = pd.read_csv(os.path.join(sample_data_dir, "TG_pseudobulk.tsv"), sep="\t", index_col=0)
        RE_pseudobulk = pd.read_csv(os.path.join(sample_data_dir, "RE_pseudobulk.tsv"), sep="\t", index_col=0)

        logging.info("\n  - Total Pseudobulk Genes and Peaks")
        logging.info(f"\tTG_pseudobulk: {TG_pseudobulk.shape[0]:,} Genes x {TG_pseudobulk.shape[1]} metacells")
        logging.info(f"\tRE_pseudobulk: {RE_pseudobulk.shape[0]:,} Peaks x {RE_pseudobulk.shape[1]} metacells")

        TG_chr_specific = TG_pseudobulk.loc[TG_pseudobulk.index.intersection(gene_tss_df['name'].unique())]
        RE_chr_specific = RE_pseudobulk[RE_pseudobulk.index.str.startswith(f"{CHROM_ID}:")]

        logging.info(f"\n  - Restricted to {CHROM_ID} Genes and Peaks: ")
        logging.info(f"\tTG_chr_specific: {TG_chr_specific.shape[0]} Genes x {TG_chr_specific.shape[1]} metacells")
        logging.info(f"\tRE_chr_specific: {RE_chr_specific.shape[0]:,} Peaks x {RE_chr_specific.shape[1]} metacells")

        peaks_df = (
            RE_chr_specific.index.to_series()
            .str.split("[:-]", expand=True)
            .rename(columns={0: "chrom", 1: "start", 2: "end"})
        )
        peaks_df["start"] = peaks_df["start"].astype(int)
        peaks_df["end"] = peaks_df["end"].astype(int)
        peaks_df["peak_id"] = RE_chr_specific.index
        
        TG_pseudobulk_global.append(TG_pseudobulk)
        TG_pseudobulk_samples.append(TG_chr_specific)
        RE_pseudobulk_samples.append(RE_chr_specific)
        peaks_df_samples.append(peaks_df)
    
# Aggregate across samples
total_TG_pseudobulk_chr = pd.concat(TG_pseudobulk_samples).groupby(level=0).sum()
total_RE_pseudobulk_chr = pd.concat(RE_pseudobulk_samples).groupby(level=0).sum()
total_peaks_df = pd.concat(peaks_df_samples).groupby(level=0).first()

# TF expression (genome-wide TFs)
total_TG_pseudobulk_global = pd.concat(TG_pseudobulk_samples).groupby(level=0).sum()
genome_wide_tf_expression = total_TG_pseudobulk_global.reindex(tf_names).fillna(0).values.astype("float32")

# Scale TG expression
scaler = StandardScaler()
TG_scaled = scaler.fit_transform(total_TG_pseudobulk_chr.values.astype("float32"))

# Save scaler for inverse-transform
joblib.dump(scaler, os.path.join(TRANSFORMER_DATA_DIR, f"tg_scaler_{CHROM_ID}.pkl"))

# Create genome windows
mm10_windows = create_or_load_genomic_windows()
mm10_windows = mm10_windows.reset_index(drop=True)
mm10_windows["win_idx"] = mm10_windows.index

# --- Calculate Peak-to-TG Distance Scores ---
genes_near_peaks = calculate_peak_to_tg_distance_score(
    mesc_atac_peak_loc_df=total_peaks_df,  # peak locations DataFrame
    gene_tss_df=gene_tss_df,
    force_recalculate=False
)

# Save as metadata for downstream use
dist_path = os.path.join(TRANSFORMER_DATA_DIR, f"genes_near_peaks_{CHROM_ID}.parquet")
genes_near_peaks.to_parquet(dist_path, compression="snappy", engine="pyarrow")
logging.info(f"Saved peak-to-TG distance scores to {dist_path}")


# Build peak -> window mapping
window_map = make_peak_to_window_map(total_peaks_df, mm10_windows)
logging.info(f"Mapped {len(window_map)} peaks to windows")

# ----- Save Precomputed Tensors -----
# TF tensor
tf_tensor_all = torch.tensor(genome_wide_tf_expression, dtype=torch.float32)
torch.save(tf_tensor_all, os.path.join(TRANSFORMER_DATA_DIR, "tf_tensor_all.pt"))

# TG tensor (scaled)
tg_tensor_all = torch.tensor(TG_scaled, dtype=torch.float32)
torch.save(tg_tensor_all, os.path.join(TRANSFORMER_DATA_DIR, f"tg_tensor_all_{CHROM_ID}.pt"))

# ATAC window tensor
rows, cols, vals = [], [], []
for peak, win_idx in window_map.items():
    if peak in total_RE_pseudobulk_chr.index:
        peak_idx = total_RE_pseudobulk_chr.index.get_loc(peak)
        rows.append(win_idx)
        cols.append(peak_idx)
        vals.append(1.0)
W = sp.csr_matrix((vals, (rows, cols)), shape=(mm10_windows.shape[0], total_RE_pseudobulk_chr.shape[0]))
atac_window_tensor_all = torch.tensor(W @ total_RE_pseudobulk_chr.values, dtype=torch.float32)
torch.save(atac_window_tensor_all, os.path.join(TRANSFORMER_DATA_DIR, f"atac_window_tensor_all_{CHROM_ID}.pt"))

# Save metadata
with open(os.path.join(TRANSFORMER_DATA_DIR, f"window_map_{CHROM_ID}.json"), "w") as f:
    json.dump(window_map, f, indent=4)
    
tg_names = total_TG_pseudobulk_chr.index.tolist()
with open(os.path.join(TRANSFORMER_DATA_DIR, f"tg_names_{CHROM_ID}.json"), "w") as f:
    json.dump(tg_names, f)
    
# Build [num_windows x num_tg] distance bias matrix
num_windows = mm10_windows.shape[0]
num_tg = len(tg_names)
dist_bias = torch.zeros((num_windows, num_tg), dtype=torch.float32)

tg_index_map = {tg: i for i, tg in enumerate(tg_names)}

for _, row in genes_near_peaks.iterrows():
    peak_id, tg, score = row["peak_id"], row["target_id"], row["TSS_dist_score"]
    if peak_id in window_map and tg in tg_index_map:
        win_idx = window_map[peak_id]
        tg_idx = tg_index_map[tg]
        # store max score if multiple peaks map to the same window–TG
        dist_bias[win_idx, tg_idx] = max(dist_bias[win_idx, tg_idx], score)

# Save to disk
torch.save(dist_bias, os.path.join(TRANSFORMER_DATA_DIR, f"dist_bias_{CHROM_ID}.pt"))
logging.info(f"Saved distance bias tensor with shape {dist_bias.shape}")
    
metacell_names = total_TG_pseudobulk_global.columns.tolist()
with open(os.path.join(TRANSFORMER_DATA_DIR, f"metacell_names.json"), "w") as f:
    json.dump(metacell_names, f)
    
with open(os.path.join(TRANSFORMER_DATA_DIR, "tf_names.pickle"), "wb") as fp:
    pickle.dump(tf_names, fp)

logging.info("\nPreprocessing complete. Saved TF, TG, ATAC tensors to TRANSFORMER_DATA_DIR")