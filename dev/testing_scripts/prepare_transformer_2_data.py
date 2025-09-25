import os
import torch
import pandas as pd
import logging
import pybedtools
import json
import pickle
import random

logging.basicConfig(level=logging.INFO, format="%(message)s")

WINDOW_SIZE = 50000

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
RAW_MESC_DATA_DIR = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS/DS014_DOI496239_MOUSE_ESC_RAW_FILES"
MESC_PEAK_MATRIX_FILE = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS/DS014_DOI496239_MOUSE_ESCDAYS7AND8/scATAC_PeakMatrix.txt"

MM10_GENOME_DIR = os.path.join(PROJECT_DIR, "data/reference_genome/mm10")
MM10_CHROM_SIZES_FILE = os.path.join(MM10_GENOME_DIR, "chrom.sizes")
MM10_GENE_TSS_FILE = os.path.join(PROJECT_DIR, "data/genome_annotation/mm10/mm10_TSS.bed")
GROUND_TRUTH_DIR = os.path.join(PROJECT_DIR, "ground_truth_files")
SAMPLE_INPUT_DIR = os.path.join(PROJECT_DIR, "input/transformer_input/mESC/")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output/transformer_testing_output")

def load_homer_tf_to_peak_results():
    assert os.path.exists(os.path.join(OUTPUT_DIR, "homer_tf_to_peak.parquet")), \
        "ERROR: Homer TF to peak output parquet file required"
        
    homer_results = pd.read_parquet(os.path.join(OUTPUT_DIR, "homer_tf_to_peak.parquet"), engine="pyarrow")
    homer_results = homer_results.reset_index(drop=True)
    homer_results["source_id"] = homer_results["source_id"].str.capitalize()
    
    return homer_results

def create_or_load_genomic_windows(chrom_id, force_recalculate=False):
    genome_window_file = os.path.join(MM10_GENOME_DIR, f"mm10_{chrom_id}_windows_{WINDOW_SIZE // 1000}kb.bed")
    if not os.path.exists(genome_window_file) or force_recalculate:
        
        logging.info("\nCreating genomic windows")
        mm10_genome_windows = pybedtools.bedtool.BedTool().window_maker(g=MM10_CHROM_SIZES_FILE, w=WINDOW_SIZE)
        mm10_windows = (
            mm10_genome_windows
            .filter(lambda x: x.chrom == chrom_id)  # TEMPORARY Restrict to one chromosome for testing
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

sample_name_list = ["E7.5_rep1", "E7.5_rep1", "E7.75_rep1", "E8.0_rep2", "E8.5_rep2",
                    "E8.75_rep2", "E7.5_rep2", "E8.0_rep1", "E8.5_rep1"]
holdout = ["E8.75_rep1"]
chrom_id = "chr19"

mm10_gene_tss_bed = pybedtools.BedTool(MM10_GENE_TSS_FILE)
gene_tss_df = (
    mm10_gene_tss_bed
    .filter(lambda x: x.chrom == chrom_id)
    .saveas(os.path.join(MM10_GENOME_DIR, "mm10_ch19_gene_tss.bed"))
    .to_dataframe()
    .sort_values(by="start", ascending=True)
    )


tf_list = list(load_homer_tf_to_peak_results()["source_id"].unique())
logging.info(f"\nHomer TFs: \t{tf_list[:5]}\n\tTotal {len(tf_list)} TFs")

TG_pseudobulk_global = []
TG_pseudobulk_samples = []
RE_pseudobulk_samples = []
peaks_df_samples = []

for sample_name in sample_name_list:
    sample_data_dir = os.path.join(SAMPLE_INPUT_DIR, sample_name)
    if os.path.exists(sample_data_dir) and len(os.listdir(sample_data_dir)) == 7 and sample_name not in holdout:
        logging.info(f"\nLoading Pseudobulk data for {sample_name}")
        TG_pseudobulk = pd.read_csv(os.path.join(sample_data_dir, "TG_pseudobulk.tsv"), sep="\t", index_col=0)
        RE_pseudobulk = pd.read_csv(os.path.join(sample_data_dir, "RE_pseudobulk.tsv"), sep="\t", index_col=0)

        logging.info("\n  - Total Pseudobulk Genes and Peaks")
        logging.info(f"\tTG_pseudobulk: {TG_pseudobulk.shape[0]:,} Genes x {TG_pseudobulk.shape[1]} metacells")
        logging.info(f"\tRE_pseudobulk: {RE_pseudobulk.shape[0]:,} Peaks x {RE_pseudobulk.shape[1]} metacells")

        TG_chr_specific = TG_pseudobulk.loc[TG_pseudobulk.index.intersection(gene_tss_df['name'].unique())]
        RE_chr_specific = RE_pseudobulk[RE_pseudobulk.index.str.startswith(f"{chrom_id}:")]

        logging.info(f"\n  - Restricted to {chrom_id} Genes and Peaks: ")
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
    
total_TG_pseudobulk_global = pd.concat(TG_pseudobulk_samples)
total_TG_pseudobulk_chr = pd.concat(TG_pseudobulk_samples)
total_RE_pseudobulk_chr = pd.concat(RE_pseudobulk_samples)
total_peaks_df = pd.concat(peaks_df_samples)

# Create genome windows and add index
mm10_windows = create_or_load_genomic_windows(chrom_id)
mm10_windows = mm10_windows.reset_index(drop=True)
mm10_windows["win_idx"] = mm10_windows.index

# Build peak -> window mapping
window_map = make_peak_to_window_map(total_peaks_df, mm10_windows)
logging.info(f"Mapped {len(window_map)} peaks to windows")

transformer_data_dir = os.path.join(PROJECT_DIR, "dev/testing_scripts/transformer_data")

with open(os.path.join(transformer_data_dir, "window_map.json"), "w") as f:
    json.dump(window_map, f, indent=4)

with open(os.path.join(transformer_data_dir, "tf_list.pickle"), "wb") as fp:
    pickle.dump(tf_list, fp)

total_TG_pseudobulk_global.to_csv(os.path.join(transformer_data_dir, f"TG_pseudobulk_global.csv"))
total_TG_pseudobulk_chr.to_csv(os.path.join(transformer_data_dir, f"TG_{chrom_id}_specific_pseudobulk_agg.csv"))
total_RE_pseudobulk_chr.to_csv(os.path.join(transformer_data_dir, f"RE_{chrom_id}_specific_pseudobulk_agg.csv"))

