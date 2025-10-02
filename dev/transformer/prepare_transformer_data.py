import os
import torch
import pandas as pd
import logging
import pybedtools
import json
import pickle
import random
import scipy.sparse as sp
from torch import logsumexp
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
from grn_inference import utils
from typing import List, Dict, Tuple, Sequence, Iterable, Optional

import sys
from dev.transformer.build_motif_mask.moods_scan import run_moods_scan

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Reads the number of CPUs from the slurm job, else defaults to 1
NUM_CPUS = int(os.getenv("NUM_CPUS", "1"))
print(f"Using {NUM_CPUS} CPUs")

SAMPLE_NAME = "mESC"
CHROM_ID = "chr1"

# Window parameters
WINDOW_SIZE = 5_000             # Aggregates peaks within WINDOW_SIZE bp genomic tiles
DISTANCE_SCALE_FACTOR = 3_000   # Weights the peak-gene TSS distance score. Lower numbers = faster dropoff
MAX_PEAK_DISTANCE = 10_000      # Masks out peaks further than this distance from the gene TSS
DIST_BIAS_MODE = "max"

# TF-peak binding calculation parameters
MOODS_PVAL_THRESHOLD=1e-3

sample_name_list = ["E7.5_rep1", "E7.75_rep1", "E8.0_rep2", "E8.5_rep2",
                    "E8.75_rep2", "E7.5_rep2", "E8.0_rep1", "E8.5_rep1"]
holdout = ["E8.75_rep1"]

# sample_name_list = ["E8.75_rep1"]
# holdout = []

# sample_name_list = ["DS011_sample1"]

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"

GENOME_DIR = os.path.join(PROJECT_DIR, "data/reference_genome/mm10")
CHROM_SIZES_FILE = os.path.join(GENOME_DIR, "chrom.sizes")
GENE_TSS_FILE = os.path.join(PROJECT_DIR, "data/genome_annotation/mm10/mm10_TSS.bed")
SAMPLE_INPUT_DIR = os.path.join(PROJECT_DIR, f"input/transformer_input/{SAMPLE_NAME}/")
OUTPUT_DIR = os.path.join(PROJECT_DIR, f"output/transformer_testing_output/{SAMPLE_NAME}/{CHROM_ID}")

COMMON_DATA = os.path.join(PROJECT_DIR, f"dev/transformer/transformer_data/common")

SAMPLE_DATA_DIR = os.path.join(PROJECT_DIR, f"dev/transformer/transformer_data/{SAMPLE_NAME}")
CHROM_SPECIFIC_DATA_DIR = os.path.join(SAMPLE_DATA_DIR, CHROM_ID)
JASPAR_PFM_DIR=os.path.join(PROJECT_DIR, "data/motif_information/JASPAR/pfm_files")

# TF and TG vocab files
common_tf_vocab_file = os.path.join(COMMON_DATA, f"tf_vocab.json")
common_tg_vocab_file = os.path.join(COMMON_DATA, f"tg_vocab_{CHROM_ID}.json")

# TF, TG, and Window tensors
atac_tensor_path = os.path.join(CHROM_SPECIFIC_DATA_DIR, f"atac_window_tensor_all_{CHROM_ID}.pt")
tg_tensor_path = os.path.join(CHROM_SPECIFIC_DATA_DIR, f"tg_tensor_all_{CHROM_ID}.pt")
tf_tensor_path = os.path.join(SAMPLE_DATA_DIR, "tf_tensor_all.pt")

# Sample-specific output files
metacell_name_file = os.path.join(SAMPLE_DATA_DIR, "metacell_names.json")
sample_tf_name_file = os.path.join(SAMPLE_DATA_DIR, "tf_names.json")
sample_tg_name_file = os.path.join(CHROM_SPECIFIC_DATA_DIR, f"tg_names_{CHROM_ID}.json")
sample_window_map_file = os.path.join(CHROM_SPECIFIC_DATA_DIR, f"window_map_{CHROM_ID}.json")
sample_scaler_file = os.path.join(CHROM_SPECIFIC_DATA_DIR, f"tg_scaler_{CHROM_ID}.pkl")
peak_to_tss_dist_path = os.path.join(CHROM_SPECIFIC_DATA_DIR, f"genes_near_peaks_{CHROM_ID}.parquet")
dist_bias_file = os.path.join(CHROM_SPECIFIC_DATA_DIR, f"dist_bias_{CHROM_ID}.pt")
tf_id_file = os.path.join(SAMPLE_DATA_DIR, "tf_ids.pt")
tg_id_file = os.path.join(CHROM_SPECIFIC_DATA_DIR, f"tg_ids_{CHROM_ID}.pt")
manifest_file = os.path.join(CHROM_SPECIFIC_DATA_DIR, "manifest.json")
motif_mask_file = os.path.join(CHROM_SPECIFIC_DATA_DIR, f"motif_mask_{CHROM_ID}.pt")
moods_sites_file = os.path.join(CHROM_SPECIFIC_DATA_DIR, f"{CHROM_ID}_moods_sites.tsv")

os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
os.makedirs(CHROM_SPECIFIC_DATA_DIR, exist_ok=True)
os.makedirs(COMMON_DATA, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_or_load_genomic_windows(genome_dir, window_size, chrom_id, chrom_sizes_file, force_recalculate=False):
    genome_window_file = os.path.join(genome_dir, f"{chrom_id}_windows_{window_size // 1000}kb.bed")
    if not os.path.exists(genome_window_file) or force_recalculate:
        
        logging.info("\nCreating genomic windows")
        mm10_genome_windows = pybedtools.bedtool.BedTool().window_maker(g=chrom_sizes_file, w=window_size)
        mm10_windows = (
            mm10_genome_windows
            .filter(lambda x: x.chrom == chrom_id)  # TEMPORARY Restrict to one chromosome for testing
            .saveas(genome_window_file)
            .to_dataframe()
        )
    else:
        
        logging.info("\nLoading existing genomic windows")
        mm10_windows = pybedtools.BedTool(genome_window_file).to_dataframe()
        
    mm10_windows = mm10_windows.reset_index(drop=True)
    mm10_windows["win_idx"] = mm10_windows.index
        
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

def calculate_peak_to_tg_distance_score(
    output_dir, 
    mesc_atac_peak_loc_df, 
    gene_tss_df, 
    max_peak_distance=1e6, 
    distance_factor_scale=25000, 
    force_recalculate=False
    ) -> pd.DataFrame:
    if not os.path.isfile(os.path.join(output_dir, "genes_near_peaks.parquet")) or force_recalculate:
        if "peak_tmp.bed" not in os.listdir(output_dir) or "tss_tmp.bed" not in os.listdir(output_dir) or force_recalculate:
        
            logging.info("Calculating peak to TG distance score")
            peak_bed = pybedtools.BedTool.from_dataframe(
                mesc_atac_peak_loc_df[["chrom", "start", "end", "peak_id"]]
                ).saveas(os.path.join(output_dir, "peak_tmp.bed"))

            tss_bed = pybedtools.BedTool.from_dataframe(
                gene_tss_df[["chrom", "start", "end", "name"]]
                ).saveas(os.path.join(output_dir, "tss_tmp.bed"))
            
        peak_bed = pybedtools.BedTool(os.path.join(output_dir, "peak_tmp.bed"))
        tss_bed = pybedtools.BedTool(os.path.join(output_dir, "tss_tmp.bed"))
    
        genes_near_peaks = utils.find_genes_near_peaks(peak_bed, tss_bed, tss_distance_cutoff=max_peak_distance)

        # Restrict to peaks within 1 Mb of a gene TSS
        genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= max_peak_distance]

        # Scale the TSS distance score by the exponential scaling factor
        genes_near_peaks = genes_near_peaks.copy()
        genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / distance_factor_scale)

        genes_near_peaks.to_parquet(os.path.join(output_dir, "genes_near_peaks.parquet"), compression="snappy", engine="pyarrow")
    else:
        genes_near_peaks = pd.read_parquet(os.path.join(output_dir, "genes_near_peaks.parquet"), engine="pyarrow")
    
    return genes_near_peaks

def build_motif_mask(tf_names, tg_names, motif_hits_df, genes_near_peaks):
    """
    tf_names        : list of TF names
    tg_names        : list of TG names
    motif_hits_df   : DataFrame with columns ['peak_id','TF','logodds']
    genes_near_peaks: DataFrame with columns ['peak_id','target_id']
    """
    TF = len(tf_names)
    TG = len(tg_names)

    # Map TFs and TGs to indices
    tf_index = {tf: i for i, tf in enumerate(tf_names)}
    tg_index = {tg: i for i, tg in enumerate(tg_names)}

    mask = np.zeros((TG, TF), dtype=np.float32)

    # Map peak → list of TGs
    peak_to_tgs = genes_near_peaks.groupby("peak_id")["target_id"].apply(list).to_dict()

    # Fill mask directly
    for _, row in motif_hits_df.iterrows():
        pid, tf, score = row["peak_id"], row["TF"], row["logodds"]
        if tf not in tf_index or pid not in peak_to_tgs:
            continue
        for tg in peak_to_tgs[pid]:
            j, i = tf_index[tf], tg_index.get(tg)
            if i is not None:
                mask[i, j] = max(mask[i, j], score)

    return mask

def precompute_input_tensors(
    output_dir: str,
    genome_wide_tf_expression: np.ndarray,   # [num_TF, num_cells]
    TG_scaled: np.ndarray,                   # [num_TG_chr, num_cells] (already standardized)
    total_RE_pseudobulk_chr,                 # pd.DataFrame: rows=peak_id, cols=metacells
    window_map,
    windows,                            # pd.DataFrame with shape[0] = num_windows
    dtype: torch.dtype = torch.float32,
):
    """
    Builds & saves:
      - tf_tensor_all.pt                        [num_TF, num_cells]
      - tg_tensor_all_{chr}.pt                  [num_TG_chr, num_cells]
      - atac_window_tensor_all_{chr}.pt         [num_windows, num_cells]

    Returns:
      (tf_tensor_all, tg_tensor_all, atac_window_tensor_all)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- TF tensor ----
    tf_tensor_all = torch.as_tensor(
        np.asarray(genome_wide_tf_expression, dtype=np.float32), dtype=dtype
    )

    # ---- TG tensor (scaled) ----
    tg_tensor_all = torch.as_tensor(
        np.asarray(TG_scaled, dtype=np.float32), dtype=dtype
    )

    # ---- ATAC window tensor ----
    num_windows = int(windows.shape[0])
    num_peaks   = int(total_RE_pseudobulk_chr.shape[0])

    rows, cols, vals = [], [], []
    peak_to_idx = {p: i for i, p in enumerate(total_RE_pseudobulk_chr.index)}
    for peak_id, win_idx in window_map.items():
        peak_idx = peak_to_idx.get(peak_id)
        if peak_idx is not None and 0 <= win_idx < num_windows:
            rows.append(win_idx)
            cols.append(peak_idx)
            vals.append(1.0)

    if not rows:
        raise ValueError("No peaks from window_map matched rows in total_RE_pseudobulk_chr.")

    W = sp.csr_matrix((vals, (rows, cols)), shape=(num_windows, num_peaks))
    atac_window = W @ total_RE_pseudobulk_chr.values  # [num_windows, num_cells]

    atac_window_tensor_all = torch.as_tensor(
        atac_window.astype(np.float32), dtype=dtype
    )

    return tf_tensor_all, tg_tensor_all, atac_window_tensor_all

def match_gene_to_vocab(
    names: Sequence[str],
    vocab: Dict[str, int],
    tensor_all,              # torch.Tensor or np.ndarray shaped [N, C]
    label: str = "genes",
) -> Tuple:
    """
    Align rows of `tensor_all` (N x C) to a common `vocab`.
    Keeps only names present in `vocab`, preserving original order.

    Returns:
      tensor_kept : filtered tensor [N_keep, C]
      names_kept  : list[str] kept, in row order
      ids         : list[int] vocab ids aligned to names_kept
    """
    assert len(names) == (tensor_all.shape[0]), \
        f"{label}: len(names) != tensor rows ({len(names)} vs {tensor_all.shape[0]})"

    ids, keep_idx, dropped = [], [], []
    for i, n in enumerate(names):
        vid = vocab.get(n)
        if vid is not None:
            ids.append(vid)
            keep_idx.append(i)
        else:
            dropped.append(n)

    if dropped:
        logging.warning(f"Dropping {len(dropped)} {label} not in common vocab (e.g. {dropped[:5]})")
        tensor_kept = tensor_all[keep_idx, :]
        names_kept  = [names[i] for i in keep_idx]
    else:
        tensor_kept = tensor_all
        names_kept  = list(names)

    return tensor_kept, names_kept, ids

def build_distance_bias(
    genes_near_peaks: pd.DataFrame,
    window_map: Dict[str, int],
    tg_names_kept: Iterable[str],
    num_windows: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    mode: str = "logsumexp",   # "max" | "sum" | "mean" | "logsumexp"
) -> torch.Tensor:
    """
    Build a [num_windows x num_tg_kept] distance-bias tensor aligned to the kept TGs.

    Args:
        genes_near_peaks: DataFrame with at least columns:
            - 'peak_id' (str): peak identifier that matches keys in window_map
            - 'target_id' (str): TG name
            - 'TSS_dist_score' (float): precomputed distance score
        window_map: dict mapping peak_id -> window index (0..num_windows-1).
        tg_names_kept: iterable of TG names kept after vocab filtering; column order target.
        num_windows: total number of genomic windows.
        dtype: torch dtype for the output tensor (default: torch.float32).
        device: optional torch device for the output tensor.
        mode: pooling strategy if multiple peaks map to the same (window, TG).
              Options = {"max", "sum", "mean", "logsumexp"}

    Returns:
        dist_bias: torch.Tensor of shape [num_windows, len(tg_names_kept)],
                   where each entry is an aggregated TSS distance score.
    """
    tg_names_kept = list(tg_names_kept)
    num_tg_kept = len(tg_names_kept)

    dist_bias = torch.zeros((num_windows, num_tg_kept), dtype=dtype, device=device)
    tg_index_map = {tg: i for i, tg in enumerate(tg_names_kept)}

    from collections import defaultdict
    scores_map = defaultdict(list)

    # Collect all scores for each (window, TG)
    for _, row in genes_near_peaks.iterrows():
        win_idx = window_map.get(row["peak_id"])
        tg_idx  = tg_index_map.get(row["target_id"])
        if win_idx is not None and tg_idx is not None:
            scores_map[(win_idx, tg_idx)].append(float(row["TSS_dist_score"]))

    # Aggregate according to pooling mode
    for (win_idx, tg_idx), scores in scores_map.items():
        scores_tensor = torch.tensor(scores, dtype=dtype, device=device)

        if mode == "max":
            dist_bias[win_idx, tg_idx] = scores_tensor.max()
        elif mode == "sum":
            dist_bias[win_idx, tg_idx] = scores_tensor.sum()
        elif mode == "mean":
            dist_bias[win_idx, tg_idx] = scores_tensor.mean()
        elif mode == "logsumexp":
            dist_bias[win_idx, tg_idx] = torch.logsumexp(scores_tensor, dim=0)
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

    return dist_bias

def update_vocab(vocab_file, new_names, label="GENE"):
    # Load existing vocab or create new
    if os.path.isfile(vocab_file):
        with open(vocab_file) as f:
            vocab = json.load(f)
    else:
        vocab = {}

    # Standardize new names
    new_names = [standardize_name(n) for n in new_names]

    # Add missing names with new indices
    updated = False
    for name in sorted(set(new_names)):
        if name not in vocab:
            vocab[name] = len(vocab)
            updated = True
            logging.info(f"Added new {label}: {name}")

    # Save only if something changed
    if updated:
        atomic_json_dump(vocab, vocab_file)
        logging.info(f"Updated {label} vocab with {len(vocab)} entries")

    return vocab

def atomic_json_dump(obj, path):
    """Safe JSON dump, avoids race conditions by making a tmp file first, then updating the name"""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def make_chrom_gene_tss_df(gene_tss_file, chrom_id, genome_dir):
    gene_tss_bed = pybedtools.BedTool(gene_tss_file)
    gene_tss_df = (
        gene_tss_bed
        .filter(lambda x: x.chrom == chrom_id)
        .saveas(os.path.join(genome_dir, f"{chrom_id}_gene_tss.bed"))
        .to_dataframe()
        .sort_values(by="start", ascending=True)
        )
    return gene_tss_df

def _agg_sum(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Sum rows across samples; rows are aligned by index."""
    if len(dfs) == 0:
        raise ValueError("No DataFrames provided to aggregate.")
    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs).groupby(level=0).sum()

def _agg_first(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Keep first occurrence per index (for metadata like peak coords)."""
    if len(dfs) == 0:
        raise ValueError("No DataFrames provided to aggregate.")
    if len(dfs) == 1:
        return dfs[0]
    return pd.concat(dfs).groupby(level=0).first()

def standardize_name(name: str) -> str:
    """Convert gene/motif name to capitalization style (e.g. 'Hoxa2')."""
    if not isinstance(name, str):
        return name
    return name.capitalize()

if __name__ == "__main__":
    # Create or load the gene TSS information for the chromosome
    gene_tss_df = make_chrom_gene_tss_df(
        gene_tss_file=GENE_TSS_FILE,
        chrom_id=CHROM_ID,
        genome_dir=GENOME_DIR
    )

    # Load the global TF vocab
    with open(os.path.join(PROJECT_DIR, f"dev/transformer/mesc_homer_tfs.pkl"), 'rb') as f:
        tf_names: list = pickle.load(f)
    
    tf_names = [standardize_name(n) for n in tf_names]
        
    logging.info(f"\nLoaded {SAMPLE_NAME} TFs: {len(tf_names)} TFs")

    # ----- Combe Pseudobulk Data into a Training Dataset -----
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
        
    # Aggregate the pseudobulk for all samples
    total_TG_pseudobulk_global = _agg_sum(TG_pseudobulk_global)
    total_TG_pseudobulk_chr    = _agg_sum(TG_pseudobulk_samples)
    total_RE_pseudobulk_chr    = _agg_sum(RE_pseudobulk_samples)
    total_peaks_df             = _agg_first(peaks_df_samples)

    tg_names = total_TG_pseudobulk_chr.index.tolist()

    # Genome-wide TF expression for all samples
    genome_wide_tf_expression = total_TG_pseudobulk_global.reindex(tf_names).fillna(0).values.astype("float32")
    metacell_names = total_TG_pseudobulk_global.columns.tolist()

    # Scale TG expression
    scaler = StandardScaler()
    TG_scaled = scaler.fit_transform(total_TG_pseudobulk_chr.T.astype("float32")).T

    # Create genome windows
    mm10_windows = create_or_load_genomic_windows(
        genome_dir=GENOME_DIR,
        window_size=WINDOW_SIZE,
        chrom_id=CHROM_ID,
        chrom_sizes_file=CHROM_SIZES_FILE
    )
    num_windows = mm10_windows.shape[0]

    # --- Calculate Peak-to-TG Distance Scores ---
    genes_near_peaks = calculate_peak_to_tg_distance_score(
        output_dir=OUTPUT_DIR,
        mesc_atac_peak_loc_df=total_peaks_df,  # peak locations DataFrame
        gene_tss_df=gene_tss_df,
        max_peak_distance= MAX_PEAK_DISTANCE,
        distance_factor_scale= DISTANCE_SCALE_FACTOR,
        force_recalculate=True
    )
    
    peaks_bed = os.path.join(OUTPUT_DIR, "peak_tmp.bed")
    jaspar_pfm_paths = [os.path.join(JASPAR_PFM_DIR, f) for f in os.listdir(JASPAR_PFM_DIR) if f.endswith(".pfm")]
    
    run_moods_scan(
        peaks_bed=peaks_bed, 
        fasta_path=os.path.join(GENOME_DIR, f"{CHROM_ID}.fa"), 
        motif_paths=jaspar_pfm_paths, 
        out_tsv=moods_sites_file, 
        n_cpus=NUM_CPUS,
        pval_threshold=MOODS_PVAL_THRESHOLD, 
        bg="auto"
    )
    
    # Build peak -> window mapping
    logging.info(f"\nCreating peak to window map")
    window_map = make_peak_to_window_map(total_peaks_df, mm10_windows)
    logging.info(f"\tMapped {len(window_map)} peaks to windows")

    # Save Precomputed Tensors 
    logging.info(f"\nPrecomputing TF, TG, and ATAC tensors")
    tf_tensor_all, tg_tensor_all, atac_window_tensor_all = precompute_input_tensors(
        output_dir=SAMPLE_DATA_DIR,
        genome_wide_tf_expression=genome_wide_tf_expression,
        TG_scaled=TG_scaled,
        total_RE_pseudobulk_chr=total_RE_pseudobulk_chr,
        window_map=window_map,
        windows=mm10_windows,
    )
    logging.info(f"\t- Done!")

    # ----- Load common TF and TG vocab -----
    # Create a common TG vocabulary for the chromosome using the gene TSS
    logging.info(f"\nMatching TFs and TGs to global gene vocabulary")
    
    # Update TG vocab with whatever TGs exist in gene_tss_df
    tg_vocab = update_vocab(common_tg_vocab_file, gene_tss_df["name"].unique(), label="TG")

    # Update TF vocab with whatever TFs exist in tf_names
    tf_vocab = update_vocab(common_tf_vocab_file, tf_names, label="TF")
            
    tg_names = [standardize_name(n) for n in total_TG_pseudobulk_chr.index.tolist()]
    
    tf_tensor_all, tf_names_kept, tf_ids = match_gene_to_vocab(
        [standardize_name(tf) for tf in tf_names],
        tf_vocab,
        tf_tensor_all,
        label="TF"
    )

    tg_tensor_all, tg_names_kept, tg_ids = match_gene_to_vocab(
        [standardize_name(tg) for tg in tg_names],
        tg_vocab,
        tg_tensor_all,
        label="TG"
    )

    logging.info(f"\tMatched {len(tf_names_kept)} TFs to global vocab")
    logging.info(f"\tMatched {len(tg_names_kept)} TGs to global vocab")
    logging.info(f"\t- Done!")
    
    logging.info(f"\nBuilding motif mask")
    moods_hits = pd.read_csv(moods_sites_file, sep="\t")
    
    # Drop rows with missing TFs just in case
    moods_hits = moods_hits.dropna(subset=["TF"])
    
    # Strip ".pfm" suffix from TF names if present
    moods_hits["TF"] = moods_hits["TF"].str.replace(r"\.pfm$", "", regex=True).apply(standardize_name)

    # Build motif mask using merged info
    motif_mask = build_motif_mask(
        tf_names=tf_names_kept,
        tg_names=tg_names_kept,
        motif_hits_df=moods_hits,
        genes_near_peaks=genes_near_peaks
    )
    logging.info(f"\t- Done!")

    
    if not tf_ids: raise ValueError("No TFs matched the common vocab.")
    if not tg_ids: raise ValueError("No TGs matched the common vocab.")
        
    # Build distance bias [num_windows x num_tg_kept] aligned to kept TGs
    logging.info(f"\nBuilding distance bias")
    dist_bias = build_distance_bias(
        genes_near_peaks=genes_near_peaks,
        window_map=window_map,
        tg_names_kept=tg_names_kept,
        num_windows=num_windows,
        dtype=torch.float32,
        mode=DIST_BIAS_MODE
    )
    logging.info(f"\t- Done!")
    
    # ----- Writing Output Files -----
    logging.info(f"\nWriting output files")
    # Save the Window, TF, and TG expression tensors
    torch.save(atac_window_tensor_all, atac_tensor_path)
    torch.save(tg_tensor_all, tg_tensor_path)
    torch.save(tf_tensor_all, tf_tensor_path)
    
    # Write the peak to gene TSS distance scores
    genes_near_peaks.to_parquet(peak_to_tss_dist_path, compression="snappy", engine="pyarrow")
    logging.info(f"Saved peak-to-TG distance scores to {peak_to_tss_dist_path}")

    # Save scaler for inverse-transform
    joblib.dump(scaler, sample_scaler_file)

    # Save the peak -> window map for the sample
    atomic_json_dump(window_map, sample_window_map_file)

    # Write TF and TG names and global vocab indices present in the sample
    atomic_json_dump(tf_names_kept, sample_tf_name_file)
    atomic_json_dump(tg_names_kept, sample_tg_name_file)

    torch.save(torch.tensor(tf_ids, dtype=torch.long), tf_id_file)
    torch.save(torch.tensor(tg_ids, dtype=torch.long), tg_id_file)

    # Write the distance bias and metacell names for the sample
    torch.save(dist_bias, dist_bias_file)
    logging.info(f"Saved distance bias tensor with shape {tuple(dist_bias.shape)}")

    atomic_json_dump(metacell_names, metacell_name_file)
    
    torch.save(torch.from_numpy(motif_mask), motif_mask_file)

    # Manifest of general sample info and file paths
    manifest = {
        "sample": SAMPLE_NAME,
        "chrom": CHROM_ID,
        "num_windows": int(num_windows),
        "num_tfs": int(len(tf_names_kept)),
        "num_tgs": int(len(tg_names_kept)),
        "Distance tau": DISTANCE_SCALE_FACTOR,
        "Max peak-TG distance": MAX_PEAK_DISTANCE,
        "paths": {
            "tf_tensor_all": tf_tensor_path,
            "tg_tensor_all": tg_tensor_path,
            "atac_window_tensor_all": atac_tensor_path,
            "dist_bias": dist_bias_file,
            "tf_ids": tf_id_file,
            "tg_ids": tg_id_file,
            "tf_names": sample_tf_name_file,                 # sample-level
            "tg_names": sample_tg_name_file,                 # chrom-level
            "common_tf_vocab": common_tf_vocab_file,
            "common_tg_vocab": common_tg_vocab_file,
            "window_map": sample_window_map_file,
            "genes_near_peaks": peak_to_tss_dist_path,
            "metacell_names": metacell_name_file,
            "tg_scaler": sample_scaler_file,
            "motif_mask": motif_mask_file,
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    logging.info("\nPreprocessing complete. Wrote per-sample/per-chrom data for MultiomicTransformerDataset.")
