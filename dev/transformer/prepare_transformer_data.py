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
from typing import List, Dict, Tuple, Sequence, Iterable, Optional

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)

logging.basicConfig(level=logging.INFO, format="%(message)")

WINDOW_SIZE = 25000
SAMPLE_NAME = "mESC_holdout"
CHROM_ID = "chr1"

# sample_name_list = ["E7.5_rep1", "E7.75_rep1", "E8.0_rep2", "E8.5_rep2",
#                     "E8.75_rep2", "E7.5_rep2", "E8.0_rep1", "E8.5_rep1"]
# holdout = ["E8.75_rep1"]

sample_name_list = ["E8.75_rep1"]
holdout = []

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

def calculate_peak_to_tg_distance_score(output_dir, mesc_atac_peak_loc_df, gene_tss_df, force_recalculate=False):
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
    

        genes_near_peaks = utils.find_genes_near_peaks(peak_bed, tss_bed)

        # Restrict to peaks within 1 Mb of a gene TSS
        genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= 1e6]

        # Scale the TSS distance score by the exponential scaling factor
        genes_near_peaks = genes_near_peaks.copy()
        genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / 250000)

        genes_near_peaks.to_parquet(os.path.join(output_dir, "genes_near_peaks.parquet"), compression="snappy", engine="pyarrow")
    else:
        genes_near_peaks = pd.read_parquet(os.path.join(output_dir, "genes_near_peaks.parquet"), engine="pyarrow")
    
    return genes_near_peaks

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

    Returns:
        dist_bias: torch.Tensor of shape [num_windows, len(tg_names_kept)],
                   where each entry is the max TSS distance score for (window, TG).
    """
    tg_names_kept = list(tg_names_kept)
    num_tg_kept = len(tg_names_kept)

    dist_bias = torch.zeros((num_windows, num_tg_kept), dtype=dtype, device=device)
    tg_index_map = {tg: i for i, tg in enumerate(tg_names_kept)}

    # Iterate rows; update with max score per (win_idx, tg_idx)
    for _, row in genes_near_peaks.iterrows():
        peak_id = row["peak_id"]
        tg      = row["target_id"]
        score   = float(row["TSS_dist_score"])

        win_idx = window_map.get(peak_id, None)
        tg_idx  = tg_index_map.get(tg, None)
        if win_idx is None or tg_idx is None:
            continue

        # keep the maximum score if multiple peaks map to the same (window, TG)
        current = dist_bias[win_idx, tg_idx].item()
        if score > current:
            dist_bias[win_idx, tg_idx] = score

    return dist_bias

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
    TG_scaled = scaler.fit_transform(total_TG_pseudobulk_chr.values.astype("float32"))

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
        force_recalculate=False
    )

    # Build peak -> window mapping
    window_map = make_peak_to_window_map(total_peaks_df, mm10_windows)
    logging.info(f"Mapped {len(window_map)} peaks to windows")

    # Save Precomputed Tensors 
    tf_tensor_all, tg_tensor_all, atac_window_tensor_all = precompute_input_tensors(
        output_dir=SAMPLE_DATA_DIR,
        genome_wide_tf_expression=genome_wide_tf_expression,
        TG_scaled=TG_scaled,
        total_RE_pseudobulk_chr=total_RE_pseudobulk_chr,
        window_map=window_map,
        windows=mm10_windows,
    )

    # ----- Load common TF and TG vocab -----
    # Create a common TG vocabulary for the chromosome using the gene TSS
    if not os.path.isfile(common_tg_vocab_file):
        all_tg = sorted(gene_tss_df["name"].unique().tolist())
        tg_vocab = {name: i for i, name in enumerate(all_tg)}
        atomic_json_dump(tg_vocab, common_tg_vocab_file)

    # Create a common global TF vocabulary using the TF names from Homer
    if not os.path.isfile(common_tf_vocab_file):
        tf_vocab = {name: i for i, name in enumerate(sorted(set(tf_names)))}
        atomic_json_dump(tf_vocab, common_tf_vocab_file)

    with open(common_tf_vocab_file) as f:
        tf_vocab = json.load(f)
        
    with open(common_tg_vocab_file) as f:
        tg_vocab = json.load(f)

    tf_tensor_all, tf_names_kept, tf_ids = match_gene_to_vocab(tf_names, tf_vocab, tf_tensor_all, label="TF")
    tg_tensor_all, tg_names_kept, tg_ids = match_gene_to_vocab(tg_names, tg_vocab, tg_tensor_all, label="TG")
    
    if not tf_ids: raise ValueError("No TFs matched the common vocab.")
    if not tg_ids: raise ValueError("No TGs matched the common vocab.")
        
    # Build distance bias [num_windows x num_tg_kept] aligned to kept TGs
    dist_bias = build_distance_bias(
        genes_near_peaks=genes_near_peaks,
        window_map=window_map,
        tg_names_kept=tg_names_kept,
        num_windows=num_windows,
        dtype=torch.float32,
    )

    # ----- Writing Output Files -----
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

    # Manifest of general sample info and file paths
    manifest = {
        "sample": SAMPLE_NAME,
        "chrom": CHROM_ID,
        "num_windows": int(num_windows),
        "num_tfs": int(len(tf_names_kept)),
        "num_tgs": int(len(tg_names_kept)),
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
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    logging.info("\nPreprocessing complete. Wrote per-sample/per-chrom data for MultiomicTransformerDataset.")
