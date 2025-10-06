import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
import pybedtools
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler

from config.settings import *
from grn_inference import utils
from multiomic_transformer.data.moods_scan import run_moods_scan
from multiomic_transformer.utils.standardize import standardize_name
from multiomic_transformer.utils.files import atomic_json_dump
from multiomic_transformer.utils.peaks import find_genes_near_peaks

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Reads the number of CPUs from the slurm job, else defaults to 1
NUM_CPUS: int = int(os.getenv("NUM_CPUS", "1"))
logging.info(f"Using {NUM_CPUS} CPUs")

# TF and TG vocab files
common_tf_vocab_file: Path =  COMMON_DATA / f"tf_vocab.json"
common_tg_vocab_file: Path =  COMMON_DATA / f"tg_vocab.json"

# Sample-specific cache files
tf_tensor_path: Path =        SAMPLE_DATA_CACHE_DIR / "tf_tensor_all.pt"
metacell_name_file: Path =    SAMPLE_DATA_CACHE_DIR / "metacell_names.json"
sample_tf_name_file: Path =   SAMPLE_DATA_CACHE_DIR / "tf_names.json"
tf_id_file: Path =            SAMPLE_DATA_CACHE_DIR / "tf_ids.pt"

# Chromosome-specific cache files
atac_tensor_path: Path =      SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"atac_window_tensor_all_{CHROM_ID}.pt"
tg_tensor_path: Path =        SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_tensor_all_{CHROM_ID}.pt"
sample_tg_name_file: Path =   SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_names_{CHROM_ID}.json"
genome_window_file: Path =    SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"{CHROM_ID}_windows_{WINDOW_SIZE // 1000}kb.bed"
sample_window_map_file: Path= SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"window_map_{CHROM_ID}.json"
sample_scaler_file: Path =    SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_scaler_{CHROM_ID}.save"
peak_to_tss_dist_path: Path = SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"genes_near_peaks_{CHROM_ID}.parquet"
dist_bias_file: Path =        SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"dist_bias_{CHROM_ID}.pt"
tg_id_file: Path =            SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_ids_{CHROM_ID}.pt"
manifest_file: Path =         SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"manifest_{CHROM_ID}.json"
motif_mask_file: Path =       SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"motif_mask_{CHROM_ID}.pt"
moods_sites_file: Path =      SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"moods_sites_{CHROM_ID}.tsv"

os.makedirs(COMMON_DATA, exist_ok=True)
os.makedirs(SAMPLE_DATA_CACHE_DIR, exist_ok=True)
os.makedirs(SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR, exist_ok=True)


def build_global_tg_vocab(gene_tss_file, vocab_file):
    """
    Build or update a global TG vocab from all genes in the genome.
    """
    # Load existing vocab if present
    if os.path.isfile(vocab_file):
        with open(vocab_file) as f:
            vocab = json.load(f)
    else:
        vocab = {}

    # Load all gene names genome-wide
    gene_tss_bed = pybedtools.BedTool(gene_tss_file)
    gene_tss_df = (
        gene_tss_bed
        .to_dataframe()
        .sort_values(by="start", ascending=True)
        )
    all_genes = [standardize_name(n) for n in gene_tss_df["name"].unique()]

    updated = False
    for name in sorted(set(all_genes)):
        if name not in vocab:
            vocab[name] = len(vocab)
            updated = True

    if updated:
        atomic_json_dump(vocab, vocab_file)
        logging.info(f"Updated TG vocab: {len(vocab)} genes")

    return vocab



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

def create_single_cell_tensors(
    gene_tss_df: pd.DataFrame,
    sample_names: list[str],
    dataset_processed_data_dir: Path,
    tg_vocab: dict[str, int],
    tf_vocab: dict[str, int],
    chrom_id: str,
):

    # --- set chromosome-specific TG list ---
    chrom_tg_names = set(gene_tss_df["name"].unique())

    for sample_name in sample_names:
        outdir = SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / "single_cell" / sample_name
        outdir.mkdir(parents=True, exist_ok=True)
        
        sample_processed_data_dir = dataset_processed_data_dir / sample_name

        tg_sc_file = sample_processed_data_dir / "TG_singlecell.tsv"
        re_sc_file = sample_processed_data_dir / "RE_singlecell.tsv"

        if not (tg_sc_file.exists() and re_sc_file.exists()):
            logging.warning(f"Skipping {sample_name}: missing TG/RE single-cell files")
            continue

        TG_sc = pd.read_csv(tg_sc_file, sep="\t", index_col=0)
        RE_sc = pd.read_csv(re_sc_file, sep="\t", index_col=0)

        # --- restrict TGs to chromosome + vocab ---
        tg_rows = [g for g in TG_sc.index if g in chrom_tg_names]
        TG_sc_chr = TG_sc.loc[tg_rows]

        tg_tensor_sc, tg_names_kept, tg_ids = align_to_vocab(
            TG_sc_chr.index.tolist(),
            tg_vocab,
            torch.tensor(TG_sc_chr.values, dtype=torch.float32),
            label="TG"
        )
        torch.save(tg_tensor_sc, outdir / f"{sample_name}_tg_tensor_singlecell_{chrom_id}.pt")
        torch.save(torch.tensor(tg_ids, dtype=torch.long), outdir / f"{sample_name}_tg_ids_singlecell_{chrom_id}.pt")
        atomic_json_dump(tg_names_kept, outdir / f"{sample_name}_tg_names_singlecell_{chrom_id}.json")

        # --- restrict ATAC peaks to chromosome ---
        re_rows = [p for p in RE_sc.index if p.startswith(f"{chrom_id}:")]
        RE_sc_chr = RE_sc.loc[re_rows]

        atac_tensor_sc = torch.tensor(RE_sc_chr.values, dtype=torch.float32)
        torch.save(atac_tensor_sc, outdir / f"{sample_name}_atac_tensor_singlecell_{chrom_id}.pt")

        # --- TF tensor (subset of TGs) ---
        tf_tensor_sc = None
        tf_rows = [g for g in TG_sc.index if g in tf_vocab]
        if tf_rows:
            TF_sc = TG_sc.loc[tf_rows]
            tf_tensor_sc, tf_names_kept, tf_ids = align_to_vocab(
                TF_sc.index.tolist(),
                tf_vocab,
                torch.tensor(TF_sc.values, dtype=torch.float32),
                label="TF"
            )
            torch.save(tf_tensor_sc, outdir / f"{sample_name}_tf_tensor_singlecell_{chrom_id}.pt")
            torch.save(torch.tensor(tf_ids, dtype=torch.long), outdir / f"{sample_name}_tf_ids_singlecell_{chrom_id}.pt")
            atomic_json_dump(tf_names_kept, outdir / f"{sample_name}_tf_names_singlecell_{chrom_id}.json")
        else:
            logging.warning(f"No TFs from global vocab found in sample {sample_name}")
            tf_tensor_sc, tf_ids = None, []

        logging.info(
            f"Saved single-cell tensors for {sample_name} | "
            f"TGs={tg_tensor_sc.shape}, "
            f"TFs={tf_tensor_sc.shape if tf_tensor_sc is not None else 'N/A'}, "
            f"RE={atac_tensor_sc.shape}"
        )


def aggregate_pseudobulk_datasets(sample_names: list[str], dataset_processed_data_dir: Path, chrom_id: str):
    
    # ----- Combine Pseudobulk Data into a Training Dataset -----
    TG_pseudobulk_global = []
    TG_pseudobulk_samples = []
    RE_pseudobulk_samples = []
    peaks_df_samples = []

    logging.info("\nLoading processed pseudobulk datasets:")
    logging.info(f"  - Sample names: {sample_names}")
    logging.info(f"  - Looking for processed samples in {dataset_processed_data_dir}")
    for sample_name in sample_names:
        sample_processed_data_dir = dataset_processed_data_dir / sample_name
        if not os.path.exists(sample_processed_data_dir):
            logging.warning(f"Skipping {sample_name}: directory not found")
            continue
        if sample_name in VALIDATION_DATASETS:
            logging.warning(f"Skipping {sample_name}: in VALIDATION_DATASETS list")
            continue
        else:
            logging.info(f"\nLoading Pseudobulk data for {sample_name}")
            TG_pseudobulk = pd.read_csv(os.path.join(sample_processed_data_dir, "TG_pseudobulk.tsv"), sep="\t", index_col=0)
            RE_pseudobulk = pd.read_csv(os.path.join(sample_processed_data_dir, "RE_pseudobulk.tsv"), sep="\t", index_col=0)

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
        
        # Aggregate the pseudobulk for all samples
        total_TG_pseudobulk_global = _agg_sum(TG_pseudobulk_global)
        total_TG_pseudobulk_chr    = _agg_sum(TG_pseudobulk_samples)
        total_RE_pseudobulk_chr    = _agg_sum(RE_pseudobulk_samples)
        total_peaks_df             = _agg_first(peaks_df_samples)
        
    return total_TG_pseudobulk_global, total_TG_pseudobulk_chr, total_RE_pseudobulk_chr, total_peaks_df


def create_or_load_genomic_windows(window_size, chrom_id, genome_window_file, chrom_sizes_file, force_recalculate=False):
    if not os.path.exists(genome_window_file) or force_recalculate:
        
        logging.info("\nCreating genomic windows")
        mm10_genome_windows = pybedtools.bedtool.BedTool().window_maker(g=chrom_sizes_file, w=window_size)
        mm10_windows = (
            mm10_genome_windows
            .filter(lambda x: x.chrom == chrom_id)  # TEMPORARY Restrict to one chromosome for testing
            .saveas(genome_window_file)
            .to_dataframe()
        )
        logging.info(f"  - Created {mm10_windows.shape[0]} windows")
    else:
        
        logging.info("\nLoading existing genomic windows")
        mm10_windows = pybedtools.BedTool(genome_window_file).to_dataframe()
        
    mm10_windows = mm10_windows.reset_index(drop=True)
    mm10_windows["win_idx"] = mm10_windows.index
    
    return mm10_windows


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
    
        genes_near_peaks = find_genes_near_peaks(peak_bed, tss_bed, tss_distance_cutoff=max_peak_distance)

        # Restrict to peaks within 1 Mb of a gene TSS
        genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= max_peak_distance]

        # Scale the TSS distance score by the exponential scaling factor
        genes_near_peaks = genes_near_peaks.copy()
        genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / distance_factor_scale)

        genes_near_peaks.to_parquet(os.path.join(output_dir, "genes_near_peaks.parquet"), compression="snappy", engine="pyarrow")
    else:
        genes_near_peaks = pd.read_parquet(os.path.join(output_dir, "genes_near_peaks.parquet"), engine="pyarrow")
    
    return genes_near_peaks


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


def build_motif_mask(tf_names, tg_names, motif_hits_df, genes_near_peaks):
    """
    Build motif mask [TG x TF] with max logodds per (TG, TF).
    """
    # map TFs and TGs to ids
    tf_index = {tf: i for i, tf in enumerate(tf_names)}
    tg_index = {tg: i for i, tg in enumerate(tg_names)}

    # restrict to known TFs
    motif_hits_df = motif_hits_df[motif_hits_df["TF"].isin(tf_index)]
    
    # merge motif hits with target genes (peak_id → TG)
    merged = motif_hits_df.merge(
        genes_near_peaks[["peak_id", "target_id"]],
        on="peak_id",
        how="inner"
    )

    # drop TGs not in tg_index
    merged = merged[merged["target_id"].isin(tg_index)]

    # map names → indices
    merged["tf_idx"] = merged["TF"].map(tf_index)
    merged["tg_idx"] = merged["target_id"].map(tg_index)

    # groupby max(logodds) per (TG, TF)
    agg = merged.groupby(["tg_idx", "tf_idx"])["logodds"].max().reset_index()

    # construct sparse COO
    mask = sp.coo_matrix(
        (agg["logodds"], (agg["tg_idx"], agg["tf_idx"])),
        shape=(len(tg_names), len(tf_names)),
        dtype=np.float32
    ).toarray()

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


def align_to_vocab(names, vocab, tensor_all, label="genes"):
    """
    Restrict to the subset of names that exist in the global vocab.
    Returns:
      aligned_tensor : [num_kept, C] (chromosome-specific subset)
      kept_names     : list[str] of kept names (order = aligned_tensor rows)
      kept_ids       : list[int] global vocab indices for kept names
    """
    kept_ids = []
    kept_names = []
    aligned_rows = []

    for i, n in enumerate(names):
        vid = vocab.get(n)
        if vid is not None:
            kept_ids.append(vid)
            kept_names.append(n)
            aligned_rows.append(tensor_all[i])

    if not kept_ids:
        raise ValueError(f"No {label} matched the global vocab.")

    aligned_tensor = torch.stack(aligned_rows, dim=0)  # [num_kept, num_cells]

    return aligned_tensor, kept_names, kept_ids


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


if __name__ == "__main__":
    logging.info(f"Preparing data for {DATASET_NAME} {CHROM_ID}")
    
    tg_vocab = build_global_tg_vocab(GENE_TSS_FILE, common_tg_vocab_file)
    
    # Create or load the gene TSS information for the chromosome
    gene_tss_df = make_chrom_gene_tss_df(
        gene_tss_file=GENE_TSS_FILE,
        chrom_id=CHROM_ID,
        genome_dir=GENOME_DIR
    )

    # Load the global TF vocab
    with open(common_tf_vocab_file, 'r') as f:
        tf_names: list = json.load(f)
    
    tf_names = [standardize_name(n) for n in tf_names]
        
    logging.info(f"\nLoaded {DATASET_NAME} TFs: {len(tf_names)} TFs")

    total_TG_pseudobulk_global, total_TG_pseudobulk_chr, total_RE_pseudobulk_chr, total_peaks_df = \
        aggregate_pseudobulk_datasets(SAMPLE_NAMES, SAMPLE_PROCESSED_DATA_DIR, CHROM_ID)
    
    tg_names = total_TG_pseudobulk_chr.index.tolist()

    # Genome-wide TF expression for all samples
    genome_wide_tf_expression = total_TG_pseudobulk_global.reindex(tf_names).fillna(0).values.astype("float32")
    metacell_names = total_TG_pseudobulk_global.columns.tolist()

    # Scale TG expression
    scaler = StandardScaler()
    TG_scaled = scaler.fit_transform(total_TG_pseudobulk_chr.values.astype("float32"))

    # Create genome windows
    mm10_windows = create_or_load_genomic_windows(
        window_size=WINDOW_SIZE,
        chrom_id=CHROM_ID,
        chrom_sizes_file=CHROM_SIZES_FILE,
        genome_window_file=genome_window_file,
        force_recalculate=FORCE_RECALCULATE
    )
    num_windows = mm10_windows.shape[0]

    # --- Calculate Peak-to-TG Distance Scores ---
    genes_near_peaks = calculate_peak_to_tg_distance_score(
        output_dir=SAMPLE_DATA_CACHE_DIR,
        mesc_atac_peak_loc_df=total_peaks_df,  # peak locations DataFrame
        gene_tss_df=gene_tss_df,
        max_peak_distance= MAX_PEAK_DISTANCE,
        distance_factor_scale= DISTANCE_SCALE_FACTOR,
        force_recalculate=FORCE_RECALCULATE
    )
    
    peaks_bed = os.path.join(SAMPLE_DATA_CACHE_DIR, "peak_tmp.bed")
    jaspar_pfm_paths = [os.path.join(JASPAR_PFM_DIR, f) for f in os.listdir(JASPAR_PFM_DIR) if f.endswith(".pfm")]
    
    if FORCE_RECALCULATE == True:
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
        output_dir=str(SAMPLE_DATA_CACHE_DIR),
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
    
    tg_names = [standardize_name(n) for n in total_TG_pseudobulk_chr.index.tolist()]
    
    if not os.path.exists(common_tf_vocab_file):
        vocab = {n: i for i, n in enumerate(tf_names)}
        with open(common_tf_vocab_file, "w") as f:
            json.dump(vocab, f)
        logging.info(f"Initialized TF vocab with {len(vocab)} entries")
    else:
        with open(common_tf_vocab_file) as f:
            vocab = json.load(f)

    # Load global vocab
    with open(common_tf_vocab_file) as f: tf_vocab = json.load(f)
    with open(common_tg_vocab_file) as f: tg_vocab = json.load(f)

    # Match TFs and TGs to global vocab
    tf_tensor_all, tf_names_kept, tf_ids = align_to_vocab(tf_names, tf_vocab, tf_tensor_all, label="TF")
    tg_tensor_all, tg_names_kept, tg_ids = align_to_vocab(tg_names, tg_vocab, tg_tensor_all, label="TG")
    
    torch.save(torch.tensor(tf_ids, dtype=torch.long), tf_id_file)
    torch.save(torch.tensor(tg_ids, dtype=torch.long), tg_id_file)

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
    
    create_single_cell_tensors(
        gene_tss_df=gene_tss_df, 
        sample_names=FINE_TUNING_DATASETS, 
        dataset_processed_data_dir=SAMPLE_PROCESSED_DATA_DIR, 
        tg_vocab=tg_vocab, 
        tf_vocab=tf_vocab, 
        chrom_id=CHROM_ID
    )
    
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


    # Write the distance bias and metacell names for the sample
    torch.save(dist_bias, dist_bias_file)
    logging.info(f"Saved distance bias tensor with shape {tuple(dist_bias.shape)}")

    atomic_json_dump(metacell_names, metacell_name_file)
    
    torch.save(torch.from_numpy(motif_mask), motif_mask_file)

    # Manifest of general sample info and file paths
    manifest = {
        "dataset_name": DATASET_NAME,
        "chrom": CHROM_ID,
        "num_windows": int(num_windows),
        "num_tfs": int(len(tf_names_kept)),
        "num_tgs": int(len(tg_names_kept)),
        "Distance tau": DISTANCE_SCALE_FACTOR,
        "Max peak-TG distance": MAX_PEAK_DISTANCE,
        "paths": {
            "tf_tensor_all": str(tf_tensor_path),
            "tg_tensor_all": str(tg_tensor_path),
            "atac_window_tensor_all": str(atac_tensor_path),
            "dist_bias": str(dist_bias_file),
            "tf_ids": str(tf_id_file),
            "tg_ids": str(tg_id_file),
            "tf_names": str(sample_tf_name_file),
            "tg_names": str(sample_tg_name_file),
            "common_tf_vocab": str(common_tf_vocab_file),
            "common_tg_vocab": str(common_tg_vocab_file),
            "window_map": str(sample_window_map_file),
            "genes_near_peaks": str(peak_to_tss_dist_path),
            "metacell_names": str(metacell_name_file),
            "tg_scaler": str(sample_scaler_file),
            "motif_mask": str(motif_mask_file),
        }
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    logging.info("\nPreprocessing complete. Wrote per-sample/per-chrom data for MultiomicTransformerDataset.")
