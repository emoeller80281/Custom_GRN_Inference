import pandas as pd
import pybedtools
import numpy as np
from pybiomart import Server
import scipy.sparse as sp
import scipy.stats as stats
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
from tqdm import tqdm
import psutil
import shutil

import math
import os
import sys
import argparse
import logging
from typing import Tuple, Union

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.lib as pa_lib
import matplotlib.pyplot as plt
import seaborn as sns

from grn_inference.normalization import (
    minmax_normalize_dask,
    clip_and_normalize_log1p_dask
)
from grn_inference.plotting import plot_feature_score_histogram

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
    argparse.Namespace: Parsed arguments containing paths for input and output files.
    """

    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument(
        "--atac_data_file",
        type=str,
        required=True,
        help="Path to the scATACseq data file"
    )
    parser.add_argument(
        "--rna_data_file",
        type=str,
        required=True,
        help="Path to the scRNAseq data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
    )
    parser.add_argument(
        "--num_cpu",
        type=str,
        required=True,
        help="Number of processors to run multithreading with"
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        required=True,
        help="Path to the sample's figure directory"
    )
    
    args: argparse.Namespace = parser.parse_args()

    return args

def row_normalize_sparse(X):
    """
    Row-normalize a sparse matrix X (CSR format) so that each row
    has zero mean and unit variance. Only nonzero entries are stored.
    """
    # Compute row means (this works even with sparse matrices)
    means = np.array(X.mean(axis=1)).flatten()
    
    # Compute row means of squared values:
    X2 = X.multiply(X)
    means2 = np.array(X2.mean(axis=1)).flatten()
    
    # Standard deviation: sqrt(E[x^2] - mean^2)
    stds = np.sqrt(np.maximum(0, means2 - means**2))
    
    # Convert to LIL format for efficient row-wise operations.
    X_norm = X.tolil(copy=True)
    for i in range(X.shape[0]):
        if stds[i] > 0:
            # For each nonzero in row i, subtract the row mean and divide by std.
            # X_norm.rows[i] gives the column indices and X_norm.data[i] the values.
            X_norm.data[i] = [(val - means[i]) / stds[i] for val in X_norm.data[i]]
        else:
            # If the row has zero variance, leave it as-is (or set to zero)
            X_norm.data[i] = [0 for _ in X_norm.data[i]]
    return X_norm.tocsr()

def select_top_dispersion_auto(df, fig_dir, target_n_features=5000, dispersion_quantile=0.75):
    """
    Automatically selects the top most dispersed features by:
    1. Filtering features with dispersion above a dataset-driven threshold.
    2. Selecting the top-N features with the highest dispersion.
    """
    def compute_dispersion(part):
        numeric_part = part.select_dtypes(include=[np.number])
        mean = numeric_part.mean(axis=1, skipna=True)
        var = numeric_part.var(axis=1, skipna=True)
        disp = var / (mean + 1e-8)
        return disp

    logging.info("    Computing feature dispersion across cells...")
    dispersions = df.map_partitions(compute_dispersion, meta=('x', 'f8')).compute()
    
    os.makedirs(fig_dir, exist_ok=True)
    thresh = dispersions.quantile(dispersion_quantile)

    plt.figure(figsize=(8, 4))
    sns.histplot(dispersions, bins=100, kde=True, stat="count", edgecolor=None)
    plt.axvline(thresh, color="red", linestyle="--", label=f"{dispersion_quantile:.2f} quantile = {thresh:.2f}")
    plt.title("Feature Dispersion Distribution")
    plt.xlabel("Dispersion")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "dispersion_histogram.png"), dpi=200)
    plt.close()

    total_features = len(dispersions)
    min_disp = dispersions.quantile(dispersion_quantile)
    filtered = dispersions[dispersions >= min_disp]
    n_above_thresh = len(filtered)

    if n_above_thresh > target_n_features:
        top_features = filtered.nlargest(target_n_features)
    else:
        top_features = filtered

    top_feature_names = top_features.index.to_list()

    logging.info(f"\t  - Initial features: {total_features:,}")
    logging.info(f"\t  - Features above dispersion threshold (quantile={dispersion_quantile}): {n_above_thresh:,}")
    logging.info(f"\t  - Final selected features (after top-N limit): {len(top_feature_names):,}")
    logging.info(f"\t  - Dispersion threshold (min_disp) used: {min_disp:.4f}")

    # Explicitly create a new column from the index to preserve feature IDs
    df = df.assign(feature_id=df.index)

    # Filter rows based on selected feature IDs
    df = df[df["feature_id"].isin(top_feature_names)]

    # Set index back to feature_id (for clarity)
    df = df.set_index("feature_id")         
    
    return df

def auto_tune_parameters(
    num_cpu: Union[int,None] = None, 
    total_memory_gb: Union[int,None] = None
    ) -> dict:
    """
    Suggests optimal chunk_size, batch_size, and number of Dask workers based on system resources.

    Parameters
    ----------
    num_cpu : int, optional
        Number of CPU cores available. If None, detects automatically.
    total_memory_gb : int, optional
        Total available RAM in GB. If None, detects automatically.

    Returns
    -------
    dict
        Dictionary with 'chunk_size', 'batch_size', and 'num_workers'.
    """
    # Auto-detect if not provided
    if num_cpu is None:
        num_cpu = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)

    if total_memory_gb is None:
        total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)  # bytes → GB

    # -------------------------------
    # Tuning chunk size (gene side)
    # -------------------------------
    if total_memory_gb >= 512:
        chunk_size = 50000
    elif total_memory_gb >= 256:
        chunk_size = 30000
    elif total_memory_gb >= 128:
        chunk_size = 20000
    else:
        chunk_size = 10000

    # -------------------------------
    # Tuning batch size (peak side)
    # -------------------------------
    if num_cpu >= 64:
        batch_size = 2000
    elif num_cpu >= 32:
        batch_size = 1000
    elif num_cpu >= 16:
        batch_size = 500
    else:
        batch_size = 200

    # -------------------------------
    # Tuning Dask number of workers
    # -------------------------------
    num_workers = max(1, num_cpu - 2)

    return {
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "num_workers": num_workers
    }

def calculate_significant_peak_to_gene_correlations(
    atac_df,
    gene_df,
    output_file=None,
    alpha=0.05,
    chunk_size=25_000,
    num_cpu=4,
    batch_size=1500,
    memory_threshold=1e8  # 100 million peak-gene pairs (default switch)
):
    """
    Calculates significant peak-to-gene correlations (p < alpha).

    If the number of peak × gene pairs is small enough (< memory_threshold),
    computes everything in memory and returns a DataFrame.
    If too large, streams to Parquet file on disk.

    Parameters
    ----------
    atac_df : pandas.DataFrame or dask.DataFrame
        ATAC matrix (peaks × cells)
    gene_df : pandas.DataFrame or dask.DataFrame
        RNA matrix (genes × cells)
    output_file : str, optional
        Output file path to save results (only required if streaming)
    alpha : float
        P-value threshold
    chunk_size : int
        Chunk size for dot products
    num_cpu : int
        Number of threads
    batch_size : int
        Batch size for Dask task execution
    memory_threshold : int
        Max number of peak-gene pairs to allow full-memory mode

    Returns
    -------
    pandas.DataFrame or str
        If using memory mode, returns a DataFrame.
        If using disk streaming mode, returns output_file path.
    """
    
    if output_file is None:
        raise ValueError("output_file must be provided.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Make sure dataframes are pandas
    logging.info('\tConverting ATAC dask dataframe to pandas')
    if isinstance(atac_df, dd.DataFrame):
        atac_df = atac_df.compute()
        
    logging.info('\tConverting RNA dask dataframe to pandas')
    if isinstance(gene_df, dd.DataFrame):
        gene_df = gene_df.compute()

    num_peaks, n = atac_df.shape
    num_genes = gene_df.shape[0]
    num_pairs = num_peaks * num_genes

    logging.info(f"\t- Number of peaks: {num_peaks}")
    logging.info(f"\t- Number of genes: {num_genes}")
    logging.info(f"\t- Total peak-gene pairs: {num_pairs:,}")

    # Convert to sparse
    X = sp.csr_matrix(atac_df.select_dtypes(include=[np.number]).values.astype(float))
    Y = sp.csr_matrix(gene_df.select_dtypes(include=[np.number]).values.astype(float))

    df_degrees = n - 2

    # Row-normalize
    X_norm = row_normalize_sparse(X)
    Y_norm = row_normalize_sparse(Y)

    def process_peak_gene_chunked(i, X_norm, Y_norm, df_degrees, n, alpha, chunk_size, min_r=0.0):
        results = []
        peak_row = X_norm.getrow(i)
        total_genes = Y_norm.shape[0]
        for start in range(0, total_genes, chunk_size):
            end = min(start + chunk_size, total_genes)
            Y_chunk = Y_norm[start:end]

            r_chunk = peak_row.dot(Y_chunk.T).toarray().ravel()
            r_chunk = r_chunk / (n - 1)

            # Optional rough correlation filtering
            if min_r > 0:
                keep = np.where(np.abs(r_chunk) >= min_r)[0]
                if len(keep) == 0:
                    continue
                r_chunk = r_chunk[keep]
                gene_indices = keep + start
            else:
                gene_indices = np.arange(start, end)

            with np.errstate(divide='ignore', invalid='ignore'):
                t_stat_chunk = r_chunk * np.sqrt(df_degrees / (1 - r_chunk**2))
            p_chunk = 2 * stats.t.sf(np.abs(t_stat_chunk), df=df_degrees)

            sig_indices = np.where(p_chunk < alpha)[0]
            for local_idx in sig_indices:
                results.append((i, gene_indices[local_idx], r_chunk[local_idx]))
                
        return results

    if num_pairs <= memory_threshold:
        # --- Version 2: Full Memory Mode ---
        logging.info("\t- Using in-memory calculation (faster)")

        tasks = [
            delayed(process_peak_gene_chunked)(i, X_norm, Y_norm, df_degrees, n, alpha, chunk_size)
            for i in range(num_peaks)
        ]

        with ProgressBar(dt=30, out=sys.stderr):
            results = compute(*tasks, scheduler="threads", num_workers=num_cpu)

        flat_results = [item for sublist in results for item in sublist]

        df_corr = pd.DataFrame(flat_results, columns=["peak_i", "gene_j", "correlation"])
        df_corr["peak_id"] = atac_df.index[df_corr["peak_i"]]
        df_corr["gene_id"] = gene_df.index[df_corr["gene_j"]]
        df_corr = df_corr[["peak_id", "gene_id", "correlation"]]

        logging.info("    Done!")
        
        df_corr.to_parquet(output_file, engine="pyarrow", compression="snappy")

        return df_corr

    else:
        # --- Version 1: Streaming-to-Parquet Mode ---
        logging.info("    Too large for memory. Streaming results to disk.")

        tasks = [
            delayed(process_peak_gene_chunked)(i, X_norm, Y_norm, df_degrees, n, alpha, chunk_size, min_r=0.2)
            for i in range(num_peaks)
        ]

        writer = None
        schema = None
        num_batches = (num_peaks + batch_size - 1) // batch_size

        for batch_start in tqdm(range(0, num_peaks, batch_size), total=num_batches, desc="Computing peak-gene batches"):
            batch = tasks[batch_start: batch_start + batch_size]
            results = compute(*batch, scheduler="threads", num_workers=num_cpu)

            for local_idx, sublist in enumerate(results):
                peak_idx = batch_start + local_idx
                if not sublist:
                    continue

                df_chunk = pd.DataFrame(sublist, columns=["peak_i", "gene_j", "correlation"])
                df_chunk["peak_id"] = atac_df.index[df_chunk["peak_i"]]
                df_chunk["gene_id"] = gene_df.index[df_chunk["gene_j"]]
                df_chunk = df_chunk[["peak_id", "gene_id", "correlation"]]

                table = pa.Table.from_pandas(df_chunk, preserve_index=False)

                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(output_file, schema)

                writer.write_table(table)
                del df_chunk, table

            del results

        if writer:
            writer.close()
            logging.info(f"    Wrote correlations to {output_file}")
        else:
            logging.warning("    No significant correlations found; no Parquet written.")

        return output_file

def prepare_inputs_for_correlation(
    atac_df: dd.DataFrame, 
    rna_df: dd.DataFrame,
    fig_dir: str
    ) -> Tuple[dd.DataFrame, dd.DataFrame]:
    
    # Use peak_id / target_id as index before dispersion selection
    atac_df = atac_df.set_index("peak_id")
    rna_df = rna_df.set_index("gene_id")

    # Compute dispersion and filter
    logging.info("Filtering out peaks with high dispersion")
    atac_df = select_top_dispersion_auto(atac_df, fig_dir, target_n_features=500_000, dispersion_quantile=0.75)
    
    logging.info("Filtering out genes with high dispersion")
    rna_df = select_top_dispersion_auto(rna_df, fig_dir, target_n_features=50_000, dispersion_quantile=0.75)

    logging.info(f"\nSelected {atac_df.shape[0].compute()} ATAC features with highest dispersion")
    logging.info(f"Selected {rna_df.shape[0].compute()} RNA features with highest dispersion")
    
    logging.info('\nSubsetting datasets to shared cell barcodes')
    # 1) find the common cell barcodes
    common_cells = [cell for cell in atac_df.columns if cell in rna_df.columns]        
    if len(common_cells) == 0:
        raise ValueError("No shared cells between ATAC and RNA!")
    else:
        logging.info(f'\t- Found {len(common_cells):,} common cells between ATAC and RNA datasets')
    
    # 2) subset and reorder both DataFrames to that same list
    atac_df = atac_df[common_cells]
    rna_df  = rna_df[common_cells]

    return atac_df, rna_df

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()
    
    ATAC_DATA_FILE = args.atac_data_file
    RNA_DATA_FILE =  args.rna_data_file
    OUTPUT_DIR = args.output_dir
    TMP_DIR = f"{OUTPUT_DIR}/tmp"
    NUM_CPU = int(args.num_cpu)
    FIG_DIR=os.path.join(args.fig_dir, "peak_gene_correlation_figures")
    
    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Auto-tune parameters based on system resources
    tuned_params = auto_tune_parameters(num_cpu=NUM_CPU)
    chunk_size = tuned_params["chunk_size"]
    batch_size = tuned_params["batch_size"]
    num_workers = tuned_params["num_workers"]

    logging.info("Auto-tuned parameters:")
    logging.info(f"  - chunk_size = {chunk_size:,}")
    logging.info(f"  - batch_size = {batch_size:,}")
    logging.info(f"  - num_workers = {num_workers}")
    
    logging.info("Loading the scRNA-seq dataset.")
    rna_df: pd.DataFrame = dd.read_parquet(RNA_DATA_FILE, engine="pyarrow")

    logging.info("Loading and parsing the ATAC-seq peaks")
    atac_df: pd.DataFrame = dd.read_parquet(ATAC_DATA_FILE, engine="pyarrow")
    
    # ============ PEAK TO GENE CORRELATION CALCULATION ============
    PARQ = os.path.join(TMP_DIR, "sig_peak_to_gene_corr.parquet")

    if not os.path.exists(PARQ):
        atac_df, rna_df = prepare_inputs_for_correlation(atac_df, rna_df, FIG_DIR)
                
        logging.info("\nCalculating significant ATAC-seq peak-to-gene correlations")
        result = calculate_significant_peak_to_gene_correlations(
            atac_df,
            rna_df,
            output_file=PARQ,
            alpha=0.5,
            chunk_size=chunk_size,
            num_cpu=num_workers,
            batch_size=batch_size
        )
        if isinstance(result, str):
            sig_peak_to_gene_corr = dd.read_parquet(result)  # streaming mode
        else:
            sig_peak_to_gene_corr = dd.from_pandas(result, npartitions=1)  # in-memory mode

    else:
        logging.info("Loading existing Parquet")
        sig_peak_to_gene_corr = dd.read_parquet(PARQ)
    
    logging.info("DataFrame after calculating sig peak to gene correlations")
    logging.info(f'Number of edges in sig_peak_to_gene_corr: {len(sig_peak_to_gene_corr)}')

    if len(sig_peak_to_gene_corr["correlation"]) > 1_000_000:
        quantile_threshold = 0.70
        logging.info("More than 1,000,000 significantly correlated edges")
        logging.info(f"Computing the {quantile_threshold*100:.0f}th percentile of correlation…")
        cutoff = sig_peak_to_gene_corr["correlation"].quantile(quantile_threshold).compute()
        logging.info(f"\tCutoff = {cutoff:.4f}")
        
    else:
        logging.info("Less than 1,000,000 significant edges, skipping quantile thresholding")
        cutoff = 0

    # Filter to only peak to TG correlations ≥ cutoff
    filtered = sig_peak_to_gene_corr[sig_peak_to_gene_corr["correlation"] >= cutoff]
    
    # logging.info("DataFrame after filtering 70th quantile")
    logging.info(f'\t- Number of edges: {len(filtered)}')
    
    # Rename so both sides use "target_id"
    filtered = filtered.rename(columns={"gene_id": "target_id"})
        
    logging.info(f'\nClipping to within the upper and lower 95th quantiles, log1p normalizing scores')
    normalized_ddf = clip_and_normalize_log1p_dask(
        ddf=filtered,
        score_cols=["correlation"],
        quantiles=(0, 1),
        apply_log1p=True,
    )
    logging.info(f'\t- Number of edges after clipping and normalizing: {len(normalized_ddf)}')
    
    logging.info(f'\nMinmax normalizing scores between 0-1')
    normalized_ddf = minmax_normalize_dask(
        ddf=normalized_ddf, 
        score_cols=["correlation"], 
        dtype=np.float32
    )
    logging.info(f'\t- Number of edges: {len(normalized_ddf)}')
    
    plot_feature_score_histogram(normalized_ddf, "correlation", OUTPUT_DIR)
        
    out_path = f"{OUTPUT_DIR}/peak_to_gene_correlation.parquet"
    normalized_ddf.to_parquet(out_path, engine="pyarrow", compression="snappy")
    
    logging.info(f"Wrote peak to gene correlation to {out_path}")
    logging.info("\n-----------------------------------------\n")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()