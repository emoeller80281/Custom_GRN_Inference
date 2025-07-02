import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import psutil

import os
import sys
import argparse
import logging
from typing import Union

from grn_inference.normalization import (
    minmax_normalize_pandas,
    clip_and_normalize_log1p_pandas
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

def calculate_peak_to_gene_correlations(
    atac_df,
    gene_df,
    output_dir,
    output_file=None,
    num_cpu=4,
):
    """
    Calculates peak-to-gene correlations (p < alpha).

    Parameters
    ----------
    atac_df : pandas.DataFrame
        ATAC matrix (peaks × cells)
    gene_df : pandas.DataFrame
        RNA matrix (genes × cells)
    output_dir : str
        Output directory for the sample
    output_file : str, optional
        Output file path to save results (only required if streaming)
    num_cpu : int
        Number of threads

    Returns
    -------
    pandas.DataFrame
    """
    
    if output_file is None:
        raise ValueError("output_file must be provided.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    peaks_near_genes: pd.DataFrame = pd.read_parquet(os.path.join(output_dir, "peaks_near_genes.parquet"), engine="pyarrow")
    
    # Map peak_id and gene_id to matrix indices
    peak_id_to_index = {pid: i for i, pid in enumerate(atac_df.index)}
    gene_id_to_index = {gid: j for j, gid in enumerate(gene_df.index)}

    # Filter only valid entries
    peaks_near_genes = peaks_near_genes[
        peaks_near_genes["peak_id"].isin(peak_id_to_index) &
        peaks_near_genes["target_id"].isin(gene_id_to_index)
    ].copy().drop_duplicates()

    # Replace IDs with matrix indices
    peaks_near_genes["peak_i"] = peaks_near_genes["peak_id"].map(peak_id_to_index)
    peaks_near_genes["gene_j"] = peaks_near_genes["target_id"].map(gene_id_to_index)

    pairs_to_compute = list(zip(peaks_near_genes["peak_i"], peaks_near_genes["gene_j"]))

    logging.info(f"\t- Number of peak-gene pairs to test: {len(pairs_to_compute):,}")
    
    assert len(pairs_to_compute) == len(peaks_near_genes)

    # Convert to sparse
    X = sp.csr_matrix(atac_df.select_dtypes(include=[np.number]).values.astype(float))
    Y = sp.csr_matrix(gene_df.select_dtypes(include=[np.number]).values.astype(float))

    n = atac_df.shape[1]
    df_degrees = n - 2

    # Row-normalize
    X_norm = row_normalize_sparse(X)
    Y_norm = row_normalize_sparse(Y)

    def compute_correlation_for_pair(pair, X_norm, Y_norm, df_degrees, n):
        i, j = pair
        r_val = X_norm.getrow(i).dot(Y_norm.getrow(j).T).toarray()[0][0]
        r_val = r_val / (n - 1)
        if np.isnan(r_val): 
            return None

        with np.errstate(divide='ignore', invalid='ignore'):
            t_stat = r_val * np.sqrt(df_degrees / (1 - r_val**2))
        p_val = 2 * stats.t.sf(np.abs(t_stat), df=df_degrees)
        return (i, j, r_val)

    tasks = [
        delayed(compute_correlation_for_pair)(pair, X_norm, Y_norm, df_degrees, n)
        for pair in pairs_to_compute
    ]

    with ProgressBar(dt=30, out=sys.stderr):
        results = compute(*tasks, scheduler="threads", num_workers=num_cpu)

    flat_results = [r for r in results if r is not None]

    df_corr = pd.DataFrame(flat_results, columns=["peak_i", "gene_j", "correlation"])
    df_corr["peak_id"] = [atac_df.index[i] for i in df_corr["peak_i"]]
    df_corr["gene_id"] = [gene_df.index[j] for j in df_corr["gene_j"]]
    df_corr = df_corr[["peak_id", "gene_id", "correlation"]]

    df_corr.to_parquet(output_file, engine="pyarrow", compression="snappy")
    return df_corr

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
    rna_df: pd.DataFrame = pd.read_parquet(RNA_DATA_FILE, engine="pyarrow")

    logging.info("Loading and parsing the ATAC-seq peaks")
    atac_df: pd.DataFrame = pd.read_parquet(ATAC_DATA_FILE, engine="pyarrow")
    
    # ============ PEAK TO GENE CORRELATION CALCULATION ============
    PARQ = os.path.join(TMP_DIR, "peak_to_gene_corr.parquet")

    if not os.path.exists(PARQ):
        # Assuming peak-gene filtering has already been applied:
        atac_df = atac_df.set_index("peak_id")
        rna_df = rna_df.set_index("gene_id")

        common_cells = [cell for cell in atac_df.columns if cell in rna_df.columns]
        if len(common_cells) == 0:
            raise ValueError("No shared cells between ATAC and RNA!")
        logging.info(f'\t- Found {len(common_cells):,} shared cells')
        
        atac_df = atac_df[common_cells]
        rna_df = rna_df[common_cells]
                
        logging.info("\nCalculating significant ATAC-seq peak-to-gene correlations")
        result = calculate_peak_to_gene_correlations(
            atac_df,
            rna_df,
            output_dir=OUTPUT_DIR,
            output_file=PARQ,
            num_cpu=num_workers,
        )
        if isinstance(result, str):
            peak_to_gene_corr = pd.read_parquet(result, engine="pyarrow")  # streaming mode
        else:
            peak_to_gene_corr = result  # in-memory mode

    else:
        logging.info("Loading existing Parquet")
        peak_to_gene_corr = pd.read_parquet(PARQ, engine="pyarrow")
    
    logging.info("DataFrame after calculating peak to gene correlations")
    logging.info(f'Number of edges in peak_to_gene_corr: {len(peak_to_gene_corr)}')

    if len(peak_to_gene_corr["correlation"]) > 1_000_000:
        quantile_threshold = 0.70
        logging.info("More than 1,000,000 correlated edges")
        logging.info(f"Computing the {quantile_threshold*100:.0f}th percentile of correlation…")
        cutoff = peak_to_gene_corr["correlation"].quantile(quantile_threshold)
        logging.info(f"\tCutoff = {cutoff:.4f}")
        
    else:
        logging.info("Less than 1,000,000 significant edges, skipping quantile thresholding")
        cutoff = 0

    # Filter to only peak to TG correlations ≥ cutoff
    filtered = peak_to_gene_corr[peak_to_gene_corr["correlation"] >= cutoff]
    
    # logging.info("DataFrame after filtering 70th quantile")
    logging.info(f'\t- Number of edges: {len(filtered)}')
    
    # Rename so both sides use "target_id"
    filtered = filtered.rename(columns={"gene_id": "target_id"})
        
    logging.info(f'\nClipping to within the upper and lower 95th quantiles, log1p normalizing scores')
    normalized_df = clip_and_normalize_log1p_pandas(
        df=filtered,
        score_cols=["correlation"],
        quantiles=(0, 1),
        apply_log1p=True,
    )
    logging.info(f'\t- Number of edges after clipping and normalizing: {len(normalized_df)}')
    
    logging.info(f'\nMinmax normalizing scores between 0-1')
    normalized_ddf = minmax_normalize_pandas(
        df=normalized_df, 
        score_cols=["correlation"], 
    )
    logging.info(f'\t- Number of edges: {len(normalized_df)}')
    
    plot_feature_score_histogram(normalized_df, "correlation", OUTPUT_DIR)
        
    out_path = f"{OUTPUT_DIR}/peak_to_gene_correlation.parquet"
    normalized_ddf.to_parquet(out_path, engine="pyarrow", compression="snappy")
    
    logging.info(f"Wrote peak to gene correlation to {out_path}")
    logging.info("\n-----------------------------------------\n")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()