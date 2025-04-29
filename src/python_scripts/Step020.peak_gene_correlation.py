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

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.lib as pa_lib

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
        "--species",
        type=str,
        required=True,
        help="Species of the sample, either 'mouse', 'human', 'hg38', or 'mm10'"
    )
    parser.add_argument(
        "--num_cpu",
        type=str,
        required=True,
        help="Number of processors to run multithreading with"
    )
    parser.add_argument(
        "--peak_dist_limit",
        type=str,
        required=True,
        help="Number of base pairs away from a genes TSS to associate with peaks"
    )
    
    args: argparse.Namespace = parser.parse_args()

    return args

# ------------------------- DATA LOADING & PREPARATION ------------------------- #
def extract_atac_peaks(atac_df, tmp_dir):
    peak_pos = atac_df["peak_id"].compute().tolist()

    peak_df = pd.DataFrame()
    peak_df["chr"] = [pos.split(":")[0].replace("chr", "") for pos in peak_pos]
    peak_df["start"] = [int(pos.split(":")[1].split("-")[0]) for pos in peak_pos]
    peak_df["end"] = [int(pos.split(":")[1].split("-")[1]) for pos in peak_pos]
    peak_df["peak_id"] = peak_pos

    peak_df["chr"] = peak_df["chr"].astype(str)
    peak_df["start"] = peak_df["start"].astype(int)
    peak_df["end"] = peak_df["end"].astype(int)
    peak_df["peak_id"] = peak_df["peak_id"].astype(str)
    
    # Write the peak DataFrame to a file
    peak_df.to_parquet(f"{tmp_dir}/peak_df.parquet", engine="pyarrow", index=False, compression="snappy")

def load_ensembl_organism_tss(organism, tmp_dir):
    # Connect to the Ensembl BioMart server
    server = Server(host='http://www.ensembl.org')

    gene_ensembl_name = f'{organism}_gene_ensembl'
    
    # Select the Ensembl Mart and the human dataset
    mart = server['ENSEMBL_MART_ENSEMBL']
    dataset: pd.DataFrame = mart[gene_ensembl_name]

    # Query for attributes: Ensembl gene ID, gene name, strand, and transcription start site (TSS)
    ensembl_df = dataset.query(attributes=[
        'external_gene_name', 
        'strand', 
        'chromosome_name',
        'transcription_start_site'
    ])

    ensembl_df.rename(columns={
        "Chromosome/scaffold name": "chr",
        "Transcription start site (TSS)": "tss",
        "Gene name": "gene_id"
    }, inplace=True)
    
    # Make sure TSS is integer (some might be floats).
    ensembl_df["tss"] = ensembl_df["tss"].astype(int)

    # In a BED file, we’ll store TSS as [start, end) = [tss, tss+1)
    ensembl_df["start"] = ensembl_df["tss"].astype(int)
    ensembl_df["end"] = ensembl_df["tss"].astype(int) + 1

    # Re-order columns for clarity: [chr, start, end, gene]
    ensembl_df = ensembl_df[["chr", "start", "end", "gene_id"]]
    
    ensembl_df["chr"] = ensembl_df["chr"].astype(str)
    ensembl_df["gene_id"] = ensembl_df["gene_id"].astype(str)
    
    # Write the peak DataFrame to a file
    ensembl_df.to_parquet(f"{tmp_dir}/ensembl.parquet", engine="pyarrow", index=False, compression="snappy")

def find_genes_near_peaks(peak_bed, tss_bed, rna_df, peak_dist_limit, tmp_dir):
    """
    Identify genes whose transcription start sites (TSS) are near scATAC-seq peaks.
    
    This function:
        1. Uses BedTools to find peaks that are within peak_dist_limit bp of each gene's TSS.
        2. Converts the BedTool result to a pandas DataFrame.
        3. Computes the absolute distance between the peak end and gene start (as a proxy for TSS distance).
        4. Scales these distances using an exponential drop-off function (e^-dist/25000),
           the same method used in the LINGER cis-regulatory potential calculation.
        5. Deduplicates the data to keep the minimum (i.e., best) peak-to-gene connection.
        6. Only keeps genes that are present in the RNA-seq dataset.
        
    Parameters
    ----------
    peak_bed : BedTool
        A BedTool object representing scATAC-seq peaks.
    tss_bed : BedTool
        A BedTool object representing gene TSS locations.
    rna_df : pandas.DataFrame
        The RNA-seq dataset, which must have a "gene_id" column.
    peak_dist_limit : int
        The maximum distance (in bp) from a TSS to consider a peak as potentially regulatory.
        
    Returns
    -------
    peak_tss_subset_df : pandas.DataFrame
        A DataFrame containing columns "peak_id", "target_id", and the scaled TSS distance "TSS_dist"
        for peak–gene pairs.
    gene_list : set
        A set of unique gene IDs (target_id) present in the DataFrame.
    """
    # 3) Find peaks that are within peak_dist_limit bp of each gene's TSS using BedTools
    logging.info(f"Locating peaks that are within {peak_dist_limit} bp of each gene's TSS")
    peak_tss_overlap = peak_bed.window(tss_bed, w=peak_dist_limit)
    
    # Define the column types for conversion to DataFrame
    dtype_dict = {
        "peak_chr": str,
        "peak_start": int,
        "peak_end": int,
        "peak_id": str,
        "gene_chr": str,
        "gene_start": int,
        "gene_end": int,
        "gene_id": str
    }
    
    # Convert the BedTool result to a DataFrame for further processing.
    peak_tss_overlap_df = peak_tss_overlap.to_dataframe(
        names=[
            "peak_chr", "peak_start", "peak_end", "peak_id",
            "gene_chr", "gene_start", "gene_end", "gene_id"
        ],
        dtype=dtype_dict,
        low_memory=False  # ensures the entire file is read in one go
    ).rename(columns={"gene_id": "target_id"}).dropna()
    
    # Calculate the absolute distance between the peak's end and gene's start.
    # This serves as a proxy for the TSS distance for the peak-to-gene pair.
    peak_tss_overlap_df["TSS_dist"] = np.abs(peak_tss_overlap_df["peak_end"] - peak_tss_overlap_df["gene_start"])
    
    # Sort by the TSS distance (lower values imply closer proximity and therefore stronger association)
    # and drop duplicates keeping only the best association for each peak-target pair.
    peak_tss_overlap_df = peak_tss_overlap_df.sort_values("TSS_dist")
    peak_tss_overlap_df = peak_tss_overlap_df.drop_duplicates(subset=["peak_id", "target_id"], keep="first")
    
    # Scale the TSS distance using an exponential drop-off function
    # e^-dist/25000, same scaling function used in LINGER Cis-regulatory potential calculation
    # https://github.com/Durenlab/LINGER
    peak_tss_overlap_df["TSS_dist_score"] = peak_tss_overlap_df["TSS_dist"].apply(lambda x: math.exp(-(x / 250000)))
    
    # Keep only the necessary columns.
    peak_tss_subset_df = peak_tss_overlap_df[["peak_id", "target_id", "TSS_dist_score"]]
    
    # Filter out any genes not found in the RNA-seq dataset.
    peak_tss_subset_df = peak_tss_subset_df[peak_tss_subset_df["target_id"].isin(rna_df["gene_id"])]
        
    peak_tss_subset_df.to_parquet(f"{tmp_dir}/peak_to_gene_map.parquet", index=False)

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

def select_top_dispersion_auto(df, target_n_features=5000, dispersion_quantile=0.75):
    """
    Automatically selects top most dispersed features by:
    1. Filtering features with dispersion above a dataset-driven threshold.
    2. Selecting the top-N features with the highest dispersion.

    Parameters
    ----------
    df : dask.DataFrame
        Rows are features (peaks or genes), columns are cells.
    target_n_features : int
        Number of features you ideally want to select (default: 5000).
    dispersion_quantile : float
        Quantile of dispersion to define filtering threshold (default: 0.75 = top 25%).

    Returns
    -------
    dask.DataFrame
        Subset of original DataFrame with selected features.
    """

    def compute_dispersion(part):
        numeric_part = part.select_dtypes(include=[np.number])
        mean = numeric_part.mean(axis=1, skipna=True)
        var = numeric_part.var(axis=1, skipna=True)
        disp = var / (mean + 1e-8)
        return disp

    # Step 1: Preserve original row identity as a column
    df_reset = df.reset_index(drop=False)  # index becomes a column, preserves row mapping
    df_reset = df_reset.persist()

    logging.info("Computing feature dispersion across cells...")    
    dispersions = df_reset.map_partitions(compute_dispersion, meta=('x', 'f8')).compute()

    total_features = len(dispersions)

    # Step 2: Compute quantile-based dispersion threshold
    min_disp = dispersions.quantile(dispersion_quantile)

    # Step 3: Keep only features above the threshold
    filtered = dispersions[dispersions >= min_disp]
    n_above_thresh = len(filtered)

    # Step 4: Select top-N features if needed
    if n_above_thresh > target_n_features:
        top_features_idx = filtered.nlargest(target_n_features).index
        n_final = target_n_features
    else:
        top_features_idx = filtered.index
        n_final = n_above_thresh

    # Log summary
    logging.info(f"Initial features: {total_features:,}")
    logging.info(f"Features above dispersion threshold (quantile={dispersion_quantile}): {n_above_thresh:,}")
    logging.info(f"Final selected features (after top-N limit): {n_final:,}")
    logging.info(f"Dispersion threshold (min_disp) used: {min_disp:.4f}")

    # Step 5: Map integer positions to index values and subset
    index_values = df_reset.index.to_numpy()[top_features_idx]
    df_filtered = df_reset.loc[index_values].drop(columns=["index"], errors="ignore")

    return df_filtered


def auto_tune_parameters(num_cpu: int = None, total_memory_gb: int = None) -> dict:
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
    alpha=0.01,
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
    logging.info("Calculating significant peak-to-gene correlations")

    # Make sure dataframes are pandas
    if isinstance(atac_df, dd.DataFrame):
        atac_df = atac_df.compute()
    if isinstance(gene_df, dd.DataFrame):
        gene_df = gene_df.compute()

    num_peaks, n = atac_df.shape
    num_genes = gene_df.shape[0]
    num_pairs = num_peaks * num_genes

    logging.info(f"    Number of peaks: {num_peaks}")
    logging.info(f"    Number of genes: {num_genes}")
    logging.info(f"    Total peak-gene pairs: {num_pairs:,}")

    # Convert to sparse
    X = sp.csr_matrix(atac_df.values.astype(float))
    Y = sp.csr_matrix(gene_df.values.astype(float))

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
        logging.info("    Using in-memory calculation (faster)")

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
        return df_corr

    else:
        # --- Version 1: Streaming-to-Parquet Mode ---
        logging.info("    Too large for memory. Streaming results to disk.")

        if output_file is None:
            raise ValueError("output_file must be provided for streaming mode.")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

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

def load_atac_dataset(atac_data_file: str) -> dd.DataFrame:
    atac_df: dd.DataFrame = dd.read_parquet(atac_data_file)
    return atac_df

def load_rna_dataset(rna_data_file: str) -> dd.DataFrame:
    rna_df: dd.DataFrame = dd.read_parquet(rna_data_file)
    return rna_df

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()
    
    if args.species == "hg38":
        ORGANISM = "hsapiens"
    elif args.species == "mm10":
        ORGANISM = "mmusculus"
    else:
        raise Exception(f'Organism not found, you entered {args.species} (must be one of: "hg38", "mm10")')
    
    ATAC_DATA_FILE = args.atac_data_file
    RNA_DATA_FILE =  args.rna_data_file
    OUTPUT_DIR = args.output_dir
    TMP_DIR = f"{OUTPUT_DIR}/tmp"
    NUM_CPU = int(args.num_cpu)
    PEAK_DIST_LIMIT=args.peak_dist_limit
    
    # Auto-tune parameters based on system resources
    tuned_params = auto_tune_parameters(num_cpu=int(args.num_cpu))
    chunk_size = tuned_params["chunk_size"]
    batch_size = tuned_params["batch_size"]
    num_workers = tuned_params["num_workers"]

    logging.info("Auto-tuned parameters:")
    logging.info(f"  - chunk_size = {chunk_size:,}")
    logging.info(f"  - batch_size = {batch_size:,}")
    logging.info(f"  - num_workers = {num_workers}")
    
    # Make the tmp dir if it does not exist
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    
    logging.info("Loading the scRNA-seq dataset.")
    rna_df: dd.DataFrame = load_rna_dataset(RNA_DATA_FILE,)

    logging.info("Loading and parsing the ATAC-seq peaks")
    atac_df: dd.DataFrame = load_atac_dataset(ATAC_DATA_FILE)

    if not os.path.exists(f"{TMP_DIR}/peak_df.parquet"):
        logging.info(f"Extracting peak information and saving as a bed file")
        extract_atac_peaks(atac_df, TMP_DIR)
    else:
        logging.info("ATAC-seq BED file exists, loading...")

    if not os.path.exists(f"{TMP_DIR}/ensembl.parquet"):
        logging.info(f"Extracting TSS locations for {ORGANISM} from Ensembl and saving as a bed file")
        load_ensembl_organism_tss(ORGANISM, TMP_DIR)
    else:
        logging.info("Ensembl gene TSS BED file exists, loading...")

    pybedtools.set_tempdir(TMP_DIR)
    
    # Load the peak and gene TSS BED files
    peak_df = dd.read_parquet(f"{TMP_DIR}/peak_df.parquet").compute()
    tss_df = dd.read_parquet(f"{TMP_DIR}/ensembl.parquet").compute()

    peak_bed = pybedtools.BedTool.from_dataframe(peak_df)
    tss_bed = pybedtools.BedTool.from_dataframe(tss_df)
    
    # ============ FINDING PEAKS NEAR GENES ============
    # Dataframe with "peak_id", "target_id" and "TSS_dist"
    if not os.path.exists(os.path.join(TMP_DIR, "peak_to_gene_map.parquet")):
        find_genes_near_peaks(
            peak_bed,
            tss_bed,
            rna_df,
            PEAK_DIST_LIMIT,
            TMP_DIR
        )
    else:
        logging.info('TSS distance file exists, loading...')
    
    peak_gene_df = dd.read_parquet(f"{TMP_DIR}/peak_to_gene_map.parquet")

    # ============ PEAK TO GENE CORRELATION CALCULATION ============
    PARQ = f"{TMP_DIR}/sig_peak_to_gene_corr.parquet"

    def prepare_inputs_for_correlation():
        logging.info("Subsetting ATAC and RNA matrices to relevant peaks/genes")
        atac_sub = atac_df.merge(peak_gene_df[["peak_id"]].drop_duplicates(), on="peak_id", how="inner")
        atac_sub = atac_sub.drop(columns="peak_id", errors="ignore")
        
        rna_sub = rna_df.merge(peak_gene_df[["target_id"]].drop_duplicates(), 
                    left_on="gene_id", right_on="target_id", how="inner")
        rna_sub = rna_sub.drop(columns=["target_id", "gene_id"], errors="ignore")

        logging.info("Filtering out genes and peaks with low variance")

        rna_sub = select_top_dispersion_auto(rna_df, target_n_features=5000, dispersion_quantile=0.75)
        atac_sub = select_top_dispersion_auto(rna_df, target_n_features=50000, dispersion_quantile=0.75)
        logging.info(f"Selected {rna_sub.shape[0]} RNA features with highest dispersion")
        logging.info(f"Selected {atac_sub.shape[0]} ATAC features with highest dispersion")
        
        return atac_sub, rna_sub

    def run_correlation_and_load():
        atac_sub, rna_sub = prepare_inputs_for_correlation()
        logging.info("Calculating significant ATAC-seq peak-to-gene correlations")
        written = calculate_significant_peak_to_gene_correlations(
            atac_sub,
            rna_sub,
            output_file=PARQ,
            alpha=0.05,
            chunk_size=chunk_size,
            num_cpu=num_workers,
            batch_size=batch_size
        )
        return dd.read_parquet(written)

    if not os.path.exists(PARQ):
        sig_peak_to_gene_corr = run_correlation_and_load()

    else:
        try:
            logging.info("Loading existing Parquet")
            sig_peak_to_gene_corr = dd.read_parquet(PARQ)
        except pa_lib.ArrowInvalid:
            logging.warning(f"Corrupt Parquet file detected at {PARQ}. Deleting and recalculating...")
            os.remove(PARQ)
            sig_peak_to_gene_corr = run_correlation_and_load()

    # Compute the 90th-percentile cutoff
    quantile_threshold = 0.70
    logging.info(f"Computing the {quantile_threshold*100:.0f}th percentile of correlation…")
    cutoff = sig_peak_to_gene_corr["correlation"].quantile(quantile_threshold).compute()
    logging.info(f"Cutoff = {cutoff:.4f}")

    # Filter to only peak to TG correlations ≥ cutoff
    filtered = sig_peak_to_gene_corr[sig_peak_to_gene_corr["correlation"] >= cutoff]
    
    # Rename so both sides use "target_id"
    filtered = filtered.rename(columns={"gene_id": "target_id"})

    # Convert the TSS distance score DataFrame to Dask
    dd_pg: dd.DataFrame = dd.from_pandas(peak_gene_df, npartitions=1)

    # Merge the TSS distance score and correlation DataFrames
    joined: dd.DataFrame = filtered.merge(
        dd_pg,
        how="inner",
        on=["peak_id", "target_id"]
    )[["peak_id", "target_id", "correlation", "TSS_dist_score"]]

    # Write the joined DataFrame to a CSV file
    out_df: pd.DataFrame = joined.compute()
    out_path = f"{OUTPUT_DIR}/peak_to_gene_correlation.parquet"
    out_df.to_parquet(out_path, engine="pyarrow", index=False, compression="snappy")

    logging.info(f"Wrote top-10% correlations + TSS scores to {out_path}")
    logging.info("\n-----------------------------------------\n")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()