import pandas as pd
import pybedtools
import numpy as np
from pybiomart import Server
import scipy.sparse as sp
import scipy.stats as stats
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import dask.dataframe as dd

import math
import os
import sys
import argparse
import logging

import pyarrow as pa
import pyarrow.parquet as pq

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
    peak_pos = atac_df["peak_id"].tolist()

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

def filter_low_variance_features(df, min_variance=0.5):
    def row_var(part):
        return part.var(axis=1, skipna=True)

    variances = df.map_partitions(row_var, meta=('x', 'f8'))
    return df[variances >= min_variance]

def calculate_significant_peak_to_gene_correlations(
    atac_df: pd.DataFrame,
    gene_df: pd.DataFrame,
    output_file: str,
    alpha: float = 0.01,
    chunk_size: int = 1000,
    num_cpu: int = 4,
    batch_size: int = 100
) -> str:
    """
    Streams all significant peak–gene correlations (p < alpha) into a single
    Parquet file at `output_file`.  Returns the path to that file.
    """
    # ensure parent directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 1) Dense→sparse
    X = sp.csr_matrix(atac_df.values.astype(float))
    Y = sp.csr_matrix(gene_df.values.astype(float))
    num_peaks, n = X.shape
    df_degrees = n - 2

    # 2) Row-normalize so Pearson r = dot/(n-1)
    X_norm = row_normalize_sparse(X)
    Y_norm = row_normalize_sparse(Y)

    # 3) Per-peak worker
    def process_peak_gene_chunked(i, Xn, Yn, df_deg, nn, α, csize):
        out = []
        prow = Xn.getrow(i)
        total_genes = Yn.shape[0]
        for start in range(0, total_genes, csize):
            end = min(start + csize, total_genes)
            chunk = Yn[start:end]
            r = prow.dot(chunk.T).toarray().ravel() / (nn - 1)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = r * np.sqrt(df_deg / (1 - r**2))
            p = 2 * stats.t.sf(np.abs(t), df=df_deg)
            sig = np.where(p < α)[0]
            for loc in sig:
                out.append((i, start + loc, r[loc]))
        return out

    # 4) Build delayed tasks
    tasks = [
        delayed(process_peak_gene_chunked)(
            i, X_norm, Y_norm, df_degrees, n, alpha, chunk_size
        )
        for i in range(num_peaks)
    ]

    writer = None
    schema = None

    # 5) Compute in batches and append to Parquet
    for batch_start in range(0, num_peaks, batch_size):
        batch = tasks[batch_start: batch_start + batch_size]
        with ProgressBar(dt=30, out=sys.stderr):
            results = compute(
                *batch,
                scheduler="threads",
                num_workers=num_cpu
            )

        for local_idx, sublist in enumerate(results):
            peak_idx = batch_start + local_idx
            if not sublist:
                continue

            # build small DataFrame
            df_chunk = pd.DataFrame(
                sublist,
                columns=["peak_i", "gene_j", "correlation"]
            )
            # map back to IDs
            df_chunk["peak_id"] = atac_df.index[df_chunk["peak_i"]]
            df_chunk["gene_id"] = gene_df.index[df_chunk["gene_j"]]
            df_chunk = df_chunk[["peak_id", "gene_id", "correlation"]]

            # convert to Arrow Table
            table = pa.Table.from_pandas(df_chunk, preserve_index=False)

            # initialize writer on first non-empty chunk
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(output_file, schema)

            writer.write_table(table)
            # free memory
            del df_chunk, table

        del results

    if writer:
        writer.close()
        logging.info(f"Wrote correlations to {output_file}")
    else:
        logging.warning("No significant correlations found; no Parquet written.")

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
    find_genes_near_peaks(
        peak_bed,
        tss_bed,
        rna_df,
        PEAK_DIST_LIMIT,
        TMP_DIR
    )
    
    peak_gene_df = dd.read_parquet(f"{TMP_DIR}/peak_to_gene_map.parquet")

    # ============ PEAK TO GENE CORRELATION CALCULATION ============
    PARQ = f"{TMP_DIR}/sig_peak_to_gene_corr.parquet"
    if not os.path.exists(PARQ):
        logging.info("Subsetting ATAC and RNA matrices to relevant peaks/genes")
        atac_sub = atac_df.merge(peak_gene_df[["peak_id"]].drop_duplicates(), on="peak_id", how="inner")

        rna_sub = rna_df.merge(peak_gene_df[["target_id"]].drop_duplicates(), 
                       left_on="gene_id", right_on="target_id", how="inner")
        rna_sub = rna_sub.drop(columns="target_id")
        
        # Filter out genes / peaks with low variance in expression / accessibility
        logging.info("Filtering out genes and peaks with low variance")
        rna_sub  = filter_low_variance_features(rna_sub,  min_variance=0.5)
        atac_sub  = filter_low_variance_features(atac_sub,  min_variance=0.5)
        
        logging.info("Calculating significant ATAC-seq peak-to-gene correlations")
        written = calculate_significant_peak_to_gene_correlations(
            atac_sub,
            rna_sub,
            output_file=PARQ,
            alpha=0.01,
            chunk_size=1000,
            num_cpu=NUM_CPU,
            batch_size=100
        )
        sig_peak_to_gene_corr = dd.read_parquet(written)
    else:
        logging.info("Loading existing Parquet")
        sig_peak_to_gene_corr = dd.read_parquet(PARQ)

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