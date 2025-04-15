import pandas as pd
import pybedtools
import numpy as np
from pybiomart import Server
import scipy.sparse as sp
import scipy.stats as stats
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import math
import os
import sys
import argparse
import logging

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
def load_atac_dataset(atac_data_file: str) -> pd.DataFrame:
    """
    Load ATAC peaks from a CSV file. Parse the chromosome, start, end, and center
    for each peak.

    Returns
    -------
    atac_df : pd.DataFrame
        The raw ATAC data with the first column containing the peak positions.
    """
    atac_df: pd.DataFrame = pd.read_csv(atac_data_file, sep="\t", header=0)
    
    return atac_df

def load_rna_dataset(rna_data_file: str) -> pd.DataFrame:
    rna_df: pd.DataFrame = pd.read_csv(rna_data_file, sep="\t", header=0)
    
    return rna_df

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
    peak_df.to_csv(f"{tmp_dir}/peak_df.bed", sep="\t", header=False, index=False)

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
    ensembl_df.to_csv(f"{tmp_dir}/ensembl.bed", sep="\t", header=False, index=False)

def find_genes_near_peaks(peak_bed, tss_bed, rna_df, peak_dist_limit):
    # 3) Find peaks that are within PEAK_DIST_LIMIT bp of each gene's TSS
    logging.info(f"Locating peaks that are within {peak_dist_limit} bp of each gene's TSS")
    peak_tss_overlap = peak_bed.window(tss_bed, w=peak_dist_limit)
    
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
    
    peak_tss_overlap_df = peak_tss_overlap.to_dataframe(
        names=[
            "peak_chr", "peak_start", "peak_end", "peak_id",
            "gene_chr", "gene_start", "gene_end", "gene_id"
        ],
        dtype=dtype_dict,
        low_memory=False  # ensures the entire file is read in one go
    ).rename(columns={"gene_id": "target_id"}).dropna()
    
    # Calculate the TSS distance for each peak - gene pair
    peak_tss_overlap_df["TSS_dist"] = np.abs(peak_tss_overlap_df["peak_end"] - peak_tss_overlap_df["gene_start"])
    
    # Scale the TSS distance score by an exponential drop-off function 
    # (e^-dist/25000, same scaling function used in LINGER Cis-regulatory potential calculation)
    # https://github.com/Durenlab/LINGER
    peak_tss_overlap_df["TSS_dist"] = peak_tss_overlap_df["TSS_dist"].apply(lambda x: math.exp(-(x/25000)))
    
    peak_tss_subset_df = peak_tss_overlap_df[["peak_id", "target_id", "TSS_dist"]]
    
    # Take the minimum peak to gene TSS distance
    peak_tss_subset_df = peak_tss_subset_df.sort_values("TSS_dist")
    peak_tss_subset_df = peak_tss_subset_df.drop_duplicates(subset=["peak_id", "target_id"], keep="first")
    
    # Only keep genes that are also in the RNA-seq dataset
    peak_tss_subset_df = peak_tss_subset_df[peak_tss_subset_df["target_id"].isin(rna_df["gene_id"])]
    
    gene_list = set(peak_tss_subset_df["target_id"].drop_duplicates().to_list())
    
    # logging.info("\n-----------------------------------------\n")
    
    return peak_tss_subset_df, gene_list

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
    """
    Filter rows of 'df' (features) that have variance < min_variance.
    Returns the filtered DataFrame and the mask of kept rows.
    """
    variances = df.var(axis=1)
    mask = variances >= min_variance
    return df.loc[mask]

def calculate_significant_peak_to_gene_correlations(atac_df, gene_df, alpha=0.05, chunk_size=1000, num_cpu=4):
    """
    Returns a DataFrame of [peak_id, target_id, correlation] for p < alpha.
    """
    # Convert to sparse
    X = sp.csr_matrix(atac_df.values.astype(float))
    Y = sp.csr_matrix(gene_df.values.astype(float))
    
    num_peaks, n = X.shape
    df_degrees = n - 2
    
    X_norm = row_normalize_sparse(X)
    Y_norm = row_normalize_sparse(Y)
    
    def process_peak_gene_chunked(i, X_norm, Y_norm, df_degrees, n, alpha, chunk_size):
        """
        For peak i, compute correlation with each gene in chunks, returning
        (i, j, r, p) for significant pairs.
        """
        results = []
        peak_row = X_norm.getrow(i)
        num_genes = Y_norm.shape[0]

        for start in range(0, num_genes, chunk_size):
            end = min(start + chunk_size, num_genes)
            Y_chunk = Y_norm[start:end]

            # Dot product => shape(1, chunk_size)
            r_chunk = peak_row.dot(Y_chunk.T).toarray().ravel()
            r_chunk = r_chunk / (n - 1)

            with np.errstate(divide='ignore', invalid='ignore'):
                t_stat_chunk = r_chunk * np.sqrt(df_degrees / (1 - r_chunk**2))
            p_chunk = 2 * stats.t.sf(np.abs(t_stat_chunk), df=df_degrees)

            # Indices where p < alpha
            sig_indices = np.where(p_chunk < alpha)[0]
            for local_j in sig_indices:
                global_j = start + local_j
                results.append((i, global_j, r_chunk[local_j]))

        return results
    
    tasks = [
        delayed(process_peak_gene_chunked)(i, X_norm, Y_norm, df_degrees, n, alpha, chunk_size)
        for i in range(num_peaks)
    ]

    with ProgressBar(dt=30, out=sys.stderr):
        results = compute(*tasks, scheduler="threads", num_workers=num_cpu)

    flat_results = [item for sublist in results for item in sublist]
    df_corr = pd.DataFrame(flat_results, columns=["peak_i", "gene_j", "correlation"])

    # Map i->peak_id, j->target_id
    df_corr["peak_id"] = atac_df.index[df_corr["peak_i"]]
    df_corr["gene_id"] = gene_df.index[df_corr["gene_j"]]
    df_corr = df_corr[["peak_id", "gene_id", "correlation"]]
    
    logging.info("\tDone!")

    return df_corr

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
    rna_df: pd.DataFrame = pd.read_csv(RNA_DATA_FILE, header=0)

    logging.info("Loading and parsing the ATAC-seq peaks")
    atac_df: pd.DataFrame = pd.read_csv(ATAC_DATA_FILE, header=0)

    if not os.path.exists(f"{TMP_DIR}/peak_df.bed"):
        logging.info(f"Extracting peak information and saving as a bed file")
        extract_atac_peaks(atac_df, TMP_DIR)
    else:
        logging.info("ATAC-seq BED file exists, loading...")

    if not os.path.exists(f"{TMP_DIR}/ensembl.bed"):
        logging.info(f"Extracting TSS locations for {ORGANISM} from Ensembl and saving as a bed file")
        load_ensembl_organism_tss(ORGANISM, TMP_DIR)
    else:
        logging.info("Ensembl gene TSS BED file exists, loading...")

    pybedtools.set_tempdir(TMP_DIR)
    
    # Load the peak and gene TSS BED files
    peak_bed = pybedtools.BedTool(f"{TMP_DIR}/peak_df.bed")
    tss_bed = pybedtools.BedTool(f"{TMP_DIR}/ensembl.bed")
    
    # ============ FINDING PEAKS NEAR GENES ============
    # Dataframe with "peak_id", "target_id" and "TSS_dist"
    peak_gene_df, gene_list = find_genes_near_peaks(peak_bed, tss_bed, rna_df, PEAK_DIST_LIMIT)

    logging.info("Subset the ATAC-seq DataFrame to only contain peak that are in range of the genes")
    atac_sub = atac_df[atac_df["peak_id"].isin(peak_gene_df["peak_id"])].set_index("peak_id")

    # ============ PEAK TO GENE CORRELATION CALCULATION ============
    if not os.path.exists(f"{TMP_DIR}/sig_peak_to_gene_corr.parquet"):
        # Subset the RNA-seq DataFrame to only contain the genes in the peak-to-gene dictionary
        rna_sub = rna_df[rna_df["gene_id"].isin(gene_list)].set_index("gene_id")
        
        # Filter out genes / peaks with low variance in expression / accessibility
        logging.info("Filtering out genes and peaks with low variance")
        rna_sub  = filter_low_variance_features(rna_sub,  min_variance=0.5)
        atac_sub  = filter_low_variance_features(atac_sub,  min_variance=0.5)
        
        logging.info("Calculating significant ATAC-seq peak-to-gene correlations")
        sig_peak_to_gene_corr = calculate_significant_peak_to_gene_correlations(atac_sub, rna_sub, alpha=0.05, num_cpu=NUM_CPU)
        logging.info(sig_peak_to_gene_corr.head())

        sig_peak_to_gene_corr.to_parquet(f"{TMP_DIR}/sig_peak_to_gene_corr.parquet")
    else:
        logging.info("sig_peak_to_gene_corr.parquet exists, loading")
        sig_peak_to_gene_corr = pd.read_parquet(f"{TMP_DIR}/sig_peak_to_gene_corr.parquet")

    quantile_threshold = 0.75
    logging.info(f"Subsetting to only retain correlations in the top {quantile_threshold} quantile")
    cutoff = sig_peak_to_gene_corr["correlation"].quantile(quantile_threshold)
    top_peak_to_gene_corr = sig_peak_to_gene_corr[sig_peak_to_gene_corr["correlation"] >= cutoff]

    # Merge the correlation and TSS distance DataFrames
    final_df = pd.merge(top_peak_to_gene_corr, peak_gene_df, how="inner", left_on=["peak_id", "gene_id"], right_on=["peak_id", "target_id"]).dropna(subset="peak_id")
    
    final_df = final_df[["peak_id", "target_id", "correlation", "TSS_dist"]]
        
    logging.info(final_df.head())
    final_df.to_csv(f"{OUTPUT_DIR}/peak_to_gene_correlation.csv", sep="\t", header=True, index=False)
    logging.info("\n-----------------------------------------\n")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()