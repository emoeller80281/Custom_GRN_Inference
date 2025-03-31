import pandas as pd
import pybedtools
import numpy as np
from pybiomart import Server
import scipy.sparse as sp
import scipy.stats as stats
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import os
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
        "--enhancer_db_file",
        type=str,
        required=True,
        help="Path to the EnhancerDB file"
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        required=True,
        help="Path to the tmp_dir for this sample"
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
def load_atac_dataset(atac_data_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ATAC peaks from a CSV file. Parse the chromosome, start, end, and center
    for each peak.

    Returns
    -------
    atac_df : pd.DataFrame
        The raw ATAC data with the first column containing the peak positions.
    peak_df : pd.DataFrame
        The chromosome, start, end, and center of each peak.
    """
    atac_df = pd.read_csv(atac_data_file, sep=",", header=0, index_col=None)
    atac_df = atac_df.rename(columns={atac_df.columns[0]: "peak_id"})
    
    # Downcast the values from float64 to float16
    numeric_cols = atac_df.columns.drop("peak_id")
    atac_df[numeric_cols] = atac_df[numeric_cols].astype('float16')
    
    return atac_df

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

def log2_cpm_normalize(df):
    """
    Log2 CPM normalize the values for each gene / peak.
    Assumes:
      - The df's first column is a non-numeric peak / gene identifier (e.g., "chr1:100-200"),
      - columns 1..end are numeric count data for samples or cells.
    """
    # Separate the non-numeric first column
    row_ids = df.iloc[:, 0]
    
    # Numeric counts
    counts = df.iloc[:, 1:]
    
    # 1. Compute library sizes (sum of each column)
    library_sizes = counts.sum(axis=0)
    
    # 2. Convert counts to CPM
    # Divide each column by its library size, multiply by 1e6
    # Add 1 to avoid log(0) issues in the next step
    cpm = (counts.div(library_sizes, axis=1) * 1e6).add(1)
    
    # 3. Log2 transform
    log2_cpm = np.log2(cpm)
    
    # Reassemble into a single DataFrame
    normalized_df = pd.concat([row_ids, log2_cpm], axis=1)
    
    return normalized_df

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
        "Gene name": "gene"
    }, inplace=True)
    
    # Make sure TSS is integer (some might be floats).
    ensembl_df["tss"] = ensembl_df["tss"].astype(int)

    # In a BED file, we’ll store TSS as [start, end) = [tss, tss+1)
    ensembl_df["start"] = ensembl_df["tss"].astype(int)
    ensembl_df["end"] = ensembl_df["tss"].astype(int) + 1

    # Re-order columns for clarity: [chr, start, end, gene]
    ensembl_df = ensembl_df[["chr", "start", "end", "gene"]]
    
    ensembl_df["chr"] = ensembl_df["chr"].astype(str)
    ensembl_df["gene"] = ensembl_df["gene"].astype(str)
    
    # Write the peak DataFrame to a file
    ensembl_df.to_csv(f"{tmp_dir}/ensembl.bed", sep="\t", header=False, index=False)

def load_enhancer_database_file(enhancer_db_file, tmp_dir):
    enhancer_db = pd.read_csv(enhancer_db_file, sep="\t", header=None, index_col=None)
    enhancer_db = enhancer_db.rename(columns={
        0 : "chr",
        1 : "start",
        2 : "end",
        3 : "enhancer",
        4 : "tissue",
        5 : "R1_value",
        6 : "R2_value",
        7 : "R3_value",
        8 : "score"
    })
    
    # Remove the "chr" before chromosome number
    enhancer_db["chr"] = enhancer_db["chr"].str.replace("^chr", "", regex=True)
    
    # Average the score of an enhancer across all tissues / cell types
    enhancer_db = enhancer_db.groupby(["chr", "start", "end", "enhancer"], as_index=False)["score"].mean()

    enhancer_db["chr"] = enhancer_db["chr"].astype(str)
    enhancer_db["start"] = enhancer_db["start"].astype(int)
    enhancer_db["end"] = enhancer_db["end"].astype(int)
    enhancer_db["enhancer"] = enhancer_db["enhancer"].astype(str)
    
    enhancer_db = enhancer_db[["chr", "start", "end", "enhancer", "score"]]
    
    # Write the peak DataFrame to a file
    enhancer_db.to_csv(f"{tmp_dir}/enhancer.bed", sep="\t", header=False, index=False)

def find_genes_near_peaks(peak_bed, tss_bed, rna_df, peak_dist_limit):
    # 3) Find peaks that are within PEAK_DIST_LIMIT bp of each gene's TSS
    logging.info(f"Locating peaks that are within {peak_dist_limit} bp of each gene's TSS")
    peak_tss_overlap = peak_bed.window(tss_bed, w=peak_dist_limit)
    
    peak_tss_overlap_df = peak_tss_overlap.to_dataframe(
        names=[
            "peak_chr", "peak_start", "peak_end", "peak_id",
            "gene_chr", "gene_start", "gene_end", "gene_id"
        ]
    ).dropna()
    
    # Calculate the TSS distance for each peak - gene pair
    peak_tss_overlap_df["TSS_dist"] = np.abs(peak_tss_overlap_df["peak_end"] - peak_tss_overlap_df["gene_start"])
    peak_tss_subset_df = peak_tss_overlap_df[["peak_id", "gene_id", "TSS_dist"]]
    
    # Take the minimum peak to gene TSS distance
    peak_tss_subset_df = peak_tss_subset_df.sort_values("TSS_dist")
    peak_tss_subset_df = peak_tss_subset_df.drop_duplicates(subset=["peak_id", "gene_id"], keep="first")
    
    # Only keep genes that are also in the RNA-seq dataset
    peak_tss_subset_df = peak_tss_subset_df[peak_tss_subset_df["gene_id"].isin(rna_df["gene"])]
    
    gene_list = set(peak_tss_subset_df["gene_id"].drop_duplicates().to_list())
    
    # logging.info("\n-----------------------------------------\n")
    
    return peak_tss_subset_df, gene_list

def find_peaks_in_known_enhancer_region(peak_bed, enh_bed):
    # 4) Find peaks that overlap with known enhancer locations from EnhancerDB
    logging.info("Locating peaks that overlap with known enhancer locations from EnhancerDB")
    peak_enh_overlap = peak_bed.intersect(enh_bed, wa=True, wb=True)
    peak_enh_overlap_df = peak_enh_overlap.to_dataframe(
        names=[
            "peak_chr", "peak_start", "peak_end", "peak_id",
            "enh_chr", "enh_start", "enh_end", "enh_id",
            "enh_score"  # only if you had a score column in your enhancers
        ]
    ).dropna()
    peak_enh_overlap_subset_df = peak_enh_overlap_df[["peak_id", "enh_id", "enh_score"]]
        
    return peak_enh_overlap_subset_df

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
    Returns a DataFrame of [peak_id, gene_id, correlation] for p < alpha.
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

    with ProgressBar():
        results = compute(*tasks, scheduler="threads", num_workers=num_cpu)

    flat_results = [item for sublist in results for item in sublist]
    df_corr = pd.DataFrame(flat_results, columns=["peak_i", "gene_j", "correlation"])

    # Map i->peak_id, j->gene_id
    df_corr["peak"] = atac_df.index[df_corr["peak_i"]]
    df_corr["gene"] = gene_df.index[df_corr["gene_j"]]
    df_corr = df_corr[["peak", "gene", "correlation"]]

    return df_corr

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()
    
    if args.species == "human" | "hg38":
        ORGANISM = "hsapiens"
    elif args.species == "mouse" | "mm10":
        ORGANISM = "mmusculus"
    else:
        raise Exception(f'Organism not found, you entered {args.species} (must be one of: "human", "hg38", "mouse", or "mm10")')
    
    ATAC_DATA_FILE = args.atac_data_file
    RNA_DATA_FILE =  args.rna_data_file
    ENHANCER_DB_FILE = args.enhancer_db_file
    OUTPUT_DIR = args.output_dir
    TMP_DIR = f"{OUTPUT_DIR}/tmp"
    NUM_CPU = args.num_cpu
    PEAK_DIST_LIMIT=args.peak_dist_limit
    
    # Make the tmp dir if it does not exist
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
    
    logging.info("Loading the scRNA-seq dataset.")
    rna_df = pd.read_csv(RNA_DATA_FILE, sep=",", header=0, index_col=None)
    rna_df = rna_df.rename(columns={rna_df.columns[0]: "gene"})
    # logging.info(rna_df.head())
    # logging.info("\n-----------------------------------------\n")

    logging.info("Log2 CPM normalizing the RNA-seq data")
    rna_df = log2_cpm_normalize(rna_df)

    # Downcast the values from float64 to float16
    numeric_cols = rna_df.columns.drop("gene")
    rna_df[numeric_cols] = rna_df[numeric_cols].astype('float16')
    # logging.info(rna_df.head())
    # logging.info("\n-----------------------------------------\n")

    logging.info("Loading and parsing the ATAC-seq peaks")
    atac_df = load_atac_dataset(ATAC_DATA_FILE)

    logging.info("Log2 CPM normalizing the ATAC-seq data")
    atac_df = log2_cpm_normalize(atac_df)
    # logging.info(atac_df.head())
    # logging.info("\n-----------------------------------------\n")

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
    
    # Load the peak, gene TSS, and enhancer BED files
    peak_bed = pybedtools.BedTool(f"{TMP_DIR}/peak_df.bed")
    tss_bed = pybedtools.BedTool(f"{TMP_DIR}/ensembl.bed")
    
    # ============ FINDING PEAKS NEAR GENES ============
    # Dataframe with "peak_id", "gene_id" and "TSS_dist"
    peak_gene_df, gene_list = find_genes_near_peaks(peak_bed, tss_bed, rna_df, PEAK_DIST_LIMIT)

    if ORGANISM == "hsapiens":
        # ============ MAPPING PEAKS TO KNOWN ENHANCERS ============
        # Dataframe with "peak_id", "enh_id", and "enh_score" columns
        if not os.path.exists(f"{TMP_DIR}/enhancer.bed"):
            logging.info("Loading known enhancer locations from EnhancerDB and saving as a bed file")
            load_enhancer_database_file(ENHANCER_DB_FILE, TMP_DIR)
        else:
            logging.info("Enhancer BED file exists, loading...")
        enh_bed = pybedtools.BedTool(f"{TMP_DIR}/enhancer.bed")
        peak_enh_df = find_peaks_in_known_enhancer_region(peak_bed, enh_bed)

        logging.info("Merging the peak to gene mapping with the known enhancer location mapping")
        peak_gene_df = pd.merge(peak_gene_df, peak_enh_df, how="left", on="peak_id")

    logging.info("Subset the ATAC-seq DataFrame to only contain peak that are in range of the genes")
    atac_sub = atac_df[atac_df["peak_id"].isin(peak_gene_df["peak_id"])]
    # logging.info(atac_sub.head())
    # logging.info("\n-----------------------------------------\n")

    # ============ PEAK TO GENE CORRELATION CALCULATION ============
    if not os.path.exists(f"{TMP_DIR}/sig_peak_to_gene_corr.parquet"):
        # Subset the RNA-seq DataFrame to only contain the genes in the peak-to-gene dictionary
        rna_sub = rna_df[rna_df["gene"].isin(gene_list)].set_index("gene")
        
        # Filter out genes / peaks with low variance in expression / accessibility
        logging.info("Filtering out genes and peaks with low variance")
        rna_sub  = filter_low_variance_features(rna_sub,  min_variance=0.5)
        atac_sub  = filter_low_variance_features(atac_sub,  min_variance=0.5)
        
        logging.info("Calculating significant ATAC-seq peak-to-gene correlations")
        sig_peak_to_gene_corr = calculate_significant_peak_to_gene_correlations(atac_sub, rna_sub, alpha=0.05, num_cpu=NUM_CPU)
        logging.info(sig_peak_to_gene_corr.head())
        # logging.info(f'Number of significant peak to gene correlations: {sig_peak_to_gene_corr.shape[0]:,}')
        # logging.info("\n-----------------------------------------\n")

        sig_peak_to_gene_corr.to_parquet(f"{TMP_DIR}/sig_peak_to_gene_corr.parquet")
    else:
        logging.info("sig_peak_to_gene_corr.parquet exists, loading")
        sig_peak_to_gene_corr = pd.read_parquet(f"{TMP_DIR}/sig_peak_to_gene_corr.parquet")
        # logging.info(sig_peak_to_gene_corr.head())
        # logging.info("\n-----------------------------------------\n")

    def subset_by_highest_correlations(df, threshold=0.9):
        
        # Compute the correlation threshold corresponding to the top 10%
        cutoff = df["correlation"].quantile(threshold)

        # Filter to only keep rows with correlation greater than or equal to the threshold
        top_10_df = df[df["correlation"] >= cutoff]
        
        return top_10_df

    quantile_threshold = 0.75
    logging.info(f"Subsetting to only retain correlations in the top {quantile_threshold}")
    top_peak_to_gene_corr = subset_by_highest_correlations(sig_peak_to_gene_corr, quantile_threshold)

    # Merge the gene and enhancer df
    final_df = pd.merge(top_peak_to_gene_corr, peak_gene_df, how="inner", left_on=["peak", "gene"], right_on=["peak_id", "gene_id"]).dropna(subset="peak_id")
    
    # EnhancerDB only has entries for human
    if ORGANISM == "hsapiens":
        final_df[["enh_score"]] = final_df[["enh_score"]].fillna(value=0)
        final_df = final_df[["peak_id", "gene_id", "correlation", "TSS_dist", "enh_score"]]
    else:
        final_df = final_df[["peak_id", "gene_id", "correlation", "TSS_dist"]]
        
    logging.info(final_df.head())
    final_df.to_csv("peak_to_gene_correlation.csv", sep="\t", header=True, index=False)
    logging.info("\n-----------------------------------------\n")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()