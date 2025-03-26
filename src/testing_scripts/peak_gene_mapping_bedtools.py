import pandas as pd
import pybedtools
import numpy as np
from pybiomart import Server
import scipy.sparse as sp
import scipy.stats as stats
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 1) Create or load peak_df and enhancer_df (each has [chr, start, end, <ID>, <score>])
# ------------------------- CONFIGURATIONS ------------------------- #
ORGANISM = "hsapiens"
ATAC_DATA_FILE = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/K562/K562_human_filtered/K562_human_filtered_ATAC.csv"
RNA_DATA_FILE =  "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/K562/K562_human_filtered/K562_human_filtered_RNA.csv"
ENHANCER_DB_FILE = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/enhancer_db/enhancer"
TMP_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/tmp"
PEAK_DIST_LIMIT = 1_000_000

# ------------------------- DATA LOADING & PREPARATION ------------------------- #
def load_and_parse_atac_peaks(atac_data_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    atac_df = atac_df.rename(columns={atac_df.columns[0]: "peak_full"})
    
    peak_pos = atac_df["peak_full"].tolist()

    peak_df = pd.DataFrame()
    peak_df["chr"] = [pos.split(":")[0].replace("chr", "") for pos in peak_pos]
    peak_df["start"] = [int(pos.split(":")[1].split("-")[0]) for pos in peak_pos]
    peak_df["end"] = [int(pos.split(":")[1].split("-")[1]) for pos in peak_pos]
    peak_df["peak_full"] = peak_pos

    peak_df["chr"] = peak_df["chr"].astype(str)
    peak_df["start"] = peak_df["start"].astype(int)
    peak_df["end"] = peak_df["end"].astype(int)
    peak_df["peak_full"] = peak_df["peak_full"].astype(str)

    return peak_df, atac_df

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

def load_ensembl_organism_tss(organism):
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

    return ensembl_df

def load_enhancer_database_file(enhancer_db_file):
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

    return enhancer_db

def set_merged_df_col_dtypes(merged_df):
    merged_df["peak_chr"] = merged_df["peak_chr"].apply(str)
    merged_df["gene_chr"] = merged_df["gene_chr"].apply(str)
    merged_df["enh_chr"] = merged_df["enh_chr"].apply(str)

    merged_df["gene_start"] = merged_df["gene_start"].astype(int)
    merged_df["gene_end"] = merged_df["gene_end"].astype(int)
    merged_df["enh_start"] = merged_df["enh_start"].astype(int)
    merged_df["enh_end"] = merged_df["enh_end"].astype(int)
    merged_df["enh_score"] = pd.to_numeric(merged_df["enh_score"], errors="coerce")
    
    return merged_df

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

def calculate_significant_peak_to_peak_correlations(atac_df, alpha=0.05):
    """
    For each pair of peaks (i, j), compute correlation and p-value, return only
    those pairs with p < alpha. Returns a DataFrame with columns:
    [peak1, peak2, correlation].
    """
    X = sp.csr_matrix(atac_df.values)
    num_peaks, n = X.shape
    df_degrees = n - 2

    X_norm = row_normalize_sparse(X)

    def process_peak(i, X_norm, df_degrees, n, alpha):
        """
        Compute correlations for row i vs. all other peaks. Return a list of
        (i, j, correlation) for each pair (i, j) with p < alpha.
        """
        results = []
        row_i = X_norm.getrow(i)
        # sparse dot product => shape(1, num_peaks)
        corr_row = row_i.dot(X_norm.T)
        indices = corr_row.indices
        correlations = corr_row.data

        # Remove self-correlation
        mask = (indices != i)
        indices = indices[mask]
        correlations = correlations[mask]

        if len(indices) == 0:
            return results  # empty list

        # Compute p-values
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stat = correlations * np.sqrt(df_degrees / (1 - correlations**2))
        p_vals = 2 * stats.t.sf(np.abs(t_stat), df=df_degrees)

        # Collect only significant pairs
        sig_mask = (p_vals < alpha)
        sig_indices = indices[sig_mask]
        sig_corrs = correlations[sig_mask]

        # Build tuples: (i, j, corr)
        for j, c in zip(sig_indices, sig_corrs):
            # We'll handle duplicates (i, j) vs. (j, i) later if desired
            results.append((i, j, c))

        return results

    # Use Dask to process each peak
    tasks = [delayed(process_peak)(i, X_norm, df_degrees, n, alpha) for i in range(num_peaks)]

    with ProgressBar():
        results = compute(*tasks, scheduler='threads', num_workers=4)

    # Flatten list of lists
    flat_results = [item for sublist in results for item in sublist]

    # Convert to DataFrame
    df_corr = pd.DataFrame(flat_results, columns=["peak_i", "peak_j", "correlation"])
    
    # Filter to only keep highly correlated peak-to-peak connections
    df_corr = df_corr[ abs(df_corr["correlation"]) > 0.2 ]
    
    # Map i,j to actual peak IDs
    df_corr["peak1"] = atac_df.index[df_corr["peak_i"]]
    df_corr["peak2"] = atac_df.index[df_corr["peak_j"]]
    
    # Drop duplicates (peakA, peakB) vs (peakB, peakA)
    df_corr[["peak1", "peak2"]] = np.sort(df_corr[["peak1", "peak2"]], axis=1)
    df_corr.drop_duplicates(subset=["peak1","peak2"], keep="first")

    # Final subset and reorder columns
    df_corr = df_corr[["peak1", "peak2", "correlation"]]
    return df_corr

def calculate_significant_peak_to_gene_correlations(atac_df, gene_df, alpha=0.05, chunk_size=1000):
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
        results = compute(*tasks, scheduler="threads", num_workers=4)

    flat_results = [item for sublist in results for item in sublist]
    df_corr = pd.DataFrame(flat_results, columns=["peak_i", "gene_j", "correlation"])

    # Map i->peak_id, j->gene_id
    df_corr["peak"] = atac_df.index[df_corr["peak_i"]]
    df_corr["gene"] = gene_df.index[df_corr["gene_j"]]
    df_corr = df_corr[["peak", "gene", "correlation"]]

    return df_corr

logging.info("Loading the scRNA-seq dataset.")
rna_df = pd.read_csv(RNA_DATA_FILE, sep=",", header=0, index_col=None)
rna_df = rna_df.rename(columns={rna_df.columns[0]: "gene"})
logging.info(rna_df.head())
logging.info("\n-----------------------------------------\n")

logging.info("Log2 CPM normalizing the RNA-seq data")
rna_df = log2_cpm_normalize(rna_df)
logging.info(rna_df.head())
logging.info("\n-----------------------------------------\n")

logging.info("Loading and parsing the ATAC-seq peaks")
peak_df, atac_df = load_and_parse_atac_peaks(ATAC_DATA_FILE)
logging.info(peak_df.head())
logging.info("\n-----------------------------------------\n")

logging.info("Log2 CPM normalizing the ATAC-seq data")
atac_df = log2_cpm_normalize(atac_df)
logging.info(atac_df.head())
logging.info("\n-----------------------------------------\n")

logging.info(f"Loading the TSS locations for {ORGANISM} from Ensembl")
ensembl_df = load_ensembl_organism_tss(ORGANISM)
logging.info(ensembl_df.head())
logging.info("\n-----------------------------------------\n")

logging.info("Loading known enhancer locations from EnhancerDB")
enhancer_df = load_enhancer_database_file(ENHANCER_DB_FILE)
logging.info(enhancer_df.head())
logging.info("\n-----------------------------------------\n")

# 2) Convert to BedTool objects
peak_bed = pybedtools.BedTool.from_dataframe(peak_df)
tss_bed = pybedtools.BedTool.from_dataframe(ensembl_df)
enh_bed = pybedtools.BedTool.from_dataframe(enhancer_df)

# 3) Find peaks that are within PEAK_DIST_LIMIT bp of each gene's TSS
logging.info(f"Find peaks that are within {PEAK_DIST_LIMIT} bp of each gene's TSS")
peak_tss_overlap = peak_bed.window(tss_bed, w=PEAK_DIST_LIMIT)
peak_tss_overlap_df = peak_tss_overlap.to_dataframe(
    names=[
        "peak_chr", "peak_start", "peak_end", "peak_id",
        "gene_chr", "gene_start", "gene_end", "gene_id"
    ]
)
logging.info(peak_tss_overlap_df.head())
logging.info("\n-----------------------------------------\n")

# 4) Find peaks that overlap with known enhancer locations from EnhancerDB
logging.info("Find peaks that overlap with known enhancer locations from EnhancerDB")
peak_enh_overlap = peak_bed.intersect(enh_bed, wa=True, wb=True)
peak_enh_overlap_df = peak_enh_overlap.to_dataframe(
    names=[
        "peak_chr", "peak_start", "peak_end", "peak_id",
        "enh_chr", "enh_start", "enh_end", "enh_id",
        "enh_score"  # only if you had a score column in your enhancers
    ]
)
logging.info(peak_enh_overlap_df.head())
logging.info("\n-----------------------------------------\n")

logging.info("Merging peaks that both overlap the TSS and which are in known enhancer regions")
merged_df = pd.merge(peak_tss_overlap_df, peak_enh_overlap_df, on=["peak_chr", "peak_start", "peak_end", "peak_id"], how="outer").dropna()

# Remove duplicate peak-enh-gene pairs where only the "gene_start" value is different. Take the closest "gene_start" value instead
merged_df = merged_df.sort_values("gene_start")
merged_df = merged_df.drop_duplicates(subset=["peak_id", "gene_id", "enh_id"], keep="first")

# Remove rows where the enhancer has no score
merged_df = merged_df[merged_df["enh_score"] != "."]

# Subsetting the RNA-seq dataset to only contain genes in the merged_df
rna_df = rna_df[rna_df["gene"].isin(merged_df["gene_id"])].set_index("gene")

logging.info(f'Num ATAC-seq peaks remaining: {merged_df["peak_id"].nunique():,} / {peak_df["peak_full"].nunique():,} ({merged_df["peak_id"].nunique() / peak_df["peak_full"].nunique()*100:.2f}%)')
logging.info(f'Num genes from Ensembl: {merged_df["gene_id"].nunique():,} / {ensembl_df["gene"].nunique():,} ({merged_df["gene_id"].nunique() / ensembl_df["gene"].nunique()*100:.2f}%)')
logging.info(f'Num enhancers from EnhancerDB: {merged_df["enh_id"].nunique():,} / {enhancer_df["enhancer"].nunique():,} ({merged_df["enh_id"].nunique() / enhancer_df["enhancer"].nunique()*100:.2f}%)')
logging.info("\n-----------------------------------------\n")

# Ensure the correct data types for each column
merged_df = set_merged_df_col_dtypes(merged_df)
print(merged_df.dtypes)

merged_df.to_parquet(f"{TMP_DIR}/merged_df.parquet", compression="gzip")

# Subset the ATAC-seq data df to only contain peaks that are in the final merged_df and set the index to the peak names
atac_df = atac_df.set_index("peak_full")

# 1) Identify relevant peaks and genes from merged_df
peaks_in_merged = merged_df["peak_id"].unique()
genes_in_merged = merged_df["gene_id"].unique()

# 2) Subset the ATAC-seq DataFrame to only contain peaks in merged_df
atac_sub = atac_df.loc[atac_df.index.intersection(peaks_in_merged)]

# 3) Subset the RNA-seq DataFrame to only contain genes in merged_df
rna_sub = rna_df.loc[rna_df.index.intersection(genes_in_merged)]

logging.info(f"Subsetting to {len(atac_sub)} peaks and {len(rna_sub)} genes from merged_df")

# 4) Filter out peaks / genes with low variance in expression
atac_sub = filter_low_variance_features(atac_sub, min_variance=0.5)
rna_sub  = filter_low_variance_features(rna_sub,  min_variance=0.5)
logging.info(f"After filtering variance: {len(atac_sub)} peaks, {len(rna_sub)} genes")

# 5) Now compute correlations only among these subsets:
logging.info("Calculating significant ATAC-seq peak-to-peak co-accessibility correlations")
sig_peak_to_peak_corr = calculate_significant_peak_to_peak_correlations(atac_sub, alpha=0.05)
logging.info(sig_peak_to_peak_corr.head())
logging.info(f'Number of significant peak to peak correlations: {sig_peak_to_peak_corr.shape[0]:,}')
logging.info("\n-----------------------------------------\n")

sig_peak_to_peak_corr.to_parquet(f"{TMP_DIR}/sig_peak_to_peak_corr.parquet", compression="gzip")

logging.info("Calculating significant ATAC-seq peak-to-gene correlations")
sig_peak_to_gene_corr = calculate_significant_peak_to_gene_correlations(atac_sub, rna_sub, alpha=0.05)
logging.info(sig_peak_to_gene_corr.head())
logging.info(f'Number of significant peak to gene correlations: {sig_peak_to_gene_corr.shape[0]:,}')
logging.info("\n-----------------------------------------\n")

sig_peak_to_gene_corr.to_parquet(f"{TMP_DIR}/sig_peak_to_gene_corr.parquet", compression="gzip")

# Only keep rows in merged_df where the peak_id is 
logging.info("Subsetting merged_df to only contain peaks and genes with a significant correlation")

# Filter out rows containing peaks with no significant peak to peak or peak to gene correlations
mask_peak = (
    merged_df["peak_id"].isin(sig_peak_to_peak_corr["peak1"]) |
    merged_df["peak_id"].isin(sig_peak_to_peak_corr["peak2"]) |
    merged_df["peak_id"].isin(sig_peak_to_gene_corr["peak"])
)

# Filter out genes that dont have any significant peaks
mask_gene = merged_df["gene_id"].isin(sig_peak_to_gene_corr["gene"])
subset_mask = mask_peak & mask_gene

sig_merged_df = merged_df[mask_peak]
logging.info(sig_merged_df.head())

print(f'Number of significant peaks: {sig_merged_df.shape[0]:,} / {merged_df["peak_id"].nunique():,} ({sig_merged_df.shape[0] / merged_df["peak_id"].nunique()*100:.2f}%)')

print(f'Num ATAC-seq peaks remaining: {sig_merged_df["peak_id"].nunique():,} / {peak_df["peak_full"].nunique():,} ({sig_merged_df["peak_id"].nunique() / peak_df["peak_full"].nunique()*100:.2f}%)')
print(f'Num genes from Ensembl: {sig_merged_df["gene_id"].nunique():,} / {ensembl_df["gene"].nunique():,} ({sig_merged_df["gene_id"].nunique() / ensembl_df["gene"].nunique()*100:.2f}%)')
print(f'Num enhancers from EnhancerDB: {sig_merged_df["enh_id"].nunique():,} / {enhancer_df["enhancer"].nunique():,} ({sig_merged_df["enh_id"].nunique() / enhancer_df["enhancer"].nunique()*100:.2f}%)')
