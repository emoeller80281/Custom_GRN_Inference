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
    ensembl_df["start"] = ensembl_df["tss"]
    ensembl_df["end"] = ensembl_df["tss"] + 1

    # Re-order columns for clarity: [chr, start, end, gene]
    ensembl_df = ensembl_df[["chr", "start", "end", "gene"]]

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

    enhancer_db["start"] = enhancer_db["start"].astype(int)
    enhancer_db["end"] = enhancer_db["end"].astype(int)
    enhancer_db = enhancer_db[["chr", "start", "end", "enhancer", "score"]]

    return enhancer_db

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

def calculate_significant_peak_to_peak_correlations(atac_df, alpha=0.05):
    # Convert to sparse matrix and normalize as before
    X = sp.csr_matrix(atac_df.values)
    num_peaks, n = X.shape
    df_degrees = n - 2
    
    X_norm = row_normalize_sparse(X)
    
    def process_peak(i, X_norm, df_degrees, n, alpha):
        """
        Compute correlations for row i and return True if any non-self correlation
        is significant (p < alpha); otherwise return False.
        """
        row = X_norm.getrow(i)
        # Compute correlations of peak i with all peaks using sparse dot product.
        corr_row = row.dot(X_norm.T)
        indices = corr_row.indices
        correlations = corr_row.data
        
        # Remove self-correlation (if present)
        mask = indices != i
        if np.any(mask):
            correlations = correlations[mask]
            with np.errstate(divide='ignore', invalid='ignore'):
                t_stat = correlations * np.sqrt(df_degrees / (1 - correlations**2))
            p_vals = 2 * stats.t.sf(np.abs(t_stat), df=df_degrees)
            if np.any(p_vals < alpha):
                return True
        return False
    
    # Create delayed tasks for each peak.
    tasks = [delayed(process_peak)(i, X_norm, df_degrees, n, alpha)
             for i in range(num_peaks)]
    
    # Compute all tasks in parallel
    with ProgressBar():
        results = compute(*tasks, scheduler='threads', num_workers=4)
    
    return np.array(results)

def calculate_significant_peak_to_gene_correlations(atac_df, gene_df, alpha=0.05, chunk_size=1000):
    """
    Given:
      - atac_df: ATAC-seq DataFrame (peaks as rows, cells as columns; index are peak IDs).
      - gene_df: RNA-seq DataFrame (genes as rows, cells as columns; index are gene IDs).
      
    Convert both DataFrames to sparse matrices, normalize them, and then compute the Pearson
    correlations (via sparse dot products) for each peak against all genes.
    
    Returns:
      A DataFrame with columns ['peak_id', 'gene_id', 'correlation', 'p_value']
      for every significant (p < alpha) peak-to-gene pair.
    """
    # Convert ATAC data to sparse matrix and normalize rows.
    X = sp.csr_matrix(atac_df.values.astype(float))
    Y = sp.csr_matrix(gene_df.values.astype(float))
    
    num_peaks, n = X.shape
    df_degrees = n - 2
    
    X_norm = row_normalize_sparse(X)
    Y_norm = row_normalize_sparse(Y)
    
    def process_peak_gene_chunked(i, X_norm, Y_norm, df_degrees, n, alpha, chunk_size):
        """
        For peak i in X_norm, compute correlations with the genes in Y_norm
        chunked by 'chunk_size'. Return list of (peak_index, gene_index, r, p).
        """
        results = []
        peak_row = X_norm.getrow(i)
        num_genes = Y_norm.shape[0]

        for start in range(0, num_genes, chunk_size):
            end = min(start + chunk_size, num_genes)
            Y_chunk = Y_norm[start:end]  # shape: (chunk_size, n)

            # Dot product is (1, n) · (chunk_size, n).T -> (1, chunk_size)
            r_chunk = peak_row.dot(Y_chunk.T).toarray().ravel()

            # For normalized data, correlation = dot / (n - 1)
            r_chunk = r_chunk / (n - 1)

            # Compute t-stats + p-values
            with np.errstate(divide='ignore', invalid='ignore'):
                t_stat_chunk = r_chunk * np.sqrt(df_degrees / (1 - r_chunk**2))
            p_chunk = 2 * stats.t.sf(np.abs(t_stat_chunk), df=df_degrees)

            # Keep only significant
            sig_indices = np.where(p_chunk < alpha)[0]
            for local_j in sig_indices:
                global_j = start + local_j
                results.append((i, global_j, r_chunk[local_j], p_chunk[local_j]))

        return results
    
    # Delayed tasks for each peak
    tasks = [
        delayed(process_peak_gene_chunked)(
            i, X_norm, Y_norm, df_degrees, n, alpha, chunk_size
        )
        for i in range(num_peaks)
    ]

    with ProgressBar():
        results = compute(*tasks, scheduler="threads", num_workers=4)

    flat_results = [item for sublist in results for item in sublist]
    res_df = pd.DataFrame(
        flat_results, columns=["peak_index", "gene_index", "correlation", "p_value"]
    )

    # Map indices back
    res_df["peak_id"] = atac_df.index[res_df["peak_index"]]
    res_df["gene_id"] = gene_df.index[res_df["gene_index"]]

    return res_df[["peak_id", "gene_id", "correlation", "p_value"]]

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
# print(f'Num genes from RNA dataset in merged_df: {rna_df["gene"].nunique():,} / {merged_df["gene_id"].nunique():,} ({rna_df["gene"].nunique() / merged_df["gene_id"].nunique()*100:.2f}%)')

# Subsetting the merged_df to only contain genes in the RNA-seq dataset
# merged_df = merged_df[merged_df["gene_id"].isin(rna_df.index)]

print(f'Num ATAC-seq peaks remaining: {merged_df["peak_id"].nunique():,} / {peak_df["peak_full"].nunique():,} ({merged_df["peak_id"].nunique() / peak_df["peak_full"].nunique()*100:.2f}%)')
print(f'Num genes from Ensembl: {merged_df["gene_id"].nunique():,} / {ensembl_df["gene"].nunique():,} ({merged_df["gene_id"].nunique() / ensembl_df["gene"].nunique()*100:.2f}%)')
print(f'Num enhancers from EnhancerDB: {merged_df["enh_id"].nunique():,} / {enhancer_df["enhancer"].nunique():,} ({merged_df["enh_id"].nunique() / enhancer_df["enhancer"].nunique()*100:.2f}%)')

# Subset the ATAC-seq data df to only contain peaks that are in the final merged_df and set the index to the peak names
atac_df = atac_df[atac_df["peak_full"].isin(merged_df["peak_id"])].set_index("peak_full")

print("Calculating significant ATAC-seq peak-to-peak co-accessibility correlations")
sig_peak_to_peak_corr = calculate_significant_peak_to_peak_correlations(atac_df, alpha=0.05)
sig_atac_df = atac_df.loc[sig_peak_to_peak_corr]

sig_peak_to_gene_corr = calculate_significant_peak_to_gene_correlations(atac_df, rna_df, alpha=0.05, chunk_size=1000)
sig_rna_df = rna_df.loc[sig_peak_to_gene_corr]

# Only keep rows in merged_df where the peak_id is 
sig_merged_df = merged_df[(merged_df["peak_id"].isin(sig_atac_df.index)) & (merged_df["gene"].isin(sig_rna_df.index))]
print(sig_merged_df.head())
print(f'Number of significant peaks: {sig_merged_df.shape[0]:,} / {merged_df["peak_id"].nunique():,} ({sig_merged_df.shape[0] / merged_df["peak_id"].nunique()*100:.2f}%)')
