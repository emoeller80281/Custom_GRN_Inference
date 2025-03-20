import pandas as pd
import numpy as np
from pybiomart import Server
import matplotlib.pyplot as plt
from itertools import combinations

organism = "hsapiens"
atac_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/K562/K562_human_filtered/K562_human_filtered_ATAC.csv"
rna_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/K562/K562_human_filtered/K562_human_filtered_RNA.csv"
peak_dist_limit = 10000

def retrieve_ensembl_gene_positions(organism):
    # Connect to the Ensembl BioMart server
    server = Server(host='http://www.ensembl.org')

    gene_ensembl_name = f'{organism}_gene_ensembl'
    
    # Select the Ensembl Mart and the human dataset
    mart = server['ENSEMBL_MART_ENSEMBL']
    dataset: pd.DataFrame = mart[gene_ensembl_name]

    # Query for attributes: Ensembl gene ID, gene name, strand, and transcription start site (TSS)
    result_df = dataset.query(attributes=[
        'external_gene_name', 
        'strand', 
        'chromosome_name',
        'transcription_start_site'
    ])

    return result_df

def load_and_parse_atac_peaks(atac_data_file):
    atac_df = pd.read_csv(atac_data_file, sep=",", header=0, index_col=None)
    peak_pos = atac_df[atac_df.columns[0]].to_list()

    peak_df = pd.DataFrame()
    peak_df["chr"] = [i.split(":")[0].strip("chr") for i in peak_pos]
    peak_df["start"] = [int(i.split(":")[1].split("-")[0]) for i in peak_pos]
    peak_df["end"] = [int(i.split(":")[1].split("-")[1]) for i in peak_pos]

    # Find the center of the peak (subtract 1/2 the length of the peak from the end)
    peak_df["center"] = peak_df["end"] - ((peak_df["end"] - peak_df["start"]) / 2)
    
    return atac_df, peak_df

print("Loading and parsing ATAC peak positions")
atac_df, peak_df = load_and_parse_atac_peaks(atac_data_file)

rna_df: pd.DataFrame = pd.read_csv(rna_data_file, sep=",", header=0, index_col=None)
rna_df = rna_df.rename(columns={rna_df.columns[0]: "gene"})

print(f"Loading ensembl genes for {organism}")
ensembl_gene_df: pd.DataFrame = retrieve_ensembl_gene_positions(organism)

# Subset to only contain genes that are in the RNA dataset
ensembl_gene_matching_genes = ensembl_gene_df[ensembl_gene_df["Gene name"].isin(rna_df["gene"])].dropna()

# Subset to only contain chromosomes and scaffolds that are present in the peak dataframe "chr" column
ensembl_gene_matching_chr = ensembl_gene_matching_genes[ensembl_gene_matching_genes["Chromosome/scaffold name"].isin(peak_df["chr"])].dropna()

from scipy.spatial import cKDTree

# Build a dictionary mapping each chromosome to a KDTree and the corresponding index list
tree_dict = {}
for chrom, group in peak_df.groupby("chr"):
    # Reshape the center positions into a 2D array (required by cKDTree)
    centers = group["center"].values.reshape(-1, 1)
    tree = cKDTree(centers)
    tree_dict[chrom] = (tree, group.index.tolist())

def find_matching_peaks_kdtree(tss, chrom, tree_dict, threshold=10000):
    # Check if the chromosome is in our KD-Tree dictionary
    if chrom not in tree_dict:
        return []
    tree, idx_list = tree_dict[chrom]
    # Query the KDTree for all peaks within 'threshold' of the TSS
    indices = tree.query_ball_point([[tss]], r=threshold)[0]
    # Convert tree indices back to the original peak_df indices
    return [idx_list[i] for i in indices]

print(f"Identifying ATACseq peaks within {peak_dist_limit} of gene TSS using KDTree")

# Apply the KDTree function row-wise
ensembl_gene_matching_chr["peaks_in_range"] = ensembl_gene_matching_chr.apply(
    lambda row: find_matching_peaks_kdtree(row["Transcription start site (TSS)"],
                                           row["Chromosome/scaffold name"],
                                           tree_dict,
                                           threshold=peak_dist_limit),
    axis=1
)

# Filter out any genes that dont have any peaks within range
ensembl_genes_within_range = ensembl_gene_matching_chr[ensembl_gene_matching_chr["peaks_in_range"].apply(lambda lst: len(lst) > 1)]

print(ensembl_genes_within_range.head())

def calculate_correlations(gene_row, rna_df_indexed, atac_df):
    """
    Calculate peak-to-gene and peak-to-peak correlations for a given gene row.
    
    Parameters:
      gene_row (Series): A row from ensembl_genes_within_range containing:
          - "Gene name": gene symbol
          - "peaks_in_range": list of peak indices
      rna_df_indexed (DataFrame): Gene expression DataFrame with index set to gene names.
      atac_df (DataFrame): ATAC-seq data (first column non-numeric).
    
    Returns:
      gene_peak_df (DataFrame): One row per peak with gene-to-peak correlation.
      peak_peak_df (DataFrame): One row per unique peak pair with their correlation.
    """
    gene_name = gene_row["Gene name"]
    peak_indices = gene_row["peaks_in_range"]
    
    # Get gene expression vector for this gene
    try:
        gene_expr = rna_df_indexed.loc[gene_name].astype(float)
    except KeyError:
        return None, None

    # Extract ATAC-seq data for the peaks (skip first column)
    selected_atac = atac_df.iloc[peak_indices, 1:].astype(float)
    # Remove peaks with zero total accessibility
    selected_atac = selected_atac[selected_atac.sum(axis=1) > 0]
    
    if selected_atac.empty:
        return None, None
    
    # Compute peak-to-gene correlation (transpose so cells are rows)
    peak_to_gene_corr = selected_atac.transpose().corrwith(gene_expr).fillna(0)
    gene_peak_df = pd.DataFrame({
        "Gene": gene_name,
        "Peak": peak_to_gene_corr.index,
        "Correlation": peak_to_gene_corr.values
    })
    
    # Compute peak-to-peak correlations if there are at least 2 peaks
    if selected_atac.shape[0] > 1:
        # Compute correlation matrix (using vectorized pandas method)
        corr_matrix = selected_atac.transpose().corr().fillna(0).values
        peaks_idx = list(selected_atac.index)
        # Use numpy triu_indices to get the upper triangle (excluding diagonal)
        iu = np.triu_indices_from(corr_matrix, k=1)
        peak_pairs = [(peaks_idx[i], peaks_idx[j]) for i, j in zip(iu[0], iu[1])]
        corr_values = corr_matrix[iu]
        peak_peak_df = pd.DataFrame({
            "Peak1": [pair[0] for pair in peak_pairs],
            "Peak2": [pair[1] for pair in peak_pairs],
            "Correlation": corr_values
        })
    else:
        peak_peak_df = pd.DataFrame(columns=["Peak1", "Peak2", "Correlation"])
    
    return gene_peak_df, peak_peak_df

def aggregate_all_correlations(genes_df, rna_df, atac_df, gene_range):
    """
    Aggregate peak-to-gene and peak-to-peak correlations for a range of genes.
    
    Parameters:
      genes_df (DataFrame): DataFrame (e.g., ensembl_genes_within_range) containing gene info.
      rna_df (DataFrame): RNA-seq expression data with a column "gene".
      atac_df (DataFrame): ATAC-seq data.
      gene_range (iterable): Row indices of genes_df to process.
      
    Returns:
      total_gene_peak_df (DataFrame): Aggregated gene-to-peak correlations.
      total_peak_peak_df (DataFrame): Aggregated peak-to-peak correlations.
    """
    # Pre-set the index for the RNA data once for efficiency
    rna_df_indexed = rna_df.set_index("gene")
    
    gene_peak_list = []
    peak_peak_list = []
    
    for i in gene_range:
        gene_row = genes_df.iloc[i]
        gene_peak_df, peak_peak_df = calculate_correlations(gene_row, rna_df_indexed, atac_df)
        if gene_peak_df is not None:
            gene_peak_list.append(gene_peak_df)
        if peak_peak_df is not None and not peak_peak_df.empty:
            peak_peak_list.append(peak_peak_df)
    
    total_gene_peak_df = pd.concat(gene_peak_list, ignore_index=True) if gene_peak_list else pd.DataFrame()
    total_peak_peak_df = pd.concat(peak_peak_list, ignore_index=True) if peak_peak_list else pd.DataFrame()
    
    return total_gene_peak_df, total_peak_peak_df

# Example usage:
# Define the gene indices you want to process (e.g., rows 1 to 14)
gene_indices = range(1, 1000)
total_gene_peak_df, total_peak_peak_df = aggregate_all_correlations(ensembl_genes_within_range, rna_df, atac_df, gene_indices)

print("Aggregated Gene-to-Peak Correlations:")
print(total_gene_peak_df.head())

print("\nAggregated Peak-to-Peak Correlations:")
print(total_peak_peak_df.head())


def plot_correlation_histogram(df, title):
    plt.figure(figsize=(5,5))
    plt.hist(df["Correlation"])
    plt.title(title)
    plt.xlabel("Correlation Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

plot_correlation_histogram(total_peak_peak_df, "Peak to Peak Correlation Scores")
plot_correlation_histogram(total_gene_peak_df, "Peak to Gene Correlation Scores")