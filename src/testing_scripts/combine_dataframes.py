import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def minmax_normalize_column(column: pd.DataFrame):
    return (column - column.min()) / (column.max() - column.min())

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

def plot_column_histograms(df, output_dir):
    # Create a figure and axes with a suitable size
    plt.figure(figsize=(15, 8))
    
    # Select only the numerical columns (those with numeric dtype)
    cols = df.select_dtypes(include=[np.number]).columns

    # Loop through each feature and create a subplot
    for i, col in enumerate(cols, 1):
        plt.subplot(2, 3, i)  # 2 rows, 4 columns, index = i
        plt.hist(df[col], bins=50, alpha=0.7, edgecolor='black')
        plt.title(f"{col} distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f'{output_dir}/column_histograms.png', dpi=300)
    plt.close()

output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/K562/K562_human_filtered/"
rna_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/K562/K562_human_filtered/K562_human_filtered_RNA.csv"
atac_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/K562/K562_human_filtered/K562_human_filtered_ATAC.csv"

print("Loading in the DataFrames")
print("\tCorrelation peak to TG DataFrame")
peak_corr_df = pd.read_csv(f'{output_dir}/peak_to_gene_correlation.csv', sep="\t", header=0, index_col=None)

print("\tCicero peak to TG DataFrame")
cicero_df = pd.read_csv(f'{output_dir}/cicero_peak_to_tg_scores.csv', sep="\t", header=0, index_col=None)

print("\tSliding Window peak to TG DataFrame")
sliding_window_df = pd.read_csv(f'{output_dir}/sliding_window_tf_to_peak_score.tsv', sep="\t", header=0, index_col=None)

print("\tRNAseq dataset")
rna_df = pd.read_csv(rna_data_file, sep=",", header=0, index_col=None)
rna_df = rna_df.rename(columns={rna_df.columns[0]: "gene_id"})
rna_df = log2_cpm_normalize(rna_df)
rna_df['mean_gene_expression'] = rna_df.iloc[:, 1:].mean(axis=1)
rna_df = rna_df[['gene_id', 'mean_gene_expression']]

print("\tATACseq dataset")
atac_df = log2_cpm_normalize(load_atac_dataset(atac_data_file))
atac_df['mean_peak_accessibility'] = atac_df.iloc[:, 1:].mean(axis=1)
atac_df = atac_df[['peak_id', 'mean_peak_accessibility']]
print("Done!")
print("\n---------------------------\n")

print("Merging DataFrames")
print("\tMerging the correlation and cicero methods for peak to target gene")
peak_to_tg_merged_df = pd.merge(peak_corr_df, cicero_df, on=["peak_id", "gene_id"], how="outer")

print("\tMerging the peak to target gene scores with the sliding window TF to peak scores")
# For the sliding window genes, change their name to "source_id" to represent that these genes are TFs
peak_to_tg_merged_df = peak_to_tg_merged_df.rename(columns={"gene_id": "source_id"})
binding_scores_merged_df = pd.merge(peak_to_tg_merged_df, sliding_window_df, on=["peak_id"], how="outer")
print(binding_scores_merged_df.head())

print("\tMerging in the ATACseq peak accessibility values")
expr_atac_df = pd.merge(atac_df, binding_scores_merged_df, on="peak_id", how="left")

print("\tMerging in the RNAseq gene expression values")
# Add the RNA-seq gene expression for the target genes
target_merged_df = pd.merge(rna_df, expr_atac_df, on="gene_id", how="left")
target_merged_df = target_merged_df.rename(columns={
    "gene_id" : "target_id",
    "mean_gene_expression": "mean_TG_expression"
    })
print(target_merged_df.head())

# Add the RNA-seq gene expression for the transcription factor genes
source_merged_df = pd.merge(rna_df, target_merged_df, left_on="gene_id", right_on="source_id", how="left")
source_merged_df = source_merged_df.rename(columns={
    "gene_id": "source_id",
    "mean_gene_expression": "mean_TF_expression"
    })
print(source_merged_df.head())

# Remove any rows that are missing a peak, TF, or TG
source_merged_df = source_merged_df.dropna(subset=["peak_id", "gene_id", "source_id"])
print("Done!")
print("\n---------------------------\n")

print("Minmax normalizing all data columns to be between 0-1")
numeric_cols = source_merged_df.select_dtypes(include=np.number).columns.tolist()
full_merged_df_norm = source_merged_df[numeric_cols].apply(lambda x: minmax_normalize_column(x),axis=0)
full_merged_df_norm[["peak_id", "gene_id", "source_id"]] = source_merged_df[["peak_id", "gene_id", "source_id"]]
print(full_merged_df_norm.head())

plot_column_histograms(full_merged_df_norm, output_dir)

full_merged_df_norm.to_csv(f'{output_dir}/merged_df.csv', sep="\t", header=True, index=None)