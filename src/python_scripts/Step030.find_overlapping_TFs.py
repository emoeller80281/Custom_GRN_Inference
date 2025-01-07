import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import argparse

# Set font to Arial and adjust font sizes
rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,  # General font size
    'axes.titlesize': 18,  # Title font size
    'axes.labelsize': 16,  # Axis label font size
    'xtick.labelsize': 14,  # X-axis tick label size
    'ytick.labelsize': 14,  # Y-axis tick label size
    'legend.fontsize': 14  # Legend font size
})

def parse_args():
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument(
        "--rna_data_file",
        type=str,
        required=True,
        help="Path to the scRNA-seq data file"
    )

    parser.add_argument(
        "--tf_motif_binding_score_file",
        type=str,
        required=True,
        help="Path to the processed TF motif binding score"
    )
    
    args = parser.parse_args()

    return args 

args = parse_args()
RNA_file = args.rna_data_file
TF_motif_binding_score_file = args.tf_motif_binding_score_file

print(f'Loading datasets')
RNA_dataset = pd.read_csv(RNA_file)
TF_motif_binding_df = pd.read_csv(TF_motif_binding_score_file, header=0, sep="\t", index_col=None)

# Find overlapping TFs
RNA_dataset = RNA_dataset.rename(columns={RNA_dataset.columns[0]: "Genes"})

print(f'Size of unfiltered RNA dataset:')
print(f'\tGenes: {RNA_dataset.shape[1]}')
print(f'\tCells: {RNA_dataset.shape[0]}')

print(f'\nSize of unfiltered TF motif dataset:')
print(f'\tTFs: {len(set(TF_motif_binding_df["Source"]))}')
print(f'\tTGs: {len(TF_motif_binding_df["Target"])}')

genes = set(RNA_dataset["Genes"])

tf_motif_dataset_genes = TF_motif_binding_df["Source"].unique()
# Print the unique genes
# print(f'Unique genes: {tf_motif_dataset_genes}')

# Print the number of unique genes
print(f'tf_motif_dataset_genes: {len(tf_motif_dataset_genes)}')
        
TF_motif_binding_df["Source"].to_csv("TF_motif_genes.txt", index=False)

# Remove any rows from the TF motif binding df that have genes not in scRNAseq dataset
overlapping_TF_motif_binding_df = TF_motif_binding_df[
    (TF_motif_binding_df["Source"].apply(lambda x: x in genes)) &
    (TF_motif_binding_df["Target"].apply(lambda x: x in genes))
    ]

# Align RNA_dataset with overlapping_TF_motif_binding_df["Source"]
aligned_RNA = RNA_dataset[RNA_dataset["Genes"].isin(overlapping_TF_motif_binding_df["Source"])]

RNA_expression_matrix = RNA_dataset.iloc[1:, 1:].values

# print(f'Mean RNA expression value = {np.mean(RNA_expression_matrix)}')

expr_above_zero = RNA_expression_matrix[RNA_expression_matrix > 0]

threshold = np.percentile(expr_above_zero, 10)  # Set to the 10th percentile
# print(f"10th Percentile Threshold = {threshold}")

row_sums = RNA_dataset.iloc[1:, 1:].sum(axis=1)  # Compute row sums for filtering
filtered_genes = RNA_dataset.iloc[1:, :][row_sums > threshold]  # Use the same row index

plt.figure(figsize=(10, 6))

filtered_expr_values = filtered_genes.iloc[:, 1:].values.flatten()
filtered_expr_values = filtered_expr_values[filtered_expr_values > 0]

# Plot a histogram of the filtered gene expression values
plt.hist(np.log10(filtered_expr_values), bins=50, color="blue", alpha=0.7, log=True)
plt.title("Distribution of Expression Values")
plt.xlabel("Log10 Expression Value")
plt.ylabel("Log10 Frequency")
plt.savefig("RNA_expression_distribution.png", dpi=200)


# Align overlapping_TF_motif_binding_df to the filtered RNA dataset
aligned_TF_binding = overlapping_TF_motif_binding_df[
    overlapping_TF_motif_binding_df["Source"].isin(aligned_RNA["Genes"])
]

# Find indices of the common genes between the RNAseq dataset and the TF motif binding 
common_indices = aligned_RNA.index.intersection(aligned_TF_binding.index)

# Filter both DataFrames
aligned_RNA = aligned_RNA.loc[common_indices].set_index("Genes", drop=False)
aligned_TF_binding = aligned_TF_binding.loc[common_indices]
print(aligned_TF_binding)
# print(aligned_RNA)

tf_set = set(aligned_TF_binding["Source"])
# print(f'TFs: {tf_set}')

# Ensure the first column is the gene name and set it as the index
aligned_RNA = aligned_RNA.set_index(aligned_RNA.columns[0])

# Normalize the formatting of gene names for consistency
aligned_RNA.index = aligned_RNA.index.str.upper().str.strip()
aligned_TF_binding["Source"] = aligned_TF_binding["Source"].str.upper().str.strip()

# Select the second column (expression values)
expression_col = aligned_RNA.columns[0]  # Assuming this is the expression column

# Map expression values from aligned_RNA to aligned_TF_binding
aligned_TF_binding["Expression"] = aligned_TF_binding["Source"].map(aligned_RNA[expression_col])

# Handle potential NaN values
aligned_TF_binding["Expression"].fillna(0, inplace=True)

# Multiply the Score by the mapped expression values
aligned_TF_binding["Weighted_Score"] = aligned_TF_binding["Score"] * aligned_TF_binding["Expression"]

# Display the resulting DataFrame
print(aligned_TF_binding)







# # Extract expression matrix (genes x cells)
# expression_matrix = aligned_RNA.iloc[:, 1:].values

# # Extract motif scores
# motif_scores = aligned_TF_binding["Score"].values.reshape(-1, 1)
# # print(motif_scores)

# # Perform element-wise multiplication
# tf_regulatory_score = aligned_RNA.values * motif_scores

# # Create a DataFrame for the weighted expression
# tf_regulatory_score_df = pd.DataFrame(
#     tf_regulatory_score,
#     index=aligned_RNA.index,
#     columns=aligned_RNA.columns
# )

# print(f'Number of TFs = {tf_regulatory_score_df.shape[0]}')
# print(f'Number of TGs = {tf_regulatory_score_df.shape[1]}')

# # print(tf_regulatory_score_df.head)

# # Create a heatmap of the TF regulatory score for each TF in each cell
# plt.figure(figsize=(15, 10))
# sns.heatmap(tf_regulatory_score_df, cmap="viridis", cbar=True)
# plt.title("Weighted TF Regulatory Score by Cell Heatmap")
# plt.xticks([])
# plt.xlabel("Cells")
# plt.ylabel("Genes")
# plt.savefig("TF_score_heatmap.png", dpi=200)

# # Plot a histogram of the weighted TF regulatory scores
# all_values = tf_regulatory_score_df.values.flatten()
# nonzero_values = all_values[all_values > 0]
# log2_values = np.log2(nonzero_values)

# plt.figure(figsize=(10, 6))
# plt.hist(log2_values, bins=50, color="blue", alpha=0.7)
# plt.title("Distribution of Weighted Expression Values")
# plt.xlabel("TF Regulatory Score")
# plt.ylabel("Frequency")
# plt.savefig("TF_score_distribution.png", dpi=200)