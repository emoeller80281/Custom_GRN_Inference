import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RNA_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/macrophage_buffer1_filtered_RNA.csv"
TF_motif_binding_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/total_motif_regulatory_scores.tsv"

print(f'Loading datasets')
RNA_dataset = pd.read_csv(RNA_file)
TF_motif_binding_df = pd.read_csv(TF_motif_binding_file, header=0, sep="\t", index_col=None)

# Find overlapping TFs
RNA_dataset = RNA_dataset.rename(columns={RNA_dataset.columns[0]: "Genes"})

print(f'Size of unfiltered RNA dataset:')
print(f'\tGenes: {RNA_dataset.shape[1]}')
print(f'\tCells: {RNA_dataset.shape[0]}')

print(f'\nSize of unfiltered TF motif dataset:')
print(f'\tTFs: {len(set(TF_motif_binding_df["Source"]))}')
print(f'\tTGs: {len(TF_motif_binding_df["Target"])}')

genes = set(RNA_dataset["Genes"])

with open(f'RNA_genes.txt', 'w') as gene_file:
    gene_file.write('Genes\n')
    for gene in genes:
        gene_file.write(f'{gene}\n')


tf_motif_dataset_genes = TF_motif_binding_df["Source"].unique()
# Print the unique genes
print(f'Unique genes: {tf_motif_dataset_genes}')

# Print the number of unique genes
print(f'tf_motif_dataset_genes: {len(tf_motif_dataset_genes)}')

# Open the file and write
with open('TF_motif_genes2.txt', 'w') as motif_file:
    motif_file.write('Genes\n')  # Header
    count = 0  # Track how many lines are written
    for gene in tf_motif_dataset_genes:
        count += 1
        motif_file.write(f'{gene}\t{count}\n')
        if count < 10:  # Debugging: Print first 10 writes
            print(f'Writing gene: {gene}')
    motif_file.flush()
    print(f'Total lines written: {count}')
        
TF_motif_binding_df["Source"].to_csv("TF_motif_genes.txt", index=False)

overlapping_TF_motif_binding_df = TF_motif_binding_df[
    (TF_motif_binding_df["Source"].apply(lambda x: x in genes)) &
    (TF_motif_binding_df["Target"].apply(lambda x: x in genes))
    ]

# Align RNA_dataset with overlapping_TF_motif_binding_df["Source"]
aligned_RNA = RNA_dataset[RNA_dataset["Genes"].isin(overlapping_TF_motif_binding_df["Source"])]

RNA_expression_matrix = RNA_dataset.iloc[1:, 1:].values

print(f'Mean RNA expression value = {np.mean(RNA_expression_matrix)}')

expr_above_zero = RNA_expression_matrix[RNA_expression_matrix > 0]

threshold = np.percentile(expr_above_zero, 10)  # Set to the 10th percentile
print(f"10th Percentile Threshold = {threshold}")

row_sums = RNA_dataset.iloc[1:, 1:].sum(axis=1)  # Compute row sums for filtering
filtered_genes = RNA_dataset.iloc[1:, :][row_sums > threshold]  # Use the same row index

plt.figure(figsize=(10, 6))

filtered_expr_values = filtered_genes.iloc[:, 1:].values.flatten()
filtered_expr_values = filtered_expr_values[filtered_expr_values > 0]

plt.hist(np.log10(filtered_expr_values), bins=50, color="blue", alpha=0.7, log=True)
plt.title("Distribution of Expression Values")
plt.xlabel("Log10 Expression Value")
plt.ylabel("Log10 Frequency")
plt.savefig("RNA_expression_distribution.png", dpi=200)


# Align overlapping_TF_motif_binding_df to the filtered RNA dataset
aligned_TF_binding = overlapping_TF_motif_binding_df[
    overlapping_TF_motif_binding_df["Source"].isin(aligned_RNA["Genes"])
]

# Find common indices
common_indices = aligned_RNA.index.intersection(aligned_TF_binding.index)

# Filter both DataFrames
aligned_RNA = aligned_RNA.loc[common_indices].set_index("Genes", drop=True)
aligned_TF_binding = aligned_TF_binding.loc[common_indices]

# Extract expression matrix (genes x cells)
expression_matrix = aligned_RNA.iloc[:, 1:].values

# Extract motif scores
motif_scores = aligned_TF_binding["Score"].values.reshape(-1, 1)

# Perform element-wise multiplication
weighted_expression = aligned_RNA.values * motif_scores

# Create a DataFrame for the weighted expression
weighted_expression_df = pd.DataFrame(
    weighted_expression,
    index=aligned_RNA.index,
    columns=aligned_RNA.columns
)

print(f'Number of TFs = {weighted_expression_df.shape[0]}')
print(f'Number of TGs = {weighted_expression_df.shape[1]}')

plt.figure(figsize=(15, 10))
sns.heatmap(weighted_expression_df, cmap="viridis", cbar=True)
plt.title("Cell Expressions Heatmap")
plt.xlabel("Cells")
plt.ylabel("Genes")
plt.savefig("TF_score_heatmap.png", dpi=200)

all_values = weighted_expression_df.values.flatten()

nonzero_values = all_values[all_values > 0]
log2_values = np.log2(nonzero_values)

plt.figure(figsize=(10, 6))
plt.hist(log2_values, bins=50, color="blue", alpha=0.7)
plt.title("Distribution of Expression Values")
plt.xlabel("Expression Value")
plt.ylabel("Frequency")
plt.savefig("TF_score_distribution.png", dpi=200)

# print(f'Mean gene score = {weighted_expression_df.mean()}')
