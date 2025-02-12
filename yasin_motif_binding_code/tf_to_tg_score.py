import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

tf_to_peak_score_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/tf_to_peak_binding_score.tsv"
peak_to_tg_score_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/peak_gene_associations.csv"
rna_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC_filtered_L2_E7.5_merged_RNA.csv"
    
# Read in the RNAseq data file and extract the gene names to find matching TFs
print("Reading and formatting expression data")
rna_data = pd.read_csv(rna_data_file, index_col=0, header=0)
rna_data = rna_data.rename(columns={rna_data.columns[0]: "gene"}).set_index("gene")
# print(rna_data.shape)
rna_data["mean_expression"] = np.log2(rna_data.values.mean(axis=1))
rna_data["mean_expression"] = (rna_data["mean_expression"] - rna_data["mean_expression"].min()) / (rna_data["mean_expression"].max() - rna_data["mean_expression"].min()) # Normalize gene expression values
rna_data = rna_data.reset_index()
rna_data = rna_data[["gene", "mean_expression"]]
# print(rna_data.head())

print("Reading and formatting TF to peak binding scores")
tf_to_peak_score = pd.read_csv(tf_to_peak_score_file, sep="\t", header=0, index_col=None)
tf_to_peak_score = tf_to_peak_score.melt(id_vars="peak", var_name="gene", value_name="binding_score")
tf_to_peak_score["binding_score"] = tf_to_peak_score["binding_score"] / tf_to_peak_score["binding_score"].max() # Normalize binding score values
# print(tf_to_peak_score.head())
# print(tf_to_peak_score.shape)

print("Reading and formatting peak to TG scores")
peak_to_tg_score = pd.read_csv(peak_to_tg_score_file, header=0, index_col=None)

# Format the peaks to match the tf_to_peak_score dataframe
peak_to_tg_score = peak_to_tg_score[["peak", "gene", "score_normalized"]]
peak_to_tg_score["peak"] = peak_to_tg_score["peak"].str.replace("_", "-")
peak_to_tg_score["peak"] = peak_to_tg_score["peak"].str.replace("-", ":", 1)
peak_to_tg_score = peak_to_tg_score.rename(columns={"score_normalized": "peak_to_target_score"})
# print(peak_to_tg_score.head())
# print(peak_to_tg_score.shape)

print("Combining TF to peak binding scores with TF expression")
tf_to_peak_score_and_expr = pd.merge(tf_to_peak_score, rna_data, on="gene", how="inner").rename(columns={"mean_expression": "TF_expression", "gene": "Source"})
print("Combining peak to TG scores with TG expression")
peak_to_tg_score_and_expr = pd.merge(peak_to_tg_score, rna_data, on="gene", how="inner").rename(columns={"mean_expression": "TG_expression", "gene": "Target"})

print("Calculating final TF to TG score")
merged_peaks = pd.merge(tf_to_peak_score_and_expr, peak_to_tg_score_and_expr, on=["peak"], how="inner")

print("Plotting subscore histograms")
# Initialize a 3x3 charts
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

# Flatten the axes array (makes it easier to iterate over)
axes = axes.flatten()

# Loop through each column and plot a histogram
selected_cols = ["TF_expression", "binding_score", "peak_to_target_score", "TG_expression"]
for i, column in enumerate(selected_cols):
    
    # Add the histogram
    merged_peaks[column].hist(ax=axes[i], # Define the current ax
                    color='cornflowerblue', # Color of the bins
                    bins=25, # Number of bins
                    grid=False
                   )
    
    # Add title and axis label
    axes[i].set_title(f'{column} distribution') 
    axes[i].set_xlabel(column) 
    axes[i].set_ylabel('Frequency') 
    axes[i].set_xlim((0,1)) # Set the xlim between 0-1 as the data is normalized

plt.tight_layout()
plt.savefig("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/merged_peaks.png", dpi=500)

# Calculate the final score column
merged_peaks["Score"] = merged_peaks["binding_score"] * merged_peaks["peak_to_target_score"] * merged_peaks["TF_expression"] * merged_peaks["TG_expression"]
merged_peaks["Score"] = (merged_peaks["Score"] - merged_peaks["Score"].min()) / (merged_peaks["Score"].max() - merged_peaks["Score"].min()) # Normalize gene expression values

merged_peaks = merged_peaks[merged_peaks["Score"] != 0]

print("Plotting TF and TG expression to score scatterplot")
plt.figure(figsize=(8, 10))
plt.scatter(x=merged_peaks["TF_expression"], y=merged_peaks["TG_expression"], c=merged_peaks["Score"], cmap="coolwarm")
plt.title("Relationship between TF and TG expression", fontsize=18)
plt.xlabel("TF Expression", fontsize=16)
plt.ylabel("TG Expression", fontsize=16)
plt.colorbar()
plt.tight_layout()
plt.savefig("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/tf_vs_tg_expression_score_scatter.png", dpi=500)

merged_peaks = merged_peaks[["Source", "Target", "Score"]]
# print(merged_peaks.head())
# print(merged_peaks.shape)

print("Plotting the final TF-TG score histogram")
plt.figure(figsize=(8, 10))
plt.hist(np.log2(merged_peaks["Score"]), bins=25)
plt.title("TF-TG binding score", fontsize=18)
plt.xlabel("Score", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.tight_layout()
plt.savefig("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/tf_to_tg_binding_score_hist.png", dpi=500)
plt.close()



merged_peaks.to_csv("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/tf_to_tg_inferred_network.tsv", sep="\t", header=True, index=False)