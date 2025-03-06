import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mESC_rna_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC/filtered_L2_E7.5_rep1/mESC_filtered_L2_E7.5_rep1_RNA.csv"
macrophage_rna_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/macrophage/macrophage_buffer1_filtered/macrophage_buffer1_filtered_RNA.csv"
K562_rna_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/K562/K562_human_filtered/K562_human_filtered_RNA.csv"

mESC_atac_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC/filtered_L2_E7.5_rep1/mESC_filtered_L2_E7.5_rep1_ATAC.csv"
macrophage_atac_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/macrophage/macrophage_buffer1_filtered/macrophage_buffer1_filtered_ATAC.csv"
K562_atac_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/K562/K562_human_filtered/K562_human_filtered_ATAC.csv"

# RNA datasets
print("Loading in RNA-seq datasets")
mesc_rna_df = pd.read_csv(mESC_rna_data_file, sep=",", header=0, index_col=0)
macrophage_rna_df = pd.read_csv(macrophage_rna_data_file, sep=",", header=0, index_col=0)
K562_rna_df = pd.read_csv(K562_rna_data_file, sep=",", header=0, index_col=0)

# ATAC datasets
print("Loading in ATAC-seq datasets")
mesc_atac_df = pd.read_csv(mESC_atac_data_file, sep=",", header=0, index_col=0)
macrophage_atac_df = pd.read_csv(macrophage_atac_data_file, sep=",", header=0, index_col=0)
K562_atac_df = pd.read_csv(K562_atac_data_file, sep=",", header=0, index_col=0)

def remove_outliers(series):
    """
    Remove outliers from a pandas Series using the IQR method.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series >= lower_bound) & (series <= upper_bound)]

def create_expression_boxplots(dfs, titles):
    # Create a subplot for each dataframe
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))

    for ax, df, title in zip(axes, dfs, titles):
        # Flatten the dataframe values to a 1D Series
        flat_series = pd.Series(df.values.flatten())
        
        # Flatten all values into one array and plot a boxplot
        ax.boxplot(flat_series, patch_artist=True, log=True)
        ax.set_title(title)
        ax.set_ylabel("Expression")
        # Hide the x-tick label because it’s just one box per plot
        ax.set_xticklabels([""])

    plt.tight_layout()
    
    return fig

# Create subplots
def create_expression_histograms(dfs, titles):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

    for ax, df, title in zip(axes, dfs, titles):
        # Plot a histogram for the flattened data
        ax.hist(df.values.flatten(), bins=50, alpha=0.7, edgecolor='black', log=True)
        ax.set_title(title)
        ax.set_xlabel("Expression")
        ax.set_ylabel("Frequency")
        
    plt.tight_layout()
    
    return fig

# Create the figures from the RNA-seq data distributions
rna_dfs = [mesc_rna_df, macrophage_rna_df, K562_rna_df]
rna_titles = ["mESC RNA", "Macrophage RNA", "K562 RNA"]

print("Creating RNA-seq expression boxplots and histograms")
rna_expr_boxplot = create_expression_boxplots(rna_dfs, rna_titles)
rna_expr_hist = create_expression_histograms(rna_dfs, rna_titles)

rna_expr_boxplot.savefig("rna_data_dist_boxplot.png", dpi=500)
rna_expr_hist.savefig("rna_data_dist_hist.png", dpi=500)


# Create the figures from the ATAC-seq data distributions
atac_dfs = [mesc_atac_df, macrophage_atac_df, K562_atac_df]
atac_titles = ["mESC ATAC", "Macrophage ATAC", "K562 ATAC"]

print("Creating ATAC-seq expression boxplots and histograms")
atac_expr_boxplot = create_expression_boxplots(atac_dfs, atac_titles)
atac_expr_hist = create_expression_histograms(atac_dfs, atac_titles)

atac_expr_boxplot.savefig("atac_data_dist_boxplot.png", dpi=500)
atac_expr_hist.savefig("atac_data_dist_hist.png", dpi=500)

print("Done!")


