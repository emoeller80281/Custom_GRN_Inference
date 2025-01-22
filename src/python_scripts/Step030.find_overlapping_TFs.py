import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
from tqdm import tqdm
import logging
from typing import Any

def filter_overlapping_genes(TF_df: pd.DataFrame, RNA_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters TF-Target pairs where both genes exist in the RNA dataset.

    Args:
        TF_df (pd.DataFrame): DataFrame containing TF-Target pairs.
        RNA_df (pd.DataFrame): DataFrame containing RNA gene expression data.

    Returns:
        pd.DataFrame: Filtered DataFrame with overlapping genes.
    """
    genes: set[str] = set(RNA_df["Genes"])
    return TF_df[
        (TF_df["Source"].apply(lambda x: x in genes)) &
        (TF_df["Target"].apply(lambda x: x in genes))
    ]

def plot_histogram(data: pd.Series, title: str, xlabel: str, ylabel: str, save_path: str) -> None:
    """
    Plots and saves a histogram for the given data.

    Args:
        data (pd.Series): Data to plot.
        title (str): Title of the histogram.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, color="blue", alpha=0.7, log=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path, dpi=200)
    plt.close()

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing paths for RNA data and TF motif binding score files.
    """
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument("--rna_data_file", type=str, required=True, help="Path to the scRNA-seq data file")
    parser.add_argument("--tf_motif_binding_score_file", type=str, required=True, help="Path to the processed TF motif binding score")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    return parser.parse_args()

def main() -> None:
    """
    Main function to process TF-TG relationships and generate a weighted regulatory network.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    args: argparse.Namespace = parse_args()

    output_dir: str = args.output_dir

    # Load data
    RNA_dataset: pd.DataFrame = pd.read_csv(args.rna_data_file)
    TF_motif_binding_df: pd.DataFrame = pd.read_csv(args.tf_motif_binding_score_file, sep="\t")

    # Rename the first column as "Genes"
    RNA_dataset.rename(columns={RNA_dataset.columns[0]: "Genes"}, inplace=True)
    RNA_dataset["Genes"] = RNA_dataset["Genes"].apply(lambda x: x.upper())
    print(RNA_dataset.head())
    print(TF_motif_binding_df.head())

    # Filter overlapping genes
    overlapping_TF_motif_binding_df: pd.DataFrame = filter_overlapping_genes(TF_motif_binding_df, RNA_dataset)
    
    print(overlapping_TF_motif_binding_df.head())

    # Align RNA data
    aligned_RNA: pd.DataFrame = RNA_dataset[RNA_dataset["Genes"].isin(overlapping_TF_motif_binding_df["Source"])]

    # Transpose RNA dataset for easier access
    gene_expression_matrix: pd.DataFrame = RNA_dataset.set_index("Genes").T  # Rows = cells, Columns = genes

    # Filter genes present in both Source and Target
    filtered_genes: np.ndarray = pd.concat([TF_motif_binding_df['Source'], TF_motif_binding_df['Target']]).unique()
    filtered_expression_matrix: pd.DataFrame = gene_expression_matrix.loc[:, gene_expression_matrix.columns.intersection(filtered_genes)]

    # Compute gene-specific metrics across cells
    gene_stats: pd.DataFrame = filtered_expression_matrix.agg(['mean', 'std', 'max', 'sum']).T  # Rows = genes, Columns = stats

    # Map TF-specific metrics
    TF_motif_binding_df['TF_Mean_Expression'] = TF_motif_binding_df['Source'].map(gene_stats['mean'])
    TF_motif_binding_df['TF_Std_Expression'] = TF_motif_binding_df['Source'].map(gene_stats['std'])
    TF_motif_binding_df['TF_Max_Expression'] = TF_motif_binding_df['Source'].map(gene_stats['max'])

    # Map TG-specific metrics
    TF_motif_binding_df['TG_Mean_Expression'] = TF_motif_binding_df['Target'].map(gene_stats['mean'])
    TF_motif_binding_df['TG_Std_Expression'] = TF_motif_binding_df['Target'].map(gene_stats['std'])
    TF_motif_binding_df['TG_Max_Expression'] = TF_motif_binding_df['Target'].map(gene_stats['max'])

    # Drop NaN values
    TF_motif_binding_df.dropna(inplace=True)

    # Generate a weighted score
    TF_motif_binding_df['Weighted_Score'] = (
        TF_motif_binding_df['TF_Mean_Expression'] *
        TF_motif_binding_df['TG_Mean_Expression'] *
        TF_motif_binding_df['Motif_Score'] *
        TF_motif_binding_df['Peak Gene Score']
    )
    
    TF_motif_binding_df = TF_motif_binding_df[TF_motif_binding_df["Weighted_Score"] > 0]

    # Normalize the Weighted_Score
    TF_motif_binding_df['Normalized_Score'] = TF_motif_binding_df['Weighted_Score'] / TF_motif_binding_df['Weighted_Score'].max()
    
    cols_of_interest = ["Source", "Target", "Peak Gene Score", "TF_Mean_Expression", "TG_Mean_Expression", "Motif_Score", "Normalized_Score"]
    TF_motif_binding_df = TF_motif_binding_df[cols_of_interest]
    
    # Save results
    TF_motif_binding_df.to_csv(f'{output_dir}/inferred_grn.tsv', sep="\t", index=False)

    # Plot histogram of scores
    plot_histogram(
        data=np.log2(TF_motif_binding_df["Normalized_Score"]),
        title="Weighted TF-TG Binding Score Distribution",
        xlabel="Weighted Score",
        ylabel="Frequency",
        save_path=f"{output_dir}/weighted_score_histogram.png"
    )

if __name__ == "__main__":
    main()
