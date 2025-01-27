import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
from tqdm import tqdm
import logging
from typing import Any
from scipy.stats import spearmanr

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
    plt.hist(data, bins=150, color="blue", alpha=0.7, log=False)
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
    parser.add_argument("--fig_dir", type=str, required=True, help="Path to the figure directory")
    
    return parser.parse_args()

def main() -> None:
    """
    Main function to process TF-TG relationships and generate a weighted regulatory network.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    args: argparse.Namespace = parse_args()

    rna_data_file = args.rna_data_file
    tf_motif_binding_score_file = args.tf_motif_binding_score_file
    output_dir: str = args.output_dir
    fig_dir = args.fig_dir

    # Load data
    logging.info(f'Loading scRNAseq dataset')
    RNA_dataset: pd.DataFrame = pd.read_csv(rna_data_file)
    logging.info(f'Loading TF motif binding dataset')
    TF_motif_binding_df: pd.DataFrame = pd.read_csv(tf_motif_binding_score_file, sep="\t")

    # Rename the first column as "Genes"
    RNA_dataset.rename(columns={RNA_dataset.columns[0]: "Genes"}, inplace=True)
    RNA_dataset["Genes"] = RNA_dataset["Genes"].apply(lambda x: x.upper())

    # Transpose RNA dataset for easier access
    gene_expression_matrix: pd.DataFrame = RNA_dataset.set_index("Genes").T  # Rows = cells, Columns = genes
    
    # Filter genes present in both Source and Target
    filtered_genes: np.ndarray = pd.concat([TF_motif_binding_df['Source'], TF_motif_binding_df['Target']]).unique()
    filtered_expression_matrix: pd.DataFrame = gene_expression_matrix.loc[:, gene_expression_matrix.columns.intersection(filtered_genes)]

    # logging.info(f'Calculating TF-TG expression Spearman correlation')
    
    # # Create a dictionary of filtered expression data for faster access
    # gene_expression_dict = filtered_expression_matrix.to_dict(orient="list")

    # # Prepare results as a DataFrame
    # def compute_spearman(pair):
    #     source_gene = pair['Source']
    #     target_gene = pair['Target']

    #     if source_gene in gene_expression_dict and target_gene in gene_expression_dict:
    #         source_expression = gene_expression_dict[source_gene]
    #         target_expression = gene_expression_dict[target_gene]
    #         # Compute Spearman correlation
    #         correlation, _ = spearmanr(source_expression, target_expression)
    #         return correlation
    #     return np.nan

    # TF_motif_binding_df['Correlation'] = TF_motif_binding_df[['Source', 'Target']].apply(
    #     compute_spearman, axis=1
    # )
    
    # # Drop NaN values to keep only valid pairs
    # TF_motif_binding_df.dropna(subset=['Correlation'], inplace=True)
    # logging.info(f'\tDone!')

    logging.info(f'Calculating summary statistics')
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
    logging.info(f'\tDone!')
    
    logging.info(f'Generating weighted score')
    # Generate a weighted score
    TF_motif_binding_df['Weighted_Score'] = (
        TF_motif_binding_df['TF_Mean_Expression'] *
        TF_motif_binding_df['TG_Mean_Expression'] *
        TF_motif_binding_df['TF_TG_Motif_Binding_Score']
    ) # Add a Spearman or Pearson correlation?
    
    TF_motif_binding_df = TF_motif_binding_df[TF_motif_binding_df["Weighted_Score"] > 0]

    # Normalize the Weighted_Score
    TF_motif_binding_df['Normalized_Score'] = TF_motif_binding_df['Weighted_Score'] / TF_motif_binding_df['Weighted_Score'].max()
    
    cols_of_interest = ["Source", "Target", "TF_Mean_Expression", "TG_Mean_Expression", "TF_TG_Motif_Binding_Score","Normalized_Score"]
    TF_motif_binding_df = TF_motif_binding_df[cols_of_interest]
    
    logging.info(f'Writing final inferred GRN to the output directory as "inferred_grn.tsv"')
    # Save results
    TF_motif_binding_df.to_csv(f'{output_dir}/inferred_grn.tsv', sep="\t", index=False)

    # Plot histogram of scores
    logging.info(f'Plotting normalized log2 TF-TG binding score histogram')
    plot_histogram(
        data=np.log2(TF_motif_binding_df["Normalized_Score"]),
        title="Normalized log2 TF-TG Binding Score Distribution",
        xlabel="Normalized log2 Score",
        ylabel="Frequency",
        save_path=f"{fig_dir}/weighted_score_histogram.png"
    )

if __name__ == "__main__":
    main()
