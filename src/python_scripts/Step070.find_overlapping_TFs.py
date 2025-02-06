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
    
    # Calculate the normalized log2 counts/million for each gene in each cell
    RNA_dataset = RNA_dataset.astype({col: float for col in RNA_dataset.columns[1:]})
    column_sum = np.array(RNA_dataset.iloc[:, 1:].sum(axis=1, numeric_only=True))
    expression_matrix = RNA_dataset.iloc[:, 1:].values
    rna_cpm = np.log2(((expression_matrix.T / column_sum).T * 1e6) + 1)
    rna_cpm = rna_cpm / np.max(rna_cpm, axis=0)
    rna_cpm_df = pd.DataFrame(rna_cpm, index=RNA_dataset.index, columns=RNA_dataset.columns[1:])
    
    RNA_dataset.iloc[:, 1:] = rna_cpm_df
        
    logging.info(RNA_dataset.head())

    # Transpose RNA dataset for easier access
    gene_expression_df: pd.DataFrame = RNA_dataset.set_index("Genes").T  # Rows = cells, Columns = genes
    
    # Find the unique "Source" and "Target" genes in the TF motif binding dataframe
    filtered_genes: np.ndarray = pd.concat([TF_motif_binding_df['Source'], TF_motif_binding_df['Target']]).unique()
    
    # Filter the scRNAseq expression dataframe to only contain genes in the TF motif binding dataframe
    filtered_expression_df: pd.DataFrame = gene_expression_df.loc[:, gene_expression_df.columns.intersection(filtered_genes)]

    def calculate_cell_population_grn(filtered_expression_df, TF_motif_binding_df):
        logging.info(f'Calculating summary statistics')
        # Compute gene-specific metrics across cells
        gene_stats: pd.DataFrame = filtered_expression_df.agg(['mean', 'std', 'max', 'sum']).T  # Rows = genes, Columns = stats

        # Map TF-specific metrics
        TF_motif_binding_df['TF_Mean_Expression'] = TF_motif_binding_df['Source'].map(gene_stats['mean'])

        # Map TG-specific metrics
        TF_motif_binding_df['TG_Mean_Expression'] = TF_motif_binding_df['Target'].map(gene_stats['mean'])

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
    
    def calculate_cell_level_grn(filtered_expression_df, TF_motif_binding_df):
        # Set display option to show all columns
        pd.set_option('display.max_columns', 5)
        
        # Transpose to get genes as index and cells as columns
        filtered_expression_df = filtered_expression_df.T

        # Extract gene names (index) and reset index to have cell barcodes as a column
        filtered_expression_df = filtered_expression_df.reset_index()

        # Initialize a list to store results for all cells
        cell_results = []

        # Loop over each cell in filtered_expression_df (excluding "Genes" column)
        print('Processing cells')
        for i, cell in enumerate(filtered_expression_df.columns[1:200]):  
            # print(f"Processing cell: {i+1}", flush=True)

            # Get the expression data for the current cell
            first_cell_expression = filtered_expression_df[["Genes", cell]].rename(columns={cell: "Expression"})

            # Rename for merging
            first_cell_expression_TF = first_cell_expression.rename(columns={"Genes": "Source", "Expression": "TF_expression"})
            first_cell_expression_TG = first_cell_expression.rename(columns={"Genes": "Target", "Expression": "TG_expression"})

            # Merge to get TF expression
            cell_TF_motif_binding_df = TF_motif_binding_df.merge(first_cell_expression_TF, on="Source", how="left")

            # Merge to get TG expression
            cell_TF_motif_binding_df = cell_TF_motif_binding_df.merge(first_cell_expression_TG, on="Target", how="left")

            cell_TF_motif_binding_df[cell] = (
                cell_TF_motif_binding_df["TF_expression"].astype(np.float16) * 
                cell_TF_motif_binding_df["TG_expression"].astype(np.float16) * 
                cell_TF_motif_binding_df["TF_TG_Motif_Binding_Score"].astype(np.float16)
            )
            
            cell_TF_motif_binding_df = cell_TF_motif_binding_df[["Source", "Target", cell]]
            # print(cell_TF_motif_binding_df, flush=True)
            
            # Store results in a list instead of merging now
            cell_results.append(cell_TF_motif_binding_df)
        print('\tDone! Concatenating cell-level GRNs')
        
        # Efficiently merge everything at the end using `pd.concat()`
        final_df = pd.concat(cell_results, ignore_index=True)

        # Pivot table to get a wide format (cells as columns)
        final_df = final_df.pivot_table(index=["Source", "Target"], columns=[], values=list(filtered_expression_df.columns[1:100]), aggfunc="first")

        # Reset index so that "Source" and "Target" remain as regular columns
        final_df.reset_index(inplace=True)
        
        final_df = final_df.fillna(1e-6)
        
        # Drop "Source" and "Target" columns to keep only the score values
        score_columns = final_df.columns[2:]

        # Avoid divide-by-zero errors
        final_df[score_columns] = final_df[score_columns].replace(0, 1e-6)  # Replace zeros with small value

        # Log2 transformation safely
        log_transformed = np.log2(final_df[score_columns])
        
        log_values_normalized = (log_transformed - log_transformed.min().min()) / (log_transformed.max().max() - log_transformed.min().min())
        
        print(final_df.head())
        print(final_df.shape)

        # Plot overlapping histograms
        plt.figure(figsize=(10, 6))
        for col in score_columns:
            # Drop NaNs and explicitly remove values equal to 0
            filtered_values = log_values_normalized[col].replace(0, np.nan).dropna()
            
            # Plot only if there are remaining values
            if not filtered_values.empty:
                plt.hist(filtered_values, bins=50, alpha=0.1, density=True, label=col)
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.title("Overlapping Histograms of TF-TG Interaction Scores")
        plt.savefig(f'{output_dir}/cell_level_overlapping_histogram.png', dpi=200)
        
        final_df.to_csv(f"{output_dir}/cell_level_inferred_grn.csv", sep='\t', index=False)

    
    # calculate_cell_population_grn(filtered_expression_df, TF_motif_binding_df)
    calculate_cell_level_grn(filtered_expression_df, TF_motif_binding_df)
    
    

if __name__ == "__main__":
    main()
