import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
from tqdm import tqdm
import logging

def filter_overlapping_genes(TF_df, RNA_df):
    genes = set(RNA_df["Genes"])
    return TF_df[
        (TF_df["Source"].apply(lambda x: x in genes)) &
        (TF_df["Target"].apply(lambda x: x in genes))
    ]

def plot_histogram(data, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, color="blue", alpha=0.7, log=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path, dpi=200)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument("--rna_data_file", type=str, required=True, help="Path to the scRNA-seq data file")
    parser.add_argument("--tf_motif_binding_score_file", type=str, required=True, help="Path to the processed TF motif binding score")
    return parser.parse_args()

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    args = parse_args()

    # Load data
    RNA_dataset = pd.read_csv(args.rna_data_file)
    TF_motif_binding_df = pd.read_csv(args.tf_motif_binding_score_file, sep="\t")

    # Rename the first column as "Genes"
    RNA_dataset.rename(columns={RNA_dataset.columns[0]: "Genes"}, inplace=True)

    # Filter overlapping genes
    overlapping_TF_motif_binding_df = filter_overlapping_genes(TF_motif_binding_df, RNA_dataset)

    # Align RNA data
    aligned_RNA = RNA_dataset[RNA_dataset["Genes"].isin(overlapping_TF_motif_binding_df["Source"])]

    # Transpose RNA dataset for easier access
    gene_expression_matrix = RNA_dataset.set_index("Genes").T  # Rows = cells, Columns = genes

    # Filter genes present in both Source and Target
    filtered_genes = pd.concat([TF_motif_binding_df['Source'], TF_motif_binding_df['Target']]).unique()
    filtered_expression_matrix = gene_expression_matrix.loc[:, gene_expression_matrix.columns.intersection(filtered_genes)]

    # Compute gene-specific metrics across cells
    gene_stats = filtered_expression_matrix.agg(['mean', 'std', 'max', 'sum']).T  # Rows = genes, Columns = stats

    # Map TF-specific metrics
    TF_motif_binding_df['TF_Mean_Expression'] = TF_motif_binding_df['Source'].map(gene_stats['mean'])
    TF_motif_binding_df['TF_Std_Expression'] = TF_motif_binding_df['Source'].map(gene_stats['std'])
    TF_motif_binding_df['TF_Max_Expression'] = TF_motif_binding_df['Source'].map(gene_stats['max'])

    # Map TG-specific metrics
    TF_motif_binding_df['TG_Mean_Expression'] = TF_motif_binding_df['Target'].map(gene_stats['mean'])
    TF_motif_binding_df['TG_Std_Expression'] = TF_motif_binding_df['Target'].map(gene_stats['std'])
    TF_motif_binding_df['TG_Max_Expression'] = TF_motif_binding_df['Target'].map(gene_stats['max'])

    # Fill NaN values (e.g., for genes not present in RNA dataset)
    # TF_motif_binding_df.fillna({'TF_Mean_Expression': 0, 'TG_Mean_Expression': 0, 'TG_Std_Expression': 0, 'TG_Max_Expression': 0}, inplace=True)
    TF_motif_binding_df = TF_motif_binding_df.dropna()

    # Generate a weighted score, taking into account the TF-TG motif score, TF mean expression, and TG mean expression
    TF_motif_binding_df['Weighted_Score'] = (
        TF_motif_binding_df['TF_Mean_Expression'] *
        TF_motif_binding_df['TG_Mean_Expression'] *
        TF_motif_binding_df['Motif_Score']
    )

    # Normalize the Weighted_Score
    TF_motif_binding_df['Normalized_Score'] = TF_motif_binding_df['Weighted_Score'] / TF_motif_binding_df['Weighted_Score'].max()

    # Save results (optional)
    # TF_motif_binding_df.to_csv('/path/to/weighted_scores_with_nuance.tsv', sep='\t', index=False)

    # Preview results
    print(TF_motif_binding_df[['Source', 'Target', 'TF_Mean_Expression', 'TG_Mean_Expression', 'Weighted_Score', 'Normalized_Score']].head())

    # Save results
    TF_motif_binding_df.to_csv("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/inferred_grn.tsv", sep="\t", index=False)

    # Plot histogram of scores
    plot_histogram(
        data=TF_motif_binding_df["Weighted_Score"],
        title="Weighted TF-TG Binding Score Distribution",
        xlabel="Weighted Score",
        ylabel="Frequency",
        save_path="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/weighted_score_histogram.png"
    )

if __name__ == "__main__":
    main()