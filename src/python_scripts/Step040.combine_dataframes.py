import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging



def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing paths for input and output files and CPU count.
    """
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument(
        "--atac_data_file",
        type=str,
        required=True,
        help="Path to the scATAC-seq dataset"
    )
    parser.add_argument(
        "--rna_data_file",
        type=str,
        required=True,
        help="Path to the scRNA-seq dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
    )

    args: argparse.Namespace = parser.parse_args()
    return args

def minmax_normalize_column(column: pd.DataFrame):
    return (column - column.min()) / (column.max() - column.min())

def plot_column_histograms(df, fig_dir):
    # Create a figure and axes with a suitable size
    plt.figure(figsize=(15, 8))
    
    # Select only the numerical columns (those with numeric dtype)
    cols = df.select_dtypes(include=[np.number]).columns

    # Loop through each feature and create a subplot
    for i, col in enumerate(cols, 1):
        plt.subplot(2, 4, i)  # 2 rows, 4 columns, index = i
        plt.hist(df[col], bins=50, alpha=0.7, edgecolor='black')
        plt.title(f"{col} distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/column_histograms.png', dpi=300)
    plt.close()

def main(atac_data_file, rna_data_file, output_dir, fig_dir):
    logging.info("Loading in the DataFrames")
    logging.info("\tCorrelation peak to TG DataFrame")
    peak_corr_df = pd.read_csv(f'{output_dir}/peak_to_gene_correlation.csv', sep="\t", header=0)

    logging.info("\tCicero peak to TG DataFrame")
    cicero_df = pd.read_csv(f'{output_dir}/cicero_peak_to_tg_scores.csv', sep="\t", header=0)

    logging.info("\tSliding Window peak to TG DataFrame")
    sliding_window_df = pd.read_csv(f'{output_dir}/sliding_window_tf_to_peak_score.tsv', sep="\t", header=0)

    logging.info("\tHomer TF to peak DataFrame")
    homer_df = pd.read_csv(f'{output_dir}/homer_tf_to_peak.tsv', sep="\t", header=0)

    logging.info("\tRNAseq dataset")
    rna_df = pd.read_csv(rna_data_file, header=0)
    rna_df['mean_gene_expression'] = rna_df.iloc[:, 1:].mean(axis=1)
    rna_df = rna_df[['gene_id', 'mean_gene_expression']]

    logging.info("\tATACseq dataset")
    atac_df = pd.read_csv(atac_data_file, header=0)
    atac_df['mean_peak_accessibility'] = atac_df.iloc[:, 1:].mean(axis=1)
    atac_df = atac_df[['peak_id', 'mean_peak_accessibility']]
    logging.info("Done!")
    logging.info("\n---------------------------\n")

    logging.info(" ============== Merging DataFrames ==============")
    logging.info("Combining the sliding window and Homer TF to peak binding scores")
    tf_to_peak_df = pd.merge(sliding_window_df, homer_df, on=["peak_id", "source_id"], how="outer")
    logging.info("tf_to_peak_df")
    logging.info(tf_to_peak_df.head())
    logging.info(tf_to_peak_df.columns)
    logging.info("\n---------------------------\n")

    logging.info("Adding RNA expression info to the TF to peak binding DataFrame")
    tf_expr_to_peak_df = pd.merge(rna_df, tf_to_peak_df, left_on="gene_id", right_on="source_id", how="outer").drop("gene_id", axis=1)
    tf_expr_to_peak_df = tf_expr_to_peak_df.rename(columns={"mean_gene_expression": "mean_TF_expression"})
    logging.info("tf_expr_to_peak_df")
    logging.info(tf_expr_to_peak_df.head())
    logging.info(tf_expr_to_peak_df.columns)
    logging.info("\n---------------------------\n")

    logging.info("Merging the correlation and cicero methods for peak to target gene")
    peak_to_tg_df = pd.merge(peak_corr_df, cicero_df, on=["peak_id", "target_id"], how="outer")
    logging.info("peak_to_tg_df")
    logging.info(peak_to_tg_df.head())
    logging.info(peak_to_tg_df.columns)
    logging.info("\n---------------------------\n")

    logging.info("Adding RNA expression info to the peak to TF DataFrame")
    peak_to_tg_expr_df = pd.merge(rna_df, peak_to_tg_df, left_on="gene_id", right_on="target_id", how="left").drop("gene_id", axis=1)
    peak_to_tg_expr_df = peak_to_tg_expr_df.rename(columns={"mean_gene_expression": "mean_TG_expression"})
    logging.info("peak_to_tg_expr_df")
    logging.info(peak_to_tg_expr_df.head())
    logging.info(peak_to_tg_expr_df.columns)
    logging.info("\n---------------------------\n")

    logging.info("Merging the peak to target gene scores with the sliding window TF to peak scores")
    # For the sliding window genes, change their name to "source_id" to represent that these genes are TFs
    tf_to_tg_score_df = pd.merge(tf_expr_to_peak_df, peak_to_tg_expr_df, on=["peak_id"], how="outer")
    logging.info("tf_to_tg_score_df")
    logging.info(tf_to_tg_score_df.head())
    logging.info(tf_to_tg_score_df.columns)
    logging.info("\n---------------------------\n")

    logging.info("Merging in the ATACseq peak accessibility values")
    final_df = pd.merge(atac_df, tf_to_tg_score_df, on="peak_id", how="left")

    # Drop columns that dont have all three of the peak, target, and source names
    final_df = final_df.dropna(subset=[
        "peak_id",
        "target_id",
        "source_id",
        ])
    logging.info("expr_atac_df")
    logging.info(final_df.head())
    logging.info(final_df.columns)
    logging.info("\n---------------------------\n")

    logging.info("Minmax normalizing all data columns to be between 0-1")
    numeric_cols = final_df.select_dtypes(include=np.number).columns.tolist()
    full_merged_df_norm: pd.DataFrame = final_df[numeric_cols].apply(lambda x: minmax_normalize_column(x),axis=0)
    full_merged_df_norm[["peak_id", "target_id", "source_id"]] = final_df[["peak_id", "target_id", "source_id"]]

    # Replace NaN values with 0 for the scores
    full_merged_df_norm = full_merged_df_norm.fillna(0)

    # Set the desired column order
    column_order = [
        "source_id",
        "target_id",
        "peak_id",
        "mean_TF_expression",
        "mean_TG_expression",
        "mean_peak_accessibility",
        "cicero_score",
        "enh_score",
        "TSS_dist",
        "correlation",
        "sliding_window_score",
        "homer_binding_score"
    ]
    full_merged_df_norm = full_merged_df_norm[column_order]
    logging.info(full_merged_df_norm.head())
    logging.info(full_merged_df_norm.columns)

    logging.info("Writing the final dataframe as 'inferred_network_score_df.csv'")
    full_merged_df_norm.to_csv(f'{output_dir}/inferred_network_score_df.csv', sep="\t", header=True, index=None)

    plot_column_histograms(full_merged_df_norm, fig_dir)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Parse command-line arguments
    args: argparse.Namespace = parse_args()
    atac_data_file: str = args.atac_data_file
    rna_data_file: str = args.rna_data_file
    output_dir: str = args.output_dir
    fig_dir: str = args.fig_dir

    main(atac_data_file, rna_data_file, output_dir, fig_dir)