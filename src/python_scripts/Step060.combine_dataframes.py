import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import argparse
import logging
import os
from tqdm import tqdm

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
    parser.add_argument(
        "--inferred_grn_dir",
        type=str,
        required=True,
        help="Path to the output directory for the inferred GRNs"
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        required=True,
        help="Path to the figure directory for the sample"
    )
    parser.add_argument(
        "--subsample",
        type=str,
        required=True,
        help="Percent of rows by which to subsample the combined DataFrame"
    )

    args: argparse.Namespace = parser.parse_args()
    return args

def get_percentile_mask(column, lower=5, upper=95):
    col_clean = column.dropna()
    q_low = np.percentile(col_clean, lower)
    q_high = np.percentile(col_clean, upper)
    return (column > q_low) & (column < q_high)

def minmax_normalize_column(column: pd.DataFrame):
    return (column - column.min()) / (column.max() - column.min())

def plot_column_histograms(df, fig_dir, df_name="inferred_net"):
    # Create a figure and axes with a suitable size
    plt.figure(figsize=(15, 8))
    
    # Select only the numerical columns (those with numeric dtype)
    cols = df.select_dtypes(include=[np.number]).columns

    # Loop through each feature and create a subplot
    for i, col in enumerate(cols, 1):
        plt.subplot(3, 4, i)  # 3 rows, 4 columns, index = i
        plt.hist(df[col], bins=50, alpha=0.7, edgecolor='black')
        plt.title(f"{col} distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{df_name}_column_histograms.png', dpi=300)
    plt.close()

def write_csv_in_chunks(df, output_dir, filename):
    logging.info(f'Writing out CSV file to {filename} in 5% chunks')
    output_file = f'{output_dir}/{filename}'
    chunksize = int(math.ceil(0.05 * df.shape[0]))

    # Remove the output file if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Write the DataFrame in chunks
    for start in tqdm(range(0, len(df), chunksize), unit="chunk"):
        chunk = df.iloc[start:start + chunksize]
        if start == 0:
            # For the first chunk, write with header in write mode
            chunk.to_csv(output_file, mode='w', header=True, index=False)
        else:
            # For subsequent chunks, append without header
            chunk.to_csv(output_file, mode='a', header=False, index=False)

def merge_score_dataframes_slow(peak_corr_df, cicero_df, sliding_window_df, homer_df, rna_df, atac_df):
    logging.info("\tCombining the sliding window and Homer TF to peak binding scores")
    tf_to_peak_df = pd.merge(sliding_window_df, homer_df, on=["peak_id", "source_id"], how="outer")
    logging.debug("tf_to_peak_df")
    logging.debug(tf_to_peak_df.head())
    logging.debug(tf_to_peak_df.columns)
    logging.debug("\n---------------------------\n")

    logging.info("\t - Adding mean RNA expression to the TF to peak binding DataFrame")
    tf_expr_to_peak_df = pd.merge(rna_df, tf_to_peak_df, left_on="gene_id", right_on="source_id", how="outer").drop("gene_id", axis=1)
    tf_expr_to_peak_df = tf_expr_to_peak_df.rename(columns={"mean_gene_expression": "mean_TF_expression"})
    logging.debug("tf_expr_to_peak_df")
    logging.debug(tf_expr_to_peak_df.head())
    logging.debug(tf_expr_to_peak_df.columns)
    logging.debug("\n---------------------------\n")

    logging.info("\tMerging the correlation and cicero methods for peak to target gene")
    peak_to_tg_df = pd.merge(peak_corr_df, cicero_df, on=["peak_id", "target_id"], how="outer")
    logging.debug("peak_to_tg_df")
    logging.debug(peak_to_tg_df.head())
    logging.debug(peak_to_tg_df.columns)
    logging.debug("\n---------------------------\n")

    logging.info("\t - Adding mean RNA expression to the peak to TF DataFrame")
    peak_to_tg_expr_df = pd.merge(rna_df, peak_to_tg_df, left_on="gene_id", right_on="target_id", how="left").drop("gene_id", axis=1)
    peak_to_tg_expr_df = peak_to_tg_expr_df.rename(columns={"mean_gene_expression": "mean_TG_expression"})
    logging.debug("peak_to_tg_expr_df")
    logging.debug(peak_to_tg_expr_df.head())
    logging.debug(peak_to_tg_expr_df.columns)
    logging.debug("\n---------------------------\n")

    logging.info("\tMerging the peak to target gene scores with the sliding window TF to peak scores")
    # For the sliding window genes, change their name to "source_id" to represent that these genes are TFs
    tf_to_tg_score_df = pd.merge(tf_expr_to_peak_df, peak_to_tg_expr_df, on=["peak_id"], how="outer")
    logging.debug("tf_to_tg_score_df")
    logging.debug(tf_to_tg_score_df.head())
    logging.debug(tf_to_tg_score_df.columns)
    logging.debug("\n---------------------------\n")

    logging.info("\t - Adding the mean ATAC-seq peak accessibility values")
    final_df = pd.merge(atac_df, tf_to_tg_score_df, on="peak_id", how="left")
    
    return final_df

def merge_score_dataframes(peak_corr_df, cicero_df, sliding_window_df, homer_df, rna_df, atac_df):
    # Merge the correlation, cicero, sliding window, and homer dfs
    logging.info("Combining correlation, Cicero, sliding window, and Homer Scores")
    merge1 = (
        peak_corr_df
        .merge(cicero_df, on=["peak_id", "target_id"], how="outer")
        .merge(sliding_window_df, on="peak_id", how="outer")
        .merge(homer_df, on="peak_id", how="outer")
    )

    # Merge on RNA & ATAC means
    logging.info("Adding gene expression and peak accessibility")
    logging.info("  - Calculating mean Gene Expression")
    rna_means = rna_df.pipe(
        lambda df: df.assign(mean_gene_expression=df.iloc[:,1:].mean(axis=1))
    ).loc[:, ["gene_id", "mean_gene_expression"]]

    logging.info("  - Calculating mean peak accessibility")
    atac_means = atac_df.pipe(
        lambda df: df.assign(mean_peak_accessibility=df.iloc[:,1:].mean(axis=1))
    ).loc[:, ["peak_id", "mean_peak_accessibility"]]

    final_df = (
        merge1
        .merge(rna_means,  left_on="target_id", right_on="gene_id", how="left")
        .merge(atac_means, on="peak_id", how="left")
    )
    
    return final_df

def main():
    # Parse command-line arguments
    args: argparse.Namespace = parse_args()
    atac_data_file: str = args.atac_data_file
    rna_data_file: str = args.rna_data_file
    output_dir: str = args.output_dir
    inferred_grn_dir: str = args.inferred_grn_dir
    fig_dir: str = args.fig_dir
    subsample: str = args.subsample
    
    subsample = float(subsample)
    
    logging.info("Loading in the DataFrames")
    logging.info("\tCorrelation peak to TG DataFrame")
    peak_corr_df = pd.read_parquet(f'{output_dir}/peak_to_gene_correlation.parquet')

    logging.info("\tCicero peak to TG DataFrame")
    cicero_df = pd.read_parquet(f'{output_dir}/cicero_peak_to_tg_scores.parquet')

    logging.info("\tSliding Window peak to TG DataFrame")
    sliding_window_df = pd.read_parquet(f'{output_dir}/sliding_window_tf_to_peak_score.parquet')

    logging.info("\tHomer TF to peak DataFrame")
    homer_df = pd.read_parquet(f'{output_dir}/homer_tf_to_peak.parquet')

    logging.info("\tRNAseq dataset")
    rna_df = pd.read_csv(rna_data_file, header=0)
    rna_df['mean_gene_expression'] = rna_df.iloc[:, 1:].mean(axis=1)
    rna_df = rna_df[['gene_id', 'mean_gene_expression']]

    logging.info("\tATACseq dataset")
    atac_df = pd.read_csv(atac_data_file, header=0)
    atac_df['mean_peak_accessibility'] = atac_df.iloc[:, 1:].mean(axis=1)
    atac_df = atac_df[['peak_id', 'mean_peak_accessibility']]
    logging.debug("Done!")
    logging.debug("\n---------------------------\n")

    logging.info("\n ============== Merging DataFrames ==============")
    final_df = merge_score_dataframes(peak_corr_df, cicero_df, sliding_window_df, homer_df, rna_df, atac_df)
    logging.info(final_df.head())
    logging.info(final_df.shape)
    

    # Drop columns that dont have all three of the peak, target, and source names
    final_df = final_df.dropna(subset=[
        "peak_id",
        "target_id",
        "source_id",
        "mean_TF_expression",
        "mean_TG_expression"
        ])
    logging.debug("final_df")
    logging.debug(final_df.head())
    logging.debug(final_df.columns)
    logging.debug("\n---------------------------\n")
        
    final_df["num_cols_w_values"] = final_df.notna().sum(axis=1)
    final_df = final_df.sort_values(ascending=False, by="num_cols_w_values")
    final_df = final_df.drop(columns=["num_cols_w_values"])
    
    # # ===== WRITE OUT THE FULL RAW DATAFRAME =====
    # logging.info("Writing a non-normalized 10% dataframe as 'inferred_network_non_normalized.csv'")
    # top_10_percent_df = final_df.head(int(len(final_df) * 0.10))    
    # write_csv_in_chunks(top_10_percent_df, output_dir, 'inferred_network_non_normalized.csv')

    logging.info("\nNormalizing feature score columns")
    num_cols = final_df.select_dtypes("number").columns.difference([
        "cicero_score", "TSS_dist_score"  # skip list
    ])

    # compute 5th/95th percentiles for each column in a single pass
    quantiles = final_df[num_cols].quantile([0.05, 0.95])

    # mask + scale with broadcasting
    low, high = quantiles.loc[0.05], quantiles.loc[0.95]
    trimmed = final_df[num_cols].where(final_df[num_cols].gt(low) & final_df[num_cols].lt(high))

    # min‐max normalize + log1p
    scaled = (trimmed - trimmed.min()) / (trimmed.max() - trimmed.min())
    final_df[num_cols] = np.log1p(scaled.fillna(0))     
    
    # MinMax normalize all feature score columns, whether or not they were normalized
    cols_to_minmax = [col for col in final_df.select_dtypes(include=np.number).columns]
    
    for col in cols_to_minmax:
        final_df[col] = minmax_normalize_column(final_df[col])

    # Set the desired column order
    column_order = [
        "source_id",
        "target_id",
        "peak_id",
        "mean_TF_expression",
        "mean_TG_expression",
        "mean_peak_accessibility",
        "cicero_score",
        "TSS_dist_score",
        "correlation",
        "sliding_window_score",
        "homer_binding_score"
    ]
    
    column_order = [col for col in column_order if col in final_df.columns]
    
    final_df = final_df[column_order]
    
    all_nan_cols = final_df.columns[final_df.isna().all()].tolist()
    if all_nan_cols:
        logging.warning(f"The following columns are entirely NaN: {all_nan_cols}")
    else:
        logging.info("No columns are entirely NaN.")

        
    logging.info(final_df.head())
    logging.info('\nColumns:')
    for col_name in final_df.columns:
        logging.info(f'\t{col_name}')
    logging.info(final_df.columns)
    logging.info(f'\nTFs: {len(final_df["source_id"].unique())}')
    logging.info(f'Peaks: {len(final_df["peak_id"].unique())}')
    logging.info(f'TGs: {len(final_df["target_id"].unique())}')
    
    # For testing, randomly downsample the rows
    logging.info(f"Creating and saving a {subsample}% downsampling of the dataset for testing")
    decimal_subsample = subsample / 100
    
    logging.info(f'\tNumber of unique non-NaN scores for each feature:')
    for column in column_order:
        logging.info(f'\t\tNumber of {column} scores: {final_df[column].nunique(dropna=True)}')
        
    # Write the final df as a parquet file
    final_df.to_parquet(f"{inferred_grn_dir}/inferred_network.parquet", engine="pyarrow", index=False, compression="snappy")

    # if you need a 10% sample for testing:
    final_df.sample(frac=decimal_subsample).to_parquet(
        f"{inferred_grn_dir}/inferred_network_{subsample}pct.parquet", engine="pyarrow", index=False, compression="snappy"
    )
        
    logging.info(f'\nSaving rows with no more than 2 missing feature scores')
    n_score_cols = len(final_df.select_dtypes(include=np.number).columns)
    feature_threshold = n_score_cols - 2
    final_df_enriched_features = final_df[final_df.count(numeric_only=True, axis=1) >= feature_threshold]
    logging.info(f'\tNumber of rows with >= {feature_threshold}/{n_score_cols} feature columns {len(final_df)}')
    
    final_df_enriched_features.to_parquet(f"{inferred_grn_dir}/inferred_network_enrich_feat.parquet", engine="pyarrow", index=False, compression="snappy")
    
    logging.info("Done!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    main()