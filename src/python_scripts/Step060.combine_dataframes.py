import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import argparse
import logging
import os
import dask.dataframe as dd

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

def merge_score_dataframes_slow(output_dir, rna_data_file, atac_data_file):
    logging.info("Loading in the DataFrames")
    logging.info("\tCorrelation peak to TG DataFrame")
    peak_corr_df = dd.read_parquet(f'{output_dir}/peak_to_gene_correlation.parquet')

    logging.info("\tCicero peak to TG DataFrame")
    cicero_df = dd.read_parquet(f'{output_dir}/cicero_peak_to_tg_scores.parquet')

    logging.info("\tSliding Window peak to TG DataFrame")
    sliding_window_df = dd.read_parquet(f'{output_dir}/sliding_window_tf_to_peak_score.parquet')

    logging.info("\tHomer TF to peak DataFrame")
    homer_df = dd.read_parquet(f'{output_dir}/homer_tf_to_peak.parquet')

    logging.info("\tRNAseq dataset")
    rna_df = dd.read_parquet(rna_data_file)
    rna_df['mean_gene_expression'] = rna_df.select_dtypes("number").mean(axis=1)
    rna_df = rna_df[['gene_id', 'mean_gene_expression']]

    logging.info("\tATACseq dataset")
    atac_df = dd.read_parquet(atac_data_file)
    atac_df['mean_peak_accessibility'] = atac_df.select_dtypes("number").mean(axis=1)
    atac_df = atac_df[['peak_id', 'mean_peak_accessibility']]
    logging.debug("Done!")
    logging.debug("\n---------------------------\n")
    
    logging.info("\nCombining the sliding window and Homer TF to peak binding scores")
    tf_to_peak_df = pd.merge(sliding_window_df, homer_df, on=["peak_id", "source_id"], how="outer")
    logging.info(f"    • tf_to_peak_df shape: {tf_to_peak_df.shape}")

    logging.info(" - Adding mean RNA expression to the TF to peak binding DataFrame")
    tf_expr_to_peak_df = pd.merge(rna_df, tf_to_peak_df, left_on="gene_id", right_on="source_id", how="right").drop("gene_id", axis=1)
    tf_expr_to_peak_df = tf_expr_to_peak_df.rename(columns={"mean_gene_expression": "mean_TF_expression"})
    logging.info(f"    • tf_expr_to_peak_df shape: {tf_expr_to_peak_df.shape}")

    logging.info("\nMerging the correlation and cicero methods for peak to target gene")
    peak_to_tg_df = pd.merge(peak_corr_df, cicero_df, on=["peak_id", "target_id"], how="outer")
    logging.info(f"    • peak_to_tg_df shape: {peak_to_tg_df.shape}")

    logging.info(" - Adding mean RNA expression to the peak to TF DataFrame")
    peak_to_tg_expr_df = pd.merge(rna_df, peak_to_tg_df, left_on="gene_id", right_on="target_id", how="right").drop("gene_id", axis=1)
    peak_to_tg_expr_df = peak_to_tg_expr_df.rename(columns={"mean_gene_expression": "mean_TG_expression"})
    logging.info(f"    • peak_to_tg_expr_df shape: {peak_to_tg_expr_df.shape}")

    logging.info("\nMerging the peak to target gene scores with the sliding window TF to peak scores")
    # For the sliding window genes, change their name to "source_id" to represent that these genes are TFs
    tf_to_tg_score_df = pd.merge(tf_expr_to_peak_df, peak_to_tg_expr_df, on=["peak_id"], how="outer")
    logging.info(f"    • tf_to_tg_score_df shape: {tf_to_tg_score_df.shape}")

    logging.info(" - Adding the mean ATAC-seq peak accessibility values")
    final_df = pd.merge(atac_df, tf_to_tg_score_df, on="peak_id", how="left")
    
    return final_df

def merge_score_dataframes_dask(output_dir: str,
                                rna_data_file: str,
                                atac_data_file: str) -> dd.DataFrame:
    logging.info("=== Starting Dask merge_score_dataframes ===")

    # 1) Read the big Parquet tables
    logging.info("1/8: Loading sliding_window_df and homer_df as Dask")
    sliding_window_dd = dd.read_parquet(f"{output_dir}/sliding_window_tf_to_peak_score.parquet")
    homer_dd          = dd.read_parquet(f"{output_dir}/homer_tf_to_peak.parquet")
    logging.info(f"    • sliding_window_dd partitions={sliding_window_dd.npartitions}, columns={list(sliding_window_dd.columns)}")
    logging.info(f"    • homer_dd          partitions={homer_dd.npartitions}, columns={list(homer_dd.columns)}")

    logging.info("2/8: Loading peak_corr_df and cicero_df as Dask")
    peak_corr_dd = dd.read_parquet(f"{output_dir}/peak_to_gene_correlation.parquet")
    cicero_dd    = dd.read_parquet(f"{output_dir}/cicero_peak_to_tg_scores.parquet")
    logging.info(f"    • peak_corr_dd partitions={peak_corr_dd.npartitions}, columns={list(peak_corr_dd.columns)}")
    logging.info(f"    • cicero_dd    partitions={cicero_dd.npartitions}, columns={list(cicero_dd.columns)}")

    # 2) Merge sliding window and Homer TF-peak scores
    logging.info("3/8: Combining the sliding window and Homer TF→peak scores")
    tf_to_peak_dd = sliding_window_dd.merge(
        homer_dd,
        on=["peak_id", "source_id"],
        how="outer"
    )
    logging.info(f"    • tf_to_peak_dd shape ~ {tf_to_peak_dd.shape} (lazy)")

    # 3) Read RNA data, compute mean TF expression
    logging.info("4/8: Adding mean RNA expression for TF (mean_TF_expression)")
    rna_df = dd.read_parquet(rna_data_file)
    rna_df["mean_TF_expression"] = rna_df.select_dtypes("number").mean(axis=1)
    dd_rna_tf = rna_df[["gene_id", "mean_TF_expression"]].set_index("gene_id")
    logging.info(f"    • dd_rna_tf partitions={dd_rna_tf.npartitions}")

    # Merge TF expression into TF-peak DataFrame
    tf_expr_to_peak_dd = tf_to_peak_dd.merge(
        dd_rna_tf,
        left_on="source_id",
        right_index=True,
        how="right"
    )
    logging.info(f"    • tf_expr_to_peak_dd columns={list(tf_expr_to_peak_dd.columns)}")

    # 4) Merge peak to gene scores (correlation + Cicero)
    logging.info("5/8: Merging the correlation and Cicero methods for peak→target gene")
    peak_to_tg_dd = peak_corr_dd.merge(
        cicero_dd,
        on=["peak_id", "target_id"],
        how="outer"
    )
    logging.info(f"    • peak_to_tg_dd partitions={peak_to_tg_dd.npartitions}")

    # 5) Add mean TG expression
    logging.info("6/8: Adding mean RNA expression for TG (mean_TG_expression)")
    rna_df = rna_df.drop(columns=["mean_TF_expression"])
    rna_df["mean_TG_expression"] = rna_df.select_dtypes("number").mean(axis=1)
    dd_rna_tg = rna_df[["gene_id", "mean_TG_expression"]].set_index("gene_id")

    peak_to_tg_expr_dd = peak_to_tg_dd.merge(
        dd_rna_tg,
        left_on="target_id",
        right_index=True,
        how="right"
    )
    logging.info(f"    • peak_to_tg_expr_dd columns={list(peak_to_tg_expr_dd.columns)}")

    # 6) Merge TF→peak and peak→TG
    logging.info("7/8: Merging TF→peak and peak→TG into tf_to_tg_score_dd")
    tf_to_tg_score_dd = tf_expr_to_peak_dd.merge(
        peak_to_tg_expr_dd,
        on=["peak_id"],
        how="outer"
    )
    logging.info(f"    • tf_to_tg_score_dd columns={list(tf_to_tg_score_dd.columns)}")

    # 7) Read ATAC data, compute mean peak accessibility
    logging.info("8/8: Adding mean ATAC‐seq peak accessibility")
    atac_df = dd.read_parquet(atac_data_file)
    atac_df["mean_peak_accessibility"] = atac_df.select_dtypes("number").mean(axis=1)
    dd_atac = atac_df[["peak_id", "mean_peak_accessibility"]].set_index("peak_id")

    final_dd = tf_to_tg_score_dd.merge(
        dd_atac,
        left_on="peak_id",
        right_index=True,
        how="left"
    )
    logging.info(f"    • final_dd columns={list(final_dd.columns)}")

    return final_dd

def main():
    # Parse command-line arguments
    args: argparse.Namespace = parse_args()
    atac_data_file: str = args.atac_data_file
    rna_data_file: str = args.rna_data_file
    output_dir: str = args.output_dir
    inferred_grn_dir: str = args.inferred_grn_dir
    fig_dir: str = args.fig_dir
    subsample: str = float(args.subsample)
    

    logging.info("\n ============== Merging DataFrames ==============")
    final_dd = merge_score_dataframes_dask(output_dir, rna_data_file, atac_data_file)
    
    # ============ L A Z Y  Processing ============ #

    logging.info("Filtering out rows missing peak_id, target_id, source_id, TF/TG expr")
    final_dd = final_dd.dropna(subset=[
        "peak_id",
        "target_id",
        "source_id",
        "mean_TF_expression",
        "mean_TG_expression"
    ])

    # Sort by number of non-null columns
    final_dd["num_cols_w_values"] = final_dd.notna().sum(axis=1)
    final_dd = final_dd.map_partitions(
        lambda df: df.sort_values(ascending=False, by="num_cols_w_values")
    ).drop(columns=["num_cols_w_values"])

    # Normalize score columns
    logging.info("\nNormalizing feature score columns")

    num_cols = final_dd.select_dtypes("number").columns.difference([
        "cicero_score", "TSS_dist_score"
    ])

    # Get 5th/95th percentiles lazily
    logging.info("Calculating 5th and 95th percentiles for normalization")
    quantiles = final_dd[num_cols].quantile([0.05, 0.95]).compute()
    low, high = quantiles.loc[0.05], quantiles.loc[0.95]

    def normalize_partition(df):
        trimmed = df[num_cols].where(df[num_cols].gt(low) & df[num_cols].lt(high))
        scaled = (trimmed - trimmed.min()) / (trimmed.max() - trimmed.min())
        df[num_cols] = np.log1p(scaled.fillna(0))
        return df

    final_dd = final_dd.map_partitions(normalize_partition)

    # Normalize again min-max to [0,1]
    def minmax_partition(df):
        for col in df.select_dtypes(include="number").columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return df

    final_dd = final_dd.map_partitions(minmax_partition)

    # Select final column order
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

    existing_columns = [col for col in column_order if col in final_dd.columns]
    final_dd = final_dd[existing_columns]
    
    # Save final merged DataFrame
    logging.info("Writing full inferred network Parquet")
    final_dd.repartition(partition_size="256MB").to_parquet(
        f"{inferred_grn_dir}/inferred_network.parquet",
        engine="pyarrow",
        compression="snappy",
        write_index=False
    )

    # Save subsampled version
    logging.info(f"Writing {subsample:.0f}% subsample of inferred network")
    final_dd.sample(frac=subsample/100).repartition(partition_size="256MB").to_parquet(
        f"{inferred_grn_dir}/inferred_network_{subsample:.0f}pct.parquet",
        engine="pyarrow",
        compression="snappy",
        write_index=False
    )

    # Save enriched features (rows with ≥ N-2 numeric scores)
    n_score_cols = len(final_dd.select_dtypes(include="number").columns)
    feature_threshold = n_score_cols - 2
    enriched_dd = final_dd[final_dd.count(axis=1, numeric_only=True) >= feature_threshold]

    logging.info(f"Writing enriched feature subset: {feature_threshold}/{n_score_cols} non-null numeric features required")
    enriched_dd.repartition(partition_size="256MB").to_parquet(
        f"{inferred_grn_dir}/inferred_network_enrich_feat.parquet",
        engine="pyarrow",
        compression="snappy",
        write_index=False
    )

    logging.info("Done!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    main()