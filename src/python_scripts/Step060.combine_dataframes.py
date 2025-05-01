import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import argparse
import logging
import os
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster


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

def tidy_tf_peak_score(df: dd.DataFrame, score_col: str) -> dd.DataFrame:
    df = df.rename(columns={score_col: "score_value"}).copy()
    df["score_type"] = score_col
    df["target_id"] = None
    return df[["source_id", "peak_id", "target_id", "score_type", "score_value"]]

def tidy_peak_tg_score(df: dd.DataFrame, score_col: str) -> dd.DataFrame:
    df = df.rename(columns={score_col: "score_value"}).copy()
    df["score_type"] = score_col
    df["source_id"] = None
    return df[["source_id", "peak_id", "target_id", "score_type", "score_value"]]

def tidy_expression_scores(df: dd.DataFrame, id_col: str, score_col: str) -> dd.DataFrame:
    df = df.rename(columns={score_col: "score_value"}).copy()
    df["score_type"] = score_col
    df["source_id"] = None
    df["peak_id"] = None
    df["target_id"] = None
    return df[["source_id", "peak_id", "target_id", "score_type", "score_value"]].assign(**{id_col: df[id_col]})

def merge_score_dataframes_dask(output_dir: str, rna_data_file: str, atac_data_file: str) -> dd.DataFrame:
    logging.info("=== Starting Dask merge_score_dataframes ===")

    # Load score data
    sliding_window_dd = dd.read_parquet(f"{output_dir}/sliding_window_tf_to_peak_score.parquet")
    homer_dd = dd.read_parquet(f"{output_dir}/homer_tf_to_peak.parquet")
    peak_corr_dd = dd.read_parquet(f"{output_dir}/peak_to_gene_correlation.parquet")
    cicero_dd = dd.read_parquet(f"{output_dir}/cicero_peak_to_tg_scores.parquet")

    # Load expression data
    rna_df = dd.read_parquet(rna_data_file)
    rna_df["mean_TF_expression"] = rna_df.select_dtypes("number").mean(axis=1)
    dd_rna_tf = rna_df[["gene_id", "mean_TF_expression"]].rename(columns={"gene_id": "source_id"})

    rna_df = rna_df.drop(columns=["mean_TF_expression"])
    rna_df["mean_TG_expression"] = rna_df.select_dtypes("number").mean(axis=1)
    dd_rna_tg = rna_df[["gene_id", "mean_TG_expression"]].rename(columns={"gene_id": "target_id"})

    # Load ATAC data
    atac_df = dd.read_parquet(atac_data_file)
    atac_df["mean_peak_accessibility"] = atac_df.select_dtypes("number").mean(axis=1)
    dd_atac = atac_df[["peak_id", "mean_peak_accessibility"]]

    # Tidy score data
    tidy_scores = dd.concat([
        tidy_tf_peak_score(sliding_window_dd, "sliding_window_score"),
        tidy_tf_peak_score(homer_dd, "homer_binding_score"),
        tidy_peak_tg_score(peak_corr_dd[["peak_id", "target_id", "correlation"]], "correlation"),
        tidy_peak_tg_score(peak_corr_dd[["peak_id", "target_id", "TSS_dist_score"]], "TSS_dist_score"),
        tidy_peak_tg_score(cicero_dd, "cicero_score"),
        tidy_expression_scores(dd_rna_tf, id_col="source_id", score_col="mean_TF_expression"),
        tidy_expression_scores(dd_rna_tg, id_col="target_id", score_col="mean_TG_expression"),
        tidy_expression_scores(dd_atac, id_col="peak_id", score_col="mean_peak_accessibility"),
    ], interleave_partitions=True)

    logging.info("Building complete source_id × peak_id × target_id keyspace")

    tf_peak = dd.concat([
        sliding_window_dd[["source_id", "peak_id"]],
        homer_dd[["source_id", "peak_id"]],
        dd_rna_tf[["source_id"]].assign(peak_id=None)
    ], interleave_partitions=True).drop_duplicates()

    peak_tg = dd.concat([
        peak_corr_dd[["peak_id", "target_id"]],
        cicero_dd[["peak_id", "target_id"]],
        dd_rna_tg[["target_id"]].assign(peak_id=None)
    ], interleave_partitions=True).drop_duplicates()

    triplet_df = tf_peak.merge(peak_tg, on="peak_id", how="outer").dropna(subset=["source_id", "target_id"], how="any")

    # Merge tidy scores back into triplets
    logging.info("Merging scores into long-format triplet table")
    result_dd = triplet_df.merge(tidy_scores, on=["source_id", "peak_id", "target_id"], how="left")

    return result_dd

def main():
    # Parse command-line arguments
    args: argparse.Namespace = parse_args()
    atac_data_file: str = args.atac_data_file
    rna_data_file: str = args.rna_data_file
    output_dir: str = args.output_dir
    inferred_grn_dir: str = args.inferred_grn_dir
    fig_dir: str = args.fig_dir
    subsample: str = float(args.subsample)
    
    # dask_tmp = os.path.join(output_dir, "tmp", "dask_tmp")
    # os.makedirs(dask_tmp, exist_ok=True)
        
    # cluster = LocalCluster(n_workers=4, threads_per_worker=4, memory_limit="64GB", local_directory=dask_tmp)
    # client = Client(cluster)

    logging.info("\n ============== Merging DataFrames ==============")
    result_dd = merge_score_dataframes_dask(output_dir, rna_data_file, atac_data_file)
    
    # Keep rows with scores
    result_filtered = result_dd.dropna(subset=["score_value"])
    
    result_filtered.to_parquet(
        os.path.join(output_dir, "tidy_result_matrix.parquet"), 
        engine="pyarrow", 
        compression="snappy"
        )

    # Count how many unique score types exist per triplet
    score_counts = (
        result_filtered
        .groupby(["source_id", "peak_id", "target_id"])["score_type"]
        .nunique()
        .reset_index()
        .rename(columns={"score_type": "n_score_types"})
    )

    # To preview:
    logging.info(score_counts.head(10))
    
    
    
    # ============ L A Z Y  Processing ============ #

#     # Sort by number of non-null columns
#     sorted_df = filtered_df.sort_values(by=["source_id", "peak_id", "target_id"])
    
#     final_dd["num_cols_w_values"] = final_dd.map_partitions(
#         lambda df: df.notna().sum(axis=1)
#     )
#     final_dd = final_dd.map_partitions(
#         lambda df: df.sort_values(ascending=False, by="num_cols_w_values")
#     ).drop(columns=["num_cols_w_values"])

#     # Select final column order
#     column_order = [
#         "source_id",
#         "target_id",
#         "peak_id",
#         "mean_TF_expression",
#         "mean_TG_expression",
#         "mean_peak_accessibility",
#         "cicero_score",
#         "TSS_dist_score",
#         "correlation",
#         "sliding_window_score",
#         "homer_binding_score"
#     ]

#     existing_columns = [col for col in column_order if col in final_dd.columns]
#     final_dd = final_dd[existing_columns]

#     # Save enriched features (rows with ≥ N-2 numeric scores)
#     n_score_cols = len(final_dd.select_dtypes(include="number").columns)
#     feature_threshold = n_score_cols - 2
#     enriched_dd = final_dd[final_dd.count(axis=1, numeric_only=True) >= feature_threshold]
    
#     logging.info(f"Total rows: {final_dd.shape[0].compute()}")
#     logging.info(f"Enriched rows: {enriched_dd.shape[0].compute()}")

#     logging.info(f"Writing enriched feature subset: {feature_threshold}/{n_score_cols} non-null numeric features required")
#     enriched_dd.repartition(partition_size="256MB").to_parquet(
#         f"{inferred_grn_dir}/inferred_network_enrich_feat.parquet",
#         engine="pyarrow",
#         compression="snappy",
#     )

#     logging.info("Done!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    main()