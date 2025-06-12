import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
import os
import dask.dataframe as dd
from dask import compute
from typing import Set, Tuple, Union
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
import dask

from grn_inference.normalization import (
    minmax_normalize_pandas,
    minmax_normalize_dask,
    clip_and_normalize_log1p_dask
)

from grn_inference.plotting import plot_feature_score_histograms_dask

dask.config.set({"dataframe.shuffle.method": "disk"})

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
        "--string_dir", 
        type=str, 
        required=True, 
        help="Path to STRING database directory"
    )


    args: argparse.Namespace = parser.parse_args()
    return args

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

def compute_expression_means(rna_dd: dd.DataFrame) -> Tuple[dd.DataFrame, dd.DataFrame]:
    """Compute mean TF and TG expression from RNA matrix."""
    
    expr_cols = [c for c in rna_dd.columns if c != "gene_id"]
    
    norm_rna_df = minmax_normalize_dask(
        ddf=rna_dd.assign(mean_expression=rna_dd[expr_cols].mean(axis=1)),
        score_cols=["mean_expression"]
    ).rename(columns={"mean_expression": "mean_expression"})
    
    tf_df = norm_rna_df[["gene_id", "mean_expression"]].rename(columns={
        "gene_id": "source_id",
        "mean_expression": "mean_TF_expression"
    })
    
    tg_df = norm_rna_df[["gene_id", "mean_expression"]].rename(columns={
        "gene_id": "target_id",
        "mean_expression": "mean_TG_expression"
    })
    
    return tf_df, tg_df

def compute_atac_mean(atac_df: dd.DataFrame) -> dd.DataFrame:
    """Compute mean peak accessibility."""
    number_cols = [c for c in atac_df.columns if c != "peak_id"]
    atac_df["mean_peak_accessibility"] = atac_df[number_cols].mean(axis=1)
    
    norm_atac_df = minmax_normalize_dask(
        ddf=atac_df, 
        score_cols=["mean_peak_accessibility"], 
    )
    
    return norm_atac_df[["peak_id", "mean_peak_accessibility"]]

def join_all_scores(full_edges_dd, score_dfs):
    df = full_edges_dd
    for label, score_df, keys in score_dfs:
        logging.info(f"  - Merging {label}")
        df = df.merge(score_df, how="left", on=keys)
    return df

def add_string_db_scores(
    inferred_net_dd: dd.DataFrame,
    string_dir: str,
    tf_peak_dd: dd.DataFrame,
    peak_tg_dd: dd.DataFrame
) -> dd.DataFrame:
    logging.info("  - Reading STRING protein_info.txt")
    protein_info_df = pd.read_csv(f"{string_dir}/protein_info.txt", sep="\t")
    logging.info("  - Reading STRING protein_links_detailed.txt")
    protein_links_df = pd.read_csv(f"{string_dir}/protein_links_detailed.txt", sep=" ")

    filtered_links_df = protein_links_df[[
        "protein1", "protein2", "experimental", "textmining", "combined_score"
    ]].copy()

    id_to_name = protein_info_df.set_index("#string_protein_id")["preferred_name"].to_dict()
    name_to_id = {v: k for k, v in id_to_name.items()}

    tf_names = tf_peak_dd["source_id"].drop_duplicates().compute()
    tg_names = peak_tg_dd["target_id"].drop_duplicates().compute()
    tf_set = set(tf_names.tolist())
    tg_set = set(tg_names.tolist())

    tf_string_ids = {name_to_id[nm] for nm in tf_set if nm in name_to_id}
    tg_string_ids = {name_to_id[nm] for nm in tg_set if nm in name_to_id}

    mask = (
        filtered_links_df["protein1"].isin(tf_string_ids) &
        filtered_links_df["protein2"].isin(tg_string_ids)
    )
    filtered_links_df = filtered_links_df.loc[mask, :].copy()
    if filtered_links_df.shape[0] == 0:
        logging.info("No matching STRING edges found; skipping STRING step.")
        return inferred_net_dd

    filtered_links_df["protein1_name"] = filtered_links_df["protein1"].map(id_to_name)
    filtered_links_df["protein2_name"] = filtered_links_df["protein2"].map(id_to_name)

    filtered_links_df = filtered_links_df[[
        "protein1_name", "protein2_name", "experimental", "textmining", "combined_score"
    ]].rename(columns={
        "protein1_name": "protein1",
        "protein2_name": "protein2",
        "experimental": "string_experimental_score",
        "textmining": "string_textmining_score",
        "combined_score": "string_combined_score"
    })

    logging.info("  - Converting filtered STRING links to Dask DataFrame")
    nrows = filtered_links_df.shape[0]
    nparts = max(1, nrows // 100_000)
    string_dd = dd.from_pandas(filtered_links_df, npartitions=nparts)

    string_dd = clip_and_normalize_log1p_dask(
        ddf=string_dd,
        score_cols=[
            "string_experimental_score",
            "string_textmining_score",
            "string_combined_score"
        ],
        quantiles=(0.05, 0.95),
        apply_log1p=True
    )

    string_dd = minmax_normalize_dask(
        ddf=string_dd,
        score_cols=[
            "string_experimental_score",
            "string_textmining_score",
            "string_combined_score"
        ]
    )

    cols_to_keep = [
        "source_id", "peak_id", "target_id",
        "sliding_window_score", "homer_binding_score",
        "correlation", "cicero_score",
        "TSS_dist_score",
        "mean_TF_expression", "mean_TG_expression",
        "mean_peak_accessibility"
    ]
    inferred_net_dd = inferred_net_dd[cols_to_keep]

    logging.info("  - Merging normalized STRING scores into inferred_net_dd")
    merged_dd = inferred_net_dd.merge(
        string_dd,
        left_on=["source_id", "target_id"],
        right_on=["protein1", "protein2"],
        how="left"
    ).drop(columns=["protein1", "protein2"])

    merged_dd = merged_dd.persist()
    logging.info("  Done merging STRING scores.")
    return merged_dd

def filter_scored_edges(df, min_valid_scores=6, score_cols=None):
    """
    Filters out rows with too many missing values.

    Parameters:
        df: DataFrame or Dask DataFrame
        min_valid_scores: minimum number of non-NaN score fields required
        score_cols: list of score column names (optional)

    Returns:
        Filtered DataFrame
    """
    if score_cols is None:
        score_cols = [
            "sliding_window_score", "homer_binding_score",
            "correlation", "TSS_dist_score", "cicero_score",
            "mean_TF_expression", "mean_TG_expression", "mean_peak_accessibility"
        ]

    return df.dropna(subset=score_cols, thresh=min_valid_scores)

def extract_edges(df: Union[pd.DataFrame, dd.DataFrame], edge_cols: list) -> Set[Tuple[str, ...]]:
    """Extract unique edge tuples from a DataFrame."""
    df = df[edge_cols]
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    return set(map(tuple, df.values))

def main():
    # Parse command-line arguments
    args: argparse.Namespace = parse_args()
    atac_data_file: str = args.atac_data_file
    rna_data_file: str = args.rna_data_file
    output_dir: str = args.output_dir
    inferred_grn_dir: str = args.inferred_grn_dir
    
    # 1) Create a temporary local folder for Dask spill files (you already do this later)
    dask_tmp_dir = os.path.join(output_dir, "tmp/dask_tmp")
    os.makedirs(dask_tmp_dir, exist_ok=True)
    
    os.environ["TMPDIR"] = dask_tmp_dir
    os.environ["DASK_TEMPORARY_DIRECTORY"] = dask_tmp_dir

    # 2) Launch a SLURMCluster
    cluster = SLURMCluster(
        cores=4,
        memory="128GB",
        queue="compute",
        walltime="02:00:00",
        interface="ib0",
        local_directory=dask_tmp_dir,

    )

    # 3) Scale out to N workers (launch N SLURM jobs)
    #    Change 10 → however many nodes/workers you want.
    cluster.scale(15)

    # 4) Connect a Dask Client to that cluster
    client = Client(cluster)
    logging.info(f"Dask dashboard: {client.dashboard_link}")
    
    

    logging.info("\n ============== Loading Score DataFrames ==============")
    # Load Dask DataFrames
    logging.info("  - (0/6) Loading TSS Distance Score DataFrame")
    tss_dist_dd: dd.DataFrame       = dd.read_parquet(f"{output_dir}/tss_distance_score.parquet")
    
    logging.info("  - (1/6) Loading Sliding Window DataFrame")
    sliding_window_dd: dd.DataFrame = dd.read_parquet(f"{output_dir}/sliding_window_tf_to_peak_score.parquet")
    
    logging.info("  - (2/6) Loading Homer DataFrame")
    homer_dd: dd.DataFrame          = dd.read_parquet(f"{output_dir}/homer_tf_to_peak.parquet")
    
    logging.info("  - (3/6) Loading Peak to TG Correlation DataFrame")
    peak_corr_dd: dd.DataFrame      = dd.read_parquet(f"{output_dir}/peak_to_gene_correlation.parquet")
    
    logging.info("  - (4/6) Loading Cicero DataFrame")
    cicero_dd: dd.DataFrame         = dd.read_parquet(f"{output_dir}/cicero_peak_to_tg_scores.parquet")
    
    logging.info("  - (5/6) Loading scRNAseq DataFrame")
    rna_df: dd.DataFrame            = dd.read_parquet(rna_data_file)
    
    logging.info("  - (6/6) Loading scATACseq DataFrame")
    atac_df: dd.DataFrame           = dd.read_parquet(atac_data_file)
    logging.info("\n  All dataframes loaded")
    
    sliding_window_dd = sliding_window_dd.repartition(partition_size="100MB")
    homer_dd          = homer_dd.repartition(partition_size="100MB")
    peak_corr_dd      = peak_corr_dd.repartition(partition_size="100MB")
    cicero_dd         = cicero_dd.repartition(partition_size="100MB")
    tss_dist_dd       = tss_dist_dd.repartition(partition_size="100MB")
    rna_df   = rna_df.repartition(partition_size="100MB")
    atac_df  = atac_df.repartition(partition_size="100MB")

    # Compute mean TF and TG expression
    logging.info("\n  - Calculating average TF and TG expression")
    dd_rna_tf, dd_rna_tg = compute_expression_means(rna_df)

    # Compute mean peak accessibility
    logging.info("  - Calculating average peak accessibility")
    dd_atac = compute_atac_mean(atac_df)

    # Extract edges from DFs
    logging.info("\n ============== Finding Unique TF-peak-TG Edges ==============")

    sliding_window_dd = sliding_window_dd.assign(
        source_id=sliding_window_dd["source_id"].astype("category"),
        peak_id=sliding_window_dd["peak_id"].astype("category"),
    )

    homer_dd = homer_dd.assign(
        source_id=homer_dd["source_id"].astype("category"),
        peak_id=homer_dd["peak_id"].astype("category"),
    )

    # b) peak_corr and cicero must be categorical on both keys
    peak_corr_dd = peak_corr_dd.assign(
        peak_id=peak_corr_dd["peak_id"].astype("category"),
        target_id=peak_corr_dd["target_id"].astype("category"),
    )

    cicero_dd = cicero_dd.assign(
        peak_id=cicero_dd["peak_id"].astype("category"),
        target_id=cicero_dd["target_id"].astype("category"),
    )

    # c) tss_dist also categorical on both keys, then index it
    tss_dist_dd = tss_dist_dd.assign(
        peak_id=tss_dist_dd["peak_id"].astype("category"),
        target_id=tss_dist_dd["target_id"].astype("category"),
    )

    logging.info("  - Extracting Sliding Window TF→peak edges")
    tf_peak_dd = dd.concat([
        sliding_window_dd[["source_id", "peak_id"]],
        homer_dd[["source_id", "peak_id"]]
    ]).drop_duplicates()

    logging.info("  - Extracting Peak→TG edges")
    peak_tg_dd = dd.concat([
        peak_corr_dd[["peak_id", "target_id"]],
        cicero_dd[["peak_id", "target_id"]]
    ]).drop_duplicates()

    logging.info("\n  - Building full_edges_dd with Dask join (inner merge on peak_id)...")
    full_edges_dd = tf_peak_dd.merge(
        peak_tg_dd,
        how="inner",
        on="peak_id"  # both sides are categorical, so join is efficient
    ).persist()
    
    logging.info("\n ============== Merging scores into full_edges_dd ==============")

    logging.info("  - Merging TF to peak score DataFrames")
    tf_scores_dd = sliding_window_dd.merge(
        homer_dd,
        how="outer",
        on=["source_id", "peak_id"]
    )

    logging.info("  - Merging peak to TG score DataFrames")
    ptg_scores_dd = peak_corr_dd.merge(
        cicero_dd,
        how="outer",
        on=["peak_id", "target_id"]
    )
    
    # ensure categories line up
    tf_scores_dd = tf_scores_dd.assign(
        peak_id=tf_scores_dd["peak_id"].astype("category"),
        source_id=tf_scores_dd["source_id"].astype("category")
    )
    ptg_scores_dd = ptg_scores_dd.assign(
        peak_id=ptg_scores_dd["peak_id"].astype("category"),
        target_id=ptg_scores_dd["target_id"].astype("category")
    )

    logging.info("  - Merging TF→peak with peak→TG scores + TSS distance")
    combined_ddf = (
        full_edges_dd
          .merge(tf_scores_dd, how="left", on=["source_id", "peak_id"])
          .merge(ptg_scores_dd, how="left", on=["peak_id", "target_id"])
          .merge(
              tss_dist_dd,
              how="left",
              on=["peak_id", "target_id"]
          )
    ).persist()

    logging.info("  - Joining mean RNA expression and ATAC accessibility")
    tf_expr_pdf = dd_rna_tf.compute()   # small Pandas DF
    tg_expr_pdf = dd_rna_tg.compute()
    atac_pdf    = dd_atac.compute()

    # Cast the Dask side’s keys back to plain strings (object) before merging:
    combined_ddf = combined_ddf.assign(
        source_id=combined_ddf["source_id"].astype("object"),
        target_id=combined_ddf["target_id"].astype("object"),
        peak_id=combined_ddf["peak_id"].astype("object"),
    )
    
    # Cast the Pandas‐side join keys to object as well:
    tf_expr_pdf["source_id"] = tf_expr_pdf["source_id"].astype(str)
    tg_expr_pdf["target_id"] = tg_expr_pdf["target_id"].astype(str)
    atac_pdf["peak_id"]      = atac_pdf["peak_id"].astype(str)

    # Now the Pandas lookups can remain as object dtype:
    # (no need to cast tf_expr_pdf["source_id"] to category)
    combined_ddf = combined_ddf.merge(tf_expr_pdf, how="left", on="source_id")
    combined_ddf = combined_ddf.merge(tg_expr_pdf, how="left", on="target_id")
    combined_ddf = combined_ddf.merge(atac_pdf,    how="left", on="peak_id")
    
    logging.info("\n ============== Adding STRING Scores ==============")
    combined_string_ddf = add_string_db_scores(
        inferred_net_dd=combined_ddf,
        string_dir=args.string_dir,
        tf_peak_dd=tf_peak_dd,
        peak_tg_dd=peak_tg_dd
    )
    
    logging.info("\n ============== Filtering and Melting Combined Score DataFrame ==============")
    score_cols = [
        "sliding_window_score", "homer_binding_score",
        "correlation", "TSS_dist_score", "cicero_score",
        "mean_TF_expression", "mean_TG_expression", "mean_peak_accessibility",
        "string_experimental_score", "string_textmining_score", "string_combined_score"
    ]
    
    num_score_col_threshold = 7
    
    logging.info(f"  - Filtering combined DataFrame to only contain edges with at least {num_score_col_threshold} / {len(score_cols)} scores")
    # Filter the combined dataframe to only contain rows with scores in num_score_col_threshold columns
    filtered_combined_string_ddf = filter_scored_edges(
        combined_string_ddf, 
         min_valid_scores=num_score_col_threshold,
         score_cols=score_cols
         ).persist()
    logging.info("      Done!")
    
    # filtered_combined_string_ddf = filtered_combined_string_ddf.repartition(npartitions=100)
    # plot_feature_score_histograms_dask(filtered_combined_string_ddf, score_cols, output_dir)
    
    # logging.info(f'  - Melting the combined DataFrame to reduce NaN values')
    # melted_df = filtered_combined_string_ddf.melt(
    #     id_vars=["source_id", "peak_id", "target_id"],
    #     value_vars=score_cols,
    #     var_name="score_type",
    #     value_name="score_value"
    # )

    # non_null_scores_ddf = filtered_combined_string_ddf.dropna(subset=["score_value"])
    # logging.info("      Done!")    
    
    logging.info(f"Writing out inferred_score_df.parquet")
    for col in ["source_id", "target_id", "peak_id"]:
        filtered_combined_string_ddf[col] = filtered_combined_string_ddf[col].astype("string")

    filtered_combined_string_ddf = filtered_combined_string_ddf.repartition(npartitions=200)
    filtered_combined_string_ddf.to_parquet(
        os.path.join(inferred_grn_dir, "inferred_score_df.parquet"),
        engine="pyarrow",
        compression="snappy",
        write_index=False
    )
    logging.info("  Done!")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    main()