import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
import os
import dask.dataframe as dd
from dask import compute
from typing import Set, Tuple, Union
from tqdm import tqdm
from dask.diagnostics import ProgressBar


from grn_inference.normalization import (
    minmax_normalize_pandas,
    minmax_normalize_dask,
    clip_and_normalize_log1p_dask
)

from grn_inference.plotting import plot_feature_score_histograms

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

def compute_expression_means(rna_df: dd.DataFrame) -> Tuple[dd.DataFrame, dd.DataFrame]:
    """Compute mean TF and TG expression from RNA matrix."""
    mean_expr = rna_df.select_dtypes("number").mean(axis=1)
    rna_df["mean_expression"] = mean_expr
    
    norm_rna_df = minmax_normalize_pandas(
        df=rna_df, 
        score_cols=["mean_expression"], 
        dtype=np.float32
    )

    tf_df = norm_rna_df[["gene_id", "mean_expression"]].rename(columns={"gene_id": "source_id", "mean_expression": "mean_TF_expression"})
    tg_df = norm_rna_df[["gene_id", "mean_expression"]].rename(columns={"gene_id": "target_id", "mean_expression": "mean_TG_expression"})
    
    return tf_df, tg_df

def compute_atac_mean(atac_df: dd.DataFrame) -> dd.DataFrame:
    """Compute mean peak accessibility."""
    atac_df["mean_peak_accessibility"] = atac_df.select_dtypes("number").mean(axis=1)
    
    norm_atac_df = minmax_normalize_pandas(
        df=atac_df, 
        score_cols=["mean_peak_accessibility"], 
        dtype=np.float32
    )
    
    return norm_atac_df[["peak_id", "mean_peak_accessibility"]]

def join_all_scores(full_edges_dd, score_dfs):
    df = full_edges_dd
    for label, score_df, keys in score_dfs:
        logging.info(f"  - Merging {label}")
        df = df.merge(score_df, how="left", on=keys)
    return df

def add_string_db_scores(inferred_net_dd, string_dir, full_edges):
    # Load STRING protein info and links (small)
    logging.info("  - Reading STRING protein info")
    protein_info_df = pd.read_csv(f"{string_dir}/protein_info.txt", sep="\t")
    
    logging.info("  - Reading STRING protein links detailed")
    protein_links_df = pd.read_csv(f"{string_dir}/protein_links_detailed.txt", sep=" ")

    logging.info("  - Mapping STRING protein IDs to preferred names")
    id_to_name = protein_info_df.set_index("#string_protein_id")["preferred_name"].to_dict()
    protein_links_df["protein1"] = protein_links_df["protein1"].map(id_to_name)
    protein_links_df["protein2"] = protein_links_df["protein2"].map(id_to_name)

    logging.info(f"  - Filtering STRING links to match TF–TG pairs")
    logging.info(f"\t\t  Computing unique source_id edges")
    with ProgressBar():
        tf_set = set(full_edges["source_id"].compute().unique())
        
    logging.info(f"\t\t  Computing unique target_id edges")
    with ProgressBar():
        tg_set = set(full_edges["target_id"].compute().unique())

    mask = protein_links_df["protein1"].isin(tf_set) & protein_links_df["protein2"].isin(tg_set)
    filtered_links_df = protein_links_df[mask]

    # Rename columns and normalize
    filtered_links_df = filtered_links_df.rename(columns={
        "experimental": "string_experimental_score",
        "textmining": "string_textmining_score",
        "combined_score": "string_combined_score"
    })[["protein1", "protein2", "string_experimental_score", "string_textmining_score", "string_combined_score"]]

    logging.info("  - Converting to Dask and normalizing STRING scores")
    string_dd = dd.from_pandas(filtered_links_df, npartitions=1)

    string_dd = clip_and_normalize_log1p_dask(
        ddf=string_dd,
        score_cols=["string_experimental_score", "string_textmining_score", "string_combined_score"],
        quantiles=(0.05, 0.95),
        apply_log1p=True,
    )

    string_dd = minmax_normalize_dask(
        ddf=string_dd,
        score_cols=["string_experimental_score", "string_textmining_score", "string_combined_score"],
        dtype=np.float32
    )

    logging.info("  - Merging normalized STRING scores into inferred network")
    merged_dd = inferred_net_dd.merge(
        string_dd,
        left_on=["source_id", "target_id"],
        right_on=["protein1", "protein2"],
        how="left"
    ).drop(columns=["protein1", "protein2"])

    logging.info("  Done!")
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

def build_full_edges(tf_peak_edges: Set[Tuple[str, str]], peak_tg_edges: Set[Tuple[str, str]]) -> Set[Tuple[str, str, str]]:
    """Build all valid (source_id, peak_id, target_id) triples based on shared peak_id."""
    peak_to_targets = {}
    for peak_id, target_id in peak_tg_edges:
        peak_to_targets.setdefault(peak_id, set()).add(target_id)

    full_edges = set()
    for source_id, peak_id in tf_peak_edges:
        if peak_id in peak_to_targets:
            for target_id in peak_to_targets[peak_id]:
                full_edges.add((source_id, peak_id, target_id))
    return full_edges

def main():
    # Parse command-line arguments
    args: argparse.Namespace = parse_args()
    atac_data_file: str = args.atac_data_file
    rna_data_file: str = args.rna_data_file
    output_dir: str = args.output_dir
    inferred_grn_dir: str = args.inferred_grn_dir

    logging.info("\n ============== Loading Score DataFrames ==============")
    # Load Dask DataFrames
    logging.info("  - (1/6) Loading Sliding Window DataFrame")
    sliding_window_dd = dd.read_parquet(f"{output_dir}/sliding_window_tf_to_peak_score.parquet")
    
    logging.info("  - (2/6) Loading Homer DataFrame")
    homer_dd          = dd.read_parquet(f"{output_dir}/homer_tf_to_peak.parquet")
    
    logging.info("  - (3/6) Loading Peak to TG Correlation DataFrame")
    peak_corr_dd      = dd.read_parquet(f"{output_dir}/peak_to_gene_correlation.parquet")
    
    logging.info("  - (4/6) Loading Cicero DataFrame")
    cicero_dd         = dd.read_parquet(f"{output_dir}/cicero_peak_to_tg_scores.parquet")
    
    logging.info("  - (5/6) Loading scRNAseq DataFrame")
    rna_df            = dd.read_parquet(rna_data_file)
    
    logging.info("  - (6/6) Loading scATACseq DataFrame")
    atac_df           = dd.read_parquet(atac_data_file)
    logging.info("\n  All dataframes loaded")

    # Compute mean TF and TG expression
    logging.info("\n  - Calculating average TF and TG expression")
    dd_rna_tf, dd_rna_tg = compute_expression_means(rna_df)

    # Compute mean peak accessibility
    logging.info("  - Calculating average peak accessibility")
    dd_atac = compute_atac_mean(atac_df)
    
    # Extract edges from DFs
    logging.info("\n ============== Finding Unique TF-peak-TG Edges ==============")
    logging.info("  - (1/4) Extracting Sliding Window TF to peak edges")
    sliding_window_edges = extract_edges(sliding_window_dd, ["source_id", "peak_id"])
    logging.info("  - (2/4) Extracting Homer TF to peak edges")
    homer_edges           = extract_edges(homer_dd,          ["source_id", "peak_id"])
    logging.info("  - (3/4) Extracting Peak to TG correlation edges")
    peak_corr_edges       = extract_edges(peak_corr_dd,      ["peak_id", "target_id"])
    logging.info("  - (4/4) Extracting Cicero Peak to TG edges")
    cicero_edges          = extract_edges(cicero_dd,         ["peak_id", "target_id"])

    # Combine edges
    logging.info('\n  - Building set of unique TF-peak-TG edges')
    tf_peak_edges  = sliding_window_edges | homer_edges
    peak_tg_edges  = peak_corr_edges | cicero_edges
    full_edges     = build_full_edges(tf_peak_edges, peak_tg_edges)
    logging.info(f"  Done! Found {len(full_edges)} full edges")
    
    logging.info("\n ============== Merging DataFrames ==============")
    
    # suppose full_edges is a set of (source_id, peak_id, target_id)
    full_edges_df = pd.DataFrame(list(full_edges),
                                columns=["source_id","peak_id","target_id"])
    # pick a reasonable partition count
    full_edges_dd = dd.from_pandas(full_edges_df, npartitions=32)
    
    # 1) First, fuse the two TF→peak merges into one shuffle
    tf_scores_dd = (
        sliding_window_dd
        .merge(homer_dd, how="outer", on=["source_id","peak_id"])
    )

    # 2) Fuse the two peak→TG merges into one shuffle
    ptg_scores_dd = (
        peak_corr_dd
        .merge(cicero_dd, how="outer", on=["peak_id","target_id"])
    )

    # 3) Build your full_edges (however you prefer—set‑based or Dask join)
    #    Assume `full_edges` is your big Dask DataFrame of (source_id,peak_id,target_id)

    # 4) Merge those two big score tables onto full_edges (2 shuffles)
    combined_ddf = (
        full_edges_dd
        .merge(tf_scores_dd,  how="left", on=["source_id","peak_id"])
        .merge(ptg_scores_dd, how="left", on=["peak_id","target_id"])
    )

    tf_expr_pdf = dd_rna_tf
    tg_expr_pdf = dd_rna_tg
    atac_pdf    = dd_atac

    # 6) Broadcast‑join those into each partition (zero additional shuffles)
    def _merge_pdf(df, pdf, on):
        return df.merge(pdf, how="left", on=on)

    combined_ddf = combined_ddf.map_partitions(_merge_pdf, tf_expr_pdf, on="source_id")
    combined_ddf = combined_ddf.map_partitions(_merge_pdf, tg_expr_pdf, on="target_id")
    combined_ddf = combined_ddf.map_partitions(_merge_pdf, atac_pdf,    on="peak_id")
    
    logging.info("\n ============== Adding STRING Scores ==============")
    combined_string_ddf = add_string_db_scores(combined_ddf, args.string_dir, full_edges_dd)
    
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
         )
    logging.info("      Done!")
    
    logging.info(f'  - Melting the combined DataFrame to reduce NaN values')
    melted_df = filtered_combined_string_ddf.melt(
        id_vars=["source_id", "peak_id", "target_id"],
        value_vars=score_cols,
        var_name="score_type",
        value_name="score_value"
    )

    non_null_scores_ddf = melted_df.dropna(subset=["score_value"])
    logging.info("      Done!")    
    
    plot_feature_score_histograms(non_null_scores_ddf, score_cols, output_dir)
    
    logging.info(f"Writing out inferred_score_df.parquet")
    pdf = non_null_scores_ddf.compute()
    pdf.to_parquet(
        os.path.join(inferred_grn_dir, "inferred_score_df.parquet"),
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    logging.info("  Done!")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    main()