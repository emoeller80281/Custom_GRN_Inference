import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
import os
import dask.dataframe as dd
import pandas as pd
import logging
from typing import Set, Tuple, Union
from tqdm import tqdm

from normalization import (
    minmax_normalize_pandas,
    minmax_normalize_dask,
    clip_and_normalize_log1p_dask
)

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

def build_scored_edges_dataframe(
    full_edges: Set[Tuple[str, str, str]],
    sliding_window_dd: dd.DataFrame,
    homer_dd: dd.DataFrame,
    peak_corr_dd: dd.DataFrame,
    cicero_dd: dd.DataFrame,
    rna_tf_dd: dd.DataFrame,
    rna_tg_dd: dd.DataFrame,
    atac_dd: dd.DataFrame
) -> dd.DataFrame:
    """Build merged scored edge table from component scores."""
    full_edges_df = pd.DataFrame(list(full_edges), columns=["source_id", "peak_id", "target_id"])
    full_edges_dd = dd.from_pandas(full_edges_df, npartitions=1)

    logging.info("  - (1/7) Adding Sliding Window TF to peak scores")
    df = full_edges_dd.merge(sliding_window_dd, on=["source_id", "peak_id"], how="left")
    logging.info("      Done!")

    logging.info("  - (2/7)Adding Homer TF to peak scores")
    df = df.merge(homer_dd, on=["source_id", "peak_id"], how="left")
    logging.info("      Done!")

    logging.info("  - (3/7)Adding Peak to TG correlation scores")
    df = df.merge(peak_corr_dd, on=["peak_id", "target_id"], how="left")
    logging.info("      Done!")

    logging.info("  - (4/7)Adding Cicero peak to TG scores")
    df = df.merge(cicero_dd, on=["peak_id", "target_id"], how="left")
    logging.info("      Done!")

    logging.info("  - (5/7)Adding mean TF expression scores")
    df = df.merge(rna_tf_dd, on="source_id", how="left")
    logging.info("      Done!")

    logging.info("  - (6/7)Adding mean TG expression")
    df = df.merge(rna_tg_dd, on="target_id", how="left")
    logging.info("      Done!")

    logging.info("  - (7/7) Adding mean peak accessibility")
    df = df.merge(atac_dd, on="peak_id", how="left")
    logging.info("      Done!")

    logging.info("\n  All merges complete. Returning final Dask DataFrame.")
    return df

def add_string_db_scores(inferred_net_dd, string_dir):
    # Load STRING metadata files (small)
    logging.info("  - Reading STRING protein info")
    protein_info_df = pd.read_csv(f"{string_dir}/protein_info.txt", sep="\t", header=0)

    logging.info("  - Reading STRING protein links detailed")
    protein_links_df = pd.read_csv(f"{string_dir}/protein_links_detailed.txt", sep=" ", header=0)

    # Map STRING protein IDs to human-readable names
    logging.info("  - Mapping STRING protein IDs to preferred names")
    id_to_name = protein_info_df.set_index("#string_protein_id")["preferred_name"].to_dict()
    protein_links_df["protein1"] = protein_links_df["protein1"].map(id_to_name)
    protein_links_df["protein2"] = protein_links_df["protein2"].map(id_to_name)

    # Select relevant STRING columns
    protein_links_df = protein_links_df.rename(columns={
        "experimental": "string_experimental_score",
        "textmining": "string_textmining_score",
        "combined_score": "string_combined_score"
    })[["protein1", "protein2", "string_experimental_score", "string_textmining_score", "string_combined_score"]]

    # Convert STRING links to Dask
    logging.info("  - Converting STRING links to Dask")
    protein_links_dd = dd.from_pandas(protein_links_df, npartitions=1)

    # Merge inferred network with STRING scores
    logging.info("  - Merging inferred network with STRING edges")
    merged_dd = inferred_net_dd.merge(
        protein_links_dd,
        left_on=["source_id", "target_id"],
        right_on=["protein1", "protein2"],
        how="left"
    ).drop(columns=["protein1", "protein2"])

    # Normalize STRING score columns
    logging.info("  - Normalizing STRING scores")
    cols_to_normalize = ["string_experimental_score", "string_textmining_score", "string_combined_score"]

    norm_string_ddf = clip_and_normalize_log1p_dask(
        ddf=merged_dd,
        score_cols=cols_to_normalize,
        quantiles=(0.05, 0.95),
        apply_log1p=True,
        dtype=np.float32
    )
    
    norm_string_ddf = minmax_normalize_dask(
        ddf=norm_string_ddf, 
        score_cols=cols_to_normalize, 
        dtype=np.float32
    )
    logging.info("  Done!")
    
    return norm_string_ddf

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
    sliding_window_dd = dd.read_parquet(f"{output_dir}/sliding_window_tf_to_peak_score.parquet").head(10000)
    
    logging.info("  - (2/6) Loading Homer DataFrame")
    homer_dd          = dd.read_parquet(f"{output_dir}/homer_tf_to_peak.parquet").head(10000)
    
    logging.info("  - (3/6) Loading Peak to TG Correlation DataFrame")
    peak_corr_dd      = dd.read_parquet(f"{output_dir}/peak_to_gene_correlation.parquet").head(10000)
    
    logging.info("  - (4/6) Loading Cicero DataFrame")
    cicero_dd         = dd.read_parquet(f"{output_dir}/cicero_peak_to_tg_scores.parquet").head(10000)
    
    logging.info("  - (5/6) Loading scRNAseq DataFrame")
    rna_df            = dd.read_parquet(rna_data_file).head(10000)
    
    logging.info("  - (6/6) Loading scATACseq DataFrame")
    atac_df           = dd.read_parquet(atac_data_file).head(10000)
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

    combined_ddf = build_scored_edges_dataframe(
        full_edges,
        sliding_window_dd,
        homer_dd,
        peak_corr_dd,
        cicero_dd,
        dd_rna_tf,
        dd_rna_tg,
        dd_atac
    )
    
    logging.info("\n ============== Adding STRING Scores ==============")
    combined_string_ddf = add_string_db_scores(combined_ddf, args.string_dir)
    
    logging.info("\n ============== Filtering and Melting Combined Score DataFrame ==============")
    score_cols = [
        "sliding_window_score", "homer_binding_score",
        "correlation", "TSS_dist_score", "cicero_score",
        "mean_TF_expression", "mean_TG_expression", "mean_peak_accessibility",
        "string_experimental_score", "string_textmining_score", "string_combined_score"
    ]
    
    num_score_col_threshold = 1
    
    logging.info(f"  - Filtering combined DataFrame to only contain edges with at least {num_score_col_threshold} / {len(score_cols)} scores")
    # Filter the combined dataframe to only contain rows with scores in num_score_col_threshold columns
    filtered_combined_string_ddf = filter_scored_edges(
        combined_string_ddf, 
         min_valid_scores=num_score_col_threshold,
         score_cols=score_cols
         )
    logging.info("      Done!")
    
    # Repartition to balance the partition size
    partition_size = "256MB"
    logging.info(f'  - Repartitioning the final Dask DataFrame to {partition_size}')
    filtered_combined_string_ddf = filtered_combined_string_ddf.repartition(partition_size=partition_size)
    logging.info("      Done!")
    
    logging.info(f'  - Melting the combined DataFrame to reduce NaN values')
    melted_ddf = filtered_combined_string_ddf.melt(
        id_vars=["source_id", "peak_id", "target_id"],
        value_vars=score_cols,
        var_name="score_type",
        value_name="score_value"
    )

    non_null_scores_ddf = melted_ddf.dropna(subset=["score_value"])
    logging.info("      Done!")    
    
    # Save the final combined network scores
    logging.info(f'  - Saving final combined score Dask DataFrame to the inferred GRN output directory')
    non_null_scores_ddf.to_parquet(
        os.path.join(inferred_grn_dir, "inferred_score_df.parquet"),
        engine="pyarrow",
        compression="snappy",
        write_index=False
    )
    logging.info("      Done!")
    
    
    
    logging.info("\nFinished")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    main()