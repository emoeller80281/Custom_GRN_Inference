import pandas as pd
import numpy as np
import logging
import csv
import os
import pybedtools
from pandas.api.types import is_numeric_dtype

from typing import Union

def read_ground_truth(ground_truth_file):
    logging.info("Reading in the ground truth")
    ground_truth = pd.read_csv(ground_truth_file, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    ground_truth = ground_truth.rename(columns={"Source": "source_id", "Target": "target_id"})
    return ground_truth

def label_edges_with_ground_truth(df_to_label: pd.DataFrame, ground_truth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a "label" column with 1 if the TF-TG edge is present in the ground truth, else 0.
    
    Both `df_to_label` and `ground_truth_df` must have the columns `"source_id"` and `"target_id"`.
    Creates a "label" column with values of 1 for TF-TG edges that are shared between the `df_to_label`
    and `ground_truth_df` and values of 0 for edges in `df_to_label` which are not in `ground_truth_df`.
    

    Args:
        df_to_label (pd.DataFrame): DataFrame containing TF-TG edges in `"source_id"` `"target_id"` 
            gene name columns.
        ground_truth_df (pd.DataFrame): Ground truth DataFrame containing True TF-TG edges in `"source_id"` `"target_id"`
            gene name columns.

    Returns:
        labeled_df (pd.DataFrame): 
            Pandas DataFrame containing a `"label"` column. The `"source_id"` and `"target_id"` columns are
            converted to uppercase.
    """
    
    # Catch if the TF column or TG columns are missing
    required_cols = ["source_id", "target_id"]
    for col in required_cols:
        if col not in ground_truth_df.columns:
            raise KeyError(f"'{col}' column not found in the ground truth DataFrame")
    
        if col not in df_to_label.columns:
            raise KeyError(f"'{col}' column not found in the ground truth DataFrame")
    
    logging.info("Creating ground truth set")
    ground_truth_pairs = set(zip(
        ground_truth_df["source_id"].str.upper(),
        ground_truth_df["target_id"].str.upper()
    ))

    logging.info("Adding labels to inferred network")
    
    df_to_label["source_id"] = df_to_label["source_id"].str.upper()
    df_to_label["target_id"] = df_to_label["target_id"].str.upper()

    def label_partition(df):
        if isinstance(df, pd.Series):
            raise ValueError("Expected DataFrame but got Series — check apply usage.")
        df = df.copy()  # <-- avoids SettingWithCopyWarning
        tf_tg_tuples = list(zip(df["source_id"], df["target_id"]))
        df.loc[:, "label"] = [1 if pair in ground_truth_pairs else 0 for pair in tf_tg_tuples]
        return df

    labeled_df = label_partition(df_to_label)

    return labeled_df

def merge_dataset_with_ground_truth(
    df: pd.DataFrame, 
    ground_truth: pd.DataFrame, 
    tf_col_name: str = "source_id",
    tg_col_name: str = "target_id",
    gt_tf_col_name: str = "source_id",
    gt_tg_col_name: str = "target_id",
    method:str="", 
    gt_name: str="", 
    show_network_size: bool=False
    ):
    df = df.copy()
    ground_truth = ground_truth.copy()
    
    df = df.rename(columns={tf_col_name: "source_id", tg_col_name: "target_id"})
    ground_truth = ground_truth.rename(columns={gt_tf_col_name: "source_id", gt_tg_col_name: "target_id"})
    
    df['source_id'] = df['source_id'].str.capitalize()
    df['target_id'] = df['target_id'].str.capitalize()
    
    shared_sources = set(df['source_id']) & set(ground_truth['source_id'])
    shared_targets = set(df['target_id']) & set(ground_truth['target_id'])

    df_filtered = df[
        df['source_id'].isin(shared_sources) &
        df['target_id'].isin(shared_targets)
    ]

    gt_filtered = ground_truth[
        ground_truth['source_id'].isin(shared_sources) &
        ground_truth['target_id'].isin(shared_targets)
    ]
    
    df_merged = pd.merge(df_filtered, gt_filtered, on=['source_id', 'target_id'], how='outer', indicator=True)
    
    if show_network_size:
        if len(method) == 0:
            method = "sliding window"
        if len(gt_name) == 0:
            gt_name = "ground truth"
        
        print(f"- **Overlap between {method} and {gt_name} edges**")
        
        edges_in_df_and_ground_truth = df_merged[df_merged["_merge"] == "both"].drop(columns="_merge")
        df_not_ground_truth_edges = df_merged[df_merged["_merge"] == "left_only"].drop(columns="_merge")
        ground_truth_edges_only = df_merged[df_merged["_merge"] == "right_only"].drop(columns="_merge")
        
        tfs_in_both = edges_in_df_and_ground_truth["source_id"].drop_duplicates()
        tgs_in_both = edges_in_df_and_ground_truth["target_id"].drop_duplicates()
        
        print(f"\t- **Both {gt_name} and {method}**")
        print(f"\t\t- TFs: {len(tfs_in_both):,}")
        print(f"\t\t- TGs: {len(tgs_in_both):,}")
        print(f"\t\t- TF-TG Edges: {len(edges_in_df_and_ground_truth.drop_duplicates(subset=['source_id', 'target_id'])):,}")
        
        tfs_only_in_sliding_window = df[~df["source_id"].isin(ground_truth["source_id"])]["source_id"].drop_duplicates()
        tgs_only_in_sliding_window = df[~df["target_id"].isin(ground_truth["target_id"])]["target_id"].drop_duplicates()

        print(f"\t- **Only {method.capitalize()}**")
        print(f"\t\t- TFs: {len(tfs_only_in_sliding_window):,}")
        print(f"\t\t- TGs: {len(tgs_only_in_sliding_window):,}")
        print(f"\t\t- TF-TG Edges: {len(df_not_ground_truth_edges.drop_duplicates(subset=['source_id', 'target_id'])):,}")
        
        tfs_only_in_ground_truth = ground_truth[~ground_truth["source_id"].isin(df["source_id"])]["source_id"].drop_duplicates()
        tgs_only_in_ground_truth = ground_truth[~ground_truth["target_id"].isin(df["target_id"])]["target_id"].drop_duplicates()

        print(f"\t- **Only {gt_name}**")
        print(f"\t\t- TFs: {len(tfs_only_in_ground_truth):,}")
        print(f"\t\t- TGs: {len(tgs_only_in_ground_truth):,}")
        print(f"\t\t- Edges: {len(ground_truth_edges_only):,}")
    
    df_merged["label"] = df_merged["_merge"] == "both"
    
    df_labeled = df_merged.drop(columns=["_merge"])
    
    df_labeled = df_labeled.rename(columns={"source_id": tf_col_name, "target_id": tg_col_name})
    
    return df_labeled

def load_dataset(dataset_file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from a csv, tsv, or parquet file.
    
    **csv and tsv files**:
    - header = row 0
    - index = None

    Args:
        dataset_file_path (str): Path to the dataset file to open.

    Raises:
        ValueError: Data file must be .csv, .tsv, or parquet
        RuntimeError: Raises an error if the resulting DataFrame is empty

    Returns:
        df (pd.DataFrame): DataFrame object loaded from the `dataset_file_path`
    """
    assert os.path.isfile(dataset_file_path), \
        "`datset_file_path` must be an existing file."
    
    assert dataset_file_path.lower().endswith((".csv", ".tsv", ".parquet")), \
        "`dataset_file_path` must end with .csv, .tsv. or .parquet"
    
    df: pd.DataFrame = pd.DataFrame()
    if dataset_file_path.lower().endswith('.parquet'):
        df = pd.read_parquet(dataset_file_path)
        
    elif dataset_file_path.lower().endswith('.csv'):
        df = pd.read_csv(dataset_file_path, sep=",", header=0, index_col=None)
        
    elif dataset_file_path.lower().endswith('.tsv'):
        df = pd.read_csv(dataset_file_path, sep="\t", header=0, index_col=None)
        
    else:
        raise ValueError(f"Data file must be .csv, .tsv or .parquet: got {dataset_file_path}")
    
    if df.empty:
        raise RuntimeError(f"Failed to load data file: {dataset_file_path}")
    
    return df

def load_atac_dataset(atac_data_file: str) -> pd.DataFrame:
    """
    Loads an ATAC-seq dataset from a csv, tsv, or parquet file
    
    - **Cell Names** should be in the **first row**, set as header
    - **Peak Location** should be in the **first column**
    - No index column

    Args:
        atac_data_file (str): Path to the scATAC-seq csv, tsv, or parquet file

    Returns:
        df (pd.DataFrame): 
            DataFrame where column 0 = `"peak_id"` and row 0 as the header
    """
    
    df = load_dataset(atac_data_file)
    
    df = df.rename(columns={df.columns[0]: "peak_id"})
    
    logging.info(f'\tNumber of peaks: {df.shape[0]}')
    logging.info(f'\tNumber of cells: {df.shape[1]-1}')
    
    return df

def load_rna_dataset(rna_data_file: str) -> pd.DataFrame:
    """
    Loads an RNA-seq dataset from a csv, tsv, or parquet file.
    
    - **Cell names** should be in the **first row**, set as header
    - **Gene names** should be in the **first column**
    - No index column

    Args:
        rna_data_file (str): Path to the scRNA-seq csv, tsv, or parquet file

    Returns:
        df (pd.DataFrame): 
            DataFrame where column 0 = `"gene_id"` and row 0 as the header
    """
    
    df = load_dataset(rna_data_file)
    
    df = df.rename(columns={df.columns[0]: "gene_id"})
    
    return df

def load_and_pivot_melted_score_dataframe(melted_score_file_path: str) -> pd.DataFrame:
    """
    Loads a melted score DataFrame and pivots it to a dense DataFrame.
    
    **Required Columns**
    1) `"source_id"`: Transcription factor gene names
    2) `"peak_id"`: ATAC-seq peak names
    3) `"target_id"`: target gene names
    4) `"score_type"`: Name of the feature score
    5) `"score_value"`: Feature score values
    
    Args:
        melted_score_file_path (str):
            Path to the melted dataframe .csv, .tsv, or .parquet file. For csv or tsv,
            assumes header = row 0 and no index column
            
    Returns:
        wide_df (pd.DataFrame):
            Score DataFrame in wide format, with columns `"source_id"`, `"peak_id"`, `"target_id"`, \
            followed by a column for each unique `"score_type"`
    """
    melted_df = load_dataset(melted_score_file_path)
    
    required_cols = {"source_id", "peak_id", "target_id", "score_type", "score_value"}
    missing_cols = required_cols - set(melted_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    grouped: pd.DataFrame = (
        melted_df
        .groupby(["source_id", "peak_id", "target_id", "score_type"])
        ["score_value"]
        .mean()
        .reset_index()
    )

    wide_df = grouped.pivot(
        index=["source_id", "peak_id", "target_id"],
        columns="score_type",
        values="score_value",
    ).reset_index()
    
    # Ensures that the edge columns come before the feature columns
    edge_cols = ["source_id", "peak_id", "target_id"]
    score_cols = [col for col in wide_df.columns if col not in edge_cols]
    wide_df = wide_df[edge_cols + score_cols]
    
    # Ensures that the score columns are numeric
    wide_df[score_cols] = wide_df[score_cols].apply(pd.to_numeric, errors="coerce")

    return wide_df

def format_peaks(peak_ids: Union[pd.Series, pd.Index]) -> pd.DataFrame:
    """
    Splits peaks from `chrN:start-end` format into a DataFrame.
    
    Creates a dataframe with the following columns:
    1) "peak_id": peakN+1 where N is the index position of the peak
    2) "chromosome": chrN
    3) "start"
    4) "end"
    5) "strand": List of "." values, we dont have strand information for our peaks.
    
    Args:
        peak_ids (pd.Series):
            Series containing the peak locations in "chrN:start-end" format.
            
    Returns:
        peak_df (pd.DataFrame):
            DataFrame of peak locations in the correct format for Homer and the sliding window method
    """
    if peak_ids.empty:
        raise ValueError("Input peak ID list is empty.")
    
    peak_ids = peak_ids.drop_duplicates()

    logging.info(f'Formatting {peak_ids.shape[0]} peaks')

    # Extract chromosome, start, and end from peak ID strings
    try:
        chromosomes = peak_ids.str.extract(r'([^:]+):')[0]
        starts = peak_ids.str.extract(r':(\d+)-')[0]
        ends = peak_ids.str.extract(r'-(\d+)$')[0]
    except Exception as e:
        raise ValueError(f"Error parsing 'peak_id' values: {e}")

    if chromosomes.isnull().any() or starts.isnull().any() or ends.isnull().any():
        raise ValueError("Malformed peak IDs. Expect format 'chr:start-end'.")

    peak_df = pd.DataFrame({
        # "peak_id": [f"peak{i + 1}" for i in range(len(peak_ids))],
        "chromosome": chromosomes,
        "start": pd.to_numeric(starts, errors='coerce').astype(int),
        "end": pd.to_numeric(ends, errors='coerce').astype(int),
        "strand": ["."] * len(peak_ids)
    })
    
    peak_df["peak_id"] = (
        peak_df["chromosome"].astype(str) + ":" +
        peak_df["start"].astype(str) + "-" +
        peak_df["end"].astype(str)
    )
    
    return peak_df

def find_peak_length(peak_id_col: pd.Series) -> pd.Series:    
    """
    Finds the base pair lengths for a Series of genomic ranges in chr:start-end format.

    Args:
        peak_id_col (pd.Series): Series of genomic locations in chr:start-end format.

    Returns:
        pd.Series: base pair lengths of the genomic ranges proviced.
    """
    peak_col_split = peak_id_col.str.extract(r'(chr[\w]+):([0-9]+)-([0-9]+)').dropna()
    return np.abs(peak_col_split[2].astype(int) - peak_col_split[1].astype(int))

def find_genes_near_peaks(
    peak_bed: pybedtools.BedTool, 
    tss_bed: pybedtools.BedTool, 
    tss_distance_cutoff: Union[int, float] = 1e6
    ):
    """
    Identify genes whose transcription start sites (TSS) are near scATAC-seq peaks.
    
    This function:
        1. Uses BedTools to find peaks that are within tss_distance_cutoff bp of each gene's TSS.
        2. Converts the BedTool result to a pandas DataFrame.
        3. Computes the absolute distance between the peak end and gene start (as a proxy for TSS distance).
        
    Args:
        peak_bed (pybedtools.BedTool):
            BedTool object representing scATAC-seq peaks.
        tss_bed (pybedtools.BedTool):
            BedTool object representing gene TSS locations.
        tss_distance_cutoff (int): 
            The maximum distance (in bp) from a TSS to consider a peak as potentially regulatory.
        
    Returns:
        peak_tss_subset_df (pandas.DataFrame): 
            A DataFrame containing columns "peak_id", "target_id", and the scaled TSS distance "TSS_dist"
            for peak–gene pairs.
    """
    
    logging.info(f"Locating peaks that are within {tss_distance_cutoff} bp of each gene's TSS")
    peak_tss_overlap = peak_bed.window(tss_bed, w=tss_distance_cutoff)
    
    
    # Define the column types for conversion to DataFrame
    dtype_dict = {
        "peak_chr": str,
        "peak_start": int,
        "peak_end": int,
        "peak_id": str,
        "gene_chr": str,
        "gene_start": int,
        "gene_end": int,
        "gene_id": str
    }
    
    # Convert the BedTool result to a DataFrame for further processing.
    peak_tss_overlap_df = peak_tss_overlap.to_dataframe(
        names = [
            "peak_chr", "peak_start", "peak_end", "peak_id",
            "gene_chr", "gene_start", "gene_end", "gene_id"
        ],
        dtype=dtype_dict,
        low_memory=False  # ensures the entire file is read in one go
    ).rename(columns={"gene_id": "target_id"}).dropna()
        
    # Calculate the absolute distance in basepairs between the peak's end and gene's start.
    distances = np.abs(peak_tss_overlap_df["peak_end"].values - peak_tss_overlap_df["gene_start"].values)
    peak_tss_overlap_df["TSS_dist"] = distances
    
    # Sort by the TSS distance (lower values imply closer proximity and therefore stronger association)
    peak_tss_overlap_df = peak_tss_overlap_df.sort_values("TSS_dist")
    
    return peak_tss_overlap_df

def minmax_norm_col(scores: pd.Series) -> pd.Series:
    """
    Min-max normalizes the scores to a value between 0-1 without changing the shape of the distribution.

    Args:
        scores (pd.Series): Score column or series to be normalized.

    Returns:
        pd.Series: Min-max normalized score column.
    """
    
    assert is_numeric_dtype(scores), \
        f"scores are not numeric, type {scores.dtype}"
    
    return (scores - min(scores)) / (max(scores) - min(scores))

def calculate_summed_tf_tg_score(
    df: pd.DataFrame, 
    score_col: str, 
    tf_col_name: str = "source_id",
    tg_col_name: str = "target_id"
    ) -> pd.DataFrame:
    # Group by TF and sum all scores
    sum_of_tf_peaks = (
        df
        .groupby(tf_col_name)[score_col]
        .sum()
        .reset_index()
        .rename(columns={score_col:"total_tf_score"})
        )
    
    # Group by TF-TG edge and sum for all peaks for that edge
    sum_of_tf_tg_peak_scores = (
        df
        .groupby([tf_col_name, tg_col_name])[score_col]
        .sum()
        .reset_index()
        .rename(columns={score_col:"tf_to_tg_peak_scores_summed"})
        )
    
    # Merge the total TF peaks and summed TF-TG edges
    sum_calculation_df = pd.merge(
        sum_of_tf_tg_peak_scores, 
        sum_of_tf_peaks, 
        how="left", 
        on=tf_col_name
        )
    
    
    sum_calculation_df[score_col] = (
        sum_calculation_df["tf_to_tg_peak_scores_summed"] / sum_calculation_df["total_tf_score"]
        ) * 1e6
    
    return sum_calculation_df

def calculate_tf_peak_tg_score(
    df: pd.DataFrame, 
    score_col: str, 
    tf_col_name: str = "source_id",
    ) -> pd.DataFrame:
    # Group by TF and sum all sliding window scores
    sum_of_tf_peaks = (
        df
        .groupby(tf_col_name)[score_col]
        .sum()
        .reset_index()
        .rename(columns={score_col:"total_tf_score"})
        )
    
    # Merge the total TF peaks with the TF-peak-TG scores
    individual_calculation_df = pd.merge(
        df, 
        sum_of_tf_peaks, 
        how="left", 
        on=tf_col_name
        ).rename(columns={score_col:"tf_peak_tg_score"})
    
    # Calculate the final sliding window score by dividing each TF-peak-TG score by the sum of scores for the TF
    individual_calculation_df[score_col] = (
        individual_calculation_df["tf_peak_tg_score"] / individual_calculation_df["total_tf_score"]
        ) * 1e6
    
    return individual_calculation_df

def set_tg_as_closest_gene_tss(df: pd.DataFrame, peaks_gene_distance_file: str):
    
    assert "peak_id" in df.columns, \
        f"'peak_id' column not in df. Columns: {df.columns}"
    
    # Read in the peaks to TG data and pick the closest gene for each peak (maximum TSS distance score)
    peaks_near_genes_df = pd.read_parquet(peaks_gene_distance_file, engine="pyarrow")
    
    assert "target_id" in peaks_near_genes_df.columns, \
        f"'target_id' column not in peaks_gene_distance_file DataFrame. Columns: {peaks_near_genes_df.columns}"
        
    assert "TSS_dist_score" in peaks_near_genes_df.columns, \
        f"'TSS_dist_score' column not in peaks_gene_distance_file DataFrame. Columns: {peaks_near_genes_df.columns}"

    closest_gene_to_peak_df = peaks_near_genes_df.sort_values("TSS_dist_score", ascending=False).groupby("peak_id").first()
    closest_gene_to_peak_df = closest_gene_to_peak_df[["target_id"]].reset_index()

    # Set the TG for each TF-peak edge as the closest gene to the peak
    sliding_window_closest_gene_df = pd.merge(df, closest_gene_to_peak_df, on=["peak_id"], how="left")
    return sliding_window_closest_gene_df

def set_tg_using_mira_peak_tg_edges(raw_sliding_window_scores: pd.DataFrame, mira_df: pd.DataFrame):
    sliding_window_mira_df = pd.merge(raw_sliding_window_scores, mira_df, on=["peak_id"], how="left")
    sliding_window_mira_df = sliding_window_mira_df[["source_id", "peak_id", "target_id", "sliding_window_score"]].dropna(subset="target_id")
    
    return sliding_window_mira_df

def set_tg_using_cicero_peak_tg_edges(raw_sliding_window_scores: pd.DataFrame, cicero_df: pd.DataFrame):
    sliding_window_cicero_df = pd.merge(raw_sliding_window_scores, cicero_df, on=["peak_id"], how="left")
    sliding_window_cicero_df = sliding_window_cicero_df[["source_id", "peak_id", "target_id", "sliding_window_score"]].dropna(subset="target_id")
    
    return sliding_window_cicero_df