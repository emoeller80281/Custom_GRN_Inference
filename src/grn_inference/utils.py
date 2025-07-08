import pandas as pd
import numpy as np
import logging
import csv
import os
import pybedtools

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
    
    df_to_label["source_id"].str.upper()
    df_to_label["target_id"].str.upper()

    def label_partition(df):
        df = df.copy()  # <-- avoids SettingWithCopyWarning
        tf_tg_tuples = list(zip(df["source_id"], df["target_id"]))
        df.loc[:, "label"] = [1 if pair in ground_truth_pairs else 0 for pair in tf_tg_tuples]
        return df

    labeled_df = df_to_label.apply(label_partition)

    return labeled_df

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

def format_peaks(peak_ids: pd.Series) -> pd.DataFrame:
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
        "peak_id": [f"peak{i + 1}" for i in range(len(peak_ids))],
        "chromosome": chromosomes,
        "start": pd.to_numeric(starts, errors='coerce').astype(int),
        "end": pd.to_numeric(ends, errors='coerce').astype(int),
        "strand": ["."] * len(peak_ids)
    })
    
    return peak_df

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