import pybedtools
import pandas as pd
import numpy as np
from typing import Union
import logging

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