import argparse
import logging
import os
from typing import Any

import pandas as pd

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
    argparse.Namespace: Parsed arguments containing paths for input and output files.
    """
    parser = argparse.ArgumentParser(description="Create a formatted peak file for Homer")

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
    )
    
    args: argparse.Namespace = parser.parse_args()

    return args

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

# ----- Input -----
def main() -> None:
    """
    Main function to read scATACseq data, convert it to HOMER peak format, and save the results.
    """
    # Parse arguments
    args: argparse.Namespace = parse_args()
    output_dir: str = args.output_dir
    
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    logging.info(f"Reading Peaks and Genes from 'peaks_near_genes.parquet'")
    assert os.path.isfile(os.path.join(output_dir, "peaks_near_genes.parquet")), FileNotFoundError("peaks_near_genes.parquet not found in output_dir")
    
    peaks_near_genes_df: pd.DataFrame = pd.read_parquet(os.path.join(output_dir, "peaks_near_genes.parquet"))
    peak_ids = peaks_near_genes_df["peak_id"].drop_duplicates()

    logging.info('Converting scATACseq peaks to Homer peak format')
    homer_df: pd.DataFrame = format_peaks(peak_ids)

    logging.info('Saving Homer peak file')
    homer_df.to_csv(f'{tmp_dir}/homer_peaks.txt', sep='\t', header=False, index=False)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    main()
