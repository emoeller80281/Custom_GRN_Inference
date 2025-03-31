import pandas as pd
import argparse
import logging
from typing import Any

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
    argparse.Namespace: Parsed arguments containing paths for input and output files.
    """
    parser = argparse.ArgumentParser(description="Create a formatted peak file for Homer")
    parser.add_argument(
        "--atac_data_file",
        type=str,
        required=True,
        help="Path to the directory containing Homer TF motif scores from Homer 'annotatePeaks.pl'"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
    )
    
    args: argparse.Namespace = parser.parse_args()

    return args

def convert_to_homer_peak_format(atac_data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts an ATAC-seq dataset into HOMER-compatible peak format.

    Parameters:
    atac_data (pd.DataFrame): DataFrame containing ATAC-seq data with a 'peak_id' column.

    Returns:
    pd.DataFrame: A DataFrame in HOMER-compatible format.
    """
    # Validate that the input DataFrame has the expected structure
    if atac_data.empty:
        raise ValueError("Input ATAC-seq data is empty.")
    
    # Extract the peak ID column
    peak_ids: pd.Series = atac_data.iloc[:, 0]

    # Split peak IDs into chromosome, start, and end
    try:
        chromosomes: pd.Series = peak_ids.str.extract(r'([^:]+):')[0]
        starts: pd.Series = peak_ids.str.extract(r':(\d+)-')[0]
        ends: pd.Series = peak_ids.str.extract(r'-(\d+)$')[0]
    except Exception as e:
        raise ValueError(f"Error parsing 'peak_id' values: {e}")

    # Check for missing or invalid values
    if chromosomes.isnull().any() or starts.isnull().any() or ends.isnull().any():
        raise ValueError("One or more peak IDs are malformed. Ensure all peak IDs are formatted as 'chr:start-end'.")

    # Create a dictionary for constructing the HOMER-compatible DataFrame
    homer_dict: dict[str, Any] = {
        "peak_id": [f"peak{i + 1}" for i in range(len(peak_ids))],  # Generate unique peak IDs
        "chromosome": chromosomes,
        "start": pd.to_numeric(starts, errors='coerce'),  # Convert to numeric and handle errors
        "end": pd.to_numeric(ends, errors='coerce'),      # Convert to numeric and handle errors
        "strand": ["."] * len(peak_ids),                 # Set strand as "."
    }

    # Construct the DataFrame
    homer_df: pd.DataFrame = pd.DataFrame(homer_dict)

    # Final validation: Ensure no NaN values in the critical columns
    if homer_df[['chromosome', 'start', 'end']].isnull().any().any():
        raise ValueError("Parsed values contain NaNs. Check input 'peak_id' format.")

    return homer_df

# ----- Input -----
def main() -> None:
    """
    Main function to read scATACseq data, convert it to HOMER peak format, and save the results.
    """
    # Parse arguments
    args: argparse.Namespace = parse_args()
    atac_data_file: str = args.atac_data_file
    output_dir: str = args.output_dir

    logging.info('Reading scATACseq data')
    atac_data: pd.DataFrame = pd.read_csv(atac_data_file)
    logging.info(atac_data.head())

    logging.info('Converting scATACseq peaks to Homer peak format')
    homer_df: pd.DataFrame = convert_to_homer_peak_format(atac_data)

    logging.info('Saving Homer peak file')
    homer_df.to_csv(f'{output_dir}/homer_peaks.txt', sep='\t', header=False, index=False)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    main()
