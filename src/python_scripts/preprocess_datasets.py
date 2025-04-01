import numpy as np
import pandas as pd
import argparse
import logging
import os

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

    args: argparse.Namespace = parser.parse_args()
    return args

def log2_cpm_normalize(df):
    """
    Log2 CPM normalize the values for each gene / peak.
    Assumes:
      - The df's first column is a non-numeric peak / gene identifier (e.g., "chr1:100-200"),
      - columns 1..end are numeric count data for samples or cells.
    """
    # Separate the non-numeric first column
    row_ids = df.iloc[:, 0]
    
    # Numeric counts
    counts = df.iloc[:, 1:]
    
    # 1. Compute library sizes (sum of each column)
    library_sizes = counts.sum(axis=0)
    
    # 2. Convert counts to CPM
    # Divide each column by its library size, multiply by 1e6
    # Add 1 to avoid log(0) issues in the next step
    cpm = (counts.div(library_sizes, axis=1) * 1e6).add(1)
    
    # 3. Log2 transform
    log2_cpm = np.log2(cpm)
    
    # Reassemble into a single DataFrame
    normalized_df = pd.concat([row_ids, log2_cpm], axis=1)
    
    return normalized_df

def load_atac_dataset(atac_data_file: str) -> pd.DataFrame:
    atac_df = pd.read_csv(atac_data_file, sep=",", header=0, index_col=None)
    atac_df = atac_df.rename(columns={atac_df.columns[0]: "peak_id"})
    
    return atac_df

def load_rna_dataset(rna_data_file: str) -> pd.DataFrame:
    rna_df = pd.read_csv(rna_data_file, sep=",", header=0)
    rna_df = rna_df.rename(columns={rna_df.columns[0]: "gene_id"})
    
    return rna_df

def main(atac_data_file, rna_data_file):
    logging.info("Loading ATACseq dataset")
    atac_df = load_atac_dataset(atac_data_file)
    
    logging.info("Loading RNAseq dataset")
    rna_df = load_rna_dataset(rna_data_file)
    
    logging.info("Log2 CPM normalizing the datasets")
    atac_df_norm = log2_cpm_normalize(atac_df)
    rna_df_norm = log2_cpm_normalize(rna_df)
    
    logging.info("Updating filenames and saving processed datasets")
    def update_name(filename):
        base, ext = os.path.splitext(filename)
        return f"{base}_processed.csv"
    
    new_atac_file = update_name(atac_data_file)
    new_rna_file = update_name(rna_data_file)
    
    print(f"Updated ATAC file: {new_atac_file}")
    print(f"Updated RNA file: {new_rna_file}")
    
    atac_df_norm.to_csv(new_atac_file, sep=",", header=True, index=False)
    rna_df_norm.to_csv(new_rna_file, sep=",", header=True, index=False)
    logging.info("Done!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Parse command-line arguments
    args: argparse.Namespace = parse_args()
    atac_data_file: str = args.atac_data_file
    rna_data_file: str = args.rna_data_file

    # Run the main function
    main(atac_data_file, rna_data_file)