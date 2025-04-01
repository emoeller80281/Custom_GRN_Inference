import numpy as np
import pandas as pd
import argparse
import logging

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
        help="Path for the processed RNAseq and ATACseq files"
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

def main(atac_data_file, rna_data_file, output_dir):
    atac_df = load_atac_dataset(atac_data_file)
    rna_df = load_rna_dataset(rna_data_file)
    
    atac_df_norm = log2_cpm_normalize(atac_df)
    rna_df_norm = log2_cpm_normalize(rna_df)
    
    atac_df_norm.to_csv(f'{output_dir}/atac_df_processed.tsv', sep="\t", header=True, index=False)
    rna_df_norm.to_csv(f'{output_dir}/rna_df_processed.tsv', sep="\t", header=True, index=False)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Parse command-line arguments
    args: argparse.Namespace = parse_args()
    atac_data_file: str = args.atac_data_file
    rna_data_file: str = args.rna_data_file
    output_dir: str = args.output_dir

    # Run the main function
    main(atac_data_file, rna_data_file, output_dir)