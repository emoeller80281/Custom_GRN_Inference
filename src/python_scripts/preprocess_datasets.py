import numpy as np
import pandas as pd
import argparse
import logging
import os

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument("--atac_data_file", type=str, required=True, help="Path to the scATAC-seq dataset")
    parser.add_argument("--rna_data_file", type=str, required=True, help="Path to the scRNA-seq dataset")
    return parser.parse_args()

def is_normalized(df: pd.DataFrame, threshold: float = 1.5) -> bool:
    """
    Heuristically check if the dataset appears normalized.
    """
    values = df.iloc[:, 1:].values
    if not np.issubdtype(values.dtype, np.floating):
        return False
    mean_val = values[values > 0].mean()
    return 0 < mean_val < threshold

def log2_cpm_normalize(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Normalize the data if it doesn't appear already normalized.
    Assumes first column contains row identifiers.
    """
    row_ids = df.iloc[:, 0]
    counts = df.iloc[:, 1:]

    if is_normalized(df):
        print(f" - {label} matrix appears already normalized. Skipping log2 CPM.", flush=True)
        return df

    print(f" - {label} matrix appears unnormalized. Applying log2 CPM normalization.", flush=True)

    library_sizes = counts.sum(axis=0)
    cpm = (counts.div(library_sizes, axis=1) * 1e6).add(1)
    log2_cpm = np.log2(cpm)

    return pd.concat([row_ids, log2_cpm], axis=1)

def load_atac_dataset(atac_data_file: str) -> pd.DataFrame:
    df = pd.read_csv(atac_data_file, sep=",", header=0, index_col=None)
    return df.rename(columns={df.columns[0]: "peak_id"})

def load_rna_dataset(rna_data_file: str) -> pd.DataFrame:
    df = pd.read_csv(rna_data_file, sep=",", header=0)
    return df.rename(columns={df.columns[0]: "gene_id"})

def main(atac_data_file, rna_data_file):
    logging.info("Loading ATAC-seq dataset")
    atac_df = load_atac_dataset(atac_data_file)

    logging.info("Loading RNA-seq dataset")
    rna_df = load_rna_dataset(rna_data_file)

    logging.info("Checking and normalizing ATAC-seq data")
    atac_df_norm = log2_cpm_normalize(atac_df, label="scATAC-seq")

    logging.info("Checking and normalizing RNA-seq data")
    rna_df_norm = log2_cpm_normalize(rna_df, label="scRNA-seq")

    def update_name(filename):
        base, ext = os.path.splitext(filename)
        return f"{base}_processed.csv"

    new_atac_file = update_name(atac_data_file)
    new_rna_file = update_name(rna_data_file)

    print(f"Updated ATAC file: {new_atac_file}", flush=True)
    print(f"Updated RNA file: {new_rna_file}", flush=True)

    logging.info('\nWriting updated scATAC-seq dataset')
    atac_df_norm.to_csv(new_atac_file, sep=",", header=True, index=False)
    logging.info("  Done!")
    
    logging.info('\nWriting updated scRNA-seq dataset')
    rna_df_norm.to_csv(new_rna_file, sep=",", header=True, index=False)
    logging.info("  Done!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args()
    main(args.atac_data_file, args.rna_data_file)
