import numpy as np
import pandas as pd
import argparse
import logging
import os
from scipy import sparse

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

def log2_cpm_normalize(df: pd.DataFrame, id_col_name: str, label: str = "dataset") -> pd.DataFrame:
    if id_col_name not in df.columns:
        raise ValueError(f"Identifier column '{id_col_name}' not found in DataFrame.")

    # Split into ID column and numeric values
    id_col = df[[id_col_name]].reset_index(drop=True)
    counts = df.drop(columns=[id_col_name]).apply(pd.to_numeric, errors="coerce").fillna(0)

    # Combine and check normalization status
    full_df = pd.concat([id_col, counts], axis=1)
    if is_normalized(full_df):
        print(f" - {label} matrix appears already normalized. Skipping log2 CPM.", flush=True)
        return full_df

    print(f" - {label} matrix appears unnormalized. Applying log2 CPM normalization.", flush=True)

    # Compute log2 CPM
    library_sizes = counts.sum(axis=0)
    cpm = (counts.div(library_sizes, axis=1) * 1e6).add(1)
    log2_cpm = np.log2(cpm).reset_index(drop=True)

    return pd.concat([id_col, log2_cpm], axis=1)


def load_atac_dataset(atac_data_file: str) -> pd.DataFrame:
    if atac_data_file.lower().endswith('.parquet'):
        df = pd.read_parquet(atac_data_file)
        
    elif atac_data_file.lower().endswith('.csv'):
        df = pd.read_csv(atac_data_file, sep=",", header=0, index_col=None)
        
    elif atac_data_file.lower().endswith('.tsv'):
        df = pd.read_csv(atac_data_file, sep="\t", header=0, index_col=None)

        
    else:
        logging.error("ERROR: ATAC data file must be a csv, tsv, or parquet format. Check column separators")
        
    return df.rename(columns={df.columns[0]: "peak_id"})

def load_rna_dataset(rna_data_file: str) -> pd.DataFrame:
    if rna_data_file.lower().endswith('.parquet'):
        df = pd.read_parquet(rna_data_file)
        
    elif rna_data_file.lower().endswith('.csv'):
        df = pd.read_csv(rna_data_file, sep=",", header=0, index_col=None)
        
    elif rna_data_file.lower().endswith('.tsv'):
        df = pd.read_csv(rna_data_file, sep="\t", header=0, index_col=None)
        
    else:
        logging.error("ERROR: RNA data file must be a csv, tsv, or parquet format. Check column separators")
    
    return df.rename(columns={df.columns[0]: "gene_id"})

def deduplicate_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    seen = {}
    new_columns = []
    
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}.{seen[col]}")
    
    if len(new_columns) != len(set(new_columns)):
        raise ValueError(f"[{label}] Still found duplicates after deduplication attempt.")
    
    if df.columns.tolist() != new_columns:
        print(f"[{label}] Duplicate column names found. Renaming to make unique.")
        df.columns = new_columns
    else:
        print(f"[{label}] No duplicates found.")
    
    return df

def main(atac_data_file, rna_data_file):
    logging.info("Loading ATAC-seq dataset")
    atac_df = load_atac_dataset(atac_data_file)

    logging.info("Loading RNA-seq dataset")
    rna_df = load_rna_dataset(rna_data_file)

    logging.info("Checking and normalizing ATAC-seq data")
    atac_df_norm = log2_cpm_normalize(atac_df, label="scATAC-seq", id_col_name="peak_id")

    logging.info("Checking and normalizing RNA-seq data")
    rna_df_norm = log2_cpm_normalize(rna_df, label="scRNA-seq", id_col_name="gene_id")

    def update_name(filename):
        base, ext = os.path.splitext(filename)
        return f"{base}_processed.parquet"

    new_atac_file = update_name(atac_data_file)
    new_rna_file = update_name(rna_data_file)

    print(f"\nUpdated ATAC file: {new_atac_file}", flush=True)
    print(f"Updated RNA file: {new_rna_file}", flush=True)
    
    # atac_count_df = deduplicate_columns(atac_df_norm, label="ATAC")

    logging.info('\nWriting ATAC-seq dataset to Parquet')
    atac_df_norm.to_parquet(new_atac_file, engine="pyarrow", compression="snappy", index=False)
    logging.info("  Done!")
    
    # rna_count_df = deduplicate_columns(rna_df_norm, label="RNA")
    
    logging.info('\nWriting RNA-seq dataset to Parquet')
    rna_df_norm.to_parquet(new_rna_file, engine="pyarrow", compression="snappy", index=False)
    logging.info("  Done!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args()
    main(args.atac_data_file, args.rna_data_file)
