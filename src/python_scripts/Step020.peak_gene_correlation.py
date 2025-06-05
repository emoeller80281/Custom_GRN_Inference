import numpy as np
import pandas as pd
import argparse
import logging
import os
import pyarrow.parquet as pq
import pyarrow as pa

import pyranges as pr
from pybiomart import Server
from scipy import sparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument("--atac_data_file", type=str, required=True, help="Path to the scATAC-seq dataset")
    parser.add_argument("--rna_data_file", type=str, required=True, help="Path to the scRNA-seq dataset")
    parser.add_argument("--organism", type=str, default="mmusculus", 
                        help="Ensembl organism prefix (e.g. mmusculus for mouse) for TSS lookup")
    parser.add_argument("--tss_distance", type=int, default=1_000_000,
                        help="Distance (bp) from TSS to filter peaks (default: 1,000,000)")
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
        raise ValueError("Unsupported ATAC file format.")
        
    logging.info(f'\tNumber of peaks: {df.shape[0]}')
    logging.info(f'\tNumber of cells: {df.shape[1] - 1}')
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
        raise ValueError("Unsupported RNA file format.")
        
    logging.info(f'\tNumber of genes: {df.shape[0]}')
    logging.info(f'\tNumber of cells: {df.shape[1] - 1}')
    return df.rename(columns={df.columns[0]: "gene_id"})

def load_ensembl_tss_df(organism: str) -> pd.DataFrame:
    """
    Query Ensembl BioMart for TSS of all genes in 'organism'
    (e.g. 'mmusculus_gene_ensembl'). Returns a DataFrame with columns:
    ['chr', 'start', 'end', 'gene_id'], where start = TSS, end = TSS+1.
    """
    server = Server(host="http://www.ensembl.org")
    dataset_name = f"{organism}_gene_ensembl"
    mart = server["ENSEMBL_MART_ENSEMBL"]
    ds = mart[dataset_name]
    
    df = ds.query(
        attributes=[
            "external_gene_name",
            "strand",
            "chromosome_name",
            "transcription_start_site"
        ]
    )
    df.rename(
        columns={
            "Chromosome/scaffold name": "chr",
            "Transcription start site (TSS)": "tss",
            "Gene name": "gene_id"
        },
        inplace=True
    )
    df["tss"] = df["tss"].astype(int)
    df["start"] = df["tss"]
    df["end"] = df["tss"] + 1
    return df[["chr", "start", "end", "gene_id"]].copy()

def filter_peaks_within_distance(atac_df: pd.DataFrame,
                                 tss_df: pd.DataFrame,
                                 distance: int) -> pd.DataFrame:
    """
    Given atac_df with a 'peak_id' column ("chr:start-end") and numeric cell columns,
    plus tss_df with columns ['chr','start','end','gene_id'], filter to keep only
    peaks whose interval overlaps (±distance bp) any TSS.

    Returns atac_df filtered to those peak_ids.
    """
    # 1) Build PyRanges for TSS expanded by ±distance
    tss_df = tss_df.copy()
    tss_df["Chromosome"] = "chr" + tss_df["chr"].astype(str)
    tss_df["Start"] = tss_df["start"] - distance
    tss_df["End"]   = tss_df["end"] + distance
    # Ensure non-negative start
    tss_df["Start"] = tss_df["Start"].clip(lower=0)
    promoter_pr = pr.PyRanges(tss_df[["Chromosome", "Start", "End", "gene_id"]])
    
    # 2) Parse peaks from atac_df['peak_id']
    peaks = atac_df["peak_id"].str.split("[:-]", expand=True)
    # peaks has columns [chr, start, end] but split by ':' and '-'
    peaks.columns = ["Chromosome", "Start", "End"]
    peaks["Chromosome"] = peaks["Chromosome"].astype(str)
    peaks["Start"] = peaks["Start"].astype(int)
    peaks["End"] = peaks["End"].astype(int)
    peaks_pr = pr.PyRanges(peaks)
    
    # 3) Find peaks overlapping any promoter window
    overlap_pr = peaks_pr.overlap(promoter_pr)
    overlap_df = overlap_pr.df.iloc[:, :3].copy()
    overlap_df.columns = ["Chromosome", "Start", "End"]
    
    # 4) Reconstruct peak_id strings for filtering
    overlap_df["peak_id"] = (
        overlap_df["Chromosome"]
        + ":"
        + overlap_df["Start"].astype(str)
        + "-"
        + overlap_df["End"].astype(str)
    )
    kept_peak_ids = set(overlap_df["peak_id"].tolist())
    
    # 5) Filter atac_df to only those peak_ids
    filtered = atac_df[atac_df["peak_id"].isin(kept_peak_ids)].copy()
    logging.info(f"Filtered peaks within ±{distance} bp of TSS: kept {filtered.shape[0]} peaks (out of {atac_df.shape[0]})")
    return filtered.reset_index(drop=True)

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

def main(atac_data_file, rna_data_file, organism, tss_distance):
    logging.info("Loading ATAC-seq dataset")
    atac_df = load_atac_dataset(atac_data_file)

    logging.info("Loading RNA-seq dataset")
    rna_df = load_rna_dataset(rna_data_file)

    logging.info(f"Loading TSS for organism '{organism}'")
    tss_df = load_ensembl_tss_df(organism)

    logging.info(f"Filtering peaks to within ±{tss_distance} bp of any TSS")
    atac_df_filtered = filter_peaks_within_distance(atac_df, tss_df, tss_distance)

    logging.info("Checking and normalizing ATAC-seq data")
    atac_df_norm = log2_cpm_normalize(atac_df_filtered, id_col_name="peak_id", label="scATAC-seq")

    logging.info("Checking and normalizing RNA-seq data")
    rna_df_norm = log2_cpm_normalize(rna_df, id_col_name="gene_id", label="scRNA-seq")

    # Deduplicate columns if necessary
    atac_df_norm = deduplicate_columns(atac_df_norm, label="ATAC")
    rna_df_norm = deduplicate_columns(rna_df_norm, label="RNA")

    def update_name(filename):
        base, ext = os.path.splitext(filename)
        return f"{base}_processed.parquet"

    new_atac_file = update_name(atac_data_file)
    new_rna_file = update_name(rna_data_file)

    logging.info('\nWriting ATAC-seq dataset to Parquet')
    atac_df_norm.to_parquet(new_atac_file, engine="pyarrow", compression="snappy", index=False)
    logging.info("  Done!")

    logging.info('\nWriting RNA-seq dataset to Parquet')
    rna_df_norm.to_parquet(new_rna_file, engine="pyarrow", compression="snappy", index=False)
    logging.info("  Done!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args()
    main(args.atac_data_file, args.rna_data_file, args.organism, args.tss_distance)
