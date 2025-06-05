import argparse
import logging
import os

import numpy as np
import pandas as pd
import pyranges as pr
import pybedtools
from pybiomart import Server
from scipy import sparse
from pyarrow.parquet import ParquetFile
import pyarrow as pa 
import math

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument("--atac_data_file", type=str, required=True, help="Path to the scATAC-seq dataset")
    parser.add_argument("--rna_data_file", type=str, required=True, help="Path to the scRNA-seq dataset")
    parser.add_argument("--species", type=str, required=True, help="Species genome, e.g. 'mm10' or 'hg38'")
    parser.add_argument("--tss_distance_cutoff", type=int, required=False, default=1_000_000, help="Distance (bp) from TSS to filter peaks (default: 1,000,000)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the sample")
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
    zero_lib = library_sizes == 0
    if zero_lib.any():
        logging.warning(f"Found {zero_lib.sum()} all-zero columns—setting library size to 1e-6 to avoid division by zero.")
        library_sizes[zero_lib] = 1e-6
    cpm = (counts.div(library_sizes, axis=1) * 1e6).add(1)
    log2_cpm = np.log2(cpm).reset_index(drop=True)

    return pd.concat([id_col, log2_cpm], axis=1)

def load_atac_dataset(atac_data_file: str) -> pd.DataFrame:
    df: pd.DataFrame = pd.DataFrame()
    if atac_data_file.lower().endswith('.parquet'):
        df = pd.read_parquet(atac_data_file)
        
    elif atac_data_file.lower().endswith('.csv'):
        df = pd.read_csv(atac_data_file, sep=",", header=0, index_col=None)
        
    elif atac_data_file.lower().endswith('.tsv'):
        df = pd.read_csv(atac_data_file, sep="\t", header=0, index_col=None)
        
    else:
        raise ValueError(f"ATAC data file must be .csv, .tsv or .parquet: got {atac_data_file}")
    
    if df.empty:
        raise RuntimeError(f"Failed to load ATAC file: {args.atac_data_file}")
    
    df = df.rename(columns={df.columns[0]: "peak_id"})
    
    logging.info(f'\tNumber of peaks: {df.shape[0]}')
    logging.info(f'\tNumber of cells: {df.shape[1]-1}')
    
    return df

def load_rna_dataset(rna_data_file: str) -> pd.DataFrame:
    if rna_data_file.lower().endswith('.parquet'):
        df = pd.read_parquet(rna_data_file)
        
    elif rna_data_file.lower().endswith('.csv'):
        df = pd.read_csv(rna_data_file, sep=",", header=0, index_col=None)
        
    elif rna_data_file.lower().endswith('.tsv'):
        df = pd.read_csv(rna_data_file, sep="\t", header=0, index_col=None)
        
    else:
        raise ValueError(f"RNA data file must be .csv, .tsv or .parquet: got {rna_data_file}")
    
    df = df.rename(columns={df.columns[0]: "gene_id"})
    
    if df.empty:
        raise RuntimeError(f"Failed to load RNA file: {args.rna_data_file}")
    
    logging.info(f'\tNumber of genes: {df.shape[0]}')
    logging.info(f'\tNumber of cells: {df.shape[1]-1}')
    
    return df

def extract_atac_peaks(atac_df, tmp_dir):
    
    if not os.path.exists(f"{tmp_dir}/peak_df.parquet"):
        logging.info(f"Extracting peak information and saving as a bed file")
        def parse_peak_str(s):
            try:
                chrom, coords = s.split(":")
                start_s, end_s = coords.split("-")
                return chrom.replace("chr", ""), int(start_s), int(end_s)
            except Exception:
                raise ValueError(f"Malformed peak_id '{s}'; expected 'chrN:start-end'.")

        # List of peak strings
        peak_pos = atac_df["peak_id"].tolist()

        # Apply parsing function to all peak strings
        parsed = [parse_peak_str(s) for s in peak_pos]

        # Construct DataFrame
        peak_df = pd.DataFrame(parsed, columns=["chr", "start", "end"])
        peak_df["peak_id"] = peak_pos
        
        # Write the peak DataFrame to a file
        peak_df.to_parquet(f"{tmp_dir}/peak_df.parquet", engine="pyarrow", index=False, compression="snappy")
    else:
        logging.info("ATAC-seq BED file exists, loading...")

def load_ensembl_organism_tss(organism, tmp_dir):
    if not os.path.exists(os.path.join(tmp_dir, "ensembl.parquet")):
        logging.info(f"Loading Ensembl TSS locations for {organism}")
        # Connect to the Ensembl BioMart server
        server = Server(host='http://www.ensembl.org')

        gene_ensembl_name = f'{organism}_gene_ensembl'
        
        # Select the Ensembl Mart and the human dataset
        mart = server['ENSEMBL_MART_ENSEMBL']
        try:
            dataset = mart[gene_ensembl_name]
        except KeyError:
            raise RuntimeError(f"BioMart dataset {gene_ensembl_name} not found. Check if ‘{organism}’ is correct.")

        # Query for attributes: Ensembl gene ID, gene name, strand, and transcription start site (TSS)
        ensembl_df = dataset.query(attributes=[
            'external_gene_name', 
            'strand', 
            'chromosome_name',
            'transcription_start_site'
        ])

        ensembl_df.rename(columns={
            "Chromosome/scaffold name": "chr",
            "Transcription start site (TSS)": "tss",
            "Gene name": "gene_id"
        }, inplace=True)
        
        # Make sure TSS is integer (some might be floats).
        ensembl_df["tss"] = ensembl_df["tss"].astype(int)

        # In a BED file, we’ll store TSS as [start, end) = [tss, tss+1)
        ensembl_df["start"] = ensembl_df["tss"].astype(int)
        ensembl_df["end"] = ensembl_df["tss"].astype(int) + 1

        # Re-order columns for clarity: [chr, start, end, gene]
        ensembl_df = ensembl_df[["chr", "start", "end", "gene_id"]]
        
        ensembl_df["chr"] = ensembl_df["chr"].astype(str)
        ensembl_df["gene_id"] = ensembl_df["gene_id"].astype(str)
        
        # Write the peak DataFrame to a file
        ensembl_df.to_parquet(f"{tmp_dir}/ensembl.parquet", engine="pyarrow", index=False, compression="snappy")
    else:
        logging.info("Ensembl gene TSS BED file exists, loading...")

def find_genes_near_peaks(peak_bed, tss_bed, rna_df, peak_dist_limit, tmp_dir):
    """
    Identify genes whose transcription start sites (TSS) are near scATAC-seq peaks.
    
    This function:
        1. Uses BedTools to find peaks that are within peak_dist_limit bp of each gene's TSS.
        2. Converts the BedTool result to a pandas DataFrame.
        3. Computes the absolute distance between the peak end and gene start (as a proxy for TSS distance).
        4. Scales these distances using an exponential drop-off function (e^-dist/250000),
           the same method used in the LINGER cis-regulatory potential calculation.
        5. Deduplicates the data to keep the minimum (i.e., best) peak-to-gene connection.
        6. Only keeps genes that are present in the RNA-seq dataset.
        
    Parameters
    ----------
    peak_bed : BedTool
        A BedTool object representing scATAC-seq peaks.
    tss_bed : BedTool
        A BedTool object representing gene TSS locations.
    rna_df : pandas.DataFrame
        The RNA-seq dataset, which must have a "gene_id" column.
    peak_dist_limit : int
        The maximum distance (in bp) from a TSS to consider a peak as potentially regulatory.
        
    Returns
    -------
    peak_tss_subset_df : pandas.DataFrame
        A DataFrame containing columns "peak_id", "target_id", and the scaled TSS distance "TSS_dist"
        for peak–gene pairs.
    gene_list : set
        A set of unique gene IDs (target_id) present in the DataFrame.
    """
    if not os.path.exists(os.path.join(tmp_dir, "peak_to_gene_map.parquet")):
        # 3) Find peaks that are within peak_dist_limit bp of each gene's TSS using BedTools
        logging.info(f"Locating peaks that are within {peak_dist_limit} bp of each gene's TSS")
        peak_tss_overlap = peak_bed.window(tss_bed, w=peak_dist_limit)
        
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
        
        # Calculate the absolute distance between the peak's end and gene's start.
        # This serves as a proxy for the TSS distance for the peak-to-gene pair.
        distances = np.abs(peak_tss_overlap_df["peak_end"].values - peak_tss_overlap_df["gene_start"].values)
        peak_tss_overlap_df["TSS_dist"] = distances
        
        # Sort by the TSS distance (lower values imply closer proximity and therefore stronger association)
        # and drop duplicates keeping only the best association for each peak-target pair.
        peak_tss_overlap_df = peak_tss_overlap_df.sort_values("TSS_dist")
        peak_tss_overlap_df = peak_tss_overlap_df.drop_duplicates(subset=["peak_id", "target_id"], keep="first")
        
        # Scale the TSS distance using an exponential drop-off function
        # e^-dist/25000, same scaling function used in LINGER Cis-regulatory potential calculation
        # https://github.com/Durenlab/LINGER
        peak_tss_overlap_df["TSS_dist_score"] = np.exp(-peak_tss_overlap_df["TSS_dist"] / 250000)
        
        # Keep only the necessary columns.
        peak_tss_subset_df: pd.DataFrame = peak_tss_overlap_df[["peak_id", "target_id", "TSS_dist_score"]]
        
        # Filter out any genes not found in the RNA-seq dataset.
        rna_genes = set(rna_df["gene_id"])
        peak_tss_subset_df = peak_tss_subset_df[peak_tss_subset_df["target_id"].isin(rna_genes)]
        
        logging.info(f'\t- Number of peaks: {len(peak_tss_subset_df.drop_duplicates(subset="target_id"))}')
            
        peak_tss_subset_df.to_parquet(f"{tmp_dir}/peak_to_gene_map.parquet", index=False)
    else:
        logging.info('TSS distance file exists, loading...')

def extract_atac_peaks_near_rna_genes(
    atac_df: pd.DataFrame, 
    rna_df: pd.DataFrame, 
    organism: str, 
    tss_distance_cutoff: int, 
    tmp_dir: str
    ) -> None:
    
    if not organism == "hsapiens" or not organism == "mmusculus":
        if organism == "hg38":
            organism = "hsapiens"
        elif organism == "mm10":
            organism = "mmusculus"
        else:
            raise Exception(f'Organism not found, you entered {organism} (must be one of: "hg38", "mm10")')

    extract_atac_peaks(atac_df, tmp_dir)
    load_ensembl_organism_tss(organism, tmp_dir)

    pybedtools.set_tempdir(tmp_dir)
    
    # Load the peak and gene TSS BED files
    peak_df = pd.read_parquet(os.path.join(tmp_dir, "peak_df.parquet"))
    tss_df = pd.read_parquet(os.path.join(tmp_dir, "ensembl.parquet"))

    peak_bed = pybedtools.BedTool.from_dataframe(peak_df)
    tss_bed = pybedtools.BedTool.from_dataframe(tss_df)
    
    find_genes_near_peaks(
        peak_bed,
        tss_bed,
        rna_df,
        tss_distance_cutoff,
        tmp_dir
    )

    pybedtools.helpers.cleanup(verbose=False, remove_all=True)

def main(args):
    args.atac_data_file = os.path.abspath(args.atac_data_file)
    args.rna_data_file = os.path.abspath(args.rna_data_file)
    output_dir = os.path.abspath(args.output_dir)
    tmp_dir = os.path.join(output_dir, "tmp")
    
    os.makedirs(tmp_dir, exist_ok=True)
    
    if not os.path.isfile(args.atac_data_file):
        raise FileNotFoundError(f"ATAC file not found: {args.atac_data_file}")

    if not os.path.isfile(args.rna_data_file):
        raise FileNotFoundError(f"RNA file not found: {args.RNA_data_file}")
    
    logging.info("Loading ATAC-seq dataset")
    atac_df: pd.DataFrame = load_atac_dataset(args.atac_data_file)

    logging.info("Loading RNA-seq dataset")
    rna_df: pd.DataFrame = load_rna_dataset(args.rna_data_file)

    extract_atac_peaks_near_rna_genes(atac_df, rna_df, args.species, args.tss_distance_cutoff, tmp_dir)
        
    peak_gene_df = pd.read_parquet(f"{tmp_dir}/peak_to_gene_map.parquet")
    
    peak_subset = set(peak_gene_df["peak_id"])
    atac_df_filtered = atac_df[atac_df["peak_id"].isin(peak_subset)]
    logging.info(f'\tNumber of peaks after filtering: {len(set(atac_df["peak_id"]))}')

    logging.info("Checking and normalizing ATAC-seq data")
    atac_df_norm: pd.DataFrame = log2_cpm_normalize(atac_df_filtered, label="scATAC-seq", id_col_name="peak_id")

    logging.info("Checking and normalizing RNA-seq data")
    rna_df_norm: pd.DataFrame = log2_cpm_normalize(rna_df, label="scRNA-seq", id_col_name="gene_id")

    def update_name(filename):
        base, ext = os.path.splitext(filename)
        return f"{base}_processed.parquet"

    new_atac_file = update_name(args.atac_data_file)
    new_rna_file = update_name(args.rna_data_file)

    print(f"\nUpdated ATAC file: {new_atac_file}", flush=True)
    print(f"Updated RNA file: {new_rna_file}", flush=True)
    
    logging.info('\nWriting ATAC-seq dataset to Parquet')
    atac_df_norm.to_parquet(new_atac_file, engine="pyarrow", compression="snappy", index=False)
    logging.info("  Done!")
        
    logging.info('\nWriting RNA-seq dataset to Parquet')
    rna_df_norm.to_parquet(new_rna_file, engine="pyarrow", compression="snappy", index=False)
    logging.info("  Done!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args()
    
    main(args)
