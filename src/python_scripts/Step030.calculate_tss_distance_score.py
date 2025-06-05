import pandas as pd
import logging
from pybiomart import Server
import pyranges as pr
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument("--atac_data_file", type=str, required=True, help="Path to the scATAC-seq dataset")
    parser.add_argument("--organism", type=str, default="mmusculus", 
                        help="Ensembl organism prefix (e.g. mmusculus for mouse) for TSS lookup")
    parser.add_argument("--tss_distance", type=int, default=1_000_000,
                        help="Distance (bp) from TSS to filter peaks (default: 1,000,000)")
    return parser.parse_args()

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

def main(atac_data_file, organism, tss_distance):
    logging.info("Loading ATAC-seq dataset")
    atac_df = load_atac_dataset(atac_data_file)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args()
    main(args.atac_data_file, args.rna_data_file, args.organism, args.tss_distance)