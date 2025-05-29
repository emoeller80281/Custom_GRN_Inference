import pandas as pd
import argparse
import os
import logging
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument("--atac_data_file", type=str, required=True, help="Path to the scATAC-seq dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for the sample")
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
        
    logging.info(f'\tNumber of peaks: {df.shape[0]}')
    logging.info(f'\tNumber of cells: {df.shape[1]-1}')
        
    return df.rename(columns={df.columns[0]: "peak_id"})

def main():
    args = parse_args()
    atac_data_file: str = args.atac_data_file
    output_dir: str = args.output_dir

    logging.info('===== CONVERTING ATAC DATASET TO CICERO SPARSE MATRIX INPUT FORMAT =====')
    # Load or normalize your matrix first
    logging.info('\t- Loading ATAC parquet file')
    atac_df = load_atac_dataset(atac_data_file)  # already has 'peak_id' in col 0
    
    logging.info("\t- Locating count values > 0")
    peak_ids = atac_df["peak_id"].values
    cell_ids = atac_df.columns[1:]

    dense_mat = atac_df.iloc[:, 1:].values  # (n_peaks, n_cells)
    
    # Get nonzero indices
    row_idx, col_idx = np.nonzero(dense_mat)
    counts = dense_mat[row_idx, col_idx]
    logging.info(f'\t\tNon-zero values: {len(counts)} / {len(dense_mat)}')
    
    logging.info('\t- Melting to peak_id, barcode, count long format')
    # Melt into long format
    atac_long = pd.DataFrame({
        "peak_id": peak_ids[row_idx],
        "cell": cell_ids[col_idx],
        "count": counts
    })

    logging.info('\t- Formatting peak names to chrN_XXX_XXX format')
    # Format peak_id to match Cicero expectations: chr1_100_200
    atac_long["peak_id"] = atac_long["peak_id"].str.replace("[:-]", "_", regex=True)

    logging.info('\t- Saving to sample output directory as "cicero_atac_input.txt"')
    save_path = os.path.join(output_dir, "cicero_atac_input.txt")

    # Save to tab-delimited file without header
    atac_long.to_csv(save_path, sep="\t", header=False, index=False)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()


