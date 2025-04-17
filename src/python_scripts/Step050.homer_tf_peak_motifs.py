import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import argparse
import logging
from typing import List

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing paths for input and output files and CPU count.
    """
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing Homer TF motif scores from Homer 'annotatePeaks.pl'"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path for the merged TF TG motif binding score TSV file"
    )
    parser.add_argument(
        "--cpu_count",
        type=str,
        required=True,
        help="The number of CPUs to utilize for multiprocessing"
    )
    
    args: argparse.Namespace = parser.parse_args()
    return args

def is_file_empty(file_path: str) -> bool:
    """
    Check if a file is empty or invalid.
    """
    return os.stat(file_path).st_size == 0

def process_TF_motif_file(path_to_file: str) -> pd.DataFrame:
    """
    Processes a single Homer TF motif file and extracts TF-TG relationships and motif scores.

    Args:
        path_to_file (str): Path to the motif file.
        peak_gene_assoc (pd.DataFrame): Dataframe of Cicero peak to gene inference results.

    Returns:
        pd.DataFrame: DataFrame containing Source (TF), Target (TG), and Motif_Score.
    """
    
    # Check if the file is empty
    if is_file_empty(path_to_file):
        print(f"Skipping empty file: {path_to_file}")
        return pd.DataFrame()  # Return an empty DataFrame

    try:
        # Read in the motif file; force "Start" and "End" to be strings
        motif_to_peak = pd.read_csv(path_to_file, sep='\t', header=0, dtype={'Start': str, 'End': str})
        
        # Reconstruct the peak_id
        motif_to_peak['peak_id'] = motif_to_peak['Chr'] + ':' + motif_to_peak['Start'] + '-' + motif_to_peak['End']
        
        # Extract the motif column
        motif_column = motif_to_peak.columns[-2]
        
        # Extract the TF name from the motif column name; chain splits to remove extraneous info
        TF_name = motif_column.split('/')[0].split('(')[0].split(':')[0]
        
        # Set source_id to the TF name
        motif_to_peak['source_id'] = TF_name
        
        # Calculate the number of motifs per peak.
        # If the motif column is not null, split by comma, count tokens, divide by 3; else 0.
        motif_to_peak['tf_motifs_in_peak'] = motif_to_peak[motif_column].apply(
            lambda x: len(x.split(',')) / 3 if pd.notnull(x) and x != '' else 0
        )
        
        # Remove rows with zero binding sites
        motif_to_peak = motif_to_peak[motif_to_peak['tf_motifs_in_peak'] > 0].copy()
        
        # Calculate the total number of binding sites across all peaks
        total_tf_binding_sites = motif_to_peak['tf_motifs_in_peak'].sum()
        
        # Calculate homer_binding_score for each peak
        motif_to_peak['homer_binding_score'] = motif_to_peak['tf_motifs_in_peak'] / total_tf_binding_sites
        
        # Select only the columns of interest and drop any rows with missing values
        df = motif_to_peak[['peak_id', 'source_id', 'homer_binding_score']].dropna()
        
        return df
    
    except Exception as e:
        print(f"Error processing file {path_to_file}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there is an error

def main(input_dir: str, output_file: str, cpu_count: int) -> None:
    
    # List all files in the input directory
    file_paths: List[str] = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    tf_scores_list = []
    with Pool(processes=cpu_count) as pool:
        with tqdm(total=len(file_paths), desc="Processing files") as pbar:
            for tf_tg_scores in pool.imap_unordered(process_TF_motif_file, file_paths):
                tf_scores_list.append(tf_tg_scores)
                pbar.update(1)

    logging.info("Finished formatting all TF motif binding sites for downstream target genes, combining")
    
    # Combine all results
    logging.info("Concatenating all TF to peak binding score DataFrames")
    tf_binding_scores_df: pd.DataFrame = pd.concat(tf_scores_list, ignore_index=True)
    
    tf_binding_scores_df.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Parse command-line arguments
    args: argparse.Namespace = parse_args()
    input_dir: str = args.input_dir
    output_file: str = args.output_file
    cpu_count: int = int(args.cpu_count)

    # Run the main function
    main(input_dir, output_file, cpu_count)