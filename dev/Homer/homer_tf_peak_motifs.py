import pandas as pd
import os
from typing import List, Union
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

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
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the sample"
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

def process_TF_motif_file(path_to_file: str, output_dir: str) -> Union[str,None]:
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
        return None  # Return an empty DataFrame

    try:
        # Read in the motif file; force "Start" and "End" to be strings
        motif_to_peak = pd.read_csv(path_to_file, sep='\t', header=0, dtype={'Start': str, 'End': str})
        
        # Reconstruct the peak_id
        motif_to_peak['peak_id'] = motif_to_peak['Chr'] + ':' + motif_to_peak['Start'] + '-' + motif_to_peak['End']
        
        # Extract the motif column
        motif_candidates = [
            c for c in motif_to_peak.columns
            if "Distance From Peak(sequence,strand,conservation)" in c
        ]

        if len(motif_candidates) != 1:
            raise ValueError(
                f"Expected exactly one motif column, found {len(motif_candidates)}. "
                f"Candidates: {motif_candidates}. "
                f"Last columns: {motif_to_peak.columns[-6:].tolist()}"
            )

        motif_column = motif_candidates[0]
        
        # Extract the TF name from the motif column name; chain splits to remove extraneous info
        TF_name = motif_column.split('/')[0].split('(')[0].split(':')[0]
        
        tmp_dir =  os.path.join(output_dir, "tmp", "homer_tf_parquet_files")
        os.makedirs(tmp_dir, exist_ok=True)
        
        output_path = os.path.join(tmp_dir, f"{TF_name}.parquet")
        
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
        motif_to_peak['homer_binding_score'] = motif_to_peak['tf_motifs_in_peak']
        
        # Select only the columns of interest and drop any rows with missing values
        df = motif_to_peak[['peak_id', 'source_id', 'homer_binding_score']].dropna()
        
        df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
        
        return output_path
    
    except Exception as e:
        print(f"Error processing file {path_to_file}: {e}")
        return None  # Return an empty DataFrame if there is an error

def main(input_dir: str, output_dir: str, cpu_count: int) -> None:
    
    file_paths: List[str] = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]

    logging.info(f"Found {len(file_paths)} files in {input_dir}")

    logging.info("Processing files using multiprocessing")
    all_parquet_paths = []
    
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        futures = [executor.submit(process_TF_motif_file, f, output_dir) for f in file_paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing TF files", ncols=80):
            res = future.result()
            if res is not None:
                all_parquet_paths.append(res)
    
    parquet_paths = [p for p in all_parquet_paths if isinstance(p, str)]
    
    if not parquet_paths:
        logging.error("No valid parquet files were written")
        return

    logging.info(f"Combining {len(parquet_paths)} Parquet files into a single Pandas DataFrame")
    
    combined_out = os.path.join(output_dir, "homer_tf_to_peak.parquet")
    combined_df = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in parquet_paths], ignore_index=True)
    combined_df.to_parquet(combined_out, engine="pyarrow", compression="snappy", index=False)
    
    logging.info("Finished writing combined TF motif binding scores to Parquet")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Parse command-line arguments
    args: argparse.Namespace = parse_args()
    input_dir: str = args.input_dir
    output_dir: str = args.output_dir
    cpu_count: int = int(args.cpu_count)

    # Run the main function
    main(input_dir, output_dir, cpu_count)