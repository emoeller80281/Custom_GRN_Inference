import dask.dataframe as dd
import dask
from dask import delayed, compute
import pandas as pd
import os
from dask.diagnostics import ProgressBar
from typing import List
import argparse
import logging
import numpy as np

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

def process_TF_motif_file(path_to_file: str, output_dir: str) -> str:
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
        motif_to_peak['homer_binding_score'] = motif_to_peak['tf_motifs_in_peak'] / total_tf_binding_sites
        
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

    logging.info("Wrapping delayed processing tasks")
    delayed_tasks = [delayed(process_TF_motif_file)(f, output_dir) for f in file_paths]
    
    # Get the stream used by the root logger or your custom logger
    log_stream = logging.getLogger().handlers[0].stream

    with ProgressBar(dt=5.0, minimum=1, width=80, out=log_stream):
        all_parquet_paths = compute(*delayed_tasks, scheduler="processes", num_workers=cpu_count)
        
    parquet_paths = [p for p in all_parquet_paths if p]
    
    if not parquet_paths:
        logging.error("No valid parquet files were written")
        return

    logging.info(f"Combining {len(parquet_paths)} Parquet files into a single Dask DataFrame")
    ddf = dd.read_parquet(parquet_paths)
    
    combined_out = os.path.join(output_dir, "homer_tf_to_peak.parquet")
    
    # Compute global 5th/95th quantiles for homer_binding_score
    score_col = "homer_binding_score"
    qs = combined_out[score_col].quantile([0.05, 0.95]).compute()
    low, high = qs.loc[0.05], qs.loc[0.95]
    logging.info(f"Global {score_col} 5th/95th cutoffs: {low:.4g}, {high:.4g}")

    # Define per-partition normalizer
    def normalize_homer(df):
        # clip into [low, high]
        df[score_col] = df[score_col].clip(lower=low, upper=high)
        # linear scale to [0,1]
        df[score_col] = (df[score_col] - low) / (high - low)
        # log1p
        df[score_col] = np.log1p(df[score_col])
        return df

    # Apply across your Dask DataFrame
    normalized_dd = combined_out.map_partitions(normalize_homer)

    # (Optional) persist or overwrite final_dd
    combined_out = normalized_dd.persist()

    # Then continue on—e.g. write back out:
    combined_out.to_parquet(
        f"{output_dir}/inferred_network_with_normalized_homer.parquet",
        engine="pyarrow",
        compression="snappy",
    )

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