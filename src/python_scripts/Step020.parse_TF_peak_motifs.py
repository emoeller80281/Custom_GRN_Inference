import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import Pool
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
        "--cicero_cis_reg_file",
        type=str,
        required=True,
        help="Path to the inferred cis-regulatory network from Cicero"
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

def process_file(path_to_file: str, peak_gene_assoc: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a single Homer TF motif file and extracts TF-TG relationships and motif scores.

    Args:
        path_to_file (str): Path to the motif file.
        peak_gene_assoc (pd.DataFrame): Dataframe of Cicero peak to gene inference results.

    Returns:
        pd.DataFrame: DataFrame containing Source (TF), Target (TG), and Motif_Score.
    """
    # Read in the motif file as a pandas DataFrame
    motif_to_peak: pd.DataFrame = pd.read_csv(path_to_file, sep='\t', header=[0], index_col=None)

    motif_to_peak["Start"] = motif_to_peak["Start"].astype(str)
    motif_to_peak["End"] = motif_to_peak["End"].astype(str)
    
    # Reconstruct the peak
    motif_to_peak["peak"] = motif_to_peak["Chr"] + "_" + motif_to_peak["Start"] + "_" + motif_to_peak["End"]
    
    # Merge the peaks that match with the peaks in the peak to gene association output from Cicero 
    merged_df = pd.merge(motif_to_peak, peak_gene_assoc, how='right', on='peak')
    
    print(merged_df.head())
    
    # Extract motif-related data
    motif_column: str = motif_to_peak.columns[-1]
    TF_name: str = motif_column.split('/')[0]
    
    # Set the columns    
    motif_to_peak['Motif Count'] = motif_to_peak[motif_column].apply(lambda x: len(x.split(',')) / 3 if pd.notnull(x) else 0)
    motif_to_peak["Source"] = TF_name.split('(')[0]
    motif_to_peak["Target"] = motif_to_peak["Gene Name"]

    # Calculate total motifs for the TF
    total_motifs: float = motif_to_peak["Motif Count"].sum()

    # Columns of interest
    cols_of_interest: List[str] = ["Source", "Target", "Motif Count"]

    # Remove rows with NA values
    df: pd.DataFrame = motif_to_peak[cols_of_interest].dropna()

    # Filter rows with no motifs
    filtered_df: pd.DataFrame = df[df["Motif Count"] > 0]

    # Group by Source and Target and sum the motifs
    filtered_df = filtered_df.groupby(["Source", "Target"])["Motif Count"].sum().reset_index()

    # Standardize gene names to uppercase
    filtered_df["Source"] = filtered_df["Source"].apply(lambda x: x.upper())
    filtered_df["Target"] = filtered_df["Target"].apply(lambda x: x.upper())

    # Calculate Motif Score
    filtered_df["Motif_Score"] = filtered_df["Motif Count"] / total_motifs

    # Return final DataFrame
    final_df: pd.DataFrame = filtered_df[["Source", "Target", "Motif_Score"]]
    return final_df

def main(input_dir: str, cicero_cis_reg_file: str, output_file: str, cpu_count: int) -> None:
    """
    Main function to process all files in a directory and output combined results.

    Args:
        input_dir (str): Path to the directory containing input files.
        output_file (str): Path to the output file.
        cpu_count (int): Number of CPUs to use for multiprocessing.
    """
    # List all files in the input directory
    file_paths: List[str] = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    peak_gene_assoc: pd.DataFrame = pd.read_csv(cicero_cis_reg_file, sep=",", header=0)
    
    results: List[pd.DataFrame] = []
    # Use multiprocessing pool to process files
    with Pool(processes=cpu_count) as pool:
        with tqdm(total=len(file_paths), desc="Processing files") as pbar:
            for result in pool.imap_unordered(process_file, file_paths, peak_gene_assoc):
                results.append(result)
                pbar.update(1)

    logging.info("Finished formatting all TF motif binding sites for downstream target genes, combining")

    # Combine all results
    combined_df: pd.DataFrame = pd.concat(results, ignore_index=True)

    logging.info("TF motif scores combined, normalizing")
    max_score: float = max(combined_df['Motif_Score'])

    # Normalize scores between 0-1
    combined_df['Motif_Score'] = combined_df['Motif_Score'].apply(lambda x: x / max_score)

    logging.info(f"Finished normalization, writing combined dataset to {output_file}")
    combined_df.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Parse command-line arguments
    args: argparse.Namespace = parse_args()
    input_dir: str = args.input_dir
    cicero_cis_reg_file: str = args.cicero_cis_reg_file
    homer_peak_file: str = args.homer_peak_file
    output_file: str = args.output_file
    cpu_count: int = int(args.cpu_count)

    # Run the main function
    main(input_dir, cicero_cis_reg_file, output_file, cpu_count)