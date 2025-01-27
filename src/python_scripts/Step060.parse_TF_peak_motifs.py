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
        "--cicero_cis_reg_file",
        type=str,
        required=True,
        help="Path to the inferred cis-regulatory network from Cicero"
    )
    parser.add_argument(
        "--homer_peak_file",
        type=str,
        required=True,
        help="Path to the formatted ATAC-seq peak file for Homer"
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

def process_TF_motif_file(path_to_file: str, peak_gene_assoc: pd.DataFrame) -> pd.DataFrame:
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
        # Read in the motif file as a pandas DataFrame
        motif_to_peak: pd.DataFrame = pd.read_csv(path_to_file, sep='\t', header=[0], index_col=None)
        motif_to_peak.rename(columns={motif_to_peak.columns[0]: "PeakID"}, inplace=True)
        motif_to_peak["Start"] = motif_to_peak["Start"].astype(str)
        motif_to_peak["End"] = motif_to_peak["End"].astype(str)
        
        # Reconstruct the peak
        motif_to_peak["peak"] = motif_to_peak["Chr"] + "_" + motif_to_peak["Start"] + "_" + motif_to_peak["End"]
            
        # Extract motif-related data
        motif_column: str = motif_to_peak.columns[-2]
        TF_name: str = motif_column.split('/')[0].split('(')[0]
        
        if ":" in TF_name:
            TF_name = TF_name.split(':')[0]
        
        # Parse the number of motifs and the name of the TF from the Homer annotatePeaks output
        motif_to_peak['Motif Count'] = motif_to_peak[motif_column].apply(lambda x: len(x.split(',')) / 3 if pd.notnull(x) else 0)
        motif_to_peak["Source"] = TF_name
        
        # Merge the peak with the peak to gene association score
        motif_to_peak = pd.merge(
            motif_to_peak,
            peak_gene_assoc[["peak", "gene", "Peak Gene Score"]],  # Use only 'peak' and 'gene' columns from peak_gene_assoc
            how="left",                         # Perform a left join to retain all rows in motif_to_peak
            on="peak"                           # Join on the 'peak' column
        )

        # Rename the associated gene column for clarity
        motif_to_peak.rename(columns={"gene": "Target"}, inplace=True)
        motif_to_peak.dropna(subset=["Target"], inplace=True)

        # Calculate total motifs for the TF
        motif_to_peak["Total Motifs"] = motif_to_peak["Motif Count"].sum()

        # Columns of interest
        cols_of_interest: List[str] = ["Source", "Target", "Motif Count", "Total Motifs", "Peak Gene Score"]

        # Remove rows with NA values
        df: pd.DataFrame = motif_to_peak[cols_of_interest].dropna()

        # Filter rows with no motifs
        filtered_df: pd.DataFrame = df[df["Motif Count"] > 0]
        filtered_df: pd.DataFrame = filtered_df[filtered_df["Peak Gene Score"] > 0]

        # Standardize gene names to uppercase
        filtered_df["Source"] = filtered_df["Source"].apply(lambda x: x.upper())
        filtered_df["Target"] = filtered_df["Target"].apply(lambda x: x.upper())
                
        # Calculate product of Motif Count and Peak Gene Score
        filtered_df['product'] = filtered_df['Motif Count'] * filtered_df['Peak Gene Score']

        # Group by Source (TF) and Target (TG), then compute the score
        grouped = filtered_df.groupby(['Source', 'Target'])
        filtered_df = grouped.apply(
            lambda x: x['product'].sum() / x['Total Motifs'].iloc[0]
        ).reset_index(name='TF_TG_Motif_Binding_Score')
        
        print(filtered_df.head())
        
        return filtered_df
    
    except Exception as e:
        print(f"Error processing file {path_to_file}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there is an error

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
    
    # Read in the peak to gene association scores from cicero
    peak_gene_assoc: pd.DataFrame = pd.read_csv(cicero_cis_reg_file, sep=",", header=0)
    
    # Normalize the cicero peak to gene scores (excluding 0 and 1)
    peak_gene_assoc['Peak Gene Score'] = peak_gene_assoc['score'].apply(
        lambda x: (x - peak_gene_assoc['score'].min()) / (peak_gene_assoc['score'].max() - peak_gene_assoc['score'].min())
        if 0 != x != 1 else x  # Retain scores of 0 and 1 as they are
    )
    
    # Process each of the files
    # Preload the additional argument using functools.partial
    process_file_partial = partial(process_TF_motif_file, peak_gene_assoc=peak_gene_assoc)

    tf_scores_list = []
    with Pool(processes=cpu_count) as pool:
        with tqdm(total=len(file_paths), desc="Processing files") as pbar:
            for tf_tg_scores in pool.imap_unordered(process_file_partial, file_paths):
                tf_scores_list.append(tf_tg_scores)
                pbar.update(1)

    logging.info("Finished formatting all TF motif binding sites for downstream target genes, combining")
    
    # Combine all results
    tf_binding_scores_df: pd.DataFrame = pd.concat(tf_scores_list, ignore_index=True)
    
    tf_binding_scores_df["TF_TG_Motif_Binding_Score"] = tf_binding_scores_df["TF_TG_Motif_Binding_Score"] / max(tf_binding_scores_df["TF_TG_Motif_Binding_Score"])

    logging.info(f"Finished normalization, writing combined dataset to {output_file}")
    tf_binding_scores_df.to_csv(output_file, sep='\t', index=False)

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