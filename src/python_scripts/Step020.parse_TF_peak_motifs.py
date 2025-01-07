import pandas as pd
import os
from tqdm import tqdm  # or thread_map
from multiprocessing import Pool
import argparse
import logging

homer_tf_motif_score_dir = "./output/homer_tf_motif_scores"

def parse_args():
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
        help="Path for the merged TF TG motif binding score tsv file"
    )
    parser.add_argument(
        "--cpu_count",
        type=str,
        required=True,
        help="The number of CPUs to utilize for multiprocessing"
    )
    
    args = parser.parse_args()

    return args   

def process_file(path_to_file) -> pd.DataFrame:
    motif_to_peak = pd.read_csv(path_to_file, sep='\t', header=[0], index_col=None)

    # Count the number of motifs found for the peak
    motif_column = motif_to_peak.columns[-1]
    TF_name = motif_column.split('/')[0]
    
    # Set the columns    
    motif_to_peak['Motif Count'] = motif_to_peak[motif_column].apply(lambda x: len(x.split(',')) / 3 if pd.notnull(x) else 0)
    motif_to_peak["Source"] = TF_name.split('(')[0]
    motif_to_peak["Target"] = motif_to_peak["Gene Name"]

    # Sum the total number of motifs for the TF
    total_motifs = motif_to_peak["Motif Count"].sum()

    # peak_binding_column = motif_to_peak[motif_to_peak.columns[-1]]
    cols_of_interest = ["Source", "Target", "Motif Count"]
    
    # Remove any rows with NA values
    df = motif_to_peak[cols_of_interest].dropna()

    # Filter out rows with no motifs
    filtered_df = df[df["Motif Count"] > 0]
    # print(filtered_df.head())

    # Sum the total motifs for each TG
    filtered_df = filtered_df.groupby(["Source", "Target"])["Motif Count"].sum().reset_index()
    
    filtered_df["Source"] = filtered_df["Source"].apply(lambda x: x.upper())
    filtered_df["Target"] = filtered_df["Target"].apply(lambda x: x.upper())

    # Calculate the score for the TG based on the total number of motifs for the TF
    filtered_df["Motif_Score"] = filtered_df["Motif Count"] / total_motifs

    # Save the final results
    final_df = filtered_df[["Source", "Target", "Motif_Score"]]
    
    return final_df
    
def main(input_dir, output_file, cpu_count):
    file_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    results = []

    # Combine progress bar with result collection
    with Pool(processes=cpu_count) as pool:
        with tqdm(total=len(file_paths), desc="Processing files") as pbar:
            for result in pool.imap_unordered(process_file, file_paths):
                results.append(result)
                pbar.update(1)

    logging.info(f'Finished formatting all TF motif binding sites for downstream target genes, combining')
        
    combined_df = pd.concat(results, ignore_index=True)
    
    logging.info(f'TF motif scores combined, normalizing')
    max_score = max(combined_df['Motif_Score'])
    
    # Clamps the scores between 0-1 
    combined_df['Motif_Score'] = combined_df['Motif_Score'].apply(lambda x: x / max_score)

    logging.info(f'Finished normalization, writing combined dataset to {output_file}')
    # print(combined_df.head())
    combined_df.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    args = parse_args()
    input_dir = args.input_dir
    output_file = args.output_file
    cpu_count = int(args.cpu_count)
    
    main(input_dir, output_file, cpu_count)

