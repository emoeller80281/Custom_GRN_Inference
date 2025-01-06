import pandas as pd
from Bio import SeqIO
from Bio import motifs
from Bio.Seq import Seq
import requests
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument(
        "--atac_data_file",
        type=str,
        required=True,
        help="Path to the directory containing Homer TF motif scores from Homer 'annotatePeaks.pl'"
    )
    
    args = parser.parse_args()

    return args   

def convert_to_homer_peak_format(atac_data):
    # Extract the peak ID column (assuming the first column contains peak IDs)
    peak_ids = atac_data.iloc[:, 0]  # Adjust column index if peak IDs are not in the first column

    # Split peak IDs into chromosome, start, and end
    chromosomes = peak_ids.str.split(':').str[0]
    starts = peak_ids.str.split(':').str[1].str.split('-').str[0]
    ends = peak_ids.str.split(':').str[1].str.split('-').str[1]

    # Create a dictionary for constructing the HOMER-compatible DataFrame
    homer_dict = {
        "peak_id": ["peak" + str(i + 1) for i in range(len(peak_ids))],  # Generate unique peak IDs
        "chromosome": chromosomes,
        "start": starts,
        "end": ends,
        "strand": ["."] * len(starts),  # Set strand as "."
    }

    # Construct the DataFrame
    homer_df = pd.DataFrame(homer_dict)

    # Convert 'start' and 'end' to numeric types
    homer_df['start'] = pd.to_numeric(homer_df['start'])
    homer_df['end'] = pd.to_numeric(homer_df['end'])

    # Print the head of the resulting DataFrame
    print(homer_df.head())

    return homer_df
    

# ----- Input -----
# Read in the ATAC data
args = parse_args()
atac_data_file = args.atac_data_file
print(f'Reading scATACseq data')
atac_data = pd.read_csv(atac_data_file)
print(atac_data.head())

print(f'Converting scATACseq peaks to Homer peak format')
homer_df = convert_to_homer_peak_format(atac_data)
print(homer_df.head())

homer_peak_file = homer_df.to_csv("/home/emoeller/github/Custom_GRN_Inference/input/Homer_peaks.txt", sep='\t', header=False, index=False)