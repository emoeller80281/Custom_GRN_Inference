#!/usr/bin/env python3
import sys
import os
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import numpy as np
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from numba import njit
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import argparse
import re

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
    argparse.Namespace: Parsed arguments containing paths for input and output files.
    """
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument(
        "--tf_names_file",
        type=str,
        required=True,
        help="Path to the tab-separated TF_Information_all_motifs.txt file containing TF name to binding motif association"
    )
    parser.add_argument(
        "--meme_dir",
        type=str,
        required=True,
        help="Path to the directory containing the motif .meme files for the organism"
    )
    parser.add_argument(
        "--reference_genome_dir",
        type=str,
        required=True,
        help="Path to the directory containing the chromosome fasta files for an organism"
    )
    parser.add_argument(
        "--atac_data_file",
        type=str,
        required=True,
        help="Path to the scATACseq data file"
    )
    parser.add_argument(
        "--rna_data_file",
        type=str,
        required=True,
        help="Path to the scRNAseq data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
    )
    parser.add_argument(
        "--species",
        type=str,
        required=True,
        help="Species of the sample, either 'mouse', 'human', 'hg38', or 'mm10'"
    )
    parser.add_argument(
        "--num_cpu",
        type=str,
        required=True,
        help="Number of processors to run multithreading with"
    )
    
    args: argparse.Namespace = parser.parse_args()

    return args

@njit
def calculate_strand_score(sequence, pwm_values, window_size):
    """
    Calculate the cumulative PWM score over all sliding windows
    for a given strand sequence. Windows containing an ambiguous base (not in mapping)
    are assigned a neutral contribution (score of 0 for that position).
    """
    # Map the sequence to indices; use -1 for ambiguous nucleotides (e.g., 'N')
    score_total = 0
    L = sequence.shape[0]
    
    # Slide over the sequence with the window size
    for i in range(L - window_size):
        window_score = 0.0
        # Sum over the PWM positions.
        for j in range(window_size):
            window_score += pwm_values[j, sequence[i + j]]
        score_total += window_score
    return score_total

def process_motif_file(file, meme_dir, chr_pos_to_seq, background_freq, tf_df):
    # Read in the motif PWM file.
    motif_df = pd.read_csv(os.path.join(meme_dir, file), sep="\t", header=0, index_col=0)
    motif_name = file.replace('.txt', '')
    
    # Calculate the log2-transformed PWM (with background correction)
    log2_motif_df_freq = np.log2(motif_df.T.div(background_freq, axis=0) + 1).T
    
    # Set scores for ambiguous base 'N' to 0.
    log2_motif_df_freq["N"] = [0] * log2_motif_df_freq.shape[0]
    
    pwm_values = log2_motif_df_freq.to_numpy()
    window_size = log2_motif_df_freq.shape[0]
    
    n_peaks = chr_pos_to_seq.shape[0]
    total_peak_score = np.zeros(n_peaks)
    
    # Loop over each peak to compute binding scores for both strands.
    # (Assume that the sequences stored in chr_pos_to_seq["+ seq"] and ["- seq"]
    # are NumPy arrays of integers.)
    for peak_num in range(n_peaks):
        peak = chr_pos_to_seq.iloc[peak_num, :]
        pos_seq = peak["+ seq"]  # already a NumPy array of ints
        neg_seq = peak["- seq"]
        
        pos_strand_score = calculate_strand_score(pos_seq, pwm_values, window_size)
        neg_strand_score = calculate_strand_score(neg_seq, pwm_values, window_size)
        total_peak_score[peak_num] = pos_strand_score + neg_strand_score

    # Get the list of TF names that correspond to this motif.
    tf_names = tf_df.loc[tf_df["Motif_ID"] == motif_name, "TF_Name"].values
    return motif_name, tf_names, total_peak_score

def get_background_freq(species):
    if species == "human" or species == "hg38":
        background_freq = pd.Series({
            "A": 0.29182,
            "C": 0.20818,
            "G": 0.20818,
            "T": 0.29182
        })
    
    elif species == "mouse" or species == "mm10":
        background_freq = pd.Series({
        "A": 0.2917,
        "C": 0.2083,
        "G": 0.2083,
        "T": 0.2917
    })
        
    else:
        raise Exception(f"Species {species} is not 'human', 'mouse', 'hg38', or 'mm10'")

    return background_freq
    

def associate_tf_with_motif_pwm(tf_names_file, meme_dir, chr_pos_to_seq, rna_data_genes, species, num_cpu):

    background_freq = get_background_freq(species)
    
    # Read in the list of TFs to extract their name and matching motif ID
    tf_df = pd.read_csv(tf_names_file, sep="\t", header=0, index_col=None)
    
    # logging.info(f'tf_df TFs: {tf_df["TF_Name"][0:5]}')
    # logging.info(f'RNA dataset TFs: {rna_data_genes}')
    
    
    tf_df = tf_df[tf_df["TF_Name"].isin(rna_data_genes)]
    logging.info(f'Number of TFs matching RNA dataset = {tf_df.shape[0]}')
    
    tf_motif_names = tf_df["Motif_ID"].unique()
    logging.info(f'Number of motifs: {len(tf_motif_names)}')
    logging.info(f'Number of peaks: {chr_pos_to_seq.shape[0]}')
    
    tf_to_peak_score_df = pd.DataFrame()
    tf_to_peak_score_df["peak_id"] = chr_pos_to_seq.apply(
        lambda row: f'{row["chr"]}:{row["start"]}-{row["end"]}', axis=1
    )
    
    
    
    # Identify motif files that match the TF motifs.
    matching_motif_files = [file for file in os.listdir(meme_dir)
                            if file.replace('.txt', '') in tf_motif_names]
    logging.info(f'Number of TF meme files matching motifs: {len(matching_motif_files)}')
    
    logging.info(f'\nCalclating motif binding scores for each ATACseq peak and matching TFs to motifs')
    logging.info(f'\tUsing {num_cpu} processors')
    logging.info(f'\tSize of calculation:') 
    logging.info(f'\t\t{len(tf_motif_names)} motifs x {chr_pos_to_seq.shape[0]} peaks = {len(tf_motif_names) * chr_pos_to_seq.shape[0]} computations')
    
    window_len_set = set()
    for file in matching_motif_files:
        motif_df = pd.read_csv(os.path.join(meme_dir, file), sep="\t", header=0, index_col=0)
        window_len = motif_df.shape[0]
        window_len_set.add(window_len)
    
    # Use ProcessPoolExecutor to parallelize processing of motif files.
    with ProcessPoolExecutor(max_workers=num_cpu) as executor:
        futures = {
            executor.submit(process_motif_file, file, meme_dir, chr_pos_to_seq,
                            background_freq, tf_df): file
            for file in matching_motif_files
        }
        
        new_columns = {}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing motifs"):
            motif_name, tf_names, total_peak_score = future.result()
            for tf_name in tf_names:
                new_columns[tf_name] = total_peak_score

        new_columns_df = pd.DataFrame(new_columns, index=tf_to_peak_score_df.index)
        tf_to_peak_score_df = pd.concat([tf_to_peak_score_df, new_columns_df], axis=1)
    
    logging.info(tf_to_peak_score_df.head())
    return tf_to_peak_score_df

def format_peaks(atac_df: pd.DataFrame, cicero_peak_names: list):
        # Validate that the input DataFrame has the expected structure
    if atac_df.empty:
        raise ValueError("Input ATAC-seq data is empty.")
    
    # Extract the peak ID column
    # logging.info(f'First column of atac_df:\n{atac_df.iloc[:, 0].head()}')  # Check first column
    # logging.info(f'First few peak names:\n{list(cicero_peak_names)[:5]}')  # Check first few peak names
    # logging.info(f'Data Type:\n{atac_df.iloc[:, 0].dtype}')  # Check data type
    peak_ids = atac_df[atac_df.iloc[:, 0].astype(str).isin(map(str, cicero_peak_names))].iloc[:, 0].astype(str)
    
    logging.info(f'Formatting {peak_ids.shape[0]} peaks')
    # logging.info(peak_ids)
    
    # Split peak IDs into chromosome, start, and end
    try:
        chromosomes: pd.Series = peak_ids.str.extract(r'([^:]+):')[0]
        starts: pd.Series = peak_ids.str.extract(r':(\d+)-')[0]
        ends: pd.Series = peak_ids.str.extract(r'-(\d+)$')[0]
    except Exception as e:
        raise ValueError(f"Error parsing 'peak_id' values: {e}")

    # Check for missing or invalid values
    if chromosomes.isnull().any() or starts.isnull().any() or ends.isnull().any():
        raise ValueError("One or more peak IDs are malformed. Ensure all peak IDs are formatted as 'chr:start-end'.")

    # Create a dictionary for constructing the HOMER-compatible DataFrame
    peak_dict: dict[str, Any] = {
        "PeakID": [f"peak{i + 1}" for i in range(len(peak_ids))],  # Generate unique peak IDs
        "chr": chromosomes,
        "start": pd.to_numeric(starts, errors='coerce'),  # Convert to numeric and handle errors
        "end": pd.to_numeric(ends, errors='coerce'),      # Convert to numeric and handle errors
        "strand": ["."] * len(peak_ids),                 # Set strand as "."
    }
    
    peak_df = pd.DataFrame(peak_dict)
    
    return peak_df

def find_ATAC_peak_sequence(peak_df, reference_genome_dir, parsed_peak_file):
    logging.info("Reading in ATACseq peak file")
    # Read in the Homer peaks dataframe
    chr_seq_list = []
    logging.info("Finding DNA sequence for each ATAC peak")
    
    logging.info("Reading in genome fasta files")
    files_to_open = []
    for file in os.listdir(reference_genome_dir):
        if ".fa" in file:
            
            file_path = os.path.join(reference_genome_dir, file)
            files_to_open.append(file_path)
    
    lookup = np.full(256, -1, dtype=np.int8)  # Default: ambiguous characters get -1.
    lookup[ord('A')] = 0
    lookup[ord('C')] = 1
    lookup[ord('G')] = 2
    lookup[ord('T')] = 3
    lookup[ord('N')] = 4
    
    logging.info(f'Extracting the peak sequences...')
    # Find the unique chromosomes in the peaks
    peak_chr_ids = set(peak_df["chr"].unique())
    
    # Iterate through each fasta file (chromosome fastas for mouse, entire genome fasta for human)
    for file in tqdm(files_to_open):
        
        # Read in the fasta
        fasta_sequences = SeqIO.parse(open(file), 'fasta')
        
        # Find the sequence for each peak in the ATACseq data
        for chr in fasta_sequences:
            if chr.id in peak_chr_ids:
                chr_seq_plus = str(chr.seq).upper()
                chr_seq_neg = str(chr.seq.complement()).upper()
                chr_peaks = peak_df[peak_df["chr"] == chr.id][["chr", "start", "end"]]
                starts = chr_peaks["start"].to_numpy()
                ends = chr_peaks["end"].to_numpy()
                
                # Convert the sequence string into a NumPy array of uint8 codes, then map:
                chr_seq_plus_mapped = lookup[np.frombuffer(chr_seq_plus.encode('ascii'), dtype=np.uint8)]
                chr_seq_neg_mapped  = lookup[np.frombuffer(chr_seq_neg.encode('ascii'), dtype=np.uint8)]
                
                chr_peaks["+ seq"] = [chr_seq_plus_mapped[start:end] for start, end in zip(starts, ends)]
                chr_peaks["- seq"] = [chr_seq_neg_mapped[start:end] for start, end in zip(starts, ends)]
                chr_peaks = chr_peaks.dropna()
                chr_seq_list.append(chr_peaks)

    # Mouse has separate fasta for each chromosome
    if len(chr_seq_list) > 1:
        chr_pos_to_seq = pd.concat(chr_seq_list)
    
    # Human fasta is one file with all chromosomes
    else:
        chr_pos_to_seq = chr_seq_list[0]
        
    logging.info(f'\tFound sequence for {chr_pos_to_seq.shape[0] / peak_df.shape[0] * 100}% of peaks ({chr_pos_to_seq.shape[0]} / {peak_df.shape[0]})')
    logging.info('Writing to pickle file')
    chr_pos_to_seq.to_pickle(parsed_peak_file)
    logging.info(f'\tDone!')
    
    return chr_pos_to_seq

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()
    tf_names_file: str = args.tf_names_file
    meme_dir: str = args.meme_dir
    reference_genome_dir: str = args.reference_genome_dir
    atac_data_file: str = args.atac_data_file
    rna_data_file: str = args.rna_data_file
    output_dir: str = args.output_dir
    species: str = args.species
    num_cpu: int = int(args.num_cpu)
    
    # Alternative: Set file names manually
    # tf_names_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/motif_information/mm10/TF_Information_all_motifs.txt"
    # meme_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/motif_information/mm10/mm10_motif_meme_files"
    # reference_genome_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/reference_genome/mm10"
    # atac_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC_filtered_L2_E7.5_merged_ATAC.csv"
    # rna_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC_filtered_L2_E7.5_merged_RNA.csv"
    # output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC"
    # num_cpu = 4
    
    logging.info("Reading in parsed Cicero peak to TG file to find associated peaks")
    cicero_peak_file = f"{output_dir}/peak_to_tg_scores.csv"
    cicero_peaks = pd.read_csv(cicero_peak_file, sep="\t", header=0, index_col=None)
    cicero_peak_names = cicero_peaks["peak_id"].to_list()
    logging.info(f'{len(cicero_peak_names)} Cicero peaks')
    
    
    logging.info('Reading scATACseq data')
    atac_df: pd.DataFrame = pd.read_csv(atac_data_file, header=0, index_col=None)
    
    # Read in the RNAseq data file and extract the gene names to find matching TFs
    logging.info('Reading gene names from scATACseq data')
    rna_data = pd.read_csv(rna_data_file, index_col=None)
    rna_data.rename(columns={rna_data.columns[0]: "Genes"}, inplace=True)
    rna_data_genes = set(rna_data["Genes"])
    
    # Read in the peak dataframe containing genomic sequences
    
    parsed_peak_file = f'{output_dir}/peak_sequences.pkl'
    if os.path.exists(parsed_peak_file):
        logging.info('Reading ATACseq peaks from pickle file')
        chr_pos_to_seq = pd.read_pickle(parsed_peak_file)
        
    # Create the peak dataframe containing genomic sequences if it doesn't exist
    else:
        # Read in the ATACseq dataframe and parse the peak locations into a dataframe of genomic locations and peak IDs
        logging.info(f'Identifying ATACseq peak sequences')
        peak_df = format_peaks(atac_df, cicero_peak_names)
        logging.info(peak_df.head())
        
        # Get the genomic sequence from the reference genome to each ATACseq peak
        chr_pos_to_seq = find_ATAC_peak_sequence(peak_df, reference_genome_dir, parsed_peak_file)
        
    # Associate the TFs from TF_Information_all_motifs.txt to the motif with the matching motifID
    tf_to_peak_score_df = associate_tf_with_motif_pwm(tf_names_file, meme_dir, chr_pos_to_seq, rna_data_genes, species, num_cpu)
    
    # Melt the wide-format dataframe to a three column long format, matching the other output files
    tf_to_peak_score_df = pd.melt(
        frame=tf_to_peak_score_df,
        id_vars="peak",
        value_vars=tf_to_peak_score_df.columns[1:],
        var_name="source_id",
        value_name="sliding_window_score"
        )

    
    tf_to_peak_score_df.to_csv(f'{output_dir}/sliding_window_tf_to_peak_score.tsv', sep='\t', header=True, index=False)
        
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()
