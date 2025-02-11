#!/usr/bin/env python3
import sys
import os
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from numba import njit
from concurrent.futures import ProcessPoolExecutor, as_completed



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

def process_motif_file(file, meme_dir, chr_pos_to_seq, mm10_background_freq, tf_df):
    # Read in the motif PWM file.
    motif_df = pd.read_csv(os.path.join(meme_dir, file), sep="\t", header=0, index_col=0)
    motif_name = file.replace('.txt', '')
    
    # Calculate the log2-transformed PWM (with background correction)
    log2_motif_df_freq = np.log2(motif_df.T.div(mm10_background_freq, axis=0) + 1).T
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

def associate_tf_with_motif_pwm(tf_names_file, meme_dir, chr_pos_to_seq, rna_data_genes):
    mm10_background_freq = pd.Series({
        "A": 0.2917,
        "C": 0.2083,
        "G": 0.2083,
        "T": 0.2917
    })
    
    # Read in the list of TFs to extract their name and matching motif ID
    tf_df = pd.read_csv(tf_names_file, sep="\t", header=0, index_col=None)
    
    tf_df = tf_df[tf_df["TF_Name"].isin(rna_data_genes)]
    print(f'Number of TFs matching RNA dataset = {tf_df.shape[0]}')
    
    tf_to_peak_score_df = pd.DataFrame()
    tf_to_peak_score_df["peak"] = chr_pos_to_seq.apply(
        lambda row: f'{row["chr"]}:{row["start"]}-{row["end"]}', axis=1
    )
    
    tf_motif_names = tf_df["Motif_ID"].unique()
    
    # Identify motif files that match the TF motifs.
    matching_motif_files = [file for file in os.listdir(meme_dir)
                            if file.replace('.txt', '') in tf_motif_names]
    
    # Use ProcessPoolExecutor to parallelize processing of motif files.
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_motif_file, file, meme_dir, chr_pos_to_seq,
                            mm10_background_freq, tf_df): file
            for file in matching_motif_files
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing motifs"):
            motif_name, tf_names, total_peak_score = future.result()
            # For each TF associated with this motif, add a new column.
            for tf_name in tf_names:
                tf_to_peak_score_df[tf_name] = total_peak_score
    
    print(tf_to_peak_score_df.head())
    return tf_to_peak_score_df

def find_ATAC_peak_sequence(peak_file, reference_genome_dir, parsed_peak_file):
    print("Reading in ATACseq peak file")
    # Read in the Homer peaks dataframe
    peaks = pd.read_csv(peak_file, sep="\t", header=None, index_col=None)
    peaks.columns = ["PeakID", "chr", "start", "end", "strand"]    
    peak_chromosomes = set(peaks["chr"])
    chr_seq_list = []
    print("Finding DNA sequence for each ATAC peak")
    print("Reading in mm10 chromosome fasta files")
    files_to_open = []
    for file in os.listdir(reference_genome_dir):
        if ".fa" in file:
            file_chr_name = file.replace(".fa", "")
            if file_chr_name in peak_chromosomes:
                file_path = os.path.join(reference_genome_dir, file)
                # Read in the mm10 genome from Homer
                files_to_open.append(file_path)
    print(f'Found {len(files_to_open)} chromosome fasta files matching the peak locations')
    
    lookup = np.full(256, -1, dtype=np.int8)  # Default: ambiguous characters get -1.
    lookup[ord('A')] = 0
    lookup[ord('C')] = 1
    lookup[ord('G')] = 2
    lookup[ord('T')] = 3
    lookup[ord('N')] = 4
    
    print(f'Extracting the peak sequences...')
    peak_chr_ids = set(peaks["chr"].unique())
    for file in tqdm(files_to_open):
        fasta_sequences = SeqIO.parse(open(file), 'fasta')
        # Find the sequence for each peak in the ATACseq data
        for chr in fasta_sequences:
            if chr.id in peak_chr_ids:
                chr_seq_plus = str(chr.seq).upper()
                chr_seq_neg = str(chr.seq.reverse_complement()).upper()
                chr_peaks = peaks[peaks["chr"] == chr.id][["chr", "start", "end"]]
                starts = chr_peaks["start"].to_numpy()
                ends = chr_peaks["end"].to_numpy()
                
                # Convert the sequence string into a NumPy array of uint8 codes, then map:
                chr_seq_plus_mapped = lookup[np.frombuffer(chr_seq_plus.encode('ascii'), dtype=np.uint8)]
                chr_seq_neg_mapped  = lookup[np.frombuffer(chr_seq_neg.encode('ascii'), dtype=np.uint8)]
                
                chr_peaks["+ seq"] = [chr_seq_plus_mapped[start:end] for start, end in zip(starts, ends)]
                chr_peaks["- seq"] = [chr_seq_neg_mapped[start:end] for start, end in zip(starts, ends)]
                chr_peaks = chr_peaks.dropna()
                chr_seq_list.append(chr_peaks)
            
    chr_pos_to_seq = pd.concat(chr_seq_list)
    print(f'\tFound sequence for {chr_pos_to_seq.shape[0] / peaks.shape[0] * 100}% of peaks ({chr_pos_to_seq.shape[0]} / {peaks.shape[0]})')
    print('Writing to pickle file')
    chr_pos_to_seq.to_pickle(parsed_peak_file)
    print(f'\tDone!')
    
    return chr_pos_to_seq

def calculate_single_peak_scores(chr_pos_to_seq, tf_pwm_dict, peak_num, tf_name, num_nucleotides):
    first_row = chr_pos_to_seq.iloc[peak_num, :]
    peak_name = f'{first_row["chr"]}:{first_row["start"]}-{first_row["start"] + num_nucleotides}'
    
    pos_seq = first_row["+ seq"]
    neg_seq = first_row["- seq"]
    
    tf_motif = tf_pwm_dict[tf_name]
    
    def calculate_partial_strand_score(sequence, tf_motif, num_nucleotides):
        window_size = tf_motif.shape[0]
        num_peak_nucleotides = len(sequence[0:num_nucleotides])
        
        peak_scores = []
        for i in range(num_peak_nucleotides):
            window = [i for i in sequence[i:i+window_size]]
            score = np.sum([tf_motif.loc[i+1, letter] for i, letter in enumerate(window)])
            peak_scores.append(score)
        
        min_score = abs(min(peak_scores))
        peak_scores = [i + min_score for i in peak_scores]
        return peak_scores
    
    pos_peak_scores = calculate_partial_strand_score(pos_seq, tf_motif, num_nucleotides)
    neg_peak_scores = calculate_partial_strand_score(neg_seq, tf_motif, num_nucleotides)
    
    total_score = sum(pos_peak_scores) + sum(neg_peak_scores)
    
    neg_peak_scores = [-1 * i for i in neg_peak_scores]
    x = range(len(pos_peak_scores))
    
    # Plot a barplot of the positive and negative strands
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_figheight(8)
    fig.set_figwidth(18)
    fig.suptitle(f"{tf_name} binding score along peak {peak_name}", fontsize=18)
    
    ax1.bar(x, pos_peak_scores, width=1, color='b')
    ax1.set_xticks(ticks=[i for i in range(len(pos_seq[0:num_nucleotides]))], labels=[i for i in pos_seq[0:num_nucleotides]], fontsize=11)
    ax1.set_ylabel("Coding strand binding potential")
    
    ax2.bar(x, neg_peak_scores, width=1, color='b')
    ax2.set_xticks(ticks=[i for i in range(len(neg_seq[0:num_nucleotides]))], labels=[i for i in neg_seq[0:num_nucleotides]], fontsize=11)
    ax2.tick_params(labelbottom=False, labeltop=True, top=True, bottom=False)
    ax2.set_ylabel("Template strand binding potential")

    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: str(abs(y))))
    
    plt.tight_layout()
    plt.savefig("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/peak_scores.png", dpi=500)

def main():
    tf_names_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/motif_pwms/TF_Information_all_motifs.txt"
    meme_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/motif_pwms/pwms_all_motifs"
    reference_genome_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/Homer/data/genomes/mm10"
    peak_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/Homer_peaks.txt"
    rna_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC_filtered_L2_E7.5_merged_RNA.csv"
    
    # Read in the RNAseq data file and extract the gene names to find matching TFs
    rna_data = pd.read_csv(rna_data_file, index_col=0)
    rna_data.rename(columns={rna_data.columns[0]: "Genes"}, inplace=True)
    rna_data_genes = set(rna_data["Genes"])
    
    # Read in the peak dataframe containing genomic sequences
    parsed_peak_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/peak_sequences.pkl"
    if os.path.exists(parsed_peak_file):
        chr_pos_to_seq = pd.read_pickle(parsed_peak_file)
        
    # Create the peak dataframe containing genomic sequences if it doesn't exist
    else:
        # Get the genomic sequence from the reference genome to each ATACseq peak
        chr_pos_to_seq = find_ATAC_peak_sequence(peak_file, reference_genome_dir, parsed_peak_file)

    # Associate the TFs from TF_Information_all_motifs.txt to the motif with the matching motifID
    tf_to_peak_score_df = associate_tf_with_motif_pwm(tf_names_file, meme_dir, chr_pos_to_seq, rna_data_genes)
    
    tf_to_peak_score_df.to_csv("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/tf_to_peak_binding_score.tsv", sep='\t', header=True, index=False)
    
    # calculate_single_peak_scores(chr_pos_to_seq, tf_pwm_dict, peak_num=0, tf_name="Hoxb1", num_nucleotides=100)
    
if __name__ == "__main__":
    main()
