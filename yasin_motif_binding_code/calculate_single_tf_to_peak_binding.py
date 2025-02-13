#!/usr/bin/env python3
import sys
import os
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


def associate_tf_with_motif_pwm(tf_names_file, meme_dir, pwm_output_dir):
    # Read in the list of TFs to extract their name and matching motif ID
    tf_df = pd.read_csv(tf_names_file, sep="\t", header=0, index_col=None)
    
    tf_pwm_dict = {}
    
    mm10_background_freq = pd.Series({
        "A": 0.2917,
        "C": 0.2083,
        "G": 0.2083,
        "T": 0.2917
    })

    # Read in each of the motifs to generate a dictionary of motif name to a dataframe containing the motif
    for file in tqdm(os.listdir(meme_dir)[0:1]):
        
        # Read in the motif pwm file
        motif_df = pd.read_csv(os.path.join(meme_dir, file), sep="\t", header=0, index_col=0)
        motif_name = file.replace('.txt', '')
        
        # Calculate the log2-transformed PWM (with background correction)
        log2_motif_df_freq = np.log2(motif_df.T.div(mm10_background_freq, axis=0) + 1).T
        
        # Set scores for ambiguous base 'N' to 0.
        log2_motif_df_freq["N"] = [0] * log2_motif_df_freq.shape[0]
        
        # Find the TF name that corresponds to the motif ID
        tf_name = tf_df.loc[tf_df["Motif_ID"] == motif_name, "TF_Name"]
        
        # For each TF that matches, write the motif to a .ppm file with the TF name
        for tf in tf_name:
            tf_pwm_dict[tf] = log2_motif_df_freq
            motif_df.to_csv(os.path.join(pwm_output_dir, f"{tf}.pwm"), sep="\t", header=True, index=True)    
    
    return tf_pwm_dict

def find_ATAC_peak_sequence(peak_file, reference_genome_dir, parsed_peak_file):
    print("Reading in ATACseq peak file")
    # Read in the Homer peaks dataframe
    peaks = pd.read_csv(peak_file, sep="\t", header=None, index_col=None)
    peaks.columns = ["PeakID", "chr", "start", "end", "strand"]
    print(peaks.head())
    
    peak_chromosomes = set(peaks["chr"])
    chr_seq_list = []
    print("Finding DNA sequence for each ATAC peak")
    print("Reading in mm10 fasta files")
    for file in os.listdir(reference_genome_dir):
        if ".fa" in file:
            file_chr_name = file.replace(".fa", "")
            if file_chr_name in peak_chromosomes:
                print(file_chr_name)
                file_path = os.path.join(reference_genome_dir, file)
                # Read in the mm10 genome from Homer
                fasta_sequences = SeqIO.parse(open(file_path), 'fasta')
                
                # Find the sequence for each peak in the ATACseq data
                for chr in fasta_sequences:
                    if chr.id in list(peaks["chr"]):
                        chr_seq_plus = str(chr.seq).upper() # Coding strand
                        chr_seq_neg = str(chr.seq.complement()).upper() # Template strand
                        chr_peaks = peaks[peaks["chr"] == chr.id][["chr", "start", "end"]]
                        starts = list(chr_peaks["start"])
                        ends = list(chr_peaks["end"])
                        chr_peaks["+ seq"] = [chr_seq_plus[start:end] for start, end in zip(starts, ends)]
                        chr_peaks["- seq"] = [chr_seq_neg[start:end] for start, end in zip(starts, ends)]
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
    
    pos_seq = first_row["+ seq"][0:num_nucleotides]
    neg_seq = first_row["- seq"][0:num_nucleotides]
    
    tf_motif = tf_pwm_dict[tf_name]
    
    def calculate_partial_strand_score(sequence, tf_motif):
        window_size = tf_motif.shape[0]
        num_peak_nucleotides = len(sequence)
        # print(f'\t{sequence}')
        
        peak_scores = []
        for i in range(num_peak_nucleotides-window_size+1):
            window = [i for i in sequence[i:i+window_size]]
            # print(f'Window {i+1}: {window}')
            score = np.sum([tf_motif.loc[i+1, letter] for i, letter in enumerate(window)])
            # print(f'\tScore = {score}\n')
            peak_scores.append(score)
        
        # min_score = abs(min(peak_scores))
        # peak_scores = [i + min_score for i in peak_scores]
        return peak_scores
    
    pos_peak_scores = calculate_partial_strand_score(pos_seq, tf_motif)
    neg_peak_scores = calculate_partial_strand_score(neg_seq, tf_motif)
    
    # print(f'\n----- Positive Peak Scores -----')
    # for score in pos_peak_scores:
    #     print(score)
        
    # print(f'\n----- Negative Peak Scores -----')
    # for score in neg_peak_scores:
    #     print(score)
    
    # print(f'\tCoding Strand = {sum(pos_peak_scores)}')
    # print(f'\tTemplate Strand = {sum(neg_peak_scores)}')
    
    total_score = sum(pos_peak_scores) + sum(neg_peak_scores)
    # print(f'\tTotal = {total_score}')
    
    return total_score
    
    # neg_peak_scores = [-1 * i for i in neg_peak_scores]
    # x = range(len(pos_peak_scores))
    
    # # Plot a barplot of the positive and negative strands
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # fig.set_figheight(8)
    # fig.set_figwidth(18)
    # fig.suptitle(f"{tf_name} binding score along peak {peak_name}", fontsize=18)
    
    # ax1.bar(x, pos_peak_scores, width=1, color='b')
    # ax1.set_xticks(ticks=[i for i in range(len(pos_seq[0:num_nucleotides]))], labels=[i for i in pos_seq[0:num_nucleotides]], fontsize=11)
    # ax1.set_ylabel("Coding strand binding potential")
    
    # ax2.bar(x, neg_peak_scores, width=1, color='b')
    # ax2.set_xticks(ticks=[i for i in range(len(neg_seq[0:num_nucleotides]))], labels=[i for i in neg_seq[0:num_nucleotides]], fontsize=11)
    # ax2.tick_params(labelbottom=False, labeltop=True, top=True, bottom=False)
    # ax2.set_ylabel("Template strand binding potential")

    # ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: str(abs(y))))
    
    # plt.tight_layout()
    # plt.savefig("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/peak_scores.png", dpi=500)
 

def main():
    tf_names_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/motif_pwms/TF_Information_all_motifs.txt"
    meme_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/motif_pwms/pwms_all_motifs"
    pwm_output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/motif_binding_output"
    reference_genome_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/Homer/data/genomes/mm10"
    peak_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/Homer_peaks.txt"
    rna_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC_filtered_L2_E7.5_merged_RNA.csv" 
    
    # Get the genomic sequence from the reference genome to each ATACseq peak
    # Read in the peak dataframe containing genomic sequences
    parsed_peak_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/ind_peak_sequences.pkl"
    if os.path.exists(parsed_peak_file):
        chr_pos_to_seq = pd.read_pickle(parsed_peak_file)
        
    # Create the peak dataframe containing genomic sequences if it doesn't exist
    else:
        # Get the genomic sequence from the reference genome to each ATACseq peak
        chr_pos_to_seq = find_ATAC_peak_sequence(peak_file, reference_genome_dir, parsed_peak_file)


    tf_pwm_dict = associate_tf_with_motif_pwm(tf_names_file, meme_dir, pwm_output_dir)    
    
    tf_name = "Hoxb1"
    num_nucleotides=21
    
    tf_to_peak_scores = []
    for i in range(20):
        peak_num=i
        # print(f'Peak {i+1}')
        peak_score = calculate_single_peak_scores(chr_pos_to_seq, tf_pwm_dict, peak_num, tf_name, num_nucleotides)
        tf_to_peak_scores.append(peak_score)
    
    # Read in the RNAseq data file and extract the gene names to find matching TFs
    rna_data = pd.read_csv(rna_data_file, index_col=0)
    rna_data = rna_data.rename(columns={rna_data.columns[0]: "gene"}).set_index("gene")  
    
    rna_data["mean_expression"] = np.log2(rna_data.values.mean(axis=1))
    print(rna_data.loc["Hoxb1", "mean_expression"])
    
    rna_data["norm_mean_expression"] = (rna_data["mean_expression"] - rna_data["mean_expression"].min()) / (rna_data["mean_expression"].max() - rna_data["mean_expression"].min())
    
    
    print(rna_data.loc["Hoxb1", "norm_mean_expression"])
      

if __name__ == "__main__":
    main()