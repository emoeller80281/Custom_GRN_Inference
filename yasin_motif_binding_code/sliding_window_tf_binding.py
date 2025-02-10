#!/usr/bin/env python3
import sys
import os
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO

def associate_tf_with_motif_pwm(tf_names_file, meme_dir, pwm_output_dir):
    # Read in the list of TFs to extract their name and matching motif ID
    tf_df = pd.read_csv(tf_names_file, sep="\t", header=0, index_col=None)
    
    tf_pwm_dict = {}

    # Read in each of the motifs to generate a dictionary of motif name to a dataframe containing the motif
    for file in tqdm(os.listdir(meme_dir)[0:1]):
        
        # Read in the motif pwm file
        motif_df = pd.read_csv(os.path.join(meme_dir, file), sep="\t", header=0, index_col=0)
        motif_name = file.replace('.txt', '')
        
        # Find the TF name that corresponds to the motif ID
        tf_name = tf_df.loc[tf_df["Motif_ID"] == motif_name, "TF_Name"]
        
        # For each TF that matches, write the motif to a .ppm file with the TF name
        for tf in tf_name:
            tf_pwm_dict[tf] = motif_df
            motif_df.to_csv(os.path.join(pwm_output_dir, f"{tf}.pwm"), sep="\t", header=True, index=True)    
    
    return tf_pwm_dict

def find_ATAC_peak_sequence(peak_file, reference_genome_dir):
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
                        chr_seq_plus = str(chr.seq)
                        chr_seq_neg = str(chr.seq.reverse_complement()).upper()
                        chr_peaks = peaks[peaks["chr"] == chr.id][["chr", "start", "end"]]
                        starts = list(chr_peaks["start"])
                        ends = list(chr_peaks["end"])
                        chr_peaks["+ seq"] = [chr_seq_plus[start:end] for start, end in zip(starts, ends)]
                        chr_peaks["- seq"] = [chr_seq_neg[start:end] for start, end in zip(starts, ends)]
                        chr_peaks = chr_peaks.dropna()
                        chr_seq_list.append(chr_peaks)
            
    chr_pos_to_seq = pd.concat(chr_seq_list)
    print(f'Original Peaks: {peaks.shape[0]}')
    print(f'Peaks with matching position: {chr_pos_to_seq.shape[0]}')
    
    return chr_pos_to_seq
    

def main():
    tf_names_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/motif_pwms/TF_Information_all_motifs.txt"
    meme_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/motif_pwms/pwms_all_motifs"
    pwm_output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/motif_binding_output"
    reference_genome_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/Homer/data/genomes/mm10"
    peak_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/Homer_peaks.txt"
    
    # Get the genomic sequence from the reference genome to each ATACseq peak
    chr_pos_to_seq = find_ATAC_peak_sequence(peak_file, reference_genome_dir)

    num_tfs = 1
    tf_pwm_dict = associate_tf_with_motif_pwm(tf_names_file, meme_dir, pwm_output_dir)
    
    print(tf_pwm_dict.keys())
    
    first_row = chr_pos_to_seq.iloc[0, :]
    print(first_row)
    
    first_tf = tf_pwm_dict["Hoxb1"]
    print(first_tf)
    
    
    
    
    
      

if __name__ == "__main__":
    main()
