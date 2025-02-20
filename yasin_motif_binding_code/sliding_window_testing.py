import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import pool

# seq = "ACTGCTANGCTATANGCATAGCTNATNCGANTCANAGNTACNAGNTACNAGANTACAGATACAG"
motif_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/mm10_motif_meme_files"

encoding = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4
}


all_motifs = {}
window_sizes = set()
window_lengths = {}
for motif_file in tqdm(os.listdir(motif_dir)[:100]):
    motif_name = motif_file.replace('.txt', '')
    # print(motif_name)

    pwm = pd.read_csv(os.path.join(motif_dir, motif_file), sep="\t", header=0, index_col=0)
    pwm['N'] = [0] * pwm.shape[0]

    # print(pwm)
    window_size = pwm.shape[0]

    pwm = pwm.to_numpy()
    window_size = pwm.shape[0]
    
    all_motifs[motif_name] = pwm
    
    if window_size > 0:
        window_sizes.add(window_size)
    
        if window_size not in window_lengths:
            window_lengths[window_size] = []

print(window_sizes)

def build_windows_vectorized(encoded_seq, w):
    L = len(encoded_seq)
    N = L - w + 1
    # Create the 2D array
    arr = np.empty((N, w), dtype=int)
    # Fill column by column
    for col in range(w):
        arr[:, col] = encoded_seq[col : col + N]
    return arr

parsed_peak_file = f'/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/ind_peak_sequences.pkl'
if os.path.exists(parsed_peak_file):
    chr_pos_to_seq = pd.read_pickle(parsed_peak_file)
    print(chr_pos_to_seq.shape)
    
    peak_names = []
    peak_windows = {}
    
    print(f'Preprocessing {chr_pos_to_seq.shape[0]} peaks')
    def sliding_windows_gen(encoded_seq, w):
        L = len(encoded_seq)
        for i in range(L - w + 1):
            yield encoded_seq[i:i+w]

    # Then, for scoring:
    for peak_name, row in tqdm(chr_pos_to_seq.iterrows(), total=chr_pos_to_seq.shape[0]):
        encoded_seq = [encoding[nuc] for nuc in row["+ seq"]]
        for motif_file, pwm in all_motifs.items():
            motif_window_size = pwm.shape[0]
            if len(encoded_seq) < motif_window_size:
                continue
            # Build score for each window on the fly:
            total_score = 0
            for window in sliding_windows_gen(encoded_seq, motif_window_size):
                # Convert window to a NumPy array to use advanced indexing:
                window_arr = np.array(window, dtype=int)
                row_indices = np.arange(motif_window_size)
                # Score for this window:
                total_score += pwm[row_indices, window_arr].sum()
            # print(f'Motif: {motif_file}, Peak: {peak_name}, Score: {total_score}')

    

