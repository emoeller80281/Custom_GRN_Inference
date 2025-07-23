import os
import cooler
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.sparse import lil_matrix, save_npz
from tqdm import tqdm
import contextlib
import logging
from math import ceil

logging.basicConfig(level=logging.INFO, format='%(message)s')

project_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/"

# Input files
peak_location_file = os.path.join(project_dir, "output/DS011_mESC/DS011_mESC_sample1/tmp/peak_df.parquet")
gene_tss_bedfile = os.path.join(project_dir, "data/genome_annotation/mm10/mm10_TSS.bed")
hic_data_file = os.path.join(project_dir, "data/Hi-C_data/4DNFITHTURR9.mcool::resolutions/1000")

# Output file
contact_matrix_file = os.path.join(project_dir, 'dev/notebooks/hic_peak_to_tg.parquet')

logging.info("Loading Cooler")
clr = cooler.Cooler(hic_data_file)

logging.info("Reading mm10 TSS file")
mm10_tss = pd.read_csv(
    gene_tss_bedfile, 
    sep="\t", 
    header=None, 
    index_col=None,
    names=["chrom", "start", "end", "name", "score", "strand"]
    )
mm10_tss.head()

logging.info("Reading ATAC Peaks file")
atac_peaks = pd.read_parquet(peak_location_file)
atac_peaks = atac_peaks.rename(columns={"chr":"chrom"})
atac_peaks['chrom'] = 'chr' + atac_peaks['chrom'].astype(str)

def find_contact_frequency_between_coords(coord1, coord2):
    bin1 = clr.bins().fetch(coord1).index[0]
    bin2 = clr.bins().fetch(coord2).index[0]

    # Get contact frequency between the two bins
    contact_value = clr.matrix(balance=True)[bin1, bin2].item()
    return contact_value

logging.info("Pre-parsing TSS and peak genomic coordinates")
mm10_tss["gene_position"] = [
    f"{c}:{s}-{e}" for c, s, e in zip(mm10_tss["chrom"], mm10_tss["start"], mm10_tss["end"])
]
atac_peaks["peak_position"] = [
    f"{c}:{s}-{e}" for c, s, e in zip(atac_peaks["chrom"], atac_peaks["start"], atac_peaks["end"])
]

n_peaks = len(atac_peaks)
n_genes = len(mm10_tss)

# Precompute for fast access
peak_positions = atac_peaks["peak_position"].tolist()
peak_chroms = atac_peaks["chrom"].to_numpy()
peak_starts = atac_peaks["start"].to_numpy()
peak_pos_to_idx = {pos: i for i, pos in enumerate(peak_positions)}

gene_positions = mm10_tss["gene_position"].tolist()
gene_pos_to_idx = {pos: j for j, pos in enumerate(gene_positions)}

# Parallel function for each gene
def process_gene(j, row):
    local_entries = []
    gene_chrom = row["chrom"]
    gene_start = row["start"]
    gene_pos = row["gene_position"]

    # Mask for peaks within 1Mb and on same chromosome
    mask = (peak_chroms == gene_chrom) & (np.abs(peak_starts - gene_start) <= 1_000_000)
    filtered_peaks = atac_peaks[mask]

    for _, peak_row in filtered_peaks.iterrows():
        peak_pos = peak_row["peak_position"]
        i = peak_pos_to_idx[peak_pos]
        try:
            val = find_contact_frequency_between_coords(peak_pos, gene_pos)
            if not np.isnan(val) and val > 0:
                local_entries.append((i, j, val))
        except Exception:
            continue
    return local_entries

# Run in parallel
logging.info("Extracting Hi-C contact values between peaks and genes")
update_interval = ceil(len(mm10_tss) / 100)  # every 1%

results = Parallel(n_jobs=64)(
    delayed(process_gene)(j, row)
    for j, row in tqdm(
        list(mm10_tss.iterrows()),
        desc="Processing genes",
        total=len(mm10_tss),
        miniters=update_interval
    )
)

# Flatten and populate sparse matrix
logging.info("Creating lil_matrix of the results")
contact_matrix = lil_matrix((n_peaks, n_genes), dtype=np.float32)
for triplets in results:
    for i, j, val in triplets:
        contact_matrix[i, j] = val

logging.info("Converting to coo matrix")
contact_matrix = contact_matrix.tocoo()

atac_peaks["peak_id"] = [
    f"{c}:{s}-{e}" for c, s, e in zip(atac_peaks["chrom"], atac_peaks["start"], atac_peaks["end"])
]

gene_names = mm10_tss["name"].values
peak_positions = atac_peaks["peak_id"].values

contact_df = pd.DataFrame({
    "peak_id": peak_positions[contact_matrix.row],
    "target_id": gene_names[contact_matrix.col],
    "contact_value": contact_matrix.data
})

contact_df.to_parquet(contact_matrix_file, engine="pyarrow", compression="snappy")