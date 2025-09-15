
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import pybedtools
from grn_inference import utils
from typing import Tuple

torch.manual_seed(1)
np.random.seed(42)

project_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
mm10_genome_dir = os.path.join(project_dir, "data/reference_genome/mm10")
mm10_gene_tss_file = os.path.join(project_dir, "data/genome_annotation/mm10/mm10_TSS.bed")
ground_truth_dir = os.path.join(project_dir, "ground_truth_files")
sample_input_dir = os.path.join(project_dir, "input/mESC/filtered_L2_E7.5_rep1")
output_dir = os.path.join(project_dir, "output/transformer_testing_output")

mm10_fasta_file = os.path.join(mm10_genome_dir, "chr1.fa")
mm10_chrom_sizes_file = os.path.join(mm10_genome_dir, "chrom.sizes")

print("Reading and formatting TSS bed file")
mm10_gene_tss_bed = pybedtools.BedTool(mm10_gene_tss_file)
gene_tss_df = (
    mm10_gene_tss_bed
    .filter(lambda x: x.chrom == "chr1")
    .saveas(os.path.join(mm10_genome_dir, "mm10_ch1_gene_tss.bed"))
    .to_dataframe()
    .sort_values(by="start", ascending=True)
    )

print("Reading processed scATAC-seq dataset")
mesc_atac_data = pd.read_parquet(os.path.join(sample_input_dir, "mESC_filtered_L2_E7.5_rep1_ATAC_processed.parquet")).set_index("peak_id")
mesc_atac_peak_loc = mesc_atac_data.index

# format the peaks to be in bed_format
mesc_atac_peak_loc = utils.format_peaks(mesc_atac_peak_loc)
mesc_atac_peak_loc = mesc_atac_peak_loc[mesc_atac_peak_loc["chromosome"] == "chr1"]
mesc_atac_peak_loc = mesc_atac_peak_loc.rename(columns={"chromosome":"chrom"})

# TEMPORARY Restrict to Chr1 for testing
mesc_atac_data_chr1 = mesc_atac_data[mesc_atac_data.index.isin(mesc_atac_peak_loc.peak_id)]

print("Reading in the scRNA-seq dataset")
mesc_rna_data = pd.read_parquet(
    os.path.join(sample_input_dir, "mESC_filtered_L2_E7.5_rep1_RNA_processed.parquet")).set_index("gene_id")

print("Creating genomic windows")
window_size = 1000
mm10_genome_windows = pybedtools.bedtool.BedTool().window_maker(g=mm10_chrom_sizes_file, w=window_size)
mm10_chr1_windows = (
    mm10_genome_windows
    .filter(lambda x: x.chrom == "chr1") #TEMPORARY Restrict to Chr1 for testing
    .saveas(os.path.join(mm10_genome_dir, f"mm10_chr1_windows_{window_size // 1000}kb.bed"))
    .to_dataframe()
    )

print("Calculating peak to TG distance score")
peak_bed = pybedtools.BedTool.from_dataframe(
    mesc_atac_peak_loc[["chrom", "start", "end", "peak_id"]]
    ).saveas(os.path.join(output_dir, "peak_tmp.bed"))

tss_bed = pybedtools.BedTool.from_dataframe(
    gene_tss_df[["chrom", "start", "end", "name"]]
    ).saveas(os.path.join(output_dir, "tss_tmp.bed"))

genes_near_peaks = utils.find_genes_near_peaks(peak_bed, tss_bed)

# Restrict to peaks within 1 Mb of a gene TSS
genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= 1e6]

# Scale the TSS distance score by the exponential scaling factor
genes_near_peaks = genes_near_peaks.copy()
genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / 250000)

print("Building Homer peaks file")
homer_peaks = genes_near_peaks[["peak_id", "peak_chr", "peak_start", "peak_end"]]
homer_peaks = homer_peaks.rename(columns={
    "peak_id":"PeakID", 
    "peak_chr":"chr",
    "peak_start":"start",
    "peak_end":"end"
    })
homer_peaks["strand"] = ["."] * len(homer_peaks)
homer_peaks["start"] = round(homer_peaks["start"].astype(int),0)
homer_peaks["end"] = round(homer_peaks["end"].astype(int),0)
homer_peaks = homer_peaks.drop_duplicates(subset="PeakID")

os.makedirs(os.path.join(output_dir, "tmp"), exist_ok=True)
homer_peak_path = os.path.join(output_dir, "tmp/homer_peaks.txt")
homer_peaks.to_csv(homer_peak_path, sep="\t", header=False, index=False)

print("Loading Homer output")
homer_results = pd.read_parquet(os.path.join(output_dir, "homer_tf_to_peak.parquet"), engine="pyarrow")
homer_results = homer_results.reset_index(drop=True)
homer_results["source_id"] = homer_results["source_id"].str.capitalize()

print("Ensuring shared cell barcodes between ATAC and RNA datasets")
atac_cell_barcodes = mesc_atac_data_chr1.columns.to_list()
rna_cell_barcodes = mesc_rna_data.columns.to_list()
atac_in_rna_shared_barcodes = [i for i in atac_cell_barcodes if i in rna_cell_barcodes]

# Make sure that the cell names are in the same order and in both datasets
shared_barcodes = sorted(set(atac_in_rna_shared_barcodes))

mesc_atac_data_chr1_shared = mesc_atac_data_chr1[shared_barcodes]
mesc_rna_data_shared = mesc_rna_data[shared_barcodes]

# TEMPORARY: Set to the first cell in the dataset
rna_first_cell = mesc_rna_data_shared.iloc[:, 0]
atac_first_cell = mesc_atac_data_chr1_shared.loc[:, rna_first_cell.name]

# Print out the dimensionality 
potential_tgs = genes_near_peaks["target_id"].unique()
print(f"Number of potential TGs: {len(potential_tgs)}")

unique_tfs = homer_results["source_id"].unique()
print(f"Number of unique TFs: {len(unique_tfs)}")

unique_peaks = homer_results["peak_id"].unique()
print(f"Number of unique peaks: {len(unique_peaks)}")

print("\nCalculating TF-peak Binding Potential")
tf_peak_binding_potential = pd.merge(homer_results, rna_first_cell, left_on="source_id", right_index=True, how="inner")
tf_peak_binding_potential["tf_peak_binding_potential"] = tf_peak_binding_potential["homer_binding_score"] * tf_peak_binding_potential.iloc[:,-1]
tf_peak_binding_potential = tf_peak_binding_potential[["source_id", "peak_id", "tf_peak_binding_potential"]]

print("\nCalculating Peak-TG Regulatory Potential")
peak_tg_regulatory_potential = pd.merge(genes_near_peaks, atac_first_cell, left_on="peak_id", right_index=True, how="inner")
peak_tg_regulatory_potential["peak_tg_regulatory_potential"] = peak_tg_regulatory_potential["TSS_dist_score"] * peak_tg_regulatory_potential.iloc[:, -1]
peak_tg_regulatory_potential = peak_tg_regulatory_potential[["peak_id", "target_id", "peak_tg_regulatory_potential"]]

print("\nJoining to create TF-Peak-TG Regulatory Potential DataFrame")
tf_peak_tg_regulatory_potential = pd.merge(tf_peak_binding_potential, peak_tg_regulatory_potential, on="peak_id", how="outer")
tf_peak_tg_regulatory_potential["tf_peak_tg_score"] = tf_peak_tg_regulatory_potential["tf_peak_binding_potential"] * tf_peak_tg_regulatory_potential["peak_tg_regulatory_potential"]
tf_peak_tg_regulatory_potential = tf_peak_tg_regulatory_potential[["source_id", "peak_id", "target_id", "tf_peak_tg_score"]]
print(tf_peak_tg_regulatory_potential.head())

print("\nAggregating peaks into genomic windows")
def peaks_to_windows(
    tf_peak_tg_regulatory_potential: pd.DataFrame, 
    mm10_chr1_windows: pd.DataFrame
    )-> pd.DataFrame:
    # Parse peak_id into genomic coords (chrom, start, end)
    coords = tf_peak_tg_regulatory_potential["peak_id"].str.extract(
        r"(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)"
    ).astype({"start":"int64","end":"int64"})

    df = pd.concat([tf_peak_tg_regulatory_potential, coords], axis=1)

    df = df[df["chrom"] == "chr1"].copy()

    window_size = int((mm10_chr1_windows["end"] - mm10_chr1_windows["start"]).mode().iloc[0])

    # Build a quick lookup of window_id strings from window indices
    # window index k -> [start=k*w, end=(k+1)*w)
    win_lut = {}
    for _, row in mm10_chr1_windows.iterrows():
        k = row["start"] // window_size
        win_lut[k] = f'{row["chrom"]}:{row["start"]}-{row["end"]}'

    # --- Assign each unique peak to the window with maximal overlap (random ties) ---
    rng = np.random.default_rng(0)  # set a seed for reproducibility; change/remove if you want different random choices

    peaks_unique = (
        df.loc[:, ["peak_id", "chrom", "start", "end"]]
        .drop_duplicates(subset=["peak_id"])
        .reset_index(drop=True)
    )

    def assign_best_window(start, end, w):
        # windows indices spanned by the peak (inclusive)
        i0 = start // w
        i1 = (end - 1) // w  # subtract 1 so exact boundary end==k*w goes to k-1 window
        if i1 < i0:
            i1 = i0
        # compute overlaps with all spanned windows
        overlaps = []
        for k in range(i0, i1 + 1):
            bin_start = k * w
            bin_end = bin_start + w
            ov = max(0, min(end, bin_end) - max(start, bin_start))
            overlaps.append((k, ov))
        # choose the k with max overlap; break ties randomly
        ov_vals = [ov for _, ov in overlaps]
        max_ov = max(ov_vals)
        candidates = [k for (k, ov) in overlaps if ov == max_ov]
        if len(candidates) == 1:
            return candidates[0]
        else:
            return rng.choice(candidates)

    peak_to_window_idx = peaks_unique.apply(
        lambda r: assign_best_window(r["start"], r["end"], window_size), axis=1
    )
    peaks_unique["window_idx"] = peak_to_window_idx
    peaks_unique["window_id"] = peaks_unique["window_idx"].map(win_lut)

    # Map window assignment back to the full TF–peak–gene table and aggregate
    df = df.merge(
        peaks_unique.loc[:, ["peak_id", "window_id"]],
        on="peak_id",
        how="left"
    )

    # Aggregate scores per TF × window × gene
    binned_scores = (
        df.groupby(["source_id", "window_id", "target_id"], observed=True)["tf_peak_tg_score"]
        .sum()
        .reset_index()
    ).rename(columns={"tf_peak_tg_score":"tf_window_tg_score"})

    print(binned_scores.head())
    
    return binned_scores

tf_window_tg_regulatory_potential = peaks_to_windows(tf_peak_tg_regulatory_potential, mm10_chr1_windows)

print("Converting dataframe to numpy matrix")
def convert_tf_window_tg_df_to_numpy(tf_window_tg_df: pd.DataFrame) -> Tuple(np.ndarray, dict):
    # Get unique IDs
    tfs = tf_window_tg_df["source_id"].unique()
    windows = tf_window_tg_df["window_id"].unique()
    genes = tf_window_tg_df["target_id"].unique()

    # Create index maps
    tf_idx = {tf: i for i, tf in enumerate(tfs)}
    window_idx = {p: i for i, p in enumerate(windows)}
    gene_idx = {g: i for i, g in enumerate(genes)}

    # Initialize 3D matrix
    data_array = np.zeros((len(tfs), len(windows), len(genes)), dtype=float)

    # Fill values
    for _, row in tf_window_tg_df.iterrows():
        i = tf_idx[row["source_id"]]
        j = window_idx[row["window_id"]]
        k = gene_idx[row["target_id"]]
        data_array[i, j, k] = row["tf_window_tg_score"]

    print(data_array.shape)  # (n_TFs, n_windows, n_genes)
    print(f"Saving to {os.path.join(output_dir, 'tf_window_gene_tensor.npz')}")
    
    np.savez_compressed(
        os.path.join(output_dir, "tf_window_gene_tensor.npz"),
        tensor=data_array,
        tfs=tfs,
        windows=windows,
        genes=genes
    )
    
    return data_array, genes

tf_window_tg_regulatory_potential_numpy, genes = convert_tf_window_tg_df_to_numpy(tf_window_tg_regulatory_potential)

X = torch.from_numpy(tf_window_tg_regulatory_potential_numpy).float()     # 106 TF x 12,480 windows x 1,425 TGs

# Standardizing the window data per TF across the TGs
print("Standardizing data distribution")
X = (X - X.mean(dim=2, keepdim=True)) / (X.std(dim=2, keepdim=True) + 1e-6)
X = torch.clamp(X, -5, 5)

TF, W, TG = X.shape
d_model = 256
tf_channels = 32
tg_channels = 64

# Project the TF, window, and TG dimensions down to a size of 
proj_tg = nn.Linear(TG, tg_channels, bias=False)
Xg = proj_tg(X)             # TF, W, 64

proj_tf = nn.Linear(TF, tf_channels, bias=False)
Xg = Xg.permute(1, 2, 0)    # W, 64, TF
Xg = proj_tf(Xg)            # W, 64, 32

window_features = Xg.reshape(W, tg_channels * tf_channels) 
proj_window = nn.Linear(tg_channels * tf_channels, d_model)
tokens = proj_window(window_features)   # W, 256

# Downsample window dimension by average pooling across windows
pool = nn.AvgPool1d(kernel_size=8, stride=8)  # along sequence length, bins and pools the data
tokens = tokens.unsqueeze(0)
tokens_ds = pool(tokens.transpose(1,2)).transpose(1,2)   # [1, W', d_model]
W_ds = tokens_ds.size(1)

# Set up the encoder
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model, nhead=8, dim_feedforward=512, batch_first=True
)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

window_key_values_encoding = encoder(tokens_ds)   # [1, W', d_model]

n_genes = len(genes)  # 1425
gene_embed = nn.Embedding(n_genes, d_model)

# Create the multi-headed attention block
cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
readout = nn.Sequential(
    nn.LayerNorm(d_model),
    nn.Linear(d_model, 1)
)

# Build gene queries (index 0..n_genes-1)
gene_ids = torch.arange(n_genes)
gene_queries = gene_embed(gene_ids).unsqueeze(0)  # [1, n_genes, d_model]

# Cross-attention: (Q=genes, K/V=windows)
Z, _ = cross_attn(
    query=gene_queries, 
    key=window_key_values_encoding, 
    value=window_key_values_encoding
    )  # [1, n_genes, d_model]

pred_expr = readout(Z).squeeze(-1)           # [1, n_genes]

# Register the parameters
params = (
    list(proj_tg.parameters()) +
    list(proj_tf.parameters()) +
    list(proj_window.parameters()) +
    list(encoder.parameters()) +
    list(gene_embed.parameters()) +
    list(cross_attn.parameters()) +
    list(readout.parameters())
    # + list(rpb.parameters())  # if you use the custom RPB blocks
)
opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

def get_true_tg_expr_vector_from_data(rna_dataset: pd.Series, genes: dict):
    dup = pd.Index(genes).duplicated()
    assert not dup.any(), f"Duplicate gene IDs in prediction axis at: {np.where(dup)[0][:10]}"

    # Align counts to prediction order from the gene to index mapping (same length and order as genes)
    true_counts = rna_dataset.reindex(genes)

    # build mask for missing genes (not present in RNA)
    mask = ~true_counts.isna().to_numpy()

    # Handle missing genes using a masked loss 
    y_true_vec = true_counts.to_numpy(dtype=float)        # shape (n_genes,)

    y_true = torch.tensor(y_true_vec, dtype=torch.float32).unsqueeze(0)   # [1, n_genes]
    mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)            # [1, n_genes]
    
    return y_true, mask_t

y_true, mask_t = get_true_tg_expr_vector_from_data(rna_first_cell, genes)

def masked_mse(pred, y, m):
    diff2 = (pred - y)**2
    return diff2[m].mean()

# pred_expr: [1, n_genes] from the model
loss = masked_mse(pred_expr, y_true, mask_t)
print(loss)