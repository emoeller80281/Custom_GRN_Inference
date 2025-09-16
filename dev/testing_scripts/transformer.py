
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
from scipy import sparse
from contextlib import nullcontext
import logging

torch.manual_seed(1)
np.random.seed(42)

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TORCH_ALLOW_TF32"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

project_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
mm10_genome_dir = os.path.join(project_dir, "data/reference_genome/mm10")
mm10_gene_tss_file = os.path.join(project_dir, "data/genome_annotation/mm10/mm10_TSS.bed")
ground_truth_dir = os.path.join(project_dir, "ground_truth_files")
sample_input_dir = os.path.join(project_dir, "input/mESC/filtered_L2_E7.5_rep1")
output_dir = os.path.join(project_dir, "output/transformer_testing_output")

mm10_fasta_file = os.path.join(mm10_genome_dir, "chr1.fa")
mm10_chrom_sizes_file = os.path.join(mm10_genome_dir, "chrom.sizes")

CHECKPOINT_DIR = os.path.join(output_dir, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

window_size = 1000

print("\n ----- Loading and Formatting Input Data -----")
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
mesc_atac_peak_loc_df = utils.format_peaks(mesc_atac_peak_loc)
mesc_atac_peak_loc_df = mesc_atac_peak_loc_df[mesc_atac_peak_loc_df["chromosome"] == "chr1"]
mesc_atac_peak_loc_df = mesc_atac_peak_loc_df.rename(columns={"chromosome":"chrom"})

# TEMPORARY Restrict to Chr1 for testing
mesc_atac_data_chr1 = mesc_atac_data[mesc_atac_data.index.isin(mesc_atac_peak_loc_df.peak_id)]

print("Reading in the scRNA-seq dataset")
mesc_rna_data = pd.read_parquet(
    os.path.join(sample_input_dir, "mESC_filtered_L2_E7.5_rep1_RNA_processed.parquet")).set_index("gene_id")

genome_window_file = os.path.join(mm10_genome_dir, f"mm10_chr1_windows_{window_size // 1000}kb.bed")
if not os.path.exists(genome_window_file):
    print("Creating genomic windows")

    mm10_genome_windows = pybedtools.bedtool.BedTool().window_maker(g=mm10_chrom_sizes_file, w=window_size)
    mm10_chr1_windows = (
        mm10_genome_windows
        .filter(lambda x: x.chrom == "chr1")  # TEMPORARY Restrict to Chr1 for testing
        .saveas(genome_window_file)
        .to_dataframe()
    )
else:
    print("Loading existing genomic windows")
    mm10_chr1_windows = pybedtools.BedTool(genome_window_file).to_dataframe()



if "peak_tmp.bed" not in os.listdir(output_dir) or "tss_tmp.bed" not in os.listdir(output_dir):
    print("Calculating peak to TG distance score")
    peak_bed = pybedtools.BedTool.from_dataframe(
        mesc_atac_peak_loc[["chrom", "start", "end", "peak_id"]]
        ).saveas(os.path.join(output_dir, "peak_tmp.bed"))

    tss_bed = pybedtools.BedTool.from_dataframe(
        gene_tss_df[["chrom", "start", "end", "name"]]
        ).saveas(os.path.join(output_dir, "tss_tmp.bed"))
    
peak_bed = pybedtools.BedTool(os.path.join(output_dir, "peak_tmp.bed"))
tss_bed = pybedtools.BedTool(os.path.join(output_dir, "tss_tmp.bed"))

genes_near_peaks = utils.find_genes_near_peaks(peak_bed, tss_bed)

# Restrict to peaks within 1 Mb of a gene TSS
genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= 1e6]

# Scale the TSS distance score by the exponential scaling factor
genes_near_peaks = genes_near_peaks.copy()
genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / 250000)

genes_near_peaks.to_parquet(os.path.join(output_dir, "genes_near_peaks.parquet"), compression="snappy", engine="pyarrow")

genes_near_peaks = pd.read_parquet(os.path.join(output_dir, "genes_near_peaks.parquet"), engine="pyarrow")


# print("Building Homer peaks file")
# homer_peaks = genes_near_peaks[["peak_id", "peak_chr", "peak_start", "peak_end"]]
# homer_peaks = homer_peaks.rename(columns={
#     "peak_id":"PeakID", 
#     "peak_chr":"chr",
#     "peak_start":"start",
#     "peak_end":"end"
#     })
# homer_peaks["strand"] = ["."] * len(homer_peaks)
# homer_peaks["start"] = round(homer_peaks["start"].astype(int),0)
# homer_peaks["end"] = round(homer_peaks["end"].astype(int),0)
# homer_peaks = homer_peaks.drop_duplicates(subset="PeakID")

# os.makedirs(os.path.join(output_dir, "tmp"), exist_ok=True)
# homer_peak_path = os.path.join(output_dir, "tmp/homer_peaks.txt")
# homer_peaks.to_csv(homer_peak_path, sep="\t", header=False, index=False)

homer_results = pd.read_parquet(os.path.join(output_dir, "homer_tf_to_peak.parquet"), engine="pyarrow")
homer_results = homer_results.reset_index(drop=True)
homer_results["source_id"] = homer_results["source_id"].str.capitalize()

atac_cell_barcodes = mesc_atac_data_chr1.columns.to_list()
rna_cell_barcodes = mesc_rna_data.columns.to_list()
atac_in_rna_shared_barcodes = [i for i in atac_cell_barcodes if i in rna_cell_barcodes]

# Make sure that the cell names are in the same order and in both datasets
shared_barcodes = sorted(set(atac_in_rna_shared_barcodes))[:25]

mesc_atac_data_chr1_shared = mesc_atac_data_chr1[shared_barcodes]
mesc_rna_data_shared = mesc_rna_data[shared_barcodes]

print("Preparing sparse components (TF×Peak and Peak×Gene)")

# Universe/order
tfs   = homer_results["source_id"].astype(str).str.capitalize().unique()
genes = genes_near_peaks["target_id"].astype(str).unique()
# union of peaks that appear in either table
peaks = np.unique(np.concatenate([
    homer_results["peak_id"].astype(str).values,
    genes_near_peaks["peak_id"].astype(str).values
]))

tf_i   = {t:i for i,t in enumerate(tfs)}
peak_i = {p:i for i,p in enumerate(peaks)}
gene_i = {g:i for i,g in enumerate(genes)}

# H: TF x Peak (HOMER). Missing -> 0 (sparse).
Hr = homer_results[["source_id","peak_id","homer_binding_score"]].copy()
Hr["source_id"] = Hr["source_id"].astype(str).str.capitalize().map(tf_i)
Hr["peak_id"]   = Hr["peak_id"].astype(str).map(peak_i)
Hr = Hr.dropna(subset=["source_id","peak_id"])
H = sparse.coo_matrix(
    (Hr["homer_binding_score"].astype(np.float32).values,
     (Hr["source_id"].astype(int).values, Hr["peak_id"].astype(int).values)),
    shape=(len(tfs), len(peaks))
).tocsr()

# D: Peak x Gene (distance score). Missing -> 0 (sparse).
Dr = genes_near_peaks[["peak_id","target_id","TSS_dist_score"]].copy()
Dr["peak_id"]   = Dr["peak_id"].astype(str).map(peak_i)
Dr["target_id"] = Dr["target_id"].astype(str).map(gene_i)
Dr = Dr.dropna(subset=["peak_id","target_id"])
D = sparse.coo_matrix(
    (Dr["TSS_dist_score"].astype(np.float32).values,
     (Dr["peak_id"].astype(int).values, Dr["target_id"].astype(int).values)),
    shape=(len(peaks), len(genes))
).tocsr()

# ----- peak -> window assignment (max-overlap; random ties) -----
print("Assigning peaks to windows")

# Make a Series with peak start/end by peak_id
peak_coord_df = mesc_atac_peak_loc.loc[mesc_atac_peak_loc["peak_id"].isin(peaks), ["peak_id","chrom","start","end"]].copy()
peak_coord_df = peak_coord_df[peak_coord_df["chrom"] == "chr1"]  # keep chr1
coord_map = peak_coord_df.set_index("peak_id")[["start","end"]].to_dict(orient="index")

w = int((mm10_chr1_windows["end"] - mm10_chr1_windows["start"]).mode().iloc[0])
win_lut = {}  # window_idx -> window_id string
for _, row in mm10_chr1_windows.iterrows():
    k = row["start"] // w
    win_lut[k] = f'{row["chrom"]}:{row["start"]}-{row["end"]}'
nW = len(win_lut)

rng = np.random.default_rng(0)
def assign_best_window(start, end, w):
    i0 = start // w
    i1 = (end - 1) // w
    if i1 < i0: i1 = i0
    best_k = i0
    best_ov = -1
    ties = []
    for k in range(i0, i1 + 1):
        bs, be = k * w, (k + 1) * w
        ov = max(0, min(end, be) - max(start, bs))
        if ov > best_ov:
            best_ov, best_k = ov, k
            ties = [k]
        elif ov == best_ov:
            ties.append(k)
    return best_k if len(ties) == 1 else int(rng.choice(ties))

# map each peak to a window_idx (or -1 if we lack coords)
peak_to_window_idx = np.full(len(peaks), -1, dtype=np.int32)
for p, idx in peak_i.items():
    info = coord_map.get(p)
    if info is None: 
        continue
    peak_to_window_idx[idx] = assign_best_window(int(info["start"]), int(info["end"]), w)

# build list of peak indices per window
peaks_by_window = [np.where(peak_to_window_idx == k)[0] for k in range(nW)]
window_ids = np.array([win_lut[k] for k in range(nW)], dtype=object)

print("\nBuilding Transformer Model")
TF = len(tfs)               # number of TFs
TG = len(genes)             # number of target genes
Wn = len(peaks_by_window)   # number of windows

# ----- Configurations -----
d_model = 96
tf_channels = 12
tg_channels = 24 
batch_size = 1
epochs = 20
kernel_and_stride_size = 32
window_channels = tg_channels * tf_channels

# Encoder Settings
encoder_nhead = 8
encoder_dim_feedforward = 512
encoder_num_layers = 4

# Model Architecture overview
# proj_tg -> proj_tf -> proj_window -> encoder -> cross-attention -> readout

# Dimensionality reduction using linear projections
proj_tg = nn.Linear(TG, tg_channels, bias=False)
proj_tf = nn.Linear(TF, tf_channels, bias=False)
proj_window = nn.Linear(window_channels, d_model)

# Pool window data by averaging bins
pool = nn.AvgPool1d(
    kernel_size=kernel_and_stride_size, 
    stride=kernel_and_stride_size
    )

nonempty = sum(1 for p in peaks_by_window if p.size > 0)
print(f"Data Size:")
print(f"  - Windows: {len(peaks_by_window):,} | non-empty: {nonempty:,}")
print(f"  - Num TFs = {len(tfs):,}")
print(f"  - Num Peaks = {len(peaks):,}")
print(f"  - Num TGs = {len(genes):,}")
print(f"  - Num Cells = {len(shared_barcodes)}")

print(f"\nModel Parameters:")
print(f"  - Model Dimensions = {d_model}")
print(f"  - TF Linear Projection = {TF:,} -> {tf_channels}")
print(f"  - TG Linear Projection = {TG:,} -> {tg_channels}")
print(f"  - TF x TG Features Per Window = {window_channels} -> Model Dimension of {d_model}")
print(f"  - Window Data Pooling: kernel={kernel_and_stride_size}, stride={kernel_and_stride_size}")

print(f"\nEncoder Settings")
print(f"  - Encoder Layers - {encoder_num_layers}")
print(f"  - Number of Heads = {encoder_nhead}")
print(f"  - Feedforward Layer Neurons = {encoder_dim_feedforward}")

# Set up the Key, Query, Value self-attention layers
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model, 
    nhead=encoder_nhead, 
    dim_feedforward=512, 
    batch_first=True
    )
encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_num_layers)

# Set up the multi-headed cross-attention layers
n_genes = len(genes)  # 1425
gene_embed = nn.Embedding(n_genes, d_model)
cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
readout = nn.Sequential(
    nn.LayerNorm(d_model),
    nn.Linear(d_model, 1)
)

def build_tokens_sparse_for_cells(
    cell_ids,
    H, D,               # CSR: [TF×P], [P×G]
    peaks_by_window,    # list of np.array peak indices per window
    tfs, peaks, genes,  # ID arrays
    mesc_rna_data_shared, mesc_atac_data_chr1_shared,
    proj_tg: nn.Linear, proj_tf: nn.Linear, proj_window: nn.Linear,
    pool: nn.AvgPool1d,
    device,
    clamp_val: float = 5.0,
):
    """
    Returns [B, W', d_model] with W' ≈ ceil(n_nonempty_windows / pool_ks).
    Streaming: never builds a [W, d_model] tensor, no concatenations.
    """
    TF = H.shape[0]; G = D.shape[1]
    pool_ks = pool.kernel_size if isinstance(pool.kernel_size, int) else pool.kernel_size[0]
    out_batches = []

    for cell in cell_ids:
        # Per-cell vectors (safe reindex; CPU)
        e = mesc_rna_data_shared.reindex(index=tfs)[cell].fillna(0).to_numpy(np.float32)
        a = mesc_atac_data_chr1_shared.reindex(index=peaks)[cell].fillna(0).to_numpy(np.float32)

        # Scale sparse and keep CSR for slicing (CPU)
        B = H.multiply(e[:, None]).tocsr()   # [TF, P]
        R = D.multiply(a[:, None]).tocsr()   # [P,  G]

        # Rolling pool buffers on GPU (track grads)
        pooled_list = []                     # will hold a few [1,1,d_model]; tiny
        sum_tok = None                       # [1, d_model] on device
        cnt = 0

        for p_idx in peaks_by_window:
            if p_idx.size == 0:
                # skip empty windows entirely; if you need positional density, create a learned "empty" token here
                continue

            # Sparse @ sparse → dense once (CPU), then to GPU
            Mw = (B[:, p_idx] @ R[p_idx, :]).toarray()   # [TF, G], CPU
            M  = torch.from_numpy(Mw).to(device)

            # stabilize per TF across genes
            M = (M - M.mean(dim=1, keepdim=True)) / (M.std(dim=1, keepdim=True) + 1e-6)
            M = torch.clamp(M, -clamp_val, clamp_val)

            # projections on device, tracked by autograd
            Xg  = proj_tg(M.unsqueeze(0))   # [1, TF, tg]
            Xg  = Xg.permute(0, 2, 1)       # [1, tg, TF]
            Xg  = proj_tf(Xg)               # [1, tg, tf]
            feat = Xg.reshape(1, -1)        # [1, tg*tf]
            tok  = proj_window(feat)        # [1, d_model]

            # rolling sum
            if sum_tok is None:
                sum_tok = tok
            else:
                sum_tok = sum_tok + tok
            cnt += 1

            # when we hit pool_ks windows, average and append a single pooled token
            if cnt == pool_ks:
                pooled = (sum_tok / float(pool_ks)).unsqueeze(1)  # [1, 1, d_model]
                pooled_list.append(pooled)
                # reset
                sum_tok = None
                cnt = 0

            # free big intermediates quickly
            del M, Xg, feat, tok
            torch.cuda.empty_cache()

        # tail: if we have leftovers < pool_ks, average them as-is
        if cnt > 0 and sum_tok is not None:
            pooled = (sum_tok / float(cnt)).unsqueeze(1)         # [1, 1, d_model]
            pooled_list.append(pooled)
            sum_tok = None; cnt = 0

        if len(pooled_list) == 0:
            out_batches.append(torch.zeros((1, 1, proj_window.out_features), device=device))
        else:
            out_batches.append(torch.cat(pooled_list, dim=1))    # [1, W', d_model]

        # free sparse per-cell
        del B, R

    return torch.cat(out_batches, dim=0)  # [B, W', d_model]

def forward_tokens_to_pred(
    tokens_batch: torch.Tensor,      # [B, W', d_model]
    encoder,                         # nn.TransformerEncoder or your custom RPB blocks
    gene_embed: nn.Embedding,
    cross_attn: nn.MultiheadAttention,
    readout: nn.Module,
) -> torch.Tensor:
    H = encoder(tokens_batch)                            # [B, W', d_model]
    B = H.size(0)
    n_genes = gene_embed.num_embeddings
    gene_ids = torch.arange(n_genes, device=H.device)
    GQ = gene_embed(gene_ids).unsqueeze(0).expand(B, -1, -1)  # [B, n_genes, d_model]
    Z, _ = cross_attn(query=GQ, key=H, value=H)         # [B, n_genes, d_model]
    pred_expr = readout(Z).squeeze(-1)                  # [B, n_genes]
    return pred_expr

def masked_mse(pred, y, m):
    diff2 = (pred - y)**2
    diff2 = diff2[m]
    return diff2.mean() if diff2.numel() > 0 else torch.tensor(0.0, device=pred.device)

def get_true_tg_expr_vector_from_data(rna_data, genes):
    dup = pd.Index(genes).duplicated()
    assert not dup.any(), f"Duplicate gene IDs in prediction axis at: {np.where(dup)[0][:10]}"

    # Align counts to prediction order from the gene to index mapping (same length and order as genes)
    true_counts = rna_data.reindex(genes)

    # build mask for missing genes (not present in RNA)
    mask = ~true_counts.isna().to_numpy()

    # Handle missing genes using a masked loss 
    y_true_vec = true_counts.to_numpy(dtype=float)        # shape (n_genes,)

    y_true = torch.tensor(y_true_vec, dtype=torch.float32).unsqueeze(0)   # [1, n_genes]
    mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)            # [1, n_genes]
    
    return y_true, mask_t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# move modules to device
for m in [proj_tg, proj_tf, proj_window, encoder, gene_embed, cross_attn, readout]:
    m.to(device)

def iter_batches(items, bs):
    for i in range(0, len(items), bs):
        yield items[i:i+bs]

# Train/val split on barcodes you already computed
all_cells = shared_barcodes
rng = np.random.default_rng(0)
rng.shuffle(all_cells)
val_frac = 0.15
n_val = max(1, int(len(all_cells)*val_frac))
val_cells = all_cells[:n_val]
train_cells = all_cells[n_val:]

opt = torch.optim.AdamW(
    list(proj_tg.parameters()) +
    list(proj_tf.parameters()) +
    list(proj_window.parameters()) +
    list(encoder.parameters()) +
    list(gene_embed.parameters()) +
    list(cross_attn.parameters()) +
    list(readout.parameters()),
    lr=1e-3, weight_decay=1e-4
)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

best_val, patience, pat = float("inf"), 5, 0

use_amp = torch.cuda.is_available()
autocast_ctx = (lambda: torch.amp.autocast(device_type="cuda")) if use_amp else (lambda: nullcontext())
scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

print("\n ----- Starting Training -----")
for epoch in range(1, epochs+1):
    # ---- TRAIN ----
    for m in [encoder, gene_embed, cross_attn, readout]:
        m.train()
    train_loss, n_train = 0.0, 0
    print("  - Running Cell Batch:")
    cell_num = 1
    for cell_batch in iter_batches(train_cells, batch_size):
        print(f"    {cell_num}/{(len(train_cells) // batch_size)+1}")
        opt.zero_grad()

        with autocast_ctx():
            tokens_b = build_tokens_sparse_for_cells(
                cell_batch, H, D, peaks_by_window, tfs, peaks, genes,
                mesc_rna_data_shared, mesc_atac_data_chr1_shared,
                proj_tg, proj_tf, proj_window, pool, device
            )
            print(f"      - Tokens batch shape: {tuple(tokens_b.shape)}")  # [B, W', d_model]
            Henc = encoder(tokens_b)
            n_genes = len(genes)
            gene_ids = torch.arange(n_genes, device=device)
            GQ = gene_embed(gene_ids).unsqueeze(0).expand(tokens_b.size(0), -1, -1)
            Z, _ = cross_attn(query=GQ, key=Henc, value=Henc)
            pred = readout(Z).squeeze(-1)

            # targets/masks for this batch
            y_list, m_list = [], []
            for c in cell_batch:
                y, m = get_true_tg_expr_vector_from_data(mesc_rna_data_shared[c], genes)
                y_list.append(y); m_list.append(m)
            y_true = torch.cat(y_list, dim=0).to(device)
            mask_t = torch.cat(m_list, dim=0).to(device)

            # masked loss
            diff2 = (pred - y_true)**2
            loss = (diff2[mask_t]).mean() if (mask_t.sum() > 0) else torch.tensor(0.0, device=device)

        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                list(proj_tg.parameters()) + list(proj_tf.parameters()) +
                list(proj_window.parameters()) + list(encoder.parameters()) +
                list(gene_embed.parameters()) + list(cross_attn.parameters()) +
                list(readout.parameters()), 1.0
            )
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(proj_tg.parameters()) + list(proj_tf.parameters()) +
                list(proj_window.parameters()) + list(encoder.parameters()) +
                list(gene_embed.parameters()) + list(cross_attn.parameters()) +
                list(readout.parameters()), 1.0
            )
            opt.step()
        
        # accumulate training loss
        train_loss += loss.item() * len(cell_batch)
        n_train    += len(cell_batch)
        
        cell_num += 1
    
    train_loss /= max(1, n_train)
    sched.step()
    
    # ---- VAL ----
    for m in [encoder, gene_embed, cross_attn, readout]:
        m.eval()
    val_loss, n_valtot = 0.0, 0
    with torch.no_grad(), autocast_ctx():
        for cell_batch in iter_batches(val_cells, batch_size):
            tokens_b = build_tokens_sparse_for_cells(
                cell_batch, H, D, peaks_by_window, tfs, peaks, genes,
                mesc_rna_data_shared, mesc_atac_data_chr1_shared,
                proj_tg, proj_tf, proj_window, pool, device
            )
            Henc = encoder(tokens_b)
            n_genes = len(genes)
            gene_ids = torch.arange(n_genes, device=device)
            GQ = gene_embed(gene_ids).unsqueeze(0).expand(tokens_b.size(0), -1, -1)
            Z, _ = cross_attn(query=GQ, key=Henc, value=Henc)
            pred = readout(Z).squeeze(-1)

            y_list, m_list = [], []
            for c in cell_batch:
                y, m = get_true_tg_expr_vector_from_data(mesc_rna_data_shared[c], genes)
                y_list.append(y); m_list.append(m)
            y_true = torch.cat(y_list, dim=0).to(device)
            mask_t = torch.cat(m_list, dim=0).to(device)

            diff2 = (pred - y_true)**2
            loss = (diff2[mask_t]).mean() if (mask_t.sum() > 0) else torch.tensor(0.0, device=device)
            val_loss += loss.item() * len(cell_batch)
            n_valtot += len(cell_batch)

    val_loss /= max(1, n_valtot)
    print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # early stopping + checkpoint
    if val_loss < best_val - 1e-5:
        best_val, pat = val_loss, 0
        torch.save({
            "proj_tg": proj_tg.state_dict(),
            "proj_tf": proj_tf.state_dict(),
            "proj_window": proj_window.state_dict(),
            "encoder": encoder.state_dict(),
            "gene_embed": gene_embed.state_dict(),
            "cross_attn": cross_attn.state_dict(),
            "readout": readout.state_dict(),
            "tfs": np.array(tfs), "genes": np.array(genes), "window_ids": window_ids,
        }, os.path.join(output_dir, "expr_predictor_best.ckpt"))
    else:
        pat += 1
        if pat >= patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
            break