
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
import numpy as np
import pandas as pd
import pybedtools
from grn_inference import utils
from typing import Optional, Tuple, Any, Dict
from scipy import sparse
from contextlib import nullcontext
from collections import OrderedDict
import logging
import time
import multiprocessing as mp
import warnings
from sc_multi_transformer import MultiomicTransformer

# ------ PyTorch Configurations ------
warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group` or `barrier `")
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(1)
np.random.seed(42)

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TORCH_ALLOW_TF32"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

USE_AMP = False
AMP_DTYPE = torch.float16  # ignored
scaler = torch.amp.GradScaler(enabled=False)

world = int(os.environ.get("WORLD_SIZE", "1"))
per = max(1, mp.cpu_count() // world)

for var in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[var] = str(per)

torch.set_num_threads(per)
torch.set_num_interop_threads(1)

# ----- User Settings -----
load_model = False
window_size = 1000 # 1000
num_cells = 100
chrom_id = "chr1"

atac_data_filename = "mESC_filtered_L2_E7.5_rep1_ATAC_processed.parquet"
rna_data_filename = "mESC_filtered_L2_E7.5_rep1_RNA_processed.parquet"

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
MM10_GENOME_DIR = os.path.join(PROJECT_DIR, "data/reference_genome/mm10")
MM10_GENE_TSS_FILE = os.path.join(PROJECT_DIR, "data/genome_annotation/mm10/mm10_TSS.bed")
GROUND_TRUTH_DIR = os.path.join(PROJECT_DIR, "ground_truth_files")
SAMPLE_INPUT_DIR = os.path.join(PROJECT_DIR, "input/mESC/filtered_L2_E7.5_rep1")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output/transformer_testing_output")

MM10_FASTA_FILE = os.path.join(MM10_GENOME_DIR, "chr1.fa")
MM10_CHROM_SIZES_FILE = os.path.join(MM10_GENOME_DIR, "chrom.sizes")

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "training_stats"), exist_ok=True)

def init_ddp():
    # torchrun sets these
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)
    return rank, world_size, local_rank

def setup_logging():
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    handlers = []
    if rank == 0:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.ERROR,
                        handlers=handlers, format="%(message)s")

def _rank0() -> bool:
    return (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized()

def _atomic_save(obj: Dict[str, Any], path: str) -> None:
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_checkpoint(model, optimizer, scheduler, epoch: int, loss: float,
                    best_val: float, fname: str) -> str:
    """Always safe to call from any rank; only rank0 writes."""
    if not _rank0():
        if dist.is_initialized(): dist.barrier()
        return fname

    logging.debug(f"Saving checkpoint -> {fname}")
    # unwrap DDP
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "epoch": epoch,
        "loss": loss,
        "best_val": best_val,
        "state_dict": raw_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "torch_version": torch.__version__,
    }
    _atomic_save(state, fname)
    if dist.is_initialized(): dist.barrier()
    return fname

def save_regular(model, optimizer, scheduler, epoch: int, loss: float, best_val: float) -> str:
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch:04d}.pth.tar")
    return save_checkpoint(model, optimizer, scheduler, epoch, loss, best_val, path)

def unwrap(m):
    return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

def load_or_create_gene_tss_df():
    gene_tss_outfile = os.path.join(MM10_GENOME_DIR, "mm10_ch1_gene_tss.bed")
    if not os.path.isfile(gene_tss_outfile):
        mm10_gene_tss_bed = pybedtools.BedTool(MM10_GENE_TSS_FILE)
        
        gene_tss_df = (
            mm10_gene_tss_bed
            .filter(lambda x: x.chrom == "chr1")
            .saveas(gene_tss_outfile)
            .to_dataframe()
            .sort_values(by="start", ascending=True)
            )
    else:
        gene_tss_df = pybedtools.BedTool(gene_tss_outfile).to_dataframe().sort_values(by="start", ascending=True)
        
    return gene_tss_df

def load_atac_dataset(atac_data_filename):
    mesc_atac_data = pd.read_parquet(os.path.join(SAMPLE_INPUT_DIR, atac_data_filename)).set_index("peak_id")
    mesc_atac_peak_loc = mesc_atac_data.index

    # format the peaks to be in bed_format
    mesc_atac_peak_loc_df = utils.format_peaks(mesc_atac_peak_loc)
    mesc_atac_peak_loc_df = mesc_atac_peak_loc_df[mesc_atac_peak_loc_df["chromosome"] == "chr1"]
    mesc_atac_peak_loc_df = mesc_atac_peak_loc_df.rename(columns={"chromosome":"chrom"})

    # TEMPORARY Restrict to Chr1 for testing
    mesc_atac_data_chr1 = mesc_atac_data[mesc_atac_data.index.isin(mesc_atac_peak_loc_df.peak_id)]
    
    return mesc_atac_data_chr1, mesc_atac_peak_loc_df

def load_rna_data(rna_data_filename):
    logging.info("Reading in the scRNA-seq dataset")
    mesc_rna_data = pd.read_parquet(
        os.path.join(SAMPLE_INPUT_DIR, rna_data_filename)).set_index("gene_id")
    return mesc_rna_data

def create_or_load_genomic_windows(force_recalculate=False):
    genome_window_file = os.path.join(MM10_GENOME_DIR, f"mm10_chr1_windows_{window_size // 1000}kb.bed")
    if not os.path.exists(genome_window_file) or force_recalculate:
        
        logging.info("Creating genomic windows")
        mm10_genome_windows = pybedtools.bedtool.BedTool().window_maker(g=MM10_CHROM_SIZES_FILE, w=window_size)
        mm10_chr1_windows = (
            mm10_genome_windows
            .filter(lambda x: x.chrom == "chr1")  # TEMPORARY Restrict to Chr1 for testing
            .saveas(genome_window_file)
            .to_dataframe()
        )
    else:
        
        logging.info("Loading existing genomic windows")
        mm10_chr1_windows = pybedtools.BedTool(genome_window_file).to_dataframe()
        
    return mm10_chr1_windows

def calculate_peak_to_tg_distance_score(mesc_atac_peak_loc_df, gene_tss_df, force_recalculate=False):
    if not os.path.isfile(os.path.join(OUTPUT_DIR, "genes_near_peaks.parquet")) or force_recalculate:
        if "peak_tmp.bed" not in os.listdir(OUTPUT_DIR) or "tss_tmp.bed" not in os.listdir(OUTPUT_DIR):
        
            logging.info("Calculating peak to TG distance score")
            peak_bed = pybedtools.BedTool.from_dataframe(
                mesc_atac_peak_loc_df[["chrom", "start", "end", "peak_id"]]
                ).saveas(os.path.join(OUTPUT_DIR, "peak_tmp.bed"))

            tss_bed = pybedtools.BedTool.from_dataframe(
                gene_tss_df[["chrom", "start", "end", "name"]]
                ).saveas(os.path.join(OUTPUT_DIR, "tss_tmp.bed"))
            
        peak_bed = pybedtools.BedTool(os.path.join(OUTPUT_DIR, "peak_tmp.bed"))
        tss_bed = pybedtools.BedTool(os.path.join(OUTPUT_DIR, "tss_tmp.bed"))

        genes_near_peaks = utils.find_genes_near_peaks(peak_bed, tss_bed)

        # Restrict to peaks within 1 Mb of a gene TSS
        genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= 1e6]

        # Scale the TSS distance score by the exponential scaling factor
        genes_near_peaks = genes_near_peaks.copy()
        genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / 250000)

        genes_near_peaks.to_parquet(os.path.join(OUTPUT_DIR, "genes_near_peaks.parquet"), compression="snappy", engine="pyarrow")
    else:
        genes_near_peaks = pd.read_parquet(os.path.join(OUTPUT_DIR, "genes_near_peaks.parquet"), engine="pyarrow")
    
    return genes_near_peaks

def create_homer_peaks_file(genes_near_peaks):
    logging.info("Building Homer peaks file")
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

    os.makedirs(os.path.join(OUTPUT_DIR, "tmp"), exist_ok=True)
    homer_peak_path = os.path.join(OUTPUT_DIR, "tmp/homer_peaks.txt")
    homer_peaks.to_csv(homer_peak_path, sep="\t", header=False, index=False)

def load_homer_tf_to_peak_results():
    assert os.path.exists(os.path.join(OUTPUT_DIR, "homer_tf_to_peak.parquet")), \
        "ERROR: Homer TF to peak output parquet file required"
        
    homer_results = pd.read_parquet(os.path.join(OUTPUT_DIR, "homer_tf_to_peak.parquet"), engine="pyarrow")
    homer_results = homer_results.reset_index(drop=True)
    homer_results["source_id"] = homer_results["source_id"].str.capitalize()
    
    return homer_results

def find_shared_barcodes(atac_df, rna_df, num_cells):
    atac_cell_barcodes = atac_df.columns.to_list()
    rna_cell_barcodes = rna_df.columns.to_list()
    atac_in_rna_shared_barcodes = [i for i in atac_cell_barcodes if i in rna_cell_barcodes]

    # Make sure that the cell names are in the same order and in both datasets
    shared_barcodes = sorted(set(atac_in_rna_shared_barcodes))[:num_cells]

    atac_df_shared = atac_df[shared_barcodes]
    rna_df_shared = rna_df[shared_barcodes]
    
    return shared_barcodes, atac_df_shared, rna_df_shared

def get_unique_tfs_peaks_genes(homer_results, genes_near_peaks):  
    tfs   = homer_results["source_id"].astype(str).str.capitalize().unique()
    
    peaks = np.unique(np.concatenate([
        homer_results["peak_id"].astype(str).values,
        genes_near_peaks["peak_id"].astype(str).values
    ]))
    
    genes = genes_near_peaks["target_id"].astype(str).unique()
    
    return tfs, peaks, genes

def create_tf_peak_gene_mapping_dicts(tfs, peaks, genes):
    tf_i   = {t:i for i,t in enumerate(tfs)}
    peak_i = {p:i for i,p in enumerate(peaks)}
    gene_i = {g:i for i,g in enumerate(genes)}
    
    return tf_i, peak_i, gene_i

def cast_homer_tf_to_peak_df_sparse(homer_results, tf_i, peak_i):
    homer_results = homer_results[["source_id","peak_id","homer_binding_score"]].copy()
    homer_results["source_id"] = homer_results["source_id"].astype(str).str.capitalize().map(tf_i)
    homer_results["peak_id"]   = homer_results["peak_id"].astype(str).map(peak_i)
    homer_results = homer_results.dropna(subset=["source_id","peak_id"])
    homer_tf_peak_sparse = sparse.coo_matrix(
        (homer_results["homer_binding_score"].astype(np.float32).values,
        (homer_results["source_id"].astype(int).values, homer_results["peak_id"].astype(int).values)),
        shape=(len(tf_i), len(peak_i))
    ).tocsr()

    return homer_tf_peak_sparse

def cast_peak_to_tg_distance_sparse(genes_near_peaks, peak_i, gene_i):
    genes_near_peaks = genes_near_peaks[["peak_id","target_id","TSS_dist_score"]].copy()
    genes_near_peaks["peak_id"]   = genes_near_peaks["peak_id"].astype(str).map(peak_i)
    genes_near_peaks["target_id"] = genes_near_peaks["target_id"].astype(str).map(gene_i)
    genes_near_peaks = genes_near_peaks.dropna(subset=["peak_id","target_id"])
    gene_distance_sparse = sparse.coo_matrix(
        (genes_near_peaks["TSS_dist_score"].astype(np.float32).values,
        (genes_near_peaks["peak_id"].astype(int).values, genes_near_peaks["target_id"].astype(int).values)),
        shape=(len(peak_i), len(gene_i))
    ).tocsr()
    
    return gene_distance_sparse

def assign_peaks_to_windows(mesc_atac_peak_loc_df, peaks, peak_i, windows_df):
    logging.info("Assigning peaks to windows")

    # Make a Series with peak start/end by peak_id
    peak_coord_df = mesc_atac_peak_loc_df.loc[mesc_atac_peak_loc_df["peak_id"].isin(peaks), ["peak_id","chrom","start","end"]].copy()
    peak_coord_df = peak_coord_df[peak_coord_df["chrom"] == "chr1"]  # keep chr1
    coord_map = peak_coord_df.set_index("peak_id")[["start","end"]].to_dict(orient="index")

    w = int((windows_df["end"] - windows_df["start"]).mode().iloc[0])
    win_lut = {}  # window_idx -> window_id string
    for _, row in windows_df.iterrows():
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
    
    return peaks_by_window, window_ids

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

def iter_batches(items, bs):
    for i in range(0, len(items), bs):
        yield items[i:i+bs]
        
def shard_list_per_rank(items, rank, world_size, pad=True):
    # simple equal split with optional padding (repeat last) so lengths match
    n = len(items)
    per = (n + world_size - 1) // world_size  # ceil
    start = rank * per
    end = min(start + per, n)
    shard = items[start:end]
    if pad and len(shard) < per and len(shard) > 0:
        shard = shard + [shard[-1]] * (per - len(shard))
    return shard

def build_train_gene_stats(rna_data_shared_barcodes, train_cells, genes, device):
    n_genes = len(genes)
    sum_y   = torch.zeros(n_genes, dtype=torch.float32)
    sum_y2  = torch.zeros(n_genes, dtype=torch.float32)
    count   = torch.zeros(n_genes, dtype=torch.int32)

    for c in train_cells:                     # use *global* train_cells, not sharded
        y, m = get_true_tg_expr_vector_from_data(rna_data_shared_barcodes[c], genes)
        y = y.squeeze(0).to(torch.float32)    # [n_genes]
        m = m.squeeze(0)                      # [n_genes], bool
        sum_y  += torch.where(m, y, 0).cpu()
        sum_y2 += torch.where(m, y*y, 0).cpu()
        count  += m.to(torch.int32).cpu()

    count_f = count.to(torch.float32)
    mu = sum_y / torch.clamp(count_f, min=1.0)           # [n_genes]
    var = sum_y2 / torch.clamp(count_f, min=1.0) - mu*mu
    var = torch.clamp(var, min=0.0)
    sd  = torch.sqrt(var)

    # For genes never observed in training, make them neutral so they don't explode the loss
    never_seen = (count == 0)
    mu[never_seen] = 0.0
    sd[never_seen] = 1.0

    seen = (count > 0).sum().item()

    # Keep ggenes that are seen in the training dataset
    seen_genes_mask = (count.to(torch.int32) > 0).to(device)  # [n_genes], True for 532 seen genes
    mu = mu.to(device); sd = sd.to(device)
    sd = torch.clamp(sd, min=1e-6)

    seen = seen_genes_mask.to(device)  # genes seen in TRAIN only

    # Pack to a single tensor for convenient broadcast
    stats = torch.stack([mu, sd], dim=0)      # [2, n_genes]

    # Move to device and broadcast
    stats = stats.to(device)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(stats, src=0)

    mu, sd = stats[0], stats[1]                   # both [n_genes], on device

    # after you compute mu, sd from TRAIN only
    mu = mu.to(device)
    sd = torch.clamp(sd.to(device), min=1e-2)
    
    return mu, sd, seen_genes_mask


def main():
    multi_gpu = torch.cuda.device_count() > 1 and os.environ.get("WORLD_SIZE", "1") != "1"
    if multi_gpu:
        rank, world, local = init_ddp()
    else:
        rank, world, local = 0, 1, 0

    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    
    setup_logging()

    # ----- Input Data Setup -----
    if rank == 0:
        logging.info("Reading processed scATAC-seq dataset")
    mesc_atac_data_chr1, mesc_atac_peak_loc_df = load_atac_dataset(atac_data_filename)
    mesc_rna_data = load_rna_data(rna_data_filename)

    gene_tss_df = load_or_create_gene_tss_df()
    if rank == 0:
        logging.info(f"Using {gene_tss_df['name'].nunique()} genes (TSS df)")
    
    # Restrict gene TSS dataframe to only use the selected chromosome
    gene_tss_df = gene_tss_df[gene_tss_df["chrom"] == chrom_id]
    
    mesc_rna_data = mesc_rna_data[mesc_rna_data.index.isin(gene_tss_df["name"])]
    if rank == 0:
        logging.info(f"Using {mesc_rna_data.index.nunique()} genes (scRNA-seq data df)")
    
    mm10_chr1_windows = create_or_load_genomic_windows(force_recalculate=False)

    genes_near_peaks = calculate_peak_to_tg_distance_score(mesc_atac_peak_loc_df, gene_tss_df)
    
    # Restrict the genes near peaks dataframe to only using TGs from genes on chr1
    genes_near_peaks = genes_near_peaks[genes_near_peaks["gene_chr"] == chrom_id]
    if rank == 0:
        logging.info(f"Using {genes_near_peaks['target_id'].nunique()} genes (genes_near_peaks_df)")

    if not os.path.isfile(os.path.join(OUTPUT_DIR, "tmp/homer_peaks.txt")):
        create_homer_peaks_file(genes_near_peaks)
        
    homer_results = load_homer_tf_to_peak_results()

    shared_barcodes, mesc_atac_data_chr1_shared, mesc_rna_data_shared = find_shared_barcodes(mesc_atac_data_chr1, mesc_rna_data, num_cells)

    tfs, peaks, genes = get_unique_tfs_peaks_genes(homer_results, genes_near_peaks)
    tf_i, peak_i, gene_i = create_tf_peak_gene_mapping_dicts(tfs, peaks, genes)
    
    if rank == 0:
        logging.info("Preparing sparse components (TF×Peak and Peak×Gene)")
    homer_tf_peak_sparse = cast_homer_tf_to_peak_df_sparse(homer_results, tf_i, peak_i)
    gene_distance_sparse = cast_peak_to_tg_distance_sparse(genes_near_peaks, peak_i, gene_i)
    peaks_by_window, window_ids = assign_peaks_to_windows(mesc_atac_peak_loc_df, peaks, peak_i, mm10_chr1_windows)

    if rank == 0:
        logging.info("\nBuilding Transformer Model")
    TF = len(tfs)               # number of TFs
    TG = len(genes)             # number of target genes
    Wn = len(peaks_by_window)   # number of windows

    # ----- Configurations -----
    d_model = 256
    tf_channels = 32
    tg_channels = 64 
    batch_size = 1
    epochs = 20
    kernel_and_stride_size = 32
    window_channels = tg_channels * tf_channels

    # Encoder Settings
    encoder_nhead = 8
    encoder_dim_feedforward = 1024
    encoder_num_layers = 3
    dropout = 0.1
    
    # Train-test split
    validation_fraction = 0.15

    assert d_model % encoder_nhead == 0, \
        "Dimension of the model must be divisible by the number of heads"

    nonempty = sum(1 for p in peaks_by_window if p.size > 0)

    if rank == 0:
        logging.info(f"Data Size:")
        logging.info(f"  - Window Size: {window_size} bp")
        logging.info(f"  - Windows: {len(peaks_by_window):,} | non-empty: {nonempty:,}")
        logging.info(f"  - Num TFs = {len(tfs):,}")
        logging.info(f"  - Num Peaks = {len(peaks):,}")
        logging.info(f"  - Num TGs = {len(genes):,}")
        logging.info(f"  - Num Cells = {len(shared_barcodes)}")

        logging.info(f"\nModel Parameters:")
        logging.info(f"  - Model Dimensions = {d_model}")
        logging.info(f"  - TF Linear Projection = {TF:,} -> {tf_channels}")
        logging.info(f"  - TG Linear Projection = {TG:,} -> {tg_channels}")
        logging.info(f"  - TF x TG Features Per Window = {window_channels} -> Model Dimension of {d_model}")
        logging.info(f"  - Window Data Pooling: kernel={kernel_and_stride_size}, stride={kernel_and_stride_size}")

        logging.info(f"\nEncoder Settings")
        logging.info(f"  - Encoder Layers = {encoder_num_layers}")
        logging.info(f"  - Number of Heads = {encoder_nhead}")
        logging.info(f"  - Feedforward Layer Neurons = {encoder_dim_feedforward}")

    model = MultiomicTransformer(
        n_genes = len(genes),
        tg_in_dim=TG,
        tf_in_dim=TF,
        window_in_dim=Wn,
        d_model=d_model,
        nhead=encoder_nhead,
        dff=encoder_dim_feedforward,
        dropout=dropout,
        n_layers=encoder_num_layers,
        kernel_stride_size=kernel_and_stride_size,
    ).to(device)
    
    if multi_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local], output_device=local, find_unused_parameters=False
        )

    gene_ids = torch.arange(len(genes), device=device)

    rna_TF  = mesc_rna_data_shared.reindex(index=tfs).fillna(0).astype("float32")          # rows: TFs, cols: cells
    atac_P  = mesc_atac_data_chr1_shared.reindex(index=peaks).fillna(0).astype("float32")   # rows: peaks, cols: cells

    assert set(rna_TF.columns) == set(atac_P.columns), "RNA/ATAC barcode sets differ"
    rna_arr  = rna_TF.values          # shape [TF, n_cells]
    atac_arr = atac_P.values          # shape [P,  n_cells]
    col_of   = {c:i for i,c in enumerate(rna_TF.columns)}

    # Train/val split on barcodes you already computed
    all_cells = shared_barcodes
    rng = np.random.default_rng(0)
    rng.shuffle(all_cells)
    n_val = max(1, int(len(all_cells)*validation_fraction))
    val_cells = all_cells[:n_val]
    train_cells = all_cells[n_val:]

    target = model.module if multi_gpu else model
    target.attach_sparse_sources(
        H=homer_tf_peak_sparse, D=gene_distance_sparse,
        peaks_by_window=peaks_by_window, col_of=col_of,
        rna_arr=rna_arr, atac_arr=atac_arr,
    )

    best_val, patience, pat = float("inf"), 10, 0

    best_val = float("inf")
    start_epoch = 1

    mu, sd, seen_genes_mask = build_train_gene_stats(mesc_rna_data_shared, train_cells, genes, device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    use_amp = torch.cuda.is_available()
    autocast_ctx = (lambda: torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_amp else (lambda: nullcontext())
    scaler = torch.amp.GradScaler(enabled=use_amp)

    loss_by_epoch: dict =  {"epoch": [], "train_loss": [], "val_loss": [], "epoch_sec": []}
    
    train_cells_rank = shard_list_per_rank(train_cells, rank, world, pad=True)
    val_cells_rank   = shard_list_per_rank(val_cells,   rank, world, pad=True)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        inner = unwrap(model)
        inner.train()
        train_loss_sum, n_train = 0.0, 0
        if rank == 0:
            logging.info("  - Running Training")
        for bi, cell_batch in enumerate(iter_batches(train_cells_rank, batch_size), start=1):
            if rank == 0 and bi % 5 == 0:
                logging.info(f"       {bi}/{(len(train_cells_rank) + batch_size - 1)//batch_size}")
            opt.zero_grad(set_to_none=True)

            tokens_b, key_mask = inner.build_tokens_streaming_for_cells(
                cell_batch, clamp_val=3.0, pad_to_max_in_batch=True
            )

            # ---- forward + loss (bf16 autocast; stable, no GradScaler needed) ----
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                pred, Z = inner.forward(
                    windows=tokens_b, 
                    gene_ids=gene_ids, 
                    key_padding_mask=key_mask,
                    attn_mask=None,
                    already_pooled=True
                    )

                # targets
                y_list, m_list = [], []
                for c in cell_batch:
                    y, m = get_true_tg_expr_vector_from_data(mesc_rna_data_shared[c], genes)
                    y_list.append(y); m_list.append(m)

                y_true = torch.cat(y_list, dim=0).to(pred.device)     # fp32
                mask_t = torch.cat(m_list, dim=0).to(pred.device)      # bool

                y_norm = ((y_true - mu) / sd)
                y_norm = torch.clamp(y_norm, -8.0, 8.0)
                y_norm = torch.nan_to_num(y_norm, 0.0).to(pred.dtype)  # *** critical: match pred dtype ***

                mask_eff = mask_t & seen_genes_mask.unsqueeze(0) & torch.isfinite(y_norm)
                pred     = torch.nan_to_num(pred, 0.0)

                if not mask_eff.any():
                    continue

                loss = F.huber_loss(pred[mask_eff], y_norm.to(pred.dtype)[mask_eff], delta=1.0)


            # ---- backward/step (bf16 -> no GradScaler) ----
            loss.backward()
                
            torch.nn.utils.clip_grad_norm_(inner.parameters(), 1.0)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)
            
            opt.step()

            train_loss_sum += float(loss.item()) * len(cell_batch)
            n_train        += len(cell_batch)

        # --- VAL ---
        model.eval()
        inner = unwrap(model)
        val_loss_sum, n_valtot = 0.0, 0
        with torch.no_grad():
            if rank == 0:
                logging.info("  - Running Validation:")

            for bi, cell_batch in enumerate(iter_batches(val_cells_rank, batch_size), start=1):
                if rank == 0 and bi % 5 == 0:
                    logging.info(f"       {bi}/{(len(val_cells_rank) + batch_size - 1)//batch_size}")

                tokens_b, key_mask = inner.build_tokens_streaming_for_cells(
                    cell_batch, clamp_val=3.0, pad_to_max_in_batch=True
                )

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    pred, Z = inner.forward(
                        windows=tokens_b, 
                        gene_ids=gene_ids, 
                        key_padding_mask=key_mask,
                        attn_mask=None,
                        already_pooled=True
                        )

                    y_list, m_list = [], []
                    for c in cell_batch:
                        y, m = get_true_tg_expr_vector_from_data(mesc_rna_data_shared[c], genes)
                        y_list.append(y); m_list.append(m)

                    y_true = torch.cat(y_list, dim=0).to(pred.device)
                    mask_t = torch.cat(m_list, dim=0).to(pred.device)

                    y_norm = ((y_true - mu) / sd)
                    y_norm = torch.clamp(y_norm, -8.0, 8.0)
                    y_norm = torch.nan_to_num(y_norm, 0.0).to(pred.dtype)

                    mask_eff = mask_t & seen_genes_mask.unsqueeze(0) & torch.isfinite(y_norm)
                    pred     = torch.nan_to_num(pred, 0.0)

                    if not mask_eff.any():
                        continue

                    loss = F.huber_loss(pred[mask_eff], y_norm.to(pred.dtype)[mask_eff], delta=1.0)

                val_loss_sum += float(loss.item()) * len(cell_batch)
                n_valtot     += len(cell_batch)

        torch.cuda.synchronize(device)
        
        # --- epoch metrics ---
        sum_t = torch.tensor([train_loss_sum], device=device)
        cnt_t = torch.tensor([n_train],        device=device)
        sum_v = torch.tensor([val_loss_sum],   device=device)
        cnt_v = torch.tensor([n_valtot],       device=device)

        if multi_gpu:
            dist.all_reduce(sum_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(sum_v, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_v, op=dist.ReduceOp.SUM)

        train_loss_avg = (sum_t / cnt_t.clamp(min=1)).item()
        val_loss_avg   = (sum_v / cnt_v.clamp(min=1)).item()

        if rank == 0:
            print(f"[Epoch {epoch:02d}] train_loss={train_loss_avg:.4f}  val_loss={val_loss_avg:.4f}")

        if dist.is_available() and dist.is_initialized():
            dist.barrier() 
        
        # Calculate early stopping (best val hasnt improved in several epochs)
        is_best = val_loss_avg < best_val - 1e-5
        if is_best:
            best_val = val_loss_avg
            pat = 0
        else:
            pat += 1
            logging.info(f"No improvement ({pat}/{patience}) — val {val_loss_avg:.4f} ≥ best {best_val:.4f}")
        stop = pat >= patience
        
        # Save a checkpoint every 5 epochs
        if rank == 0 and epoch % 5 == 0:
            save_regular(model, opt, sched, epoch, loss=train_loss_avg, best_val=best_val)
        
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        if stop:
            if rank == 0:
                logging.info(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
                training_stats = pd.DataFrame(loss_by_epoch).set_index("epoch")
                training_stats.to_csv(os.path.join(OUTPUT_DIR, f"training_stats/training_loss_window_{window_size // 1000}kb_{num_cells}_cells.csv"), header=True, index=True)

                break

        if rank == 0:
            time_sec = time.time() - epoch_start_time
            loss_by_epoch["epoch"].append(epoch)
            loss_by_epoch["train_loss"].append(train_loss_avg)
            loss_by_epoch["val_loss"].append(val_loss_avg)      
            loss_by_epoch["epoch_sec"].append(time_sec)  

        sched.step()
            
    logging.info("\nTRAINING COMPLETE, ENDING PROCESS")
    training_stats = pd.DataFrame(loss_by_epoch).set_index("epoch")
    training_stats.to_csv(os.path.join(OUTPUT_DIR, f"training_stats/training_loss_window_{window_size // 1000}kb_{num_cells}_cells.csv"), header=True, index=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()