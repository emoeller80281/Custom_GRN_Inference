
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

warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group` or `barrier `")
torch.autograd.set_detect_anomaly(True)

torch.manual_seed(1)
np.random.seed(42)

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["TORCH_ALLOW_TF32"] = "1"
os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

world = int(os.environ.get("WORLD_SIZE", "1"))
per = max(1, mp.cpu_count() // world)

for var in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[var] = str(per)

torch.set_num_threads(per)
torch.set_num_interop_threads(1)

project_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
mm10_genome_dir = os.path.join(project_dir, "data/reference_genome/mm10")
mm10_gene_tss_file = os.path.join(project_dir, "data/genome_annotation/mm10/mm10_TSS.bed")
ground_truth_dir = os.path.join(project_dir, "ground_truth_files")
sample_input_dir = os.path.join(project_dir, "input/mESC/filtered_L2_E7.5_rep1")
output_dir = os.path.join(project_dir, "output/transformer_testing_output")

mm10_fasta_file = os.path.join(mm10_genome_dir, "chr1.fa")
mm10_chrom_sizes_file = os.path.join(mm10_genome_dir, "chrom.sizes")

load_model = False

CHECKPOINT_DIR = os.path.join(output_dir, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.join(output_dir, "training_stats"), exist_ok=True)

window_size = 1000 # 1000

num_cells = 300

CHECKPOINT_DIR = os.path.join(output_dir, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

def save_best(model, optimizer, scheduler, epoch: int, loss: float, best_val: float) -> str:
    path = os.path.join(CHECKPOINT_DIR, "best_model.pth.tar")
    return save_checkpoint(model, optimizer, scheduler, epoch, loss, best_val, path)

def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return OrderedDict((k.replace("module.", "", 1), v) for k, v in state_dict.items())
    return state_dict

def load_checkpoint_into(model, optimizer=None, scheduler=None, path: str = "") -> Tuple[int, float, float]:
    """Returns (epoch, last_loss, best_val). Loads onto CPU; move model to device after."""
    assert os.path.isfile(path), f"Checkpoint not found: {path}"
    logging.info(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    sd = strip_module_prefix(ckpt["state_dict"])
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(sd)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    # (Optional) restore RNG for exact reproducibility
    if "rng_state" in ckpt: torch.set_rng_state(ckpt["rng_state"])
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    if "numpy_rng_state" in ckpt: np.random.set_state(ckpt["numpy_rng_state"])

    epoch = int(ckpt.get("epoch", 0))
    loss  = float(ckpt.get("loss", float("inf")))
    best  = float(ckpt.get("best_val", float("inf")))
    return epoch, loss, best

def get_latest_checkpoint() -> Optional[str]:
    files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth.tar")), key=os.path.getmtime)
    return files[-1] if files else None

def load_best_checkpoint(model, optimizer=None, scheduler=None) -> Tuple[int, float]:
    best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth.tar")
    if os.path.isfile(best_path):
        epoch, last_loss, best_val = load_checkpoint_into(model, optimizer, scheduler, best_path)
        return epoch, best_val
    logging.info("No best_model.pth.tar found.")
    return 0, float("inf")

def clean_old_checkpoints(keep_last: int = 3) -> None:
    if not _rank0(): 
        if dist.is_initialized(): dist.barrier()
        return
    pattern = os.path.join(CHECKPOINT_DIR, "checkpoint_epoch_*.pth.tar")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    for f in files[:-keep_last]:
        try:
            logging.debug(f"Removing old checkpoint: {f}")
            os.remove(f)
        except OSError:
            pass
    if dist.is_initialized(): dist.barrier()

def setup_logging():
    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    handlers = []
    if rank == 0:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.ERROR,
                        handlers=handlers, format="%(message)s")

def init_distributed():
    """
    Initialize torch.distributed from torchrun or SLURM env.
    Returns (rank, world_size, local_rank).
    """
    if all(k in os.environ for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]):
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:                                                               # srun --mpi=pmi2
        rank       = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    
    print(f"[rank {rank}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[rank {rank}] torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"[rank {rank}] Setting CUDA device to local_rank = {local_rank}")
    
    return rank, world_size, local_rank

ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1 or int(os.environ.get("SLURM_NTASKS", "1")) > 1

# Initialize the distributed computing to utilize training across multiple GPUs
if ddp:
    rank, world_size, local_rank = init_distributed()
else:
    rank, world_size, local_rank = 0, 1, 0
    
setup_logging()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if rank==0:
    logging.info("\n ----- Loading and Formatting Input Data -----")
    logging.info("Reading and formatting TSS bed file")
    
mm10_gene_tss_bed = pybedtools.BedTool(mm10_gene_tss_file)
gene_tss_df = (
    mm10_gene_tss_bed
    .filter(lambda x: x.chrom == "chr1")
    .saveas(os.path.join(mm10_genome_dir, "mm10_ch1_gene_tss.bed"))
    .to_dataframe()
    .sort_values(by="start", ascending=True)
    )
if rank==0:
    logging.info("Reading processed scATAC-seq dataset")
mesc_atac_data = pd.read_parquet(os.path.join(sample_input_dir, "mESC_filtered_L2_E7.5_rep1_ATAC_processed.parquet")).set_index("peak_id")
mesc_atac_peak_loc = mesc_atac_data.index

# format the peaks to be in bed_format
mesc_atac_peak_loc_df = utils.format_peaks(mesc_atac_peak_loc)
mesc_atac_peak_loc_df = mesc_atac_peak_loc_df[mesc_atac_peak_loc_df["chromosome"] == "chr1"]
mesc_atac_peak_loc_df = mesc_atac_peak_loc_df.rename(columns={"chromosome":"chrom"})

# TEMPORARY Restrict to Chr1 for testing
mesc_atac_data_chr1 = mesc_atac_data[mesc_atac_data.index.isin(mesc_atac_peak_loc_df.peak_id)]

if rank==0:
    logging.info("Reading in the scRNA-seq dataset")
mesc_rna_data = pd.read_parquet(
    os.path.join(sample_input_dir, "mESC_filtered_L2_E7.5_rep1_RNA_processed.parquet")).set_index("gene_id")

genome_window_file = os.path.join(mm10_genome_dir, f"mm10_chr1_windows_{window_size // 1000}kb.bed")
if not os.path.exists(genome_window_file):
    if rank==0:
        logging.info("Creating genomic windows")

    mm10_genome_windows = pybedtools.bedtool.BedTool().window_maker(g=mm10_chrom_sizes_file, w=window_size)
    mm10_chr1_windows = (
        mm10_genome_windows
        .filter(lambda x: x.chrom == "chr1")  # TEMPORARY Restrict to Chr1 for testing
        .saveas(genome_window_file)
        .to_dataframe()
    )
else:
    if rank==0:
        logging.info("Loading existing genomic windows")
    mm10_chr1_windows = pybedtools.BedTool(genome_window_file).to_dataframe()



if "peak_tmp.bed" not in os.listdir(output_dir) or "tss_tmp.bed" not in os.listdir(output_dir):
    if rank==0:
        logging.info("Calculating peak to TG distance score")
    peak_bed = pybedtools.BedTool.from_dataframe(
        mesc_atac_peak_loc_df[["chrom", "start", "end", "peak_id"]]
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


# if rank==0:
#    logging.info("Building Homer peaks file")
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
shared_barcodes = sorted(set(atac_in_rna_shared_barcodes))[:num_cells]

mesc_atac_data_chr1_shared = mesc_atac_data_chr1[shared_barcodes]
mesc_rna_data_shared = mesc_rna_data[shared_barcodes]

if rank==0:
    logging.info("Preparing sparse components (TF×Peak and Peak×Gene)")

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
if rank==0:
    logging.info("Assigning peaks to windows")

# Make a Series with peak start/end by peak_id
peak_coord_df = mesc_atac_peak_loc_df.loc[mesc_atac_peak_loc_df["peak_id"].isin(peaks), ["peak_id","chrom","start","end"]].copy()
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

if rank==0:
    logging.info("\nBuilding Transformer Model")
TF = len(tfs)               # number of TFs
TG = len(genes)             # number of target genes
Wn = len(peaks_by_window)   # number of windows

class CheckpointedEncoder(nn.Module):
    """
    Wrap a stack of encoder layers, checkpointing each layer during training.
    Works with batch_first=True (input [B, S, D]).
    """
    def __init__(self, layers, norm=None, use_checkpoint=True, use_reentrant=False):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # pass the flag explicitly and avoid the lambda
                x = checkpoint(layer, x, use_reentrant=self.use_reentrant)
            else:
                x = layer(x)
        return self.norm(x) if self.norm is not None else x

# ----- Configurations -----
d_model = 192
tf_channels = 32
tg_channels = 64 
batch_size = 1
epochs = 20
kernel_and_stride_size = 32
window_channels = tg_channels * tf_channels

# Encoder Settings
encoder_nhead = 6
encoder_dim_feedforward = 1024
encoder_num_layers = 3

assert d_model % encoder_nhead == 0, \
    "Dimension of the model must be divisible by the number of heads"

nonempty = sum(1 for p in peaks_by_window if p.size > 0)
if rank==0:
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

# Create individual layers
encoder_layers = [
    nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=encoder_nhead,
        dim_feedforward=encoder_dim_feedforward,
        batch_first=True,    # important: your tokens are [B, W', D]
        norm_first=True,
        dropout = 0.1
    )
    for _ in range(encoder_num_layers)
]

encoder_norm = nn.LayerNorm(d_model)

# Wrap with checkpointing
encoder = CheckpointedEncoder(encoder_layers, norm=encoder_norm, use_checkpoint=True, use_reentrant=False).to(device)

# Dimensionality reduction using linear projections
proj_tg = nn.Linear(TG, tg_channels, bias=False)
proj_tf = nn.Linear(TF, tf_channels, bias=False)
proj_window = nn.Sequential(nn.Linear(window_channels, d_model), nn.Dropout(0.1))

# Pool window data by averaging bins
pool = nn.AvgPool1d(
    kernel_size=kernel_and_stride_size, 
    stride=kernel_and_stride_size
    )

# Set up the multi-headed cross-attention layers
n_genes = len(genes)  # 1425
gene_embed = nn.Embedding(n_genes, d_model)
cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=encoder_nhead, batch_first=True)
readout = nn.Sequential(
    nn.LayerNorm(d_model),
    nn.Linear(d_model, 1)
)

def build_tokens_sparse_for_cells(
    cell_ids,
    H, D,               # CSR: [TF×P], [P×G]
    peaks_by_window,    # list of np.array peak indices per window
    col_of,
    rna_arr, atac_arr,
    proj_tg: nn.Linear, proj_tf: nn.Linear, proj_window: nn.Linear,
    pool: nn.AvgPool1d,
    device,
    clamp_val: float = 3.0,
):
    """
    Returns [B, W', d_model] with W' ≈ ceil(n_nonempty_windows / pool_ks).
    Streaming: never builds a [W, d_model] tensor, no concatenations.
    """
    pool_ks = pool.kernel_size if isinstance(pool.kernel_size, int) else pool.kernel_size[0]
    out_batches = []

    for cell in cell_ids:
        col = col_of[cell]
        
        # Per-cell vectors (safe reindex; CPU)
        e = rna_arr[:, col]
        a = atac_arr[:, col]

        # Scale sparse and keep CSR for slicing (CPU)
        B = H.multiply(e[:, None]).tocsr()   # [TF, P]
        R = D.multiply(a[:, None]).tocsr()   # [P,  G]

        # Rolling pool buffers on GPU (track grads)
        pooled_list = []                     # will hold a few [1,1,d_model]; tiny
        sum_tok = None                       # [1, d_model] on device
        cnt = 0

        for p_idx in peaks_by_window:
            p_active = p_idx[a[p_idx] > 0]
            if p_active.size == 0:
                continue  # no work, no token

            Mw = (B[:, p_active] @ R[p_active, :]).toarray()
            M  = torch.as_tensor(Mw, device=device, dtype=torch.float32)
            
            if not torch.isfinite(M).all():
                logging.error("Non-finite in M before standardization")
                M = torch.nan_to_num(M, 0.0)

            # stabilize per TF across genes
            M = (M - M.mean(dim=1, keepdim=True)) / (M.std(dim=1, keepdim=True) + 1e-6)
            M = torch.clamp(M, -clamp_val, clamp_val)
            
            M = torch.nan_to_num(M, 0.0)

            # projections on device, tracked by autograd
            # Project TF first, then G (genes)
            with torch.amp.autocast(enabled=False, device_type="cuda"):
                Xtf  = proj_tf(M.t().unsqueeze(0))        # [1, G, tf]  (fp32)
                Xtg  = proj_tg(Xtf.transpose(1, 2))       # [1, tf, tg]
                feat = Xtg.reshape(1, -1)
                tok  = proj_window(feat)                  # [1, d_model]

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
            del M, Xtg, feat, tok

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

train_cells_rank = shard_list_per_rank(train_cells, rank, world_size, pad=True)
val_cells_rank   = shard_list_per_rank(val_cells,   rank, world_size, pad=True)

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

best_val, patience, pat = float("inf"), 5, 0

class ExprPredictor(nn.Module):
    def __init__(self, proj_tg, proj_tf, proj_window, encoder, gene_embed, cross_attn, readout):
        super().__init__()
        self.proj_tg = proj_tg
        self.proj_tf = proj_tf
        self.proj_window = proj_window
        self.encoder = encoder
        self.gene_embed = gene_embed
        self.cross_attn = cross_attn
        self.readout = readout

model = ExprPredictor(proj_tg, proj_tf, proj_window, encoder, gene_embed, cross_attn, readout).to(device)
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
        find_unused_parameters=False
    )
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
use_amp = torch.cuda.is_available()
autocast_ctx = (lambda: torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_amp else (lambda: nullcontext())
scaler = torch.amp.GradScaler(enabled=use_amp)

def unwrap(m):
    return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

best_val = float("inf")
start_epoch = 1

# (optional) resume:
if load_model:
    latest = get_latest_checkpoint()
    if latest:
        e, last_loss, best_val = load_checkpoint_into(
            model=model,  # or pass a simple object with your modules
            optimizer=opt, scheduler=sched, path=latest
        )
        start_epoch = e + 1
    else:
        start_epoch = 1
else:
    start_epoch = 1

gene_ids = torch.arange(n_genes, device=device)

rna_TF  = mesc_rna_data_shared.reindex(index=tfs).fillna(0).astype("float32")          # rows: TFs, cols: cells
atac_P  = mesc_atac_data_chr1_shared.reindex(index=peaks).fillna(0).astype("float32")   # rows: peaks, cols: cells

assert set(rna_TF.columns) == set(atac_P.columns), "RNA/ATAC barcode sets differ"
rna_arr  = rna_TF.values          # shape [TF, n_cells]
atac_arr = atac_P.values          # shape [P,  n_cells]
col_of   = {c:i for i,c in enumerate(rna_TF.columns)}

loss_by_epoch: dict =  {"epoch": [], "train_loss": [], "val_loss": [], "epoch_sec": []}

# --- build robust per-gene stats on the TRAIN SPLIT only ---
n_genes = len(genes)
sum_y   = torch.zeros(n_genes, dtype=torch.float32)
sum_y2  = torch.zeros(n_genes, dtype=torch.float32)
count   = torch.zeros(n_genes, dtype=torch.int32)

for c in train_cells:                     # use *global* train_cells, not sharded
    y, m = get_true_tg_expr_vector_from_data(mesc_rna_data_shared[c], genes)
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
logging.info(f"Genes seen in training: {seen}/{len(genes)}")

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

assert len(genes) == gene_ids.numel()

if rank==0:
    logging.info("\n ----- Starting Training -----")
for epoch in range(start_epoch, epochs+1):
    did_optim_step = False
    
    p0 = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
    
    epoch_start_time = time.time()
    # ---- TRAIN ----
    model.train()  # <— instead of toggling each submodule
    inner = unwrap(model)
    train_loss_sum, n_train = 0.0, 0
    if rank == 0:
        logging.info("  - Running Training:")
    for bi, cell_batch in enumerate(iter_batches(train_cells_rank, batch_size), start=1):
        if rank == 0 and bi % 5 == 0:
            logging.info(f"       {bi}/{(len(train_cells_rank) + batch_size - 1)//batch_size}")
        opt.zero_grad(set_to_none=True)

        tokens_b = build_tokens_sparse_for_cells(
            cell_batch, H, D, peaks_by_window, col_of, rna_arr, atac_arr,
            inner.proj_tg, inner.proj_tf, inner.proj_window, pool, device
        )
        
        # Build predictions (keep encoder + cross-attn in fp32 to avoid fp16 overflows)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            Henc = inner.encoder(tokens_b.float())
            GQ   = inner.gene_embed(gene_ids).unsqueeze(0).expand(tokens_b.size(0), -1, -1).float()
            Z, _ = inner.cross_attn(query=GQ, key=Henc, value=Henc, need_weights=False)
            pred = inner.readout(Z).squeeze(-1)  # [B, n_genes], fp32

        # targets
        y_list, m_list = [], []
        for c in cell_batch:
            y, m = get_true_tg_expr_vector_from_data(mesc_rna_data_shared[c], genes)
            y_list.append(y); m_list.append(m)

        y_true = torch.cat(y_list, dim=0).to(device)     # [B, n_genes]
        mask_t = torch.cat(m_list, dim=0).to(device)     # [B, n_genes] (True where target exists)
        
        y_norm = (y_true - mu) / sd
        y_norm = torch.clamp(y_norm, -8.0, 8.0)  # optional, very stabilizing
        
        # final mask: seen in training, present in this cell, and finite
        mask_eff = mask_t & seen_genes_mask.unsqueeze(0) & torch.isfinite(y_norm)
        pred     = torch.nan_to_num(pred, 0.0)           # guard against rare NaNs anyway

        if not mask_eff.any():
            continue

        loss = torch.nn.functional.huber_loss(pred[mask_eff], y_norm[mask_eff], delta=1.0)
        
        if rank == 0 and epoch == start_epoch and bi == 1:
            print("dbg:",
                "seen_genes", int(seen_genes_mask.sum()),
                "sd_min", float(sd.min()),
                "sd_p1",  float(torch.quantile(sd, 0.01)),
                "absmax(y_norm)", float(y_norm.abs().max()),
                "mask_eff_sum", int(mask_eff.sum()))

        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)

        # check grads sanity
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(opt)
        scaler.update()
        scale_after = scaler.get_scale()
        did_optim_step = True

        if rank == 0 and bi % 5 == 0:
            logging.debug(f"[b{bi}] loss={loss.item():.4f} mask_sum={int(mask_t.sum())} "
                        f"grad_norm={float(gn):.3e} scale {scale_before:.1e}->{scale_after:.1e}")

        train_loss_sum += loss.item() * len(cell_batch)
        n_train        += len(cell_batch)

    # ---- VAL ----
    model.eval()
    val_loss_sum, n_valtot = 0.0, 0
    with torch.no_grad():
        if rank == 0:
            logging.info("  - Running Validation:")

        for bi, cell_batch in enumerate(iter_batches(val_cells_rank, batch_size), start=1):
            if rank == 0 and bi % 5 == 0:
                logging.info(f"       {bi}/{(len(val_cells_rank) + batch_size - 1)//batch_size}")
            opt.zero_grad(set_to_none=True)
            
            tokens_b = build_tokens_sparse_for_cells(
                cell_batch, H, D, peaks_by_window, col_of, rna_arr, atac_arr,
                inner.proj_tg, inner.proj_tf, inner.proj_window, pool, device
            )
            
            # Build predictions (keep encoder + cross-attn in fp32 to avoid fp16 overflows)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                Henc = inner.encoder(tokens_b.float())
                GQ   = inner.gene_embed(gene_ids).unsqueeze(0).expand(tokens_b.size(0), -1, -1).float()
                Z, _ = inner.cross_attn(query=GQ, key=Henc, value=Henc, need_weights=False)
                pred = inner.readout(Z).squeeze(-1)  # [B, n_genes], fp32

            # targets
            y_list, m_list = [], []
            for c in cell_batch:
                y, m = get_true_tg_expr_vector_from_data(mesc_rna_data_shared[c], genes)
                y_list.append(y); m_list.append(m)

            y_true = torch.cat(y_list, dim=0).to(device)     # [B, n_genes]
            mask_t = torch.cat(m_list, dim=0).to(device)     # [B, n_genes] (True where target exists)

            y_norm = (y_true - mu) / sd
            y_norm = torch.clamp(y_norm, -8.0, 8.0)  # optional, very stabilizing
            
            # final mask: seen in training, present in this cell, and finite
            mask_eff = mask_t & seen_genes_mask.unsqueeze(0) & torch.isfinite(y_norm)
            pred     = torch.nan_to_num(pred, 0.0)           # guard against rare NaNs anyway

            if not mask_eff.any():
                continue

            loss = torch.nn.functional.huber_loss(pred[mask_eff], y_norm[mask_eff], delta=1.0)
            
            if rank == 0 and epoch == start_epoch and bi == 1:
                print("dbg:",
                    "seen_genes", int(seen_genes_mask.sum()),
                    "sd_min", float(sd.min()),
                    "sd_p1",  float(torch.quantile(sd, 0.01)),
                    "absmax(y_norm)", float(y_norm.abs().max()),
                    "mask_eff_sum", int(mask_eff.sum()))

            # accumulate like training
            val_loss_sum += loss.item() * len(cell_batch)
            n_valtot     += len(cell_batch)

        
    torch.cuda.synchronize(device)
    
    if rank == 0:
        logging.info("  - Validation complete, updating calculating train/val average loss")
    
    # ---- per-rank sums -> global weighted averages ----
    train_sum_t = torch.tensor([train_loss_sum], device=device, dtype=torch.float32)
    train_cnt_t = torch.tensor([n_train],        device=device, dtype=torch.float32)
    val_sum_t   = torch.tensor([val_loss_sum],   device=device, dtype=torch.float32)
    val_cnt_t   = torch.tensor([n_valtot],       device=device, dtype=torch.float32)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(train_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_cnt_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_sum_t,   op=dist.ReduceOp.SUM)
        dist.all_reduce(val_cnt_t,   op=dist.ReduceOp.SUM)

    train_loss_avg = (train_sum_t / torch.clamp(train_cnt_t, min=1)).item()
    val_loss_avg   = (val_sum_t   / torch.clamp(val_cnt_t,   min=1)).item()
    
    is_best = (val_loss_avg < best_val - 1e-5)
    stop    = (pat+1 >= patience) if not is_best else False

    # compute early-stop control on rank 0, then broadcast a small CPU object
    is_best = (val_loss_avg < best_val - 1e-5)
    if dist.is_available() and dist.is_initialized():
        payload = [is_best, best_val, val_loss_avg, pat]
        if rank != 0:
            payload = [None, None, None, None]
        # broadcast the decision using object list to avoid extra CUDA traffic here
        dist.broadcast_object_list(payload, src=0)
        is_best, best_val_b, val_loss_avg_b, pat_b = payload
        # sync the scalars we use
        val_loss_avg = float(val_loss_avg_b)
        if is_best:
            best_val = min(best_val, val_loss_avg)
            pat = 0
        else:
            pat = int(pat_b) + 1
    else:
        best_val = min(best_val, val_loss_avg) if is_best else best_val
        pat = 0 if is_best else (pat + 1)

    stop = (pat >= patience)

    # ---- logging (rank 0 only) ----
    if rank == 0:
        print(f"[Epoch {epoch:02d}] train_loss={train_loss_avg:.4f}  val_loss={val_loss_avg:.4f}")
        
    if dist.is_available() and dist.is_initialized():
        dist.barrier() 

    # ---- checkpointing (rank 0 only), using GLOBAL val_loss_avg and best_val ----
    if rank == 0:
        # save_regular(model, opt, sched, epoch, loss=train_loss_avg, best_val=best_val)
        if is_best:
            logging.info(f"New best: {val_loss_avg:.4f} at epoch {epoch}")
            model_cpu_state = {k: v.cpu() for k, v in unwrap(model).state_dict().items()}
            ckpt = {
                "epoch": epoch,
                "model": model_cpu_state,
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "best_val": best_val,
            }
            torch.save(ckpt, os.path.join(output_dir, "checkpoints/best_model.pth.tar"))
            # clean_old_checkpoints(keep_last=3)
        else:
            logging.info(f"No improvement ({pat}/{patience}) — val {val_loss_avg:.4f} ≥ best {best_val:.4f}")
    
    # ensure all ranks reach the same point before next epoch (one barrier per epoch)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # early stop: everyone uses the SAME 'stop'
    if stop:
        if rank == 0:
            logging.info(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
            training_stats = pd.DataFrame(loss_by_epoch).set_index("epoch")
            training_stats.to_csv(os.path.join(output_dir, f"training_stats/training_loss_window_{window_size // 1000}kb_{num_cells}_cells.csv"), header=True, index=True)

        break
    
    if rank == 0:
        time_sec = time.time() - epoch_start_time
        loss_by_epoch["epoch"].append(epoch)
        loss_by_epoch["train_loss"].append(train_loss_avg)
        loss_by_epoch["val_loss"].append(val_loss_avg)      
        loss_by_epoch["epoch_sec"].append(time_sec)  

    # step the scheduler once per epoch on every rank (keeps LR in sync)
    if did_optim_step:
        sched.step()

if rank == 0:
    logging.info("\nTRAINING COMPLETE, ENDING PROCESS")
    training_stats = pd.DataFrame(loss_by_epoch).set_index("epoch")
    training_stats.to_csv(os.path.join(output_dir, f"training_stats/training_loss_window_{window_size // 1000}kb_{num_cells}_cells.csv"), header=True, index=True)

if dist.is_available() and dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()