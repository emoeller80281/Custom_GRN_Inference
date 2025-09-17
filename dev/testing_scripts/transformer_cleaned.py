from __future__ import annotations
import os, glob, time, logging, warnings
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

import pybedtools
from grn_inference import utils

# ------------------------------
# Global defaults (override via env or edit)
# ------------------------------
PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
MM10_DIR     = f"{PROJECT_DIR}/data/reference_genome/mm10"
MM10_TSS_BED = f"{PROJECT_DIR}/data/genome_annotation/mm10/mm10_TSS.bed"
GROUND_TRUTH = f"{PROJECT_DIR}/ground_truth_files"  # reserved
SAMPLE_DIR   = f"{PROJECT_DIR}/input/mESC/filtered_L2_E7.5_rep1"
OUTPUT_DIR   = f"{PROJECT_DIR}/output/transformer_testing_output"

CHROM_SIZES  = f"{MM10_DIR}/chrom.sizes"
FASTA_EX     = f"{MM10_DIR}/chr1.fa"  # not used directly here yet

WINDOW_BP    = 1000
N_CELLS      = 100
EPOCHS       = 20
VAL_FRAC     = 0.15
BATCH_SIZE   = 1
PATIENCE     = 5
LR           = 1e-3
WD           = 1e-4

d_model              = 192
encoder_nhead        = 6
encoder_dim_ff       = 1024
encoder_num_layers   = 3
kernel_and_stride    = 32

# ------------------------------
# Housekeeping & determinism
# ------------------------------
warnings.filterwarnings(
    "ignore",
    message=r"No device id is provided via `init_process_group` or `barrier `",
)

torch.autograd.set_detect_anomaly(True)

def _setup_threads() -> None:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    per = max(1, os.cpu_count() // max(1, world))
    for var in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[var] = str(per)
    torch.set_num_threads(per)
    torch.set_num_interop_threads(1)


def _set_seeds() -> None:
    torch.manual_seed(1)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["TORCH_ALLOW_TF32"] = "1"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"


# ------------------------------
# DDP / logging
# ------------------------------

def is_ddp() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1 or int(os.environ.get("SLURM_NTASKS", "1")) > 1


def init_distributed() -> Tuple[int, int, int]:
    if all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank       = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    if rank == 0:
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}  nGPU={torch.cuda.device_count()}")
    return rank, world_size, local_rank


def setup_logging(rank: int) -> None:
    handlers: List[logging.Handler] = []
    if rank == 0:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.ERROR,
                        handlers=handlers, format="%(message)s")


def _rank0() -> bool:
    return (dist.is_available() and dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized()


# ------------------------------
# Checkpoint helpers
# ------------------------------
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/training_stats", exist_ok=True)


def _atomic_save(obj: Dict[str, Any], path: str) -> None:
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in sd):
        return OrderedDict((k.replace("module.", "", 1), v) for k, v in sd.items())
    return sd


def save_best(model: nn.Module, opt, sched, epoch: int, best_val: float) -> None:
    if not _rank0():
        if dist.is_initialized():
            dist.barrier()
        return
    state = {
        "epoch": epoch,
        "model": {k: v.cpu() for k, v in unwrap(model).state_dict().items()},
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "best_val": best_val,
        "torch_version": torch.__version__,
    }
    _atomic_save(state, f"{CHECKPOINT_DIR}/best_model.pth.tar")
    if dist.is_initialized():
        dist.barrier()


def get_latest_checkpoint() -> Optional[str]:
    files = sorted(glob.glob(f"{CHECKPOINT_DIR}/checkpoint_epoch_*.pth.tar"), key=os.path.getmtime)
    return files[-1] if files else None


# ------------------------------
# Data prep
# ------------------------------

def load_inputs(rank: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    if rank == 0:
        logging.info("\n----- Loading and Formatting Input Data -----")
        logging.info("Reading and formatting TSS bed file")
    tss_bed = pybedtools.BedTool(MM10_TSS_BED)
    gene_tss_df = (
        tss_bed.filter(lambda x: x.chrom == "chr1")
               .saveas(f"{MM10_DIR}/mm10_ch1_gene_tss.bed")
               .to_dataframe().sort_values("start")
    )

    if rank == 0:
        logging.info("Reading processed scATAC-seq dataset")
    atac_df = pd.read_parquet(f"{SAMPLE_DIR}/mESC_filtered_L2_E7.5_rep1_ATAC_processed.parquet").set_index("peak_id")
    atac_df = atac_df.loc[utils.format_peaks(atac_df.index).query("chromosome=='chr1'").peak_id]

    if rank == 0:
        logging.info("Reading in the scRNA-seq dataset")
    rna_df = pd.read_parquet(f"{SAMPLE_DIR}/mESC_filtered_L2_E7.5_rep1_RNA_processed.parquet").set_index("gene_id")

    # windows
    genome_window_file = f"{MM10_DIR}/mm10_chr1_windows_{WINDOW_BP//1000}kb.bed"
    if not os.path.exists(genome_window_file):
        if rank == 0:
            logging.info("Creating genomic windows")
        win_bt = pybedtools.bedtool.BedTool().window_maker(g=CHROM_SIZES, w=WINDOW_BP)
        windows = win_bt.filter(lambda x: x.chrom == "chr1").saveas(genome_window_file).to_dataframe()
    else:
        if rank == 0:
            logging.info("Loading existing genomic windows")
        windows = pybedtools.BedTool(genome_window_file).to_dataframe()

    # Peak/TSS pairing
    out_dir = OUTPUT_DIR
    peak_tmp = f"{out_dir}/peak_tmp.bed"
    tss_tmp  = f"{out_dir}/tss_tmp.bed"

    peak_locs = utils.format_peaks(atac_df.index.to_numpy())
    peak_locs = peak_locs.rename(columns={"chromosome": "chrom"})
    peak_locs = peak_locs.query("chrom=='chr1'")

    if not (os.path.exists(peak_tmp) and os.path.exists(tss_tmp)):
        if rank == 0:
            logging.info("Calculating peak to TG distance score")
        pybedtools.BedTool.from_dataframe(peak_locs[["chrom", "start", "end", "peak_id"]]).saveas(peak_tmp)
        pybedtools.BedTool.from_dataframe(gene_tss_df[["chrom", "start", "end", "name"]]).saveas(tss_tmp)

    peak_bed = pybedtools.BedTool(peak_tmp)
    tss_bed  = pybedtools.BedTool(tss_tmp)
    genes_near_peaks = utils.find_genes_near_peaks(peak_bed, tss_bed)
    genes_near_peaks = genes_near_peaks.loc[genes_near_peaks["TSS_dist"] <= 1e6].copy()
    genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / 250_000)
    gnp_path = f"{out_dir}/genes_near_peaks.parquet"
    genes_near_peaks.to_parquet(gnp_path, compression="snappy", engine="pyarrow")
    genes_near_peaks = pd.read_parquet(gnp_path, engine="pyarrow")

    homer = pd.read_parquet(f"{out_dir}/homer_tf_to_peak.parquet", engine="pyarrow").reset_index(drop=True)
    homer["source_id"] = homer["source_id"].astype(str).str.capitalize()

    # cell alignment
    atac_cells = atac_df.columns.tolist()
    rna_cells  = rna_df.columns.tolist()
    shared = sorted(set(c for c in atac_cells if c in rna_cells))[:N_CELLS]
    return gene_tss_df, atac_df[shared], rna_df[shared], genes_near_peaks, peak_locs, windows, shared


def build_sparse_components(homer: pd.DataFrame, genes_near_peaks: pd.DataFrame) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    tfs   = homer["source_id"].astype(str).str.capitalize().unique()
    genes = genes_near_peaks["target_id"].astype(str).unique()
    peaks = np.unique(np.concatenate([
        homer["peak_id"].astype(str).values,
        genes_near_peaks["peak_id"].astype(str).values,
    ]))

    tf_i   = {t: i for i, t in enumerate(tfs)}
    peak_i = {p: i for i, p in enumerate(peaks)}
    gene_i = {g: i for i, g in enumerate(genes)}

    Hr = homer[["source_id", "peak_id", "homer_binding_score"]].copy()
    Hr["source_id"] = Hr["source_id"].astype(str).str.capitalize().map(tf_i)
    Hr["peak_id"]   = Hr["peak_id"].astype(str).map(peak_i)
    Hr = Hr.dropna(subset=["source_id", "peak_id"]).astype({"source_id": int, "peak_id": int})
    H = sparse.csr_matrix((Hr["homer_binding_score"].astype(np.float32).values,
                           (Hr["source_id"].values, Hr["peak_id"].values)),
                          shape=(len(tfs), len(peaks)))

    Dr = genes_near_peaks[["peak_id", "target_id", "TSS_dist_score"]].copy()
    Dr["peak_id"]   = Dr["peak_id"].astype(str).map(peak_i)
    Dr["target_id"] = Dr["target_id"].astype(str).map(gene_i)
    Dr = Dr.dropna(subset=["peak_id", "target_id"]).astype({"peak_id": int, "target_id": int})
    D = sparse.csr_matrix((Dr["TSS_dist_score"].astype(np.float32).values,
                           (Dr["peak_id"].values, Dr["target_id"].values)),
                          shape=(len(peaks), len(genes)))
    return H, D, tfs, peaks, genes


def assign_peaks_to_windows(peak_locs: pd.DataFrame, windows: pd.DataFrame, peaks: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, int]:
    peak_coords = peak_locs.loc[peak_locs["peak_id"].isin(peaks), ["peak_id", "chrom", "start", "end"]].copy()
    peak_coords = peak_coords.query("chrom=='chr1'")
    coord_map = peak_coords.set_index("peak_id")[{"start", "end"}].to_dict(orient="index")

    w = int((windows["end"] - windows["start"]).mode().iloc[0])
    win_lut: Dict[int, str] = {}
    for _, row in windows.iterrows():
        k = row["start"] // w
        win_lut[k] = f"{row['chrom']}:{row['start']}-{row['end']}"
    nW = len(win_lut)

    rng = np.random.default_rng(0)

    def assign_best_window(start: int, end: int, w: int) -> int:
        i0 = start // w
        i1 = max(i0, (end - 1) // w)
        best_k, best_ov, ties = i0, -1, []
        for k in range(i0, i1 + 1):
            bs, be = k * w, (k + 1) * w
            ov = max(0, min(end, be) - max(start, bs))
            if ov > best_ov:
                best_k, best_ov, ties = k, ov, [k]
            elif ov == best_ov:
                ties.append(k)
        return best_k if len(ties) == 1 else int(rng.choice(ties))

    peak_to_window_idx = np.full(len(peaks), -1, dtype=np.int32)
    peak_i = {p: i for i, p in enumerate(peaks)}
    for p, idx in peak_i.items():
        info = coord_map.get(p)
        if info is None: continue
        peak_to_window_idx[idx] = assign_best_window(int(info["start"]), int(info["end"]), w)

    peaks_by_window = [np.where(peak_to_window_idx == k)[0] for k in range(nW)]
    window_ids = np.array([win_lut[k] for k in range(nW)], dtype=object)
    return peaks_by_window, window_ids, nW


# ------------------------------
# Model blocks
# ------------------------------
class CheckpointedEncoder(nn.Module):
    def __init__(self, layers: Sequence[nn.Module], norm: Optional[nn.Module] = None,
                 use_checkpoint: bool = True, use_reentrant: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, use_reentrant=self.use_reentrant)
            else:
                x = layer(x)
        return self.norm(x) if self.norm is not None else x


class ExprPredictor(nn.Module):
    def __init__(self, TF: int, TG: int, d_model: int, nhead: int, dim_ff: int, n_layers: int,
                 kernel_stride: int) -> None:
        super().__init__()
        tf_ch, tg_ch = 32, 64
        self.proj_tf = nn.Linear(TF, tf_ch, bias=False)
        self.proj_tg = nn.Linear(TG, tg_ch, bias=False)
        self.proj_window = nn.Sequential(nn.Linear(tf_ch * tg_ch, d_model), nn.Dropout(0.1))
        enc_layers = [
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                       batch_first=True, norm_first=True, dropout=0.1)
            for _ in range(n_layers)
        ]
        self.encoder = CheckpointedEncoder(enc_layers, norm=nn.LayerNorm(d_model), use_checkpoint=True)
        self.pool_ks = kernel_stride
        self.gene_embed = nn.Embedding(TG, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.readout = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))


# ------------------------------
# Token building & forward
# ------------------------------
@torch.no_grad()
def _safe_nan_to_num(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, 0.0)


def build_tokens_sparse_for_cells(
    cell_ids: Sequence[str],
    H: sparse.csr_matrix, D: sparse.csr_matrix,
    peaks_by_window: Sequence[np.ndarray],
    col_of: Dict[str, int],
    rna_arr: np.ndarray, atac_arr: np.ndarray,
    model: ExprPredictor,
    device: torch.device,
    clamp_val: float = 3.0,
) -> torch.Tensor:
    pool_ks = model.pool_ks
    out_batches: List[torch.Tensor] = []
    for cell in cell_ids:
        col = col_of[cell]
        e = rna_arr[:, col]  # TF
        a = atac_arr[:, col] # Peak
        B = H.multiply(e[:, None]).tocsr()  # [TF,P]
        R = D.multiply(a[:, None]).tocsr()  # [P,G]
        pooled_list: List[torch.Tensor] = []
        sum_tok: Optional[torch.Tensor] = None
        cnt = 0
        for p_idx in peaks_by_window:
            p_active = p_idx[a[p_idx] > 0]
            if p_active.size == 0:
                continue
            Mw = (B[:, p_active] @ R[p_active, :]).toarray()  # [TF,G]
            M = torch.as_tensor(Mw, device=device, dtype=torch.float32)
            # stabilize across genes per TF
            M = (M - M.mean(dim=1, keepdim=True)) / (M.std(dim=1, keepdim=True) + 1e-6)
            M = torch.clamp(M, -clamp_val, clamp_val)
            M = _safe_nan_to_num(M)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                Xtf  = model.proj_tf(M.t().unsqueeze(0))   # [1,G,tf]
                Xtg  = model.proj_tg(Xtf.transpose(1, 2))  # [1,tf,tg]
                tok  = model.proj_window(Xtg.reshape(1, -1))  # [1,d]
            sum_tok = tok if sum_tok is None else (sum_tok + tok)
            cnt += 1
            if cnt == pool_ks:
                pooled_list.append((sum_tok / float(pool_ks)).unsqueeze(1))  # [1,1,d]
                sum_tok, cnt = None, 0
            del M, Xtg, tok
        if cnt > 0 and sum_tok is not None:
            pooled_list.append((sum_tok / float(cnt)).unsqueeze(1))
        out_batches.append(torch.zeros((1, 1, d_model), device=device) if not pooled_list
                            else torch.cat(pooled_list, dim=1))
        del B, R
    return torch.cat(out_batches, dim=0)  # [B,W',d]


def forward_tokens_to_pred(tokens: torch.Tensor, model: ExprPredictor, device: torch.device) -> torch.Tensor:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        H = model.encoder(tokens.float())
        B = H.size(0)
        gene_ids = torch.arange(model.gene_embed.num_embeddings, device=device)
        GQ = model.gene_embed(gene_ids).unsqueeze(0).expand(B, -1, -1).float()
        Z, _ = model.cross_attn(query=GQ, key=H, value=H, need_weights=False)
        pred = model.readout(Z).squeeze(-1)  # [B,G]
    return pred


# ------------------------------
# Training utilities
# ------------------------------

def unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m


def iter_batches(items: Sequence[str], bs: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), bs):
        yield items[i:i+bs]


def train_val_split(cells: Sequence[str], val_frac: float, seed: int = 0) -> Tuple[List[str], List[str]]:
    rng = np.random.default_rng(seed)
    cells = list(cells)
    rng.shuffle(cells)
    n_val = max(1, int(len(cells) * val_frac))
    return cells[n_val:], cells[:n_val]


def shard_list_per_rank(items: Sequence[str], rank: int, world_size: int, pad: bool = True) -> List[str]:
    n = len(items)
    per = (n + world_size - 1) // world_size
    start = rank * per
    end = min(start + per, n)
    shard = list(items[start:end])
    if pad and len(shard) < per and len(shard) > 0:
        shard += [shard[-1]] * (per - len(shard))
    return shard


def get_true_vec(rna_data_col: pd.Series, genes: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    dup = pd.Index(genes).duplicated()
    assert not dup.any(), "Duplicate gene IDs in prediction axis"
    vec = rna_data_col.reindex(genes)
    mask = ~vec.isna().to_numpy()
    y = torch.tensor(vec.to_numpy(dtype=float), dtype=torch.float32).unsqueeze(0)
    m = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
    return y, m


def compute_train_stats(rna_df: pd.DataFrame, train_cells: Sequence[str], genes: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nG = len(genes)
    sum_y = torch.zeros(nG, dtype=torch.float32)
    sum_y2 = torch.zeros(nG, dtype=torch.float32)
    count = torch.zeros(nG, dtype=torch.int32)
    for c in train_cells:
        y, m = get_true_vec(rna_df[c], genes)
        y = y.squeeze(0).to(torch.float32)
        m = m.squeeze(0)
        sum_y  += torch.where(m, y, 0).cpu()
        sum_y2 += torch.where(m, y*y, 0).cpu()
        count  += m.to(torch.int32).cpu()
    count_f = count.to(torch.float32).clamp_min(1.0)
    mu = (sum_y / count_f).clone()
    var = (sum_y2 / count_f) - mu * mu
    var.clamp_(min=0.0)
    sd = var.sqrt_()
    never_seen = (count == 0)
    mu[never_seen] = 0.0
    sd[never_seen] = 1.0
    seen_mask = (count > 0)
    return mu, sd.clamp_min(1e-6), seen_mask


# ------------------------------
# Main train loop
# ------------------------------

def main() -> None:
    _setup_threads()
    _set_seeds()

    ddp = is_ddp()
    if ddp:
        rank, world, local = init_distributed()
    else:
        rank, world, local = 0, 1, 0
    setup_logging(rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # I/O
    gene_tss_df, atac_df, rna_df, gnp, peak_locs, windows, shared = load_inputs(rank)

    # HOMER table is needed for H; load now to pass into builder
    homer = pd.read_parquet(f"{OUTPUT_DIR}/homer_tf_to_peak.parquet", engine="pyarrow").reset_index(drop=True)
    homer["source_id"] = homer["source_id"].astype(str).str.capitalize()

    # Sparse blocks
    H, D, tfs, peaks, genes = build_sparse_components(homer, gnp)

    # Peak→window mapping
    peaks_by_window, window_ids, nW = assign_peaks_to_windows(peak_locs, windows, peaks)

    # Shapes/info
    if rank == 0:
        nonempty = sum(1 for p in peaks_by_window if p.size > 0)
        logging.info("Data Size:")
        logging.info(f"  - Window Size: {WINDOW_BP} bp")
        logging.info(f"  - Windows: {len(peaks_by_window):,} | non-empty: {nonempty:,}")
        logging.info(f"  - Num TFs = {len(tfs):,}")
        logging.info(f"  - Num Peaks = {len(peaks):,}")
        logging.info(f"  - Num TGs = {len(genes):,}")
        logging.info(f"  - Num Cells = {len(shared)}")
        logging.info("\nModel Parameters:")
        logging.info(f"  - Model Dimensions = {d_model}")
        logging.info(f"  - TF×TG Features/Window -> {d_model}")
        logging.info(f"  - Window Pooling: kernel=stride={kernel_and_stride}")
        logging.info("\nEncoder Settings")
        logging.info(f"  - Layers={encoder_num_layers}  Heads={encoder_nhead}  FF={encoder_dim_ff}")

    # Align matrices
    rna_TF = rna_df.reindex(index=tfs).fillna(0).astype("float32")
    atac_P = atac_df.reindex(index=peaks).fillna(0).astype("float32")
    assert set(rna_TF.columns) == set(atac_P.columns)
    rna_arr = rna_TF.values
    atac_arr = atac_P.values
    col_of = {c: i for i, c in enumerate(rna_TF.columns)}

    # Model
    model = ExprPredictor(TF=len(tfs), TG=len(genes), d_model=d_model,
                          nhead=encoder_nhead, dim_ff=encoder_dim_ff,
                          n_layers=encoder_num_layers, kernel_stride=kernel_and_stride).to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local] if torch.cuda.is_available() else None,
            output_device=local if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS))

    # Split cells
    train_cells, val_cells = train_val_split(shared, VAL_FRAC, seed=0)
    train_cells_rank = shard_list_per_rank(train_cells, rank, world, pad=True)
    val_cells_rank   = shard_list_per_rank(val_cells,   rank, world, pad=True)

    # Train stats (from global train split only)
    mu, sd, seen_mask = compute_train_stats(rna_df, train_cells, genes)
    mu = mu.to(device)
    sd = sd.to(device)
    seen_mask = seen_mask.to(device)
    if dist.is_available() and dist.is_initialized():
        stats = torch.stack([mu, sd, seen_mask.float()], dim=0)
        dist.broadcast(stats, src=0)
        mu, sd, seen_mask = stats[0], stats[1], (stats[2] > 0.5)

    loss_log = {"epoch": [], "train_loss": [], "val_loss": [], "epoch_sec": []}

    if rank == 0:
        logging.info("\n----- Starting Training -----")

    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    def masked_mse(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        y_norm = (y - mu) / sd
        y_norm = torch.nan_to_num(y_norm, 0.0)
        mask_eff = m & seen_mask.unsqueeze(0) & torch.isfinite(y_norm)
        if not mask_eff.any():
            return torch.tensor(0.0, device=pred.device)
        diff2 = (torch.nan_to_num(pred, 0.0) - y_norm) ** 2
        return diff2[mask_eff].mean()

    best_val, pat = float("inf"), 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        # ---- TRAIN ----
        unwrap(model).train()
        train_sum, n_train = 0.0, 0
        for bi, cell_batch in enumerate(iter_batches(train_cells_rank, BATCH_SIZE), start=1):
            opt.zero_grad(set_to_none=True)
            tokens = build_tokens_sparse_for_cells(cell_batch, H, D, peaks_by_window, col_of, rna_arr, atac_arr, unwrap(model), device)
            pred = forward_tokens_to_pred(tokens, unwrap(model), device)
            y_list, m_list = [], []
            for c in cell_batch:
                y, m = get_true_vec(rna_df[c], genes)
                y_list.append(y); m_list.append(m)
            y_true = torch.cat(y_list, dim=0).to(device)
            mask_t = torch.cat(m_list, dim=0).to(device)
            loss = masked_mse(pred, y_true, mask_t)
            if not torch.isfinite(loss):
                logging.error("Non-finite loss; skipping batch")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_sum += loss.item() * len(cell_batch)
            n_train   += len(cell_batch)
        # ---- VAL ----
        unwrap(model).eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for bi, cell_batch in enumerate(iter_batches(val_cells_rank, BATCH_SIZE), start=1):
                tokens = build_tokens_sparse_for_cells(cell_batch, H, D, peaks_by_window, col_of, rna_arr, atac_arr, unwrap(model), device)
                pred = forward_tokens_to_pred(tokens, unwrap(model), device)
                y_list, m_list = [], []
                for c in cell_batch:
                    y, m = get_true_vec(rna_df[c], genes)
                    y_list.append(y); m_list.append(m)
                y_true = torch.cat(y_list, dim=0).to(device)
                mask_t = torch.cat(m_list, dim=0).to(device)
                loss = masked_mse(pred, y_true, mask_t)
                val_sum += loss.item() * len(cell_batch)
                n_val   += len(cell_batch)

        # ---- reduce across ranks ----
        dev = device if device.type == "cuda" else torch.device("cpu")
        t_sum = torch.tensor([train_sum], device=dev)
        t_cnt = torch.tensor([n_train], device=dev)
        v_sum = torch.tensor([val_sum], device=dev)
        v_cnt = torch.tensor([n_val], device=dev)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_cnt, op=dist.ReduceOp.SUM)
            dist.all_reduce(v_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(v_cnt, op=dist.ReduceOp.SUM)
        train_loss = (t_sum / torch.clamp(t_cnt, min=1)).item()
        val_loss   = (v_sum / torch.clamp(v_cnt, min=1)).item()

        if rank == 0:
            print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        # scheduler step per-epoch
        sched.step()

        # early stop logic (broadcast from rank0)
        if dist.is_available() and dist.is_initialized():
            payload: List[Any] = [None, None]
            if rank == 0:
                improved = val_loss < best_val - 1e-5
                if improved:
                    best_val, pat = val_loss, 0
                else:
                    pat += 1
                payload = [best_val, pat]
            dist.broadcast_object_list(payload, src=0)
            best_val, pat = float(payload[0]), int(payload[1])
        else:
            if val_loss < best_val - 1e-5:
                best_val, pat = val_loss, 0
            else:
                pat += 1

        # checkpoints
        if rank == 0:
            if val_loss <= best_val + 1e-9:
                logging.info(f"New best: {val_loss:.4f} at epoch {epoch}")
                save_best(model, opt, sched, epoch, best_val)
            else:
                logging.info(f"No improvement ({pat}/{PATIENCE}) — val {val_loss:.4f} ≥ best {best_val:.4f}")

        # logging
        if rank == 0:
            loss_log["epoch"].append(epoch)
            loss_log["train_loss"].append(train_loss)
            loss_log["val_loss"].append(val_loss)
            loss_log["epoch_sec"].append(time.time() - t0)

        # early stop
        if pat >= PATIENCE:
            if rank == 0:
                logging.info(f"Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
            break

    if rank == 0:
        logging.info("\nTRAINING COMPLETE, ENDING PROCESS")
        pd.DataFrame(loss_log).set_index("epoch").to_csv(
            f"{OUTPUT_DIR}/training_stats/training_loss_window_{WINDOW_BP//1000}kb_{N_CELLS}_cells.csv",
            header=True, index=True,
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
