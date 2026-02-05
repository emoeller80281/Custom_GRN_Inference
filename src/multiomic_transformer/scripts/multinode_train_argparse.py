import csv
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import sys

from numpy import save
sys.path.append(Path(__file__).resolve().parent.parent.parent)

import signal
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.distributed as dist
from typing import Any
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from multiomic_transformer.datasets.dataset_refactor import (
    MultiomicTransformerDataset,
    MultiChromosomeDataset,
    DistributedBatchSampler,
    fit_simple_scalers,
    SimpleScaler,
    IndexedChromBucketBatchSampler,
)
from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.utils.files import unique_path
from multiomic_transformer.utils import ewc_utils
from multiomic_transformer.scripts import gradient_attribution, tf_knockout

warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group`")

STOP_REQUESTED = False

# ----- Argument Parser Setup -----
def parse_training_args():
    """
    Parse command-line arguments for model training configuration.
    All global settings can be overridden via command-line arguments.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Train MultiomicTransformer model with distributed data parallel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ----- Data Paths -----
    parser.add_argument("--sample_data_cache_dir", type=Path, required=True,
                        help="Directory containing cached sample data")
    parser.add_argument("--common_data", type=Path, required=True,
                        help="Directory containing common data (vocabs, etc.)")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Base output directory for experiments")
    parser.add_argument("--chrom_id", type=str, default="chr19",
                        help="Chromosome ID for single-chromosome mode")
    parser.add_argument("--chrom_ids", type=str, nargs="+", default=None,
                        help="List of chromosome IDs for multi-chromosome mode")
    
    # ----- Training Parameters -----
    parser.add_argument("--total_epochs", type=int, default=250,
                        help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per GPU")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--save_every_n_epochs", type=int, default=5,
                        help="Save checkpoint every N epochs")
    
    # ----- Loss Weights -----
    parser.add_argument("--corr_loss_weight", type=float, default=1.0,
                        help="Weight for correlation loss")
    parser.add_argument("--edge_loss_weight", type=float, default=0.0,
                        help="Weight for edge loss (not used in this version)")
    parser.add_argument("--cos_weight", type=float, default=0.0,
                        help="Weight for cosine contrastive loss")
    parser.add_argument("--shortcut_reg_weight", type=float, default=0.0,
                        help="Weight for shortcut regularization")
    
    # ----- Optimization Parameters -----
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--use_grad_accumulation", action="store_true",
                        help="Use gradient accumulation")
    parser.add_argument("--use_grad_checkpointing", action="store_true",
                        help="Use gradient checkpointing to save memory")
    
    # ----- Learning Rate Schedule -----
    parser.add_argument("--mode", type=str, default="min",
                        choices=["min", "max"],
                        help="Mode for learning rate scheduler")
    parser.add_argument("--initial_learning_rate", type=float, default=2.5e-4,
                        help="Initial learning rate")
    parser.add_argument("--scheduler_factor", type=float, default=0.25,
                        help="Factor to reduce LR on plateau")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Scheduler patience (epochs)")
    parser.add_argument("--threshold", type=float, default=1e-3,
                        help="Threshold for measuring improvement")
    parser.add_argument("--threshold_mode", type=str, default="rel",
                        choices=["rel", "abs"],
                        help="Threshold mode: relative or absolute")
    parser.add_argument("--cooldown", type=int, default=4,
                        help="Cooldown period after LR reduction")
    parser.add_argument("--min_lr", type=float, default=2.5e-6,
                        help="Minimum learning rate")
    
    # ----- Model Architecture -----
    parser.add_argument("--d_model", type=int, default=192,
                        help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of transformer layers")
    parser.add_argument("--d_ff", type=int, default=None,
                        help="Feedforward dimension (default: 4 * d_model)")
    parser.add_argument("--dropout", type=float, default=0.10,
                        help="Dropout rate")
    
    # ----- Model Features -----
    parser.add_argument("--use_distance_bias", action="store_true",
                        help="Use distance bias in attention")
    parser.add_argument("--use_shortcut", action="store_true",
                        help="Use TF-to-TG shortcut connections")
    parser.add_argument("--use_motif_mask", action="store_true",
                        help="Use motif mask for TF-TG connections")
    parser.add_argument("--motif_mask_thresh", type=float, default=0.0,
                        help="Motif mask threshold")
    parser.add_argument("--motif_prior_scale", type=float, default=0.0,
                        help="Scale for motif prior scores")
    parser.add_argument("--attn_bias_scale", type=float, default=0.0,
                        help="Scale for attention bias")
    
    # ----- Shortcut Parameters -----
    parser.add_argument("--shortcut_l1", type=float, default=0.0,
                        help="L1 regularization for shortcut")
    parser.add_argument("--shortcut_l2", type=float, default=0.0,
                        help="L2 regularization for shortcut")
    parser.add_argument("--shortcut_topk", type=int, default=None,
                        help="Top-k sparsity for shortcut")
    parser.add_argument("--shortcut_dropout", type=float, default=0.0,
                        help="Dropout rate for shortcut")
    
    # ----- Data Subsampling -----
    parser.add_argument("--subsample_seed", type=int, default=42,
                        help="Random seed for subsampling")
    parser.add_argument("--allowed_samples", type=str, nargs="*", default=None,
                        help="List of allowed sample names (None = all)")
    
    # ----- Checkpoint Resumption -----
    parser.add_argument("--resume_checkpoint_path", type=Path, default=None,
                        help="Path to checkpoint to resume from")
    
    # ----- Model Compilation -----
    parser.add_argument("--use_torch_compile", action="store_true",
                        help="Use torch.compile() to optimize model (requires PyTorch 2.0+)")
    
    # ---- Profiling -----
    parser.add_argument("--use_profiler", action="store_true",
                    help="Enable TensorBoard profiler for performance analysis")
    parser.add_argument("--profiler_start_step", type=int, default=5,
                        help="Step to start profiling")
    parser.add_argument("--profiler_active_steps", type=int, default=3,
                        help="Number of steps to actively profile")
    
    return parser.parse_args()

def setup_training_globals(args):
    """
    Convert argparse arguments to global variables.
    This maintains backward compatibility with code that expects global variables.
    """
    global SAMPLE_DATA_CACHE_DIR, COMMON_DATA, OUTPUT_DIR, CHROM_ID, CHROM_IDS
    global TOTAL_EPOCHS, BATCH_SIZE, PATIENCE, SAVE_EVERY_N_EPOCHS
    global CORR_LOSS_WEIGHT, EDGE_LOSS_WEIGHT, COS_WEIGHT, SHORTCUT_REG_WEIGHT
    global GRAD_ACCUM_STEPS, USE_GRAD_ACCUMULATION, USE_GRAD_CHECKPOINTING
    global MODE, INITIAL_LEARNING_RATE, SCHEDULER_FACTOR, SCHEDULER_PATIENCE
    global THRESHOLD, THRESHOLD_MODE, COOLDOWN, MIN_LR
    global D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT
    global USE_DISTANCE_BIAS, USE_SHORTCUT, USE_MOTIF_MASK
    global MOTIF_MASK_THRESH, MOTIF_PRIOR_SCALE, ATTN_BIAS_SCALE
    global SHORTCUT_L1, SHORTCUT_L2, SHORTCUT_TOPK, SHORTCUT_DROPOUT
    global SUBSAMPLE_SEED, ALLOWED_SAMPLES
    global RESUME_CHECKPOINT_PATH, USE_TORCH_COMPILE
    global USE_PROFILER, PROFILER_START_STEP, PROFILER_ACTIVE_STEPS
    
    # ----- Data Paths -----
    SAMPLE_DATA_CACHE_DIR = args.sample_data_cache_dir
    COMMON_DATA = args.common_data
    OUTPUT_DIR = args.output_dir
    CHROM_ID = args.chrom_id
    USE_PROFILER = args.use_profiler
    PROFILER_START_STEP = args.profiler_start_step
    PROFILER_ACTIVE_STEPS = args.profiler_active_steps  
    
    # Handle chromosome list - default to a list if not provided
    if args.chrom_ids:
        CHROM_IDS = args.chrom_ids
    else:
        # Default for mouse chromosomes
        CHROM_IDS = [f"chr{i}" for i in range(1, 20)]
    
    # ----- Training Parameters -----
    TOTAL_EPOCHS = args.total_epochs
    BATCH_SIZE = args.batch_size
    PATIENCE = args.patience
    SAVE_EVERY_N_EPOCHS = args.save_every_n_epochs
    
    # ----- Loss Weights -----
    CORR_LOSS_WEIGHT = args.corr_loss_weight
    EDGE_LOSS_WEIGHT = args.edge_loss_weight
    COS_WEIGHT = args.cos_weight
    SHORTCUT_REG_WEIGHT = args.shortcut_reg_weight
    
    # ----- Optimization Parameters -----
    GRAD_ACCUM_STEPS = args.grad_accum_steps
    USE_GRAD_ACCUMULATION = args.use_grad_accumulation
    USE_GRAD_CHECKPOINTING = args.use_grad_checkpointing
    
    # ----- Learning Rate Schedule -----
    MODE = args.mode
    INITIAL_LEARNING_RATE = args.initial_learning_rate
    SCHEDULER_FACTOR = args.scheduler_factor
    SCHEDULER_PATIENCE = args.scheduler_patience
    THRESHOLD = args.threshold
    THRESHOLD_MODE = args.threshold_mode
    COOLDOWN = args.cooldown
    MIN_LR = args.min_lr
    
    # ----- Model Architecture -----
    D_MODEL = args.d_model
    NUM_HEADS = args.num_heads
    NUM_LAYERS = args.num_layers
    # D_FF defaults to 4 * D_MODEL if not specified
    D_FF = args.d_ff if args.d_ff is not None else (D_MODEL * 4)
    DROPOUT = args.dropout
    
    # ----- Model Features -----
    USE_DISTANCE_BIAS = args.use_distance_bias
    USE_SHORTCUT = args.use_shortcut
    USE_MOTIF_MASK = args.use_motif_mask
    MOTIF_MASK_THRESH = args.motif_mask_thresh
    MOTIF_PRIOR_SCALE = args.motif_prior_scale
    ATTN_BIAS_SCALE = args.attn_bias_scale
    
    # ----- Shortcut Parameters -----
    SHORTCUT_L1 = args.shortcut_l1
    SHORTCUT_L2 = args.shortcut_l2
    SHORTCUT_TOPK = args.shortcut_topk
    SHORTCUT_DROPOUT = args.shortcut_dropout
    
    # ----- Data Subsampling -----
    SUBSAMPLE_SEED = args.subsample_seed
    ALLOWED_SAMPLES = args.allowed_samples
    
    # ----- Checkpoint Resumption -----
    RESUME_CHECKPOINT_PATH = args.resume_checkpoint_path
    
    # ----- Model Compilation -----
    USE_TORCH_COMPILE = args.use_torch_compile

def update_info_file(info_file: Path, key: str, value: Any) -> None:
    """
    Update a JSON info file with a new key-value pair.
    If the file does not exist, it will be created.
    """
    if info_file.exists():
        with open(info_file, 'r') as f:
            info_data = json.load(f)
    else:
        info_data = {}
    
    info_data[key] = value
    
    with open(info_file, 'w') as f:
        json.dump(info_data, f, indent=4)

def _signal_handler(signum, frame):
    # Mark that we should shut down gracefully.
    global STOP_REQUESTED
    STOP_REQUESTED = True

# Register for both Ctrl+C and elastic's terminate
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

def ddp_setup(rank: int, world_size: int, local_rank: int):
    """
    Args:
        rank: Global identifier for this process (RANK)
        world_size: Total number of processes across all nodes
        local_rank: GPU index for this process on the current node
    """
    
    torch.cuda.set_device(local_rank)
    
    torch.backends.cuda.enable_flash_sdp(True)

    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)
    
def setup_logging(rank: int):
    # Remove existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    level = logging.INFO if rank == 0 else logging.ERROR
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.flush = sys.stdout.flush  # ensure flush

    logging.basicConfig(
        level=level,
        format=f"%(message)s",
        handlers=[handler],
        force=True
    )

def load_run_params_from_json(run_dir: Path) -> dict:
    """
    Load run_parameters.json from an existing training directory, if present.
    Returns {} if not found.
    """
    param_path = run_dir / "run_parameters.json"
    if not param_path.is_file():
        logging.warning(f"No run_parameters.json found in {run_dir}, using current config.")
        return {}
    with open(param_path, "r") as f:
        params = json.load(f)
    logging.info(f"Loaded run parameters from {param_path}")
    return params

def save_tf_tg_embeddings_from_model(model, out_dir, vocab_dir, epoch=None):
    model = getattr(model, "module", model)  # unwrap DDP if needed
    
    # Save embeddings 
    if epoch is not None:
        emb_save_path = os.path.join(out_dir, f"tf_tg_embeddings_{epoch}.pt")
    else:
        emb_save_path = os.path.join(out_dir, f"tf_tg_embeddings_final.pt")
    torch.save(
        {
            "tf_emb":     model.tf_identity_emb.weight.detach().cpu(),   # [T, D]
            "tg_query_emb":     model.tg_query_emb.weight.detach().cpu(),   # [G, D]
            "tg_emb": model.tg_identity_emb.weight.detach().cpu()
        },
        emb_save_path
    )

    # Prefer the copies we already wrote into the training_output_dir
    with open(os.path.join(vocab_dir, "tf_vocab.json")) as f:
        tf_vocab_obj = json.load(f)
    with open(os.path.join(vocab_dir, "tg_vocab.json")) as f:
        tg_vocab_obj = json.load(f)

    tf_name2id = tf_vocab_obj.get("name_to_id", tf_vocab_obj)
    tg_name2id = tg_vocab_obj.get("name_to_id", tg_vocab_obj)

    tf_id2name = [None] * len(tf_name2id)
    for name, idx in tf_name2id.items():
        tf_id2name[idx] = name
    tg_id2name = [None] * len(tg_name2id)
    for name, idx in tg_name2id.items():
        tg_id2name[idx] = name

    torch.save(
        {"tf_id2name": tf_id2name, "tg_id2name": tg_id2name},
        os.path.join(out_dir, "tf_tg_vocab_id2name.pt")
    )
    
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        global_rank: int,
        save_every: int,
        patience: int = 20,
        min_delta: float = 1e-3,
        grad_accum_steps: int = 1,
        use_grad_accumulation: bool = False,
        use_profiler: bool = False,
        profiler_dir: str = None,

    ) -> None:
        self.gpu_id = gpu_id
        self.global_rank = global_rank
        self.is_main = (global_rank == 0)
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.stop_requested = False
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.use_grad_accumulation = use_grad_accumulation
        # Loss warmup
        self.corr_sq_warmup_epochs = 3 
        
        self.best_val_loss = float("inf")
        
        # Profiler
        self.use_profiler = use_profiler
        self.profiler_dir = profiler_dir
        self.profiler = None
        
        self.scaler = GradScaler(init_scale=1024, growth_factor=1.5, backoff_factor=0.5, growth_interval=200)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode=MODE, 
            factor=SCHEDULER_FACTOR, 
            patience=SCHEDULER_PATIENCE,
            threshold=THRESHOLD,
            threshold_mode=THRESHOLD_MODE,
            cooldown=COOLDOWN,
            min_lr=MIN_LR,
        )
        
        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
    
    def _should_stop(self):
        global STOP_REQUESTED
        return self.stop_requested or STOP_REQUESTED
    
    def _save_trained_model(self, path: str):
        if not self.is_main:
            return

        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model

        model.eval()
        ckpt = {
            "model_state_dict": model.state_dict(),
        }

        if hasattr(self, "tf_scaler") and self.tf_scaler is not None:
            ckpt["tf_scaler_mean"] = self.tf_scaler.mean.detach().cpu()
            ckpt["tf_scaler_std"]  = self.tf_scaler.std.detach().cpu()
        if hasattr(self, "tg_scaler") and self.tg_scaler is not None:
            ckpt["tg_scaler_mean"] = self.tg_scaler.mean.detach().cpu()
            ckpt["tg_scaler_std"]  = self.tg_scaler.std.detach().cpu()

        out_path = os.path.join(path, "trained_model.pt")
        torch.save(ckpt, out_path)
        logging.info(f"Saved trained model to {out_path}")

    def _handle_abort(self, epoch, path, history, reason: str):
        # Only rank 0 writes, but all ranks should sync.
        if self.is_main:
            logging.info(f"{reason}: saving checkpoint and logs before exit.")
            last_epoch = max(0, epoch)
            self._save_checkpoint(last_epoch, path)
            self._save_trained_model(path)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _run_batch(self, batch):
        # Data transfer to GPU with profiler annotation
        with record_function("data_transfer"):
            atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
            atac_wins = atac_wins.to(self.gpu_id)
            tf_tensor = tf_tensor.to(self.gpu_id)
            targets   = targets.to(self.gpu_id)
            bias      = bias.to(self.gpu_id)
            tf_ids    = tf_ids.to(self.gpu_id)
            tg_ids    = tg_ids.to(self.gpu_id)
            motif_mask= motif_mask.to(self.gpu_id)

        # Optional feature scaling (id-aware)
        with record_function("feature_scaling"):
            if getattr(self, "tf_scaler", None) is not None:
                tf_tensor = self.tf_scaler.transform(tf_tensor, tf_ids)
            if getattr(self, "tg_scaler", None) is not None:
                targets  = self.tg_scaler.transform(targets, tg_ids)
            
        tf_tensor  = torch.nan_to_num(tf_tensor,  nan=0.0, posinf=1e6, neginf=-1e6)
        atac_wins  = torch.nan_to_num(atac_wins,  nan=0.0, posinf=1e6, neginf=-1e6)
        bias       = torch.nan_to_num(bias,       nan=0.0, posinf=5.0, neginf=-5.0)
        motif_mask = torch.nan_to_num(motif_mask, nan=0.0)

        for name, t in {
            "atac_wins": atac_wins, "tf_tensor": tf_tensor, "targets": targets,
            "bias": bias, "motif_mask": motif_mask
        }.items():
            if not torch.isfinite(t).all():
                bad = (~torch.isfinite(t)).nonzero(as_tuple=False)[:5]
                raise RuntimeError(f"{name} has non-finite values; examples idx={bad}")

        # ------------------------------------------------------------------
        # Forward pass
        # ------------------------------------------------------------------
        with record_function("forward_pass"):
            with autocast(device_type="cuda"):
                mask_arg = motif_mask if USE_MOTIF_MASK else None
                preds, attn, shortcut_contrib = self.model(
                    atac_wins,
                    tf_tensor,
                    tf_ids=tf_ids,
                    tg_ids=tg_ids,
                    bias=bias,
                    motif_mask=mask_arg,
                    return_shortcut_contrib=False,
                )

        # ---- loss & penalty in fp32 for stability ----
        with record_function("loss_computation"):
            preds32   = torch.nan_to_num(preds.float(),   nan=0.0, posinf=1e6, neginf=-1e6)
            targets32 = torch.nan_to_num(targets.float(), nan=0.0, posinf=1e6, neginf=-1e6)
            mse_loss = self.loss_fn(preds32, targets32)
            
            # ----- Unscaled MSE for logging (no grad) -----
            if getattr(self, "tg_scaler", None) is not None:
                with torch.no_grad():
                    targets_u = self.tg_scaler.inverse_transform(targets32, tg_ids)
                    preds_u   = self.tg_scaler.inverse_transform(preds32,   tg_ids)

                    targets_u = torch.nan_to_num(targets_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)
                    preds_u   = torch.nan_to_num(preds_u.float(),   nan=0.0, posinf=1e6, neginf=-1e6)

                    mse_loss_unscaled = F.mse_loss(preds_u, targets_u).detach()
            else:
                mse_loss_unscaled = mse_loss.detach()

            # ---------- Correlation penalty via per-gene Pearson r^2 ----------
            # shapes: [B, G]
            x = preds32
            y = targets32
            x = x - x.mean(dim=0, keepdim=True)
            y = y - y.mean(dim=0, keepdim=True)

            eps = 1e-8
            x_ss = (x * x).sum(dim=0)     # per gene
            y_ss = (y * y).sum(dim=0)     # per gene
            denom = (x_ss * y_ss).clamp_min(eps).sqrt()
            # valid genes need variance in both preds and targets
            valid = (x_ss > eps) & (y_ss > eps)

            corr = torch.zeros_like(denom)
            corr[valid] = (x[ :, valid] * y[ :, valid]).sum(dim=0) / denom[valid]
            corr = corr.clamp_(-1.0, 1.0)
            r2_g = corr * corr

            if valid.any():
                mean_r2 = r2_g[valid].mean()
            else:
                mean_r2 = torch.tensor(0.0, device=preds32.device, dtype=preds32.dtype)

            # --- r2 warmup ---
            corr_warm   = max(1, getattr(self, "corr_sq_warmup_epochs", 3))
            cur_epoch   = getattr(self, "epoch", 0)
            corr_anneal = float(min(1.0, (cur_epoch + 1) / corr_warm))
            r2_penalty  = (CORR_LOSS_WEIGHT * corr_anneal) * (1.0 - mean_r2)

            # --- shortcut regularization ---
            shortcut_reg = torch.tensor(0.0, device=self.gpu_id, dtype=torch.float32)
            if getattr(self.model, "use_shortcut", False) and hasattr(self.model, "shortcut_layer"):
                reg_val = self.model.shortcut_layer.regularization()
                shortcut_reg = reg_val.to(dtype=torch.float32, device=self.gpu_id) * SHORTCUT_REG_WEIGHT

            # ---------- Total loss ----------
            total_loss = (
                mse_loss
                + r2_penalty
                + shortcut_reg
            )

        # Safety: if something went numerically wrong, skip the step entirely.
        # Returning None ensures the caller can drop the batch without trying
        # to log detached scalars that don't exist.
        if not torch.isfinite(total_loss):
            logging.warning("Non-finite loss encountered; skipping batch.")
            return None

        # For logging: note we detach scalars we only log
        return (
            total_loss,
            mse_loss.detach(),
            mse_loss_unscaled,
            mean_r2.detach(),
            r2_penalty.detach(),
        )
        
    def _validate(self):
        self.model.eval()

        total_loss_scaled_t = torch.zeros(1, device=self.gpu_id)
        total_loss_unscaled_t = torch.zeros(1, device=self.gpu_id)
        n_batches = 0

        # global accumulators (scaled space)
        sse_s   = torch.zeros(1, device=self.gpu_id)
        sumy_s  = torch.zeros(1, device=self.gpu_id)
        sumy2_s = torch.zeros(1, device=self.gpu_id)
        n_s     = torch.zeros(1, device=self.gpu_id)

        # global accumulators (UNSCALED space)
        sse_u   = torch.zeros(1, device=self.gpu_id)
        sumy_u  = torch.zeros(1, device=self.gpu_id)
        sumy2_u = torch.zeros(1, device=self.gpu_id)
        n_u     = torch.zeros(1, device=self.gpu_id)

        with torch.no_grad():
            for batch in self.val_data:
                if self._should_stop():
                    raise KeyboardInterrupt()
                (atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask) = batch

                atac_wins  = atac_wins.to(self.gpu_id, non_blocking=True)
                tf_tensor  = tf_tensor.to(self.gpu_id, non_blocking=True)
                targets    = targets.to(self.gpu_id, non_blocking=True)
                bias       = bias.to(self.gpu_id, non_blocking=True)
                tf_ids     = tf_ids.to(self.gpu_id, non_blocking=True)
                tg_ids     = tg_ids.to(self.gpu_id, non_blocking=True)
                motif_mask = motif_mask.to(self.gpu_id, non_blocking=True)

                # scale inputs/targets as in training (scaled space)
                if getattr(self, "tf_scaler", None) is not None:
                    tf_tensor = self.tf_scaler.transform(tf_tensor, tf_ids)
                if getattr(self, "tg_scaler", None) is not None:
                    targets_s = self.tg_scaler.transform(targets, tg_ids)
                else:
                    targets_s = targets
                mask_arg = motif_mask if USE_MOTIF_MASK else None

                preds, _, _ = self.model(
                    atac_wins, tf_tensor,
                    tf_ids=tf_ids, tg_ids=tg_ids,
                    bias=bias, motif_mask=mask_arg,
                    return_shortcut_contrib=False,
                )

                # numeric safety before metrics
                preds_s   = torch.nan_to_num(preds.float(),   nan=0.0, posinf=1e6, neginf=-1e6)
                targets_s = torch.nan_to_num(targets_s.float(), nan=0.0, posinf=1e6, neginf=-1e6)

                # --- MSE in scaled space (status quo) ---
                loss_s = F.mse_loss(preds_s, targets_s)
                total_loss_scaled_t += loss_s.detach()
                n_batches += 1

                # accumulate for scaled R²
                y_s = targets_s.reshape(-1)
                p_s = preds_s.reshape(-1)
                sse_s   += torch.sum((y_s - p_s) ** 2)
                sumy_s  += torch.sum(y_s)
                sumy2_s += torch.sum(y_s ** 2)
                n_s     += y_s.numel()
            

                # ---------- Unscaled metrics ----------
                if getattr(self, "tg_scaler", None) is not None:
                    targets_u = self.tg_scaler.inverse_transform(targets_s, tg_ids)
                    preds_u   = self.tg_scaler.inverse_transform(preds_s,   tg_ids)
                else:
                    targets_u, preds_u = targets_s, preds_s

                targets_u = torch.nan_to_num(targets_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)
                preds_u   = torch.nan_to_num(preds_u.float(),   nan=0.0, posinf=1e6, neginf=-1e6)

                loss_u = F.mse_loss(preds_u, targets_u)
                total_loss_unscaled_t += loss_u.detach()
                
                y_u = targets_u.reshape(-1)
                p_u = preds_u.reshape(-1)
                sse_u   += torch.sum((y_u - p_u) ** 2)
                sumy_u  += torch.sum(y_u)
                sumy2_u += torch.sum(y_u ** 2)
                n_u     += y_u.numel()

        # Create tensor copies for reduction
        n_batches_t = torch.tensor(n_batches, device=self.gpu_id, dtype=torch.long)

        # DDP all-reduce: ensure every rank participates (including those with zero local batches)
        if dist.is_available() and dist.is_initialized():
            for t in (sse_s, sumy_s, sumy2_s, n_s, sse_u, sumy_u, sumy2_u, n_u, 
                      total_loss_scaled_t, total_loss_unscaled_t):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            # reduce the integer batch count too
            dist.all_reduce(n_batches_t, op=dist.ReduceOp.SUM)

        # Use global counts (if DDP reduced) or local if not initialized
        global_n_batches = int(n_batches_t.item()) if dist.is_available() and dist.is_initialized() else int(n_batches)

        if global_n_batches == 0 or n_s.item() == 0:
            # No validation data globally
            return 0.0, 0.0, 0.0, 0.0

        eps = 1e-12

        # scaled R²
        ybar_s = sumy_s / torch.clamp(n_s, min=1.0)
        sst_s  = sumy2_s - n_s * (ybar_s ** 2)
        r2_s   = torch.where(sst_s <= eps, torch.zeros_like(sst_s), 1.0 - sse_s / torch.clamp(sst_s, min=eps))

        # unscaled R²
        ybar_u = sumy_u / torch.clamp(n_u, min=1.0)
        sst_u  = sumy2_u - n_u * (ybar_u ** 2)
        r2_u   = torch.where(sst_u <= eps, torch.zeros_like(sst_u), 1.0 - sse_u / torch.clamp(sst_u, min=eps))

        avg_loss_scaled = float(total_loss_scaled_t.item()) / max(1, global_n_batches)
        avg_loss_unscaled = float(total_loss_unscaled_t.item()) / max(1, global_n_batches)


        # Return both: (scaled MSE, scaled R2, unscaled R2)
        return float(avg_loss_scaled), float(avg_loss_unscaled), float(r2_s.item()), float(r2_u.item())

    
    def _run_epoch(self, epoch):
        sampler = getattr(self.train_data, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        bs = getattr(self.train_data, "batch_sampler", None)
        if hasattr(bs, "set_epoch"):
            bs.set_epoch(epoch)

        total_loss_sum         = 0.0
        total_mse_scaled_sum   = 0.0
        total_mse_unscaled_sum = 0.0
        n_batches              = 0

        self.optimizer.zero_grad(set_to_none=True)
        progress_marks = [25, 50, 75]
        next_mark_idx  = 0

        for iteration, batch in enumerate(self.train_data):
            if self._should_stop():
                raise KeyboardInterrupt()

            out = self._run_batch(batch)

            # ---- Skip batches that were flagged as bad/NaN ----
            if out is None:
                self.optimizer.zero_grad(set_to_none=True)
                continue

            (total_loss_val,
            mse_scaled,
            mse_unscaled,
            mean_corr,
            corr_weight) = out

            if not total_loss_val.requires_grad:
                raise RuntimeError("Bug: total_loss_val has no grad_fn")

            loss_for_backprop = total_loss_val / self.grad_accum_steps
            
            # Backward pass with profiler annotation
            with record_function("backward_pass"):
                self.scaler.scale(loss_for_backprop).backward()

            if ((iteration + 1) % self.grad_accum_steps == 0
                or (iteration + 1) == len(self.train_data)):
                with record_function("optimizer_step"):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

            total_loss_sum          += float(total_loss_val.detach())
            total_mse_scaled_sum    += float(mse_scaled)
            total_mse_unscaled_sum  += float(mse_unscaled)
            n_batches += 1
            
            # ----- Step the profiler after each training iteration -----
            if self.profiler is not None:
                self.profiler.step()
            
            # ----- sparse progress logging -----
            if (
                self.is_main
                and len(self.train_data) > 0
                and next_mark_idx < len(progress_marks)
            ):
                pct = int(100 * (iteration + 1) / len(self.train_data))
                # log when we *cross* the next mark
                if pct >= progress_marks[next_mark_idx]:
                    logging.info(
                        f"    [{progress_marks[next_mark_idx]}%] Iter {iteration}"
                    )
                    next_mark_idx += 1
                            # f"MSE + pearson r2 penalty: {total_loss_val:.4f} | "
                            # f"MSE Loss: {mse_loss_scaled:.4f} | "
                            # f"pearson r2 penalty: {corr_weight:.4f} | "
                            # f"Mean pearson r2: {mean_corr:.2f}"     

        avg_train_loss          = total_loss_sum / max(1, n_batches)
        avg_train_mse_scaled    = total_mse_scaled_sum / max(1, n_batches)
        avg_train_mse_unscaled  = total_mse_unscaled_sum / max(1, n_batches)

        avg_val_mse_scaled, avg_val_mse_unscaled, r2_s, r2_u = self._validate()
        self.scheduler.step(avg_val_mse_unscaled)

        return (
            avg_train_loss,
            avg_train_mse_scaled,
            avg_train_mse_unscaled,
            avg_val_mse_scaled,
            avg_val_mse_unscaled,
            r2_s,
            r2_u,
        )


    def _save_checkpoint(self, epoch: int, path: str):
        # Only main rank writes (see Fix 2 below!)
        if not getattr(self, "is_main", self.gpu_id == 0):
            return

        # Make sure directory exists
        os.makedirs(path, exist_ok=True)

        # Get the unwrapped model for saving both state_dict and embeddings
        if hasattr(self.model, "module"):
            model_for_save = self.model.module
        else:
            model_for_save = self.model

        model_state = model_for_save.state_dict()

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_state,
        }

        if hasattr(self, "optimizer") and self.optimizer is not None:
            ckpt["optimizer_state_dict"] = self.optimizer.state_dict()
        if hasattr(self, "scheduler") and self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        if hasattr(self, "tf_scaler") and self.tf_scaler is not None:
            ckpt["tf_scaler_mean"] = self.tf_scaler.mean.detach().cpu()
            ckpt["tf_scaler_std"]  = self.tf_scaler.std.detach().cpu()
        if hasattr(self, "tg_scaler") and self.tg_scaler is not None:
            ckpt["tg_scaler_mean"] = self.tg_scaler.mean.detach().cpu()
            ckpt["tg_scaler_std"]  = self.tg_scaler.std.detach().cpu()

        # ---- Save embeddings from the model (not the state dict) ----
        save_tf_tg_embeddings_from_model(
            model_for_save,
            out_dir=path,
            vocab_dir=path,   # or training_output_dir if your vocabs live elsewhere
        )

        out_path = os.path.join(path, f"checkpoint_{epoch}.pt")
        torch.save(ckpt, out_path)
        logging.info(f"\tTraining checkpoint saved to {out_path}")
        
    def train(self, max_epochs: int, path: str, start_epoch: int = 0):
        best_r2 = float("-inf")
        patience_counter = 0
        history = []  # store per-epoch logs
        
        # Initialize profiler (only on rank 0 to avoid conflicts)
        if self.use_profiler and self.is_main:
            profiler_log_dir = os.path.join(self.profiler_dir or path, "profiler_logs")
            os.makedirs(profiler_log_dir, exist_ok=True)
            
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    wait=PROFILER_START_STEP,
                    warmup=1,
                    active=PROFILER_ACTIVE_STEPS,
                    repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_log_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.profiler.start()
            logging.info(f"TensorBoard profiler enabled. Logs: {profiler_log_dir}")
        
        
        try:
            total_train_start_time = time.time()
            for epoch in range(start_epoch, max_epochs):
                epoch_start_time = time.time()
                if self._should_stop():
                    raise KeyboardInterrupt()
                
                (avg_train_loss,
                 avg_train_mse_scaled, 
                 avg_train_mse_unscaled, 
                 avg_val_mse_scaled, 
                 avg_val_mse_unscaled, 
                 r2_s, 
                 r2_u) = self._run_epoch(epoch)
                epoch_end_time = time.time()
                
                epoch_dur_sec = epoch_end_time - epoch_start_time
                
                if self._should_stop():
                    raise KeyboardInterrupt()

                if self.is_main:
                    lr = self.optimizer.param_groups[0]['lr']
                    logging.info(
                        f"Epoch {epoch+1} | Train Total Loss: {avg_train_loss:.4f} | "
                        f"Train MSE: {avg_train_mse_unscaled:.4f} | "
                        f"Val MSE: {avg_val_mse_unscaled:.4f} | "
                        f"R2 (Unscaled): {r2_u:.3f} | "
                        f"R2 (Scaled): {r2_s:.3f} | "
                        f"LR: {lr:.2e} | "
                        f"Time: {epoch_dur_sec:.0f}s" 
                    )

                    epoch_log = {
                        "Epoch": epoch+1,
                        "Train Total Loss": avg_train_loss,
                        "Train MSE": avg_train_mse_unscaled,
                        "Val MSE": avg_val_mse_unscaled,
                        "R2_u": r2_u,
                        "R2_s": r2_s,
                        "LR": lr,
                        "Time": round(epoch_dur_sec, 0),
                    }
                    history.append(epoch_log)

                    self._write_log_csv(epoch_log, path)
                                    
                # Checkpoint + CSV log
                if epoch % self.save_every == 0:
                    if self.is_main:
                        self._save_checkpoint(epoch, path)
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()

                # Checkpoint + CSV log
                stop_tensor = torch.tensor(0, device=self.gpu_id)

                # --- Early stopping check (only rank 0 sets flag) ---
                if self.is_main:
                    if epoch > 5: # wait a few epochs before checking
                        improved = False

                        if avg_val_mse_unscaled < self.best_val_loss - self.min_delta:
                            self.best_val_loss = avg_val_mse_unscaled
                            improved = True
                        if r2_s > best_r2 + self.min_delta:
                            best_r2 = r2_s
                            improved = True

                        if improved:
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= self.patience:
                                logging.info("Early stopping triggered (no improvement).")
                                self._save_checkpoint(epoch, path)
                                stop_tensor.fill_(1)

                            else:
                                logging.info(f"    Loss did not improve {patience_counter}/{self.patience}")

                # --- Broadcast stop flag from rank 0 to all ranks ---
                dist.broadcast(stop_tensor, src=0)

                # --- All ranks see the same value now ---
                if stop_tensor.item() == 1:
                    if self.is_main:
                        logging.info("All ranks stopping training.")
                    break
            
            total_train_end_time = time.time()
            
            total_training_time_min = total_train_end_time - total_train_start_time
            
            # Final save if not early stopped
            if self.is_main and patience_counter < self.patience:
                logging.info("Training loop exited normally.")
                
                # Convert elapsed_seconds into hours, minutes, and seconds
                hours, remainder = divmod(total_training_time_min, 3600)  # 3600 seconds in an hour
                minutes, seconds = divmod(remainder, 60)         # 60 seconds in a minute
                logging.info(f"Total Training Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

        except KeyboardInterrupt:
            # graceful Ctrl+C
            epoch = locals().get("epoch", start_epoch)
            self._handle_abort(epoch, path, history, "KeyboardInterrupt")
            raise

        except RuntimeError as e:
            # catch CUDA OOM and save before dying
            if "out of memory" in str(e).lower():
                epoch = locals().get("epoch", start_epoch)
                self._handle_abort(epoch, path, history, "CUDA OOM")
            raise
        
        finally:
            # Stop profiler
            if self.profiler is not None:
                self.profiler.stop()
                logging.info("Profiler stopped")
        
    def _write_log_csv(self, history, path):
        fieldnames = ["Epoch", "Train Total Loss", "Train MSE",
                    "Val MSE", "R2_u", "R2_s", "LR", "Time"]
        log_path = os.path.join(path, "training_log.csv")

        file_exists = os.path.isfile(log_path)
        mode = "a" if file_exists else "w"

        with open(log_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            # --- allow a single dict OR a list of dicts ---
            if isinstance(history, dict):
                writer.writerow(history)          # single epoch
            else:
                writer.writerows(history)         # list of epochs

def load_checkpoint(checkpoint_path, trainer, device):
    rank = int(os.environ.get("RANK", -1))

    # Load checkpoint on rank 0 only
    if rank == 0:
        logging.info(f"[RANK {rank}] Starting torch.load('{checkpoint_path}')")
        ckpt = torch.load(checkpoint_path, map_location=device)
        logging.info(f"[RANK {rank}] Finished torch.load")
    else:
        ckpt = None

    # Broadcast checkpoint object
    if dist.is_available() and dist.is_initialized():
        ckpt_list = [ckpt]
        dist.broadcast_object_list(ckpt_list, src=0)
        ckpt = ckpt_list[0]

    model = trainer.model
    state_dict = ckpt["model_state_dict"]

    # ---- Load model state ----
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    # ---- Load optimizer state and move tensors to device ----
    if "optimizer_state_dict" in ckpt and hasattr(trainer, "optimizer"):
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Move optimizer state tensors to the correct device
        for state in trainer.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    # ---- Load AMP scaler (stays on CPU except its unscale tensor) ----
    if "scaler_state_dict" in ckpt and hasattr(trainer, "scaler"):
        try:
            trainer.scaler.load_state_dict(ckpt["scaler_state_dict"])

            # Fix unscale tensor device mismatch
            st = trainer.scaler._scale
            if torch.is_tensor(st):
                trainer.scaler._scale = st.to(device)
        except Exception:
            pass
        
    if "tf_scaler_mean" in ckpt and "tf_scaler_std" in ckpt:
        trainer.tf_scaler = SimpleScaler(
            ckpt["tf_scaler_mean"].to(device),
            ckpt["tf_scaler_std"].to(device),
        )
    if "tg_scaler_mean" in ckpt and "tg_scaler_std" in ckpt:
        trainer.tg_scaler = SimpleScaler(
            ckpt["tg_scaler_mean"].to(device),
            ckpt["tg_scaler_std"].to(device),
        )
    
    model.to(device)

    start_epoch = ckpt.get("epoch", 0) + 1
    return start_epoch

def detect_gpu_type():
    """
    Detect GPU type and return compatibility information.
    Returns: (gpu_name, compute_capability, is_a100_or_better)
    """
    if not torch.cuda.is_available():
        return "CPU", 0.0, False
    
    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    cc = float(f"{compute_capability[0]}.{compute_capability[1]}")
    
    # A100 and newer (8.0+), H100 (9.0), etc.
    is_a100_or_better = cc >= 8.0
    
    return gpu_name, cc, is_a100_or_better

def load_train_objs(run_cfg):
    # Dataset does not depend on d_model etc., but we keep it here
    dataset = MultiChromosomeDataset(
        data_dir=SAMPLE_DATA_CACHE_DIR,
        chrom_ids=CHROM_IDS,
        tf_vocab_path=os.path.join(COMMON_DATA, "tf_vocab.json"),
        tg_vocab_path=os.path.join(COMMON_DATA, "tg_vocab.json"),
        max_cached=2,
        subset_seed=SUBSAMPLE_SEED,
        allowed_samples=ALLOWED_SAMPLES
    )

    tf_vocab_size = int(dataset.tf_ids.numel())
    tg_vocab_size = int(dataset.tg_ids.numel())

    model = MultiomicTransformer(
        d_model=run_cfg["d_model"],
        num_heads=run_cfg["num_heads"],
        num_layers=run_cfg["num_layers"],
        d_ff=run_cfg["d_ff"],
        dropout=run_cfg["dropout"],
        tf_vocab_size=tf_vocab_size,
        tg_vocab_size=tg_vocab_size,
        bias_scale=ATTN_BIAS_SCALE,
        use_bias=run_cfg["use_dist_bias"],
        use_shortcut=run_cfg["use_shortcut"],
        use_motif_mask=run_cfg["use_motif_mask"],
        motif_mask_threshold=run_cfg["motif_mask_threshold"],
        motif_prior_scale=run_cfg["motif_prior_scale"],
        lambda_l1=run_cfg["shortcut_l1"],
        lambda_l2=run_cfg["shortcut_l2"],
        topk=run_cfg["shortcut_topk"],
        shortcut_dropout=run_cfg["shortcut_dropout"],
        use_gradient_checkpointing=run_cfg["use_grad_ckpt"]
    )
    
    # Compile model if requested (before DDP wrapping) - GPU-aware
    if USE_TORCH_COMPILE:
        gpu_name, cc, is_a100_or_better = detect_gpu_type()
        logging.info(f"GPU: {gpu_name}, Compute Capability: {cc}")
        
        if is_a100_or_better:
            try:
                model = torch.compile(
                    model,
                    mode="max-autotune",  # Optimal for A100+
                    fullgraph=False,
                )
                logging.info("Model compiled with max-autotune (A100+)")
            except Exception as e:
                logging.warning(f"torch.compile() failed: {e}. Continuing without compilation.")
        else:
            logging.info(f"Compute Capability {cc} detected - torch.compile not recommended. Skipping.")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=run_cfg["lr"],
    )

    return dataset, model, optimizer

def prepare_dataloader(dataset, batch_size, world_size=1, rank=0,
                       num_workers=4, pin_memory=True, seed=42, drop_last=True):
    """
    Build train/val/test loaders.

    For MultiChromosomeDataset:
      - Use ONE shared dataset instance.
      - For EACH chromosome:
          * split its indices into train/val/test subsets
      - For EACH split:
          * use an IndexedChromBucketBatchSampler over its per-chrom index subsets
          * -> every split sees all chromosomes (by indices),
             but each batch is still single-chromosome (shape-safe).

    For other datasets:
      - Fallback to legacy random_split + DistributedSampler.
    """
    import random
    import zlib
    g = torch.Generator()
    g.manual_seed(seed)

    # ---------- Multi-chromosome path ----------
    if isinstance(dataset, MultiChromosomeDataset):
        # 1) Build per-chrom index ranges from dataset._offsets
        chrom_to_indices = {}
        for i, chrom in enumerate(dataset.chrom_ids):
            start = dataset._offsets[i]
            end = dataset._offsets[i + 1] if i + 1 < len(dataset._offsets) else len(dataset)
            if end > start:
                chrom_to_indices[chrom] = list(range(start, end))

        # 2) For each chrom, split its indices into train/val/test
        train_map = {}
        val_map = {}
        test_map = {}

        for chrom, idxs in chrom_to_indices.items():
            n = len(idxs)
            if n == 0:
                continue

            # deterministic per-chrom shuffle
            chrom_hash = zlib.crc32(str(chrom).encode("utf-8")) & 0xFFFFFFFF
            rnd = random.Random(seed + chrom_hash % 10_000_000)
            idxs_shuf = idxs[:]
            rnd.shuffle(idxs_shuf)

            # 70% train, 15% val, 15% test
            n_train = int(0.70 * n)
            n_val   = int(0.15 * n)
            n_test  = n - n_train - n_val

            # ensure we don't drop everything for tiny chromosomes
            if n_val == 0 and n_train > 1:
                n_val += 1
                n_train -= 1
            if n_test == 0 and n_train > 1:
                n_test += 1
                n_train -= 1

            train_idx = idxs_shuf[:n_train]
            val_idx   = idxs_shuf[n_train:n_train + n_val]
            test_idx  = idxs_shuf[n_train + n_val:]

            if train_idx:
                train_map[chrom] = train_idx
            if val_idx:
                val_map[chrom] = val_idx
            if test_idx:
                test_map[chrom] = test_idx

        # Split 
        base_train_bs = IndexedChromBucketBatchSampler(
            train_map, batch_size=batch_size, shuffle=True, seed=seed
        )
        base_val_bs = IndexedChromBucketBatchSampler(
            val_map, batch_size=batch_size, shuffle=False, seed=seed
        )
        base_test_bs = IndexedChromBucketBatchSampler(
            test_map, batch_size=batch_size, shuffle=False, seed=seed
        )

        # Creates distributed batch samplers if needed
        if world_size > 1:
            train_bs = DistributedBatchSampler(base_train_bs, world_size, rank, drop_last=drop_last)
            val_bs   = DistributedBatchSampler(base_val_bs,   world_size, rank, drop_last=False)
            test_bs  = DistributedBatchSampler(base_test_bs,  world_size, rank, drop_last=False)
        else:
            train_bs, val_bs, test_bs = base_train_bs, base_val_bs, base_test_bs

        # 5) Single shared dataset; samplers decide which indices belong to which split
        train_loader = DataLoader(
            dataset,
            batch_sampler=train_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            dataset,
            batch_sampler=val_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            dataset,
            batch_sampler=test_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, test_loader

    # ---------- Single-chromosome / legacy path (unchanged) ----------
    n_total = len(dataset)
    n_train = int(n_total * 0.70)
    n_val   = int(n_total * 0.15)
    n_test  = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=g)

    train_sampler = val_sampler = test_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, drop_last=drop_last)
        val_sampler   = DistributedSampler(val_set,   num_replicas=world_size, rank=rank, drop_last=False)
        test_sampler  = DistributedSampler(test_set,  num_replicas=world_size, rank=rank, drop_last=False)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=MultiomicTransformerDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=MultiomicTransformerDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=MultiomicTransformerDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader

def write_run_parameters(dataset, out_dir, world_size):

    logging.info("\n===== MultiomicTransformerDataset Loaded =====")
    if ALLOWED_SAMPLES is not None:
        logging.info(f"Samples:             {ALLOWED_SAMPLES}")
    logging.info(f"Chromosome:          {CHROM_IDS}")
    logging.info(f"Genes:               {dataset.num_tgs}")
    logging.info(f"Windows (RE):        {dataset.num_windows}")
    logging.info(f"TFs:                 {dataset.num_tfs}")
    logging.info(f"Model Size:          {D_MODEL}")
    logging.info(f"Metacells:           {len(dataset.metacell_names)}")
    logging.info(f"Epochs:              {TOTAL_EPOCHS}")
    logging.info(f"Batch Size:          {BATCH_SIZE}")
    logging.info(f"GPUs:                {world_size}")
    logging.info(f"Grad Accum Steps:    {GRAD_ACCUM_STEPS}")   
    if USE_GRAD_ACCUMULATION:
        logging.info(f"Effctve Batch Size:  {BATCH_SIZE * GRAD_ACCUM_STEPS * world_size}")   
    else:
        logging.info(f"Effctve Batch Size:  {BATCH_SIZE * world_size}")   
    logging.info(f"Use Grad Accum?:     {USE_GRAD_ACCUMULATION}")   
    logging.info(f"Use Grad Chkpt?:     {USE_GRAD_CHECKPOINTING}") 
    logging.info(f"Model Dimension:     {D_MODEL}")
    logging.info(f"Attention Heads:     {NUM_HEADS}")
    logging.info(f"Attention Layers:    {NUM_LAYERS}")
    logging.info(f"Feedforward Layers:  {D_FF}")
    logging.info(f"Dropout:             {DROPOUT}")
    logging.info(f"TF-TG Shortcut?:     {USE_SHORTCUT}")
    logging.info(f"Dist bias?:          {USE_DISTANCE_BIAS}")
    logging.info(f"Motif Mask?:         {USE_MOTIF_MASK}")
    logging.info(f"Mask Thresh:         {MOTIF_MASK_THRESH}")
    logging.info(f"Mask Soft Scale:     {MOTIF_PRIOR_SCALE}")
    logging.info(f"Shortcut L1:         {SHORTCUT_L1}")
    logging.info(f"Shortcut L2:         {SHORTCUT_L2}")
    logging.info(f"Shortcut Dropout:    {SHORTCUT_DROPOUT}")
    logging.info(f"Shortcut Top K:      {SHORTCUT_TOPK}")
    logging.info("================================================")
    
    run_params = {
        "allowed_samples": ALLOWED_SAMPLES,
        "epochs": TOTAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "use_grad_accum": USE_GRAD_ACCUMULATION,
        "use_grad_ckpt": USE_GRAD_CHECKPOINTING,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "d_ff": D_FF,
        "dropout": DROPOUT,
        "use_shortcut": USE_SHORTCUT,
        "use_dist_bias": USE_DISTANCE_BIAS,
        "use_motif_mask": USE_MOTIF_MASK,
        "motif_mask_threshold": MOTIF_MASK_THRESH,
        "motif_prior_scale": MOTIF_PRIOR_SCALE,
        "shortcut_l1": SHORTCUT_L1,
        "shortcut_l2": SHORTCUT_L2,
        "shortcut_dropout": SHORTCUT_DROPOUT,
        "shortcut_topk": SHORTCUT_TOPK,
        "lr": INITIAL_LEARNING_RATE,
        "genes": dataset.num_tgs,
        "windows": dataset.num_windows,
        "tfs": dataset.num_tfs,
        "metacells": len(dataset.metacell_names),
    }

    path = os.path.join(out_dir, "run_parameters.json")
    with open(path, "w") as f:
        json.dump(run_params, f, indent=4)  # indent=4 for readability
    logging.info(f"Run parameters written to {path}")

def _mapping_to_ordered_list(name2id: dict):
    # convert {name: id} → [names] in id order
    return [k for k, _ in sorted(name2id.items(), key=lambda kv: kv[1])]

def save_test_loader_portable(
    out_dir: Path,
    test_loader,
):
    """
    Save everything needed to reconstruct the test DataLoader
    without pickling DataLoader / Dataset objects.

    Parameters
    ----------
    out_dir : Path
        Training output directory
    test_loader : DataLoader
        Test DataLoader with IndexedChromBucketBatchSampler
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract batch sampler (handle DistributedBatchSampler wrapper)
    batch_sampler = test_loader.batch_sampler
    if hasattr(batch_sampler, 'batch_sampler'):
        # Wrapped in DistributedBatchSampler
        batch_sampler = batch_sampler.batch_sampler
    
    # 1) Save test split indices (chrom -> list of indices)
    test_map = batch_sampler.chrom_to_indices
    split_path = out_dir / "test_split_indices.json"
    with open(split_path, "w") as f:
        json.dump(test_map, f)

    # 2) Save loader config
    loader_cfg = {
        "batch_size": test_loader.batch_size,
        "seed": batch_sampler.seed,
        "num_workers": test_loader.num_workers,
        "pin_memory": test_loader.pin_memory,
    }

    cfg_path = out_dir / "test_loader_config.json"
    with open(cfg_path, "w") as f:
        json.dump(loader_cfg, f, indent=4)

    logging.info(
        f"Saved portable test loader:\n"
        f"  - {split_path}\n"
        f"  - {cfg_path}"
    )

def write_experiment_settings_and_objects(training_output_dir: Path, dataset, test_loader, world_size: int):
    """
    Works for both MultiChromosomeDataset and single-chrom MultiomicTransformerDataset.
    Writes tf/tg vocab mappings and run parameters. Skips scaler unless present.
    """
    os.makedirs(training_output_dir, exist_ok=True)

    # Persist full vocab mappings (no subsampling)
    tf_map = getattr(dataset, "tf_name2id", None)
    tg_map = getattr(dataset, "tg_name2id", None)

    if tf_map is None or tg_map is None:
        raise RuntimeError("Dataset is missing TF/TG name→id mappings (tf_name2id/tg_name2id).")

    # Persist mappings (name→id)
    with open(os.path.join(training_output_dir, "tf_vocab.json"), "w") as f:
        json.dump(tf_map, f)
    with open(os.path.join(training_output_dir, "tg_vocab.json"), "w") as f:
        json.dump(tg_map, f)

    # Also persist ordered name lists (useful for plotting/inspection)
    tf_names_ordered = _mapping_to_ordered_list(tf_map)
    tg_names_ordered = _mapping_to_ordered_list(tg_map)
    
    with open(os.path.join(training_output_dir, "tf_names_ordered.json"), "w") as f:
        json.dump(tf_names_ordered, f)
        
    with open(os.path.join(training_output_dir, "tg_names_ordered.json"), "w") as f:
        json.dump(tg_names_ordered, f)
    
    # Persist test loader
    torch.save(test_loader, os.path.join(training_output_dir, "test_loader.pt"))
    
    # Saves the inidces and config needed to reconstruct the test loader without pickling
    save_test_loader_portable(
        out_dir=training_output_dir,
        test_loader=test_loader,
    )

    # Your existing run-parameter writer is fine to call here if it doesn’t assume single-chrom only
    write_run_parameters(dataset, training_output_dir, world_size)
    logging.info("Wrote experiment settings and objects to training output directory")
    
def main(rank: int, local_rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    
    # Early check to make sure the model dimension can be split evenly among the number of heads
    assert D_MODEL % NUM_HEADS == 0, f"{D_MODEL} not divisible by {NUM_HEADS}"
    
    ddp_setup(rank, world_size, local_rank)
    
    print(
        f"[HOST {os.environ.get('HOSTNAME','?')}] "
        f"RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}",
        flush=True,
    )
    
    setup_logging(rank)
    
    resume_ckpt = RESUME_CHECKPOINT_PATH
    
    try:
        if resume_ckpt and os.path.isfile(resume_ckpt):
            resume_ckpt = Path(resume_ckpt)
            training_output_dir = resume_ckpt.parent
            logging.info(f"\n=========== RESUMING FROM {resume_ckpt} ===========")

            # Load previous run parameters (if present)
            prev_params = load_run_params_from_json(training_output_dir)

            def g(key, default):
                return prev_params.get(key, default) if prev_params else default

            run_cfg = {
                "allowed_samples":      g("allowed_samples", ALLOWED_SAMPLES),
                "epochs":               g("epochs", TOTAL_EPOCHS),
                "batch_size":           g("batch_size", BATCH_SIZE),
                "grad_accum_steps":     g("grad_accum_steps", GRAD_ACCUM_STEPS),
                "d_model":              g("d_model", D_MODEL),
                "num_layers":           g("num_layers", NUM_LAYERS),
                "num_heads":            g("num_heads", NUM_HEADS),
                "d_ff":                 g("d_ff", D_FF),
                "dropout":              g("dropout", DROPOUT),
                "use_grad_ckpt":        g("use_grad_ckpt", USE_GRAD_CHECKPOINTING),
                "use_shortcut":         g("use_shortcut", USE_SHORTCUT),
                "use_dist_bias":        g("use_dist_bias", USE_DISTANCE_BIAS),
                "use_motif_mask":       g("use_motif_mask", USE_MOTIF_MASK),
                "motif_mask_threshold": g("motif_mask_threshold", MOTIF_MASK_THRESH),
                "motif_prior_scale":    g("motif_prior_scale", MOTIF_PRIOR_SCALE),
                "shortcut_l1":          g("shortcut_l1", SHORTCUT_L1),
                "shortcut_l2":          g("shortcut_l2", SHORTCUT_L2),
                "shortcut_topk":        g("shortcut_topk", SHORTCUT_TOPK),
                "shortcut_dropout":     g("shortcut_dropout", SHORTCUT_DROPOUT),
                "lr":                   g("lr", INITIAL_LEARNING_RATE),
            }

        else:
            # New experiment
            training_file_iter_format = "model_training_{:03d}"
            training_output_dir = unique_path(OUTPUT_DIR / CHROM_ID, training_file_iter_format)
            logging.info(f"\n=========== EXPERIMENT {training_output_dir.name.upper()} ===========")

            run_cfg = {
                "allowed_samples":      ALLOWED_SAMPLES,
                "epochs":               TOTAL_EPOCHS,
                "batch_size":           BATCH_SIZE,
                "grad_accum_steps":     GRAD_ACCUM_STEPS,
                "d_model":              D_MODEL,
                "num_layers":           NUM_LAYERS,
                "num_heads":            NUM_HEADS,
                "d_ff":                 D_FF,
                "dropout":              DROPOUT,
                "use_grad_ckpt":        USE_GRAD_CHECKPOINTING,
                "use_shortcut":         USE_SHORTCUT,
                "use_dist_bias":        USE_DISTANCE_BIAS,
                "use_motif_mask":       USE_MOTIF_MASK,
                "motif_mask_threshold": MOTIF_MASK_THRESH,
                "motif_prior_scale":    MOTIF_PRIOR_SCALE,
                "shortcut_l1":          SHORTCUT_L1,
                "shortcut_l2":          SHORTCUT_L2,
                "shortcut_topk":        SHORTCUT_TOPK,
                "shortcut_dropout":     SHORTCUT_DROPOUT,
                "lr":                   INITIAL_LEARNING_RATE,
            }

        os.makedirs(training_output_dir, exist_ok=True)

        dataset, model, optimizer = load_train_objs(run_cfg)
        
        # After dataset/model are created and you know these:
        T = int(dataset.tf_ids.numel())
        G = int(dataset.tg_ids.numel())

        if rank == 0:
            logging.info("Preparing dataloader")

        train_loader, val_loader, test_loader = prepare_dataloader(
            dataset,
            batch_size=run_cfg["batch_size"],
            world_size=world_size,
            rank=rank,
        )
        
        if rank == 0 and not (resume_ckpt and os.path.isfile(resume_ckpt)):
            write_experiment_settings_and_objects(training_output_dir, dataset, test_loader, world_size)
            logging.info("Wrote experiment settings and objects to training output directory")

        if rank == 0:
            logging.info("Creating Trainer")

        loss_fn = nn.MSELoss()
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            loss_fn,
            optimizer,
            gpu_id=local_rank,
            global_rank=rank,
            save_every=SAVE_EVERY_N_EPOCHS,
            patience=PATIENCE,
            grad_accum_steps=run_cfg["grad_accum_steps"],
            use_grad_accumulation=USE_GRAD_ACCUMULATION,
            use_profiler=USE_PROFILER,
            profiler_dir=str(training_output_dir),
        )

        rank_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        # ---- Resume or new scalers ----
        if resume_ckpt and os.path.isfile(resume_ckpt):
            start_epoch = load_checkpoint(str(resume_ckpt), trainer, rank_device)
            logging.info(f"Resuming training from epoch {start_epoch}")
        else:
            T = int(dataset.tf_ids.numel())
            G = int(dataset.tg_ids.numel())
            use_ddp_reduce = torch.distributed.is_initialized()

            tf_s, tg_s = fit_simple_scalers(
                train_loader,
                T_expected=T,
                G_expected=G,
                device_for_reduce=rank_device,
                use_ddp_reduce=use_ddp_reduce,
            )
            trainer.tf_scaler = SimpleScaler(tf_s.mean.to(rank_device), tf_s.std.to(rank_device))
            trainer.tg_scaler = SimpleScaler(tg_s.mean.to(rank_device), tg_s.std.to(rank_device))
            start_epoch = 0

        if rank == 0:
            logging.info("\n ----- TRAINING STARTED -----")

        trainer.train(
            max_epochs=TOTAL_EPOCHS,
            path=training_output_dir,
            start_epoch=start_epoch,
        )

        # ---------- Post-training: unwrap, save, etc. ----------
        model_for_eval = getattr(trainer.model, "module", trainer.model).to(rank_device)

        if rank == 0:
            model_for_eval.eval()
            # embeddings
            save_tf_tg_embeddings_from_model(
                model_for_eval,
                out_dir=training_output_dir,
                vocab_dir=training_output_dir,
            )
            # final checkpoint
            torch.save(
                {
                    "epoch": TOTAL_EPOCHS - 1,
                    "model_state_dict": model_for_eval.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "scheduler_state_dict": trainer.scheduler.state_dict(),
                    "best_val_loss": trainer.best_val_loss,
                    "tf_scaler_mean": trainer.tf_scaler.mean,
                    "tf_scaler_std": trainer.tf_scaler.std,
                    "tg_scaler_mean": trainer.tg_scaler.mean,
                    "tg_scaler_std": trainer.tg_scaler.std,
                },
                training_output_dir / "trained_model.pt",
            )
            logging.info("Saved final trained model")

        # ----- EWC fisher (rank 0 only) -----
        if rank == 0:
            ewc_bundle_path = training_output_dir / "ewc_bundle.pth"
            fisher = ewc_utils.compute_fisher_diag(
                model_for_eval, train_loader,
                device=rank_device,
                n_batches=100,
            )
            ewc_utils.save_ewc_bundle(ewc_bundle_path, model_for_eval, fisher)

            if rank == 0:
                logging.info("\nIterations complete")
                            
        state = {
            "epoch": TOTAL_EPOCHS - 1,
            "model_state_dict": model_for_eval.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
            "best_val_loss": trainer.best_val_loss,
            "tf_scaler_mean": trainer.tf_scaler.mean,
            "tf_scaler_std": trainer.tf_scaler.std,
            "tg_scaler_mean": trainer.tg_scaler.mean,
            "tg_scaler_std": trainer.tg_scaler.std,
        }
                
        # ----- Compute Gradient Attribution and TF Knockout -----
        gradient_attribution.run_gradient_attribution(
            training_output_dir,
            model_for_eval,
            test_loader,
            tg_scaler=trainer.tg_scaler,
            tf_scaler=trainer.tf_scaler,
            state=state,
            device=rank_device,
            use_amp=True,
            rank=rank,
            world_size=world_size,
            distributed=torch.distributed.is_initialized(),
            max_batches=None,
            use_dataloader=True,
        )
        
        tf_knockout.run_tf_knockout(
            training_output_dir,
            model_for_eval,
            test_loader,
            tg_scaler=trainer.tg_scaler,
            tf_scaler=trainer.tf_scaler,
            state=state,
            device=rank_device,
            use_amp=True,
            rank=rank,
            world_size=world_size,
            distributed=torch.distributed.is_initialized(),
            max_batches=None,
            use_dataloader=True,
        )
    
    finally:
        if dist.is_initialized():
            dist.barrier()
            if rank == 0:
                logging.info("\nDestroying process group")
            dist.destroy_process_group()
    
if __name__ == "__main__":
    # Parse command-line arguments and setup global variables
    args = parse_training_args()
    setup_training_globals(args)
    
    # Get distributed training environment variables
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    main(
        rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        save_every=SAVE_EVERY_N_EPOCHS,
        total_epochs=TOTAL_EPOCHS,
        batch_size=BATCH_SIZE,
    )
