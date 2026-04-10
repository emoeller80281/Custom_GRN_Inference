import json
import os
import random
import re
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
import sys
import torch.nn.functional as F
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from typing import Set, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
import seaborn as sns
import random
import zlib
import time
from cycler import cycler
import pickle

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import multiomic_transformer.utils.data_formatter as data_formatter
from multiomic_transformer.datasets.dataset_refactor import SimpleScaler
from multiomic_transformer.models.model_simplified import MultiomicTransformer
from multiomic_transformer.datasets.dataset_refactor import (
    MultiomicTransformerDataset,
    MultiChromosomeDataset,
    DistributedBatchSampler,
    fit_simple_scalers_fast_gpu,
    SimpleScaler,
    IndexedChromBucketBatchSampler,
    InterleavedChromBatchSampler,
)
from multiomic_transformer.scripts.multinode_train_simplified import Trainer
import multiomic_transformer.utils.auroc_refactored as auroc_utils

logging.basicConfig(level=logging.INFO, format='%(message)s')

COLOR_PALETTE = {
    "blue_light": "#18A6ED",
    "orange_light": "#EEA700",
    "red_light": "#EF767A",
    "green_light": "#7EE3BA",
    "purple_light": "#C798CC",
    "grey_light": "#BCBCBF",
    "blue_dark": "#2E70B9",
    "orange_dark": "#D18A3D",
    "red_dark": "#BC3E1A",
    "green_dark": "#32936F",
    "purple_dark": "#9D5ED4",
    "grey_dark": "#434B4E",
    }


plt.rcParams.update({

    # figure
    "figure.figsize": (6,4),
    "figure.dpi": 300,

    # fonts
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    # axes
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "grid.alpha": 0.25,

    # lines
    "lines.linewidth": 2,

    # legend
    "legend.frameon": False,
    "axes.prop_cycle": cycler(color=COLOR_PALETTE.values()),
})

order = ["LINGER", "SCENIC+", "CellOracle", "GRaNIE", "Pando", "TRIPOD", "FigR"]

def load_experiment_handler(
    tdf_settings_path: str | Path,
    experiment_dir: str | Path,
    model_num: int | None = None,
    silence_warnings: bool = False,
    ):
    """
    Utility function to load an existing ExperimentHandler from disk.
    """
    tdf = data_formatter.load_tdf(Path(tdf_settings_path))
        
    # Load the ExperimentHandler settings and state from disk
    exp_handler = ExperimentHandler(
        training_data_formatter=tdf,
        experiment_dir=experiment_dir,
        model_num=model_num,
        silence_warnings=silence_warnings,
    )
    
    exp_handler.load_handler()
    
    return exp_handler

class ExperimentHandler:
    def __init__(
        self, 
        training_data_formatter: data_formatter.TrainingDataFormatter,
        experiment_dir: None | str = None, 
        model_num: None | int = 1, 
        silence_warnings: bool = False
        ):
        
        self.tdf = training_data_formatter
        
        assert os.path.exists(experiment_dir), f"Experiment directory {experiment_dir} does not exist."
        
        # Set up experiment directory to store model checkpoints, results, and figures
        if experiment_dir is None:
            self.experiment_dir = self.tdf.project_dir / "experiments"
            
            if not os.path.exists(self.experiment_dir):
                logging.info(f"Experiment directory {self.experiment_dir} does not exist. Creating it.")
                self.experiment_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.experiment_dir = Path(experiment_dir)
            
        self.experiment_name = self.tdf.experiment_name
        self.model_num = model_num
        self.silence_warnings = silence_warnings
        
        # Either load an existing model training directory or set up a new one if no number is provided
        model_str = f"{model_num:03d}"
        self.model_training_dir = Path(f"{self.experiment_dir}/{self.experiment_name}/model_training_{model_str}")
        if not self.model_training_dir.exists():
            self._create_model_training_dir(allow_overwrite=False)
        
        # Set up blank file paths
        self.common_data_dir = self.tdf.file_paths["training_cache"]["common"]["dir"]
        self.data_cache_dir = self.tdf.file_paths["training_cache"]["dataset_dir"]
        self.chrom_ids = self.tdf.chrom_list
        self.tf_names = self.tdf.tf_names
        self.tg_names = self.tdf.tg_names
        self.num_windows = self.tdf.num_windows
        self.num_metacells = self.tdf.num_metacells
        
        # Default training parameters
        self.epochs = 250
        self.batch_size = 32
        self.grad_accum_steps = 1
        self.use_grad_ckpt = True
        self.d_model = 128
        self.num_heads = 4
        self.num_layers = 3
        self.d_ff = 512
        self.kernel_size = 64
        self.dropout = 0.1
        self.bias_scale = 2.0
        self.use_dist_bias = True
        self.initial_lr = 0.00025
        self.starting_epoch = None
        
        self.dataset = None
        
        # Model and training state will be loaded when load_trained_model is called
        self.model = None
        self.state = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.tg_scaler = None
        self.tf_scaler = None
                
        self.scheduler_reduction_factor = 0.5
        self.scheduler_patience_epochs = 5
        self.scheduler_cooldown_epochs = 2
        
        # Gradient Attribution dataframe will be loaded when load_gradient_attribution is called
        self.grn = None
        
        # Model forward pass predictions vs true values
        self.tg_prediction_df = None
        self.tg_true_df = None
        
        # Model evaluation metric results
        self.raw_results_df = None
        self.results_df = None
        self.per_tf_all_df = None
        self.per_tf_summary_df = None
        
        # Model evaluation metric results with ground truth
        self.auroc_auprc_scores = None
        self.gpu_mem_log_df = None
        self.batch_profile_log_df = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        
        self.color_palette = COLOR_PALETTE
        
        self.method_color_dict = {
            "Gradient Attribution": "#4195DF",
            "LINGER": "#7EE3BA",
            "SCENIC+": "#EF767A",
            "CellOracle": "#F9C60D",
            "Pando": "#EF9CFA",
            "TRIPOD": "#82EC32",
            "FigR": "#FDA7BB",
            "GRaNIE": "#F98637"
            }
        
    def save_handler(self):
        self.tdf.save_settings()
        
        settings = {
            "experiment_dir": str(self.experiment_dir),
            "experiment_name": self.experiment_name,
            "model_training_dir": str(self.model_training_dir),
            "common_data_dir": str(self.common_data_dir),
            "data_cache_dir": str(self.data_cache_dir),
            "chrom_ids": self.chrom_ids,
            "tf_names": self.tf_names,
            "tg_names": self.tg_names,
            "num_windows": self.num_windows,
            "num_metacells": self.num_metacells,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "use_grad_ckpt": self.use_grad_ckpt,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "bias_scale": self.bias_scale,
            "use_dist_bias": self.use_dist_bias,
            "initial_lr": self.initial_lr,
            "scheduler_reduction_factor": self.scheduler_reduction_factor,
            "scheduler_patience_epochs": self.scheduler_patience_epochs,
            "scheduler_cooldown_epochs": self.scheduler_cooldown_epochs,
        }
        
        # Save the settings to disk
        self.tdf.atomic_json_dump(settings, self.model_training_dir / "experiment_handler_save.json")
        
        if self.tf_scaler is not None:
            scaler_state = {
                "tf_scaler_mean": self.tf_scaler.mean.detach().cpu().tolist(),
                "tf_scaler_std": self.tf_scaler.std.detach().cpu().tolist(),
                "tg_scaler_mean": self.tg_scaler.mean.detach().cpu().tolist(),
                "tg_scaler_std": self.tg_scaler.std.detach().cpu().tolist(),
            }
            scaler_state_path = self.model_training_dir / "scaler_state.pt"
            torch.save(scaler_state, scaler_state_path)
        
        # Save the loaders to disk
        if self.train_loader is not None:
            torch.save(self.train_loader, self.model_training_dir / "train_loader.pt")
        if self.val_loader is not None:
            torch.save(self.val_loader, self.model_training_dir / "val_loader.pt")
        if self.test_loader is not None:
            torch.save(self.test_loader, self.model_training_dir / "test_loader.pt")
    
    def load_handler(self):
        # Load the settings from disk
        settings_path = self.model_training_dir / "experiment_handler_save.json"
        if not settings_path.exists():
            logging.warning(f"Settings file {settings_path} does not exist. Cannot load ExperimentHandler state.")
            return
        
        with open(settings_path, "r") as f:
            logging.info(f"Loading ExperimentHandler state from {settings_path}...")
            settings = json.load(f)
        
        # Load the settings into the ExperimentHandler instance
        for key, value in settings.items():
            setattr(self, key, value)
        
        self.experiment_dir = Path(self.experiment_dir)
        self.model_training_dir = Path(self.model_training_dir)
        self.common_data_dir = Path(self.common_data_dir)
        self.data_cache_dir = Path(self.data_cache_dir)
        
        self.train_loader = torch.load(self.model_training_dir / "train_loader.pt", weights_only=False)
        self.val_loader = torch.load(self.model_training_dir / "val_loader.pt", weights_only=False)
        self.test_loader = torch.load(self.model_training_dir / "test_loader.pt", weights_only=False)
        
        scaler_state_path = self.model_training_dir / "scaler_state.pt"
        if scaler_state_path.exists():
            scaler_state = torch.load(scaler_state_path)
        
            # Rebuild the scalers from the training parameters
            self.tg_scaler = SimpleScaler(
                mean=torch.as_tensor(scaler_state["tg_scaler_mean"], device=self.device, dtype=torch.float32),
                std=torch.as_tensor(scaler_state["tg_scaler_std"],  device=self.device, dtype=torch.float32),
            )
            self.tf_scaler = SimpleScaler(
                mean=torch.as_tensor(scaler_state["tf_scaler_mean"], device=self.device, dtype=torch.float32),
                std=torch.as_tensor(scaler_state["tf_scaler_std"],  device=self.device, dtype=torch.float32),
            )
            
        if (self.model_training_dir / "inferred_grn.csv").is_file():
            self.grn = self.load_grn()
        
        # Load the model from disk
        if (self.model_training_dir / "trained_model.pt").exists():
            self.load_model()
        
        if (self.model_training_dir / "epoch_log.csv").is_file():
            self.epoch_log_df = pd.read_csv(self.model_training_dir / "epoch_log.csv")
        
        if (self.model_training_dir / "gpu_memory_log.csv").is_file():
            self.gpu_mem_log_df = pd.read_csv(self.model_training_dir / "gpu_memory_log.csv")
        
        if (self.model_training_dir / "batch_profile.log.csv").is_file():
            self.batch_profile_log_df = pd.read_csv(self.model_training_dir / "batch_profile.log.csv")

    def _setup_cuda_ddp(self, local_rank, rank, world_size):
        use_cuda = torch.cuda.is_available()
        self.use_ddp = world_size > 1

        if use_cuda:
            torch.cuda.set_device(local_rank)
            torch.backends.cuda.enable_flash_sdp(True)
            device = torch.device(f"cuda:{local_rank}")
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
            
        self.device = device
        
        # Only initialize distributed when actually using multi-process training
        if self.use_ddp and not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )
    
    def create_multichrom_dataset(self, max_cached=None):
        self.dataset = MultiChromosomeDataset(
            data_dir=self.data_cache_dir,
            chrom_ids=self.chrom_ids,
            tf_vocab_path=os.path.join(self.common_data_dir, "tf_vocab.json"),
            tg_vocab_path=os.path.join(self.common_data_dir, "tg_vocab.json"),
            max_cached=len(self.chrom_ids) if max_cached is None else max_cached,
            subset_seed=42,
        )
        
        return self.dataset
    
    def create_new_model(
        self, 
        use_dist_bias=None,
        bias_scale=None, 
        d_model=None,
        num_heads=None,
        num_layers=None,
        d_ff=None,
        dropout=None,
        kernel_size=None,
        local_rank=0, 
        rank=0, 
        world_size=1, 
        ):
        
        self._setup_cuda_ddp(local_rank, rank, world_size)

        self.d_model = int(self.d_model) if d_model is None else d_model
        self.num_heads = int(self.num_heads) if num_heads is None else num_heads
        self.num_layers = int(self.num_layers) if num_layers is None else num_layers
        self.d_ff = int(self.d_ff) if d_ff is None else d_ff
        self.dropout = float(self.dropout) if dropout is None else dropout
        self.bias_scale = float(self.bias_scale) if bias_scale is None else bias_scale
        self.use_dist_bias = bool(self.use_dist_bias) if use_dist_bias is None else use_dist_bias
        self.kernel_size = int(self.kernel_size) if kernel_size is None else kernel_size

        tf_vocab_size = int(self.dataset.tf_ids.numel())
        tg_vocab_size = int(self.dataset.tg_ids.numel())

        self.model = MultiomicTransformer(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            tf_vocab_size=tf_vocab_size,
            tg_vocab_size=tg_vocab_size,
            use_bias=self.use_dist_bias,
            bias_scale=self.bias_scale,
            window_pool_size=self.kernel_size,
        ).to(self.device)   

        return self.model
    
    def load_model(self, model_name: str = "trained_model.pt"):
        model_path = self.model_training_dir / model_name
        if not model_path.exists():
            logging.warning(f"WARNING: Trained model file {model_path} does not exist.")
            return None
                
        state_dict = torch.load(model_path, map_location=self.device)
        model_params = self.tdf.load_json(self.model_training_dir / "model_params.json")
        training_state = self.tdf.load_json(self.model_training_dir / "training_state.json")
        
        # Set the starting epoch and initial learning rate for continuing training
        self.starting_epoch = training_state.get("last_epoch", 0)
        self.initial_lr = training_state.get("last_lr", self.initial_lr)
        
        # Create a new model with the model parameters
        self.model = MultiomicTransformer(**model_params).to(self.device)
        
        # Load the saved model weights
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        
        return self.model
    
    def create_scalers(self, dataloader, max_batches: int | None = None):
        T = int(self.dataset.tf_ids.numel())
        G = int(self.dataset.tg_ids.numel())
        use_ddp_reduce = dist.is_initialized()

        tf_s, tg_s = fit_simple_scalers_fast_gpu(
            dataloader,
            T_expected=T,
            G_expected=G,
            device=self.device,
            use_ddp_reduce=use_ddp_reduce,
            max_batches=max_batches,
        )

        tf_scaler = SimpleScaler(tf_s.mean.to(self.device), tf_s.std.to(self.device))
        tg_scaler = SimpleScaler(tg_s.mean.to(self.device), tg_s.std.to(self.device))
                
        self.tf_scaler = tf_scaler
        self.tg_scaler = tg_scaler

    def prepare_dataloader(self, batch_size=None, world_size=1, rank=0,
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
        g = torch.Generator()
        g.manual_seed(seed)
        
        self.batch_size = self.batch_size if batch_size is None else batch_size
        
        dataset = self.dataset

        # ---------- Multi-chromosome path ----------
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
        base_train_bs = InterleavedChromBatchSampler(
            train_map, batch_size=self.batch_size, shuffle=True, seed=seed, drop_last=False
        )
        base_val_bs = InterleavedChromBatchSampler(
            val_map, batch_size=self.batch_size, shuffle=False, seed=seed, drop_last=False
        )
        base_test_bs = InterleavedChromBatchSampler(
            test_map, batch_size=self.batch_size, shuffle=False, seed=seed, drop_last=False
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
            persistent_workers=False,
        )
        val_loader = DataLoader(
            dataset,
            batch_sampler=val_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
        )
        test_loader = DataLoader(
            dataset,
            batch_sampler=test_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        return train_loader, val_loader, test_loader

    def train(
        self,
        train_loader,
        val_loader=None,
        num_epochs: int = 10,
        learning_rate: float | None = None,
        verbose: bool = True,
        improvement_patience: int = 10,
        max_batches: int | None = None,
        use_amp: bool = True,
        grad_accum_steps=4,
        save_every_n_epochs: int = 5,
        monitor_gpu_memory: bool = False,
        profile_batches: bool = False,
        silence_tqdm: bool = False,
        allow_overwrite: bool = False,
    ):
        self.initial_lr = self.initial_lr if learning_rate is None else learning_rate
        self.starting_epoch = self.starting_epoch if self.starting_epoch is not None else 0
        self.epochs = num_epochs
        self.grad_accum_steps = grad_accum_steps
        
        model = self.model.to(self.device)
        
        if "trained_model.pt" in os.listdir(self.model_training_dir) and not allow_overwrite:
            self._create_model_training_dir(allow_overwrite=False)
            logging.info(f"Creating model in new training directory: {self.model_training_dir}")
            
        self.save_handler()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.initial_lr)
        loss_fn = nn.MSELoss()

        tf_scaler = getattr(self, "tf_scaler", None)
        tg_scaler = getattr(self, "tg_scaler", None)

        amp_enabled = use_amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_reduction_factor,
            patience=self.scheduler_patience_epochs,
            cooldown=self.scheduler_cooldown_epochs, 
            threshold=1e-4,
            min_lr=1e-7,
        )

        gpu_mem_log = []
        batch_profile_log = []
        epoch_log = []

        final_epoch = self.starting_epoch + num_epochs
        epoch_iter = range(self.starting_epoch, final_epoch)

        if not verbose and not silence_tqdm:
            epoch_iter = tqdm(epoch_iter, desc="Training", ncols=100, unit="epoch", total=num_epochs)

        best_loss = float("inf")
        best_r2 = float("-inf")
        no_improvement_epochs = 0
        
        if verbose:
            logging.info(f"\n===== Model {self.model_num} Training Started =====")
        for epoch in epoch_iter:
            epoch_start_time = time.time()
            model.train()
            total_loss = 0.0
            n_batches = 0
            
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            optimizer.zero_grad(set_to_none=True)

            data_iter = iter(train_loader)
            i = 0
            
            if self.device.type == "cuda":
                peak_alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
                peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
            else:
                peak_alloc_mb = None
                peak_reserved_mb = None

            try:
                total_batches = max_batches if max_batches is not None else len(train_loader)
            except TypeError:
                total_batches = max_batches

            if verbose and not silence_tqdm:
                pbar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{final_epoch}", ncols=100, leave=False)
            else:
                pbar = None

            while True:
                if max_batches is not None and i >= max_batches:
                    break

                t0 = time.perf_counter()
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                t1 = time.perf_counter()

                atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, _ = batch

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t2 = time.perf_counter()

                atac_wins = atac_wins.to(self.device, non_blocking=True)
                tf_tensor = tf_tensor.to(self.device, non_blocking=True)
                targets   = targets.to(self.device, non_blocking=True)
                bias      = bias.to(self.device, non_blocking=True) if bias is not None else None
                tf_ids    = tf_ids.to(self.device, non_blocking=True)
                tg_ids    = tg_ids.to(self.device, non_blocking=True)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t3 = time.perf_counter()

                if tf_scaler is not None or tg_scaler is not None:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t4 = time.perf_counter()

                    if tf_scaler is not None:
                        tf_tensor = tf_scaler.transform(tf_tensor, tf_ids)
                    if tg_scaler is not None:
                        targets = tg_scaler.transform(targets, tg_ids)

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t5 = time.perf_counter()
                else:
                    t4 = t5 = time.perf_counter()

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t6 = time.perf_counter()

                with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                    preds = model(
                        atac_wins,
                        tf_tensor,
                        tf_ids=tf_ids,
                        tg_ids=tg_ids,
                        bias=bias,
                    )
                    loss = loss_fn(preds, targets)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t7 = time.perf_counter()

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t8 = time.perf_counter()

                loss_for_backward = loss / grad_accum_steps
                scaler.scale(loss_for_backward).backward()

                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t9 = time.perf_counter()

                stepped = False
                if (i + 1) % grad_accum_steps == 0:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t10 = time.perf_counter()

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    stepped = True

                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t11 = time.perf_counter()
                else:
                    t10 = t11 = time.perf_counter()

                if profile_batches:
                    batch_profile_log.append({
                        "epoch": epoch,
                        "step": i,
                        "loader_s": t1 - t0,
                        "transfer_s": t3 - t2,
                        "scaler_s": t5 - t4,
                        "forward_s": t7 - t6,
                        "backward_s": t9 - t8,
                        "optim_s": t11 - t10 if stepped else 0.0,
                        "total_step_s": (t11 if stepped else t9) - t0,
                        "loss": float(loss.detach().item()),
                        "batch_size": int(atac_wins.shape[0]),
                        "num_windows": int(atac_wins.shape[1]),
                        "num_tfs": int(tf_tensor.shape[1]),
                        "num_tgs": int(targets.shape[1]),
                    })

                if monitor_gpu_memory and self.device.type == "cuda":
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    allocated_mb = torch.cuda.memory_allocated() / 1024**2
                    reserved_mb = torch.cuda.memory_reserved() / 1024**2
                    free_mb = free_bytes / 1024**2
                    total_mb = total_bytes / 1024**2

                    gpu_mem_log.append({
                        "epoch": epoch,
                        "step": i,
                        "allocated_mb": allocated_mb,
                        "reserved_mb": reserved_mb,
                        "free_mb": free_mb,
                        "total_memory_mb": total_mb,
                        "allocated_pct_total": 100 * allocated_mb / total_mb,
                        "reserved_pct_total": 100 * reserved_mb / total_mb,
                        "free_pct_total": 100 * free_mb / total_mb,
                    })

                total_loss += loss.detach().item()
                n_batches += 1
                i += 1
                
                if pbar is not None:
                    pbar.update(1)
            
            if pbar is not None:
                pbar.close()

            if n_batches % grad_accum_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            avg_loss = total_loss / max(1, n_batches)
            epoch_end_time = time.time()

            avg_val_mse_scaled = None
            avg_val_mse_unscaled = None
            r2_s = None
            r2_u = None

            avg_val_mse_scaled, avg_val_mse_unscaled, r2_s, r2_u = self._validate_simple(
                model=model,
                val_loader=val_loader,
                device=self.device,
                amp_enabled=amp_enabled,
            )
            scheduler.step(avg_val_mse_unscaled)

            current_lr = optimizer.param_groups[0]["lr"]
            
            if self.device.type == "cuda":
                peak_alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
                peak_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
            else:
                peak_alloc_mb = None
                peak_reserved_mb = None
                
            if avg_val_mse_unscaled < best_loss:
                best_loss = avg_val_mse_unscaled
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
            
            chkpt_saved = False
            if epoch % save_every_n_epochs == 0:
                chkpt_name = f"checkpoint_{epoch:03d}.pt"
                self.save_model(epoch_log=epoch_log, model_name=chkpt_name, verbose=False)
                chkpt_saved = True
                
            epoch_log.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_mse_unscaled": avg_val_mse_unscaled,
                "r2_unscaled": r2_u,
                "r2_scaled": r2_s,
                "lr": current_lr,
                "epoch_time_s": epoch_end_time - epoch_start_time,
                "peak_allocated_mb": peak_alloc_mb,
                "peak_reserved_mb": peak_reserved_mb,
            })

            chkpt_saved_str = "Checkpoint Saved" if chkpt_saved else ""
            
            if verbose:
                logging.info(
                    f"Epoch {epoch}/{final_epoch} | "
                    f"Train Loss: {avg_loss:.4f} | "
                    f"Val MSE: {avg_val_mse_unscaled:.4f} | "
                    f"R2 (Unscaled): {r2_u:.3f} | "
                    f"R2 (Scaled): {r2_s:.3f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_end_time - epoch_start_time:.1f}s | "
                    f"Last Improved: {no_improvement_epochs} epochs ago | "
                    f"{chkpt_saved_str}"
                )

            if no_improvement_epochs >= improvement_patience:
                if verbose:
                    logging.info(f"No improvement in validation loss for {no_improvement_epochs} epochs. Stopping early.")
                break
            
        self.gpu_mem_log_df = pd.DataFrame(gpu_mem_log)
        self.batch_profile_df = pd.DataFrame(batch_profile_log)
        self.epoch_log_df = pd.DataFrame(epoch_log)
        
        # If resuming from a previous training run, concatenate the new logs with the old ones so we have a complete history
        if self.starting_epoch != 0:
            def _concat_logs(new_df, log_filename):
                previous_log_path = self.model_training_dir / log_filename
                if previous_log_path.exists():
                    previous_log_df = pd.read_csv(previous_log_path)
                    return pd.concat([previous_log_df, new_df], ignore_index=True)
                else:
                    return new_df
            self.epoch_log_df = _concat_logs(self.epoch_log_df, "epoch_log.csv")
            self.gpu_mem_log_df = _concat_logs(self.gpu_mem_log_df, "gpu_memory_log.csv")
            self.batch_profile_df = _concat_logs(self.batch_profile_df, "batch_profile_log.csv")
        
        self.gpu_mem_log_df.to_csv(self.model_training_dir / "gpu_memory_log.csv", index=False)
        self.batch_profile_df.to_csv(self.model_training_dir / "batch_profile_log.csv", index=False)
        self.epoch_log_df.to_csv(self.model_training_dir / "epoch_log.csv", index=False)
        
        logging.info(f"\nTraining Complete. Saving final model")
        self.save_model(epoch_log=epoch_log, model_name="trained_model.pt")

        return model
    
    def save_model(self, epoch_log, model_name: str = "trained_model.pt", verbose: bool = True):
        if self.model is None:
            logging.warning("WARNING: No model to save.")
            return
        
        model_path = self.model_training_dir / model_name
        torch.save(self.model.state_dict(), model_path)
        
        model_params = {
            "d_model": self.model.d_model,
            "num_heads": self.model.num_heads,
            "num_layers": self.model.num_layers,
            "d_ff": self.model.d_ff,
            "dropout": self.model.dropout,
            "tf_vocab_size": self.model.tf_vocab_size,
            "tg_vocab_size": self.model.tg_vocab_size,
            "use_bias": self.model.use_bias,
            "bias_scale": self.model.bias_scale,
            "window_pool_size": self.model.window_pool_size,
        }
                
        training_state = {
            "last_epoch": epoch_log[-1]["epoch"] if epoch_log else None,
            "last_lr": epoch_log[-1]["lr"] if epoch_log else None
        }

        self.tdf.atomic_json_dump(training_state, self.model_training_dir / "training_state.json")
        self.tdf.atomic_json_dump(model_params, self.model_training_dir / "model_params.json")
        
        if verbose:
            logging.info(f"Model saved to {model_path}")
    
    def _validate_simple(self, model, val_loader, device, amp_enabled):
        model.eval()

        tf_scaler = getattr(self, "tf_scaler", None)
        tg_scaler = getattr(self, "tg_scaler", None)

        total_loss_scaled = 0.0
        total_loss_unscaled = 0.0
        n_batches = 0

        sse_s = 0.0
        sumy_s = 0.0
        sumy2_s = 0.0
        n_s = 0

        sse_u = 0.0
        sumy_u = 0.0
        sumy2_u = 0.0
        n_u = 0

        with torch.no_grad():
            for batch in val_loader:
                atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, _ = batch

                atac_wins = atac_wins.to(device, non_blocking=True)
                tf_tensor = tf_tensor.to(device, non_blocking=True)
                targets   = targets.to(device, non_blocking=True)
                bias      = bias.to(device, non_blocking=True) if bias is not None else None
                tf_ids    = tf_ids.to(device, non_blocking=True)
                tg_ids    = tg_ids.to(device, non_blocking=True)

                if tf_scaler is not None:
                    tf_tensor = tf_scaler.transform(tf_tensor, tf_ids)

                if tg_scaler is not None:
                    targets_s = tg_scaler.transform(targets, tg_ids)
                else:
                    targets_s = targets

                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    preds_s = model(
                        atac_wins,
                        tf_tensor,
                        tf_ids=tf_ids,
                        tg_ids=tg_ids,
                        bias=bias,
                    )

                preds_s = torch.nan_to_num(preds_s.float(), nan=0.0, posinf=1e6, neginf=-1e6)
                targets_s = torch.nan_to_num(targets_s.float(), nan=0.0, posinf=1e6, neginf=-1e6)

                loss_s = F.mse_loss(preds_s, targets_s)
                total_loss_scaled += float(loss_s.item())

                y_s = targets_s.reshape(-1)
                p_s = preds_s.reshape(-1)
                sse_s += float(torch.sum((y_s - p_s) ** 2).item())
                sumy_s += float(torch.sum(y_s).item())
                sumy2_s += float(torch.sum(y_s ** 2).item())
                n_s += y_s.numel()

                if tg_scaler is not None:
                    targets_u = tg_scaler.inverse_transform(targets_s, tg_ids)
                    preds_u = tg_scaler.inverse_transform(preds_s, tg_ids)
                else:
                    targets_u, preds_u = targets_s, preds_s

                targets_u = torch.nan_to_num(targets_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)
                preds_u = torch.nan_to_num(preds_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)

                loss_u = F.mse_loss(preds_u, targets_u)
                total_loss_unscaled += float(loss_u.item())

                y_u = targets_u.reshape(-1)
                p_u = preds_u.reshape(-1)
                sse_u += float(torch.sum((y_u - p_u) ** 2).item())
                sumy_u += float(torch.sum(y_u).item())
                sumy2_u += float(torch.sum(y_u ** 2).item())
                n_u += y_u.numel()

                n_batches += 1

        if n_batches == 0 or n_s == 0 or n_u == 0:
            return 0.0, 0.0, 0.0, 0.0

        eps = 1e-12

        ybar_s = sumy_s / max(n_s, 1)
        sst_s = sumy2_s - n_s * (ybar_s ** 2)
        r2_s = 0.0 if sst_s <= eps else 1.0 - (sse_s / max(sst_s, eps))

        ybar_u = sumy_u / max(n_u, 1)
        sst_u = sumy2_u - n_u * (ybar_u ** 2)
        r2_u = 0.0 if sst_u <= eps else 1.0 - (sse_u / max(sst_u, eps))

        avg_loss_scaled = total_loss_scaled / max(1, n_batches)
        avg_loss_unscaled = total_loss_unscaled / max(1, n_batches)

        return avg_loss_scaled, avg_loss_unscaled, r2_s, r2_u
    
    def run_gradient_attribution(
        self,
        test_loader,
        model=None,
        tf_scaler=None,
        tg_scaler=None,
        use_amp=True,
        max_batches: int | None = None,
        save_every_n_batches: int = 20,
        max_tgs_per_batch: int | None = None,
        chunk_size = 64,
    ):

        model = self.model if model is None else model
        max_batches = len(test_loader) if max_batches is None else max_batches
        
        tf_names = self.tf_names
        tg_names = self.tg_names
        device = self.device
        
        max_tgs_per_batch = len(tg_names) if max_tgs_per_batch is None else max_tgs_per_batch
        max_tgs_per_batch = min(max_tgs_per_batch, len(tg_names))

        T_total = len(tf_names)
        G_total = len(tg_names)
        
        # Creates empty tensors to accumulate gradients across batches. The shape is [TF total, Genes total]
        grad_sum = torch.zeros(T_total, G_total, device=device, dtype=torch.float32)
        grad_count = torch.zeros_like(grad_sum)
        
        tf_scaler = self.tf_scaler if tf_scaler is None else tf_scaler
        tg_scaler = self.tg_scaler if tg_scaler is None else tg_scaler
        
        model.to(device).eval()

        iterator = tqdm(
            test_loader,
            desc=f"Gradient attributions",
            unit="batches",
            total=max_batches,
            ncols=100,
        )

        batch_grad_dfs = {}
        for b_idx, batch in enumerate(iterator):
            if max_batches is not None and b_idx >= max_batches:
                break

            atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
            
            atac_wins = atac_wins.to(device)
            tf_tensor = tf_tensor.to(device)
            bias = bias.to(device) if bias is not None else None
            tf_ids = tf_ids.to(device)
            tg_ids = tg_ids.to(device)
            motif_mask = motif_mask.to(device) if motif_mask is not None else None

            # Shapes
            if tf_tensor.dim() == 2:
                B, T_eval = tf_tensor.shape
                F_dim = 1
            else:
                B, T_eval, F_dim = tf_tensor.shape
                
            if bias is not None:
                if bias.dim() == 2:
                    # [G, W] -> [1, G, W]
                    bias = bias.unsqueeze(0)

            # Flatten TF IDs over batch for aggregation later
            if tf_ids.dim() == 1:  # [T_eval]
                tf_ids_flat = tf_ids.view(1, T_eval).expand(B, T_eval).reshape(-1)
            else:                  # [B, T_eval]
                tf_ids_flat = tf_ids.reshape(-1)

            G_eval = tg_ids.shape[-1]

            # Assign TGs to this rank and optionally chunk them to control memory.
            if G_eval > max_tgs_per_batch:
                perm = torch.randperm(G_eval, device=device)[:max_tgs_per_batch]
                owned_tg_indices = perm.sort().values
            else:
                owned_tg_indices = torch.arange(G_eval, device=device)

            # ---------- METHOD 1: plain saliency (grad * input) ----------
            total_owned = owned_tg_indices.numel()

            for chunk_start in range(0, total_owned, chunk_size):
                tg_chunk = owned_tg_indices[chunk_start : chunk_start + chunk_size]

                if bias is not None:
                    if bias.dim() == 3:
                        bias_chunk = bias[:, tg_chunk, :]
                    elif bias.dim() == 4:
                        bias_chunk = bias[:, :, tg_chunk, :]
                    else:
                        raise ValueError(f"Unexpected bias shape: {tuple(bias.shape)}")
                else:
                    bias_chunk = None

                if tg_ids.dim() == 1:
                    tg_ids_chunk = tg_ids[tg_chunk]
                else:
                    tg_ids_chunk = tg_ids[:, tg_chunk]

                tf_tensor_chunk = tf_tensor.detach().clone().requires_grad_(True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    tf_scaled = tf_scaler.transform(tf_tensor_chunk, tf_ids)
                    preds_s = model(
                        atac_wins,
                        tf_scaled,
                        tf_ids=tf_ids,
                        tg_ids=tg_ids_chunk,
                        bias=bias_chunk,
                    )
                    if isinstance(preds_s, tuple):
                        preds_s = preds_s[0]

                    preds_u = tg_scaler.inverse_transform(preds_s, tg_ids_chunk) if tg_scaler is not None else preds_s
                    preds_u = torch.nan_to_num(preds_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)

                grad_output_j = torch.zeros_like(preds_u)

                for offset in range(preds_u.shape[1]):
                    grad_output_j.zero_()
                    grad_output_j[:, offset] = 1.0

                    grads = torch.autograd.grad(
                        outputs=preds_u,
                        inputs=tf_tensor_chunk,
                        grad_outputs=grad_output_j,
                        retain_graph=(offset < preds_u.shape[1] - 1),
                        create_graph=False,
                    )[0]

                    grad_abs = grads[..., 0].abs() if grads.dim() == 3 else grads.abs()
                    grad_flat = grad_abs.reshape(-1)

                    tg_global = int(tg_ids_chunk[offset].item()) if tg_ids_chunk.dim() == 1 else int(tg_ids_chunk[0, offset].item())

                    grad_sum[:, tg_global].index_add_(0, tf_ids_flat, grad_flat)
                    grad_count[:, tg_global].index_add_(0, tf_ids_flat, torch.ones_like(grad_flat))
                    
                # cleanup per chunk
                del (
                    preds_u,
                    preds_s,
                    tf_scaled,
                    tf_tensor_chunk,
                    bias_chunk,
                    tg_ids_chunk,
                )
                    
            # Inside the loop - periodic saves
            if save_every_n_batches is not None:
                if b_idx % save_every_n_batches == 0:
                    
                    edge_seen = grad_count > 0
                    tf_idx, tg_idx = torch.nonzero(edge_seen, as_tuple=True)

                    scores = (grad_sum[tf_idx, tg_idx] / grad_count[tf_idx, tg_idx]).detach().cpu().numpy()

                    batch_df_long = pd.DataFrame({
                        "Source": [tf_names[i] for i in tf_idx.cpu().numpy()],
                        "Target": [tg_names[j] for j in tg_idx.cpu().numpy()],
                        "Score": scores,
                    })
                    
                    batch_grad_dfs[b_idx] = batch_df_long
        
        edge_seen = grad_count > 0
        tf_idx, tg_idx = torch.nonzero(edge_seen, as_tuple=True)

        scores = (grad_sum[tf_idx, tg_idx] / grad_count[tf_idx, tg_idx]).detach().cpu().numpy()

        df_long = pd.DataFrame({
            "Source": [tf_names[i] for i in tf_idx.cpu().numpy()],
            "Target": [tg_names[j] for j in tg_idx.cpu().numpy()],
            "Score": scores,
        })
        
        self.grn = self.format_grn(df_long)
        
        self.grn.to_csv(self.model_training_dir / "inferred_grn.csv", index=False)
        
        return self.grn, batch_grad_dfs
    
    def format_grn(self, df):
        
        def inverse_normal_transform(x):
            r = x.rank(method="average")
            n = len(x)
            p = (r - 0.5) / n          # avoids 0 and 1
            return norm.ppf(p)
        
        # Apply rank-based inverse normal transform (INT)
        df["Score"] = df.groupby("Source")["Score"].transform(inverse_normal_transform)
        
        df = df.dropna()
        
        df["Source"] = df["Source"].astype(str).str.upper()
        df["Target"] = df["Target"].astype(str).str.upper()
        
        return df
    
    def load_grn(self):
        grn_path = self.model_training_dir / "inferred_grn.csv"
        if not grn_path.exists():
            if self.silence_warnings is not None:
                logging.warning(f"GRN file not found at {grn_path}. Please run run_gradient_attribution() first to generate the GRN.")
            return None
        
        self.grn = pd.read_csv(grn_path)
        return self.grn
    
    def load_eval_results(self):      
        eval_files = [
            "pooled_auroc_auprc_raw_results.csv",
            "pooled_auroc_auprc_results.csv",
            "per_tf_auroc_auprc_results.csv",
            "per_tf_auroc_auprc_summary.csv",
        ]
        
        assert all([(self.model_training_dir / f).exists() for f in eval_files]), \
            f"Not all evaluation result files exist in {self.model_training_dir}. Expected files: {eval_files}"
        
        self.raw_results_df = pd.read_csv(self.model_training_dir / "pooled_auroc_auprc_raw_results.csv")
        self.results_df = pd.read_csv(self.model_training_dir / "pooled_auroc_auprc_results.csv")
        self.per_tf_all_df = pd.read_csv(self.model_training_dir / "per_tf_auroc_auprc_results.csv")
        self.per_tf_summary_df = pd.read_csv(self.model_training_dir / "per_tf_auroc_auprc_summary.csv")
    
    def run_forward_pass(self, num_batches: int = 1):
        device = self.device
        self.model.eval()

        global_tg_names = self.test_loader.dataset.tg_names

        pred_blocks = []
        true_blocks = []
        
        dataset = self.test_loader.dataset

        with torch.no_grad():
            for b, (batch_indices, batch) in tqdm(
                enumerate(
                    zip(self.test_loader.batch_sampler, self.test_loader)
                    ), 
                total=min(num_batches, len(self.test_loader.batch_sampler)), 
                desc="Running forward pass",
                ncols=80,
                ):
                if b >= num_batches:
                    break
                
                # Gets the cell indices for the batch, which is used to align the metacell names
                if hasattr(dataset, "_locate"):
                    local_indices = [dataset._locate(i)[1] for i in batch_indices]
                    if dataset._cell_idx is not None:
                        col_indices = [int(dataset._cell_idx[i]) for i in local_indices]
                    else:
                        col_indices = [int(i) for i in local_indices]
                else:
                    # Single-chrom dataset
                    if getattr(dataset, "_cell_idx", None) is not None:
                        col_indices = [int(dataset._cell_idx[i]) for i in batch_indices]
                    else:
                        col_indices = [int(i) for i in batch_indices]

                metacell_names = [dataset.metacell_names[i] for i in col_indices]
                
                atac_wins, tf_tensor, tg_expr_true, bias, tf_ids, tg_ids, motif_mask = batch
                atac_wins    = atac_wins.to(device, non_blocking=True)
                tf_tensor    = tf_tensor.to(device, non_blocking=True)
                tg_expr_true = tg_expr_true.to(device, non_blocking=True)
                bias         = bias.to(device, non_blocking=True)
                tf_ids       = tf_ids.to(device, non_blocking=True)
                tg_ids       = tg_ids.to(device, non_blocking=True)
                motif_mask   = motif_mask.to(device, non_blocking=True)   
                
                if getattr(self, "tf_scaler", None) is not None:
                    tf_tensor = self.tf_scaler.transform(tf_tensor, tf_ids)

                out = self.model(
                    atac_wins, tf_tensor,
                    tf_ids=tf_ids, tg_ids=tg_ids,
                    bias=bias
                )

                pred = self.tg_scaler.inverse_transform(out, ids=tg_ids).detach().cpu().numpy()
                true = tg_expr_true.detach().cpu().numpy()

                tg_ids_cpu = tg_ids.detach().cpu().numpy().astype(int)
                tg_names_batch = [global_tg_names[i] for i in tg_ids_cpu]

                pred_df = pd.DataFrame(pred.T, index=tg_names_batch, columns=metacell_names)
                true_df = pd.DataFrame(true.T, index=tg_names_batch, columns=metacell_names)

                pred_df = pred_df.reindex(index=global_tg_names)
                true_df = true_df.reindex(index=global_tg_names)

                pred_blocks.append(pred_df)
                true_blocks.append(true_df)

        pred_df = pd.concat(pred_blocks, axis=1, copy=False) if pred_blocks else pd.DataFrame()
        true_df = pd.concat(true_blocks, axis=1, copy=False) if true_blocks else pd.DataFrame()

        pred_df = pred_df.dropna(axis=0, how="all")
        true_df = true_df.loc[pred_df.index]
        
        self.tg_prediction_df = pred_df
        self.tg_true_df = true_df

        return pred_df, true_df
    
    def load_ground_truth(self, ground_truth_file: Tuple[str, Path]):
        if type(ground_truth_file) == str:
            ground_truth_file = Path(ground_truth_file)
            
        if ground_truth_file.suffix == ".csv":
            sep = ","
        elif ground_truth_file.suffix == ".tsv":
            sep="\t"
            
        ground_truth_df = pd.read_csv(ground_truth_file, sep=sep, on_bad_lines="skip", engine="python")
        
        if "chip" in ground_truth_file.name and "atlas" in ground_truth_file.name:
            ground_truth_df = ground_truth_df[["source_id", "target_id"]]

        if ground_truth_df.columns[0] != "Source" or ground_truth_df.columns[1] != "Target":
            ground_truth_df = ground_truth_df.rename(columns={ground_truth_df.columns[0]: "Source", ground_truth_df.columns[1]: "Target"})
        ground_truth_df["Source"] = ground_truth_df["Source"].astype(str).str.upper()
        ground_truth_df["Target"] = ground_truth_df["Target"].astype(str).str.upper()
        
        # Build TF, TG, and edge sets for quick lookup later
        gt = ground_truth_df[["Source", "Target"]].dropna()

        gt_tfs = set(gt["Source"].unique())
        gt_tgs = set(gt["Target"].unique())
        
        gt_pairs = (gt["Source"] + "\t" + gt["Target"]).drop_duplicates()
        
        gt_lookup = (gt_tfs, gt_tgs, set(gt_pairs))
            
        return ground_truth_df, gt_lookup
    
    def create_ground_truth_comparison_df(self, score_df, ground_truth_lookup, ground_truth_name):
        # Normalize once
        gt_tfs, gt_tgs, gt_pairs_set = ground_truth_lookup

        src = score_df["Source"]
        tgt = score_df["Target"]

        mask = src.isin(gt_tfs) & tgt.isin(gt_tgs)

        df = score_df.loc[mask].copy()
        # re-use normalized versions so we don't upper twice
        df["Source"] = src.loc[mask].values
        df["Target"] = tgt.loc[mask].values

        key = df["Source"] + "\t" + df["Target"]
        df["_in_gt"] = key.isin(gt_pairs_set).astype("int8")
        df["ground_truth_name"] = ground_truth_name

        return df
    
    def create_grn_ground_truth_overlap_comparison_df(
        self, 
        unlabeled_df: pd.DataFrame, 
        labeled_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame, 
        ground_truth_name: str,
        ):
        """
        Creates a DataFrame that compares the TF, TG, and edge sets of GRN and Ground Truth.
        
        Parameters
        ----------
        unlabeled_df : pd.DataFrame
            DataFrame with at least ['Source', 'Target'] columns that contains the GRN edges
        labeled_df : pd.DataFrame
            DataFrame with at least ['Source', 'Target'] columns that contains the overlap between GRN and Ground Truth
        ground_truth_df : pd.DataFrame
            DataFrame with at least ['Source', 'Target'] columns that contains the Ground Truth edges
        ground_truth_name : str
            The name of the Ground Truth dataset
        
        Returns
        -------
        overlap_info_df : pd.DataFrame
            DataFrame containing the comparison information
        """
        grn_unique_tfs = unlabeled_df["Source"].nunique()
        grn_unique_tgs = unlabeled_df["Target"].nunique()
        grn_unique_edges = len(unlabeled_df)

        gt_unique_tfs = ground_truth_df["Source"].nunique()
        gt_unique_tgs = ground_truth_df["Target"].nunique()
        gt_unique_edges = len(ground_truth_df)

        overlap_tfs = labeled_df["Source"].nunique()
        overlap_tgs = labeled_df["Target"].nunique()
        overlap_edges = len(labeled_df)
        
        comparison_dict = {
            "TFs": [grn_unique_tfs, gt_unique_tfs, overlap_tfs],
            "TGs": [grn_unique_tgs, gt_unique_tgs, overlap_tgs],
            "edges": [grn_unique_edges, gt_unique_edges, overlap_edges],
        }
        
        def pct(num, den):
            return np.where(den == 0, np.nan, (num / den) * 100)

        overlap_info_df = pd.DataFrame.from_dict(comparison_dict, orient="index", columns=["GRN", f"Ground Truth {ground_truth_name}", "Overlap (Score DF in GT)"])
        overlap_info_df["Pct of GRN in GT"] = pct(overlap_info_df["Overlap (Score DF in GT)"], overlap_info_df["GRN"]).round(2)
        overlap_info_df["Pct of GT in GRN"] = pct(overlap_info_df["Overlap (Score DF in GT)"], overlap_info_df[f"Ground Truth {ground_truth_name}"]).round(2)
        return overlap_info_df
    
    def report_grn_overlap_with_gt(self, ground_truth_name, ground_truth):
        ground_truth_df, gt_lookup = ground_truth
        labeled_df = self.create_ground_truth_comparison_df(self.grn, gt_lookup, ground_truth_name)
        unlabeled_df = self.grn

        grn_unique_tfs = unlabeled_df["Source"].nunique()
        grn_unique_tgs = unlabeled_df["Target"].nunique()
        grn_unique_edges = len(unlabeled_df)

        gt_unique_tfs = ground_truth_df["Source"].nunique()
        gt_unique_tgs = ground_truth_df["Target"].nunique()
        gt_unique_edges = len(ground_truth_df)

        overlap_tfs = labeled_df["Source"].nunique()
        overlap_tgs = labeled_df["Target"].nunique()
        overlap_edges = len(labeled_df)

        
        logging.info(ground_truth_name)
        logging.info(f"  - TF Overlap: {overlap_tfs:,} ({gt_unique_tfs:,} in GT)")
        logging.info(f"  - TG Overlap: {overlap_tgs:,} ({gt_unique_tgs:,} in GT)")
        logging.info(f"  - Edge Overlap: {overlap_edges:,} ({gt_unique_edges:,} in GT)")
    
    def evaluate_single_method_setwise(
        self,
        method_name: str,
        score_df: pd.DataFrame,
        ground_truth: tuple[pd.DataFrame, dict[str, set[str]]],
        ground_truth_name: str,
        max_edges: int = 10000,
    ) -> dict | None:
        """
        Mimics the R evaluate_single_method() logic:

        1. Uppercase TF/target names
        2. Filter inferred network to GT TFs and GT targets
        3. Take top max_edges edges overall
        4. Compare inferred edge set to GT edge set
        5. Compute precision, recall, F1

        Returns
        -------
        dict with:
            - metrics: dict(precision, recall, f1)
            - summary: one-row DataFrame
            - final_edges: filtered top-edge DataFrame
        """
        ground_truth_df, gt_lookup = ground_truth

        if score_df is None or len(score_df) == 0:
            return None

        # ---- Build GT TF/target sets ----
        gt_tfs = set(ground_truth_df["Source"].astype(str).str.upper())
        gt_targets = set(ground_truth_df["Target"].astype(str).str.upper())

        # GT edge pairs as strings, matching the R behavior
        gt_pairs = set(
            ground_truth_df["Source"].astype(str).str.upper()
            + "_"
            + ground_truth_df["Target"].astype(str).str.upper()
        )

        # ---- Filter inferred network by GT TFs and GT targets ----
        df_filtered = score_df.copy()
        df_filtered["source_upper"] = df_filtered["Source"].astype(str).str.upper()
        df_filtered["target_upper"] = df_filtered["Target"].astype(str).str.upper()

        df_filtered = df_filtered[
            df_filtered["source_upper"].isin(gt_tfs)
            & df_filtered["target_upper"].isin(gt_targets)
        ].copy()

        # ---- Sort by score descending before taking top max_edges ----
        if "Score" in df_filtered.columns:
            df_filtered = df_filtered.sort_values("Score", ascending=False, kind="stable")

        n_edges = min(max_edges, len(df_filtered))
        if n_edges == 0:
            print(f"    {method_name}: No edges after filtering, skipping")
            return None

        df_final = df_filtered.head(n_edges).copy()

        # ---- Create inferred edge pairs ----
        inferred_pairs = set(
            df_final["source_upper"].astype(str) + "_" + df_final["target_upper"].astype(str)
        )

        # ---- Set-based evaluation ----
        tp = inferred_pairs.intersection(gt_pairs)
        fp = inferred_pairs.difference(gt_pairs)
        fn = gt_pairs.difference(inferred_pairs)

        tp_n = len(tp)
        fp_n = len(fp)
        fn_n = len(fn)

        precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
        recall = tp_n / (tp_n + fn_n) if (tp_n + fn_n) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        inferred_tfs = set(df_final["source_upper"].unique())
        inferred_targets = set(df_final["target_upper"].unique())

        summary_df = pd.DataFrame([{
            "Method": method_name,
            "GT_name": ground_truth_name,
            "GT_total_edges": len(gt_pairs),
            "GT_TFs": len(gt_tfs),
            "GT_targets": len(gt_targets),
            "Original_network_size": len(score_df),
            "Filtered_network_size": len(df_filtered),
            "Final_network_size": len(df_final),
            "Inferred_TFs": len(inferred_tfs),
            "Inferred_targets": len(inferred_targets),
            "TP": tp_n,
            "FP": fp_n,
            "FN": fn_n,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_score": round(f1, 4),
            "Common_TFs": len(gt_tfs.intersection(inferred_tfs)),
            "Common_targets": len(gt_targets.intersection(inferred_targets)),
        }])

        return summary_df
    
    def generate_pooled_metrics(
        self,
        method_name: str,
        score_df: pd.DataFrame,
        ground_truth: tuple[pd.DataFrame, dict[str, set[str]]],
        ground_truth_name: str,
        top_fracs=(0.001, 0.005, 0.01, 0.05), 
        balance=True
        ) -> pd.DataFrame:
        
        ground_truth_df, gt_lookup = ground_truth
        
        # Uses a fast lookup to label GRN edges as 1 or 0 depending on whether they are in the ground truth or not
        # (only compares TFs and TGs that are in both the GRN and the ground truth)
        labeled_df = self.create_ground_truth_comparison_df(score_df, gt_lookup, ground_truth_name)
            
        if len(labeled_df) == 0 or labeled_df["_in_gt"].nunique() < 2:
            logging.info(f"Need at least one positive and one negative, got {labeled_df['_in_gt'].value_counts().to_dict()}")
            return None
        
        y_all = labeled_df["_in_gt"].fillna(0).astype(int).to_numpy()
        s_all = labeled_df["Score"].to_numpy()
        
        # --- ranking metrics on unbalanced
        pos_rate = y_all.mean()
        order = np.argsort(s_all)[::-1]
        y_sorted = y_all[order]
        tp = np.cumsum(y_sorted)
        k = np.arange(1, len(y_sorted) + 1)
        prec = tp / k
        
        prec_at = {}
        for frac in top_fracs:
            K = max(1, int(frac * len(y_sorted)))
            K = min(K, len(y_sorted))
            precK = float(prec[K-1])
            prec_at[f"precision@{frac*100:.2f}%"] = precK
            prec_at[f"lift@{frac*100:.2f}%"] = float(precK / pos_rate) if pos_rate > 0 else np.nan

        # --- AUROC/AUPRC: choose which set
        if balance:
            balanced = self._balance_pos_neg(labeled_df, random_state=42)
            y = balanced["_in_gt"].astype(int).to_numpy()
            s = balanced["Score"].to_numpy()
        else:
            y, s = y_all, s_all
            
        auroc = roc_auc_score(y, s)
        auprc = average_precision_score(y, s)
            
        pooled_metrics_df = pd.DataFrame({
            "method": method_name,
            "gt": ground_truth_name,
            "auroc": float(auroc) if not np.isnan(auroc) else np.nan,
            "auprc": float(auprc) if not np.isnan(auprc) else np.nan,
            "pos_rate": float(pos_rate) if not np.isnan(pos_rate) else np.nan,
            "lift_auprc": float(auprc / pos_rate) if (not np.isnan(auprc) and pos_rate > 0) else np.nan,
            **prec_at
        }, index=[0])
        return pooled_metrics_df
    
    def generate_per_tf_metrics(
        self,
        method_name: str,
        score_df: pd.DataFrame, 
        ground_truth: tuple[pd.DataFrame, dict[str, set[str]]],
        ground_truth_name: str,
        top_fracs=(0.001, 0.005, 0.01, 0.05), 
        min_edges=10, min_pos=1,
        balance=True
        ) -> pd.DataFrame:
        """
        Returns a per-TF dataframe with:
        TF, AUROC, n_pos, n_neg, pos_rate, Precision@K, Lift@K (for each K)
        """
        ground_truth_df, gt_lookup = ground_truth
        
        # Uses a fast lookup to label GRN edges as 1 or 0 depending on whether they are in the ground truth or not
        # (only compares TFs and TGs that are in both the GRN and the ground truth)
        labeled_df = self.create_ground_truth_comparison_df(score_df, gt_lookup, ground_truth_name)
            
        if len(labeled_df) == 0 or labeled_df["_in_gt"].nunique() < 2:
            logging.info(f"Need at least one positive and one negative, got {labeled_df['_in_gt'].value_counts().to_dict()}")
            return None

        rows = []
        tf_curves = {}
        for tf, g in labeled_df.groupby("Source", sort=False):
            y = g["_in_gt"].to_numpy()
            s = g["Score"].to_numpy()

            if balance:
                balanced = self._balance_pos_neg(g, random_state=42)
                y = balanced["_in_gt"].astype(int).to_numpy()
                s = balanced["Score"].to_numpy()
            else:
                y = g["_in_gt"].fillna(0).astype(int).to_numpy()
                s = g["Score"].to_numpy()
                
            n = len(y)
            n_pos = int(y.sum())
            n_neg = int(n - n_pos)
            pos_rate = (n_pos / n) if n > 0 else np.nan
            
            # basic filters to avoid degenerate metrics
            if n < min_edges:
                continue
            if n_pos < min_pos or n_neg == 0:
                continue
            
            # --- ROC/PR metrics and curves ---
            auroc = roc_auc_score(y, s)
            fpr, tpr, _ = roc_curve(y, s)
            rand_fpr, rand_tpr, _ = roc_curve(y, self._create_random_distribution(s))

            auprc = average_precision_score(y, s)
            prec, rec, _ = precision_recall_curve(y, s)
            rand_prec, rand_rec, _ = precision_recall_curve(y, self._create_random_distribution(s))
            
            tf_curves[tf] = {
                "auroc": auroc,
                "auprc": auprc,
                "prec": prec,
                "rec": rec,
                "fpr": fpr,
                "tpr": tpr,
                "rand_prec": rand_prec,
                "rand_rec": rand_rec,
                "rand_fpr": rand_fpr,
                "rand_tpr": rand_tpr,
            }
            
            # Pre-sort once for precision@K
            order = np.argsort(s)[::-1]
            y_sorted = y[order]
            tp = np.cumsum(y_sorted)

            row = {
                "tf": tf,
                "n_edges": n,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "pos_rate": pos_rate,
                "auroc": float(auroc) if not np.isnan(auroc) else np.nan,
                "auprc": float(auprc) if not np.isnan(auprc) else np.nan,
            }

            for frac in top_fracs:
                K = max(1, int(frac * n))
                K = min(K, n)
                prec_k = float(tp[K-1] / K) if n > 0 else np.nan
                row[f"precision@{frac*100:.2f}%"] = prec_k
                row[f"lift@{frac*100:.2f}%"] = (prec_k / pos_rate) if (pos_rate and pos_rate > 0) else np.nan

            rows.append(row)
            
        per_tf_df = pd.DataFrame(rows)
                
        # Skip if no TFs passed the filtering criteria
        if len(per_tf_df) == 0 or "auroc" not in per_tf_df.columns:
            if not self.silence_warnings:
                logging.warning(f"WARNING: No TFs passed filtering criteria for {ground_truth_name}")
            return None
            
        per_tf_df.insert(0, "gt", ground_truth_name)
        per_tf_df.insert(0, "method", method_name)

        return per_tf_df, tf_curves
    
    def quick_pooled_auroc(self, labeled_df):
        balanced = self._balance_pos_neg(labeled_df, random_state=42)
        y = balanced["_in_gt"].astype(int).to_numpy()
        s = balanced["Score"].to_numpy()
        
        auroc = roc_auc_score(y, s)
        
        return auroc

    def quick_per_tf_auroc(self, labeled_df):
        per_tf_auroc = []
        
        for tf, group in labeled_df.groupby("Source"):
            balanced = self._balance_pos_neg(group, random_state=42)
            y = balanced["_in_gt"].astype(int).to_numpy()
            s = balanced["Score"].to_numpy()
            
            if len(np.unique(y)) > 1:
                auroc = roc_auc_score(y, s)
                per_tf_auroc.append(auroc)
            else:
                per_tf_auroc.append(np.nan)  # or some default value for TFs with only pos or neg examples
        
        median_per_tf_auroc = np.nanmedian(per_tf_auroc)
        
        return median_per_tf_auroc

    def calculate_auroc_all_sample_gts(self, grad_attr_df, ground_truth_dict):    
        pooled_auroc = []
        per_tf_auroc = []
        for gt_name, ground_truth in ground_truth_dict.items():
            _, gt_lookup = ground_truth
            
            labeled_df = self.create_ground_truth_comparison_df(self.grn, gt_lookup, gt_name)
            
            gt_pooled_auroc = self.quick_pooled_auroc(labeled_df)
            gt_per_tf_auroc = self.quick_per_tf_auroc(labeled_df)
            
            pooled_auroc.append(gt_pooled_auroc)
            per_tf_auroc.append(gt_per_tf_auroc)

        pooled_median_auroc = np.median(pooled_auroc)
        per_tf_median_auroc = np.median(per_tf_auroc)
            
        auroc_df = pd.DataFrame({
            "pooled_median_auroc": pooled_median_auroc,
            "per_tf_median_auroc": per_tf_median_auroc,
        }, index=[0])
        
        return auroc_df
    
    def calculate_auroc_all_methods(self, sample_list, dataset_type, ground_truth_dict, grn=None, use_muon_grn=True):        
        grn = self.grn if grn is None else grn
        
        assert grn is not None, "GRN must be provided either through argument or by running gradient attribution first"
        
        # Loop through each ground truth dataset and load each file
        ground_truth_edges_dict = {gt: auroc_utils.prep_gt_edges(df) for gt, (df, _) in ground_truth_dict.items()}
        
        # Load the other method GRNs (TF-TG scores averaged across samples)
        if use_muon_grn:
            standardized_method_dict = auroc_utils.load_other_method_muon_grns(sample_list, dataset_type)
        else:
            standardized_method_dict = auroc_utils.load_other_method_grns(sample_list, dataset_type)
        
        standardized_method_dict["Gradient Attribution"] = grn
            
        # Pooled AUROC/AUPRC
        logging.info("\nEvaluating pooled methods across samples")
        results_df, raw_results_df = auroc_utils.calculate_pooled_auroc(
            standardized_method_dict, ground_truth_edges_dict, 
            per_tf_methods={"Gradient Attribution"}
            )
        
        logging.info(results_df.groupby("method")["auroc"].median().sort_values(ascending=False))

        # Per-TF AUROC/AUPRC
        logging.info("\nPer-TF evaluation of pooled methods across samples")
        per_tf_all_df, per_tf_summary_df = auroc_utils.calculate_per_tf_auroc(
            standardized_method_dict, ground_truth_edges_dict, [0.001, 0.01, 0.1], 
            per_tf_methods={"Gradient Attribution"}
            )    
        
        logging.info(per_tf_summary_df.groupby("method")["median_per_tf_auroc"].median().sort_values(ascending=False))
        
        # Save results
        raw_results_df.to_csv(self.model_training_dir / "pooled_auroc_auprc_raw_results.csv", index=False)
        results_df.to_csv(self.model_training_dir / "pooled_auroc_auprc_results.csv", index=False)
        per_tf_all_df.to_csv(self.model_training_dir / "per_tf_auroc_auprc_results.csv", index=False)
        per_tf_summary_df.to_csv(self.model_training_dir / "per_tf_auroc_auprc_summary.csv", index=False)
    
    def plot_auroc_auprc(
        self, 
        score_df: pd.DataFrame, 
        ground_truth: Tuple[pd.DataFrame, Tuple[Set[str], Set[str], Set[str]]],
        ground_truth_name: str, 
        return_overlap_info: bool = True,
        balance: bool = True,
        no_fig: bool = False,
        save_fig: bool = False,
        
        ):
        
        ground_truth_df, gt_lookup = ground_truth
        
        # Uses a fast lookup to label GRN edges as 1 or 0 depending on whether they are in the ground truth or not
        # (only compares TFs and TGs that are in both the GRN and the ground truth)
        labeled_df = self.create_ground_truth_comparison_df(score_df, gt_lookup, ground_truth_name)
            
        if len(labeled_df) == 0 or labeled_df["_in_gt"].nunique() < 2:
            logging.info(f"Need at least one positive and one negative, got {labeled_df['_in_gt'].value_counts().to_dict()}")
            return None
        
        if balance:
            balanced = self._balance_pos_neg(labeled_df, random_state=42)
            y = balanced["_in_gt"].astype(int).to_numpy()
            s = balanced["Score"].to_numpy()
        else:
            y = labeled_df["_in_gt"].fillna(0).astype(int).to_numpy()
            s = labeled_df["Score"].to_numpy()
        
        auroc = roc_auc_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        rand_fpr, rand_tpr, _ = roc_curve(y, self._create_random_distribution(s))
        
        auprc = average_precision_score(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        rand_prec, rand_rec, _ = precision_recall_curve(y, self._create_random_distribution(s))
        
        metric_df = pd.DataFrame({
            "experiment": self.experiment_name,
            "ground_truth": ground_truth_name,
            "auroc": auroc,
            "auprc": auprc,
        }, index=[0])
        
        
        if self.auroc_auprc_scores is None:
            self.auroc_auprc_scores = metric_df
        else:
            self.auroc_auprc_scores = pd.concat([self.auroc_auprc_scores, metric_df], ignore_index=True)
                
        if not no_fig:
            # ROC plot
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
            ax[0].plot(rand_fpr, rand_tpr, color="#747474", linestyle="--", lw=2)
            ax[0].plot(fpr, tpr, lw=2, color="#4195df", label=f"AUROC = {auroc:.3f}")
            ax[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
            ax[0].set_xlabel("False Positive Rate", fontsize=12)
            ax[0].set_ylabel("True Positive Rate", fontsize=12)
            ax[0].set_title(f"AUROC", fontsize=12)
            ax[0].legend(
                bbox_to_anchor=(0.5, -0.28),
                loc="upper center",
                borderaxespad=0.0
            )
            ax[0].set_xlim(0, 1)
            ax[0].set_ylim(0, 1)
            
            # Precision-Recall plot
            ax[1].plot(rand_rec, rand_prec, color="#747474", linestyle="--", lw=2)
            ax[1].plot(rec, prec, lw=2, color="#4195df", label=f"AUPRC = {auprc:.3f}")
            ax[1].set_xlabel("Recall", fontsize=12)
            ax[1].set_ylabel("Precision", fontsize=12)
            ax[1].set_title(f"AUPRC", fontsize=12)
            ax[1].legend(
                bbox_to_anchor=(0.5, -0.28),
                loc="upper center",
                borderaxespad=0.0
            )
            ax[1].set_ylim(0, 1.0)
            ax[1].set_xlim(0, 1.0)
            plt.suptitle(f"{self.experiment_name} vs {ground_truth_name}", fontsize=14)
            plt.tight_layout()
            
            if save_fig:
                fig_dir = os.path.join(self.experiment_dir, self.experiment_name, ground_truth_name)
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)
                auroc_fig_path = os.path.join(fig_dir, f"{ground_truth_name}_auroc_auprc.png")

                fig.savefig(auroc_fig_path, dpi=300)
        else:
            fig = None
            
        if return_overlap_info:
            overlap_info_df = self.create_grn_ground_truth_overlap_comparison_df(
                score_df, labeled_df, ground_truth_df, ground_truth_name
            )
        else:
            overlap_info_df = None
        
        return fig, overlap_info_df
    
    def plot_hist_roc_pr(
        self,
        score_df, 
        ground_truth_files,
        method_name,
        n_bins=75, 
        random_state=42, 
        y_log=False,
        panel_kind="kde",  # "hist" or "kde"
        density=False,
        selected_batch_num=None,
        ):
        
        nrows = len(ground_truth_files)
        fig, ax = plt.subplots(
            nrows=nrows, 
            ncols=3, 
            figsize=(12, 2.5*nrows + 1),
            squeeze=False
            )
        
        for i, (ground_truth_name, ground_truth) in enumerate(ground_truth_files.items()):
            ground_truth_df, gt_lookup = ground_truth
                
            # Uses a fast lookup to label GRN edges as 1 or 0 depending on whether they are in the ground truth or not
            # (only compares TFs and TGs that are in both the GRN and the ground truth)
            labeled_df = self.create_ground_truth_comparison_df(score_df, gt_lookup, ground_truth_name)

            # --- balance pos/neg once so all three panels use same data ---
            balanced = self._balance_pos_neg(labeled_df, random_state=random_state).copy()
            
            # safety: remove NaN/inf scores
            balanced = balanced[np.isfinite(balanced["Score"].to_numpy())].copy()

            if balanced["_in_gt"].nunique() < 2:
                raise ValueError("Need both positive and negative examples to plot ROC/PR.")

            y = balanced["_in_gt"].astype(int).to_numpy()
            s = balanced["Score"].to_numpy()

            # --- ROC/PR metrics and curves ---
            auroc = roc_auc_score(y, s)
            fpr, tpr, _ = roc_curve(y, s)
            rand_fpr, rand_tpr, _ = roc_curve(y, self._create_random_distribution(s))

            auprc = average_precision_score(y, s)
            prec, rec, _ = precision_recall_curve(y, s)
            rand_prec, rand_rec, _ = precision_recall_curve(y, self._create_random_distribution(s))

            # --- Histogram data (balanced counts already, but keep code explicit) ---
            true_vals = balanced.loc[balanced["_in_gt"] == 1, "Score"].dropna()
            false_vals = balanced.loc[balanced["_in_gt"] == 0, "Score"].dropna()

            min_len = min(len(true_vals), len(false_vals))
            if min_len == 0:
                raise ValueError("Not enough positives/negatives to plot histograms.")

            true_vals = true_vals.sample(n=min_len, random_state=random_state)
            false_vals = false_vals.sample(n=min_len, random_state=random_state)

            combined = pd.concat([true_vals, false_vals])
            bins = np.linspace(combined.min(), combined.max(), n_bins)

            # (1) Histogram or KDE
            if panel_kind == "hist":
                ax[i, 0].hist(false_vals, bins=n_bins, alpha=0.6, color="#747474", label="False",
                            density=density)
                ax[i, 0].hist(true_vals,  bins=n_bins, alpha=0.6, color="#4195df", label="True",
                            density=density)

                ax[i, 0].set_title("True vs False Scores", fontsize=12)
                ax[i, 0].set_xlabel("Score", fontsize=12)
                ax[i, 0].set_ylabel(f"{ground_truth_name}\n{'Density' if density else 'Count'}", fontsize=12)
                ax[i, 0].legend(fontsize=9)

            elif panel_kind == "kde":
                sns.kdeplot(false_vals, ax=ax[i, 0], color="#747474", label="False",
                            fill=True, common_norm=False, bw_adjust=1.0)
                sns.kdeplot(true_vals,  ax=ax[i, 0], color="#4195df", label="True",
                            fill=True, common_norm=False, bw_adjust=1.0)

                ax[i, 0].set_title("True vs False Score Density", fontsize=12)
                ax[i, 0].set_xlabel("Score", fontsize=12)
                ax[i, 0].set_ylabel(f"{ground_truth_name}\nDensity", fontsize=12)
                ax[i, 0].legend(fontsize=9)
                
            if y_log == True:
                ax[i, 0].set_yscale("log")
                ax[i, 0].set_ylim(bottom=0.1)

            # (2) ROC
            ax[i, 1].plot(rand_fpr, rand_tpr, color="#747474", linestyle="--", lw=2)
            ax[i, 1].plot(fpr, tpr, color="#4195df", lw=2, label=f"AUROC = {auroc:.3f}")
            # ax[i, 1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
            ax[i, 1].set_title("ROC", fontsize=12)
            ax[i, 1].set_xlabel("False Positive Rate", fontsize=12)
            ax[i, 1].set_ylabel("True Positive Rate", fontsize=12)
            ax[i, 1].set_xlim(0, 1)
            ax[i, 1].set_ylim(0, 1)
            ax[i, 1].legend(fontsize=9, loc="lower right")

            # (3) PR
            ax[i, 2].plot(rand_rec, rand_prec, color="#747474", linestyle="--", lw=2)
            ax[i, 2].plot(rec, prec, color="#4195df", lw=2, label=f"AUPRC = {auprc:.3f}")
            ax[i, 2].set_title("Precision–Recall", fontsize=12)
            ax[i, 2].set_xlabel("Recall", fontsize=12)
            ax[i, 2].set_ylabel("Precision", fontsize=12)
            ax[i, 2].set_xlim(0, 1)
            ax[i, 2].set_ylim(0, 1)
            ax[i, 2].legend(fontsize=9, loc="lower right")

        if selected_batch_num is not None:
            fig.suptitle(f"{method_name}\n{self.experiment_name}\n{selected_batch_num} Batches", fontsize=14)
        else:
            fig.suptitle(f"{method_name}\n{self.experiment_name}", fontsize=14)
        
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        return fig
    
    def plot_relative_improvement(self, per_tf_plot_df, experiment_name, override_title=None):
        median_per_tf_score = per_tf_plot_df.groupby(["method"])["auroc"].median().reset_index()
        median_per_tf_score["diff_from_ga"] = median_per_tf_score.loc[median_per_tf_score["method"] == "Gradient Attribution", "auroc"].values[0] - median_per_tf_score["auroc"]

        median_per_tf_score = median_per_tf_score.drop(median_per_tf_score[median_per_tf_score["method"] == "Gradient Attribution"].index)
        median_per_tf_score = median_per_tf_score.drop(median_per_tf_score[median_per_tf_score["method"] == "TF Knockout"].index)


        median_per_tf_score["method"] = pd.Categorical(
            median_per_tf_score["method"],
            categories=order,
            ordered=True
        )

        median_per_tf_score = median_per_tf_score.sort_values("method", ascending=False)

        colors = [self.method_color_dict.get(m, "#BBBBBB") for m in median_per_tf_score["method"]]

        name = re.sub(r"_hvg_filter_disp_.*", "", experiment_name).replace("_", " ")

        fig = plt.figure(figsize=(4, 4.5))
        plt.barh(median_per_tf_score["method"], median_per_tf_score["diff_from_ga"], color=colors)
        ax = plt.gca()
        ax.axvline(0, color=self.color_palette["grey_light"], linestyle="--", linewidth=1)
        plt.xlabel("AUROC Improvement")
        
        if max(median_per_tf_score["diff_from_ga"]) > 0.15 or min(median_per_tf_score["diff_from_ga"]) < -0.15:
            plt.xlim(-0.3, 0.3)
        else:
            plt.xlim(-0.15, 0.15)
        if override_title is not None:
            plt.title(override_title)
        else:
            plt.title(f"{name}\nPer-TF AUROC Improvement")
        plt.tight_layout()
        return fig
    
    def plot_true_vs_predicted_tg_expression(
        self, 
        num_batches: int=15, 
        rerun_forward_pass: bool = False, 
        set_axis_logscale: bool = False,
        title: Optional[str] = None,
        ):
        fig, ax = plt.subplots(figsize=(4,4))

        if self.tg_prediction_df is None or self.tg_true_df is None or rerun_forward_pass:
            logging.info("Running forward pass to get predicted vs true TG expression for a subset of test batches...")
            self.tg_prediction_df, self.tg_true_df = self.run_forward_pass(num_batches=num_batches)
        
        x = self.tg_true_df.median(axis=1).values
        y = self.tg_prediction_df.median(axis=1).values
        
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        from sklearn.metrics import r2_score
        # Calculate the R^2 value on all values, not the means
        x_flat = self.tg_true_df.values.ravel()
        y_flat = self.tg_prediction_df.values.ravel()
        mask = np.isfinite(x_flat) & np.isfinite(y_flat)

        x_flat = x_flat[mask]
        y_flat = y_flat[mask]
        
        r2 = r2_score(x_flat, y_flat)
        
        # correlation = np.corrcoef(x, y)[0, 1]
        if title is None:
            ax.set_title(f"Model Prediction vs True TG Expression", fontsize=16)
        else:
            ax.set_title(title, fontsize=16)

        ax.scatter(x, y, s=10, alpha=0.6, color="#4195df")
        
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())

        pad = 1.2
        lims = [min_val / pad, max_val * pad]

        if set_axis_logscale:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1)
        
        ax.text(
            0.98, 0.02,
            f"$R^2$ = {r2:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
                pad=2
            )
        )
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("True Expression", fontsize=13)
        ax.set_ylabel("Predicted Expression", fontsize=13)
        
        plt.tight_layout()
        
        return fig
    
    def plot_per_tf_auroc_boxplot(self, agg_by_gt: bool = True, ylim: tuple = (0.3, 0.7)):
        """
        Plots a boxplot of per-TF AUROC scores for each GRN inference method.

        Parameters
        ----------
        agg_by_gt : bool, optional
            * If **True**, plots the average per-TF AUROC score for all TFs for each ground truth
            * If **False**, plots the per-TF AUROC scores for each TF for each ground truth

        """
        if self.per_tf_all_df is None:
            
            assert (self.model_training_dir / "per_tf_auroc_auprc_results.csv").exists(), \
                f"Per-TF AUROC/AUPRC results file does not exist for {self.experiment_name}, model_training_00{self.model_num}"
            
            self.per_tf_all_df = pd.read_csv(self.model_training_dir / "per_tf_auroc_auprc_results.csv")
        
        if not agg_by_gt:
            # Average the per-TF AUROC scores across ground truths for each method
            per_tf_plot_df = (
                self.per_tf_all_df.dropna(subset=["auroc"])
                .groupby(['method', 'tf'], as_index=False)
                .agg(
                    auroc=('auroc', 'median'),
                    n_gt=('gt', 'nunique'),
                )
            )
            
        elif agg_by_gt:
            # Average the per-TF AUROC scores across ground truths for each method
            per_tf_plot_df = (
                self.per_tf_all_df
                .dropna(subset=["auroc"])
                .groupby(["method", "gt"], as_index=False)
                .agg(
                    auroc=("auroc", "median"),
                    n_gt=("gt", "nunique"),
                )
            )


        fig = self._plot_all_results_auroc_boxplot(
            per_tf_plot_df, 
            per_tf=True,
            ylim=ylim
            )
        
        fig.tight_layout()
        return fig
        
    def plot_pooled_auroc_boxplot(self, ylim: tuple = (0.3, 0.7)):
        if self.results_df is None:
            assert (self.model_training_dir / "pooled_auroc_auprc_results.csv").exists(), \
                f"Pooled AUROC/AUPRC results file does not exist for {self.experiment_name}, model_training_00{self.model_num}"
            self.results_df = pd.read_csv(self.model_training_dir / "pooled_auroc_auprc_results.csv")
            
        fig = self._plot_all_results_auroc_boxplot(
            self.results_df, 
            per_tf=False,
            ylim=ylim,
            )
        fig.tight_layout()
        
        return fig
    
    def plot_top_n_tf_roc_curves(
        self,
        score_df, 
        gt_df, 
        ground_truth_name, 
        exp, 
        method_name="Gradient Attribution", 
        num_top_tfs_to_plot=25,
        min_edges=10,
        min_pos=1,
        balance=True,
        name_clean=None,
        override_title=None,
        ):
        per_tf_df, tf_curves = self.generate_per_tf_metrics(
            method_name=method_name, 
            score_df=score_df, 
            ground_truth=gt_df, 
            ground_truth_name=ground_truth_name, 
            top_fracs=(0.001, 0.005, 0.01, 0.05),
            min_edges=min_edges,
            min_pos=min_pos,
            balance=balance,
            )
        
        fig = plt.figure(figsize=(5,4))

        top_tfs = per_tf_df.sort_values("auroc", ascending=False)["tf"].unique()[:num_top_tfs_to_plot]
        tf_curves_subset = {tf: curves for tf, curves in tf_curves.items() if tf in top_tfs}
        tf_curves_sorted = dict(sorted(tf_curves_subset.items(), key=lambda item: per_tf_df.set_index("tf").loc[item[0], "auroc"], reverse=True))

        if name_clean is not None:
            name = name_clean
        else:
            name = exp.experiment_name
        
        for i, tf in enumerate(tf_curves_sorted.keys()):
            if i < 11:
                label = f"{tf} (AUROC={per_tf_df.set_index('tf').loc[tf, 'auroc']:.3f})"
            else:
                label = None  # Only label the top 10 for clarity
                
            plt.plot(
                tf_curves_sorted[tf]["fpr"], 
                tf_curves_sorted[tf]["tpr"], 
                lw=1.5, 
                label=label,
                alpha=per_tf_df.set_index("tf").loc[tf, "auroc"])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            if override_title is not None:
                plt.title(override_title)
            else:
                plt.title(f"{name} vs {ground_truth_name}\n{method_name} Top {num_top_tfs_to_plot} Per-TF AUROC")
            
        plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
            
        fig.legend(
            bbox_to_anchor=(1.0, 0.5), 
            loc="center left", 
            borderaxespad=0.0, 
            title=f"Top 10 TFs:", 
        )
        plt.tight_layout()

        return fig, per_tf_df, tf_curves
    
    def plot_method_gt_heatmap(
        self,
        df: pd.DataFrame, 
        metric: str = "auroc", 
        per_tf: bool = False,
        x_scale: float = 1.5,
        y_scale: float = 0.6,
        override_title: str | None = None,
        ) -> plt.Figure:
        """
        Plot a heatmap of METHOD (rows) x GROUND TRUTH (cols) for AUROC or AUPRC.

        Rows are sorted by the median metric across all ground truth datasets.
        """
        metric = metric.lower()
        if metric not in ("auroc", "auprc"):
            raise ValueError(f"metric must be 'auroc' or 'auprc', got {metric}")

        metric_col = metric  # 'auroc' or 'auprc'
        
        df["method"] = df["method"].str.replace(" ", "\n")

        # 1) Order methods by median metric across all GTs (descending)
        method_order = (
            df.groupby("method")[metric_col]
            .median()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        
        # 2) Pivot to METHOD x GT matrix
        heat_df = (
            df
            .pivot_table(index="method", columns="gt", values=metric_col)
            .loc[method_order]  # apply sorted method order
        )

        # 3) Plot heatmap with better sizing
        n_methods = len(heat_df.index)
        n_gts = len(heat_df.columns)
        
        fig, ax = plt.subplots(
            figsize=(
                max(n_gts * x_scale, 4),      # Width: 1.5 inches per GT, min 6
                max(n_methods * y_scale, 3),  # Height: 0.5 inches per method, min 4
            )
        )
        
        sns.heatmap(
            heat_df,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            vmin=0.3,
            vmax=0.7,
            cbar_kws={"label": metric.upper()},
            ax=ax,
        )
        
        plt.xlabel(None)
        plt.ylabel(None)
        
        if override_title is not None:
            ax.set_title(override_title, pad=10)
        
        else:
            if per_tf == True:
                ax.set_title(
                    f"Median per-TF {metric.upper()} score × ground truth\n"
                    f"(methods sorted by Median {metric.upper()} across GTs)",
                    pad=10,
                )
            else:
                ax.set_title(
                    f"Median Pooled {metric.upper()} score × ground truth\n"
                    f"(methods sorted by Median {metric.upper()} across GTs)",
                    pad=10,
                )
        
        # Improve tick label readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        fig.tight_layout()
        return fig
    
    def _plot_all_results_auroc_boxplot(
        self,
        df: pd.DataFrame, 
        per_tf: bool = False, 
        override_color: bool = False,
        ylim: tuple = (0.3, 0.7),
        sort_by_median: bool = True,
        override_title: Optional[str] = None,
        method_color_dict: Optional[Dict[str, str]] = None,
        ) -> plt.Figure:
        """
        Plots AUROC boxplots for all GRN inference methods in the provided DataFrame.
        
        Parameters
        -------------
        df : pd.DataFrame
            DataFrame containing AUROC results with columns 'method' and 'auroc'.
        per_tf : bool, optional
            If True, indicates that the DataFrame contains per-TF AUROC scores. Default is False.
        override_color : bool, optional
            If True, overrides the default coloring scheme for methods to plot all boxes as blue. Default is False.
        """
        
        
        # 1. Order methods by median AUROC (highest → lowest)
        if sort_by_median:
            method_order = (
                df.groupby("method")["auroc"]
                .median()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
        else:
            method_order = (
                df.groupby("method")["auroc"]
                .median()
                .index
                .tolist()
            )

        if "No Filtering" in method_order:
            method_order = [m for m in method_order if m != "No Filtering"] + ["No Filtering"]
        
        mean_by_method = (
            df.groupby("method")["auroc"]
            .median()
        )
        
        # 2. Prepare data in that order
        data = [df.loc[df["method"] == m, "auroc"].values for m in method_order]

        feature_list = [
            "Gradient Attribution",
            "TF Knockout",
        ]
        my_color = "#4195df"
        other_color = "#747474"

        fig, ax = plt.subplots(figsize=(8, 5))

        # Baseline random line
        ax.axhline(y=0.5, color="#2D2D2D", linestyle='--', linewidth=1)

        # --- Boxplot (existing styling) ---
        bp = ax.boxplot(
            data,
            tick_labels=method_order,
            patch_artist=True,
            showfliers=False
        )

        # Color boxes: light blue for your methods, grey for others
        for box, method in zip(bp["boxes"], method_order):
            if method_color_dict and method in method_color_dict:
                box.set_facecolor(method_color_dict[method])
            else:
                if method in feature_list or override_color:
                    box.set_facecolor(my_color)
                else:
                    box.set_facecolor(other_color)

        # Medians in black
        for median in bp["medians"]:
            median.set_color("black")

        # --- NEW: overlay jittered points for each method ---
        for i, method in enumerate(method_order, start=1):
            y = df.loc[df["method"] == method, "auroc"].values
            if len(y) == 0:
                continue

            # Small horizontal jitter around the box center (position i)
            x = np.random.normal(loc=i, scale=0.06, size=len(y))

            # Match point color to box color
            if method_color_dict and method in method_color_dict:
                point_color = method_color_dict[method]
            else:
                point_color = my_color if method in feature_list or override_color else other_color

            ax.scatter(
                x, y,
                color=point_color,
                alpha=0.7,
                s=18,
                edgecolor="k",
                linewidth=0.3,
                zorder=3,
            )

        legend_handles = []

        for method in method_order:
            # Match EXACT box color logic
            if method_color_dict and method in method_color_dict:
                color = method_color_dict[method]
            elif method in feature_list or override_color:
                color = my_color
            else:
                color = other_color
            label = method.replace(" ", "\n")
            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=color,
                    markeredgecolor="k",
                    markersize=7,
                    label=f"{label}: {mean_by_method.loc[method]:.3f}"
                )
            )
            
        
        ax.legend(
            handles=legend_handles,
            title="Median AUROC",
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            borderaxespad=0.0,
            ncol=1,
        )

        ax.set_ylabel("AUROC across ground truths")
        
        if override_title is not None:
            ax.set_title(override_title)
        else:
            if per_tf == True:
                ax.set_title("per-TF AUROC Scores per method")
            else:
                ax.set_title("AUROC Scores per method")
            
        ax.set_ylim(ylim)

        labels = [t.get_text().replace(" ", "\n") for t in ax.get_xticklabels()]
        ax.set_xticklabels(labels)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        
        return fig
    
    def _load_json(self, path: Path) -> dict:
        with open(path, "r") as f:
            data = json.load(f)
        return data
        
    def _balance_pos_neg(self, df, random_state=42):
        """Balances positive and negative edges by downsampling the majority class."""
        rng = np.random.default_rng(random_state)

        y = df["_in_gt"].to_numpy() == 1
        pos_idx = np.flatnonzero(y)
        neg_idx = np.flatnonzero(~y)

        n_pos = pos_idx.size
        n_neg = neg_idx.size
        if n_pos == 0 or n_neg == 0:
            return df  # no copy

        n = min(n_pos, n_neg)
        if n_pos > n:
            pos_idx = rng.choice(pos_idx, size=n, replace=False)
        if n_neg > n:
            neg_idx = rng.choice(neg_idx, size=n, replace=False)

        # iloc keeps column dtypes; copy only the subset
        return df.iloc[np.concatenate([pos_idx, neg_idx])].reset_index(drop=True)
    
    def _create_random_distribution(self, scores, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        arr = np.asarray(scores)   # works for Series or ndarray, no copy if already ndarray
        return rng.uniform(arr.min(), arr.max(), size=arr.shape[0])
    
    def _create_model_training_dir(self, allow_overwrite=False):
        while True:
            model_str = f"{self.model_num:03d}"
            candidate_dir = self.experiment_dir / self.experiment_name / f"model_training_{model_str}"

            if allow_overwrite or not candidate_dir.exists():
                self.model_training_dir = candidate_dir
                self.model_training_dir.mkdir(parents=True, exist_ok=True)
                break
            else:
                self.model_num += 1

        if not allow_overwrite:
            logging.info(f"Using model_num={self.model_num} at {self.model_training_dir}")
            