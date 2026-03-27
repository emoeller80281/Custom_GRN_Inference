import json
import os
import random
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

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

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

logging.basicConfig(level=logging.INFO, format='%(message)s')

class ExperimentLoader:
    def __init__(self, experiment_dir: str, experiment_name: str, model_num: int, silence_warnings: bool = False):
        
        assert os.path.exists(experiment_dir), f"Experiment directory {experiment_dir} does not exist."
        
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = experiment_name
        self.model_num = model_num
        self.silence_warnings = silence_warnings
        
        if "chr19" in [p.name for p in (Path(experiment_dir) / experiment_name).iterdir()]:
            self.model_training_dir = Path(f"{experiment_dir}/{experiment_name}/chr19/model_training_00{model_num}")
        else:
            self.model_training_dir = Path(f"{experiment_dir}/{experiment_name}/model_training_00{model_num}")
                
        assert self.model_training_dir.exists(), f"Model training directory {self.model_training_dir} does not exist."
        
        # Load the run parameters saved during model training
        if not (self.model_training_dir / "run_parameters.json").exists():
            if not self.silence_warnings:
                logging.warning(f"WARNING: Run parameters file {self.model_training_dir / 'run_parameters.json'} does not exist.")
            self.model_training_params = None
        else:
            self.model_training_params = self._load_json(self.model_training_dir / "run_parameters.json")
        
        # Open the full experiment settings file
        if not (self.experiment_dir / self.experiment_name / "run_parameters_long.csv").exists():
            if not self.silence_warnings:
                logging.warning(f"WARNING: Experiment settings file {self.experiment_dir / self.experiment_name / 'run_parameters_long.csv'} does not exist.")
            self.experiment_settings_df = None
        else:
            self.experiment_settings_df = pd.read_csv(self.experiment_dir / self.experiment_name / "run_parameters_long.csv")
        
        # Open the preprocessing and pseudobulk file
        if not (self.experiment_dir / self.experiment_name / "experiment_info.json").exists():
            if not self.silence_warnings:
                logging.warning(f"WARNING: Experiment info file {self.experiment_dir / self.experiment_name / 'experiment_info.json'} does not exist.")
            self.preprocessing_info = None
        else:
            self.preprocessing_info = self._load_json(self.experiment_dir / self.experiment_name / "experiment_info.json")
            
        # Open the preprocessing settings file
        if not (self.experiment_dir / self.experiment_name / "preprocessing_config.json").exists():
            if not self.silence_warnings:
                logging.warning(f"WARNING: Preprocessing settings file {self.experiment_dir / self.experiment_name / 'preprocessing_config.json'} does not exist.")
            self.preprocessing_settings = None
        else:
            self.preprocessing_settings = self._load_json(self.experiment_dir / self.experiment_name / "preprocessing_config.json")

        # Load the GPU usage log and format it into a more usable format
        self.gpu_usage_df, self.gpu_usage_mean_per_sec_df, self.gpu_memory_limit_gib = self._format_gpu_usage_file()
        
        # Load the model training log data
        if not (self.model_training_dir / "training_log.csv").exists():
            if not self.silence_warnings:
                logging.warning(f"WARNING: Training log file {self.model_training_dir / 'training_log.csv'} does not exist.")
            self.training_df = None
        else:
            self.training_df = pd.read_csv(self.model_training_dir / "training_log.csv")
        
        # Loads the TF and TG names in order by their index
        if not (self.model_training_dir / "tf_names_ordered.json").exists():
            if not self.silence_warnings:
                logging.warning(f"WARNING: TF names file {self.model_training_dir / 'tf_names_ordered.json'} does not exist.")
            self.tf_names = None
        else:
            self.tf_names = self._load_json(self.model_training_dir / "tf_names_ordered.json")

        if not (self.model_training_dir / "tg_names_ordered.json").exists():
            if not self.silence_warnings:
                logging.warning(f"WARNING: TG names file {self.model_training_dir / 'tg_names_ordered.json'} does not exist.")
            self.tg_names = None
        else:
            self.tg_names = self._load_json(self.model_training_dir / "tg_names_ordered.json")
        
        # Model and training state will be loaded when load_trained_model is called
        self.model = None
        self.state = None
        self.test_loader = None
        self.tg_scaler = None
        self.tf_scaler = None
        
        # Gradient Attribution dataframe will be loaded when load_gradient_attribution is called
        self.gradient_attribution_df = None
        
        # Transcription Factor Knockout score dataframe will be loaded when load_tf_knockout is called
        self.tf_knockout_df = None
        
        # Model forward pass predictions vs true values
        self.tg_prediction_df = None
        self.tg_true_df = None
        
        # Model evaluation metric results
        self.raw_results_df = None
        self.results_df = None
        self.per_tf_all_df = None
        self.per_tf_summary_df = None
        
        # Model evaluation metric results with ground truth
        self.df_with_ground_truth = None
        self.gt_labeled_dfs = {}
        self.auroc_auprc_scores = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # GPU memory log will be populated during training if monitor_gpu_memory is True
        self.gpu_mem_log_df = None
        
    def load_trained_model(self, checkpoint_file):
        """
        Loads a trained model given a checkpoint file and loads the corresponding model parameters

        Parameters:
        checkpoint_file (str): The name of the checkpoint file to load

        Returns:
        None
        """
        if not os.path.exists(self.model_training_dir / checkpoint_file):
            if not self.silence_warnings:
                logging.warning(f"Checkpoint file {checkpoint_file} does not exist in {self.model_training_dir}. Attempting to locate the last checkpoint.")
            checkpoint_file = self._locate_last_checkpoint()
            logging.info(f"Located checkpoint file: {checkpoint_file}")

        # Pull out architecture hyperparameters
        params = self.model_training_params
        d_model   = params.get("d_model")
        num_heads = params.get("num_heads")
        num_layers = params.get("num_layers")
        d_ff      = params.get("d_ff")
        dropout   = params.get("dropout", 0.0)
        use_shortcut   = params.get("use_shortcut", False)
        use_dist_bias  = params.get("use_dist_bias", False)
        use_motif_mask = params.get("use_motif_mask", False)

        # Load test loader
        self.test_loader = torch.load(self.model_training_dir / "test_loader.pt", weights_only=False)

        # Load the model checkpoint and state dictionary
        ckpt_path = os.path.join(self.model_training_dir, checkpoint_file)
        self.state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # Recreate the model from the training parameters
        self.model = MultiomicTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            tf_vocab_size=len(self.state["tf_scaler_mean"]),
            tg_vocab_size=len(self.state["tg_scaler_mean"]),
            use_bias=use_dist_bias,
            # use_shortcut=use_shortcut,
            # use_motif_mask=use_motif_mask,
        )

        if isinstance(self.state, dict) and "model_state_dict" in self.state:
            missing, unexpected = self.model.load_state_dict(
                self.state["model_state_dict"], strict=False
            )
            if len(missing) > 0:
                logging.info(f"Missing keys: {missing}")
            if len(unexpected) > 0:
                logging.info(f"Unexpected keys: {unexpected}")
        elif isinstance(self.state, dict) and "model_state_dict" not in self.state:
            missing, unexpected = self.model.load_state_dict(self.state, strict=False)
            if len(missing) > 0:
                logging.info(f"Missing keys: {missing}")
            if len(unexpected) > 0:
                logging.info(f"Unexpected keys: {unexpected}")
        else:
            missing, unexpected = self.model.load_state_dict(self.state, strict=False)
            if len(missing) > 0:
                logging.info(f"Missing keys: {missing}")
            if len(unexpected) > 0:
                logging.info(f"Unexpected keys: {unexpected}")

        self.model.to(self.device).eval()

        # Rebuild the scalers from the training parameters
        self.tg_scaler = SimpleScaler(
            mean=torch.as_tensor(self.state["tg_scaler_mean"], device=self.device, dtype=torch.float32),
            std=torch.as_tensor(self.state["tg_scaler_std"],  device=self.device, dtype=torch.float32),
        )
        self.tf_scaler = SimpleScaler(
            mean=torch.as_tensor(self.state["tf_scaler_mean"], device=self.device, dtype=torch.float32),
            std=torch.as_tensor(self.state["tf_scaler_std"],  device=self.device, dtype=torch.float32),
        )
    
    def locate_param_value(self, param_name):
        """
        Locate the value of a given parameter in an experiment's settings dataframe.

        Parameters:
        param_name (str): The name of the parameter to locate

        Returns:
        value (str or None): The value of the parameter if found, otherwise None
        """
        mask = self.experiment_settings_df["parameter"] == param_name
        
        if mask.any():
            return self.experiment_settings_df.loc[mask, "value"].iloc[0]
        
        return None
    
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
        common_data = Path(self.preprocessing_settings["COMMON_DATA"])
        sample_data_cache_dir = Path(self.preprocessing_settings["SAMPLE_DATA_CACHE_DIR"])
        chrom_ids = self.preprocessing_settings["CHROM_IDS"]
        subsample_seed = self.preprocessing_settings.get("SUBSAMPLE_SEED", 42)
        allowed_samples = ""

        dataset = MultiChromosomeDataset(
            data_dir=sample_data_cache_dir,
            chrom_ids=chrom_ids,
            tf_vocab_path=os.path.join(common_data, "tf_vocab.json"),
            tg_vocab_path=os.path.join(common_data, "tg_vocab.json"),
            max_cached=len(chrom_ids) if max_cached is None else max_cached,
            subset_seed=subsample_seed,
            allowed_samples=allowed_samples,
        )
        
        return dataset
    
    def create_new_model(
        self, 
        dataset,
        use_dist_bias=None,
        bias_scale=2.0, 
        d_model=None,
        num_heads=None,
        num_layers=None,
        d_ff=None,
        dropout=None,
        window_pool_size=16,
        local_rank=0, 
        rank=0, 
        world_size=1, 
        ):
        
        self._setup_cuda_ddp(local_rank, rank, world_size)

        d_model = int(self.locate_param_value("D_MODEL")) if d_model is None else d_model
        num_heads = int(self.locate_param_value("NUM_HEADS")) if num_heads is None else num_heads
        num_layers = int(self.locate_param_value("NUM_LAYERS")) if num_layers is None else num_layers
        d_ff = int(self.locate_param_value("D_FF")) if d_ff is None else d_ff
        dropout = float(self.locate_param_value("DROPOUT")) if dropout is None else dropout

        # Safer bool parsing if values come from CSV as strings
        def as_bool(x):
            if isinstance(x, bool):
                return x
            return str(x).strip().lower() in {"1", "true", "yes", "y"}

        use_dist_bias = as_bool(self.locate_param_value("USE_DIST_BIAS")) if use_dist_bias is not None else False
        
        tf_vocab_size = int(dataset.tf_ids.numel())
        tg_vocab_size = int(dataset.tg_ids.numel())

        model = MultiomicTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            tf_vocab_size=tf_vocab_size,
            tg_vocab_size=tg_vocab_size,
            use_bias=use_dist_bias,
            bias_scale=bias_scale,
            window_pool_size=window_pool_size,
        ).to(self.device)   

        return model
    
    def create_dataloaders(self, dataset, batch_size, world_size=1, rank=0):
        batch_size = self.model_training_params.get("batch_size", 32) if batch_size is None else batch_size
        
        train_loader, val_loader, test_loader = self.prepare_dataloader(
            dataset,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
        )
        
        dataloaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }
        
        return dataloaders
    
    def create_scalers(self, dataset, dataloader, max_batches: int | None = 25):
        T = int(dataset.tf_ids.numel())
        G = int(dataset.tg_ids.numel())
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
        
        scalers = {
            "tf_scaler": tf_scaler,
            "tg_scaler": tg_scaler,
        }

        return scalers

    def prepare_dataloader(self, dataset, batch_size, world_size=1, rank=0,
                        num_workers=0, pin_memory=True, seed=42, drop_last=True):
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
            train_map, batch_size=batch_size, shuffle=True, seed=seed, drop_last=False
        )
        base_val_bs = InterleavedChromBatchSampler(
            val_map, batch_size=batch_size, shuffle=False, seed=seed, drop_last=False
        )
        base_test_bs = InterleavedChromBatchSampler(
            test_map, batch_size=batch_size, shuffle=False, seed=seed, drop_last=False
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
            persistent_workers=num_workers > 0,
        )
        val_loader = DataLoader(
            dataset,
            batch_sampler=val_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        test_loader = DataLoader(
            dataset,
            batch_sampler=test_bs,
            collate_fn=MultiChromosomeDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )
        return train_loader, val_loader, test_loader


    
    def train(
        self,
        model,
        train_loader,
        val_loader=None,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        verbose: bool = True,
        validate_every: int = 1,
        use_amp: bool = True,
        grad_accum_steps = 4,
        monitor_gpu_memory: bool = False,
    ):
        """
        Simple single-GPU training loop with optional AMP.
        No DDP, no checkpointing.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        tf_scaler = getattr(self, "tf_scaler", None)
        tg_scaler = getattr(self, "tg_scaler", None)

        amp_enabled = use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        gpu_mem_log = []
        
        for epoch in range(num_epochs):
            if monitor_gpu_memory and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            
            epoch_start_time = time.time()
            model.train()
            total_loss = 0.0
            n_batches = 0

            optimizer.zero_grad(set_to_none=True)
            
            for i, batch in tqdm(
                enumerate(train_loader), 
                total=len(train_loader), 
                desc=f"Epoch {epoch+1}/{num_epochs}", 
                leave=False,
                ncols=100,
                ):
                atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, _ = batch

                atac_wins = atac_wins.to(device, non_blocking=True)
                tf_tensor = tf_tensor.to(device, non_blocking=True)
                targets   = targets.to(device, non_blocking=True)
                bias      = bias.to(device, non_blocking=True)
                tf_ids    = tf_ids.to(device, non_blocking=True)
                tg_ids    = tg_ids.to(device, non_blocking=True)

                if tf_scaler is not None:
                    tf_tensor = tf_scaler.transform(tf_tensor, tf_ids)
                if tg_scaler is not None:
                    targets = tg_scaler.transform(targets, tg_ids)

                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    preds = model(
                        atac_wins,
                        tf_tensor,
                        tf_ids=tf_ids,
                        tg_ids=tg_ids,
                        bias=bias,
                    )
                    loss = loss_fn(preds, targets)

                loss_for_backward = loss / grad_accum_steps
                scaler.scale(loss_for_backward).backward()

                if (i + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    if monitor_gpu_memory and device.type == "cuda":
                        gpu_mem_log.append({
                            "epoch": epoch,
                            "step": i,
                            "time": time.time(),
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                        })

            # ---- epoch summary ----
            if monitor_gpu_memory and device.type == "cuda":
                gpu_mem_log.append({
                    "epoch": epoch,
                    "step": -1,
                    "type": "epoch_summary",
                    "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                    "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
                })
                
            # handle leftover batches at end of epoch
            if n_batches % grad_accum_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            avg_loss = total_loss / max(1, n_batches)
            epoch_end_time = time.time()

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_loss:.4f} | Time: {epoch_end_time - epoch_start_time:.1f}s")

            if val_loader is not None and ((epoch + 1) % validate_every == 0):
                model.eval()
                val_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for batch in val_loader:
                        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, _ = batch

                        atac_wins = atac_wins.to(device, non_blocking=True)
                        tf_tensor = tf_tensor.to(device, non_blocking=True)
                        targets   = targets.to(device, non_blocking=True)
                        bias      = bias.to(device, non_blocking=True)
                        tf_ids    = tf_ids.to(device, non_blocking=True)
                        tg_ids    = tg_ids.to(device, non_blocking=True)

                        if tf_scaler is not None:
                            tf_tensor = tf_scaler.transform(tf_tensor, tf_ids)
                        if tg_scaler is not None:
                            targets = tg_scaler.transform(targets, tg_ids)

                        with torch.autocast(device_type=device.type, enabled=amp_enabled):
                            preds = model(
                                atac_wins,
                                tf_tensor,
                                tf_ids=tf_ids,
                                tg_ids=tg_ids,
                                bias=bias,
                            )
                            loss = loss_fn(preds, targets)

                        val_loss += loss.detach().item()
                        val_batches += 1

                val_loss /= max(1, val_batches)

                if verbose:
                    print(f"                Val Loss: {val_loss:.4f}")
    
        self.gpu_mem_log_df = pd.DataFrame(gpu_mem_log)
    
        return model

    def train_timed(
        self,
        model,
        train_loader,
        val_loader=None,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        verbose: bool = True,
        validate_every: int = 1,
        max_batches: int | None = None,
        use_amp: bool = True,
        grad_accum_steps=4,
        monitor_gpu_memory: bool = False,
        profile_batches: bool = False,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        tf_scaler = getattr(self, "tf_scaler", None)
        tg_scaler = getattr(self, "tg_scaler", None)

        amp_enabled = use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            threshold=1e-4,
            min_lr=1e-7,
        )

        gpu_mem_log = []
        batch_profile_log = []
        epoch_log = []

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            model.train()
            total_loss = 0.0
            n_batches = 0

            optimizer.zero_grad(set_to_none=True)

            data_iter = iter(train_loader)
            i = 0

            try:
                total_batches = max_batches if max_batches is not None else len(train_loader)
            except TypeError:
                total_batches = max_batches

            with tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100, leave=False) as pbar:
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

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t2 = time.perf_counter()

                    atac_wins = atac_wins.to(device, non_blocking=True)
                    tf_tensor = tf_tensor.to(device, non_blocking=True)
                    targets   = targets.to(device, non_blocking=True)
                    bias      = bias.to(device, non_blocking=True) if bias is not None else None
                    tf_ids    = tf_ids.to(device, non_blocking=True)
                    tg_ids    = tg_ids.to(device, non_blocking=True)

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t3 = time.perf_counter()

                    if tf_scaler is not None or tg_scaler is not None:
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t4 = time.perf_counter()

                        if tf_scaler is not None:
                            tf_tensor = tf_scaler.transform(tf_tensor, tf_ids)
                        if tg_scaler is not None:
                            targets = tg_scaler.transform(targets, tg_ids)

                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t5 = time.perf_counter()
                    else:
                        t4 = t5 = time.perf_counter()

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t6 = time.perf_counter()

                    with torch.autocast(device_type=device.type, enabled=amp_enabled):
                        preds = model(
                            atac_wins,
                            tf_tensor,
                            tf_ids=tf_ids,
                            tg_ids=tg_ids,
                            bias=bias,
                        )
                        loss = loss_fn(preds, targets)

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t7 = time.perf_counter()

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t8 = time.perf_counter()

                    loss_for_backward = loss / grad_accum_steps
                    scaler.scale(loss_for_backward).backward()

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t9 = time.perf_counter()

                    stepped = False
                    if (i + 1) % grad_accum_steps == 0:
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        t10 = time.perf_counter()

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        stepped = True

                        if device.type == "cuda":
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
                        batch_profile_df = pd.DataFrame(batch_profile_log)
                        print(batch_profile_df[["loader_s", "transfer_s", "forward_s", "backward_s", "optim_s"]].mean())

                    if monitor_gpu_memory and device.type == "cuda":
                        gpu_mem_log.append({
                            "epoch": epoch,
                            "step": i,
                            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
                        })

                    total_loss += loss.detach().item()
                    n_batches += 1
                    i += 1

                    pbar.update(1)

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

            if val_loader is not None and ((epoch + 1) % validate_every == 0):
                avg_val_mse_scaled, avg_val_mse_unscaled, r2_s, r2_u = self._validate_simple(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    amp_enabled=amp_enabled,
                )
                scheduler.step(avg_val_mse_unscaled)

            current_lr = optimizer.param_groups[0]["lr"]

            epoch_log.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_mse_unscaled": avg_val_mse_unscaled,
                "r2_unscaled": r2_u,
                "r2_scaled": r2_s,
                "lr": current_lr,
                "epoch_time_s": epoch_end_time - epoch_start_time,
            })

            if verbose:
                if avg_val_mse_unscaled is not None:
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Train Loss: {avg_loss:.4f} | "
                        f"Val MSE: {avg_val_mse_unscaled:.4f} | "
                        f"R2 (Unscaled): {r2_u:.3f} | "
                        f"R2 (Scaled): {r2_s:.3f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Time: {epoch_end_time - epoch_start_time:.1f}s"
                    )
                else:
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Train Loss: {avg_loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Time: {epoch_end_time - epoch_start_time:.1f}s"
                    )

        self.gpu_mem_log_df = pd.DataFrame(gpu_mem_log)
        self.batch_profile_df = pd.DataFrame(batch_profile_log)
        self.epoch_log_df = pd.DataFrame(epoch_log)

        return model
    
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
        selected_experiment_dir,
        model,
        test_loader,
        tf_scaler,
        tg_scaler,
        tf_names,
        tg_names,
        device,
        use_amp,
        max_batches: int = None,
        save_every_n_batches: int = 20,
        max_tgs_per_batch = 128,
        chunk_size = 64,
        zero_tf_expr: bool = False,
    ):

        T_total = len(tf_names)
        G_total = len(tg_names)

        # Creates empty tensors to accumulate gradients across batches. The shape is [TF total, Genes total]
        grad_sum = torch.zeros(T_total, G_total, device=device, dtype=torch.float32)
        grad_count = torch.zeros_like(grad_sum)

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

                if zero_tf_expr:
                    tf_tensor_chunk = torch.zeros_like(tf_tensor, device=device, requires_grad=True)
                else:
                    tf_tensor_chunk = tf_tensor.detach().clone().requires_grad_(True)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    tf_scaled = tf_scaler.transform(tf_tensor_chunk, tf_ids) if tf_scaler is not None else tf_tensor_chunk
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
        
        return df_long, batch_grad_dfs
    
    def load_grn_old(self, method="gradient attribution", zscore_method="median_mad"):
        """
        Loads a GRN dataframe given a method. The dataframe contains the source transcription factor, target gene, and score.

        Parameters:
        method (str): The method to use. Must be 'Gradient Attribution' or 'TF Knockout'.

        Returns:
        pd.DataFrame: The GRN dataframe containing the source transcription factor, target gene, and score.
        """
        method = method.lower()
        
        assert method in ["gradient attribution", "tf knockout"], \
            f"Invalid GRN method {method}. Must be 'Gradient Attribution' or 'TF Knockout'."        
        
        if method == "gradient attribution":
            score_file = self.model_training_dir / "tf_tg_grad_attribution.npy"
            
        elif method == "tf knockout":
            score_file = self.model_training_dir / "tf_tg_fullmodel_knockout.npy"
            
        assert score_file.exists(), f"GRN file for method {method} {score_file} does not exist."
        
        score = np.load(score_file).astype(np.float32)
        assert score.shape == (len(self.tf_names), len(self.tg_names))

        score = np.nan_to_num(score, nan=0.0)
        score_abs = np.abs(score)

        # Calculate per-TF robust z-score
        if zscore_method == "median_mad":
            median_val = np.median(score_abs, axis=1, keepdims=True)
            mad = np.median(np.abs(score_abs - median_val), axis=1, keepdims=True) + 1e-6
            score = (score_abs - median_val) / mad
            
        elif zscore_method == "mean_std":
            mean_val = np.mean(score_abs, axis=1, keepdims=True)
            std_val = np.std(score_abs, axis=1, keepdims=True) + 1e-6
            score = (score_abs - mean_val) / std_val
        
        score = np.clip(score, 0, None)
        
        log1p_score = np.log1p(score)
        log1p_score = np.clip(log1p_score, 1e-12, None)  # ensure > 0

        score_log = np.log10(score_abs)
        
        T, G = score.shape
        tf_idx, tg_idx = np.meshgrid(np.arange(T), np.arange(G), indexing="ij")
        
        tf_idx = tf_idx.ravel()
        tg_idx = tg_idx.ravel()

        df = pd.DataFrame({
            "Source": np.asarray(self.tf_names, dtype=object)[tf_idx],
            "Target": np.asarray(self.tg_names, dtype=object)[tg_idx],
            "Score": score.ravel(),
        })
        
        df["Source"] = df["Source"].astype(str).str.upper()
        df["Target"] = df["Target"].astype(str).str.upper()
        
        # Removes the 1e-12 pseudonumbers used for safe log transformation (originally scores of 0)
        df = df[df["Score"] > 0]
        
        return df
    
    def load_grn(self, method="gradient attribution"):
        method = method.lower()
        
        assert method in ["gradient attribution", "tf knockout"], \
            f"Invalid GRN method {method}. Must be 'Gradient Attribution' or 'TF Knockout'."   
        
        if method == "gradient attribution":
            df_file = self.model_training_dir / "gradient_attribution_raw.parquet"
        elif method == "tf knockout":
            df_file = self.model_training_dir / "tf_knockout_raw.parquet"
        else:
            raise ValueError(f"Invalid method {method}. Must be 'Gradient Attribution' or 'TF Knockout'.")
        
        assert df_file.exists(), \
            f"GRN file for method {method} {df_file} does not exist."
        
        df_wide = pd.read_parquet(df_file) 
        
        df = (
            df_wide
            .reset_index(names="Source")
            .melt(id_vars="Source", var_name="Target", value_name="Score")
        )
        
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
        if self.model is None:
            self.load_trained_model("trained_model.pt")

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
                    bias=bias#, motif_mask=motif_mask,
                    #return_shortcut_contrib=False,
                )
                if type(out) == tuple:
                    out = out[0]

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
        
        def _compare_pred_true(pred_df, true_df):
            pred_df["mean_expr"] = pred_df.mean(axis=1).values
            pred_df["std_expr"] = pred_df.std(axis=1).values
            predicted_expr = pred_df[["mean_expr", "std_expr"]]

            true_df["mean_expr"] = true_df.mean(axis=1).values
            true_df["std_expr"] = true_df.std(axis=1).values
            true_expr = true_df[["mean_expr", "std_expr"]]

            merged = predicted_expr.merge(
                true_expr,
                left_index=True,
                right_index=True,
                suffixes=("_pred", "_true")
            )

            merged["diff"] = merged["mean_expr_pred"] - merged["mean_expr_true"]
            
            return merged
        
        pred_vs_true_expr_comparison_df = _compare_pred_true(pred_df, true_df)
        
        self.tg_prediction_df = pred_df
        self.tg_true_df = true_df

        return pred_df, true_df, pred_vs_true_expr_comparison_df

    def visualize_model_structure(self):
        if self.model is None:
            self.load_trained_model("trained_model.pt")

        return self.model.module
    
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
    
    def plot_train_val_loss(self):
        fig = plt.figure(figsize=(6, 5))
        df = self.training_df.iloc[5:, :]
        
        plt.plot(df["Epoch"], df["Train MSE"], label="Train MSE", linewidth=2)
        plt.plot(df["Epoch"], df["Val MSE"], label="Val MSE", linewidth=2)
        plt.title(f"Train Val Loss Curves", fontsize=17)
        plt.xlabel("Epoch", fontsize=17)
        plt.ylabel("Loss", fontsize=17)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.legend(fontsize=15)
        plt.tight_layout()
        
        return fig
    
    def plot_train_correlation(self):
        fig = plt.figure(figsize=(6, 5))
        
        df = self.training_df
        plt.plot(df.index, df["R2_u"], linewidth=2, label=f"Best R2 (unscaled) = {df['R2_u'].max():.2f}")
        plt.plot(df.index, df["R2_s"], linewidth=2, label=f"Best R2 (scaled)     = {df['R2_s'].max():.2f}")

        plt.title(f"TG Expression R2 Across Training", fontsize=17)
        plt.ylim((0,1))
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.xlabel("Epoch", fontsize=17)
        plt.ylabel("R2", fontsize=17)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        return fig
    
    def plot_gpu_usage(self, smooth=None):
        """
        align_to_common_duration: if True, truncate each run to the shortest duration so curves end together.
        smooth: optional int window (in seconds) for a centered rolling mean on memory (e.g., smooth=5).
        """
        if self.gpu_usage_df is None:
            if not self.silence_warnings:
                logging.warning(f"GPU usage data not available for {self.experiment_name}. Cannot plot GPU usage.")
            return
        
        fig, ax = plt.subplots(figsize=(7,4))

        if smooth and smooth > 1:
            self.gpu_usage_df["memory_used_gib"] = self.gpu_usage_df["memory_used_gib"].rolling(smooth, center=True, min_periods=1).mean()

        ax.plot(self.gpu_usage_df["elapsed_hr"], self.gpu_usage_df["memory_used_gib"], label=f"Avg GPU RAM Used", linewidth=2)

        ax.axhline(self.gpu_memory_limit_gib, linestyle="--", label=f"Max RAM")
        ax.set_ylabel("GiB")
        ax.set_xlabel("Minutes since start")
        ax.set_ylim(0, self.gpu_memory_limit_gib + 5)
        ax.xaxis.set_major_locator(MultipleLocator(1))  # tick every 1 hour
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
        ax.set_xlabel("Hours since start")

        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )
        ax.set_title(
            f"Average GPU Memory During Training"
        )
        plt.tight_layout()
        plt.show()
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
            self.tg_prediction_df, self.tg_true_df, _ = self.run_forward_pass(num_batches=num_batches)
        
        x = self.tg_true_df.median(axis=1).values
        y = self.tg_prediction_df.median(axis=1).values
        
        # mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        # x = x[mask]
        # y = y[mask]
        
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
            
            # mean_val = y.mean()
            # ax.scatter(
            #     i, mean_val,
            #     color="white",
            #     edgecolor="k",
            #     s=30,
            #     zorder=4,
            # )
            
            # # Annotate the mean value above the mean point
            # ax.text(i, y.max() + 0.015, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=12)

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
    
    def _locate_last_checkpoint(self):
        """
        Locate the checkpoint_<N>.pt file with the largest N in self.model_training_dir.
        
        Returns:
            str: The name of the checkpoint file (not full path)
        """
        checkpoint_files = sorted(self.model_training_dir.glob("checkpoint_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {self.model_training_dir}")
        last_checkpoint = checkpoint_files[-1]
        return last_checkpoint.name
    
    def _load_json(self, path: Path) -> dict:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    
    def _load_run_params(self) -> dict:
        """
        Load the run parameters saved during model training.

        Parameters:
        None

        Returns:
        dict: The run parameters saved during model training
        """
        run_params_path = self.model_training_dir / "run_params.json"
        if not run_params_path.exists():
            raise FileNotFoundError(f"Run parameters file {run_params_path} does not exist.")
        
        with open(run_params_path, "r") as f:
            run_params = json.load(f)
        
        return run_params
    
    def _format_gpu_usage_file(self):
        """
        Format the GPU usage log file into a more usable format.

        The function assumes that the input file contains columns for "timestamp", "memory.used [MiB]", and "memory.total [MiB]" and
        that the timestamp is in seconds since the epoch.

        memory usage per second, and a float containing the total memory available on the GPU in GiB.

        """
        try:
            gpu_usage_file = Path(f"{self.experiment_dir}/{self.experiment_name}") / "logs" / "gpu_usage.csv"

            gpu_usage_df = pd.read_csv(gpu_usage_file)
            gpu_usage_df.columns = gpu_usage_df.columns.str.strip()
            gpu_usage_df["timestamp"] = pd.to_datetime(gpu_usage_df["timestamp"], errors="coerce")
            gpu_usage_df["tsec"] = gpu_usage_df["timestamp"].dt.floor("s")

            gpu_usage_df["memory_used_gib"]  = gpu_usage_df["memory.used [MiB]"].astype(str).str.extract(r"(\d+)").astype(float) / 1024
            gpu_usage_df["memory_total_gib"] = gpu_usage_df["memory.total [MiB]"].astype(str).str.extract(r"(\d+)").astype(float) / 1024

            t0 = gpu_usage_df["tsec"].min()
            gpu_usage_df["elapsed_s"] = (gpu_usage_df["tsec"] - t0).dt.total_seconds().astype(int)
            gpu_usage_df["elapsed_min"] = gpu_usage_df["elapsed_s"] / 60.0
            gpu_usage_df["elapsed_hr"] = gpu_usage_df["elapsed_s"] / 3600.0
            

            # mean per second, then carry minutes as a column
            mean_per_sec_df = (
                gpu_usage_df.groupby("elapsed_s", as_index=False)["memory_used_gib"]
                .mean()
                .sort_values("elapsed_s")
            )
            mean_per_sec_df["elapsed_min"] = mean_per_sec_df["elapsed_s"] / 60.0
            mean_per_sec_df["elapsed_hr"] = mean_per_sec_df["elapsed_s"] / 3600.0

            gpu_memory_limit_gib = float(gpu_usage_df["memory_total_gib"].iloc[0])
            return gpu_usage_df, mean_per_sec_df, gpu_memory_limit_gib
        
        except FileNotFoundError:
            if not self.silence_warnings:
                logging.warning(f"WARNING: GPU usage file not found for {self.experiment_name}. GPU usage plotting will be unavailable.")
            return None, None, None
        
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
    
    