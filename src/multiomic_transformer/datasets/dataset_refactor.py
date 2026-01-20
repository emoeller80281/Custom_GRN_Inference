import os
import json
import numpy as np
import pandas as pd
import torch
from typing import Optional, Union, Sequence
from torch.utils.data import Dataset, Sampler
from pathlib import Path
import logging
from collections import OrderedDict
from dataclasses import dataclass

@dataclass
class SimpleScaler:
    """
    Class to scale TF/TG data when there is a variable number of genes per batch. 
    
    Different chromosomes may have different subsets of TFs/TGs, so we need to
    be able to slice the global mean/std to match the current batch's genes. 
    """
    mean: torch.Tensor  # shape [D] on the correct device
    std:  torch.Tensor  # shape [D] on the correct device

    def transform(self, x: torch.Tensor, ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        eps = 1e-6
        
        # Slice mean/std if ids are provided
        if ids is not None:
            mu  = self.mean.index_select(0, ids)
            sig = self.std.index_select(0, ids).clamp_min(eps)
        else:
            # fallback: whole-dim scaling
            mu, sig = self.mean, self.std.clamp_min(eps)

        # broadcast over batch/time dims (subtract along last axis)
        return (x - mu) / sig
    
    def inverse_transform(self, x: torch.Tensor, ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        eps = 1e-6
        if ids is not None:
            mu  = self.mean.index_select(0, ids)
            sig = self.std.index_select(0, ids).clamp_min(eps)
        else:
            mu, sig = self.mean, self.std.clamp_min(eps)

        # ensure same device as x, rely on broadcasting over leading dims
        mu  = mu.to(x.device)
        sig = sig.to(x.device)
        return x * sig + mu

@torch.no_grad()
def fit_simple_scalers(
    train_loader,
    T_expected: int,
    G_expected: int,
    device_for_reduce: Union[str, torch.device] = "cuda",
    use_ddp_reduce: bool = False,
):
    """
    Fits SimpleScaler objects for TF and TG data from a DataLoader. 
    
    Different chromosomes may have different subsets of TFs/TGs, so we need to
    accumulate sums, squared sums, and counts for each TF/TG across batches,
    then compute means and stds at the end.
    """
    # global accumulators on CPU (memory-light, avoids GPU growth)
    tf_sum   = torch.zeros(T_expected, dtype=torch.float64, device="cpu")
    tf_sqsum = torch.zeros(T_expected, dtype=torch.float64, device="cpu")
    tf_count = torch.zeros(T_expected, dtype=torch.float64, device="cpu")

    tg_sum   = torch.zeros(G_expected, dtype=torch.float64, device="cpu")
    tg_sqsum = torch.zeros(G_expected, dtype=torch.float64, device="cpu")
    tg_count = torch.zeros(G_expected, dtype=torch.float64, device="cpu")

    for batch in train_loader:
        # unpack like your collate_fn returns
        atac_wins, tf_tensor, tg_tensor, bias, tf_ids, tg_ids, motif_mask = batch
        
        # move minimal things to GPU to sum efficiently, then bring back to CPU
        tf_tensor = tf_tensor.to(device_for_reduce, non_blocking=True)   # [B,T_eval]
        tg_tensor = tg_tensor.to(device_for_reduce, non_blocking=True)   # [B,G_eval]
        tf_ids    = tf_ids.to(device_for_reduce, non_blocking=True)      # [T_eval] global ids 0..T'-1
        tg_ids    = tg_ids.to(device_for_reduce, non_blocking=True)      # [G_eval] global ids 0..G'-1

        # per-batch reductions over batch dimension
        tf_batch_sum   = tf_tensor.sum(dim=0)                # [T_eval]
        tf_batch_sqsum = (tf_tensor**2).sum(dim=0)           # [T_eval]
        tf_batch_cnt   = torch.full_like(tf_batch_sum, fill_value=tf_tensor.shape[0], dtype=torch.float64)

        tg_batch_sum   = tg_tensor.sum(dim=0)                # [G_eval]
        tg_batch_sqsum = (tg_tensor**2).sum(dim=0)           # [G_eval]
        tg_batch_cnt   = torch.full_like(tg_batch_sum, fill_value=tg_tensor.shape[0], dtype=torch.float64)

        # move to CPU for index_add_
        tf_ids_cpu = tf_ids.to("cpu")
        tg_ids_cpu = tg_ids.to("cpu")

        tf_sum.index_add_(0, tf_ids_cpu, tf_batch_sum.to("cpu", dtype=torch.float64))
        tf_sqsum.index_add_(0, tf_ids_cpu, tf_batch_sqsum.to("cpu", dtype=torch.float64))
        tf_count.index_add_(0, tf_ids_cpu, tf_batch_cnt.to("cpu", dtype=torch.float64))

        tg_sum.index_add_(0, tg_ids_cpu, tg_batch_sum.to("cpu", dtype=torch.float64))
        tg_sqsum.index_add_(0, tg_ids_cpu, tg_batch_sqsum.to("cpu", dtype=torch.float64))
        tg_count.index_add_(0, tg_ids_cpu, tg_batch_cnt.to("cpu", dtype=torch.float64))

    # Optionally all-reduce across DDP ranks (sum the accumulators)
    if use_ddp_reduce and torch.distributed.is_initialized():
        for acc in (tf_sum, tf_sqsum, tf_count, tg_sum, tg_sqsum, tg_count):
            acc_cuda = acc.to(device_for_reduce)
            torch.distributed.all_reduce(acc_cuda, op=torch.distributed.ReduceOp.SUM)
            acc.copy_(acc_cuda.to("cpu"))

    # Calculate the TF means/stds
    tf_mean = torch.zeros(T_expected, dtype=torch.float32)
    tf_std  = torch.ones(T_expected,  dtype=torch.float32)
    mask_tf = tf_count > 0
    tf_mean[mask_tf] = (tf_sum[mask_tf] / tf_count[mask_tf]).to(torch.float32)
    tf_var = torch.zeros_like(tf_mean)
    tf_var[mask_tf] = (tf_sqsum[mask_tf] / tf_count[mask_tf]).to(torch.float32) - tf_mean[mask_tf]**2
    tf_std = torch.sqrt(torch.clamp(tf_var, min=1e-6))

    # Calculate the TG means/stds
    tg_mean = torch.zeros(G_expected, dtype=torch.float32)
    tg_std  = torch.ones(G_expected,  dtype=torch.float32)
    mask_tg = tg_count > 0
    tg_mean[mask_tg] = (tg_sum[mask_tg] / tg_count[mask_tg]).to(torch.float32)
    tg_var = torch.zeros_like(tg_mean)
    tg_var[mask_tg] = (tg_sqsum[mask_tg] / tg_count[mask_tg]).to(torch.float32) - tg_mean[mask_tg]**2
    tg_std = torch.sqrt(torch.clamp(tg_var, min=1e-6))

    return SimpleScaler(tf_mean, tf_std), SimpleScaler(tg_mean, tg_std)

    

class IndexedChromBucketBatchSampler(Sampler):
    """
    Batches over a provided {chrom: [indices]} mapping.

    Guarantees:
      - Only uses the given indices (so you can pre-split train/val/test).
      - Each batch contains indices from a single chromosome (shape-safe).
      - Optionally shuffles chromosomes and indices per epoch.
    """
    def __init__(self, chrom_to_indices, batch_size, shuffle=True, seed=0):
        self.chrom_to_indices = {
            chrom: list(idxs)
            for chrom, idxs in chrom_to_indices.items()
            if len(idxs) > 0
        }
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        """
        Yields batches of indices from a single chromosome.

        Each batch is a contiguous block of indices from the same chromosome.
        The order of chromosomes and indices is shuffled randomly each epoch if `shuffle=True`.
        """
        rng = np.random.RandomState(self.seed + self.epoch)

        chroms = list(self.chrom_to_indices.keys())
        if self.shuffle:
            rng.shuffle(chroms)
        
        # Gets a batch of indices from each chromosome
        # Keeps windows and TGs aligned within each batch
        for chrom in chroms:
            idxs = self.chrom_to_indices[chrom][:]
            if self.shuffle:
                rng.shuffle(idxs)
            n = len(idxs)
            for s in range(0, n, self.batch_size):
                batch = idxs[s:s + self.batch_size]
                if batch:
                    yield batch

    def __len__(self):
        # Returns the number of batches per epoch
        total = 0
        for _, idxs in self.chrom_to_indices.items():
            n = len(idxs)
            total += (n + self.batch_size - 1) // self.batch_size
        return total

class DistributedBatchSampler(torch.utils.data.Sampler):
    """
    Shards a *batch sampler* across DDP ranks (each rank gets a disjoint
    subset of batches). This avoids duplicating work without requiring
    per-sample DistributedSampler, which would break chrom-homogeneous batches.
    """
    def __init__(self, batch_sampler, world_size, rank, drop_last=True):
        self.batch_sampler = batch_sampler
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)

    def set_epoch(self, epoch: int):
        # Sets the epoch for the underlying batch sampler
        # (Used for setting the epoch in IndexedChromBucketBatchSampler for shuffling)
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)

    def __iter__(self):
        batches = list(self.batch_sampler)
        if self.drop_last:
            size = (len(batches) // self.world_size) * self.world_size
            batches = batches[:size]
        for i in range(self.rank, len(batches), self.world_size):
            yield batches[i]

    def __len__(self):
        base = len(self.batch_sampler)
        return (base // self.world_size) if self.drop_last else (base + self.world_size - 1) // self.world_size


class MultiChromosomeDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        chrom_ids,
        tf_vocab_path: Optional[Union[str, Path]] = None,
        tg_vocab_path: Optional[Union[str, Path]] = None,
        sample_name: Optional[str] = None,
        max_cached: int = 2,
        subset_seed: int = 42,
        allowed_samples: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.chrom_ids = list(chrom_ids)
        self.tf_vocab_path = Path(tf_vocab_path) if tf_vocab_path is not None else None
        self.tg_vocab_path = Path(tg_vocab_path) if tg_vocab_path is not None else None
        self.sample_name = sample_name
        self.max_cached = max_cached
        self.subset_seed = subset_seed
        self.allowed_samples = (
            set(str(s) for s in allowed_samples) if allowed_samples is not None else None
        )

        self._tf_name2id = self._load_vocab_dict(self.tf_vocab_path)
        self._tg_name2id = self._load_vocab_dict(self.tg_vocab_path)

        if self._tf_name2id is None or self._tg_name2id is None:
            raise ValueError("tf_vocab_path and tg_vocab_path must be provided and exist")

        # Public aliases used by training scripts/utilities
        self.tf_name2id = self._tf_name2id
        self.tg_name2id = self._tg_name2id
        
        # Track the gene IDs and names
        self.tf_ids = torch.arange(len(self._tf_name2id), dtype=torch.long)
        self.tg_ids = torch.arange(len(self._tg_name2id), dtype=torch.long)
        
        self.tf_names = [n for n, _ in sorted(self._tf_name2id.items(), key=lambda kv: kv[1])]
        self.tg_names = [n for n, _ in sorted(self._tg_name2id.items(), key=lambda kv: kv[1])]
        
        self._cell_idx: Optional[np.ndarray] = None
        
        # Small LRU cache of per-chrom datasets (Keeps track of windows and TGs for each chromosome)
        self._cache: OrderedDict[str, MultiomicTransformerDataset] = OrderedDict()
        
        tf_global = torch.load(self.data_dir / "tf_tensor_all.pt", map_location="cpu") if self.tf_vocab_path is not None else None
        full_num_cells = int(tf_global.shape[1]) if tf_global is not None else 0
        
        assert full_num_cells > 0, "No cells found in tf_tensor_all.pt"
        
        if self.allowed_samples:
            base_idx = self._load_allowed_samples(full_num_cells)
        else:
            base_idx = np.arange(full_num_cells, dtype=int)
            
        self._num_cells = int(base_idx.size)
        self._cell_idx = base_idx
        
        # offsets tell us where each chromosome's indices start in the concatenated space
        self._offsets = []
        running = 0
        for _ in self.chrom_ids:
            self._offsets.append(running)
            running += self._num_cells
        self._length = running
        
        self._windows_per_chrom = {}
        _total_windows = 0
        self.metacell_names = None
        for cid in self.chrom_ids:
            ds = self._load_chrom(cid)   # uses cache + applies global sub-vocab + max_windows_per_chrom
            w = int(ds.num_windows)
            self._windows_per_chrom[cid] = w
            _total_windows += w
            self._evict_if_needed()
            
            # Check metacell names consistency between chromosome datasets
            if self.metacell_names is None:
                self.metacell_names = list(ds.metacell_names)  # keep order
            else:
                if list(ds.metacell_names) != self.metacell_names:
                    logging.warning(f"metacell_names differ on {cid}; using the first set.")
                    
        self.num_windows = int(_total_windows)
            
    def _evict_if_needed(self):
        """
        Evicts the oldest item in the cache if it has reached its maximum size.

        This ensures that the cache does not grow indefinitely, and that the most recently used items are always kept.
        """
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)  # evict oldest
            
    def __len__(self):
        return self._length
    
    def _locate(self, idx: int):
        """
        Map a global index into (chrom_id, local_index) using the
        precomputed offsets. Works for both pseudobulk and fine-tune
        modes as long as self._offsets / self._length are consistent.
        """
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)

        lo, hi = 0, len(self.chrom_ids) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self._offsets[mid]
            end = self._offsets[mid + 1] if mid + 1 < len(self._offsets) else self._length

            if start <= idx < end:
                chrom = self.chrom_ids[mid]
                local = idx - start
                return chrom, local
            elif idx < start:
                hi = mid - 1
            else:
                lo = mid + 1

        raise IndexError(idx)
    
    def __getitem__(self, idx):
        chrom, local = self._locate(idx)
        ds = self._load_chrom(chrom)
        return ds[local]

    @staticmethod
    def collate_fn(batch):
        # Reuses the per-chrom collate (expects within-batch consistent W, G, T)
        return MultiomicTransformerDataset.collate_fn(batch)
    
    @staticmethod
    def standardize_name(name: str) -> str:
        if not isinstance(name, str):
            return name
        return name.upper()

    def _load_vocab_dict(self, path: Optional[Path]) -> Optional[dict]:
        if not path or not os.path.exists(path):
            return None
        with open(path) as f:
            obj = json.load(f)
        # support either {"name_to_id": {...}} or flat dict
        raw = obj.get("name_to_id", obj)
        return {self.standardize_name(k): int(v) for k, v in raw.items()}
        
    def _load_allowed_samples(self, full_num_cells: int) -> np.ndarray:
        metacell_path = self.data_dir / "metacell_names.json"
        try:
            with open(metacell_path) as f:
                all_names = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"allowed_samples was provided, but metacell_names.json is missing at {metacell_path}"
            )

        if not isinstance(all_names, list) or len(all_names) != full_num_cells:
            logging.warning(
                f"metacell_names.json length "
                f"{len(all_names) if isinstance(all_names, list) else 'N/A'} "
                f"!= num_cells {full_num_cells}; sample-based filtering may be misaligned."
            )

        # vectorized parsing and filtering
        names = pd.Series(all_names, dtype=str)
        sample_tags = names.str.rsplit(".", n=1).str[0]
        mask = sample_tags.isin(self.allowed_samples)
        keep = np.flatnonzero(mask)

        if len(keep) == 0:
            raise ValueError(
                f"No metacells matched allowed_samples={sorted(self.allowed_samples)} "
                f"in metacell_names.json."
            )

        base_idx = np.array(sorted(keep), dtype=int)
        
        return base_idx

    def _load_chrom(self, chrom_id: str) -> "MultiomicTransformerDataset":
        # If the chromosome is already cached, move it to the end (most recently used) and return it
        if chrom_id in self._cache:
            ds = self._cache.pop(chrom_id)
            self._cache[chrom_id] = ds
            return ds

        # If the chromosome isn't in the cache, create a new dataset
        else:
            
            ds = MultiomicTransformerDataset(
                data_dir=self.data_dir,
                chrom_id=chrom_id,
                tf_vocab_path=self.tf_vocab_path,
                tg_vocab_path=self.tg_vocab_path,
                sample_name=self.sample_name,
                cell_idx=self._cell_idx,
            )
            self._cache[chrom_id] = ds
            self._evict_if_needed()
            
            return ds


class MultiomicTransformerDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        chrom_id: str,
        tf_vocab_path: Optional[Path] = None,
        tg_vocab_path: Optional[Path] = None,
        sample_name: Optional[str] = None,
        cell_idx: Optional[np.ndarray] = None,
        subset_seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.chrom_id = chrom_id
        self.sample_name = sample_name
        self._subset_rng = np.random.RandomState(subset_seed)
        self._cell_idx: Optional[np.ndarray] = None

        chrom_dir = self.data_dir / chrom_id
        if not chrom_dir.is_dir():
            raise FileNotFoundError(f"Chromosome directory missing: {chrom_dir}")

        # --- load vocab dicts ---
        self.tf_name2id, self.tg_name2id = None, None
        if tf_vocab_path and os.path.exists(tf_vocab_path):
            with open(tf_vocab_path) as f: obj = json.load(f)
            self.tf_name2id = {self.standardize_name(k): v for k, v in obj.get("name_to_id", obj).items()}
        if tg_vocab_path and os.path.exists(tg_vocab_path):
            with open(tg_vocab_path) as f: obj = json.load(f)
            self.tg_name2id = {self.standardize_name(k): v for k, v in obj.get("name_to_id", obj).items()}
            
        tf_path   = self.data_dir / f"tf_tensor_all.pt"
        tg_path   = chrom_dir / f"tg_tensor_all_{chrom_id}.pt"
        atac_path = chrom_dir / f"atac_window_tensor_all_{chrom_id}.pt"
        window_map_path = chrom_dir / f"window_map_{chrom_id}.json"
        dist_bias_path  = chrom_dir / f"dist_bias_{chrom_id}.pt"
        tf_ids_path     = self.data_dir / f"tf_ids.pt"
        tg_ids_path     = chrom_dir / f"tg_ids_{chrom_id}.pt"
        tf_names_json   = self.data_dir / f"tf_names.json"
        tg_names_json   = chrom_dir / f"tg_names_{chrom_id}.json"
        metacell_names_path = self.data_dir / "metacell_names.json"
        motif_mask_path = chrom_dir / f"motif_mask_{chrom_id}.pt"

        required = [
            tf_path, tg_path, atac_path,
            window_map_path, metacell_names_path,
            tf_names_json, tg_names_json,
            tf_ids_path, tg_ids_path
        ]
        for f in required:
            if not f.exists():
                raise FileNotFoundError(f"Required file not found: {f}")
            
        # load tensors
        self.tf_tensor_all = torch.load(tf_path).float()
        self.tg_tensor_all = torch.load(tg_path).float()
        self.atac_window_tensor_all = torch.load(atac_path).float()
        self.num_cells   = self.tf_tensor_all.shape[1]
        self.num_windows = self.atac_window_tensor_all.shape[0]
        
        # ids + names
        self.tf_ids = torch.load(tf_ids_path).long()
        self.tg_ids = torch.load(tg_ids_path).long()
        with open(tf_names_json) as f: self.tf_names = [self.standardize_name(n) for n in json.load(f)]
        with open(tg_names_json) as f: self.tg_names = [self.standardize_name(n) for n in json.load(f)]
        
        # Load the map of ATAC peaks to windows
        with open(window_map_path, "r") as f:
            self.window_map = json.load(f)
            
        # Load the metacell names
        with open(metacell_names_path, "r") as f:
            self.metacell_names = json.load(f)
            
        self.num_cells = self.tf_tensor_all.shape[1]  # already set
        self.num_windows = self.atac_window_tensor_all.shape[0]
        
        # Load Distance Bias tensor
        if dist_bias_path.exists():
            bias_WG = torch.load(dist_bias_path).float()
            if bias_WG.shape[0] == self.atac_window_tensor_all.shape[0]:
                self.dist_bias_tensor = bias_WG.T.contiguous()
            elif bias_WG.shape[1] == self.atac_window_tensor_all.shape[0]:
                self.dist_bias_tensor = bias_WG.contiguous()
            else:
                raise ValueError(f"dist_bias_{chrom_id}.pt shape mismatch: {tuple(bias_WG.shape)}")
        else:
            self.dist_bias_tensor = None
            
        # Load Motif Mask tensor
        if motif_mask_path.exists():
            self.motif_mask_tensor = torch.load(motif_mask_path).bool()
        else:
            self.motif_mask_tensor = torch.zeros((self.tg_ids.numel(), self.tf_ids.numel()), dtype=torch.float32)

        # Optional: restrict to a subset of cells (column indices)
        if cell_idx is not None:
            self.set_cell_idx(cell_idx)
            
        self._validate_sizes()
    
    def __len__(self):
        return int(self._cell_idx.size) if self._cell_idx is not None else int(self.num_cells)
    
    def __getitem__(self, idx):
        if self._cell_idx is not None:
            idx = int(self._cell_idx[int(idx)])

        # Get the TF, TG, and ATAC window tensors for a given cell index
        tf_tensor = self.tf_tensor_all[:, idx]                      # [T_eval]
        tg_tensor = self.tg_tensor_all[:, idx]                      # [G_eval]
        atac_wins = self.atac_window_tensor_all[:, idx].unsqueeze(-1)  # [W,1]

        # Distance bias
        if self.dist_bias_tensor is not None:
            dist_bias = self.dist_bias_tensor                     # [G_eval, W]
        else:
            dist_bias = torch.zeros((self.tg_ids.numel(), self.num_windows), dtype=torch.float32)
            
        # Motif mask
        if self.motif_mask_tensor is not None:
            motif_mask = self.motif_mask_tensor                       # [G_eval, T_eval]
        else:
            motif_mask = torch.zeros((self.tg_ids.numel(), self.tf_ids.numel()), dtype=torch.float32)

        return (
            atac_wins,
            tf_tensor,
            tg_tensor,
            dist_bias,
            self.tf_ids,
            self.tg_ids,
            motif_mask,
        )

    def set_cell_idx(self, cell_idx: np.ndarray):
        cell_idx = np.asarray(cell_idx, dtype=int)
        if cell_idx.ndim != 1:
            raise ValueError("cell_idx must be a 1D array of column indices")
        if cell_idx.size == 0:
            raise ValueError("cell_idx cannot be empty")
        if cell_idx.min() < 0 or cell_idx.max() >= int(self.num_cells):
            raise IndexError("cell_idx out of bounds for dataset")
        self._cell_idx = cell_idx
        
    def _validate_sizes(self):
        # Check that the TF, TG, and ATAC window tensor sizes match the ids and names
        assert self.tf_tensor_all.shape[0] == len(self.tf_ids) == len(self.tf_names), "TF tensor size mismatch with names or ids"
        assert self.tg_tensor_all.shape[0] == len(self.tg_ids) == len(self.tg_names), "TG tensor size mismatch with names or ids"
        assert self.atac_window_tensor_all.shape[0] == self.num_windows, "ATAC window tensor size mismatch"
        
        # Check that the metecell names match the cell count and tensors
        assert self.metacell_names is not None and len(self.metacell_names) == self.num_cells, "Metacell names count mismatch"
        assert len(self.metacell_names) == self.tf_tensor_all.shape[1], "Cell count mismatch between TF tensor and metacell names"
        assert len(self.metacell_names) == self.tg_tensor_all.shape[1], "Cell count mismatch between TG tensor and metacell names"
        assert self.tf_tensor_all.shape[1] == self.tg_tensor_all.shape[1] == self.atac_window_tensor_all.shape[1], "Cell count mismatch across tensors"
            
        # Make sure counts align with tensors
        assert self.tf_tensor_all.shape[0] == len(self.tf_names) == self.tf_ids.numel(), \
            "TF count mismatch between tensor, names, and ids"
        assert self.tg_tensor_all.shape[0] == len(self.tg_names) == self.tg_ids.numel(), \
            "TG count mismatch between tensor, names, and ids"
        
        # Ensure ids fit the common vocab if we loaded it
        if self.tf_name2id is not None:
            tf_vocab_size = len(self.tf_name2id)
            assert int(self.tf_ids.max()) < tf_vocab_size and int(self.tf_ids.min()) >= 0, \
                f"tf_ids out of range for common vocab (max={int(self.tf_ids.max())}, vocab={tf_vocab_size})"
                
        if self.tg_name2id is not None:
            tg_vocab_size = len(self.tg_name2id)
            assert int(self.tg_ids.max()) < tg_vocab_size and int(self.tg_ids.min()) >= 0, \
                f"tg_ids out of range for common vocab (max={int(self.tg_ids.max())}, vocab={tg_vocab_size})"
    
    @staticmethod
    def collate_fn(batch):
        """
        Returns:
          atac_wins: [B, W, 1]
          tf_tensor: [B, T_eval]
          tg_tensor: [B, G_eval]
          bias:      [B, G_eval, W]
          tf_ids:    [T_eval]
          tg_ids:    [G_eval]
          motif_mask:[B, G_eval, T_eval]
        """
        atac_list, tf_list, tg_list, bias_list, tf_ids_list, tg_ids_list, mask_list = zip(*batch)

        atac_wins = torch.stack(atac_list, dim=0)
        tf_tensor = torch.stack(tf_list,  dim=0)
        tg_tensor = torch.stack(tg_list,  dim=0)
        bias      = torch.stack(bias_list, dim=0)

        tf_ids     = tf_ids_list[0]
        tg_ids     = tg_ids_list[0]
        motif_mask = mask_list[0]

        return atac_wins, tf_tensor, tg_tensor, bias, tf_ids, tg_ids, motif_mask
    
    def standardize_name(self, name: str) -> str:
        """Convert gene/motif name to capitalization style (e.g. 'Hoxa2')."""
        if not isinstance(name, str):
            return name
        return name.upper()

        
