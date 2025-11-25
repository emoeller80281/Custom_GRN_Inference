import os
import json
import numpy as np
import torch
from typing import Optional, Union
from torch.utils.data import Dataset, Sampler
from pathlib import Path
import logging
from collections import OrderedDict

from dataclasses import dataclass

# in datasets/dataset.py (or wherever you defined fit_simple_scalers)
import torch
from dataclasses import dataclass

@dataclass
class SimpleScaler:
    mean: torch.Tensor  # shape [D] on the correct device
    std:  torch.Tensor  # shape [D] on the correct device

    def transform(self, x: torch.Tensor, ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        If ids is provided (shape [D_batch]), we slice global stats to match x's last dim.
        x can be [B, D_batch] (TF/TG) or any tensor with last-dim == len(ids).
        """
        eps = 1e-6
        if ids is not None:
            # ids and mean/std must be on the same device
            mu  = self.mean.index_select(0, ids)
            sig = self.std.index_select(0, ids).clamp_min(eps)
        else:
            # fallback: whole-dim scaling
            mu, sig = self.mean, self.std.clamp_min(eps)

        # broadcast over batch/time dims (subtract along last axis)
        return (x - mu) / sig
    
    def inverse_transform(self, x: torch.Tensor, ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Undo z-scoring that was applied by `transform`.
        Works with either global stats or a sliced subset via `ids`.
        - x:   [..., D_batch]
        - ids: [D_batch] (global ids used during transform), or None for full-dim inverse
        """
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
    Computes per-dimension mean/std for TF and TG over the entire training set,
    even if batches have different T_eval/G_eval. Uses tf_ids/tg_ids to scatter-add
    into fixed-size accumulators of length T_expected/G_expected.
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

    # means and stds (unseen dims keep mean=0, std=1)
    tf_mean = torch.zeros(T_expected, dtype=torch.float32)
    tf_std  = torch.ones(T_expected,  dtype=torch.float32)
    mask_tf = tf_count > 0
    tf_mean[mask_tf] = (tf_sum[mask_tf] / tf_count[mask_tf]).to(torch.float32)
    tf_var = torch.zeros_like(tf_mean)
    tf_var[mask_tf] = (tf_sqsum[mask_tf] / tf_count[mask_tf]).to(torch.float32) - tf_mean[mask_tf]**2
    tf_std = torch.sqrt(torch.clamp(tf_var, min=1e-6))

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
        import numpy as np

        rng = np.random.RandomState(self.seed + self.epoch)

        chroms = list(self.chrom_to_indices.keys())
        if self.shuffle:
            rng.shuffle(chroms)

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
        total = 0
        for _, idxs in self.chrom_to_indices.items():
            n = len(idxs)
            total += (n + self.batch_size - 1) // self.batch_size
        return total


class ChromSubsetBatchSampler(Sampler):
    """
    Batch sampler that:
        - Operates on a shared MultiChromosomeDataset.
        - Restricts indices to a given subset of chromosomes.
        - Keeps batches chromosome-homogeneous (no shape mismatches).
    """
    def __init__(self, ds, chrom_subset, batch_size, shuffle=True, seed=0):
        self.ds = ds
        self.chrom_subset = set(chrom_subset)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._build_ranges()

    def _build_ranges(self):
        self._chrom_ranges = []
        # dataset._offsets[i] is the start index for ds.chrom_ids[i]
        for i, chrom in enumerate(self.ds.chrom_ids):
            if chrom not in self.chrom_subset:
                continue
            start = self.ds._offsets[i]
            end = self.ds._offsets[i+1] if i + 1 < len(self.ds._offsets) else len(self.ds)
            if end > start:
                idxs = list(range(start, end))
                self._chrom_ranges.append((chrom, idxs))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        if not self._chrom_ranges:
            return
        rng = np.random.RandomState(self.seed + self.epoch)
        chrom_blocks = self._chrom_ranges[:]
        if self.shuffle:
            rng.shuffle(chrom_blocks)
        for _, idxs in chrom_blocks:
            idxs = idxs[:]  # copy
            if self.shuffle:
                rng.shuffle(idxs)
            for s in range(0, len(idxs), self.batch_size):
                batch = idxs[s:s + self.batch_size]
                if batch:  # non-empty
                    yield batch

    def __len__(self):
        count = 0
        for _, idxs in self._chrom_ranges:
            if not idxs:
                continue
            count += (len(idxs) + self.batch_size - 1) // self.batch_size
        return count

class DistributedBatchSampler(torch.utils.data.Sampler):
    """
    Shards a *batch sampler* across DDP ranks (each rank gets a disjoint
    subset of batches). This avoids duplicating work without requiring
    per-sample DistributedSampler, which would break chrom-homogeneous batches.
    """
    def __init__(self, batch_sampler, num_replicas, rank, drop_last=True):
        self.batch_sampler = batch_sampler
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)

    def set_epoch(self, epoch: int):
        # delegate if inner supports it
        if hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)

    def __iter__(self):
        batches = list(self.batch_sampler)
        if self.drop_last:
            size = (len(batches) // self.num_replicas) * self.num_replicas
            batches = batches[:size]
        for i in range(self.rank, len(batches), self.num_replicas):
            yield batches[i]

    def __len__(self):
        base = len(self.batch_sampler)
        return (base // self.num_replicas) if self.drop_last else (base + self.num_replicas - 1) // self.num_replicas

class MultiChromosomeDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        chrom_ids,
        tf_vocab_path: Optional[Union[str, Path]] = None,
        tg_vocab_path: Optional[Union[str, Path]] = None,
        fine_tuner: bool = False,
        sample_name: Optional[str] = None,
        max_cached: int = 2,
        max_tfs: Optional[int] = None,
        max_tgs: Optional[int] = None,
        max_windows_per_chrom: Optional[int] = None,
        max_cells: Optional[int] = None,
        subset_seed: int = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.chrom_ids = list(chrom_ids)
        self.tf_vocab_path = Path(tf_vocab_path)
        self.tg_vocab_path = Path(tg_vocab_path)
        self.fine_tuner = fine_tuner
        self.sample_name = sample_name
        self.max_cached = max_cached
        self.max_tfs = max_tfs
        self.max_tgs = max_tgs
        self.max_windows_per_chrom = max_windows_per_chrom
        self.max_cells = max_cells
        self.subset_seed = subset_seed
        
        self._tf_name2id_full = self._load_vocab_dict(tf_vocab_path)
        self._tg_name2id_full = self._load_vocab_dict(tg_vocab_path)

        # In pseudobulk mode, num_cells is shared (tf_tensor_all is global).
        # We can read it once from data_dir (global tf tensor) to avoid loading each chromosome.
        if not fine_tuner:
            tf_global = torch.load(self.data_dir / "tf_tensor_all.pt", map_location="cpu")
            full_num_cells = int(tf_global.shape[1])

            # Start with all cell indices
            base_idx = np.arange(full_num_cells)

            # --- Optional: restrict by sample tag from metacell_names.json ---
            self._cell_idx = None
            if self.allowed_samples:
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

                # interpret sample tag as prefix before first '.' (e.g. "E7.5_REP1")
                keep = []
                for i, name in enumerate(all_names):
                    tag = str(name).split(".")[0]
                    if tag in self.allowed_samples:
                        keep.append(i)

                if not keep:
                    raise ValueError(
                        f"No metacells matched allowed_samples={sorted(self.allowed_samples)} "
                        f"in metacell_names.json."
                    )

                base_idx = np.array(sorted(keep), dtype=int)

            # --- Optional: downsample to max_cells ---
            self._num_cells = int(base_idx.size)
            if self.max_cells is not None and self._num_cells > self.max_cells:
                rng = np.random.RandomState(self.subset_seed or 42)
                chosen_local = np.sort(
                    rng.choice(self._num_cells, size=self.max_cells, replace=False)
                )
                base_idx = base_idx[chosen_local]
                self._num_cells = int(base_idx.size)

            # global indices into C dimension of tf/tg/atac tensors
            self._cell_idx = base_idx


        # offsets tell us where each chromosome's indices start in the concatenated space
        self._offsets = []
        running = 0
        if not fine_tuner:
            for _ in self.chrom_ids:
                self._offsets.append(running)
                running += self._num_cells
            self._length = running
        else:
            # Fine-tuner: compute per-chrom lengths on demand
            self._per_chrom_len = {}
            for chrom in self.chrom_ids:
                ds = self._load_chrom(chrom)  # temporary open
                n = len(ds)
                self._per_chrom_len[chrom] = n
                self._offsets.append(running)
                running += n
                self._evict_if_needed()  # keep cache small
            self._length = running

        # Small LRU cache of per-chrom datasets
        self._cache: OrderedDict[str, MultiomicTransformerDataset] = OrderedDict()
        
        # Build per-chrom name inventories (fast JSON reads, not big tensors)
        per_chrom_tf = {}
        per_chrom_tg = {}
        for cid in self.chrom_ids:
            tf_names, tg_names = self._peek_names_for_chrom(cid)
            per_chrom_tf[cid] = tf_names
            per_chrom_tg[cid] = tg_names

        rng = np.random.RandomState(self.subset_seed or 42)

        # Guarantee at least one per chromosome, then top up to your global caps
        self.tf_keep_names = self._build_global_keep(
            per_chrom_names=per_chrom_tf,
            max_k=self.max_tfs,
            rng=rng,
            ensure_per_chrom=True,
            min_per_chrom=1,
        )
        self.tg_keep_names = self._build_global_keep(
            per_chrom_names=per_chrom_tg,
            max_k=self.max_tgs,
            rng=rng,
            ensure_per_chrom=True,
            min_per_chrom=1,
        )

        # Global contiguous id maps (these define the embedding table sizes)
        self.tf_name2id_sub = {n: i for i, n in enumerate(self.tf_keep_names)}
        self.tg_name2id_sub = {n: i for i, n in enumerate(self.tg_keep_names)}
        
        # Contiguous 0..T'-1 and 0..G'-1 that match the sub-vocabs above.
        self.tf_ids_sub = torch.arange(len(self.tf_name2id_sub), dtype=torch.long)
        self.tg_ids_sub = torch.arange(len(self.tg_name2id_sub), dtype=torch.long)

        # Ordered names (by id) are handy for reports/plots
        self.tf_names_sub = [n for n, _ in sorted(self.tf_name2id_sub.items(), key=lambda kv: kv[1])]
        self.tg_names_sub = [n for n, _ in sorted(self.tg_name2id_sub.items(), key=lambda kv: kv[1])]

        # Expose legacy attribute names expected by logging code
        # (These are "global across chromosomes", not per-batch)
        self.tf_ids = self.tf_ids_sub
        self.tg_ids = self.tg_ids_sub
        self.tf_names = self.tf_names_sub
        self.tg_names = self.tg_names_sub
        
        # --- Total number of ATAC windows across all chromosomes (after subsampling) ---
        # We load each per-chrom dataset via the same path used for training (apply_global_subsample),
        # read its num_windows, and sum them. This is deterministic across ranks if subset_seed is fixed.
        self._windows_per_chrom = {}
        _total_windows = 0
        self.metacell_names = None
        for cid in self.chrom_ids:
            ds = self._load_chrom(cid)   # uses cache + applies global sub-vocab + max_windows_per_chrom
            w = int(ds.num_windows)
            self._windows_per_chrom[cid] = w
            _total_windows += w
            self._evict_if_needed()
            
            if self.metacell_names is None:
                self.metacell_names = list(ds.metacell_names)  # keep order
            else:
                if list(ds.metacell_names) != self.metacell_names:
                    logging.warning(f"metacell_names differ on {cid}; using the first set.")
            
        self.num_windows = int(_total_windows)

        # store window cap too
        self.max_windows_per_chrom = max_windows_per_chrom

    def _evict_if_needed(self):
        while len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)  # evict oldest

    def __len__(self):
        return self._length

    def _locate(self, idx):
        # Binary search over offsets to find chromosome and local idx
        lo, hi = 0, len(self.chrom_ids)-1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self._offsets[mid]
            if mid == len(self.chrom_ids)-1:
                end = self._length
            else:
                end = self._offsets[mid+1]
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
        # Reuse your per-chrom collate (expects within-batch consistent W, G, T)
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

    def _load_chrom_lazy(self, chrom_id: str) -> "MultiomicTransformerDataset":
        """Instantiate a child without forcing subsample (internal helper)."""
        ds = MultiomicTransformerDataset(
            data_dir=self.data_dir,
            chrom_id=chrom_id,
            tf_vocab_path=self.tf_vocab_path,
            tg_vocab_path=self.tg_vocab_path,
            fine_tuner=self.fine_tuner,
            sample_name=self.sample_name
        )
        
        return ds

    def _load_chrom(self, chrom_id: str) -> "MultiomicTransformerDataset":
        # cache hit
        if chrom_id in self._cache:
            ds = self._cache.pop(chrom_id)
            self._cache[chrom_id] = ds
            return ds

        # miss -> create
        ds = self._load_chrom_lazy(chrom_id)

        # Apply the global subsample and contiguous id mapping
        ds.apply_global_subsample(
            tf_name2id_sub=self.tf_name2id_sub,
            tg_name2id_sub=self.tg_name2id_sub,
            max_windows=self.max_windows_per_chrom,
            rng_seed=self.subset_seed,
            cell_idx=self._cell_idx,
        )

        self._cache[chrom_id] = ds
        self._evict_if_needed()
        return ds
    
    def _peek_names_for_chrom(self, chrom_id: str):
        """
        Fast, lightweight peek: read only the TF/TG names JSON for a chromosome,
        without loading big tensors.
        """
        chrom_dir = self.data_dir / chrom_id
        tf_names_json = self.data_dir / "tf_names.json"
        tg_names_json = chrom_dir / f"tg_names_{chrom_id}.json"

        with open(tf_names_json) as f:
            tf_names = [self.standardize_name(n) for n in json.load(f)]
        with open(tg_names_json) as f:
            tg_names = [self.standardize_name(n) for n in json.load(f)]
        return tf_names, tg_names

    def _build_global_keep(
        self,
        per_chrom_names: dict[str, list[str]],
        max_k: Optional[int],
        rng: np.random.RandomState,
        ensure_per_chrom: bool = True,
        min_per_chrom: int = 1,
    ) -> list[str]:
        """
        Build a global keep list given per-chrom name lists. Optionally guarantees
        at least `min_per_chrom` names from each chromosome (if available), then
        tops up to max_k from the remaining union.
        """
        # union over all chromosomes
        union = set()
        for names in per_chrom_names.values():
            union.update(names)

        if not max_k or max_k >= len(union):
            # keep everything
            return sorted(union)

        chosen = set()

        if ensure_per_chrom and min_per_chrom > 0:
            # seed with a small quota from each chromosome
            for cid, names in per_chrom_names.items():
                if not names:
                    continue
                take = min(min_per_chrom, len(names))
                picks = rng.choice(names, size=take, replace=False)
                chosen.update(picks)

        # top up to max_k from the leftover pool
        remaining = list(union - chosen)
        need = max_k - len(chosen)
        if need > 0 and remaining:
            extra = rng.choice(remaining, size=min(need, len(remaining)), replace=False)
            chosen.update(extra)

        # If we still came up short (all chromosomes tiny), just return whatever exists
        return sorted(chosen)



class ChromBucketBatchSampler(Sampler[list]):
    """
    Produces batches grouped by chromosome so that sequence length W
    and bias shapes match within the batch (no padding needed).
    """
    def __init__(self, dataset: MultiChromosomeDataset, batch_size: int, shuffle: bool = True, seed: int = 0):
        self.ds = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self._chrom_ranges: list[tuple[str, list[int]]] = []
        self._refresh_ranges()
        
    def _refresh_ranges(self):
        self._chrom_ranges = []
        # Recompute from dataset offsets/length (works for train/val)
        for i, chrom in enumerate(self.ds.chrom_ids):
            start = self.ds._offsets[i]
            end = self.ds._offsets[i+1] if i+1 < len(self.ds._offsets) else len(self.ds)
            if end > start:
                self._chrom_ranges.append((chrom, list(range(start, end))))


    def set_epoch(self, epoch: int):
        self.epoch = epoch

        # Build per-chrom index ranges
        self._chrom_ranges = []
        for i, chrom in enumerate(self.ds.chrom_ids):
            start = self.ds._offsets[i]
            end = self.ds._offsets[i+1] if i+1 < len(self.ds._offsets) else len(self.ds)
            idxs = list(range(start, end))
            self._chrom_ranges.append((chrom, idxs))

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        chrom_blocks = self._chrom_ranges[:]
        if self.shuffle:
            rng.shuffle(chrom_blocks)
        for chrom, idxs in chrom_blocks:
            idxs = idxs[:]  # copy
            if self.shuffle:
                rng.shuffle(idxs)
            for s in range(0, len(idxs), self.batch_size):
                yield idxs[s:s+self.batch_size]

    def __len__(self):
        count = 0
        for _, idxs in self._chrom_ranges:
            count += (len(idxs) + self.batch_size - 1) // self.batch_size
        return count

def _subsample_vocab(name2id: dict, max_n: int, rng: np.random.RandomState):
    if max_n is None or max_n >= len(name2id):
        return name2id
    names = list(name2id.keys())
    keep = set(rng.choice(names, size=max_n, replace=False))
    # rebuild compact 0..N-1 id map
    return {n:i for i,n in enumerate(sorted(keep))}

def _reindex_tensor_rows(tensor, old_name2id, new_name2id, axis=0):
    # assumes rows (axis=0) correspond to vocab order; select rows that survive + in new order
    keep_names = sorted(new_name2id, key=new_name2id.get)
    idx = [old_name2id[n] for n in keep_names]
    return tensor.index_select(axis, torch.tensor(idx, dtype=torch.long))

class MultiomicTransformerDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        chrom_id: str,
        tf_vocab_path: Optional[Path] = None,
        tg_vocab_path: Optional[Path] = None,
        fine_tuner: bool = False,
        sample_name: Optional[str] = None,
        max_tfs: Optional[int] = None,
        max_tgs: Optional[int] = None,
        max_windows: Optional[int] = None,
        allowed_samples: Optional[list[str]] = None,
        subset_seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.chrom_id = chrom_id
        self.sample_name = sample_name
        self._max_tfs = max_tfs
        self._max_tgs = max_tgs
        self._max_windows = max_windows
        self.allowed_samples = set(allowed_samples) if allowed_samples else None
        self._subset_rng = np.random.RandomState(subset_seed)

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

        # --- Fine-tune mode ---
        if fine_tuner:
            if sample_name is None:
                raise ValueError("fine_tuner=True requires sample_name (e.g. 'E7.5_rep1')")

            sc_dir = chrom_dir / "single_cell" / sample_name

            tf_tensor_file   = sc_dir / f"{sample_name}_tf_tensor_singlecell_{chrom_id}.pt"
            tg_tensor_file   = sc_dir / f"{sample_name}_tg_tensor_singlecell_{chrom_id}.pt"
            atac_tensor_file = sc_dir / f"{sample_name}_atac_tensor_singlecell_{chrom_id}.pt"
            tf_ids_file      = sc_dir / f"{sample_name}_tf_ids_singlecell_{chrom_id}.pt"
            tg_ids_file      = sc_dir / f"{sample_name}_tg_ids_singlecell_{chrom_id}.pt"
            tf_names_file    = sc_dir / f"{sample_name}_tf_names_singlecell_{chrom_id}.json"
            tg_names_file    = sc_dir / f"{sample_name}_tg_names_singlecell_{chrom_id}.json"

            required = [tf_tensor_file, tg_tensor_file, atac_tensor_file,
                        tf_ids_file, tg_ids_file, tf_names_file, tg_names_file]
            for f in required:
                if not f.exists():
                    raise FileNotFoundError(f"Required single-cell tensor missing: {f}")

            # load tensors
            self.tf_tensor_all   = torch.load(tf_tensor_file).float()
            self.tg_tensor_all   = torch.load(tg_tensor_file).float()
            self.atac_window_tensor_all = torch.load(atac_tensor_file).float()

            # ids + names
            self.tf_ids = torch.load(tf_ids_file).long()
            self.tg_ids = torch.load(tg_ids_file).long()
            with open(tf_names_file) as f: self.tf_names = [self.standardize_name(n) for n in json.load(f)]
            with open(tg_names_file) as f: self.tg_names = [self.standardize_name(n) for n in json.load(f)]
            
            # --- REMAP ids to common vocab if available ---
            if self.tf_name2id is not None:
                missing_tf = [n for n in self.tf_names if n not in self.tf_name2id]
                if missing_tf:
                    logging.warning(f"{len(missing_tf)} TFs not in common vocab (e.g. {missing_tf[:10]})")
                self.tf_ids = torch.tensor(
                    [self.tf_name2id.get(n, -1) for n in self.tf_names], dtype=torch.long
                )
                if (self.tf_ids < 0).any():
                    bad = [n for n in self.tf_names if self.tf_name2id.get(n, -1) < 0][:10]
                    raise ValueError(f"Unmapped TFs present (e.g. {bad})")

            if self.tg_name2id is not None:
                missing_tg = [n for n in self.tg_names if n not in self.tg_name2id]
                if missing_tg:
                    logging.warning(f"{len(missing_tg)} TGs not in common vocab (e.g. {missing_tg[:10]})")
                self.tg_ids = torch.tensor(
                    [self.tg_name2id.get(n, -1) for n in self.tg_names], dtype=torch.long
                )
                if (self.tg_ids < 0).any():
                    bad = [n for n in self.tg_names if self.tg_name2id.get(n, -1) < 0][:10]
                    raise ValueError(f"Unmapped TGs present (e.g. {bad})")

            self.num_cells   = self.tg_tensor_all.shape[1]
            self.num_windows = self.atac_window_tensor_all.shape[0]
            
            # in fine-tune mode we skip pseudobulk dist_bias + motif_mask
            self.dist_bias_tensor = None
            self.motif_mask_tensor = None

            logging.info(f"[Fine-tune mode] Loaded single-cell sample {sample_name}: "
                         f"{self.num_cells} cells, "
                         f"{self.tf_tensor_all.shape[0]} TFs, "
                         f"{self.tg_tensor_all.shape[0]} TGs, "
                         f"{self.atac_window_tensor_all.shape[0]} peaks/windows")

        # --- Pseudobulk mode ---
        else:
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
            
            # -------- metadata --------
            with open(window_map_path, "r") as f:
                self.window_map = json.load(f)
            with open(metacell_names_path, "r") as f:
                self.metacell_names = json.load(f)

            self.num_cells = self.tf_tensor_all.shape[1]  # already set
            if not isinstance(self.metacell_names, list) or len(self.metacell_names) != self.num_cells:
                logging.warning(
                    f"metacell_names length ({len(self.metacell_names) if isinstance(self.metacell_names, list) else 'N/A'}) "
                    f"!= num_cells ({self.num_cells}); regenerating labels."
                )
                self.metacell_names = [f"cell_{i}" for i in range(self.num_cells)]

            # distance bias
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

            # motif mask
            if motif_mask_path.exists():
                self.motif_mask_tensor = torch.load(motif_mask_path).float()
            else:
                self.motif_mask_tensor = torch.zeros((self.tg_ids.numel(), self.tf_ids.numel()), dtype=torch.float32)
                
            # Make sure counts align with tensors
            assert self.tf_tensor_all.shape[0] == len(self.tf_names) == self.tf_ids.numel(), \
                "TF count mismatch between tensor, names, and ids"
            assert self.tg_tensor_all.shape[0] == len(self.tg_names) == self.tg_ids.numel(), \
                "TG count mismatch between tensor, names, and ids"

            # Ensure ids fit the *common* vocab if we loaded it
            if self.tf_name2id is not None:
                tf_vocab_size = len(self.tf_name2id)
                assert int(self.tf_ids.max()) < tf_vocab_size and int(self.tf_ids.min()) >= 0, \
                    f"tf_ids out of range for common vocab (max={int(self.tf_ids.max())}, vocab={tf_vocab_size})"
                    
            if self.tg_name2id is not None:
                tg_vocab_size = len(self.tg_name2id)
                assert int(self.tg_ids.max()) < tg_vocab_size and int(self.tg_ids.min()) >= 0, \
                    f"tg_ids out of range for common vocab (max={int(self.tg_ids.max())}, vocab={tg_vocab_size})"

        self._apply_subsampling()

    # -------- Dataset API --------
    def __len__(self):
        return self.num_cells

    def __getitem__(self, idx):
        # expressions for this metacell
        tf_tensor = self.tf_tensor_all[:, idx]                      # [T_eval]
        tg_tensor = self.tg_tensor_all[:, idx]                      # [G_eval]
        atac_wins = self.atac_window_tensor_all[:, idx].unsqueeze(-1)  # [W,1]

        # distance bias (shared across items)
        if self.dist_bias_tensor is not None:
            dist_bias = self.dist_bias_tensor                       # [G_eval, W]
        else:
            dist_bias = torch.zeros((self.tg_ids.numel(), self.num_windows), dtype=torch.float32)
            
        # motif mask (shared across items)
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
        
    def _apply_subsampling(self):
        """
        Optionally subsample windows (W), TFs (T), TGs (G) while keeping all
        tensors, ids, names, bias, and motif_mask consistent.
        """
        # 1) Subsample windows (affects atac_window_tensor_all rows, dist_bias columns)
        if self._max_windows is not None and self.num_windows > self._max_windows:
            keep_w = np.sort(self._subset_rng.choice(self.num_windows, size=self._max_windows, replace=False))
            keep_w_t = torch.as_tensor(keep_w, dtype=torch.long)

            # atac windows: [W, num_cells]
            self.atac_window_tensor_all = self.atac_window_tensor_all.index_select(0, keep_w_t)
            self.num_windows = int(self.atac_window_tensor_all.shape[0])

            # distance bias: [G, W] -> slice columns
            if getattr(self, "dist_bias_tensor", None) is not None:
                self.dist_bias_tensor = self.dist_bias_tensor.index_select(1, keep_w_t)

        # Build name->row index maps from current names
        tf_old_map = {self.standardize_name(n): i for i, n in enumerate(getattr(self, "tf_names", []))}
        tg_old_map = {self.standardize_name(n): i for i, n in enumerate(getattr(self, "tg_names", []))}

        # Bail out early if names aren’t present (shouldn’t happen with your pipeline)
        if not getattr(self, "tf_names", None) or not getattr(self, "tg_names", None):
            return

        # 2) Subsample TFs (affects tf_tensor_all rows, motif_mask columns, ids/names)
        if self._max_tfs is not None and len(self.tf_names) > self._max_tfs:
            keep_tf_names = list(self._subset_rng.choice(self.tf_names, size=self._max_tfs, replace=False))
            keep_tf_names = sorted(set(map(self.standardize_name, keep_tf_names)))
            keep_tf_idx = [tf_old_map[n] for n in keep_tf_names]
            keep_tf_t = torch.as_tensor(keep_tf_idx, dtype=torch.long)

            # tf tensor: [T, num_cells]
            self.tf_tensor_all = self.tf_tensor_all.index_select(0, keep_tf_t)
            self.tf_names = [self.tf_names[i] for i in keep_tf_idx]

            # rebuild compact vocab/id mapping 0..T'-1
            new_tf_map = {n: i for i, n in enumerate(keep_tf_names)}
            self.tf_name2id = new_tf_map
            self.tf_ids = torch.as_tensor([new_tf_map[n] for n in keep_tf_names], dtype=torch.long)

            # motif mask: [G, T] -> slice columns
            if getattr(self, "motif_mask_tensor", None) is not None:
                self.motif_mask_tensor = self.motif_mask_tensor.index_select(1, keep_tf_t)

        # 3) Subsample TGs (affects tg_tensor_all rows, dist_bias rows, motif_mask rows, ids/names)
        if self._max_tgs is not None and len(self.tg_names) > self._max_tgs:
            keep_tg_names = list(self._subset_rng.choice(self.tg_names, size=self._max_tgs, replace=False))
            keep_tg_names = sorted(set(map(self.standardize_name, keep_tg_names)))
            keep_tg_idx = [tg_old_map[n] for n in keep_tg_names]
            keep_tg_t = torch.as_tensor(keep_tg_idx, dtype=torch.long)

            # tg tensor: [G, num_cells]
            self.tg_tensor_all = self.tg_tensor_all.index_select(0, keep_tg_t)
            self.tg_names = [self.tg_names[i] for i in keep_tg_idx]

            # rebuild compact vocab/id mapping 0..G'-1
            new_tg_map = {n: i for i, n in enumerate(keep_tg_names)}
            self.tg_name2id = new_tg_map
            self.tg_ids = torch.as_tensor([new_tg_map[n] for n in keep_tg_names], dtype=torch.long)

            # dist bias: [G, W] -> slice rows
            if getattr(self, "dist_bias_tensor", None) is not None:
                self.dist_bias_tensor = self.dist_bias_tensor.index_select(
                    0, torch.arange(len(keep_tg_names))
                )

            # motif mask: [G, T] -> slice rows
            if getattr(self, "motif_mask_tensor", None) is not None:
                self.motif_mask_tensor = self.motif_mask_tensor.index_select(
                    0, torch.arange(len(keep_tg_names))
                )

    # -------- utilities --------
    def apply_global_subsample(
        self,
        tf_name2id_sub: dict[str, int],
        tg_name2id_sub: dict[str, int],
        max_windows: Optional[int] = None,
        rng_seed: int = 42,
        cell_idx: Optional[np.ndarray] = None,
    ):
        """
        Intersect this chromosome's TF/TG with the global sub-vocab and remap IDs so
        that returned tf_ids/tg_ids index into the global contiguous spaces used by the model.
        """
        rng = np.random.RandomState(rng_seed or 42)
        
        # Subsample the number of cells
        if cell_idx is not None:
            keep_c = torch.as_tensor(cell_idx, dtype=torch.long)
            # [T, C] / [G, C] / [W, C]
            self.tf_tensor_all          = self.tf_tensor_all.index_select(1, keep_c)
            self.tg_tensor_all          = self.tg_tensor_all.index_select(1, keep_c)
            self.atac_window_tensor_all = self.atac_window_tensor_all.index_select(1, keep_c)

            # IMPORTANT: update counts
            self.num_cells = int(self.tf_tensor_all.shape[1])

            # Keep metacell_names aligned (regenerate if missing or wrong length)
            if hasattr(self, "metacell_names") and isinstance(self.metacell_names, list):
                if len(self.metacell_names) != self.tf_tensor_all.shape[1] + (len(keep_c) if False else 0):
                    # before slicing, length should have matched original C; if not, reinit
                    # since we just sliced to len(keep_c), simply rebuild labels here
                    pass  # fall through to reinit
                try:
                    self.metacell_names = [self.metacell_names[i] for i in keep_c.tolist()]
                except Exception:
                    self.metacell_names = [f"cell_{i}" for i in range(self.num_cells)]
            else:
                self.metacell_names = [f"cell_{i}" for i in range(self.num_cells)]

        # ---- Intersect names with global keep sets ----
        tf_keep_names_local = [n for n in self.tf_names if n in tf_name2id_sub]
        tg_keep_names_local = [n for n in self.tg_names if n in tg_name2id_sub]

        if len(tf_keep_names_local) == 0 or len(tg_keep_names_local) == 0:
            raise ValueError(
                f"Subsample produced empty TF or TG set for {self.chrom_id}. "
                f"TFs={len(tf_keep_names_local)} TGs={len(tg_keep_names_local)}"
            )

        # Build index lists in the *current* tensor order
        tf_name_to_local = {n: i for i, n in enumerate(self.tf_names)}
        tg_name_to_local = {n: i for i, n in enumerate(self.tg_names)}

        tf_keep_idx = [tf_name_to_local[n] for n in tf_keep_names_local]
        tg_keep_idx = [tg_name_to_local[n] for n in tg_keep_names_local]

        # ---- Slice tensors to the intersection & preserve order of the *local* names ----
        # Shapes before: tf_tensor_all [T, C], tg_tensor_all [G, C], dist_bias [G, W], motif_mask [G, T]
        self.tf_tensor_all = self.tf_tensor_all[tf_keep_idx, :]
        self.tg_tensor_all = self.tg_tensor_all[tg_keep_idx, :]

        if self.motif_mask_tensor is not None:
            self.motif_mask_tensor = self.motif_mask_tensor[np.ix_(tg_keep_idx, tf_keep_idx)]
        else:
            self.motif_mask_tensor = torch.zeros(
                (len(tg_keep_idx), len(tf_keep_idx)), dtype=torch.float32
            )

        if self.dist_bias_tensor is not None:
            self.dist_bias_tensor = self.dist_bias_tensor[tg_keep_idx, :]

        # Update name lists to match sliced tensors
        self.tf_names = [self.tf_names[i] for i in tf_keep_idx]
        self.tg_names = [self.tg_names[i] for i in tg_keep_idx]

        # ---- Remap IDs using the *global* contiguous maps (0..|sub|-1) ----
        self.tf_ids = torch.tensor([tf_name2id_sub[n] for n in self.tf_names], dtype=torch.long)
        self.tg_ids = torch.tensor([tg_name2id_sub[n] for n in self.tg_names], dtype=torch.long)

        # ---- Optionally subsample windows (by accessibility) ----
        if max_windows is not None and self.num_windows > max_windows:
            # self.atac_window_tensor_all: [W, C] (windows x metacells)
            atac = self.atac_window_tensor_all

            # Compute an accessibility score per window.
            # Using mean over cells; sum would be equivalent for ranking.
            # Stay on CPU to avoid extra GPU use.
            if isinstance(atac, torch.Tensor):
                window_scores = atac.float().mean(dim=1)
            else:
                # in case it's a numpy array
                window_scores = torch.as_tensor(atac, dtype=torch.float32).mean(dim=1)

            # Number of windows to keep
            k = int(min(max_windows, self.num_windows))

            # Indices of top-k most accessible windows (unsorted genomic-wise)
            topk = torch.topk(window_scores, k=k, largest=True, sorted=False).indices

            # Sort to preserve original genomic order
            keep_w_t = torch.sort(topk).values.to(dtype=torch.long)

            # ---- Apply selection consistently across tensors ----

            # atac windows: [W, C] -> [W', C]
            if isinstance(atac, torch.Tensor):
                self.atac_window_tensor_all = atac.index_select(0, keep_w_t)
            else:
                self.atac_window_tensor_all = atac[keep_w_t.numpy(), :]

            # dist bias: [G, W] -> [G, W']
            if getattr(self, "dist_bias_tensor", None) is not None:
                # dist_bias_tensor is [G, W]
                self.dist_bias_tensor = self.dist_bias_tensor.index_select(1, keep_w_t)

            # Update num_windows
            self.num_windows = int(self.atac_window_tensor_all.shape[0])

            # window_map: remap peak_id -> new window index, if present
            if hasattr(self, "window_map") and isinstance(self.window_map, dict):
                # original: peak_id -> old_w_idx
                # build inverse: old_w_idx -> [peak_ids]
                idx_to_peaks = {}
                for peak_id, old_idx in self.window_map.items():
                    idx_to_peaks.setdefault(int(old_idx), []).append(peak_id)

                new_window_map = {}
                for new_idx, old_idx in enumerate(keep_w_t.tolist()):
                    for peak_id in idx_to_peaks.get(int(old_idx), []):
                        new_window_map[peak_id] = new_idx

                self.window_map = new_window_map

        # ---- Sanity logs (useful during bring-up) ----
        logging.debug(
            f"[Subsample] {self.chrom_id}: TFs={len(self.tf_names)} "
            f"TGs={len(self.tg_names)} Windows={self.num_windows} | "
            f"tf_ids max={int(self.tf_ids.max())} tg_ids max={int(self.tg_ids.max())}"
        )


    def filter_genes(self, subset_genes):
        subset = set(subset_genes)
        if not subset:
            raise ValueError("subset_genes must be non-empty.")
        name_to_idx = {g: i for i, g in enumerate(self.tg_names)}
        keep = [name_to_idx[g] for g in subset if g in name_to_idx]
        missing = [g for g in subset if g not in name_to_idx]
        if missing:
            logging.warning(f"{len(missing)} genes not present (e.g. {missing[:10]})")
        if not keep:
            raise ValueError("No matching genes found to keep.")

        self.tg_tensor_all = self.tg_tensor_all[keep, :]
        if self.dist_bias_tensor is not None:
            self.dist_bias_tensor = self.dist_bias_tensor[keep, :]
        self.tg_names = [self.tg_names[i] for i in keep]

        # re-map tg_ids if we have a vocab
        if self.tg_name2id is not None:
            self.tg_ids = torch.tensor([self.tg_name2id[n] for n in self.tg_names if n in self.tg_name2id],
                                       dtype=torch.long)
        else:
            # keep previous ids length-aligned to tensor
            self.tg_ids = self.tg_ids[:len(self.tg_names)]

        logging.info(f"Filtered TGs: kept {len(self.tg_names)}")

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