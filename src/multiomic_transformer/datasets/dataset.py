import os, sys
import json
import joblib
import pandas as pd
import numpy as np
from sympy import O
import torch
from typing import Optional, Union
from torch.utils.data import Dataset, Sampler
from pathlib import Path
import logging
from collections import OrderedDict
import random


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
        self.subset_seed = subset_seed
        
        self._tf_name2id_full = self._load_vocab_dict(tf_vocab_path)
        self._tg_name2id_full = self._load_vocab_dict(tg_vocab_path)

        # In pseudobulk mode, num_cells is shared (tf_tensor_all is global).
        # We can read it once from data_dir (global tf tensor) to avoid loading each chromosome.
        if not fine_tuner:
            tf_global = torch.load(self.data_dir / "tf_tensor_all.pt", map_location="cpu")
            self._num_cells = int(tf_global.shape[1])
        else:
            # Fine-tune single-cell mode: number of cells can vary per chromosome.
            # We'll open a tiny handle per chrom to read shape; still keep it light.
            self._num_cells = None

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
        subset_seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.chrom_id = chrom_id
        self.sample_name = sample_name
        self._max_tfs = max_tfs
        self._max_tgs = max_tgs
        self._max_windows = max_windows
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
            
            self.scaler = None

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
            tf_path   = self.data_dir / "tf_tensor_all.pt"
            tg_path   = chrom_dir / f"tg_tensor_all_{chrom_id}.pt"
            atac_path = chrom_dir / f"atac_window_tensor_all_{chrom_id}.pt"
            scaler_path = chrom_dir / f"tg_scaler_{chrom_id}.save"
            window_map_path = chrom_dir / f"window_map_{chrom_id}.json"
            dist_bias_path  = chrom_dir / f"dist_bias_{chrom_id}.pt"
            tf_ids_path     = self.data_dir / "tf_ids.pt"
            tg_ids_path     = chrom_dir / f"tg_ids_{chrom_id}.pt"
            tf_names_json   = self.data_dir / "tf_names.json"
            tg_names_json   = chrom_dir / f"tg_names_{chrom_id}.json"
            metacell_names_path = self.data_dir / "metacell_names.json"
            motif_mask_path = chrom_dir / f"motif_mask_{chrom_id}.pt"

            required = [
                tf_path, tg_path, atac_path, scaler_path,
                window_map_path, metacell_names_path,
                tf_names_json, tg_names_json,
                tf_ids_path, tg_ids_path
            ]
            for f in required:
                if not f.exists():
                    raise FileNotFoundError(f"Required file not found: {f}")
                
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                logging.warning(f"Scaler not found at {scaler_path}, setting to None")
                self.scaler = None

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
        tensors, ids, names, bias, motif_mask, and scaler consistent.
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
                self.dist_bias_tensor = self.dist_bias_tensor.index_select(0, keep_tg_t)

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

        # 3) Subsample TGs (affects tg_tensor_all rows, dist_bias rows, motif_mask rows, ids/names, scaler)
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

            # scaler is per-gene: subset it
            if getattr(self, "scaler", None) is not None:
                try:
                    self.scaler = self.subset_scaler(self.scaler, keep_tg_idx)
                except Exception as e:
                    logging.warning(f"Could not subset scaler to kept TGs: {e}")


    # -------- utilities --------
    def subset_scaler(self, original_scaler, kept_indices):
        from sklearn.preprocessing import StandardScaler
        new_scaler = StandardScaler()
        new_scaler.mean_ = original_scaler.mean_[kept_indices]
        new_scaler.scale_ = original_scaler.scale_[kept_indices]
        new_scaler.var_ = original_scaler.var_[kept_indices]
        new_scaler.n_features_in_ = len(kept_indices)
        return new_scaler
    
    def apply_global_subsample(
        self,
        tf_name2id_sub: dict[str, int],
        tg_name2id_sub: dict[str, int],
        max_windows: Optional[int] = None,
        rng_seed: int = 42,
    ):
        """
        Intersect this chromosome's TF/TG with the global sub-vocab and remap IDs so
        that returned tf_ids/tg_ids index into the global contiguous spaces used by the model.
        """
        rng = np.random.RandomState(rng_seed or 42)

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

        # ---- Optionally subsample windows ----
        if max_windows is not None and self.num_windows > max_windows:
            # consistent window subsample across tensors that have W
            W = self.num_windows
            keep_w = np.sort(rng.choice(np.arange(W), size=max_windows, replace=False))
            keep_w_t = torch.tensor(keep_w, dtype=torch.long)

            # atac windows: [W, C] -> [W', C]
            self.atac_window_tensor_all = self.atac_window_tensor_all[keep_w_t, :]

            # dist bias: [G, W] -> [G, W']
            if self.dist_bias_tensor is not None:
                self.dist_bias_tensor = self.dist_bias_tensor[:, keep_w_t]

            self.num_windows = int(self.atac_window_tensor_all.shape[0])

        # ---- Sanity logs (useful during bring-up) ----
        logging.debug(
            f"[Subsample] {self.chrom_id}: TFs={len(self.tf_names)} "
            f"TGs={len(self.tg_names)} Windows={self.num_windows} | "
            f"tf_ids max={int(self.tf_ids.max())} tg_ids max={int(self.tg_ids.max())}"
        )

    
    def inverse_transform(self, preds, tg_ids=None) -> np.ndarray:
        """
        Inverse-transform predictions back to raw scale.
        preds: [num_samples, num_genes]
        tg_ids: tensor or array of target gene IDs (indices into vocab).
        """
        if tg_ids is None:
            # assume same number of TGs as scaler
            return self.scaler.inverse_transform(preds)

        tg_ids = np.array(tg_ids, dtype=int)
        sub_scaler = self.subset_scaler(self.scaler, tg_ids)
        return sub_scaler.inverse_transform(preds)

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
