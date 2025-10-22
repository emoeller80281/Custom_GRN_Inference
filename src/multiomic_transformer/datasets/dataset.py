import os
import json
import joblib
import pandas as pd
import numpy as np
from sympy import O
import torch
from typing import Optional
from torch.utils.data import Dataset
from pathlib import Path
import logging

class MultiomicTransformerDataset(Dataset):
    """
    Yields:
      atac_wins: [W, 1]
      tf_tensor: [T_eval]
      tg_tensor: [G_eval]
      dist_bias: [G_eval, W]   (same for every item; stacked in collate)
      tf_ids   : [T_eval] (Long) - indices into COMMON TF vocab
      tg_ids   : [G_eval] (Long) - indices into COMMON TG vocab
    """
    def __init__(self,
                 data_dir: Path,
                 chrom_id: str,
                 tf_vocab_path: Optional[Path] = None,
                 tg_vocab_path: Optional[Path] = None,
                 fine_tuner: bool = False,
                 sample_name: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.chrom_id = chrom_id
        self.sample_name = sample_name

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

    # -------- utilities --------
    def subset_scaler(self, original_scaler, kept_indices):
        from sklearn.preprocessing import StandardScaler
        new_scaler = StandardScaler()
        new_scaler.mean_ = original_scaler.mean_[kept_indices]
        new_scaler.scale_ = original_scaler.scale_[kept_indices]
        new_scaler.var_ = original_scaler.var_[kept_indices]
        new_scaler.n_features_in_ = len(kept_indices)
        return new_scaler
    
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
