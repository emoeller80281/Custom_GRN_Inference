import os
import json
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
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
                 data_dir: str,
                 chrom_id: str,
                 tf_vocab_path: str = None,   # e.g. ".../transformer_data/common/tf_vocab.json"
                 tg_vocab_path: str = None,   # e.g. ".../transformer_data/common/tg_vocab.json"
                 ):
        self.data_dir = data_dir
        self.chrom_id = chrom_id
        
        chrom_dir = os.path.join(data_dir, chrom_id)
        if not os.path.isdir(chrom_dir):
            raise FileNotFoundError(f"Chromosome directory missing: {chrom_dir}")

        # -------- per-dataset file paths --------
        tf_path            = os.path.join(data_dir, "tf_tensor_all.pt")
        tg_path            = os.path.join(chrom_dir, f"tg_tensor_all_{chrom_id}.pt")
        atac_path          = os.path.join(chrom_dir, f"atac_window_tensor_all_{chrom_id}.pt")
        scaler_path        = os.path.join(chrom_dir, f"tg_scaler_{chrom_id}.pkl")
        window_map_path    = os.path.join(chrom_dir, f"window_map_{chrom_id}.json")
        dist_bias_path     = os.path.join(chrom_dir, f"dist_bias_{chrom_id}.pt")
        tf_ids_path        = os.path.join(data_dir, "tf_ids.pt")
        tg_ids_path        = os.path.join(chrom_dir, f"tg_ids_{chrom_id}.pt")
        tf_names_json      = os.path.join(data_dir, "tf_names.json")
        tg_names_json      = os.path.join(chrom_dir, f"tg_names_{chrom_id}.json")
        metacell_names_path= os.path.join(data_dir, "metacell_names.json")
        motif_mask_path    = os.path.join(chrom_dir, f"motif_mask_{chrom_id}.pt")

        # required tensors/metadata
        required = [
            tf_path, tg_path, atac_path, scaler_path,
            window_map_path, metacell_names_path,
            tf_names_json, tg_names_json,
            tf_ids_path, tg_ids_path
        ]
        for f in required:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required file not found: {f}")

        # -------- load tensors --------
        # stored [rows, cells]
        self.tf_tensor_all = torch.load(tf_path).float()            # [T_eval, C]
        self.tg_tensor_all = torch.load(tg_path).float()            # [G_eval, C]
        self.atac_window_tensor_all = torch.load(atac_path).float() # [W, C]
        
        self._paths = {
            "tf": tf_path, "tg": tg_path, "atac": atac_path,
            "scaler": scaler_path, "window_map": window_map_path,
            "dist_bias": dist_bias_path,
            "tf_ids": tf_ids_path, "tg_ids": tg_ids_path,
            "tf_names": tf_names_json, "tg_names": tg_names_json,
            "metacells": metacell_names_path,
            "tf_vocab": tf_vocab_path, "tg_vocab": tg_vocab_path,
            "motif_mask": motif_mask_path
        }

        # ------ load distance bias -------
        if os.path.exists(dist_bias_path):
            bias_WG = torch.load(dist_bias_path).float()
            if bias_WG.shape[0] == self.atac_window_tensor_all.shape[0]:
                self.dist_bias_tensor = bias_WG.T.contiguous()      # [G_eval, W]
            elif bias_WG.shape[1] == self.atac_window_tensor_all.shape[0]:
                self.dist_bias_tensor = bias_WG.contiguous()        # already [G_eval, W]
            else:
                raise ValueError(f"dist_bias_{chrom_id}.pt shape mismatch: {tuple(bias_WG.shape)}")
        else:
            self.dist_bias_tensor = None
        
        # -------- metadata --------
        with open(window_map_path, "r") as f:
            self.window_map = json.load(f)
        with open(metacell_names_path, "r") as f:
            self.metacell_names = json.load(f)

        # names (for logging/eval)
        if os.path.exists(tf_names_json):
            with open(tf_names_json, "r") as f:
                self.tf_names = json.load(f)
            self.tf_names = [self.standardize_name(i) for i in self.tf_names]
        else:
            # last resort: make placeholders
            self.tf_names = [f"TF_{i}" for i in range(self.tf_tensor_all.shape[0])]
            logging.warning("TF names not found; using placeholders.")

        with open(tg_names_json, "r") as f:
            self.tg_names = json.load(f)
        self.tg_names = [self.standardize_name(i) for i in self.tg_names]

        # counts
        self.num_windows = self.atac_window_tensor_all.shape[0]
        self.num_cells   = self.tf_tensor_all.shape[1]

        # -------- load common vocabs (for reference / debugging) --------
        # not strictly required at runtime if tf_ids/tg_ids exist, but handy to validate
        self.tf_vocab_path = tf_vocab_path
        self.tg_vocab_path = tg_vocab_path
        self.tf_name2id = None
        self.tg_name2id = None

        if self.tf_vocab_path and os.path.exists(self.tf_vocab_path):
            with open(self.tf_vocab_path, "r") as f:
                obj = json.load(f)
            # handle both {"name_to_id":...} or flat dict
            if "name_to_id" in obj:
                self.tf_name2id = {self.standardize_name(k): v for k, v in obj["name_to_id"].items()}
            else:
                self.tf_name2id = {self.standardize_name(k): v for k, v in obj.items()}

        if self.tg_vocab_path and os.path.exists(self.tg_vocab_path):
            with open(self.tg_vocab_path, "r") as f:
                obj = json.load(f)
            if "name_to_id" in obj:
                self.tg_name2id = {self.standardize_name(k): v for k, v in obj["name_to_id"].items()}
            else:
                self.tg_name2id = {self.standardize_name(k): v for k, v in obj.items()}


        # -------- load per-dataset ids (preferred) or derive from names --------
        if os.path.exists(tf_ids_path):
            self.tf_ids = torch.load(tf_ids_path).long()  # [T_eval]
        else:
            # derive from names using tf_name2id (if provided)
            if self.tf_name2id is None:
                raise FileNotFoundError(
                    f"tf_ids.pt not found and no tf_vocab_path provided to map names → ids."
                )
            missing = [n for n in self.tf_names if n not in self.tf_name2id]
            if missing:
                logging.warning(f"TF: {len(missing)} names missing in common vocab (e.g. {missing[:10]})")
            self.tf_ids = torch.tensor([self.tf_name2id[n] for n in self.tf_names if n in self.tf_name2id],
                                       dtype=torch.long)

        if os.path.exists(tg_ids_path):
            self.tg_ids = torch.load(tg_ids_path).long()  # [G_eval]
        else:
            if self.tg_name2id is None:
                raise FileNotFoundError(
                    f"tg_ids.pt not found and no tg_vocab_path provided to map names → ids."
                )
            missing = [n for n in self.tg_names if n not in self.tg_name2id]
            if missing:
                logging.warning(f"TG: {len(missing)} names missing in common vocab (e.g. {missing[:10]})")
            self.tg_ids = torch.tensor([self.tg_name2id[n] for n in self.tg_names if n in self.tg_name2id],
                                       dtype=torch.long)

        # sanity: lengths must match row counts of tensors
        if self.tf_tensor_all.shape[0] != self.tf_ids.numel():
            logging.warning(
                f"TF rows ({self.tf_tensor_all.shape[0]}) != tf_ids ({self.tf_ids.numel()}). "
                "Make sure you saved tf_tensor_all AFTER dropping TFs not in vocab."
            )
        if self.tg_tensor_all.shape[0] != self.tg_ids.numel():
            logging.warning(
                f"TG rows ({self.tg_tensor_all.shape[0]}) != tg_ids ({self.tg_ids.numel()}). "
                "Make sure you saved tg_tensor_all AFTER dropping TGs not in vocab."
            )
        if self.dist_bias_tensor is not None and self.dist_bias_tensor.shape[0] != self.tg_ids.numel():
            logging.warning(
                f"Bias G dimension ({self.dist_bias_tensor.shape[0]}) != tg_ids ({self.tg_ids.numel()})."
            )
            
        # -------- load motif mask (TG x TF) --------
        if os.path.exists(motif_mask_path):
            motif_mask_tensor = torch.load(motif_mask_path).float()
            # sanity check: TG dimension must match tg_ids, TF dimension must match tf_ids
            if motif_mask_tensor.shape[0] != self.tg_ids.numel() or motif_mask_tensor.shape[1] != self.tf_ids.numel():
                logging.warning(
                    f"Motif mask shape {tuple(motif_mask_tensor.shape)} "
                    f"!= (TG {self.tg_ids.numel()}, TF {self.tf_ids.numel()})"
                )
            self.motif_mask_tensor = motif_mask_tensor
        else:
            logging.warning("Motif mask file not found — using zeros.")
            self.motif_mask_tensor = torch.zeros((self.tg_ids.numel(), self.tf_ids.numel()), dtype=torch.float32)


        logging.info(
            f"Loaded dataset {data_dir} [{chrom_id}]\n"
            f" - TFs: {self.tf_tensor_all.shape[0]} (ids: {self.tf_ids.numel()})\n"
            f" - TGs: {self.tg_tensor_all.shape[0]} (ids: {self.tg_ids.numel()})\n"
            f" - Windows: {self.num_windows}\n"
            f" - Metacells: {self.num_cells}"
        )

        # scaler for inverse-transform
        self.scaler = joblib.load(scaler_path)

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
    def inverse_transform(self, preds: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(preds)

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
        return name.capitalize()
