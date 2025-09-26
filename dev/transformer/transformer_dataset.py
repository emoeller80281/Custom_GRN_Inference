import os
import json
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import scipy.sparse as sp
import logging

class MultiomicTransformerDataset(Dataset):
    def __init__(self, data_dir, chrom_id):
        self.data_dir = data_dir
        self.chrom_id = chrom_id

        # ----- Paths to required files -----
        tf_path   = os.path.join(data_dir, "tf_tensor_all.pt")
        tg_path   = os.path.join(data_dir, f"tg_tensor_all_{chrom_id}.pt")
        atac_path = os.path.join(data_dir, f"atac_window_tensor_all_{chrom_id}.pt")
        scaler_path = os.path.join(data_dir, f"tg_scaler_{chrom_id}.pkl")
        window_map_path = os.path.join(data_dir, "window_map.json")
        tf_names_path = os.path.join(data_dir, "tf_names.pickle")
        tg_names_path = os.path.join(data_dir, f"tg_names_{chrom_id}.json")
        metacell_names_path = os.path.join(data_dir, f"metacell_names.json")
        
        for f in [tf_path, tg_path, atac_path, scaler_path, window_map_path,
            tf_names_path, tf_names_path, metacell_names_path]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Required precomputed file not found: {f}")

        # Load tensors
        self.tf_tensor_all = torch.load(tf_path)          # [num_tf, num_cells]
        self.tg_tensor_all = torch.load(tg_path)          # [num_tg, num_cells]
        self.atac_window_tensor_all = torch.load(atac_path)  # [num_windows, num_cells]

        # Load scaler for inverse-transform
        self.scaler = joblib.load(scaler_path)

        # Load metadata
        with open(window_map_path, 'r') as f:
            self.window_map = json.loads(f.read())
            
        with open(tg_names_path, 'r') as f:
            self.tg_names = json.loads(f.read())
        
        with open(metacell_names_path, 'r') as f:
            self.metacell_names = json.loads(f.read())
            
        with open(tf_names_path, "rb") as fp:
            self.tf_names = pickle.load(fp)

        # Infer metadata from shapes
        self.num_tf = self.tf_tensor_all.shape[0]
        self.num_windows = self.atac_window_tensor_all.shape[0]
        self.num_tg = self.tg_tensor_all.shape[0]

        logging.info(
            f"Loaded dataset from {data_dir}\n"
            f" - TFs: {self.num_tf}\n"
            f" - TGs: {self.num_tg}\n"
            f" - Windows: {self.num_windows}\n"
            f" - Metacells: {len(self.metacell_names)}"
        )

    def __len__(self):
        return self.tf_tensor_all.shape[1]  # number of metacells

    def __getitem__(self, idx):
        tf_tensor   = self.tf_tensor_all[:, idx]                            # [num_tf]
        atac_wins   = self.atac_window_tensor_all[:, idx].unsqueeze(-1)     # [num_windows, 1]
        tg_tensor   = self.tg_tensor_all[:, idx]                            # [num_tg]
        return atac_wins, tf_tensor, tg_tensor

    def load_window_map(self):
        with open(os.path.join(self.data_dir, "window_map.json")) as f:
            return json.load(f)

    def load_tf_list(self):
        with open(os.path.join(self.data_dir, "tf_list.pickle"), "rb") as fp:
            return pickle.load(fp)

    def inverse_transform(self, preds: np.ndarray) -> np.ndarray:
        """Inverse transform predictions back to original scale"""
        return self.scaler.inverse_transform(preds)