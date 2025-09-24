import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pathlib import Path
import pybedtools
import pickle

class MultiomicTransformerDataset(Dataset):
    def __init__(self, data_dir, chrom_id):
        self.data_dir = data_dir

        # Load pseudobulk (genes x metacells, peaks x metacells)
        self.TG_pseudobulk = pd.read_csv(
            os.path.join(data_dir, f"TG_{chrom_id}_specific_pseudobulk.csv"),
            index_col=0
        )
        self.RE_pseudobulk = pd.read_csv(
            os.path.join(data_dir, f"RE_{chrom_id}_specific_pseudobulk.csv"),
            index_col=0
        )

        self.window_map = self.load_window_map()
        self.tf_list = self.load_tf_list()

        self.metacell_names = self.TG_pseudobulk.columns
        self.num_samples = len(self.metacell_names)
        
        self.num_tf = len(set(self.tf_list).intersection(self.TG_pseudobulk.index))
        self.num_windows = max(self.window_map.values()) + 1
        self.num_tg = self.TG_pseudobulk.shape[0]   # or restrict further

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        col_name = self.metacell_names[idx]

        # --- TF expression ---
        tf_expr = self.TG_pseudobulk.loc[
            self.TG_pseudobulk.index.intersection(self.tf_list), col_name
        ].values.astype("float32")
        tf_tensor = torch.tensor(tf_expr)  # [num_tf]

        # --- Collapse peaks into windows ---
        num_windows = max(self.window_map.values()) + 1
        atac_wins = torch.zeros((num_windows, 1), dtype=torch.float32)
        for peak, win_idx in self.window_map.items():
            if peak in self.RE_pseudobulk.index:
                atac_wins[win_idx, 0] += self.RE_pseudobulk.at[peak, col_name]

        # --- TG expression (targets) ---
        tg_expr = self.TG_pseudobulk[col_name].values.astype("float32")
        tg_tensor = torch.tensor(tg_expr)  # [num_genes]

        return tf_tensor, atac_wins, tg_tensor

    def load_window_map(self):
        with open(os.path.join(self.data_dir, "window_map.json")) as f:
            return json.load(f)

    def load_tf_list(self):
        with open(os.path.join(self.data_dir, "tf_list.pickle"), "rb") as fp:
            return pickle.load(fp)