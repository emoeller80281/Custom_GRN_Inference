import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pickle
import logging

class MultiomicTransformerDataset(Dataset):
    def __init__(self, data_dir, chrom_id):
        self.data_dir = data_dir

        # Load pseudobulk (genes x metacells, peaks x metacells)
        self.TG_pseudobulk = pd.read_csv(
            os.path.join(data_dir, f"TG_{chrom_id}_specific_pseudobulk_agg.csv"),
            index_col=0
        )
        self.TG_pseudobulk = self.TG_pseudobulk.groupby(self.TG_pseudobulk.index).sum()
        
        # Scale the target gene expression to a mean of 0 and stdev of 1
        self.scaler = StandardScaler()
        self.TG_scaled = self.scaler.fit_transform(self.TG_pseudobulk.values)
        
        # Replace TG expression with scaled values for training
        self.TG_pseudobulk.iloc[:, :] = self.TG_scaled

        self.RE_pseudobulk = pd.read_csv(
            os.path.join(data_dir, f"RE_{chrom_id}_specific_pseudobulk_agg.csv"),
            index_col=0
        )
        self.RE_pseudobulk = self.RE_pseudobulk.groupby(self.RE_pseudobulk.index).sum()

        self.window_map = self.load_window_map()
        self.tf_list = self.load_tf_list()
        
        # Get the global gene expression for finding TF expression
        self.TG_pseudobulk_global = pd.read_csv(
            os.path.join(data_dir, "TG_pseudobulk_global.csv"),
            index_col=0
        )
        self.TG_pseudobulk_global = self.TG_pseudobulk_global[~self.TG_pseudobulk_global.index.duplicated(keep="first")]

        # Save the fitted scaler so you can inverse-transform later
        joblib.dump(self.scaler, os.path.join(self.data_dir, "tg_scaler.pkl"))

        # Metadata
        self.metacell_names = self.TG_pseudobulk.columns.tolist()
        self.num_tf = len(set(self.tf_list))
        self.num_windows = max(self.window_map.values()) + 1
        self.num_tg = self.TG_pseudobulk.shape[0]   # number of genes
        

    def __len__(self):
        return len(self.metacell_names)

    def __getitem__(self, idx):
        col_name = self.metacell_names[idx]

        # --- TF expression (genome-wide TFs) ---
        tf_expr = self.TG_pseudobulk_global.reindex(
            self.tf_list
        )[col_name].fillna(0).values.astype("float32")
        tf_tensor = torch.tensor(tf_expr)

        # --- Collapse peaks into windows [num_windows, 1] ---
        atac_wins = torch.zeros((self.num_windows, 1), dtype=torch.float32)
        for peak, win_idx in self.window_map.items():
            if peak in self.RE_pseudobulk.index:
                val = self.RE_pseudobulk.loc[peak, col_name]
                # If multiple rows (Series), reduce to scalar
                if isinstance(val, pd.Series):
                    val = val.sum()   # or mean(), depending on biology
                atac_wins[win_idx, 0] += float(val)

        # --- TG expression (chr-specific targets) ---
        tg_expr = self.TG_pseudobulk[col_name].values.astype("float32")
        tg_tensor = torch.tensor(tg_expr)

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