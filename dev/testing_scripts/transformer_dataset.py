import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class OnDiskWindowsDataset(torch.utils.data.Dataset):
    """
    Expects a manifest parquet with columns: barcode, path, wlen
    Each .npz at 'path' contains: tf_windows [W',TF], gene_biases [W',G]
    """
    def __init__(self, manifest_path: str, drop_empty: bool = True):
        self.manifest = pd.read_parquet(manifest_path)
        if drop_empty:
            self.manifest = self.manifest[self.manifest["wlen"] > 0].reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, i):
        row = self.manifest.iloc[i]
        d = np.load(row.path, allow_pickle=False, mmap_mode="r")  # fast & zero-copy
        return {
            "barcode": row.barcode,
            "tf_windows": d["tf_windows"],     # [W', TF] float32
            "gene_biases": d["gene_biases"],   # [W', G]  float32
        }

class WindowsWithTargets(Dataset):
    """
    Wraps OnDiskWindowsDataset (cached features) and attaches per-cell target vector [G].
    Requires that each OnDisk sample includes 'barcode'.
    """
    def __init__(self, manifest_path: str, rna_parquet_path: str, genes_npy_path: str, drop_empty: bool = True):
        self.base = OnDiskWindowsDataset(manifest_path, drop_empty=drop_empty)
        self.manifest = self.base.manifest.copy()       # has 'barcode', 'path', 'wlen'
        self.genes = np.load(genes_npy_path, allow_pickle=True)
        gene_index = pd.Index(self.genes.astype(str))

        rna_df = pd.read_parquet(rna_parquet_path).set_index("gene_id")  # genes in index
        rna_df = rna_df.reindex(gene_index).astype("float32")            # align + order to cached genes

        # Keep only barcodes present in RNA columns
        mask_df = rna_df.notna().astype(np.float32)
        # Fill NaNs with zeros; they will be ignored by the mask in loss
        rna_df = rna_df.fillna(0.0).astype(np.float32)

        # Reorder columns to match manifest
        cols = self.manifest["barcode"].tolist()
        self.rna_df = rna_df[cols]
        self.mask_df = mask_df[cols]

        self.rna_arr  = self.rna_df.to_numpy(dtype=np.float32, copy=False)   # [G, N]
        self.mask_arr = self.mask_df.to_numpy(dtype=np.float32, copy=False)  # [G, N]
        self.b2col = {b: i for i, b in enumerate(cols)}

    def __len__(self): 
        return len(self.manifest)

    def __getitem__(self, i: int):
        row = self.manifest.iloc[i]
        rec = self.base[i]  # expects dict with tf_windows [W',TF], gene_biases [W',G], barcode
        b = row.barcode
        col = self.b2col[b]
        target = self.rna_arr[:, col]  # [G] float32
        tmask  = self.mask_arr[:, col]     # [G], float32 (1.0 if observed else 0.0)
        return {
            "barcode": b,
            "tf_windows": rec["tf_windows"],
            "gene_biases": rec["gene_biases"],
            "target": target,
            "target_mask": tmask,
        }


def make_collate(pad_to_max=True):
    def _collate(batch):
        B = len(batch)
        wlens = [b["tf_windows"].shape[0] for b in batch]
        if not any(wlens):
            raise ValueError("Batch has no active windows; adjust sampling/filters.")

        TF = G = None
        for b in batch:
            if b["tf_windows"].shape[0] > 0:
                TF = b["tf_windows"].shape[1]
                G  = b["gene_biases"].shape[1]
                break

        Wmax = max(wlens) if pad_to_max else max(wlens)  # you already pad_to_max=True
        tf_pad    = torch.zeros((B, Wmax, TF), dtype=torch.float32)
        bias_pad  = torch.zeros((B, Wmax, G),  dtype=torch.float32)
        kpm       = torch.ones((B, Wmax),      dtype=torch.bool)     # True=PAD
        targets   = torch.zeros((B, G),        dtype=torch.float32)
        tmask_pad = torch.zeros((B, G),        dtype=torch.float32)  # 1.0=keep, 0.0=ignore

        for i, b in enumerate(batch):
            w = b["tf_windows"].shape[0]
            if w > 0:
                tf_pad[i, :w]   = torch.from_numpy(b["tf_windows"])
                bias_pad[i, :w] = torch.from_numpy(b["gene_biases"])
                kpm[i, :w] = False
            targets[i]   = torch.from_numpy(b["target"])
            tmask_pad[i] = torch.from_numpy(b["target_mask"])

        return tf_pad, bias_pad, kpm, targets, tmask_pad
    return _collate
