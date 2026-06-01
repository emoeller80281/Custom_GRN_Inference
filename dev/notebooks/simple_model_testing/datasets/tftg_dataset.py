import torch
from torch.utils.data import Dataset

class TFTGEdgeBagDataset(Dataset):
    def __init__(
        self,
        inputs,
        *,
        tf_embeddings_tensor,
        tf_mask_tensor,
        atac_peak_tensor,
    ):
        self.inputs = inputs
        self.tf_embeddings_tensor = tf_embeddings_tensor
        self.tf_mask_tensor = tf_mask_tensor
        self.atac_peak_tensor = atac_peak_tensor

    def __len__(self):
        return len(self.inputs["label"])

    def __getitem__(self, idx):
        tf_idx = self.inputs["tf_idx"][idx]
        tg_idx = self.inputs["tg_idx"][idx]

        peak_indices = self.inputs["peak_indices"][idx]          # [P]
        peak_sequences = self.atac_peak_tensor[peak_indices]     # [P, L, 4]

        tf_embedding = self.tf_embeddings_tensor[tf_idx]         # [T, D]
        tf_mask = self.tf_mask_tensor[tf_idx]                    # [T]

        return {
            "label": self.inputs["label"][idx],

            "tf_name": self.inputs["tf_name"][idx],
            "tg_name": self.inputs["tg_name"][idx],
            "cell_ids": self.inputs["cell_ids"][idx],
            "tf_idx": tf_idx,
            "tg_idx": tg_idx,
            "tf_embedding": tf_embedding.float(),
            "tf_mask": tf_mask.bool(),
            "peak_indices": peak_indices,
            "peak_sequences": peak_sequences,
            "peak_distance": self.inputs["peak_distance"][idx].float(),
            "peak_mask": self.inputs["peak_mask"][idx].bool(),
            "peak_accessibility": self.inputs["peak_accessibility"][idx].float(),
            "tf_expression": self.inputs["tf_expression"][idx].float(),
            "tg_expression": self.inputs["tg_expression"][idx].float(),
        }
        
def collate_tftg_edge_bags(batch):
    output = {
        "label": torch.stack([b["label"] for b in batch]).float(),

        "tf_idx": torch.stack([b["tf_idx"] for b in batch]).long(),
        "tg_idx": torch.stack([b["tg_idx"] for b in batch]).long(),

        "tf_embedding": torch.stack([b["tf_embedding"] for b in batch]),
        "tf_mask": torch.stack([b["tf_mask"] for b in batch]),

        "peak_indices": torch.stack([b["peak_indices"] for b in batch]),
        "peak_sequences": torch.stack([b["peak_sequences"] for b in batch]),
        "peak_distance": torch.stack([b["peak_distance"] for b in batch]),
        "peak_mask": torch.stack([b["peak_mask"] for b in batch]),

        "peak_accessibility": torch.stack([b["peak_accessibility"] for b in batch]),
        "tf_expression": torch.stack([b["tf_expression"] for b in batch]),
        "tg_expression": torch.stack([b["tg_expression"] for b in batch]),

        "tf_name": [b["tf_name"] for b in batch],
        "tg_name": [b["tg_name"] for b in batch],
        "cell_ids": [b["cell_ids"] for b in batch],
    }

    E, C = output["tf_expression"].shape
    output["cell_mask"] = torch.ones(E, C, dtype=torch.bool)

    return output