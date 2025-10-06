#!/usr/bin/env python

import os
import csv
import torch
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

from config.settings import *
from multiomic_transformer.datasets.dataset import MultiomicTransformerDataset
from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.utils import plotting

def load_model_from_run_params(run_dir, tf_vocab_size, tg_vocab_size, device="cpu"):
    """
    Load MultiomicTransformer with the exact parameters from run_parameters.json
    """
    params_path = os.path.join(run_dir, "run_parameters.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Could not find {params_path}")

    with open(params_path, "r") as f:
        run_params = json.load(f)

    model = MultiomicTransformer(
        d_model=run_params.get("d_model", 384),
        num_heads=run_params.get("num_heads", 6),
        num_layers=run_params.get("num_layers", 3),
        d_ff=run_params.get("d_feedforward", 768),  # note the key
        dropout=run_params.get("dropout", 0.1),
        tf_vocab_size=tf_vocab_size,
        tg_vocab_size=tg_vocab_size,
        bias_scale=run_params.get("attn_bias_scale", 1.0),
        use_shortcut=run_params.get("use_shortcut", True),
        use_motif_mask=run_params.get("use_motif_mask", False),
        lambda_l1=run_params.get("shortcut_l1", 0.0),
        lambda_l2=run_params.get("shortcut_l2", 0.0),
        topk=run_params.get("shortcut_topk", None),
        shortcut_dropout=run_params.get("shortcut_dropout", 0.0),
    )

    model = model.to(device)
    return model

def subset_scaler(original_scaler, kept_indices):
    from sklearn.preprocessing import StandardScaler
    new_scaler = StandardScaler()
    new_scaler.mean_ = original_scaler.mean_[kept_indices]
    new_scaler.scale_ = original_scaler.scale_[kept_indices]
    new_scaler.var_ = original_scaler.var_[kept_indices]
    new_scaler.n_features_in_ = len(kept_indices)
    return new_scaler

def run_test(checkpoint_path, out_dir, batch_size=BATCH_SIZE, gpu_id=0):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    ckpt_dir = os.path.dirname(checkpoint_path)

    # ----- Load dataset -----
    dataset = MultiomicTransformerDataset(
        data_dir=SAMPLE_DATA_CACHE_DIR,
        chrom_id=CHROM_ID,
        tf_vocab_path=os.path.join(COMMON_DATA, "tf_vocab.json"),
        tg_vocab_path=os.path.join(COMMON_DATA, "tg_vocab.json"),
    )

    tf_vocab_size = len(dataset.tf_name2id)
    tg_vocab_size = len(dataset.tg_name2id)

    # ----- Load model -----
    model = load_model_from_run_params(ckpt_dir, tf_vocab_size, tg_vocab_size, device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # ----- Dataloader -----
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MultiomicTransformerDataset.collate_fn,
    )

    # ----- Evaluate -----
    preds_list, tgts_list, tg_ids_list = [], [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = [
                x.to(device) for x in batch
            ]
            mask_arg = motif_mask if USE_MOTIF_MASK else None

            preds, _ = model(atac_wins, tf_tensor,
                             tf_ids=tf_ids, tg_ids=tg_ids,
                             bias=bias, motif_mask=mask_arg)

            loss = torch.nn.functional.mse_loss(preds, targets)
            total_loss += loss.item(); n_batches += 1

            preds_list.append(preds.cpu().numpy())
            tgts_list.append(targets.cpu().numpy())
            tg_ids_list.append(tg_ids.cpu().numpy())

    val_loss = total_loss / max(1, n_batches)

    # Stack results
    all_preds = np.concatenate(preds_list, axis=0)
    all_tgts  = np.concatenate(tgts_list, axis=0)
    all_tg_ids = np.concatenate(tg_ids_list, axis=0)

    # scaler = subset_scaler(dataset.scaler, all_tg_ids)
    # preds_rescaled = scaler.inverse_transform(all_preds)
    # tgts_rescaled  = scaler.inverse_transform(all_tgts)
    
    preds_rescaled = all_preds
    tgts_rescaled  = all_tgts

    pearson_corr, _ = pearsonr(preds_rescaled.ravel(), tgts_rescaled.ravel())
    spearman_corr, _ = spearmanr(preds_rescaled.ravel(), tgts_rescaled.ravel())

    logging.info(
        f"[Test] Val Loss: {val_loss:.4f} | "
        f"Pearson: {pearson_corr:.3f} | Spearman: {spearman_corr:.3f}"
    )

    # ----- Save CSV -----
    csv_path = os.path.join(out_dir, "test_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Val Loss", "Pearson", "Spearman"])
        writer.writerow([val_loss, pearson_corr, spearman_corr])

    # ----- Plot scatter -----
    scatter_fig = plotting.plot_per_gene_correlation_scatterplot(
        model, loader, mask_arg=USE_MOTIF_MASK, gpu_id=gpu_id
    )
    scatter_fig.savefig(os.path.join(out_dir, "test_scatter.png"), dpi=300)

    return val_loss, pearson_corr, spearman_corr



if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    # Output directory
    test_out = OUTPUT_DIR / "model_training_014"

    # Run test on final checkpoint
    ckpt_path = test_out / "trained_model.pt"
    run_test(ckpt_path, test_out, batch_size=BATCH_SIZE, gpu_id=0)
