import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def plot_per_gene_correlation_scatterplot(model, dataloader, mask_arg, gpu_id=0):
    model.eval()
    preds_list, tgts_list = [], []
    first_tg_ids = None
    with torch.no_grad():
        for atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask in dataloader:
            atac_wins = atac_wins.to(gpu_id)
            tf_tensor = tf_tensor.to(gpu_id)
            targets   = targets.to(gpu_id)
            bias      = bias.to(gpu_id)
            tf_ids    = tf_ids.to(gpu_id)
            tg_ids    = tg_ids.to(gpu_id)
            motif_mask= motif_mask.to(gpu_id)
            
            preds, _ = model(
                    atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias, motif_mask=mask_arg
                )
            preds_list.append(preds.cpu().numpy())
            tgts_list.append(targets.cpu().numpy())
            
            if first_tg_ids is None:
                first_tg_ids = tg_ids.cpu().numpy()

    total_preds = np.concatenate(preds_list, axis=0)
    total_tgts  = np.concatenate(tgts_list, axis=0)

    corr, _ = pearsonr(total_preds.ravel(), total_tgts.ravel())
    print(f"Test Pearson correlation: {corr:.3f}")

    fig = plt.figure(figsize=(6,6))
    plt.scatter(total_tgts, total_preds, alpha=0.5, s=5)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(f"Predicted vs True (r={corr:.3f})")
    plt.plot([total_tgts.min(), total_tgts.max()],
             [total_tgts.min(), total_tgts.max()], 'r--')
    
    return fig

def plot_pearson_corr_across_epochs(df, dataset_name, chrom_id):
    fig = plt.figure(figsize=(6, 5))
    plt.plot(df.index, df["Pearson"], linewidth=2, label="Pearson Correlation")

    plt.title(f"Training {dataset_name} {chrom_id} Pearson Correlation", fontsize=14)
    plt.ylim((0,1))
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Pearson Correlation", fontsize=13)
    plt.legend(fontsize=13)
    plt.tight_layout()
    
    return fig

def plot_train_val_loss(df, dataset_name, chrom_id):
    fig = plt.figure(figsize=(6, 5))
    plt.plot(df.index, df["Train Loss"], linewidth=2, label="Validation Loss")
    plt.plot(df.index, df["Val Loss"], linewidth=2, label="Training Loss")

    plt.title(f"Training {dataset_name} {chrom_id} Training/Validation Loss", fontsize=14)
    plt.ylim((0,1.5))
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)
    plt.xlabel("Epoch", fontsize=13)
    plt.ylabel("Loss", fontsize=13)
    plt.legend(fontsize=13)
    plt.tight_layout()
    
    return fig