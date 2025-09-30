import torch
import pandas as pd
import numpy as np
import logging
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os

def plot_per_gene_correlation_scatterplot(
    model,
    dataloader,
    device="cuda:0",
    *,
    zscore_tf: bool = False,
    inverse_scaler=None,           # e.g. dataset.scaler (StandardScaler) or None
    title: str = "Predicted vs True",
    max_points: int = 5000,
    outpath: str = None
):
    """
    Scatter plot of predicted vs. true TG expression.

    Parameters
    ----------
    model : torch.nn.Module
        Your MultiomicTransformer.
    dataloader : torch.utils.data.DataLoader
        Should yield (atac_wins, tf_tensor, tg_true, bias, tf_ids, tg_ids)
        or (atac_wins, tf_tensor, tg_true, bias).
    device : str or torch.device
        Device for inference.
    zscore_tf : bool
        Whether to z-score TF inputs per cell (recommended for cross-dataset).
    inverse_scaler : object or None
        Optional scaler with .mean_ and .scale_ (e.g., StandardScaler). If provided,
        both preds and truth will be inverse-transformed feature-wise.
    title : str
        Plot title.
    max_points : int
        Randomly subsample to this many rows for plotting (to keep the figure readable).
    outpath : str or None
        If given, save the figure to this path; else show().

    Returns
    -------
    corr : float
        Pearson correlation on all points (in the space actually plotted:
        raw if inverse_scaler is given; otherwise the z-space).
    """
    model.eval()
    preds_all, true_all = [], []

    with torch.no_grad():
        for batch in dataloader:
            # Support both 4-tuple and 6-tuple batches
            if len(batch) == 6:
                atac_wins, tf_tensor, tg_true, bias, tf_ids, tg_ids = batch
            elif len(batch) == 4:
                atac_wins, tf_tensor, tg_true, bias = batch
                tf_ids = tg_ids = None
            else:
                raise ValueError(
                    f"Unexpected batch size {len(batch)}. "
                    "Expected 4 or 6 elements from the dataloader."
                )

            atac_wins = atac_wins.to(device)
            tf_tensor = tf_tensor.to(device)
            tg_true   = tg_true.to(device)
            bias      = bias.to(device)
            if tf_ids is not None: tf_ids = tf_ids.to(device)
            if tg_ids is not None: tg_ids = tg_ids.to(device)

            # Per-cell TF z-score (recommended)
            if zscore_tf:
                mu = tf_tensor.mean(dim=1, keepdim=True)
                sd = tf_tensor.std(dim=1, keepdim=True).clamp_min(1e-6)
                tf_tensor = (tf_tensor - mu) / sd

            # Forward
            outputs = model(
                atac_wins, tf_tensor,
                tf_ids=tf_ids, tg_ids=tg_ids, bias=bias
            )  # [B, G]

            preds_all.append(outputs.cpu().numpy())
            true_all.append(tg_true.cpu().numpy())

    preds = np.vstack(preds_all)  # [N_cells, G]
    tgts  = np.vstack(true_all)   # [N_cells, G]

    # Optional inverse-transform to raw space (feature-wise)
    if inverse_scaler is not None:
        # Expecting sklearn StandardScaler-like object with mean_/scale_
        if getattr(inverse_scaler, "scale_", None) is not None:
            preds = preds * inverse_scaler.scale_
            tgts  = tgts  * inverse_scaler.scale_
        if getattr(inverse_scaler, "mean_", None) is not None:
            preds = preds + inverse_scaler.mean_
            tgts  = tgts  + inverse_scaler.mean_

    # Pearson on all points
    corr = float(pearsonr(preds.ravel(), tgts.ravel())[0])
    logging.info(f"Pred vs True Pearson: {corr:.3f} "
                 f"({ 'raw' if inverse_scaler is not None else 'z-space' })")

    # Subsample rows for plotting (keeps gene ordering intact)
    n_rows = preds.shape[0]
    if max_points is not None and n_rows > max_points:
        sel = np.random.default_rng(42).choice(n_rows, size=max_points, replace=False)
        y_true = tgts[sel].ravel()
        y_pred = preds[sel].ravel()
    else:
        y_true = tgts.ravel()
        y_pred = preds.ravel()

    # Plot
    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(y_true, y_pred, alpha=0.25, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlim(lims); plt.ylim(lims)
    space_tag = "raw" if inverse_scaler is not None else "z-space"
    plt.title(f"{title}\nPearson r = {corr:.2f} ({space_tag})")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=150)
        plt.close()
    else:
        plt.show()

    return corr

        
def per_gene_correlation(model, dataloader, gpu_id=0, gene_names=None):
    """
    Compute Pearson & Spearman correlation per gene across the test set.

    Args:
        model       : trained PyTorch model
        dataloader  : DataLoader over test set
        gpu_id      : device id
        gene_names  : list of gene names for annotation (optional)

    Returns:
        DataFrame with [gene, pearson, spearman]
    """
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for atac_wins, tf_tensor, targets, bias in dataloader:
            atac_wins, tf_tensor, targets, bias = (
                atac_wins.to(gpu_id),
                tf_tensor.to(gpu_id),
                targets.to(gpu_id),
                bias.to(gpu_id)
            )
            output = model(atac_wins, tf_tensor, bias=bias)
            preds.append(output.cpu().numpy())
            tgts.append(targets.cpu().numpy())

    preds = np.concatenate(preds, axis=0)   # [samples, num_genes]
    tgts  = np.concatenate(tgts, axis=0)
    
    # inverse-transform
    # preds_rescaled = scaler.inverse_transform(preds)
    # tgts_rescaled  = scaler.inverse_transform(tgts)
    preds_rescaled = preds
    tgts_rescaled  = tgts

    results = []
    for i in range(tgts_rescaled.shape[1]):  # loop over genes
        if np.std(tgts_rescaled[:, i]) < 1e-8:   # avoid constant targets
            pear, spear = np.nan, np.nan
        else:
            pear, _ = pearsonr(preds_rescaled[:, i], tgts_rescaled[:, i])
            spear, _ = spearmanr(preds_rescaled[:, i], tgts_rescaled[:, i])
        results.append((pear, spear))

    df = pd.DataFrame(results, columns=["pearson", "spearman"])
    if gene_names is not None:
        df.insert(0, "gene", gene_names)
    df["label"] = (df["pearson"] > 0.2).astype(int)
    
    return df

def train_classifier(feature_matrix, labels, out_prefix):
    
    X = feature_matrix
    y = labels
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    clf = RandomForestClassifier(
        n_estimators=200, 
        random_state=42,
        class_weight="balanced"   # upweight minority class
    )
    clf.fit(X_train_res, y_train_res)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    avg_prec = average_precision_score(y_test, y_score)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # ROC curve
    axes[0].plot(fpr, tpr, color="#4195df", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    # Precision-Recall curve
    axes[1].plot(recall, precision, color="#4195df", lw=2, label=f"AP = {avg_prec:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_can_train_classifier_auroc_auprc.png", dpi=300)
    plt.close()
    
    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_score))
    
def plot_gene_correlation_distribution(corr_df, out_prefix):
    """
    Plot distributions of per-gene Pearson & Spearman correlations.
    """
    plt.figure(figsize=(5,4))
    sns.violinplot(data=corr_df[['pearson','spearman']], inner="quartile", palette="Set2")
    plt.ylabel("Correlation")
    plt.title("Distribution of per-gene correlations")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_violin.png", dpi=300)
    plt.close()
