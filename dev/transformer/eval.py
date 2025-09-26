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

def plot_per_gene_correlation_scatterplot(model, dataloader, scaler, gpu_id=0, outpath=None):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for atac_wins, tf_tensor, targets in dataloader:
            atac_wins, tf_tensor = atac_wins.to(gpu_id), tf_tensor.to(gpu_id)
            output = model(atac_wins, tf_tensor)
            preds.append(output.cpu().numpy())
            tgts.append(targets.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    tgts  = np.concatenate(tgts, axis=0)

    # inverse-transform
    # preds_rescaled = scaler.inverse_transform(preds)
    # tgts_rescaled  = scaler.inverse_transform(tgts)
    preds_rescaled = preds
    tgts_rescaled  = tgts

    corr, _ = pearsonr(preds_rescaled.ravel(), tgts_rescaled.ravel())
    logging.info(f"Test Pearson correlation: {corr:.3f}")

    plt.figure(figsize=(6,6))
    plt.scatter(tgts_rescaled, preds_rescaled, alpha=0.5, s=5)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title(f"Predicted vs True (r={corr:.3f})")
    plt.plot([tgts_rescaled.min(), tgts_rescaled.max()],
             [tgts_rescaled.min(), tgts_rescaled.max()], 'r--')
    if outpath:
        plt.savefig(outpath, dpi=300)
    else:
        plt.show()
        
def per_gene_correlation(model, dataloader, scaler, gpu_id=0, gene_names=None):
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
        for atac_wins, tf_tensor, targets in dataloader:
            atac_wins, tf_tensor = atac_wins.to(gpu_id), tf_tensor.to(gpu_id)
            output = model(atac_wins, tf_tensor)
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
