import os, sys
from typing import Literal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    precision_recall_curve, 
    roc_curve,
)

def _create_random_distribution(scores, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.asarray(scores)   # works for Series or ndarray, no copy if already ndarray
    return rng.uniform(arr.min(), arr.max(), size=arr.shape[0])

def _balance_pos_neg(labels, scores):
    true_scores = scores[labels == 1]
    false_scores = scores[labels == 0]
    
    n_pos = len(true_scores)
    n_neg = len(false_scores)
    
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative examples are required for balancing.")
    
    if n_pos > n_neg:
        true_scores_balanced = np.random.choice(true_scores, size=n_neg, replace=False)
        false_scores_balanced = false_scores    
    elif n_neg > n_pos:
        false_scores_balanced = np.random.choice(false_scores, size=n_pos, replace=False)
        true_scores_balanced = true_scores
        
    balanced_labels = np.concatenate([np.ones_like(true_scores_balanced), np.zeros_like(false_scores_balanced)])
    balanced_scores = np.concatenate([true_scores_balanced, false_scores_balanced])
    
    # Shuffle the balanced dataset
    indices = np.arange(len(balanced_labels))
    np.random.shuffle(indices)
    balanced_labels = balanced_labels[indices]
    balanced_scores = balanced_scores[indices]
    
    return balanced_labels, balanced_scores


def plot_auroc_auprc(
    labels,
    scores,
    roc_line_color="#4195df",
    prc_line_color="#4195df",
    rand_line_color="#747474",
    title=None,
    plot_type: Literal["both", "roc", "prc"] = "both",
) -> plt.Figure:
    """
    labels: array-like of 0/1 labels
    scores: array-like of predicted probabilities after sigmoid

    plot_type:
        "both" -> plot ROC and PRC
        "roc"  -> plot ROC only
        "prc"  -> plot PRC only
    """

    if plot_type not in {"both", "roc", "prc"}:
        raise ValueError("plot_type must be one of: 'both', 'roc', or 'prc'.")

    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores).astype(float).ravel()

    rand_scores = _create_random_distribution(scores)

    plots_to_make = []
    if plot_type in {"both", "roc"}:
        plots_to_make.append("roc")
    if plot_type in {"both", "prc"}:
        plots_to_make.append("prc")

    ncols = len(plots_to_make)
    figsize = (7, 4) if ncols == 2 else (4, 4)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=ncols,
        figsize=figsize,
    )

    if ncols == 1:
        axes = [axes]

    for ax, current_plot in zip(axes, plots_to_make):
        
        ax.set_box_aspect(1)
        
        if current_plot == "roc":
            auroc = roc_auc_score(labels, scores)
            fpr, tpr, _ = roc_curve(labels, scores)

            rand_fpr, rand_tpr, _ = roc_curve(labels, rand_scores)
            rand_auroc = roc_auc_score(labels, rand_scores)

            roc_line, = ax.plot(
                fpr,
                tpr,
                lw=2,
                color=roc_line_color,
                label=f"AUROC = {auroc:.3f}",
                zorder=3,
            )

            rand_roc_line, = ax.plot(
                rand_fpr,
                rand_tpr,
                color=rand_line_color,
                linestyle="--",
                lw=2,
                label=f"Random = {rand_auroc:.3f}",
                zorder=2,
            )

            ax.plot(
                [0, 1],
                [0, 1],
                "k--",
                lw=1,
                alpha=0.5,
                zorder=1,
            )

            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.set_title("AUROC", fontsize=12)

            ax.legend(
                handles=[roc_line, rand_roc_line],
                labels=[f"AUROC = {auroc:.3f}", f"Random = {rand_auroc:.3f}"],
                bbox_to_anchor=(0.5, -0.28),
                loc="upper center",
                borderaxespad=0.0,
                facecolor="none",
                edgecolor="none",
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        elif current_plot == "prc":
            auprc = average_precision_score(labels, scores)
            prec, rec, _ = precision_recall_curve(labels, scores)

            rand_prec, rand_rec, _ = precision_recall_curve(labels, rand_scores)
            rand_auprc = average_precision_score(labels, rand_scores)

            pr_line, = ax.plot(
                rec,
                prec,
                lw=2,
                color=prc_line_color,
                label=f"AUPRC = {auprc:.3f}",
                zorder=3,
            )

            rand_pr_line, = ax.plot(
                rand_rec,
                rand_prec,
                color=rand_line_color,
                linestyle="--",
                lw=2,
                label=f"Random = {rand_auprc:.3f}",
                zorder=2,
            )

            ax.set_xlabel("Recall", fontsize=12)
            ax.set_ylabel("Precision", fontsize=12)
            ax.set_title("AUPRC", fontsize=12)

            ax.legend(
                handles=[pr_line, rand_pr_line],
                labels=[f"AUPRC = {auprc:.3f}", f"Random = {rand_auprc:.3f}"],
                bbox_to_anchor=(0.5, -0.28),
                loc="upper center",
                borderaxespad=0.0,
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    if title is not None:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()

    return fig

def plot_score_histograms(
    labels,
    scores, 
    n_bins=75, 
    random_state=42, 
    y_log=False,
    panel_kind="kde",
    density=False,
    title=None,
    y_lim=None,
    x_lim=None,
    balance_pos_neg: bool = True,
):
    
    fig, ax = plt.subplots(
        nrows=1, 
        ncols=1, 
        figsize=(4, 3),
        squeeze=False,
    )

    y = np.asarray(labels).astype(int).ravel()
    s = np.asarray(scores).astype(float).ravel()

    if balance_pos_neg:
        balanced_labels, balanced_scores = _balance_pos_neg(y, s)

        true_vals = balanced_scores[balanced_labels == 1]
        false_vals = balanced_scores[balanced_labels == 0]
    else:
        true_vals = s[y == 1]
        false_vals = s[y == 0]

    min_len = min(len(true_vals), len(false_vals))
    if min_len == 0:
        raise ValueError("Not enough positives/negatives to plot histograms.")

    rng = np.random.default_rng(random_state)
    true_vals = rng.choice(true_vals, size=min_len, replace=False)
    false_vals = rng.choice(false_vals, size=min_len, replace=False)

    combined = np.concatenate([true_vals, false_vals])
    bins = np.linspace(combined.min(), combined.max(), n_bins)

    plot_ax = ax[0, 0]

    if panel_kind == "hist":
        plot_ax.hist(
            false_vals,
            bins=bins,
            alpha=0.6,
            label="False",
            density=density,
        )
        plot_ax.hist(
            true_vals,
            bins=bins,
            alpha=0.6,
            label="True",
            density=density,
        )

        plot_ax.set_title("True vs False Scores", fontsize=12)
        plot_ax.set_xlabel("Score", fontsize=12)
        plot_ax.set_ylabel("Density" if density else "Count", fontsize=12)
        plot_ax.legend(fontsize=9)

    elif panel_kind == "kde":
        sns.kdeplot(
            false_vals,
            ax=plot_ax,
            label="False",
            fill=True,
            common_norm=False,
            bw_adjust=1.0,
            color="#747474",
        )
        sns.kdeplot(
            true_vals,
            ax=plot_ax,
            label="True",
            fill=True,
            common_norm=False,
            bw_adjust=1.0,
            color="#4195df"
        )

        plot_ax.set_title("True vs False Score Density", fontsize=12)
        plot_ax.set_xlabel("Score", fontsize=12)
        plot_ax.set_ylabel("Density", fontsize=12)
        plot_ax.legend(fontsize=9)

    if y_log:
        plot_ax.set_yscale("log")
        plot_ax.set_ylim(bottom=0.1)
        
    if y_lim:
        plot_ax.set_ylim(y_lim[0], y_lim[1])
    if x_lim:
        plot_ax.set_xlim(x_lim[0], x_lim[1])
        
    if title is not None:
        plt.suptitle(title, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    return fig