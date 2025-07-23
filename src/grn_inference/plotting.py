import os
import math
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import dask.dataframe as dd
import dask.array as da

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dask_ml.model_selection import train_test_split
from dask.distributed import Client

from typing import Tuple

from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed

import xgboost as xgb

from grn_inference.model import (
    train_xgboost_dask, xgb_classifier_from_booster
)

def plot_xgboost_prediction_histogram(booster, X_test, fig_dir):
    logging.info("\tPlotting model prediction histogram")

    # Convert to DMatrix (required for Booster prediction)
    dtest = xgb.DMatrix(X_test)

    # Get predicted probabilities for the positive class
    y_pred_prob = booster.predict(dtest)
    
    os.makedirs(fig_dir, exist_ok=True)

    # Plot histogram of predicted probabilities
    plt.figure(figsize=(8, 6))
    plt.hist(y_pred_prob, bins=50, color='skyblue', edgecolor='black')
    plt.title("Histogram of XGBoost Prediction Probabilities", fontsize=18)
    plt.xlabel("Prediction Probability", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_prediction_probability_histogram.png", dpi=200)
    plt.close()

def plot_feature_importance(features: list, model, fig_dir: str):
    logging.info("\tPlotting feature importance barplot")
    
    os.makedirs(fig_dir, exist_ok=True)
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    else:
        booster = model  # assume already a Booster

    importance_dict = booster.get_score(importance_type="weight")

    # Build DataFrame ensuring all input features are represented (default to 0 if missing)
    feature_importances = pd.DataFrame({
        "Feature": features,
        "Importance": [importance_dict.get(f, 0) for f in features]
    }).sort_values(by="Importance", ascending=False)

    if feature_importances["Importance"].sum() == 0:
        logging.warning("All feature importances are zero; model may not have learned from any features.")
    
    # Plot the feature importances
    plt.figure(figsize=(8, 6))
    plt.barh(feature_importances["Feature"], feature_importances["Importance"], color="skyblue")
    plt.xlabel("Importance", fontsize=16)
    plt.ylabel("Feature", fontsize=16)
    plt.title("Feature Importance", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_feature_importance.png", dpi=200)
    plt.close()

def plot_feature_score_histograms(inferred_network, features, fig_dir):
    logging.info("\tPlotting feature score histograms")
    
    os.makedirs(fig_dir, exist_ok=True)

    # Step 1: Convert only necessary columns to pandas
    if isinstance(inferred_network, dd.DataFrame):
        logging.info("\tConverting feature columns from Dask to pandas for plotting")
        inferred_network_series = inferred_network[features].compute()
    else:
        inferred_network_series = inferred_network[features]

    ncols = 4
    nrows = math.ceil(len(features) / ncols)

    plt.figure(figsize=(5 * ncols, 4 * nrows))

    for i, feature in enumerate(features, 1):
        plt.subplot(nrows, ncols, i)
        plt.hist(inferred_network_series[feature].dropna(), bins=50, alpha=0.7, edgecolor='black')
        plt.title(f"{feature}", fontsize=14)
        plt.xlabel(feature, fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xlim((0, 1))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_feature_score_hist.png", dpi=300)
    plt.close()
    
def plot_feature_score_histograms_split_by_label(inferred_network, features, fig_dir, label_col="label"):
    """
    Plot overlapping histograms of feature scores split by binary label (True vs False).
    """
    logging.info("\tPlotting split histograms by label")

    os.makedirs(fig_dir, exist_ok=True)

    # Convert to pandas if needed
    if isinstance(inferred_network, dd.DataFrame):
        logging.info("\tConverting Dask dataframe to pandas")
        inferred_network_pd = inferred_network[features + [label_col]].compute()
    else:
        inferred_network_pd = inferred_network[features + [label_col]].copy()

    # Separate true/false label subsets
    true_df = inferred_network_pd[inferred_network_pd[label_col] == 1]
    false_df = inferred_network_pd[inferred_network_pd[label_col] == 0]

    ncols = 4
    nrows = math.ceil(len(features) / ncols)

    plt.figure(figsize=(5 * ncols, 4 * nrows))

    for i, feature in enumerate(features, 1):
        plt.subplot(nrows, ncols, i)
        
        # Drop NaNs
        true_vals = true_df[feature].dropna()
        false_vals = false_df[feature].dropna()

        # Determine min count
        min_len = min(len(true_vals), len(false_vals))

        # Randomly sample both to the same size
        true_vals_sampled = true_vals.sample(n=min_len, random_state=42)
        false_vals_sampled = false_vals.sample(n=min_len, random_state=42)

        # Compute common bin edges
        combined_vals = pd.concat([true_vals_sampled, false_vals_sampled])
        bins = np.linspace(combined_vals.min(), combined_vals.max(), 75)  # 150 equal-width bins

        # Plot histograms using the same bin edges
        plt.hist(false_vals_sampled, bins=bins, alpha=0.6, color="#747474", label="False Scores")
        plt.hist(true_vals_sampled, bins=bins, alpha=0.6, color='#4195df', label="True Scores")

        plt.title(feature, fontsize=14)
        plt.xlabel("Score", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend(fontsize=10)
        plt.xlim((0,1))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

    plt.tight_layout()
    output_path = f"{fig_dir}/xgboost_feature_score_hist_by_label.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logging.info(f"\tSaved histogram figure to: {output_path}")
    
def plot_feature_score_histogram(df, score_col, fig_dir):
    logging.info("\tPlotting feature score histogram")
    
    if isinstance(df, dd.DataFrame):
        logging.info("\tConverting feature columns from Dask to pandas for plotting")
        df_series = df[score_col].compute()
    else:
        # If it’s already a Pandas DataFrame, pull out the column as a Series:
        df_series = df[score_col]
    
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))

    plt.hist(df_series.dropna(), bins=50, alpha=0.7, edgecolor='black')
    plt.title(f"{score_col}", fontsize=14)
    plt.xlabel("Score", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xlim((0, 1))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{score_col}_hist.png"), dpi=300)
    plt.close()
    
def plot_multi_sample_feature_score_histograms(
    features,
    inferred_network1,
    inferred_network2,
    label1_name,
    label2_name, 
    ncols
):
    print("\tPlotting feature score histograms")
    
    # materialize only needed columns
    if isinstance(inferred_network1, dd.DataFrame):
        print("\tConverting feature columns from Dask to pandas for plotting")
        inferred_network1 = inferred_network1[features].compute()
    if isinstance(inferred_network2, dd.DataFrame):
        inferred_network2 = inferred_network2[features].compute()

    nrows = math.ceil(len(features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    # flatten axes for easy indexing
    axes_flat = axes.flat

    for ax, feature in zip(axes_flat, features):
        # draw into this axis explicitly:
        sns.histplot(
            inferred_network1[feature].dropna(),
            bins=50, alpha=0.7,
            color='#4195df', edgecolor="#032b5f",
            stat='proportion',
            label=label1_name,
            ax=ax
        )
        sns.histplot(
            inferred_network2[feature].dropna(),
            bins=50, alpha=0.7,
            color="#747474", edgecolor="#2D2D2D",
            stat='proportion',
            label=label2_name,
            ax=ax
        )

        # set titles/labels on the same ax
        ax.set_title(feature, fontsize=14)
        ax.set_xlabel(feature, fontsize=14)
        ax.set_ylabel("Proportion", fontsize=14)
        ax.set_xlim(0, 1)
        ax.tick_params(axis='both', labelsize=12)

    # turn off any leftover empty subplots
    for ax in axes_flat[len(features):]:
        ax.set_visible(False)

    # figure-level legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        fontsize=14,
        frameon=False
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    plt.show()

def plot_feature_boxplots(features, inferred_network_ddf, fig_dir):
    logging.info("Plotting feature importance boxplots")
    os.makedirs(fig_dir, exist_ok=True)

    # 0) Make sure you have a pandas DataFrame
    if hasattr(inferred_network_ddf, "compute"):
        net = inferred_network_ddf.compute()
    else:
        net = inferred_network_ddf

    n = len(features)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    axes = axes.flatten()

    for idx, feat in enumerate(features):
        ax = axes[idx]
        if feat not in net.columns:
            logging.warning(f"Missing feature `{feat}`, skipping.")
            ax.set_visible(False)
            continue

        # split by label and drop NA
        data_0 = net.loc[net.label==0, feat].dropna()
        data_1 = net.loc[net.label==1, feat].dropna()

        # if there's nothing to plot on one side, skip
        if data_0.empty and data_1.empty:
            ax.set_visible(False)
            continue

        ax.boxplot(
            [data_0, data_1],
            labels=["False","True"],
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor="lightgray"),
            medianprops=dict(color="black"),
        )
        ax.set_title(feat, fontsize=14)
        ax.set_ylabel("Score", fontsize=12)

    # hide any leftover axes beyond your features
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = os.path.join(fig_dir, "feature_boxplots.png")
    plt.savefig(out, dpi=200)
    plt.close()
    logging.info(f"Saved boxplots to {out}")

def partition_hist(feature_series, bins):
    counts, _ = np.histogram(feature_series.dropna().values, bins=bins)
    return counts

def plot_feature_score_histograms_dask(ddf, score_cols, fig_dir, df_name="inferred_net"):
    logging.info(f"  - Plotting feature score histograms using Dask")

    # Define common bins: assume normalized scores between 0 and 1
    bins = np.linspace(0, 1, 51)  # 50 bins → 51 edges

    ncols = 4
    nrows = int(np.ceil(len(score_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    axes_flat = axes.flat

    for ax, feat in zip(axes_flat, score_cols):
        logging.info(f"    → Computing histogram for {feat}")

        counts_per_part = ddf.map_partitions(
            lambda df, col=feat: partition_hist(df[col], bins),
            meta=(None, "i8")
        )

        total_counts = counts_per_part.sum(axis=0).compute()

        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(bin_centers, total_counts, width=(bins[1] - bins[0]), align="center")
        ax.set_title(f"{feat} histogram", fontsize=14)
        ax.set_xlabel(feat)
        ax.set_ylabel("Frequency (count)")

        # Hide unused axes
        for j in range(len(score_cols), len(axes_flat)):
            axes_flat[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{df_name}_score_histograms.png"), dpi=300)
    plt.close()
    logging.info("  - Done plotting histograms")

def plot_permutation_importance_plot(xgb_model, X_test, y_test, fig_dir):
    logging.info("\tPlotting permutation importance plot")
    
    os.makedirs(fig_dir, exist_ok=True)

    # Ensure input is clean and has no NaNs
    X_test_clean = X_test.copy()
    y_test_clean = y_test.copy()
    valid_rows = X_test_clean.notna().all(axis=1) & y_test_clean.notna()
    X_test_clean = X_test_clean[valid_rows]
    y_test_clean = y_test_clean[valid_rows]

    if X_test_clean.empty:
        logging.warning("No valid data for permutation importance plot.")
        return

    result = permutation_importance(
        estimator=xgb_model,
        X=X_test_clean,
        y=y_test_clean,
        n_repeats=10,
        random_state=42,
        scoring="roc_auc"
    )

    # Extract means and standard deviations
    importances = result.importances_mean
    std = result.importances_std
    feature_names = X_test_clean.columns.to_numpy()
    sorted_idx = np.argsort(importances)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.barh(
        y=range(len(importances)),
        width=importances[sorted_idx],
        xerr=std[sorted_idx],
        align='center',
        color='#747474',
        edgecolor='black'
    )
    plt.yticks(ticks=range(len(importances)), labels=feature_names[sorted_idx], fontsize=12)
    plt.xlabel("Decrease in ROC-AUC", fontsize=14)
    plt.title("Permutation Feature Importance", fontsize=16)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_permutation_importance.png", dpi=300)
    plt.close()
    
def plot_stability_boxplot(X: pd.DataFrame, y: pd.Series, feature_names: list[str], fig_dir: str):
    logging.info("\tPlotting stability boxplot using Dask-trained XGBoost")
    
    os.makedirs(fig_dir, exist_ok=True)

    # Start local Dask client for multithreaded parallelism
    client = Client(processes=False)
    logging.info(f"\tDask client started: {client}")

    n_runs = 20
    auroc_scores = []

    for i in range(n_runs):
        # Split the data with a different random seed
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        # Convert to Dask arrays for training
        X_train_da = da.from_array(X_train.values, chunks="auto")
        y_train_da = da.from_array(y_train.values, chunks="auto")

        # Convert test set to DMatrix for prediction (local)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Train using Dask
        dtrain = xgb.dask.DaskDMatrix(client, X_train_da, y_train_da, feature_names=feature_names)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 1,
            "reg_alpha": 0.5,
            "reg_lambda": 1,
            "random_state": 42 + i,
        }
        booster = xgb.dask.train(
            client,
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, "train")],
        )["booster"]

        # Predict and evaluate
        y_pred_prob = booster.predict(dtest)
        auroc = roc_auc_score(y_test, y_pred_prob)
        auroc_scores.append(auroc)
        logging.info(f"\tRun {i+1}: AUROC = {auroc:.4f}")

    # Shutdown Dask client
    client.close()

    # Plotting AUROC distribution
    plt.figure(figsize=(8, 6))
    plt.boxplot(auroc_scores, patch_artist=True, boxprops=dict(facecolor="#4195df"))
    plt.ylabel("AUROC", fontsize=16)
    plt.title(f"Stability Analysis of AUROC over {n_runs} runs", fontsize=18)
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_stability_boxplot_dask.png", dpi=300)
    plt.close()
    
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os

def plot_model_auroc_auprc(X_dd, y_dd, model, fig_dir):
    """
    Plot ROC and Precision-Recall curves for a trained model.
    
    Parameters
    ----------
    X_dd : dask.DataFrame
        Input features (Dask).
    y_dd : dask.Series
        Ground truth labels (Dask).
    model : xgboost.Booster or sklearn-style classifier
        Trained model with a `.predict_proba()` or `.predict()` method.
    fig_dir : str
        Directory to save the plots.
    """
    os.makedirs(fig_dir, exist_ok=True)

    # Convert Dask → pandas
    X = X_dd.compute()
    y = y_dd.compute()
    
    dtest = xgb.DMatrix(X)
    y_scores = model.predict(dtest)

    # Compute metrics
    fpr, tpr, _ = roc_curve(y, y_scores)
    precision, recall, _ = precision_recall_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)
    avg_prec = average_precision_score(y, y_scores)

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
    plt.savefig(f"{fig_dir}/xgboost_model_auroc_auprc.png", dpi=300)
    plt.close()

    
def plot_overlapping_roc_pr_curves(X_dd, y_dd, feature_names, fig_dir, n_runs=10, n_jobs=-1):
    """
    Plots overlapping ROC and Precision-Recall curves for multiple runs using Dask with parallel training.
    """
    logging.info("\tPlotting stability AUROC and AUPRC curves in parallel")
    
    os.makedirs(fig_dir, exist_ok=True)

    def run_single_split(i):
        logging.info(f"\t[Job {i+1}] Splitting and training")
        X_train_dd, X_test_dd, y_train_dd, y_test_dd = train_test_split(
            X_dd, y_dd, test_size=0.2, shuffle=True, random_state=i
        )

        booster = train_xgboost_dask(X_train_dd, y_train_dd, feature_names)
        clf = xgb_classifier_from_booster(booster, feature_names)

        X_test = X_test_dd.compute()
        y_test = y_test_dd.compute()
        y_pred_prob = clf.predict_proba(X_test)[:, 1]

        return y_test.to_numpy(), y_pred_prob

    # Run all model training and evaluation jobs in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_split)(i) for i in range(n_runs)
    )

    y_true_list, y_score_list = zip(*results)
    labels = [f"Run {i+1}" for i in range(n_runs)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- ROC Curves ---
    for i, (y_true, y_score) in enumerate(zip(y_true_list, y_score_list)):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, lw=1, alpha=0.8, label=f"{labels[i]} (AUC={roc_auc:.2f})")

    axes[0].plot([0, 1], [0, 1], color="#4195df", lw=1, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.0])
    axes[0].set_xlabel("False Positive Rate", fontsize=14)
    axes[0].set_ylabel("True Positive Rate", fontsize=14)
    axes[0].set_title("ROC Curve", fontsize=16)
    axes[0].legend(loc="lower right")

    # --- Precision-Recall Curves ---
    for i, (y_true, y_score) in enumerate(zip(y_true_list, y_score_list)):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_prec = average_precision_score(y_true, y_score)
        axes[1].plot(recall, precision, lw=1, alpha=0.8, color="4195df", label=f"{labels[i]} (PR={avg_prec:.2f})")

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall", fontsize=14)
    axes[1].set_ylabel("Precision", fontsize=14)
    axes[1].set_title("Precision-Recall Curve", fontsize=16)
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_stability_auroc_auprc.png", dpi=300)
    plt.close()

def plot_feature_ablation(feature_names, X_train, X_test, y_train, y_test, full_model, fig_dir, n_jobs=-1):
    logging.info("\tPlotting feature ablation for each feature")
    
    os.makedirs(fig_dir, exist_ok=True)

    # Evaluate full model AUROC
    y_pred_prob_full = full_model.predict_proba(X_test)[:, 1]
    full_auroc = roc_auc_score(y_test, y_pred_prob_full)
    logging.info(f"\t\tFull model AUROC: {full_auroc:.4f}")

    def evaluate_feature_removal(feature):
        features_subset = [f for f in feature_names if f != feature]
        X_train_subset = X_train[features_subset]
        X_test_subset = X_test[features_subset]

        model = xgb.XGBClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=6,
            gamma=1,
            reg_alpha=0.5,
            reg_lambda=1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False
        )
        model.fit(X_train_subset, y_train)
        y_pred = model.predict_proba(X_test_subset)[:, 1]
        auroc = roc_auc_score(y_test, y_pred)
        return feature, auroc

    # Run in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_feature_removal)(feature)
        for feature in feature_names
    )

    # Sort and plot results
    feature_set, auroc_score_set = zip(*results)
    features: list = list(feature_set)
    auroc_scores: list = list(auroc_score_set)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, auroc_scores, color='#4195df')
    plt.axhline(full_auroc, color='black', linestyle='--', label=f'Full model AUROC = {full_auroc:.2f}')
    plt.xlabel("Removed Feature", fontsize=14)
    plt.ylabel("AUROC", fontsize=14)
    plt.title("Feature Ablation Analysis", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_feature_ablation.png", dpi=300)
    plt.close()

    # Log results
    for f, a in zip(features, auroc_scores):
        logging.info(f"\t\tAUROC without '{f}': {a:.4f}")
        
def plot_enhancer_to_target_arcs(df, gene, tss_reference_file, gene_body_anno_file, figsize=(10, 2)):
    gene_df = df[df["target_id"] == gene].copy()
    if gene_df.empty:
        print(f"No enhancers found for {gene}")
        return
    
    gene_df = gene_df.drop_duplicates(subset="peak_id")
    gene_df["peak_center"] = (gene_df["Start"] + gene_df["End"]) // 2
    
    def find_target_gene_tss(gene) -> Tuple[float, float]:
        """Standardizes the target gene TSS using a reference dataset"""
        if not os.path.isfile(tss_reference_file):
            raise FileNotFoundError(f"Path {tss_reference_file} is not a file")
        
        tss_reference = pd.read_csv(
            tss_reference_file, 
            sep="\t", 
            header=None, 
            index_col=None,
            names=["Chr", "Start", "End", "Name", "Strand", "Strand2"]
        )
        tss_reference["tss"] = (tss_reference["Start"] + tss_reference["End"]) // 2
        
        tss_entry = tss_reference[tss_reference["Name"].str.upper() == gene.upper()]
        if tss_entry.empty:
            raise ValueError(f"Gene '{gene}' not found in TSS reference.")
        tss_start = float(tss_entry["Start"].iloc[0])
        tss_end = float(tss_entry["End"].iloc[0])
        return tss_start, tss_end
    
    def find_target_gene_body(gene) -> Tuple[float, float]:
        """Standardizes the target gene length using a reference dataset"""
        if not os.path.isfile(gene_body_anno_file):
            raise FileNotFoundError(f"Path {gene_body_anno_file} is not a file")
        gene_body_anno = pd.read_csv(
            gene_body_anno_file, 
            sep="\t", 
            header=None, 
            index_col=None,
            names=["Chr", "Start", "End", "Name", "Strand", "Strand2"]
        )
        
        gene_entry = gene_body_anno[gene_body_anno["Name"].str.upper() == gene.upper()]
        if gene_entry.empty:
            raise ValueError(f"Gene '{gene}' not found in gene body reference")
        gene_start = float(gene_entry["Start"].iloc[0])
        gene_end = float(gene_entry["End"].iloc[0])
        return gene_start, gene_end
    
    gene_start, gene_end = find_target_gene_body(gene)
    tss_start, tss_end = find_target_gene_tss(gene)

    # Set a distance cutoff so the figure doesn't get too squished
    max_dist = 250000
    gene_df["distance_to_gene"] = abs(gene_df["peak_center"] - ((gene_start + gene_end) / 2))
    gene_df = gene_df[gene_df["distance_to_gene"] <= max_dist]
    
    if gene_df.empty:
        print(f"No enhancers within ±{max_dist:,} bp of {gene}")
        return

    fig, ax = plt.subplots(figsize=figsize)
    
    # Blue Rectangle the length of the gene body
    start_height = 25
    gene_height = 10
    
    ax.add_patch(
        patches.Rectangle((gene_start, start_height), gene_end - gene_start, gene_height,
                          facecolor='skyblue', edgecolor='black', label=gene)
    )
    
    if abs(tss_start - gene_start) < abs(tss_start - gene_end):
        tss_rect_start = min(tss_start, gene_start)
        tss_rect_width = abs(tss_start - gene_start)
    else:
        tss_rect_start = min(tss_start, gene_end)
        tss_rect_width = abs(tss_start - gene_end)

    ax.add_patch(
        patches.Rectangle((tss_rect_start, start_height), tss_rect_width, gene_height,
                        facecolor='green', edgecolor='black')
    )

    xmin = int(np.ceil(tss_start - 25000))
    xmax = int(np.ceil(gene_end + 25000))
    xrange = xmax - xmin
    h = xrange * 0.02     # gene height
    arc_height = max(50, xrange * 0.001)
    
    # Add a line along the bottom to show the genomic position
    plt.plot([xmin, xmax], [start_height, start_height], 'k-', lw=2)
    
    # Add 10 evenly spaced marks along the line to show the genomic position
    mark_points = [int(np.ceil(np.mean(i))) for i in np.array_split(range(xmin, xmax), 10)]
    for xloc in mark_points:
        # Plots a vertical dash for each of the 10 points
        plt.plot([xloc, xloc], [15, start_height], 'k-', lw=2)
        # Plots the genomic position for each of the 10 points
        plt.text(xloc, 0, s=f'{xloc}', ha='center', va='bottom', fontsize=10)
    
    for _, row in gene_df.iterrows():
        
        # Plot a grey rectangle for the enhancer locations
        ax.add_patch(
        patches.Rectangle((row["Start"], start_height), row["End"] - row['Start'], gene_height / 2,
                          facecolor='grey', edgecolor='black')
        )
        
        enh_center = row["peak_center"]
        tss = tss_start
        center = (enh_center + tss) / 2
        
        radius = max(abs(enh_center - tss) / 2, xrange * 0.001)  # avoid tiny lines

        arc = patches.Arc((center, start_height + 10), radius * 2, arc_height,
                          angle=0, theta1=0, theta2=180)
        ax.add_patch(arc)

    ax.set_xlim(tss_start - 25000, gene_end + 25000)
    ax.set_ylim(0, 75)
    ax.set_title(f"Enhancers targeting {gene}")
    ax.axis("off")
    
    plt.tight_layout()
    plt.show()
