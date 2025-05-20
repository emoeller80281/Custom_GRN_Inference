import os
import math
import logging

import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da

import matplotlib.pyplot as plt

from dask_ml.model_selection import train_test_split
from dask.distributed import Client

from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed

import xgboost as xgb

from model import (
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
  
def plot_feature_score_histograms(features, inferred_network, fig_dir):
    logging.info("\tPlotting feature score histograms")
    
    os.makedirs(fig_dir, exist_ok=True)

    # Step 1: Convert only necessary columns to pandas
    if isinstance(inferred_network, dd.DataFrame):
        logging.info("\tConverting feature columns from Dask to pandas for plotting")
        inferred_network = inferred_network[features].compute()

    ncols = 4
    nrows = math.ceil(len(features) / ncols)

    plt.figure(figsize=(5 * ncols, 4 * nrows))

    for i, feature in enumerate(features, 1):
        plt.subplot(nrows, ncols, i)
        plt.hist(inferred_network[feature].dropna(), bins=50, alpha=0.7, edgecolor='black')
        plt.title(f"{feature}", fontsize=14)
        plt.xlabel(feature, fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xlim((0, 1))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_feature_score_hist.png", dpi=300)
    plt.close()

def plot_feature_boxplots(features, inferred_network, fig_dir):
    logging.info("\tPlotting feature importance boxplots")
    
    os.makedirs(fig_dir, exist_ok=True)

    def remove_outliers(series: pd.Series) -> pd.Series:
        """
        Remove outliers using the IQR method.
        NaN values are preserved.
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return series[(series >= lower) & (series <= upper)]

    n_features = len(features)
    ncols = 3
    nrows = math.ceil(n_features / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]

        try:
            data_0 = remove_outliers(inferred_network[inferred_network["label"] == 0][feature].dropna())
            data_1 = remove_outliers(inferred_network[inferred_network["label"] == 1][feature].dropna())
        except KeyError as e:
            logging.warning(f"Feature '{feature}' not found in inferred network. Skipping.")
            continue

        ax.boxplot([data_0, data_1], patch_artist=True,
                   boxprops=dict(facecolor='lightgray', color='black'),
                   medianprops=dict(color='red'),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='black'),
                   flierprops=dict(markerfacecolor='red', markersize=4, linestyle='none'))

        ax.set_title(feature, fontsize=16)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["False", "True"], fontsize=14)
        ax.set_ylabel("Score", fontsize=14)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/xgboost_feature_boxplots.png', dpi=300)
    plt.close()

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
        color='skyblue',
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
    plt.boxplot(auroc_scores, patch_artist=True, boxprops=dict(facecolor="blue"))
    plt.ylabel("AUROC", fontsize=16)
    plt.title(f"Stability Analysis of AUROC over {n_runs} runs", fontsize=18)
    plt.ylim((0, 1))
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_stability_boxplot_dask.png", dpi=300)
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
            X_dd, y_dd, test_size=0.2, shuffle=True, stratify=y_dd, random_state=i
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

    axes[0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
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
        axes[1].plot(recall, precision, lw=1, alpha=0.8, label=f"{labels[i]} (PR={avg_prec:.2f})")

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
    features, auroc_scores = zip(*results)
    features = list(features)
    auroc_scores = list(auroc_scores)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, auroc_scores, color='steelblue')
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
