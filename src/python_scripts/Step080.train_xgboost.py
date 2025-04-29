import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask.distributed import Client
import math
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import csv
import os
import joblib
import logging
import argparse
import xgboost as xgb
import dask.array as da
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing paths for input and output files.
    """
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")

    parser.add_argument(
        "--ground_truth_file",
        type=str,
        required=True,
        help="Path to the ChIPseq ground truth file, formatted as 'source_id'\\t'target_id'"
    )
    parser.add_argument(
        "--inferred_network_file",
        type=str,
        required=True,
        help="Path to the inferred network file for the sample"
    )
    parser.add_argument(
        "--trained_model_dir",
        type=str,
        required=True,
        help="Path to the output directory for the trained XGBoost model"
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        required=True,
        help="Path to the figure directory for the sample"
    )
    parser.add_argument(
        "--model_save_name",
        type=str,
        required=True,
        help="Name of the output .pkl file for the trained model"
    )
    
    args: argparse.Namespace = parser.parse_args()
    return args

def read_inferred_network(inferred_network_file):
    logging.info("Reading inferred network with Dask")
    inferred_network = dd.read_parquet(inferred_network_file)
    inferred_network["source_id"] = inferred_network["source_id"].str.upper()
    inferred_network["target_id"] = inferred_network["target_id"].str.upper()
    return inferred_network

def read_ground_truth(ground_truth_file):
    logging.info("Reading in the ground truth")
    ground_truth = pd.read_csv(ground_truth_file, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    ground_truth = ground_truth.rename(columns={"Source": "source_id", "Target": "target_id"})
    return ground_truth

def read_merged_ground_truth(merged_ground_truth_file):
    merged_ground_truth = pd.read_csv(merged_ground_truth_file, sep='\t', header=0)
    logging.info(merged_ground_truth)
    return merged_ground_truth

def train_xgboost_dask(X_train_dd, y_train_dd, feature_names):
    """
    Train an XGBoost model using DaskDMatrix (distributed).
    
    Args:
        X_train_dd (dask.dataframe.DataFrame): Training features (Dask)
        y_train_dd (dask.dataframe.Series): Training labels (Dask)
        feature_names (list): List of feature column names
        
    Returns:
        booster (xgboost.Booster): Trained XGBoost Booster object
    """
    logging.info("Training XGBoost model with Dask")

    # Create a DaskDMatrix
    dtrain = xgb.dask.DaskDMatrix(
        client=None,  # If using local CPU, otherwise pass Dask client
        data=X_train_dd,
        label=y_train_dd,
        feature_names=feature_names
    )

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',    # highly recommended for large data
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 1,
        'reg_alpha': 0.5,
        'reg_lambda': 1,
        'random_state': 42
    }

    output = xgb.dask.train(
        client=None,       # None uses threads; you could pass a Dask client
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train')],
    )

    booster = output['booster']  # This is the trained model
    booster.set_attr(feature_names=",".join(feature_names))  # Save feature names

    return booster

def parameter_grid_search(X_train_dd, y_train_dd, features, cpu_count, fig_dir):
    logging.info("⚙️ Starting XGBoost hyperparameter grid search")

    # Convert Dask → pandas (required for GridSearchCV)
    X_train = X_train_dd.compute()
    y_train = y_train_dd.compute()

    # Combine into single DataFrame for balancing
    train_data = X_train.copy()
    train_data["label"] = y_train

    # Balance classes
    pos_train = train_data[train_data["label"] == 1]
    neg_train = train_data[train_data["label"] == 0]
    neg_train_sampled = neg_train.sample(n=len(pos_train), random_state=42)
    train_data_balanced = pd.concat([pos_train, neg_train_sampled])

    X_bal = train_data_balanced[features]
    y_bal = train_data_balanced["label"]

    # Define parameter grid
    param_grid = {
        "n_estimators":      [50, 100, 200],
        "max_depth":         [4, 6, 8],
        "gamma":             [0, 1, 5],
        "reg_alpha":         [0.0, 0.5, 1.0],
        "reg_lambda":        [1.0, 2.0, 5.0],
        "subsample":         [0.8, 1.0],
        "colsample_bytree":  [0.8, 1.0],
    }

    # Initialize classifier (use hist method for speed)
    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        tree_method="hist",
        use_label_encoder=False
    )

    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform grid search
    grid = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=cpu_count,
        verbose=2
    )

    logging.info("Running GridSearchCV on balanced training set")
    grid.fit(X_bal, y_bal)

    logging.info(f"  Best CV score:  {grid.best_score_:.4f}")
    logging.info(f"  Best parameters: {grid.best_params_}")

    best_model = grid.best_estimator_

    # Save plot
    output_dir = os.path.join(fig_dir, "parameter_search")
    os.makedirs(output_dir, exist_ok=True)

    plot_feature_importance(
        features=features,
        model=best_model,
        fig_dir=output_dir
    )

def xgb_classifier_from_booster(booster: xgb.Booster, feature_names: list | np.ndarray | pd.Index) -> xgb.XGBClassifier:
    """
    Converts a trained XGBoost Booster (e.g., from Dask) to a scikit-learn XGBClassifier.
    This allows use of sklearn-compatible APIs like predict_proba, permutation_importance, etc.

    Parameters:
    -----------
    booster : xgb.Booster
        Trained Booster object (e.g. from xgb.dask.train)

    feature_names : list or array
        List of feature names used during training

    Returns:
    --------
    xgb.XGBClassifier
        Fully compatible sklearn-style classifier loaded from booster
    """
    clf = xgb.XGBClassifier()
    clf._Booster = booster
    clf.n_features_in_ = len(feature_names)
    clf.feature_names_in_ = np.array(feature_names)
    clf.classes_ = np.array([0, 1])
    return clf

def plot_xgboost_prediction_histogram(booster, X_test, fig_dir):
    logging.info("\tPlotting model prediction histogram")

    # Convert to DMatrix (required for Booster prediction)
    dtest = xgb.DMatrix(X_test)

    # Get predicted probabilities for the positive class
    y_pred_prob = booster.predict(dtest)

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

    # Extract raw importance scores from Booster
    importance_dict = model.get_score(importance_type="weight")

    # Build DataFrame ensuring all input features are represented (default to 0 if missing)
    feature_importances = pd.DataFrame({
        "Feature": features,
        "Importance": [importance_dict.get(f, 0) for f in features]
    }).sort_values(by="Importance", ascending=False)

    if feature_importances["Importance"].sum() == 0:
        logging.warning("All feature importances are zero; model may not have learned from any features.")

    # Plot
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
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_feature_score_hist.png", dpi=300)
    plt.close()

def plot_feature_boxplots(features, inferred_network, fig_dir):
    logging.info("\tPlotting feature importance boxplots")

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
    
def plot_overlapping_roc_pr_curves(X, y, feature_names, fig_dir):
    """
    Plots overlapping ROC and Precision-Recall curves for multiple runs.
    
    """
    logging.info("\tPlotting stability AUROC and AUPRC curves")
    # --- Generate y_true_list and y_score_list over multiple runs ---
    n_runs = 10  # Number of runs for stability analysis; adjust as needed

    y_true_list = []
    y_score_list = []

    for i in range(n_runs):
        # Split the data with a different random seed each run
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.7, random_state=i)
        
        # Train your XGBoost model; ensure your train_xgboost function is defined
        xgb_model = train_xgboost(X_train, y_train, feature_names)
        # Save feature names if needed for prediction
        xgb_model.feature_names = list(X_train.columns.values)
        
        # Predict probabilities on the test set (for the positive class)
        y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
        
        # Store the true labels and predicted probabilities
        y_true_list.append(y_test.to_numpy())
        y_score_list.append(y_pred_prob)
    
    labels = [f"Run {i+1}" for i in range(len(y_true_list))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- ROC Curves ---
    for i, (y_true, y_score) in enumerate(zip(y_true_list, y_score_list)):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, lw=1, alpha=0.8, label=f"{labels[i]} (AUC={roc_auc:.2f})")

    # Diagonal line for random guessing
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.0])
    axes[0].set_xlabel("False Positive Rate", fontsize=14)
    axes[0].set_ylabel("True Positive Rate", fontsize=14)
    axes[0].set_title(f"ROC Curve", fontsize=16)
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
    axes[1].set_title(f"Precision-Recall Curve", fontsize=16)
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_stability_auroc_auprc.png", dpi=300)
    plt.close()

def plot_feature_ablation(feature_names, X_train, X_test, y_train, y_test, full_model, fig_dir, n_jobs=-1):
    logging.info("\tPlotting feature ablation for each feature")

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

def main():
    args = parse_args()

    ground_truth_file: str = args.ground_truth_file
    inferred_network_file: str = args.inferred_network_file
    trained_model_dir: str = args.trained_model_dir
    fig_dir: str = args.fig_dir
    model_save_name: str = args.model_save_name

    inferred_network_dd = read_inferred_network(inferred_network_file)
    ground_truth_df = read_ground_truth(ground_truth_file)

    # Create set of (TF, TG) from ground truth
    logging.info("Creating ground truth set")
    ground_truth_pairs = set(zip(ground_truth_df["source_id"], ground_truth_df["target_id"]))

    logging.info("Adding labels to inferred network")
    inferred_network_dd = inferred_network_dd.map_partitions(
        lambda df: df.assign(label=df.apply(
            lambda row: 1 if (row["source_id"], row["target_id"]) in ground_truth_pairs else 0,
            axis=1
        )),
        meta=inferred_network_dd._meta.assign(label=np.int64(0))
    )

    # Drop unnecessary columns
    drop_cols = ["source_id", "peak_id", "target_id", "label"]
    feature_names = [col for col in inferred_network_dd.columns if col not in drop_cols]

    # Only keep columns needed for modeling
    logging.info(f"Keeping {len(feature_names)} feature columns + labels")
    model_dd = inferred_network_dd[feature_names + ["label"]]

    logging.info(f"Splitting {model_dd.shape[0].compute():,} rows into train/test with stratification")

    # Dask-ML's split works directly on Dask DataFrames
    X_dd = model_dd[feature_names]
    y_dd = model_dd["label"]

    X_train_dd, X_test_dd, y_train_dd, y_test_dd = train_test_split(
        X_dd,
        y_dd,
        test_size=0.2,
        shuffle=True,
        stratify=y_dd,   # Pure Dask stratification!
        random_state=42
    )

    logging.info(f"Done splitting: {X_train_dd.shape[0].compute():,} train / {X_test_dd.shape[0].compute():,} test rows")
    logging.info("Training XGBoost Model")
    xgb_booster = train_xgboost_dask(X_train_dd, y_train_dd, feature_names)

    # Save the feature names
    xgb_booster.set_attr(feature_names=",".join(feature_names))
    
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)

    model_save_path = os.path.join(trained_model_dir, f"{model_save_name}.json")
    xgb_booster.save_model(model_save_path)
    logging.info(f"Saved trained XGBoost booster to {model_save_path}")

    importance_dict = xgb_booster.get_score(importance_type="weight")
    feature_importances = pd.DataFrame({
        "Feature": list(importance_dict.keys()),
        "Importance": list(importance_dict.values())
    })
    feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

    logging.info("\n----- Plotting Figures -----")
    model_df = model_dd.compute()
    X = model_df[feature_names]
    y = model_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_cls = xgb_classifier_from_booster(xgb_booster, X.columns)

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    logging.info("\n----- Plotting Figures -----")
    plot_feature_score_histograms(feature_names, model_df, fig_dir)
    plot_feature_importance(feature_names, xgb_booster, fig_dir)
    plot_feature_boxplots(feature_names, model_df, fig_dir)
    plot_xgboost_prediction_histogram(xgb_booster, X_test, fig_dir)
    
    # --- Note: The following plots take a long time to run for large models, as they test re-training the model ---

    plot_overlapping_roc_pr_curves(X, y, feature_names, fig_dir)
    plot_permutation_importance_plot(model_cls, X_test, y_test, fig_dir)
    plot_feature_ablation(feature_names, X_train, X_test, y_train, y_test, model_cls, fig_dir)
    plot_stability_boxplot(X, y, fig_dir)
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
