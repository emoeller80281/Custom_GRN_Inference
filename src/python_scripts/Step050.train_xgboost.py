import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import random
import csv
import os
import joblib
import logging
import argparse
import xgboost as xgb  # Import XGBoost

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
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        required=True,
        help="Path to the figure directory for the sample"
    )
    
    args: argparse.Namespace = parser.parse_args()
    return args

def read_inferred_network(inferred_network_file):
    logging.info("Reading in inferred network")
    inferred_network = pd.read_csv(inferred_network_file, header=0)
    
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

def train_xgboost(X_train, y_train, features):
    logging.info("Training XGBoost Model")
    # Combine training features and labels for resampling
    train_data = X_train.copy()
    train_data["label"] = y_train

    # Separate positive and negative examples
    pos_train = train_data[train_data["label"] == 1]
    neg_train = train_data[train_data["label"] == 0]

    # Balance the dataset between positive and negative label values
    neg_train_sampled = neg_train.sample(n=len(pos_train), random_state=42)
    train_data_balanced = pd.concat([pos_train, neg_train_sampled])

    X_train_balanced = train_data_balanced[features]
    y_train_balanced = train_data_balanced["label"]

    logging.info(f"Balanced training set: {len(pos_train)} positives and {len(neg_train_sampled)} negatives.")

    # Train the XGBoost Classifier
    # XGBoost automatically handles NaN values (missing=np.nan is the default)
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=10,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_balanced, y_train_balanced)

    return xgb_model

def plot_xgboost_prediction_histogram(model, X_test, fig_dir):
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability for the positive class

    plt.figure(figsize=(8, 6))
    plt.hist(y_pred_prob, bins=50)
    plt.title("Histogram of XGBoost Prediction Probabilities", fontsize=18)
    plt.xlabel("Prediction Probability", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_prediction_probability_histogram.png", dpi=200)
    plt.close()

def plot_feature_importance(features: list, model, fig_dir: str):
    # Feature Importance Analysis
    feature_importances = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 6))
    plt.barh(feature_importances["Feature"], feature_importances["Importance"], color="skyblue")
    plt.xlabel("Importance", fontsize=16)
    plt.ylabel("Feature", fontsize=16)
    plt.title("Feature Importance", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_feature_importance.png", dpi=200)
    plt.close()
    
def plot_feature_score_histograms(features, inferred_network, fig_dir):
    plt.figure(figsize=(15, 8))
    for i, feature in enumerate(features, 1):
        plt.subplot(3, 4, i)  # 3 rows, 4 columns, index = i
        plt.hist(inferred_network[feature], bins=50, alpha=0.7, edgecolor='black')
        plt.title(f"{feature} distribution", fontsize=18)
        plt.xlabel(feature, fontsize=16)
        plt.ylabel("Frequency", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/xgboost_feature_score_hist.png', dpi=300)
    plt.close()
    
def plot_feature_boxplots(features, inferred_network, fig_dir):
    def remove_outliers(series):
        """
        Remove outliers from a pandas Series using the IQR method.
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series >= lower_bound) & (series <= upper_bound)]
    
    n_features = len(features)
    ncols = 3
    nrows = math.ceil(n_features / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()  # Flatten the 2D array of axes
    
    for i, feature in enumerate(features):
        ax = axes[i]
        data_label0 = remove_outliers(inferred_network.loc[inferred_network["label"] == 0, feature])
        data_label1 = remove_outliers(inferred_network.loc[inferred_network["label"] == 1, feature])
        ax.boxplot([data_label0, data_label1], patch_artist=True)
        ax.set_title(feature, fontsize=18)
        ax.set_xticklabels(["True", "False"], fontsize=16)
        ax.set_ylabel("score", fontsize=16)
        ax.set_yticklabels(fontsize=14)
    
    # Hide any unused subplots if they exist
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/xgboost_feature_boxplots.png', dpi=300)
    plt.close()

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()

    ground_truth_file: str = args.ground_truth_file
    inferred_network_file: str = args.inferred_network_file
    output_dir: str = args.output_dir
    fig_dir: str = args.fig_dir

    inferred_network = read_inferred_network(inferred_network_file)
    ground_truth = read_ground_truth(ground_truth_file)

    logging.info("Creating set of TF-TG pairs for ground truth")
    ground_truth_pairs = set(zip(ground_truth["source_id"], ground_truth["target_id"]))

    logging.info("Adding labels to inferred network: 1 if predicted edge is in ground truth, else 0")
    inferred_network["label"] = inferred_network.apply(
        lambda row: 1 if (row["source_id"], row["target_id"]) in ground_truth_pairs else 0,
        axis=1
    )

    logging.info(inferred_network.head())
    logging.info(f'Number of True predictions: {len(inferred_network[inferred_network["label"] == 1])}')
    logging.info(f'Number of False predictions: {len(inferred_network[inferred_network["label"] == 0])}')

    aggregated_features_new = [
        "mean_TF_expression",
        "mean_TG_expression",
        "mean_peak_accessibility",
        "cicero_score",
        "enh_score",
        "TSS_dist",
        "correlation",
        "sliding_window_score",
        "homer_binding_score"
    ]

    X = inferred_network[aggregated_features_new]
    y = inferred_network["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_model = train_xgboost(X_train, y_train, aggregated_features_new)
    
    # Save feature names for reference
    xgb_model.feature_names = list(X_train.columns.values)
    
    logging.info("Done! Saving trained XGBoost model.")
    joblib.dump(xgb_model, f"{output_dir}/trained_xgboost_model.pkl")

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    plot_feature_importance(aggregated_features_new, xgb_model, fig_dir)
    plot_feature_score_histograms(aggregated_features_new, inferred_network, fig_dir)
    plot_feature_boxplots(aggregated_features_new, inferred_network, fig_dir)
    plot_xgboost_prediction_histogram(xgb_model, X_test, fig_dir)
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
