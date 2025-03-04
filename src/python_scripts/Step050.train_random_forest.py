import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import random
import csv
import os
import joblib
import logging
import argparse

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
        help="Path to the ChIPseq ground truth file, formatted as 'Source'\\t'Target'"
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
    inferred_network = pd.read_pickle(inferred_network_file)
    logging.info("Read in inferred network")
    inferred_network["Source"] = inferred_network["Source"].str.upper()
    inferred_network["Target"] = inferred_network["Target"].str.upper()
    
    return inferred_network

def read_ground_truth(ground_truth_file):
    ground_truth = pd.read_csv(ground_truth_file, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    logging.info("Read in ground truth")
    
    return ground_truth

def read_merged_ground_truth(merged_ground_truth_file):
    merged_ground_truth = pd.read_csv(merged_ground_truth_file, sep='\t', header=0)
    logging.info(merged_ground_truth)
    
    return merged_ground_truth

def compute_aggregated_cell_level_features(df: pd.DataFrame, desired_n=5000):
    """
    For each row in df, resample (with replacement) the cell-level values in cell_cols 
    until we have desired_n values. Then compute aggregated statistics (mean, std, min, max, median).
    
    Returns a new DataFrame with these aggregated features.
    """    
    
    # Randomly sample desired_n columns with replacement
    df_copy = df.copy()
    data_cols = df_copy.drop(["Source", "Target", "Label"], axis=1)
    sampled_columns = data_cols.sample(n=desired_n, axis='columns', replace=True)

    # Compute aggregated statistics
    sampled_columns["mean_score"]   = sampled_columns.mean(axis=1)
    sampled_columns["std_score"]    = sampled_columns.std(axis=1)
    sampled_columns["min_score"]    = sampled_columns.min(axis=1)
    sampled_columns["max_score"]   = sampled_columns.max(axis=1)
    sampled_columns["median_score"] = sampled_columns.median(axis=1)
    
    df_with_agg_feature_cols = pd.concat([df, sampled_columns], axis=1)

    return df_with_agg_feature_cols

def train_random_forest(X_train, y_train, features):
    logging.info("Training Random Forest Model")
    # Combine training features and labels for resampling
    train_data = X_train.copy()
    train_data["Label"] = y_train

    # Separate positive and negative examples
    pos_train = train_data[train_data["Label"] == 1]
    neg_train = train_data[train_data["Label"] == 0]

    # Balance the dataset between positive and negative label values
    neg_train_sampled = neg_train.sample(n=len(pos_train), random_state=42)
    train_data_balanced = pd.concat([pos_train, neg_train_sampled])

    X_train_balanced = train_data_balanced[features]
    y_train_balanced = train_data_balanced["Label"]

    logging.info(f"Balanced training set: {len(pos_train)} positives and {len(neg_train_sampled)} negatives.")

    # Train the Random Forest Classifier
    rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    rf.fit(X_train_balanced, y_train_balanced)

    return rf

def plot_random_forest_prediction_histogram(rf: RandomForestClassifier, X_test, fig_dir):
    y_pred_prob = rf.predict_proba(X_test)[:, 1]  # Probability for the positive class

    plt.figure(figsize=(8, 6))
    plt.hist(y_pred_prob, bins=50)
    plt.title("Histogram of Random Forest Prediction Probabilities")
    plt.xlabel("Prediction Probability")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/random_forest_prediction_probability_histogram.png", dpi=200)
    plt.close()

def plot_feature_importance(features: list, rf: RandomForestClassifier, fig_dir: str):
    # Step 4: Feature Importance Analysis
    feature_importances = pd.DataFrame({
        "Feature": features,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(8, 6))
    plt.barh(feature_importances["Feature"], feature_importances["Importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/random_forest_feature_importance.png", dpi=200)
    plt.close()
    
def plot_feature_score_histograms(features, inferred_network, fig_dir):
    # Create a figure and axes with a suitable size
    plt.figure(figsize=(15, 10))

    # Loop through each feature and create a subplot
    for i, feature in enumerate(features, 1):
        plt.subplot(3, 4, i)  # 3 rows, 4 columns, index = i
        plt.hist(inferred_network[feature], bins=50, alpha=0.7, edgecolor='black')
        plt.title(f"{feature} distribution")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        # plt.xlim((0,1))

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/rf_feature_score_hist.png', dpi=300)
    plt.close()

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()

    ground_truth_file: str = args.ground_truth_file
    output_dir: str = args.output_dir
    fig_dir: str = args.fig_dir
    
    # Alternatively: Set the input file paths directly
    # ground_truth_file = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/RN111.tsv"
    # output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC"
    
    inferred_network_file = f"{output_dir}/inferred_network_raw.pkl"

    inferred_network = read_inferred_network(inferred_network_file)
    ground_truth = read_ground_truth(ground_truth_file)

    # Create a set of tuples from ground_truth for faster lookup
    logging.info("Creating set of TF-TG pairs for ground truth")
    ground_truth_pairs = set(zip(ground_truth["Source"], ground_truth["Target"]))

    logging.info("Adding labels to inferred network, 1 if predicted edge is in ground truth else 0")
    # Add the "Label" column to inferred_network
    inferred_network["Label"] = inferred_network.apply(
        lambda row: 1 if (row["Source"], row["Target"]) in ground_truth_pairs else 0,
        axis=1
    )

    # logging.info the resulting DataFrame
    logging.info(inferred_network.head())
    logging.info(f'Number of True predictions: {len(inferred_network[inferred_network["Label"] == 1])}')
    logging.info(f'Number of False predictions: {len(inferred_network[inferred_network["Label"] == 0])}')

    # Define the list of aggregated features for training
    aggregated_features_new = [
        "TF_mean_expression",
        "TF_std_expression",
        "TF_min_expression",
        "TF_median_expression",
        "tf_to_tg_score",
        "TG_mean_expression",
        "TG_std_expression",
        "TG_min_expression",
        "TG_median_expression",
        "pearson_correlation"
    ]

    # Define X (features) and y (target)
    X = inferred_network[aggregated_features_new]
    y = inferred_network["Label"]

    # Split into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = train_random_forest(X_train, y_train, aggregated_features_new)
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    plot_feature_importance(aggregated_features_new, rf, fig_dir)
    
    plot_feature_score_histograms(aggregated_features_new, inferred_network, fig_dir)
    
    # Save the feature names of the trained model
    rf.feature_names = list(X_train.columns.values)

    # Save the trained model as a pickle file
    logging.info("Saving trained model")
    joblib.dump(rf, f"{output_dir}/trained_random_forest_model.pkl")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()
