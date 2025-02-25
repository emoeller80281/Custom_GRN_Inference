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
    inferred_network = pd.read_csv(inferred_network_file, sep="\t")
    logging.info(inferred_network.head())
    inferred_network["Source"] = inferred_network["Source"].str.upper()
    inferred_network["Target"] = inferred_network["Target"].str.upper()
    
    return inferred_network

def read_ground_truth(ground_truth_file):
    ground_truth = pd.read_csv(ground_truth_file, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    logging.info(ground_truth.head())
    
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

# def cell_level_grn_test_summary_stat_features(rf: RandomForestClassifier):
#     new_data = pd.read_csv("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/cell_level_inferred_grn_testing.csv", sep="\t")

#     # Identify the cell columns; for example, assume everything after "Target" is cell data:
#     cell_columns_new = new_data.columns[2:]  # if there's no "Label" yet

#     # Compute the aggregated features
#     new_data["mean_score"]   = new_data[cell_columns_new].mean(axis=1)
#     new_data["std_score"]    = new_data[cell_columns_new].std(axis=1)
#     new_data["min_score"]    = new_data[cell_columns_new].min(axis=1)
#     new_data["max_score"]    = new_data[cell_columns_new].max(axis=1)
#     new_data["median_score"] = new_data[cell_columns_new].median(axis=1)

#     aggregated_features = ["mean_score", "std_score", "min_score", "max_score", "median_score"]

#     # Extract the aggregated feature vector
#     X_new = new_data[aggregated_features]

#     # Make predictions with your trained model
#     new_data["Score"] = rf.predict_proba(X_new)[:, 1]

#     new_data = new_data[["Source", "Target", "Score"]]

#     new_data.to_csv(f'{output_dir}/summary_stat_rf_inferred_grn.tsv', sep='\t', index=False)

# def calculate_randomly_permuted_subsamples(rf: RandomForestClassifier, subsamples: int, num_cols: int, inferred_network: pd.DataFrame, features: list):
#     """Calculates n subsamples of num_cols cell-level GRNs for stability analysis"""
    
#     subsamples = 10
#     num_cols = 500
#     for i in range(subsamples):
#         # Set the features as the randomly permuted 100 columns to match the size of the training data
#         features = [
#             "tf_to_peak_binding_score",
#             "TF_mean_expression",
#             "TF_std_expression",
#             "TF_min_expression",
#             "TF_median_expression",
#             "peak_to_target_score",
#             "TG_mean_expression",
#             "TG_std_expression",
#             "TG_min_expression",
#             "TG_median_expression",
#             "pearson_correlation"
#         ]
        
#         # Randomly reindex all columns containing cell data
#         inferred_network_permuted = inferred_network.reindex(np.random.permutation(inferred_network[features]), axis='columns')
#         logging.info(f'Permuted inferred network shape: {inferred_network_permuted.shape}')
        
#         # Take a 90% subsample of the cell data

#         inferred_network_subsample = inferred_network_permuted.iloc[:, 0:num_cols+1]
#         logging.info(f'Randomized inferred network shape: {inferred_network_subsample.shape}')
            

        
#         logging.info(f'Num features: {len(features)}')
#         X = inferred_network_subsample[features]
#         inferred_network_subsample["Score"] = rf.predict_proba(X)[:, 1]

#         inferred_network_subsample = inferred_network[["Source", "Target", "Score"]]
#         logging.info(inferred_network_subsample.head())
        
#         sample_dir_path = f'{output_dir}/rf_stability_analysis/inferred_network_subsample_{i+1}'
#         if not os.path.exists(sample_dir_path):
#             os.makedirs(sample_dir_path)

#         inferred_network_subsample.to_csv(f'{sample_dir_path}/rf_inferred_grn.tsv', sep='\t', index=False)

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()

    ground_truth_file: str = args.ground_truth_file
    output_dir: str = args.output_dir
    fig_dir: str = args.fig_dir
    
    # Alternatively: Set the input file paths directly
    # ground_truth_file = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/RN111.tsv"
    # output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC"
    
    inferred_network_file = f"{output_dir}/inferred_network_raw.tsv"

    inferred_network = read_inferred_network(inferred_network_file)
    ground_truth = read_ground_truth(ground_truth_file)

    # Create a set of tuples from ground_truth for faster lookup
    ground_truth_pairs = set(zip(ground_truth["Source"], ground_truth["Target"]))

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

    # Save the feature names of the trained model
    rf.feature_names = list(X_train.columns.values)

    # Save the trained model as a pickle file
    joblib.dump(rf, f"{output_dir}/trained_random_forest_model.pkl")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()
