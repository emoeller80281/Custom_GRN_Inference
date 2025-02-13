import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import random
import csv
import os

inferred_network_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/raw_scores_tf_to_tg_inferred_network.tsv"
ground_truth_file = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/RN111.tsv"
merged_ground_truth_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/reference_networks/merged_reference_networks.csv"

output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output"

inferred_network = pd.read_csv(inferred_network_file, sep="\t")
print(inferred_network.head())
inferred_network["Source"] = inferred_network["Source"].str.upper()
inferred_network["Target"] = inferred_network["Target"].str.upper()

ground_truth = pd.read_csv(ground_truth_file, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
print(ground_truth.head())

merged_ground_truth = pd.read_csv(merged_ground_truth_file, sep='\t', header=0)
print(merged_ground_truth)

# Create a set of tuples from ground_truth for faster lookup
merged_ground_truth_pairs = set(zip(merged_ground_truth["Source"], merged_ground_truth["Target"]))

# Add the "Label" column to inferred_network
inferred_network["Label"] = inferred_network.apply(
    lambda row: 1 if (row["Source"], row["Target"]) in merged_ground_truth_pairs else 0,
    axis=1
)

# Print the resulting DataFrame
print(inferred_network.head())

print(f'Number of True predictions: {len(inferred_network[inferred_network["Label"] == 1])}')
print(f'Number of False predictions: {len(inferred_network[inferred_network["Label"] == 0])}')

# cell_columns = inferred_network.columns[2:-1]

# def compute_aggregated_features(df: pd.DataFrame, desired_n=5000):
#     """
#     For each row in df, resample (with replacement) the cell-level values in cell_cols 
#     until we have desired_n values. Then compute aggregated statistics (mean, std, min, max, median).
    
#     Returns a new DataFrame with these aggregated features.
#     """    
    
#     # Randomly sample desired_n columns with replacement
#     df_copy = df.copy()
#     data_cols = df_copy.drop(["Source", "Target", "Label"], axis=1)
#     sampled_columns = data_cols.sample(n=desired_n, axis='columns', replace=True)

#     # Compute aggregated statistics
#     sampled_columns["mean_score"]   = sampled_columns.mean(axis=1)
#     sampled_columns["std_score"]    = sampled_columns.std(axis=1)
#     sampled_columns["min_score"]    = sampled_columns.min(axis=1)
#     sampled_columns["max_score"]   = sampled_columns.max(axis=1)
#     sampled_columns["median_score"] = sampled_columns.median(axis=1)

#     return sampled_columns

# # Compute the aggregated features from a resampled set of 5000 cell-level values per row
# print("Randomly resampling to create 5000 randomly permuted cell columns")
# agg_features = compute_aggregated_features(inferred_network, desired_n=5000)
# Append the new aggregated features to the original DataFrame
# inferred_network = pd.concat([inferred_network, agg_features], axis=1)

# Define the list of aggregated features for training
# aggregated_features_new = ["mean_score", "std_score", "min_score", "max_score", "median_score"]
aggregated_features_new = [
    "tf_to_peak_binding_score",
    "TF_mean_expression",
    "TF_std_expression",
    "TF_min_expression",
    "TF_median_expression",
    "peak_to_target_score",
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

# Combine training features and labels for resampling
train_data = X_train.copy()
train_data["Label"] = y_train

# Separate positive and negative examples
pos_train = train_data[train_data["Label"] == 1]
neg_train = train_data[train_data["Label"] == 0]

# Undersample the negatives to match the number of positives
neg_train_sampled = neg_train.sample(n=len(pos_train), random_state=42)
train_data_balanced = pd.concat([pos_train, neg_train_sampled])

X_train_balanced = train_data_balanced[aggregated_features_new]
y_train_balanced = train_data_balanced["Label"]

print(f"Balanced training set: {len(pos_train)} positives and {len(neg_train_sampled)} negatives.")

# === Step 3: Train the Random Forest Classifier ===
rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf.fit(X_train_balanced, y_train_balanced)

# Evaluate on the (imbalanced) test set
y_pred = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

plt.figure(figsize=(8, 6))
plt.hist(y_pred_prob, bins=50)
plt.title("Histogram of Random Forest Prediction Probabilities")
plt.xlabel("Prediction Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("random_forest_prediction_probability_histogram.png", dpi=200)

# ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.3f}")

# Step 4: Feature Importance Analysis
feature_importances = pd.DataFrame({
    "Feature": aggregated_features_new,
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
plt.savefig("random_forest_feature_importance.png", dpi=200)

inferred_network["Score"] = rf.predict_proba(X)[:, 1]
inferred_network = inferred_network[["Source", "Target", "Score"]]
print(inferred_network.head())

inferred_network.to_csv(f'{output_dir}/rf_inferred_grn.tsv', sep='\t', index=False)

# new_data = pd.read_csv("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/cell_level_inferred_grn_testing.csv", sep="\t")

# # Identify the cell columns; for example, assume everything after "Target" is cell data:
# cell_columns_new = new_data.columns[2:]  # if there's no "Label" yet

# # Compute the aggregated features
# new_data["mean_score"]   = new_data[cell_columns_new].mean(axis=1)
# new_data["std_score"]    = new_data[cell_columns_new].std(axis=1)
# new_data["min_score"]    = new_data[cell_columns_new].min(axis=1)
# new_data["max_score"]    = new_data[cell_columns_new].max(axis=1)
# new_data["median_score"] = new_data[cell_columns_new].median(axis=1)

# aggregated_features = ["mean_score", "std_score", "min_score", "max_score", "median_score"]

# # Extract the aggregated feature vector
# X_new = new_data[aggregated_features]

# # Make predictions with your trained model
# new_data["Score"] = rf.predict_proba(X_new)[:, 1]

# new_data = new_data[["Source", "Target", "Score"]]

# new_data.to_csv(f'{output_dir}/rf_inferred_grn.tsv', sep='\t', index=False)

# ------ old code below -----

# # Select features and target
# num_cols = 500
# # Extracts the first 100 cells for training (skips 'Source' and 'Target' columns), includes the "Label" column
# dataset_subsample = pd.concat([inferred_network.iloc[:, 2:num_cols+2], inferred_network["Label"]], axis=1)

# def set_colnames_to_col_index(df: pd.DataFrame):
#     """Sets the column names to their column index (0, 1, 2, 3, etc.)"""
#     return df.rename(columns={str(x):y for x,y in zip(df.columns,range(0,len(df.columns)))})

# dataset_subsample_cols = set_colnames_to_col_index(dataset_subsample)

# # Set the training features as the de-identified cell column numbers
# features = [
#     column for column in dataset_subsample_cols
# ]

# print(f'Num features: {len(features)}')
# print(features)
# X = dataset_subsample_cols[features]
# y = dataset_subsample["Label"]

# # Split into train and test sets (80-20 split)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 2: Train the Random Forest Classifier
# rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
# rf.fit(X_train, y_train)

subsamples = 10
num_cols = 500
subsample_list = []
for i in range(subsamples):
    # Set the features as the randomly permuted 100 columns to match the size of the training data
    features = [
        "tf_to_peak_binding_score",
        "TF_mean_expression",
        "TF_std_expression",
        "TF_min_expression",
        "TF_median_expression",
        "peak_to_target_score",
        "TG_mean_expression",
        "TG_std_expression",
        "TG_min_expression",
        "TG_median_expression",
        "pearson_correlation"
    ]
    
    # Randomly reindex all columns containing cell data
    inferred_network_permuted = inferred_network.reindex(np.random.permutation(inferred_network[features]), axis='columns')
    print(f'Permuted inferred network shape: {inferred_network_permuted.shape}')
    
    # Take a 90% subsample of the cell data

    inferred_network_subsample = inferred_network_permuted.iloc[:, 0:num_cols+1]
    print(f'Randomized inferred network shape: {inferred_network_subsample.shape}')
        

    
    print(f'Num features: {len(features)}')
    X = inferred_network_subsample[features]
    inferred_network_subsample["Score"] = rf.predict_proba(X)[:, 1]
    # inferred_network_subsample[["Source", "Target"]] = inferred_network[["Source", "Target"]]

    # inferred_network_subsample = inferred_network_subsample[["Source", "Target", "Prediction"]]
    # inferred_network_subsample = inferred_network_subsample.rename(columns={"Prediction": "Score"})

    inferred_network_subsample = inferred_network[["Source", "Target", "Score"]]
    print(inferred_network_subsample.head())
    
    sample_dir_path = f'{output_dir}/rf_stability_analysis/inferred_network_subsample_{i+1}'
    if not os.path.exists(sample_dir_path):
        os.makedirs(sample_dir_path)

    inferred_network_subsample.to_csv(f'{sample_dir_path}/rf_inferred_grn.tsv', sep='\t', index=False)



# # # Step 3: Evaluate the Model
# # y_pred = rf.predict(X_test)
# # y_pred_prob = rf.predict_proba(X_test)[:, 1]  # Probability for the positive class

# # plt.figure(figsize=(8, 6))
# # plt.hist(y_pred_prob, bins=50)
# # plt.title("Histogram of Random Forest Prediction Probabilities")
# # plt.xlabel("Prediction Probability")
# # plt.ylabel("Frequency")
# # plt.tight_layout()
# # plt.savefig("random_forest_prediction_probability_histogram.png", dpi=200)

# # # Classification report
# # print(classification_report(y_test, y_pred))

# # # ROC-AUC score
# # roc_auc = roc_auc_score(y_test, y_pred_prob)
# # print(f"ROC-AUC Score: {roc_auc:.3f}")

# # # Step 4: Feature Importance Analysis
# # feature_importances = pd.DataFrame({
# #     "Feature": features,
# #     "Importance": rf.feature_importances_
# # }).sort_values(by="Importance", ascending=False)

# # # Plot feature importance
# # plt.figure(figsize=(8, 6))
# # plt.barh(feature_importances["Feature"], feature_importances["Importance"], color="skyblue")
# # plt.xlabel("Importance")
# # plt.ylabel("Feature")
# # plt.title("Feature Importance")
# # plt.gca().invert_yaxis()  # Highest importance at the top
# # plt.tight_layout()
# # plt.savefig("random_forest_feature_importance.png", dpi=200)

# # print("inferred network")
# # print(inferred_network.head())

# # # inferred_network["Normalized_Score"] = inferred_network["Prediction"] * inferred_network["Normalized_Score"]
# # importance_dict = feature_importances.set_index("Feature")["Importance"].to_dict()

# # # Weight each feature in inferred_network by its importance
# # for feature in feature_importances["Feature"]:
# #     if feature in inferred_network.columns:  # Ensure the feature exists in the DataFrame
# #         inferred_network[feature] = inferred_network[feature] * importance_dict[feature]