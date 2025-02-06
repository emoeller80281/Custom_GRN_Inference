import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import random
import csv

inferred_network_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/cell_level_inferred_grn.csv"
ground_truth_file = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/RN111.tsv"
merged_ground_truth_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/reference_networks/merged_reference_networks.csv"

output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output"

inferred_network = pd.read_csv(inferred_network_file, sep="\t")
print(inferred_network.head())

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

# Select features and target
# Extracts the first 100 cells for training (skips 'Source' and 'Target' columns), includes the "Label" column
dataset_subsample = pd.concat([inferred_network.iloc[:, 2:103], inferred_network["Label"]], axis=1)

def set_colnames_to_col_index(df: pd.DataFrame):
    """Sets the column names to their column index (0, 1, 2, 3, etc.)"""
    return df.rename(columns={x:y for x,y in zip(df.columns,range(0,len(df.columns)))})

# Set the training features as the de-identified cell column numbers
features = [
    column for column in set_colnames_to_col_index(dataset_subsample)
]

print(f'Num features: {len(features)}')
X = dataset_subsample[features]
y = dataset_subsample["Label"]

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the Random Forest Classifier
rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)

# # Step 3: Evaluate the Model
# y_pred = rf.predict(X_test)
# y_pred_prob = rf.predict_proba(X_test)[:, 1]  # Probability for the positive class

# plt.figure(figsize=(8, 6))
# plt.hist(y_pred_prob, bins=50)
# plt.title("Histogram of Random Forest Prediction Probabilities")
# plt.xlabel("Prediction Probability")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.savefig("random_forest_prediction_probability_histogram.png", dpi=200)

# # Classification report
# print(classification_report(y_test, y_pred))

# # ROC-AUC score
# roc_auc = roc_auc_score(y_test, y_pred_prob)
# print(f"ROC-AUC Score: {roc_auc:.3f}")

# # Step 4: Feature Importance Analysis
# feature_importances = pd.DataFrame({
#     "Feature": features,
#     "Importance": rf.feature_importances_
# }).sort_values(by="Importance", ascending=False)

# # Plot feature importance
# plt.figure(figsize=(8, 6))
# plt.barh(feature_importances["Feature"], feature_importances["Importance"], color="skyblue")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.title("Feature Importance")
# plt.gca().invert_yaxis()  # Highest importance at the top
# plt.tight_layout()
# plt.savefig("random_forest_feature_importance.png", dpi=200)

# print("inferred network")
# print(inferred_network.head())

subsamples = 10
subsample_list = []
for i in range(subsamples):
    # Randomly reindex all columns containing cell data
    inferred_network_permuted = inferred_network.reindex(np.random.permutation(inferred_network.columns[2:-1]), axis='columns')
    print(f'Permuted inferred network shape: {inferred_network_permuted.shape}')
    
    # Take a 90% subsample of the cell data
    num_subset_col = int(round(len(inferred_network_permuted.columns) * 0.9, 0))
    print(num_subset_col)
    print(type(num_subset_col))
    inferred_network_subsample = inferred_network_permuted.iloc[:, 0:num_subset_col]
    print(f'Randomized inferred network shape: {inferred_network_subsample.shape}')
    
    dataset_columns = set_colnames_to_col_index(inferred_network_subsample[2:101])
    
    # Set the features as the randomly permuted 100 columns to match the size of the training data
    features = [
        column for column in dataset_columns
    ]
    
    print(f'Num features: {len(features)}')
    X = dataset_columns[features]
    inferred_network_subsample["Prediction"] = rf.predict_proba(X)[:, 1]

    # # inferred_network["Normalized_Score"] = inferred_network["Prediction"] * inferred_network["Normalized_Score"]
    # importance_dict = feature_importances.set_index("Feature")["Importance"].to_dict()

    # # Weight each feature in inferred_network by its importance
    # for feature in feature_importances["Feature"]:
    #     if feature in inferred_network.columns:  # Ensure the feature exists in the DataFrame
    #         inferred_network[feature] = inferred_network[feature] * importance_dict[feature]

    inferred_network_subsample = inferred_network_subsample[["Source", "Target", "Prediction"]]
    print(inferred_network_subsample.head())
    inferred_network_subsample = inferred_network_subsample.rename(columns={"Prediction": "Score"})

    print(inferred_network_subsample.head())

    inferred_network_subsample.to_csv(f'{output_dir}/rf_stability_analysis/inferred_network_subsample_{i+1}.tsv', sep='\t', index=False)