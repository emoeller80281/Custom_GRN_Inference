import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import csv

inferred_network_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/inferred_grn.tsv"
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
features = [
    "Peak Gene Score",
    "TF_Mean_Expression",
    "TG_Mean_Expression",
    "Motif_Score",
    "Correlation",
    "Normalized_Score"
]
X = inferred_network[features]
y = inferred_network["Label"]

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the Random Forest Classifier
rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)

# Step 3: Evaluate the Model
y_pred = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]  # Probability for the positive class

plt.figure(figsize=(8, 6))
plt.hist(y_pred_prob, bins=50)
plt.title("Histogram of Random Forest Prediction Probabilities")
plt.xlabel("Prediction Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("random_forest_prediction_probability_histogram.png", dpi=200)

# Classification report
print(classification_report(y_test, y_pred))

# ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.3f}")

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
plt.savefig("random_forest_feature_importance.png", dpi=200)

inferred_network["Prediction"] = rf.predict_proba(X)[:, 1]

# inferred_network["Normalized_Score"] = inferred_network["Prediction"] * inferred_network["Normalized_Score"]
importance_dict = feature_importances.set_index("Feature")["Importance"].to_dict()

# Weight each feature in inferred_network by its importance
for feature in feature_importances["Feature"]:
    if feature in inferred_network.columns:  # Ensure the feature exists in the DataFrame
        inferred_network[feature] = inferred_network[feature] * importance_dict[feature]
        
inferred_network["Normalized_Score"] = (
    inferred_network["Peak Gene Score"] * \
    inferred_network["TF_Mean_Expression"] * \
    inferred_network["TG_Mean_Expression"] * \
    inferred_network["Motif_Score"] * \
    inferred_network["Correlation"] * \
    inferred_network["Prediction"]
)

print(inferred_network.head())

inferred_network.to_csv(f'{output_dir}/rf_inferred_grn.tsv', sep='\t', index=False)