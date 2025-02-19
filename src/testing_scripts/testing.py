import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import shap
import xgboost as xgb
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, roc_curve, auc, precision_recall_curve

rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,  # General font size
    'axes.titlesize': 18,  # Title font size
    'axes.labelsize': 16,  # Axis label font size
    'xtick.labelsize': 14,  # X-axis tick label size
    'ytick.labelsize': 14,  # Y-axis tick label size
    'legend.fontsize': 12  # Legend font size
})

# File paths
peak_to_gene_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/peak_gene_associations.csv"
motif_reg_score_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/total_motif_regulatory_scores.tsv"
inferred_grn_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/inferred_grn.tsv"

# Sample files
rna_dataset_sample1_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/multiomic_data_filtered_L2_E7.5_rep1_RNA.csv"
atac_dataset_sample1_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/multiomic_data_filtered_L2_E7.5_rep1_ATAC.csv"

rna_dataset_sample2_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC_filtered_L2_E7.5_rep2_RNA.csv"
atac_dataset_sample2_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC_filtered_L2_E7.5_rep2_ATAC.csv"

ground_truth_file = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/RN111.tsv"

fig_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/figures"
input_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input"

cell_oracle_grn_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/other_method_grns/cell_oracle_mESC_E7.5_rep1_inferred_grn.csv"
linger_grn_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/other_method_grns/linger_mESC_E7.5_rep1_inferred_grn.tsv"
tripod_grn_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/other_method_grns/tripod_mESC_E7.5_rep1_inferred_grn.csv"
scenic_plus_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/other_method_grns/scenic_plus_mESC_E7.5_rep1_inferred_grn.tsv"

def combine_mesc_samples_1_and_2(rna_dataset_sample1_file, rna_dataset_sample2_file, atac_dataset_sample1_file, atac_dataset_sample2_file):
    sample_1_rna = pd.read_csv(rna_dataset_sample1_file)
    # print("Sample 1 RNA")
    # print(sample_1_rna.head())
    # print(sample_1_rna.shape)

    sample_2_rna = pd.read_csv(rna_dataset_sample2_file)
    # print("Sample 2 RNA")
    # print(sample_2_rna.head())
    # print(sample_2_rna.shape)

    merged_rna = pd.merge(sample_1_rna, sample_2_rna, how='outer', on='Unnamed: 0').fillna(0)
    merged_rna.iloc[1:, 1:] = merged_rna.iloc[1:, 1:].astype(int)
    print("RNA from samples 1 and 2 merged")
    print(merged_rna.head())
    print(merged_rna.shape)

    sample_1_atac = pd.read_csv(atac_dataset_sample1_file)
    print("Sample 1 ATAC")
    print(sample_1_atac.head())
    print(sample_1_atac.shape)

    sample_2_atac = pd.read_csv(atac_dataset_sample2_file)
    print("Sample 2 ATAC")
    print(sample_2_atac.head())
    print(sample_2_atac.shape)

    merged_atac = pd.merge(sample_1_atac, sample_2_atac, how='outer', on='Unnamed: 0').fillna(0)
    # merged_atac.iloc[1:, 1:] = merged_atac.iloc[1:, 1:].astype(int)
    print("ATAC from samples 1 and 2 merged")
    print(merged_atac.head())
    print(merged_atac.shape)

    merged_rna.to_csv(f"{input_dir}/mESC_filtered_L2_E7.5_merged_RNA.csv")
    merged_atac.to_csv(f"{input_dir}/mESC_filtered_L2_E7.5_merged_ATAC.csv")

def plot_histogram(non_ground_truth_db, ground_truth_db, score_name, log2):
    plt.figure(figsize=(10, 7))
    
    if log2:
        non_ground_truth_score = np.log2(non_ground_truth_db[score_name])
        ground_truth_score = np.log2(ground_truth_db[score_name])
    
    else:
        non_ground_truth_score = non_ground_truth_db[score_name]
        ground_truth_score = ground_truth_db[score_name]
    
    plt.hist(non_ground_truth_score, bins=50, color='#4195df', alpha=0.7, label="Non-Ground Truth")
    plt.hist(ground_truth_score, bins=50, color='#dc8634', alpha=0.7, label="Ground Truth")
    
    plt.title(f"{score_name} distribution", fontsize=16)
    plt.xlabel(f"{score_name} Score")
    plt.ylabel("Frequency")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/ground_truth_{score_name}_dist.png", dpi=300)
    plt.close()
    
def find_same_genes(gene_source_db, target_db):
    tfs = set(gene_source_db["Source"])
    tgs = set(gene_source_db["Target"])
    
    mask = target_db["Source"].isin(tfs) & target_db["Target"].isin(tgs)
    target_db = target_db[mask]
    
    return target_db

def investigate_ground_truth_overlap(ground_truth, inferred_grn):
    ground_truth = find_same_genes(inferred_grn, ground_truth)
    inferred_grn = find_same_genes(ground_truth, inferred_grn)
    
    merged = pd.merge(
        inferred_grn,
        ground_truth,
        on=['Source', 'Target'],  # Specify key columns explicitly
        how='left',
        indicator=True
    )

    # Step 2: Filter rows that exist ONLY in the left DataFrame (inferred_grn)
    non_ground_truth_db = merged[merged['_merge'] == 'left_only']

    # Step 3: Drop the indicator column (optional)
    non_ground_truth_db = non_ground_truth_db.drop(columns=['_merge'])
        
    ground_truth_db = pd.merge(ground_truth, inferred_grn,
             how='inner', on=["Target", "Source"])
    
    return non_ground_truth_db, ground_truth_db


# Load data
# peak_to_gene = pd.read_csv(peak_to_gene_file)
# peak_to_gene["Target"] = peak_to_gene["gene"].apply(lambda x: x.upper())
# print("Peak to Gene Associations:")
# print(peak_to_gene.head())
# print()

# motif_reg_scores = pd.read_csv(motif_reg_score_file, sep='\t')
# print("Motif Regulatory Scores:")
# print(motif_reg_scores.head())
# print()

# inferred_grn = pd.read_csv(inferred_grn_file, sep='\t')
# print("Inferred GRN:")
# print(inferred_grn.head())
# print()

def load_cell_oracle_grn(cell_oracle_grn_file):
    cell_oracle_grn = pd.read_csv(cell_oracle_grn_file, sep=",", index_col=0, header=0)
    cell_oracle_grn = cell_oracle_grn[["source", "target", "coef_abs"]]
    cell_oracle_grn.rename(columns={"source": "Source", "target": "Target", "coef_abs": "Score"}, inplace=True)
    cell_oracle_grn["Source"] = cell_oracle_grn["Source"].str.upper()
    cell_oracle_grn["Target"] = cell_oracle_grn["Target"].str.upper()
    
    return cell_oracle_grn

def load_linger_grn(linger_grn_file):
    linger_grn = pd.read_csv(linger_grn_file, sep='\t')
    linger_grn = linger_grn.melt(id_vars=[linger_grn.columns[0]], value_vars=[i for i in linger_grn.columns[1:]])
    linger_grn.rename(columns={"Unnamed: 0": "Target", "variable": "Source", "value": "Score"}, inplace=True)
    linger_grn = linger_grn[["Source", "Target", "Score"]]
    linger_grn["Source"] = linger_grn["Source"].str.upper()
    linger_grn["Target"] = linger_grn["Target"].str.upper()
    return linger_grn

def load_tripod_grn(tripod_grn_file):
    tripod_grn = pd.read_csv(tripod_grn_file, sep=",", index_col=None, header=0)
    tripod_grn.rename(columns={"gene": "Target", "TF": "Source", "abs_coef": "Score"}, inplace=True)
    tripod_grn = tripod_grn[["Source", "Target", "Score"]]
    tripod_grn["Source"] = tripod_grn["Source"].str.upper()
    tripod_grn["Target"] = tripod_grn["Target"].str.upper()
    return tripod_grn

def load_custom_grn(inferred_grn_file):
    custom_grn = pd.read_csv(inferred_grn_file, sep='\t')
    custom_grn.rename(columns={"Normalized_Score": "Score"}, inplace=True)
    custom_grn = custom_grn[["Source", "Target", "Score"]]
    custom_grn["Source"] = custom_grn["Source"].str.upper()
    custom_grn["Target"] = custom_grn["Target"].str.upper()
    return custom_grn

def load_scenic_plus_grn(scenic_plus_file):
    scenic_plus_grn = pd.read_csv(scenic_plus_file, sep='\t')
    scenic_plus_grn["Source"] = scenic_plus_grn["Source"].str.upper()
    scenic_plus_grn["Target"] = scenic_plus_grn["Target"].str.upper()
    return scenic_plus_grn

def combine_method_grns(cell_oracle_grn_file, linger_grn_file, tripod_grn_file, scenic_plus_grn, custom_grn_file):
    cell_oracle_grn = load_cell_oracle_grn(cell_oracle_grn_file)
    print("Cell oracle GRN")
    print(cell_oracle_grn.head())
    print(cell_oracle_grn.shape)
    print()

    linger_grn = load_linger_grn(linger_grn_file)
    print("Linger GRN")
    print(linger_grn.head())
    print(linger_grn.shape)
    print()

    tripod_grn = load_tripod_grn(tripod_grn_file)
    print("Tripod GRN")
    print(tripod_grn.head())
    print(tripod_grn.shape)
    print()

    scenic_plus_grn = load_scenic_plus_grn(scenic_plus_file)
    print("SCENIC+ GRN")
    print(scenic_plus_grn.head())
    print(scenic_plus_grn.shape)
    print()
    
    custom_grn = load_custom_grn(custom_grn_file)
    print("Custom GRN")
    print(custom_grn.head())
    print(custom_grn.shape)
    print()
        
    # Process each GRN and rename the score column to method-specific names
    cell_oracle_grn = cell_oracle_grn.rename(columns={"Score": "CellOracle Score"})
    linger_grn = linger_grn.rename(columns={"Score": "Linger Score"})
    tripod_grn = tripod_grn.rename(columns={"Score": "Tripod Score"})
    scenic_plus_grn = scenic_plus_grn.rename(columns={"Score": "SCENIC+ Score"})
    custom_grn = custom_grn.rename(columns={"Score": "Custom Score"})  # Ensure custom_grn has "Score" column

    # Merge all DataFrames using outer joins
    combined_df = cell_oracle_grn.merge(
        linger_grn, on=["Source", "Target"], how="outer"
    ).merge(
        tripod_grn, on=["Source", "Target"], how="outer"
    ).merge(
        scenic_plus_grn, on=["Source", "Target"], how="outer"
    ).merge(
        custom_grn, on=["Source", "Target"], how="outer"
    )

    combined_df.fillna(0, inplace=True)
    
    def normalize(scores):
        return scores / max(scores)

    combined_df["Score"] = \
        (normalize(combined_df["CellOracle Score"]) + \
        normalize(combined_df["Linger Score"]) + \
        normalize(combined_df["Tripod Score"]) + \
        normalize(combined_df["SCENIC+ Score"]) + \
        normalize(combined_df["Custom Score"])) / 5

    print("Combined GRN")
    print(combined_df.head())
    print(combined_df.shape)
    
    return combined_df

combined_df = combine_method_grns(cell_oracle_grn_file, linger_grn_file, tripod_grn_file, scenic_plus_file, inferred_grn_file)

ground_truth = pd.read_csv(ground_truth_file, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
print("Ground truth:")
print(ground_truth)
print()

non_ground_truth_db, ground_truth_db = investigate_ground_truth_overlap(ground_truth, combined_df)

print("Ground truth")
print(ground_truth_db)
print()

print("Non Ground Truth")
print(non_ground_truth_db)

plot_histogram(non_ground_truth_db, ground_truth_db, "Score", log2=True)

non_ground_truth_db["Label"] = 0
ground_truth_db["Label"] = 1

print(ground_truth_db)

combined_labeled_df = pd.merge(ground_truth_db, non_ground_truth_db, how='outer')
print(combined_labeled_df)

print(f'Number of True predictions: {len(combined_labeled_df[combined_labeled_df["Label"] == 1])}')
print(f'Number of False predictions: {len(combined_labeled_df[combined_labeled_df["Label"] == 0])}')

features = [
    "CellOracle Score",
    "Linger Score",
    "Tripod Score",
    "SCENIC+ Score",
    "Custom Score",
]
X = combined_labeled_df[features]
y = combined_labeled_df["Label"]

weights = class_weight.compute_class_weight('balanced', classes=np.unique(combined_labeled_df["Label"]), y=combined_labeled_df["Label"])

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train XGBoost with class weights
model = xgb.XGBClassifier(
    scale_pos_weight=weights[1]/weights[0],  # Adjust for imbalance
    objective="binary:logistic",
    n_estimators=200,
    max_depth=5
)
model.fit(X_train, y_train)

# Step 3: Evaluate the Model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")

xgb.plot_importance(model, importance_type="weight")

non_ground_truth_db["XGBoost Prediction"] = model.predict_proba(non_ground_truth_db[features])[:, 1]
ground_truth_db["XGBoost Prediction"] = model.predict_proba(ground_truth_db[features])[:, 1]
plot_histogram(non_ground_truth_db, ground_truth_db, "XGBoost Prediction", log2=False)


plt.figure(figsize=(8, 6))
plt.hist(y_proba, bins=50)
plt.title("Histogram of XGBoost Prediction Probabilities")
plt.xlabel("Prediction Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{fig_dir}/multiple_method_xgboost_prediction_probability_histogram.png", dpi=200)
plt.close()

# Classification report
print(classification_report(y_test, y_pred))

def plot_auroc_auprc(y_true, y_pred, filename):
    # Normal Distribution
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    prc_auc = auc(recall, precision)
    
    # Randomized Distribution
    uniform_distribution = np.random.uniform(low = 0.0, high = 1.0, size = len(y_pred))
    rand_y_pred = np.random.choice(uniform_distribution, size=len(y_pred), replace=True)
    rand_y_true = np.random.choice([0,1], size=len(y_true), replace=True)
    
    rand_fpr, rand_tpr, _ = roc_curve(rand_y_true, rand_y_pred)
    rand_precision, rand_recall, _ = precision_recall_curve(rand_y_true, rand_y_pred)
    rand_roc_auc = auc(rand_fpr, rand_tpr)
    rand_prc_auc = auc(rand_recall, rand_precision)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(fpr, tpr, label=f'AUROC = {roc_auc:.2f}', linestyle='-')
    axes[0].plot(rand_fpr, rand_tpr, label=f'Randomized AUROC = {rand_roc_auc:.2f}', linestyle='--')
    axes[0].plot([0, 1], [0, 1], 'k--')  # Diagonal line for random performance
    
    axes[1].plot(recall, precision, label=f'AUPRC = {prc_auc:.2f}', linestyle='-')
    axes[1].plot(rand_recall, rand_precision, label=f'Randomized AUPRC = {rand_prc_auc:.2f}', linestyle='--')

    axes[0].set_title(f"AUROC")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_ylim((0,1))
    axes[0].set_xlim((0,1))
        
    axes[1].set_title(f"AUPRC")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim((0,1))
    axes[1].set_xlim((0,1))

    # Place the ROC legend below its plot
    axes[0].legend(
        loc="upper center",  # Anchor the legend at the upper center of the bounding box
        bbox_to_anchor=(0.5, -0.2),  # Move it below the axes
        ncol=1,  # Number of columns in the legend
    )

    # Place the PR legend below its plot
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=1,
    )

    # Adjust layout and display the figure
    fig.tight_layout()
    plt.savefig(f'{fig_dir}/{filename}', dpi=200)
    plt.close()

plot_auroc_auprc(y_test, y_proba, "multiple_method_xg_boost_auroc_auprc.png")

prediction_scores = combined_labeled_df["Score"]
prediction_labels = combined_labeled_df["Label"]
plot_auroc_auprc(prediction_labels, prediction_scores, "multiple_method_average_score_auroc_auprc_with_custom.png")
plot_histogram(non_ground_truth_db, ground_truth_db, "Score", log2=True)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
plt.figure(figsize=(10,12))
shap.summary_plot(shap_values, features=X_train, feature_names=features, plot_type="layered_violin", show=False, plot_size=[10,8])
plt.title("XGBoost SHAP Values")
plt.savefig(f"{fig_dir}/xg_boost_shap_summary.png", dpi=200)
plt.close()

# # Plot feature importance
# plt.figure(figsize=(8, 6))
# plt.barh(feature_importances["Feature"], feature_importances["Importance"], color="skyblue")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.title("Feature Importance")
# plt.gca().invert_yaxis()  # Highest importance at the top
# plt.tight_layout()
# plt.savefig(f"{fig_dir}/multiple_method_random_forest_feature_importance.png", dpi=200)

# non_ground_truth_db, ground_truth_db = investigate_ground_truth_overlap(ground_truth, inferred_grn)
# plot_histogram(non_ground_truth_db, ground_truth_db, "TF_Mean_Expression", log2=False)
# plot_histogram(non_ground_truth_db, ground_truth_db, "TG_Mean_Expression", log2=False)
# plot_histogram(non_ground_truth_db, ground_truth_db, "TF_TG_Motif_Binding_Score", log2=False)
# plot_histogram(non_ground_truth_db, ground_truth_db, "Normalized_Score", log2=True)


# def parse_rna_dataset(rna_dataset_file):
#     rna_dataset = pd.read_csv(rna_dataset_file)
#     rna_dataset.rename(columns={rna_dataset.columns[0]: "Genes"}, inplace=True)
#     rna_dataset["Genes"] = rna_dataset["Genes"].apply(lambda x: x.upper())
#     print("RNA Dataset:")
#     print(rna_dataset.head())
#     return rna_dataset

# rna_dataset1 = parse_rna_dataset(rna_dataset_sample1_file)
# rna_dataset2 = parse_rna_dataset(rna_dataset_sample2_file)

# combined_rna_dataset = pd.merge(
#     rna_dataset1, rna_dataset2, 
#     on="Genes", 
#     how="outer", 
#     suffixes=("_rep1", "_rep2")
# )

# # Fill NaN values with 0 before adding
# combined_rna_dataset = combined_rna_dataset.fillna(0)

# # Verify the combined dataset
# print("Combined RNA Dataset:")
# print(combined_rna_dataset.head())
# print(f"Shape of combined dataset: {combined_rna_dataset.shape}")

# # Combine all targets and sources
# total_targets = set(
#     pd.concat([peak_to_gene["Target"], motif_reg_scores["Target"], inferred_grn["Target"]])
# )
# total_sources = set(
#     pd.concat([motif_reg_scores["Source"], inferred_grn["Source"]])
# )

# # Print stats for targets
# print(f'Target genes')
# print(f'\tpeak_to_gene: {len(set(peak_to_gene["Target"]))}')
# print(f'\tmotif_reg_scores: {len(set(motif_reg_scores["Target"]))}')
# print(f'\tinferred_grn: {len(set(inferred_grn["Target"]))}')
# print(f'Total Unique Target Genes: {len(total_targets)}')

# # Mark genes in RNA dataset
# combined_rna_dataset["Target Genes"] = combined_rna_dataset["Genes"].apply(lambda x: x in total_targets)

# print()

# # Print stats for sources
# print(f'Source genes')
# print(f'\tmotif_reg_scores: {len(set(motif_reg_scores["Source"]))}')
# print(f'\tinferred_grn: {len(set(inferred_grn["Source"]))}')
# print(f'Total Unique Source Genes: {len(total_sources)}')

# combined_rna_dataset["Source Genes"] = combined_rna_dataset["Genes"].apply(lambda x: x in total_sources)

# # Final counts in RNA dataset
# print(f'Target genes in the Combined RNAseq dataset: {combined_rna_dataset["Target Genes"].sum()}')
# print(f'Source genes in the Combined RNAseq dataset: {combined_rna_dataset["Source Genes"].sum()}')
# print()




