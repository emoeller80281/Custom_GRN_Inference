import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cell_rf_net_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC/filtered_L2_E7.5_rep1/cell_networks_rf"

cell_dfs = []
cell_names = []  # To record which cell each dataframe belongs to

# Read each cell's inferred GRN TSV file and add a column for cell name
for cell_dir in os.listdir(cell_rf_net_dir):
    cell_net = os.path.join(cell_rf_net_dir, cell_dir, "rf_inferred_grn.tsv")
    if os.path.exists(cell_net):
        # Extract cell name from directory name
        cell_name = cell_dir
        df = pd.read_csv(cell_net, sep="\t", header=0)
        df["cell"] = cell_name  # add a column for the cell
        cell_dfs.append(df)
        cell_names.append(cell_name)

# Concatenate dataframes from all cells
merged_cell_grns = pd.concat(cell_dfs, ignore_index=True)
print(merged_cell_grns.head())

# For consensus, we can compute the fraction of cells with a high score per TF-TG pair.
# Define a threshold for a "high-confidence" interaction. For example:
threshold = 0.5

# Create a binary column for high confidence
merged_cell_grns["is_high"] = merged_cell_grns["Score"] > threshold

# Group by TF-TG pair and compute:
# 1. The average Score across cells.
# 2. The fraction of cells where the interaction is "high confidence".
consensus_df = (
    merged_cell_grns
    .groupby(["Source", "Target"], as_index=False)
    .agg(avg_score=("Score", "mean"),
         consensus_fraction=("is_high", "mean"),
         cell_count=("cell", "nunique"))
)

print(consensus_df.head())

# Visualize the distribution of consensus fractions
plt.hist(consensus_df["consensus_fraction"], bins=50, color="skyblue", edgecolor="k")
plt.xlabel("Consensus Fraction (Fraction of cells with high Score)")
plt.ylabel("Number of TF-TG pairs")
plt.title("Distribution of Consensus Scores Across Cells")
plt.savefig("consensus_fraction.png")

# Now, you can use consensus_df to refine your training data.
# Option A: Merge consensus_fraction as a new feature into your original training set.
# Option B: Select only pairs with consensus_fraction above (or below) certain thresholds as pseudo-labels.
#
# For example, to use as pseudo-labels:
positive_threshold = 0.7  # consensus_fraction >= 0.7 as pseudo-positive
negative_threshold = 0.3  # consensus_fraction <= 0.3 as pseudo-negative

def label_pseudo(row):
    if row["consensus_fraction"] >= positive_threshold:
        return 1
    elif row["consensus_fraction"] <= negative_threshold:
        return 0
    else:
        return np.nan

consensus_df["pseudo_label"] = consensus_df.apply(label_pseudo, axis=1)
pseudo_df = consensus_df.dropna(subset=["pseudo_label"])

print(pseudo_df.head())

# You could then merge these pseudo-labeled interactions with your original training features,
# and retrain your random forest model. For example, if you have an original training dataframe 'training_df':


training_df = pd.merge(training_df, consensus_df[["Source", "Target", "avg_score", "consensus_fraction"]], on=["Source", "Target"], how="left")

# Or, create a new training set:
# X = pseudo_df[["avg_score", "consensus_fraction"]]  # plus any additional features you have
# y = pseudo_df["pseudo_label"]
#
# Then retrain using your favorite ML framework.
