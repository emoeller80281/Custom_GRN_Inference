import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

peak_gene_assoc_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/peak_gene_associations.csv"
fig_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/figures"

df = pd.read_csv(peak_gene_assoc_file)
print(df.head())

def normalize_peak_to_peak_scores(df):
    # Identify scores that are not 0 or 1
    mask = (df['score'] != 0) & (df['score'] != 1)
    filtered_scores = df.loc[mask, 'score']

    if not filtered_scores.empty:
        # Compute min and max of non-0/1 scores
        score_min = filtered_scores.min()
        score_max = filtered_scores.max()
        
        # Handle edge case where all non-0/1 scores are the same
        if score_max == score_min:
            # Set all non-0/1 scores to 0 (or another default value)
            score_normalized = np.where(mask, 0, df['score'])
        else:
            # Normalize non-0/1 scores and retain 0/1 values
            score_normalized = np.where(
                mask,
                (df['score'] - score_min) / (score_max - score_min),
                df['score']
            )
    else:
        # All scores are 0 or 1; no normalization needed
        score_normalized = df['score']
    
    return score_normalized

df['score_normalized'] = normalize_peak_to_peak_scores(df)

plt.figure(figsize=(10, 7))
plt.hist(df['score'], bins=50, color='blue')
plt.title("Cicero Peak to Gene Association Score Distribution", fontsize=16)
plt.xlabel("Normalized Peak to Gene Association Score")
plt.ylabel("Frequency")
plt.savefig(f"{fig_dir}/cicero_hist_peak_gene_association.png", dpi=300)

plt.figure(figsize=(10, 7))
plt.hist(df['score_normalized'], bins=50, color='blue')
plt.title("Normalized Cicero Peak to Gene Association Score Distribution", fontsize=16)
plt.xlabel("Normalized Peak to Gene Association Score")
plt.ylabel("Frequency")
plt.savefig(f"{fig_dir}/cicero_hist_peak_gene_association_norm.png", dpi=300)

df.to_csv(peak_gene_assoc_file, index=False)