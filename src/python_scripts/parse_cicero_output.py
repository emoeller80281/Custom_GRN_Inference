import pandas as pd
import matplotlib.pyplot as plt

peak_gene_assoc_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/peak_gene_associations.csv"
fig_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/figures"

df = pd.read_csv(peak_gene_assoc_file)
print(df.head())

# Normalize the scores excluding 0 and 1
df['score_normalized'] = df['score'].apply(
    lambda x: (x - df['score'].min()) / (df['score'].max() - df['score'].min())
    if 0 != x != 1 else x  # Retain scores of 0 and 1 as they are
)

plt.figure(figsize=(10, 7))
plt.hist(df['score_normalized'], bins=50, color='blue')
plt.title("Normalized Cicero Peak to Gene Association Score Distribution", fontsize=16)
plt.xlabel("Normalized Peak to Gene Association Score")
plt.ylabel("Frequency")
plt.savefig(f"{fig_dir}/hist_peak_gene_association.png", dpi=300)

df.to_csv(peak_gene_assoc_file, index=False)