import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/peak_gene_associations.csv")
print(df.head())

plt.figure(figsize=(10, 7))
plt.hist(df['score'], bins=20, color='blue')
plt.title("Normalized Peak to Gene Association Score Distribution", fontsize=16)
plt.xlabel("Normalized Peak to Gene Association Score")
plt.ylabel("Frequency")
plt.savefig("hist_peak_gene_association.png", dpi=300)