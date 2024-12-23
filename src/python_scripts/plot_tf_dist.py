import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(f'./output/inferred_grn.tsv', sep='\t')

df['Source'] = df['Source'].apply(lambda x: x.split('(')[0])

print(df.head())
print(df.describe())

fig = plt.hist(np.log2(df["Score"]), bins=100)
plt.xlabel("Motifs for TG / All Motifs")
plt.ylabel("Count")
plt.savefig("histogram.png")