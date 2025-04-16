import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

mESC_tss_filepath="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC/filtered_L2_E7.5_rep1/peak_to_gene_correlation.csv"

K562_tss_filepath="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/K562/K562_human_filtered/peak_to_gene_correlation.csv"

macrophage_tss_filepath="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/macrophage/macrophage_buffer1_filtered/peak_to_gene_correlation.csv"

def apply_different_tss_scaling(file_path, original_scaling, new_scaling):
    df = pd.read_csv(file_path, sep="\t", header=0)
    df["TSS_dist"] = -original_scaling * np.log(df["TSS_dist"])
    df["TSS_dist_score"] = df["TSS_dist"].apply(lambda x: math.exp(-(x / new_scaling)))

    # plt.figure(figsize=(4, 3))
    # plt.scatter(df["TSS_dist"], df["TSS_score"], alpha=0.5, s=10, edgecolors='none')
    # plt.xlabel("TSS Distance")
    # plt.ylabel("TSS Score")
    # plt.title("Scatterplot of TSS Distance vs TSS Score")
    # plt.grid(False)
    # plt.tight_layout()
    # plt.show()

    # plt.figure(figsize=(4, 3))
    # plt.hist(df["TSS_score"], bins=25)
    # plt.xlabel("TSS Score")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of TSS Scores")
    # plt.grid(False)
    # plt.tight_layout()
    # plt.show()
    
    df.to_csv(file_path, sep="\t", header=True, index=False)

original_scaling = 250_000
new_scaling = 250_000

apply_different_tss_scaling(mESC_tss_filepath, original_scaling, new_scaling)
apply_different_tss_scaling(K562_tss_filepath, original_scaling, new_scaling)
apply_different_tss_scaling(macrophage_tss_filepath, original_scaling, new_scaling)