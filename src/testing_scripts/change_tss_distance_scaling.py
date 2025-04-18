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

    plt.figure(figsize=(4, 3))
    plt.scatter(df["TSS_dist"], df["TSS_score"], alpha=0.5, s=10, edgecolors='none')
    plt.xlabel("TSS Distance")
    plt.ylabel("TSS Score")
    plt.title("Scatterplot of TSS Distance vs TSS Score")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(4, 3))
    # plt.hist(df["TSS_score"], bins=25)
    # plt.xlabel("TSS Score")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of TSS Scores")
    # plt.grid(False)
    # plt.tight_layout()
    # plt.show()
    
    # df.to_csv(file_path, sep="\t", header=True, index=False)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def test_scaling_factors(file_path, original_scaling, scaling_tests):
    df = pd.read_csv(file_path, sep="\t", header=0)
    # df["TSS_dist"] = -original_scaling * np.log(df["TSS_dist_score"])
    
    cmap = get_cmap(len(scaling_tests))
    
    plt.figure(figsize=(4, 3))
    for i, scaling in enumerate(scaling_tests):
        df[str(scaling)] = df["TSS_dist"].apply(lambda x: math.exp(-(x / scaling)))
        plt.scatter(df[str(scaling)], df["TSS_dist"], alpha=0.5, s=10, edgecolors=cmap(i), label=str(scaling))
    
    plt.xlabel("TSS Distance Score")
    plt.ylabel("TSS Distance")
    plt.title("Impact of TSS scaling on TSS distance score")
    plt.legend(loc='best')
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    

original_scaling = 250_000
new_scaling = 250_000

scaling_tests = [
    10_000,
    25_000,
    250_000,
    500_000,
    750_000,
    1_000_000
    
]

test_scaling_factors(mESC_tss_filepath, original_scaling, scaling_tests)

# apply_different_tss_scaling(mESC_tss_filepath, original_scaling, new_scaling)
# apply_different_tss_scaling(K562_tss_filepath, original_scaling, new_scaling)
# apply_different_tss_scaling(macrophage_tss_filepath, original_scaling, new_scaling)

