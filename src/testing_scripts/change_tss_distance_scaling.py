import pandas as pd
import numpy as np
import math

mESC_tss_filepath="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC/filtered_L2_E7.5_rep1/peak_to_gene_correlation.csv"

K562_tss_filepath="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/K562/K562_human_filtered/peak_to_gene_correlation.csv"

macrophage_tss_filepath="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/macrophage/macrophage_buffer1_filtered/peak_to_gene_correlation.csv"

def apply_different_tss_scaling(file_path):
    df = pd.read_csv(file_path, sep="\t", header=0)
    df["TSS_dist"] = -25000 * np.log(df["TSS_dist"])
    df["TSS_dist"] = df["TSS_dist"].apply(lambda x: math.exp(-(x / 1000000)))
    df.to_csv(file_path, sep="\t", header=True, index=False)


apply_different_tss_scaling(mESC_tss_filepath)
apply_different_tss_scaling(K562_tss_filepath)
apply_different_tss_scaling(macrophage_tss_filepath)