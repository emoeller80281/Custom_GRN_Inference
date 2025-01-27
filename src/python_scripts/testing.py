import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

# File paths
peak_to_gene_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/peak_gene_associations.csv"
motif_reg_score_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/total_motif_regulatory_scores.tsv"
inferred_grn_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/inferred_grn.tsv"
rna_dataset_sample1_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/multiomic_data_filtered_L2_E7.5_rep1_RNA.csv"
rna_dataset_sample2_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/SCENIC_PLUS/input/mESC/filtered_L2_E7.5_rep2/mESC_filtered_L2_E7.5_rep2_RNA.csv"

ground_truth_file = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/RN111.tsv"

fig_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/figures"


# Load data
peak_to_gene = pd.read_csv(peak_to_gene_file)
peak_to_gene["Target"] = peak_to_gene["gene"].apply(lambda x: x.upper())
# print("Peak to Gene Associations:")
# print(peak_to_gene.head())
# print()

motif_reg_scores = pd.read_csv(motif_reg_score_file, sep='\t')
# print("Motif Regulatory Scores:")
# print(motif_reg_scores.head())
# print()

inferred_grn = pd.read_csv(inferred_grn_file, sep='\t')
# print("Inferred GRN:")
# print(inferred_grn.head())
# print()

ground_truth = pd.read_csv(ground_truth_file, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
# print("Ground truth:")
# print(ground_truth)
# print()

def plot_histogram(non_ground_truth_db, ground_truth_db, score_name):
    plt.figure(figsize=(10, 7))
    
    plt.hist(non_ground_truth_db[score_name], bins=50, color='#4195df', alpha=0.7, label="Non-Ground Truth")
    plt.hist(ground_truth_db[score_name], bins=50, color='#dc8634', alpha=0.7, label="Ground Truth")
    
    plt.title(f"{score_name} distribution", fontsize=16)
    plt.xlabel(f"{score_name} Score")
    plt.ylabel("Frequency")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=18)
    
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
    
    plot_histogram(non_ground_truth_db, ground_truth_db, "TF_Mean_Expression")
    plot_histogram(non_ground_truth_db, ground_truth_db, "TG_Mean_Expression")
    plot_histogram(non_ground_truth_db, ground_truth_db, "TF_TG_Motif_Binding_Score")
    plot_histogram(non_ground_truth_db, ground_truth_db, "Normalized_Score")

    print("Ground truth")
    print(ground_truth)
    print()
    
    print("Non Ground Truth")
    print(non_ground_truth_db)
    

investigate_ground_truth_overlap(ground_truth, inferred_grn)

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




