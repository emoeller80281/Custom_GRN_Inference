import pandas as pd

# File paths
peak_to_gene_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/peak_gene_associations.csv"
motif_reg_score_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/total_motif_regulatory_scores.tsv"
inferred_grn_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/inferred_grn.tsv"
rna_dataset_sample1_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/multiomic_data_filtered_L2_E7.5_rep1_RNA.csv"
rna_dataset_sample2_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/SCENIC_PLUS/input/mESC/filtered_L2_E7.5_rep2/mESC_filtered_L2_E7.5_rep2_RNA.csv"

# Load data
peak_to_gene = pd.read_csv(peak_to_gene_file)
peak_to_gene["Target"] = peak_to_gene["gene"].apply(lambda x: x.upper())
print("Peak to Gene Associations:")
print(peak_to_gene.head())

motif_reg_scores = pd.read_csv(motif_reg_score_file, sep='\t')
print("Motif Regulatory Scores:")
print(motif_reg_scores.head())

inferred_grn = pd.read_csv(inferred_grn_file, sep='\t')
print("Inferred GRN:")
print(inferred_grn.head())


def parse_rna_dataset(rna_dataset_file):
    rna_dataset = pd.read_csv(rna_dataset_file)
    rna_dataset.rename(columns={rna_dataset.columns[0]: "Genes"}, inplace=True)
    rna_dataset["Genes"] = rna_dataset["Genes"].apply(lambda x: x.upper())
    print("RNA Dataset:")
    print(rna_dataset.head())
    return rna_dataset

rna_dataset1 = parse_rna_dataset(rna_dataset_sample1_file)
rna_dataset2 = parse_rna_dataset(rna_dataset_sample2_file)

combined_rna_dataset = pd.merge(
    rna_dataset1, rna_dataset2, 
    on="Genes", 
    how="outer", 
    suffixes=("_rep1", "_rep2")
)

# Fill NaN values with 0 before adding
combined_rna_dataset = combined_rna_dataset.fillna(0)

# Verify the combined dataset
print("Combined RNA Dataset:")
print(combined_rna_dataset.head())
print(f"Shape of combined dataset: {combined_rna_dataset.shape}")

# Combine all targets and sources
total_targets = set(
    pd.concat([peak_to_gene["Target"], motif_reg_scores["Target"], inferred_grn["Target"]])
)
total_sources = set(
    pd.concat([motif_reg_scores["Source"], inferred_grn["Source"]])
)

# Print stats for targets
print(f'Target genes')
print(f'\tpeak_to_gene: {len(set(peak_to_gene["Target"]))}')
print(f'\tmotif_reg_scores: {len(set(motif_reg_scores["Target"]))}')
print(f'\tinferred_grn: {len(set(inferred_grn["Target"]))}')
print(f'Total Unique Target Genes: {len(total_targets)}')

# Mark genes in RNA dataset
combined_rna_dataset["Target Genes"] = combined_rna_dataset["Genes"].apply(lambda x: x in total_targets)

print()

# Print stats for sources
print(f'Source genes')
print(f'\tmotif_reg_scores: {len(set(motif_reg_scores["Source"]))}')
print(f'\tinferred_grn: {len(set(inferred_grn["Source"]))}')
print(f'Total Unique Source Genes: {len(total_sources)}')

combined_rna_dataset["Source Genes"] = combined_rna_dataset["Genes"].apply(lambda x: x in total_sources)

# Final counts in RNA dataset
print(f'Target genes in the Combined RNAseq dataset: {combined_rna_dataset["Target Genes"].sum()}')
print(f'Source genes in the Combined RNAseq dataset: {combined_rna_dataset["Source Genes"].sum()}')
print()


