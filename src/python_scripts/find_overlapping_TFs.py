import pandas as pd

RNA_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/macrophage_buffer1_filtered_RNA.csv"
TF_motif_binding_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/total_motif_regulatory_scores.tsv"

print(f'Loading datasets')
RNA_dataset = pd.read_csv(RNA_file)
TF_motif_binding_df = pd.read_csv(TF_motif_binding_file, header=0, sep="\t", index_col=None)

# Find overlapping TFs
RNA_source = pd.DataFrame({
    "Source": RNA_dataset.iloc[1:, 0],
    "TF Expression": RNA_dataset.iloc[1:, 1:].sum(axis=1)
})

RNA_target = pd.DataFrame({
    "Target": RNA_dataset.iloc[1:, 0],
    "TG Expression": RNA_dataset.iloc[1:, 1:].sum(axis=1)
})

print(RNA_target)

overlapping_TFs = pd.merge(RNA_source, TF_motif_binding_df, how="inner", on="Source")
overlapping_TGs = pd.merge(RNA_target, TF_motif_binding_df, how="inner", on="Target")

# cell_num = 1

# # Rename the cell expression columns for the TF and TG datasets
# overlapping_TFs.rename(columns={overlapping_TFs.columns[cell_num]: "TF Expression"}, inplace=True)
# overlapping_TGs.rename(columns={overlapping_TGs.columns[cell_num]: "TG Expression"}, inplace=True)

# Construct the cell-level gene expression dataframe from the overlapping TF and TG dataframes
print(f'Construcing cell-level expression dataframes')
first_cell = pd.DataFrame({
    "Source": overlapping_TFs["Source"],
    "TF Expression": overlapping_TFs["TF Expression"],
    "Target": overlapping_TGs["Target"],
    "TG Expression": overlapping_TGs["TG Expression"],
    "Motif Score": overlapping_TFs["Score"]
})

# Only keep rows where both the TFs and TGs are expressed in the cell
first_cell_dataset = first_cell[(first_cell["TF Expression"] > 0) & (first_cell["TG Expression"] > 0)].reset_index()
first_cell_dataset["Total Score"] = first_cell_dataset["Motif Score"] * first_cell_dataset["TF Expression"]
print(first_cell_dataset.head())
print(f'TFs: {len(set(first_cell["Source"]))}')
print(f'TGs: {len(set(first_cell["Target"]))}')
# print(first_cell_dataset.describe())

inferred_grn = pd.DataFrame({
    "Source": first_cell_dataset["Source"],
    "Target": first_cell_dataset["Target"],
    "Score": first_cell_dataset["Total Score"]
})

inferred_grn.to_csv("./output/inferred_grn.tsv", sep="\t", index=False)

