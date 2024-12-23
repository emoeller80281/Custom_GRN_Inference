import pandas as pd
import os
from tqdm import tqdm

homer_tf_motif_score_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/homer_tf_motif_scores"

dataframes = []
for file in tqdm(os.listdir("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/homer_tf_motif_scores")):

    motif_to_peak = pd.read_csv(f'{homer_tf_motif_score_dir}/{file}', sep='\t', header=[0], index_col=None)

    # Count the number of motifs found for the peak
    motif_column = motif_to_peak.columns[-1]
    TF_name = motif_column.split('/')[0]
    # print(f'Processing TF {i+1} / {len(os.listdir("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/homer_tf_motif_scores"))}')
    
    motif_to_peak['Motif Count'] = motif_to_peak[motif_column].apply(lambda x: len(x.split(',')) / 3 if pd.notnull(x) else 0)
    motif_to_peak["TF"] = TF_name.split('(')[0]
    motif_to_peak["TG"] = motif_to_peak["Gene Name"]

    # Sum the total number of motifs for the TF
    total_motifs = motif_to_peak["Motif Count"].sum()

    # peak_binding_column = motif_to_peak[motif_to_peak.columns[-1]]
    cols_of_interest = ["TF", "TG", "Motif Count"]
    
    # Remove any rows with NA values
    df = motif_to_peak[cols_of_interest].dropna()

    # Filter out rows with no motifs
    filtered_df = df[df["Motif Count"] > 0]
    # print(filtered_df.head())

    # Sum the total motifs for each TG
    filtered_df = filtered_df.groupby(["TF", "TG"])["Motif Count"].sum().reset_index()

    # Calculate the score for the TG based on the total number of motifs for the TF
    filtered_df["Score"] = filtered_df["Motif Count"] / total_motifs

    # Save the final results
    final_df = filtered_df[["TF", "TG", "Score"]]

    if not "tf_motif_scores" in os.listdir(f'./output/'):
        os.makedirs("./output/tf_motif_scores")

    final_df.to_csv(f'./output/tf_motif_scores/{TF_name}_motif_regulatory_scores.tsv', sep='\t')

    dataframes.append(final_df)
    
combined_df = pd.concat(dataframes)
print(combined_df.head())
combined_df.to_csv(f'./output/total_motif_regulatory_scores.tsv', sep='\t')

