import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import requests ## python -m pip install requests
response = requests.get("https://string-db.org/api/image/network?identifiers=PTCH1%0dSHH%0dGLI1%0dSMO%0dGLI3")
with open('string_network.png', 'wb') as fh:
    fh.write(response.content)

string_dataset = pd.read_csv(
    "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/string_protein_links.txt",
    sep=" ",
    header=0,
    index_col=None
    )

string_dataset["protein1"] = string_dataset["protein1"].str.replace("9606.", "")
string_dataset["protein2"] = string_dataset["protein2"].str.replace("9606.", "")

print(string_dataset.head())

tf_to_tg = pd.read_csv(
    "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/K562/K562_human_filtered/inferred_network.tsv",
    sep="\t",
    header=0,
    index_col=None
    )

print(tf_to_tg.head())

print(string_dataset["protein1"].isin(tf_to_tg["Source"]).sum())
print(string_dataset["protein2"].isin(tf_to_tg["Target"]).sum())

# Gets the protein interaction partners for the TFs
import stringdb
genes = tf_to_tg["Source"].to_list()[:10] + tf_to_tg["Target"].to_list()[:500]
string_ids = stringdb.get_string_ids(genes)
interaction_df = stringdb.get_interaction_partners(string_ids.queryItem)
functional_annot_df = stringdb.get_functional_annotation(string_ids.queryItem)
functional_annot_df = functional_annot_df[functional_annot_df["description"] == "Transcription regulator complex"]["inputGenes"]
gene_str_list = functional_annot_df.tolist()
tf_complex_genes = [gene for item in gene_str_list for gene in item.split(",")]

interaction_df = interaction_df[["preferredName_A", "preferredName_B"]].rename(columns={"preferredName_A": "Source", "preferredName_B": "Target"})
interaction_df = interaction_df[interaction_df["Target"].isin(tf_complex_genes)]

print(interaction_df.head())

merged_df = pd.merge(interaction_df, tf_to_tg, how='right')
merged_df = merged_df.fillna(0)
print(merged_df.head())


# score_df = tf_to_tg.groupby(["Source", "Target"]).apply(
