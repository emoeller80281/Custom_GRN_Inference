import pandas as pd
import numpy as np
import os
from tqdm import tqdm

tf_to_tg = pd.read_csv(
    "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC/tf_to_tg_inferred_network.tsv",
    sep="\t",
    header=0,
    index_col=None
    )

score_df = tf_to_tg.groupby(["Source", "Target"]).apply(
    lambda x: (x["tf_to_peak_binding_score"] * x["peak_to_target_score"]).sum()
).reset_index(name="tf_to_tg_score")
print(score_df.head())

tf_to_tg_w_scores = pd.merge(tf_to_tg, score_df, how="right", on=["Source", "Target"])
print(tf_to_tg_w_scores.head())

tf_to_tg_w_scores["Score"] = tf_to_tg_w_scores["TF_mean_expression"] * tf_to_tg_w_scores["tf_to_tg_score"] * tf_to_tg_w_scores["TG_mean_expression"]

tf_to_tg_w_scores = tf_to_tg_w_scores[["Source", "Target", "Score"]]
print(tf_to_tg_w_scores.head())

# edges = tf_to_tg[["Source", "Target"]].drop_duplicates()
# tf_col_list = edges["Source"].to_list()
# tg_col_list = edges["Target"].to_list()

# # Iterate through each unique edge pair
# score_dict = {"Source": [], "Target": [], "score": []}
# for tf, tg in tqdm(zip(tf_col_list, tg_col_list), total=len(tf_col_list)):
#     # Extract rows with just these pairs
#     current_edge_subset = tf_to_tg.loc[(tf_to_tg["Source"] == tf) & (tf_to_tg["Target"] == tg)]
#     # print(current_edge_subset)
#     current_edge_score = sum(current_edge_subset["tf_to_peak_binding_score"] * current_edge_subset["peak_to_target_score"])
#     # print(f'\t{current_edge_score}\n')
#     score_dict["Source"].append(tf)
#     score_dict["Target"].append(tg)
#     score_dict["score"].append(current_edge_score)

# score_df = pd.DataFrame(score_dict)
# print(score_df.head())

# tf_to_tg_w_scores = pd.merge(tf_to_tg, score_df, how="left", on=["Source", "Target"])
    
    
    