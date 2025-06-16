import os
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm

homer_tf_motif_score_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/DS011_mESC/DS011_mESC_sample1/homer_results/homer_tf_motif_scores"
ground_truth_path = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"

def read_ground_truth(ground_truth_file):
    """Read ground truth TF--target pairs from a tab-delimited file.

    Parameters
    ----------
    ground_truth_file : str
        File path to a TSV with columns ``Source`` and ``Target``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardized column names ``source_id`` and ``target_id``.
    """
    ground_truth = pd.read_csv(
        ground_truth_file,
        sep='\t',
        quoting=csv.QUOTE_NONE,
        on_bad_lines='skip',
        header=0
    )
    ground_truth = ground_truth.rename(columns={"Source": "source_id", "Target": "target_id"})
    ground_truth = ground_truth[["source_id", "target_id"]]
    
    ground_truth["source_id"] = ground_truth["source_id"].str.upper()
    ground_truth["target_id"] = ground_truth["target_id"].str.upper()
    
    return ground_truth

ground_truth_df = read_ground_truth(ground_truth_path)
ground_truth_tfs = set(ground_truth_df["source_id"].unique())

tf_targets: list[pd.DataFrame] = []
for known_motif_file in tqdm(os.listdir(homer_tf_motif_score_dir)):
    motif_path = os.path.join(homer_tf_motif_score_dir, known_motif_file)
    
    # Only read the header row to get column names
    try:
        with open(motif_path, "r") as f:
            header = f.readline().strip().split("\t")
            TF_column = header[-1]
    except Exception as e:
        print(f"Error reading file {known_motif_file}: {e}")
        continue

    # Extract TF name robustly
    TF_name = TF_column.split('/')[0].split('(')[0].split(':')[0].upper()

    if TF_name in ground_truth_tfs:    
        print(TF_name)

        homer_df = pd.read_csv(
            motif_path, 
            sep="\t", 
            header=0, 
            index_col=0 # Setting the PeakID column as the index
            )
        homer_df.index.names = ["PeakID"]
        
        homer_df["peak_id"] = homer_df["Chr"].astype(str) + ":" + homer_df["Start"].astype(str) + "-" + homer_df["End"].astype(str)
        homer_df["source_id"] = TF_name
        homer_df = homer_df.rename(columns={"Gene Name":"target_id"})
        
        cols_of_interest = [
            "source_id",
            "peak_id",
            "target_id",
            "Annotation",
            "Distance to TSS",
            "Gene Type",
            "CpG%",
            "GC%"
        ]
        homer_df = homer_df[cols_of_interest]
        homer_df["target_id"] = homer_df["target_id"].str.upper()

        valid_targets = set(ground_truth_df["target_id"]) & set(homer_df["target_id"])
        valid_sources = set(ground_truth_df["source_id"]) & set(homer_df["source_id"])
        
        ground_truth_tf_edges = ground_truth_df[ground_truth_df["source_id"] == TF_name]

        homer_shared_genes = homer_df[
            homer_df["source_id"].isin(valid_sources) &
            homer_df["target_id"].isin(valid_targets)
        ].reset_index(drop=True)
        
        ground_truth_shared_genes = ground_truth_tf_edges[
            ground_truth_tf_edges["source_id"].isin(valid_sources) &
            ground_truth_tf_edges["target_id"].isin(valid_targets)
        ].reset_index(drop=True)
                
        tf_tg_overlap_w_ground_truth = pd.merge(
            ground_truth_shared_genes,
            homer_shared_genes,
            on=["source_id", "target_id"],
            how="outer",
            indicator=True
        )
        
        tf_targets.append(tf_tg_overlap_w_ground_truth)
    
total_tf_to_tg = pd.concat(tf_targets)
print(total_tf_to_tg)
print(total_tf_to_tg.shape)

overlapping_edges = total_tf_to_tg[total_tf_to_tg["_merge"] == "both"]
edges_only_in_homer = total_tf_to_tg[total_tf_to_tg["_merge"] == "right_only"]
edges_only_in_ground_truth = total_tf_to_tg[total_tf_to_tg["_merge"] == "left_only"]

print(f"Number of edges in both Homer and ground truth: {len(overlapping_edges):,.0f}")
print(f"Number of edges only in Homer: {len(edges_only_in_homer):,.0f}")
print(f"Number of edges only in the ground truth: {len(edges_only_in_ground_truth):,.0f}")

tp = (total_tf_to_tg["_merge"] == "both").sum()
fn = (total_tf_to_tg["_merge"] == "left_only").sum()
fp = (total_tf_to_tg["_merge"] == "right_only").sum()
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"Recall = {recall:.3f} ({tp} / {tp+fn})")

    
    
    
    