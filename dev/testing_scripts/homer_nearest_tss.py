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
        
        homer_df["Peak Location"] = homer_df["Chr"].astype(str) + ":" + homer_df["Start"].astype(str) + "-" + homer_df["End"].astype(str)
            
        tf_to_nearest_gene_map = {
            "source_id":[TF_name.upper() for i in range(len(homer_df))],
            "target_id":homer_df.loc[:, "Gene Name"].str.upper().to_list()
            }
        index_col = np.arange(len(tf_to_nearest_gene_map["source_id"]))
            
        tf_to_nearest_gene = pd.DataFrame(tf_to_nearest_gene_map, index=index_col)
        
        shared_sources = set(tf_to_nearest_gene["source_id"]) & set(ground_truth_df["source_id"])
        shared_targets = set(tf_to_nearest_gene["target_id"]) & set(ground_truth_df["target_id"])

        tf_to_nearest_gene = tf_to_nearest_gene[
            tf_to_nearest_gene["source_id"].isin(shared_sources) &
            tf_to_nearest_gene["target_id"].isin(shared_targets)
        ].reset_index(drop=True)

        ground_truth_df = ground_truth_df[
            ground_truth_df["source_id"].isin(shared_sources) &
            ground_truth_df["target_id"].isin(shared_targets)
        ].reset_index(drop=True)

        tf_tg_overlap_w_ground_truth = pd.merge(
            ground_truth_df,
            tf_to_nearest_gene,
            on=["source_id", "target_id"],
            how="outer",
            indicator=True
        ).drop_duplicates()
        
        tf_targets.append(tf_tg_overlap_w_ground_truth)
    
total_tf_to_tg = pd.concat(tf_targets)
print(total_tf_to_tg)

ground_truth_only = total_tf_to_tg[total_tf_to_tg["_merge"] == "left_only"]
homer_only = total_tf_to_tg[total_tf_to_tg["_merge"] == "right_only"]
overlap = total_tf_to_tg[total_tf_to_tg["_merge"] == "both"]

print(f"Num edges only in ground truth = {ground_truth_only.shape[0]}")
print(f"Num edges only in Homer = {homer_only.shape[0]}")
print(f"Num edges in both = {overlap.shape[0]}")
    
    
    
    