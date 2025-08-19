import os
import pandas as pd
from typing import Union

def merge_dataset_with_ground_truth(df: pd.DataFrame, ground_truth: pd.DataFrame, method:str="", gt_name: str="", show_network_size: bool=False):
    df['source_id'] = df['source_id'].str.capitalize()
    df['target_id'] = df['target_id'].str.capitalize()
    
    shared_sources = set(df['source_id']) & set(ground_truth['source_id'])
    shared_targets = set(df['target_id']) & set(ground_truth['target_id'])

    df_filtered = df[
        df['source_id'].isin(shared_sources) &
        df['target_id'].isin(shared_targets)
    ]

    gt_filtered = ground_truth[
        ground_truth['source_id'].isin(shared_sources) &
        ground_truth['target_id'].isin(shared_targets)
    ]
    
    df_merged = pd.merge(df_filtered, gt_filtered, on=['source_id', 'target_id'], how='outer', indicator=True)
    
    if show_network_size:
        if len(method) == 0:
            method = "sliding window"
        if len(gt_name) == 0:
            gt_name = "ground truth"
        
        print(f"- **Overlap between {method} and {gt_name} edges**")
        
        edges_in_df_and_ground_truth = df_merged[df_merged["_merge"] == "both"].drop(columns="_merge")
        df_not_ground_truth_edges = df_merged[df_merged["_merge"] == "left_only"].drop(columns="_merge")
        ground_truth_edges_only = df_merged[df_merged["_merge"] == "right_only"].drop(columns="_merge")
        
        tfs_in_both = edges_in_df_and_ground_truth["source_id"].drop_duplicates()
        tgs_in_both = edges_in_df_and_ground_truth["target_id"].drop_duplicates()
        
        print(f"\t- **Both {gt_name} and {method}**")
        print(f"\t\t- TFs: {len(tfs_in_both):,}")
        print(f"\t\t- TGs: {len(tgs_in_both):,}")
        print(f"\t\t- TF-TG Edges: {len(edges_in_df_and_ground_truth.drop_duplicates(subset=['source_id', 'target_id'])):,}")
        
        tfs_only_in_sliding_window = df[~df["source_id"].isin(ground_truth["source_id"])]["source_id"].drop_duplicates()
        tgs_only_in_sliding_window = df[~df["target_id"].isin(ground_truth["target_id"])]["target_id"].drop_duplicates()

        print(f"\t- **Only {method.capitalize()}**")
        print(f"\t\t- TFs: {len(tfs_only_in_sliding_window):,}")
        print(f"\t\t- TGs: {len(tgs_only_in_sliding_window):,}")
        print(f"\t\t- TF-TG Edges: {len(df_not_ground_truth_edges.drop_duplicates(subset=['source_id', 'target_id'])):,}")
        
        tfs_only_in_ground_truth = ground_truth[~ground_truth["source_col"].isin(df["source_id"])]["source_col"].drop_duplicates()
        tgs_only_in_ground_truth = ground_truth[~ground_truth["target_id"].isin(df["target_id"])]["target_id"].drop_duplicates()

        print(f"\t- **Only {gt_name}**")
        print(f"\t\t- TFs: {len(tfs_only_in_ground_truth):,}")
        print(f"\t\t- TGs: {len(tgs_only_in_ground_truth):,}")
        print(f"\t\t- Edges: {len(ground_truth_edges_only):,}")
    
    df_merged["label"] = df_merged["_merge"] == "both"
    
    df_labeled = df_merged.drop(columns=["_merge"])
    
    return df_labeled