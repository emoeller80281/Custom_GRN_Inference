import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from typing import Union

COMBINED_SCORE_DF_FEATURES = [
    "mean_TF_expression", 
    "mean_peak_accessibility", 
    "mean_TG_expression",
    "string_combined_score",
    "string_experimental_score",
    "string_textmining_score"
    ]

ground_truth_file = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/"

TMP_DIR = os.path.join(PROJECT_DIR, "src/testing_scripts/tmp")
os.makedirs(TMP_DIR, exist_ok=True)

ds011_input_dir = os.path.join(PROJECT_DIR, "input/DS011_mESC/DS011_mESC_sample1")
ds011_output_dir = os.path.join(PROJECT_DIR, "output/DS011_mESC/DS011_mESC_sample1")

mesc_input_dir = os.path.join(PROJECT_DIR, "input/mESC/filtered_L2_E7.5_rep2")
mesc_output_dir = os.path.join(PROJECT_DIR, "output/mESC/filtered_L2_E7.5_rep2")

def read_ground_truth(ground_truth_file):
    ground_truth = pd.read_csv(ground_truth_file, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    ground_truth = ground_truth.rename(columns={"Source": "source_id", "Target": "target_id"})
    return ground_truth

def create_score_path_dict(
    selected_features: list[str], 
    output_dir: str
    ) -> dict[str, str]:
    """
    Creates a dictionary of file paths to each score file for a given set of selected features.

    Arguments:
        selected_features (list[str]): List of selected feature score names
        output_dir (str): Output directory for the sample

    Returns:
        selected_feature_path_dict (dict[str:str]): A dictionary containing the selected feature names
        along with the path to the data file for that feature
    """
    
    feature_score_file_path_dict = {
        'mean_TF_expression' : os.path.join(output_dir, "inferred_grns/inferred_score_df.parquet"),
        'mean_peak_accessibility' : os.path.join(output_dir, "inferred_grns/inferred_score_df.parquet"),
        'mean_TG_expression' : os.path.join(output_dir, "inferred_grns/inferred_score_df.parquet"),
        'cicero_score' : os.path.join(output_dir, "cicero_peak_to_tg_scores.parquet"),
        'TSS_dist_score' : os.path.join(output_dir, "peak_to_gene_correlation.parquet"), 
        'correlation' : os.path.join(output_dir, "peak_to_gene_correlation.parquet"),
        'homer_binding_score' : os.path.join(output_dir, "homer_tf_to_peak.parquet"), 
        'sliding_window_score' : os.path.join(output_dir, "sliding_window_tf_to_peak_score.parquet"), 
        'string_combined_score' : os.path.join(output_dir, "inferred_grns/inferred_score_df.parquet"), 
        'string_experimental_score' : os.path.join(output_dir, "inferred_grns/inferred_score_df.parquet"), 
        'string_textmining_score' : os.path.join(output_dir, "inferred_grns/inferred_score_df.parquet")
    }

    selected_feature_path_dict = {}
    for feature_name in selected_features:
        if feature_name in feature_score_file_path_dict.keys():
            selected_feature_path_dict[feature_name] = feature_score_file_path_dict[feature_name]
            
    for feature_name, path in selected_feature_path_dict.items():
        assert os.path.isfile(path) | os.path.isdir(path), f'Error: {path} is not a file or directory'
        
    return selected_feature_path_dict

def check_for_features_in_combined_score_df(feature_score_dict: dict[str,str]):
    features = []
    for score_name in feature_score_dict.keys():
        if score_name in COMBINED_SCORE_DF_FEATURES:
            print(f'  - {score_name} in feature score path')
            features.append(score_name)
            
    return features

def load_melted_inferred_grn_ddf(inferred_net_path: str, feature_scores: list[str]):

    melted_tf_score_ddf = dd.read_parquet(inferred_net_path, engine="pyarrow")
    
    melted_tf_score_ddf = melted_tf_score_ddf[melted_tf_score_ddf["score_type"].isin(feature_scores)]
    
    grouped = (
        melted_tf_score_ddf
        .groupby(["source_id", "peak_id", "target_id", "score_type"])
        ["score_value"]
        .mean()
        .reset_index()
    )
    
    pdf = grouped.compute()
    
    wide_df = pdf.pivot_table(
        index=["source_id", "peak_id", "target_id"],
        columns="score_type",
        values="score_value",
        aggfunc="first"
    ).reset_index()
    
    
    return wide_df

def load_individual_score_dataframes(score_path_dict):
    individual_feature_score_dataframes = {}
    for feature_name, path in score_path_dict.items():
        if feature_name not in COMBINED_SCORE_DF_FEATURES:
            print(f'  - Loading {feature_name} DataFrame')
            df = pd.read_parquet(path, engine="pyarrow")
            df = df.reset_index(drop=True)
            individual_feature_score_dataframes[feature_name] = df

    return individual_feature_score_dataframes

def build_tf_peak_tg_edge_dataframe():

    # Build a DataFrame of TF -> peak -> TG edges
    print("\nBuilding a DataFrame of TF-peak-TG edges")
    tf_peak_tg_edges_save_path = os.path.join(TMP_DIR, "tf_peak_tg_edges.parquet")
    if not os.path.isfile(tf_peak_tg_edges_save_path):
        
        selected_features = [
            'mean_TF_expression',
            'mean_peak_accessibility',
            'mean_TG_expression',
            'cicero_score',
            'TSS_dist_score', 
            'correlation',
            'homer_binding_score', 
            'sliding_window_score', 
            'string_combined_score', 
            'string_experimental_score', 
            'string_textmining_score'
            ]

        mesc_score_paths: dict = create_score_path_dict(selected_features, mesc_output_dir)

        print("\nLoading and combining individual feature scores")
        feature_score_dataframes: dict = load_individual_score_dataframes(mesc_score_paths)

        print('\nChecking for scores originating from the combined feature score DataFrame')
        combined_df_features: list[str] = check_for_features_in_combined_score_df(mesc_score_paths)

        if len(combined_df_features) > 0:
            print('\nLoading combined feature score dataframe')
            inferred_df_path = mesc_score_paths[combined_df_features[0]] # get path for the first feature name in combined_df_feature
            inferred_df = load_melted_inferred_grn_ddf(inferred_df_path, combined_df_features)
            print('\tDone!')
            
            print("\nSplitting off combined scores into individual dataframes")
            for feature_name in combined_df_features:
                feature_df = inferred_df[["source_id", "peak_id", "target_id", feature_name]]
                feature_score_dataframes[feature_name] = feature_df
            
        else:
            print('  - No scores from the combined feature score DataFrame')
        
        # Build peak -> TG edges from Cicero and correlation scores
        peak_to_tg_save_path = os.path.join(TMP_DIR, "peak_to_tg_edges.parquet")
        if not os.path.isfile(peak_to_tg_save_path):
            print('\tBuilding peak to TG edges')
            cicero_peak_to_tg = feature_score_dataframes["cicero_score"][["peak_id", "target_id"]]
            corr_peak_to_tg = feature_score_dataframes["correlation"][["peak_id", "target_id"]]
            
            peak_to_tg_edges = pd.merge(cicero_peak_to_tg, corr_peak_to_tg, how="outer")
            
            print('\t\tDone! Saving to parquet file')
            peak_to_tg_edges.to_parquet(peak_to_tg_save_path, engine="pyarrow", compression="snappy")
            
        else:
            print("\t- Peak to TG edge file exists, loading...")
            peak_to_tg_edges = pd.read_parquet(peak_to_tg_save_path, engine="pyarrow")
            
        # Build TF -> peak edges from Homer and sliding window scores
        tf_to_peak_save_path = os.path.join(TMP_DIR, "tf_to_peak_edges.parquet")
        if not os.path.isfile(tf_to_peak_save_path):
            print('\t- Building TF to peak edges')
            sliding_window_tf_to_peak = feature_score_dataframes["sliding_window_score"][["source_id", "peak_id"]]
            homer_tf_to_peak = feature_score_dataframes["homer_binding_score"][["source_id", "peak_id"]]
            
            tf_to_peak_edges = pd.merge(sliding_window_tf_to_peak, homer_tf_to_peak, how="outer")
            
            print('\t\tDone! Saving to parquet file')
            tf_to_peak_edges.to_parquet(tf_to_peak_save_path, engine="pyarrow", compression="snappy")
            
        else:
            print("\t- TF to peak edge file exists, loading...")
            tf_to_peak_edges = pd.read_parquet(tf_to_peak_save_path, engine="pyarrow")
        
        print("Merging TF to peak and peak to TG edges")
        tf_peak_tg_edges = pd.merge(tf_to_peak_edges, peak_to_tg_edges, on=["peak_id"], how="inner")
        print("\tDone!")
        tf_peak_tg_edges.to_parquet(tf_peak_tg_edges_save_path, engine="pyarrow", compression="snappy")
        
        return dd.from_pandas(tf_peak_tg_edges, npartitions=1)
        
    else:
        print("\t- TF-peak-TG edge file exists, loading...")
        tf_peak_tg_edges = dd.read_parquet(tf_peak_tg_edges_save_path, engine="pyarrow")
        
        return tf_peak_tg_edges

def label_edges_with_ground_truth(inferred_network_dd, ground_truth_df):
    import dask.dataframe as dd
    import numpy as np
    ground_truth_pairs = set(zip(
        ground_truth_df["source_id"].str.upper(),
        ground_truth_df["target_id"].str.upper()
    ))
    
    inferred_network_dd["source_id"] = inferred_network_dd["source_id"].str.upper()
    inferred_network_dd["target_id"] = inferred_network_dd["target_id"].str.upper()


    def label_partition(df):
        df = df.copy()
        tf_tg_tuples = list(zip(df["source_id"], df["target_id"]))
        df.loc[:, "label"] = ["True" if pair in ground_truth_pairs else "False" for pair in tf_tg_tuples]
        return df

    inferred_network_dd = inferred_network_dd.map_partitions(
        label_partition,
        meta=inferred_network_dd._meta.assign(label=np.int64(0))
    )

    return inferred_network_dd


tf_tg_edges_save_file = os.path.join(TMP_DIR, "tf_tg_edges.parquet")
if not os.path.isfile(tf_tg_edges_save_file):
    ground_truth_df = read_ground_truth(ground_truth_file)

    tf_peak_tg_edges: dd.DataFrame = build_tf_peak_tg_edge_dataframe()

    print('\nBuilding TF to TG edges with peak counts')
    tf_tg_edges = (
        tf_peak_tg_edges
        .groupby(["source_id", "target_id"])["peak_id"]
        .count()
        .reset_index()
        .rename(columns={"peak_id": "edge_count"})
    )

    print("\nLabeling ground truth TF to TG edges")
    tf_tg_edges_in_gt: dd.DataFrame = label_edges_with_ground_truth(tf_tg_edges, ground_truth_df)

    print(tf_tg_edges_in_gt.head())
    
    print("\nSaving labeled peak counts for each TF-TG edge to parquet file")
    tf_tg_edges_in_gt.to_parquet(tf_tg_edges_save_file, engine="pyarrow", compression="snappy")
else:
    print("Loading labeled TF-TG peak count parquet file")
    tf_tg_edges_in_gt = dd.read_parquet(tf_tg_edges_save_file, engine="pyarrow")

        
plt.figure(figsize=(8,8))
sns.histplot(data=tf_tg_edges_in_gt, x="edge_count", hue="label", bins=50, log_scale=True, element="step", stat="count")
plt.title("Distribution of peak counts for TF-TG edges", fontsize=16)
plt.xlabel("Number of peaks", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.show()