import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output"
print(output_dir)

inferred_net_paths = {}
feature_set_filename = "inferred_network_w_string.csv"

for cell_type in os.listdir(output_dir):
    cell_type_path = os.path.join(output_dir, cell_type)
    print(f'\n----- {cell_type.upper()} -----')
    
    
    for sample in os.listdir(cell_type_path):
        sample_path = os.path.join(cell_type_path, sample)
        print(f'{sample}')
        
        for folder in os.listdir(sample_path):
            if folder == "inferred_grns":
                folder_path = os.path.join(sample_path, folder)
                
                for inferred_net in os.listdir(folder_path):
                    inferred_net_path = os.path.join(folder_path, inferred_net)
                    if os.path.isfile(inferred_net_path):
                        print(f'    |____{inferred_net}')
                        if inferred_net == feature_set_filename:
                            if not cell_type in inferred_net_paths:
                                inferred_net_paths[cell_type] = {}
                                
                            if not sample in inferred_net_paths[cell_type]:
                                inferred_net_paths[cell_type][sample] = inferred_net_path


print(f'\n===== Found {feature_set_filename} in: =====')             
dataframes = {}                   

for cell_type, sample_path_dict in inferred_net_paths.items():
    for sample_name, sample_grn_path in sample_path_dict.items():
        df = pd.read_csv(sample_grn_path, header=0, index_col=None, nrows=100000)
        feature_counts = df.count(numeric_only=True, axis=1).value_counts().sort_index(ascending=False)
        
        n_score_cols = len(df.select_dtypes(include=np.number).columns)
        feature_threshold = n_score_cols - 2
        df = df[df.count(numeric_only=True, axis=1) >= feature_threshold]
        print(f'\tNumber of rows with >= {feature_threshold}/{n_score_cols} feature columns {len(df)}')
        print(feature_counts.values)

def plot_non_nan_feature_scores(inferred_net_paths):
    cell_types = list(inferred_net_paths.keys())
    n_cells = len(cell_types)

    # One row of subplots, one per cell_type
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_cells,
        figsize=(6*n_cells, 5),
        squeeze=False  # ensures axes is 2D even if n_cells=1
    )

    for ax, cell_type in zip(axes[0], cell_types):
        sample_dict = inferred_net_paths[cell_type]
        
        # build a DataFrame of feature_counts for this cell_type
        all_counts = {}
        for sample_name, sample_path in sample_dict.items():
            df = pd.read_csv(sample_path, nrows=1000000)
            counts = (
                df
                .count(numeric_only=True, axis=1)
                .value_counts()
                .sort_index(ascending=False)
            )
            all_counts[sample_name] = counts
        
        feature_df = pd.DataFrame(all_counts).fillna(0)
        feature_df.index.name = "# non‑NaN features"
        feature_df.sort_index(ascending=True, inplace=True)
        
        # stacked bar chart
        feature_df.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            width=0.8,
        )
        # place legend below
        n_samples = len(sample_dict)
        ax.legend(
            feature_df.columns,
            title="Sample",
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=n_samples,
            fontsize='medium'
        )
        
        ax.set_title(f"{cell_type}", fontsize=14)
        ax.set_xlabel(feature_df.index.name, fontsize=14)
        ax.set_ylabel("Number of edges", fontsize=14)
        ax.tick_params(axis='x', rotation=0)
        # ax.set_ylim((0, 200000))

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # make room at bottom for legends
    plt.suptitle("Number of non‑NaN feature columns", y=1.02, fontsize=18)
    plt.show()

def combine_ground_truth_datasets():
    reference_net_dir="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS"
    
    ground_truths = [
        "RN117_ChIPSeq_PMID37486787_Human_K562.tsv",
        "RN204_ChIPSeq_ChIPAtlas_Human_Macrophages.tsv",
        "RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"
    ]
    
    gt_dfs = []
    for file in ground_truths:
        gt_dfs.append(
            pd.read_csv(
                f'{reference_net_dir}/{file}', 
                sep='\t', 
                quoting=csv.QUOTE_NONE, 
                on_bad_lines='skip', 
                header=0, 
                index_col=None,
                usecols=["Source", "Target"]
            )
        )
    
    merged_gt_df = pd.concat(gt_dfs)
    print(merged_gt_df)
    print(merged_gt_df.shape)