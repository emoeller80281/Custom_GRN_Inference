import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import dask.dataframe as dd
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing paths for input and output files and CPU count.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_set_filename",
        type=str,
        required=True,
        help="Name of the feature set to combine"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory"
    )
    parser.add_argument(
        "--combined_dataframe_dir",
        type=str,
        required=True,
        help="Path to the output directory for the final combined dataframe"
    )

    args: argparse.Namespace = parser.parse_args()
    return args

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
            df = pd.read_csv(sample_path, nrows=1000)
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
    logging.info(merged_gt_df)
    logging.info(merged_gt_df.shape)

# args: argparse.Namespace = parse_args()

# feature_set_filename = args.feature_set_filename
# output_dir = args.output_dir
# combined_dataframe_dir = args.combined_dataframe_dir

feature_set_filename = "inferred_score_df.parquet"
output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output"
combined_dataframe_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/combined_inferred_dfs"
cell_types = ["mESC"]

logging.info("\n============ LOADING INFERRED SCORE FILES ============")
inferred_net_paths = {}
for cell_type in cell_types:
    if cell_type in os.listdir(output_dir):
        cell_type_path = os.path.join(output_dir, cell_type)
        logging.info(f'\n  ----- {cell_type.upper()} -----')
        
        
        for sample in os.listdir(cell_type_path):
            sample_path = os.path.join(cell_type_path, sample)
            
            for folder in os.listdir(sample_path):
                if folder == "inferred_grns":
                    folder_path = os.path.join(sample_path, folder)
                    
                    for inferred_net in os.listdir(folder_path):
                        inferred_net_path = os.path.join(folder_path, inferred_net)
                        if os.path.exists(inferred_net_path):
                            if inferred_net == feature_set_filename:
                                logging.info(f'\t- Found combined score dataframe for {sample}')
                                if not cell_type in inferred_net_paths:
                                    inferred_net_paths[cell_type] = {}
                                    
                                if not sample in inferred_net_paths[cell_type]:
                                    inferred_net_paths[cell_type][sample] = inferred_net_path

       
logging.info("\n============ COMBINING INFERRED SCORES ============")

# Combine the sampled Dask DataFrames
sampled_enriched_feature_dfs = []
total_samples = sum(len(samples) for samples in inferred_net_paths.values())
sample_fraction = 1
logging.info(f"\t- Sample fraction set to: {sample_fraction*100:.4f} based on {total_samples} inferred networks.")

for cell_type, sample_path_dict in inferred_net_paths.items():
    logging.info(f"\n- Processing cell type: {cell_type} with {len(sample_path_dict)} sample(s).")
    for sample_name, sample_grn_path in sample_path_dict.items():
        try:
            df = dd.read_parquet(sample_grn_path)

            # Sample the pivoted DataFrame
            df_sample = df.sample(frac=sample_fraction, random_state=42)
            
            sample_row_count = df_sample.shape[0].compute()
            logging.info(f"\t- Sampled {sample_row_count:,} rows ({sample_fraction*100:.2f}%) from {sample_name}")

            sampled_enriched_feature_dfs.append(df_sample)
        except Exception as e:
            logging.error(f"\t- [ERROR] Failed to process {sample_grn_path}: {e}")

logging.info(f"\nCombining {len(sampled_enriched_feature_dfs)} sampled Dask DataFrames.")
combined_ddf = dd.concat(sampled_enriched_feature_dfs)
logging.info(f'\t- Number of unique TFs: {combined_ddf["source_id"].nunique().compute():,}')
logging.info(f'\t- Number of unique Peaks: {combined_ddf["peak_id"].nunique().compute():,}')
logging.info(f'\t- Number of unique TGs: {combined_ddf["target_id"].nunique().compute():,}')

output_path = os.path.join(combined_dataframe_dir, 'combined_enrich_feat_w_string.parquet')
logging.info(f"\nWriting combined DataFrame to {output_path}")

try:
    combined_ddf.to_parquet(
        output_path,
        compression='snappy',
        engine='pyarrow'
    )
    logging.info("Successfully wrote combined Dask DataFrame to disk.")
except Exception as e:
    logging.error(f"Failed to write Parquet output: {e}")