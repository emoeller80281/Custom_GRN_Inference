import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import dask.dataframe as dd
import logging
from typing import Union

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

def combine_ground_truth_datasets(
    reference_net_dir: str, 
    cell_type: str,
    ground_truths: list[str], 
    save_dir: str,
    excluded_files: Union[list[str], None] = None
    ) -> None:
    
    logging.info(f'============ COMBINING GROUND TRUTH FILES FOR {cell_type.upper()} ============')
    
    if excluded_files:
        logging.info(f'NOTE: Excluding {excluded_files} for testing the trained XGBoost model on a naive ground truth\n')
    
    if os.path.isdir(reference_net_dir):
        
        gt_dfs = []
        # Read in the ground truth files
        for file in ground_truths:
            
            # Exclude any ground truth files that are in the testing_
            if not file in excluded_files:
                file_path = os.path.join(reference_net_dir, file)
                if os.path.isfile(file_path):
                    logging.info(f'\t- Loading "{file}"')
                    gt_dfs.append(
                        pd.read_csv(
                            filepath_or_buffer=file_path, 
                            sep='\t', 
                            quoting=csv.QUOTE_NONE, 
                            on_bad_lines='skip', 
                            header=0, 
                            index_col=None,
                            usecols=["Source", "Target"]
                        )
                    )
                else:
                    logging.warning(f'WARNING: "{file_path}" is not a file')
            else:
                logging.info(f'\t- Not combining excluded file {file}')
    else:
        logging.error(f'ERROR: {reference_net_dir} is not a directory')
    
    # Combine the individual ground truth files together
    logging.info(f'\nCombining ground truth networks')
    merged_gt_df = pd.concat(gt_dfs)
    
    combined_gt_path = os.path.join(save_dir, f'combined_{cell_type}_ground_truth.tsv')
    
    logging.info(f'\nSaving combined ground truth file to "{combined_gt_path}"')
    # Write the merged ground truth dataframes to the reference network directory
    merged_gt_df.to_csv(
        combined_gt_path, 
        sep='\t', 
        header=True, 
        index=False
        )

def locate_inferred_score_files(
    inferred_score_filename: str, 
    output_dir: str, 
    cell_type: str,
    excluded_samples: Union[None, list[str]] = None
) -> dict[list]:
    logging.info(f"\n============ LOADING INFERRED SCORE FILES FOR {cell_type.upper()} ============")    
    cell_type_paths = {}
    if cell_type in os.listdir(output_dir):
        cell_type_path = os.path.join(output_dir, cell_type)        
        
        for sample in os.listdir(cell_type_path):
                
            sample_path = os.path.join(cell_type_path, sample)
            
            for folder in os.listdir(sample_path):
                if folder == "inferred_grns":
                    folder_path = os.path.join(sample_path, folder)
                    
                    for inferred_net in os.listdir(folder_path):
                        inferred_net_path = os.path.join(folder_path, inferred_net)
                        if os.path.isfile(inferred_net_path):
                            if inferred_net == inferred_score_filename:
                                logging.info(f'\t- Found combined score dataframe for {sample}')

                                if not sample in cell_type_paths:
                                    cell_type_paths[sample] = inferred_net_path
    
    if excluded_samples:
        logging.info("\nRemoving excluded samples: ")
        for excluded_sample_name in excluded_samples:
            if excluded_sample_name in cell_type_paths.keys():
                removed_sample = cell_type_paths.pop(excluded_sample_name)
                
                logging.info(f'\t- {removed_sample}')
    
    return cell_type_paths
    
def combine_inferred_score_files(
    cell_type_paths: dict[list],
    cell_type: str,
    excluded_inferred_score_files: Union[list[str], None] = None
    ) -> dd.DataFrame:
    
    logging.info("\n============ COMBINING INFERRED SCORES ============")
    sample_fraction = 1
    logging.info(f"\nSampling {sample_fraction*100:.0f}% of each inferred network.")
    
    if excluded_inferred_score_files:
        for excluded_file in excluded_inferred_score_files:
            logging.info(f'\nExcluding sample "{excluded_file}"')
            cell_type_paths.pop(excluded_file)
    
    # Combine the sampled Dask DataFrames
    sampled_enriched_feature_dfs = []

    logging.info(f"\nLoading inferred score DataFrames for cell type {cell_type} with {len(cell_type_paths)} sample(s).")
    
    # Read in and sample the sample parquet files
    for sample_name, sample_grn_path in cell_type_paths.items():
        try:
            df = dd.read_parquet(sample_grn_path)

            # Sample the pivoted DataFrame
            df_sample = df.sample(frac=sample_fraction, random_state=42)
            
            sample_row_count = df_sample.shape[0].compute()
            logging.info(f"\t- Sampled {sample_row_count:,} rows ({sample_fraction*100:.0f}%) from {sample_name}")

            sampled_enriched_feature_dfs.append(df_sample)

        except Exception as e:
            logging.error(f"\t- [ERROR] Failed to process {sample_grn_path}: {e}")

    # Combine the sampels into a single combined Dask DataFrame for the cell-type
    logging.info(f"\nCombining {len(sampled_enriched_feature_dfs)} DataFrames.")
    combined_ddf = dd.concat(sampled_enriched_feature_dfs)
    
    return combined_ddf

def write_combined_dataframe(combined_ddf: dd.DataFrame, cell_type: str, output_dir: str) -> None:
        combined_dataframe_dir = os.path.join(output_dir, "combined_inferred_dfs")
        os.makedirs(combined_dataframe_dir, exist_ok=True)

        output_path = os.path.join(combined_dataframe_dir, f'{cell_type}_combined_inferred_score_df.parquet')
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
            
        logging.info('\tDONE!')


def main():
    inferred_score_filename = "inferred_score_df.parquet"
    output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output"
    cell_types = ["mESC"]
    
    reference_net_dir="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS"
    
    ground_truth_files = {
        "mESC": ["RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv", "RN112_LOGOF_BEELINE_Mouse_ESC.tsv", "RN114_ChIPX_ESCAPE_Mouse_ESC.tsv", "RN115_LOGOF_ESCAPE_Mouse_ESC.tsv"],
        "K562": ["RN117_ChIPSeq_PMID37486787_Human_K562.tsv", "RN118_KO_KnockTF_Human_K562.tsv", "RN119_ChIPSeqandKO_PMID37486787andKnockTF_Human_K562.tsv"],
        "macrophage":["RN204_ChIPSeq_ChIPAtlas_Human_Macrophages.tsv"]
        }
    
    # Excluding this ground truth, keeping it separate for testing the model with a naive ground truth
    excluded_ground_truth_files = ["RN111_ChIPSeq_BEELINE_Mouse_ESC.tsv"]
    excluded_inferred_score_files = ["filtered_L2_E7.5_rep1"]
    
    ground_truth_save_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/ground_truth_files"
    
    for cell_type in cell_types:
        # Combine the ground truth files together
        combine_ground_truth_datasets(
            reference_net_dir,
            cell_type,
            ground_truth_files[cell_type],
            ground_truth_save_dir,
            excluded_ground_truth_files
        )
        
        # find the path to the inferred score files for each cell type
        cell_type_paths: dict[list] = locate_inferred_score_files(inferred_score_filename, output_dir, cell_type)
        
        # Combine the inferred score files together
        combined_ddf = combine_inferred_score_files(cell_type_paths, cell_type, excluded_inferred_score_files)
        
        logging.info(f'\t- Number of unique TFs: {combined_ddf["source_id"].nunique().compute():,}')
        logging.info(f'\t- Number of unique Peaks: {combined_ddf["peak_id"].nunique().compute():,}')
        logging.info(f'\t- Number of unique TGs: {combined_ddf["target_id"].nunique().compute():,}')
        
        # Write out the final combined score file
        write_combined_dataframe(combined_ddf, cell_type, output_dir)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()