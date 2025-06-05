import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask.distributed import Client
import xgboost as xgb
import argparse
import logging
import os
import csv

from grn_inference.plotting import (
    plot_feature_score_histograms,
    plot_feature_importance,
    plot_feature_boxplots,
    plot_xgboost_prediction_histogram,
    plot_overlapping_roc_pr_curves,
    plot_permutation_importance_plot,
    plot_feature_ablation,
    plot_stability_boxplot,
)
from grn_inference.model import (
    train_xgboost_dask,
    parameter_grid_search,
)


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing paths for input and output files.
    """
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")

    parser.add_argument(
        "--ground_truth_file",
        type=str,
        required=True,
        help="Path to the ChIPseq ground truth file, formatted as 'source_id'\\t'target_id'"
    )
    parser.add_argument(
        "--inferred_network_file",
        type=str,
        required=True,
        help="Path to the inferred network file for the sample"
    )
    parser.add_argument(
        "--trained_model_dir",
        type=str,
        required=True,
        help="Path to the output directory for the trained XGBoost model"
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        required=True,
        help="Path to the figure directory for the sample"
    )
    parser.add_argument(
        "--model_save_name",
        type=str,
        required=True,
        help="Name of the output .pkl file for the trained model"
    )
    parser.add_argument(
        "--num_cpu",
        type=int,
        required=True,
        help="Number of CPUs allocated to the job"
    )
    
    args: argparse.Namespace = parser.parse_args()
    return args

def read_inferred_network(inferred_network_file: str) -> dd.DataFrame:
    """
    Loads a melted sparse inferred network from Parquet and pivots it into a Dask DataFrame
    where each row is (source_id, target_id) and columns are score_types (mean-aggregated).
    """
    logging.info(f"Loading melted sparse network from: {inferred_network_file}")
    melted_ddf = dd.read_parquet(inferred_network_file, engine="pyarrow")

    # Standardize IDs
    melted_ddf["source_id"] = melted_ddf["source_id"].str.upper()
    melted_ddf["target_id"] = melted_ddf["target_id"].str.upper()

    # Aggregate scores
    grouped_ddf = (
        melted_ddf
        .groupby(["source_id", "peak_id", "target_id", "score_type"])["score_value"]
        .mean()
        .reset_index()
    )

    # Pivot manually by converting to pandas (if dataset is small enough)
    def pivot_partition(df):
        return df.pivot_table(
            index=["source_id", "peak_id", "target_id"],
            columns="score_type",
            values="score_value",
            aggfunc="first"
        ).reset_index()

    # Apply pivot in a single partition (best if you've already aggregated)
    pivot_df = grouped_ddf.compute()  # convert to Pandas here
    pivot_df = pivot_partition(pivot_df)
    return dd.from_pandas(pivot_df, npartitions=1)

def read_ground_truth(ground_truth_file):
    logging.info("Reading in the ground truth")
    ground_truth = pd.read_csv(ground_truth_file, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    ground_truth = ground_truth.rename(columns={"Source": "source_id", "Target": "target_id"})
    return ground_truth

def read_merged_ground_truth(merged_ground_truth_file):
    merged_ground_truth = pd.read_csv(merged_ground_truth_file, sep='\t', header=0)
    logging.info(merged_ground_truth)
    return merged_ground_truth

def label_edges_with_ground_truth(inferred_network_dd, ground_truth_df):
    logging.info("Creating ground truth set")
    ground_truth_pairs = set(zip(
        ground_truth_df["source_id"].str.upper(),
        ground_truth_df["target_id"].str.upper()
    ))

    logging.info("Adding labels to inferred network")

    def label_partition(df):
        df = df.copy()  # <-- avoids SettingWithCopyWarning
        tf_tg_tuples = list(zip(df["source_id"], df["target_id"]))
        df.loc[:, "label"] = [1 if pair in ground_truth_pairs else 0 for pair in tf_tg_tuples]
        return df

    inferred_network_dd = inferred_network_dd.map_partitions(
        label_partition,
        meta=inferred_network_dd._meta.assign(label=np.int64(0))
    )

    return inferred_network_dd
    

def main():
    args = parse_args()

    ground_truth_file: str = args.ground_truth_file
    inferred_network_file: str = args.inferred_network_file
    trained_model_dir: str = args.trained_model_dir
    fig_dir: str = args.fig_dir
    model_save_name: str = args.model_save_name
    num_cpu: int = int(args.num_cpu)

    inferred_network_dd = read_inferred_network(inferred_network_file)
    ground_truth_df = read_ground_truth(ground_truth_file)

    inferred_network_dd = label_edges_with_ground_truth(inferred_network_dd, ground_truth_df)

    # Drop unnecessary columns
    # drop_cols = ["source_id", "peak_id", "target_id", "label", "correlation", "cicero_score"]
    # feature_names = [col for col in inferred_network_dd.columns if col not in drop_cols]
    feature_names = [
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


    # Only keep columns needed for modeling
    logging.info(f"Keeping {len(feature_names)} feature columns + labels")
    model_dd = inferred_network_dd[feature_names + ["label"]].persist()

    logging.info(f"Splitting {model_dd.shape[0].compute():,} rows into train/test with stratification")

    # Dask-ML's split works directly on Dask DataFrames
    X_dd = model_dd[feature_names]
    y_dd = model_dd["label"]
    
    label_dist = y_dd.value_counts().compute()
    logging.info(f"Label distribution: {label_dist.to_dict()}")

    X_train_dd, X_test_dd, y_train_dd, y_test_dd = train_test_split(
        X_dd,
        y_dd,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )
    
    model_save_path = os.path.join(trained_model_dir, f"{model_save_name}.json")

    logging.info(f"Done splitting: {X_train_dd.shape[0].compute():,} train / {X_test_dd.shape[0].compute():,} test rows")
    logging.info(f"\tSaving Train / Test splits to disk in {trained_model_dir}")
    train_test_dir = os.path.join(trained_model_dir, f"train_test_splits/{model_save_name}")
    os.makedirs(train_test_dir, exist_ok=True)
    
    X_train_dd.to_parquet(os.path.join(train_test_dir, "X_train.parquet"))
    X_test_dd.to_parquet(os.path.join(train_test_dir, "X_test.parquet"))
    y_train_dd.to_frame(name="label").to_parquet(os.path.join(train_test_dir, "y_train.parquet"))
    y_test_dd.to_frame(name="label").to_parquet(os.path.join(train_test_dir, "y_test.parquet"))
    
    logging.info("\n")
    xgb_booster = train_xgboost_dask(train_test_dir, feature_names)

    # Save the feature names
    xgb_booster.set_attr(feature_names=",".join(feature_names))
    
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)

    
    xgb_booster.save_model(model_save_path)
    logging.info(f"Saved trained XGBoost booster to {model_save_path}")

    importance_dict = xgb_booster.get_score(importance_type="weight")
    feature_importances = pd.DataFrame({
        "Feature": list(importance_dict.keys()),
        "Importance": list(importance_dict.values())
    })
    feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

    logging.info("\n----- Plotting Figures -----")

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    X_train = X_train_dd.compute()
    X_test = X_test_dd.compute()
    
    y_train = y_train_dd.compute()
    y_test = y_test_dd.compute()
    model_df = model_dd.compute()
    
    plot_feature_importance(feature_names, xgb_booster, fig_dir)
    plot_feature_score_histograms(model_df, feature_names, fig_dir)
    plot_feature_boxplots(feature_names, model_df, fig_dir)
    plot_xgboost_prediction_histogram(xgb_booster, X_test, fig_dir)
    
    # prarm_grid_out_dir = train_test_dir = os.path.join(trained_model_dir, f"parameter_grid_search/{model_save_name}")
    
    # # Run the parameter grid search
    # logging.info("Starting parameter grid search")
    # grid = parameter_grid_search(
    #     X_train, 
    #     y_train, 
    #     X_test,
    #     y_test,
    #     prarm_grid_out_dir, 
    #     cpu_count=num_cpu, 
    #     fig_dir=fig_dir
    #     )

    # logging.info("Plotting grid search best estimator feature importances")
    # plot_feature_importance(
    #     features=feature_names,
    #     model=grid,
    #     fig_dir=os.path.join(fig_dir, "parameter_search")
    # )

    # --- Note: The following plots take a long time to run for large models, as they test re-training the model ---

    # plot_overlapping_roc_pr_curves(X, y, feature_names, fig_dir)
    # plot_permutation_importance_plot(xgb_booster, X_test, y_test, fig_dir)
    # plot_feature_ablation(feature_names, X_train, X_test, y_train, y_test, xgb_booster, fig_dir)
    # plot_stability_boxplot(X, y, fig_dir)
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
