import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask.distributed import Client
import xgboost as xgb
import argparse
import logging
import os

from plotting import (
    plot_feature_score_histograms,
    plot_feature_importance,
    plot_feature_boxplots,
    plot_xgboost_prediction_histogram,
    plot_overlapping_roc_pr_curves,
    plot_permutation_importance_plot,
    plot_feature_ablation,
    plot_stability_boxplot,
)
from model import (
    train_xgboost_dask,
    xgb_classifier_from_booster,
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
    
    args: argparse.Namespace = parser.parse_args()
    return args

def read_inferred_network(inferred_network_file):
    logging.info("Reading inferred network with Dask")
    inferred_network = dd.read_parquet(inferred_network_file)
    inferred_network["source_id"] = inferred_network["source_id"].str.upper()
    inferred_network["target_id"] = inferred_network["target_id"].str.upper()
    return inferred_network

def read_ground_truth(ground_truth_file):
    logging.info("Reading in the ground truth")
    ground_truth = pd.read_csv(ground_truth_file, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    ground_truth = ground_truth.rename(columns={"Source": "source_id", "Target": "target_id"})
    return ground_truth

def read_merged_ground_truth(merged_ground_truth_file):
    merged_ground_truth = pd.read_csv(merged_ground_truth_file, sep='\t', header=0)
    logging.info(merged_ground_truth)
    return merged_ground_truth

def main():
    args = parse_args()

    ground_truth_file: str = args.ground_truth_file
    inferred_network_file: str = args.inferred_network_file
    trained_model_dir: str = args.trained_model_dir
    fig_dir: str = args.fig_dir
    model_save_name: str = args.model_save_name

    inferred_network_dd = read_inferred_network(inferred_network_file)
    ground_truth_df = read_ground_truth(ground_truth_file)

    # Create set of (TF, TG) from ground truth
    logging.info("Creating ground truth set")
    ground_truth_pairs = set(zip(ground_truth_df["source_id"], ground_truth_df["target_id"]))

    logging.info("Adding labels to inferred network")
    inferred_network_dd = inferred_network_dd.map_partitions(
        lambda df: df.assign(label=df.apply(
            lambda row: 1 if (row["source_id"], row["target_id"]) in ground_truth_pairs else 0,
            axis=1
        )),
        meta=inferred_network_dd._meta.assign(label=np.int64(0))
    )

    # Drop unnecessary columns
    drop_cols = ["source_id", "peak_id", "target_id", "label"]
    feature_names = [col for col in inferred_network_dd.columns if col not in drop_cols]

    # Only keep columns needed for modeling
    logging.info(f"Keeping {len(feature_names)} feature columns + labels")
    model_dd = inferred_network_dd[feature_names + ["label"]]

    logging.info(f"Splitting {model_dd.shape[0].compute():,} rows into train/test with stratification")

    # Dask-ML's split works directly on Dask DataFrames
    X_dd = model_dd[feature_names]
    y_dd = model_dd["label"]

    X_train_dd, X_test_dd, y_train_dd, y_test_dd = train_test_split(
        X_dd,
        y_dd,
        test_size=0.2,
        shuffle=True,
        stratify=y_dd,   # Pure Dask stratification!
        random_state=42
    )

    logging.info(f"Done splitting: {X_train_dd.shape[0].compute():,} train / {X_test_dd.shape[0].compute():,} test rows")
    logging.info("Training XGBoost Model")
    xgb_booster = train_xgboost_dask(X_train_dd, y_train_dd, feature_names)

    # Save the feature names
    xgb_booster.set_attr(feature_names=",".join(feature_names))
    
    if not os.path.exists(trained_model_dir):
        os.makedirs(trained_model_dir)

    model_save_path = os.path.join(trained_model_dir, f"{model_save_name}.json")
    xgb_booster.save_model(model_save_path)
    logging.info(f"Saved trained XGBoost booster to {model_save_path}")

    importance_dict = xgb_booster.get_score(importance_type="weight")
    feature_importances = pd.DataFrame({
        "Feature": list(importance_dict.keys()),
        "Importance": list(importance_dict.values())
    })
    feature_importances = feature_importances.sort_values(by="Importance", ascending=False)

    logging.info("\n----- Plotting Figures -----")
    model_df = model_dd.compute()
    X = model_df[feature_names]
    y = model_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_cls = xgb_classifier_from_booster(xgb_booster, X.columns)

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    logging.info("\n----- Plotting Figures -----")
    plot_feature_score_histograms(feature_names, model_df, fig_dir)
    plot_feature_importance(feature_names, xgb_booster, fig_dir)
    plot_feature_boxplots(feature_names, model_df, fig_dir)
    plot_xgboost_prediction_histogram(xgb_booster, X_test, fig_dir)
    
    # --- Note: The following plots take a long time to run for large models, as they test re-training the model ---

    plot_overlapping_roc_pr_curves(X, y, feature_names, fig_dir)
    plot_permutation_importance_plot(model_cls, X_test, y_test, fig_dir)
    plot_feature_ablation(feature_names, X_train, X_test, y_train, y_test, model_cls, fig_dir)
    plot_stability_boxplot(X, y, fig_dir)
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
