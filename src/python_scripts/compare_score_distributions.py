import pandas as pd
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import argparse
import logging

def parse_args() -> argparse.Namespace:
    """
    Parses command‐line arguments.

    Returns:
        argparse.Namespace: 
        - model_training_inferred_net: path to training Parquet
        - prediction_target_inferred_net: path to prediction Parquet
        - model_training_sample_name: sample name for training data
        - prediction_target_sample_name: sample name for prediction data
        - fig_dir: directory to save the plot
    """

    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument(
        "--model_training_inferred_net",
        type=str,
        required=True,
        help="Path to the inferred network feature score file used to train the model"
    )
    parser.add_argument(
        "--prediction_target_inferred_net",
        type=str,
        required=True,
        help="Path to the inferred network feature score file to run model predictions on"
    )
    parser.add_argument(
        "--model_training_sample_name",
        type=str,
        required=True,
        help="Name of the sample used to train the model"
    )
    parser.add_argument(
        "--prediction_target_sample_name",
        type=str,
        required=True,
        help="Name of the sample the predictions are being run on"
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        required=True,
        help="Directory to save the overlapping feature score histogram figure to"
    )
    
    args: argparse.Namespace = parser.parse_args()

    return args

def read_inferred_network(inferred_network_file: str) -> dd.DataFrame:
    """
    Loads the melted sparse inferred-network parquet (source_id, peak_id, target_id, score_type, score_value)
    and pivots it back to wide form *including* peak_id in the index.
    """
    melted_ddf = dd.read_parquet(inferred_network_file, engine="pyarrow")

    # Standardize IDs
    melted_ddf["source_id"] = melted_ddf["source_id"].str.upper()
    melted_ddf["target_id"] = melted_ddf["target_id"].str.upper()
    # peak_id probably doesn't need uppercasing but you could if you like:
    # melted_ddf["peak_id"]   = melted_ddf["peak_id"].str.upper()

    # 1) group on THREE id-columns + score_type
    grouped = (
        melted_ddf
        .groupby(["source_id", "peak_id", "target_id", "score_type"])
        ["score_value"]
        .mean()
        .reset_index()
    )

    # 2) pivot in pandas (safe since it's already aggregated)
    pdf = grouped.compute()
    wide = pdf.pivot_table(
        index=["source_id", "peak_id", "target_id"],
        columns="score_type",
        values="score_value",
        aggfunc="first"       # now that each is unique per id-triple
    ).reset_index()

    # 3) back to Dask if you want
    return wide

def plot_feature_score_histograms(
    features,
    inferred_network1,
    inferred_network2,
    label1_name,
    label2_name,
    fig_dir
):
    """
    Plot overlapping histograms of features from two Dask/Pandas DataFrames,
    saving a PNG under fig_dir/overlapping_feature_score_histograms.png.
    """
    logging.info("\t- Plotting feature score histograms")
    
    # materialize only needed columns
    if isinstance(inferred_network1, dd.DataFrame):
        logging.info("\t- Converting feature columns from Dask to pandas for plotting")
        inferred_network1 = inferred_network1[features].compute()
    if isinstance(inferred_network2, dd.DataFrame):
        inferred_network2 = inferred_network2[features].compute()

    ncols = 4
    nrows = math.ceil(len(features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    # flatten axes for easy indexing
    axes_flat = axes.flat

    for ax, feature in zip(axes_flat, features):
        # draw into this axis explicitly:
        sns.histplot(
            inferred_network1[feature].dropna().sample(frac=0.1, random_state=0),
            bins=50, alpha=0.7,
            color='#1682b1', edgecolor="#032b5f",
            stat='proportion',
            label=f'{label1_name} (model)',
            ax=ax
        )
        sns.histplot(
            inferred_network2[feature].dropna().sample(frac=0.1, random_state=0),
            bins=50, alpha=0.7,
            color="#cb5f17", edgecolor="#b13301",
            stat='proportion',
            label=f'{label2_name} (target)',
            ax=ax
        )

        # set titles/labels on the same ax
        ax.set_title(feature, fontsize=14)
        ax.set_xlabel(feature, fontsize=14)
        ax.set_ylabel("Proportion", fontsize=14)
        ax.set_xlim(0, 1)
        ax.tick_params(axis='both', labelsize=12)

    # turn off any leftover empty subplots
    for ax in axes_flat[len(features):]:
        ax.set_visible(False)

    # figure-level legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        fontsize=14,
        bbox_to_anchor=(0.5, 0.02)
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(fig_dir, "overlapping_feature_score_histograms.png"), dpi=200)
    plt.close(fig)

def main():
    args = parse_args()

    # inferred_network1_file: str = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/combined_inferred_dfs/mESC_combined_inferred_score_df.parquet"
    model_training_inferred_net_path: str = args.model_training_inferred_net
    prediction_target_inferred_net_path: str = args.prediction_target_inferred_net

    model_training_sample_name: str = args.model_training_sample_name
    prediction_target_sample_name: str = args.prediction_target_sample_name
    
    fig_dir: str = args.fig_dir

    logging.info("----- Plotting overlapping feature score histograms -----")
    logging.info("  Reading the feature scores used to train the model")
    model_training_inferred_net = read_inferred_network(model_training_inferred_net_path)
    logging.info("    Done!")

    logging.info('  Reading the feature scores used for predictions')
    prediction_target_inferred_net = read_inferred_network(prediction_target_inferred_net_path)
    logging.info('    Done!')

    all_cols = set(model_training_inferred_net.columns) - {'source_id','peak_id','target_id'}
    feature_names = sorted(list(all_cols))

    logging.info('  Plotting overlapping feature score histograms')
    plot_feature_score_histograms(
        feature_names, 
        model_training_inferred_net, 
        prediction_target_inferred_net, 
        model_training_sample_name, 
        prediction_target_sample_name,
        fig_dir
        )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()