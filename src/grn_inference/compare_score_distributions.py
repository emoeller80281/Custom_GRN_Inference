import os
import math
import argparse
import logging

import dask.dataframe as dd
import dask.array as da
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: 
          - model_training_inferred_net: path to training Parquet
          - prediction_target_inferred_net: path to prediction Parquet
          - model_training_sample_name: sample name for training data
          - prediction_target_sample_name: sample name for prediction data
          - fig_dir: directory to save the plot
    """
    parser = argparse.ArgumentParser(description="Plot overlapping feature score histograms using Dask")
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
    return parser.parse_args()


def read_inferred_network(inferred_network_file: str) -> dd.DataFrame:
    """
    Loads the melted sparse inferred-network parquet (source_id, peak_id, target_id, score_type, score_value)
    and pivots it back to wide form with columns = score_type.
    WARNING: This pivot still materializes the wide DataFrame in memory.
    """
    melted_ddf = dd.read_parquet(inferred_network_file, engine="pyarrow")

    # Standardize IDs
    melted_ddf["source_id"] = melted_ddf["source_id"].str.upper()
    melted_ddf["target_id"] = melted_ddf["target_id"].str.upper()

    # 1) group on (source_id, peak_id, target_id, score_type) and take mean
    grouped = (
        melted_ddf
        .groupby(["source_id", "peak_id", "target_id", "score_type"])["score_value"]
        .mean()
        .reset_index()
    )

    # 2) pivot in pandas (collects into memory)
    pdf = grouped.compute()
    wide = pdf.pivot_table(
        index=["source_id", "peak_id", "target_id"],
        columns="score_type",
        values="score_value",
        aggfunc="first"
    ).reset_index()

    # 3) return as a single-partition Dask DataFrame
    return dd.from_pandas(wide, npartitions=1)


def compute_dask_histogram(ddf_col, bins=50, value_range=(0, 1)):
    """
    Given a Dask Series of numeric values (possibly with NaNs), drop NaNs,
    convert to a Dask Array, and compute a histogram with `bins` bins over `value_range`.
    Returns (counts, bin_edges) as numpy arrays.

    This avoids materializing the full column in memory.
    """
    # 1) Drop NaN
    no_na = ddf_col.dropna()

    # 2) Convert to dask array
    darr = no_na.to_dask_array(lengths=True)

    # 3) Compute histogram lazily
    counts_d, edges_d = da.histogram(darr, bins=bins, range=value_range)

    # 4) Materialize the small arrays
    counts, edges = da.compute(counts_d, edges_d)
    return counts, edges


def plot_feature_score_histograms_dask(
    features,
    ddf1,
    ddf2,
    label1_name,
    label2_name,
    fig_dir,
    bins=50,
    value_range=(0, 1)
):
    """
    For each feature in `features`, compute two histograms from two wide-form Dask DataFrames
    (ddf1, ddf2) without converting entire columns to pandas. Then plot them in a grid
    and save the figure to fig_dir.

    Assumes all feature columns exist in both ddf1 and ddf2, and values lie in `value_range`.
    """
    os.makedirs(fig_dir, exist_ok=True)
    logging.info("  Plotting feature score histograms (Dask-based)")

    ncols = 4
    nrows = math.ceil(len(features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flat

    for ax, feature in zip(axes_flat, features):
        logging.info(f"    Computing histogram for feature: '{feature}'")
        
        counts1, edges = compute_dask_histogram(ddf1[feature], bins=bins, value_range=value_range)
        counts2, _     = compute_dask_histogram(ddf2[feature], bins=bins, value_range=value_range)

        total1 = counts1.sum()
        total2 = counts2.sum()
        prop1  = counts1 / total1 if total1 > 0 else counts1 * 0
        prop2  = counts2 / total2 if total2 > 0 else counts2 * 0

        # compute bin centers and half‐width
        bin_centers = (edges[:-1] + edges[1:]) / 2
        bin_width   = edges[1] - edges[0]
        width       = bin_width

        # Plot as side-by-side bars
        ax.bar(
            bin_centers,
            prop1,
            width=width,
            alpha=0.6,
            edgecolor="#032b5f",
            label=f"{label1_name} (model)",
            color="#1682b1"
        )
        ax.bar(
            bin_centers,
            prop2,
            width=width,
            alpha=0.6,
            edgecolor="#b13301",
            label=f"{label2_name} (target)",
            color="#cb5f17"
        )

        ax.set_title(feature, fontsize=14)
        ax.set_xlabel("Score", fontsize=12)
        ax.set_ylabel("Proportion", fontsize=12)
        ax.set_xlim(value_range)
        ax.tick_params(axis="both", labelsize=12)

    # Turn off any leftover empty subplots
    for ax in axes_flat[len(features):]:
        ax.set_visible(False)

    # One legend for the entire figure
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=2,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.02)
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    outpath = os.path.join(fig_dir, "overlapping_feature_score_histograms.png")
    plt.savefig(outpath, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    # Read & pivot training data
    logging.info("  Reading the feature scores used to train the model")
    ddf_train = read_inferred_network(args.model_training_inferred_net)
    logging.info("    Done!")

    # Read & pivot prediction data
    logging.info("  Reading the feature scores used for predictions")
    ddf_pred = read_inferred_network(args.prediction_target_inferred_net)
    logging.info("    Done!")

    # List of features to plot (must exist as columns in both wide Dask DataFrames)
    feature_names = [
        'mean_TF_expression',
        'mean_peak_accessibility',
        'mean_TG_expression',
        'cicero_score',
        'correlation',
        'TSS_dist_score',
        'homer_binding_score',
        'sliding_window_score',
        'string_combined_score',
        'string_experimental_score',
        'string_textmining_score'
    ]

    # Plot and save using the Dask-based histogram approach
    plot_feature_score_histograms_dask(
        feature_names,
        ddf_train,
        ddf_pred,
        args.model_training_sample_name,
        args.prediction_target_sample_name,
        args.fig_dir
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main()
