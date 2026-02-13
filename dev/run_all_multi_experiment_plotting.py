import pandas as pd
import numpy as np
import json
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from matplotlib.lines import Line2D
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')


FIGURE_DIR = Path("/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/FIGURES")
EXPERIMENT_DATA_DIR = Path("/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments")

def format_path(experiment_name: str, model_training_number: int) -> Path:
    """
    Reads in the AUROC results from a given experiment and model training number.
    
    Uses a fixed base directory.
    
    Parameters
    -------------
    experiment_name : str
        Name of the experiment (e.g., "K562_test_experiment")
    model_training_number : int
        Model training number (e.g. 1 for "model_training_001")
        
    Returns
    -------------
    Path
        Path to the experiment directory
    """
    base_dir = EXPERIMENT_DATA_DIR
    exp_dir = base_dir / experiment_name / "chr19" / f"model_training_{model_training_number:03d}"
    if not exp_dir.exists():
        exp_dir = base_dir / experiment_name / f"model_training_{model_training_number:03d}"
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found for {experiment_name} and model training number {model_training_number}")
    
    return exp_dir

def read_exp_auroc_results(auroc_exp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads evaluation metric results from a given experiment directory.
    
    Parameters
    -------------
    auroc_exp_dir : Path
        Path to the experiment directory containing AUROC results.
        
    Returns
    -------------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple containing:
        - raw_results_df: DataFrame with raw AUROC and AUPRC results.
        - results_df: DataFrame with pooled AUROC and AUPRC results.
        - per_tf_all_df: DataFrame with per-TF AUROC and AUPRC results.
        - per_tf_summary_df: DataFrame with summary statistics of per-TF AUROC and AUPRC results.
    """
    files = ["pooled_auroc_auprc_raw_results.csv", "pooled_auroc_auprc_results.csv", "per_tf_auroc_auprc_results.csv", "per_tf_auroc_auprc_summary.csv"]
    
    if not all((auroc_exp_dir / file).exists() for file in files):
        missing_files = [file for file in files if not (auroc_exp_dir / file).exists()]
        logging.info(f"Missing AUROC result files in {auroc_exp_dir}: {', '.join(missing_files)}")
        return None, None, None, None
    
    raw_results_df = pd.read_csv(auroc_exp_dir / "pooled_auroc_auprc_raw_results.csv")
    results_df = pd.read_csv(auroc_exp_dir / "pooled_auroc_auprc_results.csv")
    per_tf_all_df = pd.read_csv(auroc_exp_dir / "per_tf_auroc_auprc_results.csv")
    per_tf_summary_df = pd.read_csv(auroc_exp_dir / "per_tf_auroc_auprc_summary.csv")
    
    return raw_results_df, results_df, per_tf_all_df, per_tf_summary_df

def load_and_combine_experiment_results(experiment_list: list[tuple[str, int]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads and combines AUROC results from multiple experiments.
    
    Parameters
    -------------
    experiment_list : list[tuple[str, int]]
        List of tuples containing experiment names and model training numbers.
        
    Returns
    -------------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple containing combined results DataFrames:
        - combined_raw_results_df: DataFrame with raw pooled AUROC and AUPRC results.
        - combined_results_df: DataFrame with pooled AUROC and AUPRC results.
        - combined_per_tf_all_df: DataFrame with per-TF AUROC and AUPRC results.
        - combined_per_tf_summary_df: DataFrame with summary statistics of per-TF AUROC and AUPRC results.
    """
    combined_experiment_list = []
    for experiment_name, model_training_number in experiment_list:
        auroc_exp_dir = format_path(experiment_name, model_training_number)
        
        raw_results_df, results_df, per_tf_all_df, per_tf_summary_df = read_exp_auroc_results(auroc_exp_dir)
        
        # Skip experiments with missing results
        if raw_results_df is None:
            continue
        
        raw_results_df["experiment"] = experiment_name
        results_df["experiment"] = experiment_name
        per_tf_all_df["experiment"] = experiment_name
        per_tf_summary_df["experiment"] = experiment_name
        
        combined_experiment_list.append((raw_results_df, results_df, per_tf_all_df, per_tf_summary_df))

    combined_raw_results_df = pd.concat([x[0] for x in combined_experiment_list], ignore_index=True)
    combined_results_df = pd.concat([x[1] for x in combined_experiment_list], ignore_index=True)
    combined_per_tf_all_df = pd.concat([x[2] for x in combined_experiment_list], ignore_index=True)
    combined_per_tf_summary_df = pd.concat([x[3] for x in combined_experiment_list], ignore_index=True)
    
    return combined_raw_results_df, combined_results_df, combined_per_tf_all_df, combined_per_tf_summary_df

def plot_all_results_auroc_boxplot(
    df: pd.DataFrame, 
    per_tf: bool = False, 
    override_color: bool = False,
    ylim: tuple = (0.3, 0.7),
    sort_by_mean: bool = True,
    ) -> plt.Figure:
    """
    Plots AUROC boxplots for all GRN inference methods in the provided DataFrame.
    
    Parameters
    -------------
    df : pd.DataFrame
        DataFrame containing AUROC results with columns 'method' and 'auroc'.
    per_tf : bool, optional
        If True, indicates that the DataFrame contains per-TF AUROC scores. Default is False.
    override_color : bool, optional
        If True, overrides the default coloring scheme for methods to plot all boxes as blue. Default is False.
    """
    # 1. Order methods by mean AUROC (highest → lowest)
    if sort_by_mean:
        method_order = (
            df.groupby("method")["auroc"]
            .mean()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
    else:
        method_order = (
            df.groupby("method")["auroc"]
            .mean()
            .index
            .tolist()
        )

    if "No Filtering" in method_order:
        method_order = [m for m in method_order if m != "No Filtering"] + ["No Filtering"]
    
    mean_by_method = (
        df.groupby("method")["auroc"]
        .mean()
    )
    
    # 2. Prepare data in that order
    data = [df.loc[df["method"] == m, "auroc"].values for m in method_order]

    feature_list = [
        "Gradient Attribution",
        "TF Knockout",
    ]
    my_color = "#4195df"
    other_color = "#747474"

    fig, ax = plt.subplots(figsize=(9, 5))

    # Baseline random line
    ax.axhline(y=0.5, color="#2D2D2D", linestyle='--', linewidth=1)

    # --- Boxplot (existing styling) ---
    bp = ax.boxplot(
        data,
        tick_labels=method_order,
        patch_artist=True,
        showfliers=False
    )

    # Color boxes: light blue for your methods, grey for others
    for box, method in zip(bp["boxes"], method_order):
        if method in feature_list or override_color:
            box.set_facecolor(my_color)
        else:
            box.set_facecolor(other_color)

    # Medians in black
    for median in bp["medians"]:
        median.set_color("black")

    # --- NEW: overlay jittered points for each method ---
    for i, method in enumerate(method_order, start=1):
        y = df.loc[df["method"] == method, "auroc"].values
        if len(y) == 0:
            continue

        # Small horizontal jitter around the box center (position i)
        x = np.random.normal(loc=i, scale=0.06, size=len(y))

        # Match point color to box color
        point_color = my_color if method in feature_list or override_color else other_color

        ax.scatter(
            x, y,
            color=point_color,
            alpha=0.7,
            s=18,
            edgecolor="k",
            linewidth=0.3,
            zorder=3,
        )
        
        mean_val = y.mean()
        ax.scatter(
            i, mean_val,
            color="white",
            edgecolor="k",
            s=30,
            zorder=4,
        )

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=(
                my_color if (method in feature_list or override_color) else other_color
            ),
            markeredgecolor="k",
            markersize=7,
            label=f"{method}: {mean_by_method.loc[method]:.3f}"
        )
        for method in method_order
    ]
    
    ax.legend(
        handles=legend_handles,
        title="Mean AUROC",
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        borderaxespad=0.0,
        ncol=1,
    )

    ax.set_ylabel("AUROC across ground truths")
    if per_tf == True:
        ax.set_title("per-TF AUROC Scores per method")
        ax.set_ylim(ylim)
    else:
        ax.set_title("AUROC Scores per method")
        ax.set_ylim(ylim)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    
    return fig

def plot_per_tf_auroc_agg_by_gt(selected_experiment: str, combined_per_tf_all_df: pd.DataFrame) -> plt.Figure:
    selected_per_tf_df = combined_per_tf_all_df.loc[combined_per_tf_all_df["experiment"] == selected_experiment].copy()

    # Average the per-TF AUROC scores across ground truths for each method
    per_tf_mean_across_gt = (
        selected_per_tf_df
        .dropna(subset=["auroc"])
        .groupby(["method", "gt"], as_index=False)
        .agg(
            auroc=("auroc", "mean"),
            n_gt=("gt", "nunique"),
        )
    )

    fig = plot_all_results_auroc_boxplot(
        per_tf_mean_across_gt, 
        per_tf=True,
        ylim=(0.2, 0.8)
        )
    
    exp_dir = FIGURE_DIR / selected_experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(exp_dir / "per_tf_auroc_agg_by_gt.svg", bbox_inches="tight")
    plt.close(fig)

def plot_per_tf_auroc_agg_by_tf(selected_experiment: str, combined_per_tf_all_df: pd.DataFrame) -> plt.Figure:
    selected_per_tf_df = combined_per_tf_all_df.loc[combined_per_tf_all_df["experiment"] == selected_experiment].copy()

    per_tf_mean_across_tfs = (
        selected_per_tf_df.dropna(subset=["auroc"])
        .groupby(['method', 'tf'], as_index=False)
        .agg(
            auroc=('auroc', 'mean'),
            n_gt=('gt', 'nunique'),
        )
    )
    fig = plot_all_results_auroc_boxplot(
        per_tf_mean_across_tfs, 
        per_tf=True,
        ylim=(0.2, 0.8)
        )
    
    exp_dir = FIGURE_DIR / selected_experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(exp_dir / "per_tf_auroc_agg_by_tf.svg", bbox_inches="tight")
    plt.close(fig)
    
def plot_pooled_auroc(selected_experiment: str, combined_results_df: pd.DataFrame) -> plt.Figure:
    selected_results_df = combined_results_df.loc[combined_results_df["experiment"] == selected_experiment].copy()
    
    fig = plot_all_results_auroc_boxplot(
        selected_results_df, 
        per_tf=False, 
        ylim=(0.2, 0.8)
        )
    
    exp_dir = FIGURE_DIR / selected_experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(exp_dir / "pooled_auroc.svg", bbox_inches="tight")
    plt.close(fig)

def plot_multiexperiment_per_tf_auroc_agg_by_gt(name: str, selected_method: str, combined_per_tf_all_df: pd.DataFrame) -> plt.Figure:
    selected_method_safe = selected_method.lower().replace(" ", "_")
    
    per_tf_mean_across_gt = (
        combined_per_tf_all_df
        .dropna(subset=["auroc"])
        .groupby(["method", "gt"], as_index=False)
        .agg(
            auroc=("auroc", "mean"),
            n_gt=("gt", "nunique"),
        )
    )
    fig = plot_all_results_auroc_boxplot(
        per_tf_mean_across_gt, 
        per_tf=True, 
        override_color=True, 
        sort_by_mean=False,
        ylim=(0.2, 0.8)
        )
    
    exp_dir = FIGURE_DIR / "multi_experiment" / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(exp_dir / f"{selected_method_safe}_per_tf_agg_by_gt.svg", bbox_inches="tight")
    plt.close(fig)

def plot_multiexperiment_per_tf_auroc_agg_by_tf(name: str, selected_method: str, combined_per_tf_all_df: pd.DataFrame) -> plt.Figure:
    selected_method_safe = selected_method.lower().replace(" ", "_")
    
    fig = plot_all_results_auroc_boxplot(
        combined_per_tf_all_df,
        per_tf=True,
        override_color=True,
        sort_by_mean=False,
        ylim=(0.2, 0.8)
        )
    
    exp_dir = FIGURE_DIR / "multi_experiment" / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(exp_dir / f"{selected_method_safe}_per_tf_agg_by_tf.svg", bbox_inches="tight")
    plt.close(fig)

def plot_multiexperiment_pooled_auroc(name: str, selected_method: str, combined_results_df: pd.DataFrame) -> plt.Figure:
    selected_method_safe = selected_method.lower().replace(" ", "_")
    
    fig = plot_all_results_auroc_boxplot(
        combined_results_df,
        per_tf=False,
        override_color=True,
        sort_by_mean=False,
        ylim=(0.2, 0.8)
        )
    
    exp_dir = FIGURE_DIR / "multi_experiment" / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(exp_dir / f"{selected_method_safe}_pooled_auroc.svg", bbox_inches="tight")
    plt.close(fig)


def format_experiment_type_names(experiment_type, combined_results_df, combined_per_tf_all_df):
    """
    Formats the method names in the combined results DataFrames based on the specified experiment type.
    Must be one of: ['mESC_sample_num', 'dispersion', 'macrophage_sample_num', 'generic']
    """
    if experiment_type == "mESC_sample_num":
        multi_sample_results_df = combined_results_df.loc[combined_results_df["method"] == selected_method].copy()
        multi_sample_results_df["method"] = multi_sample_results_df["experiment"].str.replace(r"mESC_", "").str.replace(r"_samples*", " Samples", regex=True).str.replace(r"_hvg_filter_disp_0.01", "", regex=True)

        multi_sample_per_tf_df = combined_per_tf_all_df.loc[combined_per_tf_all_df["method"] == selected_method].copy()
        multi_sample_per_tf_df["method"] = multi_sample_per_tf_df["experiment"].str.replace(r"mESC_", "").str.replace(r"_samples*", " Samples", regex=True).str.replace(r"_hvg_filter_disp_0.01", "", regex=True)
        
        return multi_sample_results_df, multi_sample_per_tf_df
    
    if experiment_type == "dispersion":
        dispersion_results_df = combined_results_df.loc[combined_results_df["method"] == selected_method].copy()
        dispersion_results_df["method"] = dispersion_results_df["experiment"].str.replace(r".*_hvg_filter_disp_", "Dispersion ", regex=True)

        dispersion_per_tf_df = combined_per_tf_all_df.loc[combined_per_tf_all_df["method"] == selected_method].copy()
        dispersion_per_tf_df["method"] = dispersion_per_tf_df["experiment"].str.replace(r".*hvg_filter_disp_", "Dispersion ", regex=True)
        
        return dispersion_results_df, dispersion_per_tf_df
    
    elif experiment_type == "macrophage_sample_num":
        multi_sample_results_df = combined_results_df.loc[combined_results_df["method"] == selected_method].copy()
        multi_sample_results_df["method"] = multi_sample_results_df["experiment"].str.replace(r"Macrophage_buffer_", "Samples ", regex=True).str.replace(r"_hvg_filter_disp_0.1", "", regex=True)

        multi_sample_per_tf_df = combined_per_tf_all_df.loc[combined_per_tf_all_df["method"] == selected_method].copy()
        multi_sample_per_tf_df["method"] = multi_sample_per_tf_df["experiment"].str.replace(r"Macrophage_buffer_", "Samples ", regex=True).str.replace(r"_hvg_filter_disp_0.1", "", regex=True)
        
        return multi_sample_results_df, multi_sample_per_tf_df
    
    elif experiment_type == "generic":
        combined_results_df = combined_results_df.loc[combined_results_df["method"] == selected_method].copy()
        combined_per_tf_all_df = combined_per_tf_all_df.loc[combined_per_tf_all_df["method"] == selected_method].copy()
        
        combined_results_df["method"] = combined_results_df["experiment"].str.replace("_", " ")
        combined_per_tf_all_df["method"] = combined_per_tf_all_df["experiment"].str.replace("_", " ")
        
        return combined_results_df, combined_per_tf_all_df
    
    
    else:
        raise ValueError(f"Unsupported experiment type: {experiment_type}. Choose from ['mESC_sample_num', 'dispersion', 'macrophage_sample_num', 'generic']")



if __name__ == "__main__":

    multi_experiment_settings = {
        "mESC_best_settings": {
            "experiment_list": [
                ("mESC_E7.5_rep1_best_settings", 1),
                ("mESC_E7.5_rep2_best_settings", 1),
                ("mESC_E8.5_rep1_best_settings", 1),
                ("mESC_E8.5_rep2_best_settings", 1),
                # ("mESC_all_bnchmk_best_settings", 1),
            ],
            "experiment_type": "generic",
            "multi_experiment_name": "mESC_best_settings",
        },
        "mESC_multiple_samples": {
            "experiment_list": [
                ("mESC_1_sample_hvg_filter_disp_0.01", 1),
                ("mESC_2_sample_hvg_filter_disp_0.01", 1),
                ("mESC_3_sample_hvg_filter_disp_0.01", 1),
                ("mESC_4_sample_hvg_filter_disp_0.01", 1),
                ("mESC_6_sample_hvg_filter_disp_0.01", 1),
                ("mESC_7_sample_hvg_filter_disp_0.01", 1),
            ],
            "experiment_type": "mESC_sample_num",
            "multi_experiment_name": "mESC_multiple_samples",
        },
        "mESC_E7.5_rep1_dispersion": {
            "experiment_list": [
                ("mESC_two_hop_no_hvg_small", 2),
                # ("mES_two_hop_hvg_small", 1),
                # ("mESC_E7.5_rep1_hvg_filter_only_rna", 1),
                ("mESC_E7.5_rep1_hvg_filter_disp_0.01", 1),
                ("mESC_E7.5_rep1_hvg_filter_disp_0.05", 1),
                ("mESC_E7.5_rep1_hvg_filter_disp_0.1", 1),
                ("mESC_E7.5_rep1_hvg_filter_disp_0.2", 1),
                ("mESC_E7.5_rep1_hvg_filter_disp_0.3", 1),
                ("mESC_E7.5_rep1_hvg_filter_disp_0.4", 1),
                ("mESC_E7.5_rep1_hvg_filter_disp_0.5", 1),
                ("mESC_E7.5_rep1_hvg_filter_disp_0.6", 1),
            ],
            "experiment_type": "dispersion",
            "multi_experiment_name": "mESC_E7.5_rep1_dispersion",
        },
        "mESC_E7.5_rep2_dispersion": {
            "experiment_list": [
                # ("mESC_E7.5_rep2_hvg_filter_only_rna", 1),
                ("mESC_E7.5_rep2_hvg_filter_disp_0.01", 1),
                ("mESC_E7.5_rep2_hvg_filter_disp_0.05", 1),
                ("mESC_E7.5_rep2_hvg_filter_disp_0.1", 1),
                ("mESC_E7.5_rep2_hvg_filter_disp_0.2", 1),
                ("mESC_E7.5_rep2_hvg_filter_disp_0.3", 1),
                ("mESC_E7.5_rep2_hvg_filter_disp_0.4", 1),
                ("mESC_E7.5_rep2_hvg_filter_disp_0.5", 1),
                ("mESC_E7.5_rep2_hvg_filter_disp_0.6", 1),
            ],
            "experiment_type": "dispersion",
            "multi_experiment_name": "mESC_E7.5_rep2_dispersion",
        },
        "mESC_E8.5_rep1_dispersion": {
            "experiment_list": [
                # ("mESC_E8.5_rep1_hvg_filter_only_rna", 1),
                ("mESC_E8.5_rep1_hvg_filter_disp_0.01", 1),
                ("mESC_E8.5_rep1_hvg_filter_disp_0.05", 1),
                ("mESC_E8.5_rep1_hvg_filter_disp_0.1", 1),
                ("mESC_E8.5_rep1_hvg_filter_disp_0.2", 1),
                ("mESC_E8.5_rep1_hvg_filter_disp_0.3", 1),
                ("mESC_E8.5_rep1_hvg_filter_disp_0.4", 1),
                ("mESC_E8.5_rep1_hvg_filter_disp_0.5", 1),
                ("mESC_E8.5_rep1_hvg_filter_disp_0.6", 1),
            ],
            "experiment_type": "dispersion",
            "multi_experiment_name": "mESC_E8.5_rep1_dispersion",
        },
        "mESC_E8.5_rep2_dispersion": {
            "experiment_list": [
                # ("mESC_E8.5_rep2_hvg_filter_only_rna", 1),
                ("mESC_E8.5_rep2_hvg_filter_disp_0.01", 1),
                ("mESC_E8.5_rep2_hvg_filter_disp_0.05", 1),
                ("mESC_E8.5_rep2_hvg_filter_disp_0.1", 1),
                ("mESC_E8.5_rep2_hvg_filter_disp_0.2", 1),
                ("mESC_E8.5_rep2_hvg_filter_disp_0.3", 1),
                ("mESC_E8.5_rep2_hvg_filter_disp_0.4", 1),
                ("mESC_E8.5_rep2_hvg_filter_disp_0.5", 1),
                ("mESC_E8.5_rep2_hvg_filter_disp_0.6", 1),
            ],
            "experiment_type": "dispersion",
            "multi_experiment_name": "mESC_E8.5_rep2_dispersion",
        },
        "macrophage_best_samples": {
            "experiment_list": [
                ("Macrophage_buffer_1_best_settings", 1),
                ("Macrophage_buffer_2_best_settings", 1),
                ("Macrophage_all_bnchmk_best_settings", 1),
            ],
            "experiment_type": "macrophage_sample_num",
            "multi_experiment_name": "macrophage_best_samples",
        },
        "macrophage_multiple_samples": {
            "experiment_list": [
                ("Macrophage_buffer_1_hvg_filter_disp_0.1", 2),
                ("Macrophage_buffer_12_hvg_filter_disp_0.1", 1),
                ("Macrophage_buffer_123_hvg_filter_disp_0.1", 1),
                ("Macrophage_buffer_1234_hvg_filter_disp_0.1", 1),
            ],
            "experiment_type": "macrophage_sample_num",
            "multi_experiment_name": "macrophage_multiple_samples",
        },
        "macrophage_buffer_1_dispersion": {
            "experiment_list": [
                ("Macrophage_buffer_1_hvg_filter_only_rna", 2),
                ("Macrophage_buffer_1_hvg_filter_none", 2),
                ("Macrophage_buffer_1_hvg_filter_disp_0.01", 2),
                ("Macrophage_buffer_1_hvg_filter_disp_0.05", 2),
                ("Macrophage_buffer_1_hvg_filter_disp_0.1", 2),
                ("Macrophage_buffer_1_hvg_filter_disp_0.2", 2),
                ("Macrophage_buffer_1_hvg_filter_disp_0.3", 2),
                ("Macrophage_buffer_1_hvg_filter_disp_0.4", 2),
                ("Macrophage_buffer_1_hvg_filter_disp_0.5", 2),
                ("Macrophage_buffer_1_hvg_filter_disp_0.6", 2),
            ],
            "experiment_type": "dispersion",
            "multi_experiment_name": "macrophage_buffer_1_dispersion",
        },
        "macrophage_buffer_2_dispersion": {
            "experiment_list": [
                # ("Macrophage_buffer_2_hvg_filter_only_rna", 1),
                ("Macrophage_buffer_2_hvg_filter_none", 1),
                ("Macrophage_buffer_2_hvg_filter_disp_0.01", 1),
                ("Macrophage_buffer_2_hvg_filter_disp_0.05", 1),
                ("Macrophage_buffer_2_hvg_filter_disp_0.1", 1),
                ("Macrophage_buffer_2_hvg_filter_disp_0.2", 1),
                ("Macrophage_buffer_2_hvg_filter_disp_0.3", 1),
                ("Macrophage_buffer_2_hvg_filter_disp_0.4", 1),
                ("Macrophage_buffer_2_hvg_filter_disp_0.5", 1),
                ("Macrophage_buffer_2_hvg_filter_disp_0.6", 1),
            ],
            "experiment_type": "dispersion",
            "multi_experiment_name": "macrophage_buffer_2_dispersion",
        },
        "macrophage_buffer_3_dispersion": {
            "experiment_list": [
                # ("Macrophage_buffer_3_hvg_filter_only_rna", 1),
                ("Macrophage_buffer_3_hvg_filter_none", 1),
                ("Macrophage_buffer_3_hvg_filter_disp_0.01", 1),
                ("Macrophage_buffer_3_hvg_filter_disp_0.05", 1),
                ("Macrophage_buffer_3_hvg_filter_disp_0.1", 1),
                ("Macrophage_buffer_3_hvg_filter_disp_0.2", 1),
                ("Macrophage_buffer_3_hvg_filter_disp_0.3", 1),
                ("Macrophage_buffer_3_hvg_filter_disp_0.4", 1),
                ("Macrophage_buffer_3_hvg_filter_disp_0.5", 1),
                ("Macrophage_buffer_3_hvg_filter_disp_0.6", 1),
            ],
            "experiment_type": "dispersion",
            "multi_experiment_name": "macrophage_buffer_3_dispersion",
        },
        "macrophage_buffer_4_dispersion": {
            "experiment_list": [
                # ("Macrophage_buffer_4_hvg_filter_only_rna", 1),
                ("Macrophage_buffer_4_hvg_filter_none", 1),
                ("Macrophage_buffer_4_hvg_filter_disp_0.01", 1),
                ("Macrophage_buffer_4_hvg_filter_disp_0.05", 1),
                ("Macrophage_buffer_4_hvg_filter_disp_0.1", 1),
                ("Macrophage_buffer_4_hvg_filter_disp_0.2", 1),
                ("Macrophage_buffer_4_hvg_filter_disp_0.3", 1),
                ("Macrophage_buffer_4_hvg_filter_disp_0.4", 1),
                ("Macrophage_buffer_4_hvg_filter_disp_0.5", 1),
                ("Macrophage_buffer_4_hvg_filter_disp_0.6", 1),
            ],
            "experiment_type": "dispersion",
            "multi_experiment_name": "macrophage_buffer_4_dispersion",
        },
        "K562_dispersion": {
            "experiment_list": [
                # ("K562_hvg_filter_only_rna", 1),
                ("K562_hvg_filter_none", 1),
                ("K562_hvg_filter_disp_0.01", 1),
                ("K562_hvg_filter_disp_0.05", 1),
                ("K562_hvg_filter_disp_0.1", 1),
                ("K562_hvg_filter_disp_0.2", 1),
                ("K562_hvg_filter_disp_0.3", 1),
                ("K562_hvg_filter_disp_0.4", 1),
                ("K562_hvg_filter_disp_0.5", 1),
                ("K562_hvg_filter_disp_0.6", 1),
            ],
            "experiment_type": "dispersion",
            "multi_experiment_name": "K562_dispersion",
        },
        "K562_best_experiment_settings": {
            "experiment_list": [
                ("K562_best_experiment_settings", 1),
            ],
            "experiment_type": "generic",
            "multi_experiment_name": "K562_best_experiment_settings",
        },
    }
    
    
    with open(EXPERIMENT_DATA_DIR / "multi_experiment_metadata.json", "r") as f:
        multi_experiment_settings = json.load(f)
    
    # Selected method options: ['Gradient Attribution', 'TF Knockout']
    # Experiment type options: ['mESC_sample_num', 'dispersion', 'macrophage_sample_num', 'generic']
    
    experiments_to_run = ["mESC_E7.5_rep1_dispersion", "mESC_multiple_samples"]
    selected_method = "Gradient Attribution"
    
    for experiment_name in experiments_to_run:
        if experiment_name in multi_experiment_settings:
            selected_experiment_list = multi_experiment_settings[experiment_name]["experiment_list"]
            experiment_type = multi_experiment_settings[experiment_name]["experiment_type"]
            multi_experiment_name = multi_experiment_settings[experiment_name]["multi_experiment_name"]
        else:
            logging.warning(f"Experiment {experiment_name} not found in settings. Skipping.")

        logging.info(f"Creating plots for experiment: {multi_experiment_name}")
        logging.info(f" - Method: {selected_method}")
        logging.info(f" - Experiment type: {experiment_type}")

        logging.info("\nLoading and combining experiment results...")
        loaded_combined_dfs = load_and_combine_experiment_results(selected_experiment_list)
        combined_raw_results_df, combined_results_df, combined_per_tf_all_df, combined_per_tf_summary_df = loaded_combined_dfs

        # Plot all individual results for each of the experiments
        logging.info("\nPlotting individual experiment results...")
        for experiment_name, _ in selected_experiment_list:
            logging.info(f"  - {experiment_name}")
            plot_pooled_auroc(experiment_name, combined_results_df)
            plot_per_tf_auroc_agg_by_gt(experiment_name, combined_per_tf_all_df)
            plot_per_tf_auroc_agg_by_tf(experiment_name, combined_per_tf_all_df)
        
        # Format the x-axis labels based on the experiment type for the combined results DataFrames
        logging.info("\nFormatting combined experiment results...")
        formatted_results_df, formatted_per_tf_all_df = format_experiment_type_names(experiment_type, combined_results_df, combined_per_tf_all_df)
        
        logging.info("\nPlotting multi-experiment combined results...")
        plot_multiexperiment_pooled_auroc(multi_experiment_name, selected_method, formatted_results_df)
        plot_multiexperiment_per_tf_auroc_agg_by_gt(multi_experiment_name, selected_method, formatted_per_tf_all_df)
        plot_multiexperiment_per_tf_auroc_agg_by_tf(multi_experiment_name, selected_method, formatted_per_tf_all_df)
        
        logging.info(f"\nPlots saved to: {FIGURE_DIR / 'multi_experiment' / multi_experiment_name}")
        logging.info(f"{'-'*60}\n")
        
    logging.info("\nDone!")