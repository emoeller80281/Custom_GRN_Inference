import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from typing import Union, Optional
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.patches as mpatches
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_bool_dtype
from scipy.stats import linregress

def balance_dataset(
    df1: Union[pd.DataFrame, pd.Series], 
    df2: Union[pd.DataFrame, pd.Series]
    ):
    
    assert type(df1) == type(df2), "df1 and df2 are not the same type"
    
    min_edges = min(len(df1), len(df2))
    
    df1_balanced = df1.sample(min_edges)
    df2_balanced = df2.sample(min_edges)
    
    return df1_balanced, df2_balanced

def plot_scores_grouped_by_tf(
    df: pd.DataFrame, 
    title: str, 
    score_col: str,
    tf_col_name: str = "source_id",
    top_tf_limit: int = 40
    ):
    
    # TF column exists
    assert tf_col_name in df.columns, f"ERROR: tf_col_name {tf_col_name} not in df.columns {df.columns}"
    
    # Set top_tf_limit to the number of TFs if it's too high
    if df[tf_col_name].nunique() < top_tf_limit:
        print(f"INFO: top_tf_limit set to {top_tf_limit}, but only {df[tf_col_name].nunique()} unique TFs in the df.\n\t- Setting top_tf_limit = {df[tf_col_name].nunique()}")
        top_tf_limit = df[tf_col_name].nunique()
    
    tf_list = df[tf_col_name].drop_duplicates().to_list()
    
    # Limit the number of TFs to plot
    if len(tf_list) > top_tf_limit:
        print(f"INFO: Limiting to scores from the top {top_tf_limit} TFs, found {len(tf_list)}")
        df_by_tf = (
            df
            .groupby(tf_col_name)
            .count()
            .sort_values(score_col, ascending=False)
            .reset_index()
            .iloc[:top_tf_limit]
            )
        tf_list = df_by_tf[tf_col_name].drop_duplicates().to_list()
        
    # Orders by the TF with the most scores -> least scores
    else:        
        df_by_tf = (
            df
            .groupby(tf_col_name)
            .count()
            .sort_values(score_col, ascending=False)
            .reset_index()
            )
        tf_list = df_by_tf[tf_col_name].drop_duplicates().to_list()
    
    plt.figure(figsize=(10,4))
    plt.bar(x=df_by_tf[tf_col_name], height=df_by_tf[score_col], color="#4195df")
    plt.title(title)
    plt.ylabel("Number of\nSliding Window Scores", fontsize=10)
    
    # If there are too many TFs, remove the individual TF name labels
    if len(tf_list) <= 40:
        plt.xticks(rotation=55, fontsize=10)
    elif len(tf_list) > 40:
        plt.xticks([])
        plt.xlabel("TFs", fontsize=10)
        
    plt.tight_layout()
    plt.show()
    
def plot_score_heatmap_by_tf_tg(
    df: pd.DataFrame,
    score_col: str,
    tf_col_name: str = "source_id",
    tg_col_name: str = "target_id",
    agg_func: str = "mean",
    max_tfs: int = 40,
    max_tgs: int = 40,
    figsize: tuple = (10, 8),
    cmap: str = "coolwarm",
    title: str = "TF–TG Score Heatmap"
    ):
    assert score_col in df.columns, f"{score_col} not in df"
    assert tf_col_name in df.columns, f"{tf_col_name} not in df"
    assert tg_col_name in df.columns, f"{tg_col_name} not in df"

    # Dynamically limit max TFs and TGs to what's in the df
    max_tfs = min(max_tfs, df[tf_col_name].nunique())
    max_tgs = min(max_tgs, df[tg_col_name].nunique())

    # Limit to top TFs and TGs with the most data
    pivot_df = df.pivot_table(
        index=tf_col_name, columns=tg_col_name,
        values=score_col, aggfunc='sum'
    ).fillna(0)

    top_tfs = pivot_df.sum(axis=1).nlargest(max_tfs).index
    top_tgs = pivot_df.sum(axis=0).nlargest(max_tgs).index

    pivot_df = pivot_df.loc[top_tfs, top_tgs]

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        pivot_df, 
        cmap=cmap, 
        annot=False, 
        cbar=True, 
        linewidths=0, 
        vmin=0, 
        vmax=np.percentile(pivot_df.values, 95)
        )
    plt.title(title, fontsize=12)
    plt.xlabel("Target Genes", fontsize=10)
    plt.ylabel("Transcription Factor", fontsize=10)
    plt.xticks(rotation=90)
    if len(top_tgs) > 50:
        plt.xticks([])
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
def plot_score_distribution_by_tf(
    df: pd.DataFrame,
    score_col: str,
    tf_col_name: str = "source_id",
    tfs_of_interest: list[str] = [],
    title: str = "",
    limit_x: bool = False,
    top_tf_limit: int = 40,
    log: bool = False
    ):
    fig = plt.figure(figsize=(7, 5))
    y_cmap = plt.get_cmap("tab20c")
    
    if len(df) > 1e6:
        df = df.sample(n=1_000_000)
    
    # TF column exists
    assert tf_col_name in df.columns, f"ERROR: tf_col_name {tf_col_name} not in df.columns {df.columns}"
    
    # All TFs are in the df
    assert all(x in df[tf_col_name].to_list() for x in tfs_of_interest), \
        f"ERROR: Found items in tfs_of_interest not in df: {[i for i in tfs_of_interest if i not in df[tf_col_name].to_list()]}"
    
    # Set top_tf_limit to the number of TFs if it's too high
    if df[tf_col_name].nunique() < top_tf_limit:
        print(f"INFO: top_tf_limit set to {top_tf_limit}, but only {df[tf_col_name].nunique()} unique TFs in the df.\n\t- Setting top_tf_limit = {df[tf_col_name].nunique()}")
        top_tf_limit = df[tf_col_name].nunique()
    
    # Use all TFs if none are specified
    if len(tfs_of_interest) == 0:
        print(f"INFO: No TFs of interest specified, defaulting to the top_tf_limit ({top_tf_limit}) TFs")
        tfs_of_interest = df[tf_col_name].drop_duplicates().to_list()
        tf_rows_in_df = df[df[tf_col_name].isin(tfs_of_interest)]
        df = (
            tf_rows_in_df
            .groupby(tf_col_name)
            .count()
            .sort_values(score_col, ascending=False)
            .reset_index()
            )
        tfs_of_interest = df[tf_col_name].drop_duplicates().to_list()

    # Limit the number of TFs to plot
    if len(tfs_of_interest) > top_tf_limit:
        tfs_of_interest = tfs_of_interest[0:top_tf_limit]
        print(f"INFO: Limiting to scores from the top {top_tf_limit} TFs, found {len(tfs_of_interest)}")
        tf_rows_in_df = df[df[tf_col_name].isin(tfs_of_interest)]
        # df = (
        #     tf_rows_in_df
        #     .groupby(tf_col_name)
        #     .count()
        #     .sort_values(score_col, ascending=False)
        #     .reset_index()
        #     )
        # tfs_of_interest = df[tf_col_name].iloc[:top_tf_limit].drop_duplicates().to_list()
        
    
    # Sets the bin widths to be equal for all TF score distributions
    scores = df[score_col].dropna()
    min_score = scores.min()
    max_score = scores.max()
    bin_width = max((max_score - min_score) / 85, 1e-3)
    bins = np.arange(min_score, max_score + bin_width, bin_width).tolist()
    
    for x, tf in tqdm(enumerate(tfs_of_interest), total=len(tfs_of_interest), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        tfs = df[df[tf_col_name] == tf]
        plt.hist(
            tfs[score_col].dropna(),
            bins=bins, alpha=0.5,
            color=y_cmap(x / len(tfs_of_interest)),
            label=tf,
            log=log
        )

    # set titles/labels on the same ax
    plt.title(title, fontsize=12)
    plt.xlabel("Sliding Window Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xlim(0, max_score)
    if limit_x:
        plt.xlim(0, 1)

    fig.legend(
        loc="lower center",
        ncol=1,
        fontsize=9,
        bbox_to_anchor=(1.10, 0.2),

    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    plt.show()

def plot_true_false_scores_by_tf_barplots(
    df: pd.DataFrame,
    score_col: str,
    tf_name_col: str,
    agg: str = "median",
    top_n: int = 20,
    title: Union[str, None] = None,
) -> plt.Figure:
    
    assert all(c in df.columns for c in [score_col, tf_name_col, "label"])
    sub = df[[score_col, tf_name_col, "label"]].dropna().copy()
    sub["label"] = sub["label"].astype(bool)

    # pick top TFs by count
    top_tfs = sub[tf_name_col].value_counts().head(top_n).index
    sub = sub[sub[tf_name_col].isin(top_tfs)]

    aggfunc = np.median if agg == "median" else np.mean
    wide = (
        sub.groupby([tf_name_col, "label"])[score_col]
        .agg(aggfunc)
        .unstack("label")
        .reindex(top_tfs)
        .rename(columns={True: "True", False: "False"})
    ).sort_values(by="True", axis=0, ascending=False)

    x = np.arange(len(wide))
    width = 0.4

    fig, ax = plt.subplots(figsize=(min(1.2*top_n, 18), 6))
    ax.bar(x - width/2, wide["True"],  width, color="#4195df", label="Edge in Ground Truth")
    ax.bar(x + width/2, wide["False"], width, color="#747474", label="Edge Not in Ground Truth")

    ax.set_xticks(x)
    ax.set_xticklabels(wide.index, rotation=65, ha="right", fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
    ax.set_ylabel(f"Median TF-Peak Binding Score", fontsize=16)
    ax.set_xlabel("TF", fontsize=16)
    if title is None:
        title = f"{agg.title()} {score_col} by TF (True vs False)"
    ax.set_title(title, fontsize=18)
    ax.legend(loc="upper center", bbox_to_anchor=(1.15, 0.50), ncol=1, fontsize=14)
    fig.tight_layout()
    return fig

def plot_true_false_scores_by_tf_boxplots(
    df: pd.DataFrame, 
    score_col: str, 
    tf_name_col: str,
    title: Union[str, None] = None,
    top_n: int = 20,
    order_by: str = "median",
    ) -> plt.Figure:
    
    # Validate columns
    assert all(c in df.columns for c in [score_col, tf_name_col, "label"]), \
        f"Missing required columns. Have: {df.columns.tolist()}"

    sub = df[[score_col, tf_name_col, "label"]].dropna().copy()

    # Ensure label is boolean-like
    if not (is_bool_dtype(sub["label"]) or is_integer_dtype(sub["label"])):
        # Try to coerce common string cases
        sub["label"] = sub["label"].map({"True": 1, "False": 0, "true": 1, "false": 0}).fillna(sub["label"])
        sub["label"] = sub["label"].astype(int)
    sub["label"] = sub["label"].astype(bool)

    assert is_numeric_dtype(sub[score_col]), f"{score_col} must be numeric"

    # Choose top TFs by either median or count
    if order_by == "median":
        tf_order = (
            sub.groupby(tf_name_col)[score_col]
            .median()
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )
    else:  # count
        tf_order = (
            sub[tf_name_col]
            .value_counts()
            .head(top_n)
            .index
            .tolist()
        )
    sub = sub[sub[tf_name_col].isin(tf_order)]
    # lock the order
    sub[tf_name_col] = pd.Categorical(sub[tf_name_col], categories=tf_order, ordered=True)

    # Plot
    fig, ax = plt.subplots(figsize=(min(1.2*top_n, 18), 6))
    sns.boxplot(
        data=sub,
        x=tf_name_col,
        y=score_col,
        hue="label",
        order=tf_order,
        showfliers=False,
        palette={True:"#4195df", False:"#747474"},
        ax=ax,
    )

    if title is None:
        title = "TF Binding Scores by TF (True vs False)"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("TF", fontsize=12)
    ax.set_ylabel(score_col, fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(labels=["Edges in Ground Truth", "Edges Not in Ground Truth"])
    fig.tight_layout()
    return fig

def plot_true_false_distribution(
    true_series: pd.Series,
    false_series: pd.Series,
    xlabel: str = "Score",
    ylabel: str = "",
    title: str = "",
    log: bool = False,
    balance: bool = False,
    density: bool = False,
    ax: Optional[plt.Axes] = None,
    use_default_legend: bool = False,
    silence_legend: bool = False,
) -> plt.Figure:
    # Optional balancing (make sure you’ve defined balance_dataset elsewhere)
    if balance and not density:
        true_series, false_series = balance_dataset(true_series, false_series)  # noqa: F821
    if balance and density:
        print("INFO: density=True normalizes counts; balance=True is unnecessary.")

    if not balance and not density:
        n_t, n_f = len(true_series), len(false_series)
        if n_t > 10*n_f or n_f > 10*n_t:
            print("WARNING: Class sizes are highly unbalanced. Consider balance=True or density=True.")

    # Combine to compute common bins
    combined = pd.concat([true_series, false_series]).astype(float).dropna()
    if combined.empty:
        fig, ax2 = plt.subplots(figsize=(5, 3))
        ax2.set_title(title or "No data")
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel or ("Density" if density else "Frequency"))
        return fig

    min_score = combined.min()
    max_score = combined.max()

    # Handle degenerate range (all values equal)
    if np.isclose(max_score, min_score):
        eps = 1e-6 if max_score == 0 else 1e-3 * abs(max_score)
        bins = np.linspace(min_score - eps, max_score + eps, 20)
    else:
        # Robust automatic binning (Freedman–Diaconis); falls back to 'auto' if needed
        try:
            bins = np.histogram_bin_edges(combined.values, bins="fd")
        except Exception:
            bins = np.histogram_bin_edges(combined.values, bins="auto")

        # Ensure a reasonable number of bins
        if len(bins) < 5:
            # manual: ~85 bins as you intended, but with correct parentheses
            width = max((max_score - min_score) / 85.0, 1e-3)
            bins = np.arange(min_score, max_score + width, width)

    # Axes setup
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        fontsize = 12
    elif isinstance(ax, plt.Axes):
        fig = ax.figure
        fontsize = 10
    else:
        raise ValueError("ax must be a matplotlib Axes or None")

    # Labels
    if silence_legend:
        true_label = None
        false_label = None
    elif use_default_legend:
        true_label = "Edge in\nGround Truth"
        false_label = "Edge not in\nGround Truth"
    else:
        true_label = true_series.name if isinstance(true_series.name, str) else None
        false_label = false_series.name if isinstance(false_series.name, str) else None

    # Plot
    ax.hist(true_series,  bins=bins, alpha=0.5, color="#4195df",
            label=true_label, density=density, log=log, edgecolor="none")
    ax.hist(false_series, bins=bins, alpha=0.5, color="#747474",
            label=false_label, density=density, log=log, edgecolor="none")

    # Axis text
    if not ylabel:
        ylabel = "Density" if density else "Frequency"
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize-1)
    ax.tick_params(axis='y', labelsize=fontsize-1)
    ax.set_xlim((bins[0], bins[-1]))

    # Legend (only if labels exist and not silenced)
    if not silence_legend and (true_label or false_label):
        ax.legend(loc="upper right", fontsize=fontsize-2)

    fig.tight_layout()
    return fig
    
def plot_auroc(df: pd.DataFrame, score_col: str, title: str="", ax: Union[plt.Axes, None]=None):
    assert score_col in df.columns, \
        f"{score_col} not in df columns. Columns: {df.columns}"
        
    assert "label" in df.columns, \
        f"label not in df columns. Columns: {df.columns}"
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    elif isinstance(ax, plt.Axes):
        fig = ax.figure
    else:
        raise ValueError("ax must be a matplotlib Axes or None")
    
    df = df.copy()
    df = df.dropna(subset=["label", score_col])
    y_true = df["label"]
    y_score = df[score_col]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
        
    ax.plot(fpr, tpr, lw=1, alpha=0.8, color="#4195df", label=f"AUROC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    ax.set_title(title, fontsize=10)
    ax.legend(bbox_to_anchor=(0.5, -0.25), loc="lower center", borderaxespad=0., fontsize=9)
    
    return fig

def plot_scores_distribution(df: pd.DataFrame, score_col: str, title: str="", log: bool = False, ax: Union[plt.Axes, None]=None):
    """Extract the true and false scores by the value of each row in the 'label' column"""
    assert "label" in df.columns, f"label column does not exist, columns: {df.columns}"
    
    true_scores = df[df["label"] == True]
    false_scores = df[df["label"] == False]

    fig = plot_true_false_distribution(
        true_series=true_scores[score_col], 
        false_series=false_scores[score_col],
        xlabel=score_col,
        title=title,
        balance=True,
        log=log,
        ax=ax,
        silence_legend=True
        )
    
    return fig

def plot_true_false_boxplots(
    df: pd.DataFrame, 
    x_axis_group_col: str = "label",
    score_col: str = "sliding_window_score",
    xlabel: str = "label", 
    ylabel: str = "TF-Peak Binding Score", 
    title: str = "True/False Scores", 
    ax: Union[plt.Axes, None] = None,
    add_side_legend: bool = False
) -> plt.Figure:

    df2 = df.dropna(subset=[x_axis_group_col, score_col]).copy()

    # Ensure exactly 2 groups
    assert df2[x_axis_group_col].nunique() == 2, \
        f"{x_axis_group_col} has {df2[x_axis_group_col].nunique()} unique values; expected 2"

    # Make order & palette match the dtype of the column
    if pd.api.types.is_bool_dtype(df2[x_axis_group_col]):
        order = [True, False]
        palette = {True: "#4195df", False: "#747474"}
    elif pd.api.types.is_integer_dtype(df2[x_axis_group_col]):
        order = [1, 0] if set([1,0]).issubset(set(df2[x_axis_group_col].unique())) \
                else sorted(df2[x_axis_group_col].unique())
        palette = {1: "#4195df", 0: "#747474"}
    else:
        # cast to string to be safe
        df2[x_axis_group_col] = df2[x_axis_group_col].astype(str)
        order = ["True", "False"] if set(["True","False"]).issubset(set(df2[x_axis_group_col].unique())) \
                else sorted(df2[x_axis_group_col].unique())
        palette = {"True": "#4195df", "False": "#747474"} if set(["True","False"]).issubset(set(order)) else None

    # Figure/Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        fontsize = 12
        add_legend = True
    else:
        fig = ax.figure
        fontsize = 10
        add_legend = True

    sns.boxplot(
        data=df2,
        x=x_axis_group_col,
        y=score_col,
        hue=x_axis_group_col,
        order=order,
        palette=palette,
        showfliers=False,
        ax=ax
    )

    ax.get_legend().set_visible(False) # We set up our own legend below
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize-1)
    ax.tick_params(axis='y', labelsize=fontsize-1)

    if add_side_legend:
        handles = [
            mpatches.Patch(facecolor="#4195df", edgecolor="#032b5f", label="Edges in Ground Truth"),
            mpatches.Patch(facecolor="#747474", edgecolor="#2D2D2D", label="Edges Not in Ground Truth")
        ]
        fig.legend(handles=handles, bbox_to_anchor=(1.03, 0.5), loc='upper left', borderaxespad=0., fontsize=fontsize-2)
        
        # If there is a legend to the side, dont label the bottom
        ax.tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)
    elif add_legend:
        ax.set_xticks(ticks=[0, 1], labels=["Edges in Ground Truth", "Edges Not in Ground Truth"])
    else:
        ax.tick_params(axis='x', which="both", bottom=False, top=False, labelbottom=False)


    fig.tight_layout()
    return fig

def tg_assignment_multiplot(nearest_tss_df, mira_df, cicero_df, suptitle):
    fig, axes = plt.subplots(4, 3, figsize=(11, 12))

    # Column 1: gene TSS
    plot_scores_distribution(nearest_tss_df,
                            title="Nearest gene TSS to peak",
                            ax=axes[0, 0])
    
    plot_grouped_score_boxplot(
        nearest_tss_df,
        group_col="target_id",
        score_col="sliding_window_score",
        title=f"Nearest gene TSS top scores by TG",
        ylabel="Sliding Window Score",
        n_top_groups=15,
        ax=axes[1,0]
        )
    
    plot_auroc(nearest_tss_df,
                        score_col="sliding_window_score",
                        title="Nearest gene TSS AUROC",
                        ax=axes[2, 0])
    
    plot_true_false_boxplots(
        nearest_tss_df,
        ax=axes[3, 0]
        )
   
    # Column 2: MIRA peak-TG
    plot_scores_distribution(mira_df,
                            title="MIRA peak-TG",
                            ax=axes[0, 1])
    
    plot_grouped_score_boxplot(
        mira_df,
        group_col="target_id",
        score_col="sliding_window_score",
        title=f"MIRA top scores by TG",
        ylabel="Sliding Window Score",
        n_top_groups=15,
        ax=axes[1,1]
        )
    
    plot_auroc(mira_df,
                        score_col="sliding_window_score",
                        title="MIRA AUROC",
                        ax=axes[2, 1])
    
    plot_true_false_boxplots(
        mira_df,
        ax=axes[3, 1])
    
    

    # Column 3: Cicero peak-TG
    plot_scores_distribution(cicero_df,
                            title="Cicero peak-TG",
                            ax=axes[0, 2])
    
    plot_grouped_score_boxplot(
        cicero_df,
        group_col="target_id",
        score_col="sliding_window_score",
        title=f"Cicero top scores by TG",
        ylabel="Sliding Window Score",
        n_top_groups=15,
        ax=axes[1,2]
        )
    
    plot_auroc(
        cicero_df,
        score_col="sliding_window_score",
        title="Cicero AUROC",
        ax=axes[2, 2]
        )
    
    plot_true_false_boxplots(
        cicero_df,
        ax=axes[3, 2])
    


    legend_handles = [
        mpatches.Patch(color="#4195df", alpha=0.5, label="Edge in Ground Truth"),
        mpatches.Patch(color="#747474", alpha=0.5, label="Edge not in Ground Truth")
    ]
    fig.legend(handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),  # centered below plots
            ncol=2,
            fontsize=9)

    plt.suptitle(suptitle)

    plt.tight_layout()
    return fig

def plot_grouped_score_boxplot(
    df: pd.DataFrame, 
    group_col: str, 
    score_col: str, 
    n_top_groups: int=10, 
    title: str="",
    xlabel: Union[str, None]=None,
    ylabel: Union[str, None]=None,
    ax: Union[plt.Axes, None]=None
    ):
    
    assert score_col in df.columns, \
        f"{score_col} not in df columns. Columns: {df.columns}"
    
    assert group_col in df.columns, \
        f"{group_col} not in df columns. Columns: {df.columns}"
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        fontsize = 15

    elif isinstance(ax, plt.Axes):
        fig = ax.figure
        fontsize = 9
    else:
        raise ValueError("ax must be a matplotlib Axes or None")
    
    top_ids = (
        df
        .groupby(group_col)[score_col]
        .median()
        .sort_values(ascending=False)
        .head(n_top_groups)
        .index
    )

    # Filter and preserve order
    top_tg_df = df[
        df[group_col].isin(top_ids)
    ]

    # Boxplot with preserved order
    sns.boxplot(
        data=top_tg_df,
        x=group_col,
        y=score_col,
        order=top_ids,
        showfliers=False,
        ax=ax
    )

    ax.set_title(title, fontsize=fontsize+1)
    
    if ylabel is None:
        ax.set_ylabel(score_col, fontsize=fontsize)
    else:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    
    if xlabel is None:
        ax.set_xlabel('')
    else:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    
    ax.set_yticklabels(ax.get_yticklabels(),fontsize=fontsize-1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=fontsize-1)
    fig.tight_layout()
    return fig
    
def plot_mean_score_differences_grouped(
    df: pd.DataFrame, 
    group_col: str,
    sample_name: str = "",
    ground_truth_name: str = "",
    tg_label_name: str = "",
    top_n: Union[int, None] = 75,
    order_by: Union[str, None] = None,
    ):
    # Pretty axis label
    group_name = {"source_id": "TF", "target_id": "TG"}.get(group_col, group_col)

    # Decide ordering metric
    if order_by is None:
        order_by = "p_value" if "p_value" in df.columns else "abs_diff"

    if order_by == "p_value":
        if "p_value" not in df.columns:
            raise ValueError("order_by='p_value' but 'p_value' not in df.columns")
        sorter = df[[group_col, "p_value"]].copy()
        sorter = sorter.sort_values("p_value", ascending=True)  # most significant first
    elif order_by == "abs_diff":
        if not {"mean_True", "mean_False"}.issubset(df.columns):
            raise ValueError("Need columns 'mean_True' and 'mean_False' for abs_diff ordering")
        sorter = df[[group_col]].copy()
        sorter["abs_diff"] = np.abs(df["mean_True"] - df["mean_False"])
        sorter = sorter.sort_values("abs_diff", ascending=False)
    else:
        raise ValueError("order_by must be 'p_value' or 'abs_diff'")

    # If top_n requested, select top-N groups BEFORE melting
    if top_n is not None:
        keep_groups = sorter[group_col].head(top_n).tolist()
        df = df[df[group_col].isin(keep_groups)]
        sorter = sorter[sorter[group_col].isin(keep_groups)]

    # Long format for plotting
    plot_df = df.melt(
        id_vars=group_col,
        value_vars=["mean_True", "mean_False"],
        var_name="label",
        value_name="mean_score"
    )
    plot_df["label"] = plot_df["label"].str.replace("mean_", "", regex=False)

    # Apply categorical order
    order = sorter[group_col].tolist()
    plot_df[group_col] = pd.Categorical(plot_df[group_col], categories=order, ordered=True)
    plot_df = plot_df.sort_values(group_col)

    # Colors
    label_colors = {"True": "#4195df", "False": "#747474"}

    fig, ax = plt.subplots(figsize=(14,5))

    # Points
    sns.scatterplot(
        data=plot_df,
        x=group_col,
        y="mean_score",
        hue="label",
        palette=label_colors,
        s=80,
        zorder=2,
        ax=ax
    )

    # Connect True/False within each group
    for _, g in plot_df.groupby(group_col, sort=False, observed=True):
        ax.plot(g[group_col], g["mean_score"], color="gray", linewidth=1, zorder=1)

    # Labels & layout
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=10)
    ax.set_xlabel(group_name, fontsize=14)
    ax.set_ylabel("Mean log1p Sliding Window Score", fontsize=14)
    title_main = f"{sample_name}, {ground_truth_name}, {tg_label_name}".strip(", ").strip()
    ax.set_title(f"{title_main}\nMean log1p True/False Scores by {group_name}", fontsize=14)
    ax.legend(title="Label", bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0., fontsize=12)
    fig.tight_layout()
    return fig

def plot_individual_true_false_distributions(
    mean_diff_df: pd.DataFrame, 
    original_score_df: pd.DataFrame,
    group_col: str,
    score_col: str = "sliding_window_score",
    sample_name: str="",
    ground_truth_name: str="",
    tg_label_name: str=""
    ):
    if group_col == "source_id":
        group_name = "TF"
    elif group_col == "target_id":
        group_name = "TG"
    else:
        group_name = group_col

    fig, axes = plt.subplots(4, 4, figsize=(11, 8))

    ax = axes.flatten()
    for i in range(16):
        selected = mean_diff_df.iloc[i, :][group_col]

        selected_data = original_score_df[original_score_df[group_col] == selected]

        true_series = selected_data[selected_data["label"] == True][score_col]
        false_series = selected_data[selected_data["label"] == False][score_col]

        plot_true_false_distribution(
            true_series=true_series,
            false_series=false_series,
            balance=True,
            xlabel="",
            ylabel=None,
            title=selected,
            silence_legend=True,
            ax=ax[i]
        )
        
    fig.suptitle(f"{sample_name}, {ground_truth_name}, {tg_label_name}\nTrue / False Score Distributions by {group_name}")
    fig.supylabel("Frequency")
    fig.supxlabel("Sliding Window Score")
    fig.tight_layout()
    
    return fig

def plot_peak_length_vs_score_scatterplot(
    df: pd.DataFrame, 
    score_col: str,
    title: Union[str, None] = None,
    ylabel: Union[str, None] = None,
) -> plt.Figure:
    # --- validations ---
    assert "label" in df.columns, \
        f"'label' column required. Columns = {df.columns.tolist()}"
    assert score_col in df.columns, \
        f"'{score_col}' column not found in df."
    assert "peak_length" in df.columns, \
        f"'peak_length' column not found in df."

    assert is_numeric_dtype(df["peak_length"]), \
        f"'peak_length' must be numeric, found {df['peak_length'].dtype}"
    assert is_numeric_dtype(df[score_col]), \
        f"'{score_col}' must be numeric, found {df[score_col].dtype}"

    # allow bool or integer labels
    assert is_bool_dtype(df["label"]) or is_integer_dtype(df["label"]), \
        f"'label' must be bool or integer-like, found {df['label'].dtype}"

    # --- prep & split ---
    sub = df.dropna(subset=["peak_length", score_col]).copy()
    # build a boolean mask regardless of dtype
    is_true = sub["label"].astype(bool)

    true_df  = sub[is_true]
    false_df = sub[~is_true]

    # need at least 2 points and non-constant x for linregress
    for name, d in {"True": true_df, "False": false_df}.items():
        assert len(d) >= 2 and d["peak_length"].nunique() >= 2, \
            f"{name} group has too few points or constant x for regression."

    # --- regressions ---
    t_slope, t_int, t_r, t_p, t_stderr = linregress(true_df["peak_length"],  true_df[score_col])
    f_slope, f_int, f_r, f_p, f_stderr = linregress(false_df["peak_length"], false_df[score_col])

    t_label = f"Edges in Ground Truth: y = {t_slope:.2e}x + {t_int:.2e}, r={t_r:.2f}, p={t_p:.2e}"
    f_label = f"Edges Not in Ground Truth: y = {f_slope:.2e}x + {f_int:.2e}, r={f_r:.2f}, p={f_p:.2e}"

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(false_df["peak_length"], false_df[score_col],
               s=28, c="#747474", edgecolors="#2D2D2D", alpha=0.30,
               label=f_label, rasterized=True)
    ax.scatter(true_df["peak_length"],  true_df[score_col],
               s=28, c="#4195df", edgecolors="#032b5f", alpha=0.90,
               label=t_label, rasterized=True)

    x_min = 0.0
    x_max = max(true_df["peak_length"].max(), false_df["peak_length"].max()) * 1.05
    xs = np.linspace(x_min, x_max, 200)

    ax.plot(xs, t_int + t_slope * xs, c="#4195df",  lw=2, ls="--")
    ax.plot(xs, f_int + f_slope * xs, c="#2D2D2D", lw=2, ls="--")

    # labels/titles
    if ylabel is None:
        ylabel = "TF–Peak Binding Score"
    if title is None:
        title = "TF–Peak Binding Score vs Peak Length"

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Peak Length (bp)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=1, fontsize=11)
    fig.tight_layout()
    return fig

def plot_calculation_method_overview(
    chipseq_plot_df,
    score_col,
    tf_col_name,
    method_name,
    calculation_method,
    log_dist= False,
    ):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    plot_scores_distribution(
        chipseq_plot_df,
        score_col=score_col,
        title="Distribution of True/False scores",
        log=log_dist,
        ax=axes[0, 0]
        )

    plot_grouped_score_boxplot(
        chipseq_plot_df,
        group_col=tf_col_name,
        score_col=score_col,
        title=f"Scores grouped by TF",
        ylabel=f"{method_name} Score",
        n_top_groups=15,
        ax=axes[1, 0]
        )

    plot_grouped_score_boxplot(
        chipseq_plot_df,
        group_col="target_id",
        score_col=score_col,
        title=f"Scores grouped by TG",
        ylabel=f"{method_name} Score",
        n_top_groups=15,
        ax=axes[1, 1]
        )


    plot_auroc(chipseq_plot_df,
                        score_col=score_col,
                        title="AUROC",
                        ax=axes[0, 1])

    plot_true_false_boxplots(
        chipseq_plot_df,
        score_col=score_col,
        ylabel=f"{method_name} Score",
        ax=axes[0, 2]
        )

    fig.delaxes(axes[1, 2])

    legend_handles = [
        mpatches.Patch(color="#4195df", alpha=0.5, label="Edge in Ground Truth"),
        mpatches.Patch(color="#747474", alpha=0.5, label="Edge not in Ground Truth")
    ]
    fig.legend(handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),  # centered below plots
            ncol=2,
            fontsize=9)

    plt.suptitle(f"{method_name} ChIP-Seq {calculation_method}, Chip-Atlas Ground Truth")

    plt.tight_layout()

    return fig
