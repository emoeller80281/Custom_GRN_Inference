import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from typing import Union
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

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
        )

    # set titles/labels on the same ax
    plt.title(title, fontsize=12)
    plt.xlabel("Sliding Window Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xlim(0, 50000)
    if limit_x:
        plt.xlim(0, 1)

    fig.legend(
        loc="lower center",
        ncol=2,
        fontsize=10,
        bbox_to_anchor=(1.15, 0.10),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    plt.show()
    
def plot_true_false_distribution(
    true_series: pd.Series, 
    false_series: pd.Series, 
    xlabel: str = "Score", 
    ylabel: str = "", 
    title: str = "", 
    log: bool = False,
    balance: bool = False,
    density: bool = False
    ):
    
    if balance and not density: # Don't balance if using the density plot, it normalizes
        true_series, false_series = balance_dataset(true_series, false_series)
    
    if balance and density:
        print("INFO: Setting density=True balances the dataset, balance can be set to False")
    
    if not balance and not density:
        if len(true_series) > 10*len(false_series) or len(false_series) > 10*len(true_series):
            print("WARNING: Number of score are unbalanced, consider setting balance=True or density=True")
    
    combined = pd.concat([true_series, false_series]).dropna()
    min_score = combined.min()
    max_score = combined.max()
    bin_width = max((max_score - min_score) / 85, 1e-3)
    bins = np.arange(min_score, max_score + bin_width, bin_width).tolist()
    
    plt.figure(figsize=(5,3))
    plt.hist(
        true_series,
        bins=bins,
        alpha=0.5,
        color="#4195df",
        label="Edge in\nGround Truth",
        density=density,
        log=log
    )
    plt.hist(
        false_series,
        bins=bins,
        alpha=0.5,
        color="#747474",
        label="Edge not in\nGround Truth",
        density=density,
        log=log
    )
    
    if len(ylabel) == 0:
        if not density:
            ylabel = "Frequency"
        if density:
            ylabel = "Density"
            
    
    plt.title(title, fontsize=10)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlim((0, max_score))
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=9)
    plt.tight_layout()
    plt.show()
    
def plot_auroc(df: pd.DataFrame, score_col: str, title: str):
    assert score_col in df.columns, \
        f"{score_col} not in df columns. Columns: {df.columns}"
        
    assert "label" in df.columns, \
        f"label not in df columns. Columns: {df.columns}"
    
    df = df.copy()
    df = df.dropna(subset=["label", score_col])
    y_true = df["label"]
    y_score = df["sliding_window_score"]

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 3))
    plt.plot(fpr, tpr, lw=1, alpha=0.8, color="#4195df", label=f"AUROC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")
    # plt.title(title, fontsize=10)
    plt.legend(bbox_to_anchor=(0.5, -0.25), loc="lower center", borderaxespad=0., fontsize=9)
    plt.show()
    