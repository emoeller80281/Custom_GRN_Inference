import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from typing import Union

def balance_dataset(
    df1: Union[pd.DataFrame, pd.Series], 
    df2: Union[pd.DataFrame, pd.Series]
    ):
    
    assert type(df1) == type(df2), "df1 and df2 are not the same type"
    
    min_edges = min(len(df1), len(df2))
    
    df1_balanced = df1.sample(min_edges)
    df2_balanced = df2.sample(min_edges)
    
    return df1_balanced, df2_balanced

def plot_scores_grouped_by_tf(df, title, height_col):
    plt.figure(figsize=(10,3))
    plt.bar(x=df['source_id'], height=df[height_col], color="#4195df")
    plt.title(title)
    plt.ylabel("Number of Targets", fontsize=10)
    plt.xticks(rotation=55, fontsize=10)
    plt.tight_layout()
    plt.show()
    
def plot_scores_grouped_by_tf_tg(df, title, score_col):
    plt.figure(figsize=(7,7))
    plt.boxplot(x=df[score_col])
    plt.title(title)
    plt.ylabel("Sliding Window Score", fontsize=10)
    plt.tight_layout()
    plt.show()
    
def plot_score_distribution_by_tf(
    df: pd.DataFrame,
    feature_col: str,
    tf_col_name: str = "source_id",
    tfs_of_interest: list[str] = [],
    title: str = "",
    limit_x: bool = False,
    top_tf_limit: int = 40,
):
    fig = plt.figure(figsize=(7, 5))
    y_cmap = plt.get_cmap("tab20c")
    
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

    # Limit the number of TFs to plot
    if len(tfs_of_interest) > top_tf_limit:

        print(f"INFO: Limiting to scores from the top {top_tf_limit} TFs, found {len(tfs_of_interest)}")
        tf_rows_in_df = df[df[tf_col_name].isin(tfs_of_interest)]
        df_by_tf = (
            tf_rows_in_df
            .groupby(tf_col_name)
            .count()
            .sort_values(feature_col, ascending=False)
            .reset_index()
            .iloc[:top_tf_limit]
            )
        tfs_of_interest = df_by_tf[tf_col_name].drop_duplicates().to_list()
        
    # Orders by the TF with the most scores -> least scores
    else:
        tf_rows_in_df = df[df[tf_col_name].isin(tfs_of_interest)]
        df_by_tf = (
            tf_rows_in_df
            .groupby(tf_col_name)
            .count()
            .sort_values(feature_col, ascending=False)
            .reset_index()
            )
        tfs_of_interest = df_by_tf[tf_col_name].drop_duplicates().to_list()
    
    # Sets the bin widths to be equal for all TF score distributions
    scores = df[feature_col].dropna()
    min_score = scores.min()
    max_score = scores.max()
    bin_width = max((max_score - min_score) / 85, 1e-3)
    bins = np.arange(min_score, max_score + bin_width, bin_width).tolist()
    
    for x, tf in enumerate(tfs_of_interest):
        tfs = df[df[tf_col_name] == tf]
        plt.hist(
            tfs[feature_col].dropna(),
            bins=bins, alpha=0.5,
            color=y_cmap(x / len(tfs_of_interest)),
            label=tf,
        )

    # set titles/labels on the same ax
    plt.title(title, fontsize=12)
    plt.xlabel("Sliding Window Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
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
    
    plt.figure(figsize=(6,4))
    plt.hist(
        true_series,
        bins=50,
        alpha=0.5,
        color="#4195df",
        label="Edge in\nGround Truth",
        density=density,
        log=log
    )
    plt.hist(
        false_series,
        bins=50,
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
            
    
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.xticks(fontsize=10, rotation=20)
    plt.yticks(fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.tight_layout()
    plt.show()