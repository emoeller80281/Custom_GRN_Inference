import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
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
    plt.bar(x=df['source_id'], height=df[height_col], color="blue")
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
    df,
    feature_col,
    tfs_of_interest,
    limit_x = True
):
    fig = plt.figure(figsize=(7, 5))
    y_cmap = plt.get_cmap("tab20c")
    
    for x, tf in enumerate(tfs_of_interest):
        tfs = df[df["source_id"] == tf]
        percent_total_true_edges = len(tfs[feature_col]) / len(df[feature_col])
        nbins=max(30, math.ceil(50*percent_total_true_edges))
        plt.hist(
            tfs[feature_col].dropna(),
            bins=nbins, alpha=0.5,
            color=y_cmap(x / len(tfs_of_interest)),
            label=tf
        )
    # set titles/labels on the same ax
    plt.title("Sliding window score distribution colored by TF", fontsize=12)
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
    ylabel: str = "Frequency", 
    title: str = "", 
    log: bool = False,
    balance: bool = False
    ):
    
    if balance == True:
        true_series, false_series = balance_dataset(true_series, false_series)
    
    plt.figure(figsize=(6,4))
    plt.hist(
        true_series,
        bins=50,
        alpha=0.5,
        color="#4195df",
        label="Edge in\nGround Truth",
        log=log
    )
    plt.hist(
        false_series,
        bins=50,
        alpha=0.5,
        color="#747474",
        label="Edge not in\nGround Truth",
        log=log
    )
    plt.title(title, fontsize=12)
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
    plt.tight_layout()
    plt.show()