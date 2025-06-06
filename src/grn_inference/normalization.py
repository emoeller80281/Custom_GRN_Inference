import numpy as np
import pandas as pd
import dask.dataframe as dd
import logging
from typing import Union

def minmax_normalize_dask(
    ddf: dd.DataFrame,
    score_cols: list[str],
    sample_frac: float = None,
    random_state: int = 42
) -> dd.DataFrame:
    """
    Min-max normalize selected columns. Optionally estimate min/max on a sample.
    """
    ddf = ddf.persist()

    if sample_frac:
        stats_df = ddf[score_cols].sample(frac=sample_frac, random_state=random_state).compute()
    else:
        stats_df = ddf[score_cols].compute()

    col_mins = stats_df.min().to_dict()
    col_maxs = stats_df.max().to_dict()

    def normalize_partition(df):
        df = df.copy()
        for col in score_cols:
            min_val = col_mins[col]
            max_val = col_maxs[col]
            if max_val != min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0.0
        return df

    return ddf.map_partitions(normalize_partition, meta=ddf._meta.copy())

def clip_and_normalize_log1p_dask(
    ddf: dd.DataFrame,
    score_cols: list[str],
    quantiles: tuple[float, float] = (0.05, 0.95),
    apply_log1p: bool = True,
    sample_frac: Union[None, float] = None,
    random_state: int = 42
) -> dd.DataFrame:
    """
    Clips, normalizes, and optionally log1p-transforms selected columns in a Dask DataFrame.
    Optionally samples a fraction of the data to estimate quantiles faster.

    Parameters
    ----------
    ddf : dd.DataFrame
        Input data.
    score_cols : list of str
        Column names to transform.
    quantiles : tuple
        Lower and upper quantiles (default 5th–95th).
    apply_log1p : bool
        Apply log1p after normalization.
    sample_frac : float or None
        If set, compute quantiles on a sample fraction to improve speed.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dd.DataFrame
        Transformed DataFrame.
    """
    ddf = ddf.persist()

    if sample_frac:
        logging.info(f"Sampling {sample_frac*100:.1f}% of data to estimate quantiles")
        sample = ddf[score_cols].sample(frac=sample_frac, random_state=random_state).compute()
    else:
        sample = ddf[score_cols].compute()

    q_lo, q_hi = quantiles
    low = sample.quantile(q_lo).to_dict()
    high = sample.quantile(q_hi).to_dict()

    def transform_partition(df):
        df = df.copy()
        for col in score_cols:
            df[col] = df[col].clip(lower=low[col], upper=high[col])
            if high[col] != low[col]:
                df[col] = (df[col] - low[col]) / (high[col] - low[col])
            else:
                df[col] = 0.0
            if apply_log1p:
                df[col] = np.log1p(df[col])
        return df

    return ddf.map_partitions(transform_partition, meta=ddf._meta.copy())

def minmax_normalize_pandas(
    df: pd.DataFrame,
    score_cols: list[str],
) -> pd.DataFrame:
    """
    Applies global min-max normalization to selected columns in a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The Pandas DataFrame containing the columns to normalize.
    score_cols : list of str
        List of column names to normalize.

    Returns
    -------
    pd.DataFrame
        Normalized Pandas DataFrame.
    """
    df = df.copy()
    
    if isinstance(df, dd.DataFrame):
        df = df.compute()
        
    for col in score_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df.loc[:, col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df.loc[:, col] = 0.0
    return df
