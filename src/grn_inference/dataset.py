import dask.dataframe as dd
import pandas as pd
import numpy as np

from typing import Tuple

# Loading functions from Step020.peak_gene_correlation.py
def load_atac_dataset(atac_data_file: str) -> dd.DataFrame:
    atac_df: dd.DataFrame = dd.read_parquet(atac_data_file, engine="pyarrow")
    return atac_df

def load_rna_dataset(rna_data_file: str) -> dd.DataFrame:
    rna_df: dd.DataFrame = dd.read_parquet(rna_data_file, engine="pyarrow")
    return rna_df


def compute_atac_mean(atac_df: dd.DataFrame) -> dd.DataFrame:
    """Compute mean peak accessibility."""
    atac_df["mean_peak_accessibility"] = atac_df.select_dtypes("number").mean(axis=1)
    
    norm_atac_df = minmax_normalize_pandas(
        df=atac_df, 
        score_cols=["mean_peak_accessibility"], 
        dtype=np.float32
    )
    
    return norm_atac_df[["peak_id", "mean_peak_accessibility"]]

def pivot_melted_inferred_network(melted_ddf: str) -> dd.DataFrame:
    """
    Loads the melted sparse inferred-network parquet (source_id, peak_id, target_id, score_type, score_value)
    and pivots it back to wide form *including* peak_id in the index.
    """
    

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
    return dd.from_pandas(wide, npartitions=1)

def compute_expression_means(rna_df: dd.DataFrame) -> Tuple[dd.DataFrame, dd.DataFrame]:
    """Compute mean TF and TG expression from RNA matrix."""
    mean_expr = rna_df.select_dtypes("number").mean(axis=1)
    rna_df["mean_expression"] = mean_expr
    
    norm_rna_df = minmax_normalize_pandas(
        df=rna_df, 
        score_cols=["mean_expression"], 
        dtype=np.float32
    )

    tf_df = norm_rna_df[["gene_id", "mean_expression"]].rename(columns={"gene_id": "source_id", "mean_expression": "mean_TF_expression"})
    tg_df = norm_rna_df[["gene_id", "mean_expression"]].rename(columns={"gene_id": "target_id", "mean_expression": "mean_TG_expression"})
    
    return tf_df, tg_df

def minmax_normalize_pandas(
    df: pd.DataFrame,
    score_cols: list[str],
    dtype: np.dtype = np.float32
) -> pd.DataFrame:
    """
    Applies global min-max normalization to selected columns in a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The Pandas DataFrame containing the columns to normalize.
    score_cols : list of str
        List of column names to normalize.
    dtype : numpy dtype
        Output data type (e.g., np.float32 or np.float16). Default is np.float32.

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
        df.loc[:, col] = df[col].astype(dtype)
    return df