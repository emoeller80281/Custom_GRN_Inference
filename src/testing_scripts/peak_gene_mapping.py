import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybedtools
from pybedtools import BedTool

from collections import defaultdict
from datetime import datetime
from joblib import Parallel, delayed
from scipy import stats
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

import ensembl
import normalization
import enhancerdb


# ------------------------- CONFIGURATIONS ------------------------- #
ORGANISM = "hsapiens"
ATAC_DATA_FILE = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/K562/K562_human_filtered/K562_human_filtered_ATAC.csv"
RNA_DATA_FILE =  "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/K562/K562_human_filtered/K562_human_filtered_RNA.csv"
ENHANCER_DB_FILE = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/enhancer_db/enhancer"
PEAK_DIST_LIMIT = 1_000_000
N_JOBS = 8


# ------------------------- DATA LOADING & PREPARATION ------------------------- #
def load_and_parse_atac_peaks(atac_data_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ATAC peaks from a CSV file. Parse the chromosome, start, end, and center
    for each peak.

    Returns
    -------
    atac_df : pd.DataFrame
        The raw ATAC data with the first column containing the peak positions.
    peak_df : pd.DataFrame
        The chromosome, start, end, and center of each peak.
    """
    atac_df = pd.read_csv(atac_data_file, sep=",", header=0, index_col=None)
    peak_pos = atac_df[atac_df.columns[0]].tolist()

    peak_df = pd.DataFrame()
    peak_df["peak_full"] = peak_pos
    peak_df["chr"] = [pos.split(":")[0].replace("chr", "") for pos in peak_pos]
    peak_df["start"] = [int(pos.split(":")[1].split("-")[0]) for pos in peak_pos]
    peak_df["end"] = [int(pos.split(":")[1].split("-")[1]) for pos in peak_pos]

    # Find the center of the peak
    peak_df["center"] = peak_df["end"] - ((peak_df["end"] - peak_df["start"]) / 2)
    return atac_df, peak_df


def build_peak_kd_tree_dict(peak_df: pd.DataFrame) -> dict:
    """
    Build a dictionary mapping each chromosome to a KDTree (based on peak centers)
    and the corresponding index list.

    Returns
    -------
    tree_dict : dict
        key: chromosome (str), value: (KDTree object, list of original peak indices)
    """
    tree_dict = {}
    for chrom, group in peak_df.groupby("chr"):
        centers = group["center"].values.reshape(-1, 1)
        tree = cKDTree(centers)
        tree_dict[chrom] = (tree, group.index.tolist())
    return tree_dict


def find_genes_near_peaks(
    tss, chrom: str, tree_dict: dict, threshold: int = 10_000
) -> list:
    """
    Given a transcription start site (tss) and chromosome, find all peak indices
    within `threshold` distance using a KDTree.

    Parameters
    ----------
    tss : float or list[float]
        Transcription start site position. If it's a list, we'll use the first value.
    chrom : str
        Chromosome name.
    tree_dict : dict
        Dictionary mapping chromosome -> (KDTree, list of peak indices).
    threshold : int
        The maximum distance from the TSS to consider a peak "in range".

    Returns
    -------
    list
        A list of peak indices within the specified threshold.
    """
    if chrom not in tree_dict:
        return []
    tree, idx_list = tree_dict[chrom]

    # If TSS is an array/list, use the first element
    tss_val = float(tss[0]) if isinstance(tss, (list, np.ndarray)) else float(tss)

    # Query the KDTree for all peaks within 'threshold' of the TSS
    indices = tree.query_ball_point(np.array([[tss_val]]), r=threshold)[0]
    # Convert KDTree indices back to original peak_df indices
    return [idx_list[i] for i in indices]


def map_peaks_to_ensembl_genes(
    tree_dict: dict, ensembl_df: pd.DataFrame, peak_dist_limit: int
) -> pd.DataFrame:
    """
    Identify which peaks fall within a certain distance of each gene's TSS.
    Only keep genes that have at least one peak in range.

    Returns
    -------
    ensembl_df : pd.DataFrame
        Updated DataFrame with an additional 'peaks_in_range' column containing
        a list of peak indices that fall within `peak_dist_limit` of that gene.
    """
    print(f"Identifying ATAC-seq peaks within {peak_dist_limit} of gene TSS using KDTree.")
    ensembl_df["peaks_in_range"] = ensembl_df.apply(
        lambda row: find_genes_near_peaks(
            row["Transcription start site (TSS)"],
            row["Chromosome/scaffold name"],
            tree_dict,
            threshold=peak_dist_limit,
        ),
        axis=1,
    )
    # Filter out genes with no peaks
    ensembl_df = ensembl_df[ensembl_df["peaks_in_range"].apply(len) > 0]
    return ensembl_df


def map_peaks_to_known_enhancers(tree_dict: dict, enhancer_db_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each enhancer in enhancer_db, find all peaks in range. Then compute
    the average enhancer score for each peak.

    Returns
    -------
    enhancer_scores_df : pd.DataFrame
        DataFrame with columns ['peak_index', 'score'] representing the average
        enhancer score for each peak that had at least one enhancer in range.
    """
    # Find peaks within known enhancer regions
    enhancer_db_df["peaks_in_range"] = enhancer_db_df.apply(
        lambda row: enhancerdb.find_peaks_in_enhancer(row, tree_dict), axis=1
    )

    # Remove enhancers with no peaks or NaN scores
    enhancer_db_df = enhancer_db_df[enhancer_db_df["peaks_in_range"].map(len) > 0].dropna()

    print("Num enhancers with mapped peaks by chromosome:")
    for chrom, group in enhancer_db_df.groupby("chr"):
        print(f"\t{chrom} = {group.shape[0]} mapped enhancers")

    # Collect scores for each peak
    peak_scores = defaultdict(list)
    for _, row in enhancer_db_df.iterrows():
        score = row["score"]
        for peak in row["peaks_in_range"]:
            peak_scores[peak].append(score)

    # Average score per peak (a peak can have multiple scores for different tissues)
    peak_avg = {peak: np.mean(scores) for peak, scores in peak_scores.items()}

    # Convert the dictionary to a DataFrame
    enhancer_scores_df = pd.DataFrame(list(peak_avg.items()), columns=["peak_index", "score"])
    return enhancer_scores_df


# ------------------------- CORRELATIONS ------------------------- #
def calculate_correlations(
    gene_row: pd.Series, rna_df_indexed: pd.DataFrame, atac_df: pd.DataFrame
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Calculate peak-to-gene and peak-to-peak correlations for a given gene row.

    Parameters
    ----------
    gene_row : pd.Series
        A row from the DataFrame that contains 'Gene name' and 'peaks_in_range'.
    rna_df_indexed : pd.DataFrame
        RNA-seq data with the index set to gene names.
    atac_df : pd.DataFrame
        ATAC-seq data, where the first column is peak identifiers, and subsequent
        columns are numeric accessibility data.

    Returns
    -------
    gene_peak_df : pd.DataFrame or None
        Each row is (Gene, Peak, Correlation) for each valid peak in range.
    peak_peak_df : pd.DataFrame or None
        Each row is (Peak1, Peak2, Correlation) for each unique peak pair.
    """
    gene_name = gene_row["Gene name"]
    peak_indices = gene_row["peaks_in_range"]

    # Get gene expression vector for this gene
    try:
        gene_expr = rna_df_indexed.loc[gene_name].astype(float)
    except KeyError:
        return None, None

    # Extract ATAC-seq data for only the peaks in range
    # (First column is non-numeric, so skip it)
    selected_atac = atac_df.iloc[peak_indices, 1:].astype(float)
    # Filter out peaks with zero total accessibility
    selected_atac = selected_atac[selected_atac.sum(axis=1) > 0]
    if selected_atac.empty:
        return None, None

    # 1) Peak-to-gene correlation
    peak_to_gene_corr = selected_atac.transpose().corrwith(gene_expr).fillna(0)
    gene_peak_df = pd.DataFrame(
        {
            "Gene": gene_name,
            "Peak": peak_to_gene_corr.index,
            "Correlation": peak_to_gene_corr.values,
        }
    )

    # 2) Peak-to-peak correlations (only if we have at least 2 peaks)
    if selected_atac.shape[0] > 1:
        corr_matrix = selected_atac.transpose().corr().fillna(0).values
        peaks_idx = selected_atac.index.tolist()

        iu = np.triu_indices_from(corr_matrix, k=1)
        peak_pairs = [(peaks_idx[i], peaks_idx[j]) for i, j in zip(iu[0], iu[1])]
        corr_values = corr_matrix[iu]

        peak_peak_df = pd.DataFrame(
            {
                "Peak1": [pair[0] for pair in peak_pairs],
                "Peak2": [pair[1] for pair in peak_pairs],
                "Correlation": corr_values,
            }
        )
    else:
        peak_peak_df = pd.DataFrame(columns=["Peak1", "Peak2", "Correlation"])

    return gene_peak_df, peak_peak_df


def calculate_correlations_parallelized(
    genes_df: pd.DataFrame,
    rna_df: pd.DataFrame,
    atac_df: pd.DataFrame,
    gene_range,
    n_jobs: int = 8,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate peak-to-gene and peak-to-peak correlations for a given range of genes using parallel processing.

    Returns
    -------
    total_gene_peak_df : pd.DataFrame
        Aggregated gene-to-peak correlations.
    total_peak_peak_df : pd.DataFrame
        Aggregated peak-to-peak correlations.
    """
    rna_df_indexed = rna_df.set_index("gene")

    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_correlations)(genes_df.iloc[i], rna_df_indexed, atac_df) for i in tqdm(gene_range)
    )

    gene_peak_dfs = [res[0] for res in results if res[0] is not None]
    peak_peak_dfs = [res[1] for res in results if res[1] is not None and not res[1].empty]

    total_gene_peak_df = pd.concat(gene_peak_dfs, ignore_index=True) if gene_peak_dfs else pd.DataFrame()
    total_peak_peak_df = pd.concat(peak_peak_dfs, ignore_index=True) if peak_peak_dfs else pd.DataFrame()

    total_gene_peak_df = total_gene_peak_df.drop_duplicates()
    total_peak_peak_df = total_peak_peak_df.drop_duplicates()

    return total_gene_peak_df, total_peak_peak_df


# ------------------------- STATISTICAL TESTING ------------------------- #
def correlation_pvals_vec(r: np.ndarray, n: int) -> np.ndarray:
    """
    Compute two-sided p-values for an array of Pearson correlation coefficients r
    given sample size n using the t-distribution.
    """
    # Avoid division by zero by clamping r slightly away from ±1
    r_clamped = np.clip(r, -0.9999999999, 0.9999999999)

    # t-statistic
    t_vals = r_clamped * np.sqrt((n - 2) / (1.0 - r_clamped**2))
    pvals = 2.0 * stats.t.sf(np.abs(t_vals), df=n - 2)
    return pvals


def compute_significance(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Given a DataFrame with columns 'Correlation', compute p-values and
    FDR-corrected q-values for each row.

    Returns
    -------
    df : pd.DataFrame
        The same DataFrame with additional columns ['pval', 'qval'].
    """
    corrs = df["Correlation"].values
    pvals = correlation_pvals_vec(corrs, n)

    # FDR correction (Benjamini-Hochberg)
    _, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    df = df.copy()
    df["pval"] = pvals
    df["qval"] = qvals
    return df


def filter_significant_correlations(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Filter correlations by p < 0.05, returning a DataFrame with an additional
    'log2_pval' column.
    """
    df = compute_significance(df, n)
    sig_df = df[df["pval"] < 0.05].copy()
    sig_df["log2_pval"] = np.log2(sig_df["pval"] + 1e-10)
    return sig_df


# ------------------------- COMBINING RESULTS ------------------------- #
def combine_peak_to_peak_with_peak_to_gene(
    peak_to_peak_df: pd.DataFrame, peak_to_gene_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine peak-to-peak correlations with peak-to-gene data.

    - Keep only peak-to-peak rows where at least one peak is in the peak-to-gene DataFrame.
    - Assign gene info to each peak in peak-to-peak, then explode so that each gene is on its own row.
    """
    print("\tRemoving peak-to-peak interactions where neither peak is associated with a gene.")
    filtered_p2p = peak_to_peak_df[
        peak_to_peak_df["peak1_full"].isin(peak_to_gene_df["peak_full"])
        | peak_to_peak_df["peak2_full"].isin(peak_to_gene_df["peak_full"])
    ]

    print("\tAssigning peak1 with matching genes for peak2.")
    merged_p1 = pd.merge(
        filtered_p2p,
        peak_to_gene_df[["peak_full", "Gene"]],
        left_on="peak1_full",
        right_on="peak_full",
        how="inner",
    ).rename(columns={"Gene": "Gene_from_peak1"}).drop(columns=["peak_full"])

    print("\tAssigning peak2 with matching genes for peak1.")
    merged_p2 = pd.merge(
        filtered_p2p,
        peak_to_gene_df[["peak_full", "Gene"]],
        left_on="peak2_full",
        right_on="peak_full",
        how="inner",
    ).rename(columns={"Gene": "Gene_from_peak2"}).drop(columns=["peak_full"])

    print("\tCombining gene mappings for peak1 and peak2.")
    combined_long = pd.concat([merged_p1, merged_p2], ignore_index=True)

    # Merge gene columns into one list
    combined_long["Gene_list"] = combined_long[["Gene_from_peak1", "Gene_from_peak2"]].apply(
        lambda row: row.dropna().tolist(), axis=1
    )

    # Explode so each gene gets its own row
    print("\tCreating list of peaks for each gene.")
    combined_long = combined_long.explode("Gene_list").rename(columns={"Gene_list": "gene"})

    # Create separate DataFrames from peak1 and peak2 columns
    df1 = combined_long[["peak1_full", "Correlation", "Gene_from_peak1"]].rename(
        columns={"peak1_full": "peak", "Correlation": "score", "Gene_from_peak1": "gene"}
    )
    df2 = combined_long[["peak2_full", "Correlation", "Gene_from_peak2"]].rename(
        columns={"peak2_full": "peak", "Correlation": "score", "Gene_from_peak2": "gene"}
    )

    # Combine the two DataFrames
    melted_df = pd.concat([df1, df2], ignore_index=True)

    # Drop rows where 'gene' is NaN
    melted_df = melted_df.dropna(subset=["gene"])

    # Join with peak_to_gene_df on peak to retrieve the original peak-gene correlation
    joined_df = melted_df[["peak", "gene"]].merge(
        peak_to_gene_df[["peak_full", "Gene", "Correlation"]],
        left_on="peak",
        right_on="peak_full",
        how="left",
    ).dropna(subset=["Correlation"])

    # Create a mapping from peak -> correlation
    peak_to_corr = dict(zip(joined_df["peak"], joined_df["Correlation"]))

    # Replace the 'score' in melted_df with the mapped correlation if it exists
    mapped_scores = melted_df["peak"].map(peak_to_corr)
    melted_df["score"] = mapped_scores.fillna(melted_df["score"])

    return melted_df


def normalize_peak_to_peak_scores(df: pd.DataFrame) -> pd.Series:
    """
    Normalize peak-to-peak scores to the range [0, 1], leaving 0 and 1 untouched.
    """
    mask = (df["score"] != 0) & (df["score"] != 1)
    filtered_scores = df.loc[mask, "score"]

    if not filtered_scores.empty:
        score_min = filtered_scores.min()
        score_max = filtered_scores.max()

        if score_max == score_min:
            # All non-0/1 scores are identical -> set them to 0 (or another default)
            score_normalized = np.where(mask, 0, df["score"])
        else:
            # Linearly map scores to [0, 1]
            score_normalized = np.where(
                mask,
                (df["score"] - score_min) / (score_max - score_min),
                df["score"],
            )
    else:
        # All scores are 0 or 1; no normalization needed
        score_normalized = df["score"]

    return score_normalized


# ------------------------- PLOTTING (OPTIONAL UTILS) ------------------------- #
def plot_column_histogram(df: pd.DataFrame, colname: str, title: str):
    """
    Utility function to show a histogram of values from 'df[colname]'.
    """
    plt.figure(figsize=(5, 5))
    plt.hist(df[colname], bins=30, edgecolor="k")
    plt.title(title)
    plt.xlabel(f"{colname} Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


# ------------------------- MAIN SCRIPT ------------------------- #
def main():
    # 1) Load scRNA-seq data
    print("Loading the scRNA-seq dataset.")
    rna_df = pd.read_csv(RNA_DATA_FILE, sep=",", header=0, index_col=None)
    rna_df = rna_df.rename(columns={rna_df.columns[0]: "gene"})

    # 2) Load and parse ATAC peak positions
    print("Loading and parsing ATAC peak positions.")
    atac_df, peak_df = load_and_parse_atac_peaks(ATAC_DATA_FILE)

    # 3) Normalize ATAC-seq data (log2 CPM)
    print("Log2 CPM normalizing the ATAC-seq dataset.")
    atac_df = normalization.log2_cpm_normalize(atac_df)

    # 4) Load enhancer database
    print("Loading enhancer database.")
    enhancer_db_df = enhancerdb.load_enhancer_database_file(ENHANCER_DB_FILE)

    # 5) Load ENSEMBL genes for the given organism
    print(f"Loading ENSEMBL genes for {ORGANISM}.")
    ensembl_gene_df = ensembl.retrieve_ensembl_gene_positions(ORGANISM)

    # 6) Filter ENSEMBL genes to match the RNA and peak chromosomes
    print("Filtering ENSEMBL genes to match RNA data and ATAC chromosomes.")
    ensembl_gene_df = ensembl_gene_df[ensembl_gene_df["Gene name"].isin(rna_df["gene"])].dropna()
    ensembl_gene_df = ensembl_gene_df[ensembl_gene_df["Chromosome/scaffold name"].isin(peak_df["chr"])].dropna()

    # 7) Build KD-Tree for peaks and map them to ENSEMBL genes
    print("Building KD-Tree for peaks.")
    tree_dict = build_peak_kd_tree_dict(peak_df)
    ensembl_gene_df = map_peaks_to_ensembl_genes(tree_dict, ensembl_gene_df, PEAK_DIST_LIMIT)

    # 8) Map peaks to known enhancers
    enhancer_scores_df = map_peaks_to_known_enhancers(tree_dict, enhancer_db_df)

    # 9) Calculate peak-to-gene and peak-to-peak correlations in parallel (subset of genes)
    print("Calculating peak-to-peak and peak-to-gene correlations.")
    gene_indices = range(1, 5000)  # Example range
    total_gene_peak_df, total_peak_peak_df = calculate_correlations_parallelized(
        ensembl_gene_df, rna_df, atac_df, gene_indices, n_jobs=N_JOBS
    )

    # 10) Filter for significant correlations
    print("Calculating significance of peak-to-gene correlations.")
    sig_gene_peak_df = filter_significant_correlations(total_gene_peak_df, total_gene_peak_df.shape[0])

    print("Calculating significance of peak-to-peak correlations.")
    sig_peak_peak_df = filter_significant_correlations(total_peak_peak_df, total_peak_peak_df.shape[0])

    # 11) Map the peak indices to their full location strings
    sig_gene_peak_df["peak_full"] = sig_gene_peak_df["Peak"].map(peak_df["peak_full"])
    sig_peak_peak_df["peak1_full"] = sig_peak_peak_df["Peak1"].map(peak_df["peak_full"])
    sig_peak_peak_df["peak2_full"] = sig_peak_peak_df["Peak2"].map(peak_df["peak_full"])

    # 12) Map the enhancer scores to each peak
    sig_gene_peak_df["enhancer_score"] = sig_gene_peak_df["Peak"].map(enhancer_scores_df["score"]).fillna(0)
    sig_peak_peak_df["peak1_enhancer_score"] = sig_peak_peak_df["Peak1"].map(enhancer_scores_df["score"]).fillna(0)
    sig_peak_peak_df["peak2_enhancer_score"] = sig_peak_peak_df["Peak2"].map(enhancer_scores_df["score"]).fillna(0)

    # 13) Prepare smaller DataFrames for output
    peak_gene_format_df = sig_gene_peak_df[["peak_full", "Gene", "Correlation", "enhancer_score"]]
    peak_peak_format_df = sig_peak_peak_df[
        ["peak1_full", "peak2_full", "Correlation", "peak1_enhancer_score", "peak2_enhancer_score"]
    ]

    # 14) Combine peak-to-peak with peak-to-gene
    print("Combining peak-to-peak with peak-to-gene scores.")
    melted_df = combine_peak_to_peak_with_peak_to_gene(peak_peak_format_df, peak_gene_format_df)

    # 15) Normalize final scores to [0,1]
    print("Normalizing peak-to-peak scores between 0 and 1.")
    melted_df["score"] = normalize_peak_to_peak_scores(melted_df)

    print("Preview of combined results:")
    print(melted_df.head())

    # Example: Save final table
    # out_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/K562/K562_human_filtered/peak_to_tg_scores.csv"
    # melted_df.to_csv(out_file, sep="\t", header=True, index=False)
    # print(f"Final results saved to {out_file}")


if __name__ == "__main__":
    main()