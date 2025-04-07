import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
    argparse.Namespace: Parsed arguments containing paths for input and output files.
    """
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")

    parser.add_argument(
        "--fig_dir",
        type=str,
        required=True,
        help="Path to the figure directory for the sample"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
    )
    
    args: argparse.Namespace = parser.parse_args()

    return args

def normalize_peak_to_peak_scores(df, colname):
    # Identify scores that are not 0 or 1
    mask = (df[colname] != 0) & (df[colname] != 1)
    filtered_scores = df.loc[mask, colname]

    if not filtered_scores.empty:
        # Compute min and max of non-0/1 scores
        score_min = filtered_scores.min()
        score_max = filtered_scores.max()
        
        # Handle edge case where all non-0/1 scores are the same
        if score_max == score_min:
            # Set all non-0/1 scores to 0 (or another default value)
            score_normalized = np.where(mask, 0, df[colname])
        else:
            # Normalize non-0/1 scores and retain 0/1 values
            score_normalized = np.where(
                mask,
                (df[colname] - score_min) / (score_max - score_min),
                df[colname]
            )
    else:
        # All scores are 0 or 1; no normalization needed
        score_normalized = df[colname]
    
    return score_normalized

def plot_peak_to_tg_scores(score_col, fig_dir):
    plt.hist(score_col, bins=50, color='blue')
    plt.title("Normalized Cicero Peak to Gene Association Score Distribution", fontsize=16)
    plt.xlabel("Normalized Peak to Gene Association Score")
    plt.ylabel("Frequency")
    plt.savefig(f"{fig_dir}/cicero_hist_peak_gene_association_norm.png", dpi=300)
    plt.close()

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()
    output_dir: str = args.output_dir
    fig_dir: str = args.fig_dir
    
    # # Alternatively: if you want to pass arguments manually
    # output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC"
    # fig_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/figures"
    
    cicero_peak_to_peak_file = f"{output_dir}/cicero_peak_to_peak.csv"
    cicero_peak_to_gene_file = f"{output_dir}/cicero_peak_to_gene.csv"

    peak_to_peak = pd.read_csv(cicero_peak_to_peak_file, header=0, index_col=None)
    peak_to_gene = pd.read_csv(cicero_peak_to_gene_file, header=0, index_col=0)

    # Merge matching peaks to get a single dataframe
    merged_peaks = pd.merge(peak_to_peak, peak_to_gene, how="outer", left_on=["Peak1", "Peak2"], right_on=[peak_to_gene.index, "site_name"])
    merged_peaks = merged_peaks.rename(columns={"coaccess": "score"})

    # Remove edges between peaks with no coaccessibility
    merged_peaks = merged_peaks[merged_peaks["score"] != 0]

    # Extract only rows where peaks have an associated gene
    promoter_peaks = merged_peaks[["Peak1", "gene"]].dropna()

    # Set scores to 1 for peaks from peak_to_gene (columns without a score but with a gene)
    merged_peaks.loc[merged_peaks["score"].isna() & merged_peaks["gene"].notna(), ['score']] = 1

    # Merge to set associated genes for the peak-to-peak scores (which dont have genes) based on if Peak1 is in promoter_peaks
    merged_with_promoter_genes = pd.merge(left=merged_peaks, right=promoter_peaks, on="Peak1", how="right")

    # Combine the gene associations for each row, removing any peaks not associated with a gene
    merged_with_promoter_genes["gene"] = merged_with_promoter_genes["gene_x"].fillna(merged_with_promoter_genes["gene_y"])

    # Peaks for the promoter sequences are in Peak1, with no value for Peak2. Set Peak2 as a copy of Peak1 for peak_to_gene rows
    merged_with_promoter_genes["Peak2"] = merged_with_promoter_genes["Peak2"].fillna(merged_with_promoter_genes["Peak1"])

    # Only keep Peak2 rows, as Peak1 contains promoter peaks and can have duplicates, but Peak2 does not.
    # As we set Peak2 for the peak_to_gene rows, we retain the peaks in the promoters
    merged_with_promoter_genes = merged_with_promoter_genes.rename(columns={"Peak2": "peak_id", "gene": "target_id", "score": "cicero_score"})
    merged_with_promoter_genes = merged_with_promoter_genes[["peak_id","target_id","cicero_score"]]
    
    # Format the peaks to chr:start-stop rather than chr_start_stop to match the ATACseq peaks
    merged_with_promoter_genes["peak_id"] = merged_with_promoter_genes["peak_id"].str.replace("_", "-")
    merged_with_promoter_genes["peak_id"] = merged_with_promoter_genes["peak_id"].str.replace("-", ":", 1)

    # Write the final merged peaks to a csv file
    merged_with_promoter_genes.to_csv(f"{output_dir}/cicero_peak_to_tg_scores.csv", header=True, index=False, sep="\t")

    # Plot the peak scores as a histogram
    plot_peak_to_tg_scores(merged_with_promoter_genes['cicero_score'], fig_dir)
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    main()