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
        "--cicero_peak_to_gene_file",
        type=str,
        required=True,
        help="Path to the output peak to gene association file of Cicero"
    )

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

def normalize_peak_to_peak_scores(df):
    # Identify scores that are not 0 or 1
    mask = (df['score'] != 0) & (df['score'] != 1)
    filtered_scores = df.loc[mask, 'score']

    if not filtered_scores.empty:
        # Compute min and max of non-0/1 scores
        score_min = filtered_scores.min()
        score_max = filtered_scores.max()
        
        # Handle edge case where all non-0/1 scores are the same
        if score_max == score_min:
            # Set all non-0/1 scores to 0 (or another default value)
            score_normalized = np.where(mask, 0, df['score'])
        else:
            # Normalize non-0/1 scores and retain 0/1 values
            score_normalized = np.where(
                mask,
                (df['score'] - score_min) / (score_max - score_min),
                df['score']
            )
    else:
        # All scores are 0 or 1; no normalization needed
        score_normalized = df['score']
    
    return score_normalized

def plot_peak_to_tg_scores(score_col, fig_dir):
    plt.figure(figsize=(10, 7))
    plt.hist(score_col, bins=50, color='blue')
    plt.title("Cicero Peak to Gene Association Score Distribution", fontsize=16)
    plt.xlabel("Peak to Gene Association Score")
    plt.ylabel("Frequency")
    plt.savefig(f"{fig_dir}/cicero_hist_peak_gene_association.png", dpi=300)
    plt.close()

def plot_normalized_peak_to_tg_scores(score_col, fig_dir):
    plt.hist(score_col, bins=50, color='blue')
    plt.title("Normalized Cicero Peak to Gene Association Score Distribution", fontsize=16)
    plt.xlabel("Normalized Peak to Gene Association Score")
    plt.ylabel("Frequency")
    plt.savefig(f"{fig_dir}/cicero_hist_peak_gene_association_norm.png", dpi=300)
    plt.close()

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()
    cicero_peak_to_gene_file: str = args.cicero_peak_to_gene_file
    fig_dir: str = args.fig_dir
    output_dir: str = args.output_dir
    
    # Alternatively: pass arguments manually
    cicero_peak_to_gene_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/peak_gene_associations.csv"
    output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/peak_gene_association_output.csv"
    fig_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/figures"
        
    df = pd.read_csv(cicero_peak_to_gene_file).sort_values("gene")

    # Plot the non-normalized peak scores as a histogram
    plot_peak_to_tg_scores(df['score'], fig_dir)

    # Normalize Cicero peak to peak scores between 0 to 1 (default -1 to 1)
    df['score_normalized'] = normalize_peak_to_peak_scores(df)
    plot_normalized_peak_to_tg_scores(df['score_normalized'], fig_dir)
    
    # Save the normalized peak to TG dataframe as a csv file
    peak_to_tg_score_output_file = f'{output_dir}/peak_to_tg_scores.csv'
    df.to_csv(peak_to_tg_score_output_file, index=False)
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()