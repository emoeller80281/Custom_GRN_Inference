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
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
    )
    
    args: argparse.Namespace = parser.parse_args()

    return args

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()
    
    output_dir: str = args.output_dir
    
    cicero_peak_to_peak_file = f"{output_dir}/cicero_peak_to_peak.csv"
    cicero_peak_to_gene_file = f"{output_dir}/cicero_peak_to_gene.csv"

    logging.info("Loading 'cicero_peak_to_peak.csv'")
    peak_to_peak = pd.read_csv(cicero_peak_to_peak_file, header=0, index_col=None)
    
    logging.info("Loading 'cicero_peak_to_gene.csv'")
    peak_to_gene = pd.read_csv(cicero_peak_to_gene_file, header=0, index_col=0)

    # Merge matching peaks to get a single dataframes
    logging.info("Merging peak to peak with peak to gene scores")
    merged_peaks = pd.merge(peak_to_peak, peak_to_gene, how="outer", left_on=["Peak1", "Peak2"], right_on=[peak_to_gene.index, "site_name"])
    merged_peaks = merged_peaks.rename(columns={"coaccess": "score"})

    # Remove edges between peaks with no coaccessibility
    logging.info("Removing 0 scores")
    merged_peaks = merged_peaks[merged_peaks["score"] != 0]

    # Extract only rows where peaks have an associated gene
    logging.info("Isolating peaks that are within the promoter region")
    promoter_peaks = merged_peaks[["Peak1", "gene"]].dropna()

    # Set scores to 1 for peaks from peak_to_gene (columns without a score but with a gene)
    logging.info("Setting scores to 1 for peaks in the target gene promoter region, otherwise keeping the peak-to-peak score")
    merged_peaks.loc[merged_peaks["score"].isna() & merged_peaks["gene"].notna(), ['score']] = 1

    # Add target genes for the peak-to-peak scores if Peak1 is in promoter_peaks
    logging.info("Adding target genes to the peak-to-peak scores")
    merged_with_promoter_genes = pd.merge(left=merged_peaks, right=promoter_peaks, on="Peak1", how="right")

    # Combine the gene associations for each row, removing any peaks not associated with a gene
    logging.info("Combining gene associations and removing peaks not associated with a gene")
    merged_with_promoter_genes["gene"] = merged_with_promoter_genes["gene_x"].fillna(merged_with_promoter_genes["gene_y"])

    # Peaks for the promoter sequences are in Peak1, with no value for Peak2. Set Peak2 as a copy of Peak1 for peak_to_gene rows
    merged_with_promoter_genes["Peak2"] = merged_with_promoter_genes["Peak2"].fillna(merged_with_promoter_genes["Peak1"])

    # Only keep Peak2 rows, as Peak1 contains promoter peaks and can have duplicates, but Peak2 does not.
    # As we set Peak2 for the peak_to_gene rows, we retain the peaks in the promoters
    logging.info("Creating the final DataFrame with 'peak_id', 'target_id', and 'cicero_score' columns")
    merged_with_promoter_genes = merged_with_promoter_genes.rename(columns={"Peak2": "peak_id", "gene": "target_id", "score": "cicero_score"})
    merged_with_promoter_genes = merged_with_promoter_genes[["peak_id","target_id","cicero_score"]]
    
    # Format the peaks to chr:start-stop rather than chr_start_stop to match the ATACseq peaks
    merged_with_promoter_genes["peak_id"] = merged_with_promoter_genes["peak_id"].str.replace("_", "-")
    merged_with_promoter_genes["peak_id"] = merged_with_promoter_genes["peak_id"].str.replace("-", ":", 1)

    # Write the final merged peaks to a csv file
    logging.info("Writing Cicero DataFrame to: 'cicero_peak_to_tg_scores.csv'")
    merged_with_promoter_genes.to_csv(f"{output_dir}/cicero_peak_to_tg_scores.csv", header=True, index=False, sep="\t")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    main()