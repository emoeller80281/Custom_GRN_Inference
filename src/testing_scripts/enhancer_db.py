import pandas as pd
import logging
import argparse
import pybedtools
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
    argparse.Namespace: Parsed arguments containing paths for input and output files.
    """

    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")

    parser.add_argument(
        "--enhancer_db_file",
        type=str,
        required=True,
        help="Path to the EnhancerDB file"
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        required=True,
        help="Path to the tmp_dir for this sample"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
    )
    parser.add_argument(
        "--species",
        type=str,
        required=True,
        help="Species of the sample, either 'mouse', 'human', 'hg38', or 'mm10'"
    )
    
    args: argparse.Namespace = parser.parse_args()

    return args

# Parse arguments
args: argparse.Namespace = parse_args()

ENHANCER_DB_FILE = args.enhancer_db_file
OUTPUT_DIR = args.output_dir
TMP_DIR = f"{OUTPUT_DIR}/tmp"

def load_enhancer_database_file(enhancer_db_file, tmp_dir):
    enhancer_db = pd.read_csv(enhancer_db_file, sep="\t", header=None, index_col=None)
    enhancer_db = enhancer_db.rename(columns={
        0 : "chr",
        1 : "start",
        2 : "end",
        3 : "enhancer",
        4 : "tissue",
        5 : "R1_value",
        6 : "R2_value",
        7 : "R3_value",
        8 : "score"
    })
    
    # Remove the "chr" before chromosome number
    enhancer_db["chr"] = enhancer_db["chr"].str.replace("^chr", "", regex=True)
    
    # Average the score of an enhancer across all tissues / cell types
    enhancer_db = enhancer_db.groupby(["chr", "start", "end", "enhancer"], as_index=False)["score"].mean()

    enhancer_db["chr"] = enhancer_db["chr"].astype(str)
    enhancer_db["start"] = enhancer_db["start"].astype(int)
    enhancer_db["end"] = enhancer_db["end"].astype(int)
    enhancer_db["enhancer"] = enhancer_db["enhancer"].astype(str)
    
    enhancer_db = enhancer_db[["chr", "start", "end", "enhancer", "score"]]
    
    # Write the peak DataFrame to a file
    enhancer_db.to_csv(f"{tmp_dir}/enhancer.bed", sep="\t", header=False, index=False)
    
def find_peaks_in_known_enhancer_region(peak_bed, enh_bed):
    # 4) Find peaks that overlap with known enhancer locations from EnhancerDB
    logging.info("Locating peaks that overlap with known enhancer locations from EnhancerDB")
    peak_enh_overlap = peak_bed.intersect(enh_bed, wa=True, wb=True)
    peak_enh_overlap_df = peak_enh_overlap.to_dataframe(
        names=[
            "peak_chr", "peak_start", "peak_end", "peak_id",
            "enh_chr", "enh_start", "enh_end", "enh_id",
            "enh_score"  # only if you had a score column in your enhancers
        ]
    ).dropna()
    peak_enh_overlap_subset_df = peak_enh_overlap_df[["peak_id", "enh_score"]]
        
    return peak_enh_overlap_subset_df



# ============ MAPPING PEAKS TO KNOWN ENHANCERS ============
# Dataframe with "peak_id", "enh_id", and "enh_score" columns
if not os.path.exists(f"{TMP_DIR}/enhancer.bed"):
    logging.info("Loading known enhancer locations from EnhancerDB and saving as a bed file")
    load_enhancer_database_file(ENHANCER_DB_FILE, TMP_DIR)
else:
    logging.info("Enhancer BED file exists, loading...")

# Load in the peak and enhancer bed files
peak_bed = pybedtools.BedTool(f"{TMP_DIR}/peak_df.bed")
enh_bed = pybedtools.BedTool(f"{TMP_DIR}/enhancer.bed")

# Find the peaks that are in known enhancer regions
peak_enh_df = find_peaks_in_known_enhancer_region(peak_bed, enh_bed)

# Read in the peak to gene correlation dataframe
peak_gene_corr_df = pd.read_csv(f'{OUTPUT_DIR}/peak_to_gene_correlation.csv', sep="\t", header=0, index=None)

# Merge the peaks overlapping the known enhancers with the peak to gene correlation dataframe
logging.info("Merging the peak to gene mapping with the known enhancer location mapping")
peak_gene_df = pd.merge(peak_gene_corr_df, peak_enh_df, how="left", on="peak_id")
peak_gene_df[["enh_score"]] = peak_gene_df[["enh_score"]].fillna(value=0)
peak_gene_df = peak_gene_df[["peak_id", "gene_id", "correlation", "TSS_dist", "enh_score"]]

# Write out the final dataframe to the output directory
peak_gene_df.to_csv(f'{OUTPUT_DIR}/peak_to_gene_known_enhancer.csv', sep="\t", header=True, index=False)