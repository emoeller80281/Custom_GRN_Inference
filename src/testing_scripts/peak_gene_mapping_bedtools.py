import pandas as pd
import pybedtools
import numpy as np

# 1) Create or load peak_df and enhancer_df (each has [chr, start, end, <ID>, <score>])
# ------------------------- CONFIGURATIONS ------------------------- #
ORGANISM = "hsapiens"
ATAC_DATA_FILE = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/K562/K562_human_filtered/K562_human_filtered_ATAC.csv"
RNA_DATA_FILE =  "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/K562/K562_human_filtered/K562_human_filtered_RNA.csv"
ENHANCER_DB_FILE = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/enhancer_db/enhancer"
PEAK_DIST_LIMIT = 1_000_000

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
    peak_df["chr"] = [pos.split(":")[0].replace("chr", "") for pos in peak_pos]
    peak_df["start"] = [int(pos.split(":")[1].split("-")[0]) for pos in peak_pos]
    peak_df["end"] = [int(pos.split(":")[1].split("-")[1]) for pos in peak_pos]
    peak_df["peak_full"] = peak_pos

    return peak_df

def log2_cpm_normalize(df):
    """
    Log2 CPM normalize the values for each gene / peak.
    Assumes:
      - The df's first column is a non-numeric peak / gene identifier (e.g., "chr1:100-200"),
      - columns 1..end are numeric count data for samples or cells.
    """
    # Separate the non-numeric first column
    row_ids = df.iloc[:, 0]
    
    # Numeric counts
    counts = df.iloc[:, 1:]
    
    # 1. Compute library sizes (sum of each column)
    library_sizes = counts.sum(axis=0)
    
    # 2. Convert counts to CPM
    # Divide each column by its library size, multiply by 1e6
    # Add 1 to avoid log(0) issues in the next step
    cpm = (counts.div(library_sizes, axis=1) * 1e6).add(1)
    
    # 3. Log2 transform
    log2_cpm = np.log2(cpm)
    
    # Reassemble into a single DataFrame
    normalized_df = pd.concat([row_ids, log2_cpm], axis=1)
    
    return normalized_df

def load_enhancer_database_file(enhancer_db_file):
    enhancer_db = pd.read_csv(enhancer_db_file, sep="\t", header=None, index_col=None)
    enhancer_db = enhancer_db.rename(columns={
        0 : "chr",
        1 : "start",
        2 : "end",
        3 : "num_enh",
        4 : "tissue",
        5 : "R1_value",
        6 : "R2_value",
        7 : "R3_value",
        8 : "score"
    })
    
    # Average the score of an enhancer across all tissues / cell types
    enhancer_db = enhancer_db.groupby(["chr", "start", "end", "num_enh"], as_index=False)["score"].mean()

    enhancer_db["start"] = enhancer_db["start"].astype(int)
    enhancer_db["end"] = enhancer_db["end"].astype(int)
    enhancer_db = enhancer_db[["chr", "start", "end", "num_enh", "score"]]

    return enhancer_db

peak_df = load_and_parse_atac_peaks(ATAC_DATA_FILE)

enhancer_df = load_enhancer_database_file(ENHANCER_DB_FILE)

print(peak_df.head())

# 2) Convert to BedTool objects
peak_bed = pybedtools.BedTool.from_dataframe(peak_df)
enh_bed = pybedtools.BedTool.from_dataframe(enhancer_df)

# 3) Find overlapping intervals
overlap_result = peak_bed.intersect(enh_bed, wa=True, wb=True)

# 4) Convert the intersection result to a DataFrame
overlap_df = overlap_result.to_dataframe(
    names=["peak_chr", "peak_start", "peak_end", "peak_id",
           "enh_chr", "enh_start", "enh_end", "enh_id", "enh_score"]
)

# 5) Aggregate the enhancer scores for each peak
peak_enh_scores = (
    overlap_df
    .groupby("peak_id")["enh_score"]
    .agg(lambda x: np.mean(x) if len(x) > 0 else np.nan)
    .reset_index()
    .rename(columns={"enh_score": "avg_enhancer_score"})
)

# 6) (Optional) Merge back with your main peak/ATAC DataFrame
final_peaks = pd.merge(
    peak_df,
    peak_enh_scores,
    on="peak_id",
    how="left"
)

print(final_peaks.head())
