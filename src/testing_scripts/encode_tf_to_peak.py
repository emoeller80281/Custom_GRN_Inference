import pandas as pd

def find_peaks_in_known_enhancer_region(peak_bed, enh_bed):
    # 4) Find peaks that overlap with known enhancer locations from EnhancerDB
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

encode_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/encRegTfbsClusteredWithCells.hg38.bed"

encode_df = pd.read_csv(
    encode_file, 
    sep="\t", 
    header=None, 
    names=["chr", "start", "end", "source_id", "id", "cell_type"]
    ).astype("str")

encode_df['peak_id'] = encode_df['chr'] + ':' + encode_df['start'] + '-' + encode_df['end']

encode_df = encode_df[["peak_id", "source_id"]]
print(encode_df.head())

homer_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/K562/K562_human_filtered/homer_tf_to_peak.tsv"
homer_df = pd.read_csv(homer_file, sep="\t", header=0)
print(homer_df.head())

# Standardize ENCODE DataFrame
encode_df['source_id'] = encode_df['source_id'].str.upper().str.strip()
encode_df['peak_id'] = encode_df['peak_id'].str.upper().str.strip()

# Standardize Homer DataFrame
homer_df['source_id'] = homer_df['source_id'].astype(str).str.upper().str.strip()
homer_df['peak_id'] = homer_df['peak_id'].astype(str).str.upper().str.strip()


encode_pairs = set(zip(encode_df["source_id"], encode_df["peak_id"]))

homer_df["label"] = homer_df.apply(
    lambda row: 1 if (row["source_id"], row["peak_id"]) in encode_pairs else 0,
    axis=1
)

print(homer_df.head())
