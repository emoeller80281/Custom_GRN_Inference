import pybedtools
import pandas as pd
import os

kidney_gt_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/ground_truth_files/Oth.Kid.05.AllAg.AllCell.bed"

# Skip the UCSC track header line
df = pd.read_csv(
    kidney_gt_file,
    sep="\t",
    header=None,
    skiprows=1,           # <- Skip the 'track name=...' header
    names=[
        "chrom", "start", "end", "info", "score", "strand",
        "thickStart", "thickEnd", "rgb"
    ],
    on_bad_lines="skip",
)

# Extract the TF name from the 'info' column using regex
df["name"] = df["info"].str.extract(r'Name=([^%]+)')

ref_df_bed = df[["chrom", "start", "end", "name", "strand"]]

gene_pos_df = pd.read_csv(
    "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/ground_truth_files/gencode.vM37.chr_patch_hapl_scaff.annotation.gtf", 
    sep="\t", 
    comment="#", 
    names=["chr", "annot_source", "feature", "start", "stop", "score", "strand", "phase", "additional_info"],
    header=None,
    )

genes = gene_pos_df[gene_pos_df["feature"] == "gene"].copy()

# Step 3: Extract gene_id from 'additional_info'
genes["gene_id"] = genes["additional_info"].str.extract(r'gene_id "([^"]+)"')
genes["gene_name"] = genes["additional_info"].str.extract(r'gene_name "([^"]+)"')

# Step 4: Build BED-like DataFrame
genes_bed = genes[["chr", "start", "stop", "gene_name", "strand"]].copy()
genes_bed.columns = ["chrom", "start", "end", "name", "strand"]

# Sort both DataFrames the same way
ref_df_bed = ref_df_bed.sort_values(by=["chrom", "start", "end"])
genes_bed = genes_bed.sort_values(by=["chrom", "start", "end"])

# Then convert to BedTool
ref_bedtool = pybedtools.BedTool.from_dataframe(ref_df_bed)
gene_bedtool = pybedtools.BedTool.from_dataframe(genes_bed)

# Step 6: Use bedtools closest
closest = ref_bedtool.closest(gene_bedtool, D="b")  # D="b" gives signed distance

# Each record now has original TF info + nearest gene info + distance
cols = [
    "tf_chr", "tf_start", "tf_end", "tf_name", "tf_strand",
    "gene_chr", "gene_start", "gene_end", "gene_name", "gene_strand", "distance"
]
closest_df = closest.to_dataframe(names=cols)

gt = closest_df[["tf_name", "gene_name"]]
gt = gt.rename(columns={"tf_name":"Source", "gene_name":"Target"})

gt_output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/ground_truth_files"
gt_save_path = os.path.join(gt_output_dir, "kidney_chipatlas.tsv")
gt.to_csv(gt_save_path, sep="\t", header=True, index=False)