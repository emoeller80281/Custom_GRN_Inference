import os
import logging
import pybedtools
import pandas as pd

from grn_inference.pipeline.preprocess_datasets import (
    extract_atac_peaks_near_rna_genes
)

organism = "mmusculus"
output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/testing"
tmp_dir = os.path.join(output_dir, "tmp")
tss_distance_cutoff = 1_000_000

rna_df = pd.read_parquet("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC/DS011_mESC_sample1/DS011_mESC_RNA_processed.parquet", engine="pyarrow")
atac_df = pd.read_parquet("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC/DS011_mESC_sample1/DS011_mESC_ATAC_processed.parquet", engine="pyarrow")


peaks_near_genes_df = extract_atac_peaks_near_rna_genes(atac_df, rna_df, organism, tss_distance_cutoff, output_dir)

print(peaks_near_genes_df)

