import scanpy as sc
import pandas as pd
import anndata
import os

# 1) Load in the RNAseq and ATACseq datasets

input_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input"

rna_data_path = os.path.join(input_dir, "DS011_mESC/DS011_mESC_sample1/DS011_mESC_RNA.parquet")
atac_data_path = os.path.join(input_dir, "DS011_mESC/DS011_mESC_sample1/DS011_mESC_ATAC.parquet")

rna_data_df = pd.read_parquet(rna_data_path, engine="pyarrow")
atac_data_path = pd.read_parquet(atac_data_path, engine="pyarrow")

