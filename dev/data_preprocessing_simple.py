import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from pathlib import Path
import logging

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed" / "mESC_preprocessing_testing" / "E7.5_rep1"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")

MIN_GENES_PER_CELL = 1000
MIN_PEAKS_PER_CELL = 1000
FILTER_OUT_LOWEST_PCT_GENES = 0.10
FILTER_OUT_LOWEST_PCT_PEAKS = 0.10
MIN_RNA_DISP = 0.5
PCA_COMPONENTS = 20

def normalize_barcodes(index_like: pd.Index) -> pd.Index:
    ix = pd.Index(index_like).astype(str)
    ix = ix.str.replace(r"_\d+$", "", regex=True)
    ix = ix.str.replace(r"[-\.]\d+$", "", regex=True)
    ix = ix.str.replace(r"(?:_RNA|_GEX|_ATAC|#GEX|#ATAC)$", "", regex=True, case=False)
    return ix.str.upper()

def make_adata_from_df(df: pd.DataFrame) -> AnnData:
    return AnnData(
        X=sp.csr_matrix(df.to_numpy()),
        obs=pd.DataFrame(index=df.index.astype(str)),
        var=pd.DataFrame(index=df.columns.astype(str)),
    )
    
organism_code = "mm10"
tf_file = PROJECT_DIR / "data" / "databases" / "motif_information" / organism_code / "TF_Information_all_motifs.txt"
tf_ref = pd.read_csv(tf_file, sep=None, engine="python")["TF_Name"].unique()
print(f"First 5 TFs for mESC:")
print(tf_ref[:5])

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/")

sample_name = "E7.5_rep1"
raw_rna_file = DATA_DIR / "raw" / "mESC" / sample_name / "scRNA_seq_raw.parquet"
raw_atac_file = DATA_DIR / "raw" / "mESC" / sample_name / "scATAC_seq_raw.parquet"

raw_rna_df = pd.read_parquet(raw_rna_file, engine="pyarrow")
raw_atac_df = pd.read_parquet(raw_atac_file, engine="pyarrow")

print(f"Raw RNA-seq data shape: {raw_rna_df.shape}")
print(f"Raw ATAC-seq data shape: {raw_atac_df.shape}")

# Load data
raw_rna_df = pd.read_parquet(raw_rna_file, engine="pyarrow")
raw_atac_df = pd.read_parquet(raw_atac_file, engine="pyarrow")

ad_rna = make_adata_from_df(raw_rna_df)
ad_atac = make_adata_from_df(raw_atac_df)

ad_rna.var_names_make_unique()
ad_atac.var_names_make_unique()

# Align cells
rna_norm = normalize_barcodes(ad_rna.obs_names)
atac_norm = normalize_barcodes(ad_atac.obs_names)

if rna_norm.duplicated().any():
    raise ValueError("Duplicate normalized RNA barcodes detected.")
if atac_norm.duplicated().any():
    raise ValueError("Duplicate normalized ATAC barcodes detected.")

r_map = pd.Series(ad_rna.obs_names, index=rna_norm, dtype="object")
a_map = pd.Series(ad_atac.obs_names, index=atac_norm, dtype="object")
common = r_map.index.intersection(a_map.index)

ad_rna = ad_rna[r_map.loc[common].values, :].copy()
ad_atac = ad_atac[a_map.loc[common].values, :].copy()
ad_rna.obs_names = common
ad_atac.obs_names = common

# Save raw counts
ad_rna.layers["counts"] = ad_rna.X.copy()
ad_atac.layers["counts"] = ad_atac.X.copy()

# RNA QC annotations
gene_lower = ad_rna.var_names.str.lower()
ad_rna.var["mt"] = gene_lower.str.startswith("mt-")
ad_rna.var["ribo"] = gene_lower.str.startswith(("rps", "rpl"))
ad_rna.var["hb"] = gene_lower.str.match(r"^hb(?!p)")

sc.pp.calculate_qc_metrics(ad_rna, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)

# ATAC QC metrics
ad_atac.obs["n_peaks_by_counts"] = np.asarray((ad_atac.X > 0).sum(axis=1)).ravel()
ad_atac.obs["total_atac_counts"] = np.asarray(ad_atac.X.sum(axis=1)).ravel()

# Cell filtering
ad_rna = ad_rna[
    (ad_rna.obs["pct_counts_mt"] < 20) &
    (ad_rna.obs["n_genes_by_counts"] >= MIN_GENES_PER_CELL),
].copy()

ad_atac = ad_atac[
    ad_atac.obs["n_peaks_by_counts"] >= MIN_PEAKS_PER_CELL,
].copy()

# RNA doublets
sc.pp.scrublet(ad_rna)
ad_rna = ad_rna[~ad_rna.obs["predicted_doublet"], :].copy()

# Re-sync paired cells
common_cells = ad_rna.obs_names.intersection(ad_atac.obs_names)
ad_rna = ad_rna[common_cells, :].copy()
ad_atac = ad_atac[common_cells, :].copy()

# Feature filtering
min_num_cells_rna = int(np.ceil(ad_rna.n_obs * FILTER_OUT_LOWEST_PCT_GENES))
min_num_cells_atac = int(np.ceil(ad_atac.n_obs * FILTER_OUT_LOWEST_PCT_PEAKS))

sc.pp.filter_genes(ad_rna, min_cells=min_num_cells_rna)
sc.pp.filter_genes(ad_atac, min_cells=min_num_cells_atac)

# RNA preprocessing
sc.pp.normalize_total(ad_rna, target_sum=1e4)
sc.pp.log1p(ad_rna)
ad_rna.layers["log1p"] = ad_rna.X.copy()

sc.pp.highly_variable_genes(ad_rna, min_mean=0.0125, max_mean=3, min_disp=MIN_RNA_DISP)
sc.tl.pca(ad_rna, n_comps=PCA_COMPONENTS, use_highly_variable=True, svd_solver="arpack")

