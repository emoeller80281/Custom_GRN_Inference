import numpy as np
import pandas as pd
import scanpy as sc
import random
import os
import warnings
import scipy.sparse as sp
import logging
from anndata import AnnData
from typing import Tuple
import multiprocessing as mp

def load_rna_adata(sample_raw_data_dir: str) -> sc.AnnData:
    # Look for features file
    features = [f for f in os.listdir(sample_raw_data_dir) if f.endswith("features.tsv.gz")]
    assert len(features) == 1, f"Expected 1 features.tsv.gz, found {features}"

    prefix = features[0].replace("features.tsv.gz", "")
    logging.info(f"Detected RNA prefix: {prefix}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Only considering the two last:")
        adata = sc.read_10x_mtx(
            sample_raw_data_dir,
            var_names="gene_symbols",
            make_unique=True,
            prefix=prefix
        )
    return adata

def pseudo_bulk(
    rna_data: AnnData,
    atac_data: AnnData,
    use_single: bool = False,
    neighbors_k: int = 20,
    resolution: float = 0.5,
    aggregate: str = "mean"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate pseudobulk RNA and ATAC profiles by clustering cells and 
    aggregating their neighbors.
    """
    
    rna_data = rna_data.copy()
    atac_data = atac_data.copy()

    # Align ordering explicitly: enforce identical order
    atac_data = atac_data[rna_data.obs_names].copy()
    
    assert (rna_data.obs_names == atac_data.obs_names).all(), "Cell barcodes must be aligned"

    # --- Joint embedding ---
    combined_pca = np.concatenate(
        (rna_data.obsm["X_pca"], atac_data.obsm["X_pca"]), axis=1
    )
    rna_data.obsm["X_combined"] = combined_pca
    atac_data.obsm["X_combined"] = combined_pca

    # --- Build joint neighbors + Leiden clusters ---
    # Create empty placeholder matrix (not used, but AnnData requires X)
    X_placeholder = sp.csr_matrix((rna_data.n_obs, 0))  

    joint = AnnData(X=X_placeholder, obs=rna_data.obs.copy())
    joint.obsm['X_combined'] = combined_pca
    sc.pp.neighbors(joint, n_neighbors=neighbors_k, use_rep="X_combined")
    sc.tl.leiden(joint, resolution=resolution, key_added="cluster")

    clusters = joint.obs["cluster"].astype(str)
    cluster_ids = clusters.unique()

    # attach clusters back
    rna_data.obs["cluster"] = clusters
    atac_data.obs["cluster"] = clusters

    # --- Connectivity graph (sparse adjacency) ---
    conn = joint.obsp["connectivities"].tocsr()

    pseudo_bulk_rna = []
    pseudo_bulk_atac = []
    bulk_names = []
    
    def aggregate_matrix(X, idx, method):
        if method == "sum":
            return sp.csr_matrix(X[idx, :].sum(axis=0))
        elif method == "mean":
            return sp.csr_matrix(X[idx, :].mean(axis=0))
        else:
            raise ValueError(f"Unknown aggregate: {method}")

    for cid in cluster_ids:
        cluster_idx = np.where(clusters == cid)[0]

        if len(cluster_idx) == 0:
            continue
        
        if use_single or len(cluster_idx) < neighbors_k:
            # Single pseudobulk
            rna_agg = aggregate_matrix(rna_data.X, cluster_idx, aggregate)
            atac_agg = aggregate_matrix(atac_data.X, cluster_idx, aggregate)
            pseudo_bulk_rna.append(rna_agg)
            pseudo_bulk_atac.append(atac_agg)
            bulk_names.append(f"cluster{cid}")
        else:
            # Multiple pseudobulks
            seeds = np.random.choice(cluster_idx, size=int(np.sqrt(len(cluster_idx))), replace=False)
            for s in seeds:
                neighbors = conn[s].indices
                group = np.append(neighbors, s)
                rna_agg = aggregate_matrix(rna_data.X, group, aggregate)
                atac_agg = aggregate_matrix(atac_data.X, group, aggregate)
                pseudo_bulk_rna.append(rna_agg)
                pseudo_bulk_atac.append(atac_agg)
                bulk_names.append(f"cluster{cid}_cell{s}")
                
    pseudo_bulk_rna = [m if m.ndim == 2 else m.reshape(1, -1) for m in pseudo_bulk_rna]
    pseudo_bulk_atac = [m if m.ndim == 2 else m.reshape(1, -1) for m in pseudo_bulk_atac]   

    # --- Convert to DataFrames ---
    pseudo_bulk_rna = sp.vstack(pseudo_bulk_rna).T
    pseudo_bulk_atac = sp.vstack(pseudo_bulk_atac).T

    pseudo_bulk_rna = pd.DataFrame(
        pseudo_bulk_rna.toarray(),
        index=rna_data.var_names,
        columns=bulk_names,
    )
    pseudo_bulk_atac = pd.DataFrame(
        pseudo_bulk_atac.toarray(),
        index=atac_data.var_names,
        columns=bulk_names,
    )

    return pseudo_bulk_rna, pseudo_bulk_atac

def get_adata_from_peakmatrix(peak_matrix_file: str, label: pd.DataFrame, sample_name: str) -> AnnData:
    # Read header only
    all_cols = pd.read_csv(peak_matrix_file, sep="\t", nrows=10).columns
    
    # # Identify barcodes shared between RNA and ATAC
    # matching_barcodes = set(label["barcode_use"]) & set(all_cols)

    # # Map from original index -> normalized barcode
    # col_map = {i: bc for i, bc in enumerate(all_cols)}

    # # Always keep the first column (peak IDs)
    # keep_indices = [0] + [i for i, bc in col_map.items() if bc in matching_barcodes]

    # Read only those columns
    peak_matrix = pd.read_csv(
        peak_matrix_file,
        sep="\t",
        # usecols=keep_indices,
        index_col=0
    )

    # Replace column names with normalized barcodes
    # new_cols = [col_map[i] for i in keep_indices[1:]]
    # peak_matrix.columns = new_cols
    new_cols = peak_matrix.columns

    # Construct AnnData
    X = sp.csr_matrix(peak_matrix.values)
    adata_ATAC = AnnData(X=X.T)

    # Assign metadata
    adata_ATAC.obs_names = new_cols
    adata_ATAC.obs["barcode"] = new_cols
    adata_ATAC.obs["sample"] = sample_name
    adata_ATAC.obs["label"] = label.set_index("barcode_use").loc[new_cols, "label"].values

    adata_ATAC.var_names = peak_matrix.index
    adata_ATAC.var["gene_ids"] = peak_matrix.index

    return adata_ATAC

def filter_and_qc(adata_RNA: AnnData, adata_ATAC: AnnData) -> Tuple[AnnData, AnnData]:
    
    adata_RNA = adata_RNA.copy()
    adata_ATAC = adata_ATAC.copy()
    
    logging.info(f"[START] RNA shape={adata_RNA.shape}, ATAC shape={adata_ATAC.shape}")

    
    # Synchronize barcodes
    adata_RNA.obs['barcode'] = adata_RNA.obs_names
    common_barcodes = adata_RNA.obs['barcode'].isin(adata_ATAC.obs['barcode'])
    n_before = (adata_RNA.n_obs, adata_ATAC.n_obs)
    adata_RNA = adata_RNA[common_barcodes].copy()
    adata_ATAC = adata_ATAC[adata_ATAC.obs['barcode'].isin(adata_RNA.obs['barcode'])].copy()
    
    logging.info(
        f"[BARCODES] before sync RNA={n_before[0]}, ATAC={n_before[1]} → after sync RNA={adata_RNA.n_obs}, ATAC={adata_ATAC.n_obs}"
    )
    
    # --- QC and filtering ---
    adata_RNA.var['mt'] = adata_RNA.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata_RNA, qc_vars=["mt"], inplace=True)
    adata_RNA = adata_RNA[adata_RNA.obs.pct_counts_mt < 5].copy()
    adata_RNA.var_names_make_unique()
    adata_RNA.var['gene_ids'] = adata_RNA.var.index
        
    sc.pp.filter_cells(adata_RNA, min_genes=200)
    sc.pp.filter_genes(adata_RNA, min_cells=3)
    sc.pp.filter_cells(adata_ATAC, min_genes=200)
    sc.pp.filter_genes(adata_ATAC, min_cells=3)
    
    # --- Preprocess RNA ---
    sc.pp.normalize_total(adata_RNA, target_sum=1e4)
    sc.pp.log1p(adata_RNA)
    sc.pp.highly_variable_genes(adata_RNA, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable]
    sc.pp.scale(adata_RNA, max_value=10)
    sc.tl.pca(adata_RNA, n_comps=25, svd_solver="arpack")

    # --- Preprocess ATAC ---
    sc.pp.log1p(adata_ATAC)
    sc.pp.highly_variable_genes(adata_ATAC, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable]
    sc.pp.scale(adata_ATAC, max_value=10, zero_center=True)
    sc.tl.pca(adata_ATAC, n_comps=25, svd_solver="arpack")
    
    # --- After filtering to common barcodes ---
    common_barcodes = adata_RNA.obs_names.intersection(adata_ATAC.obs_names)
    
    adata_RNA = adata_RNA[common_barcodes].copy()
    adata_ATAC = adata_ATAC[common_barcodes].copy()
    
    return adata_RNA, adata_ATAC
    
def process_sample(sample_name: str):
    sample_data_dir = os.path.join(SAMPLE_INPUT_DIR, sample_name)
    os.makedirs(sample_data_dir, exist_ok=True)

    if os.path.exists(os.path.join(sample_data_dir, f"{sample_name}_RNA_qc.h5ad")) \
       and os.path.exists(os.path.join(sample_data_dir, f"{sample_name}_ATAC_qc.h5ad")):
        adata_RNA = sc.read_h5ad(os.path.join(sample_data_dir, f"{sample_name}_RNA_qc.h5ad"))
        adata_ATAC = sc.read_h5ad(os.path.join(sample_data_dir, f"{sample_name}_ATAC_qc.h5ad"))
    else:
        # --- load raw data ---
        sample_raw_data_dir = os.path.join(RAW_MESC_DATA_DIR, sample_name)
        adata_RNA = load_rna_adata(sample_raw_data_dir)
        adata_RNA.obs_names = [(sample_name + "." + i).replace("-", ".") for i in adata_RNA.obs_names]
        logging.info(f"[{sample_name}] Found {len(adata_RNA.obs_names)} RNA barcodes")

        label = pd.DataFrame({"barcode_use": adata_RNA.obs_names,
                              "label": ["mESC"] * len(adata_RNA.obs_names)})

        adata_ATAC = get_adata_from_peakmatrix(MESC_PEAK_MATRIX_FILE, label, sample_name)

        adata_RNA, adata_ATAC = filter_and_qc(adata_RNA, adata_ATAC)

        adata_RNA.write_h5ad(os.path.join(sample_data_dir, f"{sample_name}_RNA_qc.h5ad"))
        adata_ATAC.write_h5ad(os.path.join(sample_data_dir, f"{sample_name}_ATAC_qc.h5ad"))
    
    singlepseudobulk = adata_RNA.n_obs < 100

    # --- pseudo-bulk ---
    TG_pseudobulk, RE_pseudobulk = pseudo_bulk(
        adata_RNA, adata_ATAC, use_single=singlepseudobulk,
        neighbors_k=20, resolution=1.0
    )
    TG_pseudobulk = TG_pseudobulk.fillna(0)
    RE_pseudobulk = RE_pseudobulk.fillna(0)
    RE_pseudobulk[RE_pseudobulk > 100] = 100

    TG_pseudobulk.to_csv(os.path.join(sample_data_dir, "TG_pseudobulk.tsv"), sep="\t")
    RE_pseudobulk.to_csv(os.path.join(sample_data_dir, "RE_pseudobulk.tsv"), sep="\t")
    pd.DataFrame(adata_ATAC.var['gene_ids']).to_csv(os.path.join(sample_data_dir, "Peaks.txt"),
                                                    header=None, index=None)

    logging.info(f"[{sample_name}] Finished processing")
    return sample_name

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
# RAW_MESC_DATA_DIR = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS/DS014_DOI496239_MOUSE_ESC_RAW_FILES"
# MESC_PEAK_MATRIX_FILE = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS/DS014_DOI496239_MOUSE_ESCDAYS7AND8/scATAC_PeakMatrix.txt"
SAMPLE_INPUT_DIR = os.path.join(PROJECT_DIR, "input/transformer_input/mESC")

RAW_MESC_DATA_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC/10X_raw_data/HIFLR_barcodes_features_matrix"
MESC_PEAK_MATRIX_FILE="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/DS011_mESC/10X_raw_data/DS011_PMID35977485_MOUSE_ESC/scATAC_PeakMatrix.HIFLR.txt"


def main():
    # sample_name_list = ["E7.5_rep1", "E7.5_rep1", "E7.75_rep1", "E8.0_rep2", "E8.5_rep2",
    #                     "E8.75_rep2", "E7.5_rep2", "E8.0_rep1", "E8.5_rep1", "E8.75_rep1"]
    
    sample_name_list = ["DS011"]
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with mp.Pool(processes=len(sample_name_list)) as pool:
        results = pool.map(process_sample, sample_name_list)

    logging.info(f"Completed samples: {results}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("anndata").setLevel(logging.ERROR)
    logging.getLogger("scanpy").setLevel(logging.ERROR)
    main()