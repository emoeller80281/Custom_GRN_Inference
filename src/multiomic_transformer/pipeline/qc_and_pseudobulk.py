#!/usr/bin/env python3
"""
qc_and_pseudobulk.py

Perform single-cell QC + pseudobulk aggregation for scRNA-seq and scATAC-seq.
All outputs are saved as Parquet files.

Outputs:
  - scRNA_processed.parquet
  - scATAC_processed.parquet
  - pseudobulk_scRNA.parquet
  - pseudobulk_scATAC.parquet
  - pseudobulk_metadata.parquet
"""

import os
import sys
import math
import logging
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from typing import Tuple, Optional
from anndata import AnnData
import scanpy as sc

from multiomic_transformer.pipeline.config import get_paths
from multiomic_transformer.pipeline.io_utils import (
    ensure_dir,
    write_parquet_safe,
    checkpoint_exists,
    write_done_flag,
)


# =====================================================================
#                            QC FILTERING
# =====================================================================
def filter_and_qc(adata_RNA: AnnData, adata_ATAC: AnnData) -> Tuple[AnnData, AnnData]:
    """
    Perform QC, filtering, and basic preprocessing on scRNA and scATAC data.
    Returns filtered AnnData objects aligned on common barcodes.
    """
    adata_RNA = adata_RNA.copy()
    adata_ATAC = adata_ATAC.copy()

    logging.info(f"[START] RNA shape={adata_RNA.shape}, ATAC shape={adata_ATAC.shape}")

    # ------------------------------------------------------------------
    # Synchronize barcodes between RNA and ATAC
    # ------------------------------------------------------------------
    adata_RNA.obs["barcode"] = adata_RNA.obs_names
    n_before = (adata_RNA.n_obs, adata_ATAC.n_obs)

    adata_RNA = adata_RNA[adata_RNA.obs["barcode"].isin(adata_ATAC.obs_names)].copy()
    adata_ATAC = adata_ATAC[adata_ATAC.obs_names.isin(adata_RNA.obs["barcode"])].copy()

    logging.info(
        f"[BARCODES] before sync RNA={n_before[0]}, ATAC={n_before[1]} → "
        f"after sync RNA={adata_RNA.n_obs}, ATAC={adata_ATAC.n_obs}"
    )

    # ------------------------------------------------------------------
    # RNA QC filtering
    # ------------------------------------------------------------------
    adata_RNA.var["mt"] = adata_RNA.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata_RNA, qc_vars=["mt"], inplace=True)
    adata_RNA = adata_RNA[adata_RNA.obs.pct_counts_mt < 5].copy()

    sc.pp.filter_cells(adata_RNA, min_genes=200)
    sc.pp.filter_genes(adata_RNA, min_cells=3)
    # Only log-transform if not already logged
    if (adata_RNA.X.min() >= 0) and (adata_RNA.X.max() <= 20):
        # heuristically detect raw counts
        sc.pp.normalize_total(adata_RNA, target_sum=1e4)
        sc.pp.log1p(adata_RNA)
    else:
        logging.warning("Skipping log1p: data appears already log-transformed.")

    # Replace NaNs or infs with zeros before HVG computation
    adata_RNA.X = np.nan_to_num(adata_RNA.X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Highly variable genes ---
    try:
        sc.pp.highly_variable_genes(
            adata_RNA,
            min_mean=0.0125,
            max_mean=3,
            min_disp=0.5,
            n_bins=20,
            flavor="seurat",
            inplace=True,
        )
    except ValueError as e:
        logging.warning(f"highly_variable_genes() failed due to non-unique bins: {e}")
        adata_RNA.var["highly_variable"] = np.ones(adata_RNA.n_vars, dtype=bool)
    adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable].copy()
    sc.pp.scale(adata_RNA, max_value=10)
    sc.tl.pca(adata_RNA, n_comps=25, svd_solver="arpack")

    # ------------------------------------------------------------------
    # ATAC QC filtering
    # ------------------------------------------------------------------
    sc.pp.filter_cells(adata_ATAC, min_genes=200)
    sc.pp.filter_genes(adata_ATAC, min_cells=3)
    # Only log-transform if not already logged
    if (adata_ATAC.X.min() >= 0) and (adata_ATAC.X.max() <= 20):
        # heuristically detect raw counts
        sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
        sc.pp.log1p(adata_ATAC)
    else:
        logging.warning("Skipping log1p: data appears already log-transformed.")

    # Replace NaNs or infs with zeros before HVG computation
    adata_ATAC.X = np.nan_to_num(adata_ATAC.X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Highly variable genes ---
    try:
        sc.pp.highly_variable_genes(
            adata_ATAC,
            min_mean=0.0125,
            max_mean=3,
            min_disp=0.5,
            n_bins=20,
            flavor="seurat",
            inplace=True,
        )
    except ValueError as e:
        logging.warning(f"highly_variable_genes() failed due to non-unique bins: {e}")
        adata_ATAC.var["highly_variable"] = np.ones(adata_ATAC.n_vars, dtype=bool)
    adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable].copy()
    sc.pp.scale(adata_ATAC, max_value=10)
    sc.tl.pca(adata_ATAC, n_comps=25, svd_solver="arpack")

    # ------------------------------------------------------------------
    # Synchronize final barcodes
    # ------------------------------------------------------------------
    common = adata_RNA.obs_names.intersection(adata_ATAC.obs_names)
    adata_RNA = adata_RNA[common].copy()
    adata_ATAC = adata_ATAC[common].copy()

    logging.info(f"[QC DONE] RNA shape={adata_RNA.shape}, ATAC shape={adata_ATAC.shape}")
    return adata_RNA, adata_ATAC


# =====================================================================
#                          PSEUDOBULK AGGREGATION
# =====================================================================
def pseudo_bulk(
    rna_data: AnnData,
    atac_data: AnnData,
    use_single: bool = False,
    neighbors_k: int = 20,
    resolution: float = 0.5,
    aggregate: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate pseudobulk RNA and ATAC profiles by clustering cells
    and aggregating neighbors within clusters.
    """

    rna_data = rna_data.copy()
    atac_data = atac_data.copy()

    # Enforce alignment
    atac_data = atac_data[rna_data.obs_names].copy()
    assert (rna_data.obs_names == atac_data.obs_names).all(), "Cell barcodes must align"

    # Combined PCA embedding
    combined_pca = np.concatenate(
        (rna_data.obsm["X_pca"], atac_data.obsm["X_pca"]), axis=1
    )
    rna_data.obsm["X_combined"] = combined_pca
    atac_data.obsm["X_combined"] = combined_pca

    # Joint graph & Leiden clustering
    X_placeholder = sp.csr_matrix((rna_data.n_obs, 0))
    joint = AnnData(X=X_placeholder, obs=rna_data.obs.copy())
    joint.obsm["X_combined"] = combined_pca
    sc.pp.neighbors(joint, n_neighbors=neighbors_k, use_rep="X_combined")
    sc.tl.leiden(joint, resolution=resolution, flavor="igraph", n_iterations=2, key_added="cluster")

    clusters = joint.obs["cluster"].astype(str)
    cluster_ids = clusters.unique()

    rna_data.obs["cluster"] = clusters
    atac_data.obs["cluster"] = clusters

    conn = joint.obsp["connectivities"].tocsr()
    pseudo_bulk_rna, pseudo_bulk_atac, bulk_names = [], [], []

    def aggregate_matrix(X, idx, method):
        if method == "sum":
            return sp.csr_matrix(X[idx, :].sum(axis=0))
        elif method == "mean":
            return sp.csr_matrix(X[idx, :].mean(axis=0))
        else:
            raise ValueError(f"Unknown aggregate method: {method}")

    for cid in cluster_ids:
        cluster_idx = np.where(clusters == cid)[0]
        if len(cluster_idx) == 0:
            continue

        if use_single or len(cluster_idx) < neighbors_k:
            rna_agg = aggregate_matrix(rna_data.X, cluster_idx, aggregate)
            atac_agg = aggregate_matrix(atac_data.X, cluster_idx, aggregate)
            pseudo_bulk_rna.append(rna_agg)
            pseudo_bulk_atac.append(atac_agg)
            bulk_names.append(f"cluster{cid}")
        else:
            seeds = np.random.choice(cluster_idx, size=int(np.sqrt(len(cluster_idx))), replace=False)
            for s in seeds:
                neighbors = conn[s].indices
                group = np.append(neighbors, s)
                rna_agg = aggregate_matrix(rna_data.X, group, aggregate)
                atac_agg = aggregate_matrix(atac_data.X, group, aggregate)
                pseudo_bulk_rna.append(rna_agg)
                pseudo_bulk_atac.append(atac_agg)
                bulk_names.append(f"cluster{cid}_cell{s}")

    pseudo_bulk_rna = sp.vstack(pseudo_bulk_rna).T
    pseudo_bulk_atac = sp.vstack(pseudo_bulk_atac).T

    df_rna = pd.DataFrame(
        pseudo_bulk_rna.toarray(), index=rna_data.var_names, columns=bulk_names
    )
    df_atac = pd.DataFrame(
        pseudo_bulk_atac.toarray(), index=atac_data.var_names, columns=bulk_names
    )

    logging.info(f"[PSEUDOBULK] RNA shape={df_rna.shape}, ATAC shape={df_atac.shape}")
    return df_rna, df_atac


# =====================================================================
#                             MAIN PIPELINE
# =====================================================================
def run_qc_and_pseudobulk(
    rna_path: str,
    atac_path: str,
    organism: str = "mm10",
    num_cpu: int = 8,
    outdir:Optional[Path] = None,
    keep_intermediate: bool = False,
    force_recompute: bool = False,
):
    paths = get_paths()
    outdir = Path(outdir or paths["processed_data"])
    ensure_dir(outdir)

    # -------------------------------------------------------------
    # Load input data
    # -------------------------------------------------------------
    logging.info(f"Loading data:\n  RNA: {rna_path}\n  ATAC: {atac_path}")
    adata_RNA = sc.read(rna_path)
    adata_ATAC = sc.read(atac_path)

    # -------------------------------------------------------------
    # Stage 1: QC
    # -------------------------------------------------------------
    qc_done_flag = outdir / ".qc.done"
    if not checkpoint_exists(qc_done_flag) or force_recompute:
        adata_RNA_qc, adata_ATAC_qc = filter_and_qc(adata_RNA, adata_ATAC)

        # Save as Parquet (cell × feature)
        def to_dense(X):
            return X.toarray() if hasattr(X, "toarray") else X

        df_rna = pd.DataFrame(to_dense(adata_RNA_qc.X), index=adata_RNA_qc.obs_names, columns=adata_RNA_qc.var_names)
        df_atac = pd.DataFrame(to_dense(adata_ATAC_qc.X), index=adata_ATAC_qc.obs_names, columns=adata_ATAC_qc.var_names)


        write_parquet_safe(df_rna, outdir / "scRNA_processed.parquet")
        write_parquet_safe(df_atac, outdir / "scATAC_processed.parquet")

        write_done_flag(qc_done_flag)
        if not keep_intermediate:
            del adata_RNA_qc, adata_ATAC_qc, df_rna, df_atac
    else:
        logging.info("QC already completed; skipping.")

    # -------------------------------------------------------------
    # Stage 2: Pseudobulk
    # -------------------------------------------------------------
    pb_done_flag = outdir / ".pseudobulk.done"
    if not checkpoint_exists(pb_done_flag) or force_recompute:
        if "adata_RNA_qc" not in locals():
            adata_RNA_qc = sc.read(rna_path)
            adata_ATAC_qc = sc.read(atac_path)

        df_rna_bulk, df_atac_bulk = pseudo_bulk(adata_RNA_qc, adata_ATAC_qc)
        write_parquet_safe(df_rna_bulk, outdir / "pseudobulk_scRNA.parquet")
        write_parquet_safe(df_atac_bulk, outdir / "pseudobulk_scATAC.parquet")

        meta = pd.DataFrame({
            "cluster": df_rna_bulk.columns,
            "n_genes": [df_rna_bulk.shape[0]] * len(df_rna_bulk.columns),
            "n_peaks": [df_atac_bulk.shape[0]] * len(df_rna_bulk.columns)
        })
        write_parquet_safe(meta, outdir / "pseudobulk_metadata.parquet")

        write_done_flag(pb_done_flag)
    else:
        logging.info("Pseudobulk already completed; skipping.")

    logging.info("QC + pseudobulk pipeline complete.")


# =====================================================================
# CLI ENTRY
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="QC and pseudobulk aggregation for scRNA-seq and scATAC-seq.")
    parser.add_argument("--rna_path", required=True, help="Path to scRNA-seq AnnData file (.h5ad or .mtx).")
    parser.add_argument("--atac_path", required=True, help="Path to scATAC-seq AnnData file (.h5ad or .mtx).")
    parser.add_argument("--organism", default="mm10", help="Genome assembly (default: mm10).")
    parser.add_argument("--num_cpu", type=int, default=8, help="Number of CPU cores (default: 8).")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default from config).")
    parser.add_argument("--keep_intermediate", action="store_true", help="Keep intermediate data files.")
    parser.add_argument("--force_recompute", action="store_true", help="Recompute even if .done files exist.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    if args.outdir is None:
        outdir = None
    else:
        outdir = Path(args.outdir)

    run_qc_and_pseudobulk(
        rna_path=args.rna_path,
        atac_path=args.atac_path,
        organism=args.organism,
        num_cpu=args.num_cpu,
        outdir=outdir,
        keep_intermediate=args.keep_intermediate,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
