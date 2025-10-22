import os
import re
import json
import torch
import joblib
import pandas as pd
import scanpy as sc
import logging
from pathlib import Path
import warnings
import numpy as np
import scipy.sparse as sp
import random
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Set, Optional, List, Iterable, Union, Dict
from anndata import AnnData
from tqdm import tqdm
import pybedtools
import argparse

import sys
sys.path.append(Path(__file__).resolve().parent.parent.parent)

from multiomic_transformer.utils.standardize import standardize_name
from multiomic_transformer.utils.files import atomic_json_dump
from multiomic_transformer.utils.peaks import find_genes_near_peaks, format_peaks
from multiomic_transformer.utils.downloads import *
from multiomic_transformer.data.sliding_window import run_sliding_window_scan
from multiomic_transformer.data.build_pkn import build_organism_pkns
from config.settings import *

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)

# ----- Data Loading and Processing -----
def pseudo_bulk(
    rna_data: AnnData,
    atac_data: AnnData,
    use_single: bool = False,
    neighbors_k: int = 20,
    resolution: float = 0.5,
    aggregate: str = "mean",
    pca_components: int = 25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate pseudobulk RNA and ATAC profiles by clustering cells and 
    aggregating their neighbors.
    """

    rna_data = rna_data.copy()
    atac_data = atac_data.copy()

    # ----- Ensure the same cells and order in both modalities -----
    # Align to common obs_names (intersection) and same order
    common = rna_data.obs_names.intersection(atac_data.obs_names)
    if len(common) == 0:
        raise ValueError("No overlapping cell barcodes between RNA and ATAC.")
    rna_data = rna_data[common].copy()
    atac_data = atac_data[common].copy()
    # strictly enforce identical order
    atac_data = atac_data[rna_data.obs_names].copy()
    assert (rna_data.obs_names == atac_data.obs_names).all(), "Cell barcodes must be aligned"

    # ----- Ensure PCA exists for both modalities -----
    def _ensure_pca(adata: AnnData, n_comps: int) -> None:
        # cap components to valid range
        max_comps = int(min(adata.n_obs, adata.n_vars))
        use_comps = max(1, min(n_comps, max_comps))
        if "X_pca" not in adata.obsm_keys() or adata.obsm.get("X_pca", np.empty((0,0))).shape[1] < use_comps:
            sc.pp.scale(adata, max_value=10, zero_center=True)
            sc.tl.pca(adata, n_comps=use_comps, svd_solver="arpack")

    _ensure_pca(rna_data, pca_components)
    _ensure_pca(atac_data, pca_components)
    
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

    rna_rows: List[sp.csr_matrix] = []
    atac_rows: List[sp.csr_matrix] = []
    bulk_names: List[str] = []
    
    def aggregate_matrix(X: sp.spmatrix, idx: np.ndarray, method: str) -> sp.csr_matrix:
        """
        Aggregate rows in `X` indexed by `idx` using sum/mean.
        Returns a 1×n_features CSR matrix.
        """
        if method == "sum":
            out = X[idx, :].sum(axis=0)
        elif method == "mean":
            out = X[idx, :].mean(axis=0)
        else:
            raise ValueError(f"Unknown aggregate: {method}")
        # ensure CSR 2D
        return sp.csr_matrix(out)

    for cid in cluster_ids:
        cluster_idx = np.where(clusters == cid)[0]
        if len(cluster_idx) == 0:
            continue
        
        if use_single or len(cluster_idx) < neighbors_k:
            # Single pseudobulk
            rna_agg = aggregate_matrix(rna_data.X, cluster_idx, aggregate)
            atac_agg = aggregate_matrix(atac_data.X, cluster_idx, aggregate)
            rna_rows.append(rna_agg)
            atac_rows.append(atac_agg)
            bulk_names.append(f"cluster{cid}")
        else:
            # Multiple pseudobulks
            seeds = np.random.choice(cluster_idx, size=int(np.sqrt(len(cluster_idx))), replace=False)
            for s in seeds:
                neighbors = conn[s].indices
                group = np.append(neighbors, s)
                rna_agg = aggregate_matrix(rna_data.X, group, aggregate)
                atac_agg = aggregate_matrix(atac_data.X, group, aggregate)
                rna_rows.append(rna_agg)
                atac_rows.append(atac_agg)
                bulk_names.append(f"cluster{cid}_cell{s}")
                
    # Stack to (n_bulks × n_features) then transpose → (n_features × n_bulks)
    rna_stack: sp.csr_matrix = sp.vstack(rna_rows) if rna_rows else sp.csr_matrix((0, rna_data.n_vars))
    atac_stack: sp.csr_matrix = sp.vstack(atac_rows) if atac_rows else sp.csr_matrix((0, atac_data.n_vars))

    rna_stack = rna_stack.T
    atac_stack = atac_stack.T

    # Final DataFrames
    pseudo_bulk_rna_df: pd.DataFrame = pd.DataFrame(
        rna_stack.toarray(),
        index=rna_data.var_names,
        columns=bulk_names,
    )
    pseudo_bulk_atac_df: pd.DataFrame = pd.DataFrame(
        atac_stack.toarray(),
        index=atac_data.var_names,
        columns=bulk_names,
    )

    return pseudo_bulk_rna_df, pseudo_bulk_atac_df

def process_10x_to_csv(raw_10x_rna_data_dir, raw_atac_peak_file, rna_outfile_path, atac_outfile_path):
    
    def _load_rna_adata(sample_raw_data_dir: str) -> sc.AnnData:
        # Look for features file
        features = [f for f in os.listdir(sample_raw_data_dir) if f.endswith("features.tsv.gz")]
        assert len(features) == 1, \
            f"Expected 1 features.tsv.gz, found {features}. Make sure the files are gunziped for sc.read_10x_mtx."

        prefix = features[0].replace("features.tsv.gz", "")
        logging.info(f"Detected File Prefix: {prefix}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Only considering the two last:")
            adata = sc.read_10x_mtx(
                sample_raw_data_dir,
                var_names="gene_symbols",
                make_unique=True,
                prefix=prefix
            )
        return adata
    
    def _get_adata_from_peakmatrix(peak_matrix_file: Path, label: pd.DataFrame, sample_name: str) -> AnnData:
        logging.info(f"[{sample_name}] Reading ATAC peaks")
        # Read header only
        all_cols = pd.read_csv(peak_matrix_file, sep="\t", nrows=10).columns[1:]
        logging.info(f"  - First ATAC Barcode: {all_cols[0]}")
        
        # Identify barcodes shared between RNA and ATAC
        matching_barcodes = set(label["barcode_use"]) & set(all_cols)
        logging.info(f"  - Matched {len(matching_barcodes):,} barcodes with scRNA-seq file")

        # Map from original index -> normalized barcode
        col_map = {i: bc for i, bc in enumerate(all_cols)}

        # Always keep the first column (peak IDs)
        keep_indices = [0] + [i for i, bc in col_map.items() if bc in matching_barcodes]
        header = pd.read_csv(peak_matrix_file, sep="\t", nrows=0)
        first_col = header.columns[0]
        keep_cols = [first_col] + [c for c in header.columns[1:] if c in matching_barcodes]
        compression = "gzip" if str(peak_matrix_file).endswith(".gz") else None

        # Read only those columns
        logging.info("  - Reading data for matching barcodes")
        peak_matrix = pd.read_csv(
            peak_matrix_file,
            sep="\t",
            usecols=keep_cols,
            index_col=0,
            compression=compression,
            low_memory=False
        )
        logging.info("\tDone reading filtered peak matrix")

        # Replace column names with normalized barcodes
        new_cols = [col_map[i] for i in keep_indices[1:]]
        peak_matrix.columns = new_cols

        # Construct AnnData
        logging.info("  - Constructing AnnData for scATAC-seq data")
        adata_ATAC = AnnData(X=sp.csr_matrix(peak_matrix.values.T))
        adata_ATAC.obs_names = peak_matrix.columns
        adata_ATAC.var_names = peak_matrix.index
        adata_ATAC.obs["barcode"] = adata_ATAC.obs_names
        adata_ATAC.obs["sample"] = sample_name
        adata_ATAC.obs["label"] = label.set_index("barcode_use").loc[peak_matrix.columns, "label"].values
        
        logging.info("\tDone!")

        return adata_ATAC
    
    # --- load raw data ---
    adata_RNA = _load_rna_adata(raw_10x_rna_data_dir)
    adata_RNA.obs_names = [(sample_name + "." + i).replace("-", ".") for i in adata_RNA.obs_names]
    # adata_RNA.obs_names = [i.replace("-", ".") for i in adata_RNA.obs_names]
    logging.info(f"[{sample_name}] Found {len(adata_RNA.obs_names)} RNA barcodes")
    logging.info(f"  - First RNA barcode: {adata_RNA.obs_names[0]}")

    label = pd.DataFrame({"barcode_use": adata_RNA.obs_names,
                            "label": ["mESC"] * len(adata_RNA.obs_names)})

    adata_ATAC = _get_adata_from_peakmatrix(raw_atac_peak_file, label, sample_name)
    
    logging.info(f"[{sample_name}] Writing raw data files")
    raw_sc_rna_df = pd.DataFrame(
        adata_RNA.X.toarray() if sp.issparse(adata_RNA.X) else adata_RNA.X,
        index=adata_RNA.obs_names,
        columns=adata_RNA.var_names,
    )
    raw_sc_atac_df = pd.DataFrame(
        adata_ATAC.X.toarray() if sp.issparse(adata_ATAC.X) else adata_ATAC.X,
        index=adata_ATAC.obs_names,
        columns=adata_ATAC.var_names,
    )

    os.makedirs(os.path.dirname(rna_outfile_path), exist_ok=True)
    os.makedirs(os.path.dirname(atac_outfile_path), exist_ok=True)
    
    raw_sc_rna_df = raw_sc_rna_df.astype("float32")
    raw_sc_atac_df = raw_sc_atac_df.astype("float32")
    
    raw_sc_rna_df.to_parquet(rna_outfile_path, engine="pyarrow", compression="snappy")
    raw_sc_atac_df.to_parquet(atac_outfile_path, engine="pyarrow", compression="snappy")

def process_or_load_rna_atac_data(
    sample_input_dir: Union[str, Path],
    ignore_processed_files: bool = False,
    raw_10x_rna_data_dir: Union[str, Path, None] = None,
    raw_atac_peak_file: Union[str, Path, None] = None,
    *,
    sample_name: Optional[str] = None,
    neighbors_k: Optional[int] = None,             # defaults to NEIGHBORS_K if None
    leiden_resolution: Optional[float] = None,     # defaults to LEIDEN_RESOLUTION if None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load or build processed scRNA/scATAC matrices and their pseudobulk aggregations.

    The function progressively searches for:
        1) processed parquet files,
        2) filtered AnnData (.h5ad),
        3) raw CSVs,
        4) raw 10x inputs (if provided),
    and materializes missing artifacts as needed.

    It also checks for pseudobulk TSV files (TG and RE). If they are missing, it
    computes pseudobulks from filtered AnnData and writes them.

    Parameters
    ----------
    sample_input_dir : str | Path
        Directory that contains/should contain the processed files for one sample.
    ignore_processed_files : bool, default False
        Force recomputation even if processed/parquet/h5ad exists.
    raw_10x_rna_data_dir : str | Path | None
        Path to the 10x RNA directory if CSVs need to be generated.
    raw_atac_peak_file : str | Path | None
        Path to a peak matrix (or peak list) used when building ATAC CSVs from 10x.
    sample_name : str | None, keyword-only
        Pretty name for logging; defaults to the directory name if None.
    neighbors_k : int | None, keyword-only
        Neighborhood size for multi-pseudobulk; defaults to NEIGHBORS_K if None.
    leiden_resolution : float | None, keyword-only
        Clustering resolution; defaults to LEIDEN_RESOLUTION if None.

    Returns
    -------
    processed_rna_df : DataFrame
        Genes × cells/pseudobulks, dense.
    processed_atac_df : DataFrame
        Peaks × cells/pseudobulks, dense.
    TG_pseudobulk_df : DataFrame
        Genes × pseudobulks.
    RE_pseudobulk_df : DataFrame
        Peaks × pseudobulks.

    Notes
    -----
    - ATAC pseudobulk values are clipped to 100 to limit extreme counts.
    - If adata_ATAC has a 'gene_ids' column in .var, it becomes var_names.
    - Will use existing artifacts when available unless `ignore_processed_files=True`.
    """
    # ---- resolve paths and names ----
    sample_input_dir = Path(sample_input_dir)
    sample_name = sample_name or sample_input_dir.name
    raw_10x_rna_data_dir = Path(raw_10x_rna_data_dir) if raw_10x_rna_data_dir is not None else None
    raw_atac_peak_file = Path(raw_atac_peak_file) if raw_atac_peak_file is not None else None

    processed_rna_file = sample_input_dir / "scRNA_seq_processed.parquet"
    processed_atac_file = sample_input_dir / "scATAC_seq_processed.parquet"

    raw_rna_file = sample_input_dir / "scRNA_seq_raw.parquet"
    raw_atac_file = sample_input_dir / "scATAC_seq_raw.parquet"

    adata_rna_file = sample_input_dir / "adata_RNA.h5ad"
    adata_atac_file = sample_input_dir / "adata_ATAC.h5ad"

    pseudobulk_TG_file = sample_input_dir / "TG_pseudobulk.tsv"
    pseudobulk_RE_file = sample_input_dir / "RE_pseudobulk.tsv"

    # lazy-import pipeline constants if not provided
    try:
        from config.settings import NEIGHBORS_K as _K, LEIDEN_RESOLUTION as _RES
    except Exception:
        _K, _RES = 15, 1.0  # safe fallbacks
    neighbors_k = neighbors_k if neighbors_k is not None else _K
    leiden_resolution = leiden_resolution if leiden_resolution is not None else _RES

    logging.info(f"\n----- Loading or Processing RNA and ATAC data for {sample_name} -----")
    logging.info("Searching for processed RNA/ATAC parquet files:")

    # helpers
    def _adata_to_dense_df(adata: AnnData) -> pd.DataFrame:
        X = adata.X
        if sp.issparse(X):
            X = X.toarray()
        else:
            X = np.asarray(X)
        # return gene/peak × cell matrix
        return pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names).T

    def _load_or_none(path: Path, loader):
        if path.is_file():
            return loader(path)
        return None
    
    def _standardize_symbols_index(
        df: pd.DataFrame,
        *,
        strip_version_suffix: bool = True,  # e.g., 'Gm12345.1' -> 'GM12345'
        uppercase: bool = True,
        deduplicate: str = "sum",          # {'sum','mean','first','max','min','median', None}
    ) -> pd.DataFrame:
        """
        Standardize gene symbols in the DataFrame index.

        - Assumes rows are genes (index), columns are cells/pseudobulks.
        - Applies simple, offline normalization:
            * strip whitespace
            * optionally strip trailing transcript/version suffixes like '.1', '.2'
            * optionally uppercase
        - Optionally aggregates duplicate indices created by normalization.

        Returns a NEW DataFrame.
        """
        x = df.copy()

        # ensure index is str
        idx = x.index.astype(str).str.strip()

        if strip_version_suffix:
            # remove final dot-number ("ENSG000..", "Gene.1", etc.)
            idx = idx.str.replace(r"\.\d+$", "", regex=True)

        if uppercase:
            idx = idx.str.upper()

        x.index = idx

        if deduplicate:
            if deduplicate == "sum":
                x = x.groupby(level=0).sum()
            elif deduplicate == "mean":
                x = x.groupby(level=0).mean()
            elif deduplicate == "first":
                x = x[~x.index.duplicated(keep="first")]
            elif deduplicate in {"max", "min", "median"}:
                x = getattr(x.groupby(level=0), deduplicate)()
            else:
                raise ValueError(f"Unknown deduplicate policy: {deduplicate}")

        return x


    def _standardize_symbols_series_index(
        s: pd.Series,
        *,
        strip_version_suffix: bool = True,
        uppercase: bool = True,
        deduplicate: str = "sum",  # aggregation for duplicates, if any
    ) -> pd.Series:
        """
        Same as above, but for a Series (used for pseudobulk TG).
        """
        df = _standardize_symbols_index(s.to_frame(), strip_version_suffix=strip_version_suffix,
                                        uppercase=uppercase, deduplicate=deduplicate)
        return df.iloc[:, 0]


    # placeholders
    processed_rna_df: Optional[pd.DataFrame] = None
    processed_atac_df: Optional[pd.DataFrame] = None
    TG_pseudobulk_df: Optional[pd.DataFrame] = None
    RE_pseudobulk_df: Optional[pd.DataFrame] = None

    # =========================
    # 1) Try processed parquet
    # =========================
    if not ignore_processed_files and processed_rna_file.is_file() and processed_atac_file.is_file():
        logging.info("Pre-processed parquet files found, loading...")
        processed_rna_df = pd.read_parquet(processed_rna_file, engine="pyarrow")
        processed_atac_df = pd.read_parquet(processed_atac_file, engine="pyarrow")
        
        processed_rna_df = _standardize_symbols_index(processed_rna_df, strip_version_suffix=True, uppercase=True, deduplicate="sum")
        
    else:
        logging.info("  - Processed parquet missing or ignored – will (re)build from earlier stages.")

        # ====================================
        # 2) Try filtered AnnData (.h5ad) pair
        # ====================================
        ad_rna = _load_or_none(adata_rna_file, sc.read_h5ad)
        ad_atac = _load_or_none(adata_atac_file, sc.read_h5ad)

        if ad_rna is None or ad_atac is None or ignore_processed_files:
            logging.info("    - Filtered AnnData missing or ignored – will look for raw CSVs.")

            # ===================
            # 3) Try raw CSV pair
            # ===================
            if not raw_rna_file.is_file() or not raw_atac_file.is_file():
                logging.info("Raw parquet files missing – will try to create from 10x inputs.")

                if raw_10x_rna_data_dir is None:
                    raise FileNotFoundError(
                        "Neither processed parquet, filtered AnnData, raw CSVs, nor raw 10x inputs are available. "
                        "Provide raw_10x_rna_data_dir and raw_atac_peak_file to proceed."
                    )
                if not raw_10x_rna_data_dir.is_dir():
                    raise FileNotFoundError(f"10x RNA directory not found: {raw_10x_rna_data_dir}")
                if raw_atac_peak_file is None or not raw_atac_peak_file.is_file():
                    raise FileNotFoundError(f"ATAC peak file not found: {raw_atac_peak_file}")

                logging.info("Raw 10X RNA and ATAC inputs found, converting to CSVs...")
                logging.info(f"  - raw_10x_rna_data_dir: {raw_10x_rna_data_dir}")
                logging.info(f"  - raw_atac_peak_file:  {raw_atac_peak_file}")
                process_10x_to_csv(raw_10x_rna_data_dir, raw_atac_peak_file, raw_rna_file, raw_atac_file)

            # Load raw data parquet files and convert to AnnData
            logging.info("Reading raw CSVs into AnnData")
            rna_df = pd.read_parquet(raw_rna_file, engine="pyarrow")
            atac_df = pd.read_parquet(raw_atac_file, engine="pyarrow")
            
            ad_rna = AnnData(rna_df)
            ad_atac = AnnData(atac_df)

            # QC/filter
            logging.info("Running filter_and_qc on RNA/ATAC AnnData")
            ad_rna, ad_atac = filter_and_qc(ad_rna, ad_atac)

            # Persist filtered AnnData for reuse
            logging.info("Writing filtered AnnData files")
            ad_rna.write_h5ad(adata_rna_file)
            ad_atac.write_h5ad(adata_atac_file)

        else:
            logging.info("Reading pre-filtered RNA/ATAC AnnData from disk")
            ad_rna = sc.read_h5ad(adata_rna_file)
            ad_atac = sc.read_h5ad(adata_atac_file)

        # Optional: if ATAC var has gene ids, adopt them as var_names
        if "gene_ids" in ad_atac.var.columns:
            ad_atac.var_names = ad_atac.var["gene_ids"].astype(str)

        logging.info("Converting AnnData to dense DataFrames")
        processed_rna_df = _adata_to_dense_df(ad_rna)
        processed_atac_df = _adata_to_dense_df(ad_atac)

        processed_rna_df = _standardize_symbols_index(processed_rna_df, strip_version_suffix=True, uppercase=True, deduplicate="sum")
        
        # Persist processed parquets
        logging.info("Writing processed parquet files")
        processed_rna_df.to_parquet(processed_rna_file, engine="pyarrow", compression="snappy")
        processed_atac_df.to_parquet(processed_atac_file, engine="pyarrow", compression="snappy")

    # =======================================
    # 4) Ensure pseudobulks exist and return
    # =======================================
    need_TG = not pseudobulk_TG_file.is_file()
    need_RE = not pseudobulk_RE_file.is_file()

    if need_TG or need_RE or ignore_processed_files:
        logging.info("Pseudobulk files missing or ignored – computing pseudobulk now.")

        # If adatas were not loaded above, reconstruct them from the processed matrices
        # to run `pseudo_bulk` (it expects AnnData + (optionally) ATAC frame).
        if 'ad_rna' not in locals() or 'ad_atac' not in locals():
            logging.info("Rebuilding AnnData objects from processed dense DataFrames for pseudobulk.")
            ad_rna = AnnData(processed_rna_df.T)
            ad_atac = AnnData(processed_atac_df.T)

        # Single-pseudobulk heuristic: keep your <100 rule
        singlepseudobulk = ad_rna.n_obs < 100

        TG_pseudobulk_df, RE_pseudobulk_df = pseudo_bulk(
            rna_data=ad_rna,
            atac_data=ad_atac,
            use_single=singlepseudobulk,
            neighbors_k=neighbors_k,
            resolution=leiden_resolution,
        )

        # Post-processing as in your original
        TG_pseudobulk_df = TG_pseudobulk_df.fillna(0)
        RE_pseudobulk_df = RE_pseudobulk_df.fillna(0)
        RE_pseudobulk_df[RE_pseudobulk_df > 100] = 100
        
        TG_pseudobulk_df = _standardize_symbols_index(TG_pseudobulk_df, strip_version_suffix=True, uppercase=True, deduplicate="sum")

        TG_pseudobulk_df.to_csv(pseudobulk_TG_file, sep="\t")
        RE_pseudobulk_df.to_csv(pseudobulk_RE_file, sep="\t")
    else:
        logging.info("Pseudobulk TSVs found, loading from disk.")
        logging.info(f"  - Pseudobulk TG Path: {pseudobulk_TG_file}")
        logging.info(f"  - Pseudobulk RE Path: {pseudobulk_RE_file}")
        TG_pseudobulk_df = pd.read_csv(pseudobulk_TG_file, sep="\t", index_col=0)
        RE_pseudobulk_df = pd.read_csv(pseudobulk_RE_file, sep="\t", index_col=0)
        
        TG_pseudobulk_df = _standardize_symbols_index(TG_pseudobulk_df, strip_version_suffix=True, uppercase=True, deduplicate="sum")

    # Final sanity checks
    for name, df in [
        ("processed_rna_df", processed_rna_df),
        ("processed_atac_df", processed_atac_df),
        ("TG_pseudobulk_df", TG_pseudobulk_df),
        ("RE_pseudobulk_df", RE_pseudobulk_df),
    ]:
        if df is None or df.empty:
            raise ValueError(f"{name} is empty or missing; upstream steps did not produce valid data.")

    logging.info("RNA/ATAC processed matrices and pseudobulks are ready.")
    return processed_rna_df, processed_atac_df, TG_pseudobulk_df, RE_pseudobulk_df

def filter_and_qc(adata_RNA: AnnData, adata_ATAC: AnnData) -> Tuple[AnnData, AnnData]:
    
    adata_RNA = adata_RNA.copy()
    adata_ATAC = adata_ATAC.copy()
    
    logging.info(f"[START] RNA shape={adata_RNA.shape}, ATAC shape={adata_ATAC.shape}")

    
    # Synchronize barcodes
    adata_RNA.obs['barcode'] = adata_RNA.obs_names
    adata_ATAC.obs['barcode'] = adata_ATAC.obs_names

    common_barcodes = adata_RNA.obs['barcode'].isin(adata_ATAC.obs['barcode'])
    n_before = (adata_RNA.n_obs, adata_ATAC.n_obs)
    adata_RNA = adata_RNA[common_barcodes].copy()
    adata_ATAC = adata_ATAC[adata_ATAC.obs['barcode'].isin(adata_RNA.obs['barcode'])].copy()
    
    logging.info(
        f"[BARCODES] before sync RNA={n_before[0]}, ATAC={n_before[1]} → after sync RNA={adata_RNA.n_obs}, ATAC={adata_ATAC.n_obs}"
    )
    
    # QC and filtering
    
    adata_RNA.var['mt'] = adata_RNA.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata_RNA, qc_vars=["mt"], inplace=True)
    adata_RNA = adata_RNA[adata_RNA.obs.pct_counts_mt < 5].copy()
    adata_RNA.var_names_make_unique()
    adata_RNA.var['gene_ids'] = adata_RNA.var.index
        
    sc.pp.filter_cells(adata_RNA, min_genes=200)
    sc.pp.filter_genes(adata_RNA, min_cells=3)
    sc.pp.filter_cells(adata_ATAC, min_genes=200)
    sc.pp.filter_genes(adata_ATAC, min_cells=3)
    
    # Preprocess RNA
    sc.pp.normalize_total(adata_RNA, target_sum=1e4)
    sc.pp.log1p(adata_RNA)
    sc.pp.highly_variable_genes(adata_RNA, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable]
    sc.pp.scale(adata_RNA, max_value=10)
    sc.tl.pca(adata_RNA, n_comps=25, svd_solver="arpack")

    # Preprocess ATAC
    sc.pp.log1p(adata_ATAC)
    sc.pp.highly_variable_genes(adata_ATAC, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable]
    sc.pp.scale(adata_ATAC, max_value=10, zero_center=True)
    sc.tl.pca(adata_ATAC, n_comps=25, svd_solver="arpack")
    
    # After filtering to common barcodes
    common_barcodes = adata_RNA.obs_names.intersection(adata_ATAC.obs_names)
    
    adata_RNA = adata_RNA[common_barcodes].copy()
    adata_ATAC = adata_ATAC[common_barcodes].copy()
    
    return adata_RNA, adata_ATAC

def create_tf_tg_combination_files(
    genes: Iterable[str],
    tf_list_file: Union[str, Path],
    dataset_dir: Union[str, Path],
    *,
    tf_name_col: Optional[str] = "TF_Name",  # if None, will auto-detect
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Build and persist TF/TG lists and the full TF–TG Cartesian product, updating any
    existing files by taking the union with newly supplied genes.

    Files written to {dataset_dir}/tf_tg_combos/:
      - total_genes.csv with column 'Gene'
      - tf_list.csv     with column 'TF'
      - tg_list.csv     with column 'TG'
      - tf_tg_combos.csv with columns ['TF','TG']

    Behavior:
      - Normalizes all symbols to uppercase and strips Ensembl-like version suffixes (.1, .2, ...).
      - TFs are defined as the intersection of `genes` with entries in `tf_list_file`.
      - TGs are defined as the remaining genes (i.e., genes - TFs).
      - If files already exist, the function unions existing + new and rewrites deterministically
        (sorted) to make runs reproducible.

    Parameters
    ----------
    genes : Iterable[str]
        Candidate gene symbols (TG or TF). Will be normalized.
    tf_list_file : str | Path
        Path to a TF reference list (tabular). If `tf_name_col` is None or missing, the function
        auto-detects a likely column or uses the only column if single-column.
    dataset_dir : str | Path
        Base directory under which the 'tf_tg_combos' folder will be created/updated.
    tf_name_col : str | None, default 'TF_Name'
        Column name that holds TF names in `tf_list_file`. If None or not present, auto-detect.

    Returns
    -------
    tfs : list[str]
        Final TF set (sorted, normalized).
    tgs : list[str]
        Final TG set (sorted, normalized, guaranteed disjoint from TFs).
    tf_tg_df : pd.DataFrame
        Cartesian product of `tfs × tgs`, columns ['TF','TG'].
    """
    dataset_dir = Path(dataset_dir)
    out_dir = dataset_dir / "tf_tg_combos"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _canon(x: str) -> str:
        # strip version suffix and uppercase
        s = str(x).strip()
        s = re.sub(r"\.\d+$", "", s)
        return s.upper()

    # --- normalize incoming genes ---
    genes_norm = sorted({_canon(g) for g in genes if pd.notna(g)})

    # --- load TF reference file robustly (auto-detect column if needed) ---
    tf_list_file = Path(tf_list_file)
    tf_ref = pd.read_csv(tf_list_file, sep=None, engine="python")  # auto-detect delim
    if tf_name_col and tf_name_col in tf_ref.columns:
        tf_col = tf_name_col
    else:
        # attempt to auto-detect a sensible TF column
        lower = {c.lower(): c for c in tf_ref.columns}
        for cand in ("tf_name", "tf", "symbol", "gene_symbol", "gene", "name"):
            if cand in lower:
                tf_col = lower[cand]
                break
        else:
            # if exactly one column, use it
            if tf_ref.shape[1] == 1:
                tf_col = tf_ref.columns[0]
            else:
                raise ValueError(
                    f"Could not locate TF name column in {tf_list_file}. "
                    f"Available columns: {list(tf_ref.columns)}"
                )

    known_tfs = {_canon(x) for x in tf_ref[tf_col].dropna().astype(str).tolist()}

    # --- new sets from this call ---
    tfs_new = sorted(set(genes_norm) & known_tfs)
    tgs_new = sorted(set(genes_norm) - set(tfs_new))

    # --- load existing lists (if any) and union ---
    def _read_list(path: Path, col: str) -> list[str]:
        if path.is_file():
            df = pd.read_csv(path)
            if col not in df.columns and df.shape[1] == 1:
                # tolerate unnamed single column
                return sorted({_canon(v) for v in df.iloc[:, 0].astype(str)})
            return sorted({_canon(v) for v in df[col].dropna().astype(str)})
        return []

    total_file = out_dir / "total_genes.csv"
    tf_file    = out_dir / "tf_list.csv"
    tg_file    = out_dir / "tg_list.csv"
    combo_file = out_dir / "tf_tg_combos.csv"

    total_existing = _read_list(total_file, "Gene")
    tf_existing    = _read_list(tf_file, "TF")
    tg_existing    = _read_list(tg_file, "TG")

    total = sorted(set(total_existing) | set(genes_norm))
    tfs   = sorted(set(tf_existing)    | set(tfs_new))
    # ensure TGs exclude any TFs
    tgs   = sorted((set(tg_existing) | set(tgs_new)) - set(tfs))

    # --- write back deterministically ---
    pd.DataFrame({"Gene": total}).to_csv(total_file, index=False)
    pd.DataFrame({"TF": tfs}).to_csv(tf_file, index=False)
    pd.DataFrame({"TG": tgs}).to_csv(tg_file, index=False)

    # full Cartesian product for current state (always rewrite)
    if tfs and tgs:
        mux = pd.MultiIndex.from_product([tfs, tgs], names=["TF", "TG"])
        tf_tg_df = mux.to_frame(index=False)
    else:
        tf_tg_df = pd.DataFrame(columns=["TF", "TG"])

    tf_tg_df.to_csv(combo_file, index=False)

    logging.info("\nCreating TF-TG combination files (updated)")
    logging.info(f"  - Number of TFs: {len(tfs):,}")
    logging.info(f"  - Number of TGs: {len(tgs):,}")
    logging.info(f"  - TF-TG combinations: {len(tf_tg_df):,}")
    logging.info(f"  - Files written under: {out_dir}")

    return tfs, tgs, tf_tg_df

# ----- MultiomicTransformer Global Dataset -----
def make_gene_tss_bed_file(gene_tss_file, genome_dir):
    gene_tss_bed = pybedtools.BedTool(gene_tss_file)
    gene_tss_df = (
        gene_tss_bed.to_dataframe(header=None, usecols=[0, 1, 2, 3])
        .rename(columns={0: "chrom", 1: "start", 2: "end", 3: "name"})
        .sort_values(by="start", ascending=True)
    )
    bed_path = os.path.join(genome_dir, "gene_tss.bed")
    gene_tss_df.to_csv(bed_path, sep="\t", header=False, index=False)
    return gene_tss_df

def build_peak_locs_from_index(
    peak_index: pd.Index,
    *,
    include_regex: str = r"^chr(\d+|X|Y)$",
    coerce_chr_prefix: bool = True,
) -> pd.DataFrame:
    """
    Parse peak ids like 'chr1:100-200' (or '1:100-200' if coerce_chr_prefix)
    into a clean DataFrame and filter to canonical chromosomes (1..n, X, Y).
    """
    rows = []
    for pid in map(str, peak_index):
        try:
            chrom_part, se = pid.split(":")
            if coerce_chr_prefix and not chrom_part.startswith("chr"):
                chrom = f"chr{chrom_part}"
            else:
                chrom = chrom_part
            s, e = se.split("-")
            s, e = int(s), int(e)
            if s > e:
                s, e = e, s
            rows.append((chrom, s, e, pid))
        except Exception:
            logging.warning(f"Skipping malformed peak ID: {pid}")
            continue

    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "peak_id"]).drop_duplicates()
    # keep only canonical chromosomes
    df = df[df["chrom"].astype(str).str.match(include_regex)].reset_index(drop=True)
    return df

def calculate_peak_to_tg_distance_score(
    peak_bed_file,
    tss_bed_file,
    peak_gene_dist_file,
    mesc_atac_peak_loc_df, 
    gene_tss_df, 
    max_peak_distance=1e6, 
    distance_factor_scale=25000, 
    force_recalculate=False
) -> pd.DataFrame:
    """
    Compute peak-to-gene distance features (BEDTools-based), ensuring BED compliance.
    """
    # Validate and convert peaks to BED format
    required_cols = {"chrom", "start", "end", "peak_id"}
    if not required_cols.issubset(mesc_atac_peak_loc_df.columns):
        logging.warning("Converting peak index to BED format (chr/start/end parsing)")
        mesc_atac_peak_loc_df = build_peak_locs_from_index(mesc_atac_peak_loc_df.index)

    print("\nmesc_atac_peak_loc_df")
    print(mesc_atac_peak_loc_df.head())
    
    # Ensure numeric types
    mesc_atac_peak_loc_df["start"] = mesc_atac_peak_loc_df["start"].astype(int)
    mesc_atac_peak_loc_df["end"] = mesc_atac_peak_loc_df["end"].astype(int)

    # Ensure proper columns for gene_tss_df
    if not {"chrom", "start", "end", "name"}.issubset(gene_tss_df.columns):
        gene_tss_df = gene_tss_df.rename(columns={"chromosome_name": "chrom", "gene_start": "start", "gene_end": "end"})

    print("\ngene_tss_df")
    print(gene_tss_df.head())
    
    # Step 1: Write valid BED files if missing
    if not os.path.isfile(peak_bed_file) or not os.path.isfile(tss_bed_file) or force_recalculate:
        logging.info("Writing BED files for peaks and gene TSSs")
        pybedtools.BedTool.from_dataframe(mesc_atac_peak_loc_df[["chrom", "start", "end", "peak_id"]]).saveas(peak_bed_file)
        pybedtools.BedTool.from_dataframe(gene_tss_df[["chrom", "start", "end", "name"]]).saveas(tss_bed_file)

    # Step 2: Run BEDTools overlap
    logging.info(f"  - Locating peaks within ±{max_peak_distance:,} bp of TSSs")
    peak_bed = pybedtools.BedTool(peak_bed_file)
    tss_bed = pybedtools.BedTool(tss_bed_file)

    genes_near_peaks = find_genes_near_peaks(peak_bed, tss_bed, tss_distance_cutoff=max_peak_distance)
    
    genes_near_peaks = genes_near_peaks.rename(columns={"gene_id": "target_id"})
    genes_near_peaks["target_id"] = genes_near_peaks["target_id"].apply(standardize_name)

    # Step 3: Compute distances and scores
    genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= max_peak_distance]
    genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / distance_factor_scale)

    # Step 4: Save and return
    genes_near_peaks.to_parquet(peak_gene_dist_file, compression="snappy", engine="pyarrow")
    logging.info(f"Saved peak–gene distance table: {genes_near_peaks.shape}")
    return genes_near_peaks

def process_single_tf(tf, tf_df, peak_to_gene_dist_df):
    # Compute softmax per peak (for this TF only)
    tf_df["sliding_window_tf_softmax"] = tf_df.groupby("peak_id")["sliding_window_score"].transform(softmax)

    merged = pd.merge(
        tf_df[["peak_id", "sliding_window_tf_softmax"]],
        peak_to_gene_dist_df,
        on="peak_id",
        how="inner"
    )
    merged["tf_tg_contrib"] = merged["sliding_window_tf_softmax"] * merged["TSS_dist_score"]

    tf_tg_reg_pot = (
        merged.groupby("target_id", as_index=False)
        .agg(reg_potential=("tf_tg_contrib", "sum"),
             motif_density=("peak_id", "nunique"))
        .rename(columns={"target_id": "TG"})
    )

    tf_tg_reg_pot["TF"] = tf
    return tf_tg_reg_pot

def calculate_tf_tg_regulatory_potential(
    sliding_window_score_file: Union[str, Path], 
    tf_tg_reg_pot_file: Union[str, Path], 
    peak_to_gene_dist_file: Union[str, Path],
    num_cpu: int=8,
    ) -> pd.DataFrame:
    
    sliding_window_score_file = Path(sliding_window_score_file)
    tf_tg_reg_pot_file = Path(tf_tg_reg_pot_file)
    peak_to_gene_dist_file = Path(peak_to_gene_dist_file)
    
    logging.info("Calculating TF–TG regulatory potential (per TF mode)")
    sliding_window_df = pd.read_parquet(sliding_window_score_file, engine="pyarrow")
    peak_to_gene_dist_df = pd.read_parquet(peak_to_gene_dist_file, engine="pyarrow")

    relevant_peaks = set(sliding_window_df["peak_id"].unique())
    peak_to_gene_dist_df = peak_to_gene_dist_df[peak_to_gene_dist_df["peak_id"].isin(relevant_peaks)]
    
    # --- Clean ---
    sliding_window_df = sliding_window_df.dropna(subset=["TF", "peak_id", "sliding_window_score"])
    
    sliding_window_df["TF"] = sliding_window_df["TF"].apply(standardize_name)
    peak_to_gene_dist_df["target_id"] = peak_to_gene_dist_df["target_id"].apply(standardize_name)

    # --- Group by TF ---
    tf_groups = {tf: df for tf, df in sliding_window_df.groupby("TF", sort=False)}

    logging.info(f"Processing {len(tf_groups)} TFs using {num_cpu} CPUs")
    results = []

    with ProcessPoolExecutor(max_workers=num_cpu) as ex:
        futures = {
            ex.submit(process_single_tf, tf, df, peak_to_gene_dist_df): tf
            for tf, df in tf_groups.items()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="TF processing"):
            tf = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                logging.error(f"TF {tf} failed: {e}")

    # --- Concatenate all TF results ---
    if not results:
        raise RuntimeError("No TF results were successfully computed.")

    tf_tg_reg_pot = pd.concat(results, ignore_index=True)
    tf_tg_reg_pot["motif_density"] = np.log1p(tf_tg_reg_pot["motif_density"].fillna(0))
    
    tf_tg_reg_pot["TF"] = tf_tg_reg_pot["TF"].apply(standardize_name)
    tf_tg_reg_pot["TG"] = tf_tg_reg_pot["TG"].apply(standardize_name)
    
    logging.info("TF-TG regulatory potential")
    logging.info(tf_tg_reg_pot.head())

    # --- Save ---
    tf_tg_reg_pot.to_parquet(tf_tg_reg_pot_file, engine="pyarrow", compression="snappy")
    logging.info(f"Saved TF–TG regulatory potential: {tf_tg_reg_pot.shape}")
    
    return tf_tg_reg_pot

def select_pkn_edges_from_df(df: pd.DataFrame, pkn_edges: set[Tuple[str, str]]):
    df['TF'] = df['TF'].str.upper()
    df['TG'] = df['TG'].str.upper()
    
    df['in_pkn'] = df.apply(
        lambda r: int((r['TF'], r['TG']) in pkn_edges or (r['TG'], r['TF']) in pkn_edges),
        axis=1
    )
    
    in_pkn_df = df[df['in_pkn'] == 1]
    not_in_pkn_df = df[df['in_pkn'] == 0]
    
    return in_pkn_df, not_in_pkn_df

def compute_minmax_expr_mean(
    tf_df: pd.DataFrame, 
    tg_df: pd.DataFrame, 
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Min–max normalize across columns (cells/pseudobulks) per row (gene),
    then return row-wise means.

    Returns
    -------
    mean_norm_tf_expr : DataFrame with columns ['TF', 'mean_tf_expr']
    mean_norm_tg_expr : DataFrame with columns ['TG', 'mean_tg_expr']
    """
    def _rowwise_minmax_mean(x: pd.DataFrame) -> pd.Series:
        xmin = x.min(axis=1)
        xmax = x.max(axis=1)
        denom = (xmax - xmin).replace(0, np.nan)          # avoid div-by-zero
        scaled = (x.sub(xmin, axis=0)).div(denom, axis=0) # row-wise scale
        scaled = scaled.fillna(0.0)                       # constant rows → 0
        return scaled.mean(axis=1)

    # Force index names so reset_index yields the correct key columns
    tf_means = _rowwise_minmax_mean(tf_df)
    tf_means.index = tf_means.index.astype(str)
    tf_means.index.name = "TF"
    mean_norm_tf_expr = tf_means.reset_index(name="mean_tf_expr")
    mean_norm_tf_expr["TF"] = mean_norm_tf_expr["TF"].str.upper()

    tg_means = _rowwise_minmax_mean(tg_df)
    tg_means.index = tg_means.index.astype(str)
    tg_means.index.name = "TG"
    mean_norm_tg_expr = tg_means.reset_index(name="mean_tg_expr")
    mean_norm_tg_expr["TG"] = mean_norm_tg_expr["TG"].str.upper()
    
    logging.debug(f"\nTF expression means: \n{mean_norm_tf_expr.head()}")
    logging.debug(f"\nTG expression means: \n{mean_norm_tg_expr.head()}")
    
    return mean_norm_tf_expr, mean_norm_tg_expr

def merge_tf_tg_attributes_with_combinations(
    tf_tg_df: pd.DataFrame, 
    tf_tg_reg_pot: pd.DataFrame, 
    mean_norm_tf_expr: pd.DataFrame, 
    mean_norm_tg_expr: pd.DataFrame,
    tf_tg_combo_attr_file: Union[str, Path],
    tf_vocab: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Merge TF-TG regulatory potential and expression means with all TF-TG combinations.
    """
    logging.info("\n  - Merging TF-TG Regulatory Potential")
    tf_tg_df = pd.merge(
        tf_tg_df,
        tf_tg_reg_pot,
        how="left",
        on=["TF", "TG"]
    ).fillna(0)
    logging.info(tf_tg_df.head())

    logging.info(f"    - Number of unique TFs: {tf_tg_df['TF'].nunique()}")

    logging.info("\n  - Merging mean min-max normalized TF expression")
    tf_tg_df = pd.merge(
        tf_tg_df,
        mean_norm_tf_expr,
        how="left",
        on=["TF"]
    ).dropna(subset="mean_tf_expr")
    logging.info(tf_tg_df.head())
    logging.info(f"    - Number of unique TFs: {tf_tg_df['TF'].nunique()}")

    logging.info("\n- Merging mean min-max normalized TG expression")

    tf_tg_df = pd.merge(
        tf_tg_df,
        mean_norm_tg_expr,
        how="left",
        on=["TG"]
    ).dropna(subset="mean_tg_expr")
    logging.info(tf_tg_df.head())
    logging.info(f"    - Number of unique TFs: {tf_tg_df['TF'].nunique()}")
    
    tf_tg_df["expr_product"] = tf_tg_df["mean_tf_expr"] * tf_tg_df["mean_tg_expr"]
    tf_tg_df["log_reg_pot"] = np.log1p(tf_tg_df["reg_potential"])
    tf_tg_df["motif_present"] = (tf_tg_df["motif_density"] > 0).astype(int)
    
    # Ensure consistent casing before validation
    tf_tg_df["TF"] = tf_tg_df["TF"].astype(str).str.upper()
    tf_tg_df["TG"] = tf_tg_df["TG"].astype(str).str.upper()
    
    def _assert_valid_tf_tg(df: pd.DataFrame, tf_vocab: Optional[set[str]] = None) -> pd.DataFrame:
        d = df.copy()
        d["TF"] = d["TF"].astype(str).str.strip()
        d["TG"] = d["TG"].astype(str).str.strip()

        # Only consider "too short" as bad if it's not in your TF vocabulary
        if tf_vocab is not None:
            bad_len = (d["TF"].str.len() <= 1) & (~d["TF"].isin(tf_vocab))
        else:
            # With no vocab, be extremely conservative: drop empties only
            bad_len = d["TF"].str.len() == 0

        if bad_len.any():
            n = int(bad_len.sum())
            examples = d.loc[bad_len, "TF"].unique()[:10]
            logging.warning(f"{n} rows dropped due to invalid TF values; examples: {examples}")
            d = d[~bad_len].copy()

        if d.empty:
            raise ValueError("All rows were dropped during TF/TG validation. Upstream mutation likely overwrote TFs.")
        return d

    # Validate and drop any damaged rows (e.g., TF == "T")
    tf_tg_df = _assert_valid_tf_tg(tf_tg_df, tf_vocab=tf_vocab)
    
    tf_tg_df.to_parquet(tf_tg_combo_attr_file, index=False)
    
    return tf_tg_df

def make_chrom_gene_tss_df(gene_tss_file, chrom_id, genome_dir):
    gene_tss_bed = pybedtools.BedTool(gene_tss_file)
    gene_tss_df = (
        gene_tss_bed
        .filter(lambda x: x.chrom == chrom_id)
        .saveas(os.path.join(genome_dir, f"{chrom_id}_gene_tss.bed"))
        .to_dataframe()
        .sort_values(by="start", ascending=True)
        )
    return gene_tss_df

def merge_tf_tg_data_with_pkn(
    df: pd.DataFrame, 
    string_csv_file: Union[str, Path], 
    trrust_csv_file: Union[str, Path], 
    kegg_csv_file: Union[str, Path],
    upscale_percent: float = 1.5,
    seed: int = 42,
    add_pkn_scores: bool = True,
    pkn_metadata_cols: Optional[dict[str, list[str]]] = None,
    *,
    normalize_tf_tg_symbols: bool = True,
    strip_version_suffix: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build union of STRING/TRRUST/KEGG edges, split positives/negatives (UNDIRECTED),
    balance per TF, and annotate per-source provenance flags. Returns:
        (tf_tg_balanced_with_pkn, tf_tg_balanced)

    tf_tg_balanced_with_pkn includes in_STRING/in_TRRUST/in_KEGG and n_sources (+ optional metadata).
    tf_tg_balanced is the same rows without the metadata columns.

    Normalization knobs:
    - normalize_tf_tg_symbols: uppercase TF/TG in the candidate DF.
    - strip_version_suffix: remove trailing '.<digits>' from TF/TG (e.g., 'TP53.1' -> 'TP53').
    """

    # -------- helpers (from your current implementation) --------
    def _load_pkn(csv_path: Union[str, Path], tf_col: str = "TF", tg_col: str = "TG") -> tuple[pd.DataFrame, Set[tuple[str, str]]]:
        pkn_df = pd.read_csv(csv_path)
        if tf_col not in pkn_df.columns or tg_col not in pkn_df.columns:
            raise ValueError(f"{csv_path} must contain columns '{tf_col}' and '{tg_col}'")
        pkn_df = pkn_df.copy()
        pkn_df[tf_col] = pkn_df[tf_col].astype(str).str.upper()
        pkn_df[tg_col] = pkn_df[tg_col].astype(str).str.upper()
        pkn_df = pkn_df.drop_duplicates(subset=[tf_col, tg_col])
        edge_set = set(map(tuple, pkn_df[[tf_col, tg_col]].to_numpy()))
        return pkn_df, edge_set

    def _make_undirected(edge_set: Set[tuple[str, str]]) -> Set[tuple[str, str]]:
        return edge_set | {(b, a) for (a, b) in edge_set}

    def _split_by_pkn_union(tf_tg_df: pd.DataFrame, pkn_union: set[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
        d = tf_tg_df.copy()
        d["TF"] = d["TF"].astype(str).str.upper()
        d["TG"] = d["TG"].astype(str).str.upper()
        undirected = pkn_union | {(b, a) for (a, b) in pkn_union}
        pairs = list(map(tuple, d[["TF", "TG"]].to_numpy()))
        mask = np.fromiter(((u, v) in undirected for (u, v) in pairs), dtype=bool, count=len(pairs))
        return d[mask].copy(), d[~mask].copy()

    def _flag_undirected(d: pd.DataFrame, s: Set[tuple[str, str]], col: str) -> None:
        undirected = _make_undirected(s)
        d[col] = d.apply(lambda r: int((r["TF"], r["TG"]) in undirected), axis=1)

    def _safe_merge_meta_bi(
        base: pd.DataFrame,
        meta_df: pd.DataFrame,
        keep_cols: list[str],
        prefix: Optional[str] = None,  # <— default: no extra prefixing
    ) -> pd.DataFrame:
        cols = ["TF", "TG"] + [c for c in keep_cols if c in meta_df.columns]
        if len(cols) <= 2:
            return base

        direct = meta_df[cols].drop_duplicates(["TF", "TG"]).copy()
        reversed_df = meta_df.rename(columns={"TF": "TG", "TG": "TF"})[cols].drop_duplicates(["TF", "TG"]).copy()

        def _pref(df_in: pd.DataFrame) -> pd.DataFrame:
            # If prefix is None/empty -> keep names as-is.
            if not prefix:
                return df_in.copy()
            # Otherwise, add prefix unless the column already starts with it (case-insensitive).
            rn = {}
            for c in df_in.columns:
                if c in ("TF", "TG"):
                    continue
                if re.match(rf"^{re.escape(prefix)}[_-]", c, flags=re.I):
                    rn[c] = c
                else:
                    rn[c] = f"{prefix}_{c}"
            return df_in.rename(columns=rn)

        direct_p = _pref(direct)
        rev_p    = _pref(reversed_df)

        out     = base.merge(direct_p, on=["TF", "TG"], how="left")
        out_rev = base.merge(rev_p,    on=["TF", "TG"], how="left")

        meta_cols = [c for c in direct_p.columns if c not in ("TF", "TG")]
        for c in meta_cols:
            out[c] = out[c].where(out[c].notna(), out_rev[c])

        return out

    # -------- candidate normalization (version-strip + uppercase) --------
    def _strip_version(s: str) -> str:
        return re.sub(r"\.\d+$", "", s)

    def _canonicalize_candidates(dfin: pd.DataFrame) -> pd.DataFrame:
        d = dfin.copy()
        for col in ("TF", "TG"):
            d[col] = d[col].astype(str).str.strip()
            if strip_version_suffix:
                d[col] = d[col].map(_strip_version)
            if normalize_tf_tg_symbols:
                d[col] = d[col].str.upper()
        return d

    # 1) Check existence of PKNs or build if missing
    missing = [p for p in [string_csv_file, trrust_csv_file, kegg_csv_file] if not os.path.isfile(p)]
    if missing:
        logging.info(f"Missing PKN files: {missing}")
        build_organism_pkns(ORGANISM_CODE, string_csv_file, trrust_csv_file, kegg_csv_file)

    if pkn_metadata_cols is None:
        pkn_metadata_cols = {
            "STRING": ["string_experimental_score", "string_database_score", "string_textmining_score", "string_combined_score"],
            "TRRUST": ["trrust_sign", "trrust_regulation", "trrust_pmids", "trrust_support_n"],
            "KEGG":   ["kegg_signal", "kegg_n_pathways", "kegg_pathways"],
        }

    # ---------------- load PKNs ----------------
    logging.info("  - Loading PKNs...")
    string_df, string_set = _load_pkn(string_csv_file)
    trrust_df, trrust_set = _load_pkn(trrust_csv_file)
    kegg_df,   kegg_set   = _load_pkn(kegg_csv_file)

    string_set_u = _make_undirected(string_set)
    trrust_set_u = _make_undirected(trrust_set)
    kegg_set_u   = _make_undirected(kegg_set)
    pkn_union_u  = string_set_u | trrust_set_u | kegg_set_u

    logging.info(f"  - Loaded {len(string_set_u):,} STRING, {len(trrust_set_u):,} TRRUST, {len(kegg_set_u):,} KEGG edges")

    # ---------------- normalize candidates ----------------
    df = _canonicalize_candidates(df)

    try:
        sample_syms = list(pd.unique(df[["TF", "TG"]].values.ravel()))[:5]
        logging.info(f"\tExample TF-TG data: {sample_syms}")
    except Exception:
        pass

    # ---------------- split ----------------
    logging.info("  - Splitting positives/negatives by PKN union")
    in_pkn_df, not_in_pkn_df = _split_by_pkn_union(df, pkn_union_u)
    logging.info("  - Splitting results:")
    logging.info(f"\tEdges in TF-TG data: {df.shape[0]:,}")
    logging.info(f"\tEdges in PKN union (undirected): {len(pkn_union_u):,}")
    logging.info(f"\tUnique TFs not in PKN: {not_in_pkn_df['TF'].nunique():,}")
    logging.info(f"\tUnique TGs not in PKN: {not_in_pkn_df['TG'].nunique():,}")
    logging.info(f"\tEdges not in PKN: {not_in_pkn_df.shape[0]:,}")
    logging.info(f"\tEdges in PKN: {in_pkn_df.shape[0]:,}")
    logging.info(f"\tFraction of TF-TG edges in PKN: {in_pkn_df.shape[0] / max(1, df.shape[0]):.2f}")

    if in_pkn_df.empty:
        raise ValueError("No TF–TG positives in PKN union after normalization.")

    # ---------------- balance per TF ----------------
    rng = np.random.default_rng(seed)
    balanced_rows: list[pd.DataFrame] = []
    logging.info("  - Balancing TF–TG pairs per TF")
    for tf, pos_grp in in_pkn_df.groupby("TF"):
        neg_cands = not_in_pkn_df[not_in_pkn_df["TF"] == tf]
        if neg_cands.empty:
            continue
        sampled_neg = neg_cands.sample(
            n=len(pos_grp), replace=True, random_state=int(rng.integers(1_000_000_000))
        )
        pos_up = pos_grp.sample(frac=upscale_percent, replace=True,
                                random_state=int(rng.integers(1_000_000_000))).assign(label=1)
        neg_up = sampled_neg.sample(frac=upscale_percent, replace=True,
                                    random_state=int(rng.integers(1_000_000_000))).assign(label=0)
        balanced_rows.append(pd.concat([pos_up, neg_up], ignore_index=True))

    if not balanced_rows:
        raise ValueError("Balancing produced no rows (no TF had negatives). Check coverage and symbol casing.")

    # ---------------- concat + shuffle ----------------
    tf_tg_balanced_no_pkn_scores = pd.concat(balanced_rows, ignore_index=True)
    tf_tg_balanced_no_pkn_scores = tf_tg_balanced_no_pkn_scores.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    tf_tg_balanced_with_pkn_scores = tf_tg_balanced_no_pkn_scores.copy()

    # ---------------- flags (undirected) ----------------
    _flag_undirected(tf_tg_balanced_with_pkn_scores, string_set, "in_STRING")
    _flag_undirected(tf_tg_balanced_with_pkn_scores, trrust_set, "in_TRRUST")
    _flag_undirected(tf_tg_balanced_with_pkn_scores,   kegg_set, "in_KEGG")
    tf_tg_balanced_with_pkn_scores["n_sources"] = (
        tf_tg_balanced_with_pkn_scores[["in_STRING","in_TRRUST","in_KEGG"]].sum(axis=1)
    )

    # ---------------- metadata (bi-directional merge with fallback) ----------------
    if add_pkn_scores:
        tf_tg_balanced_with_pkn_scores = _safe_merge_meta_bi(
            tf_tg_balanced_with_pkn_scores, string_df, pkn_metadata_cols.get("STRING", [])
        )
        tf_tg_balanced_with_pkn_scores = _safe_merge_meta_bi(
            tf_tg_balanced_with_pkn_scores, trrust_df, pkn_metadata_cols.get("TRRUST", [])
        )
        tf_tg_balanced_with_pkn_scores = _safe_merge_meta_bi(
            tf_tg_balanced_with_pkn_scores, kegg_df,   pkn_metadata_cols.get("KEGG", [])
        )
    else:
        logging.warning("Skipping PKN metadata merge (add_pkn_scores=False). tf_tg_balanced_with_pkn_scores == tf_tg_balanced_no_pkn_scores")

    return tf_tg_balanced_with_pkn_scores, tf_tg_balanced_no_pkn_scores

# ----- MultiomicTransformer Chromsome-Specific Dataset -----
def build_global_tg_vocab(gene_tss_file: Union[str, Path], vocab_file: Union[str, Path]) -> dict[str, int]:
    """
    Build a global TG vocab from the TSS file with contiguous IDs [0..N-1].
    Overwrites existing vocab if it's missing or non-contiguous.
    """
    # 1) Load all genes genome-wide (bed: chrom start end name)
    gene_tss_bed = pybedtools.BedTool(gene_tss_file)
    gene_tss_df = gene_tss_bed.to_dataframe().sort_values(by="start", ascending=True)

    # 2) Canonical symbol list (MUST match downstream normalization)
    names = [standardize_name(n) for n in gene_tss_df["name"].astype(str).unique()]
    names = sorted(set(names))  # unique + stable order

    # 3) Build fresh contiguous mapping
    vocab = {name: i for i, name in enumerate(names)}

    # 4) Atomic overwrite
    tmp = str(vocab_file) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(vocab, f)
    os.replace(tmp, vocab_file)

    logging.info(f"Rebuilt TG vocab with {len(vocab)} genes (contiguous).")
    return vocab

def create_single_cell_tensors(
    gene_tss_df: pd.DataFrame,
    sample_names: list[str],
    dataset_processed_data_dir: Path,
    tg_vocab: dict[str, int],
    tf_vocab: dict[str, int],
    chrom_id: str,
    single_cell_dir: Path
):

    # --- set chromosome-specific TG list ---
    chrom_tg_names = set(gene_tss_df["name"].unique())

    for sample_name in sample_names:        
        sample_processed_data_dir = dataset_processed_data_dir / sample_name

        tg_sc_file = sample_processed_data_dir / "TG_singlecell.tsv"
        re_sc_file = sample_processed_data_dir / "RE_singlecell.tsv"

        if not (tg_sc_file.exists() and re_sc_file.exists()):
            logging.warning(f"Skipping {sample_name}: missing TG/RE single-cell files")
            continue

        TG_sc = pd.read_csv(tg_sc_file, sep="\t", index_col=0)
        RE_sc = pd.read_csv(re_sc_file, sep="\t", index_col=0)

        # --- restrict TGs to chromosome + vocab ---
        tg_rows = [g for g in TG_sc.index if g in chrom_tg_names]
        TG_sc_chr = TG_sc.loc[tg_rows]

        tg_tensor_sc, tg_names_kept, tg_ids = align_to_vocab(
            TG_sc_chr.index.tolist(),
            tg_vocab,
            torch.tensor(TG_sc_chr.values, dtype=torch.float32),
            label="TG"
        )
        torch.save(tg_tensor_sc, single_cell_dir / f"{sample_name}_tg_tensor_singlecell_{chrom_id}.pt")
        torch.save(torch.tensor(tg_ids, dtype=torch.long), single_cell_dir / f"{sample_name}_tg_ids_singlecell_{chrom_id}.pt")
        atomic_json_dump(tg_names_kept, single_cell_dir / f"{sample_name}_tg_names_singlecell_{chrom_id}.json")

        # --- restrict ATAC peaks to chromosome ---
        re_rows = [p for p in RE_sc.index if p.startswith(f"{chrom_id}:")]
        RE_sc_chr = RE_sc.loc[re_rows]

        atac_tensor_sc = torch.tensor(RE_sc_chr.values, dtype=torch.float32)
        torch.save(atac_tensor_sc, single_cell_dir / f"{sample_name}_atac_tensor_singlecell_{chrom_id}.pt")

        # --- TF tensor (subset of TGs) ---
        tf_tensor_sc = None
        tf_rows = [g for g in TG_sc.index if g in tf_vocab]
        if tf_rows:
            TF_sc = TG_sc.loc[tf_rows]
            tf_tensor_sc, tf_names_kept, tf_ids = align_to_vocab(
                TF_sc.index.tolist(),
                tf_vocab,
                torch.tensor(TF_sc.values, dtype=torch.float32),
                label="TF"
            )
            torch.save(tf_tensor_sc, single_cell_dir / f"{sample_name}_tf_tensor_singlecell_{chrom_id}.pt")
            torch.save(torch.tensor(tf_ids, dtype=torch.long), single_cell_dir / f"{sample_name}_tf_ids_singlecell_{chrom_id}.pt")
            atomic_json_dump(tf_names_kept, single_cell_dir / f"{sample_name}_tf_names_singlecell_{chrom_id}.json")
        else:
            logging.warning(f"No TFs from global vocab found in sample {sample_name}")
            tf_tensor_sc, tf_ids = None, []

        logging.info(
            f"Saved single-cell tensors for {sample_name} | "
            f"TGs={tg_tensor_sc.shape}, "
            f"TFs={tf_tensor_sc.shape if tf_tensor_sc is not None else 'N/A'}, "
            f"RE={atac_tensor_sc.shape}"
        )

def aggregate_pseudobulk_datasets(gene_tss_df: pd.DataFrame, sample_names: list[str], dataset_processed_data_dir: Path, chrom_id: str):
    
    # ----- Combine Pseudobulk Data into a Training Dataset -----
    TG_pseudobulk_global = []
    TG_pseudobulk_samples = []
    RE_pseudobulk_samples = []
    peaks_df_samples = []

    logging.info("\nLoading processed pseudobulk datasets:")
    logging.info(f"  - Sample names: {sample_names}")
    logging.info(f"  - Looking for processed samples in {dataset_processed_data_dir}")
    for sample_name in sample_names:
        sample_processed_data_dir = dataset_processed_data_dir / sample_name
        if not os.path.exists(sample_processed_data_dir):
            logging.warning(f"Skipping {sample_name}: directory not found")
            continue
        if sample_name in VALIDATION_DATASETS:
            logging.warning(f"Skipping {sample_name}: in VALIDATION_DATASETS list")
            continue
        else:
            logging.info(f"  - Processing Pseudobulk data for {sample_name}")
            TG_pseudobulk = pd.read_csv(os.path.join(sample_processed_data_dir, "TG_pseudobulk.tsv"), sep="\t", index_col=0)
            RE_pseudobulk = pd.read_csv(os.path.join(sample_processed_data_dir, "RE_pseudobulk.tsv"), sep="\t", index_col=0)

            logging.debug("\n  - Total Pseudobulk Genes and Peaks")
            logging.debug(f"\tTG_pseudobulk: {TG_pseudobulk.shape[0]:,} Genes x {TG_pseudobulk.shape[1]} metacells")
            logging.debug(f"\tRE_pseudobulk: {RE_pseudobulk.shape[0]:,} Peaks x {RE_pseudobulk.shape[1]} metacells")

            TG_chr_specific = TG_pseudobulk.loc[TG_pseudobulk.index.intersection(gene_tss_df['name'].unique())]
            RE_chr_specific = RE_pseudobulk[RE_pseudobulk.index.str.startswith(f"{chrom_id}:")]

            logging.debug(f"\n  - Restricted to {chrom_id} Genes and Peaks: ")
            logging.debug(f"\tTG_chr_specific: {TG_chr_specific.shape[0]} Genes x {TG_chr_specific.shape[1]} metacells")
            logging.debug(f"\tRE_chr_specific: {RE_chr_specific.shape[0]:,} Peaks x {RE_chr_specific.shape[1]} metacells")

            peaks_df = (
                RE_chr_specific.index.to_series()
                .str.split("[:-]", expand=True)
                .rename(columns={0: "chrom", 1: "start", 2: "end"})
            )
            peaks_df["start"] = peaks_df["start"].astype(int)
            peaks_df["end"] = peaks_df["end"].astype(int)
            peaks_df["peak_id"] = RE_chr_specific.index
            
            TG_pseudobulk_global.append(TG_pseudobulk)
            TG_pseudobulk_samples.append(TG_chr_specific)
            RE_pseudobulk_samples.append(RE_chr_specific)
            peaks_df_samples.append(peaks_df)
            
        def _agg_sum(dfs: list[pd.DataFrame]) -> pd.DataFrame:
            """Sum rows across samples; rows are aligned by index."""
            if len(dfs) == 0:
                raise ValueError("No DataFrames provided to aggregate.")
            if len(dfs) == 1:
                return dfs[0]
            return pd.concat(dfs).groupby(level=0).sum()

        def _agg_first(dfs: list[pd.DataFrame]) -> pd.DataFrame:
            """Keep first occurrence per index (for metadata like peak coords)."""
            if len(dfs) == 0:
                raise ValueError("No DataFrames provided to aggregate.")
            if len(dfs) == 1:
                return dfs[0]
            return pd.concat(dfs).groupby(level=0).first()
        
        # Aggregate the pseudobulk for all samples
        total_TG_pseudobulk_global = _agg_sum(TG_pseudobulk_global)
        total_TG_pseudobulk_chr    = _agg_sum(TG_pseudobulk_samples)
        total_RE_pseudobulk_chr    = _agg_sum(RE_pseudobulk_samples)
        total_peaks_df             = _agg_first(peaks_df_samples)
        
    return total_TG_pseudobulk_global, total_TG_pseudobulk_chr, total_RE_pseudobulk_chr, total_peaks_df

def create_or_load_genomic_windows(window_size, chrom_id, genome_window_file, chrom_sizes_file, force_recalculate=False):
    if not os.path.exists(genome_window_file) or force_recalculate:
        
        logging.info("\nCreating genomic windows")
        genome_windows = pybedtools.bedtool.BedTool().window_maker(g=chrom_sizes_file, w=window_size)
        chrom_windows = (
            genome_windows
            .filter(lambda x: x.chrom == chrom_id)  # TEMPORARY Restrict to one chromosome for testing
            .saveas(genome_window_file)
            .to_dataframe()
        )
        logging.info(f"  - Created {chrom_windows.shape[0]} windows")
    else:
        
        logging.info("\nLoading existing genomic windows")
        chrom_windows = pybedtools.BedTool(genome_window_file).to_dataframe()
        
    chrom_windows = chrom_windows.reset_index(drop=True)
    chrom_windows["win_idx"] = chrom_windows.index
    
    return chrom_windows

def make_peak_to_window_map(peaks_bed: pd.DataFrame, windows_bed: pd.DataFrame) -> dict[str, int]:
    """
    Map each peak to the window it overlaps the most.
    If a peak ties across multiple windows, assign randomly.
    
    Parameters
    ----------
    peaks_bed : DataFrame
        Must have ['chrom','start','end','peak_id'] columns
    windows_bed : DataFrame
        Must have ['chrom','start','end','win_idx'] columns
    
    Returns
    -------
    mapping : dict[str, int]
        peak_id -> window index
    """
    bedtool_peaks = pybedtools.BedTool.from_dataframe(peaks_bed)
    bedtool_windows = pybedtools.BedTool.from_dataframe(windows_bed)

    overlaps = {}
    for interval in bedtool_peaks.intersect(bedtool_windows, wa=True, wb=True):
        peak_id = interval.name
        win_idx = int(interval.fields[-1])  # window index
        peak_start, peak_end = int(interval.start), int(interval.end)
        win_start, win_end = int(interval.fields[-3]), int(interval.fields[-2])

        # Compute overlap length
        overlap_len = min(peak_end, win_end) - max(peak_start, win_start)

        # Track best overlap for each peak
        if peak_id not in overlaps:
            overlaps[peak_id] = []
        overlaps[peak_id].append((overlap_len, win_idx))

    # Resolve ties by max overlap, then random choice
    mapping = {}
    for peak_id, ov_list in overlaps.items():
        max_overlap = max(ov_list, key=lambda x: x[0])[0]
        candidates = [win_idx for ol, win_idx in ov_list if ol == max_overlap]
        mapping[peak_id] = random.choice(candidates)  # pick randomly if tie

    return mapping

def align_to_vocab(names, vocab, tensor_all, label="genes"):
    """
    Restrict to the subset of names that exist in the global vocab.
    Returns:
      aligned_tensor : [num_kept, C] (chromosome-specific subset)
      kept_names     : list[str] of kept names (order = aligned_tensor rows)
      kept_ids       : list[int] global vocab indices for kept names
    """
    kept_ids = []
    kept_names = []
    aligned_rows = []

    for i, n in enumerate(names):
        vid = vocab.get(n)
        if vid is not None:
            kept_ids.append(vid)
            kept_names.append(n)
            aligned_rows.append(tensor_all[i])

    if not kept_ids:
        raise ValueError(f"No {label} matched the global vocab.")

    aligned_tensor = torch.stack(aligned_rows, dim=0)  # [num_kept, num_cells]

    return aligned_tensor, kept_names, kept_ids

def build_motif_mask(tf_names, tg_names, sliding_window_df, genes_near_peaks):
    """
    Build motif mask [TG x TF] with max logodds per (TG, TF).
    """
    # map TFs and TGs to ids
    tf_index = {tf: i for i, tf in enumerate(tf_names)}
    tg_index = {tg: i for i, tg in enumerate(tg_names)}

    # restrict to known TFs
    sliding_window_df = sliding_window_df[sliding_window_df["TF"].isin(tf_index)]
    
    # merge motif hits with target genes (peak_id → TG)
    merged = sliding_window_df.merge(
        genes_near_peaks[["peak_id", "target_id"]],
        on="peak_id",
        how="inner"
    )

    # drop TGs not in tg_index
    merged = merged[merged["target_id"].isin(tg_index)]

    # map names → indices
    merged["tf_idx"] = merged["TF"].map(tf_index)
    merged["tg_idx"] = merged["target_id"].map(tg_index)

    # groupby max(sliding_window_score) per (TG, TF)
    agg = merged.groupby(["tg_idx", "tf_idx"])["sliding_window_score"].max().reset_index()

    # construct sparse COO
    mask = sp.coo_matrix(
        (agg["sliding_window_score"], (agg["tg_idx"], agg["tf_idx"])),
        shape=(len(tg_names), len(tf_names)),
        dtype=np.float32
    ).toarray()

    return mask

def precompute_input_tensors(
    output_dir: str,
    genome_wide_tf_expression: np.ndarray,   # [num_TF, num_cells]
    TG_scaled: np.ndarray,                   # [num_TG_chr, num_cells] (already standardized)
    total_RE_pseudobulk_chr,                 # pd.DataFrame: rows=peak_id, cols=metacells
    window_map,
    windows,                            # pd.DataFrame with shape[0] = num_windows
    dtype: torch.dtype = torch.float32,
):
    """
    Builds & saves:
      - tf_tensor_all.pt                        [num_TF, num_cells]
      - tg_tensor_all_{chr}.pt                  [num_TG_chr, num_cells]
      - atac_window_tensor_all_{chr}.pt         [num_windows, num_cells]

    Returns:
      (tf_tensor_all, tg_tensor_all, atac_window_tensor_all)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- TF tensor ----
    tf_tensor_all = torch.as_tensor(
        np.asarray(genome_wide_tf_expression, dtype=np.float32), dtype=dtype
    )

    # ---- TG tensor (scaled) ----
    tg_tensor_all = torch.as_tensor(
        np.asarray(TG_scaled, dtype=np.float32), dtype=dtype
    )

    # ---- ATAC window tensor ----
    num_windows = int(windows.shape[0])
    num_peaks   = int(total_RE_pseudobulk_chr.shape[0])

    rows, cols, vals = [], [], []
    peak_to_idx = {p: i for i, p in enumerate(total_RE_pseudobulk_chr.index)}
    for peak_id, win_idx in window_map.items():
        peak_idx = peak_to_idx.get(peak_id)
        if peak_idx is not None and 0 <= win_idx < num_windows:
            rows.append(win_idx)
            cols.append(peak_idx)
            vals.append(1.0)

    if not rows:
        raise ValueError("No peaks from window_map matched rows in total_RE_pseudobulk_chr.")

    W = sp.csr_matrix((vals, (rows, cols)), shape=(num_windows, num_peaks))
    atac_window = W @ total_RE_pseudobulk_chr.values  # [num_windows, num_cells]

    atac_window_tensor_all = torch.as_tensor(
        atac_window.astype(np.float32), dtype=dtype
    )

    return tf_tensor_all, tg_tensor_all, atac_window_tensor_all

def build_distance_bias(
    genes_near_peaks: pd.DataFrame,
    window_map: Dict[str, int],
    tg_names_kept: Iterable[str],
    num_windows: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    mode: str = "logsumexp",   # "max" | "sum" | "mean" | "logsumexp"
) -> torch.Tensor:
    """
    Build a [num_windows x num_tg_kept] distance-bias tensor aligned to the kept TGs.

    Args:
        genes_near_peaks: DataFrame with at least columns:
            - 'peak_id' (str): peak identifier that matches keys in window_map
            - 'target_id' (str): TG name
            - 'TSS_dist_score' (float): precomputed distance score
        window_map: dict mapping peak_id -> window index (0..num_windows-1).
        tg_names_kept: iterable of TG names kept after vocab filtering; column order target.
        num_windows: total number of genomic windows.
        dtype: torch dtype for the output tensor (default: torch.float32).
        device: optional torch device for the output tensor.
        mode: pooling strategy if multiple peaks map to the same (window, TG).
              Options = {"max", "sum", "mean", "logsumexp"}

    Returns:
        dist_bias: torch.Tensor of shape [num_windows, len(tg_names_kept)],
                   where each entry is an aggregated TSS distance score.
    """
    tg_names_kept = list(tg_names_kept)
    num_tg_kept = len(tg_names_kept)

    dist_bias = torch.zeros((num_windows, num_tg_kept), dtype=dtype, device=device)
    tg_index_map = {tg: i for i, tg in enumerate(tg_names_kept)}

    from collections import defaultdict
    scores_map = defaultdict(list)

    # Collect all scores for each (window, TG)
    for _, row in genes_near_peaks.iterrows():
        win_idx = window_map.get(row["peak_id"])
        tg_idx  = tg_index_map.get(row["target_id"])
        if win_idx is not None and tg_idx is not None:
            scores_map[(win_idx, tg_idx)].append(float(row["TSS_dist_score"]))

    # Aggregate according to pooling mode
    for (win_idx, tg_idx), scores in scores_map.items():
        scores_tensor = torch.tensor(scores, dtype=dtype, device=device)

        if mode == "max":
            dist_bias[win_idx, tg_idx] = scores_tensor.max()
        elif mode == "sum":
            dist_bias[win_idx, tg_idx] = scores_tensor.sum()
        elif mode == "mean":
            dist_bias[win_idx, tg_idx] = scores_tensor.mean()
        elif mode == "logsumexp":
            dist_bias[win_idx, tg_idx] = torch.logsumexp(scores_tensor, dim=0)
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

    return dist_bias

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Main preprocessing script.")
    parser.add_argument("--num_cpu", required=True, help="Number of cores for parallel processing")
    args = parser.parse_args()
    num_cpu = int(args.num_cpu)
    
    # ----- GLOBAL SETTINGS -----
    # TF and TG vocab files
    common_tf_vocab_file: Path =  COMMON_DATA / f"tf_vocab.json"
    common_tg_vocab_file: Path =  COMMON_DATA / f"tg_vocab.json"
    
    os.makedirs(COMMON_DATA, exist_ok=True)
    
    # Genome files
    genome_fasta_file = GENOME_DIR / (ORGANISM_CODE + ".fa.gz")
    chrom_sizes_file = GENOME_DIR / (ORGANISM_CODE + ".chrom.sizes")
    
    if not os.path.isdir(GENOME_DIR):
        os.makedirs(GENOME_DIR)
    
    if not os.path.isfile(genome_fasta_file):
        download_genome_fasta(
            organism_code=ORGANISM_CODE,
            save_dir=GENOME_DIR
        )
        
    if not os.path.isfile(chrom_sizes_file):
        download_chrom_sizes(
            organism_code=ORGANISM_CODE,
            save_dir=GENOME_DIR
        )
        
    # Check organism code
    if ORGANISM_CODE == "mm10":
        ensemble_dataset_name = "mmusculus_gene_ensembl"
    elif ORGANISM_CODE == "hg38":
        ensemble_dataset_name = "hsapiens_gene_ensembl"
    else:
        raise ValueError(f"Organism not recognized: {ORGANISM_CODE} (must be 'mm10' or 'hg38').")
    
    if not os.path.isfile(GENE_TSS_FILE):
        download_gene_tss_file(
            save_file=GENE_TSS_FILE,
            gene_dataset_name=ensemble_dataset_name,
        )
        
    # Format the gene TSS file to BED format (chrom, start, end, name)
    gene_tss_df = make_gene_tss_bed_file(
        gene_tss_file=GENE_TSS_FILE,
        genome_dir=GENOME_DIR
    )
    
    # ----- BUILD GLOBAL TG VOCAB FROM THE GENE TSS FILE-----
    tg_vocab = build_global_tg_vocab(GENE_TSS_FILE, common_tg_vocab_file)

    # PKN files
    string_csv_file = STRING_DIR / f"string_{ORGANISM_CODE}_pkn.csv"
    trrust_csv_file = TRRUST_DIR / f"trrust_{ORGANISM_CODE}_pkn.csv"
    kegg_csv_file = KEGG_DIR / f"kegg_{ORGANISM_CODE}_pkn.csv"
    
    IGNORE_PROCESSED_FILES = False
    logging.info(f"IGNORE_PROCESSED_FILES: {IGNORE_PROCESSED_FILES}")
    
    PROCESS_CHROMOSOME_SPECIFIC_DATA = True
    
    # Sample-specific preprocessing
    total_tf_set: Set[str] = set()
    chrom_set: Set[str] = set()
    for sample_name in SAMPLE_NAMES:
        sample_input_dir = RAW_DATA / sample_name
        
        # Input Files (raw or processed scRNA-seq and scATAC-seq data)
        processed_rna_file = sample_input_dir / "scRNA_seq_processed.parquet"
        processed_atac_file = sample_input_dir / "scATAC_seq_processed.parquet"
        
        # Output Files
        peak_bed_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "peaks.bed"
        peak_to_gene_dist_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "peak_to_gene_dist.parquet"
        sliding_window_score_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "sliding_window.parquet"
        tf_tg_reg_pot_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "tf_tg_regulatory_potential.parquet"
        tf_tg_combo_attr_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "tf_tg_combos_with_attributes.parquet"
        gat_training_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "tf_tg_data.parquet"
        
        # Sample-specific cache files
        tf_tensor_path: Path =        SAMPLE_DATA_CACHE_DIR / "tf_tensor_all.pt"
        metacell_name_file: Path =    SAMPLE_DATA_CACHE_DIR / "metacell_names.json"
        sample_tf_name_file: Path =   SAMPLE_DATA_CACHE_DIR / "tf_names.json"
        tf_id_file: Path =            SAMPLE_DATA_CACHE_DIR / "tf_ids.pt"
        
        logging.info(f"\nOutput Directory: {SAMPLE_PROCESSED_DATA_DIR / sample_name}")
        os.makedirs(SAMPLE_PROCESSED_DATA_DIR / sample_name, exist_ok=True)
        os.makedirs(SAMPLE_DATA_CACHE_DIR, exist_ok=True)
        
        sample_raw_10x_rna_data_dir = RAW_10X_RNA_DATA_DIR / sample_name
    
        # ----- LOAD AND PROCESS RNA AND ATAC DATA -----
        processed_rna_df, processed_atac_df, pseudobulk_rna_df, pseudobulk_atac_df = process_or_load_rna_atac_data(
            sample_input_dir, 
            ignore_processed_files=IGNORE_PROCESSED_FILES, 
            raw_10x_rna_data_dir=sample_raw_10x_rna_data_dir, 
            raw_atac_peak_file=RAW_ATAC_PEAK_MATRIX_FILE,
            sample_name=sample_name,
            neighbors_k=NEIGHBORS_K,
            leiden_resolution=LEIDEN_RESOLUTION
        )
        
                
        # ----- GET TFs, TGs, and TF-TG combinations -----
        genes = processed_rna_df.index.to_list()
        peaks = processed_atac_df.index.to_list()
        
        logging.info("\nProcessed RNA and ATAC files loaded")
        logging.info(f"  - Number of genes: {processed_rna_df.shape[0]}: {genes[:3]}")
        logging.info(f"  - Number of peaks: {processed_atac_df.shape[0]}: {peaks[:3]}")

        tfs, tgs, tf_tg_df = create_tf_tg_combination_files(genes, TF_FILE, SAMPLE_PROCESSED_DATA_DIR)
        
        total_tf_set.update(tfs)

        # Format the peaks to BED format (chrom, start, end, peak_id)
        peak_locs_df = format_peaks(pd.Series(processed_atac_df.index)).rename(columns={"chromosome": "chrom"})
        
        chrom_set.update(peak_locs_df["chrom"].unique())

        if not os.path.isfile(peak_bed_file):
            # Write the peak BED file
            peak_bed_file.parent.mkdir(parents=True, exist_ok=True)
            pybedtools.BedTool.from_dataframe(
                peak_locs_df[["chrom", "start", "end", "peak_id"]]
            ).saveas(peak_bed_file)
        
        # ----- CALCULATE PEAK TO TG DISTANCE -----
        # Calculate the distance from each peak to each gene TSS
        if not os.path.isfile(peak_to_gene_dist_file) or IGNORE_PROCESSED_FILES:
            # Download the gene TSS file from Ensembl if missing

            logging.info("\nCalculating peak to TG distance score")
            peak_to_gene_dist_df = calculate_peak_to_tg_distance_score(
                peak_bed_file=peak_bed_file,
                tss_bed_file=GENE_TSS_FILE,
                peak_gene_dist_file=peak_to_gene_dist_file,
                mesc_atac_peak_loc_df=peak_locs_df,
                gene_tss_df=gene_tss_df,
                force_recalculate=IGNORE_PROCESSED_FILES
            )
            
        else:
            logging.info("Peak to gene distance file found, loading...")
            peak_to_gene_dist_df = pd.read_parquet(peak_to_gene_dist_file, engine="pyarrow")
        
        logging.info("\nPeak to gene distance")
        logging.info("  - Number of peaks to gene distances: " + str(peak_to_gene_dist_df.shape[0]))
        logging.debug("  - Example peak to gene distances: \n" + str(peak_to_gene_dist_df.head()))
        
        # ----- SLIDING WINDOW TF-PEAK SCORE -----
        if not os.path.isfile(sliding_window_score_file):

            peaks_df = pybedtools.BedTool(peak_bed_file)

            logging.info("\nRunning sliding window scan")
            run_sliding_window_scan(
                tf_name_list=tfs,
                tf_info_file=str(TF_FILE),
                motif_dir=str(MOTIF_DIR),
                genome_fasta=str(genome_fasta_file),
                peak_bed_file=str(peak_bed_file),
                output_file=sliding_window_score_file,
                num_cpu=num_cpu
            )

        tf_df = processed_rna_df[processed_rna_df.index.isin(tfs)]
        logging.debug("\nTFs in RNA data")
        logging.debug(tf_df.head())
        logging.info(f"  - TFs in RNA data: {tf_df.shape[0]}")
        
        tg_df = processed_rna_df[processed_rna_df.index.isin(tgs)]
        logging.debug("\nTGs in RNA data")
        logging.debug(tg_df.head())
        logging.info(f"  - TGs in RNA data: {tg_df.shape[0]}")
        sliding_window_df = pd.read_parquet(sliding_window_score_file, engine="pyarrow")
        logging.debug("  - Example sliding window scores: \n" + str(sliding_window_df.head()))
        
        # Build tf_df as you do:
        tf_df = processed_rna_df[processed_rna_df.index.isin(tfs)]
        logging.debug(f"tf_df shape: {tf_df.shape}")
        logging.debug(f"tf_df index sample: {list(tf_df.index[:10])}")

        
        # ----- COMPUTE TF/TG EXPRESSION MEANS -----
        mean_norm_tf_expr, mean_norm_tg_expr = compute_minmax_expr_mean(tf_df, tg_df)

        # ----- CALCULATE TF-TG REGULATORY POTENTIAL -----
        if not os.path.isfile(tf_tg_reg_pot_file):
            tf_tg_reg_pot = calculate_tf_tg_regulatory_potential(
                sliding_window_score_file, tf_tg_reg_pot_file, peak_to_gene_dist_file, num_cpu)
        else:
            logging.info("\nLoading TF-TG regulatory potential scores")
            tf_tg_reg_pot = pd.read_parquet(tf_tg_reg_pot_file, engine="pyarrow")
            logging.debug("  - Example TF-TG regulatory potential: " + str(tf_tg_reg_pot.head()))

        # ----- MERGE TF-TG ATTRIBUTES WITH COMBINATIONS -----
        if not os.path.isfile(tf_tg_combo_attr_file):
            logging.info("\nMerging TF-TG attributes with all combinations")
            tf_tg_df = merge_tf_tg_attributes_with_combinations(
                tf_tg_df, tf_tg_reg_pot, mean_norm_tf_expr, mean_norm_tg_expr, tf_tg_combo_attr_file, set(tfs))
        else:
            logging.info("\nLoading TF-TG attributes with all combinations")
            tf_tg_df = pd.read_parquet(tf_tg_combo_attr_file, engine="pyarrow")

        # ----- MERGE TF-TG DATA WITH PKN -----
        logging.info("\nMerging TF-TG data with PKN")
        tf_tg_balanced_with_pkn, tf_tg_balanced_no_pkn = merge_tf_tg_data_with_pkn(
            tf_tg_df, 
            string_csv_file, 
            trrust_csv_file, 
            kegg_csv_file
        )
        logging.debug(f"  - Example of TF-TG data with PKN: {tf_tg_balanced_with_pkn.head()}")

        logging.info("\nWriting Final TF-TG GAT training data to parquet")
        tf_tg_balanced_with_pkn.to_parquet(gat_training_file, engine="pyarrow", compression="snappy")
        logging.info(f"  - Wrote TF-TG features to {gat_training_file}")
    
    # ----- CHROMOSOME-SPECIFIC PREPROCESSING -----
    chrom_list = sorted(list(chrom_set))
    tf_names = list(total_tf_set)
    
    if PROCESS_CHROMOSOME_SPECIFIC_DATA:
        logging.info(f"  - Number of chromosomes: {len(chrom_list)}: {chrom_list}")
        for chrom_id in chrom_list:
            logging.info(f"\n----- Preparing MultiomicTransformer data for {chrom_id} -----")
            make_chrom_gene_tss_df(GENE_TSS_FILE, chrom_id, GENOME_DIR)
            
            SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR = SAMPLE_DATA_CACHE_DIR / chrom_id
        
            single_cell_dir = SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / "single_cell"
            
            # Chromosome-specific cache files
            atac_tensor_path: Path =            SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"atac_window_tensor_all_{chrom_id}.pt"
            tg_tensor_path: Path =              SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_tensor_all_{chrom_id}.pt"
            sample_tg_name_file: Path =         SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_names_{chrom_id}.json"
            genome_window_file: Path =          SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"{chrom_id}_windows_{WINDOW_SIZE // 1000}kb.bed"
            sample_window_map_file: Path =      SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"window_map_{chrom_id}.json"
            sample_scaler_file: Path =          SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_scaler_{chrom_id}.save"
            peak_to_tss_dist_path: Path =       SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"genes_near_peaks_{chrom_id}.parquet"
            dist_bias_file: Path =              SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"dist_bias_{chrom_id}.pt"
            tg_id_file: Path =                  SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_ids_{chrom_id}.pt"
            manifest_file: Path =               SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"manifest_{chrom_id}.json"
            motif_mask_file: Path =             SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"motif_mask_{chrom_id}.pt"
            chrom_sliding_window_file: Path =   SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"sliding_window_{chrom_id}.tsv"
            chrom_peak_bed_file: Path =         SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"peak_tmp_{chrom_id}.bed"
            tss_bed_file: Path =                SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tss_tmp_{chrom_id}.bed"

            os.makedirs(COMMON_DATA, exist_ok=True)
            os.makedirs(SAMPLE_DATA_CACHE_DIR, exist_ok=True)
            os.makedirs(SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR, exist_ok=True)
            os.makedirs(single_cell_dir, exist_ok=True)
            
            logging.info(f"\nPreparing data for {DATASET_NAME} {chrom_id}")
            
            # Create or load the gene TSS information for the chromosome
            if not os.path.isfile(os.path.join(GENOME_DIR, f"{chrom_id}_gene_tss.bed")):
                gene_tss_df = make_chrom_gene_tss_df(
                    gene_tss_file=GENE_TSS_FILE,
                    chrom_id=chrom_id,
                    genome_dir=GENOME_DIR
                )
            else:
                logging.info(f"  - Loading existing gene TSS file for {chrom_id}")
                gene_tss_df = pd.read_csv(os.path.join(GENOME_DIR, f"{chrom_id}_gene_tss.bed"), sep="\t", header=None, usecols=[0, 1, 2, 3])
                gene_tss_df.columns = ["chrom", "start", "end", "name"]
            
            
            
            logging.info(f"  - Aggregating pseudobulk datasets for {chrom_id}")
            total_TG_pseudobulk_global, total_TG_pseudobulk_chr, total_RE_pseudobulk_chr, total_peaks_df = \
                aggregate_pseudobulk_datasets(gene_tss_df, SAMPLE_NAMES, RAW_DATA, chrom_id)
        
            tg_names = total_TG_pseudobulk_chr.index.tolist()
            
            # Genome-wide TF expression for all samples
            genome_wide_tf_expression = total_TG_pseudobulk_global.reindex(tf_names).fillna(0).values.astype("float32")
            metacell_names = total_TG_pseudobulk_global.columns.tolist()
            
            # Scale TG expression
            scaler = StandardScaler()
            TG_scaled = scaler.fit_transform(total_TG_pseudobulk_chr.values.astype("float32"))
            
            # Create genome windows
            logging.info(f"  - Creating genomic windows for {chrom_id}")
            genome_windows = create_or_load_genomic_windows(
                window_size=WINDOW_SIZE,
                chrom_id=chrom_id,
                chrom_sizes_file=CHROM_SIZES_FILE,
                genome_window_file=genome_window_file,
                force_recalculate=FORCE_RECALCULATE
            )
            num_windows = genome_windows.shape[0]
            
                # --- Calculate Peak-to-TG Distance Scores ---
            genes_near_peaks = calculate_peak_to_tg_distance_score(
                peak_bed_file=chrom_peak_bed_file,
                tss_bed_file=tss_bed_file,
                peak_gene_dist_file=peak_to_tss_dist_path,
                mesc_atac_peak_loc_df=total_peaks_df,  # peak locations DataFrame
                gene_tss_df=gene_tss_df,
                max_peak_distance= MAX_PEAK_DISTANCE,
                distance_factor_scale= DISTANCE_SCALE_FACTOR,
                force_recalculate=FORCE_RECALCULATE
            )
            
            # ----- SLIDING WINDOW TF-PEAK SCORE -----
            if not os.path.isfile(chrom_sliding_window_file):

                peaks_df = pybedtools.BedTool(peak_bed_file)

                logging.info("\nRunning sliding window scan")
                run_sliding_window_scan(
                    tf_name_list=tfs,
                    tf_info_file=str(TF_FILE),
                    motif_dir=str(MOTIF_DIR),
                    genome_fasta=str(genome_fasta_file),
                    peak_bed_file=str(chrom_peak_bed_file),
                    output_file=chrom_sliding_window_file,
                    num_cpu=num_cpu
                )
                logging.info(f"  - Wrote sliding window scores to {chrom_sliding_window_file}")
                sliding_window_df = pd.read_parquet(chrom_sliding_window_file, engine="pyarrow")
            else:
                logging.info("\nLoading existing sliding window scores")
                sliding_window_df = pd.read_parquet(chrom_sliding_window_file, engine="pyarrow")
            
            logging.info(f"  - Creating peak to window map")
            window_map = make_peak_to_window_map(
                peaks_bed=genes_near_peaks,
                windows_bed=genome_windows,
            )
            
            # Save Precomputed Tensors 
            logging.info(f"\nPrecomputing TF, TG, and ATAC tensors")
            tf_tensor_all, tg_tensor_all, atac_window_tensor_all = precompute_input_tensors(
                output_dir=str(SAMPLE_DATA_CACHE_DIR),
                genome_wide_tf_expression=genome_wide_tf_expression,
                TG_scaled=TG_scaled,
                total_RE_pseudobulk_chr=total_RE_pseudobulk_chr,
                window_map=window_map,
                windows=genome_windows,
            )
            logging.info(f"\t- Done!")
            
            # ----- Load common TF and TG vocab -----
            # Create a common TG vocabulary for the chromosome using the gene TSS
            logging.info(f"\nMatching TFs and TGs to global gene vocabulary")
            
            tg_names = [standardize_name(n) for n in total_TG_pseudobulk_chr.index.tolist()]
            
            if not os.path.exists(common_tf_vocab_file):
                vocab = {n: i for i, n in enumerate(tf_names)}
                with open(common_tf_vocab_file, "w") as f:
                    json.dump(vocab, f)
                logging.info(f"Initialized TF vocab with {len(vocab)} entries")
            else:
                with open(common_tf_vocab_file) as f:
                    vocab = json.load(f)

            # Load global vocab
            with open(common_tf_vocab_file) as f: tf_vocab = json.load(f)
            with open(common_tg_vocab_file) as f: tg_vocab = json.load(f)
            
            # Match TFs and TGs to global vocab
            tf_tensor_all, tf_names_kept, tf_ids = align_to_vocab(tf_names, tf_vocab, tf_tensor_all, label="TF")
            tg_tensor_all, tg_names_kept, tg_ids = align_to_vocab(tg_names, tg_vocab, tg_tensor_all, label="TG")
            
            torch.save(torch.tensor(tf_ids, dtype=torch.long), tf_id_file)
            torch.save(torch.tensor(tg_ids, dtype=torch.long), tg_id_file)

            logging.info(f"\tMatched {len(tf_names_kept)} TFs to global vocab")
            logging.info(f"\tMatched {len(tg_names_kept)} TGs to global vocab")
            logging.info(f"\t- Done!")
            
            # Build motif mask using merged info
            motif_mask = build_motif_mask(
                tf_names=tf_names_kept,
                tg_names=tg_names_kept,
                sliding_window_df=sliding_window_df,
                genes_near_peaks=genes_near_peaks
            )
            logging.info(f"\t- Done!")

            if not tf_ids: raise ValueError("No TFs matched the common vocab.")
            if not tg_ids: raise ValueError("No TGs matched the common vocab.")
            
            # Build distance bias [num_windows x num_tg_kept] aligned to kept TGs
            logging.info(f"\nBuilding distance bias")
            dist_bias = build_distance_bias(
                genes_near_peaks=genes_near_peaks,
                window_map=window_map,
                tg_names_kept=tg_names_kept,
                num_windows=num_windows,
                dtype=torch.float32,
                mode=DIST_BIAS_MODE
            )
            logging.info(f"\t- Done!")
            
            create_single_cell_tensors(
                gene_tss_df=gene_tss_df, 
                sample_names=FINE_TUNING_DATASETS, 
                dataset_processed_data_dir=SAMPLE_PROCESSED_DATA_DIR, 
                tg_vocab=tg_vocab, 
                tf_vocab=tf_vocab, 
                chrom_id=chrom_id,
                single_cell_dir=single_cell_dir
            )
            
            # ----- Writing Output Files -----
            logging.info(f"\nWriting output files")
            # Save the Window, TF, and TG expression tensors
            torch.save(atac_window_tensor_all, atac_tensor_path)
            torch.save(tg_tensor_all, tg_tensor_path)
            torch.save(tf_tensor_all, tf_tensor_path)
            
            # Write the peak to gene TSS distance scores
            genes_near_peaks.to_parquet(peak_to_tss_dist_path, compression="snappy", engine="pyarrow")
            logging.info(f"Saved peak-to-TG distance scores to {peak_to_tss_dist_path}")

            # Save scaler for inverse-transform
            joblib.dump(scaler, sample_scaler_file)

            # Save the peak -> window map for the sample
            atomic_json_dump(window_map, sample_window_map_file)

            # Write TF and TG names and global vocab indices present in the sample
            atomic_json_dump(tf_names_kept, sample_tf_name_file)
            atomic_json_dump(tg_names_kept, sample_tg_name_file)


            # Write the distance bias and metacell names for the sample
            torch.save(dist_bias, dist_bias_file)
            logging.info(f"Saved distance bias tensor with shape {tuple(dist_bias.shape)}")

            atomic_json_dump(metacell_names, metacell_name_file)
            
            torch.save(torch.from_numpy(motif_mask), motif_mask_file)

            # Manifest of general sample info and file paths
            manifest = {
                "dataset_name": DATASET_NAME,
                "chrom": chrom_id,
                "num_windows": int(num_windows),
                "num_tfs": int(len(tf_names_kept)),
                "num_tgs": int(len(tg_names_kept)),
                "Distance tau": DISTANCE_SCALE_FACTOR,
                "Max peak-TG distance": MAX_PEAK_DISTANCE,
                "paths": {
                    "tf_tensor_all": str(tf_tensor_path),
                    "tg_tensor_all": str(tg_tensor_path),
                    "atac_window_tensor_all": str(atac_tensor_path),
                    "dist_bias": str(dist_bias_file),
                    "tf_ids": str(tf_id_file),
                    "tg_ids": str(tg_id_file),
                    "tf_names": str(sample_tf_name_file),
                    "tg_names": str(sample_tg_name_file),
                    "common_tf_vocab": str(common_tf_vocab_file),
                    "common_tg_vocab": str(common_tg_vocab_file),
                    "window_map": str(sample_window_map_file),
                    "genes_near_peaks": str(peak_to_tss_dist_path),
                    "metacell_names": str(metacell_name_file),
                    "tg_scaler": str(sample_scaler_file),
                    "motif_mask": str(motif_mask_file),
                }
            }
            with open(manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)

            logging.info("\nPreprocessing complete. Wrote per-sample/per-chrom data for MultiomicTransformerDataset.")
