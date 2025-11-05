import os
import re
import json
from sympy import dsolve
import torch
import joblib
import pandas as pd
import scanpy as sc
import logging
from pathlib import Path
import warnings
import numpy as np
import traceback
import scipy.sparse as sp
import random
from scipy.special import softmax
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Set, Optional, List, Iterable, Union, Dict
from anndata import AnnData
from tqdm import tqdm
import pybedtools
import argparse
import shutil, tempfile
from functools import partial
from pybedtools import helpers as pbt_helpers
import pyarrow as pa
from pyarrow import dataset as ds, compute as pc, parquet as pq, table as pat

import sys
sys.path.append(Path(__file__).resolve().parent.parent.parent)

from multiomic_transformer.utils.standardize import standardize_name
from multiomic_transformer.utils.files import atomic_json_dump
from multiomic_transformer.utils.peaks import find_genes_near_peaks, format_peaks, set_tg_as_closest_gene_tss
from multiomic_transformer.utils.downloads import *
from multiomic_transformer.data.sliding_window import run_sliding_window_scan
from multiomic_transformer.data.build_pkn import build_organism_pkns
from multiomic_transformer.utils.gene_canonicalizer import GeneCanonicalizer
from config.settings_hpc import *

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)

# ----- Data Loading and Processing -----
def pseudo_bulk(
    rna_data: AnnData,
    atac_data: AnnData,
    neighbors_k: int = 20,
    pca_components: int = 25,
    hops: int = 1,
    self_weight: float = 1.0,
    renormalize_each_hop: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Soft 'metacells' that DO NOT reduce the number of cells.

    Builds a joint KNN graph on concatenated PCA (RNA||ATAC), forms a row-stochastic
    weight matrix W (including self loops), and returns neighborhood-averaged profiles:
        X_soft = W @ X

    Args
    ----
    neighbors_k: K for the joint KNN graph.
    pca_components: # PCs per modality before concatenation.
    hops: how many times to diffuse (W^h). Larger = smoother.
    self_weight: diagonal weight (>=0) added before row-normalization.
    renormalize_each_hop: if True, row-normalize after every hop to keep rows summing to 1.

    Returns
    -------
    soft_rna_df:  genes × cells (same cells as input)
    soft_atac_df: peaks × cells (same cells as input)
    """
    # --- copy and align cells across modalities (same as in your function) ---
    rna = rna_data.copy()
    atac = atac_data.copy()

    common = rna.obs_names.intersection(atac.obs_names)
    if len(common) == 0:
        raise ValueError("No overlapping cell barcodes between RNA and ATAC.")
    rna = rna[common].copy()
    atac = atac[common].copy()
    atac = atac[rna.obs_names].copy()
    assert (rna.obs_names == atac.obs_names).all()

    # --- ensure PCA in both modalities (caps to valid range) ---
    def _ensure_pca(adata: AnnData, n_comps: int) -> None:
        max_comps = int(min(adata.n_obs, adata.n_vars))
        use_comps = max(1, min(n_comps, max_comps))
        if "X_pca" not in adata.obsm_keys() or adata.obsm.get("X_pca", np.empty((0,0))).shape[1] < use_comps:
            sc.pp.scale(adata, max_value=10, zero_center=True)
            sc.tl.pca(adata, n_comps=use_comps, svd_solver="arpack")

    _ensure_pca(rna, pca_components)
    _ensure_pca(atac, pca_components)

    # --- joint embedding, neighbors on it ---
    combined_pca = np.concatenate((rna.obsm["X_pca"], atac.obsm["X_pca"]), axis=1)
    # tiny placeholder X to satisfy AnnData; graph lives in .obsp
    joint = AnnData(X=sp.csr_matrix((rna.n_obs, 0)), obs=rna.obs.copy())
    joint.obsm["X_combined"] = combined_pca
    sc.pp.neighbors(joint, n_neighbors=neighbors_k, use_rep="X_combined")

    # base connectivities (symmetric KNN graph; values ~[0,1])
    W = joint.obsp["connectivities"].tocsr().astype(np.float32)

    # add self-loops
    if self_weight > 0:
        W = W + sp.diags(np.full(W.shape[0], self_weight, dtype=np.float32), format="csr")

    # row-normalize to make W row-stochastic
    def _row_norm(mat: sp.csr_matrix) -> sp.csr_matrix:
        row_sum = np.asarray(mat.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        inv = sp.diags(1.0 / row_sum, dtype=np.float32)
        return inv @ mat

    W = _row_norm(W)

    # multi-hop diffusion: W <- W^h (staying row-stochastic)
    if hops > 1:
        W_h = W
        for _ in range(1, int(hops)):
            W_h = W_h @ W
            if renormalize_each_hop:
                W_h = _row_norm(W_h)
        W = W_h
    # one last normalization for safety (keeps rows summing to 1 exactly)
    W = _row_norm(W)

    # --- apply smoothing per modality ---
    def _as_csr(X, dtype=np.float32):
        """Return X as CSR sparse matrix (no-copy when already CSR)."""
        if sp.issparse(X):
            return X.tocsr().astype(dtype, copy=False)
        # X is dense (numpy ndarray / matrix)
        return sp.csr_matrix(np.asarray(X, dtype=dtype, order="C"))
    
    # X matrices are cells × features (CSR recommended)
    X_rna  = _as_csr(rna.layers["log1p"])     # cells × genes
    X_atac = _as_csr(atac.layers["log1p"])    # cells × peaks

    X_rna_soft = W @ X_rna      # cells × genes
    X_atac_soft = W @ X_atac    # cells × peaks

    # return DataFrames as features × cells (to mirror your pseudo_bulk shapes)
    pseudo_bulk_rna_df = pd.DataFrame(
        X_rna_soft.T.toarray(),
        index=rna.var_names,
        columns=rna.obs_names,
    )
    pseudo_bulk_atac_df = pd.DataFrame(
        X_atac_soft.T.toarray(),
        index=atac.var_names,
        columns=atac.obs_names,
    )

    return pseudo_bulk_rna_df, pseudo_bulk_atac_df

def process_10x_to_csv(raw_10x_rna_data_dir, raw_atac_peak_file, rna_outfile_path, atac_outfile_path, sample_name):
    
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

def _configured_path(
    sample_input_dir: Path,
    key: str,
    default_name: str,
    sample_name: str,
) -> Path:
    """
    Look up `key` among names imported via `from config.settings import *`.
    Falls back to `default_name`. Relative paths are resolved under sample_input_dir.
    Supports {sample} templating. Accepts str or Path in config.
    """
    # settings injected by the star-import live in this module's globals
    value = globals().get(key, None)

    # accept Path or str from config; fall back to default
    raw = value if value is not None else default_name
    if isinstance(raw, Path):
        raw = str(raw)

    # allow e.g. "outputs/{sample}_RNA.parquet"
    if isinstance(raw, str):
        try:
            raw = raw.format(sample=sample_name)
        except Exception:
            pass

    p = Path(raw)
    return p if p.is_absolute() else (sample_input_dir / p)

def process_or_load_rna_atac_data(
    sample_input_dir: Union[str, Path],
    force_recalculate: bool = False,
    raw_10x_rna_data_dir: Union[str, Path, None] = None,
    raw_atac_peak_file: Union[str, Path, None] = None,
    *,
    sample_name: Optional[str] = None,
    neighbors_k: Optional[int] = None,
    pca_components: Optional[int] = None,
    hops: Optional[int] = None,
    self_weight: Optional[float] = None,
    load: Optional[bool] = True
) -> Tuple[
        Optional[pd.DataFrame], 
        Optional[pd.DataFrame], 
        Optional[pd.DataFrame], 
        Optional[pd.DataFrame]
        ]:
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
    force_recalculate : bool, default False
        Force recomputation even if processed/parquet/h5ad exists.
    raw_10x_rna_data_dir : str | Path | None
        Path to the 10x RNA directory if CSVs need to be generated.
    raw_atac_peak_file : str | Path | None
        Path to a peak matrix (or peak list) used when building ATAC CSVs from 10x.
    sample_name : str | None, keyword-only
        Pretty name for logging; defaults to the directory name if None.
    neighbors_k : int | None, keyword-only
        Number of nearest neighbors per cell for the joint KNN graph used by
        pseudobulk grouping and/or soft metacell smoothing. Larger values yield
        stronger smoothing / broader neighborhoods. If None, defaults to NEIGHBORS_K.

    pca_components : int | None, keyword-only
        Number of principal components per modality (RNA and ATAC) before
        concatenation for the joint neighbor graph. If None, defaults to
        PCA_COMPONENTS. Capped internally by min(n_cells, n_features).

    hops : int | None, keyword-only
        Diffusion depth for soft metacells (applies W^h over the KNN graph).
        Higher values spread information to neighbors-of-neighbors. Typical 1–3.
        If None, defaults to SOFT_HOPS.

    self_weight : float | None, keyword-only
        Additional diagonal weight added before row-normalization of the graph
        (self-loops). Higher values keep a cell closer to its original profile.
        If None, defaults to SOFT_SELF_WEIGHT.
    load : bool | None
        Chooses whether to return the processed datasets if they are found. Skipped for
        parallelized pre-processing to avoid reading every processed data file in at once.
        
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
    - Will use existing artifacts when available unless `force_recalculate=True`.
    """
    # ---- resolve paths and names ----
    sample_input_dir = Path(sample_input_dir)
    sample_name = sample_name or sample_input_dir.name
    raw_10x_rna_data_dir = Path(raw_10x_rna_data_dir) if raw_10x_rna_data_dir is not None else None
    raw_atac_peak_file = Path(raw_atac_peak_file) if raw_atac_peak_file is not None else None

    processed_rna_file  = _configured_path(sample_input_dir, "PROCESSED_RNA_FILENAME",  "scRNA_seq_processed.parquet", sample_name)
    processed_atac_file = _configured_path(sample_input_dir, "PROCESSED_ATAC_FILENAME", "scATAC_seq_processed.parquet", sample_name)

    raw_rna_file  = _configured_path(sample_input_dir, "RAW_RNA_FILENAME",  "scRNA_seq_raw.parquet", sample_name)
    raw_atac_file = _configured_path(sample_input_dir, "RAW_ATAC_FILENAME", "scATAC_seq_raw.parquet", sample_name)

    adata_rna_file  = _configured_path(sample_input_dir, "ADATA_RNA_FILENAME",  "adata_RNA.h5ad", sample_name)
    adata_atac_file = _configured_path(sample_input_dir, "ADATA_ATAC_FILENAME", "adata_ATAC.h5ad", sample_name)

    pseudobulk_TG_file = _configured_path(sample_input_dir, "PSEUDOBULK_TG_FILENAME", "TG_pseudobulk.parquet", sample_name)
    pseudobulk_RE_file = _configured_path(sample_input_dir, "PSEUDOBULK_RE_FILENAME", "RE_pseudobulk.parquet", sample_name)

    neighbors_k = neighbors_k if neighbors_k is not None else NEIGHBORS_K
    
    # If load is set to False, check that all of the preprocessed files exist
    if load == False:
        files_missing = False
        
        files = [
            processed_rna_file,
            processed_atac_file,
            adata_rna_file,
            adata_atac_file,
            pseudobulk_TG_file,
            pseudobulk_RE_file
            ]
        
        for file in files:
            if not file.is_file():
                files_missing = True
    
        if not files_missing:
            logging.info("All preprocessed files exist")
            return None, None, None, None
        

    logging.info(f"\n----- Loading or Processing RNA and ATAC data for {sample_name} -----")
    logging.info("Searching for processed RNA/ATAC parquet files:")

    # helpers
    def _adata_to_dense_df(adata: AnnData) -> pd.DataFrame:
        X = adata.layers["log1p"]
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
    
    def _normalize_barcodes(index_like) -> pd.Index:
        """
        Normalize 10x barcodes so RNA/ATAC match.
        - Strip sample prefixes like 'S1:' or 'rna-' at start
        - Strip modality tags at end: _RNA/_GEX/_ATAC, #GEX/#ATAC
        - Strip trailing dash/dot + digits: -1, -2, .1, .2
        - Uppercase to avoid case mismatches
        """
        ix = pd.Index(index_like).astype(str)

        # strip leading sample prefixes "S1:", "RNA:", "rna-", etc.
        ix = ix.str.replace(r'^[A-Za-z0-9]+[:\-]','', regex=True)  # keep if too aggressive, comment out

        # strip modality tags at end
        ix = ix.str.replace(r'(?:_RNA|_GEX|_ATAC|#GEX|#ATAC)$', '', regex=True, case=False)

        # strip trailing -digits or .digits (10x suffix)
        ix = ix.str.replace(r'[-\.]\d+$', '', regex=True)

        # unify case
        ix = ix.str.upper()

        return ix

    def harmonize_and_intersect(ad_rna: AnnData, ad_atac: AnnData, *, verbose: bool=True):
        rna_norm = _normalize_barcodes(ad_rna.obs_names)
        atac_norm = _normalize_barcodes(ad_atac.obs_names)

        if verbose:
            print(f"[SYNC] RNA n={ad_rna.n_obs}, ATAC n={ad_atac.n_obs}")
            print("[SYNC] Examples:")
            print("  RNA:", list(rna_norm[:5]))
            print("  ATAC:", list(atac_norm[:5]))

        # build maps from normalized→original to subset accurately
        r_map = pd.Series(ad_rna.obs_names, index=rna_norm, dtype="object")
        a_map = pd.Series(ad_atac.obs_names, index=atac_norm, dtype="object")

        common = r_map.index.intersection(a_map.index)
        if verbose:
            print(f"[SYNC] common={len(common)}")

        if len(common) == 0:
            # Help the user debug: show top 10 of each set difference
            r_only = r_map.index.difference(a_map.index)[:10]
            a_only = a_map.index.difference(r_map.index)[:10]
            raise RuntimeError(
                "No overlapping barcodes after normalization.\n"
                f"RNA-only examples: {list(r_only)}\n"
                f"ATAC-only examples: {list(a_only)}\n"
                "Adjust normalization rules to your dataset’s naming."
            )

        # subset both to common, in the same order
        ad_rna2  = ad_rna[ r_map.loc[common].values, : ].copy()
        ad_atac2 = ad_atac[ a_map.loc[common].values, : ].copy()

        # set synchronized obs_names to the normalized common IDs
        ad_rna2.obs_names  = common
        ad_atac2.obs_names = common
        return ad_rna2, ad_atac2
    
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

    # placeholders
    processed_rna_df: Optional[pd.DataFrame] = None
    processed_atac_df: Optional[pd.DataFrame] = None
    TG_pseudobulk_df: Optional[pd.DataFrame] = None
    RE_pseudobulk_df: Optional[pd.DataFrame] = None

    # =========================
    # 1) Try processed parquet
    # =========================
    if not force_recalculate and processed_rna_file.is_file() and processed_atac_file.is_file():
        logging.info(f"[{sample_name}] Pre-processed data files found, loading...")
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

        if ad_rna is None or ad_atac is None or force_recalculate:
            logging.info("    - Filtered AnnData missing or ignored – will look for raw CSVs.")

            # ===================
            # 3) Try raw CSV pair
            # ===================
            if not raw_rna_file.is_file() or not raw_atac_file.is_file():
                logging.info("      - Raw parquet files missing – will try to create from 10x inputs.")
                logging.info(raw_rna_file)
                logging.info(raw_atac_file)

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
                process_10x_to_csv(raw_10x_rna_data_dir, raw_atac_peak_file, raw_rna_file, raw_atac_file, sample_name)

            # Load raw data parquet files and convert to AnnData
            logging.info("Reading raw CSVs into AnnData")
            rna_df = pd.read_parquet(raw_rna_file, engine="pyarrow")
            atac_df = pd.read_parquet(raw_atac_file, engine="pyarrow")
            
            def _norm_names(names):
                out = []
                for s in map(str, names):
                    s = s.strip().lower()
                    s = re.sub(r"\s+", "", s)
                    s = re.sub(r"[-\.]\d+$", "", s)
                    out.append(s)
                return set(out)

            def _obs_overlap(ad1, ad2) -> int:
                return len(_norm_names(ad1.obs_names) & _norm_names(ad2.obs_names))

            try:
                # Try as-is (no transpose)
                ad_rna = AnnData(rna_df)
                ad_atac = AnnData(atac_df)
                
                if _obs_overlap(ad_rna, ad_atac) == 0:
                    raise RuntimeError("No overlapping barcodes after initial orientation.")
                ad_rna, ad_atac = harmonize_and_intersect(ad_rna, ad_atac, verbose=True)

            except RuntimeError as e1:
                logging.warning("No overlapping barcodes. Testing transposed matrices...")
                # Retry with transpose
                ad_rna = AnnData(rna_df.T)
                ad_atac = AnnData(atac_df.T)
                if _obs_overlap(ad_rna, ad_atac) == 0:
                    raise RuntimeError(
                        "Failed to align cell barcodes in both orientations. "
                        "If your DataFrames are genes×cells, use non-transposed; "
                        "if cells×genes, use transposed. Adjust normalization if needed."
                    ) from e1
                ad_rna, ad_atac = harmonize_and_intersect(ad_rna, ad_atac, verbose=True)
            
            # Save the expression to a layer so that scaling for PCA does not apply to the gene expression
            # (we want to scale all of the samples together later so they are in the same z-space)
            ad_rna.layers["log1p"]  = ad_rna.X.copy()
            ad_atac.layers["log1p"] = ad_atac.X.copy()

            # QC/filter
            logging.info("Running filter_and_qc on RNA/ATAC AnnData")
            ad_rna, ad_atac = filter_and_qc(ad_rna, ad_atac)

            # Persist
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

    if need_TG or need_RE or force_recalculate:
        logging.info("Pseudobulk files missing or ignored – computing pseudobulk now.")

        ad_rna = _load_or_none(adata_rna_file, sc.read_h5ad)
        ad_atac = _load_or_none(adata_atac_file, sc.read_h5ad)
        
        assert (ad_rna != None) and (ad_atac != None) \
            f"AnnData RNA or ATAC must exist!\nad_rna = {ad_rna}\nad_atac = {ad_atac}"

        TG_pseudobulk_df, RE_pseudobulk_df = pseudo_bulk(
            rna_data=ad_rna,
            atac_data=ad_atac,
            neighbors_k=neighbors_k,
            pca_components=pca_components,
            hops=hops,
            self_weight=self_weight,
            renormalize_each_hop=True
        )

        # Post-processing as in your original
        TG_pseudobulk_df = TG_pseudobulk_df.fillna(0)
        RE_pseudobulk_df = RE_pseudobulk_df.fillna(0)
        RE_pseudobulk_df[RE_pseudobulk_df > 100] = 100
        
        TG_pseudobulk_df = _standardize_symbols_index(TG_pseudobulk_df, strip_version_suffix=True, uppercase=True, deduplicate="sum")

        TG_pseudobulk_df.to_parquet(pseudobulk_TG_file, engine="pyarrow", compression="snappy")
        RE_pseudobulk_df.to_parquet(pseudobulk_RE_file, engine="pyarrow", compression="snappy")
    else:
        logging.info("Pseudobulk TSVs found, loading from disk.")
        logging.info(f"  - Pseudobulk TG Path: {pseudobulk_TG_file}")
        logging.info(f"  - Pseudobulk RE Path: {pseudobulk_RE_file}")
        TG_pseudobulk_df = pd.read_parquet(pseudobulk_TG_file, engine="pyarrow")
        RE_pseudobulk_df = pd.read_parquet(pseudobulk_RE_file, engine="pyarrow")
        
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
    
    common_barcodes = adata_RNA.obs_names.isin(adata_ATAC.obs_names)
    assert len(common_barcodes) > 10, \
        f"No common barcodes. \n  - RNA: {adata_RNA.obs_names[:2]}\n  - ATAC: {adata_ATAC.obs_names[:2]}"
    
    # Synchronize barcodes
    adata_RNA.obs['barcode'] = adata_RNA.obs_names
    adata_ATAC.obs['barcode'] = adata_ATAC.obs_names

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
    adata_RNA.layers["log1p"] = adata_RNA.X.copy()
    sc.pp.scale(adata_RNA, max_value=10)
    adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable]
    sc.tl.pca(adata_RNA, n_comps=25, svd_solver="arpack")

    # Preprocess ATAC
    sc.pp.log1p(adata_ATAC)
    sc.pp.highly_variable_genes(adata_ATAC, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_ATAC.layers["log1p"] = adata_ATAC.X.copy()
    adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable]
    sc.pp.scale(adata_ATAC, max_value=10, zero_center=True)
    sc.tl.pca(adata_ATAC, n_comps=25, svd_solver="arpack")
    
    # After filtering to common barcodes
    common_barcodes = adata_RNA.obs_names.intersection(adata_ATAC.obs_names)
    
    adata_RNA = adata_RNA[common_barcodes].copy()
    adata_ATAC = adata_ATAC[common_barcodes].copy()
    
    return adata_RNA, adata_ATAC

def _canon(x: str) -> str:
    # strip version suffix and uppercase
    s = str(x).strip()
    s = re.sub(r"\.\d+$", "", s)
    return s.upper()


# --- load existing lists (if any) and union ---
def _read_list(path: Path, col: str) -> list[str]:
    if path.is_file():
        df = pd.read_csv(path)
        if col not in df.columns and df.shape[1] == 1:
            # tolerate unnamed single column
            return sorted({_canon(v) for v in df.iloc[:, 0].astype(str)})
        return sorted({_canon(v) for v in df[col].dropna().astype(str)})
    return []

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
        gene_tss_bed
        .to_dataframe(header=None, usecols=[0, 1, 2, 3])
        .rename(columns={0: "chrom", 1: "start", 2: "end", 3: "name"})
        .sort_values(by="start", ascending=True)
    )
    # Normalize symbols to match downstream vocab
    gene_tss_df["name"] = gene_tss_df["name"].astype(str).map(standardize_name)
    # Drop any duplicated gene symbols, keep first TSS row
    gene_tss_df = gene_tss_df.drop_duplicates(subset=["name"], keep="first")

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
    force_recalculate=False,
    filter_to_nearest_gene=False,
    promoter_bp=None
) -> pd.DataFrame:
    """
    Compute peak-to-gene distance features (BEDTools-based), ensuring BED compliance.
    """
    # Validate and convert peaks to BED format
    required_cols = {"chrom", "start", "end", "peak_id"}
    if not required_cols.issubset(mesc_atac_peak_loc_df.columns):
        logging.warning("Converting peak index to BED format (chr/start/end parsing)")
        mesc_atac_peak_loc_df = build_peak_locs_from_index(mesc_atac_peak_loc_df.index)

    # print("\nmesc_atac_peak_loc_df")
    # print(mesc_atac_peak_loc_df.head())
    
    # Ensure numeric types
    mesc_atac_peak_loc_df["start"] = mesc_atac_peak_loc_df["start"].astype(int)
    mesc_atac_peak_loc_df["end"] = mesc_atac_peak_loc_df["end"].astype(int)

    # Ensure proper columns for gene_tss_df
    if not {"chrom", "start", "end", "name"}.issubset(gene_tss_df.columns):
        gene_tss_df = gene_tss_df.rename(columns={"chromosome_name": "chrom", "gene_start": "start", "gene_end": "end"})

    # Step 1: Write valid BED files if missing
    if not os.path.isfile(peak_bed_file) or not os.path.isfile(tss_bed_file) or force_recalculate:
        pybedtools.BedTool.from_dataframe(mesc_atac_peak_loc_df[["chrom", "start", "end", "peak_id"]]).saveas(peak_bed_file)
        pybedtools.BedTool.from_dataframe(gene_tss_df[["chrom", "start", "end", "name"]]).saveas(tss_bed_file)

    # Step 2: Run BEDTools overlap
    peak_bed = pybedtools.BedTool(peak_bed_file)
    tss_bed = pybedtools.BedTool(tss_bed_file)

    genes_near_peaks = find_genes_near_peaks(peak_bed, tss_bed, tss_distance_cutoff=max_peak_distance)
    
    genes_near_peaks = genes_near_peaks.rename(columns={"gene_id": "target_id"})
    genes_near_peaks["target_id"] = genes_near_peaks["target_id"].apply(standardize_name)
    
    if "TSS_dist" not in genes_near_peaks.columns:
        raise ValueError("Expected column 'TSS_dist' missing from find_genes_near_peaks output.")
    
    # Compute the distance score
    genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / float(distance_factor_scale))
    
    # Drop rows where the peak is too far from the gene
    genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= max_peak_distance]
    
    if promoter_bp is not None:
        # Subset to keep genes that are near gene promoters
        genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= int(promoter_bp)]
        
    if filter_to_nearest_gene:
        # Filter to use the gene closest to the peak
        genes_near_peaks = (genes_near_peaks.sort_values(["TSS_dist_score","TSS_dist","target_id"],
                             ascending=[False, True, True], kind="mergesort")
                .drop_duplicates(subset=["peak_id"], keep="first"))
        
    # Save and return the dataframe
    genes_near_peaks.to_parquet(peak_gene_dist_file, compression="snappy", engine="pyarrow")
    return genes_near_peaks

def _softmax_1d_stable(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    z = (x / float(tau)).astype(float)
    z -= z.max()              # numerical stability
    p = np.exp(z)
    s = p.sum()
    return p / s if s > 0 else np.full_like(p, 1.0 / len(p))

def _process_single_tf(tf, tf_df, peak_to_gene_dist_df, *, temperature: float = 1.0):
    # tf_df contains only one TF
    scores = tf_df["sliding_window_score"].to_numpy()
    tf_df = tf_df.copy()
    tf_df["tf_peak_prob"] = _softmax_1d_stable(scores, tau=temperature)

    merged = pd.merge(
        tf_df[["peak_id", "tf_peak_prob"]],
        peak_to_gene_dist_df[["peak_id", "target_id", "TSS_dist_score"]],
        on="peak_id",
        how="inner",
    )
    merged["tf_tg_contrib"] = merged["tf_peak_prob"] * merged["TSS_dist_score"]

    out = (
        merged.groupby("target_id", as_index=False)
            .agg(reg_potential=("tf_tg_contrib", "sum"),
                motif_density=("peak_id", "nunique"))
            .rename(columns={"target_id": "TG"})
    )
    out["TF"] = tf
    return out

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

    # Subset to only peaks in both the peak-TG distance dataset
    relevant_peaks = set(sliding_window_df["peak_id"].unique())
    peak_to_gene_dist_df = peak_to_gene_dist_df[peak_to_gene_dist_df["peak_id"].isin(relevant_peaks)]
    
    # Clean the column names and drop na
    sliding_window_df = sliding_window_df.dropna(subset=["TF", "peak_id", "sliding_window_score"])
    sliding_window_df["TF"] = sliding_window_df["TF"].apply(standardize_name)
    peak_to_gene_dist_df["target_id"] = peak_to_gene_dist_df["target_id"].apply(standardize_name)

    # Group the sliding window scores by TF
    tf_groups = {tf: df for tf, df in sliding_window_df.groupby("TF", sort=False)}

    logging.info(f"Processing {len(tf_groups)} TFs using {num_cpu} CPUs")
    results = []
    
    with ProcessPoolExecutor(max_workers=num_cpu) as ex:
        futures = {
            ex.submit(_process_single_tf, tf, df, peak_to_gene_dist_df): tf
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
    logging.debug(tf_tg_df.head())

    logging.info(f"    - Number of unique TFs: {tf_tg_df['TF'].nunique()}")

    logging.info("\n  - Merging mean min-max normalized TF expression")
    tf_tg_df = pd.merge(
        tf_tg_df,
        mean_norm_tf_expr,
        how="left",
        on=["TF"]
    ).dropna(subset="mean_tf_expr")
    logging.debug(tf_tg_df.head())
    logging.info(f"    - Number of unique TFs: {tf_tg_df['TF'].nunique()}")

    logging.info("\n- Merging mean min-max normalized TG expression")

    tf_tg_df = pd.merge(
        tf_tg_df,
        mean_norm_tg_expr,
        how="left",
        on=["TG"]
    ).dropna(subset="mean_tg_expr")
    logging.debug(tf_tg_df.head())
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
    gene_tss_df["name"] = gene_tss_df["name"].astype(str).map(standardize_name)
    gene_tss_df = gene_tss_df.drop_duplicates(subset=["name"], keep="first")
    return gene_tss_df

def merge_tf_tg_data_with_pkn(
    df: pd.DataFrame, 
    string_csv_file: Union[str, Path], 
    trrust_csv_file: Union[str, Path], 
    kegg_csv_file: Union[str, Path],
    seed: int = 42,
    add_pkn_scores: bool = True,
    pkn_metadata_cols: Optional[dict[str, list[str]]] = None,
    *,
    normalize_tf_tg_symbols: bool = True,
    strip_version_suffix: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build union of STRING/TRRUST/KEGG edges, split positives/negatives (UNDIRECTED),
    label all edges WITHOUT any balancing, and annotate per-source provenance flags.
    Returns:
        (tf_tg_labeled_with_pkn, tf_tg_labeled)

    tf_tg_labeled_with_pkn includes in_STRING/in_TRRUST/in_KEGG and n_sources (+ optional metadata).
    tf_tg_labeled is the same rows without the metadata columns.

    Normalization knobs:
    - normalize_tf_tg_symbols: uppercase TF/TG in the candidate DF.
    - strip_version_suffix: remove trailing '.<digits>' from TF/TG (e.g., 'TP53.1' -> 'TP53').
    """
    import os, re
    from typing import Set, Tuple

    # -------- helpers --------
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
        prefix: Optional[str] = None,  # default: no extra prefixing
    ) -> pd.DataFrame:
        cols = ["TF", "TG"] + [c for c in keep_cols if c in meta_df.columns]
        if len(cols) <= 2:
            return base

        direct = meta_df[cols].drop_duplicates(["TF", "TG"]).copy()
        reversed_df = meta_df.rename(columns={"TF": "TG", "TG": "TF"})[cols].drop_duplicates(["TF", "TG"]).copy()

        def _pref(df_in: pd.DataFrame) -> pd.DataFrame:
            if not prefix:
                return df_in.copy()
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
    df["TF"] = gc.canonicalize_series(df["TF"])
    df["TG"] = gc.canonicalize_series(df["TG"])

    for pkn_df in (string_df, trrust_df, kegg_df):
        pkn_df["TF"] = gc.canonicalize_series(pkn_df["TF"])
        pkn_df["TG"] = gc.canonicalize_series(pkn_df["TG"])

    try:
        sample_syms = list(pd.unique(df[["TF", "TG"]].values.ravel()))[:5]
        logging.info(f"\tExample TF-TG data: {sample_syms}")
    except Exception:
        pass

    # ---------------- split ----------------
    logging.info("  - Splitting positives/negatives by PKN union")
    in_pkn_df, not_in_pkn_df = _split_by_pkn_union(df, pkn_union_u)
    logging.info("  - Splitting results:")
    logging.info(f"\t  Edges in TF-TG data: {df.shape[0]:,}")
    logging.info(f"\t  Edges in PKN union (undirected): {len(pkn_union_u):,}")
    logging.info(f"\t  Unique TFs in PKN: {in_pkn_df['TF'].nunique():,}")
    logging.info(f"\t  Unique TGs in PKN: {in_pkn_df['TG'].nunique():,}")
    logging.info(f"\t  Unique TFs not in PKN: {not_in_pkn_df['TF'].nunique():,}")
    logging.info(f"\t  Unique TGs not in PKN: {not_in_pkn_df['TG'].nunique():,}")
    logging.info(f"\t  Edges not in PKN: {not_in_pkn_df.shape[0]:,}")
    logging.info(f"\t  Edges in PKN: {in_pkn_df.shape[0]:,}")
    logging.info(f"\t  Fraction of TF-TG edges in PKN: {in_pkn_df.shape[0] / max(1, df.shape[0]):.2f}")

    if in_pkn_df.empty:
        raise ValueError("No TF–TG positives in PKN union after normalization.")

    # ---------------- label (NO balancing) ----------------
    in_pkn_df  = in_pkn_df.copy()
    not_in_pkn_df = not_in_pkn_df.copy()
    in_pkn_df["label"] = 1
    not_in_pkn_df["label"] = 0

    # Union all labeled edges (unbalanced)
    tf_tg_labeled = pd.concat([in_pkn_df, not_in_pkn_df], ignore_index=True)
    # Optional: shuffle for downstream convenience (no effect on stats)
    tf_tg_labeled = tf_tg_labeled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Keep a copy before adding flags/metadata
    tf_tg_labeled_no_pkn_scores = tf_tg_labeled.copy()

    # ---------------- flags (undirected) ----------------
    tf_tg_labeled_with_pkn_scores = tf_tg_labeled.copy()
    _flag_undirected(tf_tg_labeled_with_pkn_scores, string_set, "in_STRING")
    _flag_undirected(tf_tg_labeled_with_pkn_scores, trrust_set, "in_TRRUST")
    _flag_undirected(tf_tg_labeled_with_pkn_scores,   kegg_set, "in_KEGG")
    tf_tg_labeled_with_pkn_scores["n_sources"] = (
        tf_tg_labeled_with_pkn_scores[["in_STRING","in_TRRUST","in_KEGG"]].sum(axis=1)
    )

    # ---------------- metadata (bi-directional merge with fallback) ----------------
    if add_pkn_scores:
        tf_tg_labeled_with_pkn_scores = _safe_merge_meta_bi(
            tf_tg_labeled_with_pkn_scores, string_df, pkn_metadata_cols.get("STRING", [])
        )
        tf_tg_labeled_with_pkn_scores = _safe_merge_meta_bi(
            tf_tg_labeled_with_pkn_scores, trrust_df, pkn_metadata_cols.get("TRRUST", [])
        )
        tf_tg_labeled_with_pkn_scores = _safe_merge_meta_bi(
            tf_tg_labeled_with_pkn_scores, kegg_df,   pkn_metadata_cols.get("KEGG", [])
        )
    else:
        logging.warning("Skipping PKN metadata merge (add_pkn_scores=False). tf_tg_labeled_with_pkn_scores == tf_tg_labeled_no_pkn_scores")

    return tf_tg_labeled_with_pkn_scores, tf_tg_labeled_no_pkn_scores

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
    gene_tss_df["name"] = gc.canonicalize_series(gene_tss_df["name"])
    names = sorted(set(gene_tss_df["name"]))  # unique + stable order
    logging.info(f"Writing global TG vocab with {len(names)} genes")

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

    # Ensure gene_tss_df names are normalized
    gene_tss_df = gene_tss_df.copy()
    gene_tss_df["name"] = gene_tss_df["name"].astype(str).map(standardize_name)
    
    # --- set chromosome-specific TG list ---
    chrom_tg_names = set(gene_tss_df["name"].unique())

    for sample_name in sample_names:        
        sample_processed_data_dir = dataset_processed_data_dir / sample_name

        tg_sc_file = sample_processed_data_dir / "TG_singlecell.tsv"
        re_sc_file = sample_processed_data_dir / "RE_singlecell.tsv"

        if not (tg_sc_file.exists() and re_sc_file.exists()):
            logging.debug(f"Skipping {sample_name}: missing TG/RE single-cell files")
            continue

        TG_sc = pd.read_csv(tg_sc_file, sep="\t", index_col=0)
        TG_sc.index = TG_sc.index.astype(str).map(standardize_name)
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

def aggregate_pseudobulk_datasets(sample_names: list[str], dataset_processed_data_dir: Path, chroms: list[str], gc: GeneCanonicalizer):
    
    # ----- Combine Pseudobulk Data into a Training Dataset -----
    def _canon_index_sum(df: pd.DataFrame, gc) -> pd.DataFrame:
        """Canonicalize df.index with GeneCanonicalizer and sum duplicate rows."""
        if df.empty:
            return df
        mapped = gc.canonicalize_series(pd.Series(df.index, index=df.index))
        out = df.copy()
        out.index = mapped.values
        out = out[out.index != ""]                 # drop unmapped
        if not out.index.is_unique:               # aggregate duplicates
            out = out.groupby(level=0).sum()
        return out

    def _canon_series_same_len(s: pd.Series, gc) -> pd.Series:
        """Canonicalize to same length; replace non-mapped with '' (then caller filters rows)."""
        cs = gc.canonicalize_series(s.astype(str))
        cs = cs.fillna("")
        return cs
    
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

    logging.info("\nLoading processed pseudobulk datasets:")
    logging.info(f"  - Sample names: {sample_names}")
    logging.info(f"  - Looking for processed samples in {dataset_processed_data_dir}")
    
    # Combine the TG pseudobulk for all samples into one dataframe
    per_sample_TG = {}
    for sample_name in sample_names:
        sdir = dataset_processed_data_dir / sample_name
        tg_path = sdir / "TG_pseudobulk.parquet"
        TG_pseudobulk = pd.read_parquet(tg_path, engine="pyarrow")
        TG_pseudobulk = _canon_index_sum(TG_pseudobulk, gc)
        per_sample_TG[sample_name] = TG_pseudobulk

    total_TG_pseudobulk_global = _agg_sum(list(per_sample_TG.values()))
    
    # Extract the chromosome-specific pseudobulk data from all samples
    pseudobulk_chrom_dict = {}
    for chrom_id in chroms:
        logging.info(f"Aggregating data for {chrom_id}")
        
        TG_pseudobulk_samples = []
        RE_pseudobulk_samples = []
        peaks_df_samples = []
        
        # Get a name of the genes on the chromosome from the gene TSS file
        chrom_tss_path = GENOME_DIR / f"{chrom_id}_gene_tss.bed"
        if not chrom_tss_path.is_file():
            gene_tss_chrom = make_chrom_gene_tss_df(
                gene_tss_file=GENE_TSS_FILE,
                chrom_id=chrom_id,
                genome_dir=GENOME_DIR
            )
        else:
            gene_tss_chrom = pd.read_csv(chrom_tss_path, sep="\t", header=None, usecols=[0, 1, 2, 3])
            gene_tss_chrom = gene_tss_chrom.rename(columns={0: "chrom", 1: "start", 2: "end", 3: "name"})

        gene_tss_chrom["name"] = _canon_series_same_len(gene_tss_chrom["name"], gc)
        gene_tss_chrom = gene_tss_chrom[gene_tss_chrom["name"] != ""]
        gene_tss_chrom = gene_tss_chrom.drop_duplicates(subset=["name"], keep="first")
        genes_on_chrom = gene_tss_chrom["name"].tolist()
        
        for sample_name in (pbar := tqdm(sample_names)):
            pbar.set_description(f"Processing {sample_name}")
            sample_processed_data_dir = dataset_processed_data_dir / sample_name
            
            RE_pseudobulk = pd.read_parquet(sample_processed_data_dir / "RE_pseudobulk.parquet", engine="pyarrow")
            
            TG_chr_specific = per_sample_TG[sample_name].loc[
                per_sample_TG[sample_name].index.intersection(genes_on_chrom)
            ]
            RE_chr_specific = RE_pseudobulk[RE_pseudobulk.index.str.startswith(f"{chrom_id}:")]

            peaks_df = (
                RE_chr_specific.index.to_series()
                .str.split("[:-]", expand=True)
                .rename(columns={0: "chrom", 1: "start", 2: "end"})
            )
            peaks_df["start"] = peaks_df["start"].astype(int)
            peaks_df["end"] = peaks_df["end"].astype(int)
            peaks_df["peak_id"] = RE_chr_specific.index
            
            # Add the chromosome-specific data for the sample to a list
            TG_pseudobulk_samples.append(TG_chr_specific)
            RE_pseudobulk_samples.append(RE_chr_specific)
            peaks_df_samples.append(peaks_df)
        
        # Aggregate the data from the samples for the current chromosome
        total_TG_pseudobulk_chr    = _agg_sum(TG_pseudobulk_samples)
        total_RE_pseudobulk_chr    = _agg_sum(RE_pseudobulk_samples)
        total_peaks_df             = _agg_first(peaks_df_samples)
    
        # Add the aggregated data to a dictionary by chrom_id
        pseudobulk_chrom_dict[chrom_id] = {
            "total_TG_pseudobulk_chr" : total_TG_pseudobulk_chr,
            "total_RE_pseudobulk_chr" : total_RE_pseudobulk_chr,
            "total_peaks_df" : total_peaks_df
            }
    
    return total_TG_pseudobulk_global, pseudobulk_chrom_dict

def create_or_load_genomic_windows(
    window_size,
    chrom_id,
    genome_window_file,
    chrom_sizes_file,
    force_recalculate=False,
    promoter_only=False,
):
    """
    Create/load fixed-size genomic windows for a chromosome.
    When `promoter_only=True`, skip windows entirely and return an empty
    DataFrame with the expected schema so callers don't break.
    """

    # Promoter-centric evaluation: no windows needed
    if promoter_only:
        return pd.DataFrame(columns=["chrom", "start", "end", "win_idx"])

    if not os.path.exists(genome_window_file) or force_recalculate:
        logging.info("\nCreating genomic windows")
        genome_windows = pybedtools.bedtool.BedTool().window_maker(g=chrom_sizes_file, w=window_size)
        # Ensure consistent column names regardless of BedTool defaults
        chrom_windows = (
            genome_windows
            .filter(lambda x: x.chrom == chrom_id)
            .saveas(genome_window_file)
            .to_dataframe(names=["chrom", "start", "end"])
        )
        logging.info(f"  - Created {chrom_windows.shape[0]} windows")
    else:
        logging.info("\nLoading existing genomic windows")
        chrom_windows = pybedtools.BedTool(genome_window_file).to_dataframe(names=["chrom", "start", "end"])

    chrom_windows = chrom_windows.reset_index(drop=True)
    chrom_windows["win_idx"] = chrom_windows.index
    return chrom_windows

def make_peak_to_window_map(peaks_bed: pd.DataFrame, windows_bed: pd.DataFrame, peaks_as_windows: bool = True,) -> dict[str, int]:
    """
    Map each peak to the window it overlaps the most.
    Ensures the BedTool 'name' field is exactly the `peak_id` column.
    """
    # Defensive copy & column order
    pb = peaks_bed.copy()
    required = ["chrom", "start", "end", "peak_id"]
    missing = [c for c in required if c not in pb.columns]
    if missing:
        raise ValueError(f"peaks_bed missing columns: {missing}")

    pb["chrom"] = pb["chrom"].astype(str)
    pb["start"] = pb["start"].astype(int)
    pb["end"]   = pb["end"].astype(int)
    pb["peak_id"] = pb["peak_id"].astype(str).str.strip()
    
    if peaks_as_windows or windows_bed is None or windows_bed.empty:
        # stable order mapping: as they appear
        return {pid: int(i) for i, pid in enumerate(pb["peak_id"].tolist())}

    # BedTool uses the 4th column as "name. make it explicitly 'peak_id'
    pb_for_bed = pb[["chrom", "start", "end", "peak_id"]].rename(columns={"peak_id": "name"})
    bedtool_peaks   = pybedtools.BedTool.from_dataframe(pb_for_bed)

    # windows: enforce expected columns & dtypes
    wb = windows_bed.copy()
    wr = ["chrom", "start", "end", "win_idx"]
    miss_w = [c for c in wr if c not in wb.columns]
    if miss_w:
        raise ValueError(f"windows_bed missing columns: {miss_w}")
    wb["chrom"] = wb["chrom"].astype(str)
    wb["start"] = wb["start"].astype(int)
    wb["end"]   = wb["end"].astype(int)
    wb["win_idx"] = wb["win_idx"].astype(int)
    bedtool_windows = pybedtools.BedTool.from_dataframe(wb[["chrom", "start", "end", "win_idx"]])

    overlaps = {}
    for iv in bedtool_peaks.intersect(bedtool_windows, wa=True, wb=True):
        # left fields (peak): chrom, start, end, name
        peak_id = iv.name  # guaranteed to be the 'name' we set = peak_id
        # right fields (window): ... chrom, start, end, win_idx (as the last field)
        win_idx = int(iv.fields[-1])

        peak_start, peak_end = int(iv.start), int(iv.end)
        win_start, win_end   = int(iv.fields[-3]), int(iv.fields[-2])
        overlap_len = max(0, min(peak_end, win_end) - max(peak_start, win_start))

        if overlap_len <= 0:
            continue
        overlaps.setdefault(peak_id, []).append((overlap_len, win_idx))

    # resolve ties by max-overlap then random
    mapping = {}
    for pid, lst in overlaps.items():
        if not lst: 
            continue
        max_ol = max(lst, key=lambda x: x[0])[0]
        candidates = [w for ol, w in lst if ol == max_ol]
        mapping[str(pid)] = int(random.choice(candidates))

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
    genome_wide_tg_expression: np.ndarray,                   # [num_TG_chr, num_cells]
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

    # ---- TG tensor ----
    tg_tensor_all = torch.as_tensor(
        np.asarray(genome_wide_tg_expression, dtype=np.float32), dtype=dtype
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
    chrom_sizes_file = GENOME_DIR / (ORGANISM_CODE + f"{ORGANISM_CODE}.chrom.sizes")
    
    if not os.path.isdir(GENOME_DIR):
        os.makedirs(GENOME_DIR)
    
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
        
    download_ncbi_gene_info_mouse()
    download_ensembl_gtf_mouse(release=115, assembly="GRCm39", decompress=False)
    
    if ORGANISM_CODE == "mm10":
        species_taxid = "10090"
    elif ORGANISM_CODE == "hg38":
        species_taxid = "9606"

    gc = GeneCanonicalizer(use_mygene=False)
    gc.load_gtf(str(GTF_FILE_DIR / "Mus_musculus.GRCm39.115.gtf.gz"))
    gc.load_ncbi_gene_info(str(NCBI_FILE_DIR / "Mus_musculus.gene_info.gz"), species_taxid=species_taxid)
    logging.info(f"Map sizes: {gc.coverage_report()}")
        
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
    
    logging.info(f"FORCE_RECALCULATE: {FORCE_RECALCULATE}")
    
    PROCESS_SAMPLE_DATA = True
    logging.info(f"PROCESS_SAMPLE_DATA: {PROCESS_SAMPLE_DATA}")
    PROCESS_CHROMOSOME_SPECIFIC_DATA = True
    logging.info(f"PROCESS_CHROMOSOME_SPECIFIC_DATA: {PROCESS_CHROMOSOME_SPECIFIC_DATA}")
    
    def _per_sample_worker(sample_name: str) -> dict:
        try:
            sample_input_dir = RAW_DATA / DATASET_NAME / sample_name
            out_dir = SAMPLE_PROCESSED_DATA_DIR / sample_name
            out_dir.mkdir(parents=True, exist_ok=True)
            SAMPLE_DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            
            process_or_load_rna_atac_data(
                sample_input_dir,
                force_recalculate=FORCE_RECALCULATE,
                raw_10x_rna_data_dir=RAW_10X_RNA_DATA_DIR / sample_name,
                raw_atac_peak_file=RAW_ATAC_PEAK_MATRIX_FILE,
                sample_name=sample_name,
                neighbors_k=NEIGHBORS_K,
                pca_components=PCA_COMPONENTS,
                hops=HOPS,
                self_weight=SELF_WEIGHT,
                load=False
            )
            return {"sample": sample_name, "ok": True}
        
        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"[{sample_name}] failed: {e}\n{tb}")
            return {"sample": sample_name, "ok": False, "error": str(e)}
    
    # Pre-process samples
    results = []
    with ProcessPoolExecutor(max_workers=num_cpu, mp_context=None) as ex:
        futs = {ex.submit(_per_sample_worker, s): s for s in SAMPLE_NAMES}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            if res["ok"]:
                print(f"{res['sample']} done")
            else:
                print(f"{res['sample']} failed: {res.get('error','')}")
    
    # ----- SAMPLE-SPECIFIC PREPROCESSING -----     
    if PROCESS_SAMPLE_DATA == True:
            
        # Sample-specific preprocessing
        for sample_name in SAMPLE_NAMES:
            sample_input_dir = RAW_DATA / DATASET_NAME / sample_name
            
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
            
            processed_rna_df, processed_atac_df, pseudobulk_rna_df, pseudobulk_atac_df = process_or_load_rna_atac_data(
                sample_input_dir,
                force_recalculate=FORCE_RECALCULATE,
                raw_10x_rna_data_dir=RAW_10X_RNA_DATA_DIR / sample_name,
                raw_atac_peak_file=RAW_ATAC_PEAK_MATRIX_FILE,
                sample_name=sample_name,
                neighbors_k=NEIGHBORS_K,
                pca_components=PCA_COMPONENTS,
                hops=HOPS,
                self_weight=SELF_WEIGHT,
            )

            processed_rna_df.index = pd.Index(
                    gc.canonicalize_series(pd.Series(processed_rna_df.index, dtype=object)).array
                )
            
            # ----- GET TFs, TGs, and TF-TG combinations -----
            genes = processed_rna_df.index.to_list()
            peaks = processed_atac_df.index.to_list()
            
            logging.info("\nProcessed RNA and ATAC files loaded")
            logging.info(f"  - Number of genes: {processed_rna_df.shape[0]}: {genes[:3]}")
            logging.info(f"  - Number of peaks: {processed_atac_df.shape[0]}: {peaks[:3]}")

            
            tfs, tgs, tf_tg_df = create_tf_tg_combination_files(genes, TF_FILE, SAMPLE_PROCESSED_DATA_DIR)
            
            tfs = sorted(set(gc.canonicalize_series(pd.Series(tfs)).tolist()))
            tgs = sorted(set(gc.canonicalize_series(pd.Series(tgs)).tolist()))
            
            # Format the peaks to BED format (chrom, start, end, peak_id)
            peak_locs_df = format_peaks(pd.Series(processed_atac_df.index)).rename(columns={"chromosome": "chrom"})
            
            if not os.path.isfile(peak_bed_file):
                # Write the peak BED file
                peak_bed_file.parent.mkdir(parents=True, exist_ok=True)
                pybedtools.BedTool.from_dataframe(
                    peak_locs_df[["chrom", "start", "end", "peak_id"]]
                ).saveas(peak_bed_file)
            
            # ----- CALCULATE PEAK TO TG DISTANCE -----
            # Calculate the distance from each peak to each gene TSS
            if not os.path.isfile(peak_to_gene_dist_file) or FORCE_RECALCULATE:
                # Download the gene TSS file from Ensembl if missing

                logging.info("\nCalculating peak to TG distance score")
                peak_to_gene_dist_df = calculate_peak_to_tg_distance_score(
                    peak_bed_file=peak_bed_file,
                    tss_bed_file=GENE_TSS_FILE,
                    peak_gene_dist_file=peak_to_gene_dist_file,
                    mesc_atac_peak_loc_df=peak_locs_df,
                    gene_tss_df=gene_tss_df,
                    max_peak_distance = MAX_PEAK_DISTANCE,
                    distance_factor_scale = DISTANCE_SCALE_FACTOR,
                    force_recalculate=FORCE_RECALCULATE
                )
            
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

            # ----- CALCULATE TF-TG REGULATORY POTENTIAL -----
            if not os.path.isfile(tf_tg_reg_pot_file):
                tf_tg_reg_pot = calculate_tf_tg_regulatory_potential(
                    sliding_window_score_file, tf_tg_reg_pot_file, peak_to_gene_dist_file, num_cpu)


            # ----- MERGE TF-TG ATTRIBUTES WITH COMBINATIONS -----
            if not os.path.isfile(tf_tg_combo_attr_file):
                logging.info("\nLoading TF-TG regulatory potential scores")
                tf_tg_reg_pot = pd.read_parquet(tf_tg_reg_pot_file, engine="pyarrow")
                logging.debug("  - Example TF-TG regulatory potential: " + str(tf_tg_reg_pot.head()))
                
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
                
                mean_norm_tf_expr, mean_norm_tg_expr = compute_minmax_expr_mean(tf_df, tg_df)
                
                logging.info("\nMerging TF-TG attributes with all combinations")
                tf_tg_df = merge_tf_tg_attributes_with_combinations(
                    tf_tg_df, tf_tg_reg_pot, mean_norm_tf_expr, mean_norm_tg_expr, tf_tg_combo_attr_file, set(tfs))            

            # ----- MERGE TF-TG DATA WITH PKN -----
            if not os.path.isfile(gat_training_file):
                logging.info("\nMerging TF-TG data with PKN")
                logging.info("  - Loading TF-TG attributes with all combinations")
                tf_tg_df = pd.read_parquet(tf_tg_combo_attr_file, engine="pyarrow")
                tf_tg_labeled_with_pkn, tf_tg_unlabeled = merge_tf_tg_data_with_pkn(
                    tf_tg_df, 
                    string_csv_file, 
                    trrust_csv_file, 
                    kegg_csv_file
                )
                logging.debug(f"  - Example of TF-TG data with PKN: {tf_tg_labeled_with_pkn.head()}")

                logging.info("\nWriting Final TF-TG GAT training data to parquet")
                tf_tg_labeled_with_pkn.to_parquet(gat_training_file, engine="pyarrow", compression="snappy")
                logging.info(f"  - Wrote TF-TG features to {gat_training_file}")
        
    # ----- CHROMOSOME-SPECIFIC PREPROCESSING -----
    if PROCESS_CHROMOSOME_SPECIFIC_DATA:        
        chrom_list = CHROM_IDS
        total_tf_list_file = SAMPLE_PROCESSED_DATA_DIR / "tf_tg_combos" / "tf_list.csv"
        tf_names = _read_list(total_tf_list_file, "TF")
        
        # Aggregate sample-level data for sliding window scores and peak to gene distance
        sample_level_sliding_window_dfs = []
        sample_level_peak_to_gene_dist_dfs = []
        for sample_name in SAMPLE_NAMES:
            sliding_window_score_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "sliding_window.parquet"
            peak_to_gene_dist_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "peak_to_gene_dist.parquet"
            
            
            sliding_window_df = pd.read_parquet(sliding_window_score_file, engine="pyarrow")
            sample_level_sliding_window_dfs.append(sliding_window_df)
            
            peak_to_gene_dist_df = pd.read_parquet(peak_to_gene_dist_file, engine="pyarrow")
            sample_level_peak_to_gene_dist_dfs.append(peak_to_gene_dist_df)

        total_sliding_window_score_df = pd.concat(sample_level_sliding_window_dfs)
        total_peak_gene_dist_df = pd.concat(sample_level_peak_to_gene_dist_dfs)
        
        logging.info(f"Aggregating pseudobulk datasets")
        dataset_processed_data_dir = RAW_DATA / DATASET_NAME
        total_TG_pseudobulk_global, pseudobulk_chrom_dict = \
            aggregate_pseudobulk_datasets(SAMPLE_NAMES, dataset_processed_data_dir, chrom_list, gc)
        
        logging.info(f"  - Number of chromosomes: {len(chrom_list)}: {chrom_list}")
        for chrom_id in chrom_list:
            logging.info(f"\n----- Preparing MultiomicTransformer data for {DATASET_NAME} {chrom_id} -----")
            make_chrom_gene_tss_df(GENE_TSS_FILE, chrom_id, GENOME_DIR)
            
            SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR = SAMPLE_DATA_CACHE_DIR / chrom_id
        
            single_cell_dir = SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / "single_cell"
            
            # Chromosome-specific cache files
            atac_tensor_path: Path =            SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"atac_window_tensor_all_{chrom_id}.pt"
            tg_tensor_path: Path =              SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_tensor_all_{chrom_id}.pt"
            sample_tg_name_file: Path =         SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_names_{chrom_id}.json"
            genome_window_file: Path =          SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"{chrom_id}_windows_{WINDOW_SIZE // 1000}kb.bed"
            sample_window_map_file: Path =      SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"window_map_{chrom_id}.json"
            peak_to_tss_dist_path: Path =       SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"genes_near_peaks_{chrom_id}.parquet"
            dist_bias_file: Path =              SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"dist_bias_{chrom_id}.pt"
            tg_id_file: Path =                  SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_ids_{chrom_id}.pt"
            manifest_file: Path =               SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"manifest_{chrom_id}.json"
            motif_mask_file: Path =             SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"motif_mask_{chrom_id}.pt"
            chrom_sliding_window_file: Path =   SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"sliding_window_{chrom_id}.parquet"
            chrom_peak_bed_file: Path =         SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"peak_tmp_{chrom_id}.bed"
            tss_bed_file: Path =                SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tss_tmp_{chrom_id}.bed"

            os.makedirs(COMMON_DATA, exist_ok=True)
            os.makedirs(SAMPLE_DATA_CACHE_DIR, exist_ok=True)
            os.makedirs(SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR, exist_ok=True)
            os.makedirs(single_cell_dir, exist_ok=True)
                        
            # Create or load the gene TSS information for the chromosome
            if not os.path.isfile(os.path.join(GENOME_DIR, f"{chrom_id}_gene_tss.bed")):
                gene_tss_df = make_chrom_gene_tss_df(
                    gene_tss_file=GENE_TSS_FILE,
                    chrom_id=chrom_id,
                    genome_dir=GENOME_DIR
                )
            else:
                logging.info(f"Loading existing gene TSS file for {chrom_id}")
                gene_tss_df = pd.read_csv(os.path.join(GENOME_DIR, f"{chrom_id}_gene_tss.bed"), sep="\t", header=None, usecols=[0, 1, 2, 3])
                gene_tss_df = gene_tss_df.rename(columns={0: "chrom", 1: "start", 2: "end", 3: "name"})
                
                
            total_TG_pseudobulk_chr = pseudobulk_chrom_dict[chrom_id]["total_TG_pseudobulk_chr"]
            total_RE_pseudobulk_chr = pseudobulk_chrom_dict[chrom_id]["total_RE_pseudobulk_chr"]
            total_peaks_df = pseudobulk_chrom_dict[chrom_id]["total_peaks_df"]
                
            logging.info(f"  - {chrom_id}: TG pseudobulk shape={total_TG_pseudobulk_chr.shape}")
            logging.info(f"  - TG Examples:{total_TG_pseudobulk_chr.index[:5].tolist()}")
            
            vals = total_TG_pseudobulk_chr.values.astype("float32")
            if vals.shape[0] == 0:
                logging.warning(f"{chrom_id}: no TG rows after aggregation; skipping this chromosome.")
                continue
        
            tg_names = total_TG_pseudobulk_chr.index.tolist()
            
            # Genome-wide TF expression for all samples
            genome_wide_tf_expression = total_TG_pseudobulk_global.reindex(tf_names).fillna(0).values.astype("float32")
            metacell_names = total_TG_pseudobulk_global.columns.tolist()
            
            # Scale TG expression
            TG_expression = total_TG_pseudobulk_chr.values.astype("float32")
            
            chrom_peak_ids = set(total_peaks_df["peak_id"].astype(str))
            
            # Create genome windows
            logging.info(f"Creating genomic windows for {chrom_id}")
            genome_windows = create_or_load_genomic_windows(
                window_size=WINDOW_SIZE,
                chrom_id=chrom_id,
                chrom_sizes_file=CHROM_SIZES_FILE,
                genome_window_file=genome_window_file,
                force_recalculate=FORCE_RECALCULATE,
                promoter_only=(PROMOTER_BP is not None)
            )
            
                # --- Calculate Peak-to-TG Distance Scores ---
            genes_near_peaks = total_peak_gene_dist_df[total_peak_gene_dist_df["peak_id"].astype(str).isin(chrom_peak_ids)].copy()
                
            # genes_near_peaks = calculate_peak_to_tg_distance_score(
            #     peak_bed_file=chrom_peak_bed_file,
            #     tss_bed_file=tss_bed_file,
            #     peak_gene_dist_file=peak_to_tss_dist_path,
            #     mesc_atac_peak_loc_df=total_peaks_df,  # peak locations DataFrame
            #     gene_tss_df=gene_tss_df,
            #     max_peak_distance= MAX_PEAK_DISTANCE,
            #     distance_factor_scale= DISTANCE_SCALE_FACTOR,
            #     force_recalculate=FORCE_RECALCULATE,
            #     filter_to_nearest_gene=FILTER_TO_NEAREST_GENE,
            #     promoter_bp=PROMOTER_BP
            # )
            genes_near_peaks["target_id"] = gc.canonicalize_series(genes_near_peaks["target_id"])
            genes_near_peaks.to_parquet(peak_to_tss_dist_path, engine="pyarrow", compression="snappy")
            logging.info(f"  - Saved peak-to-TG distance scores to {peak_to_tss_dist_path}")

            
            # ----- SLIDING WINDOW TF-PEAK SCORE -----
            if not os.path.isfile(chrom_sliding_window_file):
                
                sliding_window_df = total_sliding_window_score_df[
                    total_sliding_window_score_df["peak_id"].astype(str).isin(chrom_peak_ids)
                ][["TF","peak_id","sliding_window_score"]].copy()

                # normalize types/names
                sliding_window_df["TF"] = sliding_window_df["TF"].astype(str)
                sliding_window_df["peak_id"] = sliding_window_df["peak_id"].astype(str)
                sliding_window_df["sliding_window_score"] = pd.to_numeric(sliding_window_df["sliding_window_score"], errors="coerce")

                # collapse duplicates across samples
                sliding_window_df = (
                    sliding_window_df
                    .groupby(["TF","peak_id"], as_index=False, sort=False)
                    .agg(sliding_window_score=("sliding_window_score","mean"))
                )
                
                sliding_window_df.to_parquet(chrom_sliding_window_file, engine="pyarrow", compression="snappy")

                # peaks_df = pybedtools.BedTool(peak_bed_file)

                # logging.info("Running sliding window scan")
                # run_sliding_window_scan(
                #     tf_name_list=tfs,
                #     tf_info_file=str(TF_FILE),
                #     motif_dir=str(MOTIF_DIR),
                #     genome_fasta=str(genome_fasta_file),
                #     peak_bed_file=str(chrom_peak_bed_file),
                #     output_file=chrom_sliding_window_file,
                #     num_cpu=num_cpu
                # )
                logging.info(f"  - Wrote sliding window scores to {chrom_sliding_window_file}")
                # sliding_window_df = pd.read_parquet(chrom_sliding_window_file, engine="pyarrow")
            else:
                logging.info("Loading existing sliding window scores")
                sliding_window_df = pd.read_parquet(chrom_sliding_window_file, engine="pyarrow")
            
            total_peaks_df["peak_id"] = total_peaks_df["peak_id"].astype(str)
            genes_near_peaks["peak_id"] = genes_near_peaks["peak_id"].astype(str)
            
            peaks_as_windows = (PROMOTER_BP is not None)

            if peaks_as_windows:
                genome_windows = total_peaks_df[["chrom", "start", "end"]].copy().reset_index(drop=True)
                
            num_windows = int(genome_windows.shape[0])

            window_map = make_peak_to_window_map(
                peaks_bed=total_peaks_df,
                windows_bed=genome_windows,
                peaks_as_windows=peaks_as_windows,
            )

            total_RE_pseudobulk_chr.index = (
                total_RE_pseudobulk_chr.index.astype(str).str.strip()
            )

            tf_tensor_all, tg_tensor_all, atac_window_tensor_all = precompute_input_tensors(
                output_dir=str(SAMPLE_DATA_CACHE_DIR),
                genome_wide_tf_expression=genome_wide_tf_expression,
                genome_wide_tg_expression=TG_expression,
                total_RE_pseudobulk_chr=total_RE_pseudobulk_chr,
                window_map=window_map,
                windows=genome_windows,   # now aligned with map
            )
            
            # ----- Load common TF and TG vocab -----
            # Create a common TG vocabulary for the chromosome using the gene TSS
            logging.info(f"Matching TFs and TGs to global gene vocabulary")
            
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
            
            # Build motif mask using merged info
            motif_mask = build_motif_mask(
                tf_names=tf_names_kept,
                tg_names=tg_names_kept,
                sliding_window_df=sliding_window_df,
                genes_near_peaks=genes_near_peaks
            )

            if not tf_ids: raise ValueError("No TFs matched the common vocab.")
            if not tg_ids: raise ValueError("No TGs matched the common vocab.")
            
            # Build distance bias [num_windows x num_tg_kept] aligned to kept TGs
            logging.info(f"Building distance bias")
            dist_bias = build_distance_bias(
                genes_near_peaks=genes_near_peaks,
                window_map=window_map,
                tg_names_kept=tg_names_kept,
                num_windows=num_windows,
                dtype=torch.float32,
                mode=DIST_BIAS_MODE
            )
            
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
            logging.info(f"Writing output files")
            # Save the Window, TF, and TG expression tensors
            torch.save(atac_window_tensor_all, atac_tensor_path)
            torch.save(tg_tensor_all, tg_tensor_path)
            torch.save(tf_tensor_all, tf_tensor_path)

            # Save the peak -> window map for the sample
            atomic_json_dump(window_map, sample_window_map_file)

            # Write TF and TG names and global vocab indices present in the sample
            atomic_json_dump(tf_names_kept, sample_tf_name_file)
            atomic_json_dump(tg_names_kept, sample_tg_name_file)


            # Write the distance bias and metacell names for the sample
            torch.save(dist_bias, dist_bias_file)
            logging.info(f"  - Saved distance bias tensor with shape {tuple(dist_bias.shape)} to {dist_bias_file}")

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
                    "motif_mask": str(motif_mask_file),
                }
            }
            with open(manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)

            logging.info("Preprocessing complete. Wrote per-sample/per-chrom data for MultiomicTransformerDataset.")
            