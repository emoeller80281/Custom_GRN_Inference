"""Helpers shared across the mouse preprocessing workflow steps.

These live here rather than in any one numbered step because several steps need them:
step 04 builds the combined object, and steps 05 and 07 rewrite it after adding labels.
Both functions exist to work around library behaviour that fails late and expensively,
so the reasons are documented at the point of use.

Import from a numbered step directory with:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.mudata_utils import sanitize_for_h5
"""

import numpy as np
import pandas as pd


def sanitize_for_h5(adata):
    """Coerce object-dtype obs/var columns so h5ad/h5mu writing cannot fail.

    Assigning a block of mixed-dtype columns (``df[[a,b,c]] = arr``) collapses them
    all to object dtype; h5py then tries to write an integer column as variable
    length strings and raises deep inside the writer, after the expensive work is
    already done.
    """
    for df in (adata.obs, adata.var):
        for c in df.columns:
            if df[c].dtype != object:
                continue
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().all():
                df[c] = num.astype(np.int64) if (num % 1 == 0).all() else num.astype(float)
            else:
                df[c] = df[c].astype(str)
    return adata


def run_harmony(adata, key, basis, adjusted_basis, seed=0, max_iter=30):
    """Harmony on ``adata.obsm[basis]``, written to ``adata.obsm[adjusted_basis]``.

    Calls harmonypy directly rather than ``sc.external.pp.harmony_integrate``: that
    wrapper transposes ``Z_corr`` for harmonypy < 2.0's (PCs x cells) convention,
    but harmonypy 2.0 already returns (cells x PCs), so the wrapper hands anndata a
    matrix with the wrong leading dimension and it raises. Orientation is checked
    here instead of assumed, so either version works.
    """
    import harmonypy

    emb = np.asarray(adata.obsm[basis], dtype=np.float64)
    n_cells = adata.n_obs
    ho = harmonypy.run_harmony(emb, adata.obs, vars_use=[key],
                               max_iter_harmony=max_iter, random_state=seed)
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != n_cells:
        Z = Z.T
    if Z.shape[0] != n_cells:
        raise RuntimeError(
            f"Harmony returned {Z.shape}, incompatible with {n_cells} cells")
    adata.obsm[adjusted_basis] = np.ascontiguousarray(Z)
    return adata
