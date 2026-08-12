"""Reference-based label transfer from the Pijuan-Sala 2019 atlas onto the integrated mESC MuData.

Replaces marker-signature scoring with whole-transcriptome similarity to annotated
reference cells. Marker scoring cannot separate nested populations that share genes
(Visceral vs ExE endoderm, Somitic vs Paraxial mesoderm, Caudal epiblast vs NMP) and
will confidently assign a label whose signature is driven by non-specific genes --
the marker run put 2,366 cells in PGC, a population that is 392 cells in the whole
116k-cell atlas.

Method: reference and query are concatenated on shared genes, normalised identically,
embedded by PCA, and integrated with Harmony using a batch key that spans both
datasets. Query cells are then labelled by a distance-weighted k-nearest-reference-cell
vote, with the winning label's weight fraction retained as a per-cell confidence.

Reference: Pijuan-Sala et al., Nature 566:490-495 (2019), doi:10.1038/s41586-019-0933-9.
Data: https://content.cruk.cam.ac.uk/jmlab/atlas_data/

Example:
    python label_transfer_pijuansala.py --ref_dir <atlas> --h5mu <combined.h5mu>
"""

import argparse
import gc
import gzip
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.mudata_utils import run_harmony, sanitize_for_h5  # noqa: E402


def log(m):
    print(f"[label_transfer] {m}", flush=True)


def load_reference(ref_dir, cache):
    """Build the reference AnnData, caching the parsed result as h5ad."""
    import anndata as ad
    import scanpy as sc

    ref_dir, cache = Path(ref_dir), Path(cache)
    if cache.exists():
        log(f"loading cached reference {cache}")
        return ad.read_h5ad(cache)

    log("parsing atlas meta/genes ...")
    meta = pd.read_csv(ref_dir / "meta.tab.gz", sep="\t")
    genes = pd.read_csv(ref_dir / "genes.tsv.gz", sep="\t", header=None,
                        names=["ensembl", "symbol"])

    mtx = ref_dir / "raw_counts.mtx"
    if not mtx.exists():
        log("decompressing raw_counts.mtx.gz (once) ...")
        with gzip.open(ref_dir / "raw_counts.mtx.gz", "rb") as fh, open(mtx, "wb") as out:
            while chunk := fh.read(1 << 24):
                out.write(chunk)

    from scipy.io import mmread
    log("reading counts matrix (large, several minutes) ...")
    X = mmread(str(mtx))
    X = sp.csr_matrix(X)
    log(f"  matrix {X.shape} ({X.nnz/1e6:.0f}M nonzero)")

    # Bioconductor writes genes x cells; transpose to AnnData's cells x genes.
    if X.shape[0] == len(genes) and X.shape[1] == len(meta):
        X = X.T.tocsr()
    elif not (X.shape[0] == len(meta) and X.shape[1] == len(genes)):
        raise RuntimeError(f"matrix {X.shape} matches neither "
                           f"{len(meta)} cells x {len(genes)} genes")

    a = ad.AnnData(X=X.astype(np.float32),
                   obs=meta.set_index("cell"),
                   var=genes.set_index("symbol"))
    a.var_names_make_unique()

    # Doublets, stripped nuclei and unlabelled cells cannot serve as references.
    keep = (~a.obs["doublet"].astype(bool)) & (~a.obs["stripped"].astype(bool)) \
        & a.obs["celltype"].notna()
    a = a[keep].copy()
    log(f"reference after QC: {a.n_obs} cells x {a.n_vars} genes, "
        f"{a.obs['celltype'].nunique()} populations")
    a.write_h5ad(cache, compression="gzip")
    return a


def transfer(ref_emb, ref_labels, qry_emb, k=30, seed=0):
    """Distance-weighted kNN vote; returns labels, confidence and runner-up."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(ref_emb)
    dist, idx = nn.kneighbors(qry_emb)
    # Gaussian kernel on distance, bandwidth = each cell's median neighbour distance,
    # so a cell in a sparse region is not dominated by one very close neighbour.
    bw = np.median(dist, axis=1, keepdims=True)
    bw[bw == 0] = 1e-12
    w = np.exp(-(dist / bw) ** 2)

    cats = pd.Categorical(ref_labels)
    codes = cats.codes[idx]                      # (n_query, k)
    n_cat = len(cats.categories)
    votes = np.zeros((len(qry_emb), n_cat), dtype=np.float64)
    for j in range(codes.shape[1]):
        np.add.at(votes, (np.arange(len(qry_emb)), codes[:, j]), w[:, j])
    votes /= votes.sum(axis=1, keepdims=True)

    order = np.argsort(-votes, axis=1)
    best, second = order[:, 0], order[:, 1]
    return (np.asarray(cats.categories)[best],
            votes[np.arange(len(votes)), best],
            np.asarray(cats.categories)[second],
            votes[np.arange(len(votes)), second])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_dir", default="/gpfs/Labs/Uzun/DATA/PROJECTS/"
                   "2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/REFERENCE/pijuan_sala_atlas")
    p.add_argument("--h5mu", default="data/processed/mESC/combined/mESC_combined.h5mu")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_hvg", type=int, default=3000)
    p.add_argument("--n_pcs", type=int, default=50)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--min_confidence", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_write_h5mu", action="store_true")
    a = p.parse_args()

    import anndata as ad
    import muon as mu
    import scanpy as sc

    sc.settings.n_jobs = 16
    out = Path(a.out_dir) if a.out_dir else Path(a.h5mu).parent
    out.mkdir(parents=True, exist_ok=True)

    ref = load_reference(a.ref_dir, Path(a.ref_dir) / "atlas_reference.h5ad")

    log(f"reading query {a.h5mu}")
    mdata = mu.read(a.h5mu)
    q = mdata["rna"]
    qry = ad.AnnData(X=q.layers["counts"].copy(),
                     obs=q.obs[["sample", "timepoint", "leiden", "leiden_wnn",
                                "celltype_pijuansala"]].copy(),
                     var=pd.DataFrame(index=q.var_names.copy()))
    log(f"query: {qry.n_obs} cells x {qry.n_vars} genes")

    shared = sorted(set(ref.var_names) & set(qry.var_names))
    log(f"shared genes: {len(shared)}")
    ref = ref[:, shared].copy()
    qry = qry[:, shared].copy()

    ref.obs["dataset"] = "atlas"
    qry.obs["dataset"] = "query"
    ref.obs["batch"] = "atlas_" + ref.obs["sample"].astype(str)
    qry.obs["batch"] = "query_" + qry.obs["sample"].astype(str)
    ref.obs["celltype"] = ref.obs["celltype"].astype(str)
    qry.obs["celltype"] = "unknown"

    keep_cols = ["dataset", "batch", "celltype"]
    joint = ad.concat([ref[:, shared], qry[:, shared]], join="inner",
                      index_unique="-", label=None,
                      keys=None, merge="unique")
    joint.obs = pd.concat([ref.obs[keep_cols], qry.obs[keep_cols]], axis=0).set_axis(
        joint.obs_names)
    log(f"joint: {joint.n_obs} cells x {joint.n_vars} genes")
    del ref
    gc.collect()

    sc.pp.normalize_total(joint, target_sum=1e4)
    sc.pp.log1p(joint)
    # flavor='seurat' works on log-normalised values, which is what `joint` holds;
    # seurat_v3 would need a raw-count layer that is deliberately not kept here.
    # batch_key='dataset' stops atlas-specific variance from dominating the HVGs.
    sc.pp.highly_variable_genes(joint, n_top_genes=a.n_hvg, flavor="seurat",
                                batch_key="dataset")
    log(f"HVGs: {int(joint.var['highly_variable'].sum())}")

    # Subset to HVGs *before* scaling: sc.pp.scale densifies, and 164k cells x 17k
    # genes would be ~11 GB dense against ~2 GB for the HVG subset.
    hv = joint[:, joint.var["highly_variable"].values].copy()
    sc.pp.scale(hv, max_value=10)
    sc.tl.pca(hv, n_comps=a.n_pcs, svd_solver="arpack")
    joint.obsm["X_pca"] = hv.obsm["X_pca"]
    del hv
    gc.collect()

    log("running Harmony over reference + query (key='batch') ...")
    run_harmony(joint, key="batch", basis="X_pca", adjusted_basis="X_pca_harmony",
                seed=a.seed)

    is_ref = (joint.obs["dataset"] == "atlas").values
    emb = joint.obsm["X_pca_harmony"]
    log(f"kNN transfer (k={a.k}) from {int(is_ref.sum())} reference cells ...")
    lab, conf, run2, conf2 = transfer(emb[is_ref], joint.obs["celltype"].values[is_ref],
                                      emb[~is_ref], k=a.k, seed=a.seed)

    qry.obs["celltype_transfer"] = lab
    qry.obs["transfer_confidence"] = conf
    qry.obs["transfer_runner_up"] = run2
    qry.obs["transfer_runner_up_conf"] = conf2
    qry.obs["transfer_confident"] = conf >= a.min_confidence

    log(f"mean confidence {conf.mean():.3f}; "
        f"{100*(conf >= a.min_confidence).mean():.1f}% of cells >= {a.min_confidence}")

    # --- per-cluster consensus -------------------------------------------
    summaries = {}
    for groupby, tag in [("leiden", "rna_leiden"), ("leiden_wnn", "wnn_leiden")]:
        rows = []
        for cl, sub in qry.obs.groupby(groupby, observed=True):
            vc = sub["celltype_transfer"].value_counts(normalize=True)
            rows.append({
                "cluster": str(cl), "n_cells": int(len(sub)),
                "consensus_celltype": vc.index[0],
                "consensus_frac": round(float(vc.iloc[0]), 3),
                "runner_up": vc.index[1] if len(vc) > 1 else "",
                "runner_up_frac": round(float(vc.iloc[1]), 3) if len(vc) > 1 else 0.0,
                "mean_confidence": round(float(sub["transfer_confidence"].mean()), 3),
            })
        df = pd.DataFrame(rows).sort_values("n_cells", ascending=False)
        df.to_csv(out / f"label_transfer_clusters_{tag}.csv", index=False)
        summaries[tag] = {"n_clusters": int(len(df)),
                          "n_populations": int(df["consensus_celltype"].nunique())}
        log(f"\n--- {tag}: {len(df)} clusters -> "
            f"{df['consensus_celltype'].nunique()} populations ---")
        for _, r in df.iterrows():
            log(f"  cl {r['cluster']:>3} (n={r['n_cells']:>6}): "
                f"{r['consensus_celltype']:<32} {100*r['consensus_frac']:.0f}% "
                f"(next {r['runner_up']} {100*r['runner_up_frac']:.0f}%) "
                f"conf={r['mean_confidence']:.2f}")
        # Cluster-level label for every cell in that cluster.
        mapping = dict(zip(df["cluster"], df["consensus_celltype"]))
        qry.obs[f"celltype_transfer_{tag}"] = qry.obs[groupby].astype(str).map(mapping)

    # --- composition & comparison ----------------------------------------
    per_cell = qry.obs["celltype_transfer"].value_counts()
    per_cell.to_csv(out / "label_transfer_population_sizes.csv")
    log("\n=== transferred population sizes (per cell) ===")
    log(per_cell.to_string())

    pd.crosstab(qry.obs["timepoint"], qry.obs["celltype_transfer"]).to_csv(
        out / "label_transfer_timepoint_composition.csv")
    pd.crosstab(qry.obs["celltype_pijuansala"], qry.obs["celltype_transfer"]).to_csv(
        out / "label_transfer_vs_markers.csv")

    agree = float((qry.obs["celltype_pijuansala"].astype(str)
                   == qry.obs["celltype_transfer"].astype(str)).mean())
    log(f"\nagreement with marker-based labels: {100*agree:.1f}% of cells")

    # --- write back --------------------------------------------------------
    cols = ["celltype_transfer", "transfer_confidence", "transfer_runner_up",
            "transfer_runner_up_conf", "transfer_confident",
            "celltype_transfer_rna_leiden", "celltype_transfer_wnn_leiden"]
    for c in cols:
        v = qry.obs[c].values
        mdata["rna"].obs[c] = pd.Categorical(v) if v.dtype == object else v
        mdata.obs[c] = mdata["rna"].obs[c].values
    qry.obs[cols + ["sample", "timepoint", "leiden", "leiden_wnn"]].to_csv(
        out / "label_transfer_per_cell.csv")

    summary = {
        "reference": {
            "citation": "Pijuan-Sala et al. Nature 566:490-495 (2019)",
            "doi": "10.1038/s41586-019-0933-9",
            "url": "https://content.cruk.cam.ac.uk/jmlab/atlas_data/",
            "n_reference_cells": int(is_ref.sum()),
            "n_populations": int(pd.Series(
                joint.obs["celltype"].values[is_ref]).nunique()),
        },
        "method": {"shared_genes": len(shared), "n_hvg": int(joint.var["highly_variable"].sum()),
                   "n_pcs": a.n_pcs, "k": a.k, "harmony_key": "batch"},
        "mean_confidence": float(conf.mean()),
        "pct_confident": float(100 * (conf >= a.min_confidence).mean()),
        "agreement_with_marker_labels": agree,
        "clusters": summaries,
    }
    (out / "label_transfer_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # --- figures -----------------------------------------------------------
    r = mdata["rna"]
    for col, fn, title in [
        ("celltype_transfer", "umap_label_transfer_percell.png",
         "Pijuan-Sala label transfer (per cell)"),
        ("celltype_transfer_rna_leiden", "umap_label_transfer_cluster.png",
         "Pijuan-Sala label transfer (cluster consensus)"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 8))
        sc.pl.umap(r, color=col, ax=ax, show=False, frameon=True, title=title,
                   legend_fontsize=7)
        for sp_ in ax.spines.values():
            sp_.set_visible(True); sp_.set_color("#3A424C"); sp_.set_linewidth(1.1)
        fig.tight_layout(); fig.savefig(out / fn, dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    sc.pl.umap(r, color="transfer_confidence", ax=ax, show=False, frameon=True,
               title="Label-transfer confidence", cmap="viridis")
    for sp_ in ax.spines.values():
        sp_.set_visible(True); sp_.set_color("#3A424C"); sp_.set_linewidth(1.1)
    fig.tight_layout()
    fig.savefig(out / "umap_transfer_confidence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if not a.no_write_h5mu:
        log("rewriting h5mu with transferred labels ...")
        for m in mdata.mod.values():
            sanitize_for_h5(m)
        mdata.write(a.h5mu)
        log("h5mu updated")
    log("Done.")


if __name__ == "__main__":
    sys.exit(main())
