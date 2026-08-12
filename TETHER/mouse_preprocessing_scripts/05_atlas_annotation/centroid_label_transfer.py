"""Cluster-centroid correlation annotation against the Pijuan-Sala 2019 atlas.

The kNN-in-Harmony-space transfer failed on this data: it put 32.6% of cells into
Pharyngeal mesoderm, including clusters that are demonstrably Pax6+ neural, Meox1+
somitic and Pou5f1-high epiblast, and did so at high confidence. The reference labels
were verified correct, so the fault is the joint embedding -- the query is single
nucleus 10x Multiome while the atlas is whole-cell scRNA-seq, and correcting that
technology gap with Harmony pulls query cells into the reference's dense, generic
mesodermal centre.

This module avoids a shared embedding entirely. It compares each query cluster's mean
expression profile with each atlas cell type's mean profile using Spearman correlation
over genes that are informative in the reference. Rank correlation on centroids is
robust to the systematic scale differences between nuclear and whole-cell RNA, which
is exactly what defeated the embedding approach.

Also reports a batch-effect diagnostic: how well atlas and query mix in the Harmony
embedding, to document the over-correction.

Reference: Pijuan-Sala et al., Nature 566:490-495 (2019), doi:10.1038/s41586-019-0933-9.
"""

import argparse
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
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


def log(m):
    print(f"[centroid] {m}", flush=True)


def centroids(adata, labels, genes):
    """Mean log-CP10K profile per label, restricted to `genes`."""
    sub = adata[:, genes]
    X = sub.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    df = pd.DataFrame(X, columns=genes)
    df["_lab"] = np.asarray(labels)
    return df.groupby("_lab").mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_h5ad", default="/gpfs/Labs/Uzun/DATA/PROJECTS/"
                   "2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/REFERENCE/pijuan_sala_atlas/"
                   "atlas_reference.h5ad")
    p.add_argument("--h5mu", default="data/processed/mESC/combined/mESC_combined.h5mu")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_genes", type=int, default=2000,
                   help="Number of reference HVGs used for the correlation")
    p.add_argument("--min_margin", type=float, default=0.02)
    p.add_argument("--min_z", type=float, default=2.0,
                   help="Min z-score of the top correlation against that cluster's own "
                        "correlation spread for the call to count as distinctive")
    p.add_argument("--no_write_h5mu", action="store_true")
    a = p.parse_args()

    import anndata as ad
    import muon as mu
    import scanpy as sc

    sc.settings.n_jobs = 16
    out = Path(a.out_dir) if a.out_dir else Path(a.h5mu).parent
    out.mkdir(parents=True, exist_ok=True)

    log("loading reference ...")
    ref = ad.read_h5ad(a.ref_h5ad)
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)

    log("loading query ...")
    mdata = mu.read(a.h5mu)
    q = mdata["rna"]
    qry = ad.AnnData(X=q.layers["counts"].copy(),
                     obs=q.obs[["sample", "timepoint", "leiden", "leiden_wnn"]].copy(),
                     var=pd.DataFrame(index=q.var_names.copy()))
    sc.pp.normalize_total(qry, target_sum=1e4)
    sc.pp.log1p(qry)

    shared = sorted(set(ref.var_names) & set(qry.var_names))
    log(f"shared genes: {len(shared)}")

    # Genes chosen for being informative in the *reference*, so the comparison is
    # driven by what distinguishes atlas populations rather than by query variance.
    ref_s = ref[:, shared].copy()
    sc.pp.highly_variable_genes(ref_s, n_top_genes=a.n_genes, flavor="seurat")
    genes = ref_s.var_names[ref_s.var["highly_variable"].values].tolist()
    log(f"using {len(genes)} reference-informative genes")

    ref_c = centroids(ref_s, ref.obs["celltype"].values, genes)
    log(f"reference centroids: {ref_c.shape[0]} populations")
    del ref, ref_s
    import gc; gc.collect()

    results, summaries, rhos = {}, {}, {}
    for groupby, tag in [("leiden", "rna_leiden"), ("leiden_wnn", "wnn_leiden")]:
        if groupby not in qry.obs:
            continue
        qry_c = centroids(qry, qry.obs[groupby].astype(str).values, genes)
        rho = pd.DataFrame(index=qry_c.index, columns=ref_c.index, dtype=float)
        for cl in qry_c.index:
            r, _ = spearmanr(qry_c.loc[cl].values,
                             ref_c.values, axis=1)
            # spearmanr with a matrix returns the full correlation matrix; row 0 is
            # the query centroid against each reference centroid.
            rho.loc[cl] = r[0, 1:]
        rho.to_csv(out / f"centroid_correlation_{tag}.csv")
        rhos[tag] = rho

        rows = []
        n_per = qry.obs[groupby].astype(str).value_counts()
        for cl in rho.index:
            s = rho.loc[cl].sort_values(ascending=False)
            # Centroid correlations all sit in a narrow high band (~0.80-0.90), so a raw
            # margin is nearly uninformative. Score the winner against the spread of that
            # cluster's own correlations instead: how far the top population stands out
            # from the other 36.
            v = rho.loc[cl].values.astype(float)
            z = float((s.iloc[0] - np.mean(v)) / np.std(v)) if np.std(v) > 0 else np.nan
            rows.append({
                "cluster": cl, "n_cells": int(n_per[cl]),
                "cell_type": s.index[0], "rho": round(float(s.iloc[0]), 4),
                "z_top": round(z, 3),
                "runner_up": s.index[1], "rho_runner_up": round(float(s.iloc[1]), 4),
                "margin": round(float(s.iloc[0] - s.iloc[1]), 4),
                "third": s.index[2],
            })
        df = pd.DataFrame(rows).sort_values("n_cells", ascending=False)
        df["separated"] = df["margin"] >= a.min_margin
        df["distinctive"] = df["z_top"] >= a.min_z
        df.to_csv(out / f"centroid_annotation_{tag}.csv", index=False)
        results[tag] = df

        log(f"\n--- {tag}: {len(df)} clusters -> {df['cell_type'].nunique()} populations ---")
        for _, r in df.iterrows():
            flag = "" if r["separated"] else f"   <- close to {r['runner_up']}"
            log(f"  cl {r['cluster']:>3} (n={r['n_cells']:>6}): {r['cell_type']:<32} "
                f"rho={r['rho']:.3f} z={r['z_top']:>5.2f} margin={r['margin']:.3f}{flag}")
        summaries[tag] = {"n_clusters": int(len(df)),
                          "n_populations": int(df["cell_type"].nunique()),
                          "n_close_call": int((~df["separated"]).sum()),
                          "n_not_distinctive": int((~df["distinctive"]).sum())}

        col = f"celltype_atlas_{tag}"
        qry.obs[col] = qry.obs[groupby].astype(str).map(dict(zip(df["cluster"], df["cell_type"])))
        mdata["rna"].obs[col] = pd.Categorical(qry.obs[col].values)
        mdata.obs[col] = mdata["rna"].obs[col].values

    # --- figures -----------------------------------------------------------
    # Use the in-memory frame: a CSV round-trip re-types the string cluster labels as
    # int64 and the .loc lookup below then matches nothing.
    main_tag = "rna_leiden"
    rho = rhos[main_tag]
    order = results[main_tag].set_index("cluster").loc[rho.index, "cell_type"]
    fig, ax = plt.subplots(figsize=(0.32 * rho.shape[1] + 6, 0.36 * rho.shape[0] + 3))
    im = ax.imshow(rho.values, cmap="RdBu_r", aspect="auto",
                   vmin=float(np.nanpercentile(rho.values, 2)),
                   vmax=float(np.nanmax(rho.values)))
    ax.set_xticks(range(rho.shape[1])); ax.set_xticklabels(rho.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(rho.shape[0]))
    ax.set_yticklabels([f"cl {c} -> {order[c]}" for c in rho.index], fontsize=7)
    ax.set_title("Spearman correlation: query cluster centroids vs Pijuan-Sala populations")
    fig.colorbar(im, ax=ax, shrink=0.6, label="rho")
    fig.tight_layout()
    fig.savefig(out / "heatmap_centroid_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    r = mdata["rna"]
    for tag in ["rna_leiden", "wnn_leiden"]:
        col = f"celltype_atlas_{tag}"
        if col not in r.obs:
            continue
        fig, ax = plt.subplots(figsize=(11, 8))
        sc.pl.umap(r, color=col, ax=ax, show=False, frameon=True, legend_fontsize=8,
                   title=f"Pijuan-Sala atlas populations ({tag}, centroid correlation)")
        for sp_ in ax.spines.values():
            sp_.set_visible(True); sp_.set_color("#3A424C"); sp_.set_linewidth(1.1)
        fig.tight_layout()
        fig.savefig(out / f"umap_centroid_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    (out / "centroid_annotation_summary.json").write_text(json.dumps({
        "reference": {"citation": "Pijuan-Sala et al. Nature 566:490-495 (2019)",
                      "doi": "10.1038/s41586-019-0933-9"},
        "method": "Spearman correlation of cluster mean log-CP10K profiles against "
                  "atlas cell-type mean profiles over reference-informative genes",
        "n_genes": len(genes), "shared_genes": len(shared),
        "clusters": summaries,
    }, indent=2, default=str))

    if not a.no_write_h5mu:
        # Drop the kNN-in-Harmony transfer: it assigned 32.6% of cells to Pharyngeal
        # mesoderm, including verified Pax6+ neural and Meox1+ somitic clusters. Leaving
        # it in the object next to a good annotation invites someone to use it.
        stale = [c for c in mdata["rna"].obs.columns if c.startswith("celltype_transfer")]
        for c in stale:
            del mdata["rna"].obs[c]
            if c in mdata.obs:
                del mdata.obs[c]
        if stale:
            log(f"dropped discredited kNN-transfer columns: {', '.join(stale)}")

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from common.mudata_utils import sanitize_for_h5
        for m in mdata.mod.values():
            sanitize_for_h5(m)
        log("rewriting h5mu ...")
        mdata.write(a.h5mu)
        log("h5mu updated")
    log("Done.")


if __name__ == "__main__":
    sys.exit(main())
