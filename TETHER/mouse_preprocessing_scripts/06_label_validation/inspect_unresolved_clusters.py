"""Raw marker evidence for the clusters where centroid and marker calls disagree.

The centroid annotation and the canonical-marker check disagree on 5 of 23 RNA clusters.
Both are summary statistics, so neither settles it. This prints the underlying numbers --
mean log-CP10K and fraction of cells expressing, per candidate gene, in the disputed
cluster versus all other clusters -- so the call is made on evidence rather than on
whichever score happened to be larger.

Deliberately narrow: only the disputed clusters, only the genes that discriminate between
their competing candidates.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# cluster -> (competing candidates, discriminating genes)
DISPUTES = {
    "0":  (["Surface ectoderm", "Gut", "Epiblast"],
           ["Pou5f1", "Utf1", "Slc7a3", "Nanog",           # epiblast
            "Krt8", "Krt18", "Trp63", "Grhl3", "Wnt6",     # surface ectoderm
            "Foxa1", "Foxa2", "Sox17", "Trh", "Cer1",      # gut / endoderm
            "T", "Fgf8", "Mixl1"]),                        # streak
    "2":  (["Somitic mesoderm", "Caudal Mesoderm", "Intermediate mesoderm"],
           ["Meox1", "Pax3", "Tcf15", "Tbx6", "Cdx2", "T", "Pax2", "Osr1", "Wt1",
            "Hoxb9", "Aldh1a2"]),
    "18": (["Mesenchyme", "ExE mesoderm", "Allantois"],
           ["Pdgfra", "Prrx1", "Twist1", "Hand1", "Bmp4", "Postn", "Tbx4", "Hoxa10",
            "Rhox5", "Ahnak"]),
    "21": (["Mesenchyme", "Surface ectoderm"],
           ["Pdgfra", "Prrx1", "Twist1", "Krt8", "Krt18", "Trp63", "Grhl3", "Ttr",
            "Afp", "Apoa1"]),
    "13": (["Blood progenitors 1", "Blood progenitors 2"],
           ["Runx1", "Tal1", "Lmo2", "Gata1", "Klf1", "Hbb-bh1", "Hba-a1", "Kit",
            "Cd34"]),
}


def log(m):
    print(f"[inspect] {m}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5mu", default="data/processed/mESC/combined/mESC_combined.h5mu")
    p.add_argument("--cluster_col", default="leiden")
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()

    import anndata as ad
    import muon as mu
    import scanpy as sc

    out = Path(a.out_dir) if a.out_dir else Path(a.h5mu).parent
    log("loading ...")
    mdata = mu.read(a.h5mu)
    q = mdata["rna"]
    adata = ad.AnnData(X=q.layers["counts"].copy(),
                       obs=q.obs[[a.cluster_col]].copy(),
                       var=pd.DataFrame(index=q.var_names.copy()))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    cl = adata.obs[a.cluster_col].astype(str).values

    genes = sorted({g for _, gs in DISPUTES.values() for g in gs})
    present = [g for g in genes if g in adata.var_names]
    log(f"genes present: {len(present)}/{len(genes)}; "
        f"absent: {', '.join(sorted(set(genes) - set(present))) or 'none'}")

    X = adata[:, present].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    expr = pd.DataFrame(X, columns=present)
    expr["_cl"] = cl
    mean_by_cl = expr.groupby("_cl").mean()
    frac_by_cl = expr.drop(columns="_cl").gt(0).assign(_cl=cl).groupby("_cl").mean()

    records = []
    for c, (cands, gs) in DISPUTES.items():
        gs = [g for g in gs if g in present]
        log(f"\n=== cluster {c}  (candidates: {', '.join(cands)}) ===")
        log(f"{'gene':<10} {'mean_in_cl':>11} {'mean_other':>11} {'ratio':>7} "
            f"{'pct_in_cl':>10} {'pct_other':>10} {'rank/23':>8}")
        for g in gs:
            m_in = float(mean_by_cl.loc[c, g])
            others = mean_by_cl.drop(index=c)[g]
            m_out = float(others.mean())
            ratio = m_in / m_out if m_out > 0 else np.inf
            rank = int((mean_by_cl[g] > m_in).sum() + 1)
            f_in = 100 * float(frac_by_cl.loc[c, g])
            f_out = 100 * float(frac_by_cl.drop(index=c)[g].mean())
            log(f"{g:<10} {m_in:>11.3f} {m_out:>11.3f} {ratio:>7.2f} "
                f"{f_in:>9.1f}% {f_out:>9.1f}% {rank:>8}")
            records.append({"cluster": c, "gene": g, "mean_in_cluster": round(m_in, 4),
                            "mean_other_clusters": round(m_out, 4),
                            "ratio": round(float(ratio), 3),
                            "pct_cells_in_cluster": round(f_in, 2),
                            "pct_cells_other": round(f_out, 2), "rank_of_23": rank})
    pd.DataFrame(records).to_csv(out / "disputed_cluster_markers.csv", index=False)
    log("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
