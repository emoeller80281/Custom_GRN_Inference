"""Independent marker check on the centroid-correlation atlas labels.

The centroid annotation ranks clusters by correlation over 2000 reference HVGs. This
script asks a different question, one the annotation never optimised for: for each
cluster, is the *canonical* marker of the type it was assigned actually enriched in that
cluster relative to every other cluster?

That is the test that caught the two previous attempts -- marker scoring called 2366 PGCs
while Dppa3 and Nanos3 were flat zero, and kNN transfer called Pax6-high neurectoderm
"Pharyngeal mesoderm". A label is only reported as supported when its own markers rank the
assigned cluster near the top.

Scoring, per (cluster, assigned type):
  - specificity z: mean panel expression in the cluster, z-scored across all clusters
  - rank: where the cluster sits among all clusters for that panel (1 = highest)
A call passes when z >= 1.0 and the cluster is in the top 3 for its own panel.

Markers are canonical mouse gastrulation genes, chosen independently of the atlas.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Canonical markers per Pijuan-Sala population. Deliberately small and specific: these
# are the genes a developmental biologist would demand to see, not the top HVGs.
CANONICAL = {
    "Epiblast":                     ["Pou5f1", "Utf1", "Slc7a3"],
    "Primitive Streak":             ["T", "Fgf8", "Mixl1", "Eomes"],
    "Anterior Primitive Streak":    ["Foxa2", "Gsc", "Chrd"],
    "Caudal epiblast":              ["T", "Cdx1", "Fgf8"],
    "PGC":                          ["Dppa3", "Nanos3", "Tfap2c"],
    "Notochord":                    ["Noto", "Shh", "Foxa2", "T"],
    "Def. endoderm":                ["Foxa2", "Sox17", "Cer1"],
    "Gut":                          ["Foxa1", "Foxa2", "Sox17", "Trh"],
    "Nascent mesoderm":             ["Mesp1", "T", "Lefty2"],
    "Mixed mesoderm":               ["Mesp1", "T", "Pdgfra"],
    "Intermediate mesoderm":        ["Pax2", "Osr1", "Wt1"],
    "Caudal Mesoderm":              ["T", "Tbx6", "Cdx2"],
    "Paraxial mesoderm":            ["Tcf15", "Meox1", "Pdgfra"],
    "Somitic mesoderm":             ["Meox1", "Pax3", "Tcf15"],
    "Pharyngeal mesoderm":          ["Tbx1", "Isl1", "Pitx2"],
    "Cardiomyocytes":               ["Tnnt2", "Myl7", "Nkx2-5", "Actc1"],
    "Allantois":                    ["Tbx4", "Hoxa10", "Hand1"],
    "ExE mesoderm":                 ["Hand1", "Bmp4", "Postn"],
    "Mesenchyme":                   ["Pdgfra", "Prrx1", "Twist1"],
    "Haematoendothelial progenitors": ["Kdr", "Etv2", "Tal1"],
    "Endothelium":                  ["Pecam1", "Cdh5", "Kdr", "Tie1"],
    "Blood progenitors 1":          ["Runx1", "Tal1", "Lmo2"],
    "Blood progenitors 2":          ["Runx1", "Gata1", "Lmo2"],
    "Erythroid1":                   ["Hba-x", "Hbb-bh1", "Gata1"],
    "Erythroid2":                   ["Hbb-bh1", "Hba-a1", "Klf1"],
    "Erythroid3":                   ["Hba-a1", "Hbb-y", "Alas2"],
    "NMP":                          ["T", "Sox2", "Cdx2", "Nkx1-2"],
    "Rostral neurectoderm":         ["Six3", "Otx2", "Sox2"],
    "Caudal neurectoderm":          ["Sox2", "Nkx1-2", "Cdx2"],
    "Neural crest":                 ["Sox10", "Foxd3", "Tfap2a"],
    "Forebrain/Midbrain/Hindbrain": ["Pax6", "Sox2", "Otx2", "En1"],
    "Spinal cord":                  ["Sox2", "Pax6", "Hoxb9", "Irx3"],
    "Surface ectoderm":             ["Krt8", "Krt18", "Trp63", "Grhl3"],
    "Visceral endoderm":            ["Ttr", "Afp", "Cubn", "Apoa1"],
    "ExE endoderm":                 ["Ttr", "Apoa1", "Rhox5"],
    "ExE ectoderm":                 ["Elf5", "Tfap2c", "Krt8"],
    "Parietal endoderm":            ["Lama1", "Sparc", "Plat"],
}


def log(m):
    print(f"[validate] {m}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5mu", default="data/processed/mESC/combined/mESC_combined.h5mu")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--label_col", default="celltype_atlas_rna_leiden")
    p.add_argument("--cluster_col", default="leiden")
    p.add_argument("--min_z", type=float, default=1.0)
    p.add_argument("--max_rank", type=int, default=3)
    a = p.parse_args()

    import anndata as ad
    import muon as mu
    import scanpy as sc

    out = Path(a.out_dir) if a.out_dir else Path(a.h5mu).parent
    out.mkdir(parents=True, exist_ok=True)

    log("loading query ...")
    mdata = mu.read(a.h5mu)
    q = mdata["rna"]

    # Re-normalise from raw counts so the check does not inherit any scaling the
    # annotation pipeline applied.
    adata = ad.AnnData(X=q.layers["counts"].copy(),
                       obs=q.obs[[a.cluster_col, a.label_col]].copy(),
                       var=pd.DataFrame(index=q.var_names.copy()))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    clusters = adata.obs[a.cluster_col].astype(str)
    assigned = adata.obs[a.label_col].astype(str)
    cl_order = sorted(clusters.unique(), key=lambda x: int(x) if x.isdigit() else x)

    # Mean expression per cluster for every gene in any panel.
    panel_genes = sorted({g for gs in CANONICAL.values() for g in gs})
    present = [g for g in panel_genes if g in adata.var_names]
    missing = sorted(set(panel_genes) - set(present))
    log(f"panel genes present: {len(present)}/{len(panel_genes)}")
    if missing:
        log(f"absent from data: {', '.join(missing)}")

    X = adata[:, present].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    expr = pd.DataFrame(X, columns=present)
    expr["_cl"] = clusters.values
    cl_mean = expr.groupby("_cl").mean().loc[cl_order]

    # Per-panel score for each cluster: mean of z-scored marker expression.
    panel_scores = {}
    for ct, genes in CANONICAL.items():
        gs = [g for g in genes if g in cl_mean.columns]
        if not gs:
            continue
        z = (cl_mean[gs] - cl_mean[gs].mean()) / cl_mean[gs].std(ddof=0).replace(0, np.nan)
        panel_scores[ct] = z.mean(axis=1)
    score = pd.DataFrame(panel_scores)          # clusters x cell types
    score.to_csv(out / "validation_panel_scores.csv")

    # Evaluate each cluster's own assignment.
    cl_to_label = assigned.groupby(clusters).agg(lambda s: s.value_counts().index[0])
    n_per = clusters.value_counts()
    rows = []
    for cl in cl_order:
        ct = cl_to_label[cl]
        if ct not in score.columns:
            rows.append({"cluster": cl, "n_cells": int(n_per[cl]), "assigned": ct,
                         "z": np.nan, "rank": np.nan, "supported": False,
                         "note": "no canonical panel"})
            continue
        col = score[ct]
        z = float(col[cl])
        rank = int((col > z).sum() + 1)
        best = col.idxmax()
        rows.append({
            "cluster": cl, "n_cells": int(n_per[cl]), "assigned": ct,
            "z": round(z, 2), "rank": rank,
            "supported": bool(z >= a.min_z and rank <= a.max_rank),
            "top_cluster_for_panel": best,
            "best_panel_for_cluster": score.loc[cl].idxmax(),
            "best_panel_z": round(float(score.loc[cl].max()), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "validation_centroid_labels.csv", index=False)

    n_ok = int(df["supported"].sum())
    cells_ok = int(df.loc[df["supported"], "n_cells"].sum())
    tot = int(df["n_cells"].sum())
    log(f"\n--- marker support for {a.label_col} ---")
    for _, r in df.sort_values("n_cells", ascending=False).iterrows():
        mark = "OK " if r["supported"] else "?? "
        extra = "" if r["supported"] else f"  (best panel here: {r.get('best_panel_for_cluster')} z={r.get('best_panel_z')})"
        log(f"  {mark}cl {r['cluster']:>3} (n={r['n_cells']:>6}) {r['assigned']:<32} "
            f"z={r['z']:>5.2f} rank={r['rank']}{extra}")
    log(f"\nsupported: {n_ok}/{len(df)} clusters, {cells_ok}/{tot} cells "
        f"({100*cells_ok/tot:.1f}%)")

    # Heatmap: clusters x panels, with the assignment marked.
    fig, ax = plt.subplots(figsize=(0.34 * score.shape[1] + 6, 0.36 * score.shape[0] + 3))
    im = ax.imshow(score.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(score.shape[1]))
    ax.set_xticklabels(score.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(score.shape[0]))
    ax.set_yticklabels([f"cl {c} -> {cl_to_label[c]}" for c in score.index], fontsize=7)
    for i, cl in enumerate(score.index):
        ct = cl_to_label[cl]
        if ct in score.columns:
            j = list(score.columns).index(ct)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="black", lw=1.8))
    ax.set_title("Canonical marker z-score per cluster (box = centroid assignment)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="mean marker z")
    fig.tight_layout()
    fig.savefig(out / "heatmap_marker_validation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (out / "validation_summary.json").write_text(json.dumps({
        "label_col": a.label_col,
        "criteria": {"min_z": a.min_z, "max_rank": a.max_rank},
        "n_clusters": int(len(df)), "n_supported": n_ok,
        "cells_supported": cells_ok, "cells_total": tot,
        "pct_cells_supported": round(100 * cells_ok / tot, 1),
        "unsupported": df.loc[~df["supported"], "cluster"].tolist(),
        "genes_absent": missing,
    }, indent=2, default=str))
    log("Done.")


if __name__ == "__main__":
    sys.exit(main())
