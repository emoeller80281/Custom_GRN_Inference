"""Final reconciled atlas labels for the integrated mESC/gastrulation object.

Combines the centroid-correlation annotation with the canonical-marker check and, for the
clusters where those disagreed, with the raw per-gene evidence in
`disputed_cluster_markers.csv`. Writes `celltype_final` plus a `celltype_final_support`
column recording how each call is justified, so downstream users can filter on evidence
rather than trusting a bare label.

Provenance of each call:

  18 of 23 clusters -- centroid label confirmed by its own canonical markers
  (validate_centroid_labels.py: z >= 1.0 and top-3 rank). Kept as-is.

  Three clusters overridden on decisive marker evidence (ratios are cluster mean vs mean
  of all other clusters; rank is out of 23 clusters):

    cl2  Somitic mesoderm -> Caudal Mesoderm
         Tbx6 14.78x rank 1, Aldh1a2 4.61x, Hoxb9 3.27x, T 2.43x. Tbx6 is the defining
         presomitic marker. Intermediate mesoderm excluded (Pax2 0.27x, Wt1 0.38x).
         Somitic markers are present but weaker (Meox1 2.24x, Tcf15 1.99x, both rank 2),
         consistent with a presomitic-to-somitic continuum; named for its dominant signal.

    cl18 Mesenchyme -> ExE mesoderm
         Postn 37.12x rank 1, Bmp4 5.44x rank 1, Hand1 5.16x rank 1, Ahnak 4.47x.
         Mesenchyme markers fail (Prrx1 0.42x rank 14, Twist1 1.14x rank 6).
         Allantois excluded (Tbx4 0.32x, Hoxa10 0.68x).

    cl21 Mesenchyme -> Surface ectoderm
         Krt8 2.35x rank 1, Krt18 2.80x rank 1, Trp63 2.46x rank 2, Grhl3 2.02x rank 3.
         Mesenchyme markers depleted (Pdgfra 0.53x, Prrx1 0.51x, Twist1 0.70x).
         Visceral endoderm excluded despite high absolute Ttr: Ttr 0.99x, Afp 0.88x,
         Apoa1 0.94x are all at background across clusters.

  Two clusters marked unresolved rather than assigned a population:

    cl0  (4468 cells, largest cluster) -- depth-driven, not a cell type.
         Median 1439 genes / 2277 counts vs 3159-5166 for every other cluster except
         cl13; mitochondrial fraction is LOW (3.1%), so this is low library complexity,
         not dying cells. It is 46% E7.75_rep1 + 31% E8.5_rep2 and near-absent from E8.0
         and E8.75 -- a sample signature, not a developmental one. Both centroid
         candidates are contradicted: Surface ectoderm markers are depleted (Krt8 0.74x,
         Krt18 0.75x, Trp63 0.63x, Wnt6 0.42x) and Gut is not enriched (Foxa1 0.62x,
         Sox17 0.88x). Real epiblast/early-streak signal is present (Pou5f1 3.26x,
         Nanog 3.19x, Mixl1 3.34x, Utf1 2.84x), but epiblast does not exist at E8.5, so
         the cluster is heterogeneous and no single population fits it.

    cl13 (156 cells) -- not blood progenitors.
         Runx1 8.02x is the only support; the rest of the haematopoietic program is
         absent or bottom-ranked: Cd34 0.00 (rank 23), Klf1 0.00 (rank 23), Kit 0.20x
         (rank 21), Tal1 0.34x, Gata1 0.54x. Lowest complexity of any cluster
         (1291 genes / 2071 counts).

Reference: Pijuan-Sala et al., Nature 566:490-495 (2019), doi:10.1038/s41586-019-0933-9.
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

# cluster -> (final label, support tier, one-line justification)
OVERRIDES = {
    "2":  ("Caudal Mesoderm", "marker_override",
           "Tbx6 14.8x rank1, Aldh1a2 4.6x, Hoxb9 3.3x; Pax2/Wt1 depleted"),
    "18": ("ExE mesoderm", "marker_override",
           "Postn 37x rank1, Bmp4 5.4x rank1, Hand1 5.2x rank1; Prrx1 0.42x rank14"),
    "21": ("Surface ectoderm", "marker_override",
           "Krt8 2.4x rank1, Krt18 2.8x rank1, Trp63 2.5x; Pdgfra/Prrx1 depleted"),
    "0":  ("Unresolved (low complexity)", "unresolved",
           "1439 median genes vs 3159-5166 elsewhere; 46% E7.75_rep1 + 31% E8.5_rep2; "
           "both centroid candidates contradicted; epiblast-like but spans E7.75-E8.5"),
    "13": ("Unresolved (low complexity)", "unresolved",
           "Runx1 8x but Cd34 0.00 rank23, Klf1 0.00 rank23, Kit rank21, Tal1 0.34x; "
           "lowest complexity of any cluster"),
}


def log(m):
    print(f"[reconcile] {m}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5mu", default="data/processed/mESC/combined/mESC_combined.h5mu")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--cluster_col", default="leiden")
    p.add_argument("--centroid_col", default="celltype_atlas_rna_leiden")
    p.add_argument("--no_write_h5mu", action="store_true")
    a = p.parse_args()

    import muon as mu
    import scanpy as sc

    out = Path(a.out_dir) if a.out_dir else Path(a.h5mu).parent
    val_path = out / "validation_centroid_labels.csv"
    val = pd.read_csv(val_path, dtype={"cluster": str}).set_index("cluster")
    log(f"loaded validation for {len(val)} clusters")

    log("loading h5mu ...")
    mdata = mu.read(a.h5mu)
    r = mdata["rna"]
    cl = r.obs[a.cluster_col].astype(str)
    centroid = r.obs[a.centroid_col].astype(str)

    cl_to_centroid = centroid.groupby(cl).agg(lambda s: s.value_counts().index[0])

    rows = []
    for c in sorted(cl_to_centroid.index, key=lambda x: int(x) if x.isdigit() else x):
        base = cl_to_centroid[c]
        supported = bool(val.loc[c, "supported"]) if c in val.index else False
        if c in OVERRIDES:
            final, tier, why = OVERRIDES[c]
        elif supported:
            final, tier, why = base, "centroid+marker", (
                f"canonical markers z={val.loc[c,'z']} rank={val.loc[c,'rank']}")
        else:
            # Any cluster that failed validation must be listed in OVERRIDES; if a future
            # rerun changes the clustering, fail loudly rather than emit a silent label.
            raise RuntimeError(
                f"cluster {c} failed marker validation but has no adjudicated override. "
                "Re-run inspect_unresolved_clusters.py and add a decision.")
        rows.append({"cluster": c, "n_cells": int((cl == c).sum()),
                     "centroid_label": base, "final_label": final,
                     "support": tier, "evidence": why})
    df = pd.DataFrame(rows)
    df.to_csv(out / "final_atlas_labels.csv", index=False)

    mapping = dict(zip(df["cluster"], df["final_label"]))
    tiers = dict(zip(df["cluster"], df["support"]))
    r.obs["celltype_final"] = pd.Categorical(cl.map(mapping).values)
    r.obs["celltype_final_support"] = pd.Categorical(cl.map(tiers).values)
    mdata.obs["celltype_final"] = r.obs["celltype_final"].values
    mdata.obs["celltype_final_support"] = r.obs["celltype_final_support"].values

    n_by_tier = df.groupby("support")["n_cells"].sum()
    tot = int(df["n_cells"].sum())
    log("\n--- final labels ---")
    for _, x in df.sort_values("n_cells", ascending=False).iterrows():
        chg = "" if x["final_label"] == x["centroid_label"] else f"  (was: {x['centroid_label']})"
        log(f"  cl {x['cluster']:>3} (n={x['n_cells']:>6}) {x['final_label']:<30} "
            f"[{x['support']}]{chg}")
    log("\ncells by support tier:")
    for t, n in n_by_tier.items():
        log(f"  {t:<18} {n:>6}  ({100*n/tot:.1f}%)")

    # Composition by timepoint, excluding unresolved cells.
    comp = pd.crosstab(r.obs["celltype_final"].astype(str),
                       r.obs["timepoint"].astype(str), normalize="columns") * 100
    comp = comp[sorted(comp.columns)]
    comp.round(2).to_csv(out / "final_timepoint_composition.csv")
    log("\n=== % of each timepoint (final labels) ===")
    log("\n" + comp.round(1).to_string())

    for basis_col, fname in [("celltype_final", "umap_final_labels.png"),
                             ("celltype_final_support", "umap_final_support.png")]:
        fig, ax = plt.subplots(figsize=(11, 8))
        sc.pl.umap(r, color=basis_col, ax=ax, show=False, frameon=True,
                   legend_fontsize=8, title=basis_col)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#3A424C"); s.set_linewidth(1.1)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    (out / "final_labels_summary.json").write_text(json.dumps({
        "reference": {"citation": "Pijuan-Sala et al. Nature 566:490-495 (2019)",
                      "doi": "10.1038/s41586-019-0933-9"},
        "method": "centroid Spearman correlation, validated against canonical markers, "
                  "disagreements adjudicated on raw per-gene evidence",
        "n_clusters": int(len(df)),
        "n_populations": int(df.loc[df["support"] != "unresolved", "final_label"].nunique()),
        "cells_by_tier": {k: int(v) for k, v in n_by_tier.items()},
        "cells_total": tot,
        "pct_confidently_labelled": round(
            100 * (tot - int(n_by_tier.get("unresolved", 0))) / tot, 1),
        "overrides": {k: v[0] for k, v in OVERRIDES.items()},
    }, indent=2, default=str))

    if not a.no_write_h5mu:
        # Orphaned per-cell metadata from the rejected kNN transfer. Its labels were
        # already removed; a "confidence" column with no labels attached is worse than
        # useless, since it reads as if it qualifies the surviving annotation.
        stale = [c for c in r.obs.columns if c.startswith("transfer_")]
        for c in stale:
            del r.obs[c]
            if c in mdata.obs:
                del mdata.obs[c]
        if stale:
            log(f"dropped orphaned kNN metadata: {', '.join(stale)}")

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from common.mudata_utils import sanitize_for_h5
        for m in mdata.mod.values():
            sanitize_for_h5(m)
        log("\nrewriting h5mu ...")
        mdata.write(a.h5mu)
        log("h5mu updated")
    log("Done.")


if __name__ == "__main__":
    sys.exit(main())
