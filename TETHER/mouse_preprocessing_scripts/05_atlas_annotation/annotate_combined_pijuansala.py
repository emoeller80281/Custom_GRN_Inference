"""Re-annotate the integrated mESC MuData against the Pijuan-Sala 2019 atlas taxonomy.

Replaces the per-sample labels (which were assigned independently per sample and are
therefore not comparable across samples) with one harmonised label set applied to the
integrated clusters.

Taxonomy: Pijuan-Sala et al., "A single-cell molecular map of mouse gastrulation and
early organogenesis", Nature 566:490-495 (2019), doi:10.1038/s41586-019-0933-9.
The atlas reports 37 populations; that count splits Blood progenitors into 1-2 and
Erythroid into 1-3. The 34 top-level names are used here, which is the finest level
this dataset's cluster count can support.

Marker genes are curated from the atlas and canonical mouse gastrulation literature --
they are not a verbatim copy of a supplementary marker table.

Both clusterings are annotated: ``leiden`` (RNA Harmony) and ``leiden_wnn`` (joint
RNA+ATAC), since the finer WNN partition can resolve types the RNA clustering merges.

Example:
    python annotate_combined_pijuansala.py \
        --h5mu data/processed/mESC/combined/mESC_combined.h5mu
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

warnings.filterwarnings("ignore")

# --- Pijuan-Sala 2019 atlas populations -----------------------------------
PIJUAN_SALA_MARKERS = {
    "Epiblast": ["Pou5f1", "Utf1", "Slc7a3", "Fgf5", "Sox2", "Nanog", "Pim2"],
    "Primitive Streak": ["T", "Fgf8", "Mixl1", "Eomes", "Evx1", "Wnt3"],
    "Caudal epiblast": ["T", "Cdx1", "Cdx2", "Wnt3a", "Nkx1-2", "Fgf8"],
    "PGC": ["Dppa3", "Prdm1", "Tfap2c", "Alpl", "Nanos3", "Ifitm3"],
    "Anterior Primitive Streak": ["Foxa2", "Gsc", "Chrd", "Lhx1", "Cer1", "Nodal"],
    "Notochord": ["Noto", "Shh", "Foxa2", "Chrd", "Nog", "T"],
    "Def. endoderm": ["Sox17", "Foxa2", "Cer1", "Cxcr4", "Hhex", "Krt19"],
    "Gut": ["Trh", "Epcam", "Foxa1", "Cldn6", "Krt8", "Krt18", "Shh"],
    "Nascent mesoderm": ["Mesp1", "Mesp2", "Lefty2", "Dll3", "Phlda2", "Snai1"],
    "Mixed mesoderm": ["T", "Mesp1", "Lhx1", "Pcdh19", "Evx1"],
    "Intermediate mesoderm": ["Pax2", "Pax8", "Lhx1", "Osr1", "Gata3", "Wt1"],
    "Caudal Mesoderm": ["Tbx6", "Dll1", "Hes7", "Msgn1", "Rspo3"],
    "Paraxial mesoderm": ["Tcf15", "Meox1", "Aldh1a2", "Pax3", "Tbx6"],
    "Somitic mesoderm": ["Meox1", "Pax3", "Tcf15", "Foxc2", "Uncx", "Ripply2"],
    "Pharyngeal mesoderm": ["Tbx1", "Isl1", "Pitx2", "Msc", "Tcf21"],
    "Cardiomyocytes": ["Nkx2-5", "Tnnt2", "Myl7", "Myl4", "Actc1", "Ttn"],
    "Allantois": ["Tbx4", "Hoxa10", "Hoxa11", "Hand1", "Vim", "Postn"],
    "ExE mesoderm": ["Bmp4", "Hand1", "Ahnak", "Pmp22", "Postn", "Lum"],
    "Mesenchyme": ["Pdgfra", "Col1a1", "Prrx1", "Twist1", "Snai2", "Postn"],
    "Haematoendothelial progenitors": ["Etv2", "Tal1", "Lmo2", "Kdr", "Runx1", "Cdh5"],
    "Endothelium": ["Cdh5", "Pecam1", "Kdr", "Tie1", "Tek", "Emcn"],
    "Blood progenitors": ["Runx1", "Gata1", "Gfi1b", "Itga2b", "Lmo2", "Tal1"],
    "Erythroid": ["Hba-x", "Hbb-y", "Hbb-bh1", "Gypa", "Klf1", "Alas2"],
    "NMP": ["T", "Sox2", "Cdx2", "Nkx1-2", "Fgf8", "Cyp26a1"],
    "Rostral neurectoderm": ["Six3", "Hesx1", "Otx2", "Foxg1", "Sox1", "Pax6"],
    "Caudal neurectoderm": ["Nkx1-2", "Cdx2", "Sox2", "Sox1", "Irx3"],
    "Neural crest": ["Sox10", "Foxd3", "Pax3", "Tfap2a", "Tfap2b", "Ets1"],
    "Forebrain/Midbrain/Hindbrain": ["Sox2", "Sox1", "Pax6", "En1", "Otx2", "Gbx2", "Hoxb1"],
    "Spinal cord": ["Sox2", "Sox1", "Pax6", "Hoxb9", "Irx3", "Nkx6-1"],
    "Surface ectoderm": ["Trp63", "Krt8", "Krt18", "Wnt6", "Grhl2", "Dlx5", "Krt19"],
    "Visceral endoderm": ["Ttr", "Apoa1", "Afp", "Rhox5", "Amn", "Cubn", "Apoe"],
    "ExE endoderm": ["Ttr", "Apoa1", "Apoe", "Rhox5", "Ctsh"],
    "ExE ectoderm": ["Tfap2c", "Elf5", "Ascl2", "Krt8", "Bmp4", "Cdx2"],
    "Parietal endoderm": ["Sparc", "Lama1", "Col4a1", "Plat", "Sox7"],
}

# Broad compartment per population, for the summary table and plot colours.
COMPARTMENT = {
    "Epiblast": "Pluripotent", "Primitive Streak": "Pluripotent",
    "Caudal epiblast": "Pluripotent", "PGC": "Germline",
    "Anterior Primitive Streak": "Endoderm", "Notochord": "Axial",
    "Def. endoderm": "Endoderm", "Gut": "Endoderm",
    "Nascent mesoderm": "Mesoderm", "Mixed mesoderm": "Mesoderm",
    "Intermediate mesoderm": "Mesoderm", "Caudal Mesoderm": "Mesoderm",
    "Paraxial mesoderm": "Mesoderm", "Somitic mesoderm": "Mesoderm",
    "Pharyngeal mesoderm": "Mesoderm", "Cardiomyocytes": "Mesoderm",
    "Allantois": "Extraembryonic", "ExE mesoderm": "Extraembryonic",
    "Mesenchyme": "Mesoderm",
    "Haematoendothelial progenitors": "Blood", "Endothelium": "Blood",
    "Blood progenitors": "Blood", "Erythroid": "Blood",
    "NMP": "Ectoderm", "Rostral neurectoderm": "Ectoderm",
    "Caudal neurectoderm": "Ectoderm", "Neural crest": "Ectoderm",
    "Forebrain/Midbrain/Hindbrain": "Ectoderm", "Spinal cord": "Ectoderm",
    "Surface ectoderm": "Ectoderm",
    "Visceral endoderm": "Extraembryonic", "ExE endoderm": "Extraembryonic",
    "ExE ectoderm": "Extraembryonic", "Parietal endoderm": "Extraembryonic",
}


def log(m):
    print(f"[annotate_combined] {m}", flush=True)


def annotate(rna, groupby, panel_present, out, tag, margin_min=0.5):
    """Assign one atlas population per cluster from z-scored signature means."""
    import scanpy as sc  # noqa: F401  (scores already computed by caller)

    score_cols = [f"ps_score_{ct}" for ct in panel_present]
    per_cluster = rna.obs.groupby(groupby, observed=True)[score_cols].mean()
    per_cluster.columns = list(panel_present)

    # Z-score down each signature so a broad panel that sits uniformly high cannot
    # win every cluster; the assignment is then the row-wise argmax.
    z = (per_cluster - per_cluster.mean(axis=0)) / per_cluster.std(axis=0).replace(0, np.nan)

    rows = []
    for cl in z.index:
        r = z.loc[cl].dropna().sort_values(ascending=False)
        counts = int((rna.obs[groupby] == cl).sum())
        rows.append({
            "cluster": str(cl), "n_cells": counts,
            "cell_type": r.index[0], "z_score": round(float(r.iloc[0]), 3),
            "runner_up": r.index[1] if len(r) > 1 else "",
            "margin_over_runner_up": round(float(r.iloc[0] - r.iloc[1]), 3) if len(r) > 1 else np.nan,
            "third": r.index[2] if len(r) > 2 else "",
            "compartment": COMPARTMENT.get(r.index[0], "?"),
        })
    ann = pd.DataFrame(rows)
    ann["confident"] = ann["margin_over_runner_up"] >= margin_min
    ann.to_csv(out / f"pijuansala_annotation_{tag}.csv", index=False)
    per_cluster.round(4).to_csv(out / f"pijuansala_scores_raw_{tag}.csv")
    z.round(4).to_csv(out / f"pijuansala_scores_zscored_{tag}.csv")

    mapping = dict(zip(ann["cluster"], ann["cell_type"]))
    labels = rna.obs[groupby].astype(str).map(mapping)
    low = set(ann.loc[~ann["confident"], "cluster"])

    log(f"\n--- {tag} ({groupby}): {len(ann)} clusters -> "
        f"{ann['cell_type'].nunique()} distinct populations ---")
    for _, r in ann.sort_values("n_cells", ascending=False).iterrows():
        flag = "" if r["confident"] else f"   <- LOW (vs {r['runner_up']})"
        log(f"  cl {r['cluster']:>3} (n={r['n_cells']:>6}): {r['cell_type']:<32} "
            f"z={r['z_score']:.2f} margin={r['margin_over_runner_up']:.2f}{flag}")
    n_low = int(rna.obs[groupby].astype(str).isin(low).sum())
    log(f"  low-confidence: {len(low)}/{len(ann)} clusters, {n_low} cells "
        f"({100*n_low/rna.n_obs:.1f}%)")
    return ann, z, labels, low


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5mu", default="data/processed/mESC/combined/mESC_combined.h5mu")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--margin_min", type=float, default=0.5)
    p.add_argument("--no_write_h5mu", action="store_true")
    a = p.parse_args()

    import muon as mu
    import scanpy as sc

    h5mu = Path(a.h5mu)
    out = Path(a.out_dir) if a.out_dir else h5mu.parent
    out.mkdir(parents=True, exist_ok=True)

    log(f"reading {h5mu}")
    mdata = mu.read(str(h5mu))
    rna = mdata["rna"]
    log(f"RNA: {rna.n_obs} cells x {rna.n_vars} genes")

    # --- marker coverage ---------------------------------------------------
    present, coverage = {}, {}
    for ct, genes in PIJUAN_SALA_MARKERS.items():
        found = [g for g in genes if g in rna.var_names]
        coverage[ct] = {"found": len(found), "total": len(genes),
                        "missing": [g for g in genes if g not in rna.var_names]}
        if len(found) < 2:
            log(f"  SKIP {ct}: only {len(found)}/{len(genes)} markers present")
            continue
        present[ct] = found
    log(f"Scoring {len(present)}/{len(PIJUAN_SALA_MARKERS)} atlas populations")
    for ct, g in coverage.items():
        if g["missing"]:
            log(f"  {ct:<32} {g['found']}/{g['total']}  missing: {', '.join(g['missing'])}")

    for ct, genes in present.items():
        sc.tl.score_genes(rna, genes, score_name=f"ps_score_{ct}", use_raw=False)

    # --- annotate both clusterings ----------------------------------------
    results = {}
    if "leiden_wnn" in mdata.obs and "leiden_wnn" not in rna.obs:
        rna.obs["leiden_wnn"] = mdata.obs.loc[rna.obs_names, "leiden_wnn"].values

    for groupby, tag in [("leiden", "rna_leiden"), ("leiden_wnn", "wnn_leiden")]:
        if groupby not in rna.obs:
            log(f"skipping {groupby}: not present")
            continue
        ann, z, labels, low = annotate(rna, groupby, present, out, tag, a.margin_min)
        col = "celltype_pijuansala" if groupby == "leiden" else "celltype_pijuansala_wnn"
        rna.obs[col] = pd.Categorical(labels)
        rna.obs[col + "_confident"] = ~rna.obs[groupby].astype(str).isin(low)
        mdata.obs[col] = rna.obs[col].values
        results[tag] = {
            "groupby": groupby, "n_clusters": int(len(ann)),
            "n_populations": int(ann["cell_type"].nunique()),
            "populations": sorted(ann["cell_type"].unique().tolist()),
            "low_conf_clusters": int((~ann["confident"]).sum()),
        }

    # --- composition tables -----------------------------------------------
    main_col = "celltype_pijuansala"
    comp = pd.crosstab(rna.obs["timepoint"], rna.obs[main_col])
    comp.to_csv(out / "pijuansala_timepoint_composition.csv")
    (pd.crosstab(rna.obs["sample"], rna.obs[main_col])
     .to_csv(out / "pijuansala_sample_composition.csv"))
    # Old labels vs new, to show what the harmonised annotation changed.
    (pd.crosstab(rna.obs["per_sample_cell_type"], rna.obs[main_col])
     .to_csv(out / "pijuansala_vs_per_sample_labels.csv"))

    log("\n=== composition by timepoint (% of timepoint) ===")
    log((comp.div(comp.sum(axis=1), axis=0) * 100).round(1).to_string())

    # --- figures -----------------------------------------------------------
    sc.settings.figdir = out
    for col, fn, title in [
        (main_col, "umap_pijuansala_rna_leiden.png", "Pijuan-Sala populations (RNA Leiden)"),
        ("celltype_pijuansala_wnn", "umap_pijuansala_wnn.png",
         "Pijuan-Sala populations (WNN Leiden)"),
    ]:
        if col not in rna.obs:
            continue
        fig, ax = plt.subplots(figsize=(11, 8))
        sc.pl.umap(rna, color=col, ax=ax, show=False, frameon=True, title=title,
                   legend_fontsize=8)
        for sp_ in ax.spines.values():
            sp_.set_visible(True); sp_.set_color("#3A424C"); sp_.set_linewidth(1.1)
        fig.tight_layout()
        fig.savefig(out / fn, dpi=150, bbox_inches="tight")
        plt.close(fig)

    z = pd.read_csv(out / "pijuansala_scores_zscored_rna_leiden.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(0.42 * z.shape[1] + 5, 0.34 * z.shape[0] + 3))
    im = ax.imshow(z.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(z.shape[1])); ax.set_xticklabels(z.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(z.shape[0]))
    ax.set_yticklabels([f"cluster {c}" for c in z.index], fontsize=8)
    ax.set_title("Pijuan-Sala signature scores (z-scored across integrated clusters)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="z-score")
    fig.tight_layout()
    fig.savefig(out / "heatmap_pijuansala_signatures.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "reference": {
            "citation": "Pijuan-Sala et al. Nature 566:490-495 (2019)",
            "doi": "10.1038/s41586-019-0933-9",
            "n_populations_in_atlas": 37,
            "n_top_level_names_used": len(PIJUAN_SALA_MARKERS),
            "note": ("Atlas reports 37 populations; that count splits Blood progenitors "
                     "into 1-2 and Erythroid into 1-3. Marker genes curated from the "
                     "atlas and canonical literature, not copied from a supplementary table."),
        },
        "marker_coverage": coverage,
        "n_scored": len(present),
        "results": results,
    }
    (out / "pijuansala_annotation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    if not a.no_write_h5mu:
        log(f"rewriting {h5mu} with the new labels ...")
        mdata.write(str(h5mu))
        log("h5mu updated")
    log("Done.")


if __name__ == "__main__":
    sys.exit(main())
