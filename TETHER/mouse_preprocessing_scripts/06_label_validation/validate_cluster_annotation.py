"""Tier-1 manual validation of a single cluster's annotation, per the annotation guide.

The centroid + canonical-panel workflow validated cluster labels using *positive* markers
only, scored on cluster means. The single-cell annotation guide requires more than that for
a cluster this size, and each additional requirement is implemented here:

* **Negative markers** (guide pitfall 1). A label needs the markers of competing identities
  to be *absent*, not just its own to be present. Cluster 1's runner-up was ExE endoderm at
  rho 0.757, so the visceral/ExE endoderm program is the specific thing that must be absent.

* **Co-expression in the same cells** (guide pitfall 1). A cluster mean cannot distinguish
  "every cell expresses Lama1 and Sparc" from "half the cells express one and half the
  other". The fraction of cells co-expressing >=2 positive markers is computed directly.

* **Small-cluster scrutiny** (guide pitfall 7). Clusters under ~500 cells need manual marker
  validation regardless of automated confidence. Cluster 1 has 543.

* **Cell-cycle confounding** (guide pitfall 6). A cluster defined by proliferation genes
  rather than lineage should be named "<type> proliferating", not treated as a lineage.

* **Batch and composition sanity** (guide pitfall 4, best practice 8). Per-sample purity and
  the timepoint profile are reported; a label whose abundance is non-monotonic in
  developmental time deserves an explanation.

Verdict follows the guide: >=2 positive markers co-expressed AND negatives absent.
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

warnings.filterwarnings("ignore")

# Tirosh et al. 2016 cell-cycle genes, mouse-cased.
S_GENES = """Mcm5 Pcna Tyms Fen1 Mcm2 Mcm4 Rrm1 Ung Gins2 Mcm6 Cdca7 Dtl Prim1 Uhrf1
Hells Rfc2 Rpa2 Nasp Rad51ap1 Gmnn Wdr76 Slbp Ccne2 Ubr7 Pold3 Msh2 Atad2 Rad51 Rrm2
Cdc45 Cdc6 Exo1 Tipin Dscc1 Blm Casp8ap2 Usp1 Clspn Pola1 Chaf1b Brip1 E2f8""".split()
G2M_GENES = """Hmgb2 Cdk1 Nusap1 Ube2c Birc5 Tpx2 Top2a Ndc80 Cks2 Nuf2 Cks1b Mki67 Tmpo
Cenpf Tacc3 Smc4 Ccnb2 Ckap2l Ckap2 Aurkb Bub1 Kif11 Anp32e Tubb4b Gtse1 Kif20b Hjurp
Cdca3 Cdc20 Ttk Cdc25c Kif2c Rangap1 Ncapd2 Dlgap5 Cdca2 Cdca8 Ect2 Kif23 Hmmr Aurka
Psrc1 Anln Lbr Ckap5 Cenpe Ctcf Nek2 G2e3 Gas2l3 Cbx5 Cenpa""".split()

# Positive markers for the assigned label, and the negatives that must be absent.
# Negatives are chosen to exclude the specific competing identities, not arbitrary genes.
PANELS = {
    "Parietal endoderm": {
        "positive": ["Lama1", "Lamb1", "Sparc", "Plat", "Dab2", "Col4a1", "Snai1",
                     "Sox7", "Thbd"],
        "negative": {
            # Runner-up identity: visceral / ExE endoderm.
            "ExE/visceral endoderm": ["Afp", "Ttr", "Apoa1", "Apoa4", "Cubn", "Amn", "Hnf4a"],
            # Other lineages that would indicate contamination or doublets.
            "epiblast": ["Pou5f1", "Nanog"],
            "mesoderm": ["T", "Mesp1"],
            "blood/endothelium": ["Hbb-bh1", "Pecam1"],
        },
    },
}


def log(m):
    print(f"[cluster_qc] {m}", flush=True)


def dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5mu", default="data/processed/mESC/combined/mESC_combined.h5mu")
    p.add_argument("--cluster", default="1")
    p.add_argument("--cluster_col", default="leiden")
    p.add_argument("--label", default=None,
                   help="Assigned label; defaults to the cluster's celltype_final")
    p.add_argument("--compare_to", default=None,
                   help="Cluster id of the runner-up identity for a direct DE contrast")
    p.add_argument("--out_dir", default=None)
    a = p.parse_args()

    import anndata as ad
    import muon as mu
    import scanpy as sc

    sc.settings.n_jobs = 8
    sc.settings.verbosity = 1
    out = Path(a.out_dir) if a.out_dir else Path(a.h5mu).parent / f"cluster{a.cluster}_qc"
    out.mkdir(parents=True, exist_ok=True)

    log("loading ...")
    mdata = mu.read(a.h5mu)
    r = mdata["rna"]
    keep = [c for c in [a.cluster_col, "sample", "timepoint", "celltype_final",
                        "n_genes_by_counts", "total_counts", "pct_counts_mt",
                        "pct_counts_ribo", "doublet_score"] if c in r.obs.columns]
    adata = ad.AnnData(X=r.layers["counts"].copy(), obs=r.obs[keep].copy(),
                       var=pd.DataFrame(index=r.var_names.copy()))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cl = adata.obs[a.cluster_col].astype(str)
    m = (cl == str(a.cluster)).values
    n_in = int(m.sum())
    label = a.label or str(adata.obs.loc[m, "celltype_final"].mode().iloc[0])
    log(f"cluster {a.cluster}: {n_in} cells, assigned '{label}'")
    if label not in PANELS:
        raise SystemExit(f"No positive/negative panel defined for '{label}'. "
                         f"Add one to PANELS before validating it.")
    panel = PANELS[label]

    def stats(gene):
        if gene not in adata.var_names:
            return None
        v = dense(adata[:, gene].X).ravel()
        det = v > 0
        mean_in, mean_out = float(v[m].mean()), float(v[~m].mean())
        per_cl = pd.Series(v).groupby(cl.values).mean()
        return {
            "gene": gene,
            "mean_in": round(mean_in, 3), "mean_other": round(mean_out, 3),
            "ratio": round(mean_in / mean_out, 2) if mean_out > 0 else np.inf,
            "pct_in": round(100 * det[m].mean(), 1),
            "pct_other": round(100 * det[~m].mean(), 1),
            "rank_of_n": int((per_cl > per_cl[str(a.cluster)]).sum() + 1),
            "n_clusters": int(per_cl.size),
        }

    # --- positive markers ----------------------------------------------------
    pos = [s for s in (stats(g) for g in panel["positive"]) if s]
    pos_df = pd.DataFrame(pos)
    log(f"\n=== POSITIVE markers for '{label}' ===")
    log(pos_df.to_string(index=False))
    # Guide: a marker supports the label if it is enriched here and this cluster ranks top.
    pos_df["supports"] = (pos_df["ratio"] > 1.5) & (pos_df["rank_of_n"] <= 3)
    n_pos_ok = int(pos_df["supports"].sum())

    # --- co-expression in the same cells ------------------------------------
    good = pos_df.loc[pos_df["supports"], "gene"].tolist()
    if good:
        M = dense(adata[:, good].X) > 0
        n_pos_per_cell = M.sum(axis=1)
        coexpr_in = float((n_pos_per_cell[m] >= 2).mean())
        coexpr_out = float((n_pos_per_cell[~m] >= 2).mean())
    else:
        coexpr_in = coexpr_out = float("nan")
    log(f"\nco-expression of >=2 supporting positive markers: "
        f"{coexpr_in:.1%} of cluster cells vs {coexpr_out:.1%} elsewhere")

    # --- negative markers ----------------------------------------------------
    neg_rows, neg_fail = [], []
    log(f"\n=== NEGATIVE markers (must be ABSENT) ===")
    for group, genes in panel["negative"].items():
        for s in (stats(g) for g in genes):
            if not s:
                continue
            s["group"] = group
            # A negative fails if it is enriched in this cluster relative to the rest.
            s["violates"] = bool(s["ratio"] > 1.5 and s["rank_of_n"] <= 3)
            neg_rows.append(s)
            if s["violates"]:
                neg_fail.append(f"{s['gene']} ({group})")
    neg_df = pd.DataFrame(neg_rows)
    log(neg_df.to_string(index=False))

    # --- cell cycle ----------------------------------------------------------
    s_g = [g for g in S_GENES if g in adata.var_names]
    g2m_g = [g for g in G2M_GENES if g in adata.var_names]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_g, g2m_genes=g2m_g)
    cyc = adata.obs.groupby(cl.values)["phase"].value_counts(normalize=True).unstack().fillna(0)
    cyc_in = (cyc.loc[str(a.cluster)] * 100).round(1)
    cyc_all = (cyc.mean() * 100).round(1)
    log(f"\n=== cell cycle ===\n  cluster {a.cluster}: "
        + ", ".join(f"{k} {v}%" for k, v in cyc_in.items()))
    log("  dataset mean: " + ", ".join(f"{k} {v}%" for k, v in cyc_all.items()))
    cycling_in = float(100 - cyc_in.get("G1", 0))
    cycling_all = float(100 - cyc_all.get("G1", 0))

    # --- composition ---------------------------------------------------------
    samp = adata.obs.loc[m, "sample"].astype(str).value_counts(normalize=True)
    tp_share = (pd.crosstab(cl, adata.obs["timepoint"].astype(str), normalize="columns")
                .loc[str(a.cluster)] * 100).round(2)
    log(f"\n=== composition ===\n  top samples: "
        + ", ".join(f"{k} {v:.0%}" for k, v in samp.head(4).items()))
    log("  % of each timepoint: " + ", ".join(f"{k} {v}%" for k, v in tp_share.items()))
    qc_in = adata.obs.loc[m, [c for c in ["n_genes_by_counts", "total_counts",
                                          "pct_counts_mt", "doublet_score"]
                              if c in adata.obs]].median()
    log("  median QC: " + ", ".join(f"{k}={v:.2f}" for k, v in qc_in.items()))

    # --- direct contrast with the runner-up cluster -------------------------
    de_top = pd.DataFrame()
    if a.compare_to:
        pair = adata[cl.isin([str(a.cluster), str(a.compare_to)])].copy()
        pair.obs["_g"] = pair.obs[a.cluster_col].astype(str)
        sc.tl.rank_genes_groups(pair, groupby="_g", groups=[str(a.cluster)],
                                reference=str(a.compare_to), method="wilcoxon",
                                use_raw=False)
        de = sc.get.rank_genes_groups_df(pair, group=str(a.cluster))
        de_top = de[de["pvals_adj"] < 0.05].copy()
        de_top.to_csv(out / f"de_cluster{a.cluster}_vs_{a.compare_to}.csv", index=False)
        up = de_top.nlargest(12, "logfoldchanges")["names"].tolist()
        dn = de_top.nsmallest(12, "logfoldchanges")["names"].tolist()
        log(f"\n=== cluster {a.cluster} vs cluster {a.compare_to} (runner-up) ===")
        log(f"  up in {a.cluster}: {', '.join(up)}")
        log(f"  up in {a.compare_to}: {', '.join(dn)}")

    # --- verdict -------------------------------------------------------------
    passes = (n_pos_ok >= 2) and (len(neg_fail) == 0) and (coexpr_in > coexpr_out)
    log("\n=== VERDICT (guide: >=2 positives co-expressed, negatives absent) ===")
    log(f"  positive markers supporting: {n_pos_ok}/{len(pos_df)}")
    log(f"  negative markers violated:   {len(neg_fail)}"
        + (f" -> {', '.join(neg_fail)}" if neg_fail else ""))
    log(f"  co-expression enriched:      {coexpr_in:.1%} vs {coexpr_out:.1%}")
    log(f"  cycling fraction:            {cycling_in:.1f}% vs {cycling_all:.1f}% dataset mean")
    log(f"  => {'SUPPORTED' if passes else 'NOT SUPPORTED'}")

    pos_df.to_csv(out / "positive_markers.csv", index=False)
    neg_df.to_csv(out / "negative_markers.csv", index=False)

    # --- figures -------------------------------------------------------------
    all_marks = panel["positive"] + [g for gs in panel["negative"].values() for g in gs]
    all_marks = [g for g in all_marks if g in adata.var_names]
    fig = sc.pl.dotplot(adata, var_names=all_marks, groupby=a.cluster_col, use_raw=False,
                        show=False, return_fig=True, standard_scale="var",
                        title=f"cluster {a.cluster} = '{label}': positives then negatives")
    fig.savefig(out / "dotplot_pos_neg_markers.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = np.arange(len(pos_df))
    ax.bar(idx - 0.2, pos_df["pct_in"], 0.4, label=f"cluster {a.cluster}", color="#1F6F5C")
    ax.bar(idx + 0.2, pos_df["pct_other"], 0.4, label="all other clusters", color="#B9C0C9")
    ax.set_xticks(idx); ax.set_xticklabels(pos_df["gene"], rotation=45, ha="right")
    ax.set_ylabel("% of cells expressing"); ax.legend(frameon=False)
    ax.set_title(f"Positive markers for '{label}' (cluster {a.cluster}, n={n_in})")
    for s_ in ax.spines.values():
        s_.set_color("#3A424C")
    fig.tight_layout()
    fig.savefig(out / "positive_marker_detection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (out / "verdict.json").write_text(json.dumps({
        "cluster": a.cluster, "n_cells": n_in, "label": label,
        "guide": "single-cell-annotation-guide Tier 1 manual validation",
        "n_positive_supporting": n_pos_ok, "n_positive_tested": int(len(pos_df)),
        "negative_violations": neg_fail,
        "coexpression_in_cluster": round(coexpr_in, 4),
        "coexpression_elsewhere": round(coexpr_out, 4),
        "cycling_pct": round(cycling_in, 1), "cycling_pct_dataset": round(cycling_all, 1),
        "top_sample_frac": round(float(samp.iloc[0]), 3),
        "top_sample": str(samp.index[0]),
        "supported": bool(passes),
    }, indent=2, default=str))
    log("Done.")


if __name__ == "__main__":
    sys.exit(main())
