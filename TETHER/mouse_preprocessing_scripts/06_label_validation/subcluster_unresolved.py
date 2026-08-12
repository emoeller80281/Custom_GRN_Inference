"""Sub-cluster an unresolved cluster and test whether it contains real cell types.

Cluster 0 (4,468 cells, the largest in the integrated object) could not be assigned an
atlas population: median 1,439 genes against 3,159-5,166 elsewhere, a *low* 3.1%
mitochondrial fraction, 46% E7.75_rep1 + 31% E8.5_rep2, and both candidate labels
contradicted by their own markers. This asks whether it is one uniform low-quality blob or
a mixture hiding recoverable populations.

Three things make naive sub-clustering untrustworthy here, so each is controlled for:

* **Depth drives structure.** In a low-complexity cluster, Leiden will happily split cells
  by library size and the split will look like biology. Clustering is therefore run twice --
  once uncorrected, once after Harmony on `sample` -- and every subcluster is reported with
  its depth so a depth-ladder is visible rather than hidden.

* **Sample confounding.** cl0 is dominated by two libraries. A subcluster drawn from a
  single sample is a batch artefact until proven otherwise, so per-subcluster sample purity
  is reported alongside every marker claim.

* **Ambient RNA mimics markers.** In sparse cells, the most-detected genes are the sample's
  most abundant transcripts, which produce confident-looking but meaningless markers. Each
  subcluster's profile is therefore correlated against its own sample's pseudobulk; a
  subcluster that just looks like bulk is flagged.

A subcluster is called *recoverable* only if it has specific markers (detected in a much
larger fraction of its own cells than elsewhere), is not confined to one sample, and is not
merely tracking depth or ambient signal.

Reference for the atlas comparison: Pijuan-Sala et al., Nature 566:490-495 (2019).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore")

# Canonical panels, shared with the label validation step.
from validate_centroid_labels import CANONICAL  # noqa: E402


def log(m):
    print(f"[subcluster] {m}", flush=True)


def dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def centroids(adata, labels, genes):
    """Mean expression profile per label over `genes`."""
    sub = adata[:, genes]
    df = pd.DataFrame(dense(sub.X), columns=list(genes))
    df["_lab"] = np.asarray(labels)
    return df.groupby("_lab").mean()


def frame_panels(fig, color="#3A424C", lw=1.1):
    """Make panel borders visible; scanpy defaults to frameon=False, which merges panels."""
    for ax in fig.get_axes():
        if getattr(ax, "_colorbar", None) is not None or ax.get_label() == "<colorbar>":
            continue
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(color)
            s.set_linewidth(lw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5mu", default="data/processed/mESC/combined/mESC_combined.h5mu")
    p.add_argument("--cluster", default="0", help="Value of --cluster_col to isolate")
    p.add_argument("--cluster_col", default="leiden")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--ref_h5ad", default="/gpfs/Labs/Uzun/DATA/PROJECTS/"
                   "2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/REFERENCE/pijuan_sala_atlas/"
                   "atlas_reference.h5ad")
    p.add_argument("--resolutions", default="0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--final_res", type=float, default=0.4)
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--n_pcs", type=int, default=30)
    p.add_argument("--min_pct_in", type=float, default=0.40,
                   help="Min fraction of a subcluster's cells expressing a marker")
    p.add_argument("--max_pct_out", type=float, default=0.20,
                   help="Max fraction of other cells expressing it, for specificity")
    p.add_argument("--max_sample_purity", type=float, default=0.80,
                   help="Above this single-sample fraction a subcluster is called batch-driven")
    p.add_argument("--max_rho_bulk", type=float, default=0.95,
                   help="Spearman vs the library pseudobulk above which a subcluster is "
                        "treated as ambient-dominated. Heuristic screen only; the decisive "
                        "test is verify_rescued_subclusters.py")
    p.add_argument("--skip_atlas", action="store_true")
    a = p.parse_args()

    import anndata as ad
    import muon as mu
    import scanpy as sc

    sc.settings.n_jobs = 8
    sc.settings.verbosity = 1
    out = Path(a.out_dir) if a.out_dir else Path(a.h5mu).parent / f"subcluster_{a.cluster}"
    out.mkdir(parents=True, exist_ok=True)

    log("loading integrated object ...")
    mdata = mu.read(a.h5mu)
    r = mdata["rna"]

    keep_obs = [c for c in [a.cluster_col, "sample", "timepoint", "replicate",
                            "n_genes_by_counts", "total_counts", "pct_counts_mt",
                            "pct_counts_ribo", "doublet_score", "celltype_final"]
                if c in r.obs.columns]
    full = ad.AnnData(X=r.layers["counts"].copy(), obs=r.obs[keep_obs].copy(),
                      var=pd.DataFrame(index=r.var_names.copy()))
    sc.pp.normalize_total(full, target_sum=1e4)
    sc.pp.log1p(full)
    log(f"full object: {full.n_obs} cells x {full.n_vars} genes")

    mask = full.obs[a.cluster_col].astype(str).values == str(a.cluster)
    sub = full[mask].copy()
    log(f"isolated cluster {a.cluster}: {sub.n_obs} cells")
    log(f"  samples: {dict(sub.obs['sample'].value_counts())}")

    # Genes must be re-filtered inside the subset: the integrated gene set was chosen
    # across all clusters, and most of it is silent in 4.5k low-depth cells.
    n_cells_gene = np.asarray((sub.X > 0).sum(axis=0)).ravel()
    keep_gene = n_cells_gene >= 3
    log(f"  genes detected in >=3 cells: {int(keep_gene.sum())} / {sub.n_vars}")
    sub_all = sub[:, keep_gene].copy()      # all detected genes, for DE and markers

    # --- embedding -----------------------------------------------------------
    work = sub_all.copy()
    sc.pp.highly_variable_genes(work, n_top_genes=a.n_hvg, flavor="seurat")
    work = work[:, work.var["highly_variable"].values].copy()
    sc.pp.scale(work, max_value=10)
    sc.tl.pca(work, svd_solver="arpack", n_comps=a.n_pcs)

    resolutions = [float(x) for x in a.resolutions.split(",")]

    # Uncorrected.
    sc.pp.neighbors(work, n_neighbors=15, n_pcs=a.n_pcs, key_added="raw")
    sc.tl.umap(work, neighbors_key="raw")
    work.obsm["X_umap_raw"] = work.obsm["X_umap"].copy()
    for res in resolutions:
        sc.tl.leiden(work, resolution=res, neighbors_key="raw",
                     key_added=f"raw_res{res}", flavor="igraph", n_iterations=2)

    # Harmony on sample, to separate shared biology from library structure.
    from common.mudata_utils import run_harmony
    log("running Harmony on 'sample' within the subset ...")
    run_harmony(work, key="sample", basis="X_pca", adjusted_basis="X_pca_harmony", seed=0)
    sc.pp.neighbors(work, n_neighbors=15, use_rep="X_pca_harmony", key_added="hm")
    sc.tl.umap(work, neighbors_key="hm")
    work.obsm["X_umap_hm"] = work.obsm["X_umap"].copy()
    for res in resolutions:
        sc.tl.leiden(work, resolution=res, neighbors_key="hm",
                     key_added=f"hm_res{res}", flavor="igraph", n_iterations=2)

    sweep = pd.DataFrame({
        "resolution": resolutions,
        "n_uncorrected": [work.obs[f"raw_res{r_}"].nunique() for r_ in resolutions],
        "n_harmony": [work.obs[f"hm_res{r_}"].nunique() for r_ in resolutions],
    })
    sweep.to_csv(out / "resolution_sweep.csv", index=False)
    log("\nresolution sweep:\n" + sweep.to_string(index=False))

    # Pick the working resolution rather than trusting the default: a resolution that
    # yields a single subcluster is not a failure to report as a crash, it means the
    # cluster has no substructure to find. Escalate to the coarsest resolution that does
    # split, and if nothing splits, that homogeneity IS the answer.
    n_by_res = {r_: work.obs[f"hm_res{r_}"].nunique() for r_ in resolutions}
    if n_by_res.get(a.final_res, 0) >= 2:
        chosen = a.final_res
    else:
        splits = [r_ for r_ in resolutions if n_by_res[r_] >= 2]
        chosen = min(splits) if splits else None
        if chosen is not None:
            log(f"res {a.final_res} gives {n_by_res.get(a.final_res)} subcluster(s); "
                f"escalating to res {chosen} ({n_by_res[chosen]} subclusters)")

    if chosen is None:
        log(f"\nCluster {a.cluster} does not split at any resolution up to "
            f"{max(resolutions)} — it is one homogeneous group.")
        (out / "subcluster_summary.json").write_text(json.dumps({
            "cluster": a.cluster, "n_cells": int(sub_all.n_obs),
            "n_subclusters": 1, "homogeneous": True, "n_recoverable": 0,
            "resolutions_tried": resolutions,
            "conclusion": "No substructure at any tested resolution; the cluster contains "
                          "no separable subpopulation to recover.",
        }, indent=2, default=str))
        log("Done.")
        return

    key = f"hm_res{chosen}"
    a.final_res = chosen
    sub_all.obs["subcluster"] = work.obs[key].values
    sub_all.obs["subcluster_raw"] = work.obs[f"raw_res{a.final_res}"].values
    for c in ["X_umap_raw", "X_umap_hm"]:
        sub_all.obsm[c] = work.obsm[c]
    subs = sub_all.obs["subcluster"].astype(str)
    order = sorted(subs.unique(), key=lambda x: int(x) if x.isdigit() else x)
    log(f"\nusing {key}: {len(order)} subclusters")

    # Does correction change the partition, i.e. was the raw split batch structure?
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(work.obs[f"raw_res{a.final_res}"].astype(str),
                              work.obs[key].astype(str))
    log(f"ARI(uncorrected, harmony) at res {a.final_res}: {ari:.3f}")

    # --- per-subcluster description -----------------------------------------
    qc_cols = [c for c in ["n_genes_by_counts", "total_counts", "pct_counts_mt",
                           "pct_counts_ribo", "doublet_score"] if c in sub_all.obs]
    desc = sub_all.obs.groupby("subcluster")[qc_cols].median()
    desc["n_cells"] = subs.value_counts()
    samp = pd.crosstab(subs, sub_all.obs["sample"].astype(str), normalize="index")
    desc["top_sample"] = samp.idxmax(axis=1)
    desc["top_sample_frac"] = samp.max(axis=1).round(3)
    desc["n_samples_ge5pct"] = (samp >= 0.05).sum(axis=1)
    desc = desc.loc[order]
    samp.round(4).to_csv(out / "subcluster_sample_composition.csv")

    # --- markers: Wilcoxon within the subset, on all detected genes ----------
    log("ranking marker genes (Wilcoxon) ...")
    # Wilcoxon needs at least two groups with enough cells; dropping tiny subclusters can
    # leave only one, so check what actually survives rather than assuming.
    too_small = [c for c in order if (subs == c).sum() < 10]
    if too_small:
        log(f"  excluding {len(too_small)} subcluster(s) with <10 cells from DE: "
            f"{', '.join(too_small)}")
    de_input = sub_all[~subs.isin(too_small)].copy() if too_small else sub_all
    de_groups = de_input.obs["subcluster"].astype(str).nunique()
    if de_groups >= 2:
        sc.tl.rank_genes_groups(de_input, groupby="subcluster", method="wilcoxon",
                                use_raw=False)
        de = sc.get.rank_genes_groups_df(de_input, group=None)
        # With a single surviving group scanpy omits the 'group' column entirely.
        if "group" not in de.columns:
            de["group"] = str(de_input.obs["subcluster"].astype(str).iloc[0])
    else:
        log("  fewer than 2 usable subclusters — skipping differential expression")
        de = pd.DataFrame(columns=["group", "names", "logfoldchanges", "pvals_adj"])
    de.to_csv(out / "subcluster_markers_all.csv", index=False)

    # Specificity: a real marker is detected in most of its own subcluster and few others.
    X = sub_all.X
    genes_idx = {g: i for i, g in enumerate(sub_all.var_names)}
    det = (X > 0)
    rows = []
    for cl in order:
        m = (subs == cl).values
        n_in, n_out = m.sum(), (~m).sum()
        top = de[(de["group"] == cl) & (de["pvals_adj"] < 0.05)
                 & (de["logfoldchanges"] > 0.5)].head(60)
        for _, g in top.iterrows():
            j = genes_idx.get(g["names"])
            if j is None:
                continue
            col = det[:, j]
            col = np.asarray(col.todense()).ravel() if sp.issparse(col) else np.asarray(col).ravel()
            pin = col[m].sum() / max(n_in, 1)
            pout = col[~m].sum() / max(n_out, 1)
            rows.append({"subcluster": cl, "gene": g["names"],
                         "logFC": round(float(g["logfoldchanges"]), 2),
                         "padj": float(g["pvals_adj"]),
                         "pct_in": round(float(pin), 3), "pct_out": round(float(pout), 3),
                         "specificity": round(float(pin - pout), 3),
                         "specific": bool(pin >= a.min_pct_in and pout <= a.max_pct_out)})
    spec = pd.DataFrame(rows, columns=["subcluster","gene","logFC","padj",
                                       "pct_in","pct_out","specificity","specific"])
    spec.sort_values(["subcluster", "specificity"], ascending=[True, False]).to_csv(
        out / "subcluster_marker_specificity.csv", index=False)
    n_spec = (spec[spec["specific"]].groupby("subcluster").size()
              if len(spec) else pd.Series(dtype=int))
    desc["n_specific_markers"] = [int(n_spec.get(c, 0)) for c in order]

    # --- ambient proxy: does the subcluster just look like its sample's bulk? -
    hv = work.var_names.tolist()
    amb = []
    for cl in order:
        m = (subs == cl).values
        s_top = desc.loc[cl, "top_sample"]
        same = (sub_all.obs["sample"].astype(str) == s_top).values
        prof = np.asarray(dense(sub_all[m][:, hv].X).mean(axis=0)).ravel()
        # Pseudobulk of that sample across the WHOLE object, not just this cluster.
        bulk_mask = (full.obs["sample"].astype(str) == s_top).values
        bulk = np.asarray(dense(full[bulk_mask][:, hv].X).mean(axis=0)).ravel()
        amb.append(round(float(spearmanr(prof, bulk).statistic), 3))
    desc["rho_vs_sample_bulk"] = amb

    # --- canonical marker panels --------------------------------------------
    panel_genes = sorted({g for gs in CANONICAL.values() for g in gs
                          if g in sub_all.var_names})
    pc = centroids(sub_all, subs.values, panel_genes)
    panel = {}
    for ct, gs in CANONICAL.items():
        gs = [g for g in gs if g in pc.columns]
        if not gs:
            continue
        z = (pc[gs] - pc[gs].mean()) / pc[gs].std(ddof=0).replace(0, np.nan)
        panel[ct] = z.mean(axis=1)
    panel = pd.DataFrame(panel).loc[order]
    panel.round(3).to_csv(out / "subcluster_canonical_panel_z.csv")
    desc["best_panel"] = panel.idxmax(axis=1)
    desc["best_panel_z"] = panel.max(axis=1).round(2)

    # --- what does each subcluster resemble, in the object and in the atlas? --
    main_genes = [g for g in hv if g in full.var_names]
    main_cent = centroids(full, full.obs[a.cluster_col].astype(str).values, main_genes)
    sub_cent = centroids(sub_all, subs.values, main_genes)
    rho_main = pd.DataFrame(index=order, columns=main_cent.index, dtype=float)
    for cl in order:
        rr, _ = spearmanr(sub_cent.loc[cl].values, main_cent.values, axis=1)
        rho_main.loc[cl] = rr[0, 1:]
    rho_main.round(4).to_csv(out / "subcluster_vs_main_clusters.csv")
    other = rho_main.drop(columns=[str(a.cluster)], errors="ignore")
    desc["closest_main_cluster"] = other.idxmax(axis=1)
    desc["rho_closest_main"] = other.max(axis=1).round(3)

    if not a.skip_atlas and Path(a.ref_h5ad).exists():
        log("loading atlas reference for centroid comparison ...")
        ref = ad.read_h5ad(a.ref_h5ad)
        sc.pp.normalize_total(ref, target_sum=1e4)
        sc.pp.log1p(ref)
        shared = sorted(set(ref.var_names) & set(sub_all.var_names))
        ref_s = ref[:, shared].copy()
        sc.pp.highly_variable_genes(ref_s, n_top_genes=2000, flavor="seurat")
        rg = ref_s.var_names[ref_s.var["highly_variable"].values].tolist()
        ref_cent = centroids(ref_s, ref.obs["celltype"].values, rg)
        del ref, ref_s
        sub_cent_a = centroids(sub_all, subs.values, rg)
        rho_atlas = pd.DataFrame(index=order, columns=ref_cent.index, dtype=float)
        for cl in order:
            rr, _ = spearmanr(sub_cent_a.loc[cl].values, ref_cent.values, axis=1)
            rho_atlas.loc[cl] = rr[0, 1:]
        rho_atlas.round(4).to_csv(out / "subcluster_vs_atlas.csv")
        desc["atlas_best"] = rho_atlas.idxmax(axis=1)
        desc["atlas_rho"] = rho_atlas.max(axis=1).round(3)
        desc["atlas_z"] = ((rho_atlas.max(axis=1) - rho_atlas.mean(axis=1))
                           / rho_atlas.std(axis=1)).round(2)

    # --- verdict -------------------------------------------------------------
    # A subcluster drawn almost entirely from one library is normally a batch artefact --
    # but not when that library is the ONLY one for its timepoint, because then a genuinely
    # stage-specific population is indistinguishable from a batch effect by purity alone.
    # (E7.75 has a single library here, so this is not hypothetical.) Those are separated
    # out rather than dismissed: they need the absolute-expression test in
    # verify_rescued_subclusters.py, which ambient contamination cannot pass.
    tp_of_sample = (sub_all.obs[["sample", "timepoint"]].astype(str)
                    .drop_duplicates().set_index("sample")["timepoint"].to_dict())
    top_tp, n_libs_tp = [], []
    for cl in order:
        s_top = desc.loc[cl, "top_sample"]
        tp = tp_of_sample.get(s_top, "NA")
        top_tp.append(tp)
        n_libs_tp.append(sum(1 for s_, t_ in tp_of_sample.items() if t_ == tp))
    desc["top_timepoint"] = top_tp
    desc["n_libraries_at_timepoint"] = n_libs_tp

    sample_pure = desc["top_sample_frac"] > a.max_sample_purity
    desc["single_library_timepoint"] = sample_pure & (desc["n_libraries_at_timepoint"] < 2)
    desc["batch_driven"] = sample_pure & (desc["n_libraries_at_timepoint"] >= 2)

    # The ambient statistic was computed above but previously ignored in the verdict: a
    # subcluster whose profile is essentially its library's pseudobulk is soup, however
    # specific its "markers" look against a comparison group from a different sample.
    desc["ambient_like"] = desc["rho_vs_sample_bulk"] >= a.max_rho_bulk

    desc["recoverable"] = ((desc["n_specific_markers"] >= 3)
                           & (~desc["batch_driven"]) & (~desc["ambient_like"]))
    # Sample purity is a screen, not a verdict, and on this data it was wrong in both
    # directions -- it dismissed a subcluster carrying Ttn/Myh6/Myl7 at 62-77% against a
    # 1-2% background. Any subcluster with specific markers that is not obviously soup goes
    # to the absolute-expression test, whatever its batch status says.
    desc["needs_absolute_test"] = ((desc["n_specific_markers"] >= 3)
                                   & (~desc["ambient_like"]))
    desc = desc.rename(columns={"n_genes_by_counts": "med_genes",
                                "total_counts": "med_counts",
                                "pct_counts_mt": "med_pct_mt",
                                "pct_counts_ribo": "med_pct_ribo",
                                "doublet_score": "med_doublet"})
    cols = [c for c in ["n_cells", "med_genes", "med_counts", "med_pct_mt", "med_doublet",
                        "top_sample", "top_sample_frac", "top_timepoint",
                        "n_libraries_at_timepoint", "n_samples_ge5pct",
                        "n_specific_markers", "rho_vs_sample_bulk", "best_panel",
                        "best_panel_z", "closest_main_cluster", "rho_closest_main",
                        "atlas_best", "atlas_rho", "atlas_z", "ambient_like",
                        "batch_driven", "single_library_timepoint",
                        "needs_absolute_test", "recoverable"] if c in desc.columns]
    desc = desc[cols]
    desc.to_csv(out / "subcluster_summary.csv")
    pd.set_option("display.width", 250)
    log("\n=== subcluster summary ===\n" + desc.to_string())

    log("\n=== top specific markers per subcluster ===")
    for cl in order:
        s = spec[(spec["subcluster"] == cl) & spec["specific"]].sort_values(
            "specificity", ascending=False).head(8)
        if len(s) == 0:
            log(f"  sub {cl:>2}: NO specific markers "
                f"(n={int(desc.loc[cl,'n_cells'])}, med_genes={int(desc.loc[cl,'med_genes'])})")
        else:
            gg = ", ".join(f"{x.gene}({x.pct_in:.0%}/{x.pct_out:.0%})"
                           for x in s.itertuples())
            log(f"  sub {cl:>2}: {gg}")

    # --- figures -------------------------------------------------------------
    colour = [c for c in ["subcluster", "sample", "timepoint", "n_genes_by_counts",
                          "total_counts", "pct_counts_mt", "doublet_score"]
              if c in sub_all.obs.columns]
    for basis, tag in [("X_umap_hm", "harmony"), ("X_umap_raw", "uncorrected")]:
        sub_all.obsm["X_umap"] = sub_all.obsm[basis]
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        for ax, col in zip(axes.ravel(), colour):
            sc.pl.umap(sub_all, color=col, ax=ax, show=False, frameon=True,
                       legend_fontsize=7, title=col)
        for ax in axes.ravel()[len(colour):]:
            ax.axis("off")
        frame_panels(fig)
        fig.suptitle(f"Cluster {a.cluster} sub-structure ({tag}); "
                     f"{sub_all.n_obs} cells", fontsize=13)
        fig.tight_layout()
        fig.savefig(out / f"umap_subcluster_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Canonical panel heatmap.
    fig, ax = plt.subplots(figsize=(0.34 * panel.shape[1] + 5, 0.4 * panel.shape[0] + 3))
    im = ax.imshow(panel.values.astype(float), cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(panel.shape[1]))
    ax.set_xticklabels(panel.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(panel.shape[0]))
    ax.set_yticklabels([f"sub {c} (n={int(desc.loc[c,'n_cells'])})" for c in panel.index],
                       fontsize=8)
    ax.set_title(f"Canonical marker z-score per subcluster of cluster {a.cluster}")
    fig.colorbar(im, ax=ax, shrink=0.6, label="mean marker z")
    fig.tight_layout()
    fig.savefig(out / "heatmap_subcluster_panels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Dotplot of the strongest specific markers.
    picks = (spec[spec["specific"]].sort_values("specificity", ascending=False)
             .groupby("subcluster").head(3)["gene"].unique().tolist()) if len(spec) else []
    if picks:
        fig = sc.pl.dotplot(sub_all, var_names=picks[:40], groupby="subcluster",
                            use_raw=False, show=False, return_fig=True)
        fig.savefig(out / "dotplot_specific_markers.png", dpi=150, bbox_inches="tight")
        plt.close("all")

    sub_all.obsm["X_umap"] = sub_all.obsm["X_umap_hm"]
    sub_all.write(out / f"cluster{a.cluster}_subclustered.h5ad", compression="gzip")

    n_rec = int(desc["recoverable"].sum())
    (out / "subcluster_summary.json").write_text(json.dumps({
        "cluster": a.cluster, "n_cells": int(sub_all.n_obs),
        "resolution": a.final_res, "n_subclusters": len(order),
        "ari_uncorrected_vs_harmony": round(float(ari), 3),
        "n_recoverable": n_rec,
        "recoverable": desc.index[desc["recoverable"]].tolist(),
        "needs_absolute_test": desc.index[desc["needs_absolute_test"]].tolist(),
        "ambient_like": desc.index[desc["ambient_like"]].tolist(),
        "batch_driven": desc.index[desc["batch_driven"]].tolist(),
        "criteria": {"min_pct_in": a.min_pct_in, "max_pct_out": a.max_pct_out,
                     "min_specific_markers": 3,
                     "max_sample_purity": a.max_sample_purity},
    }, indent=2, default=str))
    log(f"\n{n_rec} of {len(order)} subclusters look recoverable.")
    log("Done.")


if __name__ == "__main__":
    sys.exit(main())
