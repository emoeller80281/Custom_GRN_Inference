"""Combine the per-sample mESC gastrulation runs into one Harmony-integrated MuData.

Takes the outputs of ``annotate_scrna_celltypes.py`` for every sample and builds a
single ``MuData`` with an ``rna`` and an ``atac`` modality, batch-corrected across
samples with Harmony and given a joint WNN embedding.

Two things about this data drive the design:

* **Genes are intersected, not unioned.** Each sample was filtered independently, so
  a gene missing from one sample was *filtered*, not observed as zero. An outer join
  would fabricate structured zeros that Harmony would read as biology.

* **Peaks are sample-specific and must be harmonised first.** CellRanger-ARC called
  peaks per sample, so the same regulatory element appears as
  ``chr1:3035460-3036350`` in one sample and ``chr1:3062557-3063384`` in another.
  Concatenating on peak name would give ~2.3M near-duplicate features with almost no
  overlap. Instead the union of all peaks is merged into consensus intervals and each
  sample's counts are summed into them. This is exact aggregation of the called peaks;
  its one limitation is that reads falling outside a sample's own peak calls are not
  recovered (that would need re-quantification from the fragment files).

Example:
    python combine_mesc_samples.py --in_root data/processed/mESC \
        --out_dir data/processed/mESC/combined
"""

import argparse
import gc
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.mudata_utils import run_harmony, sanitize_for_h5  # noqa: E402

warnings.filterwarnings("ignore")

# The wild-type gastrulation timecourse. The E8.5 CRISPR T-KO/T-WT pair is a
# perturbation experiment, not a timepoint, and is deliberately excluded: mixing a
# genotype contrast into the reference trajectory would let Harmony regress away
# the very difference that experiment exists to measure.
SAMPLES = [
    "E7.5_rep1", "E7.5_rep2", "E7.75_rep1", "E8.0_rep1", "E8.0_rep2",
    "E8.5_rep1", "E8.5_rep2", "E8.75_rep1", "E8.75_rep2",
]

# Samples whose QC leaves them too small or too stressed to weight equally.
# Flagged rather than dropped so the choice stays with the analyst.
LOW_QUALITY = {"E7.5_rep2"}


def log(m):
    print(f"[combine] {m}", flush=True)


def parse_sample_meta(s):
    """Split a sample name into timepoint / replicate / genotype covariates."""
    if "CRISPR" in s:
        tp = s.split("_")[0]
        return {"timepoint": tp, "replicate": "NA",
                "genotype": "T_KO" if s.endswith("KO") else "T_WT"}
    tp, rep = s.rsplit("_", 1)
    return {"timepoint": tp, "replicate": rep, "genotype": "WT"}


# ---------------------------------------------------------------------------
# Consensus peaks
# ---------------------------------------------------------------------------
def build_consensus_peaks(peak_frames, max_width):
    """Merge per-sample peak intervals into a non-overlapping consensus set.

    Union-merging can chain overlapping peaks into very wide intervals; those are
    dropped above ``max_width`` because a multi-kb "peak" is no longer a usable
    regulatory feature.
    """
    allp = pd.concat(peak_frames, ignore_index=True)
    # Cast away per-sample Categorical dtypes: samples carry different category sets,
    # and sorting by category code rather than name would interleave chromosomes.
    allp["chrom"] = allp["chrom"].astype(str)
    allp = allp.sort_values(["chrom", "start"], kind="mergesort").reset_index(drop=True)

    chrom = allp["chrom"].values
    start = allp["start"].values.astype(np.int64)
    end = allp["end"].values.astype(np.int64)

    # A new consensus interval starts at a chromosome change or at a gap. The running
    # max of `end` MUST be accumulated per chromosome: done globally, chr1's ~195 Mb
    # coordinates dominate every chromosome sorted after it, no gap is ever detected,
    # and each later chromosome collapses into a single interval.
    grp_start = np.zeros(len(allp), dtype=bool)
    for ch in pd.unique(chrom):
        m = np.nonzero(chrom == ch)[0]      # contiguous, start-sorted after the sort
        rm = np.maximum.accumulate(end[m])
        g = np.empty(len(m), dtype=bool)
        g[0] = True                          # a new chromosome always opens a group
        g[1:] = start[m][1:] > rm[:-1]
        grp_start[m] = g
    grp = np.cumsum(grp_start) - 1

    cons = pd.DataFrame({"chrom": chrom, "start": start, "end": end, "grp": grp})
    cons = cons.groupby("grp").agg(chrom=("chrom", "first"), start=("start", "min"),
                                   end=("end", "max")).reset_index(drop=True)
    cons["width"] = cons["end"] - cons["start"]
    n_all = len(cons)
    cons = cons[cons["width"] <= max_width].reset_index(drop=True)
    med = int(cons["width"].median()) if len(cons) else 0
    log(f"Consensus peaks: {n_all} merged intervals, {len(cons)} kept "
        f"(<= {max_width} bp); median width {med}")
    cons["peak"] = (cons["chrom"].astype(str) + ":" + cons["start"].astype(str)
                    + "-" + cons["end"].astype(str))

    # Sanity guard. Peaks overlap heavily across samples, so the consensus set should
    # land somewhere near a single sample's peak count -- never a small fraction of it.
    # A collapse here is silent otherwise: the wide chained intervals get removed by
    # the width filter and the run continues with most of the genome missing.
    largest = max(len(f) for f in peak_frames)
    if len(cons) < 0.5 * largest:
        raise RuntimeError(
            f"Consensus collapse: {len(cons)} intervals from a largest input of "
            f"{largest} peaks across {cons['chrom'].nunique()} chromosomes. "
            "Expected at least half the largest sample's peak count.")
    log(f"  spans {cons['chrom'].nunique()} chromosomes; widest {int(cons['width'].max())} bp")
    return cons


def map_peaks_to_consensus(var, cons):
    """Sparse (n_sample_peaks x n_consensus) indicator of which consensus peak each falls in."""
    out = np.full(len(var), -1, dtype=np.int64)
    # Positional indices throughout: var is indexed by peak name, so groupby().groups
    # would hand back labels rather than row positions.
    chroms = var["chrom"].values
    starts = var["start"].values.astype(np.int64)
    cons_chrom = cons["chrom"].values
    cons_start = cons["start"].values
    cons_end = cons["end"].values
    for ch in pd.unique(chroms):
        idx = np.nonzero(chroms == ch)[0]
        sub_pos = np.nonzero(cons_chrom == ch)[0]
        if len(idx) == 0 or len(sub_pos) == 0:
            continue
        s = starts[idx]
        # cons is sorted by (chrom, start), so sub_pos is contiguous and ordered.
        pos = np.searchsorted(cons_start[sub_pos], s, side="right") - 1
        ok = pos >= 0
        cand = sub_pos[np.clip(pos, 0, len(sub_pos) - 1)]
        # Keep only peaks genuinely contained in the candidate consensus interval.
        inside = ok & (s >= cons_start[cand]) & (s < cons_end[cand])
        out[idx[inside]] = cand[inside]
    keep = out >= 0
    rows = np.nonzero(keep)[0]
    cols = out[keep]
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(var), len(cons)))
    return M, int(keep.sum())


# ---------------------------------------------------------------------------
def batch_mixing(emb, labels, n_neighbors=30, seed=0, max_cells=20000):
    """Mean fraction of a cell's neighbours drawn from a *different* sample.

    1.0 would be perfect mixing relative to the null; the expected value under
    random mixing is reported alongside so the number is interpretable.
    """
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    n = emb.shape[0]
    idx = rng.choice(n, size=min(max_cells, n), replace=False)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(emb)
    _, ind = nn.kneighbors(emb[idx])
    lab = pd.Categorical(labels).codes
    same = (lab[ind[:, 1:]] == lab[idx][:, None]).mean()
    p = pd.Series(labels).value_counts(normalize=True)
    expected_same = float((p ** 2).sum())
    return {"observed_same_sample_frac": float(same),
            "expected_same_sample_frac": expected_same,
            "mixing_ratio": float(expected_same / same) if same > 0 else float("nan")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_root", default="data/processed/mESC")
    p.add_argument("--out_dir", default="data/processed/mESC/combined")
    p.add_argument("--n_hvg", type=int, default=3000)
    p.add_argument("--n_pcs", type=int, default=50)
    p.add_argument("--n_lsi", type=int, default=50)
    p.add_argument("--n_neighbors", type=int, default=30)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--max_peak_width", type=int, default=10000)
    p.add_argument("--min_cells_per_peak", type=int, default=10)
    p.add_argument("--skip_atac", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import anndata as ad
    import muon as mu
    import scanpy as sc

    sc.settings.n_jobs = 16
    in_root, out = Path(a.in_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"samples": [], "params": vars(a)}

    # ---------------- RNA ----------------------------------------------
    log("=== RNA ===")
    rnas = []
    for s in SAMPLES:
        f = in_root / s / f"{s}_rna_processed.h5ad"
        if not f.exists():
            log(f"  MISSING {f}, skipping"); continue
        x = ad.read_h5ad(f)
        # Rewind to raw counts: X currently holds log-normalised values.
        r = ad.AnnData(X=x.layers["counts"].copy(),
                       obs=x.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt",
                                  "pct_counts_ribo", "doublet_score", "cell_type"]].copy(),
                       var=x.var[["gene_ids", "chrom", "start", "end"]].copy())
        r.obs["per_sample_leiden"] = x.obs["leiden"].astype(str).values
        r.obs["per_sample_cell_type"] = x.obs["cell_type"].astype(str).values
        r.obs = r.obs.drop(columns=["cell_type"])
        r.obs["sample"] = s
        for k, v in parse_sample_meta(s).items():
            r.obs[k] = v
        r.obs["qc_flag"] = "low_quality" if s in LOW_QUALITY else "ok"
        r.obs_names = [f"{s}_{b}" for b in r.obs_names]
        rnas.append(r)
        summary["samples"].append({"sample": s, "n_cells": int(r.n_obs),
                                   "n_genes": int(r.n_vars)})
        log(f"  {s}: {r.n_obs} cells x {r.n_vars} genes")
        del x; gc.collect()

    # join='inner': a gene absent from one sample was filtered there, not observed
    # as zero, so unioning would invent structured zeros.
    rna = ad.concat(rnas, join="inner", index_unique=None)
    del rnas; gc.collect()
    log(f"Combined RNA: {rna.n_obs} cells x {rna.n_vars} genes (gene intersection)")

    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    rna.raw = rna
    try:
        import skmisc  # noqa: F401
        sc.pp.highly_variable_genes(rna, n_top_genes=a.n_hvg, flavor="seurat_v3",
                                    layer="counts", batch_key="sample")
        flavor = "seurat_v3"
    except ImportError:
        sc.pp.highly_variable_genes(rna, n_top_genes=a.n_hvg, batch_key="sample")
        flavor = "seurat"
    log(f"HVGs: {int(rna.var['highly_variable'].sum())} (flavor={flavor}, batch-aware)")

    sc.pp.scale(rna, max_value=10)
    sc.tl.pca(rna, n_comps=a.n_pcs, svd_solver="arpack", mask_var="highly_variable")
    rna.X = rna.raw.X.copy()   # keep log-normalised values for DE/plots

    pre = batch_mixing(rna.obsm["X_pca"], rna.obs["sample"].values, seed=a.seed)
    log(f"Batch mixing before Harmony: {pre}")

    log("Running Harmony on RNA PCA (key='sample') ...")
    run_harmony(rna, key="sample", basis="X_pca", adjusted_basis="X_pca_harmony",
                seed=a.seed)
    post = batch_mixing(rna.obsm["X_pca_harmony"], rna.obs["sample"].values, seed=a.seed)
    log(f"Batch mixing after Harmony:  {post}")
    summary["rna_batch_mixing"] = {"before": pre, "after": post}

    # Uncorrected UMAP retained for the before/after comparison figure.
    sc.pp.neighbors(rna, n_neighbors=a.n_neighbors, use_rep="X_pca",
                    random_state=a.seed, key_added="uncorrected")
    sc.tl.umap(rna, neighbors_key="uncorrected", random_state=a.seed)
    rna.obsm["X_umap_uncorrected"] = rna.obsm["X_umap"].copy()

    sc.pp.neighbors(rna, n_neighbors=a.n_neighbors, use_rep="X_pca_harmony",
                    random_state=a.seed)
    sc.tl.umap(rna, random_state=a.seed)
    sc.tl.leiden(rna, resolution=a.resolution, key_added="leiden", flavor="igraph",
                 n_iterations=2, directed=False, random_state=a.seed)
    log(f"RNA Leiden (harmony): {rna.obs['leiden'].nunique()} clusters")
    summary["rna"] = {"n_cells": int(rna.n_obs), "n_genes": int(rna.n_vars),
                      "n_hvg": int(rna.var["highly_variable"].sum()),
                      "hvg_flavor": flavor,
                      "n_clusters": int(rna.obs["leiden"].nunique())}

    keep_cells = {s: set() for s in SAMPLES}
    for bc, s in zip(rna.obs_names, rna.obs["sample"].astype(str)):
        keep_cells[s].add(bc[len(s) + 1:])

    mods = {"rna": rna}

    # ---------------- ATAC ---------------------------------------------
    if not a.skip_atac:
        log("=== ATAC ===")
        var_frames, paths = {}, {}
        for s in SAMPLES:
            f = in_root / s / f"{s}_atac_raw.h5ad"
            if not f.exists():
                log(f"  MISSING {f}"); continue
            v = ad.read_h5ad(f, backed="r").var[["chrom", "start", "end"]].copy()
            v["start"] = v["start"].astype(np.int64)
            v["end"] = v["end"].astype(np.int64)
            var_frames[s] = v
            paths[s] = f
            log(f"  {s}: {len(v)} peaks")

        cons = build_consensus_peaks(list(var_frames.values()), a.max_peak_width)
        summary["atac_consensus_peaks"] = int(len(cons))

        atacs = []
        for s, f in paths.items():
            x = ad.read_h5ad(f)
            sel = [b for b in x.obs_names if b in keep_cells[s]]
            x = x[sel].copy()
            M, n_mapped = map_peaks_to_consensus(var_frames[s], cons)
            Xc = (x.X.tocsr().astype(np.float32) @ M).tocsr()
            aa = ad.AnnData(X=Xc, obs=pd.DataFrame(index=[f"{s}_{b}" for b in x.obs_names]),
                            var=pd.DataFrame(index=cons["peak"].values))
            aa.obs["sample"] = s
            atacs.append(aa)
            log(f"  {s}: {x.n_obs} cells, {n_mapped}/{len(var_frames[s])} peaks mapped")
            del x, M, Xc; gc.collect()

        atac = ad.concat(atacs, join="outer", index_unique=None)
        del atacs; gc.collect()
        # Column by column, aligned on peak name: a block assignment would make every
        # one of these object dtype and break the h5mu write.
        cons_idx = cons.set_index("peak")
        atac.var["chrom"] = cons_idx.loc[atac.var_names, "chrom"].astype(str).values
        for col in ("start", "end", "width"):
            atac.var[col] = cons_idx.loc[atac.var_names, col].astype(np.int64).values
        log(f"Combined ATAC: {atac.n_obs} cells x {atac.n_vars} consensus peaks")

        n_cells_peak = np.asarray((atac.X > 0).sum(axis=0)).ravel()
        atac = atac[:, n_cells_peak >= a.min_cells_per_peak].copy()
        log(f"After min_cells_per_peak={a.min_cells_per_peak}: {atac.n_vars} peaks")

        atac.layers["counts"] = atac.X.copy()
        mu.atac.pp.tfidf(atac, scale_factor=1e4)
        mu.atac.tl.lsi(atac, n_comps=a.n_lsi)
        # LSI component 1 tracks sequencing depth rather than biology.
        depth = np.log1p(np.asarray(atac.layers["counts"].sum(1)).ravel())
        r1 = abs(np.corrcoef(atac.obsm["X_lsi"][:, 0], depth)[0, 1])
        log(f"LSI1 vs log depth |r| = {r1:.3f}")
        if r1 > 0.8:
            atac.obsm["X_lsi"] = atac.obsm["X_lsi"][:, 1:]
            atac.varm["LSI"] = atac.varm["LSI"][:, 1:]
            atac.uns["lsi"]["stdev"] = atac.uns["lsi"]["stdev"][1:]
            log("  dropped LSI1")
        summary["atac_lsi1_depth_corr"] = float(r1)

        pre_a = batch_mixing(atac.obsm["X_lsi"], atac.obs["sample"].values, seed=a.seed)
        log("Running Harmony on ATAC LSI (key='sample') ...")
        run_harmony(atac, key="sample", basis="X_lsi", adjusted_basis="X_lsi_harmony",
                    seed=a.seed)
        post_a = batch_mixing(atac.obsm["X_lsi_harmony"], atac.obs["sample"].values,
                              seed=a.seed)
        log(f"ATAC batch mixing before/after: {pre_a} / {post_a}")
        summary["atac_batch_mixing"] = {"before": pre_a, "after": post_a}

        sc.pp.neighbors(atac, n_neighbors=a.n_neighbors, use_rep="X_lsi_harmony",
                        random_state=a.seed)
        sc.tl.umap(atac, random_state=a.seed)
        summary["atac"] = {"n_cells": int(atac.n_obs), "n_peaks": int(atac.n_vars)}
        mods["atac"] = atac

    # ---------------- MuData -------------------------------------------
    log("=== MuData ===")
    for m in mods.values():
        sanitize_for_h5(m)
    mdata = mu.MuData(mods)
    mu.pp.intersect_obs(mdata)
    log(f"MuData: {mdata.n_obs} cells across {len(mods)} modalities")

    if "atac" in mods:
        # WNN runs on a slim MuData holding nothing but the two corrected embeddings.
        # Run against the full objects it dies inside muon on the accumulated
        # uns/obs metadata; the graph only needs the embeddings, so this sidesteps it.
        try:
            import traceback
            slim = {}
            for name, m in mods.items():
                rep = "X_pca_harmony" if name == "rna" else "X_lsi_harmony"
                s = ad.AnnData(
                    sp.csr_matrix((m.n_obs, 1), dtype=np.float32),
                    obs=pd.DataFrame(index=m.obs_names.copy()),
                )
                s.obsm[rep] = np.asarray(m.obsm[rep], dtype=np.float32)
                sc.pp.neighbors(s, n_neighbors=a.n_neighbors, use_rep=rep,
                                random_state=a.seed)
                slim[name] = s
            sm = mu.MuData(slim)
            mu.pp.neighbors(sm, key_added="wnn", n_neighbors=a.n_neighbors,
                            random_state=a.seed)
            # muon records use_rep as a per-modality dict; scanpy's downstream
            # _choose_representation does `use_rep in adata.obsm`, and a dict is
            # unhashable, so umap/leiden raise TypeError unless it is removed.
            sm.uns["wnn"].get("params", {}).pop("use_rep", None)
            sc.tl.umap(sm, neighbors_key="wnn", random_state=a.seed)
            sc.tl.leiden(sm, neighbors_key="wnn", resolution=a.resolution,
                         key_added="leiden_wnn", flavor="igraph", n_iterations=2,
                         directed=False, random_state=a.seed)
            mdata.obsm["X_umap_wnn"] = np.asarray(sm.obsm["X_umap"])
            mdata.obs["leiden_wnn"] = pd.Categorical(
                sm.obs.loc[mdata.obs_names, "leiden_wnn"].astype(str).values)
            # Per-cell modality weights come back as 'rna:mod_weight' / 'atac:mod_weight'.
            for c in [c for c in sm.obs.columns if "mod_weight" in c]:
                mdata.obs[c] = sm.obs.loc[mdata.obs_names, c].values
            log(f"WNN Leiden: {mdata.obs['leiden_wnn'].nunique()} clusters")
            summary["wnn_clusters"] = int(mdata.obs["leiden_wnn"].nunique())
        except Exception as exc:  # noqa: BLE001
            log(f"WARNING: WNN failed ({type(exc).__name__}: {exc}); "
                "per-modality Harmony embeddings are still present and usable")
            log(traceback.format_exc())
            summary["wnn_error"] = f"{type(exc).__name__}: {exc}"

    h5mu = out / "mESC_combined.h5mu"
    mdata.write(h5mu)
    log(f"Wrote {h5mu} ({h5mu.stat().st_size/1e9:.2f} GB)")

    # ---------------- Figures ------------------------------------------
    sc.settings.figdir = out
    rna_m = mdata["rna"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    rna_m.obsm["X_umap_tmp"] = rna_m.obsm["X_umap_uncorrected"]
    sc.pl.embedding(rna_m, basis="X_umap_tmp", color="sample", ax=axes[0], show=False,
                    frameon=True, title="Before Harmony (PCA)")
    sc.pl.umap(rna_m, color="sample", ax=axes[1], show=False, frameon=True,
               title="After Harmony")
    for ax in fig.axes:
        if ax.get_label() != "<colorbar>":
            for sp_ in ax.spines.values():
                sp_.set_visible(True); sp_.set_color("#3A424C"); sp_.set_linewidth(1.1)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(out / "umap_harmony_before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    del rna_m.obsm["X_umap_tmp"]

    for color, fn in [("sample", "umap_by_sample.png"),
                      ("timepoint", "umap_by_timepoint.png"),
                      ("per_sample_cell_type", "umap_by_celltype.png"),
                      ("leiden", "umap_by_leiden.png")]:
        fig, ax = plt.subplots(figsize=(9, 7))
        sc.pl.umap(mdata["rna"], color=color, ax=ax, show=False, frameon=True, title=color)
        for sp_ in ax.spines.values():
            sp_.set_visible(True); sp_.set_color("#3A424C"); sp_.set_linewidth(1.1)
        fig.tight_layout()
        fig.savefig(out / fn, dpi=150, bbox_inches="tight")
        plt.close(fig)

    comp = pd.crosstab(mdata["rna"].obs["sample"], mdata["rna"].obs["per_sample_cell_type"])
    comp.to_csv(out / "sample_by_celltype_counts.csv")
    (out / "combine_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("Done.")


if __name__ == "__main__":
    sys.exit(main())
