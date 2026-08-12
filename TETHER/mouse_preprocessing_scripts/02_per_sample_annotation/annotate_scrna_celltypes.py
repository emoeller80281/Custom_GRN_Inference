"""Standard scRNA-seq workflow for a 10x Multiome sample: load -> QC -> normalize
-> cluster -> marker genes -> cell type annotation.

Operates on the Gene Expression modality of a CellRanger-ARC
``matrix.mtx.gz`` / ``features.tsv.gz`` / ``barcodes.tsv.gz`` triplet. The Peaks
modality is split off and written out untouched so downstream ATAC work does not
have to re-read the (very large) combined matrix.

QC thresholds default to the per-sample values recorded in
``data/qc_filtering_settings.tsv`` so this stays consistent with the rest of the
pipeline; MAD-based data-driven thresholds are reported alongside for comparison
but are not applied unless ``--use_mad_thresholds`` is passed.

Example:
    python annotate_scrna_celltypes.py \
        --input_dir /gpfs/Labs/Uzun/DATA/.../mESC_10x_raw/E7.5_rep1 \
        --sample_name E7.5_rep1 --marker_panel mouse_gastrulation
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import sparse

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"


# ---------------------------------------------------------------------------
# Marker panels
# ---------------------------------------------------------------------------
# E7.5-E8.5 mouse embryo (gastrulation / early organogenesis) lineages.
# Sourced from Pijuan-Sala et al. 2019 (Nature) and Argelaguet et al. 2022
# (Nature) mouse gastrulation atlases.
MOUSE_GASTRULATION_MARKERS = {
    "Epiblast": ["Pou5f1", "Utf1", "Slc7a3", "Fgf5", "Sox2", "Nanog", "Pim2"],
    "Primitive streak": ["T", "Tbxt", "Fgf8", "Mixl1", "Eomes", "Evx1"],
    "Nascent mesoderm": ["Mesp1", "Mesp2", "Lefty2", "Mixl1", "Dll3", "Phlda2"],
    "Caudal / somitic mesoderm": ["Tbx6", "Meox1", "Aldh1a2", "Pax3", "Dll1", "Hes7"],
    "Pharyngeal / cardiac mesoderm": ["Nkx2-5", "Tnnt2", "Myl7", "Isl1", "Tbx1", "Hand2"],
    "ExE mesoderm": ["Bmp4", "Postn", "Hand1", "Ahnak", "Pmp22", "Lum"],
    "Allantois": ["Tbx4", "Hoxa10", "Hand1", "Hoxa11", "Vim"],
    "Haematoendothelial prog.": ["Runx1", "Etv2", "Tal1", "Lmo2", "Kdr", "Cdh5", "Pecam1"],
    "Erythroid / blood prog.": ["Hba-x", "Hbb-y", "Hbb-bh1", "Gypa", "Gata1", "Klf1"],
    "Definitive endoderm": ["Sox17", "Foxa2", "Cer1", "Gsc", "Cxcr4", "Hhex"],
    "Gut / visceral endoderm": ["Ttr", "Apoa1", "Afp", "Rhox5", "Trap1a", "Foxa1", "Epcam", "Apoe"],
    "Parietal endoderm": ["Sparc", "Lama1", "Postn", "Col4a1", "Plat"],
    "Notochord": ["Noto", "Shh", "Foxa2", "Chrd", "Nog"],
    "Surface ectoderm": ["Trp63", "Krt8", "Krt18", "Wnt6", "Grhl2", "Dlx5", "Krt19"],
    "Neurectoderm": ["Sox1", "Sox2", "Pax6", "Six3", "En1", "Hesx1", "Nkx1-2", "Sox3"],
    "ExE ectoderm": ["Tfap2c", "Elf5", "Ascl2", "Krt8", "Bmp4", "Cdx2"],
    "Primordial germ cells": ["Dppa3", "Prdm1", "Tfap2c", "Alpl", "Nanos3"],
}

# Cultured mouse embryonic stem cells (2i/LIF or serum/LIF).
MOUSE_ESC_MARKERS = {
    "Naive pluripotent": ["Nanog", "Zfp42", "Klf4", "Klf2", "Esrrb", "Tbx3", "Prdm14"],
    "Core pluripotent": ["Pou5f1", "Sox2", "Utf1", "Sall4", "Lin28a"],
    "Formative / primed": ["Fgf5", "Otx2", "Pou3f1", "Dnmt3b", "Lef1", "Utf1"],
    "Differentiating": ["T", "Tbxt", "Eomes", "Mixl1", "Gata6", "Sox17"],
    "2C-like": ["Zscan4d", "Dux", "Tcstv1", "Zfp352"],
    "Proliferating": ["Mki67", "Top2a", "Ccnb1", "Cdk1"],
}

MARKER_PANELS = {
    "mouse_gastrulation": MOUSE_GASTRULATION_MARKERS,
    "mouse_esc": MOUSE_ESC_MARKERS,
}


def log(msg):
    print(f"[annotate_scrna] {msg}", flush=True)


def frame_panels(fig, color="#3A424C", lw=1.1):
    """Draw a visible border around every data panel of a figure.

    scanpy's UMAP helpers default to ``frameon=False``. In a multi-panel figure
    that leaves neighbouring embeddings visually merged: with points plotted edge
    to edge there is no cue for where one panel stops and the next starts, so it
    is ambiguous which clusters belong to which plot. Colorbar axes are skipped --
    they already read as their own element and a box around them looks wrong.
    """
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>" or getattr(ax, "_colorbar", None) is not None:
            continue
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(color)
            sp.set_linewidth(lw)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_multiome_mtx(input_dir, cache_dir):
    """Read the CellRanger-ARC mtx triplet and split it into RNA and ATAC AnnDatas.

    ``sc.read_10x_mtx`` only keeps the first three columns of ``features.tsv.gz``;
    CellRanger-ARC writes three more (chrom, start, end) that we need for peaks,
    so the feature table is re-read separately and attached to ``.var``.
    """
    input_dir = Path(input_dir)
    sc.settings.cachedir = Path(cache_dir)

    log(f"Reading 10x matrix from {input_dir} (this takes several minutes) ...")
    adata = sc.read_10x_mtx(
        input_dir,
        var_names="gene_symbols",
        make_unique=True,
        gex_only=False,
        cache=True,
    )
    log(f"Loaded combined matrix: {adata.n_obs} barcodes x {adata.n_vars} features")

    features = pd.read_csv(
        input_dir / "features.tsv.gz",
        sep="\t",
        header=None,
        names=["feature_id", "feature_symbol", "feature_type", "chrom", "start", "end"],
    )
    # read_10x_mtx preserves feature order, so positional assignment is safe.
    for col in ["chrom", "start", "end"]:
        adata.var[col] = features[col].values

    rna = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
    atac = adata[:, adata.var["feature_types"] == "Peaks"].copy()
    del adata

    if not sparse.isspmatrix_csr(rna.X):
        rna.X = sparse.csr_matrix(rna.X)
    log(f"RNA modality:  {rna.n_obs} cells x {rna.n_vars} genes")
    log(f"ATAC modality: {atac.n_obs} cells x {atac.n_vars} peaks")
    return rna, atac


# ---------------------------------------------------------------------------
# QC
# ---------------------------------------------------------------------------
def compute_qc_metrics(rna):
    """Annotate mitochondrial / ribosomal / haemoglobin gene sets and run scanpy QC."""
    # Mouse gene symbols: mitochondrial genes are 'mt-', ribosomal 'Rps'/'Rpl'.
    rna.var["mt"] = rna.var_names.str.startswith("mt-")
    rna.var["ribo"] = rna.var_names.str.match(r"^Rp[sl]")
    rna.var["hb"] = rna.var_names.str.match(r"^Hb[ab]-")

    sc.pp.calculate_qc_metrics(
        rna,
        qc_vars=["mt", "ribo", "hb"],
        percent_top=[20],
        log1p=True,
        inplace=True,
    )
    log(
        f"QC gene sets -- mito: {int(rna.var['mt'].sum())}, "
        f"ribo: {int(rna.var['ribo'].sum())}, haemoglobin: {int(rna.var['hb'].sum())}"
    )
    return rna


def mad_outlier_bounds(values, n_mads=5.0, log_transform=False):
    """Median-absolute-deviation bounds, the data-driven alternative to fixed cutoffs."""
    v = np.log1p(values) if log_transform else np.asarray(values, dtype=float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    lo, hi = med - n_mads * mad, med + n_mads * mad
    if log_transform:
        lo, hi = np.expm1(lo), np.expm1(hi)
    return float(lo), float(hi)


QC_TABLE = f"{PROJECT_DIR}/data/qc_filtering_settings.tsv"

# qc_filtering_settings.tsv column -> argparse dest
_QC_COLS = {
    "Min Cells per Gene": "min_cells_per_gene",
    "Min Genes per Cell": "min_genes",
    "Max Genes per Cell": "max_genes",
    "Min Total Counts": "min_counts",
    "Max Total Counts": "max_counts",
    "Max Pct MT": "max_pct_mt",
}


def load_qc_thresholds(sample_name, path=QC_TABLE):
    """Return this sample's row of the project QC table as an argparse-dest dict.

    Returns None when the sample has no row, so the caller can fall back rather
    than silently inheriting another sample's numbers -- inheriting the
    E7.5_rep1 row is exactly how the earlier runs acquired thresholds nobody
    had chosen for them.
    """
    p = Path(path)
    if not p.exists():
        log(f"WARNING: QC table {p} not found")
        return None
    df = pd.read_csv(p, sep="\t")
    row = df[df["Sample"].astype(str) == str(sample_name)]
    if row.empty:
        return None
    r = row.iloc[0]
    vals = {dest: r[col] for col, dest in _QC_COLS.items() if col in row.columns}
    return {k: (float(v) if k == "max_pct_mt" else int(v)) for k, v in vals.items()}


def report_mad_thresholds(rna, n_mads=5.0):
    """Report MAD-derived cutoffs so fixed project thresholds can be sanity checked."""
    report = {}
    for key, logt in [
        ("n_genes_by_counts", True),
        ("total_counts", True),
        ("pct_counts_mt", False),
    ]:
        lo, hi = mad_outlier_bounds(rna.obs[key].values, n_mads=n_mads, log_transform=logt)
        report[key] = {"lower": round(lo, 2), "upper": round(hi, 2)}
        log(f"  MAD({n_mads}) {key}: [{lo:.1f}, {hi:.1f}]")
    return report


def plot_qc(rna, out_dir, tag):
    """Violin + scatter QC panels, written before and after filtering."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, key in zip(
        axes,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo"],
    ):
        sc.pl.violin(rna, key, jitter=0.4, ax=ax, show=False, stripplot=False)
        ax.set_title(key)
    fig.suptitle(f"QC metrics ({tag} filtering)")
    frame_panels(fig)
    fig.tight_layout()
    fig.savefig(out_dir / f"qc_violin_{tag}_filtering.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Drawn with matplotlib rather than sc.pl.scatter: scanpy attaches the colorbar
    # in figure coordinates, which lands it between the two panels and over the
    # right-hand one. append_axes steals the space from the left panel only, so the
    # bar can never intersect its neighbour.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    pts = axes[0].scatter(rna.obs["total_counts"], rna.obs["n_genes_by_counts"],
                          c=rna.obs["pct_counts_mt"], s=3, alpha=0.5, cmap="viridis",
                          linewidths=0, rasterized=True)
    axes[0].set_xlabel("total_counts")
    axes[0].set_ylabel("n_genes_by_counts")
    axes[0].set_title("counts vs genes (coloured by % mito)")
    cax = make_axes_locatable(axes[0]).append_axes("right", size="4%", pad=0.09)
    fig.colorbar(pts, cax=cax).set_label("pct_counts_mt", fontsize=9)

    axes[1].scatter(rna.obs["total_counts"], rna.obs["pct_counts_mt"], s=3, alpha=0.5,
                    color="#57626E", linewidths=0, rasterized=True)
    axes[1].set_xlabel("total_counts")
    axes[1].set_ylabel("pct_counts_mt")
    axes[1].set_title("counts vs % mito")

    frame_panels(fig)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(out_dir / f"qc_scatter_{tag}_filtering.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def qc_filter(rna, args, out_dir):
    """Apply the per-sample QC thresholds and record how many cells each one removes."""
    n_start = rna.n_obs
    obs = rna.obs

    fails = {
        f"n_genes < {args.min_genes}": obs["n_genes_by_counts"] < args.min_genes,
        f"n_genes > {args.max_genes}": obs["n_genes_by_counts"] > args.max_genes,
        f"total_counts < {args.min_counts}": obs["total_counts"] < args.min_counts,
        f"total_counts > {args.max_counts}": obs["total_counts"] > args.max_counts,
        f"pct_mt > {args.max_pct_mt}": obs["pct_counts_mt"] > args.max_pct_mt,
    }
    log("Cells failing each criterion (criteria overlap):")
    for name, mask in fails.items():
        log(f"  {name:<28} {int(mask.sum()):>6} ({100 * mask.mean():.1f}%)")

    keep = ~np.logical_or.reduce([m.values for m in fails.values()])
    rna = rna[keep].copy()
    log(f"Cells: {n_start} -> {rna.n_obs} ({100 * rna.n_obs / n_start:.1f}% retained)")

    n_genes_start = rna.n_vars
    # Marker-panel genes are exempt from the detection floor. A rare lineage's
    # defining genes are precisely the ones a per-gene min_cells cut deletes
    # (Dppa3/Nanos3 for PGCs, Hba-x/Hbb-y for primitive erythroid), which makes
    # that lineage unannotatable even when it is present.
    panel_genes = {g for genes in MARKER_PANELS[args.marker_panel].values() for g in genes}
    n_cells_per_gene = np.asarray((rna.X > 0).sum(axis=0)).ravel()
    keep_gene = n_cells_per_gene >= args.min_cells_per_gene
    rescued = rna.var_names.isin(panel_genes) & ~keep_gene & (n_cells_per_gene > 0)
    keep_gene = keep_gene | rescued
    rna = rna[:, keep_gene].copy()
    log(f"Genes: {n_genes_start} -> {rna.n_vars} (min_cells={args.min_cells_per_gene}, "
        f"{int(rescued.sum())} marker genes kept below the floor)")

    summary = {
        "n_cells_start": int(n_start),
        "n_cells_after_qc": int(rna.n_obs),
        "n_genes_start": int(n_genes_start),
        "n_genes_after_qc": int(rna.n_vars),
        "failed_per_criterion": {k: int(v.sum()) for k, v in fails.items()},
        "n_marker_genes_rescued": int(rescued.sum()),
    }
    return rna, summary


def detect_doublets(rna, args):
    """Scrublet doublet scoring, with a fallback when its automatic cutoff fails.

    Scrublet picks its threshold from the valley of the *simulated* doublet score
    histogram. When that histogram is not bimodal it silently returns a threshold
    above every observed score, calls ~zero doublets, and the sample then looks
    pristine in the run summary. That failure mode hit 4 of the 11 mESC samples
    (E8.5_rep2 called 0/11,107), so a low call rate is treated as a failure and
    the threshold is re-derived from the expected rate instead.
    """
    if args.skip_doublets:
        log("Doublet detection skipped (--skip_doublets)")
        return rna, None, "skipped"
    try:
        sc.pp.scrublet(rna, expected_doublet_rate=args.expected_doublet_rate,
                       random_state=args.seed)
    except Exception as exc:  # noqa: BLE001 - optional step, keep pipeline running
        log(f"WARNING: scrublet failed ({type(exc).__name__}: {exc}); skipping doublet removal")
        return rna, None, "failed"

    method = "scrublet_auto"
    n_doublet = int(rna.obs["predicted_doublet"].sum())
    rate = n_doublet / rna.n_obs
    auto_thr = rna.uns.get("scrublet", {}).get("threshold", float("nan"))
    log(f"Scrublet auto: {n_doublet} doublets ({100*rate:.1f}%), threshold={auto_thr:.3f}, "
        f"score max={rna.obs['doublet_score'].max():.3f}")

    # A call rate far under the expected rate means the cutoff never landed.
    if rate < args.min_doublet_rate:
        q = 1.0 - args.expected_doublet_rate
        thr = float(np.quantile(rna.obs["doublet_score"].values, q))
        rna.obs["predicted_doublet"] = rna.obs["doublet_score"].values > thr
        n_doublet = int(rna.obs["predicted_doublet"].sum())
        method = "expected_rate_quantile"
        log(f"  WARNING: auto threshold failed ({100*rate:.2f}% < "
            f"{100*args.min_doublet_rate:.1f}% floor). Re-thresholded at the "
            f"{100*q:.1f}th score percentile ({thr:.3f}) -> {n_doublet} doublets "
            f"({100*n_doublet/rna.n_obs:.1f}%)")
        rna.uns["scrublet_fallback_threshold"] = thr

    if args.remove_doublets:
        rna = rna[~rna.obs["predicted_doublet"]].copy()
        log(f"Removed doublets -> {rna.n_obs} cells")
    return rna, n_doublet, method


# ---------------------------------------------------------------------------
# Normalize / cluster
# ---------------------------------------------------------------------------
def normalize_and_reduce(rna, args):
    """Library-size normalize, log1p, HVG selection, PCA, neighbours, UMAP."""
    rna.layers["counts"] = rna.X.copy()  # raw counts kept for DE / scvi-tools

    sc.pp.normalize_total(rna, target_sum=args.target_sum)
    sc.pp.log1p(rna)
    rna.raw = rna  # log-normalized all-gene snapshot for marker plots
    log(f"Normalized to {args.target_sum:g} counts/cell and log1p-transformed")

    # flavor='seurat_v3' needs scikit-misc; fall back to the log-space 'seurat' flavor.
    try:
        import skmisc  # noqa: F401

        sc.pp.highly_variable_genes(
            rna, n_top_genes=args.n_hvg, flavor="seurat_v3", layer="counts"
        )
        hvg_flavor = "seurat_v3"
    except ImportError:
        sc.pp.highly_variable_genes(rna, n_top_genes=args.n_hvg, flavor="seurat")
        hvg_flavor = "seurat"
    log(f"HVGs: {int(rna.var['highly_variable'].sum())} (flavor={hvg_flavor})")

    sc.pp.scale(rna, max_value=10)
    sc.tl.pca(rna, n_comps=args.n_pcs, svd_solver="arpack", mask_var="highly_variable")
    sc.pp.neighbors(rna, n_neighbors=args.n_neighbors, n_pcs=args.n_pcs, random_state=args.seed)
    sc.tl.umap(rna, random_state=args.seed)
    log(f"PCA ({args.n_pcs} PCs) + neighbours ({args.n_neighbors}) + UMAP done")

    # Scaled values are only needed for PCA; restore log-normalized X for plots/DE.
    rna.layers["scaled"] = rna.X.copy()
    rna.X = rna.raw.X[:, [rna.raw.var_names.get_loc(g) for g in rna.var_names]].copy()
    return rna, hvg_flavor


def cluster(rna, args):
    """Leiden at the primary resolution plus a sweep for granularity comparison."""
    resolutions = sorted({args.resolution, *args.extra_resolutions})
    for res in resolutions:
        key = "leiden" if res == args.resolution else f"leiden_res{res:g}"
        sc.tl.leiden(
            rna, resolution=res, key_added=key, flavor="igraph", n_iterations=2,
            directed=False, random_state=args.seed,
        )
        log(f"Leiden res={res:g} -> {rna.obs[key].nunique()} clusters (key='{key}')")
    return rna


# ---------------------------------------------------------------------------
# Markers / annotation
# ---------------------------------------------------------------------------
def find_markers(rna, out_dir, groupby="leiden"):
    """Wilcoxon rank-sum marker genes per cluster, with BH-FDR adjusted p-values."""
    sc.tl.rank_genes_groups(
        rna, groupby=groupby, method="wilcoxon", use_raw=False, pts=True,
        tie_correct=False,
    )
    df = sc.get.rank_genes_groups_df(rna, group=None)
    df = df.rename(columns={"group": "cluster", "names": "gene"})
    df.to_csv(out_dir / "marker_genes_all.csv", index=False)

    sig = df[(df["pvals_adj"] < 0.05) & (df["logfoldchanges"] > 0.5)]
    top = sig.groupby("cluster", observed=True).head(25)
    top.to_csv(out_dir / "marker_genes_top25.csv", index=False)
    log(f"Markers: {len(df)} tests, {len(sig)} significant (FDR<0.05, log2FC>0.5)")
    return df, top


def score_marker_panels(rna, panel, out_dir, groupby="leiden"):
    """Score each cell against every lineage panel, then assign clusters by z-scored mean.

    Z-scoring each signature across clusters keeps broad panels (which produce
    uniformly high raw scores) from winning every cluster.
    """
    present = {}
    for ct, genes in panel.items():
        found = [g for g in genes if g in rna.var_names]
        if not found:
            log(f"  WARNING: no marker genes found for '{ct}' -- skipping")
            continue
        present[ct] = found
        sc.tl.score_genes(rna, found, score_name=f"score_{ct}", use_raw=False)
        missing = sorted(set(genes) - set(found))
        log(f"  {ct}: {len(found)}/{len(genes)} markers found" + (f" (missing: {', '.join(missing)})" if missing else ""))

    score_cols = [f"score_{ct}" for ct in present]
    per_cluster = rna.obs.groupby(groupby, observed=True)[score_cols].mean()
    per_cluster.columns = list(present)

    # Z-score each signature across clusters, then take the argmax per cluster.
    z = (per_cluster - per_cluster.mean(axis=0)) / per_cluster.std(axis=0).replace(0, np.nan)
    assignment = z.idxmax(axis=1)
    confidence = z.max(axis=1)
    runner_up = z.apply(lambda r: r.nlargest(2).iloc[-1], axis=1)
    margin = confidence - runner_up

    ann = pd.DataFrame({
        "cluster": per_cluster.index,
        "n_cells": rna.obs[groupby].value_counts().reindex(per_cluster.index).values,
        "cell_type": assignment.values,
        "z_score": confidence.round(3).values,
        "margin_over_runner_up": margin.round(3).values,
        "runner_up": z.apply(lambda r: r.nlargest(2).index[-1], axis=1).values,
    })
    ann.to_csv(out_dir / "cluster_cell_type_annotation.csv", index=False)
    per_cluster.round(4).to_csv(out_dir / "cluster_marker_scores_raw.csv")
    z.round(4).to_csv(out_dir / "cluster_marker_scores_zscored.csv")

    rna.obs["cell_type"] = rna.obs[groupby].map(assignment).astype("category")
    # Clusters whose top signature barely beats the runner-up are flagged, not renamed.
    low_conf = set(ann.loc[ann["margin_over_runner_up"] < 0.5, "cluster"])
    rna.obs["cell_type_confident"] = (~rna.obs[groupby].isin(low_conf)).values

    log("\nCluster -> cell type assignment:")
    for _, r in ann.iterrows():
        flag = "" if r["margin_over_runner_up"] >= 0.5 else "   <- LOW CONFIDENCE"
        log(f"  cluster {r['cluster']:>3} (n={r['n_cells']:>5}): {r['cell_type']:<30} "
            f"z={r['z_score']:.2f} margin={r['margin_over_runner_up']:.2f}{flag}")
    return rna, ann, per_cluster, z, present


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(rna, top_markers, panel_present, z_scores, out_dir, groupby="leiden"):
    sc.settings.figdir = out_dir

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    sc.pl.umap(rna, color=groupby, legend_loc="on data", legend_fontsize=9,
               title="Leiden clusters", ax=axes[0], show=False, frameon=True)
    sc.pl.umap(rna, color="cell_type", title="Annotated cell type",
               ax=axes[1], show=False, frameon=True)
    frame_panels(fig)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(out_dir / "umap_clusters_and_celltypes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    sc.pl.umap(rna, color=["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo"],
               ncols=4, show=False, frameon=True)
    fig = plt.gcf()
    frame_panels(fig)
    fig.subplots_adjust(wspace=0.45)
    plt.savefig(out_dir / "umap_qc_metrics.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Top 5 DE genes per cluster.
    sc.pl.rank_genes_groups_dotplot(rna, n_genes=5, groupby=groupby, standard_scale="var",
                                    show=False)
    plt.savefig(out_dir / "dotplot_top_markers_per_cluster.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Canonical panel markers, grouped by lineage.
    sc.pl.dotplot(rna, panel_present, groupby=groupby, standard_scale="var",
                  show=False, figsize=(22, 0.45 * rna.obs[groupby].nunique() + 3))
    plt.savefig(out_dir / "dotplot_canonical_lineage_markers.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # Signature score heatmap (z-scored across clusters).
    fig, ax = plt.subplots(figsize=(0.55 * z_scores.shape[1] + 4, 0.35 * z_scores.shape[0] + 3))
    im = ax.imshow(z_scores.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    ax.set_xticks(range(z_scores.shape[1]))
    ax.set_xticklabels(z_scores.columns, rotation=90, fontsize=9)
    ax.set_yticks(range(z_scores.shape[0]))
    ax.set_yticklabels([f"cluster {c}" for c in z_scores.index], fontsize=9)
    ax.set_title("Lineage signature scores (z-scored across clusters)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="z-score")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_lineage_signature_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    n_res = [c for c in rna.obs.columns if c.startswith("leiden_res")]
    if n_res:
        sc.pl.umap(rna, color=n_res, ncols=3, legend_loc="on data", legend_fontsize=7,
                   show=False, frameon=True)
        fig = plt.gcf()
        frame_panels(fig)
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.savefig(out_dir / "umap_resolution_sweep.png", dpi=150, bbox_inches="tight")
        plt.close("all")
    log(f"Plots written to {out_dir}")


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True, help="Dir with matrix/features/barcodes .gz")
    p.add_argument("--sample_name", required=True)
    p.add_argument("--out_dir", default=None,
                   help="Default: <PROJECT_DIR>/data/processed/<sample_name>_scrna")
    p.add_argument("--cache_dir", default=None, help="scanpy mtx read cache (default: out_dir/cache)")

    # QC thresholds -- these are an arbitrary last-resort fallback, not tied to any one
    # sample's row. load_qc_thresholds() overwrites them via setattr for every sample present
    # in --qc_table (every production sample is), so in practice they only fire if a sample is
    # missing from the TSV. Don't resync these to a specific sample's row -- the TSV changes
    # independently and any such pairing will just go stale again.
    p.add_argument("--min_genes", type=int, default=1500)
    p.add_argument("--max_genes", type=int, default=6000)
    p.add_argument("--min_counts", type=int, default=1000)
    p.add_argument("--max_counts", type=int, default=25000)
    p.add_argument("--max_pct_mt", type=float, default=20.0)
    p.add_argument("--min_cells_per_gene", type=int, default=20)
    p.add_argument("--use_mad_thresholds", action="store_true",
                   help="Use MAD data-driven cutoffs instead of the QC table")
    p.add_argument("--n_mads", type=float, default=5.0)
    p.add_argument("--mito_ceiling", type=float, default=30.0,
                   help="Absolute cap the MAD mitochondrial bound may never exceed. "
                        "MAD is a relative rule, so on a globally high-mito sample it "
                        "lands above the bulk of the data and filters nothing.")
    p.add_argument("--qc_table", default=QC_TABLE,
                   help="TSV of per-sample thresholds (default: data/qc_filtering_settings.tsv)")
    p.add_argument("--no_qc_table", action="store_true",
                   help="Ignore the QC table and use the CLI threshold values")

    p.add_argument("--skip_doublets", action="store_true", help="Do not run scrublet at all")
    p.add_argument("--keep_doublets", dest="remove_doublets", action="store_false",
                   help="Score doublets but keep them (they are still flagged in .obs)")
    p.add_argument("--expected_doublet_rate", type=float, default=0.08)
    p.add_argument("--min_doublet_rate", type=float, default=0.02,
                   help="If scrublet's automatic threshold calls fewer than this "
                        "fraction, treat it as a failed fit and re-threshold at the "
                        "expected-rate quantile of the score distribution.")

    p.add_argument("--target_sum", type=float, default=1e4)
    p.add_argument("--n_hvg", type=int, default=3000)
    p.add_argument("--n_pcs", type=int, default=50)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--extra_resolutions", type=float, nargs="*", default=[0.4, 0.6, 2.0])

    p.add_argument("--marker_panel", default="mouse_gastrulation", choices=list(MARKER_PANELS))
    p.add_argument("--no_save_atac", dest="save_atac", action="store_false",
                   help="Skip writing the split-off raw ATAC modality")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    sc.settings.verbosity = 1
    sc.settings.n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir or f"{PROJECT_DIR}/data/processed/{args.sample_name}_scrna")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir or out_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"Sample: {args.sample_name}   out_dir: {out_dir}   n_jobs: {sc.settings.n_jobs}")

    # 1. Load -------------------------------------------------------------
    rna, atac = load_multiome_mtx(args.input_dir, cache_dir)
    rna.obs["sample"] = args.sample_name
    if args.save_atac:
        atac.write_h5ad(out_dir / f"{args.sample_name}_atac_raw.h5ad", compression="gzip")
        log(f"Wrote raw ATAC modality -> {args.sample_name}_atac_raw.h5ad")
    del atac

    # 2. QC ---------------------------------------------------------------
    rna = compute_qc_metrics(rna)
    plot_qc(rna, out_dir, "pre")
    mad_report = report_mad_thresholds(rna, n_mads=args.n_mads)

    threshold_source = "cli_defaults"
    if args.use_mad_thresholds:
        # Clamp to the observed range: 5 MADs on a log-scaled skewed distribution
        # can land past every real value, leaving a bound that filters nothing
        # (E8.5_rep2 got max_genes=39,114 against 32,285 genes in the reference).
        obs_max_g = int(rna.obs["n_genes_by_counts"].max())
        obs_max_c = int(rna.obs["total_counts"].max())
        args.min_genes = max(1, int(mad_report["n_genes_by_counts"]["lower"]))
        args.max_genes = min(int(mad_report["n_genes_by_counts"]["upper"]), obs_max_g)
        args.min_counts = max(1, int(mad_report["total_counts"]["lower"]))
        args.max_counts = min(int(mad_report["total_counts"]["upper"]), obs_max_c)
        args.max_pct_mt = min(float(mad_report["pct_counts_mt"]["upper"]), args.mito_ceiling)
        threshold_source = "mad"
        log(f"Using {args.n_mads}-MAD thresholds (clamped to data; mito ceiling "
            f"{args.mito_ceiling}%)")
    elif not args.no_qc_table:
        tsv = load_qc_thresholds(args.sample_name, args.qc_table)
        if tsv is None:
            log(f"WARNING: no row for '{args.sample_name}' in {args.qc_table}; "
                "falling back to CLI defaults")
        else:
            for k, v in tsv.items():
                setattr(args, k, v)
            threshold_source = "qc_filtering_settings.tsv"
            log(f"Using thresholds from {args.qc_table} for {args.sample_name}")

    log("Thresholds applied: " + ", ".join(
        f"{k}={getattr(args, k)}" for k in
        ["min_genes", "max_genes", "min_counts", "max_counts", "max_pct_mt",
         "min_cells_per_gene"]))

    # Doublet detection first. Scrublet models the observed cell population, so
    # discarding a third of it beforehand starves the simulated-doublet
    # distribution its threshold is read from; the earlier ordering under-called
    # doublets badly (1.5% on E7.5_rep1 against an 8% expectation).
    n_barcodes_in = int(rna.n_obs)
    rna, n_doublets, doublet_method = detect_doublets(rna, args)
    n_after_doublets = int(rna.n_obs)
    rna, qc_summary = qc_filter(rna, args, out_dir)
    # qc_filter measured its criteria against the post-doublet population; keep both
    # counts so retention is reported against the barcodes actually loaded.
    qc_summary["n_cells_criteria_denominator"] = qc_summary["n_cells_start"]
    qc_summary["n_cells_start"] = n_barcodes_in
    qc_summary["n_cells_after_doublets"] = n_after_doublets
    qc_summary["n_predicted_doublets"] = n_doublets
    qc_summary["doublet_method"] = doublet_method
    qc_summary["n_cells_final"] = int(rna.n_obs)
    plot_qc(rna, out_dir, "post")

    # 3. Normalize + cluster ----------------------------------------------
    rna, hvg_flavor = normalize_and_reduce(rna, args)
    rna = cluster(rna, args)

    # 4. Markers -----------------------------------------------------------
    _, top_markers = find_markers(rna, out_dir)

    # 5. Annotate ----------------------------------------------------------
    log(f"\nScoring '{args.marker_panel}' marker panel:")
    rna, ann, _, z_scores, panel_present = score_marker_panels(
        rna, MARKER_PANELS[args.marker_panel], out_dir
    )

    # 6. Plots + outputs ---------------------------------------------------
    make_plots(rna, top_markers, panel_present, z_scores, out_dir)

    h5ad_path = out_dir / f"{args.sample_name}_rna_processed.h5ad"
    rna.write_h5ad(h5ad_path, compression="gzip")
    log(f"Wrote processed RNA AnnData -> {h5ad_path}")

    summary = {
        "sample": args.sample_name,
        "input_dir": str(args.input_dir),
        "qc": qc_summary,
        "mad_thresholds_reported": mad_report,
        "threshold_source": threshold_source,
        "thresholds_applied": {
            k: getattr(args, k) for k in
            ["min_genes", "max_genes", "min_counts", "max_counts", "max_pct_mt",
             "min_cells_per_gene", "use_mad_thresholds", "mito_ceiling"]
        },
        "hvg_flavor": hvg_flavor,
        "n_hvg": int(rna.var["highly_variable"].sum()),
        "n_pcs": args.n_pcs,
        "resolution": args.resolution,
        "n_clusters": int(rna.obs["leiden"].nunique()),
        "marker_panel": args.marker_panel,
        "cell_type_counts": rna.obs["cell_type"].value_counts().to_dict(),
        "scanpy_version": sc.__version__,
        "seed": args.seed,
    }
    with open(out_dir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    log("\n=== Cell type composition ===")
    for ct, n in rna.obs["cell_type"].value_counts().items():
        log(f"  {ct:<32} {n:>5} cells ({100 * n / rna.n_obs:.1f}%)")
    log(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
