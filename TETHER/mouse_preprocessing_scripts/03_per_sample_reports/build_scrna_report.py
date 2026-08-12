"""Build a self-contained HTML report for one ``annotate_scrna_celltypes.py`` run.

Reads the output directory of a single sample, recomputes a few diagnostics that
the pipeline does not save (cell-cycle phase scores, marker-panel attrition from
the ``min_cells`` gene filter, ambient-RNA proxy), embeds the PNGs as data URIs
and writes ``report.html`` plus a machine-readable ``report_stats.json``.

The HTML is written as a body fragment (no <html>/<head>/<body>) because the
Artifact publisher wraps it in its own document skeleton.

Example:
    python build_scrna_report.py --sample_dir data/processed/mESC/E7.5_rep1 \
        --sample_name E7.5_rep1
"""

import argparse
import base64
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_per_sample_annotation"))
from annotate_scrna_celltypes import MOUSE_GASTRULATION_MARKERS  # noqa: E402

# Tirosh et al. 2016 cell-cycle genes, mouse-cased.
S_GENES = """Mcm5 Pcna Tyms Fen1 Mcm2 Mcm4 Rrm1 Ung Gins2 Mcm6 Cdca7 Dtl Prim1 Uhrf1
Hells Rfc2 Rpa2 Nasp Rad51ap1 Gmnn Wdr76 Slbp Ccne2 Ubr7 Pold3 Msh2 Atad2 Rad51 Rrm2
Cdc45 Cdc6 Exo1 Tipin Dscc1 Blm Casp8ap2 Usp1 Clspn Pola1 Chaf1b Brip1 E2f8""".split()
G2M_GENES = """Hmgb2 Cdk1 Nusap1 Ube2c Birc5 Tpx2 Top2a Ndc80 Cks2 Nuf2 Cks1b Mki67 Tmpo
Cenpf Tacc3 Smc4 Ccnb2 Ckap2l Ckap2 Aurkb Bub1 Kif11 Anp32e Tubb4b Gtse1 Kif20b Hjurp
Cdca3 Cdc20 Ttk Cdc25c Kif2c Rangap1 Ncapd2 Dlgap5 Cdca2 Cdca8 Ect2 Kif23 Hmmr Aurka
Psrc1 Anln Lbr Ckap5 Cenpe Ctcf Nek2 G2e3 Gas2l3 Cbx5 Cenpa""".split()

# Classic visceral-endoderm genes: very high expressers, so a uniform low-level
# floor across every cluster is the signature of ambient ("soup") contamination.
AMBIENT_PROBES = ["Ttr", "Apoa1", "Afp", "Apoe"]

# Germ-layer grouping drives the colour coding in the annotation table.
LAYER = {
    "Epiblast": "ecto", "Neurectoderm": "ecto", "Surface ectoderm": "ecto",
    "Primitive streak": "meso", "Nascent mesoderm": "meso",
    "Caudal / somitic mesoderm": "meso", "Pharyngeal / cardiac mesoderm": "meso",
    "Notochord": "meso", "Haematoendothelial prog.": "meso",
    "Erythroid / blood prog.": "meso",
    "Definitive endoderm": "endo", "Gut / visceral endoderm": "endo",
    "Parietal endoderm": "endo",
    "ExE mesoderm": "exe", "ExE ectoderm": "exe", "Allantois": "exe",
    "Primordial germ cells": "exe",
}


def log(m):
    print(f"[report] {m}", flush=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def compute_stats(sample_dir, sample_name):
    import anndata as ad
    import scanpy as sc

    d = Path(sample_dir)
    summary = json.loads((d / "run_summary.json").read_text())
    ann = pd.read_csv(d / "cluster_cell_type_annotation.csv")
    ann["cluster"] = ann["cluster"].astype(str)
    z = pd.read_csv(d / "cluster_marker_scores_zscored.csv", index_col=0)
    z.index = z.index.astype(str)
    top = pd.read_csv(d / "marker_genes_top25.csv")
    top["cluster"] = top["cluster"].astype(str)

    log("loading h5ad ...")
    a = ad.read_h5ad(d / f"{sample_name}_rna_processed.h5ad")

    # --- cell cycle -------------------------------------------------------
    s = [g for g in S_GENES if g in a.var_names]
    g2m = [g for g in G2M_GENES if g in a.var_names]
    sc.tl.score_genes_cell_cycle(a, s_genes=s, g2m_genes=g2m)
    cc = a.obs.groupby("leiden", observed=True)[["S_score", "G2M_score"]].mean()
    cc["pct_S"] = (
        a.obs.assign(p=a.obs["phase"] == "S")
        .groupby("leiden", observed=True)["p"].mean() * 100
    )
    cc.index = cc.index.astype(str)

    # A cluster is "cycling-driven" when its S score stands far above the rest.
    others = cc["S_score"]
    cycling = []
    for cl in cc.index:
        rest = others.drop(cl)
        if cc.loc[cl, "S_score"] > rest.max() + 0.08 and cc.loc[cl, "pct_S"] > 50:
            cycling.append(cl)

    # --- marker attrition from the min_cells gene filter -------------------
    present = set(a.var_names)
    dropped = {}
    coverage = {}
    for ct, genes in MOUSE_GASTRULATION_MARKERS.items():
        found = [g for g in genes if g in present]
        miss = [g for g in genes if g not in present]
        coverage[ct] = (len(found), len(genes))
        if miss:
            dropped[ct] = miss

    # --- ambient proxy ----------------------------------------------------
    probes = [g for g in AMBIENT_PROBES if g in present]
    ambient = {}
    if probes:
        sub = a[:, probes].X
        sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        det = pd.DataFrame(sub > 0, columns=probes, index=a.obs_names)
        det["cl"] = a.obs["leiden"].values
        per = det.groupby("cl", observed=True).mean() * 100
        # Floor = the *lowest* per-cluster detection rate. A high floor means even
        # the least-endodermal cluster still detects these genes -> soup.
        ambient = {g: float(per[g].min()) for g in probes}

    # --- did the MAD mito cap actually remove anything? -------------------
    # MAD is a *relative* outlier rule. On a sample whose mitochondrial fraction
    # is high throughout, the derived cap lands above the bulk of the
    # distribution and filters almost nothing, which is invisible from the
    # retention percentage alone.
    mt = a.obs["pct_counts_mt"].values
    mito = {
        "median_kept": float(np.median(mt)),
        "cap": float(summary["thresholds_applied"]["max_pct_mt"]),
        "n_gt20": int((mt > 20).sum()),
        "pct_gt20": round(100 * float((mt > 20).mean()), 1),
        "n_gt35": int((mt > 35).sum()),
    }

    # --- did scrublet actually pick a threshold? --------------------------
    # Scrublet's automatic cutoff needs a bimodal simulated-doublet histogram.
    # When it fails it silently calls ~nothing, which looks like a clean sample.
    nd = summary["qc"].get("n_predicted_doublets") or 0
    ds = a.obs["doublet_score"].values if "doublet_score" in a.obs else np.array([np.nan])
    doublet = {
        "n_called": int(nd),
        "pct_called": round(100 * nd / a.n_obs, 2),
        "score_median": float(np.nanmedian(ds)),
        "score_max": float(np.nanmax(ds)),
        # Called <1% yet the score distribution has a long right tail -> suspect.
        "suspect": bool(nd / a.n_obs < 0.01 and np.nanmax(ds) > 0.4),
    }

    # --- vacuous upper bounds --------------------------------------------
    n_genes_start = summary["qc"]["n_genes_start"]
    vacuous = summary["thresholds_applied"]["max_genes"] > n_genes_start

    # --- confidence -------------------------------------------------------
    conf = a.obs["cell_type_confident"].astype(bool)
    low_cells = int((~conf).sum())
    low_clusters = sorted(ann.loc[ann["margin_over_runner_up"] < 0.5, "cluster"],
                          key=lambda x: int(x))

    assigned = set(ann["cell_type"])
    never = sorted(set(z.columns) - assigned)
    never_detail = [(c, float(z[c].max()), str(z[c].idxmax())) for c in never]
    never_detail.sort(key=lambda t: -t[1])

    n_tests = len(pd.read_csv(d / "marker_genes_all.csv", usecols=["gene"]))

    # top DE genes per cluster for the annotation table
    top_by_cl = {
        cl: ", ".join(g.head(6).tolist())
        for cl, g in top.groupby("cluster", observed=True)["gene"]
    }

    # --- driver-gene check ------------------------------------------------
    # The z-score margin only says one marker panel outscored the others in aggregate; it
    # says nothing about whether that panel's own genes are what's actually differential in
    # this cluster. That gap is exactly what let the rejected annotate_combined_pijuansala.py
    # (05_atlas_annotation, see handoff.md's annotation-history section) call cells PGC on a
    # *confident* margin while the panel's two most specific genes, Dppa3 and Nanos3, were
    # flat zero -- driven instead by non-specific Prdm1/Ifitm3. This per-sample scorer uses
    # the identical statistical pattern, so the same failure is possible here and the existing
    # margin<0.5 caveat below would not catch it (that fires in the opposite direction).
    top_genes_by_cl = {
        cl: set(g.tolist()) for cl, g in top.groupby("cluster", observed=True)["gene"]
    }
    driver_mismatch = sorted(
        (
            r["cluster"] for _, r in ann.iterrows()
            if MOUSE_GASTRULATION_MARKERS.get(r["cell_type"])
            and not (set(MOUSE_GASTRULATION_MARKERS[r["cell_type"]])
                     & top_genes_by_cl.get(r["cluster"], set()))
        ),
        key=lambda x: int(x),
    )

    stats = {
        "sample": sample_name,
        "summary": summary,
        "n_cells": int(a.n_obs),
        "n_genes": int(a.n_vars),
        "n_clusters": int(a.obs["leiden"].nunique()),
        "n_lineages": len(assigned),
        "low_conf_cells": low_cells,
        "low_conf_pct": round(100 * low_cells / a.n_obs, 1),
        "low_conf_clusters": low_clusters,
        "driver_mismatch_clusters": driver_mismatch,
        "mito": mito,
        "doublet": doublet,
        "vacuous_max_genes": vacuous,
        "cycling_clusters": cycling,
        "cell_cycle": cc.round(3).to_dict("index"),
        "dropped_markers": dropped,
        "coverage": coverage,
        "ambient_floor": ambient,
        "never_assigned": never_detail,
        "n_tests": n_tests,
        "resolution_sweep": {
            c: int(a.obs[c].nunique())
            for c in ["leiden_res0.4", "leiden_res0.6", "leiden", "leiden_res2"]
            if c in a.obs
        },
        "annotation": ann.to_dict("records"),
        "top_markers": top_by_cl,
    }
    (d / "report_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    return stats


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = """
:root{
  --ground:#F5F7FA; --surface:#FFFFFF; --surface-2:#EBF0F6;
  --ink:#131A24; --muted:#57626E; --faint:#7C8794; --rule:#D9E0E9;
  --ecto:#2E5EA6; --meso:#AF4438; --endo:#9C7412; --exe:#6A5789;
  --accent:#2E5EA6; --accent-soft:#E3EAF6; --accent-line:#B9CBE7;
  --warn:#8C5410; --warn-soft:#F8EEDD; --warn-line:#E3C89A;
  --ok:#2F6B4F;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --measure:74ch;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0E141A; --surface:#161E27; --surface-2:#1D2731;
    --ink:#E2E8F0; --muted:#9AA7B4; --faint:#7C8794; --rule:#26313C;
    --ecto:#7BA5E4; --meso:#DE8378; --endo:#D2A63F; --exe:#A793C9;
    --accent:#7BA5E4; --accent-soft:#1B2839; --accent-line:#31486B;
    --warn:#D7A45C; --warn-soft:#2A2214; --warn-line:#4E3E22;
    --ok:#6FBF95;
  }
}
:root[data-theme="dark"]{
  --ground:#0E141A; --surface:#161E27; --surface-2:#1D2731;
  --ink:#E2E8F0; --muted:#9AA7B4; --faint:#7C8794; --rule:#26313C;
  --ecto:#7BA5E4; --meso:#DE8378; --endo:#D2A63F; --exe:#A793C9;
  --accent:#7BA5E4; --accent-soft:#1B2839; --accent-line:#31486B;
  --warn:#D7A45C; --warn-soft:#2A2214; --warn-line:#4E3E22;
  --ok:#6FBF95;
}
*{box-sizing:border-box;}
body{
  background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:16.5px; line-height:1.65;
  margin:0; padding:0 24px 96px; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:var(--measure); margin:0 auto; display:flex; flex-direction:column; gap:34px;}
.bleed{width:min(1180px,calc(100vw - 48px)); margin-left:50%; transform:translateX(-50%);}
h1,h2,h3{font-family:var(--serif); font-weight:600; text-wrap:balance; margin:0; letter-spacing:-.01em;}
h1{font-size:clamp(2rem,4.6vw,2.9rem); line-height:1.14;}
h2{font-size:1.45rem; line-height:1.25;}
h3{font-size:1.06rem; line-height:1.3;}
p{margin:0;}
a{color:var(--accent);}
.eyebrow{font-family:var(--mono); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:var(--faint);}
.lede{font-size:1.12rem; color:var(--muted); max-width:62ch;}
code,.mono{font-family:var(--mono); font-size:.88em;}
code{background:var(--surface-2); padding:.12em .38em; border-radius:3px;}
header{padding:56px 0 6px; display:flex; flex-direction:column; gap:14px;}
.meta{display:flex; flex-wrap:wrap; gap:8px 22px; font-family:var(--mono); font-size:.78rem;
  color:var(--muted); padding-top:14px; border-top:1px solid var(--rule);}
.meta b{color:var(--ink); font-weight:600;}
section{display:flex; flex-direction:column; gap:16px;}
.sec-head{display:flex; align-items:baseline; gap:12px; border-bottom:1px solid var(--rule); padding-bottom:9px;}
.stage-no{font-family:var(--mono); font-size:.74rem; color:var(--accent);
  border:1px solid var(--accent-line); border-radius:2px; padding:2px 6px; flex:none;}
.callout{background:var(--warn-soft); border:1px solid var(--warn-line); border-left:3px solid var(--warn);
  border-radius:4px; padding:20px 22px; display:flex; flex-direction:column; gap:10px;}
.callout .eyebrow{color:var(--warn);}
.callout h3{color:var(--warn);}
.stats{display:grid; grid-template-columns:repeat(auto-fit,minmax(132px,1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); border-radius:4px; overflow:hidden;}
.stat{background:var(--surface); padding:15px 16px; display:flex; flex-direction:column; gap:3px;}
.stat .n{font-family:var(--serif); font-size:1.72rem; line-height:1.1; font-variant-numeric:tabular-nums;}
.stat .l{font-family:var(--mono); font-size:.68rem; letter-spacing:.09em; text-transform:uppercase; color:var(--faint);}
.tbl-scroll{overflow-x:auto; border:1px solid var(--rule); border-radius:4px; background:var(--surface);}
table{border-collapse:collapse; width:100%; font-size:.87rem;}
th,td{padding:9px 13px; text-align:left; border-bottom:1px solid var(--rule); white-space:nowrap;}
thead th{font-family:var(--mono); font-size:.68rem; letter-spacing:.08em; text-transform:uppercase;
  color:var(--faint); font-weight:500; background:var(--surface-2);}
tbody tr:last-child td{border-bottom:none;}
.num{text-align:right; font-variant-numeric:tabular-nums; font-family:var(--mono); font-size:.83rem;}
td.genes{font-family:var(--mono); font-size:.78rem; color:var(--muted); white-space:normal; min-width:24ch;}
.chip{display:inline-flex; align-items:center; gap:7px;}
.dot{width:9px; height:9px; border-radius:50%; flex:none;}
.d-ecto{background:var(--ecto);} .d-meso{background:var(--meso);}
.d-endo{background:var(--endo);} .d-exe{background:var(--exe);}
.flag{font-family:var(--mono); font-size:.68rem; letter-spacing:.05em; padding:2px 7px;
  border-radius:2px; border:1px solid currentColor;}
.f-low{color:var(--warn); background:var(--warn-soft);}
.f-ok{color:var(--ok); background:transparent; border-color:transparent; padding-left:0;}
figure{margin:0; display:flex; flex-direction:column; gap:10px;}
figure img{width:100%; height:auto; display:block; border:1px solid var(--rule); border-radius:4px; background:var(--surface);}
figcaption{font-size:.85rem; color:var(--muted); max-width:70ch;}
.legend{display:flex; flex-wrap:wrap; gap:6px 20px; font-size:.8rem; color:var(--muted);}
ul{margin:0; padding-left:1.15em; display:flex; flex-direction:column; gap:9px;}
li::marker{color:var(--faint);}
.files{background:var(--surface); border:1px solid var(--rule); border-radius:4px; overflow:hidden;}
.file{display:flex; gap:14px; padding:9px 14px; border-bottom:1px solid var(--rule); font-size:.85rem; align-items:baseline;}
.file:last-child{border-bottom:none;}
.file .fn{font-family:var(--mono); font-size:.79rem; color:var(--ink); flex:none; min-width:27ch;}
.file .fd{color:var(--muted);}
@media (max-width:620px){ .file{flex-direction:column; gap:2px;} .file .fn{min-width:0;} }
pre{background:var(--surface); border:1px solid var(--rule); border-radius:4px; padding:14px 16px;
  overflow-x:auto; margin:0; font-family:var(--mono); font-size:.79rem; line-height:1.55; color:var(--ink);}
footer{border-top:1px solid var(--rule); padding-top:16px; font-size:.82rem; color:var(--faint);}
"""


def img_tag(path):
    if not Path(path).exists():
        return ""
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="">'


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def fmt(n):
    return f"{int(n):,}"


def build_html(st, sample_dir, sample_name, title):
    d = Path(sample_dir)
    s = st["summary"]
    qc = s["qc"]
    mad = s["mad_thresholds_reported"]
    ta = s["thresholds_applied"]
    n_start = qc["n_cells_start"]
    n_final = qc["n_cells_final"]
    retention = 100 * n_final / n_start

    # ---- QC table -------------------------------------------------------
    # Doublets are removed before the cell filters, so the criteria were evaluated
    # against the post-doublet population, not the loaded barcodes.
    denom = qc.get("n_cells_criteria_denominator", n_start)
    rows = []
    nd = qc.get("n_predicted_doublets")
    if nd:
        how = qc.get("doublet_method", "scrublet_auto")
        note = "" if how == "scrublet_auto" else " <span class=\"flag f-low\">re-thresholded</span>"
        rows.append(
            f'<tr><td>Doublets (removed first){note}</td><td class="num">{fmt(nd)}</td>'
            f'<td class="num">{100*nd/n_start:.1f}%</td></tr>'
        )
    for k, v in sorted(qc["failed_per_criterion"].items(), key=lambda x: -x[1]):
        rows.append(
            f'<tr><td><span class="mono">{esc(k)}</span></td>'
            f'<td class="num">{fmt(v)}</td>'
            f'<td class="num">{100*v/denom:.1f}%</td></tr>'
        )
    qc_rows = "\n".join(rows)

    # ---- annotation table ----------------------------------------------
    arows = []
    for r in sorted(st["annotation"], key=lambda r: -r["n_cells"]):
        cl = str(r["cluster"])
        ct = r["cell_type"]
        layer = LAYER.get(ct, "ecto")
        margin = r["margin_over_runner_up"]
        if cl in st["cycling_clusters"]:
            flag = '<span class="flag f-low">cell cycle</span>'
        elif cl in st["driver_mismatch_clusters"]:
            flag = '<span class="flag f-low">no driver gene</span>'
        elif margin < 0.5:
            flag = f'<span class="flag f-low">vs {esc(r["runner_up"])}</span>'
        else:
            flag = '<span class="flag f-ok">good</span>'
        arows.append(
            f'<tr><td class="num">{cl}</td><td class="num">{fmt(r["n_cells"])}</td>'
            f'<td><span class="chip"><span class="dot d-{layer}"></span>{esc(ct)}</span></td>'
            f'<td class="num">{r["z_score"]:.2f}</td><td class="num">{margin:.2f}</td>'
            f'<td class="genes">{esc(st["top_markers"].get(cl, "—"))}</td>'
            f"<td>{flag}</td></tr>"
        )
    ann_rows = "\n".join(arows)

    # ---- caveats --------------------------------------------------------
    mito = st["mito"]
    dbl = st["doublet"]
    cav = []
    if mito["pct_gt20"] > 25:
        sev = "dominates this sample" if mito["pct_gt20"] > 60 else "is substantial"
        cav.append(
            f'<li><b>High mitochondrial content {sev}: {mito["pct_gt20"]}% of retained '
            f'cells ({fmt(mito["n_gt20"])}) exceed 20% mitochondrial reads'
            + (f', and {fmt(mito["n_gt35"])} exceed 35%' if mito["n_gt35"] else "")
            + f'.</b> The median retained cell sits at {mito["median_kept"]:.1f}%. '
            "The 5-MAD rule is a <i>relative</i> outlier test, so on a sample whose "
            f'mitochondrial fraction is high throughout it lands at {mito["cap"]:.1f}% and '
            "removes almost nothing — high retention here reflects a permissive cutoff, "
            "not clean data. Stressed or dying cells distort clustering and marker "
            "detection; treat every lineage call on this sample as provisional and "
            "consider an absolute cap alongside the MAD rule.</li>"
        )
    if dbl["suspect"]:
        cav.append(
            f'<li><b>Doublet detection almost certainly failed here: scrublet called '
            f'{dbl["n_called"]} doublets ({dbl["pct_called"]}%) despite scores reaching '
            f'{dbl["score_max"]:.2f}.</b> Its automatic cutoff needs a bimodal '
            "simulated-doublet histogram; when that fails it silently calls nothing, which "
            "is indistinguishable from a clean sample in the summary. Multiplets are "
            "therefore still present and can appear as spurious intermediate clusters "
            "co-expressing two lineages. Set the threshold manually before trusting any "
            "transitional population.</li>"
        )
    if st["vacuous_max_genes"]:
        cav.append(
            f'<li><b>The upper genes-per-cell bound was vacuous.</b> 5 MADs on the '
            f'log-scaled distribution put it at '
            f'{fmt(ta["max_genes"])} genes, above the '
            f'{fmt(qc["n_genes_start"])} genes that exist in the reference, so no cell was '
            "ever removed for having too many genes. The usual multiplet ceiling was "
            "absent from this run.</li>"
        )
    cav.append(
        f'<li><b>{st["low_conf_pct"]}% of cells ({fmt(st["low_conf_cells"])} of '
        f'{fmt(st["n_cells"])}) sit in a low-confidence cluster.</b> '
        f'{len(st["low_conf_clusters"])} of {st["n_clusters"]} clusters '
        f'(<span class="mono">{", ".join(st["low_conf_clusters"]) or "none"}</span>) '
        "have a top lineage that beats the runner-up by less than 0.5 z. Treat the "
        "labels on those clusters as provisional.</li>"
    )
    if st["driver_mismatch_clusters"]:
        dm = st["driver_mismatch_clusters"]
        cav.append(
            f'<li><b>{len(dm)} cluster{"s" if len(dm) != 1 else ""} '
            f'(<span class="mono">{", ".join(dm)}</span>) show no evidence of their assigned '
            "lineage in their own top differentially-expressed genes.</b> A z-score margin "
            "only means one marker panel outscored the others in aggregate — it says nothing "
            "about whether that panel's own genes are what's differential in this cluster's "
            "data, margin size included. This is the exact failure mode that made the "
            "rejected <span class=\"mono\">annotate_combined_pijuansala.py</span> call cells "
            "PGC on a confident margin while its two most specific markers sat at zero "
            "(see handoff.md). Treat these labels as unverified regardless of margin.</li>"
        )
    if st["cycling_clusters"]:
        cy = ", ".join(st["cycling_clusters"])
        det = "; ".join(
            f'cluster {c}: S={st["cell_cycle"][c]["S_score"]:.2f}, '
            f'{st["cell_cycle"][c]["pct_S"]:.0f}% in S phase'
            for c in st["cycling_clusters"]
        )
        cav.append(
            f'<li><b>Cluster {cy} is driven by cell cycle, not lineage.</b> {det} — '
            "far above every other cluster. Its lineage label reflects proliferation, "
            "not identity; regress out cycle scores or relabel it S-phase.</li>"
        )
    if st["never_assigned"]:
        items = ", ".join(
            f'{esc(c)} (best z={zz:.2f} in cluster {cl})'
            for c, zz, cl in st["never_assigned"][:4]
        )
        cav.append(
            f'<li><b>{len(st["never_assigned"])} panel lineages were never assigned.</b> '
            f"{items}. One label per cluster cannot represent nested identities — a "
            "haematoendothelial cluster that is also erythroid has to pick one.</li>"
        )
    if st["dropped_markers"]:
        n_drop = sum(len(v) for v in st["dropped_markers"].values())
        worst = sorted(st["coverage"].items(), key=lambda kv: kv[1][0] / kv[1][1])[:3]
        worst_s = "; ".join(f"{esc(k)} {v[0]}/{v[1]}" for k, v in worst)
        allg = sorted({g for v in st["dropped_markers"].values() for g in v})
        cav.append(
            f'<li><b>The <span class="mono">min_cells={ta["min_cells_per_gene"]}</span> '
            f"gene filter removed {n_drop} panel markers before scoring.</b> "
            f"Weakest panels: {worst_s}. Dropped: "
            f'<span class="mono">{esc(", ".join(allg))}</span>. Markers for the rarest '
            "lineages are exactly the ones a per-gene detection floor deletes, so those "
            "lineages cannot be called even when present.</li>"
        )
    if st["ambient_floor"]:
        fl = ", ".join(f"{g} {v:.0f}%" for g, v in st["ambient_floor"].items())
        cav.append(
            "<li><b>Ambient RNA was not corrected.</b> Visceral-endoderm transcripts are "
            f"detected in every cluster including the least endodermal one ({fl} of cells). "
            "That uniform floor is the signature of droplet soup. It is low-level and does "
            "not overturn the annotation, but run SoupX or CellBender before using these "
            "counts quantitatively.</li>"
        )
    cav.append(
        "<li><b>No batch correction</b> — this is a single sample. Combining it with the "
        "other timepoints will need integration (Harmony, scVI) before joint clustering.</li>"
    )
    caveats = "\n".join(cav)

    # The lead callout carries whichever fact most changes how the page is read.
    if mito["pct_gt20"] > 60:
        lead = (
            '<div class="eyebrow">Finding &middot; read this first</div>'
            "<h3>This sample is dominated by high-mitochondrial cells</h3>"
            f'<p>{mito["pct_gt20"]}% of the {fmt(n_final)} retained cells exceed 20% '
            f'mitochondrial reads and {fmt(mito["n_gt35"])} exceed 35%; the median retained '
            f'cell sits at {mito["median_kept"]:.1f}%. The 5-MAD rule set the cap at '
            f'{mito["cap"]:.1f}%, so it removed almost nothing — the {retention:.1f}% '
            "retention figure reflects a permissive, sample-relative cutoff rather than "
            "clean data.</p>"
            "<p>Everything downstream — clustering, markers, lineage calls — inherits that. "
            "Read this report as a description of what the pipeline produced, not as a "
            "trustworthy lineage map, and re-run with an absolute mitochondrial cap before "
            "using these labels.</p>"
        )
    else:
        lead = (
            '<div class="eyebrow">Read this first</div>'
            "<h3>Thresholds were chosen per sample, not inherited</h3>"
            f"<p>Cutoffs come from this sample's row of "
            "<code>data/qc_filtering_settings.tsv</code>, picked from its own pre-filter "
            "distributions (see <code>data/qc_scan/</code>). Pure 5-MAD selection was "
            "abandoned: it is a <i>relative</i> rule, so on a sample whose mitochondrial "
            "fraction is high throughout it set the cap above the bulk of the data and "
            "filtered almost nothing. The mitochondrial cap here is "
            f"<span class=\"mono\">{ta['max_pct_mt']:.0f}%</span>.</p>"
            "<p>The sample is a <b>gastrulating embryo, not cultured mESC</b>, despite "
            "living under <code>mESC_10x_raw/</code>. Cells were annotated against a mouse "
            "gastrulation panel (Pijuan-Sala 2019 / Argelaguet 2022 lineages).</p>"
        )

    sweep = " / ".join(str(v) for v in st["resolution_sweep"].values())
    sweep_res = " / ".join(
        c.replace("leiden_res", "").replace("leiden", "1.0")
        for c in st["resolution_sweep"]
    )

    comp = sorted(s["cell_type_counts"].items(), key=lambda kv: -kv[1])
    comp_s = ", ".join(f"{esc(k)} ({fmt(v)})" for k, v in comp[:4])

    captions = {
        "qc_violin_pre_filtering.png":
            "<b>Before filtering.</b> Genes per cell, total counts, mitochondrial percent and "
            "ribosomal percent across all "
            f"{fmt(n_start)} barcodes. The thresholds for this sample were read off these "
            "distributions.",
        "qc_violin_post_filtering.png":
            "<b>After filtering.</b> The same four metrics over the "
            f"{fmt(n_final)} retained cells. Compare the mitochondrial panel against the one "
            "above: if its shape barely changes, the cutoff removed a tail rather than a "
            "population.",
        "qc_scatter_post_filtering.png":
            "Counts against genes per cell (coloured by mitochondrial percent) and counts "
            "against mitochondrial percent, after filtering.",
        "umap_clusters_and_celltypes.png":
            "Leiden clusters at resolution 1.0 (left) and the lineage label assigned to each "
            "cluster (right). Both panels share the same UMAP embedding.",
        "heatmap_lineage_signature_scores.png":
            "Mean signature score per cluster, z-scored down each column. The assignment is "
            "the row-wise argmax; a row with two similar reds is a cluster the method could "
            "not separate.",
        "dotplot_canonical_lineage_markers.png":
            "Canonical panel markers grouped by lineage, scaled per gene.",
    }

    def fig(name, cls="bleed"):
        """Embed a figure by filename, or return '' when the plot is absent."""
        t = img_tag(d / name)
        if not t:
            return ""
        cap = captions.get(name, "")
        return f'<figure class="{cls}">{t}<figcaption>{cap}</figcaption></figure>'

    body = f"""<title>{esc(title)}</title>
<style>{CSS}</style>
<div class="wrap">
  <header>
    <div class="eyebrow">10x Multiome &middot; Gene expression modality</div>
    <h1>{esc(title)}</h1>
    <p class="lede">Single-cell RNA workflow over <code>{esc(sample_name)}</code> — load, QC,
      normalize, cluster, marker genes, cell-type annotation. {st['n_clusters']} clusters
      resolve into {st['n_lineages']} gastrulation lineages; the largest are
      {comp_s}.</p>
    <div class="meta">
      <span><b>Sample</b> {esc(sample_name)}</span>
      <span><b>Genome</b> mm10</span>
      <span><b>QC</b> per-sample thresholds</span>
      <span><b>scanpy</b> {esc(s.get('scanpy_version','—'))}</span>
      <span><b>Seed</b> {esc(s.get('seed',0))}</span>
    </div>
  </header>

  <div class="callout">{lead}</div>

  <div class="stats">
    <div class="stat"><span class="n">{fmt(n_start)}</span><span class="l">Barcodes in</span></div>
    <div class="stat"><span class="n">{fmt(n_final)}</span><span class="l">Cells kept</span></div>
    <div class="stat"><span class="n">{fmt(st['n_genes'])}</span><span class="l">Genes kept</span></div>
    <div class="stat"><span class="n">{st['n_clusters']}</span><span class="l">Leiden clusters</span></div>
    <div class="stat"><span class="n">{st['n_lineages']}</span><span class="l">Lineages called</span></div>
    <div class="stat"><span class="n">{st['low_conf_pct']}%</span><span class="l">Low-confidence cells</span></div>
    <div class="stat"><span class="n">{mito['median_kept']:.1f}%</span><span class="l">Median mito, kept</span></div>
  </div>

  <section>
    <div class="sec-head"><span class="stage-no">01</span><h2>Quality control</h2></div>
    <p>Doublets are detected and removed <i>first</i>, then the cell filters are applied to
      what remains — scrublet models the observed cell population, so discarding a third of
      it beforehand starves the distribution its threshold is read from. Criteria overlap,
      so the per-criterion counts sum to more than the total removed. Retention was
      <b>{retention:.1f}%</b> ({fmt(n_start)} &rarr; {fmt(n_final)}).</p>
    <div class="tbl-scroll">
      <table>
        <thead><tr><th>Criterion applied</th><th class="num">Cells failing</th><th class="num">% of input</th></tr></thead>
        <tbody>{qc_rows}</tbody>
      </table>
    </div>
    <p>Applied: genes/cell
      <span class="mono">[{fmt(ta['min_genes'])}, {fmt(ta['max_genes'])}]</span>, counts/cell
      <span class="mono">[{fmt(ta['min_counts'])}, {fmt(ta['max_counts'])}]</span>,
      mitochondrial fraction &le; <span class="mono">{ta['max_pct_mt']:.0f}%</span>. For
      comparison, 5-MAD on this sample would have put the mitochondrial cap at
      <span class="mono">{mad['pct_counts_mt']['upper']:.1f}%</span> and the gene ceiling at
      <span class="mono">{mad['n_genes_by_counts']['upper']:.0f}</span>.</p>
    <p>Genes were filtered at <span class="mono">min_cells={ta['min_cells_per_gene']}</span>,
      leaving {fmt(st['n_genes'])} of {fmt(qc['n_genes_start'])}
      {f"({qc['n_marker_genes_rescued']} marker-panel genes were kept below that floor so "
        "rare lineages stay annotatable)" if qc.get('n_marker_genes_rescued') else ""}.</p>
    {fig("qc_violin_pre_filtering.png")}
    {fig("qc_violin_post_filtering.png")}
  </section>

  <section>
    <div class="sec-head"><span class="stage-no">02</span><h2>Normalize &amp; cluster</h2></div>
    <p>Counts-per-{s.get('n_hvg') and '10,000'} normalization, <span class="mono">log1p</span>,
      {fmt(s.get('n_hvg', 3000))} highly variable genes (flavour
      <span class="mono">{esc(s.get('hvg_flavor','seurat'))}</span>), scaling to unit variance,
      {s.get('n_pcs',50)} PCs, a 15-neighbour graph, UMAP, then Leiden at resolution
      {s.get('resolution',1.0)}. Raw counts are preserved in
      <code>.layers["counts"]</code> and the log-normalized all-gene matrix in <code>.raw</code>.</p>
    <p>A resolution sweep ({sweep_res} &rarr; {sweep} clusters) is stored alongside, so the
      granularity choice can be revisited without recomputing the embedding.</p>
  </section>

  <section>
    <div class="sec-head"><span class="stage-no">03</span><h2>Structure</h2></div>
    {fig("umap_clusters_and_celltypes.png")}
    <div class="legend">
      <span class="chip"><span class="dot d-ecto"></span>Ectoderm</span>
      <span class="chip"><span class="dot d-meso"></span>Mesoderm</span>
      <span class="chip"><span class="dot d-endo"></span>Endoderm</span>
      <span class="chip"><span class="dot d-exe"></span>Extraembryonic</span>
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="stage-no">04</span><h2>Markers &amp; annotation</h2></div>
    <p>Wilcoxon rank-sum against all other clusters: {fmt(st['n_tests'])} tests. Each cluster
      was then scored against 17 curated lineage signatures; scores are z-scored
      <i>across clusters</i> before taking the argmax, so broad panels cannot win everywhere by
      sitting uniformly high. Clusters whose top signature beats the runner-up by less than
      0.5 z are flagged rather than silently labelled.</p>
    <div class="tbl-scroll">
      <table>
        <thead><tr><th class="num">Cl.</th><th class="num">n</th><th>Assigned lineage</th>
          <th class="num">z</th><th class="num">Margin</th><th>Top marker genes</th>
          <th>Confidence</th></tr></thead>
        <tbody>{ann_rows}</tbody>
      </table>
    </div>
    {fig("heatmap_lineage_signature_scores.png")}
  </section>

  <section>
    <div class="sec-head"><span class="stage-no">05</span><h2>Where this needs a second look</h2></div>
    <ul>{caveats}</ul>
  </section>

  <section>
    <div class="sec-head"><span class="stage-no">06</span><h2>Diagnostics</h2></div>
    {fig("qc_scatter_post_filtering.png")}
    {fig("dotplot_canonical_lineage_markers.png")}
  </section>

  <section>
    <div class="sec-head"><span class="stage-no">07</span><h2>Outputs</h2></div>
    <p>Everything lands in <code>data/processed/mESC/{esc(sample_name)}/</code>.</p>
    <div class="files">
      <div class="file"><span class="fn">{esc(sample_name)}_rna_processed.h5ad</span><span class="fd">cells &times; genes with QC, PCA/UMAP, all four Leiden resolutions, <span class="mono">cell_type</span>, signature scores</span></div>
      <div class="file"><span class="fn">{esc(sample_name)}_atac_raw.h5ad</span><span class="fd">the peak modality, unprocessed, with coordinates</span></div>
      <div class="file"><span class="fn">marker_genes_all.csv</span><span class="fd">all {fmt(st['n_tests'])} Wilcoxon tests</span></div>
      <div class="file"><span class="fn">marker_genes_top25.csv</span><span class="fd">top 25 significant markers per cluster</span></div>
      <div class="file"><span class="fn">cluster_cell_type_annotation.csv</span><span class="fd">assignment, z-score, margin, runner-up per cluster</span></div>
      <div class="file"><span class="fn">cluster_marker_scores_*.csv</span><span class="fd">raw and z-scored lineage signature matrices</span></div>
      <div class="file"><span class="fn">run_summary.json</span><span class="fd">every parameter, threshold and count from the run</span></div>
      <div class="file"><span class="fn">report_stats.json</span><span class="fd">the diagnostics computed for this report</span></div>
    </div>
    <pre>source activate my_env
sbatch TETHER/bash_scripts/run_scrna_annotation_mESC.sh   # all 11 samples</pre>
  </section>

  <footer>Generated with scanpy {esc(s.get('scanpy_version','—'))} on mm10. Lineage panels after
    Pijuan-Sala et al. 2019 and Argelaguet et al. 2022. Per-sample QC thresholds. Random seed
    {esc(s.get('seed',0))} throughout.</footer>
</div>
"""
    return body


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_dir", required=True)
    p.add_argument("--sample_name", required=True)
    p.add_argument("--title", default=None)
    a = p.parse_args()

    title = a.title or f"{a.sample_name.replace('_', ' ')} Lineage Atlas"
    log(f"{a.sample_name}: computing diagnostics")
    st = compute_stats(a.sample_dir, a.sample_name)
    log(f"{a.sample_name}: building html")
    html = build_html(st, a.sample_dir, a.sample_name, title)
    out = Path(a.sample_dir) / "report.html"
    out.write_text(html)
    log(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
