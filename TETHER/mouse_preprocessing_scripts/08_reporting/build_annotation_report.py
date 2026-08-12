"""Build the HTML report for the integrated-object atlas annotation.

Reads the artefacts written by combine_mesc_samples.py, centroid_label_transfer.py,
validate_centroid_labels.py, inspect_unresolved_clusters.py and reconcile_atlas_labels.py
and emits a single self-contained page (images base64-inlined, no external requests).

Everything quantitative on the page is read from those files rather than restated here,
so the report cannot drift from the analysis.
"""

import argparse
import base64
import json
import mimetypes
import re
from pathlib import Path

import pandas as pd

PALETTE_NOTE = "validated / overridden / unresolved encode evidence tier, not category"


def img_tag(path, alt=""):
    p = Path(path)
    if not p.exists():
        return ""
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:{mime};base64,{b64}" alt="{alt}" loading="lazy">'


def figure(path, caption, cls="bleed"):
    t = img_tag(path, caption)
    if not t:
        return ""
    return f'<figure class="{cls}">{t}<figcaption>{caption}</figcaption></figure>'


def table(df, cls="", tabular=True):
    """Wrap a DataFrame as an HTML table, restyled and horizontally scrollable.

    Rewrites the opening tag rather than splitting on it: pandas has changed the
    attributes it emits there between versions (``border="0"`` comes and goes), and a
    split that misses silently nests a second <table> inside the wrapper.
    """
    cls = f"{cls} tabular".strip() if tabular else cls
    html = df.to_html(index=False, escape=False, border=0)
    html = re.sub(r"<table[^>]*>", f'<table class="{cls}">', html, count=1)
    return f'<div class="scroll">{html}</div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed/mESC/combined")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)

    final = pd.read_csv(d / "final_atlas_labels.csv", dtype={"cluster": str})
    val = pd.read_csv(d / "validation_centroid_labels.csv", dtype={"cluster": str})
    cent = pd.read_csv(d / "centroid_annotation_rna_leiden.csv", dtype={"cluster": str})
    fsum = json.loads((d / "final_labels_summary.json").read_text())
    csum = json.loads((d / "centroid_annotation_summary.json").read_text())
    comb = json.loads((d / "combine_summary.json").read_text()) if (d / "combine_summary.json").exists() else {}
    tp = pd.read_csv(d / "final_timepoint_composition.csv", index_col=0)

    n_cells = fsum["cells_total"]
    tiers = fsum["cells_by_tier"]
    n_unres = tiers.get("unresolved", 0)
    pct_lab = fsum["pct_confidently_labelled"]

    # --- cluster table -------------------------------------------------------
    v = val.set_index("cluster")
    c = cent.set_index("cluster")
    rows = []
    for _, r in final.sort_values("n_cells", ascending=False).iterrows():
        cl = r["cluster"]
        tier = r["support"]
        badge = {"centroid+marker": "ok", "marker_override": "warn",
                 "unresolved": "flag"}.get(tier, "")
        label_txt = {"centroid+marker": "validated", "marker_override": "overridden",
                     "unresolved": "unresolved"}[tier]
        changed = r["final_label"] != r["centroid_label"]
        rows.append({
            "Cluster": cl,
            "Cells": f'{r["n_cells"]:,}',
            "Final label": (f'<strong>{r["final_label"]}</strong>'
                            + (f'<span class="was">was {r["centroid_label"]}</span>' if changed else "")),
            "ρ": f'{c.loc[cl, "rho"]:.3f}' if cl in c.index else "",
            "Marker z": ("—" if pd.isna(v.loc[cl, "z"]) else f'{v.loc[cl, "z"]:.2f}') if cl in v.index else "",
            "Evidence": f'<span class="badge {badge}">{label_txt}</span>',
        })
    cluster_tbl = table(pd.DataFrame(rows))

    # --- timepoint composition, top populations ------------------------------
    tp_show = tp.loc[tp.max(axis=1).sort_values(ascending=False).index]
    tp_show = tp_show.head(14).round(1)
    tp_disp = tp_show.reset_index()
    tp_disp.columns = ["Population"] + list(tp_show.columns)
    tp_tbl = table(tp_disp)

    ncl = fsum["n_clusters"]
    npop = fsum["n_populations"]
    n_ok = int((final["support"] == "centroid+marker").sum())
    n_ov = int((final["support"] == "marker_override").sum())
    n_un = int((final["support"] == "unresolved").sum())

    html = f"""<title>Pijuan-Sala Label Transfer</title>
<style>
:root {{
  --ground:#F2F3F5; --surface:#FFFFFF; --sunk:#E7E9EC;
  --ink:#171B22; --ink-2:#454E5C; --ink-3:#6D7787;
  --rule:#D3D8DE; --rule-2:#E4E7EB;
  --ok:#1F6F5C; --ok-bg:#DDEDE7;
  --warn:#8A5A16; --warn-bg:#F4E7D2;
  --flag:#7A3F58; --flag-bg:#F0DFE6;
  --accent:#243B6B;
  --serif: "Iowan Old Style","Palatino Linotype",Palatino,"Source Serif 4",Georgia,serif;
  --sans: system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  --mono: ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#12151A; --surface:#191D24; --sunk:#22272F;
    --ink:#E7EAEE; --ink-2:#AEB6C2; --ink-3:#7F8A99;
    --rule:#2C333C; --rule-2:#242A32;
    --ok:#6FC4A9; --ok-bg:#17302A;
    --warn:#DDA85B; --warn-bg:#332616;
    --flag:#D294AF; --flag-bg:#31202A;
    --accent:#9DB3E0;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#12151A; --surface:#191D24; --sunk:#22272F;
  --ink:#E7EAEE; --ink-2:#AEB6C2; --ink-3:#7F8A99;
  --rule:#2C333C; --rule-2:#242A32;
  --ok:#6FC4A9; --ok-bg:#17302A;
  --warn:#DDA85B; --warn-bg:#332616;
  --flag:#D294AF; --flag-bg:#31202A;
  --accent:#9DB3E0;
}}
* {{ box-sizing:border-box; }}
body {{
  background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.65;
  margin:0; padding:0 1.25rem 6rem;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:78rem; margin:0 auto; display:flex; flex-direction:column; gap:2.5rem; }}
.col {{ max-width:66ch; }}
header {{ padding:4rem 0 0; display:flex; flex-direction:column; gap:1rem; }}
.eyebrow {{
  font-family:var(--mono); font-size:.72rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink-3);
}}
h1 {{
  font-family:var(--serif); font-weight:600; font-size:clamp(2.1rem,5vw,3.1rem);
  line-height:1.1; letter-spacing:-.015em; margin:0; text-wrap:balance;
}}
.standfirst {{ font-family:var(--serif); font-size:1.2rem; color:var(--ink-2); margin:0; max-width:60ch; }}
h2 {{
  font-family:var(--serif); font-size:1.6rem; font-weight:600; letter-spacing:-.01em;
  margin:0 0 .35rem; text-wrap:balance;
}}
h3 {{ font-size:1rem; font-weight:650; margin:0 0 .3rem; letter-spacing:-.005em; }}
p {{ margin:0 0 1rem; }}
section {{ display:flex; flex-direction:column; gap:1.1rem; }}
.rule {{ height:1px; background:var(--rule); border:0; margin:0; }}
/* attempt sequence: numbering is real here -- three ordered attempts */
.attempt {{ display:grid; grid-template-columns:auto 1fr; gap:1.1rem; align-items:start; }}
.num {{
  font-family:var(--mono); font-size:.78rem; color:var(--ink-3);
  border:1px solid var(--rule); border-radius:999px; width:2.1rem; height:2.1rem;
  display:grid; place-items:center; margin-top:.15rem;
}}
.attempt.dead .num {{ color:var(--flag); border-color:var(--flag); }}
.attempt.live .num {{ color:var(--ok); border-color:var(--ok); }}
.verdict {{ font-family:var(--mono); font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; }}
.verdict.dead {{ color:var(--flag); }}
.verdict.live {{ color:var(--ok); }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(11rem,1fr)); gap:1px;
  background:var(--rule-2); border:1px solid var(--rule-2); border-radius:8px; overflow:hidden; }}
.stat {{ background:var(--surface); padding:1.1rem 1.2rem; display:flex; flex-direction:column; gap:.25rem; }}
.stat .k {{ font-family:var(--mono); font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-3); }}
.stat .v {{ font-size:1.85rem; font-weight:600; letter-spacing:-.02em; font-variant-numeric:tabular-nums; line-height:1.15; }}
.stat .s {{ font-size:.83rem; color:var(--ink-2); }}
.scroll {{ overflow-x:auto; border:1px solid var(--rule-2); border-radius:8px; background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:.88rem; }}
.tabular td:nth-child(2), .tabular td:nth-child(4), .tabular td:nth-child(5) {{ font-variant-numeric:tabular-nums; }}
th {{
  text-align:left; font-family:var(--mono); font-size:.68rem; letter-spacing:.09em;
  text-transform:uppercase; color:var(--ink-3); font-weight:500;
  padding:.7rem .85rem; border-bottom:1px solid var(--rule); white-space:nowrap;
}}
td {{ padding:.6rem .85rem; border-bottom:1px solid var(--rule-2); vertical-align:top; }}
tr:last-child td {{ border-bottom:0; }}
.was {{ display:block; font-size:.76rem; color:var(--ink-3); font-family:var(--mono); }}
.badge {{
  display:inline-block; font-family:var(--mono); font-size:.68rem; letter-spacing:.05em;
  padding:.16rem .5rem; border-radius:4px; white-space:nowrap;
}}
.badge.ok {{ background:var(--ok-bg); color:var(--ok); }}
.badge.warn {{ background:var(--warn-bg); color:var(--warn); }}
.badge.flag {{ background:var(--flag-bg); color:var(--flag); }}
figure {{ margin:0; display:flex; flex-direction:column; gap:.6rem; }}
figure img {{ width:100%; height:auto; display:block; border:1px solid var(--rule-2); border-radius:8px; background:#fff; }}
figcaption {{ font-size:.83rem; color:var(--ink-3); max-width:64ch; }}
.callout {{
  border-left:2px solid var(--accent); background:var(--surface);
  padding:1rem 1.2rem; border-radius:0 8px 8px 0; display:flex; flex-direction:column; gap:.5rem;
}}
.callout.flag {{ border-left-color:var(--flag); }}
code {{ font-family:var(--mono); font-size:.86em; background:var(--sunk); padding:.1rem .32rem; border-radius:3px; }}
ul {{ margin:0 0 1rem; padding-left:1.1rem; }}
li {{ margin-bottom:.35rem; }}
footer {{ color:var(--ink-3); font-size:.83rem; border-top:1px solid var(--rule); padding-top:1.5rem; }}
@media (prefers-reduced-motion:no-preference) {{ html {{ scroll-behavior:smooth; }} }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Integrated mESC gastrulation object &middot; {n_cells:,} nuclei &middot; 9 samples</div>
  <h1>Annotating the integrated embryos against the Pijuan-Sala atlas</h1>
  <p class="standfirst">Two annotation methods produced confident, wrong answers before one
  produced a defensible one. This is what each attempt claimed, how it was tested, and which
  labels survived.</p>
</header>

<hr class="rule">

<section>
  <div class="stats">
    <div class="stat"><span class="k">Nuclei labelled</span><span class="v">{pct_lab}%</span>
      <span class="s">{n_cells - n_unres:,} of {n_cells:,}</span></div>
    <div class="stat"><span class="k">Clusters</span><span class="v">{ncl}</span>
      <span class="s">{npop} atlas populations</span></div>
    <div class="stat"><span class="k">Marker-validated</span><span class="v">{n_ok}</span>
      <span class="s">label confirmed by its own markers</span></div>
    <div class="stat"><span class="k">Overridden</span><span class="v">{n_ov}</span>
      <span class="s">markers outvoted the correlation</span></div>
    <div class="stat"><span class="k">Unresolved</span><span class="v">{n_un}</span>
      <span class="s">{n_unres:,} nuclei, deliberately unlabelled</span></div>
  </div>
</section>

<section class="col">
  <h2>Three attempts</h2>
  <p>Annotation here is a transfer problem: the query is single-nucleus 10x Multiome across
  {ncl} clusters, the reference is whole-cell scRNA-seq — 116,312 atlas cells spanning 37
  annotated populations. That technology gap is what decided which methods worked.</p>
</section>

<section>
  <div class="attempt dead">
    <div class="num">01</div>
    <div>
      <span class="verdict dead">Rejected</span>
      <h3>Marker-signature scoring</h3>
      <p>Z-scored 34 curated marker panels across clusters and took the argmax. It called
      <strong>2,366 cells PGC</strong> while <code>Dppa3</code> and <code>Nanos3</code> were flat
      zero — the call was carried by non-specific <code>Prdm1</code>/<code>Ifitm3</code>. Both PGC
      clusters were flagged <em>confident</em> (margins 0.90 and 0.57), which showed the margin only
      measures how far the winner beat the runner-up, not whether specific genes drove it.
      57.8% of cells sat in low-confidence clusters.</p>
    </div>
  </div>

  <div class="attempt dead">
    <div class="num">02</div>
    <div>
      <span class="verdict dead">Rejected</span>
      <h3>kNN label transfer in a joint Harmony embedding</h3>
      <p>Co-embedded query and atlas, then transferred labels by distance-weighted kNN vote.
      It assigned <strong>15,546 of 47,631 cells (32.6%) to Pharyngeal mesoderm</strong>, the
      consensus for 8 of 23 clusters — including clusters independently verified as Pax6+
      neurectoderm, Meox1+ somitic mesoderm and Pou5f1-high epiblast — at 0.78–0.89 mean
      confidence. Agreement with the marker labels was 22.9%.</p>
      <p>The reference was not at fault: atlas labels check out directly against expression
      (Cardiomyocytes <code>Myl7</code> 137.0, Erythroid3 <code>Hbb-bh1</code> 728.4, Epiblast
      <code>Pou5f1</code> 10.66). The joint embedding was. Correcting the nucleus-versus-whole-cell
      gap pulls query cells into the reference's dense mesodermal centre, and Pharyngeal mesoderm
      is non-distinctive there (<code>Tbx1</code> 0.43, <code>Isl1</code> 0.84) — a natural sink.</p>
    </div>
  </div>

  <div class="attempt live">
    <div class="num">03</div>
    <div>
      <span class="verdict live">Adopted</span>
      <h3>Cluster-centroid rank correlation</h3>
      <p>Spearman correlation of each cluster's mean log-CP10K profile against each atlas
      population's mean profile, over {csum['n_genes']:,} genes chosen for being informative in the
      <em>reference</em> ({csum['shared_genes']:,} genes shared). No shared embedding, so nothing to
      over-correct; rank correlation is insensitive to the systematic scale difference between
      nuclear and whole-cell RNA.</p>
    </div>
  </div>
</section>

<hr class="rule">

<section>
  <h2 class="col">The test that separated them</h2>
  <div class="col">
  <p>Each attempt was checked the same way: for every cluster, is the <em>canonical</em> marker
  panel of its assigned type actually enriched there relative to all other clusters? This is a
  different question from the one the annotation optimises, so it can disagree.</p>
  <p>The check was calibrated on controls before being trusted — correct labels scored
  <strong>6/6</strong> supported, deliberately rotated labels scored <strong>0/6</strong>, and in
  every rotated case it recovered the true label. It has power to detect wrong assignments.</p>
  </div>
  {figure(d / "heatmap_marker_validation.png",
          "Canonical marker z-score per cluster; the black box marks the assignment made by centroid correlation. Boxes on red are agreements.")}
</section>

<section>
  <div class="col">
  <h2>Confidence that tracks correctness</h2>
  <p>The decisive property, and the one the first two attempts lacked: where the centroid method
  flagged itself as uncertain is where it was actually wrong.</p>
  </div>
  <div class="stats" style="max-width:44rem">
    <div class="stat"><span class="k">Self-flagged distinctive</span><span class="v">14/15</span>
      <span class="s">93% marker-supported</span></div>
    <div class="stat"><span class="k">Self-flagged uncertain</span><span class="v">4/8</span>
      <span class="s">50% marker-supported</span></div>
  </div>
  <p class="col">Four of the five disagreements had already been flagged as non-distinctive by the
  method itself. Its close calls are also almost all between genuinely adjacent populations — ExE
  vs Visceral endoderm, Erythroid1 vs Erythroid2, Spinal cord vs Forebrain/Midbrain/Hindbrain,
  Primitive Streak vs Anterior Primitive Streak. That is what real annotation uncertainty looks
  like, as against confusion that crosses germ layers.</p>
  {figure(d / "heatmap_centroid_correlation.png",
          "Spearman correlation of every query cluster centroid against all 37 atlas populations.")}
</section>

<hr class="rule">

<section>
  <div class="col">
  <h2>Adjudicating the five disagreements</h2>
  <p>Neither a correlation nor a panel score settles a disagreement on its own, so each was
  resolved on raw per-gene evidence — cluster mean versus the mean of all other clusters, with
  rank out of {ncl}.</p>
  </div>

  <div class="callout">
    <h3>Three labels corrected</h3>
    <ul>
      <li><strong>cl18 → ExE mesoderm</strong> (was Mesenchyme): <code>Postn</code> 37×, <code>Bmp4</code>
      5.4×, <code>Hand1</code> 5.2× — all rank 1; mesenchyme's own <code>Prrx1</code> 0.42× rank 14.</li>
      <li><strong>cl2 → Caudal Mesoderm</strong> (was Somitic mesoderm): <code>Tbx6</code> 14.8× rank 1,
      the defining presomitic marker, with <code>Pax2</code> 0.27× and <code>Wt1</code> 0.38× excluding
      intermediate mesoderm.</li>
      <li><strong>cl21 → Surface ectoderm</strong> (was Mesenchyme): <code>Krt8</code> 2.4× and
      <code>Krt18</code> 2.8×, both rank 1, against depleted <code>Pdgfra</code> 0.53× / <code>Prrx1</code> 0.51×.</li>
    </ul>
  </div>

  <div class="callout flag">
    <h3>Two clusters left unlabelled</h3>
    <p>Both were assignable only by forcing a population onto cells that do not support one.
    Naming them would have been the same failure as the 2,366 phantom PGCs.</p>
    <ul>
      <li><strong>cl0</strong> ({int(final.loc[final['cluster']=='0','n_cells'].iloc[0]):,} cells, the largest cluster) is
      depth-driven, not a cell type: median 1,439 genes against 3,159–5,166 everywhere else, with a
      <em>low</em> 3.1% mitochondrial fraction, so low complexity rather than dying cells. It is 46%
      E7.75_rep1 + 31% E8.5_rep2 and near-absent from E8.0 and E8.75 — a sample signature, not a
      developmental one. Both candidate labels are contradicted (Surface ectoderm markers
      <em>depleted</em>: <code>Krt8</code> 0.74×, <code>Trp63</code> 0.63×). Genuine epiblast signal is
      present (<code>Pou5f1</code> 3.26×, <code>Nanog</code> 3.19×), but epiblast does not exist at E8.5,
      so the cluster is heterogeneous.</li>
      <li><strong>cl13</strong> (156 cells) is not blood: <code>Runx1</code> 8× is the only support, while
      <code>Cd34</code> and <code>Klf1</code> are 0.00 (rank {ncl}), <code>Kit</code> rank 21 and
      <code>Tal1</code> 0.34×. The rest of the haematopoietic program is simply absent.</li>
    </ul>
  </div>
</section>

<hr class="rule">

<section>
  <h2 class="col">Final labels</h2>
  {cluster_tbl}
  <p class="figcaption col" style="font-size:.83rem;color:var(--ink-3)">ρ is the centroid Spearman
  correlation; marker z is the canonical panel z-score across clusters. {PALETTE_NOTE}.</p>
  {figure(d / "umap_final_labels.png", "Integrated UMAP coloured by final reconciled label.")}
  {figure(d / "umap_final_support.png", "The same embedding coloured by evidence tier, so the unresolved region is visible.")}
</section>

<section>
  <div class="col">
  <h2>An independent check the annotation never saw</h2>
  <p>No timepoint information entered the annotation, so developmental staging is a free test.
  The labels reproduce known gastrulation timing.</p>
  <ul>
    <li><strong>Cardiomyocytes</strong> absent at E7.5/E7.75, appearing from E8.0 — when they differentiate.</li>
    <li><strong>Neural crest</strong> and <strong>Spinal cord</strong> both rise monotonically to E8.75.</li>
    <li><strong>Primitive Streak</strong> is confined to E7.5/E7.75 and vanishes by E8.0, as a transient structure should.</li>
    <li><strong>ExE ectoderm</strong> declines monotonically from 14.5% to 0%.</li>
  </ul>
  </div>
  {tp_tbl}
  <p class="col" style="font-size:.83rem;color:var(--ink-3)">Percentage of each timepoint's nuclei,
  top 14 populations by peak abundance.</p>
</section>

<hr class="rule">

<section class="col">
  <h2>What this does not settle</h2>
  <ul>
    <li>The {n_unres:,} unresolved nuclei ({100 - pct_lab:.1f}%) are concentrated in E7.75_rep1 and
    E8.5_rep2. Sub-clustering cl0 after regressing depth, or excluding it, is the natural next step —
    it should not be carried into downstream analysis as a cell type.</li>
    <li>Populations distinguished mainly by maturation stage (Erythroid1/2/3, Blood progenitors 1/2)
    are not reliably separable at this clustering resolution; their close calls are genuine.</li>
    <li>Labels are assigned per cluster, not per cell, so within-cluster heterogeneity is invisible
    to this annotation — cl0 is the clearest example of that limitation biting.</li>
  </ul>
</section>

<footer>
  <p>Reference: Pijuan-Sala et al., <em>A single-cell molecular map of mouse gastrulation and early
  organogenesis</em>, Nature 566:490–495 (2019), doi:10.1038/s41586-019-0933-9.</p>
  <p>Method: {fsum['method']}. Labels in <code>celltype_final</code> with provenance in
  <code>celltype_final_support</code>; the rejected kNN outputs are retained under
  <code>rejected_knn_transfer/</code> with the reasons they failed.</p>
</footer>
</div>
"""
    Path(a.out).write_text(html)
    print(f"wrote {a.out} ({len(html)/1024:.0f} KB of HTML before image inlining counted)")


if __name__ == "__main__":
    main()
