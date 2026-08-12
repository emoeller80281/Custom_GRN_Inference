"""Pre-filter QC scan for one 10x Multiome sample.

Loads only the Gene Expression modality, computes per-cell QC metrics and writes
a quantile summary to JSON. This is the evidence used to choose per-sample
thresholds; it does no filtering, clustering or annotation of its own.

Reports, for each metric, the empirical quantiles alongside the 3-MAD and 5-MAD
bounds so the two can be compared directly -- MAD is a relative rule and on some
samples lands outside the data entirely.

Example:
    python qc_scan.py --input_dir .../mESC_10x_raw/E7.5_rep1 --sample_name E7.5_rep1 \
        --out_dir data/qc_scan
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
import scanpy as sc

QUANTILES = [0.001, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 0.999]


def log(m):
    print(f"[qc_scan] {m}", flush=True)


def mad_bounds(v, n_mads, log_transform):
    x = np.log1p(v) if log_transform else np.asarray(v, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    lo, hi = med - n_mads * mad, med + n_mads * mad
    if log_transform:
        lo, hi = np.expm1(lo), np.expm1(hi)
    return float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--sample_name", required=True)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log(f"reading {a.input_dir}")
    ad = sc.read_10x_mtx(a.input_dir, var_names="gene_symbols", make_unique=True,
                         gex_only=False, cache=False)
    rna = ad[:, ad.var["feature_types"] == "Gene Expression"].copy()
    del ad
    log(f"RNA: {rna.n_obs} barcodes x {rna.n_vars} genes")

    rna.var["mt"] = rna.var_names.str.startswith("mt-")
    rna.var["ribo"] = rna.var_names.str.match(r"^Rp[sl]")
    sc.pp.calculate_qc_metrics(rna, qc_vars=["mt", "ribo"], percent_top=None,
                               log1p=False, inplace=True)

    o = rna.obs
    res = {"sample": a.sample_name, "n_barcodes": int(rna.n_obs),
           "n_genes_total": int(rna.n_vars), "metrics": {}}

    for key, logt in [("n_genes_by_counts", True), ("total_counts", True),
                      ("pct_counts_mt", False), ("pct_counts_ribo", False)]:
        v = o[key].values
        q = {f"p{int(x*1000)/10:g}": round(float(np.quantile(v, x)), 3) for x in QUANTILES}
        m3, m5 = mad_bounds(v, 3, logt), mad_bounds(v, 5, logt)
        res["metrics"][key] = {
            "quantiles": q,
            "min": round(float(v.min()), 3), "max": round(float(v.max()), 3),
            "mean": round(float(v.mean()), 3),
            "mad3": [round(m3[0], 2), round(m3[1], 2)],
            "mad5": [round(m5[0], 2), round(m5[1], 2)],
        }

    # How much of the sample each candidate mitochondrial ceiling would remove.
    mt = o["pct_counts_mt"].values
    res["mito_survival"] = {
        f"cap_{c}": round(100 * float((mt <= c).mean()), 1)
        for c in [5, 10, 15, 20, 25, 30, 40, 50]
    }
    # Joint survival under a realistic combined filter, to catch interaction effects.
    ng, tc = o["n_genes_by_counts"].values, o["total_counts"].values
    res["joint_survival"] = {}
    for cap in [10, 15, 20, 25, 30]:
        keep = (mt <= cap) & (ng >= 500) & (ng <= 9000) & (tc >= 1000)
        res["joint_survival"][f"mt{cap}_g500_9000_c1000"] = {
            "n": int(keep.sum()), "pct": round(100 * float(keep.mean()), 1)}

    (out / f"{a.sample_name}_qc_scan.json").write_text(json.dumps(res, indent=2))

    # Diagnostic panel: distributions on the scale the thresholds are chosen on.
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    ax[0].hist(np.log10(tc + 1), bins=120, color="#2E5EA6")
    ax[0].set_xlabel("log10 total_counts"); ax[0].set_title("counts/cell")
    ax[1].hist(np.log10(ng + 1), bins=120, color="#2E5EA6")
    ax[1].set_xlabel("log10 n_genes"); ax[1].set_title("genes/cell")
    ax[2].hist(mt, bins=120, color="#AF4438")
    for c in (10, 20, 30):
        ax[2].axvline(c, color="k", ls="--", lw=.8)
    ax[2].set_xlabel("pct_counts_mt"); ax[2].set_title("mitochondrial %")
    ax[3].scatter(np.log10(tc + 1), mt, s=1, alpha=.15, color="#57626E")
    ax[3].set_xlabel("log10 total_counts"); ax[3].set_ylabel("pct mt")
    ax[3].set_title("counts vs mito")
    fig.suptitle(f"{a.sample_name} — pre-filter QC ({rna.n_obs} barcodes)")
    fig.tight_layout()
    fig.savefig(out / f"{a.sample_name}_qc_scan.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {out}/{a.sample_name}_qc_scan.json")


if __name__ == "__main__":
    sys.exit(main())
