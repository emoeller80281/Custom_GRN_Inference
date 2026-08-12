"""Estimate the ambient (soup) contamination fraction without empty droplets.

The standard estimators need the raw droplet table: SoupX profiles the soup from empty
droplets, CellBender models them explicitly. Only GEO-deposited *filtered* matrices exist
for this dataset and there are no BAM/FASTQ to regenerate from, so the soup profile has to
come from the cells themselves.

The substitute is cell-type-exclusive genes. If a gene is genuinely off in a cell type,
every UMI of it observed in those cells came from the soup. Under the usual model a cell's
observed fraction for gene g is

    f_obs(g) = (1 - rho) * f_true(g) + rho * f_soup(g)

so for cells where f_true(g) = 0 this reduces to f_obs(g) = rho * f_soup(g), giving

    rho = f_obs(g) in known-negative cells / f_soup(g)

with the soup profile approximated by the global expression profile (the standard
approximation -- soup is lysed-cell material, so it resembles average expression weighted
by abundance).

Genes are chosen automatically: for each candidate, the source cluster is the one with the
highest mean, and the negative set is the clusters in the bottom half. Reporting per-gene
estimates rather than a single number matters, because the approximation is only as good as
the gene's exclusivity, and the spread shows how much to trust it.
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

# Genes that should be exclusive to one lineage in a gastrulation dataset. Chosen to be
# highly expressed in their source (so the soup signal is measurable) and genuinely off
# elsewhere.
EXCLUSIVE = [
    "Hbb-bh1", "Hba-a1", "Hbb-y", "Alas2",      # primitive erythroid
    "Myh6", "Ttn", "Myl7", "Tnnt2", "Nppa",     # cardiomyocyte
    "Ttr", "Apoa1", "Afp", "Apoa4",             # visceral / ExE endoderm
    "Elf5", "Rhox5",                            # ExE ectoderm
    "Sox10",                                    # neural crest
    "Hand1", "Postn",                           # ExE mesoderm
    "Krt8", "Krt18",                            # epithelial
    "Pou5f1",                                   # epiblast
]


def log(m):
    print(f"[ambient] {m}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5mu", default="data/processed/mESC/combined/mESC_combined.h5mu")
    p.add_argument("--cluster_col", default="leiden")
    p.add_argument("--out_dir", default="data/processed/mESC/combined/ambient")
    p.add_argument("--neg_quantile", type=float, default=0.5,
                   help="Clusters below this quantile of mean expression are the negatives")
    a = p.parse_args()

    import muon as mu

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log("loading ...")
    mdata = mu.read(a.h5mu)
    r = mdata["rna"]
    counts = r.layers["counts"]
    counts = counts.tocsc() if sp.issparse(counts) else counts
    total = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    total[total == 0] = np.nan
    cl = r.obs[a.cluster_col].astype(str).values
    samples = r.obs["sample"].astype(str).values
    log(f"{r.n_obs} cells, median {np.nanmedian(total):.0f} UMIs")

    rows = []
    for g in EXCLUSIVE:
        if g not in r.var_names:
            continue
        j = int(np.nonzero(r.var_names == g)[0][0])
        v = np.asarray(counts[:, j].todense()).ravel().astype(float) if sp.issparse(counts) \
            else np.asarray(counts[:, j]).ravel().astype(float)
        frac = v / total                       # per-cell UMI fraction of this gene
        per_cl = pd.Series(frac).groupby(cl).mean()
        source = per_cl.idxmax()
        thr = per_cl.quantile(a.neg_quantile)
        neg_clusters = per_cl[per_cl <= thr].index.tolist()
        neg = np.isin(cl, neg_clusters)

        f_soup = float(np.nanmean(frac))       # global profile approximates the soup
        f_neg = float(np.nanmean(frac[neg]))
        f_src = float(per_cl[source])
        if f_soup <= 0:
            continue
        rho = f_neg / f_soup
        rows.append({
            "gene": g, "source_cluster": source,
            "frac_in_source": f_src, "frac_in_negatives": f_neg, "frac_global": f_soup,
            "rho_estimate": round(float(rho), 4),
            "pct_cells_detected_negatives": round(100 * float((v[neg] > 0).mean()), 1),
            "neg_to_source_ratio": round(float(f_neg / f_src), 4) if f_src > 0 else np.nan,
            "n_negative_clusters": len(neg_clusters),
        })

    df = pd.DataFrame(rows).sort_values("rho_estimate")
    df.to_csv(out / "ambient_contamination_by_gene.csv", index=False)
    pd.set_option("display.width", 220)
    log("\n=== per-gene contamination estimates ===")
    log(df[["gene", "source_cluster", "frac_in_source", "frac_in_negatives",
            "rho_estimate", "pct_cells_detected_negatives",
            "neg_to_source_ratio"]].to_string(index=False))

    rho_med = float(df["rho_estimate"].median())
    log(f"\nmedian rho across {len(df)} genes: {rho_med:.3f}  "
        f"(IQR {df['rho_estimate'].quantile(.25):.3f}-{df['rho_estimate'].quantile(.75):.3f})")
    log(f"=> roughly {100*rho_med:.0f}% of a typical cell's UMIs look like soup")

    # Per-sample, since contamination is a property of the library prep.
    per_sample = {}
    log("\n=== per-sample (median rho over the same genes) ===")
    for s in sorted(set(samples)):
        ms = samples == s
        vals = []
        for _, rr in df.iterrows():
            g = rr["gene"]
            j = int(np.nonzero(r.var_names == g)[0][0])
            v = np.asarray(counts[:, j].todense()).ravel().astype(float) if sp.issparse(counts) \
                else np.asarray(counts[:, j]).ravel().astype(float)
            frac = v / total
            per_cl = pd.Series(frac[ms]).groupby(cl[ms]).mean()
            if per_cl.empty:
                continue
            thr = per_cl.quantile(a.neg_quantile)
            neg = np.isin(cl, per_cl[per_cl <= thr].index.tolist()) & ms
            f_soup = float(np.nanmean(frac[ms]))
            if f_soup > 0 and neg.sum() > 0:
                vals.append(float(np.nanmean(frac[neg])) / f_soup)
        if vals:
            per_sample[s] = round(float(np.median(vals)), 3)
            log(f"  {s:<14} rho ~ {per_sample[s]:.3f}")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(df["gene"], df["rho_estimate"], color="#7A3F58")
    ax.axhline(rho_med, ls="--", color="#243B6B", label=f"median {rho_med:.2f}")
    ax.set_ylabel("estimated soup fraction (rho)")
    ax.set_title("Ambient contamination estimated from lineage-exclusive genes")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(frameon=False)
    for s_ in ax.spines.values():
        s_.set_color("#3A424C")
    fig.tight_layout()
    fig.savefig(out / "ambient_contamination.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (out / "ambient_summary.json").write_text(json.dumps({
        "method": "rho = mean UMI fraction in known-negative clusters / global mean UMI "
                  "fraction, per lineage-exclusive gene; soup profile approximated by the "
                  "global expression profile",
        "caveat": "No empty droplets available (GEO filtered matrices only), so SoupX "
                  "autoEstCont and CellBender cannot be used; this is a proxy.",
        "n_genes": int(len(df)), "rho_median": round(rho_med, 3),
        "rho_iqr": [round(float(df["rho_estimate"].quantile(.25)), 3),
                    round(float(df["rho_estimate"].quantile(.75)), 3)],
        "per_sample_rho": per_sample,
    }, indent=2, default=str))
    log("Done.")


if __name__ == "__main__":
    sys.exit(main())
