"""Distinguish genuinely recovered cells from ambient contamination in rescued subclusters.

Sub-clustering cluster 0 produced two candidate populations with strong, specific markers:
sub 3 (94 cells; Ttn, Myh6, Myl7, Actc1) and sub 1 (1,689 cells; Pou5f1, Tdgf1). Both are
dominated by a single library, so before either is promoted to a real label the ambient
explanation has to be excluded -- abundant transcripts from a population present elsewhere
in the same library contaminate every droplet from it.

Marker detection *frequency* cannot separate the two, because the comparison group in the
sub-clustering was the rest of cluster 0, which is mostly a different sample. Absolute
expression can:

* **Ambient** produces a low, roughly uniform floor across all droplets from that library.
  Rescued cells would then sit near the same-library floor and far below the established
  cluster of that cell type.
* **Genuine cells** express their markers at a level comparable to the established cluster,
  many-fold above the same-library floor -- even at low sequencing depth, because these
  transcripts are extremely abundant in the real cell type.

So each candidate is compared on three fronts: the same-library floor (other cells of that
sample, excluding the candidate), the established reference cluster for that identity, and
the candidate itself.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# subcluster -> (identity, markers, reference cluster in the integrated object)
#
# "reference_cluster" IDs are pinned to the specific Leiden run documented in handoff.md; the
# genuine/ambient verdict itself only compares against the same-library floor and doesn't
# depend on this ID being current, but the printed "vs reference cluster" context does. Re-verify
# before reusing after any re-clustering.
CANDIDATES = {
    "3": {"identity": "Cardiomyocytes",
          "markers": ["Ttn", "Myh6", "Myl7", "Actc1", "Tnnt2", "Nkx2-5", "Slc8a1"],
          "reference_cluster": "22"},
    "1": {"identity": "Epiblast / Primitive Streak",
          "markers": ["Pou5f1", "Tdgf1", "Pim2", "Utf1", "Fgf5", "T", "Mixl1"],
          "reference_cluster": "20"},
    "0": {"identity": "ExE mesoderm (claimed)",
          "markers": ["Hbb-bh1", "Ptn", "Dlk1", "Vcan", "Meis2", "Postn", "Hand1"],
          "reference_cluster": "3"},
}


def log(m):
    print(f"[verify] {m}", flush=True)


def dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5mu", default="data/processed/mESC/combined/mESC_combined.h5mu")
    p.add_argument("--sub_h5ad",
                   default="data/processed/mESC/combined/subcluster_0/cluster0_subclustered.h5ad")
    p.add_argument("--cluster_col", default="leiden")
    p.add_argument("--out_dir", default="data/processed/mESC/combined/subcluster_0")
    # The median fold over the floor is the wrong statistic. Soup is made of the cell
    # type's own most abundant transcripts, so those genes have the HIGHEST floor and the
    # LOWEST fold; low-expressed transcription factors are poorly captured at low depth and
    # also score low. Both drag the median down for genuine cells. Ambient contamination
    # instead makes every gene sit at ~1.0x the floor, so what discriminates is whether any
    # markers rise clearly above it.
    p.add_argument("--floor_fold", type=float, default=5.0,
                   help="Fold over the same-library floor that counts as clearly above it")
    p.add_argument("--min_markers_above", type=int, default=2,
                   help="How many markers must clear --floor_fold to call cells genuine")
    a = p.parse_args()

    import anndata as ad
    import muon as mu
    import scanpy as sc

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log("loading ...")
    mdata = mu.read(a.h5mu)
    r = mdata["rna"]
    full = ad.AnnData(X=r.layers["counts"].copy(),
                      obs=r.obs[[a.cluster_col, "sample", "timepoint"]].copy(),
                      var=pd.DataFrame(index=r.var_names.copy()))
    sc.pp.normalize_total(full, target_sum=1e4)
    sc.pp.log1p(full)

    sub = ad.read_h5ad(a.sub_h5ad)
    subs = sub.obs["subcluster"].astype(str)
    # Map subcluster assignment back onto the full object by barcode.
    lab = pd.Series("other", index=full.obs_names, dtype=object)
    lab.loc[sub.obs_names] = ("sub" + subs).values
    full.obs["sublabel"] = lab.values
    cl = full.obs[a.cluster_col].astype(str)
    samples = full.obs["sample"].astype(str)

    rows, verdicts = [], {}
    for sc_id, spec in CANDIDATES.items():
        tag = f"sub{sc_id}"
        m_cand = (full.obs["sublabel"] == tag).values
        if m_cand.sum() == 0:
            continue
        top_sample = samples[m_cand].value_counts().index[0]
        # Same-library floor: every other cell from that library, candidate excluded.
        m_floor = (samples == top_sample).values & ~m_cand
        m_ref = (cl == spec["reference_cluster"]).values & ~m_cand

        log(f"\n=== {tag}: {int(m_cand.sum())} cells, claimed '{spec['identity']}' ===")
        log(f"  library floor: {top_sample} ({int(m_floor.sum())} other cells)")
        log(f"  reference: cluster {spec['reference_cluster']} ({int(m_ref.sum())} cells)")
        log(f"  {'gene':<10}{'candidate':>11}{'lib_floor':>11}{'reference':>11}"
            f"{'vs_floor':>10}{'vs_ref':>9}")
        fold_floor, fold_ref = [], []
        for g in spec["markers"]:
            if g not in full.var_names:
                continue
            v = dense(full[:, g].X).ravel()
            c_m, f_m, r_m = float(v[m_cand].mean()), float(v[m_floor].mean()), float(v[m_ref].mean())
            vf = c_m / f_m if f_m > 0 else np.inf
            vr = c_m / r_m if r_m > 0 else np.inf
            fold_floor.append(vf); fold_ref.append(vr)
            log(f"  {g:<10}{c_m:>11.3f}{f_m:>11.3f}{r_m:>11.3f}{vf:>10.2f}{vr:>9.2f}")
            rows.append({"subcluster": tag, "identity": spec["identity"], "gene": g,
                         "mean_candidate": round(c_m, 4), "mean_library_floor": round(f_m, 4),
                         "mean_reference_cluster": round(r_m, 4),
                         "fold_over_floor": round(float(vf), 2),
                         "fold_vs_reference": round(float(vr), 2)})
        fold_floor = np.array(fold_floor, dtype=float)
        n_above = int((fold_floor >= a.floor_fold).sum())
        med_floor = float(np.median(fold_floor)) if fold_floor.size else np.nan
        max_floor = float(np.nanmax(fold_floor)) if fold_floor.size else np.nan
        med_ref = float(np.median(fold_ref)) if fold_ref else np.nan
        genuine = n_above >= a.min_markers_above
        log(f"  fold over library floor: median {med_floor:.2f}, max {max_floor:.2f}, "
            f"{n_above} marker(s) >= {a.floor_fold}x (needs {a.min_markers_above})")
        log(f"  expression relative to reference cluster: median {med_ref:.2f}x "
            f"(lower is expected at lower depth)")
        log(f"  => {'GENUINE CELLS' if genuine else 'AMBIENT / NOT A DISTINCT POPULATION'}")
        verdicts[tag] = {"identity": spec["identity"], "n_cells": int(m_cand.sum()),
                         "top_sample": top_sample,
                         "median_fold_over_library_floor": round(med_floor, 2),
                         "max_fold_over_library_floor": round(max_floor, 2),
                         "n_markers_above_floor_fold": n_above,
                         "median_fold_vs_reference": round(med_ref, 2),
                         "genuine": bool(genuine)}

    pd.DataFrame(rows).to_csv(out / "rescued_subcluster_verification.csv", index=False)
    (out / "rescued_subcluster_verdicts.json").write_text(json.dumps({
        "test": "absolute expression vs same-library ambient floor and vs the established "
                "reference cluster for that identity",
        "criteria": {"floor_fold": a.floor_fold,
                     "min_markers_above_floor_fold": a.min_markers_above},
        "verdicts": verdicts,
    }, indent=2, default=str))
    log("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
