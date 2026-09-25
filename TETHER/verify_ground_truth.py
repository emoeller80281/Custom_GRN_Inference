"""Assert the built ground truth only contains sources that were deliberately allowed.

The point is the Kupffer decision. `KCs` takes ReMap `kupffer-cell` and nothing else: no
bone-marrow-derived macrophage, no whole-tissue liver. That is a judgement a string matcher
would get wrong -- ChIP-Atlas `Macrophages` looks like a match for `KCs` on the name alone --
so it lives in cell_type_map.tsv as an explicit EXCLUDED row rather than as an omission.

An omission is invisible. A later edit that adds a source, or a rerun with --include_optin,
would quietly put hepatocyte binding under a Kupffer label and nothing would complain. This
turns that into a build failure.

Run:  python3 verify_ground_truth.py --organism mm10
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# Sources that must never reach these cell types, whatever the map says. Belt and braces:
# the map is data and can be edited; this is the decision itself.
FORBIDDEN = {
    "KCs": {
        "reason": "bone-marrow-derived or whole-tissue, not resident Kupffer cells",
        "terms": {"Macrophages", "macrophage", "primary-macrophage", "BMDM", "iBMDM",
                  "RAW264-7", "peritoneal-macrophage", "alveolar-macrophage",
                  "granulocyte-macrophage-progenitor", "Liver", "liver"},
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organism", default="mm10")
    ap.add_argument("--map", type=Path, default=HERE / "cell_type_map.tsv")
    ap.add_argument("--gt_dir", type=Path, default=None)
    args = ap.parse_args()

    gt_dir = args.gt_dir or (HERE / "ground_truth" / args.organism)
    files = sorted(gt_dir.glob("*_ground_truth.parquet"))
    if not files:
        print(f"No ground truth in {gt_dir}. Run build_celltype_ground_truth.py first.")
        return 1

    m = pd.read_csv(args.map, sep="\t", comment="#", dtype=str)
    m = m[m["organism"] == args.organism]
    declared = {(r.dataset_cell_type, r.source, r.source_term): int(r.proxy_tier)
                for r in m.itertuples()}
    excluded = {(r.dataset_cell_type, r.source, r.source_term)
                for r in m.itertuples() if str(r.note).startswith("EXCLUDED")}

    failures, n_edges = [], 0
    print(f"{'cell type':14s} {'sources':>8s} {'edges':>10s}  tiers")
    for f in files:
        d = pd.read_parquet(f)
        # The builder writes <name with spaces as underscores>_ground_truth.parquet, so a file
        # name cannot be turned back into a name that itself contains underscores (the mESC
        # slices, e.g. Epiblast_lineage). Take the name from the file's own cell_type column;
        # fall back to the file name only for an empty file.
        stem = f.name.replace("_ground_truth.parquet", "")
        ct = (str(d["cell_type"].iloc[0]) if not d.empty and "cell_type" in d.columns
              else stem.replace("_", " "))
        if ct.replace(" ", "_") != stem:
            failures.append(f"{f.name}: holds cell_type {ct!r}, which does not match its name")
        if d.empty:
            print(f"{ct:14s} {0:8d} {0:10d}  (no edges)")
            continue
        n_edges += len(d)

        for col in ("Source", "Target", "cell_type", "source", "source_term", "proxy_tier"):
            if col not in d.columns:
                failures.append(f"{ct}: built file has no {col!r} column")
        if "cell_type" in d.columns and set(d.cell_type.unique()) != {ct}:
            failures.append(f"{ct}: file contains cell_type values {set(d.cell_type.unique())}")

        used = d[["source", "source_term", "proxy_tier"]].drop_duplicates()
        print(f"{ct:14s} {len(used):8d} {len(d):10,d}  "
              f"{sorted(set(d.proxy_tier))}")

        for r in used.itertuples():
            key = (ct, r.source, r.source_term)
            if key not in declared:
                failures.append(
                    f"{ct}: source ({r.source}, {r.source_term}) is in the built ground "
                    f"truth but has no row in cell_type_map.tsv -- unmapped sources are "
                    f"how a wrong cell type's binding gets in unnoticed"
                )
            elif key in excluded:
                failures.append(
                    f"{ct}: source ({r.source}, {r.source_term}) is marked EXCLUDED in "
                    f"cell_type_map.tsv but its edges are in the built ground truth"
                )
            elif declared[key] != int(r.proxy_tier):
                failures.append(
                    f"{ct}: ({r.source}, {r.source_term}) is tier {r.proxy_tier} in the "
                    f"built file but tier {declared[key]} in the map"
                )

            forb = FORBIDDEN.get(ct)
            if forb and r.source_term in forb["terms"]:
                failures.append(
                    f"{ct}: source ({r.source}, {r.source_term}) is forbidden -- "
                    f"{forb['reason']}"
                )

    print(f"\n{len(files)} cell types, {n_edges:,} edges checked.")
    for ct, forb in FORBIDDEN.items():
        path = gt_dir / f"{ct.replace(' ', '_')}_ground_truth.parquet"
        if path.exists():
            d = pd.read_parquet(path)
            terms = sorted(set(d.source_term)) if not d.empty else []
            print(f"{ct} sources: {terms or 'none'}  "
                  f"(forbidden: {forb['reason']})")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nEvery source in the built ground truth is declared in cell_type_map.tsv "
          "with a matching tier, and no forbidden source is present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
