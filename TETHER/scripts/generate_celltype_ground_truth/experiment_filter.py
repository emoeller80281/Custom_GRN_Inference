"""Per-experiment filter: drop single bad ChIP experiments instead of whole source terms.

A cell_type_map.tsv row is a whole source term (a ChIP-Atlas cell type, a ReMap biotype).
Without this filter, one bad experiment in a term (forced expression of the TF, an input
control filed under a TF name, a ChIP in the TF's own knockout) forces the whole term off,
and every good experiment in it is lost with it.

The bad experiments are listed by ID in <organism>_excluded_experiments.tsv, next to the maps:
  srx           ChIP-Atlas experiment (SRX...), or the ReMap dataset name GSE.TF.biotype[_condition]
  source_class  ChIP-Atlas cell-type class, or "remap"
  source_term   the map's source_term the experiment belongs to
With the list given, download_ground_truth.py and build_celltype_ground_truth.py:
  1. drop the peaks of every listed experiment from every source they read, enabled or not;
  2. also use rows that are disabled only because of listed experiments: default_enabled=0,
     a note starting with one of RESCUE_PREFIXES, and at least one experiment of that term on
     the list. A term disabled for forced expression whose experiments were never listed
     (BJ, Hep_G2, hTERT_RPE-1, BLaER1 in the hg38 maps) stays off, because the filter would
     not remove anything from it.
EXCLUDED rows are never used. Without the list, row selection is exactly as before.
"""
from pathlib import Path

import pandas as pd

# Note prefixes of rows that are disabled only because of experiments on the list:
# "forced expression - ..." (guide B1) and "B3 - ..." (an input control or knockout ChIP filed
# under a TF; the map notes keep the pre-2026-11-02 gate number).
RESCUE_PREFIXES = ("forced expression", "B3 - ")


def default_list(gt_dir: Path, organism: str) -> Path:
    return gt_dir / f"{organism}_cell_type_maps" / f"{organism}_excluded_experiments.tsv"


def load_excluded(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str).fillna("")
    missing = {"srx", "source_class", "source_term"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    if df["srx"].duplicated().any():
        raise SystemExit(f"{path}: duplicated IDs {df.loc[df['srx'].duplicated(), 'srx'].tolist()[:5]}")
    return df


def term_key(source: str, source_class: str, source_term: str) -> tuple:
    """Key that matches a map row to its listed experiments ("remap" stands in for the class)."""
    return ("remap" if source == "remap" else source_class, source_term)


def listed_by_term(excluded: pd.DataFrame) -> dict:
    """{(class or "remap", source_term): set of listed IDs}."""
    return {k: set(g["srx"]) for k, g in excluded.groupby(["source_class", "source_term"])}


def select_rows(m: pd.DataFrame, include_optin: bool, excluded: pd.DataFrame = None) -> pd.DataFrame:
    """Rows to use, with a boolean column `enabled_by_filter` for rows only the filter turns on."""
    m = m.copy()
    is_excluded = m["note"].fillna("").str.startswith("EXCLUDED")
    wanted = m["default_enabled"].astype(int) == 1
    rescued = pd.Series(False, index=m.index)
    if excluded is not None:
        listed = listed_by_term(excluded)
        has_listed = pd.Series([term_key(s, c, t) in listed for s, c, t in
                                zip(m["source"], m["source_class"].fillna(""), m["source_term"])], index=m.index)
        rescued = (~wanted & ~is_excluded & m["note"].fillna("").str.startswith(RESCUE_PREFIXES) & has_listed)
    if include_optin:
        wanted = wanted | ~is_excluded
    m["enabled_by_filter"] = rescued & ~wanted
    return m[(wanted | rescued) & ~is_excluded]
