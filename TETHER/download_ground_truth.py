"""Fetch the per-cell-type ChIP-seq peak sets named in a <dataset>_cell_type_map.tsv.

Output goes to data/ground_truth_files/cell_type_specific/{chipatlas,remap}/, which is the
--gt_dir that scripts/build_celltype_ground_truth.py reads. Files already there are reused.

Two sources, two very different access patterns:

  ChIP-Atlas  one assembled BED per (cell type class, cell type), downloaded over HTTP.
              The URL encodes the cell type with ChIP-Atlas's own rules -- space becomes
              "_" and "+" becomes "PULUS" -- so source_term in the map is stored already
              encoded and is used verbatim.

  ReMap 2022  no working per-biotype endpoint any more (the documented
              storage/remap2022/mm10/MACS2/TF/<biotype>/ path returns 404). The whole-genome
              "all" BED is downloaded once and every biotype is cut out of it in a SINGLE
              streaming pass, because scanning 3.2 GB once per biotype would be ~60 scans.

The ReMap name field is "GSE.TF.biotype", and the biotype half carries an optional
experimental-condition suffix joined by "_" ("pre-B-cell_48h", "MEF_3T3"). No biotype in
ReMap's own biotype list contains an underscore (checked: 0 of 374), so splitting the third
dot-field on its first "_" recovers the biotype and keeps every condition variant of it.
Dropping that suffix instead of matching it exactly is the difference between 3 Kupffer
datasets and 0.

Only rows with default_enabled=1 are fetched unless --include_optin is passed. Rows whose
note starts with EXCLUDED are never fetched -- they are in the map so the provenance check
can name the decision rather than stay silent about it.

Usage:
    python3 download_ground_truth.py --organism mm10 \
        --map ../data/ground_truth_files/cell_type_specific/GSE209610_kidney_controls_cell_type_map.tsv
    python3 download_ground_truth.py --organism mm10 --map <map> --include_optin
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
DATA_DIR = PROJECT_DIR / "data"

# ChIP-Atlas BEDs already downloaded for earlier work; reuse instead of refetching.
SHARED_GT_DIR = DATA_DIR / "ground_truth_files"
REMAP_ALL_BED = SHARED_GT_DIR / "remap" / "remap2022_all_macs2_{organism}_v1_0.bed.gz"

# Per-cell-type peak sets land in chipatlas/ and remap/ under here, next to the maps.
CELL_TYPE_GT_DIR = SHARED_GT_DIR / "cell_type_specific"

CHIPATLAS_URL = (
    "https://chip-atlas.dbcls.jp/data/{organism}/assembled/"
    "Oth.{cls}.05.AllAg.{term}.bed"
)


def read_map(map_path: Path, organism: str, include_optin: bool) -> pd.DataFrame:
    """Load cell_type_map.tsv, dropping comment lines, and select the rows to fetch."""
    df = pd.read_csv(map_path, sep="\t", comment="#", dtype=str)
    df["proxy_tier"] = df["proxy_tier"].astype(int)
    df["default_enabled"] = df["default_enabled"].astype(int)
    df = df[df["organism"] == organism].copy()

    excluded = df["note"].str.startswith("EXCLUDED")
    wanted = df["default_enabled"] == 1
    if include_optin:
        wanted = wanted | (~excluded)

    logging.info(
        "cell_type_map: %d rows for %s -- %d to fetch, %d opt-in, %d excluded by decision",
        len(df), organism, wanted.sum(), ((df["default_enabled"] == 0) & ~excluded).sum(),
        excluded.sum(),
    )
    return df[wanted].copy()


def fetch_chipatlas(rows: pd.DataFrame, organism: str, out_dir: Path, force: bool) -> dict:
    """Download one assembled BED per ChIP-Atlas row. Returns {source_term: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    got = {}
    for _, r in rows.iterrows():
        fname = f"Oth.{r.source_class}.05.AllAg.{r.source_term}.bed"
        dest = out_dir / fname.replace("/", "_")

        # An earlier notebook already pulled some of these into data/ground_truth_files/.
        shared = SHARED_GT_DIR / fname
        if not dest.exists() and shared.exists() and not force:
            shutil.copy(shared, dest)
            logging.info("  reused %s from ground_truth_files/", fname)
            got[r.source_term] = dest
            continue

        if dest.exists() and not force:
            got[r.source_term] = dest
            continue

        url = CHIPATLAS_URL.format(organism=organism, cls=r.source_class, term=r.source_term)
        try:
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                tmp = dest.with_suffix(".partial")
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                tmp.rename(dest)
            logging.info("  downloaded %s (%.1f MB)", fname, dest.stat().st_size / 1e6)
            got[r.source_term] = dest
        except requests.HTTPError as exc:
            # Not fatal: a cell type can be absent at this q-value. The breadth report is
            # what records the consequence, so log loudly and carry on.
            logging.warning("  MISSING %s -- %s", fname, exc)
    return got


def fetch_remap(rows: pd.DataFrame, organism: str, out_dir: Path, force: bool) -> dict:
    """Cut every requested biotype out of the ReMap all-peaks BED in one streaming pass."""
    out_dir.mkdir(parents=True, exist_ok=True)
    biotypes = sorted(rows["source_term"].unique())
    if not biotypes:
        return {}

    pending = [b for b in biotypes if force or not (out_dir / f"{b}.bed").exists()]
    if not pending:
        logging.info("  all %d ReMap biotypes already extracted", len(biotypes))
        return {b: out_dir / f"{b}.bed" for b in biotypes}

    src = Path(str(REMAP_ALL_BED).format(organism=organism))
    if not src.exists():
        raise FileNotFoundError(
            f"ReMap all-peaks BED not found at {src}. Download it first:\n"
            f"  curl -C - -o {src} https://remap.univ-amu.fr/storage/remap2022/"
            f"{organism}/MACS2/remap2022_all_macs2_{organism}_v1_0.bed.gz"
        )

    wanted_file = out_dir / ".wanted_biotypes.txt"
    wanted_file.write_text("\n".join(pending) + "\n")
    logging.info("  streaming %s once for %d biotypes...", src.name, len(pending))

    decomp = "pigz -dc" if shutil.which("pigz") else "gunzip -c"
    awk = r'''
      NR==FNR { want[$0]=1; next }
      {
        n = split($4, a, ".");
        bt = a[3]; for (i = 4; i <= n; i++) bt = bt "." a[i];
        u = index(bt, "_"); if (u > 0) bt = substr(bt, 1, u - 1);
        if (bt in want) print > (outdir "/" bt ".bed");
      }
    '''
    cmd = (
        f"{decomp} {src} | awk -v outdir={out_dir} -F'\\t' '{awk}' {wanted_file} -"
    )
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
    wanted_file.unlink()

    got = {}
    for b in biotypes:
        p = out_dir / f"{b}.bed"
        if p.exists() and p.stat().st_size > 0:
            got[b] = p
            logging.info("  %-28s %8.1f MB", b, p.stat().st_size / 1e6)
        else:
            logging.warning("  MISSING ReMap biotype %s -- no peaks matched", b)
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organism", default="mm10")
    ap.add_argument("--map", type=Path, required=True,
                    help="a <dataset>_cell_type_map.tsv, e.g. in " + str(CELL_TYPE_GT_DIR))
    ap.add_argument("--out_dir", type=Path, default=CELL_TYPE_GT_DIR)
    ap.add_argument("--include_optin", action="store_true",
                    help="also fetch default_enabled=0 rows that are not EXCLUDED")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    rows = read_map(args.map, args.organism, args.include_optin)

    logging.info("\nChIP-Atlas:")
    ca = fetch_chipatlas(rows[rows["source"] == "chipatlas"], args.organism,
                         args.out_dir / "chipatlas", args.force)
    logging.info("\nReMap:")
    rm = fetch_remap(rows[rows["source"] == "remap"], args.organism,
                     args.out_dir / "remap", args.force)

    n_ca = (rows["source"] == "chipatlas").sum()
    n_rm = rows[rows["source"] == "remap"]["source_term"].nunique()
    logging.info("\nFetched %d/%d ChIP-Atlas and %d/%d ReMap sources.",
                 len(ca), n_ca, len(rm), n_rm)


if __name__ == "__main__":
    main()
