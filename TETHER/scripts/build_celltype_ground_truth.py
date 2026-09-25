"""Turn per-cell-type ChIP-seq peaks into per-cell-type TF-TG edges.

Output schema is (Source, Target, cell_type) plus the provenance the evaluation stratifies
on, which is the whole point of the project: today `utils.load_ground_truth` drops every
column but two and `load_ground_truth_files` concatenates cell types together, so an edge
True in hepatocytes is labeled True in Kupffer cells too.

Pipeline per (cell_type, source_term):

  1. awk reduces the BED to 4 columns (chrom, start, end, TF) in one streaming pass. This
     happens before anything reaches Python because the ChIP-Atlas assembled files carry a
     long URL-encoded metadata blob per peak -- B_cells is 1.2 GB, of which the coordinates
     and antigen are a few percent.
  2. TF names are normalized with .capitalize(), which is what utils.load_ground_truth does
     and what the embedding table uses (Arid5a, Tfap2a, Smad4).
  3. TFs absent from the FROZEN embedding table are dropped and counted. Most such drops are
     correct -- cohesin, remodelers and coactivators have no motif and so no DNA-binding
     embedding. Names that fail only because the antigen is ambiguous are handled by
     tf_alias.tsv, which defaults to dropping them rather than guessing a paralog.
  4. bedtools closest -d -t first assigns each peak its single nearest TSS. `closest` rather
     than `window` because these peak sets run to millions of rows and an all-pairs window
     over a 100 kb radius does not finish.

The TSS annotation is mm10_gene_tss.bed (238,516 transcript entries), NOT gene_tss.bed
(25,120 gene entries) -- build_peak_to_gene_dist.py:13-25 records that using the wrong one
changes the answer completely.

Usage:
    python3 build_celltype_ground_truth.py --organism mm10
    python3 build_celltype_ground_truth.py --organism mm10 --max_tss_dist 1000000
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent.parent
DATA_DIR = PROJECT_DIR / "data"
TF_IDX = PROJECT_DIR / "TETHER" / "cached_data" / "{organism}" / "tf_dna_cache" / "tf_name_to_idx.csv"

# Matches MAX_PEAK_DISTANCE in build_peak_to_gene_dist.py, so the label side and the input
# side of the model use the same peak-to-gene radius.
DEFAULT_MAX_TSS_DIST = 100_000

# ChIP-Atlas: col4 is "ID=SRX...;Name=Tox3%20(@%20Hepatocytes);Title=...". Take the Name
# value and cut it at the "%20(@" that introduces the cell type.
AWK_CHIPATLAS = r'''
  /^track/ || /^browser/ || /^#/ { next }
  {
    n = split($4, parts, ";");
    tf = "";
    for (i = 1; i <= n; i++) {
      if (substr(parts[i], 1, 5) == "Name=") {
        tf = substr(parts[i], 6);
        p = index(tf, "%20(@");
        if (p > 0) tf = substr(tf, 1, p - 1);
        break;
      }
    }
    if (tf != "") print $1 "\t" $2 "\t" $3 "\t" tf;
  }
'''

# ReMap: col4 is "GSE.TF.biotype[_condition]".
AWK_REMAP = r'''
  /^track/ || /^browser/ || /^#/ { next }
  { n = split($4, a, "."); if (n >= 2) print $1 "\t" $2 "\t" $3 "\t" a[2]; }
'''


def load_embedding_tfs(organism: str) -> set:
    path = Path(str(TF_IDX).format(organism=organism))
    if not path.exists():
        raise SystemExit(f"Missing frozen TF embedding table: {path}")
    tfs = pd.read_csv(path)["tf_name"]
    logging.info("Frozen embedding table: %d TFs (%s)", len(tfs), path.name)
    return set(tfs.str.upper())


def reduce_bed(bed_path: Path, source: str, out_path: Path) -> int:
    """Stream a ChIP BED down to chrom/start/end/TF. Returns the row count written."""
    awk = AWK_CHIPATLAS if source == "chipatlas" else AWK_REMAP
    cmd = f"awk -F'\\t' '{awk}' {bed_path} > {out_path}"
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
    return int(subprocess.run(f"wc -l < {out_path}", shell=True, capture_output=True,
                              text=True, check=True).stdout.strip())


def closest_tss(peaks_bed: Path, tss_sorted: Path, tmp: Path) -> pd.DataFrame:
    """bedtools closest -d, one nearest TSS per peak. Returns tf/target/dist."""
    srt = tmp / (peaks_bed.stem + ".sorted.bed")
    out = tmp / (peaks_bed.stem + ".closest.bed")
    subprocess.run(f"sort -k1,1 -k2,2n {peaks_bed} > {srt}", shell=True, check=True,
                   executable="/bin/bash")
    subprocess.run(
        f"bedtools closest -a {srt} -b {tss_sorted} -d -t first > {out}",
        shell=True, check=True, executable="/bin/bash",
    )
    df = pd.read_csv(
        out, sep="\t", header=None, usecols=[3, 7, 8],
        names=["tf", "target", "dist"],
        dtype={3: str, 7: str}, na_filter=False,
    )
    srt.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    return df


def build_cell_type(cell_type: str, rows: pd.DataFrame, gt_dir: Path, tss_sorted: Path,
                    emb_tfs: set, max_dist: int, tmp: Path) -> tuple:
    """Return (edges_df, per-source breadth records) for one dataset cell type."""
    edges, records = [], []

    for r in rows.itertuples():
        if r.source == "chipatlas":
            fname = f"Oth.{r.source_class}.05.AllAg.{r.source_term}.bed".replace("/", "_")
            bed = gt_dir / "chipatlas" / fname
        else:
            bed = gt_dir / "remap" / f"{r.source_term}.bed"

        rec = {"cell_type": cell_type, "source": r.source, "source_term": r.source_term,
               "proxy_tier": r.proxy_tier, "resolution": r.resolution,
               "n_peaks": 0, "n_tfs_raw": 0, "n_tfs_usable": 0,
               "n_tfs_dropped_no_embedding": 0, "n_edges": 0, "status": "ok"}

        if not bed.exists():
            rec["status"] = "source_file_missing"
            records.append(rec)
            logging.warning("    %-34s MISSING FILE", r.source_term)
            continue

        reduced = tmp / f"{cell_type.replace(' ', '_')}__{r.source}__" \
                        f"{r.source_term.replace('/', '_')}.bed"
        n_peaks = reduce_bed(bed, r.source, reduced)
        rec["n_peaks"] = n_peaks
        if n_peaks == 0:
            rec["status"] = "no_peaks_parsed"
            records.append(rec)
            logging.warning("    %-34s parsed 0 peaks", r.source_term)
            reduced.unlink(missing_ok=True)
            continue

        hits = closest_tss(reduced, tss_sorted, tmp)
        reduced.unlink(missing_ok=True)

        # Normalize gene names to uppercase
        raw_tfs = set(hits["tf"].str.upper())
        rec["n_tfs_raw"] = len(raw_tfs)

        # Filter to TFs that have embeddings
        hits = hits[hits["tf"].str.upper().isin(emb_tfs)]
        rec["n_tfs_dropped_no_embedding"] = len(raw_tfs) - hits["tf"].str.upper().nunique()

        # Filter to peaks within the max distance of a valid target gene
        hits["dist"] = pd.to_numeric(hits["dist"], errors="coerce")
        hits = hits[(hits["dist"] >= 0) & (hits["dist"] <= max_dist)]
        hits = hits[hits["target"] != "."]

        # Build the edge list
        edge_df = pd.DataFrame({
            "Source": hits["tf"].str.upper(),
            "Target": hits["target"].str.upper(),
        }).drop_duplicates()

        rec["n_tfs_usable"] = edge_df["Source"].nunique()
        rec["n_edges"] = len(edge_df)
        records.append(rec)
        logging.info("    %-34s %10s peaks  %3d/%3d TFs  %9s edges",
                     r.source_term, f"{n_peaks:,}", rec["n_tfs_usable"], rec["n_tfs_raw"],
                     f"{len(edge_df):,}")

        if len(edge_df) > 0:
            edge_df["cell_type"] = cell_type
            edge_df["source"] = r.source
            edge_df["source_term"] = r.source_term
            edge_df["proxy_tier"] = r.proxy_tier
            edges.append(edge_df)

    if not edges:
        return pd.DataFrame(columns=["Source", "Target", "cell_type", "source",
                                     "source_term", "proxy_tier"]), records

    all_edges = pd.concat(edges, ignore_index=True)
    
    # One row per (Source, Target). When several sources give the same edge, keep the best
    # (lowest) tier, so an edge is never reported as looser evidence than it actually has.
    all_edges = all_edges.sort_values("proxy_tier").drop_duplicates(subset=["Source", "Target"],
                                                          keep="first")
    return all_edges.reset_index(drop=True), records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--organism", default="mm10")
    ap.add_argument("--map", type=Path, default=HERE / "cell_type_map.tsv")
    ap.add_argument("--gt_dir", type=Path, default=HERE / "ground_truth")
    ap.add_argument("--out_dir", type=Path, default=HERE / "ground_truth")
    ap.add_argument("--dataset", default=None,
                    help="write to <out_dir>/<organism>/<dataset>/ instead of "
                         "<out_dir>/<organism>/, so datasets that share slice names "
                         "(B cells, Endothelial) do not overwrite each other")
    ap.add_argument("--reports", type=Path, default=HERE / "reports")
    ap.add_argument("--tss_bed", type=Path, default=None)
    ap.add_argument("--max_tss_dist", type=int, default=DEFAULT_MAX_TSS_DIST)
    ap.add_argument("--include_optin", action="store_true")
    ap.add_argument("--cell_types", nargs="*", default=None)
    ap.add_argument("--report_tag", default=None,
                    help="breadth report name, gt_breadth_<tag>.tsv (default: the organism). "
                         "Give the mESC build its own tag so it does not overwrite the liver one.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if not shutil.which("bedtools"):
        raise SystemExit("bedtools is not on PATH.")

    tss = args.tss_bed or (DATA_DIR / "genome_data" / "genome_annotation" / args.organism
                           / f"{args.organism}_gene_tss.bed")
    if not tss.exists():
        raise SystemExit(f"Missing TSS annotation: {tss}")

    # Read in the cell type ChIP-seq source map
    cell_type_gt_map = pd.read_csv(args.map, sep="\t", comment="#", dtype=str)
    cell_type_gt_map["proxy_tier"] = cell_type_gt_map["proxy_tier"].astype(int)
    cell_type_gt_map["default_enabled"] = cell_type_gt_map["default_enabled"].astype(int)
    
    # Filter the map to the organism
    cell_type_gt_map = cell_type_gt_map[cell_type_gt_map["organism"] == args.organism]
    
    # Exclude any sources where the note starts with "EXCLUDED" unless --include_optin is specified
    excluded = cell_type_gt_map["note"].str.startswith("EXCLUDED")
    keep = (cell_type_gt_map["default_enabled"] == 1) | (args.include_optin & ~excluded)
    cell_type_gt_map = cell_type_gt_map[keep]
    
    # Optional: filter to a specific list of cell types if provided
    if args.cell_types:
        cell_type_gt_map = cell_type_gt_map[cell_type_gt_map["dataset_cell_type"].isin(args.cell_types)]

    # Load the frozen TF embedding table
    emb_tfs = load_embedding_tfs(args.organism)

    out_dir = args.out_dir / args.organism
    if args.dataset:
        out_dir = out_dir / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="gtbuild_", dir=out_dir))
    tss_sorted = tmp / "tss.sorted.bed"
    subprocess.run(f"sort -k1,1 -k2,2n {tss} > {tss_sorted}", shell=True, check=True,
                   executable="/bin/bash")
    logging.info("TSS annotation: %s  |  max TSS distance: %s bp\n",
                 tss.name, f"{args.max_tss_dist:,}")

    # For each cell type, build the TF-TG edges and write them to a parquet file.
    all_records = []
    for cell_type, rows in cell_type_gt_map.groupby("dataset_cell_type", sort=True):
        logging.info("%s  (%d source(s))", cell_type, len(rows))
        edges, records = build_cell_type(cell_type, rows, args.gt_dir, tss_sorted,
                                         emb_tfs, args.max_tss_dist, tmp)
        all_records.extend(records)
        dest = out_dir / f"{cell_type.replace(' ', '_')}_ground_truth.parquet"
        edges.to_parquet(dest, index=False)
        logging.info("  -> %-46s %s edges, %d TFs, %d TGs\n", dest.name,
                     f"{len(edges):,}", edges.Source.nunique(), edges.Target.nunique())

    shutil.rmtree(tmp, ignore_errors=True)

    rep = pd.DataFrame(all_records)
    rep_path = args.reports / f"gt_breadth_{args.report_tag or args.organism}.tsv"
    rep.to_csv(rep_path, sep="\t", index=False)
    logging.info("Per-source breadth report -> %s", rep_path)


if __name__ == "__main__":
    main()
