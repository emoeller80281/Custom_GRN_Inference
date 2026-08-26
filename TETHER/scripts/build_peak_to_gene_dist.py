"""Build peak_to_gene_dist.parquet for samples that lack one.

Every TETHER entry point that reads a sample (`generate_all_predictions.py`,
`build_tf_to_tg_train_data.py`, `plot_auprc_all_methods.py`, ...) requires this file, but
`muon_preprocessing.py` does not produce it -- it was previously made by hand in
`peak_to_gene.ipynb`, whose driver cell is hardcoded to one dataset at a time. This is that
same logic (`format_peaks` / `find_genes_near_peaks` are copied unchanged from the
notebook's first cell) with a CLI, so a batch of new samples can be filled in at once.

The peak universe comes from the sample's own RE_pseudobulk.parquet index, so the output
always matches the pseudobulk it sits next to -- rebuild this whenever that is rebuilt.

Two details were reverse-engineered from the existing mESC files rather than taken from the
notebook, which had drifted from what actually produced them:

  * the TSS annotation is ``<species>_gene_tss.bed`` (238,516 transcript-level 1 bp entries),
    NOT the ``gene_tss.bed`` the notebook's driver cell points at (25,120 zero-width,
    gene-level entries). Using the wrong one changes the answer completely.
  * exactly one row per peak -- its **nearest** TSS. Keeping every link inside the window
    instead gives ~27 peaks per gene rather than ~4.5, which would silently reshape the
    edge bags that `max_peaks_per_tg` slices.

Rebuilding E8.5_rep1 this way reproduces its existing file exactly: 221,047 rows, 100.00%
identical TSS_dist, 99.98% identical target_id (the remainder are ties between equidistant
TSS entries).

Usage:
    python3 scripts/build_peak_to_gene_dist.py --species mm10 --cell_type mESC
    python3 scripts/build_peak_to_gene_dist.py --species mm10 --cell_type mESC \
        --samples E8.5_CRISPR_T_KO E8.5_CRISPR_T_WT --force
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pybedtools

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
DATA_DIR = PROJECT_DIR / "data"

# Matches the existing mESC/hepatocyte files: max TSS_dist in those is ~99,998, and
# build_tether_inputs.py uses the same 20 kb decay for the score column.
MAX_PEAK_DISTANCE = 100_000
TSS_DECAY = 20_000
NEAREST_GENE_ONLY = True


def format_peaks(peak_ids: pd.Series | pd.Index) -> pd.DataFrame:
    """Split 'chrN:start-end' / 'chrN-start-end' peak IDs into BED-ish columns."""
    if peak_ids.empty:
        raise ValueError("Input peak ID list is empty.")

    peak_ids = peak_ids[peak_ids.str.contains("chr", regex=False)]
    if peak_ids.empty:
        raise ValueError("No peak IDs containing 'chr' were found after filtering.")

    parsed = peak_ids.str.extract(
        r'.*?(?P<chromosome>chr[^\s:-]+)(?::|-)(?P<start>\d+)-(?P<end>\d+)\s*$'
    )
    missing = parsed["chromosome"].isnull() | parsed["start"].isnull() | parsed["end"].isnull()
    if missing.any():
        bed_like = peak_ids[missing].str.extract(
            r'^\s*(?P<chromosome>\S+)\s+(?P<start>\d+)\s+(?P<end>\d+)(?:\s|$)'
        )
        parsed.loc[missing, ["chromosome", "start", "end"]] = bed_like[["chromosome", "start", "end"]].values

    parsed["chromosome"] = parsed["chromosome"].str.extract(r'(chr[^\s:]*)$', expand=False)

    if parsed[["chromosome", "start", "end"]].isnull().any().any():
        bad = peak_ids[parsed[["chromosome", "start", "end"]].isnull().any(axis=1)].head(3).tolist()
        raise ValueError(f"Malformed peak IDs, e.g. {bad}")

    peak_df = pd.DataFrame({
        "chromosome": parsed["chromosome"],
        "start": pd.to_numeric(parsed["start"], errors="coerce").astype(int),
        "end": pd.to_numeric(parsed["end"], errors="coerce").astype(int),
        "strand": ["."] * len(peak_ids),
    })
    peak_df["peak_id"] = (
        peak_df["chromosome"].astype(str) + ":"
        + peak_df["start"].astype(str) + "-"
        + peak_df["end"].astype(str)
    )
    return peak_df


def find_genes_near_peaks(peak_bed, tss_bed, tss_distance_cutoff=1e6):
    """Peaks within tss_distance_cutoff bp of a gene TSS, with |peak_end - gene_start|."""
    peak_tss_overlap = peak_bed.window(tss_bed, w=tss_distance_cutoff)

    cols = ["peak_chr", "peak_start", "peak_end", "peak_id",
            "gene_chr", "gene_start", "gene_end", "gene_id"]
    df = peak_tss_overlap.to_dataframe(names=cols, low_memory=False)

    for c in ["peak_start", "peak_end", "gene_start", "gene_end"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["peak_start", "peak_end", "gene_start", "gene_end"]).copy()

    df["TSS_dist"] = np.abs(df["peak_end"].values - df["gene_start"].values)
    return df.sort_values("TSS_dist")


def default_tss_bed(species):
    """Prefer <species>_gene_tss.bed -- that is what the existing samples were built with."""
    annotation_dir = DATA_DIR / "genome_data" / "genome_annotation" / species
    preferred = annotation_dir / f"{species}_gene_tss.bed"
    return preferred if preferred.exists() else annotation_dir / "gene_tss.bed"


def build_for_sample(sample_dir, tss_bed_file, force=False):
    out_file = sample_dir / "peak_to_gene_dist.parquet"
    if out_file.exists() and not force:
        logging.info(f"  {sample_dir.name}: already has one, skipping")
        return None

    atac_file = sample_dir / "RE_pseudobulk.parquet"
    if not atac_file.exists():
        logging.warning(f"  {sample_dir.name}: no RE_pseudobulk.parquet, skipping")
        return None

    # columns=[] reads only the index, not the (multi-GB) cell columns.
    peak_index = pd.read_parquet(atac_file, columns=[]).index
    logging.info(f"  {sample_dir.name}: {len(peak_index):,} peaks")

    peak_locs = format_peaks(pd.Series(peak_index)).rename(columns={"chromosome": "chrom"})
    peak_bed = pybedtools.BedTool.from_dataframe(peak_locs[["chrom", "start", "end", "peak_id"]])
    tss_bed = pybedtools.BedTool(str(tss_bed_file))

    hits = find_genes_near_peaks(peak_bed, tss_bed, tss_distance_cutoff=MAX_PEAK_DISTANCE)
    hits = hits.rename(columns={"gene_id": "target_id"})
    hits["target_id"] = hits["target_id"].str.upper()
    hits = hits[hits["TSS_dist"] <= MAX_PEAK_DISTANCE].copy()
    if NEAREST_GENE_ONLY:
        hits = hits.sort_values("TSS_dist").drop_duplicates("peak_id", keep="first")
    hits["TSS_dist_score"] = np.exp(-hits["TSS_dist"] / TSS_DECAY)

    hits.to_parquet(out_file, index=False)
    logging.info(
        f"  {sample_dir.name}: wrote {len(hits):,} links "
        f"({hits.peak_id.nunique():,} peaks, {hits.target_id.nunique():,} genes) -> {out_file.name}"
    )
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", default="mm10")
    parser.add_argument("--cell_type", required=True, help="subdirectory of data/sample_input_data")
    parser.add_argument("--samples", nargs="*", default=None, help="default: every sample missing the file")
    parser.add_argument("--tss_bed", default=None, help="override the TSS bed (default: <species>_gene_tss.bed)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the file exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Keep pybedtools' scratch off /tmp -- these intersections spill GBs.
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "pybedtools_p2g"
    scratch.mkdir(parents=True, exist_ok=True)
    pybedtools.helpers.set_tempdir(str(scratch))

    tss_bed_file = Path(args.tss_bed) if args.tss_bed else default_tss_bed(args.species)
    if not tss_bed_file.exists():
        raise SystemExit(f"Missing TSS annotation: {tss_bed_file}")
    logging.info(f"TSS annotation: {tss_bed_file.name}")

    root = DATA_DIR / "sample_input_data" / args.cell_type
    if args.samples:
        sample_dirs = [root / s for s in args.samples]
    else:
        sample_dirs = sorted(
            d for d in root.iterdir()
            if d.is_dir() and (d / "RE_pseudobulk.parquet").exists()
            and not (d / "peak_to_gene_dist.parquet").exists()
        )

    logging.info(f"{len(sample_dirs)} sample(s) to build in {root}")
    for sample_dir in sample_dirs:
        build_for_sample(sample_dir, tss_bed_file, force=args.force)
    logging.info("Done.")


if __name__ == "__main__":
    main()
