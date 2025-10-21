#!/usr/bin/env python3
"""
peak_gene_mapping.py
--------------------
Stage 2: Create a peak–gene distance table using BedTools windowing,
consistent with peaks.py and preprocess.py.

Output:
  - peak_gene_distance.parquet with columns:
      peak_id, gene_id, TSS_dist, TSS_dist_score
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas import Index
import pybedtools

from multiomic_transformer.pipeline.io_utils import (
    ensure_dir,
    write_parquet_safe,
    checkpoint_exists,
    write_done_flag,
    StageTimer,
)
from multiomic_transformer.pipeline.peaks import (
    format_peaks,
    find_genes_near_peaks,
)


def run_peak_gene_mapping(
    peaks_file: str,
    tss_bed_file: str,
    organism: str = "mm10",
    outdir: Optional[Path] = None,
    max_distance: int = 1_000_000,
    keep_intermediate: bool = True,   # reserved for future cleanup hook
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Build a peak–gene distance dataframe.

    Parameters
    ----------
    peaks_file : .bed/.bed.gz or .parquet with at least [chrom, start, end]
    tss_bed_file : BED with 4 columns [chrom, start, end, gene_id]
    organism : genome tag (unused here; logged for consistency)
    outdir : output directory
    max_distance : window size for BedTools (bp)
    force_recompute : ignore checkpoint if True
    """
    # --- Normalize paths ---
    peaks_file = str(peaks_file)
    tss_bed_file = str(tss_bed_file)

    outdir = ensure_dir(Path(outdir or "."))
    checkpoint_flag = outdir / ".peak_gene_mapping.done"
    output_file = outdir / "peak_gene_distance.parquet"

    # --- Checkpoint reuse ---
    if checkpoint_exists(checkpoint_flag) and not force_recompute:
        logging.info(f"[peak_gene_mapping] Using cached result: {output_file}")
        return pd.read_parquet(output_file)

    with StageTimer("peak_gene_mapping"):
        logging.info(f"[peak_gene_mapping] Organism={organism}")
        logging.info(f"  • peaks_file: {peaks_file}")
        logging.info(f"  • tss_bed_file: {tss_bed_file}")
        logging.info(f"  • max_distance: {max_distance:,} bp")

        # --------------------------------------------------------------
        # Load peaks
        # --------------------------------------------------------------
        if peaks_file.endswith(".parquet"):
            peaks_df = pd.read_parquet(peaks_file)
            # If this parquet comes from pseudobulk ATAC, it may not have BED-style columns
            if not {"chrom", "start", "end"}.issubset(peaks_df.columns):
                # Reconstruct BED-format from index like 'chr1:123-456'
                logging.info("[peak_gene_mapping] Formatting peaks from index...")
                formatted = format_peaks(pd.Series(peaks_df.index))
            else:
                formatted = format_peaks(
                    pd.Series([f"{c}:{s}-{e}" for c, s, e in zip(peaks_df["chrom"], peaks_df["start"], peaks_df["end"])])
                )
        elif peaks_file.endswith(".bed") or peaks_file.endswith(".bed.gz"):
            peaks_df = pd.read_csv(peaks_file, sep="\t", header=None, names=["chrom", "start", "end"])
            formatted = format_peaks(pd.Series([f"{c}:{s}-{e}" for c, s, e in zip(peaks_df["chrom"], peaks_df["start"], peaks_df["end"])]))
        else:
            raise ValueError("Unsupported peaks file format. Use .bed, .bed.gz, or .parquet.")

        # format_peaks returns a DataFrame with 'chromosome', 'start', 'end', 'peak_id'
        if not isinstance(formatted, pd.DataFrame):
            raise ValueError("format_peaks() must return a DataFrame with a 'peak_id' column.")
        formatted = formatted.rename(columns={"chromosome": "chrom"})
        peak_bed_df = formatted[["chrom", "start", "end", "peak_id"]].copy()

        # Convert to BedTool
        peak_bed = pybedtools.BedTool.from_dataframe(peak_bed_df)

        # --------------------------------------------------------------
        # Load TSS BED
        # --------------------------------------------------------------
        tss_df = pd.read_csv(tss_bed_file, sep="\t", header=None)
        if tss_df.shape[1] < 4:
            raise ValueError("TSS BED must have 4 columns: chrom, start, end, gene_id")
        tss_df.columns = Index(["chrom", "start", "end", "gene_id"], name="gene_id")
        tss_bed = pybedtools.BedTool.from_dataframe(tss_df)

        # --------------------------------------------------------------
        # BedTools window + distance
        # --------------------------------------------------------------
        peak_gene_df = find_genes_near_peaks(
            peak_bed=peak_bed,
            tss_bed=tss_bed,
            tss_distance_cutoff=max_distance,
        )
        # Expect columns from peaks.py: ..., 'peak_id', ..., 'gene_id', 'TSS_dist'

        if "peak_id" not in peak_gene_df.columns or "gene_id" not in peak_gene_df.columns or "TSS_dist" not in peak_gene_df.columns:
            raise ValueError("find_genes_near_peaks() output missing required columns.")

        # --------------------------------------------------------------
        # Distance score (exponential decay)
        # --------------------------------------------------------------
        peak_gene_df["TSS_dist_score"] = np.exp(-peak_gene_df["TSS_dist"] / max_distance)
        
        if "gene_id" in peak_gene_df.columns and "TG" not in peak_gene_df.columns:
            peak_gene_df = peak_gene_df.rename(columns={"gene_id": "TG"})

        # Keep a tidy, downstream-friendly schema
        result = peak_gene_df[["peak_id", "TG", "TSS_dist", "TSS_dist_score"]].copy()

        # --------------------------------------------------------------
        # Save + checkpoint
        # --------------------------------------------------------------
        write_parquet_safe(result, output_file)
        write_done_flag(checkpoint_flag)
        logging.info(f"[peak_gene_mapping] Wrote {len(result):,} rows → {output_file}")

        return result


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Peak–gene mapping")
    parser.add_argument("--peaks_file", required=True, help="Path to peaks (.bed/.bed.gz or .parquet)")
    parser.add_argument("--tss_bed_file", required=True, help="Path to gene TSS BED (chrom, start, end, gene_id)")
    parser.add_argument("--organism", default="mm10", help="Genome identifier (default: mm10)")
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--max_distance", type=int, default=1_000_000, help="Max peak–gene distance (bp)")
    parser.add_argument("--keep_intermediate", action="store_true", help="Keep intermediate outputs")
    parser.add_argument("--force_recompute", action="store_true", help="Force recomputation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_peak_gene_mapping(
        peaks_file=args.peaks_file,
        tss_bed_file=args.tss_bed_file,
        organism=args.organism,
        outdir=Path(args.outdir),
        max_distance=args.max_distance,
        keep_intermediate=args.keep_intermediate,
        force_recompute=args.force_recompute,
    )


if __name__ == "__main__":
    main()
