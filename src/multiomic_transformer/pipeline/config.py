#!/usr/bin/env python3
"""
config.py
-----------
Configuration and argument parsing for the MultiomicTransformer + GAT unified preprocessing pipeline.

This module defines:
  • Command-line arguments
  • Default constants (paths, genome references, etc.)
  • Helper functions for path construction and sanity checks
"""

import argparse
import os
from pathlib import Path
import socket
import datetime


# ============================================================
# 🔧 Default constants
# ============================================================

DEFAULT_ORGANISM = "mm10"
SUPPORTED_ORGANISMS = ["mm10", "hg38"]

DEFAULT_BASE_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/")
DEFAULT_DATA_DIR = DEFAULT_BASE_DIR / "data"
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_LOG_DIR = DEFAULT_PROCESSED_DIR / "logs"

# Reference genome paths (extendable)
GENOME_REFERENCES = {
    "mm10": {
        "tss_bed": str(DEFAULT_DATA_DIR / "genome_annotation/mm10/gene_tss.bed"),
        "annotation": str(DEFAULT_DATA_DIR / "reference_genome/mm10/Mus_musculus.GRCm39.113.gtf"),
    },
    "hg38": {
        "tss_bed": str(DEFAULT_DATA_DIR / "genome_annotation/hg38/gene_tss.bed"),
        "annotation": str(DEFAULT_DATA_DIR / "reference_genome/hg38/Homo_sapiens.GRCh38.113.gtf"),
    },
}


# ============================================================
# CLI Argument Parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Unified preprocessing pipeline for GRN inference — prepares data for both "
            "MultiomicTransformer and GAT models."
        )
    )

    # --- core dataset information ---
    parser.add_argument("--dataset", type=str, default="PBMC",
                        help="Dataset name (used for directory naming).")
    parser.add_argument("--sample", type=str, default="LINGER_PBMC_SC_DATA",
                        help="Sample or replicate identifier.")
    parser.add_argument("--organism", type=str, default=DEFAULT_ORGANISM, choices=SUPPORTED_ORGANISMS,
                        help="Reference genome ID (mm10 or hg38).")

    # --- runtime control ---
    parser.add_argument("--mode", type=str, default="both", choices=["transformer", "gat", "both"],
                        help="Which downstream target to prepare data for.")
    parser.add_argument("--num_cpu", type=int, default=8,
                        help="Number of CPU cores for parallel tasks.")
    parser.add_argument("--keep_intermediate", type=lambda x: str(x).lower() in ["true", "1", "yes"],
                        default=True, help="Whether to keep intermediate files after completion.")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Force recomputation of all stages, overwriting cached outputs.")

    # --- paths ---
    parser.add_argument("--base_dir", type=str, default=str(DEFAULT_BASE_DIR),
                        help="Base project directory (default: Uzun lab project root).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Custom output directory (optional).")

    args = parser.parse_args()
    return args


# ============================================================
# Helper functions for path resolution
# ============================================================

def get_paths(args=None, base_dir=None):
    """
    Returns standardized output paths.
    Accepts either:
      - argparse.Namespace (from CLI)
      - explicit base_dir (for testing)
    """
    if args is not None and hasattr(args, "outdir") and args.outdir:
        base_dir = Path(args.outdir)
    elif base_dir is not None:
        base_dir = Path(base_dir)
    else:
        base_dir = Path.cwd() / "outputs"

    base_dir.mkdir(parents=True, exist_ok=True)
    processed_data = base_dir / "processed_data"
    processed_data.mkdir(exist_ok=True)

    return {
        "base": base_dir,
        "processed_data": processed_data,
    }


# ============================================================
# Metadata utilities
# ============================================================

def get_run_metadata(args):
    """Return run metadata including timestamp, host, and job info."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    host = socket.gethostname()
    slurm_job_id = os.getenv("SLURM_JOB_ID", "N/A")

    return {
        "timestamp": timestamp,
        "host": host,
        "slurm_job_id": slurm_job_id,
        "dataset": args.dataset,
        "sample": args.sample,
        "mode": args.mode,
        "organism": args.organism,
    }


# ============================================================
# Example usage (testing only)
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    paths = get_paths(args)
    meta = get_run_metadata(args)
    print("Parsed arguments:")
    print(args)
    print("\nResolved paths:")
    for k, v in paths.items():
        print(f"  {k:20s}: {v}")
    print("\nRun metadata:")
    print(meta)
