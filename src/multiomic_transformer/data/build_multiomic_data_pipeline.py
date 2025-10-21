#!/usr/bin/env python3
"""
build_multiomic_data_pipeline.py
Unified orchestration for the multiomic GRN pipeline.

Stages:
  1. run_qc_and_pseudobulk()
  2. run_peak_gene_mapping()
  3. run_tf_tg_feature_construction()
  4. run_tensorization()
  5. run_dataset_assembly()
  6. run_export_gat_data()

Usage:
  poetry run python build_multiomic_data_pipeline.py \
      --dataset mESC --organism mm10 --outdir /path/to/output \
      --start 1 --stop 6 --rna /path/to/rna.h5ad --atac /path/to/atac.h5ad --force
"""

import argparse
import logging
from pathlib import Path

# --- Stage imports ---
from multiomic_transformer.pipeline.qc_and_pseudobulk import run_qc_and_pseudobulk
from multiomic_transformer.pipeline.peak_gene_mapping import run_peak_gene_mapping
from multiomic_transformer.pipeline.tf_tg_feature_construction import run_tf_tg_feature_construction
from multiomic_transformer.pipeline.tensorization import run_tensorization
from multiomic_transformer.pipeline.dataset_assembly import run_dataset_assembly
from multiomic_transformer.pipeline.export_gat_data import run_export_gat_data
from multiomic_transformer.pipeline.io_utils import StageTimer, ensure_dir, checkpoint_exists

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ---------------------------------------------------------------------
# Stage execution helpers
# ---------------------------------------------------------------------
def stage(func, *args, force=False, **kwargs):
    """Execute a pipeline stage with timing and checkpoint skip."""
    name = func.__name__
    with StageTimer(name):
        try:
            func(*args, force=force, **kwargs)
        except Exception as e:
            logging.error(f"[ERROR] Stage {name} failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Run multiomic data pipeline end-to-end.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., mESC)")
    parser.add_argument("--organism", required=True, help="Organism ID (e.g., mm10, hg38)")
    parser.add_argument("--outdir", required=True, help="Root output directory")
    parser.add_argument("--start", type=int, default=1, help="Start stage index (1–6)")
    parser.add_argument("--stop", type=int, default=6, help="Stop stage index (1–6)")
    parser.add_argument("--rna", required=True, help="Path to scRNA-seq AnnData file (.h5ad or .mtx)")
    parser.add_argument("--atac", required=True, help="Path to scATAC-seq AnnData file (.h5ad or .mtx)")
    parser.add_argument("--force", action="store_true", help="Force re-run even if checkpoints exist")
    args = parser.parse_args()

    base_outdir = Path(args.outdir)
    ensure_dir(base_outdir)

    # --- Stage map ---
    stages = {
        1: ("QC and pseudobulk", run_qc_and_pseudobulk),
        2: ("Peak–gene mapping", run_peak_gene_mapping),
        3: ("TF–TG feature construction", run_tf_tg_feature_construction),
        4: ("Tensorization", run_tensorization),
        5: ("Dataset assembly", run_dataset_assembly),
        6: ("GAT data export", run_export_gat_data),
    }

    logging.info(f"Running stages {args.start} → {args.stop} for dataset '{args.dataset}' ({args.organism})")

    for i in range(args.start, args.stop + 1):
        stage_name, func = stages[i]
        logging.info(f"\n=== [Stage {i}] {stage_name} ===")

        # Stage-specific arguments
        if i == 1:
            stage(func, rna_path=args.rna, atac_path=args.atac, organism=args.organism,
                  outdir=base_outdir, force=args.force)
        elif i == 2:
            stage(func, base_outdir / "pseudobulk_scRNA.parquet",
                        base_outdir / "pseudobulk_scATAC.parquet",
                        output_dir=base_outdir, force=args.force)
        elif i == 3:
            stage(func,
                  pseudobulk_file=base_outdir / "pseudobulk_scRNA.parquet",
                  reg_potential_file=base_outdir / "tf_tg_regulatory_potential.parquet",
                  peak_gene_links_file=base_outdir / "peak_gene_links.parquet",
                  output_file=base_outdir / "tf_tg_features.parquet",
                  force=args.force)
        elif i == 4:
            stage(func,
                  tf_tg_features_file=base_outdir / "tf_tg_features.parquet",
                  output_dir=base_outdir / "tensorization",
                  force=args.force)
        elif i == 5:
            stage(func,
                  tensor_dir=base_outdir / "tensorization",
                  tf_tg_features_file=base_outdir / "tf_tg_features.parquet",
                  output_dir=base_outdir / "assembled",
                  chrom_id="chrAll",
                  force=args.force)
        elif i == 6:
            stage(func,
                  dataset_dir=base_outdir / "assembled",
                  out_dir=base_outdir / "gat_export",
                  force=args.force)
        else:
            logging.warning(f"Skipping unknown stage {i}")

    logging.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
