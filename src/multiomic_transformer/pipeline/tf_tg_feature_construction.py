#!/usr/bin/env python3
"""
tf_tg_feature_construction.py
-----------------------------
Stage 3: Integrate TF–TG regulatory potential, pseudobulk expression,
and peak–gene distance features into a unified TF–TG feature table.

Inputs:
  - pseudobulk_expr.parquet
  - peak_gene_links.parquet
  - tf_tg_regulatory_potential.parquet

Outputs:
  - tf_tg_features.parquet
  - .done checkpoint

Features include:
  - mean_tf_expr, mean_tg_expr
  - expr_product
  - pearson_corr, spearman_corr
  - reg_potential, log_reg_pot
  - motif_density, motif_present
  - neg_log_tss_dist
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

from multiomic_transformer.pipeline.io_utils import (
    ensure_dir,
    write_parquet_safe,
    checkpoint_exists,
    write_done_flag,
    StageTimer,
)
from multiomic_transformer.pipeline.config import get_paths


# =====================================================================
# Utility: compute correlations across pseudobulk samples
# =====================================================================
def compute_tf_tg_correlations(expr_df: pd.DataFrame, tfs: list[str], tgs: list[str]) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations between TFs and TGs
    across pseudobulk samples.

    Args:
        expr_df : DataFrame (genes × samples)
        tfs : list of TF gene names present in expr_df.index
        tgs : list of TG gene names present in expr_df.index

    Returns:
        pd.DataFrame with columns:
          ["TF", "TG", "pearson_corr", "spearman_corr"]
    """
    tf_tg_corr = []
    expr_df = expr_df.loc[expr_df.index.intersection(tfs + tgs)]

    # Convert to float32 for efficiency
    expr_mat = expr_df.astype(np.float32)

    for tf in tfs:
        if tf not in expr_mat.index:
            continue
        tf_values = expr_mat.loc[tf].values
        for tg in tgs:
            if tg not in expr_mat.index:
                continue
            tg_values = expr_mat.loc[tg].values
            if np.all(tf_values == 0) or np.all(tg_values == 0):
                pear, spear = np.nan, np.nan
            else:
                pear, _ = pearsonr(tf_values, tg_values)
                spear, _ = spearmanr(tf_values, tg_values)
            tf_tg_corr.append((tf, tg, pear, spear))

    return pd.DataFrame(tf_tg_corr, columns=["TF", "TG", "pearson_corr", "spearman_corr"])


# =====================================================================
# Main function: integrate features
# =====================================================================
def run_tf_tg_feature_construction(
    pseudobulk_file: Path,
    reg_potential_file: Path,
    peak_gene_links_file: Path,
    output_file: Path,
    force: bool = False,
):
    """
    Main entrypoint for Stage 3 TF–TG feature construction.
    """
    if checkpoint_exists(output_file) and not force:
        logging.info(f"[SKIP] {output_file} already exists.")
        return

    with StageTimer("TF–TG Feature Construction"):

        # ---------------------------------------------------------------
        # 1. Load inputs
        # ---------------------------------------------------------------
        logging.info(f"Loading pseudobulk expression: {pseudobulk_file}")
        expr_df = pd.read_parquet(pseudobulk_file)
        if expr_df.shape[0] < expr_df.shape[1]:
            # ensure genes × samples
            logging.warning("Transposing pseudobulk expression to [genes × samples]")
            expr_df = expr_df.T

        logging.info(f"Loading regulatory potential: {reg_potential_file}")
        tf_tg_reg = pd.read_parquet(reg_potential_file)

        logging.info(f"Loading peak–gene links: {peak_gene_links_file}")
        peak_gene_links = pd.read_parquet(peak_gene_links_file)

        # ---------------------------------------------------------------
        # 2. Derive TF and TG sets
        # ---------------------------------------------------------------
        tfs = sorted(tf_tg_reg["TF"].unique().tolist())
        tgs = sorted(tf_tg_reg["TG"].unique().tolist())
        logging.info(f"Found {len(tfs)} TFs and {len(tgs)} TGs in regulatory potential file")

        # ---------------------------------------------------------------
        # 3. Mean expression features (across pseudobulk samples)
        # ---------------------------------------------------------------
        common_tfs = [tf for tf in tfs if tf in expr_df.index]
        common_tgs = [tg for tg in tgs if tg in expr_df.index]

        mean_tf_expr = expr_df.loc[common_tfs].mean(axis=1).rename("mean_tf_expr")
        mean_tg_expr = expr_df.loc[common_tgs].mean(axis=1).rename("mean_tg_expr")

        mean_tf_expr = mean_tf_expr.reset_index().rename(columns={"index": "TF"})
        mean_tg_expr = mean_tg_expr.reset_index().rename(columns={"index": "TG"})

        # ---------------------------------------------------------------
        # 4. Correlation features (across pseudobulk samples)
        # ---------------------------------------------------------------
        logging.info("Computing TF–TG correlations across pseudobulk samples")
        corr_df = compute_tf_tg_correlations(expr_df, common_tfs, common_tgs)
        logging.info(f"Correlation matrix computed for {len(corr_df):,} TF–TG pairs")

        # ---------------------------------------------------------------
        # 5. Merge all feature sources
        # ---------------------------------------------------------------
        merged = tf_tg_reg.merge(mean_tf_expr, on="TF", how="left")
        merged = merged.merge(mean_tg_expr, on="TG", how="left")
        merged = merged.merge(corr_df, on=["TF", "TG"], how="left")

        # Merge in distance-based features if available
        if {"peak_id", "TSS_dist"}.issubset(peak_gene_links.columns):
            # normalize TG naming
            if "TG" not in peak_gene_links.columns and "gene_id" in peak_gene_links.columns:
                peak_gene_links = peak_gene_links.rename(columns={"gene_id": "TG"})

            dist_df = (
                peak_gene_links[["peak_id", "TG", "TSS_dist"]]
                .groupby("TG", as_index=False)
                .agg(TSS_dist=("TSS_dist", "mean"))
            )
            merged = merged.merge(dist_df, on="TG", how="left")
            merged["neg_log_tss_dist"] = -np.log1p(merged["TSS_dist"].fillna(0))
        else:
            logging.warning("No TSS distance column found — skipping distance features.")

        # ---------------------------------------------------------------
        # 6. Derived features
        # ---------------------------------------------------------------
        merged["expr_product"] = merged["mean_tf_expr"] * merged["mean_tg_expr"]
        merged["log_reg_pot"] = np.log1p(merged.get("reg_potential", 0))
        merged["motif_present"] = (merged.get("motif_density", 0) > 0).astype(int)
        
        numeric_casts = [
            "reg_potential", "expr_product", "log_reg_pot", "neg_log_tss_dist",
            "mean_tf_expr", "mean_tg_expr", "pearson_corr", "spearman_corr", "motif_density"
        ]
        for col in numeric_casts:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)


        # ---------------------------------------------------------------
        # 7. Save output
        # ---------------------------------------------------------------
        if merged is None or merged.empty:
            logging.warning("No TF–TG features generated; skipping Parquet write.")
            return

        try:
            ensure_dir(output_file.parent)
            write_parquet_safe(merged, output_file)
            write_done_flag(output_file)
            logging.info(f"[DONE] Stage 3 complete → {output_file} ({merged.shape[0]:,} rows)")
        except Exception as e:
            logging.error(f"Failed to write TF–TG features: {e}")
            raise

# =====================================================================
# CLI
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Stage 3: TF–TG Feature Construction")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. PBMC)")
    parser.add_argument("--sample", required=True, help="Sample or replicate name")
    parser.add_argument("--organism", default="mm10", help="Reference genome ID")
    parser.add_argument("--num_cpu", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Force recompute even if done flag exists")
    args = parser.parse_args()

    paths = get_paths(args=args)
    outdir = paths["processed_dir"]

    pseudobulk_file = outdir / "pseudobulk_expr.parquet"
    reg_potential_file = outdir / "tf_tg_regulatory_potential.parquet"
    peak_gene_links_file = outdir / "peak_gene_links.parquet"
    output_file = outdir / "tf_tg_features.parquet"

    run_tf_tg_feature_construction(
        pseudobulk_file,
        reg_potential_file,
        peak_gene_links_file,
        output_file,
        force=args.force,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
