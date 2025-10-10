import os
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr


def compute_expression_stats(tf_tensor, tg_tensor, tf_names, tg_names):
    """
    Compute mean/variance per TF/TG and pairwise TF–TG correlation.
    """
    tf_expr = tf_tensor.cpu().numpy()
    tg_expr = tg_tensor.cpu().numpy()

    tf_mean = tf_expr.mean(axis=1)
    tf_var = tf_expr.var(axis=1)
    tg_mean = tg_expr.mean(axis=1)
    tg_var = tg_expr.var(axis=1)

    tf_df = pd.DataFrame({
        "TF_name": tf_names,
        "TF_mean_expr": tf_mean,
        "TF_var_expr": tf_var
    })
    tg_df = pd.DataFrame({
        "TG_name": tg_names,
        "TG_mean_expr": tg_mean,
        "TG_var_expr": tg_var
    })

    # Compute pairwise correlations for all TF–TG combinations
    corr_records = []
    for tf_i, tf_name in enumerate(tf_names):
        for tg_i, tg_name in enumerate(tg_names):
            corr, _ = pearsonr(tf_expr[tf_i], tg_expr[tg_i])
            corr_records.append((tf_name, tg_name, corr))

    corr_df = pd.DataFrame(corr_records, columns=["TF_name", "TG_name", "TF_TG_expr_corr"])
    corr_df["TF_TG_expr_corr"] = corr_df["TF_TG_expr_corr"].fillna(0.0)

    return tf_df, tg_df, corr_df


def build_tf_tg_features_from_cache(training_data_cache, out_csv):
    """
    Aggregates motif-, distance-, and expression-based TF–TG features across all chromosomes.
    """
    chr_dirs = [d for d in os.listdir(training_data_cache) if d.startswith("chr")]
    all_features = []

    # Load global TF expression (shared across chromosomes)
    tf_tensor_path = os.path.join(training_data_cache, "tf_tensor_all.pt")
    tf_names_path = os.path.join(training_data_cache, "tf_names.json")
    if not os.path.exists(tf_tensor_path) or not os.path.exists(tf_names_path):
        raise FileNotFoundError("Missing global TF expression or TF names file.")
    tf_tensor = torch.load(tf_tensor_path, map_location="cpu")
    with open(tf_names_path) as f:
        tf_names = json.load(f)

    for chr_dir in tqdm(chr_dirs, desc="Building TF–TG features across chromosomes"):
        chr_path = os.path.join(training_data_cache, chr_dir)
        moods_path = os.path.join(chr_path, f"moods_sites_{chr_dir}.tsv")
        dist_path = os.path.join(chr_path, f"genes_near_peaks_{chr_dir}.parquet")
        tg_tensor_path = os.path.join(chr_path, f"tg_tensor_all_{chr_dir}.pt")
        tg_names_path = os.path.join(chr_path, f"tg_names_{chr_dir}.json")

        if not (os.path.isfile(moods_path) and os.path.isfile(dist_path) and os.path.isfile(tg_tensor_path)):
            print(f"Skipping {chr_dir} — missing required files.")
            continue

        moods_df = pd.read_csv(moods_path, sep="\t")
        dist_df = pd.read_parquet(dist_path)
        tg_tensor = torch.load(tg_tensor_path, map_location="cpu")
        with open(tg_names_path) as f:
            tg_names = json.load(f)

        # Normalize column names
        moods_df = moods_df.rename(columns={"TF": "TF_name"})
        dist_df = dist_df.rename(columns={"target_id": "TG_name"})

        # Merge motif and distance info by peak
        merged = moods_df.merge(dist_df, on="peak_id", how="inner")

        # Aggregate per TF–TG pair
        agg = (
            merged.groupby(["TF_name", "TG_name"])
            .agg(
                n_peaks_linking=("peak_id", "nunique"),
                n_motifs_linking=("logodds", "count"),
                mean_motif_score=("logodds", "mean"),
                max_motif_score=("logodds", "max"),
                min_tss_dist=("TSS_dist", "min"),
                mean_tss_score=("TSS_dist_score", "mean")
            )
            .reset_index()
        )

        # Compute expression stats and correlations
        tf_df, tg_df, corr_df = compute_expression_stats(tf_tensor, tg_tensor, tf_names, tg_names)

        # Merge in TF/TG summaries and correlations
        agg = (
            agg.merge(tf_df, on="TF_name", how="left")
               .merge(tg_df, on="TG_name", how="left")
               .merge(corr_df, on=["TF_name", "TG_name"], how="left")
        )

        agg["chrom"] = chr_dir
        all_features.append(agg)

    # Combine all chromosomes
    all_features_df = pd.concat(all_features, ignore_index=True)

    # Derived ratios / log-transforms
    all_features_df["motif_density"] = (
        all_features_df["n_motifs_linking"] / (all_features_df["n_peaks_linking"] + 1e-6)
    )
    all_features_df["log_mean_score"] = np.log1p(all_features_df["mean_motif_score"])
    all_features_df["neg_log_tss"] = -np.log1p(all_features_df["min_tss_dist"])

    all_features_df.to_csv(out_csv, index=False)
    print(f"Saved global TF–TG feature table: {out_csv}  shape={all_features_df.shape}")

    return all_features_df


if __name__ == "__main__":
    training_data_cache = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/training_data_cache/mESC"
    out_csv = os.path.join(training_data_cache, "tf_tg_features_all_chr.csv")
    build_tf_tg_features_from_cache(training_data_cache, out_csv)
