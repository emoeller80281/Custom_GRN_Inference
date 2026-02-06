from json import load
import os, sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import argparse
import logging

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
GROUND_TRUTH_DIR = Path(PROJECT_DIR, "data/ground_truth_files")
OTHER_METHOD_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/testing_bear_grn/INFERRED.GRNS")

SRC_DIR = str(Path(PROJECT_DIR) / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
    
logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_other_method_grns(sample_name_list, dataset_type):
    
    other_method_grns = {}
    for sample_name in sample_name_list:
        logging.info(f"\nProcessing sample: {sample_name}")
        if dataset_type.lower() == "mesc":
            cell_oracle_path  = OTHER_METHOD_DIR / f"{sample_name}/CellOracle/filtered_L2_{sample_name}_out_E7.5_rep1_final_GRN.csv"
            directnet_path    = OTHER_METHOD_DIR / f"{sample_name}/DIRECTNET/{sample_name}_all_cells_Network_links.csv"
            figr_path         = OTHER_METHOD_DIR / f"{sample_name}/FigR/{sample_name}_all_cells_filtered_network.csv"
            granie_path       = OTHER_METHOD_DIR / f"{sample_name}/GRaNIE/GRN_connections_filtered_sorted_sc{sample_name}_all_cells_selected_uniq.csv"
            linger_path       = OTHER_METHOD_DIR / f"{sample_name}/LINGER/filtered_L2_{sample_name}.csv"
            pando_path        = OTHER_METHOD_DIR / f"{sample_name}/Pando/{sample_name}_all_cells_raw_network.csv"
            scenic_plus_path  = OTHER_METHOD_DIR / f"{sample_name}/SCENIC+/scenic_plus_inferred_grn_mESC_filtered_L2_{sample_name}.tsv"
            tripod_path       = OTHER_METHOD_DIR / f"{sample_name}/TRIPOD/gene_TF_highest_abs_coef.csv"
        
        elif dataset_type.lower() == "macrophage":
            if sample_name == "buffer_1":
                cell_oracle_path  = OTHER_METHOD_DIR / f"Macrophage_S1/CellOracle/Macrophase_buffer1_filtered_out_E7.5_rep1_final_GRN.csv"
                directnet_path    = OTHER_METHOD_DIR / f"Macrophage_S1/DIRECTNET/Network_links.csv"
                figr_path         = OTHER_METHOD_DIR / f"Macrophage_S1/FigR/Buffer1_filtered_network.csv"
                granie_path       = OTHER_METHOD_DIR / f"Macrophage_S1/GRaNIE/GRN_connections_filtered_sorted_scBuffer1_uniq.csv"
                linger_path       = OTHER_METHOD_DIR / f"Macrophage_S1/LINGER/cell_type_TF_gene.csv"
                pando_path        = OTHER_METHOD_DIR / f"Macrophage_S1/Pando/Macrophage_buffer1_raw_network.csv"
                scenic_plus_path  = OTHER_METHOD_DIR / f"Macrophage_S1/SCENIC+/scenic_plus_inferred_grn_macrophage_macrophage_buffer1_filtered.tsv"
                tripod_path       = OTHER_METHOD_DIR / f"Macrophage_S1/TRIPOD/gene_TF_highest_abs_coef.csv"

            elif sample_name == "buffer_2":
                cell_oracle_path  = OTHER_METHOD_DIR / f"Macrophage_S2/CellOracle/Macrophase_buffer2_filtered_out_E7.5_rep1_final_GRN.csv"
                directnet_path    = OTHER_METHOD_DIR / f"Macrophage_S2/DIRECTNET/Network_links.csv"
                figr_path         = OTHER_METHOD_DIR / f"Macrophage_S2/FigR/Buffer2_filtered_network.csv"
                granie_path       = OTHER_METHOD_DIR / f"Macrophage_S2/GRaNIE/GRN_connections_filtered_sorted_scBuffer2_uniq.csv"
                linger_path       = OTHER_METHOD_DIR / f"Macrophage_S2/LINGER/cell_type_TF_gene_buffer2.csv"
                pando_path        = OTHER_METHOD_DIR / f"Macrophage_S2/Pando/Macrophage_buffer2_filtered_network.csv"
                scenic_plus_path  = OTHER_METHOD_DIR / f"Macrophage_S2/SCENIC+/scenic_plus_inferred_grn_macrophage_macrophage_buffer2_filtered.tsv"
                tripod_path       = OTHER_METHOD_DIR / f"Macrophage_S2/TRIPOD/gene_TF_highest_abs_coef.csv"
        
        elif dataset_type.lower() == "k562":
            cell_oracle_path  = OTHER_METHOD_DIR / f"{sample_name}/CellOracle/K562_human_filtered_out_E7.5_rep1_final_GRN.csv"
            directnet_path    = OTHER_METHOD_DIR / f"{sample_name}/DIRECTNET/Network_links.csv"
            figr_path         = OTHER_METHOD_DIR / f"{sample_name}/FigR/K562_filtered_network.csv"
            granie_path       = OTHER_METHOD_DIR / f"{sample_name}/GRaNIE/GRN_connections_filtered_sorted_scK562_uniq.csv"
            linger_path       = OTHER_METHOD_DIR / f"{sample_name}/LINGER/K562_LINGER_GRN_long.tsv"
            pando_path        = OTHER_METHOD_DIR / f"{sample_name}/Pando/K562_raw_network.csv"
            scenic_plus_path  = OTHER_METHOD_DIR / f"{sample_name}/SCENIC+/scenic_plus_inferred_grn_K562_K562_human_filtered.tsv"
            tripod_path       = OTHER_METHOD_DIR / f"{sample_name}/TRIPOD/gene_TF_highest_abs_coef.csv"
            
        method_info = {
            "CellOracle": {"path": cell_oracle_path, "tf_col": "source",    "target_col": "target",    "score_col": "coef_mean"},
            "SCENIC+":    {"path": scenic_plus_path, "tf_col": "Source",    "target_col": "Target",    "score_col": "Score"},
            "Pando":      {"path": pando_path,       "tf_col": "tf",        "target_col": "target",    "score_col": "estimate"},
            "LINGER":     {"path": linger_path,      "tf_col": "Source",    "target_col": "Target",    "score_col": "Score"},
            "FigR":       {"path": figr_path,        "tf_col": "Motif",     "target_col": "DORC",      "score_col": "Score"},
            "TRIPOD":     {"path": tripod_path,      "tf_col": "TF",        "target_col": "gene",      "score_col": "abs_coef"},
            "GRaNIE":     {"path": granie_path,      "tf_col": "TF.name",   "target_col": "gene.name", "score_col": "TF_gene.r"},
            # "DirectNet":   {"path": directnet_path,   "tf_col": "TF",        "target_col": "Gene",    "score_col": "types"},
        }
                
        standardized_method_dict = {}

        other_method_grns[sample_name] = {}
        for method_name, info in method_info.items():
            logging.info(f"  - Loading {method_name}")
            df_std = load_and_standardize_method(method_name, info)
            other_method_grns[sample_name][method_name] = df_std.copy()
    
    # Calculate the mean TF-TG score across samples for each method
    standardized_method_dict = {}
    for method_name in method_info.keys():
        dfs = []
        for sample_name in sample_name_list:
            df = other_method_grns[sample_name][method_name]
            
            df_sample = (
                df.groupby(["Source", "Target"], as_index=False)["Score"]
                .mean()
            )
            
            dfs.append(df_sample)
        
        # Concatenate all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Group by Source and Target to calculate mean Score
        df_std = combined_df.groupby(['Source', 'Target'], as_index=False)['Score'].mean()
        
        standardized_method_dict[method_name] = df_std
            
    return standardized_method_dict

def load_vocab(selected_experiment_dir):
    id2name = torch.load(selected_experiment_dir / "tf_tg_vocab_id2name.pt", map_location="cpu")
    tf_names = list(id2name["tf_id2name"])
    tg_names = list(id2name["tg_id2name"])

    return tf_names, tg_names

def load_ground_truth(ground_truth_file):
    if ground_truth_file.suffix == ".csv":
        sep = ","
    elif ground_truth_file.suffix == ".tsv":
        sep="\t"
        
    ground_truth_df = pd.read_csv(ground_truth_file, sep=sep, on_bad_lines="skip", engine="python")
    
    if "chip" in ground_truth_file.name and "atlas" in ground_truth_file.name:
        ground_truth_df = ground_truth_df[["source_id", "target_id"]]

    ground_truth_df = ground_truth_df.rename(columns={ground_truth_df.columns[0]: "Source", ground_truth_df.columns[1]: "Target"})
    ground_truth_df["Source"] = ground_truth_df["Source"].astype(str).str.upper()
    ground_truth_df["Target"] = ground_truth_df["Target"].astype(str).str.upper()
        
    return ground_truth_df

def prep_gt_edges(gt_df: pd.DataFrame) -> pd.DataFrame:
    gt = gt_df[["Source", "Target"]].dropna().copy()
    gt["Source"] = gt["Source"].astype(str).str.upper()
    gt["Target"] = gt["Target"].astype(str).str.upper()
    gt = gt.drop_duplicates()
    gt["_in_gt"] = 1
    return gt

def eval_method_vs_gt(
    method_df: pd.DataFrame, 
    gt_edges: pd.DataFrame, 
    top_fracs=(0.001, 0.005, 0.01, 0.05), 
    balance=True,
    use_abs_scores=True,
    ) -> dict:
    if method_df is None or len(method_df) == 0:
        # return NaNs but keep consistent columns
        out = {
            "auroc": np.nan, "auprc": np.nan, "pos_rate": np.nan, "lift_auprc": np.nan
        }
        for frac in top_fracs:
            out[f"precision@{frac*100:.2f}%"] = np.nan
            out[f"lift@{frac*100:.2f}%"] = np.nan
        return out

    # Merge GRN with ground truth to label edges
    d = method_df.merge(gt_edges, on=["Source", "Target"], how="left")
    
    def balance_pos_neg(df, random_state=42):
        """Balance the scores for positive and negative classes by inverting negative scores."""
        rng = np.random.default_rng(random_state)
        df = df.copy()
        pos_df = df[df["_in_gt"] == 1]
        neg_df = df[df["_in_gt"] != 1]
        
        n_pos = len(pos_df)
        n_neg = len(neg_df)
        if n_pos == 0 or n_neg == 0:
            logging.info("No positives or negatives, skipping balance")
            return df

        if n_neg < n_pos:
            pos_idx = rng.choice(pos_df.index.to_numpy(), size=n_neg, replace=False)
            pos_sample = pos_df.loc[pos_idx]
            neg_sample = neg_df
        else:
            pos_sample = pos_df
            neg_idx = rng.choice(neg_df.index.to_numpy(), size=n_pos, replace=False)
            neg_sample = neg_df.loc[neg_idx]
        
        balanced = pd.concat([pos_sample, neg_sample], axis=0)
        return balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    if balance == True:
        d = balance_pos_neg(d, random_state=42)
    
    y = d["_in_gt"].fillna(0).astype(int).to_numpy()
    s = d["Score"].to_numpy()
    
    if use_abs_scores:
        s = np.abs(s)
    
    # Calculate AUROC and AUPRC scores
    auroc = roc_auc_score(y, s) if np.unique(y).size == 2 else np.nan
    auprc = average_precision_score(y, s) if y.sum() > 0 else np.nan
    pos_rate = y.mean()

    # Pre-sort once for precision@K
    order = np.argsort(s)[::-1]
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    k = np.arange(1, len(y_sorted) + 1)
    prec = tp / k

    # Calculate precision@K and lift@K for each K
    prec_at = {}
    for frac in top_fracs:
        K = int(frac * len(y_sorted))
        if K < 1:
            K = 1
        if K > len(prec):
            K = len(prec)
        prec_at[f"precision@{frac*100:.2f}%"] = float(prec[K-1]) if len(prec) else np.nan
        prec_at[f"lift@{frac*100:.2f}%"] = float(prec[K-1] / pos_rate) if (len(prec) and pos_rate > 0) else np.nan

    return {
        "auroc": float(auroc) if auroc == auroc else np.nan,
        "auprc": float(auprc) if auprc == auprc else np.nan,
        "pos_rate": float(pos_rate) if pos_rate == pos_rate else np.nan,
        "lift_auprc": float(auprc / pos_rate) if (pos_rate > 0 and auprc == auprc) else np.nan,
        **prec_at
    }, d

def restrict_to_gt_universe(method_df: pd.DataFrame, gt_edges: pd.DataFrame) -> pd.DataFrame:
    gt_tfs = set(gt_edges["Source"])
    gt_tgs = set(gt_edges["Target"])
    return method_df[method_df["Source"].isin(gt_tfs) & method_df["Target"].isin(gt_tgs)].copy()

def load_grad_df_with_two_scores(selected_experiment_dir, tf_names, tg_names):
    grad = np.load(selected_experiment_dir / "tf_tg_grad_attribution.npy").astype(np.float32)
    assert grad.shape == (len(tf_names), len(tg_names))

    grad = np.nan_to_num(grad, nan=0.0)
    grad_abs = np.abs(grad)

    score_pooled = np.log1p(grad_abs)

    # Calculate per-TF robust z-score
    median_val = np.median(grad_abs, axis=1, keepdims=True)
    mad = np.median(np.abs(grad_abs - median_val), axis=1, keepdims=True) + 1e-6
    score_per_tf = (grad_abs - median_val) / mad

    T, G = grad_abs.shape
    tf_idx, tg_idx = np.meshgrid(np.arange(T), np.arange(G), indexing="ij")

    df = pd.DataFrame({
        "Source": np.asarray(tf_names, dtype=object)[tf_idx.ravel()],
        "Target": np.asarray(tg_names, dtype=object)[tg_idx.ravel()],
        "Score_pooled": score_pooled.ravel(),
        "Score_per_tf": score_per_tf.ravel(),
    })
    df["Source"] = df["Source"].astype(str).str.upper()
    df["Target"] = df["Target"].astype(str).str.upper()
    return df

def load_tf_knockout_scores_with_two_scores(
    selected_experiment_dir,
    tf_names,
    tg_names,
    positive_only: bool = True,
    eps: float = 1e-6,
):
    """
    Loads TF-knockout effects and returns a long-form DF with two scores:

      - Score_pooled: log1p(effect_used)  (global magnitude-compressed score)
      - Score_per_tf: robust per-TF score = (effect_used - median_tf) / MAD_tf

    Where effect_used is either:
      - clip(effect, 0, inf) if positive_only=True
      - effect (signed) if positive_only=False

    Notes:
      - Unobserved entries (counts==0) are set to NaN and dropped in the output.
      - If positive_only=True, effects at 0 are valid and retained.
    """
    effect = np.load(selected_experiment_dir / "tf_tg_fullmodel_knockout.npy").astype(np.float32)         # [T, G]
    counts = np.load(selected_experiment_dir / "tf_tg_fullmodel_knockout_count.npy").astype(np.int32)    # [T, G]
    assert effect.shape == (len(tf_names), len(tg_names))
    assert counts.shape == effect.shape

    # Mark unobserved as NaN
    mask_observed = counts > 0
    effect = effect.copy()
    effect[~mask_observed] = np.nan

    # Choose effect representation
    if positive_only:
        effect_used = np.clip(effect, 0, None)  # keep NaNs
    else:
        effect_used = effect  # signed, keep NaNs

    # --- pooled score ---
    # If signed, use abs for pooled magnitude (keeps "strength" notion comparable to gradient pooled)
    pooled_base = effect_used if positive_only else np.abs(effect_used)
    score_pooled = np.log1p(pooled_base)

    # --- per-TF robust score ---
    med = np.nanmedian(effect_used, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(effect_used - med), axis=1, keepdims=True) + eps
    score_per_tf = (effect_used - med) / mad

    # --- build long-form DF ---
    T, G = effect_used.shape
    tf_idx, tg_idx = np.meshgrid(np.arange(T), np.arange(G), indexing="ij")

    df = pd.DataFrame({
        "Source": np.asarray(tf_names, dtype=object)[tf_idx.ravel()],
        "Target": np.asarray(tg_names, dtype=object)[tg_idx.ravel()],
        "Score_pooled": score_pooled.ravel(),
        "Score_per_tf": score_per_tf.ravel(),
        "counts": counts.ravel(),
    })

    # Drop unobserved (Score_pooled will be NaN there)
    df = df.dropna(subset=["Score_pooled"]).reset_index(drop=True)

    df["Source"] = df["Source"].astype(str).str.upper()
    df["Target"] = df["Target"].astype(str).str.upper()
    return df

def load_grad_and_tf_ko_df(selected_experiment_dir):
    tf_names, tg_names = load_vocab(selected_experiment_dir)
    logging.info("Loading Gradient Attribution")
    grad_attrib_df = load_grad_df_with_two_scores(
        selected_experiment_dir=selected_experiment_dir,
        tf_names=tf_names,
        tg_names=tg_names,
    )

    grad_pooled_df = grad_attrib_df.copy()
    grad_pooled_df["Score"] = grad_pooled_df["Score_pooled"]
    grad_pooled_df = grad_pooled_df[["Source", "Target", "Score"]]
    logging.info("  - Pooled Score DataFrame:")
    logging.info(f"    - TFs: {len(tf_names)}, TGs: {len(tg_names)}, Edges: {len(grad_pooled_df)}")
    
    grad_per_tf_df = grad_attrib_df.copy()
    grad_per_tf_df["Score"] = grad_per_tf_df["Score_per_tf"]
    grad_per_tf_df = grad_per_tf_df[["Source", "Target", "Score"]]
    logging.info("  - Per-TF Score DataFrame:")
    logging.info(f"    - TFs: {len(tf_names)}, TGs: {len(tg_names)}, Edges: {len(grad_per_tf_df)}")
    
    logging.info("Loading TF Knockout")
    tf_ko_df = load_tf_knockout_scores_with_two_scores(
        selected_experiment_dir=selected_experiment_dir,
        tf_names=tf_names,
        tg_names=tg_names,
        positive_only=True,
        eps=1e-6,
    )
    
    tf_ko_pooled_df = tf_ko_df.copy()
    tf_ko_pooled_df["Score"] = tf_ko_pooled_df["Score_pooled"]
    tf_ko_pooled_df = tf_ko_pooled_df[["Source", "Target", "Score"]]
    logging.info("  - Pooled Score DataFrame:")
    logging.info(f"    - TFs: {len(tf_names)}, TGs: {len(tg_names)}, Edges: {len(tf_ko_pooled_df)}")
    
    tf_ko_per_tf_df = tf_ko_df.copy()
    tf_ko_per_tf_df["Score"] = tf_ko_per_tf_df["Score_per_tf"]
    tf_ko_per_tf_df = tf_ko_per_tf_df[["Source", "Target", "Score"]]
    logging.info("  - Per-TF Score DataFrame:")
    logging.info(f"    - TFs: {len(tf_names)}, TGs: {len(tg_names)}, Edges: {len(tf_ko_per_tf_df)}")
    
    return grad_pooled_df, grad_per_tf_df, tf_ko_pooled_df, tf_ko_per_tf_df

def load_and_standardize_method(name: str, info: dict) -> pd.DataFrame:
    """
    Load a GRN CSV and rename tf_col/target_col/score_col -> Source/Target/Score.
    Extra columns are preserved.
    """
    if info["path"].suffix == ".tsv":
        sep = "\t"
    elif info["path"].suffix == ".csv":
        sep = ","
    
    df = pd.read_csv(info["path"], sep=sep, header=0, index_col=None)

    tf_col     = info["tf_col"]
    target_col = info["target_col"]
    score_col  = info["score_col"]

    rename_map = {
        tf_col: "Source",
        target_col: "Target",
        score_col: "Score",
    }

    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing expected columns: {missing}. Got: {list(df.columns)}")

    df = df.rename(columns=rename_map)

    df = df[["Source", "Target", "Score"]]
    df["Source"] = df["Source"].astype(str).str.upper()
    df["Target"] = df["Target"].astype(str).str.upper()

    return df

def balance_pos_neg(df, label_col="_in_gt", random_state=0):
    rng = np.random.default_rng(random_state)
    df = df.copy()
    pos_df = df[df[label_col] == 1]
    neg_df = df[df[label_col] != 1]

    n_pos = len(pos_df)
    n_neg = len(neg_df)
    if n_pos == 0 or n_neg == 0:
        logging.info("No positives or negatives, skipping balance")
        return df

    if n_neg < n_pos:
        pos_idx = rng.choice(pos_df.index.to_numpy(), size=n_neg, replace=False)
        pos_sample = pos_df.loc[pos_idx]
        neg_sample = neg_df
    else:
        pos_sample = pos_df
        neg_idx = rng.choice(neg_df.index.to_numpy(), size=n_pos, replace=False)
        neg_sample = neg_df.loc[neg_idx]

    balanced = pd.concat([pos_sample, neg_sample], axis=0)
    return balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def per_tf_metrics(
    method_df: pd.DataFrame, 
    gt_edges: pd.DataFrame, 
    top_fracs=(0.001, 0.005, 0.01, 0.05), 
    min_edges=10, min_pos=1,
    balance_for_auprc=False
    ) -> pd.DataFrame:
    """
    Returns a per-TF dataframe with:
      TF, AUROC, n_pos, n_neg, pos_rate, Precision@K, Lift@K (for each K)
    Assumes method_df has Source/Target/Score and is already restricted to GT universe (recommended).
    """
    # Label edges
    d = method_df.merge(gt_edges, on=["Source", "Target"], how="left")
    d["_in_gt"] = d["_in_gt"].fillna(0).astype(int)

    rows = []
    for tf, g in d.groupby("Source", sort=False):
        y = g["_in_gt"].to_numpy()
        s = g["Score"].to_numpy()
        n = len(y)
        n_pos = int(y.sum())
        n_neg = int(n - n_pos)
        pos_rate = (n_pos / n) if n > 0 else np.nan
        
        # basic filters to avoid degenerate metrics
        if n < min_edges:
            continue
        if n_pos < min_pos or n_neg == 0:
            continue

        # AUROC defined only if both classes present
        auc = roc_auc_score(y, s) if (n_pos > 0 and n_neg > 0) else np.nan
        
        if balance_for_auprc:
            balanced = balance_pos_neg(g, label_col="_in_gt", random_state=42)
            y_bal = balanced["_in_gt"].astype(int).to_numpy()
            s_bal = balanced["Score"].to_numpy()
            auprc = average_precision_score(y_bal, s_bal) if (y_bal.sum() > 0 and y_bal.sum() < len(y_bal)) else np.nan
        else:
            auprc = average_precision_score(y, s) if n_pos > 0 else np.nan
        
        # Pre-sort once for precision@K
        order = np.argsort(s)[::-1]
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)

        row = {
            "tf": tf,
            "n_edges": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_rate": pos_rate,
            "auroc": float(auc) if auc == auc else np.nan,
            "auprc": float(auprc) if auprc == auprc else np.nan,
        }

        for frac in top_fracs:
            K = max(1, int(frac * n))
            K = min(K, n)
            prec_k = float(tp[K-1] / K) if n > 0 else np.nan
            row[f"precision@{frac*100:.2f}%"] = prec_k
            row[f"lift@{frac*100:.2f}%"] = (prec_k / pos_rate) if (pos_rate and pos_rate > 0) else np.nan

        rows.append(row)

    return pd.DataFrame(rows)

def _select_df(method_name, method_obj, per_tf_methods):
    if isinstance(method_obj, dict):
        if method_name in per_tf_methods and "per_tf" in method_obj:
            return method_obj["per_tf"]
        if "pooled" in method_obj:
            return method_obj["pooled"]
    return method_obj

def calculate_pooled_auroc(standardized_method_dict, ground_truth_edges_dict, per_tf_methods=None):
    per_tf_methods = per_tf_methods or set()
    all_results = []
    for method_name, method_obj in standardized_method_dict.items():
        method_df = _select_df(method_name, method_obj, per_tf_methods)
        logging.info(f"  - Evaluating {method_name}")
        for gt_name, gt_edges in ground_truth_edges_dict.items():
            d_eval = restrict_to_gt_universe(method_df, gt_edges)
            if len(d_eval) == 0:
                logging.info(f"  - {gt_name}: no overlap, skipping")
                continue
            metrics, raw_results_df = eval_method_vs_gt(d_eval, gt_edges)
            all_results.append({"method": method_name, "gt": gt_name, **metrics})

    results_df = pd.DataFrame(all_results)
    
    return results_df, raw_results_df

def calculate_per_tf_auroc(standardized_method_dict, ground_truth_edges_dict, top_k_fracs, per_tf_methods=None):
    per_tf_methods = per_tf_methods or set()
    per_tf_all, per_tf_summary = [], []
    for method_name, method_obj in standardized_method_dict.items():
        method_df = _select_df(method_name, method_obj, per_tf_methods)
        logging.info(f"  - Per-TF evaluating {method_name}")
        for gt_name, gt_edges in ground_truth_edges_dict.items():
            d_eval = restrict_to_gt_universe(method_df, gt_edges)
            if len(d_eval) == 0:
                continue

            tf_df = per_tf_metrics(
                d_eval, 
                gt_edges, 
                top_fracs=top_k_fracs, 
                min_edges=50, 
                min_pos=10,
                balance_for_auprc=True
                )
            
            # Skip if no TFs passed the filtering criteria
            if len(tf_df) == 0 or "auroc" not in tf_df.columns:
                logging.info(f"    No TFs passed filtering criteria for {gt_name}")
                continue
                
            tf_df.insert(0, "gt", gt_name)
            tf_df.insert(0, "method", method_name)
            per_tf_all.append(tf_df)

            defined = tf_df.dropna(subset=["auroc"])
            frac_defined = len(defined) / len(tf_df) if len(tf_df) else np.nan

            row = {
                "method": method_name,
                "gt": gt_name,
                "n_tf_total": int(len(tf_df)),
                "n_tf_auroc_defined": int(len(defined)),
                "frac_tf_auroc_defined": float(frac_defined) if frac_defined == frac_defined else np.nan,
                "mean_per_tf_auroc": float(defined["auroc"].mean()) if len(defined) else np.nan,
                "median_per_tf_auroc": float(defined["auroc"].median()) if len(defined) else np.nan,
            }

            for frac in top_k_fracs:
                lift_col = f"lift@{frac*100:.2f}%"
                lift_vals = tf_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[lift_col])[lift_col]
                row[f"median_{lift_col}"] = float(lift_vals.median()) if len(lift_vals) else np.nan
                row[f"mean_{lift_col}"] = float(lift_vals.mean()) if len(lift_vals) else np.nan

            per_tf_summary.append(row)

    per_tf_all_df = pd.concat(per_tf_all, ignore_index=True) if per_tf_all else pd.DataFrame()
    per_tf_summary_df = pd.DataFrame(per_tf_summary)
    
    return per_tf_all_df, per_tf_summary_df

if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser(description="Run AUROC and AUPR testing for trained models")

    arg_parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment to test")
    arg_parser.add_argument("--training_num", type=str, required=False, default="model_training_001", help="Training number folder to test")
    arg_parser.add_argument("--experiment_dir", type=Path, required=True, help="Full path to the experiment directory to test")
    arg_parser.add_argument("--dataset_type", type=str, required=True, choices=["mESC", "macrophage", "k562"], help="Type of dataset: mESC, macrophage, or k562")
    arg_parser.add_argument("--sample_name_list", type=str, nargs='+', required=False, default=[], help="List of sample names to include in the evaluation (optional)")
    arg_parser.add_argument("--top_k_fracs", type=float, nargs='+', required=False, default=[0.001, 0.005, 0.01, 0.05], help="List of top K fractions for precision@K evaluation")

    args = arg_parser.parse_args()

    experiment = args.experiment
    training_num = args.training_num if args.training_num else "model_training_001" 
    experiment_dir = Path(args.experiment_dir)
    dataset_type = args.dataset_type
    top_k_fracs = tuple(args.top_k_fracs)
    
    sample_name_list = args.sample_name_list
    
    logging.info(f"Experiment: {experiment}, Training Num: {training_num}, Dataset Type: {dataset_type}")
    
    if "chr19" in [p.name for p in Path(experiment_dir / experiment).iterdir()] and experiment != "mESC_no_scale_linear":
        EXPERIMENT_DIR = experiment_dir / experiment / "chr19" / training_num
    else:
        EXPERIMENT_DIR = experiment_dir / experiment / training_num

    if EXPERIMENT_DIR is None:
        EXPERIMENT_DIR = Path(f"/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/{experiment}/chr19") / training_num

    
    # mESC ground truth files
    if "mesc" == dataset_type.lower():
        ground_truth_file_dict = {
            "ChIP-Atlas": GROUND_TRUTH_DIR / "chip_atlas_tf_peak_tg_dist.csv",
            "RN111": GROUND_TRUTH_DIR / "RN111.tsv",
            "RN112": GROUND_TRUTH_DIR / "RN112.tsv",
            "RN114": GROUND_TRUTH_DIR / "RN114.tsv",
            "RN116": GROUND_TRUTH_DIR / "RN116.tsv",
        }
    
    elif dataset_type.lower() == "macrophage":
        ground_truth_file_dict = {
            "ChIP-Atlas": GROUND_TRUTH_DIR / "chipatlas_macrophage.csv",
            "RN204": GROUND_TRUTH_DIR / "rn204_macrophage_human_chipseq.tsv",
        }
        
    elif "k562" == dataset_type.lower():
        ground_truth_file_dict = {
            "ChIP-Atlas": GROUND_TRUTH_DIR / "chipatlas_K562.csv",
            "RN117": GROUND_TRUTH_DIR / "RN117.tsv",
            # "RN118": GROUND_TRUTH_DIR / "RN118.tsv",
            # "RN119": GROUND_TRUTH_DIR / "RN119.tsv",
        }
        
    FIG_DIR = Path("/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/FIGURES")
    FIG_DATA = Path("/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/FIGURE_DATA")

    exp_fig_dir = FIG_DIR / experiment / training_num
    exp_fig_data_dir = FIG_DATA / experiment / training_num

    if not os.path.exists(exp_fig_data_dir):
        os.makedirs(exp_fig_data_dir)
        
    if not os.path.exists(exp_fig_dir):
        os.makedirs(exp_fig_dir)

    # Loop through each ground truth dataset and load each file
    ground_truth_df_dict = {}
    for i, (gt_name, ground_truth_file) in enumerate(ground_truth_file_dict.items(), start=1):
        logging.info(f"Loading {gt_name} ({i}/{len(ground_truth_file_dict)})")

        # --- Ground truth & sets ---
        ground_truth_df = load_ground_truth(ground_truth_file)
        
        ground_truth_df_dict[gt_name] = ground_truth_df
        logging.info(f"  - TFs: {ground_truth_df['Source'].nunique():,}, TGs: {ground_truth_df['Target'].nunique():,}, Edges: {len(ground_truth_df):,}")

    ground_truth_edges_dict = {gt: prep_gt_edges(df) for gt, df in ground_truth_df_dict.items()}
    
    grad_pooled_df, grad_per_tf_df, tf_ko_pooled_df, tf_ko_per_tf_df = load_grad_and_tf_ko_df(
        selected_experiment_dir=EXPERIMENT_DIR,
    )
    
    grad_pooled_df.to_csv(EXPERIMENT_DIR / "gradient_attribution_pooled_scores.csv", index=False)
    grad_per_tf_df.to_csv(EXPERIMENT_DIR / "gradient_attribution_per_tf_scores.csv", index=False)
    tf_ko_pooled_df.to_csv(EXPERIMENT_DIR / "tf_knockout_pooled_scores.csv", index=False)
    tf_ko_per_tf_df.to_csv(EXPERIMENT_DIR / "tf_knockout_per_tf_scores.csv", index=False)
    
    # Load the other method GRNs (TF-TG scores averaged across samples)
    standardized_method_dict = load_other_method_grns(sample_name_list, dataset_type)

    # Add the two transformer-based methods to the dictionary of method score dataframes
    standardized_method_dict["Gradient Attribution"] = {"pooled": grad_pooled_df, "per_tf": grad_per_tf_df}

    standardized_method_dict["TF Knockout"] = {"pooled": tf_ko_pooled_df, "per_tf": tf_ko_per_tf_df}

    
    # Pooled AUROC/AUPRC
    logging.info("\nEvaluating pooled methods across samples")
    results_df, raw_results_df = calculate_pooled_auroc(
        standardized_method_dict, ground_truth_edges_dict, 
        per_tf_methods={"Gradient Attribution", "TF Knockout"}
        )
    
    logging.info(results_df.groupby("method")["auroc"].mean().sort_values(ascending=False))

        
    # Per-TF AUROC/AUPRC
    logging.info("\nPer-TF evaluation of pooled methods across samples")
    per_tf_all_df, per_tf_summary_df = calculate_per_tf_auroc(
        standardized_method_dict, ground_truth_edges_dict, top_k_fracs, 
        per_tf_methods={"Gradient Attribution", "TF Knockout"}
        )    
    
    logging.info(per_tf_summary_df.groupby("method")["mean_per_tf_auroc"].mean().sort_values(ascending=False))
    
    # Save results
    raw_results_df.to_csv(EXPERIMENT_DIR / "pooled_auroc_auprc_raw_results.csv", index=False)
    results_df.to_csv(EXPERIMENT_DIR / "pooled_auroc_auprc_results.csv", index=False)
    per_tf_all_df.to_csv(EXPERIMENT_DIR / "per_tf_auroc_auprc_results.csv", index=False)
    per_tf_summary_df.to_csv(EXPERIMENT_DIR / "per_tf_auroc_auprc_summary.csv", index=False)
    
    
    
    
        
    
    
    
    
