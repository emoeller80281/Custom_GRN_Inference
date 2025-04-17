#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import csv
import joblib
import logging
import argparse
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import average_precision_score, roc_auc_score

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom XGBoost parameter search and evaluation.")
    parser.add_argument("--ground_truth_file", type=str, required=True,
                        help="Path to the ChIPseq ground truth file (source_id, target_id)")
    parser.add_argument("--inferred_network_file", type=str, required=True,
                        help="Path to the inferred network CSV file")
    parser.add_argument("--fig_dir", type=str, required=True,
                        help="Directory to save figures/results")
    parser.add_argument("--cpu_count", type=int, required=True,
                        help="Number of CPU cores for parallel search")
    return parser.parse_args()


def read_inferred_network(path: str) -> pd.DataFrame:
    logging.info(f"Reading inferred network from {path}")
    df = pd.read_csv(path)
    df["source_id"] = df["source_id"].str.upper()
    df["target_id"] = df["target_id"].str.upper()
    logging.info(f"Loaded {len(df)} rows")
    return df


def read_ground_truth(path: str) -> pd.DataFrame:
    logging.info(f"Reading ground truth from {path}")
    gt = pd.read_csv(path, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    gt = gt.rename(columns={"Source": "source_id", "Target": "target_id"})
    logging.info(f"Loaded {len(gt)} ground-truth pairs")
    return gt


def setup_labels(df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    logging.info("Assigning labels based on ground truth membership...")
    gt_pairs = set(zip(gt.source_id, gt.target_id))
    df['label'] = df.apply(lambda r: 1 if (r.source_id, r.target_id) in gt_pairs else 0, axis=1)
    n_pos = df['label'].sum(); n_neg = len(df) - n_pos
    logging.info(f"Labels assigned: {n_pos} positives, {n_neg} negatives")
    return df


def balance_data(df: pd.DataFrame, features: list):
    logging.info("Balancing dataset: downsampling negatives to match positives...")
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    neg_down = neg.sample(n=len(pos), random_state=42)
    balanced = pd.concat([pos, neg_down]).reset_index(drop=True)
    logging.info(f"Balanced set size: {len(balanced)} rows")
    X = balanced[features]; y = balanced['label']
    return X, y


def parameter_grid_search(X: pd.DataFrame, y: pd.Series, features: list, fig_dir: str):
    logging.info("Starting manual parameter search with flatness metrics...")

    # train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # parameter grid
    param_grid = {
        'n_estimators':      [50, 100, 200],
        'max_depth':         [4, 6, 8],
        'gamma':             [0, 1, 5],
        'reg_alpha':         [0.0, 0.5, 1.0],
        'reg_lambda':        [1.0, 2.0, 5.0],
        'subsample':         [0.8, 1.0],
        'colsample_bytree':  [0.8, 1.0],
        'learning_rate':     [0.01, 0.1]
    }
    logging.info(f"Parameter grid: {param_grid}")

    results = []
    for params in ParameterGrid(param_grid):
        logging.info(f"Training with params: {params}")
        model = xgb.XGBClassifier(
            **params,
            n_jobs=1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )
        # performance
        y_pred = model.predict_proba(X_val)[:,1]
        val_ap = average_precision_score(y_val, y_pred)
        val_auc = roc_auc_score(y_val, y_pred)
        # flatness metrics
        imp = model.feature_importances_
        p = imp / np.sum(imp + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))
        cv = np.std(p) / (np.mean(p) + 1e-12)

        results.append({**params,
                        'val_ap': val_ap,
                        'val_auc': val_auc,
                        'imp_entropy': entropy,
                        'imp_cv': cv})

    df_res = pd.DataFrame(results)
    out_dir = os.path.join(fig_dir, 'parameter_search')
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, 'grid_search_results.csv')
    df_res.to_csv(csv_out, index=False)
    logging.info(f"Saved parameter search results to {csv_out}")

    # identify best
    best_ap = df_res.loc[df_res.val_ap.idxmax()]
    best_flat = df_res.loc[df_res.imp_entropy.idxmax()]
    logging.info(f"Best AP={best_ap.val_ap:.4f} at {best_ap.to_dict()}")
    logging.info(f"Highest entropy={best_flat.imp_entropy:.4f} at {best_flat.to_dict()}")

    # retrain best by AP on full X/y
    best_params = {k: best_ap[k] for k in param_grid.keys()}
    best_model = xgb.XGBClassifier(**best_params, random_state=42, use_label_encoder=False)
    best_model.fit(X, y)
    joblib.dump(best_model, os.path.join(out_dir, 'best_model_by_ap.pkl'))
    logging.info(f"Saved best model by AP to {out_dir}/best_model_by_ap.pkl")

    # scatter plot AP vs entropy
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.scatter(df_res.imp_entropy, df_res.val_ap, alpha=0.7)
    plt.xlabel('Importance Entropy')
    plt.ylabel('Validation Average Precision')
    plt.title('Flatness vs Performance')
    plt.tight_layout()
    plot_out = os.path.join(out_dir, 'flatness_vs_ap.png')
    plt.savefig(plot_out, dpi=200)
    plt.close()
    logging.info(f"Saved flatness vs AP plot to {plot_out}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    logging.info("=== Parameter Search Started ===")

    df = read_inferred_network(args.inferred_network_file)
    gt = read_ground_truth(args.ground_truth_file)
    df = setup_labels(df, gt)

    drop = ['source_id','target_id','label']
    feats = [c for c in df.columns if c not in drop]
    logging.info(f"Features for modeling: {feats}")

    X_bal, y_bal = balance_data(df, feats)
    parameter_grid_search(X_bal, y_bal, feats, args.fig_dir)

    logging.info("=== Parameter Search Completed ===")

if __name__ == '__main__':
    main()
