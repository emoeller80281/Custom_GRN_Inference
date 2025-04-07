import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import random
import csv
import os
import joblib
import argparse
import logging
import xgboost as xgb  # Import XGBoost

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing paths for input and output files.
    """
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained XGBoost model file"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to the inferred network pickle file to apply the model to"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        required=True,
        help="Name used to save the predicted network"
    )
    
    args: argparse.Namespace = parser.parse_args()
    return args

def read_inferred_network(inferred_network_file):
    inferred_network = pd.read_csv(inferred_network_file, header=0)
    inferred_network["source_id"] = inferred_network["source_id"].str.upper()
    inferred_network["target_id"] = inferred_network["target_id"].str.upper()
    return inferred_network

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()

    output_dir: str = args.output_dir
    model_path: str = args.model
    target: str = args.target
    save_name: str = args.save_name
    
    logging.info(f'Running XGBoost predictions for {save_name}')
    
    logging.info("Reading inferred network")
    inferred_network = read_inferred_network(target)
    logging.info("    Done!")
    
    logging.info("Loading trained XGBoost model")
    xgb_model = joblib.load(model_path)
    # Use the feature names stored in the model to select the correct columns
    X = inferred_network[xgb_model.feature_names]
    logging.info("    Done!")
    
    logging.info("Making predictions")
    inferred_network["score"] = xgb_model.predict_proba(X)[:, 1]
    inferred_network = inferred_network[["source_id", "target_id", "score"]]
    
    inferred_network = inferred_network.drop_duplicates()
    logging.info(f'Num unique TFs: {len(inferred_network["source_id"].unique())}')
    logging.info(f'Num unique TGs: {len(inferred_network["target_id"].unique())}')
    logging.info("    Done!")
    
    logging.info("Saving inferred GRN")
    output_file = os.path.join(output_dir, f'{save_name}_xgb_inferred_grn.tsv')
    inferred_network.to_csv(output_file, sep='\t', index=False)
    logging.info("    Done!\n")
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
