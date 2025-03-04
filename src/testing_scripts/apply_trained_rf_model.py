import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import random
import csv
import os
import joblib
import argparse
import logging

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
        help="Name of the sample to apply the model to"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of the sample to apply the model to"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        required=True,
        help="Name of the sample to apply the model to"
    )
    
    args: argparse.Namespace = parser.parse_args()

    return args

def read_inferred_network(inferred_network_file):
    inferred_network = pd.read_pickle(inferred_network_file)
    # logging.info(inferred_network.head())
    inferred_network["Source"] = inferred_network["Source"].str.upper()
    inferred_network["Target"] = inferred_network["Target"].str.upper()
    
    return inferred_network

def read_ground_truth(ground_truth_file):
    ground_truth = pd.read_csv(ground_truth_file, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    
    return ground_truth

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()

    output_dir: str = args.output_dir
    model: str = args.model
    target: str = args.target
    save_name: str = args.save_name
    
    logging.info(f'Running random forest predictions for {save_name}')
    
    logging.info("Reading inferred network")
    inferred_network = read_inferred_network(target)
    logging.info("    Done!")
    
    logging.info("Loading trained Random Forest model")
    rf = joblib.load(model)
    # logging.info(f'Feature Names')
    # for feature in rf.feature_names:
    #     logging.info(f"\t{feature}")
    X = inferred_network[rf.feature_names]
    logging.info("    Done!")
    
    logging.info("Making predictions")
    inferred_network["Score"] = rf.predict_proba(X)[:, 1]
    inferred_network = inferred_network[["Source", "Target", "Score"]]
    # logging.info(inferred_network.head())
    
    inferred_network = inferred_network.drop_duplicates()
    # logging.info(inferred_network.head())
    logging.info(f'Num unique TFs: {len(inferred_network["Source"].unique())}')
    logging.info(f'Num unique TGs: {len(inferred_network["Target"].unique())}')
    
    logging.info("    Done!")

    logging.info("Saving inferred GRN")
    inferred_network.to_csv(f'{output_dir}/{save_name}_rf_inferred_grn.tsv', sep='\t', index=False)
    logging.info("    Done!\n")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()