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
        "--sample_name",
        type=str,
        required=True,
        help="Name of the sample to apply the model to"
    )
    
    args: argparse.Namespace = parser.parse_args()

    return args

def read_inferred_network(inferred_network_file):
    inferred_network = pd.read_csv(inferred_network_file, sep="\t")
    print(inferred_network.head())
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
    sample_name: str = args.dataset_name
    
    inferred_network_file = f"{output_dir}/inferred_network_raw.tsv"

    inferred_network = read_inferred_network(inferred_network_file)
    
    rf = joblib.load(f"{output_dir}/trained_random_forest_model.pkl")
    X = inferred_network[rf.feature_names]
    
    inferred_network["Score"] = rf.predict_proba(X)[:, 1]
    inferred_network = inferred_network[["Source", "Target", "Score"]]

    inferred_network.to_csv(f'{output_dir}/{sample_name}_rf_inferred_grn.tsv', sep='\t', index=False)
    logging.info(inferred_network.head())
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()