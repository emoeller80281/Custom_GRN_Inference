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
import concurrent.futures

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
        "--cell_level_net_dir",
        type=str,
        required=True,
        help="Path to the directory with the cell-level inferred network score pickle files"
    )
    parser.add_argument(
        "--num_cpu",
        type=str,
        required=True,
        help="Number of processors to run multithreading with"
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

def process_cell_network(file_path, rf, output_dir):
    # Extract cell name from file_path
    cell_name = file_path.split("/")[-1].split(".pkl")[0]
    logging.info(f'Processing cell {cell_name}')

    # Read the inferred network dataframe
    inferred_network = read_inferred_network(file_path)
    
    # Select features for predictions
    X = inferred_network[rf.feature_names]
    
    # Make predictions using the random forest model
    inferred_network["Score"] = rf.predict_proba(X)[:, 1]
    
    # Keep only relevant columns and drop duplicates
    inferred_network = inferred_network[["Source", "Target", "Score"]].drop_duplicates()
    
    # Create the output directory for this cell if it doesn't exist
    cell_output_dir = os.path.join(output_dir, "cell_networks_rf", cell_name)
    if not os.path.exists(cell_output_dir):
        os.makedirs(cell_output_dir)
    
    # Save the inferred network as a TSV file
    output_file = os.path.join(cell_output_dir, "rf_inferred_grn.tsv")
    inferred_network.to_csv(output_file, sep='\t', index=False)
    return cell_name

def infer_cell_level(cell_level_net_dir, rf, output_dir, num_cpu):
    # List all pkl files that match the expected pattern
    cell_network_files = [
        os.path.join(cell_level_net_dir, file)
        for file in os.listdir(cell_level_net_dir)
        if ".pkl" in file
    ]
    
    total_files = len(cell_network_files)
    logging.info(f'Found {total_files} cell network files to process.')
    
    # Use ProcessPoolExecutor (or ThreadPoolExecutor if rf is not picklable)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpu) as executor:
        # Submit each file to be processed in parallel
        futures = {
            executor.submit(process_cell_network, file_path, rf, output_dir): file_path
            for file_path in cell_network_files
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                cell_name = future.result()
                logging.info(f'Completed processing for cell {cell_name} ({i+1}/{total_files})')
            except Exception as exc:
                logging.error(f'File {futures[future]} generated an exception: {exc}')
        

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()

    output_dir: str = args.output_dir
    model: str = args.model
    cell_level_net_dir: str = args.cell_level_net_dir
    num_cpu: int = int(args.num_cpu)
    
    # output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC/filtered_L2_E7.5_rep1"
    # model = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC/filtered_L2_E7.5_rep1/trained_random_forest_model.pkl"
    # cell_level_net_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC/filtered_L2_E7.5_rep1/cell_networks_raw"
    # num_cpu=4
    
    logging.info("Loading trained Random Forest model")
    rf = joblib.load(model)
    
    infer_cell_level(cell_level_net_dir, rf, output_dir, num_cpu)
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()