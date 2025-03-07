import os
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
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
        "--ground_truth_file",
        type=str,
        required=True,
        help="Path to the ground truth file for training the random forest"
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        required=True,
        help="Path to the figure directory for the sample"
    )
    parser.add_argument(
        "--cell_rf_net_dir",
        type=str,
        required=True,
        help="Path to the directory with the cell-level inferred network score random forest prediction models"
    )
    parser.add_argument(
        "--num_cpu",
        type=str,
        required=True,
        help="Number of processors to run multithreading with"
    )
    
    args: argparse.Namespace = parser.parse_args()

    return args

def load_inferred_network(inferred_network_file):
    inferred_network: pd.DataFrame = pd.read_pickle(inferred_network_file)
    inferred_network["Source"] = inferred_network["Source"].str.upper()
    inferred_network["Target"] = inferred_network["Target"].str.upper()
    
    return inferred_network

def load_ground_truth(ground_truth_file):
    ground_truth: pd.DataFrame = pd.read_csv(ground_truth_file, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    ground_truth["Source"] = ground_truth["Source"].str.upper()
    ground_truth["Target"] = ground_truth["Target"].str.upper()
    
    return ground_truth

def read_cell_level_rf_inferred_grns(cell_rf_net_dir):
    cell_dfs = []
    cell_names = []  # To record which cell each dataframe belongs to

    # Read each cell's inferred GRN TSV file and add a column for cell name
    for cell_dir in os.listdir(cell_rf_net_dir):
        cell_net = os.path.join(cell_rf_net_dir, cell_dir, "rf_inferred_grn.tsv")
        if os.path.exists(cell_net):
            # Extract cell name from directory name
            cell_name = cell_dir
            df = pd.read_csv(cell_net, sep="\t", header=0)
            df["cell"] = cell_name  # add a column for the cell
            cell_dfs.append(df)
            cell_names.append(cell_name)

    return cell_dfs

def plot_cell_consensus_fraction_hist(consensus_df):
    # Visualize the distribution of consensus fractions
    plt.hist(consensus_df["consensus_fraction"], bins=50, color="skyblue", edgecolor="k")
    plt.xlabel("Consensus Fraction (Fraction of cells with high Score)")
    plt.ylabel("Number of TF-TG pairs")
    plt.title("Distribution of Consensus Scores Across Cells")
    plt.savefig("consensus_fraction.png")
    plt.close()

def plot_feature_importance(features: list, rf: RandomForestClassifier, fig_dir: str):
    # Step 4: Feature Importance Analysis
    feature_importances = pd.DataFrame({
        "Feature": features,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(8, 6))
    plt.barh(feature_importances["Feature"], feature_importances["Importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/random_forest_feature_importance.png", dpi=200)
    plt.close()


def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()

    output_dir: str = args.output_dir
    ground_truth_file: str = args.ground_truth_file
    cell_rf_net_dir: str = args.cell_rf_net_dir
    fig_dir: str = args.fig_dir
    num_cpu: int = int(args.num_cpu)

    # output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC/filtered_L2_E7.5_rep1"
    # ground_truth_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/ground_truth_files/RN111.tsv"
    # cell_rf_net_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC/filtered_L2_E7.5_rep1/cell_networks_rf"
    # fig_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/figures/mm10/filtered_L2_E7.5_rep1"
    # num_cpu = 4

    inferred_network_file: str = f"{output_dir}/inferred_network_raw.pkl"

    # ----- Loading and Preparing Data -----
    logging.info('Loading in required files:')
    
    # Load in the random forest prediction dataframes for the individual cell inferred GRNs
    logging.info(f'\tLoading in cell-level random forest GRNs')
    cell_dfs = read_cell_level_rf_inferred_grns(cell_rf_net_dir)

    # Read in the bulk network with mean TF and mean TG scores
    logging.info(f'\tLoading the averaged inferred network')
    inferred_network = load_inferred_network(inferred_network_file)

    # Load in the same ground truth used to create the original random forest model
    logging.info(f'\tLoading ground truth')
    ground_truth = load_ground_truth(ground_truth_file)

    # Get each a set of edges
    ground_truth_pairs = set(zip(ground_truth["Source"], ground_truth["Target"]))

    # Concatenate dataframes from all cells
    merged_cell_grns = pd.concat(cell_dfs, ignore_index=True)
    logging.info(f'Done!\n')

    # ----- Creating a Consensus of Edges From the Cell-Level GRNs -----
    logging.info(f'Computing consensus of edge scores from cell-level GRNs')
    # Define a threshold for a "high-confidence" interaction. For example:
    threshold = 0.5

    # Create a binary column for high confidence
    merged_cell_grns["is_high"] = merged_cell_grns["Score"] > threshold

    # Group by TF-TG pair and compute:
    # 1. The average Score across cells.
    # 2. The fraction of cells where the interaction is "high confidence".
    consensus_df = (
        merged_cell_grns
        .groupby(["Source", "Target"], as_index=False)
        .agg(avg_score=("Score", "mean"),
            consensus_fraction=("is_high", "mean"),
            cell_count=("cell", "nunique"))
    )

    plot_cell_consensus_fraction_hist(consensus_df)
    logging.info(f'Done!\n')

    # ----- Creating Pseudo-labels of High and Low Confidence Edges -----
    logging.info(f'Creating pseudo-labels for the edges based on consensus scores')
    # For example, to use as pseudo-labels:
    positive_threshold = 0.7  # consensus_fraction >= 0.7 as pseudo-positive
    negative_threshold = 0.3  # consensus_fraction <= 0.3 as pseudo-negative

    def label_pseudo(row):
        if row["consensus_fraction"] >= positive_threshold:
            return 1
        elif row["consensus_fraction"] <= negative_threshold:
            return 0
        else:
            return np.nan

    consensus_df["pseudo_label"] = consensus_df.apply(label_pseudo, axis=1)
    pseudo_df = consensus_df.dropna(subset=["pseudo_label"])

    inferred_network["Label"] = inferred_network.apply(
        lambda row: 1 if (row["Source"], row["Target"]) in ground_truth_pairs else 0,
        axis=1
    )
    logging.info(f'Done!\n')

    # ----- Merging the Cell-Level Consensus with the Bulk Model -----
    logging.info(f'Merging the cell-level consensus results with the averaged inferred network')
    # Merge the cell-level GRN metrics with the bulk inferred network
    refined_df = pd.merge(
        inferred_network,
        consensus_df[["Source", "Target", "consensus_fraction", "avg_score"]], 
        on=["Source", "Target"],
        how="left"
    )
    logging.debug("Refined df")
    logging.debug(refined_df.head())

    # Fill missing consensus scores with 0 (or some default) if not found in cell-level predictions
    refined_df["consensus_fraction"] = refined_df["consensus_fraction"].fillna(0)
    refined_df["avg_score"] = refined_df["avg_score"].fillna(0)
    logging.info(f'Done!\n')

    # ----- Training a New Random Forest with Cell-Level Information -----
    logging.info('Re-training the random forest model with single-cell random forest score consensus')
    aggregated_features_new = [
        "TF_expression",
        "tf_to_peak_binding_score",
        "atac_expression",
        "peak_to_target_score",
        "TG_expression",
    ]

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 5],
    }
    
    logging.info('\tSetting up the random forest with the new features')
    X = refined_df[aggregated_features_new]
    y = refined_df["Label"]

    # 7) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_random_forest(X_train, y_train, features, num_cpu):
        logging.info("\tTraining the random forest")
        # Combine training features and labels for resampling
        train_data = X_train.copy()
        train_data["Label"] = y_train

        # Separate positive and negative examples
        pos_train = train_data[train_data["Label"] == 1]
        neg_train = train_data[train_data["Label"] == 0]

        # Balance the dataset between positive and negative label values
        neg_train_sampled = neg_train.sample(n=len(pos_train), random_state=42)
        train_data_balanced = pd.concat([pos_train, neg_train_sampled])

        X_train_balanced = train_data_balanced[features]
        y_train_balanced = train_data_balanced["Label"]

        # Train the Random Forest Classifier
        rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
        
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=num_cpu, verbose=3)
        grid_search.fit(X_train, y_train)
        logging.info("\t\tBest parameters: %s", grid_search.best_params_)
        logging.info("\t\tBest score: %s", grid_search.best_score_)

        # Use the best estimator
        rf_best = grid_search.best_estimator_
        
        # Fit the random forest with the updated 
        rf_best.fit(X_train_balanced, y_train_balanced)

        return rf_best

    # 8) Train random forest
    rf = train_random_forest(X_train, y_train, aggregated_features_new, num_cpu)
    logging.info(f'Done!\n')
    
    logging.info(f'Saving the refined random forest model')
    # 9) Plot, save model, etc. as before
    plot_feature_importance(aggregated_features_new, rf, fig_dir)

    # Save the feature names of the trained model
    rf.feature_names = list(X_train.columns.values)

    joblib.dump(rf, f"{output_dir}/refined_trained_random_forest_model.pkl")
    logging.info(f'Done!\n')

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()
