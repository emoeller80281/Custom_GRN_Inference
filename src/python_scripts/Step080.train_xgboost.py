import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import csv
import os
import joblib
import logging
import argparse
import xgboost as xgb  # Import XGBoost

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing paths for input and output files.
    """
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")

    parser.add_argument(
        "--ground_truth_file",
        type=str,
        required=True,
        help="Path to the ChIPseq ground truth file, formatted as 'source_id'\\t'target_id'"
    )
    parser.add_argument(
        "--inferred_network_file",
        type=str,
        required=True,
        help="Path to the inferred network file for the sample"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        required=True,
        help="Path to the figure directory for the sample"
    )
    parser.add_argument(
        "--model_save_name",
        type=str,
        required=True,
        help="Name of the output .pkl file for the trained model"
    )
    
    args: argparse.Namespace = parser.parse_args()
    return args

def read_inferred_network(inferred_network_file):
    logging.info("Reading in inferred network")
    inferred_network = pd.read_csv(inferred_network_file, header=0)
    
    inferred_network["source_id"] = inferred_network["source_id"].str.upper()
    inferred_network["target_id"] = inferred_network["target_id"].str.upper()
    return inferred_network

def read_ground_truth(ground_truth_file):
    logging.info("Reading in the ground truth")
    ground_truth = pd.read_csv(ground_truth_file, sep='\t', quoting=csv.QUOTE_NONE, on_bad_lines='skip', header=0)
    ground_truth = ground_truth.rename(columns={"Source": "source_id", "Target": "target_id"})
    return ground_truth

def read_merged_ground_truth(merged_ground_truth_file):
    merged_ground_truth = pd.read_csv(merged_ground_truth_file, sep='\t', header=0)
    logging.info(merged_ground_truth)
    return merged_ground_truth

def train_xgboost(X_train, y_train, features):
    # Combine training features and labels for resampling
    train_data = X_train.copy()
    train_data["label"] = y_train

    # Separate positive and negative examples
    pos_train = train_data[train_data["label"] == 1]
    neg_train = train_data[train_data["label"] == 0]

    # Balance the dataset between positive and negative label values
    neg_train_sampled = neg_train.sample(n=len(pos_train), random_state=42)
    train_data_balanced = pd.concat([pos_train, neg_train_sampled])

    X_train_balanced = train_data_balanced[features]
    y_train_balanced = train_data_balanced["label"]

    # logging.info(f"Balanced training set: {len(pos_train)} positives and {len(neg_train_sampled)} negatives.")

    # Train the XGBoost Classifier
    # XGBoost automatically handles NaN values (missing=np.nan is the default)
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=10,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_balanced, y_train_balanced)

    return xgb_model

def plot_xgboost_prediction_histogram(model, X_test, fig_dir):
    logging.info("\tPlotting model prediction histogram")
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probability for the positive class

    plt.figure(figsize=(8, 6))
    plt.hist(y_pred_prob, bins=50)
    plt.title("Histogram of XGBoost Prediction Probabilities", fontsize=18)
    plt.xlabel("Prediction Probability", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_prediction_probability_histogram.png", dpi=200)
    plt.close()

def plot_feature_importance(features: list, model, fig_dir: str):
    logging.info("\tPlotting feature importance barplot")
    # Feature Importance Analysis
    feature_importances = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 6))
    plt.barh(feature_importances["Feature"], feature_importances["Importance"], color="skyblue")
    plt.xlabel("Importance", fontsize=16)
    plt.ylabel("Feature", fontsize=16)
    plt.title("Feature Importance", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_feature_importance.png", dpi=200)
    plt.close()
    
def plot_feature_score_histograms(features, inferred_network, fig_dir):
    logging.info("\tPlotting feature score histograms")
    
    num_cols = 4
    num_rows = math.ceil(len(features) / num_cols)
    
    # Dynamically change the height of the figure based on the number of rows
    height = 7 * num_rows
    plt.figure(figsize=(18, height))
    
    for i, feature in enumerate(features, 1):
        plt.subplot(num_rows, num_cols, i)
        plt.hist(inferred_network[feature], bins=50, alpha=0.7, edgecolor='black')
        plt.title(f"{feature}", fontsize=14)
        plt.xlabel(feature, fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/xgboost_feature_score_hist.png', dpi=300)
    plt.close()
    
def plot_feature_boxplots(features, inferred_network, fig_dir):
    logging.info("\tPlotting feature importance boxplots")
    def remove_outliers(series):
        """
        Remove outliers from a pandas Series using the IQR method.
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series >= lower_bound) & (series <= upper_bound)]
    
    n_features = len(features)
    ncols = 3
    nrows = math.ceil(n_features / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()  # Flatten the 2D array of axes
    
    for i, feature in enumerate(features):
        ax = axes[i]
        data_label0 = remove_outliers(inferred_network.loc[inferred_network["label"] == 1, feature])
        data_label1 = remove_outliers(inferred_network.loc[inferred_network["label"] == 0, feature])
        ax.boxplot([data_label0, data_label1], patch_artist=True)
        ax.set_title(feature, fontsize=18)
        ax.set_xticklabels(["True", "False"], fontsize=16)
        ax.set_ylabel("score", fontsize=16)
    
    # Hide any unused subplots if they exist
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/xgboost_feature_boxplots.png', dpi=300)
    plt.close()
    
def plot_permutation_importance_plot(xgb_model, X_test, y_test, fig_dir):
    logging.info("\tPlotting permutation importance plot")
    result = permutation_importance(xgb_model, X_test, y_test, 
                                n_repeats=10, random_state=42, scoring='roc_auc')

    # Extract mean importance and standard deviation for each feature
    importances = result.importances_mean
    std = result.importances_std

    # Get the feature names (assuming X_test is a DataFrame)
    feature_names = X_test.columns

    # Sort the feature importances in ascending order for plotting
    indices = np.argsort(importances)

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(importances)), importances[indices], xerr=std[indices],
            align='center')
    plt.yticks(range(len(importances)), feature_names[indices], fontsize=14)
    plt.xlabel("Decrease in ROC-AUC", fontsize=16)
    plt.xticks(fontsize=14)
    plt.title("Permutation Feature Importance", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_permutation_importance.png", dpi=300)
    plt.close()

def plot_stability_boxplot(X, y, fig_dir):
    logging.info("\tPlotting stability boxplot")
    n_runs = 20
    auroc_scores = []

    for i in range(n_runs):
        # Use different random seeds for splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        model = xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=10, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_pred_prob)
        auroc_scores.append(auroc)

    # Plotting the AUROC distribution
    plt.figure(figsize=(8, 6))
    plt.boxplot(auroc_scores, patch_artist=True, boxprops=dict(facecolor='blue'))
    plt.ylabel('AUROC', fontsize=16)
    plt.title('Stability Analysis of AUROC over {} runs'.format(n_runs), fontsize=18)
    plt.ylim((0, 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_stability_boxplot.png", dpi=300)
    plt.close()
    
def plot_overlapping_roc_pr_curves(X, y, feature_names, fig_dir):
    """
    Plots overlapping ROC and Precision-Recall curves for multiple runs.
    
    """
    logging.info("\tPlotting stability AUROC and AUPRC curves")
    # --- Generate y_true_list and y_score_list over multiple runs ---
    n_runs = 10  # Number of runs for stability analysis; adjust as needed

    y_true_list = []
    y_score_list = []

    for i in range(n_runs):
        # Split the data with a different random seed each run
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.7, random_state=i)
        
        # Train your XGBoost model; ensure your train_xgboost function is defined
        xgb_model = train_xgboost(X_train, y_train, feature_names)
        # Save feature names if needed for prediction
        xgb_model.feature_names = list(X_train.columns.values)
        
        # Predict probabilities on the test set (for the positive class)
        y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
        
        # Store the true labels and predicted probabilities
        y_true_list.append(y_test.to_numpy())
        y_score_list.append(y_pred_prob)
    
    labels = [f"Run {i+1}" for i in range(len(y_true_list))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- ROC Curves ---
    for i, (y_true, y_score) in enumerate(zip(y_true_list, y_score_list)):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, lw=1, alpha=0.8, label=f"{labels[i]} (AUC={roc_auc:.2f})")

    # Diagonal line for random guessing
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.0])
    axes[0].set_xlabel("False Positive Rate", fontsize=14)
    axes[0].set_ylabel("True Positive Rate", fontsize=14)
    axes[0].set_title(f"ROC Curve", fontsize=16)
    axes[0].legend(loc="lower right")

    # --- Precision-Recall Curves ---
    for i, (y_true, y_score) in enumerate(zip(y_true_list, y_score_list)):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_prec = average_precision_score(y_true, y_score)
        axes[1].plot(recall, precision, lw=1, alpha=0.8, label=f"{labels[i]} (PR={avg_prec:.2f})")

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall", fontsize=14)
    axes[1].set_ylabel("Precision", fontsize=14)
    axes[1].set_title(f"Precision-Recall Curve", fontsize=16)
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_stability_auroc_auprc.png", dpi=300)
    plt.close()

def plot_feature_ablation(feature_names, X_train, X_test, y_train, y_test, full_model, fig_dir):
    logging.info(f'\tPlotting feature ablation for each feature')
    y_pred_prob_full = full_model.predict_proba(X_test)[:, 1]
    full_auroc = roc_auc_score(y_test, y_pred_prob_full)
    logging.info(f"\t\tFull model AUROC: {full_auroc:.2f}")
    
    # Initialize a dictionary to store AUROC for each ablated model
    feature_performance = {}

    # For each feature, remove it, retrain the model, and compute AUROC
    for feature in feature_names:
        # Create a list of features excluding the current one
        features_subset = [f for f in feature_names if f != feature]
        
        # Subset training and test sets
        X_train_subset = X_train[features_subset]
        X_test_subset = X_test[features_subset]
        
        # Train a model on the subset
        model_subset = train_xgboost(X_train_subset, y_train, features_subset)
        model_subset.feature_names = list(X_train_subset.columns.values)
        
        # Evaluate performance on the test set
        y_pred_prob_subset = model_subset.predict_proba(X_test_subset)[:, 1]
        auroc_subset = roc_auc_score(y_test, y_pred_prob_subset)
        feature_performance[feature] = auroc_subset
        logging.info(f"\t\t\tAUROC without {feature}: {auroc_subset:.2f}")

    # Plot the results: AUROC for each ablated model versus the full model
    features = list(feature_performance.keys())
    auroc_values = [feature_performance[f] for f in features]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, auroc_values, color='blue')
    plt.axhline(full_auroc, color='black', linestyle='--')
    plt.xlabel("Removed Feature", fontsize=16)
    plt.ylabel("AUROC", fontsize=16)
    plt.title("Feature Ablation Analysis", fontsize=18)
    plt.ylim((0,1))
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/xgboost_feature_ablation.png", dpi=300)
    plt.close()

def plot_combined_figure(xgb_model, X, y, X_train, X_test, y_train, y_test, 
                           inferred_network, feature_names, fig_dir, full_model):
    """
    Combines all individual analysis plots into a single large image.
    This function creates:
      1. XGBoost Prediction Histogram
      2. Feature Importance Barplot
      3. Permutation Importance Plot
      4. Feature Score Histograms (multi-panel)
      5. Feature Boxplots (multi-panel)
      6. Stability Analysis Boxplot
      7. Overlapping ROC & Precision-Recall Curves (each in its own subplot)
      8. Feature Ablation Analysis Barplot
    """
    logging.info("\tPlotting the combined figure")
    # Create a master figure using GridSpec with adjusted size and spacing.
    # You can further tweak figsize, hspace, and wspace to your liking.
    fig = plt.figure(figsize=(20, 32))
    gs = GridSpec(9, 3, figure=fig, hspace=0.2, wspace=0.2)
    
    # -------------------------------
    # Panel 1: XGBoost Prediction Histogram (gs[0,0])
    logging.info("\t\t1. XGBoost Prediction Histogram")
    ax1 = fig.add_subplot(gs[0, 0])
    y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
    ax1.hist(y_pred_prob, bins=50, color='skyblue', edgecolor='black')
    ax1.set_title("Histogram of XGBoost Prediction Probabilities", fontsize=18)
    ax1.set_xlabel("Prediction Probability", fontsize=16)
    ax1.set_ylabel("Frequency", fontsize=16)
    ax1.tick_params(axis='both', labelsize=14)
    
    # -------------------------------
    # Panel 2: Feature Importance (gs[0,1])
    logging.info("\t\t2. Feature Importance Barplot")
    ax2 = fig.add_subplot(gs[0, 2])
    feature_importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": xgb_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    ax2.barh(feature_importances["Feature"], feature_importances["Importance"], color="skyblue")
    ax2.set_xlabel("Importance", fontsize=16)
    ax2.set_ylabel("Feature", fontsize=16)
    ax2.set_title("Feature Importance", fontsize=18)
    ax2.tick_params(axis='both', labelsize=14)
    ax2.invert_yaxis()
    
    # -------------------------------
    # Panel 4: Feature Score Histograms (multi-panel)
    logging.info("\t\t4. Feature Score Histograms")
    # We embed a 4x3 grid inside one subplot (spanning row 1 entirely)
    gs4 = GridSpecFromSubplotSpec(4, 3, subplot_spec=gs[1:4, :], hspace=0.7, wspace=0.4)
    for i, feature in enumerate(feature_names):
        ax = fig.add_subplot(gs4[i])
        ax.hist(inferred_network[feature].dropna(), bins=50, alpha=0.7, edgecolor='black', color='lightgreen')
        ax.set_title(f"{feature} distribution", fontsize=16)
        ax.set_xlabel(feature, fontsize=16)
        ax.set_ylabel("Frequency", fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
    
    # -------------------------------
    # Panel 5: Feature Boxplots (multi-panel)
    logging.info("\t\t5. Feature Boxplots")
    gs5 = GridSpecFromSubplotSpec(math.ceil(len(feature_names)/3), 3, subplot_spec=gs[4:7, :], 
                                  hspace=0.5, wspace=0.4)
    def remove_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series[(series >= lower_bound) & (series <= upper_bound)]
    
    for i, feature in enumerate(feature_names):
        ax = fig.add_subplot(gs5[i])
        data_label0 = remove_outliers(inferred_network.loc[inferred_network["label"] == 0, feature])
        data_label1 = remove_outliers(inferred_network.loc[inferred_network["label"] == 1, feature])
        ax.boxplot([data_label0, data_label1], patch_artist=True)
        ax.set_title(feature, fontsize=18)
        ax.set_xticklabels(["True", "False"], fontsize=16)
        ax.set_ylabel("Score", fontsize=16)
    
    # -------------------------------
    # Panel 6: Stability Boxplot (gs[3,0])
    logging.info("\t\t6. Stability Boxplot")
    ax6 = fig.add_subplot(gs[7, 0])
    n_runs = 20
    auroc_scores = []
    for i in range(n_runs):
        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, y, test_size=0.2, random_state=i)
        model_i = xgb.XGBClassifier(random_state=42, n_estimators=100, max_depth=10, eval_metric='logloss')
        model_i.fit(X_train_i, y_train_i)
        y_pred_prob_i = model_i.predict_proba(X_test_i)[:, 1]
        auroc = roc_auc_score(y_test_i, y_pred_prob_i)
        auroc_scores.append(auroc)
    ax6.boxplot(auroc_scores, patch_artist=True, boxprops=dict(facecolor='blue'))
    ax6.set_ylabel("AUROC", fontsize=16)
    ax6.set_title("Stability Analysis of AUROC", fontsize=18)
    ax6.tick_params(axis='both', labelsize=14)
    
    # -------------------------------
    # Panel 7: Overlapping ROC & Precision-Recall Curves (two sub-panels)
    logging.info("\t\t7. Overlapping ROC & Precision-Recall Curves")
    gs7 = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[7, 1:3], wspace=0.4)
    
    # Generate predictions over multiple runs
    n_runs = 10
    y_true_list = []
    y_score_list = []
    for i in range(n_runs):
        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, y, test_size=0.2, random_state=i)
        model_i = train_xgboost(X_train_i, y_train_i, feature_names)
        model_i.feature_names = list(X_train_i.columns.values)
        y_pred_prob_i = model_i.predict_proba(X_test_i)[:, 1]
        y_true_list.append(y_test_i.to_numpy())
        y_score_list.append(y_pred_prob_i)
    
    logging.info("\t\t7. AUROC and AUPRC")
    # ROC Curve subplot (left of Panel 7)
    ax7_roc = fig.add_subplot(gs7[0])
    for i, (y_true, y_score) in enumerate(zip(y_true_list, y_score_list)):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax7_roc.plot(fpr, tpr, lw=1, alpha=0.8, label=f"Run {i+1} (AUC={roc_auc:.2f})")
    ax7_roc.plot([0,1], [0,1], color="navy", lw=1, linestyle="--")
    ax7_roc.set_xlim([0.0,1.0])
    ax7_roc.set_ylim([0.0,1.0])
    ax7_roc.set_xlabel("False Positive Rate", fontsize=14)
    ax7_roc.set_ylabel("True Positive Rate", fontsize=14)
    ax7_roc.set_title("ROC Curve", fontsize=16)
    ax7_roc.legend(loc="lower right", fontsize=8)
    
    # Precision-Recall Curve subplot (right of Panel 7)
    ax7_pr = fig.add_subplot(gs7[1])
    for i, (y_true, y_score) in enumerate(zip(y_true_list, y_score_list)):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_prec = average_precision_score(y_true, y_score)
        ax7_pr.plot(recall, precision, lw=1, alpha=0.8, label=f"Run {i+1} (AP={avg_prec:.2f})")
    ax7_pr.set_xlim([0.0,1.0])
    ax7_pr.set_ylim([0.0,1.05])
    ax7_pr.set_xlabel("Recall", fontsize=14)
    ax7_pr.set_ylabel("Precision", fontsize=14)
    ax7_pr.set_title("Precision-Recall Curve", fontsize=16)
    ax7_pr.legend(loc="lower left", fontsize=8)
    
    # -------------------------------
    # Panel 8: Feature Ablation Analysis (gs[4, :])
    logging.info("\t\t8. Feature Ablation Analysis")
    ax8 = fig.add_subplot(gs[8, :])
    y_pred_prob_full = full_model.predict_proba(X_test)[:, 1]
    full_auroc = roc_auc_score(y_test, y_pred_prob_full)
    feature_performance = {}
    for feature in feature_names:
        features_subset = [f for f in feature_names if f != feature]
        X_train_subset = X_train[features_subset]
        X_test_subset = X_test[features_subset]
        model_subset = train_xgboost(X_train_subset, y_train, features_subset)
        model_subset.feature_names = list(X_train_subset.columns.values)
        y_pred_prob_subset = model_subset.predict_proba(X_test_subset)[:, 1]
        auroc_subset = roc_auc_score(y_test, y_pred_prob_subset)
        feature_performance[feature] = auroc_subset
    features_ablate = list(feature_performance.keys())
    auroc_values = [feature_performance[f] for f in features_ablate]
    ax8.bar(features_ablate, auroc_values, color='blue')
    ax8.axhline(full_auroc, color='black', linestyle='--')
    ax8.set_xlabel("Removed Feature", fontsize=16)
    ax8.set_ylabel("AUROC", fontsize=16)
    ax8.set_title("Feature Ablation Analysis", fontsize=18)
    ax8.set_xticklabels(features_ablate, rotation=45, fontsize=14)
    ax8.tick_params(axis='y', labelsize=14)
    
    # plt.tight_layout()
    plt.savefig(f"{fig_dir}/combined_xgboost_analysis.png", dpi=300)
    plt.close()


def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()

    ground_truth_file: str = args.ground_truth_file
    inferred_network_file: str = args.inferred_network_file
    output_dir: str = args.output_dir
    fig_dir: str = args.fig_dir
    model_save_name: str = args.model_save_name

    inferred_network = read_inferred_network(inferred_network_file)
    ground_truth = read_ground_truth(ground_truth_file)

    logging.info("Creating set of TF-TG pairs for ground truth")
    ground_truth_pairs = set(zip(ground_truth["source_id"], ground_truth["target_id"]))

    logging.info("Adding labels to inferred network: 1 if predicted edge is in ground truth, else 0")
    inferred_network["label"] = inferred_network.apply(
        lambda row: 1 if (row["source_id"], row["target_id"]) in ground_truth_pairs else 0,
        axis=1
    )

    logging.info(inferred_network.head())
    logging.info(f'Number of True labels: {len(inferred_network[inferred_network["label"] == 1])}')
    logging.info(f'Number of False labels: {len(inferred_network[inferred_network["label"] == 0])}')
    logging.info(f'Balancing the number of True and False labels in the training set')

    drop_cols = ["source_id", "peak_id", "target_id", "label"]
    feature_names = [col for col in inferred_network.columns if col not in drop_cols]
    logging.info(f'Features:')
    for feature in feature_names:
        logging.info(f'\t{feature}')

    X = inferred_network[feature_names]
    y = inferred_network["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info("Training XGBoost Model")
    xgb_model = train_xgboost(X_train, y_train, feature_names)
    
    # Save feature names for reference
    xgb_model.feature_names = list(X_train.columns.values)
    
    logging.info("Done! Saving trained XGBoost model.")
    joblib.dump(xgb_model, f"{output_dir}/{model_save_name}.pkl")

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    logging.info("\n----- Plotting Figures -----")
    # plot_combined_figure(xgb_model, X, y, X_train, X_test, y_train, y_test, inferred_network, feature_names, fig_dir, xgb_model)
    plot_feature_score_histograms(feature_names, inferred_network, fig_dir)
    plot_feature_importance(feature_names, xgb_model, fig_dir)
    plot_xgboost_prediction_histogram(xgb_model, X_test, fig_dir)
    plot_feature_boxplots(feature_names, inferred_network, fig_dir)
    # plot_feature_ablation(feature_names, X_train, X_test, y_train, y_test, xgb_model, fig_dir)
    # plot_overlapping_roc_pr_curves(X, y, feature_names, fig_dir)
    # plot_permutation_importance_plot(xgb_model, X_test, y_test, fig_dir)
    # plot_stability_boxplot(X, y, fig_dir)
    
    
    
    
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
