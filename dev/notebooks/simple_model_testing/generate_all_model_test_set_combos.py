
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,

)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
DATA_DIR = PROJECT_DIR / "data"
CHKPT_DIR = PROJECT_DIR / "checkpoints"
RESULT_DIR = PROJECT_DIR / "testing_results"

sys.path.append(str(PROJECT_DIR))

import models.tf_to_tg as tf_to_tg_module
import plotting_utils
import stat_utils
import utils
import warnings
import config

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

tf_tg_input_cache_dir = DATA_DIR / "tf_tg_training_cache"

all_evaluation_plot_dir = PROJECT_DIR / "plots" / "model_vs_test_set_evaluation_figs"
all_evaluation_plot_dir.mkdir(exist_ok=True)

def run_prediction_vs_test_set(
    tf_tg_model_checkpoints: dict,
    model_cell_type: str,
    model_training_sample: str,
    test_set_cell_type: str,
    evaluation_sample: str,
    dataset_split_type: str = "test",
    subset_size: int | None = None,
    show_progress_bar: bool = True
    ):
    
    tf_tg_model_chkpt = tf_tg_model_checkpoints[model_cell_type][model_training_sample]
    tf_dna_model_chkpt = config.tf_dna_model_checkpoints[model_cell_type]
    
    if tf_tg_model_chkpt is None:
        logging.warning(f"Skipping evaluation for {model_cell_type} {model_training_sample} → {test_set_cell_type} {evaluation_sample} due to missing TF-TG checkpoint")
        return None


    cell_type_cache_dir = DATA_DIR / f"{test_set_cell_type}_cache"

    # logging.info(f"Loading cached dataset with subset size: {subset_size}")
    data_loader, metadata, manifest, tf_embeddings_tensor, tf_mask_tensor = utils.load_training_cache_dataset(
        sample_name=evaluation_sample,
        cell_type_cache_dir=cell_type_cache_dir,
        split_type=dataset_split_type,
        subset_size=subset_size
        )
    
    tf_tg_model = utils.load_tf_tg_regulation_model(
        tf_dna_model_chkpt, 
        tf_tg_model_chkpt, 
        tf_embeddings_tensor, 
        tf_mask_tensor
        )

    # logging.info("Moving model to device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tf_tg_model.model
    model = model.to(device)
    
    # compile the model
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    criterion = torch.nn.BCEWithLogitsLoss()
    score_threshold = 0.5
    pooling_mode = "lse"
    pooling_temperature = 1.0

    model.eval()

    total_loss = 0.0
    n_edges = 0

    all_scores = []
    all_labels = []
    plot_data = {}

    # logging.info(f"Evaluating on {dataset_split_type} set")
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating", ncols=100, disable=not show_progress_bar):            
            batch = tf_to_tg_module.move_batch_to_device(batch, device)

            labels = batch["label"]
            cell_mask = batch["cell_mask"]
            E, C = cell_mask.shape
            
            
            edge_logits, _ = model.forward(
                tf_embedding=batch["tf_embedding"],
                tf_mask=batch["tf_mask"],
                peak_sequences=batch["peak_sequences"],
                peak_accessibility=batch["peak_accessibility"],
                peak_distance=batch["peak_distance"],
                tf_expression=batch["tf_expression"],
                tg_expression=batch["tg_expression"],
                peak_mask=batch.get("peak_mask", None),
                cell_mask=cell_mask,
                pooling_mode=pooling_mode,
                pooling_temperature=pooling_temperature,
            )

            loss = criterion(edge_logits, labels)

            total_loss += loss.item() * E
            n_edges += E

            scores = torch.sigmoid(edge_logits)

            all_scores.append(scores.detach().cpu().numpy().ravel())
            all_labels.append(labels.detach().cpu().numpy().ravel())

    all_scores_flat = np.concatenate(all_scores)
    all_labels_flat = np.concatenate(all_labels)

    metrics = stat_utils.compute_binary_classification_metrics(
        labels=all_labels_flat,
        scores=all_scores_flat,
        score_threshold=score_threshold,
        random_state=42,
    )

    metrics["Model"] = model_training_sample
    metrics["Test Set"] = evaluation_sample

    metric_df = pd.DataFrame([metrics])

    col_order = ["Model", "Test Set", "auroc", "auprc", "accuracy", "precision", "recall", "rand_auroc", "rand_auprc"]

    metric_df = metric_df[col_order]
    
    title = f"{model_cell_type} {model_training_sample} Model → {test_set_cell_type} {evaluation_sample} Test Set"
    
    plot_data = [all_labels_flat, all_scores_flat]
    
    return {
        "metric_df": metric_df,
        "plot_data": plot_data,
        "title": title
    }
    
if __name__ == "__main__":
    tf_tg_model_checkpoints = {
        "mESC": {
            # "E7.5_rep1": CHKPT_DIR / "mESC" / "E7.5_rep1" / "tf_tg_train_E7.5_rep1_3675131" / "epoch_11_best_model.ckpt",
            "E7.5_rep1": utils.find_latest_checkpoint(CHKPT_DIR, "mESC", "E7.5_rep1"),
            "E7.5_rep2": utils.find_latest_checkpoint(CHKPT_DIR, "mESC", "E7.5_rep2"),
            "E8.5_rep1": utils.find_latest_checkpoint(CHKPT_DIR, "mESC", "E8.5_rep1", training_number="3691937"),
            "E8.5_rep2": utils.find_latest_checkpoint(CHKPT_DIR, "mESC", "E8.5_rep2"),
        },
        "iPSC": {
            "WT_D13_rep1": utils.find_latest_checkpoint(CHKPT_DIR, "iPSC", "WT_D13_rep1"),
        },
        "Macrophage": {
            "buffer_1": utils.find_latest_checkpoint(CHKPT_DIR, "Macrophage", "buffer_1", training_number="3685893"),
            "buffer_2": utils.find_latest_checkpoint(CHKPT_DIR, "Macrophage", "buffer_2"),
            "buffer_3": utils.find_latest_checkpoint(CHKPT_DIR, "Macrophage", "buffer_3"),
            "buffer_4": utils.find_latest_checkpoint(CHKPT_DIR, "Macrophage", "buffer_4"),
        },
        "K562": {
            "sample_1": utils.find_latest_checkpoint(CHKPT_DIR, "K562", "sample_1"),
        },
        "mouse_liver": {
            "liver_1": utils.find_latest_checkpoint(CHKPT_DIR, "mouse_liver", "liver_1"),
            "liver_3": utils.find_latest_checkpoint(CHKPT_DIR, "mouse_liver", "liver_3")
        }
    }
    
    # All combinations of models vs all test sets
    sample_list_all = [
        ("mESC", "E7.5_rep1"),
        # ("mESC", "E7.5_rep2"),
        ("mESC", "E8.5_rep1"),
        # ("mESC", "E8.5_rep2"),
        # ("iPSC", "WT_D13_rep1"),
        ("Macrophage", "buffer_1"),
        ("Macrophage", "buffer_2"),
        # ("Macrophage", "buffer_3"),
        # ("Macrophage", "buffer_4"),
        ("K562", "sample_1"),
        ("mouse_liver", "liver_1"),
        ("mouse_liver", "liver_3")
    ]
    
    evaluation_cache = RESULT_DIR / "evaluation_cache"
    evaluation_cache.mkdir(exist_ok=True)
    
    # Check if intermediate results exist and load them
    intermediate_results_path = evaluation_cache / "intermediate_results.pkl"
    if intermediate_results_path.exists():
        with open(intermediate_results_path, "rb") as f:
            intermediate_results = pickle.load(f)
            all_comparison_df_list = intermediate_results["all_comparison_df_list"]
            all_plot_data = intermediate_results["all_plot_data"]
            
            # Skip already evaluated combinations
            evaluated_combos = set(
                (df["Model"].iloc[0], df["Test Set"].iloc[0])
                for df in all_comparison_df_list
            )
            logging.info(f"Skipping {len(evaluated_combos)} already evaluated combinations")
            
            sample_list_all = [
                (model_cell_type, model_training_sample)
                for model_cell_type, model_training_sample in sample_list_all
                if (model_cell_type, model_training_sample) not in evaluated_combos
            ]
    else:
        all_comparison_df_list = []
        all_plot_data = {}

    all_evaluation_combos = []
    for model_cell_type, model_training_sample in sample_list_all:
        for test_set_cell_type, evaluation_sample in sample_list_all:
            all_evaluation_combos.append((model_cell_type, model_training_sample, test_set_cell_type, evaluation_sample))
            
    logging.info(f"Total evaluation combinations: {len(all_evaluation_combos)}")
    
    all_comparison_df_list = []
    all_plot_data = {}

    subset_size = 5000
    for model_cell_type, model_training_sample, test_set_cell_type, evaluation_sample in tqdm(all_evaluation_combos, desc="Evaluating all model vs test set combinations", ncols=100):
        use_val = ["sample_1", "buffer_3", "buffer_4"]
        
        if evaluation_sample in use_val:
            dataset_split_type = "val"
        else:
            dataset_split_type = "test"
            
        comparison_result = run_prediction_vs_test_set(
            tf_tg_model_checkpoints=tf_tg_model_checkpoints,
            model_cell_type=model_cell_type,
            model_training_sample=model_training_sample,
            test_set_cell_type=test_set_cell_type,
            evaluation_sample=evaluation_sample,
            dataset_split_type=dataset_split_type,
            subset_size=subset_size,
            show_progress_bar=False,
        )
            
        metric_df = comparison_result["metric_df"]
        plot_data = comparison_result["plot_data"]
        
        all_labels_flat = plot_data[0]
        all_scores_flat = plot_data[1]
        
        title = comparison_result["title"]
        
        all_plot_data[title] = (all_labels_flat, all_scores_flat)
        
        all_comparison_df_list.append(metric_df)
        
        # Save the intermediate results to pickle after each evaluation to avoid losing progress
        intermediate_results = {
            "all_comparison_df_list": all_comparison_df_list,
            "all_plot_data": all_plot_data,
        }
        with open(intermediate_results_path, "wb") as f:
            pickle.dump(intermediate_results, f)
        
    full_comparison_df = pd.concat(all_comparison_df_list, ignore_index=True)

    logging.info("Saving full comparison metrics to CSV")
    full_comparison_df.to_csv(RESULT_DIR / "full_comparison_metrics.csv", index=False)
    
    # Generate a grid of ROC curves for all model vs test set combinations
    model_samples = sorted(set([sample for _, sample in sample_list_all]))
    fig, axes = plt.subplots(
        nrows=len(model_samples),
        ncols=len(model_samples),
        figsize=(12, 12),
        sharex=True,
        sharey=True,
    )

    for i, model_sample in enumerate(model_samples):
        for j, test_sample in enumerate(model_samples):

            ax = axes[i, j]

            matching_titles = [
                key for key in all_plot_data
                if f"{model_sample} Model" in key and f"{test_sample} Test Set" in key
            ]

            if len(matching_titles) == 0:
                ax.axis("off")
                continue

            title = matching_titles[0]

            if title not in all_plot_data:
                ax.axis("off")
                ax.set_title("Missing", fontsize=8)
                continue

            labels = all_plot_data[title][0]
            scores = all_plot_data[title][1]

            labels = np.asarray(labels).astype(int).ravel()
            scores = np.asarray(scores).astype(float).ravel()

            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = roc_auc_score(labels, scores)

            rand_scores = plotting_utils._create_random_distribution(scores)
            rand_fpr, rand_tpr, _ = roc_curve(labels, rand_scores)

            ax.plot(
                fpr,
                tpr,
                lw=2,
                color="#4195df",
                zorder=3,
            )

            ax.plot(
                rand_fpr,
                rand_tpr,
                color="#747474",
                linestyle="--",
                lw=1.5,
                zorder=2,
            )
            
            # Interpolate random TPR onto model FPR grid
            rand_tpr_interp = np.interp(fpr, rand_fpr, rand_tpr)

            # Fill only where model ROC is above random ROC
            ax.fill_between(
                fpr,
                tpr,
                rand_tpr_interp,
                where=(tpr > rand_tpr_interp),
                interpolate=True,
                color="green",
                alpha=0.25,
                zorder=1,
            )

            # ax.plot(
            #     [0, 1],
            #     [0, 1],
            #     color="black",
            #     linestyle=":",
            #     lw=0.8,
            #     zorder=1,
            # )

            ax.text(
                0.20,
                0.07,
                f"AUROC={auroc:.3f}",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(facecolor="none", edgecolor="none"),
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.tick_params(bottom=False, left=False, right=False)

            if i == 0:
                ax.set_title(test_sample, fontsize=12, ha="center")

            if j == 0:
                ax.set_ylabel(model_sample, fontsize=12)
                ax.set_yticks([])

            if i == len(model_samples) - 1:
                ax.set_xlabel("")
                ax.set_xticks([])

    fig.suptitle("ROC Curves for TF-TG Model Evaluations", fontsize=16, y=1.02)

    fig.text(
        0.5,
        0.02,
        "Test Set",
        ha="center",
        fontsize=14,
    )

    fig.text(
        -0.01,
        0.5,
        "Model",
        va="center",
        rotation="vertical",
        fontsize=14,
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        bottom=0.06,
        top=0.96,
        wspace=0.05,
        hspace=0.05,
    )
    plt.show()

    fig.savefig(all_evaluation_plot_dir / "model_vs_test_set_roc_curves.png", dpi=300, bbox_inches="tight")

    # Heatmap of AUROC values for each model vs test set combination
    test_comparison_df = full_comparison_df.copy()

    test_comparison_df["AUPRC Lift"] = test_comparison_df["auprc"] - test_comparison_df["rand_auprc"]
    test_comparison_df = test_comparison_df.rename(columns={
        "auroc": "AUROC", 
        "auprc": "AUPRC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        })

    evaluation_metrics = ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall", "F1 Score", "AUPRC Lift"]

    for selected_metric in evaluation_metrics:
        test_comparison_df = test_comparison_df[["Model", "Test Set", selected_metric]]

        test_comparison_df_pivot = test_comparison_df.pivot(index="Model", columns="Test Set", values=selected_metric)

        fig = plt.figure(figsize=(10, 8))
        heatmap_fig = sns.heatmap(
            test_comparison_df_pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={'label': selected_metric},
            linewidths=0.5,
            linecolor='gray',
            annot_kws={"size": 12}
        )
        heatmap_fig.set_title(f"{selected_metric} Heatmap for TF-TG Model Evaluations", fontsize=16)
        heatmap_fig.set_xlabel("Test Set", fontsize=14)
        heatmap_fig.set_ylabel("Model", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(rotation=0, fontsize=14)
        plt.tight_layout()
        plt.show()

        fig.savefig(all_evaluation_plot_dir / f"model_vs_test_set_{selected_metric.lower().replace(' ', '_')}_heatmap.png", dpi=300, bbox_inches="tight")