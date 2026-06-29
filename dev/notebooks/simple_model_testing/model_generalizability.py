
import sys
import pandas as pd
import numpy as np
import torch
import importlib
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker, HPacker, DrawingArea
from matplotlib.patches import Rectangle

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
DATA_DIR = PROJECT_DIR / "data"
CHKPT_DIR = PROJECT_DIR / "checkpoints"
RESULT_DIR = PROJECT_DIR / "testing_results"

sys.path.append(str(PROJECT_DIR))

import models.tf_to_tg as tf_to_tg_module
import models.tf_to_dna as tf_to_dna_module
from scripts.train_tf_to_tg_model import TFTGEdgeBagDataset, collate_tftg_edge_bags
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

tf_tg_input_cache_dir = DATA_DIR / "tf_tg_training_cache"

all_evaluation_plot_dir = PROJECT_DIR / "plots" / "model_vs_test_set_evaluation_figs"
all_evaluation_plot_dir.mkdir(exist_ok=True)

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
    },
    "mouse_hepatocytes": {
        "hepatocytes_1": utils.find_latest_checkpoint(CHKPT_DIR, "mouse_hepatocytes", "hepatocytes_1"),
        "hepatocytes_3": utils.find_latest_checkpoint(CHKPT_DIR, "mouse_hepatocytes", "hepatocytes_3"),
    }
}

def run_prediction_vs_test_set(
    tf_tg_model_checkpoints: dict,
    model_cell_type: str,
    model_training_sample: str,
    test_set_cell_type: str,
    evaluation_sample: str,
    dataset_split_type: str = "test",
    subset_size: int | None = None,
    show_progress_bar: bool = True,
    compile_model: bool = True,
    ):
    
    tf_tg_model_chkpt = tf_tg_model_checkpoints[model_cell_type][model_training_sample]
    tf_dna_model_chkpt = config.tf_dna_model_checkpoints[model_cell_type]
    
    if tf_tg_model_chkpt is None:
        logging.warning(f"Skipping evaluation for {model_cell_type} {model_training_sample} → {test_set_cell_type} {evaluation_sample} due to missing TF-TG checkpoint")
        return None


    cell_type_cache_dir = DATA_DIR / f"{test_set_cell_type}_cache"

    # print(f"Loading cached dataset with subset size: {subset_size}")
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

    # print("Moving model to device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tf_tg_model.model
    model = model.to(device)

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
    
    if compile_model:
        print("Compiling model for faster evaluation")
        model = torch.compile(model)

    # print(f"Evaluating on {dataset_split_type} set")
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating", ncols=100, disable=not show_progress_bar):
            # set batch peak_distance values to zero to eliminate distance information from predictions
            batch["peak_distance"] = torch.zeros_like(batch["peak_distance"])
            
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

    col_order = [
        "Model", 
        "Test Set", 
        "auroc", 
        "auprc", 
        "accuracy", 
        "precision", 
        "early_precision", 
        "recall", 
        "f1", 
        "rand_auroc", 
        "rand_auprc",
        "n_edges",
        "n_pos",
        "n_neg",
        "score_threshold"
        ]

    metric_df = metric_df[col_order]
    
    title = f"{model_cell_type} {model_training_sample} Model → {test_set_cell_type} {evaluation_sample} Test Set"
    
    plot_data = [all_labels_flat, all_scores_flat]
    
    return {
        "metric_df": metric_df,
        "plot_data": plot_data,
        "title": title
    }
    
all_comparison_df_list = []

evaluations = [

    # === mESC Evaluations ====
    # Same cell-type, same sample evaluations with own sample test sets
    ("mESC", "E7.5_rep1", "mESC", "E7.5_rep1"),
    ("mESC", "E8.5_rep1", "mESC", "E8.5_rep1"),
    
    # Same cell-type, different sample evaluations with mouse hepatocyte test sets
    ("mESC", "E7.5_rep1", "mESC", "E8.5_rep1"),
    ("mESC", "E8.5_rep1", "mESC", "E7.5_rep1"),
    
    # Cross cell-type, same organism evaluations with mESC test sets
    ("mESC", "E7.5_rep1", "mouse_hepatocytes", "hepatocytes_1"),
    ("mESC", "E7.5_rep1", "mouse_hepatocytes", "hepatocytes_3"),
    ("mESC", "E8.5_rep1", "mouse_hepatocytes", "hepatocytes_1"),
    ("mESC", "E8.5_rep1", "mouse_hepatocytes", "hepatocytes_3"),
    
    # Cross cell-type, different organism evaluations with Macrophage test sets
    ("mESC", "E7.5_rep1", "Macrophage", "buffer_1"),
    ("mESC", "E7.5_rep1", "Macrophage", "buffer_2"),
    ("mESC", "E8.5_rep1", "Macrophage", "buffer_1"),
    ("mESC", "E8.5_rep1", "Macrophage", "buffer_2"),
    
    # ==== Hepatocyte Evaluations ====
    # Same cell-type, same sample evaluations with own sample test sets
    ("mouse_hepatocytes", "hepatocytes_1", "mouse_hepatocytes", "hepatocytes_1"),
    ("mouse_hepatocytes", "hepatocytes_3", "mouse_hepatocytes", "hepatocytes_3"),
    
    # Same cell-type, different sample evaluations with mouse hepatocyte test sets
    ("mouse_hepatocytes", "hepatocytes_1", "mouse_hepatocytes", "hepatocytes_3"),
    ("mouse_hepatocytes", "hepatocytes_3", "mouse_hepatocytes", "hepatocytes_1"),
    
    # Cross cell-type, same organism evaluations with mESC test sets
    ("mouse_hepatocytes", "hepatocytes_1", "mESC", "E7.5_rep1"),
    ("mouse_hepatocytes", "hepatocytes_1", "mESC", "E8.5_rep1"),
    ("mouse_hepatocytes", "hepatocytes_3", "mESC", "E7.5_rep1"),
    ("mouse_hepatocytes", "hepatocytes_3", "mESC", "E8.5_rep1"),
    
    # Cross cell-type, different organism evaluations with Macrophage test sets
    ("mouse_hepatocytes", "hepatocytes_1", "Macrophage", "buffer_1"),
    ("mouse_hepatocytes", "hepatocytes_1", "Macrophage", "buffer_2"),
    ("mouse_hepatocytes", "hepatocytes_3", "Macrophage", "buffer_1"),
    ("mouse_hepatocytes", "hepatocytes_3", "Macrophage", "buffer_2"),
    
    # === Macrophage Evaluations ====
    # Same cell-type, same sample evaluations with own sample test sets
    ("Macrophage", "buffer_1", "Macrophage", "buffer_1"),
    ("Macrophage", "buffer_2", "Macrophage", "buffer_2"),
    
    # Same cell-type, different sample evaluations with Macrophage test sets
    ("Macrophage", "buffer_1", "Macrophage", "buffer_2"),
    ("Macrophage", "buffer_2", "Macrophage", "buffer_1"),

    # Cross cell-type, different organism evaluations with mESC test sets
    ("Macrophage", "buffer_1", "mESC", "E7.5_rep1"),
    ("Macrophage", "buffer_1", "mESC", "E8.5_rep1"),
    ("Macrophage", "buffer_2", "mESC", "E7.5_rep1"),
    ("Macrophage", "buffer_2", "mESC", "E8.5_rep1"),
    
    # Cross-cell type, different organism evaluations with mouse hepatocyte test sets
    ("Macrophage", "buffer_1", "mouse_hepatocytes", "hepatocytes_1"),
    ("Macrophage", "buffer_1", "mouse_hepatocytes", "hepatocytes_3"),
    ("Macrophage", "buffer_2", "mouse_hepatocytes", "hepatocytes_1"),
    ("Macrophage", "buffer_2", "mouse_hepatocytes", "hepatocytes_3"),
    
]

all_plot_data = {}

subset_size = None
# for model_cell_type, model_training_sample, test_set_cell_type, evaluation_sample in tqdm(evaluations, desc="Evaluating all model vs test set combinations", ncols=100):
for model_cell_type, model_training_sample, test_set_cell_type, evaluation_sample in tqdm(evaluations, desc="Evaluating model vs test set combinations", ncols=100):
    logging.info(f"Evaluating {model_cell_type} {model_training_sample} Model → {test_set_cell_type} {evaluation_sample} Test Set")

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
        compile_model=False
    )
        
    metric_df = comparison_result["metric_df"]
    plot_data = comparison_result["plot_data"]
    
    all_labels_flat = plot_data[0]
    all_scores_flat = plot_data[1]
    
    title = comparison_result["title"]
    
    all_plot_data[title] = (all_labels_flat, all_scores_flat)
    
    all_comparison_df_list.append(metric_df)
    
full_comparison_df = pd.concat(all_comparison_df_list, ignore_index=True)

full_comparison_df.to_csv(RESULT_DIR / "model_generalizability_results.csv", index=False)