
import json
import re
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
DATA_DIR = PROJECT_DIR / "cached_data"
CHKPT_DIR = PROJECT_DIR / "checkpoints"
RESULT_DIR = PROJECT_DIR / "testing_results" / "stability_evaluation"

sys.path.append(str(PROJECT_DIR))

import models.tf_to_tg as tf_to_tg_module
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

all_evaluation_plot_dir = PROJECT_DIR / "plots" / "stability_evaluation"
all_evaluation_plot_dir.mkdir(exist_ok=True)

def run_prediction_vs_test_set(
    latest_checkpoint: Path,
    cell_type: str,
    sample_name: str,
    subsample_num: int,
    dataset_split_type: str = "test",
    subset_size: int | None = None,
    show_progress_bar: bool = True,
    compile_model: bool = True,
    batch_size: int = 512,
    tf_idx_to_name: dict | None = None,
    tg_idx_to_name: dict | None = None
    ):
    
    tf_tg_model_chkpt = latest_checkpoint
    tf_dna_model_chkpt = config.tf_dna_model_checkpoints[cell_type]
    
    if tf_tg_model_chkpt is None:
        logging.warning(f"Skipping evaluation for {cell_type} {sample_name} {subsample_num} due to missing TF-TG checkpoint")
        return None

    cell_type_cache_dir = DATA_DIR / f"{cell_type}_cache"

    # Load the training data for the test set 
    data_loader, metadata, manifest, tf_embeddings_tensor, tf_mask_tensor = utils.load_training_cache_dataset(
        sample_name=sample_name,
        cell_type_cache_dir=cell_type_cache_dir,
        split_type=dataset_split_type,
        subset_size=subset_size,
        batch_size=batch_size
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the TF-TG regulation model
    tf_tg_model = utils.load_tf_tg_regulation_model(
        tf_dna_model_chkpt, 
        tf_tg_model_chkpt, 
        tf_embeddings_tensor, 
        tf_mask_tensor,
        compile_model=compile_model,
        device=device
        )

    model = tf_tg_model.model
    model = model.to(device)

    # Run evaluation on the test set and compute metrics
    criterion = torch.nn.BCEWithLogitsLoss()
    score_threshold = 0.5
    pooling_mode = "lse"
    pooling_temperature = 1.0

    model.eval()

    total_loss = 0.0
    n_edges = 0

    tf_indices_list = []
    tg_indices_list = []
    all_scores = []
    all_labels = []

    # print(f"Evaluating on {dataset_split_type} set")
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating", ncols=100, disable=not show_progress_bar):
            tf_indices = batch["tf_idx"].detach().cpu().numpy().ravel()
            tg_indices = batch["tg_idx"].detach().cpu().numpy().ravel()
            
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
            
            tf_indices_list.append(tf_indices)
            tg_indices_list.append(tg_indices)

    all_tf_indices_flat = np.concatenate(tf_indices_list)
    all_tg_indices_flat = np.concatenate(tg_indices_list)
    all_scores_flat = np.concatenate(all_scores)
    all_labels_flat = np.concatenate(all_labels)

    tf_names = [tf_idx_to_name[int(idx)].upper() for idx in all_tf_indices_flat]
    tg_names = [tg_idx_to_name[int(idx)].upper() for idx in all_tg_indices_flat]
    
    prediction_df = pd.DataFrame({
        "Source": tf_names,
        "Target": tg_names,
        "Score": all_scores_flat,
        "Label": all_labels_flat
    })

    metrics = stat_utils.compute_binary_classification_metrics(
        labels=all_labels_flat,
        scores=all_scores_flat,
        score_threshold=score_threshold,
        random_state=42,
    )

    metrics["sample_name"] = sample_name
    metrics["subsample_num"] = subsample_num

    metric_df = pd.DataFrame([metrics])
    
    # Get info about the dataset size for the test set
    peaks_per_tg = metadata.get("max_peaks_per_tg", None)
    cells_per_pair = metadata.get("max_cells_per_pair", None)
    max_peaks_real = metadata.get("max_peaks_real", None)
    
    num_tfs = len(metadata["tf_name_to_idx"])
    num_tgs = len(metadata["tg_id_to_idx"])
    
    metric_df["peaks_per_tg"] = peaks_per_tg
    metric_df["cells_per_pair"] = cells_per_pair
    metric_df["max_peaks_real"] = max_peaks_real
    metric_df["num_tfs"] = num_tfs
    metric_df["num_tgs"] = num_tgs
    metric_df["subset_size"] = subset_size
    metric_df["batch_size"] = batch_size

    col_order = [
        "sample_name", 
        "subsample_num", 
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
        "score_threshold",
        "peaks_per_tg",
        "cells_per_pair",
        "max_peaks_real",
        "num_tfs",
        "num_tgs",
        "subset_size",
        "batch_size"
        ]

    metric_df = metric_df[col_order]
            
    return {
        "metric_df": metric_df,
        "prediction_df": prediction_df
    }
    
import argparse

_CHECKPOINT_PATTERN = re.compile(
    r"^epoch=(?P<epoch>\d+)-val_auroc=(?P<val_auroc>[\d.]+)-val_loss=(?P<val_loss>[\d.]+)$"
)


def find_best_stability_checkpoint(
    checkpoint_dir: Path,
    epoch_num: int|None =None,
    verbose: bool = True
    ) -> Path:
    """
    Find the best checkpoint file for a given cell type and sample name.

    Selection is on the validation AUROC encoded in the filename, not on the epoch
    number. Training keeps only the top-2 checkpoints by val/auroc (plus last.ckpt),
    so the highest-epoch surviving file is not reliably the best model -- across the
    checkpoint directories in this repo the two criteria disagree about half the time.

    Kept identical to the copy in generate_stability_test_set_grns.py so the two
    scripts cannot select different checkpoints for the same run.

    Parameters
    ----------
    checkpoint_dir : Path
        The base directory where checkpoints are stored.
    epoch_num : int, optional
        Pin selection to this specific epoch instead of taking the best val AUROC.

    Returns
    -------
    Path or None
        The path to the selected checkpoint file, or None if no checkpoint is found.

    """

    if not checkpoint_dir.exists():
        logging.warning(f"No checkpoints found in {checkpoint_dir}")
        return None

    # Find all checkpoint files in the checkpoint directory
    chkpt_files = list(checkpoint_dir.glob("epoch=*-val_auroc=*-val_loss=*.ckpt"))
    if not chkpt_files:
        logging.warning(f"No checkpoint files found in {checkpoint_dir}")
        return None

    parsed_chkpts = []
    for chkpt_file in chkpt_files:
        match = _CHECKPOINT_PATTERN.match(chkpt_file.stem)
        if match is None:
            logging.warning(f"Skipping checkpoint with unrecognized filename: {chkpt_file.name}")
            continue
        parsed_chkpts.append((int(match.group("epoch")), float(match.group("val_auroc")), chkpt_file))

    if not parsed_chkpts:
        logging.warning(f"No parseable checkpoint files found in {checkpoint_dir}")
        return None

    # If epoch_num is specified, pin to that epoch. Otherwise take the best val AUROC.
    if epoch_num is not None:
        matching = [chkpt for chkpt in parsed_chkpts if chkpt[0] == epoch_num]
        if not matching:
            available_epochs = sorted(chkpt[0] for chkpt in parsed_chkpts)
            logging.warning(f"Checkpoint for epoch {epoch_num} not found in {checkpoint_dir}. Available epochs: {available_epochs}")
            return None
        epoch, val_auroc, best_chkpt_file = matching[0]
    else:
        # Tie-break on the later epoch so the choice is deterministic
        epoch, val_auroc, best_chkpt_file = max(parsed_chkpts, key=lambda chkpt: (chkpt[1], chkpt[0]))

    if verbose:
        logging.info(f"Selected checkpoint: epoch {epoch}, val AUROC {val_auroc:.4f}")

    return best_chkpt_file

def parse_args():
    
    parser = argparse.ArgumentParser(description="Evaluate model generalizability across different cell types and samples.")
    parser.add_argument("--cell_type", type=str, default=None, help="Model cell type for evaluation.")
    parser.add_argument("--sample_name", type=str, default=None, help="Model training sample for evaluation.")
    parser.add_argument("--subsample_num", type=int, default=None, help="Subsample number for evaluation.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for evaluation.")
    parser.add_argument("--force_reload", action="store_true", help="Force reload of the dataset even if cached.")
    return parser.parse_args()

def create_tf_tg_index_to_name_mappings(metadata):
    tf_idx_to_name = {idx: name for name, idx in metadata["tf_name_to_idx"].items()}
    tg_idx_to_name = {idx: name for name, idx in metadata["tg_id_to_idx"].items()}
    return tf_idx_to_name, tg_idx_to_name

if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size

    cell_type = args.cell_type
    sample_name = args.sample_name
    subsample_num = args.subsample_num
    force_reload = args.force_reload

    # for cell_type, sample_name, test_set_cell_type, evaluation_sample in tqdm(evaluations, desc="Evaluating model vs test set combinations", ncols=100):
    logging.info(f"Evaluating {cell_type} {sample_name} Model")

    dataset_split_type = "test"
    
    cell_type_cache_dir = DATA_DIR / f"{cell_type}_cache"
    
    prediction_save_file = RESULT_DIR / "labeled_grns" / f"{sample_name}_stability_{subsample_num}_grn.csv"
    metric_save_file = RESULT_DIR / "comparison_metric_files" / f"{sample_name}_stability_{subsample_num}_metrics.csv"

    if not prediction_save_file.parent.exists():
        prediction_save_file.parent.mkdir(parents=True, exist_ok=True)
    if not metric_save_file.parent.exists():
        metric_save_file.parent.mkdir(parents=True, exist_ok=True)

    if prediction_save_file.exists() and metric_save_file.exists() and not force_reload:
        logging.info(f"Prediction and metric files already exist for {cell_type} {sample_name}. Skipping evaluation.")
        sys.exit(0)
        
    # Load the TF and TG name to index mappings from the training cache metadata
    with open(cell_type_cache_dir / "tf_tg_training_cache" / sample_name / "metadata.json", "r") as f:
        metadata = json.load(f)
        
    tf_name_to_idx = metadata["tf_name_to_idx"]
    tg_id_to_idx = metadata["tg_id_to_idx"]

    tf_idx_to_name, tg_idx_to_name = create_tf_tg_index_to_name_mappings(metadata)
        
    stability_checkpoints = {}

    stability_sample_dir = CHKPT_DIR / "stability" / cell_type / sample_name / f"stability_{subsample_num}"

    assert stability_sample_dir.exists(), f"Stability checkpoint directory does not exist: {stability_sample_dir}"

    latest_checkpoint = find_best_stability_checkpoint(
        stability_sample_dir,
        verbose=True
        )

    comparison_result = run_prediction_vs_test_set(
        latest_checkpoint=latest_checkpoint,
        cell_type=cell_type,
        sample_name=sample_name,
        subsample_num=subsample_num,
        dataset_split_type=dataset_split_type,
        subset_size=None,
        show_progress_bar=True,
        compile_model=False,
        batch_size=batch_size,
        tf_idx_to_name=tf_idx_to_name,
        tg_idx_to_name=tg_idx_to_name
    )
        
    metric_df = comparison_result["metric_df"]
    prediction_df = comparison_result["prediction_df"]
    
    prediction_df.to_csv(prediction_save_file, index=False)
    
    metric_df.to_csv(metric_save_file, index=False)

    logging.info("Done!")