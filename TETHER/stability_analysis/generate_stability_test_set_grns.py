
import hashlib
import json
import re
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader, Subset

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
DATA_DIR = PROJECT_DIR / "cached_data"
CHKPT_DIR = PROJECT_DIR / "checkpoints"
RESULT_DIR = PROJECT_DIR / "testing_results" / "stability_results"

sys.path.append(str(PROJECT_DIR))

from scripts.train_tf_to_tg_model import TFTGEdgeBagDataset, collate_tftg_edge_bags
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

# Seed for the --subset_size draw. Fixed so every subsample evaluates the same edges.
SUBSET_SEED = 42

all_evaluation_plot_dir = PROJECT_DIR / "plots" / "model_vs_test_set_evaluation_figs"
all_evaluation_plot_dir.mkdir(parents=True, exist_ok=True)

def load_stability_training_cache_dataset(
    cell_type_cache_dir: Path,
    stability_cache_dir: Path,
    split_type: str = "test",
    subset_size: int = None,
    batch_size: int = 512,
    subset_seed: int = SUBSET_SEED,
    ) -> DataLoader:
    
    assert split_type in ["train", "val", "test"], \
        "split_type must be one of 'train', 'val', or 'test'"
        
    # Load the compact split inputs
    tftg_inputs_test = torch.load(
        stability_cache_dir / f"tftg_inputs_{split_type}.pt",
        weights_only=False,
    )

    # Load the lookup tensors
    tf_embeddings_tensor = torch.load(
        cell_type_cache_dir / "tf_embeddings.pt",
        weights_only=True,
    )
    tf_mask_tensor = torch.load(
        cell_type_cache_dir / "tf_masks.pt",
        weights_only=True,
    )
    atac_peak_tensor = torch.load(
        stability_cache_dir / "atac_peak_tensor.pt",
        weights_only=True,
    )

    # Load the metadata
    with open(stability_cache_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Load the manifest and verify tensor shapes and dtypes match expectations
    with open(stability_cache_dir / "manifest.json") as f:
        manifest = json.load(f)
    
    assert tuple(manifest["atac_peak_tensor_shape"]) == tuple(atac_peak_tensor.shape)
    assert manifest["atac_peak_tensor_dtype"] == str(atac_peak_tensor.dtype)

    dataset = TFTGEdgeBagDataset(
        tftg_inputs_test,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor
    )
    
    if subset_size is not None:
        subset_size = min(subset_size, len(dataset))
        # Draw the subset at random rather than taking the first N. The cached edges
        # are ordered by (tf_name, tg_id), so range(subset_size) would keep only the
        # alphabetically-first TFs. The seed is fixed and the cache order is canonical
        # across subsamples, so every subsample still evaluates the same edges.
        subset_rng = np.random.default_rng(subset_seed)
        subset_indices = sorted(
            subset_rng.choice(len(dataset), size=subset_size, replace=False).tolist()
        )
        dataset = Subset(dataset, subset_indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_tftg_edge_bags,
        )

    return loader, metadata, manifest, tf_embeddings_tensor, tf_mask_tensor

_CHECKPOINT_PATTERN = re.compile(
    r"^epoch=(?P<epoch>\d+)-val_auroc=(?P<val_auroc>[\d.]+)-val_loss=(?P<val_loss>[\d.]+)$"
)


def find_best_stability_checkpoint(
    checkpoint_dir: Path,
    epoch_num: int | None = None,
    verbose: bool = True,
    ) -> Path:
    """
    Find the best retrained-subsample checkpoint for a stability run.

    Selection is on the validation AUROC encoded in the filename, not on the epoch
    number. Training keeps only the top-2 checkpoints by val/auroc (plus last.ckpt),
    so the highest-epoch surviving file is not reliably the best model -- across the
    checkpoint directories in this repo the two criteria disagree about half the time.

    Parameters
    ----------
    checkpoint_dir : Path
        The stability subsample directory, e.g.
        checkpoints/stability/{cell_type}/{sample}/stability_{N}.
    epoch_num : int, optional
        Pin selection to this specific epoch instead of taking the best val AUROC.

    Returns
    -------
    Path or None
        The path to the selected checkpoint file, or None if none is found.
    """
    if not checkpoint_dir.exists():
        logging.warning(f"No checkpoints found in {checkpoint_dir}")
        return None

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
        logging.info(f"Selected stability checkpoint: epoch {epoch}, val AUROC {val_auroc:.4f}")

    return best_chkpt_file

def run_prediction_vs_test_set(
    tf_tg_model_chkpt: Path,
    model_cell_type: str,
    model_training_sample: str,
    test_set_cell_type: str,
    cell_type_cache_dir: Path,
    stability_cache_dir: Path,
    evaluation_sample: str,
    stability_number: int | None = None,
    dataset_split_type: str = "test",
    subset_size: int | None = None,
    show_progress_bar: bool = True,
    compile_model: bool = True,
    batch_size: int = 512,
    tf_idx_to_name: dict | None = None,
    tg_idx_to_name: dict | None = None
    ):

    tf_dna_model_chkpt = config.tf_dna_model_checkpoints[model_cell_type]

    if tf_tg_model_chkpt is None:
        logging.warning(f"Skipping evaluation for {model_cell_type} {model_training_sample} → {test_set_cell_type} {evaluation_sample} due to missing TF-TG checkpoint")
        return None

    # print(f"Loading cached dataset with subset size: {subset_size}")
    data_loader, metadata, manifest, tf_embeddings_tensor, tf_mask_tensor = load_stability_training_cache_dataset(
        cell_type_cache_dir=cell_type_cache_dir,
        stability_cache_dir=stability_cache_dir,
        split_type=dataset_split_type,
        subset_size=subset_size,
        batch_size=batch_size
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

    score_threshold = 0.5
    pooling_mode = "lse"
    pooling_temperature = 1.0

    model.eval()

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
    
    return prediction_df
    
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description="Evaluate model stability across different cell types and samples.")
    parser.add_argument("--model_cell_type", type=str, default=None, help="Model cell type for evaluation.")
    parser.add_argument("--model_training_sample", type=str, default=None, help="Model training sample for evaluation.")
    parser.add_argument("--test_set_cell_type", type=str, default=None, help="Test set cell type for evaluation.")
    parser.add_argument("--evaluation_sample", type=str, default=None, help="Evaluation sample for the test set.")
    parser.add_argument("--stability_number", type=int, default=None, help="Stability number for the evaluation.")
    parser.add_argument("--subset_size", type=int, default=None, help="Subset size for evaluation. If None, use the full dataset.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for evaluation.")
    parser.add_argument("--force_reload", action="store_true", help="Force reload of the dataset even if cached.")
    return parser.parse_args()

def create_tf_tg_index_to_name_mappings(metadata):
    tf_idx_to_name = {idx: name for name, idx in metadata["tf_name_to_idx"].items()}
    tg_idx_to_name = {idx: name for name, idx in metadata["tg_id_to_idx"].items()}
    return tf_idx_to_name, tg_idx_to_name

if __name__ == "__main__":
    args = parse_args()
    subset_size = args.subset_size
    batch_size = args.batch_size

    model_cell_type = args.model_cell_type
    model_training_sample = args.model_training_sample
    test_set_cell_type = args.test_set_cell_type
    evaluation_sample = args.evaluation_sample
    stability_number = args.stability_number
    force_reload = args.force_reload

    # for model_cell_type, model_training_sample, test_set_cell_type, evaluation_sample in tqdm(evaluations, desc="Evaluating model vs test set combinations", ncols=100):
    logging.info(f"Evaluating {model_cell_type} {model_training_sample} Model → {test_set_cell_type} {evaluation_sample} Test Set")

    dataset_split_type = "test"
    
    cell_type_cache_dir = DATA_DIR / f"{test_set_cell_type}_cache"
    stability_cache_dir = cell_type_cache_dir / f"{evaluation_sample}_stability_cache" / f"stability_{stability_number}"
    
    prediction_save_file = RESULT_DIR / "stability_grns" / f"{model_training_sample}_model_vs_{evaluation_sample}_test_grn_{subset_size}_stability_{stability_number}.csv"

    prediction_save_file.parent.mkdir(parents=True, exist_ok=True)

    if prediction_save_file.exists() and not force_reload:
        logging.info(f"Prediction file already exists for {model_cell_type} {model_training_sample} → {test_set_cell_type} {evaluation_sample}. Skipping evaluation.")
        sys.exit(0)
        
    # Load the TF and TG name to index mappings from the training cache metadata
    with open(stability_cache_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
        
    tf_name_to_idx = metadata["tf_name_to_idx"]
    tg_id_to_idx = metadata["tg_id_to_idx"]

    tf_idx_to_name, tg_idx_to_name = create_tf_tg_index_to_name_mappings(metadata)

    # Use the model retrained on subsample `stability_number` of the model's own
    # training sample. For the cross-trained comparison this is the other sample's
    # model at the same subsample number.
    stability_model_dir = CHKPT_DIR / "stability" / model_cell_type / model_training_sample / f"stability_{stability_number}"
    assert stability_model_dir.exists(), f"Stability checkpoint directory does not exist: {stability_model_dir}"

    tf_tg_model_chkpt = find_best_stability_checkpoint(stability_model_dir, verbose=True)

    prediction_df = run_prediction_vs_test_set(
        tf_tg_model_chkpt=tf_tg_model_chkpt,
        model_cell_type=model_cell_type,
        model_training_sample=model_training_sample,
        test_set_cell_type=test_set_cell_type,
        cell_type_cache_dir=cell_type_cache_dir,
        stability_cache_dir=stability_cache_dir,
        evaluation_sample=evaluation_sample,
        stability_number=stability_number,
        dataset_split_type=dataset_split_type,
        subset_size=subset_size,
        show_progress_bar=True,
        compile_model=False,
        batch_size=batch_size,
        tf_idx_to_name=tf_idx_to_name,
        tg_idx_to_name=tg_idx_to_name
    )
        
    prediction_df.to_csv(prediction_save_file, index=False)

    logging.info("Done!")