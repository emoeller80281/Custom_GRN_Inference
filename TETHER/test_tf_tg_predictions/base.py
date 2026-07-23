"""Base configuration and shared functionality for the TF-TG prediction analysis.

This package is a class-based refactor of ``test_tf_tg_predictions.ipynb``. The
notebook remains the canonical, unmodified exploratory version; this package
splits the same work into a parent class (:class:`TFTGBase`) that holds the
shared configuration (paths, formatting dictionaries, fonts, model checkpoints)
and the shared "main" functions, plus one child class per section of related
plots (see the other modules in this package).

Each child class subclasses :class:`TFTGBase` and is responsible for the data
generation, caching, and plotting of a single analysis section. Import-time work
here is limited to configuration setup (matching the notebook); heavy compute
only runs when a child class's ``generate_data``/``run`` method is called.

Module-level names (configuration constants, formatting dictionaries, path
objects, and shared functions) are kept identical to the notebook so the child
modules can import them directly. :class:`TFTGBase` re-exposes them as class
attributes / staticmethods so section classes can also reach them via ``self``.
"""

# ---------------------------------------------------------------------------
# The block below is copied verbatim from the notebook export (configuration
# dictionaries, path setup, model checkpoint discovery, and shared functions).
# ---------------------------------------------------------------------------


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
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker, HPacker, DrawingArea
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
import matplotlib as mpl
import matplotlib.font_manager as fm
import random

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    auc
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
DATA_DIR = PROJECT_DIR / "cached_data"
CHKPT_DIR = PROJECT_DIR / "checkpoints"
RESULT_DIR = PROJECT_DIR / "testing_results"

sys.path.append(str(PROJECT_DIR))

import models.tf_to_tg as tf_to_tg_module
import plotting_utils
import stat_utils
import utils
import warnings
import config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

font_path = PROJECT_DIR / "fonts" / "Arial.ttf"

fm.fontManager.addfont(font_path)
arial_font = fm.FontProperties(fname=font_path).get_name()

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [arial_font, "Helvetica", "Liberation Sans", "DejaVu Sans"]

logging.info(f"Using font: {arial_font}")

rng = np.random.default_rng()

OWN_MODEL_METHOD = "TF-TG Model (own test set)"
CROSS_MODEL_METHOD = "TF-TG Model (cross-trained)"

TFTG_MODEL_METHODS = [
    OWN_MODEL_METHOD,
    CROSS_MODEL_METHOD,
]

method_display_name_map = {
    OWN_MODEL_METHOD: "TF-TG Own",
    CROSS_MODEL_METHOD: "TF-TG Cross",
}

sample_order = [
    "E7.5_rep1",
    "E8.5_rep1",
    "buffer_1",
    "buffer_2",
    "sample_1",
    "hepatocytes_1",
    "hepatocytes_3",
]

sample_rename_map = {
    "E7.5_rep1": "mESC-1",
    "E8.5_rep1": "mESC-2",
    "buffer_1": "Macrophage-1",
    "buffer_2": "Macrophage-2",
    "sample_1": "K562",
    "hepatocytes_1": "Hepatocytes-1",
    "hepatocytes_3": "Hepatocytes-2",

    OWN_MODEL_METHOD: "MTGRN-STM", #"TF-TG Model\n(Own)",
    CROSS_MODEL_METHOD: "MTGRN-CTM", #"TF-TG Model\n(Cross)",
}

sample_color_map = {
    "E7.5_rep1": "#4195df",
    "E8.5_rep1": "#86C7E7",
    "buffer_1": "#EF767A",
    "buffer_2": "#F9C60D",
    "sample_1": "#EF9CFA",
    "hepatocytes_1": "#82EC32",
    "hepatocytes_3": "#F98637",
}

method_color_dict = {
    OWN_MODEL_METHOD: "#4195df",
    CROSS_MODEL_METHOD: "#86C7E7",
    "LINGER": "#EF767A",
    "CellOracle": "#F9C60D",
    "Pando": "#EF9CFA",
    "SCENIC+": "#82EC32",
    "FigR": "#FDA7BB",
    "GRaNIE": "#F98637",
}

# method_color_dict = {
#     OWN_MODEL_METHOD: "#1f78b4",
#     CROSS_MODEL_METHOD: "#a6cee3",
#     "LINGER": "#fb9a99",
#     "CellOracle": "#fdbf6f",
#     "Pando": "#cab2d6",
#     "SCENIC+": "#b2df8a",
#     "FigR": "#ffff99",
# }

org_dict = {
    "E7.5_rep1": ("mouse", "mESC"),
    "E8.5_rep1": ("mouse", "mESC"),
    "buffer_1": ("human", "Macrophage"),
    "buffer_2": ("human", "Macrophage"),
    "hepatocytes_1": ("mouse", "mouse_hepatocytes"),
    "hepatocytes_3": ("mouse", "mouse_hepatocytes"),
    "sample_1": ("human", "K562"),
    "WT_D13_rep1": ("human", "iPSC")
}

models_to_plot = [
    "E7.5_rep1",
    "E8.5_rep1",
    "buffer_1",
    "buffer_2",
    "sample_1",
    "hepatocytes_1",
    "hepatocytes_3",
]

tf_tg_input_cache_dir = DATA_DIR / "tf_tg_training_cache"

score_label_save_dir = RESULT_DIR / "model_generalizability" / "labeled_grns"
score_label_save_dir.mkdir(parents=True, exist_ok=True)

method_comparison_boxplot_dir = PROJECT_DIR / "plots" / "model_vs_other_method_boxplots"
method_comparison_boxplot_dir.mkdir(exist_ok=True, parents=True)

lift_boxplot_dir = method_comparison_boxplot_dir / "lift_boxplots"
lift_boxplot_dir.mkdir(parents=True, exist_ok=True)

model_vs_other_method_curve_dir = PROJECT_DIR / "plots" / "model_vs_other_method_auroc_auprc_curves"
model_vs_other_method_curve_dir.mkdir(exist_ok=True, parents=True)

generalizability_plot_dir = PROJECT_DIR / "plots" / "generalizability"
generalizability_plot_dir.mkdir(parents=True, exist_ok=True)

feature_ablation_plot_dir = PROJECT_DIR / "plots" / "feature_ablation"
feature_ablation_plot_dir.mkdir(parents=True, exist_ok=True)

tf_dna_plots = PROJECT_DIR / "plots" / "tf_dna_binding_plots"
tf_dna_plots.mkdir(parents=True, exist_ok=True)

roc_plot_dir = PROJECT_DIR / "plots" / "roc_curves"
roc_plot_dir.mkdir(parents=True, exist_ok=True)

auprc_plot_dir = PROJECT_DIR / "plots" / "auprc_plots"
auprc_plot_dir.mkdir(parents=True, exist_ok=True)

rank_plot_dir = PROJECT_DIR / "plots" / "rank_plots"
rank_plot_dir.mkdir(parents=True, exist_ok=True)

rank_bar_plot_dir = rank_plot_dir / f"rank_barplots"
rank_lollipop_plot_dir = rank_plot_dir / f"rank_lollipop_plots"
rank_heatmap_plot_dir = rank_plot_dir / f"rank_heatmaps"
rank_boxplot_plot_dir = rank_plot_dir / f"rank_boxplots"

rank_bar_plot_dir.mkdir(parents=True, exist_ok=True)
rank_lollipop_plot_dir.mkdir(parents=True, exist_ok=True)
rank_heatmap_plot_dir.mkdir(parents=True, exist_ok=True)
rank_boxplot_plot_dir.mkdir(parents=True, exist_ok=True)

grn_sizes_by_method_dir = PROJECT_DIR / "plots" / "grn_sizes_by_method"
grn_sizes_by_method_dir.mkdir(parents=True, exist_ok=True)

# File path for the full metrics DataFrame CSV
full_metric_df_path = RESULT_DIR / "full_method_comparison_metrics.csv"

# Plot showing how the model performs on its own test set, on other samples, on other cell types, and on other species
performance_across_evaluation_sets_plot_path = generalizability_plot_dir / "mean_performance_across_evaluation_sets.png"

# One subplot for each sample
auroc_models_vs_test_set_individual_plot_path = roc_plot_dir / "models_vs_own_test_set_auroc.png"

# Combined plot for all samples (one curve per sample)
auroc_models_vs_test_set_all_samples_plot_path = roc_plot_dir / "models_vs_own_test_set_auroc_combined.png"
auprc_models_vs_test_set_all_samples_plot_path = auprc_plot_dir / "models_vs_own_test_set_auprc_combined.png"
auprc_lift_models_vs_test_set_all_samples_plot_path = auprc_plot_dir / "models_vs_own_test_set_auprc_lift_combined.png"

# PRC curves for our model vs other methods, one figure per sample
model_vs_other_method_prc_curve_fig_dir = model_vs_other_method_curve_dir / "prc_curve_figs"
model_vs_other_method_prc_curve_fig_dir.mkdir(parents=True, exist_ok=True)

# ROC curves for our model vs other methods, one figure per sample
model_vs_other_method_roc_curve_fig_dir = model_vs_other_method_curve_dir / "roc_curve_figs"
model_vs_other_method_roc_curve_fig_dir.mkdir(parents=True, exist_ok=True)

tf_tg_model_checkpoints = {
    "mESC": {
        "E7.5_rep1": CHKPT_DIR / "mESC" / "E7.5_rep1" / "tf_tg_train_E7.5_rep1_3675131" / "epoch_11_best_model.ckpt",
        # "E7.5_rep1": utils.find_latest_checkpoint(CHKPT_DIR, "mESC", "E7.5_rep1"),
        "E7.5_rep2": utils.find_latest_checkpoint(CHKPT_DIR, "mESC", "E7.5_rep2"),
        "E8.5_rep1": utils.find_latest_checkpoint(CHKPT_DIR, "mESC", "E8.5_rep1", training_number="3691937"),
        "E8.5_rep2": utils.find_latest_checkpoint(CHKPT_DIR, "mESC", "E8.5_rep2", training_number="3691937"),
    },
    "iPSC": {
        "WT_D13_rep1": utils.find_latest_checkpoint(CHKPT_DIR, "iPSC", "WT_D13_rep1"),
    },
    "Macrophage": {
        "buffer_1": utils.find_latest_checkpoint(CHKPT_DIR, "Macrophage", "buffer_1", training_number="3685893"),
        "buffer_2": utils.find_latest_checkpoint(CHKPT_DIR, "Macrophage", "buffer_2", training_number="3713132"),
        "buffer_3": utils.find_latest_checkpoint(CHKPT_DIR, "Macrophage", "buffer_3"),
        "buffer_4": utils.find_latest_checkpoint(CHKPT_DIR, "Macrophage", "buffer_4"),
    },
    "K562": {
        "sample_1": utils.find_latest_checkpoint(CHKPT_DIR, "K562", "sample_1", training_number="3692409"),
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
    ) -> dict | None:
    
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
        tf_mask_tensor,
        compile_model=compile_model
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

    # print(f"Evaluating on {dataset_split_type} set")
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
    
def generate_model_predictions(
    model, 
    data_loader, 
    device, 
    tf_idx_to_name, 
    tg_idx_to_name,
    compile_model=False,
    ):
    pooling_mode = "lse"
    pooling_temperature = 1.0

    model = model.to(device)
    model.eval()
    
    if compile_model and device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")

    tf_indices_list = []
    tg_indices_list = []
    all_scores = []

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating", ncols=100):
            tf_indices = batch["tf_idx"].detach().cpu().numpy().ravel()
            tg_indices = batch["tg_idx"].detach().cpu().numpy().ravel()

            batch = tf_to_tg_module.move_batch_to_device(batch, device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                edge_logits, _ = model(
                    tf_embedding=batch["tf_embedding"],
                    tf_mask=batch["tf_mask"],
                    peak_sequences=batch["peak_sequences"],
                    peak_accessibility=batch["peak_accessibility"],
                    peak_distance=batch["peak_distance"],
                    tf_expression=batch["tf_expression"],
                    tg_expression=batch["tg_expression"],
                    peak_mask=batch.get("peak_mask", None),
                    cell_mask=batch["cell_mask"],
                    pooling_mode=pooling_mode,
                    pooling_temperature=pooling_temperature,
                )

            scores = torch.sigmoid(edge_logits.float())

            tf_indices_list.append(tf_indices)
            tg_indices_list.append(tg_indices)
            all_scores.append(scores.detach().cpu().numpy().ravel())

    all_tf_indices_flat = np.concatenate(tf_indices_list)
    all_tg_indices_flat = np.concatenate(tg_indices_list)
    all_scores_flat = np.concatenate(all_scores)

    tf_names = [tf_idx_to_name[int(idx)].upper() for idx in all_tf_indices_flat]
    tg_names = [tg_idx_to_name[int(idx)].upper() for idx in all_tg_indices_flat]

    prediction_df = pd.DataFrame({
        "Source": tf_names,
        "Target": tg_names,
        "Score": all_scores_flat,
    })

    prediction_df = (
        prediction_df.groupby(["Source", "Target"], as_index=False)["Score"]
        .median()
    )

    return prediction_df

def create_tf_tg_index_to_name_mappings(metadata):
    tf_idx_to_name = {idx: name for name, idx in metadata["tf_name_to_idx"].items()}
    tg_idx_to_name = {idx: name for name, idx in metadata["tg_id_to_idx"].items()}
    return tf_idx_to_name, tg_idx_to_name

def create_tf_tg_label_df(tftg_inputs_test):
    # Create the TF-TG label DataFrame
    tftg_inputs_test.keys()
    test_tf_input = tftg_inputs_test["tf_name"]
    test_tg_input = tftg_inputs_test["tg_name"]
    test_labels = tftg_inputs_test["label"]

    # create TF-TG label DataFrame
    tf_tg_label_df = pd.DataFrame({
        "Source": test_tf_input,
        "Target": test_tg_input,
        "Label": test_labels,
    })

    tf_tg_label_df = tf_tg_label_df.drop_duplicates(["Source", "Target"])

    gt_df: pd.DataFrame = tf_tg_label_df[tf_tg_label_df["Label"] == 1] 
    gt_tfs = set(gt_df["Source"].str.upper().unique()) 
    gt_targets = set(gt_df["Target"].str.upper().unique()) 
    gt_pairs = set(gt_df["Source"].str.upper() + "\t" + gt_df["Target"].str.upper())
    
    return tf_tg_label_df, gt_pairs, gt_tfs, gt_targets

def load_and_standardize_method(name: str, info: dict) -> pd.DataFrame:
    """
    Load a GRN CSV and rename tf_col/target_col/score_col -> Source/Target/Score.
    Extra columns are preserved.
    """
    if info["path"].suffix == ".tsv":
        sep = "\t"
    elif info["path"].suffix == ".csv":
        sep = ","
    
    df = pd.read_csv(info["path"], sep=sep, header=0, index_col=None)

    tf_col     = info["tf_col"]
    target_col = info["target_col"]
    score_col  = info["score_col"]

    sample_rename_map = {
        tf_col: "Source",
        target_col: "Target",
        score_col: "Score",
    }

    missing = [c for c in sample_rename_map if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing expected columns: {missing}. Got: {list(df.columns)}")

    df = df.rename(columns=sample_rename_map)

    df = df[["Source", "Target", "Score"]]
    df["Source"] = df["Source"].astype(str).str.upper()
    df["Target"] = df["Target"].astype(str).str.upper()
    
    df["Score"] = np.abs(df["Score"])

    return df

def precision_at_k(y_true, y_score, k=1_000):
    """Calculate precision among the k highest-scoring predictions."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()

    if len(y_true) != len(y_score):
        raise ValueError(
            f"y_true and y_score have different lengths: "
            f"{len(y_true)} and {len(y_score)}"
        )

    if len(y_true) == 0:
        return np.nan

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    effective_k = min(k, len(y_true))

    top_k_indices = np.argsort(-y_score, kind="stable")[:effective_k]
    top_k_labels = y_true[top_k_indices]

    return float(top_k_labels.mean())

def compute_metrics(method_name: str, sample_name: str, df: pd.DataFrame, gt_pairs: set, score_threshold: float):
    if len(df) == 0:
        labels = np.array([], dtype=int)
        scores = np.array([], dtype=float)
    else:
        labels = np.asarray(
            [1 if pair in gt_pairs else 0 for pair in df["Source"] + "\t" + df["Target"]]
        ).astype(int).ravel()
        scores = np.asarray(df["Score"].tolist()).astype(float).ravel()

    n_edges = len(labels)
    has_both_classes = n_edges > 0 and len(np.unique(labels)) >= 2

    if n_edges == 0:
        early_precision = accuracy = precision = recall = f1 = np.nan
        auroc = random_auroc = np.nan
    else:
        preds = (scores >= score_threshold).astype(int)

        early_precision = precision_at_k(labels, scores, k=10_000)
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)

        if has_both_classes:
            auroc = roc_auc_score(labels, scores)
            random_auroc = roc_auc_score(labels, np.random.rand(n_edges))
        else:
            auroc = np.nan
            random_auroc = np.nan

    return pd.DataFrame([{
        "method_name": method_name,
        "sample_name": sample_name,
        "auroc": auroc,
        "rand_auroc": random_auroc,
        "early_precision": early_precision,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_edges": n_edges,
        "n_pos": int(labels.sum()),
        "n_neg": int((labels == 0).sum()),
        "score_threshold": score_threshold,
    }])

def load_auprc_grns_all_methods(sample_list=None):
    auprc_all_method_dfs = {}

    if sample_list is not None:
        sample_list = set(sample_list)

    method_name_from_safe = {
        "celloracle": "CellOracle",
        "scenic": "SCENIC+",
        "linger": "LINGER",
        "pando": "Pando",
        "figr": "FigR",
        "tf-tg model (own test set)": OWN_MODEL_METHOD,
        "tf-tg model (cross-trained)": CROSS_MODEL_METHOD,
    }

    for cell_type_dir in (RESULT_DIR / "labeled_auprc_grns").iterdir():
        if not cell_type_dir.is_dir():
            continue

        for sample_dir in cell_type_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            if sample_list is not None and sample_dir.name not in sample_list:
                continue

            # Create nested dict before assigning methods
            auprc_all_method_dfs.setdefault(sample_dir.name, {})

            for method_file in sample_dir.glob("*.tsv"):
                method_name_safe = method_file.stem

                if method_name_safe not in method_name_from_safe:
                    logging.warning(
                        f"Unrecognized method name: {method_name_safe}. Using as-is."
                    )

                method_name = method_name_from_safe.get(method_name_safe, method_name_safe)

                auprc_df = pd.read_csv(method_file, sep="\t")
                auprc_all_method_dfs[sample_dir.name][method_name] = auprc_df

    return auprc_all_method_dfs

def load_generalizability_df(model_sample_name, test_set_sample_name, sample_size=10000):
    score_label_save_dir = RESULT_DIR / "model_generalizability" / "labeled_grns"
    
    score_label_file = (
        score_label_save_dir
        / f"{model_sample_name}_model_vs_{test_set_sample_name}_grn_{sample_size}.csv"
    )
    
    if not score_label_file.exists():
        logging.warning(f"Score-label file not found for {model_sample_name} vs {test_set_sample_name}. Skipping.")
        return None
    
    score_label_df = pd.read_csv(score_label_file)

    return score_label_df

def interpolate_precision_at_recall(source_rec, source_prec, target_rec):
    source_rec = np.asarray(source_rec, dtype=float)
    source_prec = np.asarray(source_prec, dtype=float)
    target_rec = np.asarray(target_rec, dtype=float)

    valid = np.isfinite(source_rec) & np.isfinite(source_prec)
    source_rec = source_rec[valid]
    source_prec = source_prec[valid]

    # np.interp expects ascending x values
    order = np.argsort(source_rec)
    source_rec = source_rec[order]
    source_prec = source_prec[order]

    # Collapse duplicate recall values
    interp_df = (
        pd.DataFrame({
            "recall": source_rec,
            "precision": source_prec,
        })
        .groupby("recall", as_index=False)["precision"]
        .mean()
    )

    return np.interp(
        target_rec,
        interp_df["recall"].to_numpy(),
        interp_df["precision"].to_numpy(),
    )

try:  # pragma: no cover - convenience for notebook parity
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print



class TFTGBase:
    """Parent class exposing shared configuration and "main" functions.

    Child classes (one per analysis section) subclass this to reach the
    configuration via ``self`` and to reuse the shared functions. The
    attributes/staticmethods below simply reference the module-level
    definitions above so there is a single source of truth.
    """

    # -- Configuration (paths, formatting dictionaries, fonts, checkpoints) --
    PROJECT_DIR = PROJECT_DIR
    DATA_DIR = DATA_DIR
    CHKPT_DIR = CHKPT_DIR
    RESULT_DIR = RESULT_DIR
    font_path = font_path
    arial_font = arial_font
    rng = rng
    OWN_MODEL_METHOD = OWN_MODEL_METHOD
    CROSS_MODEL_METHOD = CROSS_MODEL_METHOD
    TFTG_MODEL_METHODS = TFTG_MODEL_METHODS
    method_display_name_map = method_display_name_map
    sample_order = sample_order
    sample_rename_map = sample_rename_map
    sample_color_map = sample_color_map
    method_color_dict = method_color_dict
    org_dict = org_dict
    models_to_plot = models_to_plot
    tf_tg_input_cache_dir = tf_tg_input_cache_dir
    score_label_save_dir = score_label_save_dir
    method_comparison_boxplot_dir = method_comparison_boxplot_dir
    lift_boxplot_dir = lift_boxplot_dir
    model_vs_other_method_curve_dir = model_vs_other_method_curve_dir
    generalizability_plot_dir = generalizability_plot_dir
    feature_ablation_plot_dir = feature_ablation_plot_dir
    tf_dna_plots = tf_dna_plots
    roc_plot_dir = roc_plot_dir
    auprc_plot_dir = auprc_plot_dir
    rank_plot_dir = rank_plot_dir
    rank_bar_plot_dir = rank_bar_plot_dir
    rank_lollipop_plot_dir = rank_lollipop_plot_dir
    rank_heatmap_plot_dir = rank_heatmap_plot_dir
    rank_boxplot_plot_dir = rank_boxplot_plot_dir
    grn_sizes_by_method_dir = grn_sizes_by_method_dir
    full_metric_df_path = full_metric_df_path
    performance_across_evaluation_sets_plot_path = performance_across_evaluation_sets_plot_path
    auroc_models_vs_test_set_individual_plot_path = auroc_models_vs_test_set_individual_plot_path
    auroc_models_vs_test_set_all_samples_plot_path = auroc_models_vs_test_set_all_samples_plot_path
    auprc_models_vs_test_set_all_samples_plot_path = auprc_models_vs_test_set_all_samples_plot_path
    auprc_lift_models_vs_test_set_all_samples_plot_path = auprc_lift_models_vs_test_set_all_samples_plot_path
    model_vs_other_method_prc_curve_fig_dir = model_vs_other_method_prc_curve_fig_dir
    model_vs_other_method_roc_curve_fig_dir = model_vs_other_method_roc_curve_fig_dir
    tf_tg_model_checkpoints = tf_tg_model_checkpoints

    # -- Shared "main" functions --
    run_prediction_vs_test_set = staticmethod(run_prediction_vs_test_set)
    generate_model_predictions = staticmethod(generate_model_predictions)
    create_tf_tg_index_to_name_mappings = staticmethod(create_tf_tg_index_to_name_mappings)
    create_tf_tg_label_df = staticmethod(create_tf_tg_label_df)
    load_and_standardize_method = staticmethod(load_and_standardize_method)
    precision_at_k = staticmethod(precision_at_k)
    compute_metrics = staticmethod(compute_metrics)
    load_auprc_grns_all_methods = staticmethod(load_auprc_grns_all_methods)
    load_generalizability_df = staticmethod(load_generalizability_df)
    interpolate_precision_at_recall = staticmethod(interpolate_precision_at_recall)
