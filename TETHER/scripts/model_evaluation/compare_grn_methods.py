"""Compare the cell-type TF-TG models against other GRN inference methods.

Standalone version of the "Cross-trained model vs other inference methods" section of
celltype_model_evaluation.ipynb (cells 17-30), so the whole pipeline can run without the
notebook after a change:

    03e_train_holdout_celltype_tf_to_tg_model.sh   (train the holdout models)
    run_generate_grns_from_model.sh                (test_set_grn.csv + <sample>_cross_grn.csv)
    run_compare_grn_methods.sh                     (this script)

Steps, as in the notebook:
  1. Load each model run's self-trained (test set) and cross-trained (holdout sample) GRNs.
  2. Label every method's GRN against the cell-type ground truth, per (sample, cell type).
     Cached as <results_dir>/all_method_grn_dict_<hash>.pkl (same file as the notebook).
  3. Metrics per (model, sample, cell type, method), and shuffled-score random baselines.
  4. Boxplots, random-baseline boxplots, per-cell-type heatmaps and average-rank plots.

The hash covers the samples, cell types, other-method names and model run names, not the GRN
file contents. Use --force_relabel after regenerating the GRNs of a run with the same name.

Example (from the repository root, on a compute node):
    python TETHER/scripts/model_evaluation/compare_grn_methods.py \\
        --runs no_liver=celltype_joint_3926071_* no_mesc=celltype_joint_3926072_* ... \\
        --plot_dir TETHER/new_plots/method_comparison_<tag>
"""
import argparse
import hashlib
import json
import logging
import math
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from scipy.stats import hypergeom
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
sys.path.append(str(PROJECT_DIR))

DATA_DIR = PROJECT_DIR.parent / "data"
CHKPT_DIR = PROJECT_DIR / "checkpoints"

SELF_MODEL_METHOD = "TF-TG Model (self-trained)"
CROSS_MODEL_METHOD = "TF-TG Model (cross-trained)"
TFTG_MODEL_METHODS = {SELF_MODEL_METHOD, CROSS_MODEL_METHOD}

# Model name of the GRN with the mean edge score over all models in the run list
MEAN_MODEL = "mean"

# Holdout models trained 2026-10-22 with TF-DNA model mm10_3925194_epoch_21. Entries can be
# glob patterns, because a run name ends in its start time.
DEFAULT_MODEL_RUNS = {
    "no_liver": "celltype_joint_3926071_20261022_211905_753626",
    "no_mesc": "celltype_joint_3926072_20261022_211905_753629",
    "no_brain": "celltype_joint_3926073_20261022_211905_757179",
    "no_skin": "celltype_joint_3926074_20261022_211905_757180",
    "no_kidney": "celltype_joint_3926075_20261022_211905_757182",
    "no_hsc": "celltype_joint_3926076_20261022_211905_757180",
}

method_display_name_map = {
    SELF_MODEL_METHOD: "MONICA-STM",
    CROSS_MODEL_METHOD: "MONICA-CTM",
    "SCENIC_PLUS": "SCENIC+",
}

method_color_dict = {
    SELF_MODEL_METHOD: "#4195df",
    CROSS_MODEL_METHOD: "#86C7E7",
    "LINGER": "#EF767A",
    "CellOracle": "#F9C60D",
    "Pando": "#EF9CFA",
    "SCENIC_PLUS": "#82EC32",
    "FigR": "#FDA7BB",
    "GRaNIE": "#F98637",
}

UNUSED_TISSUE_DIRS = {"GSE246464_HSC_10x_raw"}

# Runs trained before the 2026-10 rename still name both brain samples "brain".
RENAMED_SAMPLES = {
    ("GSE140203_shareseq", "brain"): "shareseq_brain",
    ("10x_E18_mouse_brain", "brain"): "E18_brain",
}

# Other-method GRN files use the renamed sample names, so no overrides are needed.
OTHER_METHOD_SAMPLE = {}

OTHER_METHOD_GRN_DIR = Path(
    "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/formatted_GRNs_celltype"
)

# Test-set chromosomes held out from training. These must match
# build_tf_to_tg_celltype_data.py, which hard-codes the split per species.
TEST_CHROMS = {"mm10": ["18", "19"], "hg38": ["20", "21", "22"]}
DEFAULT_GENE_REF_FILES = {
    "mm10": "Mus_musculus.GRCm39.115.gtf.gz",
    "hg38": "Homo_sapiens.GRCh38.113.gtf.gz",
}

# Metrics that get a random baseline. Each one depends on how the edges are ranked.
RANDOM_BASELINE_METRICS = [
    "auroc",
    "auprc",
    "early_precision",
    "early_precision_rate",
    "accuracy",
    "precision",
    "recall",
    "fpr",
    "f1",
    "jaccard_index",
    "perint",
    "predictable_tfs",
    "predictable_tf_fraction",
]

# Shuffles per (sample, cell type, method). The baseline is the mean over the shuffles.
N_RANDOM_SHUFFLES = 10

# Metric names on the boxplot axes and titles. Precision, recall, and FPR are
# calculated on the top-edge set A, so they are "Early" metrics.
boxplot_metric_labels = {
    "auroc": "ROC",
    "auprc": "PRC",
    "early_precision": "Early Precision",
    "precision": "Early Precision",
    "f1": "Early F1",
    "recall": "Early Recall",
    "fpr": "Early FPR",
}

# Metrics in the random-baseline similarity table
BOXPLOT_METRICS = [
    "auroc",
    "auprc",
    "accuracy",
    "early_precision",
    "early_precision_rate",
    "precision",
    "recall",
    "fpr",
    "f1",
    "perint",
    "predictable_tf_fraction",
]

# The methods' random baselines for a metric are similar when their range is at most
# this fraction of the metric scale (1 for metrics in [0, 1]). The boxplots then show
# one line at the mean baseline over the methods, else one line over each method's box.
BASELINE_SIMILARITY_TOLERANCE = 0.075

RANDOM_BASELINE_STYLE = {"color": "0.35", "linestyle": (0, (4, 2)), "linewidth": 1.5}
RANDOM_BASELINE_HANDLE = Line2D([0], [0], label="Random baseline", **RANDOM_BASELINE_STYLE)

sample_tissue_map = {
    "E7.5_rep1": "Embryo",
    "E7.5_rep2": "Embryo",
    "E8.0_rep1": "Embryo",
    "E8.0_rep2": "Embryo",
    "E8.5_rep1": "Embryo",
    "E8.5_rep2": "Embryo",
    "Ctrl_4weeks_1": "Kidney",
    "Ctrl_4weeks_2": "Kidney",
    "Ctrl_6months_1": "Kidney",
    "skin_late_anagen": "Skin",
    "shareseq_brain": "Brain",
    "E18_brain": "Brain",
    "young_rep1": "HSC",
    "young_rep2": "HSC",
    "old_rep1": "HSC",
    "old_rep2": "HSC",
    "liver_sample": "Liver",
}

# Heatmap rows: grouped by which TF-TG model scored them, or sorted by mean metric
GROUP_BY_SELF_VS_EVAL = False


# ======================================================================
# Load and label the GRNs
# ======================================================================

def load_run_config(model_checkpoint: Path):
    run_dir = model_checkpoint.parent.parent
    with open(run_dir / "run_config.json") as handle:
        return json.load(handle)


def load_celltype_ground_truth(gt_file_index, celltype):
    celltype_file_name = celltype.replace(" ", "_")

    gt_filename = f"{celltype_file_name}_ground_truth.parquet"
    gt_path = gt_file_index.get(gt_filename)

    if gt_path is None:
        print(f"No ground truth found for {celltype}")
        return None

    # Load and prepare this cell type's GT once.
    # Note: GT is already filtered to TFs with embeddings
    gt = pd.read_parquet(
        gt_path,
        columns=["Source", "Target"],
    )

    gt = (
        gt.assign(
            Source=gt["Source"].astype(str).str.upper(),
            Target=gt["Target"].astype(str).str.upper(),
        )
        .drop_duplicates(["Source", "Target"])
    )

    return gt


def normalize_pairs(df):
    """
    Normalize the Source and Target columns in the DataFrame.
    """
    df = df.dropna(subset=["Source", "Target"]).copy()
    for col in ["Source", "Target"]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    return df


def generate_edge_universe_from_gt(gt):
    """
    Generate a universe of all possible edges from the ground truth.
    """

    gt = normalize_pairs(gt).drop_duplicates(["Source", "Target"])

    universe = pd.MultiIndex.from_product(
        [
            sorted(gt["Source"].unique()),
            sorted(gt["Target"].unique()),
        ],
        names=["Source", "Target"],
    ).to_frame(index=False)

    positive_pairs = pd.MultiIndex.from_frame(gt[["Source", "Target"]])
    candidate_pairs = pd.MultiIndex.from_frame(universe[["Source", "Target"]])
    universe["_in_gt"] = candidate_pairs.isin(positive_pairs).astype("int8")
    return universe


def sample_auprc_10x_negatives(full_universe, random_state=42):
    """Keep all positives, sample negatives at 10x the positive count."""
    positives = full_universe[full_universe["_in_gt"] == 1]
    negatives = full_universe[full_universe["_in_gt"] == 0]

    n_pos = len(positives)
    n_neg_sample = min(n_pos * 10, len(negatives))

    if n_pos == 0 or n_neg_sample == 0:
        return full_universe.iloc[0:0].copy()

    neg_sampled = negatives.sample(
        n=n_neg_sample,
        replace=False,
        random_state=random_state,
    )

    auprc_df = pd.concat([positives, neg_sampled], axis=0)
    auprc_df = auprc_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return auprc_df


def filter_df_to_gene_set(df, gene_set1, gene_set2):
    df = df.copy()
    df = df[
        df["Source"].isin(gene_set1) & df["Target"].isin(gene_set2)
    ]
    return df


def label_grn_with_universe(grn, universe, missing_score=np.nan):
    predictions = normalize_pairs(grn[["Source", "Target", "Score"]])
    # Absolute scores: signed methods (e.g. regression coefficients) rank by strength.
    predictions["Score"] = pd.to_numeric(predictions["Score"], errors="coerce").abs()
    predictions = predictions.loc[np.isfinite(predictions["Score"])].copy()

    # Keep the maximum score for duplicate TF-TG pairs.
    predictions = predictions.groupby(["Source", "Target"], as_index=False)["Score"].max()

    labeled = universe[["Source", "Target", "_in_gt"]].merge(
        predictions,
        on=["Source", "Target"],
        how="left",
        validate="one_to_one",
    )
    labeled["Score"] = labeled["Score"].fillna(missing_score)
    labeled = labeled.dropna(subset=["Score"])

    return labeled


def generate_hash_key(sample_celltype_df, other_method_names, model_run_dict):
    # Create a hash of the sample, cell type, and method lists
    rows = [
        {
            "sample": str(sample),
            "eval_type": str(eval_type),
            "celltypes": sorted(set(map(str, celltypes))),
            "model_names": sorted(model_run_dict.values())
        }
        for sample, eval_type, celltypes
        in sample_celltype_df.itertuples(index=False, name=None)
    ]

    payload = {
        "rows": sorted(rows, key=lambda row: json.dumps(row, sort_keys=True)),
        "other_methods": sorted(set(map(str, other_method_names))),
        # The cache holds one labeled GRN per model and one for the mean over the models.
        "model_grn_keys": "per_model_and_mean",
    }

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    hash_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]

    return hash_key


def read_sample_genes(data_path):
    """Read only the RNA gene names from the .h5mu, not the expression matrix."""
    with h5py.File(data_path, "r") as f:
        var = f["mod/rna/var"]
        names = var[var.attrs["_index"]][:]
    return {(n.decode() if isinstance(n, bytes) else str(n)).upper() for n in names}


def label_one_celltype(sample_name, celltype, sample_genes, model_grns,
                       gt_file_index, other_method_grn_dir, other_method_names,
                       other_method_sample):
    """Load, filter, and label every method's GRN for one (sample, cell type).

    other_method_sample is the sample name in the other-method GRN file names,
    or None when the other methods were not run on this sample.
    """
    messages, celltype_results = [], {}

    celltype_gt = load_celltype_ground_truth(gt_file_index, celltype)
    if celltype_gt is None:
        messages.append(f"{sample_name} / {celltype}: no ground truth found")
        return sample_name, celltype, None, celltype_results, messages

    celltype_gt = filter_df_to_gene_set(celltype_gt, sample_genes, sample_genes)
    if celltype_gt.empty:
        messages.append(f"{sample_name} / {celltype}: ground truth has no edges in sample; skipping")
        return sample_name, celltype, None, celltype_results, messages
    celltype_gt = normalize_pairs(celltype_gt).drop_duplicates(["Source", "Target"])

    full_universe = generate_edge_universe_from_gt(celltype_gt)
    auprc_eval_universe = sample_auprc_10x_negatives(full_universe)
    if auprc_eval_universe["_in_gt"].nunique() < 2:
        messages.append(f"{sample_name} / {celltype}: AUPRC universe lacks positives or negatives")

    # model_grns is keyed by (model name, eval type). MEAN_MODEL is the mean score over the models.
    for model_key, model_grn in model_grns.items():
        model_grn = filter_df_to_gene_set(model_grn, sample_genes, sample_genes)
        celltype_results[model_key] = {
            "full_universe": label_grn_with_universe(model_grn, full_universe),
            "auprc_universe": label_grn_with_universe(model_grn, auprc_eval_universe),
        }

    if other_method_sample is None:
        messages.append(f"{sample_name} / {celltype}: no other-method GRNs for this sample")
        return sample_name, celltype, celltype_gt, celltype_results, messages

    for other_method_name in other_method_names:
        celltype_results[other_method_name] = {}
        grn_filename = f"{other_method_name.lower().replace('_', '')}_{celltype.replace(' ', '_')}_{other_method_sample}.tsv"
        grn_path = other_method_grn_dir / other_method_name / grn_filename
        if not grn_path.exists():
            messages.append(f"{sample_name} / {celltype}: no {other_method_name} GRN found")
            continue

        grn = pd.read_csv(grn_path, sep="\t", usecols=["Source", "Target", "Score"], engine="pyarrow")

        full = label_grn_with_universe(grn, full_universe)
        full["Score"] = (full["Score"] - full["Score"].min()) / (full["Score"].max() - full["Score"].min())
        celltype_results[other_method_name] = {
            "full_universe": full,
            "auprc_universe": label_grn_with_universe(grn, auprc_eval_universe),
        }

    return sample_name, celltype, celltype_gt, celltype_results, messages


def load_grns_from_model_run(model_checkpoint, uid_of):
    # Use the GRNs written by generate_grns_from_model.py: the test set GRN of the
    # training samples, and one cross GRN for each holdout sample of the run.
    # uid_of maps (tissue, sample) to the unique SampleID used here.
    from scripts.generate_grns_from_model import holdout_grn_paths

    # Load the run config for the model
    model_run_dir = model_checkpoint.parent.parent
    run_config = load_run_config(model_checkpoint)

    # Determine the holdout GRN files for this model run
    holdout_specs = [
        tuple(spec.split(":", 1)) for spec in run_config.get("holdout_sample", [])
    ]
    holdout_grn_files = holdout_grn_paths(model_run_dir, holdout_specs)

    test_set_grn_file = model_run_dir / "test_set_grn.csv"
    missing_grn_files = [
        path for path in [test_set_grn_file, *holdout_grn_files.values()] if not path.is_file()
    ]
    if missing_grn_files:
        raise FileNotFoundError(
            f"Run generate_grns_from_model.py --run_dir {model_run_dir.name} first. Missing: "
            + ", ".join(path.name for path in missing_grn_files)
        )

    # Load the test set GRN
    self_trained_grn = pd.read_csv(test_set_grn_file, engine="pyarrow")

    # Remove any edges involving holdout cell types from the self-trained GRN
    holdout_cts = set(run_config.get("holdout_celltype", []))
    self_trained_grn = self_trained_grn[~self_trained_grn["CellType"].isin(holdout_cts)].copy()

    # Load the cross-trained GRNs for each holdout sample
    cross_grns = []
    for (tissue, sample), path in holdout_grn_files.items():
        grn = pd.read_csv(path, engine="pyarrow")
        grn["SampleID"] = uid_of[(tissue, RENAMED_SAMPLES.get((tissue, sample), sample))]
        cross_grns.append(grn)

    # Add the sample IDs to the self-trained GRN based on the tissue and sample mapping
    self_trained_grn["SampleID"] = [
        uid_of[(t, RENAMED_SAMPLES.get((t, s), s))]
        for t, s in zip(self_trained_grn["Tissue"], self_trained_grn["SampleID"])
    ]
    self_trained_grn = self_trained_grn.drop(columns="Tissue")

    # Concatenate the cross-trained GRNs for each holdout sample
    evaluation_grn = (
        pd.concat(cross_grns, ignore_index=True) if cross_grns
        else self_trained_grn.iloc[0:0].copy()
    )

    logging.info(
        f"{model_run_dir.name}: self-trained GRN {len(self_trained_grn):,} edges from "
        f"{sorted(self_trained_grn['SampleID'].unique())}; cross-trained GRN "
        f"{len(evaluation_grn):,} edges from {sorted(evaluation_grn['SampleID'].unique())}"
    )

    return self_trained_grn, evaluation_grn


def mean_over_models(grn):
    """One row per (sample, cell type, edge), with the mean score of the models that predicted it."""
    return grn.groupby(
        ["SampleID", "CellType", "Source", "Target"], as_index=False, observed=True, sort=False,
    )["Score"].mean()


def find_samples(data_dir):
    """
    Map each processed sample to a unique ID. A sample keeps its own name unless another
    tissue has a sample with the same name; then it becomes <tissue>__<sample>, as in
    holdout_grn_paths.
    """
    found = [
        (tissue_dir.name, path.parent.name, path)
        for tissue_dir in (data_dir / "processed").iterdir()
        if tissue_dir.name not in UNUSED_TISSUE_DIRS
        for path in tissue_dir.rglob("*/multiome_processed.h5mu")
    ]

    name_counts = Counter(sample for _, sample, _ in found)

    sample_sources = {}   # unique ID -> where the sample's data is
    for tissue, sample, path in found:
        uid = sample if name_counts[sample] == 1 else f"{tissue}__{sample}"
        sample_sources[uid] = {"tissue": tissue, "sample": sample, "h5mu": path}

    # Create a (tissue, sample) map to the unique sample ID for each sample
    uid_of = {(src["tissue"], src["sample"]): uid for uid, src in sample_sources.items()}
    return sample_sources, uid_of


def resolve_runs(run_specs):
    """{model name: run directory name}; each value may be a glob that matches one run directory."""
    runs = {}
    for model_name, pattern in run_specs.items():
        matches = sorted(p for p in (CHKPT_DIR / "celltype_tf_tg").glob(pattern) if p.is_dir())
        if len(matches) != 1:
            raise ValueError(f"{model_name}={pattern} matches {len(matches)} run directories, expected one: "
                             f"{[m.name for m in matches]}")
        runs[model_name] = matches[0].name
    return runs


def label_method_grns(model_run_dict, sample_sources, uid_of, species, results_dir,
                      other_method_grn_dir, n_jobs, force_relabel=False):
    """
    Labeled GRN of every model and method per (sample, cell type), and the cell-type
    ground truths, cached under results_dir. Returns (method_grn_dict, celltype_gt_dict, hash_key).
    """
    # Load and concatenate the self-trained and cross-trained GRNs for each model
    model_self_grn_list = []
    model_cross_grn_list = []
    for model_name, run_name in model_run_dict.items():
        model_checkpoint = CHKPT_DIR / "celltype_tf_tg" / run_name / "checkpoints" / "last.ckpt"
        model_self_grn, model_cross_grn = load_grns_from_model_run(model_checkpoint, uid_of)

        model_self_grn["ModelName"] = model_name
        model_cross_grn["ModelName"] = model_name

        model_self_grn_list.append(model_self_grn)
        model_cross_grn_list.append(model_cross_grn)

    self_trained_grn = pd.concat(model_self_grn_list, ignore_index=True)
    evaluation_grn = pd.concat(model_cross_grn_list, ignore_index=True)

    self_trained_grn["EvalType"] = SELF_MODEL_METHOD
    evaluation_grn["EvalType"] = CROSS_MODEL_METHOD

    # Combine the self-trained and cross-trained GRNs into a single DataFrame
    sample_celltype_df = pd.concat([
        evaluation_grn[["EvalType", "SampleID", "CellType"]].drop_duplicates(),
        self_trained_grn[["EvalType", "SampleID", "CellType"]].drop_duplicates()
    ])

    sample_celltype_df = sample_celltype_df.groupby(["SampleID", "EvalType"])["CellType"].apply(list).reset_index()

    other_method_dirs = sorted(
        path for path in other_method_grn_dir.iterdir()
        if path.is_dir()
    )
    other_method_names = [method_dir.name for method_dir in other_method_dirs]
    logging.info(f"Other methods: {other_method_names}")

    species_gt_dir = DATA_DIR / "ground_truth_files" / "cell_type_specific" / species
    gt_file_index = {path.name: path for path in species_gt_dir.rglob("*_ground_truth.parquet")}

    hash_key = generate_hash_key(sample_celltype_df, other_method_names, model_run_dict)

    # Cache the method GRN dictionary and cell type ground truth dictionary in the
    # results directory. This allows for caching dictionaries with multiple models.
    method_dict_file = results_dir / f"all_method_grn_dict_{hash_key}.pkl"

    if method_dict_file.is_file() and not force_relabel:
        try:
            with open(method_dict_file, "rb") as f:
                cached = pickle.load(f)
            logging.info(f"Loaded labeled GRNs from {method_dict_file}")
            return cached["method_grn_dict"], cached["celltype_gt_dict"], hash_key
        except Exception as e:
            logging.warning(f"Error loading cache: {e}")
            os.remove(method_dict_file)

    logging.info(f"Labeling the GRNs (cache {method_dict_file.name})...")

    # Split each model GRN once, so each worker receives only its own slice.
    # Each (sample, cell type) gets the GRN of every model that predicted it,
    # keyed (model name, eval type), plus the mean score over those models.
    model_slices = {}
    for eval_type, grn in {SELF_MODEL_METHOD: self_trained_grn, CROSS_MODEL_METHOD: evaluation_grn}.items():
        for (sample, celltype), sub in mean_over_models(grn).groupby(["SampleID", "CellType"], sort=False, observed=True):
            model_slices.setdefault((sample, celltype), {})[(MEAN_MODEL, eval_type)] = sub
        for (model_name, sample, celltype), sub in grn.groupby(["ModelName", "SampleID", "CellType"], sort=False, observed=True):
            model_slices.setdefault((sample, celltype), {})[(model_name, eval_type)] = sub

    # Read each sample's gene names once.
    sample_genes_dict = {}
    for sample_name in {sample for sample, _ in model_slices}:
        assert sample_name in sample_sources, f"No processed data for sample: {sample_name}"
        sample_genes_dict[sample_name] = read_sample_genes(sample_sources[sample_name]["h5mu"])
        logging.info(f"{sample_name}: {len(sample_genes_dict[sample_name])} genes")

    outputs = Parallel(n_jobs=min(len(model_slices), n_jobs), backend="loky", verbose=5)(
        delayed(label_one_celltype)(
            sample_name, celltype, sample_genes_dict[sample_name], model_grns,
            gt_file_index, other_method_grn_dir, other_method_names,
            OTHER_METHOD_SAMPLE.get(sample_name, sample_sources[sample_name]["sample"]),
        )
        for (sample_name, celltype), model_grns in model_slices.items()
    )

    method_grn_dict, celltype_gt_dict = {}, {}
    for sample_name, celltype, gt, results, messages in outputs:
        for message in messages:
            logging.warning(message)
        method_grn_dict.setdefault(sample_name, {})[celltype] = results
        if gt is not None:
            celltype_gt_dict.setdefault(sample_name, {})[celltype] = gt

    with open(method_dict_file, "wb") as f:
        pickle.dump({"method_grn_dict": method_grn_dict, "celltype_gt_dict": celltype_gt_dict}, f)

    return method_grn_dict, celltype_gt_dict, hash_key


# ======================================================================
# Metrics
# ======================================================================

def _safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p-values without statsmodels."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.array([], dtype=float)

    order = np.argsort(p_values, kind="stable")
    ranked_p = p_values[order]
    ranks = np.arange(1, len(ranked_p) + 1, dtype=float)

    ranked_q = ranked_p * len(ranked_p) / ranks
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0.0, 1.0)

    q_values = np.empty_like(ranked_q)
    q_values[order] = ranked_q
    return q_values


def _pr_auc_integral(labels, scores):
    """
    AUPRC with continuous interpolation between PR points.

    Matches PRROC ``pr.curve(...)$auc.integral`` (checked against PRROC 1.4,
    including heavily tied scores). Between two PR points, FP grows linearly
    with TP, so precision(x) = x / (a*x + b) and each segment integrates exactly.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = labels.sum()
    if n_pos == 0 or n_pos == len(labels):
        return np.nan

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # Cumulative TP/FP at the end of each group of tied scores.
    last_of_tie = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tp = np.r_[0, np.cumsum(sorted_labels)[last_of_tie]].astype(float)
    fp = np.r_[0, np.cumsum(~sorted_labels)[last_of_tie]].astype(float)

    tp0, tp1, fp0, fp1 = tp[:-1], tp[1:], fp[:-1], fp[1:]
    keep = tp1 > tp0
    tp0, tp1, fp0, fp1 = tp0[keep], tp1[keep], fp0[keep], fp1[keep]

    slope = (fp1 - fp0) / (tp1 - tp0)
    a = 1.0 + slope
    b = fp0 - slope * tp0

    def antiderivative(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.where(b != 0, (b / a**2) * np.log(a * x + b), 0.0)
        return x / a - log_term

    area = antiderivative(tp1) - antiderivative(tp0)
    return float(area.sum() / n_pos)


def _sample_10x_negatives(labels, rng):
    """Indices of all positives plus 10x as many randomly sampled negatives."""
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.array([], dtype=np.int64)
    neg_idx = rng.choice(neg_idx, size=min(10 * len(pos_idx), len(neg_idx)), replace=False)
    return np.r_[pos_idx, neg_idx]


def compute_metrics(
    method_name: str,
    sample_name: str,
    celltype: str,
    labeled_df: pd.DataFrame,
    celltype_gt: pd.DataFrame,
    universe: pd.DataFrame,
    top_k_edges: int | None = None,
    top_fraction: float = 0.10,
    ptf_alpha: float = 0.05,
    random_state: int = 42,
    return_curve_data: bool = False,
):
    """
    Calculate ranking and top-edge GRN metrics for one method, sample, and
    cell type, following the benchmark paper's accuracy assessment.

    Inputs
    ------
    labeled_df : output of ``label_grn_with_universe``. One row per predicted
        pair in the universe, with Source, Target, _in_gt, and |Score|.
    celltype_gt : the cell type's GT edges (Source, Target), filtered to the
        sample's genes.
    universe : ``generate_edge_universe_from_gt(celltype_gt)``, every GT TF x
        every GT target with its _in_gt label.

    Ranking metrics
    ---------------
    AUROC : predicted edges only, with positives and negatives sampled 1:1.
    AUPRC : the complete universe U, with unpredicted pairs scored 0, all
        positives, and 10x as many sampled negatives. Uses the PRROC integral.
    Random AUPRC : the same, with uniform random scores over U.

    With ``return_curve_data=True``, also returns the labels and scores of the
    exact AUROC and AUPRC samples, for plotting ROC and PR curves.

    Set definitions
    ---------------
    A : the method's ``top_k_edges`` highest-scoring edges (10k in the paper),
        or its top ``top_fraction`` if ``top_k_edges`` is None
    B : GT edges
    U : every GT TF x every GT target

    A is the predicted-positive set for every binary metric (precision,
    recall, F1, accuracy, FPR, Jaccard, PerInt, EPR, and predictable TFs).
    No score cutoff is used, so methods with different score scales are
    compared on their most confident edges.
    """
    rng = np.random.default_rng(random_state)

    if top_k_edges is None and not 0 < top_fraction <= 1:
        raise ValueError(f"top_fraction must be in the interval (0, 1]. Got {top_fraction}")
    if top_k_edges is not None and top_k_edges <= 0:
        raise ValueError(f"top_k_edges must be > 0. Got {top_k_edges}")

    # U is the full GT TF x GT target grid, so every TF has the same candidate targets.
    gt_out_degree = celltype_gt.groupby("Source")["Target"].nunique()
    n_gt_targets = celltype_gt["Target"].nunique()
    n_universe = len(gt_out_degree) * n_gt_targets
    n_ground_truth_edges = len(celltype_gt)

    # Rank the predicted edges. Ties keep the universe order.
    df = labeled_df.sort_values("Score", ascending=False, kind="stable").reset_index(drop=True)
    labels = df["_in_gt"].astype(int).to_numpy()
    scores = df["Score"].astype(float).to_numpy()

    n_edges = len(df)
    n_pos = int(labels.sum())
    has_both_classes = 0 < n_pos < n_edges

    # ------------------------------------------------------------------
    # Top-edge set A.
    # ------------------------------------------------------------------
    if top_k_edges is not None:
        top_k = min(top_k_edges, n_edges)
    else:
        top_k = min(n_edges, max(1, int(round(top_fraction * n_edges))))
    top_df = df.head(top_k)

    early_precision = float(top_df["_in_gt"].mean()) if top_k else np.nan
    epr = _safe_divide(early_precision, n_pos / n_edges) if n_edges else np.nan

    # ------------------------------------------------------------------
    # AUROC: predicted edges only, balanced 1:1 by downsampling the larger class.
    # ------------------------------------------------------------------
    roc_labels = roc_scores = np.array([])
    if has_both_classes:
        pos_idx = np.flatnonzero(labels == 1)
        neg_idx = np.flatnonzero(labels == 0)
        n_balanced = min(len(pos_idx), len(neg_idx))
        balanced_idx = np.r_[
            rng.choice(pos_idx, size=n_balanced, replace=False),
            rng.choice(neg_idx, size=n_balanced, replace=False),
        ]
        roc_labels, roc_scores = labels[balanced_idx], scores[balanced_idx]
        auroc = roc_auc_score(roc_labels, roc_scores)
        random_auroc = roc_auc_score(roc_labels, rng.random(len(balanced_idx)))
    else:
        auroc = random_auroc = np.nan

    # ------------------------------------------------------------------
    # AUPRC: complete universe, unpredicted pairs scored 0, 1:10 negatives.
    # ------------------------------------------------------------------
    universe_scores = (
        universe[["Source", "Target", "_in_gt"]]
        .merge(df[["Source", "Target", "Score"]], on=["Source", "Target"], how="left", validate="one_to_one")
    )
    universe_labels = universe_scores["_in_gt"].to_numpy(dtype=int)
    universe_score_values = universe_scores["Score"].fillna(0.0).to_numpy(dtype=float)

    pr_idx = _sample_10x_negatives(universe_labels, rng)
    pr_labels, pr_scores = universe_labels[pr_idx], universe_score_values[pr_idx]
    auprc = _pr_auc_integral(pr_labels, pr_scores) if len(pr_idx) else np.nan

    random_idx = _sample_10x_negatives(universe_labels, rng)
    random_auprc = (
        _pr_auc_integral(universe_labels[random_idx], rng.random(len(random_idx)))
        if len(random_idx) else np.nan
    )

    # ------------------------------------------------------------------
    # Binary network metrics: A vs B within U.
    # ------------------------------------------------------------------
    tp = int(top_df["_in_gt"].sum())
    fp = top_k - tp
    fn = n_ground_truth_edges - tp
    tn = n_universe - n_ground_truth_edges - fp

    if tn < 0:
        raise ValueError(
            f"Computed a negative TN count for {method_name}/{sample_name}/{celltype}. "
            "Check that labeled_df and celltype_gt come from the same universe."
        )

    jaccard_index = _safe_divide(tp, tp + fp + fn)
    perint = 100.0 * _safe_divide(tp, min(top_k, n_ground_truth_edges))
    top_edge_recall = _safe_divide(tp, tp + fn)
    top_edge_fpr = _safe_divide(fp, fp + tn)

    # A is the predicted-positive set; everything else in U is predicted negative.
    precision = early_precision
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    accuracy = _safe_divide(tp + tn, n_universe)

    # ------------------------------------------------------------------
    # Predictable TFs: one-sided hypergeometric enrichment of each TF's
    # targets in A, then Benjamini-Hochberg across the TFs in A.
    # ------------------------------------------------------------------
    top_by_tf = (
        top_df.groupby("Source")["_in_gt"]
        .agg(n_predicted="size", n_overlap="sum")
        .join(gt_out_degree.rename("n_true"), how="inner")
    )
    ptf_p_values = np.asarray(
        hypergeom.sf(
            top_by_tf["n_overlap"].to_numpy() - 1,
            n_gt_targets,
            top_by_tf["n_true"].to_numpy(),
            top_by_tf["n_predicted"].to_numpy(),
        ),
        dtype=float,
    )
    ptf_q_values = _benjamini_hochberg(ptf_p_values)

    n_evaluable_tfs = len(ptf_p_values)
    predictable_tfs = int(np.sum(ptf_q_values < ptf_alpha))
    predictable_tfs_raw_p = int(np.sum(ptf_p_values < ptf_alpha))
    predictable_tf_fraction = _safe_divide(predictable_tfs, n_evaluable_tfs)

    metrics_df = pd.DataFrame([{
        "method_name": method_name,
        "sample_name": sample_name,
        "cell_type": celltype,
        "auroc": auroc,
        "rand_auroc": random_auroc,
        "auprc": auprc,
        "rand_auprc": random_auprc,
        "early_precision": early_precision,
        "early_precision_rate": epr,
        "accuracy": accuracy,
        "precision": precision,

        # Binary metrics from the top-edge set A.
        "recall": top_edge_recall,
        "fpr": top_edge_fpr,
        "f1": f1,
        "jaccard_index": jaccard_index,
        "perint": perint,

        # Full predicted network
        "n_tfs": df["Source"].nunique(),
        "n_targets": df["Target"].nunique(),
        "n_edges": n_edges,

        # Top-edge network
        "top_n_tfs": top_df["Source"].nunique(),
        "top_n_targets": top_df["Target"].nunique(),
        "top_k_edges": top_k,
        "max_edges": top_k_edges,

        "top_fraction": top_fraction,
        "predictable_tfs": predictable_tfs,
        "predictable_tfs_raw_p": predictable_tfs_raw_p,
        "predictable_tf_fraction": predictable_tf_fraction,
        "n_evaluable_tfs": n_evaluable_tfs,
        "top_tp": tp,
        "top_fp": fp,
        "top_fn": fn,
        "top_tn": tn,
        "n_universe_pairs": n_universe,
        "n_pos": n_pos,
        "n_neg": n_edges - n_pos,
        "n_ground_truth_edges": n_ground_truth_edges,
        "ptf_alpha": ptf_alpha,
        "ptf_multiple_testing": "Benjamini-Hochberg FDR",
    }])

    if return_curve_data:
        curve_data = {
            "roc_labels": roc_labels.astype(np.int8),
            "roc_scores": roc_scores.astype(np.float32),
            "pr_labels": pr_labels.astype(np.int8),
            "pr_scores": pr_scores.astype(np.float32),
        }
        return metrics_df, curve_data
    return metrics_df


def compute_all_metrics(method_grn_dict, celltype_gt_dict, test_genes, top_fraction=0.1):
    """
    Metrics of every (model, sample, cell type, method), and the compute_random_baseline_metrics
    inputs of the combined comparison (mean model and other methods).
    """
    metric_frames = []
    random_baseline_jobs = []
    for sample_name, celltype_dict in method_grn_dict.items():
        for celltype, method_dict in celltype_dict.items():
            celltype_gt = celltype_gt_dict.get(sample_name, {}).get(celltype)
            if celltype_gt is None:
                continue
            universe = generate_edge_universe_from_gt(celltype_gt)

            for method_key, method_grns in method_dict.items():
                # Methods with no GRN file for this cell type have an empty entry.
                if "full_universe" not in method_grns:
                    continue

                # TF-TG model GRNs are keyed (model name, eval type); other methods by method name.
                model_name, method_name = method_key if isinstance(method_key, tuple) else (None, method_key)

                # The self-trained model only predicts test-chromosome targets, so it is
                # scored on the test-chromosome universe. All other methods use all chromosomes.
                use_test_chroms = method_name == SELF_MODEL_METHOD
                labeled_df = method_grns["full_universe"]
                if use_test_chroms:
                    method_gt = celltype_gt[celltype_gt["Target"].isin(test_genes)]
                    if method_gt.empty:
                        logging.warning(f"No test-chromosome GT edges for {celltype} in {sample_name}; skipping")
                        continue
                    method_universe = generate_edge_universe_from_gt(method_gt)
                    labeled_df = labeled_df[
                        labeled_df["Source"].isin(method_gt["Source"])
                        & labeled_df["Target"].isin(method_gt["Target"])
                    ]
                else:
                    method_gt, method_universe = celltype_gt, universe

                metrics_df = compute_metrics(
                    method_name=method_name,
                    sample_name=sample_name,
                    celltype=celltype,
                    labeled_df=labeled_df,
                    celltype_gt=method_gt,
                    universe=method_universe,
                    top_fraction=top_fraction,
                )
                metrics_df["universe"] = "test_chromosomes" if use_test_chroms else "all_chromosomes"
                metrics_df["model_name"] = model_name
                metric_frames.append(metrics_df)

                # Random baseline inputs of the combined comparison only (mean model and other methods)
                if model_name in (None, MEAN_MODEL):
                    random_baseline_jobs.append(dict(
                        method_name=method_name,
                        sample_name=sample_name,
                        celltype=celltype,
                        labeled_df=labeled_df,
                        celltype_gt=method_gt,
                        universe=method_universe,
                        top_fraction=top_fraction,
                    ))

    # One row per (model, sample, cell type, method). model_name is MEAN_MODEL for the
    # mean score over the models, a model run key for one model, and None for other methods.
    return pd.concat(metric_frames, ignore_index=True), random_baseline_jobs


def metrics_for_model(all_model_metrics_df, model_name):
    """One model's metrics and the other methods' metrics on the same (sample, cell type) pairs."""
    model_rows = all_model_metrics_df[all_model_metrics_df["model_name"] == model_name]
    model_pairs = model_rows[["sample_name", "cell_type"]].drop_duplicates()
    other_rows = all_model_metrics_df[all_model_metrics_df["model_name"].isna()].merge(
        model_pairs, on=["sample_name", "cell_type"],
    )
    return pd.concat([model_rows, other_rows], ignore_index=True).drop(columns="model_name")


def compute_random_baseline_metrics(
    method_name: str,
    sample_name: str,
    celltype: str,
    labeled_df: pd.DataFrame,
    celltype_gt: pd.DataFrame,
    universe: pd.DataFrame,
    n_shuffles: int = N_RANDOM_SHUFFLES,
    random_state: int = 42,
    **metric_kwargs,
):
    """
    Random baseline of each metric in RANDOM_BASELINE_METRICS for one method,
    sample, and cell type.

    The method keeps its own predicted edges and its own score values, but the
    scores are shuffled across those edges. compute_metrics then scores each
    shuffled GRN in the same way as the real GRN, with the same 1:10 negative
    sample. The baseline is the mean over ``n_shuffles`` shuffles.

    This baseline can differ between methods because each method predicts a
    different set of edges:
      - Precision, recall, F1, accuracy, FPR, Jaccard, and PerInt depend on the
        fraction of the method's predicted edges that are GT edges, and on the
        size of the top-edge set A.
      - AUPRC depends on how many GT edges the method predicts at all, because
        unpredicted pairs are scored 0.
      - AUROC is 0.5 for every method, because its sample is balanced 1:1.

    The ``rand_auroc`` and ``rand_auprc`` columns of compute_metrics use random
    scores over the full universe U instead, so they do not depend on the method.
    """
    rng = np.random.default_rng(random_state)
    scores = labeled_df["Score"].to_numpy()

    shuffle_frames = [
        compute_metrics(
            method_name=method_name,
            sample_name=sample_name,
            celltype=celltype,
            labeled_df=labeled_df.assign(Score=rng.permutation(scores)),
            celltype_gt=celltype_gt,
            universe=universe,
            **metric_kwargs,
        )
        for _ in range(n_shuffles)
    ]
    shuffle_means = pd.concat(shuffle_frames, ignore_index=True)[RANDOM_BASELINE_METRICS].mean()

    return pd.DataFrame([{
        "method_name": method_name,
        "sample_name": sample_name,
        "cell_type": celltype,
        **shuffle_means.to_dict(),
        "n_shuffles": n_shuffles,
    }])


# ======================================================================
# Boxplots
# ======================================================================

def setup_font():
    font_path = PROJECT_DIR / "fonts" / "Arial.ttf"
    fm.fontManager.addfont(font_path)
    arial_font = fm.FontProperties(fname=font_path).get_name()
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [arial_font, "Helvetica", "Liberation Sans", "DejaVu Sans"]
    logging.info(f"Using font: {arial_font}")


def method_baselines(baseline_df, metric):
    """Median random baseline of each method for one metric (the box centers are medians too)."""
    return baseline_df.groupby("method_name")[metric].median().dropna()


def baselines_are_similar(baselines, tolerance=BASELINE_SIMILARITY_TOLERANCE):
    scale = max(1.0, baselines.abs().max())
    return baselines.max() - baselines.min() <= tolerance * scale


def draw_random_baselines(ax, baseline_df, metric, method_order, line_width=0.8):
    """
    Draw the random baselines of the methods in method_order as dashed grey lines:
    one line at the mean baseline when the methods' baselines are similar, else one
    line across each method's box. Box i is centered at x = i.
    """
    baselines = method_baselines(baseline_df, metric)
    baselines = baselines[baselines.index.isin(method_order)]
    if baselines.empty:
        return

    if baselines_are_similar(baselines):
        ax.axhline(baselines.mean(), zorder=4, **RANDOM_BASELINE_STYLE)
        return

    for position, method in enumerate(method_order):
        if method in baselines.index:
            ax.plot(
                [position - line_width / 2, position + line_width / 2],
                [baselines[method]] * 2,
                zorder=4,
                **RANDOM_BASELINE_STYLE,
            )


def _style_xticklabels(ax, originals, method_color_dict,
                       sample_rename_map=None, color_xticks=True,
                       rotation=45, fontsize=15):
    sample_rename_map = sample_rename_map or {}
    labels = [sample_rename_map.get(o, o) for o in originals]

    ax.set_xticks(range(len(originals)))
    ax.set_xticklabels(labels, rotation=rotation, ha="right",
                       fontsize=fontsize, rotation_mode="anchor")

    for tick, original in zip(ax.get_xticklabels(), originals):
        if original in TFTG_MODEL_METHODS:
            tick.set_fontsize(fontsize)
            if color_xticks:
                tick.set_color(method_color_dict.get(original, "black"))
            tick.set_path_effects([pe.withStroke(linewidth=0.6, foreground=tick.get_color())])
        else:
            tick.set_color("black")
            tick.set_fontweight("normal")


def plot_selected_box_and_whisker_plots(
    full_metric_df,
    method_color_dict,
    sample_rename_map,
    show_values_above_boxes=True,
    title_suffix_str="",
    baseline_df=None,
):
    selected_columns = ["auroc", "auprc", "precision", "f1", "recall", "fpr"]

    # Fixed y-limits of some panels. The other panels scale to their data.
    fixed_ylims = {"f1": (0, 0.6), "recall": (0, 0.4), "fpr": (0, 0.2)}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for ax, selected_column in zip(axes.flat, selected_columns):
        plot_df = full_metric_df.dropna(
            subset=["method_name", selected_column]
        )

        medians = (
            plot_df.groupby("method_name", observed=True)[selected_column]
            .median()
            .sort_values(ascending=False, kind="stable")
        )
        method_order = medians.index.tolist()

        if plot_df.empty:
            ax.set_title(f"{boxplot_metric_labels[selected_column]} by Method")
            ax.text(
                0.5, 0.5, "No data",
                transform=ax.transAxes,
                ha="center", va="center",
            )
            continue

        if baseline_df is not None:
            draw_random_baselines(ax, baseline_df, selected_column, method_order)

        sns.boxplot(
            data=plot_df,
            x="method_name",
            y=selected_column,
            hue="method_name",
            order=method_order,
            hue_order=method_order,
            palette=method_color_dict,
            dodge=False,
            width=0.6,
            whiskerprops={
                "linestyle": (5, (10, 3)),
                "linewidth": 1,
                "color": "black",
            },
            boxprops={"linewidth": 0},
            capprops={"linewidth": 1, "color": "black"},
            medianprops={"linewidth": 1, "color": "black"},
            showfliers=False,
            ax=ax,
        )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        # Keep descending order to match the reference.
        # For rank annotations, lower FPR is better.
        ranks = medians.rank(
            method="min",
            ascending=(selected_column == "fpr"),
        ).astype(int)

        upper_whiskers = []

        for position, method in enumerate(method_order):
            vals = plot_df.loc[
                plot_df["method_name"] == method, selected_column
            ]

            q1, q3 = vals.quantile([0.25, 0.75])
            upper_whisker = vals[
                vals <= q3 + 1.5 * (q3 - q1)
            ].max()
            upper_whiskers.append(upper_whisker)

            annotation = (
                f"{medians.loc[method]:.3f}"
                if show_values_above_boxes
                else str(ranks.loc[method])
            )

            ax.annotate(
                annotation,
                xy=(position, upper_whisker),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        _style_xticklabels(
            ax,
            method_order,
            method_color_dict=method_color_dict,
            sample_rename_map=sample_rename_map,
            color_xticks=True,
            rotation=45,
            fontsize=11,
        )

        label = boxplot_metric_labels[selected_column]
        title = f"{label} by Method"

        if title_suffix_str:
            title += f" {title_suffix_str.strip()}"

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=13)
        ax.tick_params(axis="y", labelsize=12)

        if selected_column in fixed_ylims:
            ax.set_ylim(*fixed_ylims[selected_column])
        else:
            ymax = plot_df[selected_column].max()
            ax.set_ylim(
                0,
                max(
                    1.10 if ymax <= 1 else ymax * 1.10,
                    max(upper_whiskers) * 1.10,
                ),
            )
        ax.grid(False)

    fig.tight_layout()

    if baseline_df is not None:
        fig.legend(
            handles=[RANDOM_BASELINE_HANDLE], loc="upper center",
            bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=13,
        )
    return fig


def baselines_for_model(random_baseline_df, all_model_metrics_df, model_name):
    """
    Random baselines of the rows in metrics_for_model(model_name). Each model scores
    the same edges as the mean-score model on every (sample, cell type), so it has
    the mean-score model's baseline.
    """
    model_rows = metrics_for_model(all_model_metrics_df, model_name)[["method_name", "sample_name", "cell_type"]]
    return random_baseline_df.merge(model_rows, on=["method_name", "sample_name", "cell_type"])


# ======================================================================
# Per-cell-type heatmaps and average ranks
# ======================================================================

def summarize_by_cell_type(metric_df):
    """Average each method's metrics over the samples that contain the cell type."""
    metric_df = metric_df.copy()
    metric_df["tissue"] = metric_df["sample_name"].map(sample_tissue_map)
    unmapped = sorted(metric_df.loc[metric_df["tissue"].isna(), "sample_name"].unique())
    if unmapped:
        raise ValueError(f"No tissue in sample_tissue_map for: {unmapped}")

    numeric_cols = metric_df.select_dtypes("number").columns
    metric_df = (
        metric_df.groupby(["cell_type", "tissue", "method_name"], as_index=False)[numeric_cols].mean()
    )
    metric_df["panel"] = metric_df["cell_type"] + " (" + metric_df["tissue"] + ")"
    return metric_df


def _display_method_name(method_name):
    return method_display_name_map.get(method_name, method_name)


def _style_method_ticklabels(ticklabels, methods, fontsize):
    """Color and embolden the MONICA labels, as in the boxplots."""
    for tick, method in zip(ticklabels, methods):
        tick.set_fontsize(fontsize)
        if method in TFTG_MODEL_METHODS:
            tick.set_color(method_color_dict.get(method, "black"))
            tick.set_path_effects([pe.withStroke(linewidth=0.6, foreground=tick.get_color())])
        else:
            tick.set_color("black")


def average_method_rank(metric_df, metric_name):
    """Rank the methods in each cell type panel, then average each method's rank."""
    rank_df = metric_df[["panel", "method_name", metric_name]].dropna().copy()
    rank_df["rank"] = (
        rank_df.groupby("panel")[metric_name]
        .rank(method="min", ascending=False)
    )
    average_rank = (
        rank_df.groupby("method_name", as_index=False)["rank"]
        .mean().rename(columns={"rank": "average_rank"})
        .sort_values("average_rank")
    )
    return rank_df, average_rank


def plot_rank_lollipop(average_rank, metric_name, title_suffix_str=""):
    ordered_methods = average_rank["method_name"].tolist()
    fig_rank, ax_rank = plt.subplots(figsize=(7, 5))
    y_positions = np.arange(len(average_rank))
    max_rank = max(float(average_rank["average_rank"].max()) + 0.5, 1.5)
    colors = [method_color_dict[method] for method in ordered_methods]
    ax_rank.hlines(
        y_positions, max_rank, average_rank["average_rank"],
        color=colors, linewidth=2, alpha=0.8,
    )
    ax_rank.scatter(
        average_rank["average_rank"], y_positions,
        color=colors, s=150, zorder=3,
    )
    ax_rank.set_yticks(y_positions)
    ax_rank.set_yticklabels([_display_method_name(method) for method in ordered_methods])
    _style_method_ticklabels(ax_rank.get_yticklabels(), ordered_methods, fontsize=15)
    ax_rank.tick_params(axis="x", labelsize=14)
    ax_rank.set_xlim(max_rank, 0.5)
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("Average rank (1 is best)", fontsize=15)
    ax_rank.set_title(f"Mean Method Rankings for {metric_name.upper()}{title_suffix_str}", fontsize=18)
    ax_rank.grid(axis="x", linestyle="--", alpha=0.3)
    fig_rank.tight_layout()
    return fig_rank, ax_rank


def plot_metric_heatmap(rank_df, ordered_methods, metric_name, model_methods):
    metric_matrix = rank_df.pivot(
        index="panel", columns="method_name", values=metric_name,
    ).reindex(columns=ordered_methods)
    # Self-trained-only rows on top, then rows scored by both models, then cross-trained-only.
    panel_models = {
        panel: tuple(m for m in model_methods if m in row and pd.notna(row[m]))
        for panel, row in metric_matrix.iterrows()
    }
    group_order = {
        (SELF_MODEL_METHOD,): 0,
        (SELF_MODEL_METHOD, CROSS_MODEL_METHOD): 1,
        (CROSS_MODEL_METHOD,): 2,
    }

    if GROUP_BY_SELF_VS_EVAL:
        metric_matrix = metric_matrix.loc[sorted(
            metric_matrix.index,
            key=lambda panel: (group_order.get(panel_models[panel], len(group_order)), panel),
        )]
    else:
        metric_matrix = metric_matrix.loc[
            metric_matrix.mean(axis=1).sort_values(ascending=False).index
        ]

    fig_heatmap, ax_heatmap = plt.subplots(
        figsize=(1.25 * len(ordered_methods) + 2, 0.3 * len(metric_matrix) + 1),
    )
    sns.heatmap(
        metric_matrix, annot=True, fmt=".2f", cmap="viridis",
        vmin=0, vmax=1, linewidths=0.5, linecolor="white", ax=ax_heatmap,
        annot_kws={"fontsize": 12},
    )
    ax_heatmap.set_xticklabels(
        [_display_method_name(method) for method in ordered_methods],
        rotation=45, ha="right", rotation_mode="anchor",
    )
    _style_method_ticklabels(ax_heatmap.get_xticklabels(), ordered_methods, fontsize=15)
    ax_heatmap.tick_params(axis="y", labelsize=14)

    ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=13)
    ax_heatmap.set_xlabel("")
    ax_heatmap.set_ylabel("")
    ax_heatmap.set_title(f"{metric_name.upper()} by Cell Type and Method", fontsize=18)
    fig_heatmap.tight_layout()
    return fig_heatmap


def save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info(f"Saved {path}")


# ======================================================================
# Main
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the cell-type TF-TG models against other GRN inference methods "
                    "(celltype_model_evaluation.ipynb, cells 17-30).",
    )
    parser.add_argument(
        "--runs", nargs="+", default=None, metavar="NAME=RUN",
        help="Model runs as name=run_directory under checkpoints/celltype_tf_tg; the run may be a glob "
             "that matches one directory (e.g. no_liver=celltype_joint_3926071_*). Default: the "
             "2026-10-22 holdout models.",
    )
    parser.add_argument("--species", default="mm10", choices=sorted(TEST_CHROMS))
    parser.add_argument("--results_dir", type=Path, default=PROJECT_DIR / "new_testing_results",
                        help="Labeled-GRN cache and metric tables (file names carry the run hash)")
    parser.add_argument("--plot_dir", type=Path, default=PROJECT_DIR / "new_plots",
                        help="Root of the plot folders (model_vs_other_method_boxplots/, rank_plots/). "
                             "Use a new folder per model set to keep earlier plots.")
    parser.add_argument("--other_method_grn_dir", type=Path, default=OTHER_METHOD_GRN_DIR,
                        help="One subfolder of formatted GRNs per other method")
    parser.add_argument("--top_fraction", type=float, default=0.1,
                        help="Top-edge set A for the early metrics: this fraction of each method's edges")
    parser.add_argument("--n_random_shuffles", type=int, default=N_RANDOM_SHUFFLES)
    parser.add_argument("--n_jobs", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--force_relabel", action="store_true",
                        help="Relabel the GRNs even when the cache for this run hash exists "
                             "(needed after regenerating GRNs under the same run names)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.runs is None:
        run_specs = DEFAULT_MODEL_RUNS
    else:
        run_specs = dict(spec.split("=", 1) for spec in args.runs)
    model_run_dict = resolve_runs(run_specs)
    for model_name, run_name in model_run_dict.items():
        logging.info(f"{model_name}: {run_name}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    boxplot_dir = args.plot_dir / "model_vs_other_method_boxplots"
    heatmap_dir = args.plot_dir / "rank_plots" / "rank_heatmaps"
    rank_lollipop_dir = args.plot_dir / "rank_plots" / "rank_lollipop_plots"

    # ---------------- Label the GRNs ----------------
    sample_sources, uid_of = find_samples(DATA_DIR)
    method_grn_dict, celltype_gt_dict, hash_key = label_method_grns(
        model_run_dict, sample_sources, uid_of, args.species, args.results_dir,
        args.other_method_grn_dir, args.n_jobs, force_relabel=args.force_relabel,
    )
    logging.info(f"Run hash: {hash_key}")

    # ---------------- Metrics ----------------
    from scripts.build_tf_to_tg_celltype_data import split_genes_by_chromosome

    # All runs use the default gene annotation (gene_ref_file is null in their configs).
    gene_ref_file = (
        DATA_DIR / "genome_data" / "genome_annotation" / args.species / DEFAULT_GENE_REF_FILES[args.species]
    )
    _, _, test_genes = split_genes_by_chromosome(
        gene_ref_file, train_chroms=[], val_chroms=[], test_chroms=TEST_CHROMS[args.species],
    )
    test_genes = set(test_genes)

    all_model_metrics_df, random_baseline_jobs = compute_all_metrics(
        method_grn_dict, celltype_gt_dict, test_genes, top_fraction=args.top_fraction,
    )

    # Combined comparison: the mean-score model and the other methods
    celltype_metrics_df = (
        all_model_metrics_df[all_model_metrics_df["model_name"].isna() | (all_model_metrics_df["model_name"] == MEAN_MODEL)]
        .drop(columns="model_name")
        .reset_index(drop=True)
    )
    celltype_metrics_df.to_csv(args.results_dir / f"celltype_metrics_{hash_key}.tsv", sep="\t", index=False)
    all_model_metrics_df.to_csv(args.results_dir / f"celltype_metrics_per_model_{hash_key}.tsv", sep="\t", index=False)
    logging.info(f"Metrics: {len(celltype_metrics_df)} combined rows, {len(all_model_metrics_df)} per-model rows")

    # ---------------- Random baselines ----------------
    # One row per (sample, cell type, method) of the combined comparison, in the same
    # order as celltype_metrics_df. The columns have the same names as the real metrics.
    random_baseline_frames = Parallel(
        n_jobs=min(len(random_baseline_jobs), args.n_jobs), backend="loky", verbose=5,
    )(
        delayed(compute_random_baseline_metrics)(**job, n_shuffles=args.n_random_shuffles)
        for job in random_baseline_jobs
    )
    random_baseline_df = pd.concat(random_baseline_frames, ignore_index=True)
    random_baseline_df.to_csv(args.results_dir / f"celltype_random_baselines_{hash_key}.tsv", sep="\t", index=False)

    # ---------------- Boxplots ----------------
    setup_font()
    sample_rename_map = method_display_name_map

    # Are the methods' random baselines similar? One row per boxplot metric, with each
    # method's median baseline, the range over the methods, and the similarity result.
    baseline_similarity_df = pd.DataFrame(
        {metric: method_baselines(random_baseline_df, metric) for metric in BOXPLOT_METRICS}
    ).T
    baseline_similarity_df["range"] = baseline_similarity_df.max(axis=1) - baseline_similarity_df.min(axis=1)
    baseline_similarity_df["similar"] = [
        baselines_are_similar(method_baselines(random_baseline_df, metric)) for metric in BOXPLOT_METRICS
    ]
    baseline_similarity_df = baseline_similarity_df.rename(columns=method_display_name_map)
    baseline_similarity_df.to_csv(args.results_dir / f"celltype_random_baseline_similarity_{hash_key}.tsv", sep="\t")

    fig = plot_selected_box_and_whisker_plots(
        celltype_metrics_df, method_color_dict, sample_rename_map, baseline_df=random_baseline_df,
    )
    save_figure(fig, boxplot_dir / "all_metrics_by_method_boxplot.png")

    for model_name in model_run_dict:
        fig = plot_selected_box_and_whisker_plots(
            metrics_for_model(all_model_metrics_df, model_name),
            method_color_dict,
            sample_rename_map,
            baseline_df=baselines_for_model(random_baseline_df, all_model_metrics_df, model_name),
        )
        save_figure(fig, boxplot_dir / model_name / f"{model_name}_full_boxplot.png")

    # ---------------- Random baselines by method ----------------
    # Median random baseline of each metric over each method's (sample, cell type) pairs
    random_baseline_by_method = (
        random_baseline_df.groupby("method_name")[RANDOM_BASELINE_METRICS]
        .median()
        .T
        .rename(columns=method_display_name_map)
    )

    # The methods do not all cover the same (sample, cell type) pairs, so the medians above
    # also differ by which pairs each method has. To compare the methods on the same pairs,
    # subtract the median baseline of all methods on a pair from each method's baseline on it.
    # Note: the self-trained model uses the test-chromosome universe, so part of its
    # difference comes from the universe and not from its predicted edges.
    pair_median_baseline = (
        random_baseline_df.groupby(["sample_name", "cell_type"])[RANDOM_BASELINE_METRICS]
        .transform("median")
    )
    random_baseline_offset = (
        (random_baseline_df[RANDOM_BASELINE_METRICS] - pair_median_baseline)
        .groupby(random_baseline_df["method_name"])
        .median()
        .T
        .rename(columns=method_display_name_map)
    )
    random_baseline_by_method.to_csv(args.results_dir / f"celltype_random_baseline_by_method_{hash_key}.tsv", sep="\t")
    random_baseline_offset.to_csv(args.results_dir / f"celltype_random_baseline_offset_{hash_key}.tsv", sep="\t")

    fig = plot_selected_box_and_whisker_plots(
        random_baseline_df, method_color_dict, sample_rename_map, title_suffix_str="(Random Baseline)",
    )
    save_figure(fig, boxplot_dir / "all_metrics_random_baseline_by_method_boxplot.png")

    # ---------------- Heatmaps and average ranks ----------------
    comparison_metric_df = summarize_by_cell_type(celltype_metrics_df)

    # Method order: TF-TG models first, then the other methods.
    available_methods = sorted(comparison_metric_df["method_name"].unique())
    model_methods = [m for m in (SELF_MODEL_METHOD, CROSS_MODEL_METHOD) if m in available_methods]
    method_order = model_methods + [m for m in available_methods if m not in model_methods]
    fallback_colors = sns.color_palette("tab10", n_colors=max(len(method_order), 1))
    for index, method in enumerate(method_order):
        method_color_dict.setdefault(method, fallback_colors[index])

    average_rank_frames = []
    for metric_name in ("auroc", "auprc"):
        rank_df, average_rank = average_method_rank(comparison_metric_df, metric_name)
        ordered_methods = average_rank["method_name"].tolist()

        fig = plot_metric_heatmap(rank_df, ordered_methods, metric_name, model_methods)
        save_figure(fig, heatmap_dir / f"{metric_name}_method_heatmap.png")

        fig, _ = plot_rank_lollipop(average_rank, metric_name)
        save_figure(fig, rank_lollipop_dir / f"average_{metric_name}_rank_lollipop.png")
        average_rank_frames.append(average_rank.assign(metric=metric_name, model_name=MEAN_MODEL))

    # Each model on its own, against the other methods on the same (sample, cell type) pairs
    for model_name in model_run_dict:
        model_comparison_df = summarize_by_cell_type(metrics_for_model(all_model_metrics_df, model_name))
        for metric_name in ("auroc", "auprc"):
            _, model_average_rank = average_method_rank(model_comparison_df, metric_name)
            fig, _ = plot_rank_lollipop(model_average_rank, metric_name, title_suffix_str=f"\n({model_name})")
            save_figure(fig, rank_lollipop_dir / model_name / f"average_{metric_name}_rank_lollipop.png")
            average_rank_frames.append(model_average_rank.assign(metric=metric_name, model_name=model_name))

    comparison_metric_df.to_csv(args.plot_dir / "comparison_metrics_by_cell_type.csv", index=False)
    pd.concat(average_rank_frames, ignore_index=True).to_csv(
        args.results_dir / f"celltype_average_method_rank_{hash_key}.tsv", sep="\t", index=False,
    )

    # Run record: which model runs and outputs belong to this hash
    (args.results_dir / f"method_comparison_{hash_key}.json").write_text(json.dumps({
        "model_runs": model_run_dict,
        "species": args.species,
        "top_fraction": args.top_fraction,
        "n_random_shuffles": args.n_random_shuffles,
        "plot_dir": str(args.plot_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }, indent=2))

    # ---------------- Summary ----------------
    pd.set_option("display.width", 200)
    summary = (
        celltype_metrics_df.groupby("method_name")[["auroc", "auprc", "precision", "f1", "recall"]]
        .median()
        .rename(index=method_display_name_map)
        .sort_values("auroc", ascending=False)
    )
    print(f"\nMedian over (sample, cell type) pairs, mean-score TF-TG model vs other methods (hash {hash_key}):")
    print(summary.round(3).to_string())
    for metric_name in ("auroc", "auprc"):
        ranks = pd.concat(average_rank_frames)
        ranks = ranks[(ranks["metric"] == metric_name) & (ranks["model_name"] == MEAN_MODEL)]
        print(f"\nAverage {metric_name.upper()} rank over cell types (1 is best):")
        print(ranks.assign(method_name=ranks["method_name"].map(_display_method_name))
              [["method_name", "average_rank"]].round(2).to_string(index=False))
    print(f"\nTables in {args.results_dir} (*_{hash_key}.*), plots in {args.plot_dir}")


if __name__ == "__main__":
    main()
