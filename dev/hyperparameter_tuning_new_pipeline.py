from math import exp
import sys, json, os, time, re
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import product
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import seaborn as sns
from tqdm import tqdm
import random
import zlib
import importlib
from cycler import cycler
from scipy.stats import norm
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(message)s')

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
SRC_DIR = str(Path(PROJECT_DIR) / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from multiomic_transformer.utils import data_formatter, experiment_handler
import multiomic_transformer.utils.experiment_loader as experiment_loader
from multiomic_transformer.models.model_simplified import MultiomicTransformer
from multiomic_transformer.datasets.dataset_refactor import (
    SimpleScaler,
)
import muon_preprocessing as muon_prep

DATA_DIR = Path("/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
GROUND_TRUTH_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/ground_truth_files")

color_palette = {
  "blue_light": "#18A6ED",
  "orange_light": "#EEA700",
  "red_light": "#EF767A",
  "green_light": "#7EE3BA",
  "purple_light": "#C798CC",
  "grey_light": "#BCBCBF",
  "blue_dark": "#2E70B9",
  "orange_dark": "#D18A3D",
  "red_dark": "#BC3E1A",
  "green_dark": "#32936F",
  "purple_dark": "#9D5ED4",
  "grey_dark": "#434B4E",
}

plt.rcParams.update({

    # figure
    "figure.figsize": (6,4),
    "figure.dpi": 300,

    # fonts
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,

    # axes
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "grid.alpha": 0.25,

    # lines
    "lines.linewidth": 2,

    # legend
    "legend.frameon": False,

    # color cycle
    "axes.prop_cycle": cycler(color=color_palette.values()),
})

method_color_dict = {
  "Gradient Attribution": "#4195DF",
  "LINGER": "#7EE3BA",
  "SCENIC+": "#EF767A",
  "CellOracle": "#F9C60D",
  "Pando": "#EF9CFA",
  "TRIPOD": "#82EC32",
  "FigR": "#FDA7BB",
  "GRaNIE": "#F98637"
}

light_colors = [v for k,v in color_palette.items() if "light" in k]

order = ["LINGER", "SCENIC+", "CellOracle", "GRaNIE", "Pando", "TRIPOD", "FigR"]

def load_ground_truth(ground_truth_file):
    if type(ground_truth_file) == str:
        ground_truth_file = Path(ground_truth_file)
        
    if ground_truth_file.suffix == ".csv":
        sep = ","
    elif ground_truth_file.suffix == ".tsv":
        sep="\t"
        
    ground_truth_df = pd.read_csv(ground_truth_file, sep=sep, on_bad_lines="skip", engine="python")
    
    if "chip" in ground_truth_file.name and "atlas" in ground_truth_file.name:
        ground_truth_df = ground_truth_df[["source_id", "target_id"]]

    if ground_truth_df.columns[0] != "Source" or ground_truth_df.columns[1] != "Target":
        ground_truth_df = ground_truth_df.rename(columns={ground_truth_df.columns[0]: "Source", ground_truth_df.columns[1]: "Target"})
    ground_truth_df["Source"] = ground_truth_df["Source"].astype(str).str.upper()
    ground_truth_df["Target"] = ground_truth_df["Target"].astype(str).str.upper()
    
    # Build TF, TG, and edge sets for quick lookup later
    gt = ground_truth_df[["Source", "Target"]].dropna()

    gt_tfs = set(gt["Source"].unique())
    gt_tgs = set(gt["Target"].unique())
    
    gt_pairs = (gt["Source"] + "\t" + gt["Target"]).drop_duplicates()
    
    gt_lookup = (gt_tfs, gt_tgs, set(gt_pairs))
        
    return ground_truth_df, gt_lookup

def expand_experiment_dict_grid(experiment_dict):
    # Separate fixed params from swept params
    sweep_keys = [k for k, v in experiment_dict.items() if len(v) > 1]
    fixed_keys = [k for k, v in experiment_dict.items() if len(v) == 1]

    # Cartesian product over all swept parameters
    sweep_values = [experiment_dict[k] for k in sweep_keys]
    combinations = list(product(*sweep_values))

    expanded = {k: [] for k in experiment_dict.keys()}

    for combo in combinations:
        combo_dict = dict(zip(sweep_keys, combo))

        for k in fixed_keys:
            expanded[k].append(experiment_dict[k][0])

        for k in sweep_keys:
            expanded[k].append(combo_dict[k])

    return expanded

def determine_experiment_differences(experiment_dict, index, num_experiments):
    max_key_len = max(len(key) for key in experiment_dict.keys())
    max_val_len = max(
        len(str(v))
        for values in experiment_dict.values()
        for v in values
    )

    if 0 < index < num_experiments:
        for key in experiment_dict.keys():
            prev = experiment_dict[key][index - 1]
            curr = experiment_dict[key][index]

            if prev != curr:
                logging.info(f"{key:<{max_key_len}} : {str(prev):<{max_val_len}} -> {curr}")
            else:
                logging.info(f"{key:<{max_key_len}} : {curr}")
    else:
        for key in experiment_dict.keys():
            logging.info(f"{key:<{max_key_len}} : {experiment_dict[key][index]}")

def aggregate_results(
    experiment_dict, 
    auroc_df_all, 
    gpu_mem_df_all, 
    batch_profile_df_all, 
    epoch_log_df_all
    ):
    group_cols = list(experiment_dict.keys())

    epoch_log_df_all_grouped = (
        epoch_log_df_all
        .groupby(group_cols)
        .agg({
            "r2_unscaled": "max",
            "r2_scaled": "max",
            "epoch_time_s": "mean",
            "peak_allocated_mb": "max",
            "peak_reserved_mb": "max",
        })
        .reset_index()
    )

    batch_profile_df_all_grouped = (
        batch_profile_df_all
        .groupby(group_cols)
        .agg({
            "total_step_s": "mean",
            "loader_s": "mean",
            "transfer_s": "mean",
            "forward_s": "mean",
            "backward_s": "mean",
            "optim_s": "mean",
        })
        .reset_index()
    )

    gpu_mem_df_all_grouped = (
        gpu_mem_df_all
        .groupby(group_cols)
        .agg({
            "allocated_mb": "mean",
            "reserved_mb": "mean",
            "free_mb": "mean",
            "total_memory_mb": "mean",
            "allocated_pct_total": "mean",
            "reserved_pct_total": "mean",
            "free_pct_total": "mean",
        })
        .reset_index()
    )

    full_summary_df = (
        auroc_df_all
        .merge(epoch_log_df_all_grouped, on=group_cols, how="left")
        .merge(batch_profile_df_all_grouped, on=group_cols, how="left")
        .merge(gpu_mem_df_all_grouped, on=group_cols, how="left")
    )

    ordered_cols = [
        # --- Experiment identifiers ---
        "experiment_name",
        "sample_type",

        # --- Model hyperparameters ---
        "kernel_size",
        "d_model",
        "d_ff",
        "num_layers",
        "num_heads",

        # --- Training params ---
        "batch_size",
        "epochs",
        "bias_scale",
        "grad_attrib_batches",
        "grad_attrib_tgs_per_batch",
        "dataloader_workers",
        "max_cached",
        
        # --- Performance ---
        "pooled_median_auroc",
        "per_tf_median_auroc",
        "r2_unscaled",
        "r2_scaled",

        # --- Timing ---
        "epoch_time_s",
        "total_step_s",
        "loader_s",
        "transfer_s",
        "forward_s",
        "backward_s",
        "optim_s",

        # --- Memory ---
        "peak_allocated_mb",
        "peak_reserved_mb",
        "allocated_mb",
        "reserved_mb",
        "free_mb",
        "total_memory_mb",
        "allocated_pct_total",
        "reserved_pct_total",
        "free_pct_total",
        
        "replicate",
    ]

    full_summary_df = full_summary_df[ordered_cols]
    
    return full_summary_df

def save_summary_df(full_summary_df, summary_save_path):
    key_cols = [
        "experiment_name",
        "sample_type",
        "kernel_size",
        "d_model",
        "d_ff",
        "num_layers",
        "num_heads",
        "batch_size",
        "epochs",
        "bias_scale",
        "grad_attrib_batches",
        "grad_attrib_tgs_per_batch",
        "dataloader_workers",
        "max_cached",
        "replicate",
    ]

    def get_safe_path(base_path: Path):
        if not base_path.exists():
            return base_path

        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent

        i = 1
        while True:
            new_path = parent / f"{stem}_{i}{suffix}"
            if not new_path.exists():
                return new_path
            i += 1

    save_path_to_use = summary_save_path

    if summary_save_path.exists():
        try:
            existing_df = pd.read_csv(summary_save_path)

            if list(existing_df.columns) != list(full_summary_df.columns):
                raise ValueError("Column mismatch between existing and new results")

            merged_df = pd.concat([existing_df, full_summary_df], ignore_index=True)

            # Keep newest version of repeated experiment configs
            merged_df = merged_df.drop_duplicates(subset=key_cols, keep="last")

            full_summary_df = merged_df

        except Exception as e:
            logging.info(f"Merge failed: {e}")
            save_path_to_use = get_safe_path(summary_save_path)

    full_summary_df.to_csv(save_path_to_use, index=False)
    logging.info(f"Saved experiment summary to: {save_path_to_use.parent}/{save_path_to_use.name}")

def plot_gpu_memory(gpu_mem_df):
    df = gpu_mem_df.copy()
    df = df.groupby("step")[[
        "allocated_mb", 
        "reserved_mb",
        "free_mb",
        "total_memory_mb",
        "allocated_pct_total", 
        "reserved_pct_total",
        "free_pct_total",
        ]].mean().reset_index()
    
    df = df.iloc[5:]
    
    fig = plt.figure(figsize=(4,3))
    plt.plot(df["step"], df["allocated_mb"], color=color_palette["blue_light"], label="Allocated")
    plt.plot(df["step"], df["reserved_mb"], linestyle="--", color=color_palette["grey_light"], label="Reserved")

    total_mem = df["total_memory_mb"].iloc[0]
    plt.hlines(
        total_mem,
        df["step"].min(),
        df["step"].max(),
        linestyles="dashed",
        label="Total",
        color=color_palette["grey_dark"],
    )

    plt.xlabel("Training Step")
    plt.ylabel("Memory (MB)")
    plt.title(f"GPU Memory Usage")
    plt.legend(
        bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.,
        title="Memory Type"
    )
    return fig
    
def plot_training_time_by_step(df_full):
    df = df_full.copy()
    df = df.groupby("step")[["loader_s", "transfer_s", "forward_s", "backward_s"]].mean().reset_index()
    df["loader_s"] = df["loader_s"].rolling(window=10, min_periods=1).mean()
    df["transfer_s"] = df["transfer_s"].rolling(window=10, min_periods=1).mean()
    df["forward_s"] = df["forward_s"].rolling(window=10, min_periods=1).mean()
    df["backward_s"] = df["backward_s"].rolling(window=10, min_periods=1).mean()
    
    df = df.iloc[10:].iloc[:-20]

    fig = plt.figure(figsize=(6, 4))
    plt.plot(df["step"], df["loader_s"], label="Data Loading")
    plt.plot(df["step"], df["transfer_s"], label="Data Transfer")
    plt.plot(df["step"], df["forward_s"], label="Forward Pass")
    plt.plot(df["step"], df["backward_s"], label="Backward Pass")
    plt.xlabel("Training Step")
    plt.ylabel("Time (s)")
    plt.title(f"Training Time by Step")
    plt.legend(
            bbox_to_anchor=(1.05, 0.5), loc='center left',
            title="Training Step", borderaxespad=0.
    )
    return fig
    
def plot_train_step_time_by_kernel_size(batch_profile_df_all):
    df = (
        batch_profile_df_all
        .groupby("kernel_size")[["loader_s", "transfer_s", "forward_s", "backward_s"]]
        .mean()
        .reset_index()
    )

    fig = plt.figure(figsize=(6,4))

    plt.plot(df["kernel_size"], df["loader_s"], color=color_palette["blue_light"], label="Data Loading")
    plt.plot(df["kernel_size"], df["transfer_s"], color=color_palette["orange_light"], label="Data Transfer")
    plt.plot(df["kernel_size"], df["forward_s"], color=color_palette["red_light"], label="Forward Pass")
    plt.plot(df["kernel_size"], df["backward_s"], color=color_palette["green_light"], label="Backward Pass")

    plt.xlabel("Kernel Size")
    plt.ylabel("Average Time (s)")
    plt.title("Batch Profile by Kernel Size")
    plt.legend(
        bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.,
        title="Training Step"
    )
    return fig

def plot_auroc_by_kernel_size(auroc_df_all):
    fig = plt.figure(figsize=(6,4))
    auroc_df_all_grouped = (
        auroc_df_all
        .groupby("kernel_size")[["pooled_median_auroc", "per_tf_median_auroc"]]
        .mean()
        .reset_index()
    )
    plt.hlines(
        0.5,
        auroc_df_all_grouped["kernel_size"].min(),
        auroc_df_all_grouped["kernel_size"].max(),
        linestyles="dashed",
        color=color_palette["grey_dark"],
    )
    plt.plot(
        auroc_df_all_grouped["kernel_size"], auroc_df_all_grouped["pooled_median_auroc"], 
        marker="o", color=color_palette["grey_light"], label="Pooled Median AUROC"
        )
    plt.plot(
        auroc_df_all_grouped["kernel_size"], auroc_df_all_grouped["per_tf_median_auroc"], 
        marker="o", color=color_palette["blue_light"], label="Per-TF Median AUROC"
        )

    plt.xlabel("Kernel Size")
    plt.ylabel("AUROC")
    plt.title("AUROC by Kernel Size")
    plt.ylim((0.3, 0.7))
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
    return fig

def load_ground_truth_dict():
    gt_by_dataset_dict = {
        # "Macrophage": {
        #     # "RN204": load_ground_truth(GROUND_TRUTH_DIR / "rn204_macrophage_human_chipseq.tsv"),
        #     "ChIP-Atlas macrophage": load_ground_truth(GROUND_TRUTH_DIR / "chipatlas_macrophage.csv"),
        # },
        "mESC": {
            "ChIP-Atlas mESC": load_ground_truth(GROUND_TRUTH_DIR / "chip_atlas_tf_peak_tg_dist.csv"),
            "RN111": load_ground_truth(GROUND_TRUTH_DIR / "RN111.tsv"),
            "RN112": load_ground_truth(GROUND_TRUTH_DIR / "RN112.tsv"),
            "RN114": load_ground_truth(GROUND_TRUTH_DIR / "RN114.tsv"),
            "RN116": load_ground_truth(GROUND_TRUTH_DIR / "RN116.tsv"),        
        },
        # "K562": {
        #     "ChIP-Atlas K562": load_ground_truth(GROUND_TRUTH_DIR / "chipatlas_K562.csv"),
        #     "RN117": load_ground_truth(GROUND_TRUTH_DIR / "RN117.tsv"),        
        # },
        # "iPSC": {
        #     # "ChIP-Atlas iPSC": load_ground_truth(GROUND_TRUTH_DIR / "chipatlas_iPSC.csv"),
        #     "ChIP-Atlas iPSC (1 Mb)": load_ground_truth(GROUND_TRUTH_DIR / "chipatlas_iPSC_1mb.csv"),
        #     # "ChIP-Atlas iPSC (100 kb)": load_ground_truth(GROUND_TRUTH_DIR / "chipatlas_iPSC_100kb.csv"),
        # }
    }
    
    return gt_by_dataset_dict

def _run_experiments_on_gpu(
    tdf,
    gpu_id,
    experiment_indices,
    lock,
    experiment_dict,
    num_experiments,
    summary_save_path,
    sample_type,
    gt_by_dataset_dict,
    experiment_dir,
):
    # Restrict this worker to a single GPU BEFORE any CUDA calls.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        current_device = torch.cuda.current_device()
    else:
        device_name = "cpu"
        current_device = "cpu"

    logging.debug(
        f"[Worker pid={os.getpid()}] gpu_id={gpu_id} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"current_device={current_device}"
    )

    exp = experiment_handler.ExperimentHandler(
        training_data_formatter=tdf,
        experiment_dir=experiment_dir,
        model_num=1,
        silence_warnings=False,
    )
    if torch.cuda.is_available():
        exp.device = torch.device("cuda:0")

    previous_experiments_df = pd.read_csv(summary_save_path) if summary_save_path.exists() else None

    experiment_dict["d_ff"] = [experiment_dict["d_model"][i] * 4 for i in range(num_experiments)]
    
    for i in experiment_indices:
        logging.info(
            f"\n[Experiment {i+1}/{num_experiments}] Starting experiment with GPU {gpu_id} "
            f"(pid={os.getpid()})..."
        )

        batch_size = experiment_dict["batch_size"][i]
        epochs = experiment_dict["epochs"][i]
        bias_scale = experiment_dict["bias_scale"][i]
        num_layers = experiment_dict["num_layers"][i]
        num_heads = experiment_dict["num_heads"][i]
        d_model = experiment_dict["d_model"][i]
        d_ff = experiment_dict["d_ff"][i]
        kernel_size = experiment_dict["kernel_size"][i]
        dataloader_workers = experiment_dict["dataloader_workers"][i]
        max_cached = experiment_dict["max_cached"][i]
        grad_attrib_batches = experiment_dict["grad_attrib_batches"][i]
        grad_attrib_tgs_per_batch = experiment_dict["grad_attrib_tgs_per_batch"][i]
        replicate = experiment_dict["replicates"][i]

        if previous_experiments_df is not None:
            config_match = (
                (previous_experiments_df["batch_size"] == batch_size) &
                (previous_experiments_df["epochs"] == epochs) &
                (previous_experiments_df["bias_scale"] == bias_scale) &
                (previous_experiments_df["num_layers"] == num_layers) &
                (previous_experiments_df["num_heads"] == num_heads) &
                (previous_experiments_df["d_model"] == d_model) &
                (previous_experiments_df["d_ff"] == d_ff) &
                (previous_experiments_df["kernel_size"] == kernel_size) &
                (previous_experiments_df["dataloader_workers"] == dataloader_workers) &
                (previous_experiments_df["max_cached"] == max_cached) &
                (previous_experiments_df["grad_attrib_batches"] == grad_attrib_batches) &
                (previous_experiments_df["grad_attrib_tgs_per_batch"] == grad_attrib_tgs_per_batch) &
                (previous_experiments_df["replicate"] == replicate)
            )

            if config_match.any():
                logging.info(f"[Experiment {i+1}] Experiment with this configuration already exists. Skipping...")
                continue
        
        exp.model_num = i + 1
        exp._create_model_training_dir(allow_overwrite=True)
        
        exp.create_multichrom_dataset(
            max_cached=max_cached,
        )

        train_loader, val_loader, test_loader = exp.prepare_dataloader(
            batch_size=batch_size,
            world_size=1,
            rank=0,
            num_workers=dataloader_workers,
            pin_memory=True,
        )

        exp.create_scalers(
            dataloader=train_loader,
        )

        exp.create_new_model(
            use_dist_bias=True,
            bias_scale=bias_scale,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=0.1,
            kernel_size=kernel_size,
            local_rank=0,
            rank=0,
            world_size=1,
        )
        
        logging.info(f"[Experiment {i+1}] Starting Training")
        train_start_time = time.time()
        exp.train(
            train_loader=train_loader, 
            val_loader=val_loader, 
            num_epochs=epochs,
            max_batches=None,
            verbose=False,
            grad_accum_steps=1,
            improvement_patience=15,
            save_every_n_epochs=10,
            monitor_gpu_memory=True,
            profile_batches=True,
            allow_overwrite=False,
            silence_tqdm=True,
            )
        train_end_time = time.time()
        logging.info(f"[Experiment {i+1}] Training finished in {train_end_time - train_start_time:.2f} seconds.")


        grad_attrib_start_time = time.time()
        exp.run_gradient_attribution(
            test_loader,
            max_batches=grad_attrib_batches,
            max_tgs_per_batch=grad_attrib_tgs_per_batch,
            )

        grad_attrib_end_time = time.time()
        logging.info(f"[Experiment {i+1}] Gradient attribution finished {grad_attrib_batches} batches in {grad_attrib_end_time - grad_attrib_start_time:.2f} seconds.")

        auroc_df = exp.calculate_auroc_all_sample_gts(exp.grn, gt_by_dataset_dict)    

        def update_dfs_with_experiment_params(df):
            df["experiment_name"] = exp.experiment_name
            df["sample_type"] = sample_type
            df["replicate"] = replicate
            for param, value in experiment_dict.items():
                df[param] = value[i]
            return df

        auroc_df = update_dfs_with_experiment_params(auroc_df)
        exp.gpu_mem_log_df = update_dfs_with_experiment_params(exp.gpu_mem_log_df)
        exp.batch_profile_df = update_dfs_with_experiment_params(exp.batch_profile_df)
        exp.epoch_log_df = update_dfs_with_experiment_params(exp.epoch_log_df)

        logging.info(f"[Experiment {i+1}] Aggregating results...")
        full_summary_df = aggregate_results(
            experiment_dict,
            auroc_df,
            exp.gpu_mem_log_df,
            exp.batch_profile_df,
            exp.epoch_log_df,
        )

        logging.info(f"[Experiment {i+1}] Saving results...")
        with lock:
            save_summary_df(full_summary_df, summary_save_path)

def run_muon_preprocessing(sample_name, sample_raw_data_dir, sample_processed_data_dir, tss_path, project_dir):
    logging.info("\n----- MUON PREPROCESSING -----")
    logging.info(f"Preprocessing data for sample {sample_name} using muon...")
    filtering_setting_df = pd.read_csv(project_dir / "dev" / "notebooks" / "muon_preprocessing" /"qc_filtering_settings.tsv", sep="\t")
    sample_filtering_settings = filtering_setting_df[filtering_setting_df["Sample"] == sample_name]    

    # ----- RNA QC thresholds -----
    MIN_CELLS_PER_GENE = muon_prep.get_threshold(sample_filtering_settings, "Min Cells per Gene")
    MIN_GENES_PER_CELL = muon_prep.get_threshold(sample_filtering_settings, "Min Genes per Cell")
    MAX_GENES_PER_CELL = muon_prep.get_threshold(sample_filtering_settings, "Max Genes per Cell")
    MIN_TOTAL_COUNTS = muon_prep.get_threshold(sample_filtering_settings, "Min Total Counts")
    MAX_TOTAL_COUNTS = muon_prep.get_threshold(sample_filtering_settings, "Max Total Counts")
    MAX_PCT_COUNTS_MT = muon_prep.get_threshold(sample_filtering_settings, "Max Pct MT")

    # ----- ATAC QC thresholds -----
    MIN_CELLS_PER_PEAK = muon_prep.get_threshold(sample_filtering_settings, "Min Cells per Peak")
    MIN_PEAKS_PER_CELL = muon_prep.get_threshold(sample_filtering_settings, "Min Peaks per Cell")
    MAX_PEAKS_PER_CELL = muon_prep.get_threshold(sample_filtering_settings, "Max Peaks per Cell")
    MIN_TOTAL_PEAK_COUNTS = muon_prep.get_threshold(sample_filtering_settings, "Min Total Peak Counts")
    MAX_TOTAL_PEAK_COUNTS = muon_prep.get_threshold(sample_filtering_settings, "Max Total Peak Counts")

    # Load the raw data using the types of files found in the sample raw data directory.
    mdata, frag_path = muon_prep.load_raw_data(sample_name, sample_raw_data_dir)

    # Write the loaded data to the processed data directory
    mdata.write(sample_processed_data_dir / f"{sample_name}.h5mu")
    
    data_processor = muon_prep.MudataProcessor(
        mdata=mdata,
        processed_data_dir=sample_processed_data_dir,
        sample_name=sample_name,
        tss_path=tss_path,
    )
    
    # RNA QC and Preprocessing
    logging.info("  - Processing RNA")
    data_processor.rna_qc_filter(
        min_cells_per_gene = MIN_CELLS_PER_GENE,
        min_genes_per_cell = MIN_GENES_PER_CELL,
        max_genes_per_cell = MAX_GENES_PER_CELL,
        min_total_counts_per_cell = MIN_TOTAL_COUNTS,
        max_total_counts_per_cell = MAX_TOTAL_COUNTS,
        max_pct_counts_mt = MAX_PCT_COUNTS_MT,
        norm_target_sum = 1e4,
        min_rna_disp = 0.5,
        filter_hvgs = False,
        tf_list_file = None,
        fig_dir=sample_processed_data_dir / "preprocessing_figures" / "rna_qc",
        )
    
    data_processor.rna_pca_and_neighbors(
        data_processor.rna, 
        n_pcs=20,
        n_neighbors=10,
        fig_dir=sample_processed_data_dir / "preprocessing_figures" / "rna_qc",
        )
    
    # ATAC QC and Preprocessing
    logging.info("  - Processing ATAC")
    data_processor.atac_qc_filter(
        min_cells_per_peak=MIN_CELLS_PER_PEAK,
        min_peaks_per_cell=MIN_PEAKS_PER_CELL,
        max_peaks_per_cell=MAX_PEAKS_PER_CELL,
        min_total_counts_per_cell=MIN_TOTAL_PEAK_COUNTS,
        max_total_counts_per_cell=MAX_TOTAL_PEAK_COUNTS,
        min_atac_disp=0.5,
        promoter_upstream=1000,
        promoter_downstream=100,
        distal_max=200_000,
        filter_hvgs=False,
        fig_dir=sample_processed_data_dir / "preprocessing_figures" / "atac_qc",
        )
    
    data_processor.nucleosome_signal(
        frag_path=frag_path, 
        fig_dir=sample_processed_data_dir / "preprocessing_figures" / "atac_qc"
        )
    
    logging.info("  - Calculating TSS enrichment")
    data_processor.tss_enrichment(
        frag_path=frag_path, 
        n_tss=500, 
        extend_upstream=1000, 
        extend_downstream=1000,
        fig_dir=sample_processed_data_dir / "preprocessing_figures" / "atac_qc"
        )
    
    # Save the processed data
    logging.info("  - Saving processed data")
    muon_prep.save_processed_data(data_processor.mdata, sample_processed_data_dir)
    
    # Integrate the RNA and ATAC modalities using MOFA+
    logging.info("  - Integrating RNA and ATAC modalities using MOFA+")
    muon_prep.integrate_rna_atac(
        data_processor.mdata, 
        sample_processed_data_dir, 
        sample_name, 
        fig_dir=sample_processed_data_dir / "integration_figures"
        )
    
    # Create metacells
    logging.info("  - Creating metacells")
    muon_prep.create_metacells(data_processor.mdata, sample_processed_data_dir, hops=2)
    logging.info("Muon Preprocessing complete.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning experiments for multiomic transformer.")
    parser.add_argument("--sample_type", type=str, default="mESC", help="Type of sample to run experiments on (e.g. mESC, Macrophage)")
    parser.add_argument("--sample_name", type=str, default="E7.5_rep1", help="Name of the sample to run experiments on (e.g. E7.5_rep1)")
    parser.add_argument("--experiment_header", type=str, default="auroc_by_kernel_size", help="Header to identify the experiment (e.g. auroc_by_kernel_size)")
    parser.add_argument("--n_chroms", type=int, default=19, help="Number of chromosomes to include in the dataset (default: 19 for mouse)")
    parser.add_argument("--organism_code", type=str, default="mm10", help="Organism code for genome annotation (default: mm10)")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to the raw data directory")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Path to the processed data directory")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the experiment directory where results will be saved")
    parser.add_argument("--training_cache_dir", type=str, required=True, help="Path to the training data cache directory")
    
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    sample_type = args.sample_type
    sample_name = args.sample_name
    experiment_header = args.experiment_header
    n_chroms = args.n_chroms
    organism_code = args.organism_code
    
    processed_data_dir = Path(args.processed_data_dir)
    raw_data_dir = Path(args.raw_data_dir)
    experiment_dir = Path(args.experiment_dir)
    training_cache_dir = Path(args.training_cache_dir)
    
    experiment_name = f"{sample_type}_{sample_name}_{experiment_header}"
    chrom_list = [f"chr{i}" for i in list(range(1,n_chroms+1))]
    
    logging.info("===== EXPERIMENT CONFIGURATION =====")
    logging.info(f"Experiment Name: {experiment_name}")
    logging.info(f"  - Sample Type: {sample_type}")
    logging.info(f"  - Sample Name: {sample_name}")
    logging.info(f"  - Experiment Header: {experiment_header}")
    logging.info(f"  - Number of Chromosomes: {n_chroms}")
    logging.info(f"  - Organism Code: {organism_code}")
    logging.info(f"  - Raw Data Directory: {raw_data_dir}")
    logging.info(f"  - Processed Data Directory: {processed_data_dir}")
    logging.info(f"  - Experiment Directory: {experiment_dir}")
    logging.info(f"  - Training Cache Directory: {training_cache_dir}\n")
    
    exp_processed_data_dir = processed_data_dir / experiment_name
    if not exp_processed_data_dir.is_dir():
        exp_processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_processed_data_dir = exp_processed_data_dir / sample_name
    sample_raw_data_dir = raw_data_dir / f"{sample_type}_10x_raw" / sample_name
    
    def missing_preprocessing_files(sample_processed_data_dir: Path):
        required_preprocessing_files = ["RE_pseudobulk.parquet","TG_pseudobulk.parquet"]
        
        for file_name in required_preprocessing_files:
            if not (sample_processed_data_dir / file_name).is_file():
                return True
    
    # Determine if we need to run the Muon preprocessing
    run_muon = False
    if not sample_processed_data_dir.exists():
        sample_processed_data_dir.mkdir(parents=True, exist_ok=True)
        run_muon = True
        
    if run_muon == False:
        if missing_preprocessing_files(sample_processed_data_dir):
            run_muon = True
    
    # Run Muon preprocessing if needed
    if run_muon == True:
        run_muon_preprocessing(
            sample_name=sample_name,
            sample_raw_data_dir=sample_raw_data_dir,
            sample_processed_data_dir=sample_processed_data_dir,
            tss_path=PROJECT_DIR / "data" / "genome_data" /"genome_annotation" / organism_code / f"gene_tss.bed",
            project_dir=PROJECT_DIR,
        )
    
    # Create or load the training data cache for this sample
    tdf = data_formatter.TrainingDataFormatter(
        project_dir=PROJECT_DIR,
        experiment_name=experiment_name,
        organism_code=organism_code,
        sample_names=[sample_name],
        chrom_list=chrom_list,
        output_dir=experiment_dir / experiment_name,
        processed_data_dir=exp_processed_data_dir,
        training_data_cache=training_cache_dir,
    )
    
    if tdf.settings_path.is_file():
        tdf.load_settings()
    
    tdf.create_or_load_data_cache(sample_name=sample_name, force_recalculate=False)

    logging.info("\nLoading ground truth datasets...")
    gt_by_dataset_dict = load_ground_truth_dict()

    experiment_dict = {
        "batch_size": [64],
        "epochs": [250],
        "bias_scale": [0.0, 1.0, 2.0],
        "num_layers": [1, 2, 3],
        "num_heads": [2, 4, 8],
        "d_model": [128, 192],
        "kernel_size": [64, 128],
        "dataloader_workers": [8],
        "max_cached": [100],
        "grad_attrib_batches": [None],
        "grad_attrib_tgs_per_batch": [None],
        "replicates": [1],
    }

    experiment_dict = expand_experiment_dict_grid(experiment_dict)
    num_experiments = [max(len(v) for v in experiment_dict.values())][0]
    logging.info(f"Total experiments to run: {num_experiments}")

    summary_save_path = PROJECT_DIR / "dev" / "notebooks" / "benchmarking_results" / f"{experiment_name}.csv"

    if torch.cuda.is_available():
        available_gpus = list(range(torch.cuda.device_count()))
    else:
        available_gpus = [0]

    if len(available_gpus) == 0:
        raise RuntimeError("No GPUs detected. Multiprocessing GPU assignment requires CUDA devices.")

    assignments = {gpu_id: [] for gpu_id in available_gpus}
    for idx in range(num_experiments):
        assignments[available_gpus[idx % len(available_gpus)]].append(idx)

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    lock = manager.Lock()
    processes = []

    for gpu_id, indices in assignments.items():
        if not indices:
            continue
        p = mp.Process(
            target=_run_experiments_on_gpu,
            args=(
                tdf,
                gpu_id,
                indices,
                lock,
                experiment_dict,
                num_experiments,
                summary_save_path,
                sample_type,
                gt_by_dataset_dict,
                experiment_dir,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

