import os
import json
import torch
import pandas as pd
import logging
import importlib
from pathlib import Path
import numpy as np
import random
import matplotlib.pyplot as plt

import sys
PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
SRC_DIR = str(Path(PROJECT_DIR) / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import multiomic_transformer.utils.data_formatter as data_formatter
import multiomic_transformer.utils.experiment_handler as experiment_handler

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)

if __name__ == "__main__":
    # Path to the project directory (same as Git repository root)
    project_dir = Path(PROJECT_DIR)

    # Path to the training output directory. Used to store the preprocessing config
    output_dir = Path("/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments")

    # List of samples in the training datset. 
    # Each of these should have its own subdirectory in the processed data directory
    sample_names = ["buffer_2"]
    
    # Name of the dataset / experiment to run
    experiment_name = f"Macrophage_{sample_names[0]}_raw_muon_preprocessing"

    # Organism code for the dataset. Supports either "mm10" or "hg38"
    organism_code = "hg38"
    
    sample_type = "Macrophage"
    ground_truth_name = "ChIP-Atlas macrophage"

    # List of chromosomes. Used to split the data by chromsome for caching and training.
    # Should be in the format "chr1", "chr2", etc. and should match the chromosome names in the processed data files.
    chrom_list = [f"chr{i}" for i in range(1, 22)]

    logging.info("Initializing training data formatter...")
    tdf = data_formatter.TrainingDataFormatter(
        project_dir=project_dir,
        experiment_name=experiment_name,
        organism_code=organism_code,
        sample_names=sample_names,
        chrom_list=chrom_list,
        output_dir=output_dir / experiment_name,
    )
    
    if tdf.settings_path.is_file():
        logging.info(f"Loading existing preprocessing config from {tdf.settings_path}...")
        tdf.load_settings()
    
    # Verify that the data cache files exist. If not, this method will create them.
    logging.info("Creating or verifying loading cache data files...")
    tdf.create_or_load_data_cache(sample_name=sample_names[0], force_recalculate=True)

    logging.info("Initializing ExperimentHandler...")
    exp = experiment_handler.ExperimentHandler(
        training_data_formatter=tdf,
        experiment_dir="/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/experiments/",
        model_num=1,
        silence_warnings=False,
    )
    

    logging.info("Loading ground truth datasets...")
    GROUND_TRUTH_DIR = Path(PROJECT_DIR) / "data" / "ground_truth_files"
    
    gt_by_dataset_dict = {
        "ChIP-Atlas macrophage": exp.load_ground_truth(GROUND_TRUTH_DIR / "chipatlas_macrophage.csv"),
    }
    
    # Creates a MultiChromosomeDataset dataset class, which handles loading data for one chromosome
    # at a time and caching it in memory. The max_cached argument controls how many chromosomes worth 
    # of data can be cached in memory at once. Each batch only contains TG and window data from one chromosome.
    # The chromosomes are loaded sequentially, starting with the first chromosome in the chrom_list.
    logging.info("Creating dataset...")
    exp.create_multichrom_dataset(max_cached=100)

    # Prepares the Train/Val/Test dataloaders, being careful to balance the number of 
    # batches from each chromosome in each set.
    logging.info("Preparing DataLoader...")
    train_loader, val_loader, test_loader = exp.prepare_dataloader(
        batch_size=32,
        num_workers=8
    )

    # Creates scalers for the RNA and ATAC data based on the training split.
    logging.info("Creating scalers...")
    exp.create_scalers(train_loader)

    # Creates a new MultiomicTransformer model. Model attributes can be set to change
    # the hyperparameters of the model.
    logging.info("Creating model")
    exp.create_new_model(kernel_size=64)

    # Runs model training and returns the trained model.
    logging.info("Training model")
    model = exp.train(
        train_loader=train_loader, 
        val_loader=val_loader, 
        num_epochs=500,
        max_batches=None,
        grad_accum_steps=1,
        improvement_patience=15,
        save_every_n_epochs=10,
        monitor_gpu_memory=True,
        profile_batches=True,
        allow_overwrite=True,
        silence_tqdm=True,
        )

    # Runs gradient attribution to calculate the gradients between each TF input and each TG output.
    logging.info("\nRunning Gradient Attribution")
    exp.run_gradient_attribution(
        test_loader,
        max_batches=None,
        max_tgs_per_batch=None,
        )

    # Calculates the AUROC of the predicted GRN against multiple ground truth datasets.
    logging.info("\nCalculating AUROC")
    auroc_df = exp.calculate_auroc_all_sample_gts(exp.grn, gt_by_dataset_dict)     
    logging.info(f"Pooled Median AUROC: {auroc_df['pooled_median_auroc'].iloc[0]:.3f}")       
    logging.info(f"Per-TF Median AUROC: {auroc_df['per_tf_median_auroc'].iloc[0]:.3f}")
    
    exp.calculate_auroc_all_methods(
        sample_names, 
        sample_type, 
        gt_by_dataset_dict, 
        grn=exp.grn
    )    
    
    exp.epoch_log_df.iloc[-1]
    final_r2u = exp.epoch_log_df.iloc[-1]["r2_unscaled"]
    final_r2s = exp.epoch_log_df.iloc[-1]["r2_scaled"]
    avg_epoch_time = exp.epoch_log_df["epoch_time_s"].mean()

    logging.info(f"\n--- Model Training Summary ---")
    logging.info(f"Final R2 (unscaled): {final_r2u:.4f}")
    logging.info(f"Final R2 (scaled): {final_r2s:.4f}")
    logging.info(f"Average Epoch Time: {avg_epoch_time:.2f} seconds")
    logging.info(f"Number of Metacells: {exp.tdf.num_metacells:,}")
    logging.info(f"Number of Windows: {exp.tdf.num_windows:,}\n")

    num_edges = len(exp.grn)
    num_tfs = exp.grn["Source"].nunique()
    num_tgs = exp.grn["Target"].nunique()

    logging.info(f"\n----- GRN Size -----")
    logging.info(f"Number of unique TFs: {num_tfs:,}")
    logging.info(f"Number of unique TGs: {num_tgs:,}")
    logging.info(f"Number of edges: {num_edges:,}")

    exp.report_grn_overlap_with_gt(ground_truth_name, gt_by_dataset_dict)

    logging.info(f"\n----- AUROC -----")
    per_tf_df_all =  pd.read_csv(exp.model_training_dir / "per_tf_auroc_auprc_results.csv")
    pooled_df_all = pd.read_csv(exp.model_training_dir / "pooled_auroc_auprc_results.csv")

    per_tf_plot_df = (
        per_tf_df_all.dropna(subset=["auroc"])
        .groupby(['method', 'gt'], as_index=False)
        .agg(
            auroc=('auroc', 'median'),
        )
    )

    pooled_df_median_auroc = pooled_df_all[pooled_df_all["method"] == "Gradient Attribution"]["auroc"].median()
    per_tf_median_auroc = per_tf_plot_df[per_tf_plot_df["method"] == "Gradient Attribution"]["auroc"].median()

    logging.info(f"Median Pooled AUROC: {pooled_df_median_auroc:.3f}")
    logging.info(f"Median Per-TF AUROC: {per_tf_median_auroc:.3f}")

    fig_dir = exp.model_training_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    name = exp.experiment_name.replace("_", " ")
    per_tf_plot, per_tf_df, tf_curves = exp.plot_top_n_tf_roc_curves(
        exp.grn, 
        gt_by_dataset_dict[ground_truth_name], 
        ground_truth_name, 
        exp, 
        method_name="Gradient Attribution", 
        num_top_tfs_to_plot=10,
        min_edges=500,
        min_pos=50,
        balance=True,
        name_clean=name,
        override_title=f"Top 10 Per-TF AUROC"
        )
    per_tf_plot.savefig(fig_dir / f"top_10_per_tf_auroc.png", dpi=250)
    plt.close(per_tf_plot)

    pooled_auroc_boxplot_fig = exp._plot_all_results_auroc_boxplot(
        pooled_df_all,
        per_tf=False,
        ylim=(0.2, 0.8),
        override_title=f"Pooled AUROC per Method",
        method_color_dict=exp.method_color_dict
    )
    pooled_auroc_boxplot_fig.savefig(fig_dir / f"pooled_auroc_boxplot.png", dpi=250)
    plt.close(pooled_auroc_boxplot_fig)

    per_tf_auroc_boxplot_fig = exp._plot_all_results_auroc_boxplot(
        per_tf_plot_df, 
        per_tf=True,
        ylim=(0.2, 0.8),
        override_title=f"Per-TF AUROC per Method",
        method_color_dict=exp.method_color_dict
        )
    per_tf_auroc_boxplot_fig.savefig(fig_dir / f"per_tf_auroc_boxplot.png", dpi=250)
    plt.close(per_tf_auroc_boxplot_fig)

    relative_improvement_fig = exp.plot_relative_improvement(
        per_tf_plot_df, 
        exp.experiment_name,
        override_title=f"Per-TF AUROC Improvement",
        )
    relative_improvement_fig.savefig(fig_dir / f"relative_improvement.png", dpi=250)
    plt.close(relative_improvement_fig)

    pooled_auroc_heatmap_fig = exp.plot_method_gt_heatmap(
        pooled_df_all, 
        per_tf=False,
        x_scale=1.2,
        y_scale=0.6,
        override_title=f"Pooled AUROC by Method and GT"
        )
    pooled_auroc_heatmap_fig.savefig(fig_dir / f"pooled_auroc_heatmap.png", dpi=250)
    plt.close(pooled_auroc_heatmap_fig)

    per_tf_auroc_heatmap_fig = exp.plot_method_gt_heatmap(
        per_tf_plot_df, 
        per_tf=True,
        x_scale=1.2,
        y_scale=0.6,
        override_title=f"Per-TF AUROC by Method and GT"
        )
    per_tf_auroc_heatmap_fig.savefig(fig_dir / f"per_tf_auroc_heatmap.png", dpi=250)
    plt.close(per_tf_auroc_heatmap_fig)

    true_vs_predicted_fig = exp.plot_true_vs_predicted_tg_expression(
        num_batches=50, 
        set_axis_logscale=False,
        title=f"Predicted vs True TG Expression"
        )
    true_vs_predicted_fig.savefig(fig_dir / f"true_vs_predicted.png", dpi=250)
    plt.close(true_vs_predicted_fig)

    logging.info(f"\n----- Saving Experiment -----")
    exp.save_handler()
    logging.info(f"Experiment saved to {exp.experiment_dir}")