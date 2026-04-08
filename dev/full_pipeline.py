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
import argparse

import sys
PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
SRC_DIR = str(Path(PROJECT_DIR) / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
    
DEV_DIR = str(Path(PROJECT_DIR) / "dev")
if DEV_DIR not in sys.path:
    sys.path.insert(0, DEV_DIR)
    
import muon_preprocessing as muon_prep

import multiomic_transformer.utils.data_formatter as data_formatter
import multiomic_transformer.utils.experiment_handler as experiment_handler

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the full training and evaluation pipeline for the MultiomicTransformer model.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment to run. If not provided, a default name will be generated based on the sample names.")
    parser.add_argument("--sample_name", type=str, required=True, help="Name of the sample to include in the training dataset.")
    parser.add_argument("--organism_code", type=str, required=True, choices=["mm10", "hg38"], help="Organism code for the dataset. Should be either 'mm10' or 'hg38'.")
    parser.add_argument("--sample_type", type=str, required=True, choices=["mESC", "Macrophage", "K562", "iPSC"], help="Type of sample being used. This is used to determine which ground truth datasets to compare against. Should be one of 'mESC', 'Macrophage', 'K562', or 'iPSC'.")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Directory containing the raw data files.")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Directory containing the processed data files. Each sample should have its own subdirectory in this directory.")
    parser.add_argument("--training_data_cache_dir", type=str, required=True, help="Directory to use for caching the training data.")
    parser.add_argument("--experiment_output_dir", type=str, required=True, help="Directory to save the experiment outputs.")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # Path to the project directory (same as Git repository root)
    project_dir = Path(PROJECT_DIR)
    raw_data_dir = Path(args.raw_data_dir)
    processed_data_dir = Path(args.processed_data_dir)
    training_data_cache_dir = Path(args.training_data_cache_dir)
    
    # Path to the training output directory. Used to store the preprocessing config
    experiment_dir = Path(args.experiment_output_dir)

    # List of samples in the training datset. 
    # Each of these should have its own subdirectory in the processed data directory
    sample_names = [args.sample_name]
    sample_name = sample_names[0]
    
    # Name of the dataset / experiment to run
    experiment_name = args.experiment_name

    # Organism code for the dataset. Supports either "mm10" or "hg38"
    organism_code = args.organism_code
    
    sample_type = args.sample_type
    
    # Path to the list of gene TSS locations, used to locate the nearest gene to each ATAC peak.
    tss_path=f"{project_dir}/data/genome_data/genome_annotation/{organism_code}/gene_tss.bed"

    # List of chromosomes. Used to split the data by chromsome for caching and training.
    # Should be in the format "chr1", "chr2", etc. and should match the chromosome names in the processed data files.
    if organism_code == "mm10":
        chrom_list = [f"chr{i}" for i in range(1, 20)]
    elif organism_code == "hg38":
        chrom_list = [f"chr{i}" for i in range(1, 23)]
    
    # ===== MUON DATA PREPROCESSING =====
    # Create the processed data directory for the experiment if it doesn't already exist
    processed_data_dir = processed_data_dir / experiment_name
    if not processed_data_dir.is_dir():
        processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the processed data directory for the sample if it doesn't already exist
    sample_processed_data_dir = processed_data_dir / sample_name
    sample_raw_data_dir = raw_data_dir / f"{sample_type}_10x_raw" / sample_name
    
    def missing_preprocessing_files(sample_processed_data_dir: Path):
        required_preprocessing_files = ["RE_pseudobulk.parquet","TG_pseudobulk.parquet"]
        
        for file_name in required_preprocessing_files:
            if not (sample_processed_data_dir / file_name).is_file():
                return True
            
    run_muon = False
    if not sample_processed_data_dir.exists():
        sample_processed_data_dir.mkdir(parents=True, exist_ok=True)
        run_muon = True
        
    if run_muon == False:
        if missing_preprocessing_files(sample_processed_data_dir):
            run_muon = True
    
    if run_muon:
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
        logging.info("  - Calculating RNA QC metrics and filtering...")
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
        
        logging.info("  - Calculating RNA PCA and neighbors...")
        data_processor.rna_pca_and_neighbors(
            data_processor.rna, 
            n_pcs=20,
            n_neighbors=10,
            fig_dir=sample_processed_data_dir / "preprocessing_figures" / "rna_qc",
            )
        
        # ATAC QC and Preprocessing
        logging.info("  - Calculating ATAC QC metrics and filtering...")
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
        
        logging.info("  - Calculating ATAC QC metrics...")
        data_processor.nucleosome_signal(
            frag_path=frag_path, 
            fig_dir=sample_processed_data_dir / "preprocessing_figures" / "atac_qc"
            )
        
        logging.info("  - Calculating TSS enrichment...")
        data_processor.tss_enrichment(
            frag_path=frag_path, 
            n_tss=500, 
            extend_upstream=1000, 
            extend_downstream=1000,
            fig_dir=sample_processed_data_dir / "preprocessing_figures" / "atac_qc"
            )
        
        # Save the processed data
        logging.info("  - Saving processed data...")
        muon_prep.save_processed_data(data_processor.mdata, sample_processed_data_dir)
        
        # Integrate the RNA and ATAC modalities using MOFA+
        logging.info("  - Integrating RNA and ATAC modalities using MOFA+...")
        muon_prep.integrate_rna_atac(
            data_processor.mdata, 
            sample_processed_data_dir, 
            sample_name, 
            fig_dir=sample_processed_data_dir / "integration_figures"
            )
        
        # Create metacells
        logging.info("  - Creating metacells...")
        muon_prep.create_metacells(data_processor.mdata, sample_processed_data_dir, hops=2)
        logging.info("Muon Preprocessing complete.")
    else:
        logging.info(f"Processed data for sample {sample_name} already exists. Skipping muon preprocessing.")

    # ===== DATA CACHE CREATION =====
    logging.info("\n----- DATA CACHE CREATION -----")
    logging.info("  - Initializing training data formatter...")
    tdf = data_formatter.TrainingDataFormatter(
        project_dir=project_dir,
        experiment_name=experiment_name,
        organism_code=organism_code,
        sample_names=sample_names,
        chrom_list=chrom_list,
        output_dir=experiment_dir / experiment_name,
        processed_data_dir=processed_data_dir,
        training_data_cache=training_data_cache_dir
    )
    
    if tdf.settings_path.is_file():
        logging.info(f"  - Loading existing preprocessing config from {tdf.settings_path}...")
        tdf.load_settings()
    
    # Verify that the data cache files exist. If not, this method will create them.
    logging.info("  - Creating or verifying loading cache data files...")
    tdf.create_or_load_data_cache(sample_name=sample_name, force_recalculate=False)

    logging.info("\n----- MODEL TRAINING -----")
    logging.info("  - Initializing ExperimentHandler...")
    exp = experiment_handler.ExperimentHandler(
        training_data_formatter=tdf,
        experiment_dir=experiment_dir / experiment_name,
        model_num=1,
        silence_warnings=False,
    )
    
    # Creates a MultiChromosomeDataset dataset class, which handles loading data for one chromosome
    # at a time and caching it in memory. The max_cached argument controls how many chromosomes worth 
    # of data can be cached in memory at once. Each batch only contains TG and window data from one chromosome.
    # The chromosomes are loaded sequentially, starting with the first chromosome in the chrom_list.
    logging.info("  - Creating dataset...")
    exp.create_multichrom_dataset(max_cached=100)

    # Prepares the Train/Val/Test dataloaders, being careful to balance the number of 
    # batches from each chromosome in each set.
    logging.info("  - Preparing DataLoader...")
    train_loader, val_loader, test_loader = exp.prepare_dataloader(
        batch_size=32,
        num_workers=8
    )

    # Creates scalers for the RNA and ATAC data based on the training split.
    logging.info("  - Creating scalers...")
    exp.create_scalers(train_loader)

    # Creates a new MultiomicTransformer model. Model attributes can be set to change
    # the hyperparameters of the model.
    logging.info("  - Creating model")
    exp.create_new_model(kernel_size=64)

    # Runs model training and returns the trained model.
    logging.info("  - Training model")
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
        allow_overwrite=False,
        silence_tqdm=True,
        )

    logging.info("\n\n----- GRADIENT ATTRIBUTION -----")
    # Runs gradient attribution to calculate the gradients between each TF input and each TG output.
    logging.info("  - Running Gradient Attribution")
    exp.run_gradient_attribution(
        test_loader,
        max_batches=None,
        max_tgs_per_batch=None,
        )
    
    # ===== GROUND TRUTH LOADING AND AUROC CALCULATION =====
    logging.info("\n\n----- GROUND TRUTH LOADING AND AUROC CALCULATION -----")
    logging.info("  - Loading ground truth datasets...")
    GROUND_TRUTH_DIR = Path(PROJECT_DIR) / "data" / "ground_truth_files"
    
    gt_by_dataset_dict_all = {
        "Macrophage": {
            # "RN204": load_ground_truth(GROUND_TRUTH_DIR / "rn204_macrophage_human_chipseq.tsv"),
            "ChIP-Atlas macrophage": exp.load_ground_truth(GROUND_TRUTH_DIR / "chipatlas_macrophage.csv"),
        },
        "mESC": {
            "ChIP-Atlas mESC": exp.load_ground_truth(GROUND_TRUTH_DIR / "chip_atlas_tf_peak_tg_dist.csv"),
            "RN111": exp.load_ground_truth(GROUND_TRUTH_DIR / "RN111.tsv"),
            "RN112": exp.load_ground_truth(GROUND_TRUTH_DIR / "RN112.tsv"),
            "RN114": exp.load_ground_truth(GROUND_TRUTH_DIR / "RN114.tsv"),
            "RN116": exp.load_ground_truth(GROUND_TRUTH_DIR / "RN116.tsv"),        
        },
        "K562": {
            "ChIP-Atlas K562": exp.load_ground_truth(GROUND_TRUTH_DIR / "chipatlas_K562.csv"),
            "RN117": exp.load_ground_truth(GROUND_TRUTH_DIR / "RN117.tsv"),        
        },
        "iPSC": {
            # "ChIP-Atlas iPSC": exp.load_ground_truth(GROUND_TRUTH_DIR / "chipatlas_iPSC.csv"),
            "ChIP-Atlas iPSC (1 Mb)": exp.load_ground_truth(GROUND_TRUTH_DIR / "chipatlas_iPSC_1mb.csv"),
            # "ChIP-Atlas iPSC (100 kb)": exp.load_ground_truth(GROUND_TRUTH_DIR / "chipatlas_iPSC_100kb.csv"),
        }
    }
    gt_by_dataset_dict = gt_by_dataset_dict_all[sample_type]

    # Calculates the AUROC of the predicted GRN against multiple ground truth datasets.
    logging.info("  - Calculating AUROC")
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

    logging.info(f"\n\n--- Model Training Summary ---")
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

    logging.info(f"\n----- GRN Overlap with Ground Truth -----")
    for ground_truth_name, ground_truth in gt_by_dataset_dict.items():
        exp.report_grn_overlap_with_gt(ground_truth_name, ground_truth)

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