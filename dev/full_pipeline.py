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
from PIL import Image

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

def combine_images_with_spans(
    layout,
    space,
    fig_dir,
    output_name="combined_images.png",
    bg_color=(255, 255, 255, 255),
):
    """
    Combine images into a manually positioned grid with optional row/column spanning.

    Parameters
    ----------
    layout : dict
        Dictionary mapping image path -> config dict

        Required keys per image:
            row : int
            col : int

        Optional keys per image:
            scale : float, default 1.0
            rowspan : int, default 1
            colspan : int, default 1

        Example:
        {
            Path("a.png"): {"row": 0, "col": 0, "scale": 1.0, "rowspan": 1, "colspan": 2},
            Path("b.png"): {"row": 0, "col": 2, "scale": 0.8},
            Path("c.png"): {"row": 1, "col": 0, "scale": 1.1, "rowspan": 2, "colspan": 2},
        }

    space : int
        Pixels between grid cells.
    fig_dir : Path
        Output directory.
    output_name : str
        Output filename.
    bg_color : tuple
        RGBA background color.

    Returns
    -------
    Path
        Path to the saved combined image.
    """
    items = []

    # Load images and parse layout
    for image_path, cfg in layout.items():
        row = cfg["row"]
        col = cfg["col"]
        scale = cfg.get("scale", 1.0)
        rowspan = cfg.get("rowspan", 1)
        colspan = cfg.get("colspan", 1)

        if rowspan < 1 or colspan < 1:
            raise ValueError(f"{image_path}: rowspan and colspan must be >= 1")

        img = Image.open(image_path).convert("RGBA")
        new_width = max(1, int(img.width * scale))
        new_height = max(1, int(img.height * scale))
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)

        items.append({
            "path": image_path,
            "row": row,
            "col": col,
            "rowspan": rowspan,
            "colspan": colspan,
            "img": img_resized,
            "width": new_width,
            "height": new_height,
        })

    if not items:
        raise ValueError("Layout is empty.")

    n_rows = max(item["row"] + item["rowspan"] for item in items)
    n_cols = max(item["col"] + item["colspan"] for item in items)

    # Check for overlapping occupied cells
    occupied = {}
    for item in items:
        for r in range(item["row"], item["row"] + item["rowspan"]):
            for c in range(item["col"], item["col"] + item["colspan"]):
                key = (r, c)
                if key in occupied:
                    raise ValueError(
                        f"Overlap detected: {item['path']} overlaps with {occupied[key]} at cell {key}"
                    )
                occupied[key] = item["path"]

    # Compute minimum column widths and row heights
    # Start from single-cell requirements
    col_widths = [0] * n_cols
    row_heights = [0] * n_rows

    for item in items:
        if item["colspan"] == 1:
            col_widths[item["col"]] = max(col_widths[item["col"]], item["width"])
        if item["rowspan"] == 1:
            row_heights[item["row"]] = max(row_heights[item["row"]], item["height"])

    # Expand widths/heights as needed for spanned items
    # A simple iterative pass is usually enough for plotting layouts
    for _ in range(10):
        changed = False

        for item in items:
            # Width across spanned columns, including inter-column spaces inside the span
            current_span_width = (
                sum(col_widths[item["col"]: item["col"] + item["colspan"]])
                + space * (item["colspan"] - 1)
            )
            if current_span_width < item["width"]:
                deficit = item["width"] - current_span_width
                add_per_col = deficit / item["colspan"]
                for c in range(item["col"], item["col"] + item["colspan"]):
                    new_val = int(round(col_widths[c] + add_per_col))
                    if new_val > col_widths[c]:
                        col_widths[c] = new_val
                        changed = True

            # Height across spanned rows, including inter-row spaces inside the span
            current_span_height = (
                sum(row_heights[item["row"]: item["row"] + item["rowspan"]])
                + space * (item["rowspan"] - 1)
            )
            if current_span_height < item["height"]:
                deficit = item["height"] - current_span_height
                add_per_row = deficit / item["rowspan"]
                for r in range(item["row"], item["row"] + item["rowspan"]):
                    new_val = int(round(row_heights[r] + add_per_row))
                    if new_val > row_heights[r]:
                        row_heights[r] = new_val
                        changed = True

        if not changed:
            break

    # Compute canvas size
    background_width = sum(col_widths) + space * (n_cols - 1)
    background_height = sum(row_heights) + space * (n_rows - 1)

    background = Image.new("RGBA", (background_width, background_height), bg_color)

    # Precompute grid start coordinates
    col_starts = []
    x = 0
    for w in col_widths:
        col_starts.append(x)
        x += w + space

    row_starts = []
    y = 0
    for h in row_heights:
        row_starts.append(y)
        y += h + space

    # Paste each image centered within its spanned rectangle
    for item in items:
        x0 = col_starts[item["col"]]
        y0 = row_starts[item["row"]]

        span_width = (
            sum(col_widths[item["col"]: item["col"] + item["colspan"]])
            + space * (item["colspan"] - 1)
        )
        span_height = (
            sum(row_heights[item["row"]: item["row"] + item["rowspan"]])
            + space * (item["rowspan"] - 1)
        )

        x_offset = (span_width - item["width"]) // 2
        y_offset = (span_height - item["height"]) // 2

        paste_x = x0 + x_offset
        paste_y = y0 + y_offset

        background.paste(item["img"], (paste_x, paste_y), item["img"])

    output_path = fig_dir / output_name
    background.save(output_path)
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full training and evaluation pipeline for the MultiomicTransformer model."
    )

    # Core experiment config
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment (should be unique per run)."
    )
    parser.add_argument(
        "--sample_name",
        type=str,
        required=True,
        help="Sample name."
    )
    parser.add_argument(
        "--organism_code",
        type=str,
        required=True,
        choices=["mm10", "hg38"],
        help="Organism code."
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        required=True,
        choices=["mESC", "Macrophage", "K562", "iPSC"],
        help="Sample type (used for ground truth selection)."
    )

    # Data inputs
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
        help="Directory containing raw data (used if raw_h5_data_file is NOT provided)."
    )

    parser.add_argument(
        "--raw_h5_data_file",
        type=str,
        default=None,
        help="Path to a specific .h5mu file (e.g., subsample). Overrides raw_data_dir."
    )

    # Output / cache
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        required=True,
        help="Directory for processed data."
    )
    parser.add_argument(
        "--training_data_cache_dir",
        type=str,
        required=True,
        help="Directory for caching training data."
    )
    parser.add_argument(
        "--experiment_output_dir",
        type=str,
        required=True,
        help="Directory to save experiment outputs."
    )

    args = parser.parse_args()

    if args.raw_h5_data_file is None and args.raw_data_dir is None:
        raise ValueError("You must provide either --raw_h5_data_file OR --raw_data_dir")

    if args.raw_h5_data_file is not None and args.raw_data_dir is not None:
        print("WARNING: Both raw_h5_data_file and raw_data_dir provided. Using raw_h5_data_file.")

    return args

if __name__ == "__main__":
    args = parse_args()
    
    # Path to the project directory (same as Git repository root)
    project_dir = Path(PROJECT_DIR)
    raw_data_dir = Path(args.raw_data_dir)
    processed_data_dir = Path(args.processed_data_dir)
    training_data_cache_dir = Path(args.training_data_cache_dir)
    raw_h5_data_file = Path(args.raw_h5_data_file) if args.raw_h5_data_file is not None else None
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

        # Load either explicit raw_h5_data_file (preferred for stability runs) or scan sample directory files.
        mdata, frag_path = muon_prep.load_raw_data(
            sample_name=sample_name,
            sample_data_dir=sample_raw_data_dir,
            raw_h5_file=raw_h5_data_file,
        )

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
        
        if frag_path is not None:
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
    
    # Verify that the data cache files exist. If not, this method will create them.
    logging.info("  - Creating or verifying loading cache data files...")
    tdf.create_or_load_data_cache(sample_name=sample_name, force_recalculate=True)

    logging.info("\n----- MODEL TRAINING -----")
    logging.info("  - Initializing ExperimentHandler...")
    exp = experiment_handler.ExperimentHandler(
        training_data_formatter=tdf,
        experiment_dir=experiment_dir,
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
        batch_size=64,
        num_workers=8
    )

    # Creates scalers for the RNA and ATAC data based on the training split.
    logging.info("  - Creating scalers...")
    exp.create_scalers(train_loader)

    # Creates a new MultiomicTransformer model. Model attributes can be set to change
    # the hyperparameters of the model.
    logging.info("  - Creating model")
    exp.create_new_model(kernel_size=128)
    
    if exp.model_training_dir.is_dir() and "trained_model.pt" in os.listdir(exp.model_training_dir):
        logging.info(f"Trained model already exists. Skipping training...")
        exp.load_model()
        exp.load_handler()

    else:
        exp._create_model_training_dir(allow_overwrite=True)
        
        exp.train(
            train_loader=train_loader, 
            val_loader=val_loader, 
            num_epochs=500,
            max_batches=None,
            verbose=True,
            grad_accum_steps=1,
            improvement_patience=15,
            save_every_n_epochs=10,
            monitor_gpu_memory=True,
            profile_batches=True,
            allow_overwrite=False,
            silence_tqdm=True,
            )

    if "inferred_grn.csv" in os.listdir(exp.model_training_dir):
        logging.info(f"Inferred GRN already exists. Skipping gradient attribution...")
        exp.grn = exp.load_grn()
    else:
        logging.info("\n\n----- GRADIENT ATTRIBUTION -----")
        logging.info(f"Starting gradient attribution for GRN inference")
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
        grn=exp.grn,
        use_muon_grn=True,
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
    per_tf_plot.savefig(fig_dir / f"top_10_per_tf_auroc.png", dpi=250, bbox_inches="tight")
    plt.close(per_tf_plot)

    pooled_auroc_boxplot_fig = exp._plot_all_results_auroc_boxplot(
        pooled_df_all,
        per_tf=False,
        ylim=(0.2, 0.8),
        override_title=f"Pooled AUROC per Method",
        method_color_dict=exp.method_color_dict
    )
    pooled_auroc_boxplot_fig.savefig(fig_dir / f"pooled_auroc_boxplot.png", dpi=250, bbox_inches="tight")
    plt.close(pooled_auroc_boxplot_fig)

    per_tf_auroc_boxplot_fig = exp._plot_all_results_auroc_boxplot(
        per_tf_plot_df, 
        per_tf=True,
        ylim=(0.2, 0.8),
        override_title=f"Per-TF AUROC per Method",
        method_color_dict=exp.method_color_dict
        )
    per_tf_auroc_boxplot_fig.savefig(fig_dir / f"per_tf_auroc_boxplot.png", dpi=250, bbox_inches="tight")
    plt.close(per_tf_auroc_boxplot_fig)

    relative_improvement_fig = exp.plot_relative_improvement(
        per_tf_plot_df, 
        exp.experiment_name,
        override_title=f"Per-TF AUROC Improvement",
        )
    relative_improvement_fig.savefig(fig_dir / f"relative_improvement.png", dpi=250, bbox_inches="tight")
    plt.close(relative_improvement_fig)

    pooled_auroc_heatmap_fig = exp.plot_method_gt_heatmap(
        pooled_df_all, 
        per_tf=False,
        x_scale=1.2,
        y_scale=0.6,
        override_title=f"Pooled AUROC by Method and GT"
        )
    pooled_auroc_heatmap_fig.savefig(fig_dir / f"pooled_auroc_heatmap.png", dpi=250, bbox_inches="tight")
    plt.close(pooled_auroc_heatmap_fig)

    per_tf_auroc_heatmap_fig = exp.plot_method_gt_heatmap(
        per_tf_plot_df, 
        per_tf=True,
        x_scale=1.2,
        y_scale=0.6,
        override_title=f"Per-TF AUROC by Method and GT"
        )
    per_tf_auroc_heatmap_fig.savefig(fig_dir / f"per_tf_auroc_heatmap.png", dpi=250, bbox_inches="tight")
    plt.close(per_tf_auroc_heatmap_fig)

    true_vs_predicted_fig = exp.plot_true_vs_predicted_tg_expression(
        num_batches=50, 
        set_axis_logscale=False,
        title=f"Predicted vs True TG Expression"
        )
    true_vs_predicted_fig.savefig(fig_dir / f"true_vs_predicted.png", dpi=250, bbox_inches="tight")
    plt.close(true_vs_predicted_fig)

    logging.info(f"\n----- Saving Experiment -----")
    exp.save_handler()
    logging.info(f"Experiment saved to {exp.experiment_dir}")
    
    layout = {
        fig_dir / "top_10_per_tf_auroc.png": {"row": 0, "col": 0, "scale": 0.9, "rowspan": 1, "colspan": 2},
        fig_dir / "pooled_auroc_boxplot.png": {"row": 0, "col": 2, "scale": 0.9},
        fig_dir / "pooled_auroc_heatmap.png": {"row": 0, "col": 3, "scale": 1.0},
        fig_dir / "true_vs_predicted.png": {"row": 1, "col": 0, "scale": 0.85},
        fig_dir / "relative_improvement.png": {"row": 1, "col": 1, "scale": 0.85},
        fig_dir / "per_tf_auroc_boxplot.png": {"row": 1, "col": 2, "scale": 0.9},
        fig_dir / "per_tf_auroc_heatmap.png": {"row": 1, "col": 3, "scale": 1.0},
    }

    combine_images_with_spans(
        layout=layout,
        space=50,
        fig_dir=fig_dir,
        output_name="combined_images.png",
    )