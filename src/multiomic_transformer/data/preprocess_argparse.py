import os
import re
import json
import torch
import pandas as pd
import scanpy as sc
import math
import logging
from pathlib import Path
import warnings
import numpy as np
import traceback
import scipy.sparse as sp
import random
from scipy.special import softmax
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Set, Optional, List, Iterable, Union, Dict, Callable, Any
from anndata import AnnData
import pybedtools
import argparse
import pickle
from sklearn.decomposition import TruncatedSVD

import sys
sys.path.append(Path(__file__).resolve().parent.parent.parent)

from multiomic_transformer.utils.standardize import standardize_name
from multiomic_transformer.utils.files import atomic_json_dump
from multiomic_transformer.utils.peaks import find_genes_near_peaks, format_peaks
from multiomic_transformer.utils.downloads import *
from multiomic_transformer.data.sliding_window import run_sliding_window_scan
from multiomic_transformer.utils.gene_canonicalizer import GeneCanonicalizer

random.seed(1337)
np.random.seed(1337)
torch.manual_seed(1337)

# ----- Argument Parser Setup -----
def parse_preprocessing_args():
    """
    Parse command-line arguments for preprocessing configuration.
    All global settings can be overridden via command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess single-cell multiomics data for MultiomicTransformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ----- Required Arguments -----
    parser.add_argument("--num_cpu", type=int, required=True,
                        help="Number of CPU cores for parallel processing")
    
    # ----- Path Configuration -----
    parser.add_argument("--root_dir", type=Path, default=None,
                        help="Root directory of the project")
    parser.add_argument("--project_data_dir", type=Path, default=None,
                        help="Project data directory")
    parser.add_argument("--project_result_dir", type=Path, default=None,
                        help="Project results directory")
    
    # ----- Sample Information -----
    parser.add_argument("--organism_code", type=str, default="mm10",
                        choices=["mm10", "hg38"],
                        help="Organism code (mm10 for mouse, hg38 for human)")
    parser.add_argument("--dataset_name", type=str, default="mESC_default",
                        help="Name of the dataset/experiment")
    parser.add_argument("--chrom_id", type=str, default="chr19",
                        help="Single chromosome ID for processing")
    parser.add_argument("--chrom_ids", type=str, nargs="+", default=None,
                        help="List of chromosome IDs for multi-chromosome processing")
    parser.add_argument("--sample_names", type=str, nargs="+", default=None,
                        help="List of sample names to process")
    parser.add_argument("--fine_tuning_datasets", type=str, nargs="+", default=None,
                        help="List of datasets for fine-tuning")
    
    # ----- Raw Data Paths -----
    parser.add_argument("--raw_single_cell_data", type=Path, default=None,
                        help="Path to raw single-cell data directory")
    parser.add_argument("--raw_10x_rna_data_dir", type=Path, default=None,
                        help="Path to raw 10x RNA data directory")
    parser.add_argument("--raw_atac_peak_matrix_file", type=Path, default=None,
                        help="Path to raw ATAC peak matrix file")
    parser.add_argument("--raw_gse218576_dir", type=Path, default=None,
                        help="Path to raw GSE218576 data directory")
    parser.add_argument("--processed_gse218576_dir", type=Path, default=None,
                        help="Path to processed GSE218576 data directory")
    
    # ----- QC Filtering Parameters -----
    parser.add_argument("--min_genes_per_cell", type=int, default=200,
                        help="Minimum number of genes expressed per cell")
    parser.add_argument("--min_peaks_per_cell", type=int, default=200,
                        help="Minimum number of peaks expressed per cell")
    parser.add_argument("--filter_type", type=str, default="count",
                        choices=["count", "pct"],
                        help="Filter type: 'count' or 'pct' (percentage)")
    parser.add_argument("--filter_out_lowest_counts_genes", type=int, default=3,
                        help="Filter out genes expressed in fewer than this many cells")
    parser.add_argument("--filter_out_lowest_counts_peaks", type=int, default=3,
                        help="Filter out peaks expressed in fewer than this many cells")
    parser.add_argument("--filter_out_lowest_pct_genes", type=float, default=0.1,
                        help="Filter out genes expressed in less than this percentage of cells")
    parser.add_argument("--filter_out_lowest_pct_peaks", type=float, default=0.01,
                        help="Filter out peaks expressed in less than this percentage of cells")
    
    # ----- File Naming Configuration -----
    parser.add_argument("--processed_rna_filename", type=str, default="scRNA_seq_processed.parquet",
                        help="Filename for processed RNA data")
    parser.add_argument("--processed_atac_filename", type=str, default="scATAC_seq_processed.parquet",
                        help="Filename for processed ATAC data")
    parser.add_argument("--raw_rna_file", type=str, default="scRNA_seq_raw.parquet",
                        help="Filename for raw RNA data")
    parser.add_argument("--raw_atac_file", type=str, default="scATAC_seq_raw.parquet",
                        help="Filename for raw ATAC data")
    parser.add_argument("--adata_rna_file", type=str, default="adata_RNA.h5ad",
                        help="Filename for RNA AnnData object")
    parser.add_argument("--adata_atac_file", type=str, default="adata_ATAC.h5ad",
                        help="Filename for ATAC AnnData object")
    parser.add_argument("--pseudobulk_tg_file", type=str, default="TG_pseudobulk.tsv",
                        help="Filename for target gene pseudobulk data")
    parser.add_argument("--pseudobulk_re_file", type=str, default="RE_pseudobulk.tsv",
                        help="Filename for regulatory element pseudobulk data")
    
    # ----- Pseudobulk and Preprocessing Parameters -----
    parser.add_argument("--neighbors_k", type=int, default=20,
                        help="Number of nearest neighbors per cell in KNN graph")
    parser.add_argument("--pca_components", type=int, default=25,
                        help="Number of PCA components per modality")
    parser.add_argument("--hops", type=int, default=0,
                        help="Number of diffusion hops for soft metacells")
    parser.add_argument("--self_weight", type=float, default=1.0,
                        help="Self-loop weight in neighborhood graph")
    
    # ----- Data Preprocessing and Caching -----
    parser.add_argument("--validation_datasets", type=str, nargs="+", default=None,
                        help="List of validation dataset names")
    parser.add_argument("--force_recalculate", action="store_true",
                        help="Force recalculation of all cached data")
    parser.add_argument("--window_size", type=int, default=1000,
                        help="Size of genomic windows in base pairs")
    parser.add_argument("--distance_scale_factor", type=int, default=20000,
                        help="Scale factor for peak-gene distance weighting")
    parser.add_argument("--max_peak_distance", type=int, default=150000,
                        help="Maximum distance from peak to gene TSS")
    parser.add_argument("--dist_bias_mode", type=str, default="logsumexp",
                        choices=["max", "sum", "mean", "logsumexp"],
                        help="Method for calculating window-to-gene distance")
    parser.add_argument("--filter_to_nearest_gene", action="store_true", default=False,
                        help="Associate peaks only to nearest gene")
    parser.add_argument("--promoter_bp", type=int, default=None,
                        help="Promoter region size in base pairs (None for no promoter filtering)")
    
    # ----- Database and Reference Files -----
    parser.add_argument("--database_dir", type=Path, default=None,
                        help="Database directory path")
    parser.add_argument("--genome_dir", type=Path, default=None,
                        help="Genome data directory")
    parser.add_argument("--chrom_sizes_file", type=Path, default=None,
                        help="Chromosome sizes file path")
    parser.add_argument("--gtf_file_dir", type=Path, default=None,
                        help="GTF annotation file directory")
    parser.add_argument("--ncbi_file_dir", type=Path, default=None,
                        help="NCBI gene info directory")
    parser.add_argument("--gene_tss_file", type=Path, default=None,
                        help="Gene TSS BED file path")
    parser.add_argument("--tf_file", type=Path, default=None,
                        help="Transcription factor information file")
    parser.add_argument("--jaspar_pfm_dir", type=Path, default=None,
                        help="JASPAR PFM directory")
    parser.add_argument("--motif_dir", type=Path, default=None,
                        help="Motif PWM directory")
    
    # ----- Ground Truth and PKN Files -----
    parser.add_argument("--chip_ground_truth", type=Path, default=None,
                        help="ChIP-seq ground truth file path")
    parser.add_argument("--chip_ground_truth_sep", type=str, default=",",
                        help="ChIP-seq ground truth file separator")
    
    # ----- Output Directories -----
    parser.add_argument("--processed_data", type=Path, default=None,
                        help="Processed data output directory")
    parser.add_argument("--training_data_cache", type=Path, default=None,
                        help="Training data cache directory")
    parser.add_argument("--raw_data", type=Path, default=None,
                        help="Raw data directory")
    parser.add_argument("--pkn_dir", type=Path, default=None,
                        help="Prior knowledge network directory")
    parser.add_argument("--string_dir", type=Path, default=None,
                        help="STRING database directory")
    parser.add_argument("--trrust_dir", type=Path, default=None,
                        help="TRRUST database directory")
    parser.add_argument("--kegg_dir", type=Path, default=None,
                        help="KEGG database directory")
    parser.add_argument("--experiment_dir", type=Path, default=None,
                        help="Experiment output directory")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Main output directory")
    
    return parser.parse_args()


def setup_global_variables(args):
    """
    Convert argparse arguments to global variables.
    This maintains backward compatibility with code that expects global variables.
    """
    global ROOT_DIR, PROJECT_DATA_DIR, PROJECT_RESULT_DIR
    global ORGANISM_CODE, DATASET_NAME, CHROM_ID, CHROM_IDS
    global SAMPLE_NAMES, FINE_TUNING_DATASETS
    global RAW_SINGLE_CELL_DATA, RAW_10X_RNA_DATA_DIR, RAW_ATAC_PEAK_MATRIX_FILE
    global RAW_GSE218576_DIR, PROCESSED_GSE218576_DIR
    global MIN_GENES_PER_CELL, MIN_PEAKS_PER_CELL
    global FILTER_TYPE, FILTER_OUT_LOWEST_COUNTS_GENES, FILTER_OUT_LOWEST_COUNTS_PEAKS
    global FILTER_OUT_LOWEST_PCT_GENES, FILTER_OUT_LOWEST_PCT_PEAKS
    global PROCESSED_RNA_FILENAME, PROCESSED_ATAC_FILENAME
    global RAW_RNA_FILE, RAW_ATAC_FILE, ADATA_RNA_FILE, ADATA_ATAC_FILE
    global PSEUDOBULK_TG_FILE, PSEUDOBULK_RE_FILE
    global NEIGHBORS_K, PCA_COMPONENTS, HOPS, SELF_WEIGHT
    global VALIDATION_DATASETS, FORCE_RECALCULATE, WINDOW_SIZE
    global DISTANCE_SCALE_FACTOR, MAX_PEAK_DISTANCE, DIST_BIAS_MODE
    global FILTER_TO_NEAREST_GENE, PROMOTER_BP
    global DATABASE_DIR, GENOME_DIR, CHROM_SIZES_FILE, GTF_FILE_DIR, NCBI_FILE_DIR
    global GENE_TSS_FILE, TF_FILE, JASPAR_PFM_DIR, MOTIF_DIR
    global CHIP_GROUND_TRUTH, CHIP_GROUND_TRUTH_SEP
    global PROCESSED_DATA, TRAINING_DATA_CACHE, RAW_DATA, PKN_DIR
    global STRING_DIR, TRRUST_DIR, KEGG_DIR, EXPERIMENT_DIR, OUTPUT_DIR
    global SAMPLE_PROCESSED_DATA_DIR, SAMPLE_DATA_CACHE_DIR, COMMON_DATA
    
    # Set ROOT_DIR from args or default to script's parent directory
    ROOT_DIR = args.root_dir if args.root_dir else Path(__file__).resolve().parent.parent.parent
    
    # Set primary directories
    PROJECT_DATA_DIR = args.project_data_dir if args.project_data_dir else ROOT_DIR / "data"
    PROJECT_RESULT_DIR = args.project_result_dir if args.project_result_dir else ROOT_DIR / "results"
    
    # Sample information
    ORGANISM_CODE = args.organism_code
    DATASET_NAME = args.dataset_name
    
    CHROM_ID = args.chrom_id
    CHROM_IDS = args.chrom_ids
    
    # Sample names
    SAMPLE_NAMES = args.sample_names
    
    FINE_TUNING_DATASETS = args.fine_tuning_datasets if args.fine_tuning_datasets else SAMPLE_NAMES
    
    # Raw data paths
    RAW_SINGLE_CELL_DATA = args.raw_single_cell_data
    RAW_10X_RNA_DATA_DIR = args.raw_10x_rna_data_dir
    RAW_ATAC_PEAK_MATRIX_FILE = args.raw_atac_peak_matrix_file
    RAW_GSE218576_DIR = args.raw_gse218576_dir if args.raw_gse218576_dir else ROOT_DIR / "data/raw/GSE218576"
    PROCESSED_GSE218576_DIR = args.processed_gse218576_dir if args.processed_gse218576_dir else ROOT_DIR / "data/processed/GSE218576"
    
    # QC Filtering
    MIN_GENES_PER_CELL = args.min_genes_per_cell
    MIN_PEAKS_PER_CELL = args.min_peaks_per_cell
    FILTER_TYPE = args.filter_type
    FILTER_OUT_LOWEST_COUNTS_GENES = args.filter_out_lowest_counts_genes
    FILTER_OUT_LOWEST_COUNTS_PEAKS = args.filter_out_lowest_counts_peaks
    FILTER_OUT_LOWEST_PCT_GENES = args.filter_out_lowest_pct_genes
    FILTER_OUT_LOWEST_PCT_PEAKS = args.filter_out_lowest_pct_peaks
    
    # File naming
    PROCESSED_RNA_FILENAME = args.processed_rna_filename
    PROCESSED_ATAC_FILENAME = args.processed_atac_filename
    RAW_RNA_FILE = args.raw_rna_file
    RAW_ATAC_FILE = args.raw_atac_file
    ADATA_RNA_FILE = args.adata_rna_file
    ADATA_ATAC_FILE = args.adata_atac_file
    PSEUDOBULK_TG_FILE = args.pseudobulk_tg_file
    PSEUDOBULK_RE_FILE = args.pseudobulk_re_file
    
    # Pseudobulk parameters
    NEIGHBORS_K = args.neighbors_k
    PCA_COMPONENTS = args.pca_components
    HOPS = args.hops
    SELF_WEIGHT = args.self_weight
    
    # Preprocessing parameters
    VALIDATION_DATASETS = args.validation_datasets if args.validation_datasets else ["E8.75_rep1"]
    FORCE_RECALCULATE = args.force_recalculate
    WINDOW_SIZE = args.window_size
    DISTANCE_SCALE_FACTOR = args.distance_scale_factor
    MAX_PEAK_DISTANCE = args.max_peak_distance
    DIST_BIAS_MODE = args.dist_bias_mode
    FILTER_TO_NEAREST_GENE = args.filter_to_nearest_gene
    PROMOTER_BP = args.promoter_bp
    
    # Database and reference directories
    DATABASE_DIR = args.database_dir if args.database_dir else ROOT_DIR / "data"
    GENOME_DIR = args.genome_dir if args.genome_dir else DATABASE_DIR / "genome_data" / "reference_genome" / ORGANISM_CODE
    CHROM_SIZES_FILE = args.chrom_sizes_file if args.chrom_sizes_file else GENOME_DIR / f"{ORGANISM_CODE}.chrom.sizes"
    GTF_FILE_DIR = args.gtf_file_dir if args.gtf_file_dir else DATABASE_DIR / "genome_data" / "genome_annotation" / ORGANISM_CODE
    NCBI_FILE_DIR = args.ncbi_file_dir if args.ncbi_file_dir else DATABASE_DIR / "genome_data" / "genome_annotation" / ORGANISM_CODE
    GENE_TSS_FILE = args.gene_tss_file if args.gene_tss_file else DATABASE_DIR / "genome_data" / "genome_annotation" / ORGANISM_CODE / "gene_tss.bed"
    TF_FILE = args.tf_file if args.tf_file else DATABASE_DIR / "databases" / "motif_information" / ORGANISM_CODE / "TF_Information_all_motifs.txt"
    JASPAR_PFM_DIR = args.jaspar_pfm_dir if args.jaspar_pfm_dir else DATABASE_DIR / "databases" / "motif_information" / "JASPAR" / "pfm_files"
    MOTIF_DIR = args.motif_dir if args.motif_dir else DATABASE_DIR / "databases" / "motif_information" / ORGANISM_CODE / "pwms_all_motifs"
    
    # Ground truth
    CHIP_GROUND_TRUTH = args.chip_ground_truth if args.chip_ground_truth else DATABASE_DIR / "ground_truth_files" / "mESC_beeline_ChIP-seq.csv"
    CHIP_GROUND_TRUTH_SEP = args.chip_ground_truth_sep
    
    # Output directories
    PROCESSED_DATA = args.processed_data if args.processed_data else DATABASE_DIR / "processed"
    TRAINING_DATA_CACHE = args.training_data_cache if args.training_data_cache else DATABASE_DIR / "training_data_cache"
    RAW_DATA = args.raw_data if args.raw_data else DATABASE_DIR / "raw"
    PKN_DIR = args.pkn_dir if args.pkn_dir else DATABASE_DIR / "prior_knowledge_network_data" / ORGANISM_CODE
    
    # PKN subdirectories
    STRING_DIR = args.string_dir if args.string_dir else DATABASE_DIR / "prior_knowledge_network_data" / ORGANISM_CODE / "STRING"
    TRRUST_DIR = args.trrust_dir if args.trrust_dir else DATABASE_DIR / "prior_knowledge_network_data" / ORGANISM_CODE / "TRRUST"
    KEGG_DIR = args.kegg_dir if args.kegg_dir else DATABASE_DIR / "prior_knowledge_network_data" / ORGANISM_CODE / "KEGG"
    
    # Experiment directories
    EXPERIMENT_DIR = args.experiment_dir if args.experiment_dir else PROJECT_DATA_DIR / "experiments"
    OUTPUT_DIR = args.output_dir if args.output_dir else EXPERIMENT_DIR / DATASET_NAME
    
    # Sample-specific paths
    SAMPLE_PROCESSED_DATA_DIR = PROCESSED_DATA / DATASET_NAME
    SAMPLE_DATA_CACHE_DIR = TRAINING_DATA_CACHE / DATASET_NAME
    COMMON_DATA = SAMPLE_DATA_CACHE_DIR / "common"
    
    # Write all variables to a config file for reference
    config_path = OUTPUT_DIR / "preprocessing_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_dict = {k: v for k, v in globals().items() if k.isupper()}
    atomic_json_dump(config_dict, config_path)

def update_info_file(info_file: Path, key: str, value: Any) -> None:
    """
    Update a JSON info file with a new key-value pair.
    If the file does not exist, it will be created.
    """
    if info_file.exists():
        with open(info_file, 'r') as f:
            info_data = json.load(f)
    else:
        info_data = {}
    
    info_data[key] = value
    
    with open(info_file, 'w') as f:
        json.dump(info_data, f, indent=4)

# ----- Data Loading and Processing -----
def normalize_peak_format(peak_id: str) -> str:
    """
    Normalize peak format from chrN-start-end or chrN:start:end to chrN:start-end.
    Handles both formats as input and always outputs chrN:start-end.
    """
    if not isinstance(peak_id, str):
        return peak_id
    
    # Try to parse chr-start-end format (with dashes)
    parts = peak_id.split('-')
    if len(parts) >= 3:
        # Assume format is chr-start-end where chr might have dashes
        # Work backwards: the last two parts are start and end
        try:
            end = int(parts[-1])
            start = int(parts[-2])
            chrom = '-'.join(parts[:-2])  # Everything before the last two parts
            return f"{chrom}:{start}-{end}"
        except (ValueError, IndexError):
            pass
    
    # Already in chr:start-end format or some other format, return as-is
    return peak_id

def pseudo_bulk(
    rna_data: AnnData,
    atac_data: AnnData,
    neighbors_k: int = 20,
    pca_components: int = 25,
    hops: int = 1,
    self_weight: float = 1.0,
    renormalize_each_hop: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Soft 'metacells' that DO NOT reduce the number of cells.

    Builds a joint KNN graph on concatenated PCA (RNA||ATAC), forms a row-stochastic
    weight matrix W (including self loops), and returns neighborhood-averaged profiles:
        X_soft = W @ X

    Parameters
    ----------
    neighbors_k : int
        K for the joint KNN graph.
    pca_components: int
        Number of PCs per modality before concatenation.
    hops: int
        How many times to diffuse (W^h). Larger = smoother.
    self_weight: float
        Diagonal weight (>=0) added before row-normalization.
    renormalize_each_hop: bool
        If True, row-normalize after every hop to keep rows summing to 1.

    Returns
    ----------
    soft_rna_df: pd.DataFrame
        genes × cells (same cells as input)
    soft_atac_df: pd.DataFrame
        peaks × cells (same cells as input)
    """
    # --- copy and align cells across modalities (same as in your function) ---
    rna = rna_data.copy()
    atac = atac_data.copy()

    common = rna.obs_names.intersection(atac.obs_names)
    if len(common) == 0:
        raise ValueError("No overlapping cell barcodes between RNA and ATAC.")
    rna = rna[common].copy()
    atac = atac[common].copy()
    atac = atac[rna.obs_names].copy()
    assert (rna.obs_names == atac.obs_names).all()

    # --- ensure PCA in both modalities (caps to valid range) ---
    def _ensure_pca(adata: AnnData, n_comps: int) -> None:
        max_comps = int(min(adata.n_obs, adata.n_vars))
        use_comps = max(1, min(n_comps, max_comps))
        if "X_pca" not in adata.obsm_keys() or adata.obsm.get("X_pca", np.empty((0,0))).shape[1] < use_comps:
            sc.pp.scale(adata, max_value=10, zero_center=True)
            sc.tl.pca(adata, n_comps=use_comps, svd_solver="arpack")

    _ensure_pca(rna, pca_components)
    _ensure_pca(atac, pca_components)

    # --- joint embedding, neighbors on it ---
    combined_pca = np.concatenate((rna.obsm["X_pca"], atac.obsm["X_pca"]), axis=1)
    # tiny placeholder X to satisfy AnnData; graph lives in .obsp
    joint = AnnData(X=sp.csr_matrix((rna.n_obs, 0)), obs=rna.obs.copy())
    joint.obsm["X_combined"] = combined_pca
    
    sc.pp.neighbors(joint, n_neighbors=neighbors_k, use_rep="X_combined")

    # base connectivities (symmetric KNN graph; values ~[0,1])
    W = joint.obsp["connectivities"].tocsr().astype(np.float32)

    # add self-loops
    if self_weight > 0:
        W = W + sp.diags(np.full(W.shape[0], self_weight, dtype=np.float32), format="csr")

    # row-normalize to make W row-stochastic
    def _row_norm(mat: sp.csr_matrix) -> sp.csr_matrix:
        row_sum = np.asarray(mat.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        inv = sp.diags(1.0 / row_sum, dtype=np.float32)
        return inv @ mat

    W = _row_norm(W)

    # multi-hop diffusion: W <- W^h
    
    if hops > 1:
        W_h = W
        for _ in range(1, int(hops)):
            W_h = W_h @ W
            if renormalize_each_hop:
                W_h = _row_norm(W_h)
        W = W_h
        
    # one last normalization to make sure rows sum to 1
    W = _row_norm(W)

    # --- apply smoothing per modality ---
    def _as_csr(X, dtype=np.float32):
        """Return X as CSR sparse matrix (no-copy when already CSR)."""
        if sp.issparse(X):
            return X.tocsr().astype(dtype, copy=False)
        # X is dense (numpy ndarray / matrix)
        return sp.csr_matrix(np.asarray(X, dtype=dtype, order="C"))

    def _select_matrix(adata: AnnData):
        """Pick the best available expression/count matrix."""
        if "log1p" in adata.layers:
            return adata.layers["log1p"]
        if "counts" in adata.layers:
            return adata.layers["counts"]
        return adata.X
    
    # X matrices are cells × features (CSR recommended)
    X_rna  = _as_csr(_select_matrix(rna))      # cells × genes
    X_atac = _as_csr(_select_matrix(atac))     # cells × peaks

    X_rna_soft = W @ X_rna      # cells × genes
    X_atac_soft = W @ X_atac    # cells × peaks

    # return DataFrames as features × cells (to mirror your pseudo_bulk shapes)
    pseudo_bulk_rna_df = pd.DataFrame(
        X_rna_soft.T.toarray(),
        index=rna.var_names,
        columns=rna.obs_names,
    )
    pseudo_bulk_atac_df = pd.DataFrame(
        X_atac_soft.T.toarray(),
        index=atac.var_names,
        columns=atac.obs_names,
    )

    return pseudo_bulk_rna_df, pseudo_bulk_atac_df

def process_10x_to_parquet(
    raw_10x_rna_data_dir: Union[str, Path], 
    raw_atac_peak_file: Union[str, Path], 
    rna_outfile_path: Union[str, Path], 
    atac_outfile_path: Union[str, Path], 
    sample_name: str):
    
    """
    Process 10x data (RNA and ATAC) into parquet files.

    Parameters
    ----------
    raw_10x_rna_data_dir : Union[str, Path]
        Path to raw 10x RNA data directory.
    raw_atac_peak_file : Union[str, Path]
        Path to raw ATAC peak matrix file.
    rna_outfile_path : Union[str, Path]
        Path to output parquet file for RNA data.
    atac_outfile_path : Union[str, Path]
        Path to output parquet file for ATAC data.
    sample_name : str
        Sample name to associate with the data.

    Returns
    -------
    None
    """
    def _load_rna_adata(sample_raw_data_dir: Union[str, Path]) -> sc.AnnData:
        """
        Load scRNA data from a 10x directory. Expects a single features.tsv.gz file.

        Parameters
        ----------
        sample_raw_data_dir : Union[str, Path]
            Path to the directory containing the 10x RNA data.

        Returns
        -------
        adata : AnnData
            Loaded scRNA data.
        """
        
        # Look for features file
        features = [f for f in os.listdir(sample_raw_data_dir) if f.endswith("features.tsv.gz")]
        assert len(features) == 1, \
            f"Expected 1 features.tsv.gz, found {features}. Make sure the files are gunziped for sc.read_10x_mtx."

        prefix = features[0].replace("features.tsv.gz", "")
        logging.debug(f"Detected File Prefix: {prefix}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Only considering the two last:")
            adata = sc.read_10x_mtx(
                sample_raw_data_dir,
                var_names="gene_symbols",
                make_unique=True,
                prefix=prefix
            )
        return adata
    

    def _get_adata_from_peakmatrix(peak_matrix_file: Union[str, Path], label: pd.DataFrame, sample_name: str, dtype=np.uint16) -> AnnData:
        """
        Load ATAC data from a peak matrix file.

        Parameters
        ----------
        peak_matrix_file : Union[str, Path]
            Path to the peak matrix file.
        label : pd.DataFrame
            DataFrame containing barcode information.
        sample_name : str
            Sample name to associate with the data.
        dtype : np.dtype, optional
            Data type to use for the data matrix, by default np.uint16.

        Returns
        -------
        adata : AnnData
            Loaded ATAC data.
        """
        logging.info(f"  - [{sample_name}] Reading ATAC peaks")

        is_gz = str(peak_matrix_file).endswith(".gz")
        opener = gzip.open if is_gz else open

        label_set = set(label["barcode_use"].astype(str))

        with opener(peak_matrix_file, "rt") as f:
            # --- header ---
            header = f.readline().rstrip("\n").split("\t")
            first_col = header[0]
            barcodes = header[1:]

            # keep barcode columns in file order
            keep_barcodes = [bc for bc in barcodes if bc in label_set]
            keep_idx = [i + 1 for i, bc in enumerate(barcodes) if bc in label_set]  # +1 because first col is peak id

            logging.debug(f"  - Total barcodes in file: {len(barcodes):,}")
            logging.debug(f"  - Matched barcodes: {len(keep_barcodes):,}")

            # --- build COO triplets ---
            rows = []
            cols = []
            data = []
            var_names = []

            peak_i = 0
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= 1:
                    continue

                var_names.append(parts[0])  # peak coord
                # parse only kept columns; store nonzeros
                for j, col_i in enumerate(keep_idx):
                    v = parts[col_i]
                    if v == "0" or v == "" or v is None:
                        continue
                    # int conversion (fast path)
                    iv = int(v)
                    if iv != 0:
                        rows.append(j)          # cell index within keep_barcodes
                        cols.append(peak_i)     # peak index
                        data.append(iv)

                peak_i += 1

        if len(data) == 0:
            raise ValueError("No nonzero entries found after filtering.")

        rows = np.asarray(rows, dtype=np.int32)
        cols = np.asarray(cols, dtype=np.int32)
        data = np.asarray(data, dtype=dtype)

        X = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(len(keep_barcodes), len(var_names)),
            dtype=dtype
        ).tocsr()

        adata = AnnData(X=X)
        adata.obs_names = keep_barcodes
        adata.var_names = var_names
        adata.obs["barcode"] = adata.obs_names
        adata.obs["sample"] = sample_name
        adata.obs["label"] = label.set_index("barcode_use").loc[keep_barcodes, "label"].values
        return adata
    
    if raw_10x_rna_data_dir is None:
        raise FileNotFoundError(
            f"Neither processed files, filtered AnnData, nor raw data files are available.\n"
            f"Expected raw RNA file: {raw_rna_file}\n"
            f"Expected raw ATAC file: {raw_atac_file}\n"
            f"Or provide raw_10x_rna_data_dir and raw_atac_peak_file to create from 10x data."
        )
    if not raw_10x_rna_data_dir.is_dir():
        raise FileNotFoundError(f"10x RNA directory not found: {raw_10x_rna_data_dir}")
    if raw_atac_peak_file is None or not raw_atac_peak_file.is_file():
        raise FileNotFoundError(f"ATAC peak file not found: {raw_atac_peak_file}")

    logging.info(f"[{sample_name}] Raw 10X RNA and ATAC inputs found, converting to parquet files")
    logging.debug(f"  - raw_10x_rna_data_dir: {raw_10x_rna_data_dir}")
    logging.debug(f"  - raw_atac_peak_file:  {raw_atac_peak_file}")
    
    # --- load raw data ---
    adata_RNA = _load_rna_adata(raw_10x_rna_data_dir)
    adata_RNA.obs_names = [(sample_name + "." + i).replace("-", ".") for i in adata_RNA.obs_names]
    # adata_RNA.obs_names = [i.replace("-", ".") for i in adata_RNA.obs_names]
    logging.info(f"[{sample_name}] Found {len(adata_RNA.obs_names)} RNA barcodes")
    logging.info(f"  - First RNA barcode: {adata_RNA.obs_names[0]}")

    label = pd.DataFrame({"barcode_use": adata_RNA.obs_names,
                            "label": ["mESC"] * len(adata_RNA.obs_names)})

    adata_ATAC = _get_adata_from_peakmatrix(raw_atac_peak_file, label, sample_name)
    
    logging.info(f"[{sample_name}] Writing raw data files")
    raw_sc_rna_df = pd.DataFrame(
        adata_RNA.X.toarray() if sp.issparse(adata_RNA.X) else adata_RNA.X,
        index=adata_RNA.obs_names,
        columns=adata_RNA.var_names,
    )
    raw_sc_atac_df = pd.DataFrame(
        adata_ATAC.X.toarray() if sp.issparse(adata_ATAC.X) else adata_ATAC.X,
        index=adata_ATAC.obs_names,
        columns=adata_ATAC.var_names,
    )

    os.makedirs(os.path.dirname(rna_outfile_path), exist_ok=True)
    os.makedirs(os.path.dirname(atac_outfile_path), exist_ok=True)
    
    raw_sc_rna_df = raw_sc_rna_df.astype("float32")
    raw_sc_atac_df = raw_sc_atac_df.astype("float32")
    
    raw_sc_rna_df.to_parquet(rna_outfile_path, engine="pyarrow", compression="snappy")
    raw_sc_atac_df.to_parquet(atac_outfile_path, engine="pyarrow", compression="snappy")

def _configured_path(
    sample_input_dir: Path,
    key: str,
    default_name: str,
    sample_name: str,
) -> Path:
    """
    Look up `key` among names imported via `from config.settings import *`.
    Falls back to `default_name`. Relative paths are resolved under sample_input_dir.
    Supports {sample} templating. Accepts str or Path in config.
    """
    # settings injected by the star-import live in this module's globals
    value = globals().get(key, None)

    # accept Path or str from config; fall back to default
    raw = value if value is not None else default_name
    if isinstance(raw, Path):
        raw = str(raw)

    # allow e.g. "outputs/{sample}_RNA.parquet"
    if isinstance(raw, str):
        try:
            raw = raw.format(sample=sample_name)
        except Exception:
            pass

    p = Path(raw)
    return p if p.is_absolute() else (sample_input_dir / p)

def process_or_load_rna_atac_data(
    sample_input_dir: Union[str, Path],
    force_recalculate: bool = False,
    raw_10x_rna_data_dir: Union[str, Path, None] = None,
    raw_atac_peak_file: Union[str, Path, None] = None,
    *,
    sample_name: Optional[str] = None,
    sample_processed_dir: Optional[Union[str, Path]] = None,
    neighbors_k: Optional[int] = None,
    pca_components: Optional[int] = None,
    hops: Optional[int] = None,
    self_weight: Optional[float] = None,
    load: Optional[bool] = True,
    summary_file: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Create (or load) per-sample pseudobulk datasets and return them.

    Resolution order (unless `force_recalculate=True`):
      1) Pseudobulk parquet files (TG/RE)
      2) Filtered AnnData files (RNA/ATAC)
      3) Processed parquet files (scRNA/scATAC)
      4) Raw data files (parquet/CSV/TSV)
      5) Raw 10x inputs (if provided)

    Returns
    -------
    TG_pseudobulk_df, RE_pseudobulk_df : pd.DataFrame | None
        DataFrames are features × metacells. If `load=False`, returns (None, None)
        after ensuring files exist.
    """
    raw_data_info_dict = None
    preprocessed_info_dict = None

    sample_input_dir = Path(sample_input_dir)
    sample_name = sample_name or sample_input_dir.name
    raw_10x_rna_data_dir = Path(raw_10x_rna_data_dir) if raw_10x_rna_data_dir is not None else None
    raw_atac_peak_file = Path(raw_atac_peak_file) if raw_atac_peak_file is not None else None

    output_dir = Path(sample_processed_dir) if sample_processed_dir else sample_input_dir
    logging.info(f"\n[{sample_name}] Using input directory: {sample_input_dir}")
    logging.info(f"[{sample_name}] Using output directory: {output_dir}")

    processed_rna_file = _configured_path(output_dir, "PROCESSED_RNA_FILENAME", "scRNA_seq_processed.parquet", sample_name)
    processed_atac_file = _configured_path(output_dir, "PROCESSED_ATAC_FILENAME", "scATAC_seq_processed.parquet", sample_name)
    raw_rna_file = _configured_path(sample_input_dir, "RAW_RNA_FILE", "scRNA_seq_raw.parquet", sample_name)
    raw_atac_file = _configured_path(sample_input_dir, "RAW_ATAC_FILE", "scATAC_seq_raw.parquet", sample_name)
    adata_rna_file = _configured_path(output_dir, "ADATA_RNA_FILE", "adata_RNA.h5ad", sample_name)
    adata_atac_file = _configured_path(output_dir, "ADATA_ATAC_FILE", "adata_ATAC.h5ad", sample_name)

    pseudobulk_TG_file = output_dir / "TG_pseudobulk.parquet"
    pseudobulk_RE_file = output_dir / "RE_pseudobulk.parquet"

    neighbors_k = neighbors_k if neighbors_k is not None else NEIGHBORS_K
    pca_components = pca_components if pca_components is not None else PCA_COMPONENTS
    hops = hops if hops is not None else HOPS
    self_weight = self_weight if self_weight is not None else SELF_WEIGHT

    # If load is False, we only care about whether pseudobulk exists.
    if load is False and (not force_recalculate) and pseudobulk_TG_file.is_file() and pseudobulk_RE_file.is_file():
        logging.info("Pseudobulk files already exist; skipping.")
        return None, None

    def _read_matrix_file(path: Path) -> pd.DataFrame:
        suf = path.suffix.lower()
        if suf in {".csv", ".tsv", ".txt"}:
            sep = "," if suf == ".csv" else "\t"
            return pd.read_csv(path, index_col=0, sep=sep)
        return pd.read_parquet(path, engine="pyarrow")

    def _standardize_symbols_index(
        df: pd.DataFrame,
        *,
        strip_version_suffix: bool = True,
        uppercase: bool = True,
        deduplicate: str = "sum",
    ) -> pd.DataFrame:
        x = df.copy()
        idx = x.index.astype(str).str.strip()
        if strip_version_suffix:
            idx = idx.str.replace(r"\.\d+$", "", regex=True)
        if uppercase:
            idx = idx.str.upper()
        x.index = idx
        if deduplicate:
            if deduplicate == "sum":
                x = x.groupby(level=0).sum()
            elif deduplicate == "mean":
                x = x.groupby(level=0).mean()
            elif deduplicate == "first":
                x = x[~x.index.duplicated(keep="first")]
            elif deduplicate in {"max", "min", "median"}:
                x = getattr(x.groupby(level=0), deduplicate)()
            else:
                raise ValueError(f"Unknown deduplicate policy: {deduplicate}")
        return x

    def _normalize_barcodes(index_like: pd.Index) -> pd.Index:
        ix = pd.Index(index_like).astype(str)
        ix = ix.str.replace(r"_\d+$", "", regex=True)
        ix = ix.str.replace(r"[-\.]\d+$", "", regex=True)
        ix = ix.str.replace(r"(?:_RNA|_GEX|_ATAC|#GEX|#ATAC)$", "", regex=True, case=False)
        return ix.str.upper()

    def harmonize_and_intersect(ad_rna: AnnData, ad_atac: AnnData) -> Tuple[AnnData, AnnData]:
        rna_norm = _normalize_barcodes(ad_rna.obs_names)
        atac_norm = _normalize_barcodes(ad_atac.obs_names)
        r_map = pd.Series(ad_rna.obs_names, index=rna_norm, dtype="object")
        a_map = pd.Series(ad_atac.obs_names, index=atac_norm, dtype="object")
        common = r_map.index.intersection(a_map.index)
        if len(common) == 0:
            raise RuntimeError("No overlapping barcodes after normalization.")
        ad_rna2 = ad_rna[r_map.loc[common].values, :].copy()
        ad_atac2 = ad_atac[a_map.loc[common].values, :].copy()
        ad_rna2.obs_names = common
        ad_atac2.obs_names = common
        return ad_rna2, ad_atac2

    # 1) Pseudobulk cache
    if not force_recalculate and pseudobulk_TG_file.is_file() and pseudobulk_RE_file.is_file():
        logging.info("Pseudobulk files found, loading from disk.")
        if not load:
            return None, None
        tg = pd.read_parquet(pseudobulk_TG_file, engine="pyarrow")
        re_df = pd.read_parquet(pseudobulk_RE_file, engine="pyarrow")
        tg = _standardize_symbols_index(tg)
        return tg, re_df

    # 2) AnnData
    ad_rna: Optional[AnnData] = None
    ad_atac: Optional[AnnData] = None
    if not force_recalculate and adata_rna_file.is_file() and adata_atac_file.is_file():
        logging.info("Filtered AnnData found. Loading to compute pseudobulk.")
        ad_rna = sc.read_h5ad(adata_rna_file)
        ad_atac = sc.read_h5ad(adata_atac_file)

    # 3) Processed parquet -> AnnData
    if (ad_rna is None or ad_atac is None) and (not force_recalculate) and processed_rna_file.is_file() and processed_atac_file.is_file():
        logging.info("Processed parquet found. Building AnnData to compute pseudobulk.")
        proc_rna = pd.read_parquet(processed_rna_file, engine="pyarrow")
        proc_atac = pd.read_parquet(processed_atac_file, engine="pyarrow")
        proc_rna = _standardize_symbols_index(proc_rna)
        ad_rna = AnnData(X=sp.csr_matrix(proc_rna.T.values.astype(np.float32, copy=False)))
        ad_rna.obs_names = proc_rna.columns.astype(str)
        ad_rna.var_names = proc_rna.index.astype(str)
        ad_atac = AnnData(X=sp.csr_matrix(proc_atac.T.values.astype(np.float32, copy=False)))
        ad_atac.obs_names = proc_atac.columns.astype(str)
        ad_atac.var_names = proc_atac.index.astype(str)

    # 4/5) Raw files (or 10x -> raw) -> AnnData
    if ad_rna is None or ad_atac is None or force_recalculate:
        if (not raw_rna_file.is_file()) or (not raw_atac_file.is_file()):
            if raw_10x_rna_data_dir is None or raw_atac_peak_file is None:
                raise FileNotFoundError(
                    "Missing raw files and no 10x inputs provided. "
                    f"Expected raw RNA file: {raw_rna_file}; raw ATAC file: {raw_atac_file}"
                )
            if not raw_10x_rna_data_dir.is_dir():
                raise FileNotFoundError(f"10x RNA directory not found: {raw_10x_rna_data_dir}")
            if not raw_atac_peak_file.is_file():
                raise FileNotFoundError(f"ATAC peak file not found: {raw_atac_peak_file}")
            logging.info(f"[{sample_name}] Building raw parquet from 10x inputs")
            process_10x_to_parquet(raw_10x_rna_data_dir, raw_atac_peak_file, raw_rna_file, raw_atac_file, sample_name)

        logging.info(f"[{sample_name}] Loading raw data files: {raw_rna_file.name}, {raw_atac_file.name}")
        rna_df = _read_matrix_file(raw_rna_file)
        atac_df = _read_matrix_file(raw_atac_file)
        raw_data_info_dict = {
            "raw_data_info": {
                "num_rna_cells": int(rna_df.shape[0]),
                "num_atac_cells": int(atac_df.shape[0]),
                "num_genes": int(rna_df.shape[1]),
                "num_peaks": int(atac_df.shape[1]),
            }
        }

        # try both orientations
        try:
            ad_rna = AnnData(rna_df)
            ad_atac = AnnData(atac_df)
            ad_rna, ad_atac = harmonize_and_intersect(ad_rna, ad_atac)
        except Exception:
            ad_rna = AnnData(rna_df.T)
            ad_atac = AnnData(atac_df.T)
            ad_rna, ad_atac = harmonize_and_intersect(ad_rna, ad_atac)

        ad_rna.layers["log1p"] = ad_rna.X.copy()
        ad_atac.layers["log1p"] = ad_atac.X.copy()
        logging.info("Running filter_and_qc on RNA/ATAC AnnData")
        ad_rna, ad_atac = filter_and_qc(ad_rna, ad_atac)
        logging.info("Writing filtered AnnData files")
        ad_rna.write_h5ad(adata_rna_file)
        ad_atac.write_h5ad(adata_atac_file)

    # Ensure processed parquet exists for downstream single-cell tensor building.
    # Only write when missing or when forcing recomputation.
    def _adata_to_feature_by_cell_df(adata: AnnData) -> pd.DataFrame:
        if "log1p" in adata.layers:
            X = adata.layers["log1p"]
        elif "counts" in adata.layers:
            X = adata.layers["counts"]
        else:
            X = adata.X
        if sp.issparse(X):
            arr = X.T.toarray()
        else:
            arr = np.asarray(X, dtype=np.float32).T
        return pd.DataFrame(arr, index=adata.var_names.astype(str), columns=adata.obs_names.astype(str))

    wrote_processed = False
    if force_recalculate or (not processed_rna_file.is_file()):
        processed_rna_file.parent.mkdir(parents=True, exist_ok=True)
        processed_rna_df = _adata_to_feature_by_cell_df(ad_rna).astype("float32")
        processed_rna_df.index = processed_rna_df.index.astype(str).map(standardize_name)
        processed_rna_df.to_parquet(processed_rna_file, engine="pyarrow", compression="snappy")
        wrote_processed = True

    if force_recalculate or (not processed_atac_file.is_file()):
        processed_atac_file.parent.mkdir(parents=True, exist_ok=True)
        processed_atac_df = _adata_to_feature_by_cell_df(ad_atac).astype("float32")
        processed_atac_df.to_parquet(processed_atac_file, engine="pyarrow", compression="snappy")
        wrote_processed = True

    if wrote_processed:
        preprocessed_info_dict = {
            "preprocessed_data_info": {
                "num_rna_cells": int(ad_rna.n_obs),
                "num_atac_cells": int(ad_atac.n_obs),
                "num_genes": int(ad_rna.n_vars),
                "num_peaks": int(ad_atac.n_vars),
            }
        }

    if ad_rna is None or ad_atac is None:
        raise RuntimeError("Failed to construct AnnData for pseudobulk computation.")

    tg_df, re_df = pseudo_bulk(
        rna_data=ad_rna,
        atac_data=ad_atac,
        neighbors_k=neighbors_k,
        pca_components=pca_components,
        hops=hops,
        self_weight=self_weight,
        renormalize_each_hop=True,
    )
    tg_df = tg_df.fillna(0)
    re_df = re_df.fillna(0)
    re_df[re_df > 100] = 100
    tg_df = _standardize_symbols_index(tg_df)

    pseudobulk_TG_file.parent.mkdir(parents=True, exist_ok=True)
    tg_df.to_parquet(pseudobulk_TG_file, engine="pyarrow", compression="snappy")
    re_df.to_parquet(pseudobulk_RE_file, engine="pyarrow", compression="snappy")

    if summary_file is None:
        summary_file = EXPERIMENT_DIR / DATASET_NAME / "experiment_info.json"

    sample_info = {"sample_name": sample_name}
    if raw_data_info_dict is not None:
        sample_info.update(raw_data_info_dict)
    if preprocessed_info_dict is not None:
        sample_info.update(preprocessed_info_dict)
    sample_info.update(
        {
            "pseudobulk_data_info": {
                "num_TG_cells": int(tg_df.shape[1]),
                "num_RE_cells": int(re_df.shape[1]),
                "num_TG_genes": int(tg_df.shape[0]),
                "num_RE_peaks": int(re_df.shape[0]),
            }
        }
    )
    update_info_file(summary_file, f"{sample_name}_pseudobulk_info", sample_info)

    if not load:
        return None, None
    logging.info("RNA/ATAC pseudobulk datasets are ready.")
    return tg_df, re_df

def filter_and_qc(adata_RNA: AnnData, adata_ATAC: AnnData) -> Tuple[AnnData, AnnData]:
    """
    Filter and quality control RNA and ATAC data.

    Parameters
    ----------
    adata_RNA : AnnData
        RNA data
    adata_ATAC : AnnData
        ATAC data

    Returns
    -------
    filtered_RNA : AnnData
        Filtered RNA data
    filtered_ATAC : AnnData
        Filtered ATAC data
    """
    
    adata_RNA = adata_RNA.copy()
    adata_ATAC = adata_ATAC.copy()
    
    logging.debug(f"[START] RNA shape={adata_RNA.shape}, ATAC shape={adata_ATAC.shape}")
    
    common_barcodes = adata_RNA.obs_names.isin(adata_ATAC.obs_names)
    assert common_barcodes.sum() > 10, \
        f"No common barcodes. \n  - RNA: {adata_RNA.obs_names[:2]}\n  - ATAC: {adata_ATAC.obs_names[:2]}"
    
    # Synchronize barcodes
    adata_RNA.obs['barcode'] = adata_RNA.obs_names
    adata_ATAC.obs['barcode'] = adata_ATAC.obs_names

    n_before = (adata_RNA.n_obs, adata_ATAC.n_obs)
    adata_RNA = adata_RNA[common_barcodes].copy()
    adata_ATAC = adata_ATAC[adata_ATAC.obs['barcode'].isin(adata_RNA.obs['barcode'])].copy()
    
    logging.debug(
        f"[BARCODES] before sync RNA={n_before[0]}, ATAC={n_before[1]} → after sync RNA={adata_RNA.n_obs}, ATAC={adata_ATAC.n_obs}"
    )
    
    # QC and filtering
    # Filter out 
    adata_RNA.var['mt'] = adata_RNA.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata_RNA, qc_vars=["mt"], inplace=True)
    adata_RNA = adata_RNA[adata_RNA.obs.pct_counts_mt < 5].copy()
    adata_RNA.var_names_make_unique()
    adata_RNA.var['gene_ids'] = adata_RNA.var.index
    num_cells_rna = adata_RNA.n_obs
    num_cells_atac = adata_ATAC.n_obs
    
    sc.pp.filter_cells(adata_RNA, min_genes=MIN_GENES_PER_CELL)
    sc.pp.filter_cells(adata_ATAC, min_genes=MIN_PEAKS_PER_CELL)
    
    if FILTER_TYPE == "pct":
        sc.pp.filter_genes(adata_RNA, min_cells=math.ceil(num_cells_rna * FILTER_OUT_LOWEST_PCT_GENES))
        sc.pp.filter_genes(adata_ATAC, min_cells=math.ceil(num_cells_atac * FILTER_OUT_LOWEST_PCT_PEAKS))
    elif FILTER_TYPE == "count":
        sc.pp.filter_genes(adata_RNA, min_counts=FILTER_OUT_LOWEST_COUNTS_GENES)
        sc.pp.filter_genes(adata_ATAC, min_counts=FILTER_OUT_LOWEST_COUNTS_PEAKS)
    else:
        raise ValueError(f"Unknown filter type: {FILTER_TYPE}")
    
    # Preprocess RNA
    sc.pp.normalize_total(adata_RNA, target_sum=1e4)
    sc.pp.log1p(adata_RNA)
    # sc.pp.highly_variable_genes(adata_RNA, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_RNA.layers["log1p"] = adata_RNA.X.copy()
    sc.pp.scale(adata_RNA, max_value=10, zero_center=True)
    # adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable]
    sc.tl.pca(adata_RNA, n_comps=PCA_COMPONENTS, svd_solver="arpack")

    # Preprocess ATAC
    sc.pp.log1p(adata_ATAC)
    # sc.pp.highly_variable_genes(adata_ATAC, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_ATAC.layers["log1p"] = adata_ATAC.X.copy()
    # adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable]
    sc.pp.scale(adata_ATAC, max_value=10, zero_center=True)
    sc.tl.pca(adata_ATAC, n_comps=PCA_COMPONENTS, svd_solver="arpack")
    
    # After filtering to common barcodes
    common_barcodes = adata_RNA.obs_names.intersection(adata_ATAC.obs_names)
    
    adata_RNA = adata_RNA[common_barcodes].copy()
    adata_ATAC = adata_ATAC[common_barcodes].copy()
    
    return adata_RNA, adata_ATAC

def _canon(x: str) -> str:
    """
    Strips version suffix and uppercases a given string.

    Parameters
    ----------
    x : str
        The input string.

    Returns
    -------
    str
        The modified string.
    """
    
    # strip version suffix and uppercase
    s = str(x).strip()
    s = re.sub(r"\.\d+$", "", s)
    return s.upper()

def _read_list(path: Path, col: str) -> list[str]:
    """
    Reads a list of elements from a CSV file.

    Parameters
    ----------
    path : Path
        The path to the CSV file.
    col : str
        The column name to read from. If the column does not exist and
        the file has only one column, the function will read from the single column.

    Returns
    -------
    list[str]
        A sorted list of elements from the file.
    """
    if path.is_file():
        df = pd.read_csv(path)
        if col not in df.columns and df.shape[1] == 1:
            # tolerate unnamed single column
            return sorted({_canon(v) for v in df.iloc[:, 0].astype(str)})
        return sorted({_canon(v) for v in df[col].dropna().astype(str)})
    return []

def create_tf_tg_combination_files(
    genes: Iterable[str],
    tf_list_file: Union[str, Path],
    dataset_dir: Union[str, Path],
    sample_name: Union[str, Path],
    *,
    tf_name_col: Optional[str] = "TF_Name",  # if None, will auto-detect
) -> Tuple[List[str], List[str]]:
    """
    Build and persist TF/TG lists and the full TF–TG Cartesian product, updating any
    existing files by taking the union with newly supplied genes.

    Files written to {dataset_dir}/tf_tg_combos/:
      - total_genes.csv with column 'Gene'
      - tf_list.csv     with column 'TF'
      - tg_list.csv     with column 'TG'
      - tf_tg_combos.csv with columns ['TF','TG']

    Behavior:
      - Normalizes all symbols to uppercase and strips Ensembl-like version suffixes (.1, .2, ...).
      - TFs are defined as the intersection of `genes` with entries in `tf_list_file`.
      - TGs are defined as the remaining genes (i.e., genes - TFs).
      - If files already exist, the function unions existing + new and rewrites deterministically
        (sorted) to make runs reproducible.

    Parameters
    ----------
    genes : Iterable[str]
        Candidate gene symbols (TG or TF). Will be normalized.
    tf_list_file : str | Path
        Path to a TF reference list (tabular). If `tf_name_col` is None or missing, the function
        auto-detects a likely column or uses the only column if single-column.
    dataset_dir : str | Path
        Base directory under which the 'tf_tg_combos' folder will be created/updated.
    tf_name_col : str | None, default 'TF_Name'
        Column name that holds TF names in `tf_list_file`. If None or not present, auto-detect.

    Returns
    -------
    tfs : list[str]
        Final TF set (sorted, normalized).
    tgs : list[str]
        Final TG set (sorted, normalized, guaranteed disjoint from TFs).

    """
    dataset_dir = Path(dataset_dir)
    sample_dir = Path(dataset_dir, sample_name)
    
    out_dir = dataset_dir / "tf_tg_combos"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- normalize incoming genes ---
    genes_norm = sorted({_canon(g) for g in genes if pd.notna(g)})

    # --- load TF reference file robustly (auto-detect column if needed) ---
    tf_list_file = Path(tf_list_file)
    tf_ref = pd.read_csv(tf_list_file, sep=None, engine="python")  # auto-detect delim
    if tf_name_col and tf_name_col in tf_ref.columns:
        tf_col = tf_name_col
    else:
        # attempt to auto-detect a sensible TF column
        lower = {c.lower(): c for c in tf_ref.columns}
        for cand in ("tf_name", "tf", "symbol", "gene_symbol", "gene", "name"):
            if cand in lower:
                tf_col = lower[cand]
                break
        else:
            # if exactly one column, use it
            if tf_ref.shape[1] == 1:
                tf_col = tf_ref.columns[0]
            else:
                raise ValueError(
                    f"Could not locate TF name column in {tf_list_file}. "
                    f"Available columns: {list(tf_ref.columns)}"
                )

    known_tfs = {_canon(x) for x in tf_ref[tf_col].dropna().astype(str).tolist()}

    # --- new sets from this call ---
    tfs_new = sorted(set(genes_norm) & known_tfs)
    tgs_new = sorted(set(genes_norm) - set(tfs_new))


    total_file = out_dir / "total_genes.csv"
    tf_file    = out_dir / "tf_list.csv"
    tg_file    = out_dir / "tg_list.csv"

    total_existing = _read_list(total_file, "Gene")
    tf_existing    = _read_list(tf_file, "TF")
    tg_existing    = _read_list(tg_file, "TG")

    total = sorted(set(total_existing) | set(genes_norm))
    tfs   = sorted(set(tf_existing)    | set(tfs_new))
    # ensure TGs exclude any TFs
    tgs   = sorted((set(tg_existing) | set(tgs_new)) - set(tfs))

    # --- write back deterministically ---
    pd.DataFrame({"Gene": total}).to_csv(total_file, index=False)
    pd.DataFrame({"TF": tfs}).to_csv(tf_file, index=False)
    pd.DataFrame({"TG": tgs}).to_csv(tg_file, index=False)

    logging.info("  - Creating TF-TG combination files")
    logging.info(f"    - Number of TFs: {len(tfs):,}")
    logging.info(f"    - Number of TGs: {len(tgs):,}")
    logging.info(f"    - Files written under: {out_dir}")

    return tfs, tgs

# ----- MultiomicTransformer Global Dataset -----
def make_gene_tss_bed_file(gene_tss_file, genome_dir):
    gene_tss_bed = pybedtools.BedTool(gene_tss_file)
    gene_tss_df = (
        gene_tss_bed
        .to_dataframe(header=None, usecols=[0, 1, 2, 3])
        .rename(columns={0: "chrom", 1: "start", 2: "end", 3: "name"})
        .sort_values(by="start", ascending=True)
    )
    # Normalize symbols to match downstream vocab
    gene_tss_df["name"] = gene_tss_df["name"].astype(str).map(standardize_name)
    # Drop any duplicated gene symbols, keep first TSS row
    gene_tss_df = gene_tss_df.drop_duplicates(subset=["name"], keep="first")

    bed_path = os.path.join(genome_dir, "gene_tss.bed")
    gene_tss_df.to_csv(bed_path, sep="\t", header=False, index=False)
    return gene_tss_df

def build_peak_locs_from_index(
    peak_index: pd.Index,
    *,
    include_regex: str = r"^chr(\d+|X|Y)$",
    coerce_chr_prefix: bool = True,
) -> pd.DataFrame:
    """
    Parse peak ids like 'chr1:100-200' or 'chr1-100-200' (or '1:100-200' / '1-100-200' if coerce_chr_prefix)
    into a clean DataFrame and filter to canonical chromosomes (1..n, X, Y).
    
    Supports two formats:
    - 'chr1:100-200' (colon separator)
    - 'chr1-100-200' (dash separator)
    """
    rows = []
    for pid in map(str, peak_index):
        try:
            # Try colon format first: 'chr1:100-200'
            if ":" in pid:
                chrom_part, se = pid.split(":", 1)
                if coerce_chr_prefix and not chrom_part.startswith("chr"):
                    chrom = f"chr{chrom_part}"
                else:
                    chrom = chrom_part
                s, e = se.split("-", 1)
                s, e = int(s), int(e)
            # Otherwise try dash format: 'chr1-100-200'
            else:
                parts = pid.split("-")
                if len(parts) == 3:
                    chrom_part, s, e = parts
                    if coerce_chr_prefix and not chrom_part.startswith("chr"):
                        chrom = f"chr{chrom_part}"
                    else:
                        chrom = chrom_part
                    s, e = int(s), int(e)
                else:
                    raise ValueError(f"Expected 3 parts separated by '-', got {len(parts)}")
            
            if s > e:
                s, e = e, s
            rows.append((chrom, s, e, pid))
        except Exception as ex:
            logging.warning(f"Skipping malformed peak ID '{pid}': {ex}")
            continue

    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "peak_id"]).drop_duplicates()
    # keep only canonical chromosomes
    df = df[df["chrom"].astype(str).str.match(include_regex)].reset_index(drop=True)
    return df

def calculate_peak_to_tg_distance_score(
    peak_bed_file,
    tss_bed_file,
    peak_gene_dist_file,
    mesc_atac_peak_loc_df, 
    gene_tss_df, 
    max_peak_distance=1e6, 
    distance_factor_scale=25000, 
    force_recalculate=False,
    filter_to_nearest_gene=False,
    promoter_bp=None
) -> pd.DataFrame:
    """
    Compute peak-to-gene distance features (BEDTools-based), ensuring BED compliance.
    """
    # Validate and convert peaks to BED format
    required_cols = {"chrom", "start", "end", "peak_id"}
    if not required_cols.issubset(mesc_atac_peak_loc_df.columns):
        logging.warning("Converting peak index to BED format (chr/start/end parsing)")
        mesc_atac_peak_loc_df = build_peak_locs_from_index(mesc_atac_peak_loc_df.index)

    # print("\nmesc_atac_peak_loc_df")
    # print(mesc_atac_peak_loc_df.head())
    
    # Ensure numeric types
    mesc_atac_peak_loc_df["start"] = mesc_atac_peak_loc_df["start"].astype(int)
    mesc_atac_peak_loc_df["end"] = mesc_atac_peak_loc_df["end"].astype(int)

    # Ensure proper columns for gene_tss_df
    if not {"chrom", "start", "end", "name"}.issubset(gene_tss_df.columns):
        gene_tss_df = gene_tss_df.rename(columns={"chromosome_name": "chrom", "gene_start": "start", "gene_end": "end"})

    # Step 1: Write valid BED files if missing
    if not os.path.isfile(peak_bed_file) or not os.path.isfile(tss_bed_file) or force_recalculate:
        pybedtools.BedTool.from_dataframe(mesc_atac_peak_loc_df[["chrom", "start", "end", "peak_id"]]).saveas(peak_bed_file)
        pybedtools.BedTool.from_dataframe(gene_tss_df[["chrom", "start", "end", "name"]]).saveas(tss_bed_file)

    # Step 2: Run BEDTools overlap
    peak_bed = pybedtools.BedTool(peak_bed_file)
    tss_bed = pybedtools.BedTool(tss_bed_file)

    genes_near_peaks = find_genes_near_peaks(peak_bed, tss_bed, tss_distance_cutoff=max_peak_distance)
    
    genes_near_peaks = genes_near_peaks.rename(columns={"gene_id": "target_id"})
    genes_near_peaks["target_id"] = genes_near_peaks["target_id"].apply(standardize_name)
    
    if "TSS_dist" not in genes_near_peaks.columns:
        raise ValueError("Expected column 'TSS_dist' missing from find_genes_near_peaks output.")
    
    # Compute the distance score
    genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / float(distance_factor_scale))
    
    # Drop rows where the peak is too far from the gene
    genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= max_peak_distance]
    
    if promoter_bp is not None:
        # Subset to keep genes that are near gene promoters
        genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= int(promoter_bp)]
        
    if filter_to_nearest_gene:
        # Filter to use the gene closest to the peak
        genes_near_peaks = (genes_near_peaks.sort_values(["TSS_dist_score","TSS_dist","target_id"],
                             ascending=[False, True, True], kind="mergesort")
                .drop_duplicates(subset=["peak_id"], keep="first"))
        
    # Save and return the dataframe
    genes_near_peaks.to_parquet(peak_gene_dist_file, compression="snappy", engine="pyarrow")
    return genes_near_peaks

def _process_single_tf(tf, tf_df, peak_to_gene_dist_df, *, temperature: float = 1.0):
    """
    Process a single TF to compute the TF-peak probability and TF-TG distance score contributions.

    Parameters
    ----------
    tf : str
        The name of the TF.
    tf_df : pd.DataFrame
        A dataframe containing the sliding window scores for the TF.
    peak_to_gene_dist_df : pd.DataFrame
        A dataframe containing the peak-to-gene distance scores.
    temperature : float, optional
        The temperature parameter for the softmax function. Defaults to 1.0.

    Returns
    -------
    out : pd.DataFrame
        A dataframe containing the TF-peak probability and TF-TG distance score contributions.
    """
    def _softmax_1d_stable(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
        z = (x / float(tau)).astype(float)
        z -= z.max()              # numerical stability
        p = np.exp(z)
        s = p.sum()
        return p / s if s > 0 else np.full_like(p, 1.0 / len(p))
    
    # tf_df contains only one TF
    scores = tf_df["sliding_window_score"].to_numpy()
    tf_df = tf_df.copy()
    tf_df["tf_peak_prob"] = _softmax_1d_stable(scores, tau=temperature)

    merged = pd.merge(
        tf_df[["peak_id", "tf_peak_prob"]],
        peak_to_gene_dist_df[["peak_id", "target_id", "TSS_dist_score"]],
        on="peak_id",
        how="inner",
    )
    merged["tf_tg_contrib"] = merged["tf_peak_prob"] * merged["TSS_dist_score"]

    out = (
        merged.groupby("target_id", as_index=False)
            .agg(reg_potential=("tf_tg_contrib", "sum"),
                motif_density=("peak_id", "nunique"))
            .rename(columns={"target_id": "TG"})
    )
    out["TF"] = tf
    return out

def calculate_tf_tg_regulatory_potential(
    sliding_window_score_file: Union[str, Path], 
    tf_tg_reg_pot_file: Union[str, Path], 
    peak_to_gene_dist_file: Union[str, Path],
    num_cpu: int=8,
    ) -> pd.DataFrame:
    
    """
    Calculate the TF-TG regulatory potential (per TF mode) by integrating the sliding window scores with the peak-to-gene distance scores.

    Parameters
    ----------
    sliding_window_score_file : Union[str, Path]
        The file containing the sliding window scores computed by calculate_sliding_window_scores.
    tf_tg_reg_pot_file : Union[str, Path]
        The file where the TF-TG regulatory potential will be saved.
    peak_to_gene_dist_file : Union[str, Path]
        The file containing the peak-to-gene distance scores computed by calculate_peak_to_gene_distance_score.
    num_cpu : int, optional
        The number of CPUs to use for parallel processing. Defaults to 8.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the TF-TG regulatory potential scores.
    """
    sliding_window_score_file = Path(sliding_window_score_file)
    tf_tg_reg_pot_file = Path(tf_tg_reg_pot_file)
    peak_to_gene_dist_file = Path(peak_to_gene_dist_file)

    sliding_window_df = pd.read_parquet(sliding_window_score_file, engine="pyarrow")
    peak_to_gene_dist_df = pd.read_parquet(peak_to_gene_dist_file, engine="pyarrow")

    # Subset to only peaks in both the peak-TG distance dataset
    relevant_peaks = set(sliding_window_df["peak_id"].unique())
    peak_to_gene_dist_df = peak_to_gene_dist_df[peak_to_gene_dist_df["peak_id"].isin(relevant_peaks)]
    
    # Clean the column names and drop na
    sliding_window_df = sliding_window_df.dropna(subset=["TF", "peak_id", "sliding_window_score"])
    sliding_window_df["TF"] = sliding_window_df["TF"].apply(standardize_name)
    peak_to_gene_dist_df["target_id"] = peak_to_gene_dist_df["target_id"].apply(standardize_name)

    # Group the sliding window scores by TF
    tf_groups = {tf: df for tf, df in sliding_window_df.groupby("TF", sort=False)}
    
    logging.info(f"  - Calculating TF–TG regulatory potential: {len(tf_groups)} TFs | {num_cpu} CPUs")

    results = []
    
    with ProcessPoolExecutor(max_workers=num_cpu) as ex:
        futures = {
            ex.submit(_process_single_tf, tf, df, peak_to_gene_dist_df): tf
            for tf, df in tf_groups.items()
        }
        for fut in as_completed(futures):
            tf = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                logging.error(f"TF {tf} failed: {e}")

    # --- Concatenate all TF results ---
    if not results:
        raise RuntimeError("No TF results were successfully computed.")

    tf_tg_reg_pot = pd.concat(results, ignore_index=True)
    tf_tg_reg_pot["motif_density"] = np.log1p(tf_tg_reg_pot["motif_density"].fillna(0))
    
    tf_tg_reg_pot["TF"] = tf_tg_reg_pot["TF"].apply(standardize_name)
    tf_tg_reg_pot["TG"] = tf_tg_reg_pot["TG"].apply(standardize_name)
    
    logging.debug("TF-TG regulatory potential")
    logging.debug(tf_tg_reg_pot.head())

    # --- Save ---
    tf_tg_reg_pot.to_parquet(tf_tg_reg_pot_file, engine="pyarrow", compression="snappy")
    logging.debug(f"Saved TF–TG regulatory potential: {tf_tg_reg_pot.shape}")
    
    return tf_tg_reg_pot

def compute_minmax_expr_mean(
    tf_df: pd.DataFrame, 
    tg_df: pd.DataFrame, 
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the mean min-max normalized expression for TFs and TGs.
    """
    def _rowwise_minmax_mean(x: pd.DataFrame) -> pd.Series:
        xmin = x.min(axis=1)
        xmax = x.max(axis=1)
        denom = (xmax - xmin).replace(0, np.nan)          # avoid div-by-zero
        scaled = (x.sub(xmin, axis=0)).div(denom, axis=0) # row-wise scale
        scaled = scaled.fillna(0.0)                       # constant rows → 0
        return scaled.mean(axis=1)

    # Force index names so reset_index yields the correct key columns
    tf_means = _rowwise_minmax_mean(tf_df)
    tf_means.index = tf_means.index.astype(str)
    tf_means.index.name = "TF"
    mean_norm_tf_expr = tf_means.reset_index(name="mean_tf_expr")
    mean_norm_tf_expr["TF"] = mean_norm_tf_expr["TF"].str.upper()

    tg_means = _rowwise_minmax_mean(tg_df)
    tg_means.index = tg_means.index.astype(str)
    tg_means.index.name = "TG"
    mean_norm_tg_expr = tg_means.reset_index(name="mean_tg_expr")
    mean_norm_tg_expr["TG"] = mean_norm_tg_expr["TG"].str.upper()
    
    logging.debug(f"\nTF expression means: \n{mean_norm_tf_expr.head()}")
    logging.debug(f"\nTG expression means: \n{mean_norm_tg_expr.head()}")
    
    return mean_norm_tf_expr, mean_norm_tg_expr

def merge_tf_tg_attributes_with_combinations(
    tf_tg_df: pd.DataFrame, 
    tf_tg_reg_pot: pd.DataFrame, 
    mean_norm_tf_expr: pd.DataFrame, 
    mean_norm_tg_expr: pd.DataFrame,
    tf_tg_combo_attr_file: Union[str, Path],
    tf_vocab: Optional[set[str]] = None,
) -> pd.DataFrame:
    """
    Merge TF-TG regulatory potential and expression means with all TF-TG combinations.
    """
    logging.debug("\n  - Merging TF-TG Regulatory Potential")
    tf_tg_df = pd.merge(
        tf_tg_df,
        tf_tg_reg_pot,
        how="left",
        on=["TF", "TG"]
    ).fillna(0)
    logging.debug(tf_tg_df.head())

    logging.debug(f"    - Number of unique TFs: {tf_tg_df['TF'].nunique()}")

    logging.debug("\n  - Merging mean min-max normalized TF expression")
    tf_tg_df = pd.merge(
        tf_tg_df,
        mean_norm_tf_expr,
        how="left",
        on=["TF"]
    ).dropna(subset="mean_tf_expr")
    logging.debug(tf_tg_df.head())
    logging.debug(f"    - Number of unique TFs: {tf_tg_df['TF'].nunique()}")

    logging.debug("\n- Merging mean min-max normalized TG expression")

    tf_tg_df = pd.merge(
        tf_tg_df,
        mean_norm_tg_expr,
        how="left",
        on=["TG"]
    ).dropna(subset="mean_tg_expr")
    logging.debug(tf_tg_df.head())
    logging.debug(f"    - Number of unique TFs: {tf_tg_df['TF'].nunique()}")
    
    tf_tg_df["expr_product"] = tf_tg_df["mean_tf_expr"] * tf_tg_df["mean_tg_expr"]
    tf_tg_df["log_reg_pot"] = np.log1p(tf_tg_df["reg_potential"])
    tf_tg_df["motif_present"] = (tf_tg_df["motif_density"] > 0).astype(int)
    
    # Ensure consistent casing before validation
    tf_tg_df["TF"] = tf_tg_df["TF"].astype(str).str.upper()
    tf_tg_df["TG"] = tf_tg_df["TG"].astype(str).str.upper()
    
    def _assert_valid_tf_tg(df: pd.DataFrame, tf_vocab: Optional[set[str]] = None) -> pd.DataFrame:
        d = df.copy()
        d["TF"] = d["TF"].astype(str).str.strip()
        d["TG"] = d["TG"].astype(str).str.strip()

        # Only consider "too short" as bad if it's not in your TF vocabulary
        if tf_vocab is not None:
            bad_len = (d["TF"].str.len() <= 1) & (~d["TF"].isin(tf_vocab))
        else:
            # With no vocab, be extremely conservative: drop empties only
            bad_len = d["TF"].str.len() == 0

        if bad_len.any():
            n = int(bad_len.sum())
            examples = d.loc[bad_len, "TF"].unique()[:10]
            logging.warning(f"{n} rows dropped due to invalid TF values; examples: {examples}")
            d = d[~bad_len].copy()

        if d.empty:
            raise ValueError("All rows were dropped during TF/TG validation. Upstream mutation likely overwrote TFs.")
        return d

    # Validate and drop any damaged rows (e.g., TF == "T")
    tf_tg_df = _assert_valid_tf_tg(tf_tg_df, tf_vocab=tf_vocab)
    
    tf_tg_df.to_parquet(tf_tg_combo_attr_file, index=False)
    
    return tf_tg_df

def make_chrom_gene_tss_df(gene_tss_file: Union[str, Path], chrom_id: str, genome_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Builds a gene TSS dataframe for a given chromosome and dataset.

    Parameters
    ----------
    gene_tss_file : str
        Gene TSS bed file with columns 'chrom', 'start', 'end', 'name'
    chrom_id : str
        Chromosome ID to process
    genome_dir : str
        Directory containing preprocessed data for this dataset

    Returns
    -------
    pd.DataFrame
        Gene TSS dataframe with columns 'chrom', 'start', 'end', 'name'
    """
    gene_tss_bed = pybedtools.BedTool(gene_tss_file)
    gene_tss_df = (
        gene_tss_bed
        .filter(lambda x: x.chrom == chrom_id)
        .saveas(os.path.join(genome_dir, f"{chrom_id}_gene_tss.bed"))
        .to_dataframe()
        .sort_values(by="start", ascending=True)
        )
    gene_tss_df["name"] = gene_tss_df["name"].astype(str).map(standardize_name)
    gene_tss_df = gene_tss_df.drop_duplicates(subset=["name"], keep="first")
    return gene_tss_df

# ----- MultiomicTransformer Chromsome-Specific Dataset -----
def build_global_tg_vocab(gene_tss_file: Union[str, Path], vocab_file: Union[str, Path]) -> dict[str, int]:
    """
    Builds a global TG vocab from the TSS file with contiguous IDs [0..N-1].
    Overwrites existing vocab if it's missing or non-contiguous.
    
    Parameters
    ----------
    gene_tss_file : str | Path
        Gene TSS bed file with columns 'chrom', 'start', 'end', 'name'
    vocab_file : str | Path
        File to write the global TG vocab to
    
    Returns
    -------
    dict[str, int]
        Global TG vocab with contiguous IDs [0..N-1]
    """
    # 1) Load all genes genome-wide (bed: chrom start end name)
    gene_tss_bed = pybedtools.BedTool(gene_tss_file)
    gene_tss_df = gene_tss_bed.to_dataframe().sort_values(by="start", ascending=True)

    # 2) Canonical symbol list (MUST match downstream normalization)
    gene_tss_df["name"] = gc.canonicalize_series(gene_tss_df["name"])
    names = sorted(set(gene_tss_df["name"]))  # unique + stable order
    logging.info(f"  - Writing global TG vocab with {len(names)} genes")

    # 3) Build fresh contiguous mapping
    vocab = {name: i for i, name in enumerate(names)}

    # 4) Atomic overwrite
    tmp = str(vocab_file) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(vocab, f)
    os.replace(tmp, vocab_file)

    return vocab

def create_single_cell_tensors(
    gene_tss_df: pd.DataFrame,
    sample_names: list[str],
    dataset_processed_data_dir: Path,
    tg_vocab: dict[str, int],
    tf_vocab: dict[str, int],
    chrom_id: str,
    single_cell_dir: Path,
):
    """
    Builds single-cell tensors for a given chromosome and dataset.

    Parameters
    ----------
    gene_tss_df : pd.DataFrame
        Gene TSS bed file with columns 'chrom', 'start', 'end', 'name'
    sample_names : list[str]
        List of sample names to process
    dataset_processed_data_dir : Path
        Directory containing preprocessed data for this dataset
    tg_vocab : dict[str, int]
        Global TG vocabulary
    tf_vocab : dict[str, int]
        Global TF vocabulary
    chrom_id : str
        Chromosome ID to process
    single_cell_dir : Path
        Directory to save single-cell tensors

    Returns
    -------
    None
    """
    gene_tss_df = gene_tss_df.copy()
    gene_tss_df["name"] = gene_tss_df["name"].astype(str).map(standardize_name)

    # chromosome-specific TG list
    chrom_tg_names = set(gene_tss_df["name"].unique())

    for sample_name in sample_names:
        sample_processed_data_dir = dataset_processed_data_dir / sample_name

        tg_sc_file = sample_processed_data_dir / "scRNA_seq_processed.parquet"
        re_sc_file = sample_processed_data_dir / "scATAC_seq_processed.parquet"

        if not (tg_sc_file.exists() and re_sc_file.exists()):
            logging.debug(f"[{sample_name}] Skipping: missing TG/RE single-cell files")
            continue

        logging.debug(f"[{sample_name} | {chrom_id}] Building single-cell tensors")

        TG_sc = pd.read_parquet(tg_sc_file)
        TG_sc.index = TG_sc.index.astype(str).map(standardize_name)
        RE_sc = pd.read_parquet(re_sc_file)

        # Output dir for this sample + chromosome
        sample_sc_dir = single_cell_dir / sample_name
        sample_sc_dir.mkdir(parents=True, exist_ok=True)

        # --- TG tensor: restrict to this chromosome + vocab ---
        tg_rows = [g for g in TG_sc.index if g in chrom_tg_names]
        TG_sc_chr = TG_sc.loc[tg_rows]

        tg_tensor_sc, tg_names_kept, tg_ids = align_to_vocab(
            TG_sc_chr.index.tolist(),
            tg_vocab,
            torch.tensor(TG_sc_chr.values, dtype=torch.float32),
            label="TG",
        )

        torch.save(
            tg_tensor_sc,
            sample_sc_dir / f"{sample_name}_tg_tensor_singlecell_{chrom_id}.pt",
        )
        torch.save(
            torch.tensor(tg_ids, dtype=torch.long),
            sample_sc_dir / f"{sample_name}_tg_ids_singlecell_{chrom_id}.pt",
        )
        atomic_json_dump(
            tg_names_kept,
            sample_sc_dir / f"{sample_name}_tg_names_singlecell_{chrom_id}.json",
        )

        # --- ATAC tensor: restrict peaks to this chromosome ---
        re_rows = [p for p in RE_sc.index if p.startswith(f"{chrom_id}:")]
        RE_sc_chr = RE_sc.loc[re_rows]

        atac_tensor_sc = torch.tensor(RE_sc_chr.values, dtype=torch.float32)
        torch.save(
            atac_tensor_sc,
            sample_sc_dir / f"{sample_name}_atac_tensor_singlecell_{chrom_id}.pt",
        )

        # --- TF tensor: subset of TGs that are TFs in the global vocab ---
        tf_rows = [g for g in TG_sc.index if g in tf_vocab]
        if tf_rows:
            TF_sc = TG_sc.loc[tf_rows]
            tf_tensor_sc, tf_names_kept, tf_ids = align_to_vocab(
                TF_sc.index.tolist(),
                tf_vocab,
                torch.tensor(TF_sc.values, dtype=torch.float32),
                label="TF",
            )

            torch.save(
                tf_tensor_sc,
                sample_sc_dir / f"{sample_name}_tf_tensor_singlecell_{chrom_id}.pt",
            )
            torch.save(
                torch.tensor(tf_ids, dtype=torch.long),
                sample_sc_dir / f"{sample_name}_tf_ids_singlecell_{chrom_id}.pt",
            )
            atomic_json_dump(
                tf_names_kept,
                sample_sc_dir / f"{sample_name}_tf_names_singlecell_{chrom_id}.json",
            )
        else:
            logging.warning(
                f"[{sample_name} | {chrom_id}] No TF rows found in single-cell TG matrix"
            )

def aggregate_pseudobulk_datasets(
    sample_names: list[str],
    dataset_processed_data_dir: Path,
    chroms: list[str],
    gc: GeneCanonicalizer,
    force_recalculate: bool = False,
):
    """
    Aggregate pseudobulk datasets across all samples and chromosomes.

    Parameters
    ----------
    sample_names : list[str]
        List of sample names to process
    dataset_processed_data_dir : Path
        Directory containing preprocessed data for this dataset
    chroms : list[str]
        List of chromosome IDs to process
    gc : GeneCanonicalizer
        Gene canonicalizer object
    force_recalculate : bool, optional
        Whether to recompute everything (default is False)

    Returns
    -------
    total_TG_pseudobulk_global : pd.DataFrame
        Global TG pseudobulk across all samples
    pseudobulk_chrom_dict : dict[str, dict]
        Per-chromosome aggregates of TG pseudobulk and RE pseudobulk
    """
    # ----- Helpers -----
    def _canon_index_sum(df: pd.DataFrame, gc) -> pd.DataFrame:
        """Canonicalize df.index with GeneCanonicalizer and sum duplicate rows."""
        if df.empty:
            return df
        mapped = gc.canonicalize_series(pd.Series(df.index, index=df.index))
        out = df.copy()
        out.index = mapped.values
        out = out[out.index != ""]  # drop unmapped
        if not out.index.is_unique:
            out = out.groupby(level=0).sum()
        return out

    def _canon_series_same_len(s: pd.Series, gc) -> pd.Series:
        """Canonicalize to same length; replace non-mapped with '' (caller filters)."""
        cs = gc.canonicalize_series(s.astype(str))
        return cs.fillna("")

    def _agg_sum(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            raise ValueError("No DataFrames provided to aggregate.")
        if len(dfs) == 1:
            return dfs[0]
        return pd.concat(dfs).groupby(level=0).sum()

    def _agg_first(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            raise ValueError("No DataFrames provided to aggregate.")
        if len(dfs) == 1:
            return dfs[0]
        return pd.concat(dfs).groupby(level=0).first()

    total_tg_pseudobulk_path = dataset_processed_data_dir / "total_TG_pseudobulk_global.parquet"
    pseudobulk_chrom_dict_path = dataset_processed_data_dir / "pseudobulk_chrom_dict.pkl"

    # Decide whether to recompute everything
    need_recalc = (
        force_recalculate
        or not total_tg_pseudobulk_path.is_file()
        or not pseudobulk_chrom_dict_path.is_file()
    )

    if need_recalc:
        logging.info("  - Loading processed pseudobulk datasets:")
        logging.info(f"   - Sample names: {sample_names}")
        logging.info(f"   - Looking for processed samples in {dataset_processed_data_dir}")

        # ---- 1) Build per-sample TG pseudobulk (canonicalized) ----
        per_sample_TG: dict[str, pd.DataFrame] = {}
        for sample_name in sample_names:
            sample_raw_dir = dataset_processed_data_dir / sample_name
            tg_path = sample_raw_dir / "TG_pseudobulk.parquet"
            TG_pseudobulk = pd.read_parquet(tg_path, engine="pyarrow")
            TG_pseudobulk = _canon_index_sum(TG_pseudobulk, gc)
            per_sample_TG[sample_name] = TG_pseudobulk

        # Global TG pseudobulk across all samples
        total_TG_pseudobulk_global = _agg_sum(list(per_sample_TG.values()))

        # ---- 2) Build per-chromosome aggregates ----
        pseudobulk_chrom_dict: dict[str, dict] = {}
        logging.info("  - Aggregating per-chromosome pseudobulk datasets:")
        for chrom_id in chroms:
            logging.info(f"   - Aggregating data for {chrom_id}")

            TG_pseudobulk_samples = []
            RE_pseudobulk_samples = []
            peaks_df_samples = []

            # gene TSS for this chromosome
            chrom_tss_path = GENOME_DIR / f"{chrom_id}_gene_tss.bed"
            if not chrom_tss_path.is_file():
                gene_tss_chrom = make_chrom_gene_tss_df(
                    gene_tss_file=GENE_TSS_FILE,
                    chrom_id=chrom_id,
                    genome_dir=GENOME_DIR,
                )
            else:
                gene_tss_chrom = pd.read_csv(
                    chrom_tss_path,
                    sep="\t",
                    header=None,
                    usecols=[0, 1, 2, 3],
                )
                gene_tss_chrom = gene_tss_chrom.rename(
                    columns={0: "chrom", 1: "start", 2: "end", 3: "name"}
                )

            gene_tss_chrom["name"] = _canon_series_same_len(gene_tss_chrom["name"], gc)
            gene_tss_chrom = gene_tss_chrom[gene_tss_chrom["name"] != ""]
            gene_tss_chrom = gene_tss_chrom.drop_duplicates(subset=["name"], keep="first")
            genes_on_chrom = gene_tss_chrom["name"].tolist()

            for sample_name in sample_names:
                sample_raw_dir = dataset_processed_data_dir / sample_name

                # RE pseudobulk: peaks x metacells (loaded from per-sample raw directory)
                re_path = sample_raw_dir / "RE_pseudobulk.parquet"
                RE_pseudobulk = pd.read_parquet(re_path, engine="pyarrow")

                # TG: restrict to genes on this chrom
                TG_chr_specific = per_sample_TG[sample_name].loc[
                    per_sample_TG[sample_name].index.intersection(genes_on_chrom)
                ]

                # RE: restrict to this chrom (handle both chr:start-end and chr-start-end formats)
                mask_colon = RE_pseudobulk.index.str.startswith(f"{chrom_id}:")
                mask_dash = RE_pseudobulk.index.str.startswith(f"{chrom_id}-")
                RE_chr_specific = RE_pseudobulk[mask_colon | mask_dash]
                
                logging.debug(f"      - Sample {sample_name}, {chrom_id}: {len(RE_chr_specific)} peaks matched")
                if len(RE_chr_specific) > 0:
                    logging.debug(f"      - First few peaks: {RE_chr_specific.index[:3].tolist()}")

                # Build peaks df from RE index
                # Handle both colon-separated (chr:start-end) and dash-separated (chr-start-end) formats
                # Normalize all peaks to chr:start-end format for consistent storage
                peaks_df = (
                    RE_chr_specific.index.to_series()
                    .apply(normalize_peak_format)
                    .str.split("[-:]", n=2, expand=True, regex=True)
                    .rename(columns={0: "chrom", 1: "start", 2: "end"})
                )
                peaks_df["start"] = peaks_df["start"].astype(int)
                peaks_df["end"] = peaks_df["end"].astype(int)
                peaks_df["peak_id"] = RE_chr_specific.index.to_series().apply(normalize_peak_format)

                TG_pseudobulk_samples.append(TG_chr_specific)
                RE_pseudobulk_samples.append(RE_chr_specific)
                peaks_df_samples.append(peaks_df)

            total_TG_pseudobulk_chr = _agg_sum(TG_pseudobulk_samples)
            total_RE_pseudobulk_chr = _agg_sum(RE_pseudobulk_samples)
            total_peaks_df = _agg_first(peaks_df_samples)
            
            # Normalize peak IDs to consistent chr:start:end format
            total_RE_pseudobulk_chr.index = total_RE_pseudobulk_chr.index.to_series().apply(normalize_peak_format)
            
            logging.debug(f"   - {chrom_id}: Aggregated {len(total_RE_pseudobulk_chr)} RE peaks, {len(total_TG_pseudobulk_chr)} genes")
            logging.debug(f"   - {chrom_id}: total_peaks_df shape: {total_peaks_df.shape}")
            if len(total_peaks_df) > 0:
                logging.debug(f"   - {chrom_id}: Sample peak IDs: {total_peaks_df['peak_id'].head(3).tolist()}")

            pseudobulk_chrom_dict[chrom_id] = {
                "total_TG_pseudobulk_chr": total_TG_pseudobulk_chr,
                "total_RE_pseudobulk_chr": total_RE_pseudobulk_chr,
                "total_peaks_df": total_peaks_df,
            }

        # ---- 3) Save aggregates ----
        total_TG_pseudobulk_global.to_parquet(
            total_tg_pseudobulk_path,
            engine="pyarrow",
            compression="snappy",
        )
        with open(pseudobulk_chrom_dict_path, "wb") as f:
            pickle.dump(pseudobulk_chrom_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        # Load both from disk
        logging.info("  - Found existing global and per-chrom pseudobulk; loading...")
        total_TG_pseudobulk_global = pd.read_parquet(
            total_tg_pseudobulk_path,
            engine="pyarrow",
        )
        with open(pseudobulk_chrom_dict_path, "rb") as f:
            pseudobulk_chrom_dict = pickle.load(f)

    return total_TG_pseudobulk_global, pseudobulk_chrom_dict

def create_or_load_genomic_windows(
    window_size: int,
    chrom_id: str,
    genome_window_file: Union[str, Path],
    chrom_sizes_file: Union[str, Path],
    force_recalculate: bool = False,
    promoter_only: bool = False,
):
    """
    Create or load fixed-size genomic windows for a chromosome.

    When `promoter_only=True`, skip windows entirely and return an empty
    DataFrame with the expected schema so callers don't break.

    Parameters
    ----------
    window_size : int
        Size of genomic windows to create.
    chrom_id : str
        Chromosome identifier.
    genome_window_file : Union[str, Path]
        Path to save/load genomic windows to/from.
    chrom_sizes_file : Union[str, Path]
        Path to chromosome sizes file (e.g. UCSC chrom.sizes.txt).
    force_recalculate : bool, optional
        Whether to recompute genomic windows even if they already exist.
        Defaults to False.
    promoter_only : bool, optional
        Whether to skip creating genomic windows and return an empty DataFrame.
        Defaults to False.

    Returns
    -------
    pd.DataFrame
        DataFrame with genomic windows, with columns "chrom", "start", "end", and "win_idx".
    """
    
    # Promoter-centric evaluation: no windows needed
    if promoter_only:
        return pd.DataFrame(columns=["chrom", "start", "end", "win_idx"])

    if not os.path.exists(genome_window_file) or force_recalculate:
        genome_windows = pybedtools.bedtool.BedTool().window_maker(g=chrom_sizes_file, w=window_size)
        # Ensure consistent column names regardless of BedTool defaults
        chrom_windows = (
            genome_windows
            .filter(lambda x: x.chrom == chrom_id)
            .saveas(genome_window_file)
            .to_dataframe(names=["chrom", "start", "end"])
        )
        logging.debug(f"  - Created {chrom_windows.shape[0]} windows")
    else:
        logging.debug("\nLoading existing genomic windows")
        chrom_windows = pybedtools.BedTool(genome_window_file).to_dataframe(names=["chrom", "start", "end"])

    chrom_windows = chrom_windows.reset_index(drop=True)
    chrom_windows["win_idx"] = chrom_windows.index
    return chrom_windows

def make_peak_to_window_map(peaks_bed: pd.DataFrame, windows_bed: pd.DataFrame, peaks_as_windows: bool = True,) -> dict[str, int]:
    """
    Map each peak to the window it overlaps the most.
    Ensures the BedTool 'name' field is exactly the `peak_id` column.
    Parameters
    ----------
    peaks_bed : pd.DataFrame
        DataFrame of peaks, with columns "chrom", "start", "end", "peak_id".
    windows_bed : pd.DataFrame
        DataFrame of genomic windows, with columns "chrom", "start", "end", "win_idx".
    peaks_as_windows : bool, optional
        Whether to treat peaks as genomic windows. Defaults to True.
    Returns
    -------
    dict[str, int]
        Mapping of peak_id to win_idx, with the peak_id as key and the win_idx as value.
    """
    
    pb = peaks_bed.copy()
    required = ["chrom", "start", "end", "peak_id"]
    missing = [c for c in required if c not in pb.columns]
    if missing:
        raise ValueError(f"peaks_bed missing columns: {missing}")

    pb["chrom"] = pb["chrom"].astype(str)
    pb["start"] = pb["start"].astype(int)
    pb["end"]   = pb["end"].astype(int)
    pb["peak_id"] = pb["peak_id"].astype(str).str.strip()
    
    if peaks_as_windows or windows_bed is None or windows_bed.empty:
        # stable order mapping: as they appear
        return {pid: int(i) for i, pid in enumerate(pb["peak_id"].tolist())}

    # BedTool uses the 4th column as "name. make it explicitly 'peak_id'
    pb_for_bed = pb[["chrom", "start", "end", "peak_id"]].rename(columns={"peak_id": "name"})
    bedtool_peaks   = pybedtools.BedTool.from_dataframe(pb_for_bed)

    # windows: enforce expected columns & dtypes
    wb = windows_bed.copy()
    wr = ["chrom", "start", "end", "win_idx"]
    miss_w = [c for c in wr if c not in wb.columns]
    if miss_w:
        raise ValueError(f"windows_bed missing columns: {miss_w}")
    wb["chrom"] = wb["chrom"].astype(str)
    wb["start"] = wb["start"].astype(int)
    wb["end"]   = wb["end"].astype(int)
    wb["win_idx"] = wb["win_idx"].astype(int)
    bedtool_windows = pybedtools.BedTool.from_dataframe(wb[["chrom", "start", "end", "win_idx"]])

    overlaps = {}
    for iv in bedtool_peaks.intersect(bedtool_windows, wa=True, wb=True):
        # left fields (peak): chrom, start, end, name
        peak_id = iv.name  # guaranteed to be the 'name' we set = peak_id
        # right fields (window): ... chrom, start, end, win_idx (as the last field)
        win_idx = int(iv.fields[-1])

        peak_start, peak_end = int(iv.start), int(iv.end)
        win_start, win_end   = int(iv.fields[-3]), int(iv.fields[-2])
        overlap_len = max(0, min(peak_end, win_end) - max(peak_start, win_start))

        if overlap_len <= 0:
            continue
        overlaps.setdefault(peak_id, []).append((overlap_len, win_idx))

    # resolve ties by max-overlap then random
    mapping = {}
    for pid, lst in overlaps.items():
        if not lst: 
            continue
        max_ol = max(lst, key=lambda x: x[0])[0]
        candidates = [w for ol, w in lst if ol == max_ol]
        mapping[str(pid)] = int(random.choice(candidates))

    return mapping

def align_to_vocab(names: list[str], vocab: dict[str, int], tensor_all: torch.Tensor, label: str = "genes") -> tuple[torch.Tensor, list[str], list[int]]:
    """
    Restrict to the subset of names that exist in the global vocab.
    Returns:
      aligned_tensor : [num_kept, C] (chromosome-specific subset)
      kept_names     : list[str] of kept names (order = aligned_tensor rows)
      kept_ids       : list[int] global vocab indices for kept names
    """
    kept_ids = []
    kept_names = []
    aligned_rows = []

    for i, n in enumerate(names):
        vid = vocab.get(n)
        if vid is not None:
            kept_ids.append(vid)
            kept_names.append(n)
            aligned_rows.append(tensor_all[i])

    if not kept_ids:
        raise ValueError(f"No {label} matched the global vocab.")

    aligned_tensor = torch.stack(aligned_rows, dim=0)  # [num_kept, num_cells]

    return aligned_tensor, kept_names, kept_ids

def build_motif_mask(tf_names: list[str], tg_names: list[str], sliding_window_df: pd.DataFrame, genes_near_peaks: pd.DataFrame) -> np.ndarray:
    """
    Build motif mask [TG x TF] with max logodds per (TG, TF).

    Parameters
    ----------
    tf_names : list[str]
        List of TF names.
    tg_names : list[str]
        List of TG names.
    sliding_window_df : pd.DataFrame
        DataFrame containing the sliding window scores.
    genes_near_peaks : pd.DataFrame
        DataFrame containing the peak-to-gene distance scores.

    Returns
    -------
    np.ndarray
        The motif mask [TG x TF] with max logodds per (TG, TF).
    """
    
    # map TFs and TGs to ids
    tf_index = {tf: i for i, tf in enumerate(tf_names)}
    tg_index = {tg: i for i, tg in enumerate(tg_names)}

    # restrict to known TFs
    sliding_window_df = sliding_window_df[sliding_window_df["TF"].isin(tf_index)]
    
    # merge motif hits with target genes (peak_id → TG)
    merged = sliding_window_df.merge(
        genes_near_peaks[["peak_id", "target_id"]],
        on="peak_id",
        how="inner"
    )

    # drop TGs not in tg_index
    merged = merged[merged["target_id"].isin(tg_index)]

    # map names → indices
    merged["tf_idx"] = merged["TF"].map(tf_index)
    merged["tg_idx"] = merged["target_id"].map(tg_index)

    # groupby max(sliding_window_score) per (TG, TF)
    agg = merged.groupby(["tg_idx", "tf_idx"])["sliding_window_score"].max().reset_index()

    # construct sparse COO
    mask = sp.coo_matrix(
        (agg["sliding_window_score"], (agg["tg_idx"], agg["tf_idx"])),
        shape=(len(tg_names), len(tf_names)),
        dtype=np.float32
    ).toarray()

    return mask

def precompute_input_tensors(
    output_dir: str,
    genome_wide_tf_expression: np.ndarray,   # [num_TF, num_cells]
    genome_wide_tg_expression: np.ndarray,                   # [num_TG_chr, num_cells]
    total_RE_pseudobulk_chr: pd.DataFrame,                 # pd.DataFrame: rows=peak_id, cols=metacells
    window_map: dict,
    windows: pd.DataFrame,                            # pd.DataFrame with shape[0] = num_windows
    dtype: torch.dtype = torch.float32,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

    """
    Builds & saves:
      - tf_tensor_all.pt                        [num_TF, num_cells]
      - tg_tensor_all_{chr}.pt                  [num_TG_chr, num_cells]
      - atac_window_tensor_all_{chr}.pt         [num_windows, num_cells]

    Returns:
      (tf_tensor_all, tg_tensor_all, atac_window_tensor_all)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---- TF tensor ----
    tf_tensor_all = torch.as_tensor(
        np.asarray(genome_wide_tf_expression, dtype=np.float32), dtype=dtype
    )

    # ---- TG tensor ----
    tg_tensor_all = torch.as_tensor(
        np.asarray(genome_wide_tg_expression, dtype=np.float32), dtype=dtype
    )

    # ---- ATAC window tensor ----
    num_windows = int(windows.shape[0])
    num_peaks   = int(total_RE_pseudobulk_chr.shape[0])
    
    logging.debug(f"  - precompute_input_tensors: {num_windows} windows, {num_peaks} peaks")
    logging.debug(f"  - window_map has {len(window_map)} entries")
    if len(window_map) > 0:
        sample_peaks = list(window_map.keys())[:3]
        logging.debug(f"  - Sample peak IDs from window_map: {sample_peaks}")
    if num_peaks > 0:
        sample_re_peaks = total_RE_pseudobulk_chr.index[:3].tolist()
        logging.debug(f"  - Sample peak IDs from RE_pseudobulk: {sample_re_peaks}")

    rows, cols, vals = [], [], []
    peak_to_idx = {p: i for i, p in enumerate(total_RE_pseudobulk_chr.index)}
    for peak_id, win_idx in window_map.items():
        peak_idx = peak_to_idx.get(peak_id)
        if peak_idx is not None and 0 <= win_idx < num_windows:
            rows.append(win_idx)
            cols.append(peak_idx)
            vals.append(1.0)

    if not rows:
        logging.warning("No peaks from window_map matched rows in total_RE_pseudobulk_chr. Returning None.")
        logging.warning(f"  - Checked {len(window_map)} window_map entries against {num_peaks} RE peaks")
        return None, None, None

    W = sp.csr_matrix((vals, (rows, cols)), shape=(num_windows, num_peaks))
    atac_window = W @ total_RE_pseudobulk_chr.values  # [num_windows, num_cells]

    atac_window_tensor_all = torch.as_tensor(
        atac_window.astype(np.float32), dtype=dtype
    )

    return tf_tensor_all, tg_tensor_all, atac_window_tensor_all

def build_distance_bias(
    genes_near_peaks: pd.DataFrame,
    window_map: Dict[str, int],
    tg_names_kept: Iterable[str],
    num_windows: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    mode: str = "logsumexp",   # "max" | "sum" | "mean" | "logsumexp"
    prune_empty_windows: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, int], List[int]]]:
    """
    Build a [num_windows x num_tg_kept] (or pruned) distance-bias tensor aligned to the kept TGs.

    If prune_empty_windows is True, windows with no peaks (no entries in genes_near_peaks)
    are removed and indices are compacted.

    Returns:
        If prune_empty_windows is False:
            dist_bias
        If True:
            dist_bias, new_window_map, kept_window_indices

        where:
            - dist_bias: [num_kept_windows, num_tg_kept]
            - new_window_map: peak_id -> new_window_idx (only for kept windows)
            - kept_window_indices: list mapping new_window_idx -> original_window_idx
    """
    tg_names_kept = list(tg_names_kept)
    num_tg_kept = len(tg_names_kept)
    tg_index_map = {tg: i for i, tg in enumerate(tg_names_kept)}

    from collections import defaultdict
    scores_map = defaultdict(list)

    # Collect all scores for each (window, TG)
    for _, row in genes_near_peaks.iterrows():
        win_idx = window_map.get(row["peak_id"])
        tg_idx  = tg_index_map.get(row["target_id"])
        if win_idx is not None and tg_idx is not None:
            scores_map[(win_idx, tg_idx)].append(float(row["TSS_dist_score"]))

    # If not pruning, keep all windows regardless of whether they have peaks
    if not prune_empty_windows:
        dist_bias = torch.zeros((num_windows, num_tg_kept), dtype=dtype, device=device)
        for (win_idx, tg_idx), scores in scores_map.items():
            scores_tensor = torch.tensor(scores, dtype=dtype, device=device)
            if mode == "max":
                dist_bias[win_idx, tg_idx] = scores_tensor.max()
            elif mode == "sum":
                dist_bias[win_idx, tg_idx] = scores_tensor.sum()
            elif mode == "mean":
                dist_bias[win_idx, tg_idx] = scores_tensor.mean()
            elif mode == "logsumexp":
                dist_bias[win_idx, tg_idx] = torch.logsumexp(scores_tensor, dim=0)
            else:
                raise ValueError(f"Unknown pooling mode: {mode}")
        return dist_bias

    # --- Pruned version: only keep windows that appear in scores_map ---
    used_windows = sorted({win for (win, _) in scores_map.keys()})
    num_kept = len(used_windows)
    old2new = {w: i for i, w in enumerate(used_windows)}

    dist_bias = torch.zeros((num_kept, num_tg_kept), dtype=dtype, device=device)

    for (win_idx, tg_idx), scores in scores_map.items():
        new_win_idx = old2new[win_idx]
        scores_tensor = torch.tensor(scores, dtype=dtype, device=device)

        if mode == "max":
            dist_bias[new_win_idx, tg_idx] = scores_tensor.max()
        elif mode == "sum":
            dist_bias[new_win_idx, tg_idx] = scores_tensor.sum()
        elif mode == "mean":
            dist_bias[new_win_idx, tg_idx] = scores_tensor.mean()
        elif mode == "logsumexp":
            dist_bias[new_win_idx, tg_idx] = torch.logsumexp(scores_tensor, dim=0)
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

    # Build a new window_map in the compressed index space
    new_window_map: Dict[str, int] = {}
    used_set = set(used_windows)
    for peak_id, old_idx in window_map.items():
        if old_idx in used_set:
            new_window_map[peak_id] = old2new[old_idx]

    kept_window_indices = used_windows

    return dist_bias, new_window_map, kept_window_indices

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # ----- SETUP -----
    # Parse command-line arguments
    args = parse_preprocessing_args()
    num_cpu = args.num_cpu
    
    # Setup global variables from parsed arguments
    setup_global_variables(args)
    
    # Create an information json file to log information about the experiment
    summary_file = EXPERIMENT_DIR / "experiment_info.json"
    
    # TF and TG vocab files
    common_tf_vocab_file: Path =  COMMON_DATA / f"tf_vocab.json"
    common_tg_vocab_file: Path =  COMMON_DATA / f"tg_vocab.json"
    
    os.makedirs(COMMON_DATA, exist_ok=True)
    
    # Genome files
    genome_fasta_file = GENOME_DIR / (ORGANISM_CODE + ".fa.gz")
    chrom_sizes_file = GENOME_DIR / f"{ORGANISM_CODE}.chrom.sizes"
    
    if not os.path.isdir(GENOME_DIR):
        os.makedirs(GENOME_DIR)
    
    download_genome_fasta(
        organism_code=ORGANISM_CODE,
        save_dir=GENOME_DIR
    )
        
    if not os.path.isfile(chrom_sizes_file):
        download_chrom_sizes(
            organism_code=ORGANISM_CODE,
            save_dir=GENOME_DIR
        )
        
    # Check organism code
    if ORGANISM_CODE == "mm10":
        ensemble_dataset_name = "mmusculus_gene_ensembl"
    elif ORGANISM_CODE == "hg38":
        ensemble_dataset_name = "hsapiens_gene_ensembl"
    else:
        raise ValueError(f"Organism not recognized: {ORGANISM_CODE} (must be 'mm10' or 'hg38').")
    
    if not os.path.isfile(GENE_TSS_FILE):
        download_gene_tss_file(
            save_file=GENE_TSS_FILE,
            gene_dataset_name=ensemble_dataset_name,
        )
    
    # Download organism-specific NCBI gene info and Ensembl GTF
    download_ncbi_gene_info(organism_code=ORGANISM_CODE)
    
    # Download GTF with organism-appropriate defaults
    if ORGANISM_CODE == "mm10":
        download_ensembl_gtf(organism_code=ORGANISM_CODE, release=115, assembly="GRCm39", decompress=False)
    elif ORGANISM_CODE == "hg38":
        download_ensembl_gtf(organism_code=ORGANISM_CODE, release=113, assembly="GRCh38", decompress=False)
    
    if ORGANISM_CODE == "mm10":
        species_taxid = "10090"
        species_file_name = "Mus_musculus"
        gtf_assembly = "GRCm39.115"
        
    elif ORGANISM_CODE == "hg38":
        species_taxid = "9606"
        species_file_name = "Homo_sapiens"
        gtf_assembly = "GRCh38.113"

    gc = GeneCanonicalizer(use_mygene=False)
    gc.load_gtf(str(GTF_FILE_DIR / f"{species_file_name}.{gtf_assembly}.gtf.gz"))
    gc.load_ncbi_gene_info(str(NCBI_FILE_DIR / f"{species_file_name}.gene_info.gz"), species_taxid=species_taxid)
    logging.info(f"Map sizes: {gc.coverage_report()}")
        
    # Format the gene TSS file to BED format (chrom, start, end, name)
    gene_tss_df = make_gene_tss_bed_file(
        gene_tss_file=GENE_TSS_FILE,
        genome_dir=GENOME_DIR
    )
    
    # ----- BUILD GLOBAL TG VOCAB FROM THE GENE TSS FILE-----
    tg_vocab = build_global_tg_vocab(GENE_TSS_FILE, common_tg_vocab_file)

    # PKN files
    string_csv_file = STRING_DIR / f"string_{ORGANISM_CODE}_pkn.csv"
    trrust_csv_file = TRRUST_DIR / f"trrust_{ORGANISM_CODE}_pkn.csv"
    kegg_csv_file = KEGG_DIR / f"kegg_{ORGANISM_CODE}_pkn.csv"
    
    logging.info(f"\nFORCE_RECALCULATE: {FORCE_RECALCULATE}")
    
    PROCESS_SAMPLE_DATA = True
    logging.info(f"PROCESS_SAMPLE_DATA: {PROCESS_SAMPLE_DATA}")
    PROCESS_CHROMOSOME_SPECIFIC_DATA = True
    logging.info(f"PROCESS_CHROMOSOME_SPECIFIC_DATA: {PROCESS_CHROMOSOME_SPECIFIC_DATA}")
    
    # ----- SAMPLE-SPECIFIC PREPROCESSING -----     
    if PROCESS_SAMPLE_DATA == True:
        def _per_sample_worker(sample_name: str) -> dict:
            try:
                sample_input_dir = (
                    RAW_SINGLE_CELL_DATA / sample_name
                    if RAW_SINGLE_CELL_DATA
                    else RAW_DATA / DATASET_NAME / sample_name
                )
                out_dir = SAMPLE_PROCESSED_DATA_DIR / sample_name
                out_dir.mkdir(parents=True, exist_ok=True)
                SAMPLE_DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                
                process_or_load_rna_atac_data(
                    sample_input_dir,
                    force_recalculate=FORCE_RECALCULATE,
                    raw_10x_rna_data_dir=RAW_10X_RNA_DATA_DIR / sample_name if RAW_10X_RNA_DATA_DIR else None,
                    raw_atac_peak_file=RAW_ATAC_PEAK_MATRIX_FILE,
                    sample_name=sample_name,
                    sample_processed_dir=out_dir,
                    neighbors_k=NEIGHBORS_K,
                    pca_components=PCA_COMPONENTS,
                    hops=HOPS,
                    self_weight=SELF_WEIGHT,
                    load=False
                )
                return {"sample": sample_name, "ok": True}
            
            except Exception as e:
                tb = traceback.format_exc()
                logging.error(f"[{sample_name}] failed: {e}\n{tb}")
                return {"sample": sample_name, "ok": False, "error": str(e)}
        
        # Pre-process samples
        results = []
        with ProcessPoolExecutor(max_workers=num_cpu, mp_context=None) as ex:
            futs = {ex.submit(_per_sample_worker, s): s for s in SAMPLE_NAMES}
            for fut in as_completed(futs):
                res = fut.result()
                results.append(res)
                if res["ok"]:
                    print(f"{res['sample']} done")
                else:
                    print(f"{res['sample']} failed: {res.get('error','')}")
            
        # Sample-specific preprocessing
        for sample_name in SAMPLE_NAMES:
            sample_input_dir = (
                RAW_SINGLE_CELL_DATA / sample_name
                if RAW_SINGLE_CELL_DATA
                else RAW_DATA / DATASET_NAME / sample_name
            )
            
            # Input Files (raw or processed scRNA-seq and scATAC-seq data from processed dir)
            processed_rna_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "scRNA_seq_processed.parquet"
            processed_atac_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "scATAC_seq_processed.parquet"
            
            # Output Files
            peak_bed_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "peaks.bed"
            peak_to_gene_dist_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "peak_to_gene_dist.parquet"
            sliding_window_score_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "sliding_window.parquet"
            tf_tg_reg_pot_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "tf_tg_regulatory_potential.parquet"
            
            # Sample-specific cache files
            tf_tensor_path: Path =        SAMPLE_DATA_CACHE_DIR / "tf_tensor_all.pt"
            sample_tf_name_file: Path =   SAMPLE_DATA_CACHE_DIR / "tf_names.json"
            tf_id_file: Path =            SAMPLE_DATA_CACHE_DIR / "tf_ids.pt"
            
            # TF-TG combination files
            tf_tg_combos_dir = SAMPLE_PROCESSED_DATA_DIR / "tf_tg_combos"
            total_genes_file = tf_tg_combos_dir / "total_genes.csv"
            tf_list_file = tf_tg_combos_dir / "tf_list.csv"
            tg_list_file = tf_tg_combos_dir / "tg_list.csv"
            tf_tg_combos_file = tf_tg_combos_dir / "tf_tg_combos.csv"
            
            # Check if all required output files exist for this sample
            required_sample_files = [
                processed_rna_file,
                processed_atac_file,
                peak_bed_file,
                peak_to_gene_dist_file,
                sliding_window_score_file,
                tf_tg_reg_pot_file,
                total_genes_file,
                tf_list_file,
                tg_list_file,
                tf_tg_combos_file,
            ]
            
            if not FORCE_RECALCULATE and all(os.path.isfile(f) for f in required_sample_files):
                logging.info(f"\n[{sample_name}] All required output files exist. Skipping sample-level preprocessing.")
                continue
            
            logging.info(f"\nOutput Directory: {SAMPLE_PROCESSED_DATA_DIR / sample_name}")
            os.makedirs(SAMPLE_PROCESSED_DATA_DIR / sample_name, exist_ok=True)
            os.makedirs(SAMPLE_DATA_CACHE_DIR, exist_ok=True)
            
            sample_raw_10x_rna_data_dir = RAW_10X_RNA_DATA_DIR / sample_name if RAW_10X_RNA_DATA_DIR else None
            
            pseudobulk_rna_df, pseudobulk_atac_df = process_or_load_rna_atac_data(
                sample_input_dir,
                force_recalculate=FORCE_RECALCULATE,
                raw_10x_rna_data_dir=sample_raw_10x_rna_data_dir,
                raw_atac_peak_file=RAW_ATAC_PEAK_MATRIX_FILE,
                sample_name=sample_name,
                sample_processed_dir=SAMPLE_PROCESSED_DATA_DIR / sample_name,
                neighbors_k=NEIGHBORS_K,
                pca_components=PCA_COMPONENTS,
                hops=HOPS,
                self_weight=SELF_WEIGHT,
            )

            # Canonicalize gene names in processed RNA data
            pseudobulk_rna_df.index = pd.Index(
                    gc.canonicalize_series(pd.Series(pseudobulk_rna_df.index, dtype=object)).array
                )
            
            # ----- GET TFs, TGs, and TF-TG combinations -----
            genes = pseudobulk_rna_df.index.to_list()
            peaks = pseudobulk_atac_df.index.to_list()
            
            logging.info("  - Processed RNA and ATAC files loaded")
            logging.info(f"    - Number of genes: {pseudobulk_rna_df.shape[0]}: {genes[:3]}")
            logging.info(f"    - Number of peaks: {pseudobulk_atac_df.shape[0]}: {peaks[:3]}")

            # Create TF-TG combination files
            tfs, tgs = create_tf_tg_combination_files(genes, TF_FILE, SAMPLE_PROCESSED_DATA_DIR, sample_name)
            
            tfs = sorted(set(gc.canonicalize_series(pd.Series(tfs)).tolist()))
            tgs = sorted(set(gc.canonicalize_series(pd.Series(tgs)).tolist()))
            
            # Format the peaks to BED format (chrom, start, end, peak_id)
            peak_locs_df = format_peaks(pd.Series(pseudobulk_atac_df.index)).rename(columns={"chromosome": "chrom"})
            
            if not os.path.isfile(peak_bed_file):
                # Write the peak BED file
                peak_bed_file.parent.mkdir(parents=True, exist_ok=True)
                pybedtools.BedTool.from_dataframe(
                    peak_locs_df[["chrom", "start", "end", "peak_id"]]
                ).saveas(peak_bed_file)
            
            # ----- CALCULATE PEAK TO TG DISTANCE -----
            # Calculate the distance from each peak to each gene TSS
            if not os.path.isfile(peak_to_gene_dist_file) or FORCE_RECALCULATE:
                # Download the gene TSS file from Ensembl if missing

                logging.info("  - Calculating peak to TG distance score")
                peak_to_gene_dist_df = calculate_peak_to_tg_distance_score(
                    peak_bed_file=peak_bed_file,
                    tss_bed_file=GENE_TSS_FILE,
                    peak_gene_dist_file=peak_to_gene_dist_file,
                    mesc_atac_peak_loc_df=peak_locs_df,
                    gene_tss_df=gene_tss_df,
                    max_peak_distance = MAX_PEAK_DISTANCE,
                    filter_to_nearest_gene = FILTER_TO_NEAREST_GENE,
                    distance_factor_scale = DISTANCE_SCALE_FACTOR,
                    force_recalculate=FORCE_RECALCULATE
                )
            
                logging.debug("\nPeak to gene distance")
                logging.debug("  - Number of peaks to gene distances: " + str(peak_to_gene_dist_df.shape[0]))
                logging.debug("  - Example peak to gene distances: \n" + str(peak_to_gene_dist_df.head()))
            
            # # ----- SLIDING WINDOW TF-PEAK SCORE -----
            # if not os.path.isfile(sliding_window_score_file):

            #     peaks_df = pybedtools.BedTool(peak_bed_file)

            #     logging.info("  - Running sliding window scan")
            #     run_sliding_window_scan(
            #         tf_name_list=tfs,
            #         tf_info_file=str(TF_FILE),
            #         motif_dir=str(MOTIF_DIR),
            #         genome_fasta=str(genome_fasta_file),
            #         peak_bed_file=str(peak_bed_file),
            #         output_file=sliding_window_score_file,
            #         num_cpu=num_cpu,
            #         inner_executor="thread",
            #         inner_workers=4
            #     )

            # ----- CALCULATE TF-TG REGULATORY POTENTIAL -----
            # if not os.path.isfile(tf_tg_reg_pot_file):
            #     tf_tg_reg_pot = calculate_tf_tg_regulatory_potential(
            #         sliding_window_score_file, tf_tg_reg_pot_file, peak_to_gene_dist_file, num_cpu)


            # # ----- MERGE TF-TG ATTRIBUTES WITH COMBINATIONS -----
            # if not os.path.isfile(tf_tg_combo_attr_file):
            #     logging.info("  - Loading TF-TG regulatory potential scores")
            #     tf_tg_reg_pot = pd.read_parquet(tf_tg_reg_pot_file, engine="pyarrow")
            #     logging.debug("  - Example TF-TG regulatory potential: " + str(tf_tg_reg_pot.head()))
                
            #     tf_df = processed_rna_df[processed_rna_df.index.isin(tfs)]
            #     logging.debug("\nTFs in RNA data")
        #     logging.debug(tf_df.head())
            #     logging.info(f"    - TFs in RNA data: {tf_df.shape[0]}")
                
            #     tg_df = processed_rna_df[processed_rna_df.index.isin(tgs)]
            #     logging.debug("\nTGs in RNA data")
            #     logging.debug(tg_df.head())
            #     logging.info(f"    - TGs in RNA data: {tg_df.shape[0]}")
            #     sliding_window_df = pd.read_parquet(sliding_window_score_file, engine="pyarrow")
            #     logging.debug("  - Example sliding window scores: \n" + str(sliding_window_df.head()))
                
            #     mean_norm_tf_expr, mean_norm_tg_expr = compute_minmax_expr_mean(tf_df, tg_df)
                
            #     logging.info("  - Merging TF-TG attributes with all combinations")
            #     tf_tg_df = merge_tf_tg_attributes_with_combinations(
            #         tf_tg_df, tf_tg_reg_pot, mean_norm_tf_expr, mean_norm_tg_expr, tf_tg_combo_attr_file, set(tfs))            
        
    # ----- CHROMOSOME-SPECIFIC PREPROCESSING -----
    if PROCESS_CHROMOSOME_SPECIFIC_DATA:        
        logging.info("\n\n===== PROCESSING PER-CHROMOSOME DATA =====")
        chrom_list = CHROM_IDS
        total_tf_list_file = SAMPLE_PROCESSED_DATA_DIR / "tf_tg_combos" / "tf_list.csv"
        tf_names = _read_list(total_tf_list_file, "TF")
        
        # Aggregate sample-level data for sliding window scores and peak to gene distance
        # sample_level_sliding_window_dfs = []
        sample_level_peak_to_gene_dist_dfs = []
        for sample_name in SAMPLE_NAMES:
            # sliding_window_score_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "sliding_window.parquet"
            peak_to_gene_dist_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "peak_to_gene_dist.parquet"
            
            # sliding_window_df = pd.read_parquet(sliding_window_score_file, engine="pyarrow")
            # sample_level_sliding_window_dfs.append(sliding_window_df)
            
            peak_to_gene_dist_df = pd.read_parquet(peak_to_gene_dist_file, engine="pyarrow")
            sample_level_peak_to_gene_dist_dfs.append(peak_to_gene_dist_df)

        # total_sliding_window_score_df = pd.concat(sample_level_sliding_window_dfs)
        total_peak_gene_dist_df = pd.concat(sample_level_peak_to_gene_dist_dfs)
        
        logging.info(f"Aggregating pseudobulk datasets")
        dataset_processed_data_dir = PROCESSED_DATA / DATASET_NAME
        total_TG_pseudobulk_global, pseudobulk_chrom_dict = \
            aggregate_pseudobulk_datasets(SAMPLE_NAMES, dataset_processed_data_dir, chrom_list, gc, force_recalculate=FORCE_RECALCULATE)
            
        global_tf_tensor_path   = SAMPLE_DATA_CACHE_DIR / "tf_tensor_all.pt"
        global_tf_ids_path      = SAMPLE_DATA_CACHE_DIR / "tf_ids.pt"
        global_tf_names_path    = SAMPLE_DATA_CACHE_DIR / "tf_names.json"
        global_metacell_path    = SAMPLE_DATA_CACHE_DIR / "metacell_names.json"

        # genome-wide TF expression for all metacells (columns)
        genome_wide_tf_expression = (
            total_TG_pseudobulk_global
            .reindex(tf_names)           # ensure row order matches your TF list
            .fillna(0)
            .values.astype("float32")
        )
        tf_tensor_all = torch.from_numpy(genome_wide_tf_expression)  # [T, C]

        # ensure common TF vocab exists, else initialize from tf_names
        if not os.path.exists(common_tf_vocab_file):
            with open(common_tf_vocab_file, "w") as f:
                json.dump({n: i for i, n in enumerate(tf_names)}, f)

        with open(common_tf_vocab_file) as f:
            tf_vocab = json.load(f)

        # align TF tensor to vocab order (and get kept names/ids)
        tf_tensor_all_aligned, tf_names_kept, tf_ids = align_to_vocab(
            tf_names, tf_vocab, tf_tensor_all, label="TF"
        )

        # save once, globally
        torch.save(tf_tensor_all_aligned, global_tf_tensor_path)
        torch.save(torch.tensor(tf_ids, dtype=torch.long), global_tf_ids_path)
        atomic_json_dump(tf_names_kept, global_tf_names_path)
        atomic_json_dump(total_TG_pseudobulk_global.columns.tolist(), global_metacell_path)
        logging.info(f"Saved GLOBAL TF tensor to {global_tf_tensor_path} "
                    f"with {len(tf_names_kept)} TFs and {tf_tensor_all_aligned.shape[1]} metacells.")
        
        logging.info(f"  - Number of chromosomes: {len(chrom_list)}: {chrom_list}")
        logging.info(f"  - Processing chromosomes for dataset: {DATASET_NAME}")
        for chrom_id in chrom_list:
            logging.info(f"  - Processing {chrom_id}")
            make_chrom_gene_tss_df(GENE_TSS_FILE, chrom_id, GENOME_DIR)
            
            SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR = SAMPLE_DATA_CACHE_DIR / chrom_id
        
            single_cell_dir = SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / "single_cell"
            
            # Chromosome-specific cache files
            atac_tensor_path: Path =            SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"atac_window_tensor_all_{chrom_id}.pt"
            tg_tensor_path: Path =              SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_tensor_all_{chrom_id}.pt"
            sample_tg_name_file: Path =         SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_names_{chrom_id}.json"
            genome_window_file: Path =          SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"{chrom_id}_windows_{WINDOW_SIZE // 1000}kb.bed"
            sample_window_map_file: Path =      SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"window_map_{chrom_id}.json"
            peak_to_tss_dist_path: Path =       SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"genes_near_peaks_{chrom_id}.parquet"
            dist_bias_file: Path =              SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"dist_bias_{chrom_id}.pt"
            tg_id_file: Path =                  SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tg_ids_{chrom_id}.pt"
            manifest_file: Path =               SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"manifest_{chrom_id}.json"
            # motif_mask_file: Path =             SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"motif_mask_{chrom_id}.pt"
            chrom_sliding_window_file: Path =   SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"sliding_window_{chrom_id}.parquet"
            chrom_peak_bed_file: Path =         SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"peak_tmp_{chrom_id}.bed"
            tss_bed_file: Path =                SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR / f"tss_tmp_{chrom_id}.bed"

            # Check if all required output files exist
            required_files = [
                global_tf_tensor_path,
                global_tf_ids_path,
                global_tf_names_path,
                global_metacell_path,
                atac_tensor_path,
                tg_tensor_path,
                sample_tg_name_file,
                sample_window_map_file,
                peak_to_tss_dist_path,
                dist_bias_file,
                tg_id_file,
                manifest_file,
                # motif_mask_file,
            ]
            
            if not FORCE_RECALCULATE and all(os.path.isfile(f) for f in required_files):
                logging.info(f"  - All required output files exist for {chrom_id}. Skipping preprocessing.")
                continue

            os.makedirs(COMMON_DATA, exist_ok=True)
            os.makedirs(SAMPLE_DATA_CACHE_DIR, exist_ok=True)
            os.makedirs(SAMPLE_CHROM_SPECIFIC_DATA_CACHE_DIR, exist_ok=True)
            os.makedirs(single_cell_dir, exist_ok=True)
                        
            # Create or load the gene TSS information for the chromosome
            if not os.path.isfile(os.path.join(GENOME_DIR, f"{chrom_id}_gene_tss.bed")):
                gene_tss_df = make_chrom_gene_tss_df(
                    gene_tss_file=GENE_TSS_FILE,
                    chrom_id=chrom_id,
                    genome_dir=GENOME_DIR
                )
            else:
                logging.debug(f"  - Loading existing gene TSS file for {chrom_id}")
                gene_tss_df = pd.read_csv(os.path.join(GENOME_DIR, f"{chrom_id}_gene_tss.bed"), sep="\t", header=None, usecols=[0, 1, 2, 3])
                gene_tss_df = gene_tss_df.rename(columns={0: "chrom", 1: "start", 2: "end", 3: "name"})
                
                
            total_TG_pseudobulk_chr = pseudobulk_chrom_dict[chrom_id]["total_TG_pseudobulk_chr"]
            total_RE_pseudobulk_chr = pseudobulk_chrom_dict[chrom_id]["total_RE_pseudobulk_chr"]
            total_peaks_df = pseudobulk_chrom_dict[chrom_id]["total_peaks_df"]
                
            logging.debug(f"      - {chrom_id}: TG pseudobulk shape={total_TG_pseudobulk_chr.shape}")
            logging.debug(f"      - TG Examples:{total_TG_pseudobulk_chr.index[:5].tolist()}")
            
            vals = total_TG_pseudobulk_chr.values.astype("float32")
            if vals.shape[0] == 0:
                logging.warning(f"{chrom_id}: no TG rows after aggregation; skipping this chromosome.")
                continue
        
            tg_names = total_TG_pseudobulk_chr.index.tolist()
            
            # Genome-wide TF expression for all samples
            genome_wide_tf_expression = total_TG_pseudobulk_global.reindex(tf_names).fillna(0).values.astype("float32")
            metacell_names = total_TG_pseudobulk_global.columns.tolist()
            
            # Downcast the TG expression to float32
            TG_expression = total_TG_pseudobulk_chr.values.astype("float32")
            
            chrom_peak_ids = set(total_peaks_df["peak_id"].astype(str))
            
            # Create genome windows
            logging.debug(f"  - Creating genomic windows for {chrom_id}")
            genome_windows = create_or_load_genomic_windows(
                window_size=WINDOW_SIZE,
                chrom_id=chrom_id,
                chrom_sizes_file=CHROM_SIZES_FILE,
                genome_window_file=genome_window_file,
                force_recalculate=FORCE_RECALCULATE,
                promoter_only=(PROMOTER_BP is not None)
            )
            
                # --- Calculate Peak-to-TG Distance Scores ---
            genes_near_peaks = total_peak_gene_dist_df[total_peak_gene_dist_df["peak_id"].astype(str).isin(chrom_peak_ids)].copy()
                
            genes_near_peaks["target_id"] = gc.canonicalize_series(genes_near_peaks["target_id"])
            genes_near_peaks.to_parquet(peak_to_tss_dist_path, engine="pyarrow", compression="snappy")
            logging.debug(f"  - Saved peak-to-TG distance scores to {peak_to_tss_dist_path}")

            
            # # ----- SLIDING WINDOW TF-PEAK SCORE -----
            # if (not os.path.isfile(chrom_sliding_window_file)) or (FORCE_RECALCULATE == True):
                
            #     sliding_window_df = total_sliding_window_score_df[
            #         total_sliding_window_score_df["peak_id"].astype(str).isin(chrom_peak_ids)
            #     ][["TF","peak_id","sliding_window_score"]].copy()

            #     # normalize types/names
            #     sliding_window_df["TF"] = sliding_window_df["TF"].astype(str)
            #     sliding_window_df["peak_id"] = sliding_window_df["peak_id"].astype(str)
            #     sliding_window_df["sliding_window_score"] = pd.to_numeric(sliding_window_df["sliding_window_score"], errors="coerce")

            #     # collapse duplicates across samples
            #     sliding_window_df = (
            #         sliding_window_df
            #         .groupby(["TF","peak_id"], as_index=False, sort=False)
            #         .agg(sliding_window_score=("sliding_window_score","mean"))
            #     )
                
            #     sliding_window_df.to_parquet(chrom_sliding_window_file, engine="pyarrow", compression="snappy")

            #     logging.debug(f"  - Wrote sliding window scores to {chrom_sliding_window_file}")
            # else:
            #     logging.debug("Loading existing sliding window scores")
            #     sliding_window_df = pd.read_parquet(chrom_sliding_window_file, engine="pyarrow")
            
            total_peaks_df["peak_id"] = total_peaks_df["peak_id"].astype(str)
            genes_near_peaks["peak_id"] = genes_near_peaks["peak_id"].astype(str)
            
            peaks_as_windows = (PROMOTER_BP is not None)

            if peaks_as_windows:
                genome_windows = total_peaks_df[["chrom", "start", "end"]].copy().reset_index(drop=True)
                
            num_windows = int(genome_windows.shape[0])

            window_map = make_peak_to_window_map(
                peaks_bed=total_peaks_df,
                windows_bed=genome_windows,
                peaks_as_windows=peaks_as_windows,
            )

            total_RE_pseudobulk_chr.index = (
                total_RE_pseudobulk_chr.index.astype(str).str.strip()
            )

            _, tg_tensor_all, atac_window_tensor_all = precompute_input_tensors(
                output_dir=str(SAMPLE_DATA_CACHE_DIR),
                genome_wide_tf_expression=genome_wide_tf_expression,
                genome_wide_tg_expression=TG_expression,
                total_RE_pseudobulk_chr=total_RE_pseudobulk_chr,
                window_map=window_map,
                windows=genome_windows,   # now aligned with map
            )
            
            # Skip this chromosome if no peaks matched
            if tg_tensor_all is None:
                logging.warning(f"{chrom_id}: No peaks matched between window_map and total_RE_pseudobulk_chr; skipping this chromosome.")
                continue
            
            # ----- Load common TF and TG vocab -----
            # Create a common TG vocabulary for the chromosome using the gene TSS
            logging.debug(f"  - Matching TFs and TGs to global gene vocabulary")
            
            tg_names = [standardize_name(n) for n in total_TG_pseudobulk_chr.index.tolist()]
            
            if not os.path.exists(common_tf_vocab_file):
                vocab = {n: i for i, n in enumerate(tf_names)}
                with open(common_tf_vocab_file, "w") as f:
                    json.dump(vocab, f)
                logging.info(f"Initialized TF vocab with {len(vocab)} entries")
            else:
                with open(common_tf_vocab_file) as f:
                    vocab = json.load(f)

            # Load global vocab
            with open(common_tf_vocab_file) as f: tf_vocab = json.load(f)
            with open(common_tg_vocab_file) as f: tg_vocab = json.load(f)
            
            # Match TFs and TGs to global vocab
            tg_tensor_all, tg_names_kept, tg_ids = align_to_vocab(tg_names, tg_vocab, tg_tensor_all, label="TG")
            
            torch.save(torch.tensor(tg_ids, dtype=torch.long), tg_id_file)

            logging.debug(f"\t- Matched {len(tg_names_kept)} TGs to global vocab")
            
            # # Build motif mask using merged info
            # motif_mask = build_motif_mask(
            #     tf_names=tf_names_kept,
            #     tg_names=tg_names_kept,
            #     sliding_window_df=sliding_window_df,
            #     genes_near_peaks=genes_near_peaks
            # )

            if not tf_ids: raise ValueError("No TFs matched the common vocab.")
            if not tg_ids: raise ValueError("No TGs matched the common vocab.")
            
            # Build distance bias [num_windows x num_tg_kept] aligned to kept TGs
            logging.debug(f"  - Building distance bias")
            dist_bias, new_window_map, kept_window_indices = build_distance_bias(
                genes_near_peaks=genes_near_peaks,
                window_map=window_map,
                tg_names_kept=tg_names_kept,
                num_windows=num_windows,
                dtype=torch.float32,
                mode=DIST_BIAS_MODE,
                prune_empty_windows=True
            )
            
            if kept_window_indices is not None:
                keep_t = torch.tensor(kept_window_indices, dtype=torch.long)

                # atac_window_tensor_all: [W, C] -> [W', C]
                atac_window_tensor_all = atac_window_tensor_all.index_select(0, keep_t)

                # also prune genome_windows to match, if you use it later / in manifest
                genome_windows = genome_windows.iloc[kept_window_indices].reset_index(drop=True)

                num_windows = len(kept_window_indices)
            
            create_single_cell_tensors(
                gene_tss_df=gene_tss_df, 
                sample_names=FINE_TUNING_DATASETS, 
                dataset_processed_data_dir=PROCESSED_DATA / DATASET_NAME, 
                tg_vocab=tg_vocab, 
                tf_vocab=tf_vocab, 
                chrom_id=chrom_id,
                single_cell_dir=single_cell_dir
            )
            
            # ----- Writing Output Files -----
            logging.debug(f"Writing output files")
            # Save the Window, TF, and TG expression tensors
            torch.save(atac_window_tensor_all, atac_tensor_path)
            torch.save(tg_tensor_all, tg_tensor_path)

            # Save the peak -> window map for the sample
            atomic_json_dump(new_window_map, sample_window_map_file)

            # Write TF and TG names and global vocab indices present in the sample
            atomic_json_dump(tg_names_kept, sample_tg_name_file)

            # Write the distance bias and metacell names for the sample
            torch.save(dist_bias, dist_bias_file)
            logging.debug(f"  - Saved distance bias tensor with shape {tuple(dist_bias.shape)} to {dist_bias_file}")
            
            # torch.save(torch.from_numpy(motif_mask), motif_mask_file)

            # Manifest of general sample info and file paths
            manifest = {
                "dataset_name": DATASET_NAME,
                "chrom": chrom_id,
                "num_windows": int(num_windows),
                "num_tfs": int(len(tf_names_kept)),
                "num_tgs": int(len(tg_names_kept)),
                "Distance tau": DISTANCE_SCALE_FACTOR,
                "Max peak-TG distance": MAX_PEAK_DISTANCE,
                "paths": {
                    "tf_tensor_all": str(global_tf_tensor_path),
                    "tg_tensor_all": str(tg_tensor_path),
                    "atac_window_tensor_all": str(atac_tensor_path),
                    "dist_bias": str(dist_bias_file),
                    "tf_ids": str(global_tf_ids_path),
                    "tg_ids": str(tg_id_file),
                    "tf_names": str(global_tf_names_path),
                    "tg_names": str(sample_tg_name_file),
                    "common_tf_vocab": str(common_tf_vocab_file),
                    "common_tg_vocab": str(common_tg_vocab_file),
                    "window_map": str(sample_window_map_file),
                    "genes_near_peaks": str(peak_to_tss_dist_path),
                    # "motif_mask": str(motif_mask_file),
                }
            }
            with open(manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)

        logging.info("Preprocessing complete. Wrote per-sample/per-chrom data for MultiomicTransformerDataset.")
            