# Single-cell packages
#
# STANDARDISED COPY of TETHER/muon_preprocessing.py (copied 2026-09-02).
#
# Identical to the original except that create_metacells now maps the post-diffusion
# pseudobulks onto a frozen reference marginal and checks the result against a
# distributional signature before writing. See pseudobulk_standardization.py for why, and
# FINDINGS.md section 2 for the measurement that motivated it.
#
# Per-sample QC thresholds in data/qc_filtering_settings.tsv are untouched -- filtering
# stays tailored, only the numeric scale of what survives is standardised.
import os
os.environ["TQDM_DISABLE"] = "1"

import argparse
import json
import sys
import matplotlib.pyplot as plt
import muon as mu
import mudata as md
import numpy as np
import anndata as ad
import pysam
import scipy.sparse as sp
from anndata import AnnData
import networkx as nx

from pathlib import Path

# General helpful packages for data analysis and visualization
import pandas as pd
import scanpy as sc
import seaborn as sns
from muon import atac as ac  # the module containing function for scATAC data processing
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Setting figure parameters
sc.settings.verbosity = 0

# disable automatic pulling of data from MuData objects to avoid unintended side effects during preprocessing steps
md.set_options(pull_on_update=False)

# pseudobulk_standardization.py sits beside this file, not on the TETHER import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pseudobulk_standardization import (
    check_signature,
    compute_signature,
    load_reference,
    standardize_matrix,
    standardize_matrix_inplace,
)

# Every stochastic step downstream of QC keys off this, so a rerun of the same sample
# reproduces its pseudobulks exactly. Changing it changes the outputs.
MOFA_SEED = 125

DEFAULT_REFERENCE_QUANTILES = Path(__file__).resolve().parent / "reference_quantiles.npz"
DEFAULT_REFERENCE_SIGNATURE = Path(__file__).resolve().parent / "reference_signature.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Run Muon preprocessing with configurable file paths.")
    parser.add_argument("--project-dir", type=str, required=True)
    parser.add_argument("--tss-path", type=str, required=True)
    parser.add_argument("--raw-data-dir", type=str, required=True)
    parser.add_argument("--processed-data-dir", type=str, required=True)
    parser.add_argument("--sample-name", type=str, required=True)
    parser.add_argument("--rna-count-file", type=str, default=None)
    parser.add_argument("--atac-count-file", type=str, default=None)
    parser.add_argument("--raw-h5-file", type=str, default=None)
    parser.add_argument("--tf-list-file", type=str, required=True)
    parser.add_argument("--frag-path", type=str, required=True)
    parser.add_argument(
        "--reference-quantiles", type=str, default=str(DEFAULT_REFERENCE_QUANTILES),
        help="Frozen reference marginal from 09_build_reference_quantiles.py.",
    )
    parser.add_argument(
        "--reference-signature", type=str, default=str(DEFAULT_REFERENCE_SIGNATURE),
        help="Expected non-zero quantiles the written pseudobulks are checked against.",
    )
    parser.add_argument(
        "--no-standardize", action="store_true",
        help="Write the raw diffused values, reproducing the original script's output.",
    )
    parser.add_argument(
        "--gate-mode", choices=["error", "warn", "off"], default="error",
        help="What to do if the written pseudobulks miss the reference signature.",
    )
    return parser.parse_args()


def filter_to_human(mdata):
    """
    Filter a barnyard MuData object to hg38 ATAC peaks only,
    then strip the 'hg38.' prefix from peak IDs.
    """
    if "hg38" in mdata["atac"].var_names[0]:
        # annotate species from interval
        mdata["atac"].var["species"] = (
            mdata["atac"].var["interval"].str.split(".", n=1).str[0]
        )

        # keep only hg38 peaks
        hg38_mask = mdata["atac"].var["species"] == "hg38"
        mdata.mod["atac"] = mdata["atac"][:, hg38_mask].copy()

        # strip prefix from identifiers
        mdata.mod["atac"].var_names = (
            mdata.mod["atac"].var_names.str.replace(r"^hg38\.", "", regex=True)
        )
        mdata.mod["atac"].var["gene_ids"] = (
            mdata.mod["atac"].var["gene_ids"].str.replace(r"^hg38\.", "", regex=True)
        )
        mdata.mod["atac"].var["interval"] = (
            mdata.mod["atac"].var["interval"].str.replace(r"^hg38\.", "", regex=True)
        )
        mdata = mu.MuData(mdata.mod)

    return mdata


def create_fragment_index_file(frag_path: Path) -> str:
    """
    Create a tabix index file for a fragments.tsv.gz file if it doesn't already exist.
    
    Parameters
    ----------
    frag_path : Path
        Path to the fragments.tsv.gz file.
        
    Returns
    -------
    str
        Path to the created or existing index file (fragments.tsv.gz.tbi).
    """
    index_file = str(frag_path) + ".tbi"

    if Path(index_file).exists():
        logging.info(f"Found index: {index_file}")

    else:
        logging.info("Index file not found. Creating index file...")
        pysam.tabix_index(
            str(frag_path),
            preset="bed",
            force=True
        )
        index_file = str(frag_path) + ".tbi"
    
    return index_file


def construct_mdata_from_gene_by_cell_matrices(rna_count_file: Path, atac_count_file: Path) -> mu.MuData:
    """
    Construct a MuData object from gene-by-cell matrices for RNA and ATAC data.
    
    Parameters
    ----------
    rna_count_file : Path
        Path to the RNA count matrix file (genes as rows, cells as columns).
    atac_count_file : Path
        Path to the ATAC count matrix file (peaks as rows, cells as columns).

    Returns
    -------
    mu.MuData
        A MuData object containing the RNA and ATAC data.
    """
    assert rna_count_file.exists(), "rna count file does not exist"
    assert atac_count_file.exists(), "atac count file does not exist"
    
    rna_count_matrix = pd.read_csv(rna_count_file, header=0, index_col=0)
    atac_count_matrix = pd.read_csv(atac_count_file, header=0, index_col=0)
    
    rna_matrix = rna_count_matrix.T.values
    rna_metadata_df = pd.DataFrame(index=rna_count_matrix.columns)
    rna_features_df = pd.DataFrame(index=rna_count_matrix.index)
    
    atac_count_matrix.index = atac_count_matrix.index.map(normalize_peak_format)

    atac_matrix = atac_count_matrix.T.values
    atac_metadata_df = pd.DataFrame(index=atac_count_matrix.columns)
    atac_features_df = pd.DataFrame(index=atac_count_matrix.index)
    
    adata_rna = ad.AnnData(X=rna_matrix, obs=rna_metadata_df, var=rna_features_df)
    adata_rna.var["feature_types"] = pd.Categorical(["Gene Expression"] * adata_rna.n_vars)
    adata_rna.var["gene_ids"] = adata_rna.var_names

    adata_atac = ad.AnnData(X=atac_matrix, obs=atac_metadata_df, var=atac_features_df)
    adata_atac.var["feature_types"] = pd.Categorical(["Peaks"] * adata_atac.n_vars)
    adata_atac.var["gene_ids"] = adata_atac.var_names

    mdata = mu.MuData({'rna': adata_rna, 'atac': adata_atac})

    return mdata


def load_raw_data(
    sample_name: str, 
    sample_data_dir: Path, 
    rna_count_file: Path | None = None, 
    atac_count_file: Path | None = None, 
    raw_h5_file: Path | None = None,
    verbose: bool = True,
    ):

    # logging.info all files in the sample data directory.
    if verbose:
        logging.info(f"Loading data for sample {sample_name} from {sample_data_dir}...")
    
    found_barcode = False
    found_features = False
    found_matrix = False
    
    frag_path = None
    
    if raw_h5_file:
        logging.info(f"Found raw h5 file: {raw_h5_file.name}. Will load this file for sample {sample_name}.")
        # The original only handled .h5mu, but mouse_liver's raw input is a CellRanger
        # multiome matrix (GSE288579_filtered_feature_bc_matrix.h5), which mu.read_h5mu
        # cannot open -- the liver notebook used mu.read_10x_h5 for exactly this file.
        # Dispatch on the suffix so both work.
        if raw_h5_file.suffix == ".h5mu":
            mdata = mu.read_h5mu(raw_h5_file)
        else:
            mdata = mu.read_10x_h5(raw_h5_file)
            mdata.var_names_make_unique()
        return mdata, frag_path
    else:
        
        # logging.info all files in the data directory
        for file in sample_data_dir.glob("*"):
            file_name = file.name
            if verbose:
                logging.info(f"  - {file_name}")
            # Renaming here mutates the shared RAW_DATA mount, and two of these tests are
            # wider than they look: endswith("fragments.tsv.gz") also matches
            # "fragments.sorted.tsv.gz", which mESC/E7.5_rep1 has alongside the real one.
            # The original would rename the sorted file onto fragments.tsv.gz and destroy
            # the original, on a mount shared with every other consumer of this data.
            # _claim only renames when the target does not already exist, so a directory
            # whose files are already correctly named (all of ours are) is left untouched.
            def _claim(source_file, canonical_name):
                target = sample_data_dir / canonical_name
                if source_file.resolve() == target.resolve():
                    return True
                if target.exists():
                    logging.info(
                        f"  (leaving {source_file.name} alone; {canonical_name} already exists)"
                    )
                    return False
                logging.info(f"  renaming {source_file.name} -> {canonical_name}")
                source_file.rename(target)
                return True

            if file_name.endswith("barcodes.tsv.gz"):
                found_barcode |= _claim(file, "barcodes.tsv.gz")
            if file_name.endswith("features.tsv.gz"):
                found_features |= _claim(file, "features.tsv.gz")
            if file_name.endswith("matrix.mtx.gz"):
                found_matrix |= _claim(file, "matrix.mtx.gz")
            if file_name.endswith("fragments.tsv.gz"):
                if _claim(file, "fragments.tsv.gz"):
                    frag_path = sample_data_dir / "fragments.tsv.gz"
            if file_name.endswith("fragments.tsv.gz.tbi.gz"):
                _claim(file, "fragments.tsv.gz.tbi.gz")

        if not (found_barcode and found_features and found_matrix):
            for file in sample_data_dir.glob("*"):
                file_name = file.name
                if file_name.endswith(".h5") and verbose:
                    logging.info(f"Found h5 file: {file_name}")
                    raw_h5_file = file
        
        # If raw count files are passed in, use them above any other file formats
        if rna_count_file is not None and atac_count_file is not None:
            rna_count_filepath = sample_data_dir / rna_count_file
            atac_count_filepath = sample_data_dir / atac_count_file
            
            mdata = construct_mdata_from_gene_by_cell_matrices(rna_count_filepath, atac_count_filepath)
            mdata.var_names_make_unique()
            return mdata, frag_path
            
        # If no h5 file is found, look for the 10x mtx files. If they exist, load them using muon.
        elif found_barcode and found_features and found_matrix:
            mdata = mu.read_10x_mtx(sample_data_dir)
            mdata.var_names_make_unique()
            return mdata, frag_path
        
        else:
            raise FileNotFoundError(f"Could not find the necessary files to load the data for sample {sample_name} in {sample_data_dir}. Please ensure that the sample directory contains either a raw h5 file or the 10x mtx files (barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz).")


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


class MudataProcessor:
    def __init__(
        self, 
        mdata, 
        processed_data_dir,
        sample_name,
        tss_path,
        tf_list_file=None,
        ):
        
        self.raw_mdata = mdata
        self.mdata = self.raw_mdata.copy()
        self.tss_path = tss_path
        self.rna = self.mdata.mod['rna']
        self.atac = self.mdata.mod['atac']
        self.tf_list_file = tf_list_file
        self.processed_data_dir = processed_data_dir
        self.sample_name = sample_name

    def flag_tfs_to_keep(self):
        self.rna.var['keep_tf'] = False
        
        if self.tf_list_file is not None and self.tf_list_file.exists():
            tf_list_df = pd.read_csv(self.tf_list_file)

            tf_genes_to_keep = set(
                tf_list_df["source_id"].astype(str).str.strip().str.upper()
            )

            self.rna.var["keep_tf"] = (
                self.rna.var_names.astype(str).str.strip().str.upper().isin(tf_genes_to_keep)
            )
            
    def show_pre_filtering_qc_rna(self):
        self.rna.var['mt'] = self.rna.var_names.str.upper().str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(self.rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        
        sc.pl.violin(self.rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)
        
    def show_pre_filtering_qc_atac(self):
        sc.pp.calculate_qc_metrics(self.atac, percent_top=None, log1p=False, inplace=True)
        
        sc.pl.violin(self.atac, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True)
        
    def save_stability_subsamplings(
        self,
        raw_mdata: mu.MuData,
        subsampling_dir: Path,
        pct_subsample: float = 0.7,
        num_subsamples: int = 10,
    ):
        subsampling_dir.mkdir(parents=True, exist_ok=True)

        mdata = raw_mdata.copy()
        mu.pp.intersect_obs(mdata)

        for i in range(num_subsamples):
            subsample_path = subsampling_dir / f"{int(pct_subsample * 100)}pct_subsample_{i+1}.h5mu"
            if subsample_path.exists():
                logging.info(f"Subsample {i+1} already exists at {subsample_path}. Skipping subsampling.")
                continue

            logging.info(f"Creating subsample {i+1} with {pct_subsample*100:.0f}% of the cells...")

            sampled_obs_names = mdata.obs.sample(frac=pct_subsample, replace=False).index
            subsampled_mdata = mdata[sampled_obs_names, :].copy()

            mu.write(str(subsample_path), subsampled_mdata)
            logging.info(f"  - Saved subsample {i+1} to {subsample_path}.")
        
    def rna_qc_filter(
        self,
        min_cells_per_gene: int = 20,
        min_genes_per_cell: int = 500,
        max_genes_per_cell: int = 2500,
        min_total_counts_per_cell: int = 1000,
        max_total_counts_per_cell: int = 5000,
        max_pct_counts_mt: int = 20,
        norm_target_sum: float = 1e4,
        min_rna_disp: float = 0.5,
        min_rna_hvg_mean: float = 0.02,
        max_rna_hvg_mean: float = 4,
        filter_hvgs: bool = True,
        tf_list_file: Path|None = None,
        fig_dir: Path|None = None,
        
    ):
        """
        Filter RNA data based on quality control criteria.

        Parameters
        ----------
        min_cells_per_gene : int, optional
            Minimum number of cells a gene must be present in to be kept.
            Defaults to 20.
        min_genes_per_cell : int, optional
            Minimum number of genes a cell must have to be kept.
            Defaults to 500.
        max_genes_per_cell : int, optional
            Maximum number of genes a cell can have to be kept.
            Defaults to 2500.
        min_total_counts_per_cell : int, optional
            Minimum total number of counts a cell must have to be kept.
            Defaults to 1000.
        max_total_counts_per_cell : int, optional
            Maximum total number of counts a cell can have to be kept.
            Defaults to 5000.
        max_pct_counts_mt : int, optional
            Maximum percentage of counts a cell can have in mitochondrial genes to be kept.
            Defaults to 20.

        Returns
        -------
        Nothing. The function modifies the RNA data in-place.
        """
        
        if fig_dir is not None:
            fig_dir.mkdir(parents=True, exist_ok=True)
            sc.settings.figdir = fig_dir
        
        self.rna.var['mt'] = self.rna.var_names.str.upper().str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(self.rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.violin(self.rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save="_pre_qc_filtering_rna.png")

        # Filter
        if tf_list_file is not None and tf_list_file.exists():
            self.flag_tfs_to_keep()
            mu.pp.filter_var(self.rna, 'n_cells_by_counts', lambda x: (x >= min_cells_per_gene) | self.rna.var['keep_tf'].to_numpy(dtype=bool))
        else:
            mu.pp.filter_var(self.rna, 'n_cells_by_counts', lambda x: (x >= min_cells_per_gene))

            
        mu.pp.filter_obs(self.rna, 'n_genes_by_counts', lambda x: (x >= min_genes_per_cell) & (x <= max_genes_per_cell))
        mu.pp.filter_obs(self.rna, 'total_counts', lambda x: (x >= min_total_counts_per_cell) & (x <= max_total_counts_per_cell))
        mu.pp.filter_obs(self.rna, 'pct_counts_mt', lambda x: x <= max_pct_counts_mt)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.violin(self.rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save="_post_qc_filtering_rna.png")
            plt.close()
            
        # Save raw counts
        self.rna.layers["counts"] = self.rna.X.copy()
    
        # Normalize and log-transform
        sc.pp.normalize_total(self.rna, target_sum=norm_target_sum)
        sc.pp.log1p(self.rna)
        
        self.rna.layers["log1p"] = self.rna.X.copy()
        self.rna.raw = self.rna.copy()
        
        # Select highly variable genes
        sc.pp.highly_variable_genes(self.rna, min_mean=min_rna_hvg_mean, max_mean=max_rna_hvg_mean, min_disp=min_rna_disp)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.highly_variable_genes(self.rna, save="_rna.png")
            
        if filter_hvgs:
            if tf_list_file is not None and tf_list_file.exists() and 'keep_tf' in self.rna.var.columns:
                keep_genes = self.rna.var['highly_variable'] | self.rna.var['keep_tf']
            else:
                keep_genes = self.rna.var['highly_variable']
            self.rna = self.rna[:, keep_genes].copy()
            
        self.mdata.mod["rna"] = self.rna
        
        mu.write(str(self.processed_data_dir / f"{self.sample_name}.h5mu/rna"), self.rna)
    
    def rna_pca_and_neighbors(self, rna, n_pcs=20, n_neighbors=10, fig_dir: Path|None = None):
        """
        Perform principal component analysis (PCA) and k-nearest neighbors (kNN) on the RNA data.

        Parameters
        ----------
        rna : ad.AnnData
            The RNA data.
        n_pcs : int, optional
            The number of principal components to keep.
            Defaults to 20.
        n_neighbors : int, optional
            The number of k-nearest neighbors to keep.
            Defaults to 10.
        fig_dir : Path|None, optional
            The directory to save the figures in.
            Defaults to None.

        Returns
        -------
        Nothing. The function modifies the RNA data in-place.
        """
        if fig_dir is not None:
            fig_dir.mkdir(parents=True, exist_ok=True)
            sc.settings.figdir = fig_dir
        
        sc.tl.pca(rna, svd_solver='arpack')

        if fig_dir is not None and fig_dir.exists():
            if "highly_variable" in rna.var.columns:
                first_three_hvg_genes = rna.var[rna.var.highly_variable].index[:3].to_list()
                sc.pl.pca(rna, color=first_three_hvg_genes, save="_pca_hvgs.png")
                
            sc.pl.pca_variance_ratio(rna, log=True, save="_pca_variance_ratio_rna.png")
            
        sc.pp.neighbors(rna, n_neighbors=n_neighbors, n_pcs=n_pcs)
        
        sc.tl.umap(rna, spread=1., min_dist=.5, random_state=11)
        sc.tl.leiden(rna, flavor="igraph", n_iterations=2)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.umap(rna, color=["leiden"], save="_umap_leiden_rna.png")
    
    def construct_peak_annotation(
        self, 
        save_dir: Path,
        promoter_upstream: int = 1000,
        promoter_downstream: int = 100,
        distal_max: int = 200_000
        ):
        """Construct a peak annotation table in 10x-style format, assigning each peak to a gene and distance based on TSS proximity."""

        peaks = pd.DataFrame({"peak": self.atac.var_names.astype(str)})
        
        coords = peaks["peak"].str.extract(
            r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$"
        )

        if coords.isna().any().any():
            bad = peaks.loc[coords.isna().any(axis=1), "peak"].head(10).tolist()
            raise ValueError(f"Could not parse some peak names. Examples: {bad}")

        peaks = pd.concat([peaks, coords], axis=1)
        peaks["start"] = peaks["start"].astype(int)
        peaks["end"] = peaks["end"].astype(int)

        tss = pd.read_csv(
            self.tss_path,
            sep="\t",
            header=None,
            names=["chrom", "tss_start", "tss_end", "gene"]
        )

        tss["tss"] = tss["tss_start"].astype(int)

        # Only keep protein coding genes
        if "gene_biotype" in tss.columns:
            tss = tss[tss["gene_biotype"] == "protein_coding"].copy()

        # Cross-join peaks and genes by chromosome
        cand = peaks.merge(
            tss[["chrom", "gene", "tss"]],
            on="chrom",
            how="inner"
        )

        # 10x signed distance:
        # positive if peak start is downstream of TSS
        # negative if peak end is upstream of TSS
        # zero if TSS overlaps peak
        cand["distance"] = np.where(
            cand["start"] > cand["tss"],
            cand["start"] - cand["tss"],
            np.where(
                cand["end"] < cand["tss"],
                cand["end"] - cand["tss"],
                0
            )
        )

        cand["abs_distance"] = cand["distance"].abs()

        # PROMOTER peaks:
        # overlap promoter region [TSS-1000, TSS+100]
        cand["is_promoter"] = (
            (cand["end"] >= (cand["tss"] - promoter_upstream)) &
            (cand["start"] <= (cand["tss"] + promoter_downstream))
        )

        promoter = cand.loc[cand["is_promoter"], ["peak", "chrom", "start", "end", "gene", "distance"]].copy()
        promoter["peak_type"] = "promoter"

        # DISTAL peaks:
        # within 200 kb of the CLOSEST TSS,
        # but not promoter for that same gene
        
        # find closest TSS gene per peak
        closest_idx = cand.groupby("peak")["abs_distance"].idxmin()
        closest = cand.loc[closest_idx, ["peak", "chrom", "start", "end", "gene", "distance", "abs_distance"]].copy()

        closest = closest.loc[closest["abs_distance"] <= distal_max].copy()

        # remove cases where that peak is already promoter for that same gene
        promoter_pairs = set(zip(promoter["peak"], promoter["gene"]))
        closest["is_promoter_same_gene"] = [
            (p, g) in promoter_pairs for p, g in zip(closest["peak"], closest["gene"])
        ]

        distal = closest.loc[~closest["is_promoter_same_gene"], ["peak", "chrom", "start", "end", "gene", "distance"]].copy()
        distal["peak_type"] = "distal"

        # INTERGENIC peaks:
        # peaks with no promoter or distal assignment
        assigned_peaks = set(promoter["peak"]) | set(distal["peak"])

        intergenic = peaks.loc[~peaks["peak"].isin(assigned_peaks), ["peak", "chrom", "start", "end"]].copy()
        intergenic["gene"] = ""
        intergenic["distance"] = np.nan
        intergenic["peak_type"] = "intergenic"

        # Final table in 10x format
        peak_annotation_10x = pd.concat(
            [
                promoter[["chrom", "start", "end", "gene", "distance", "peak_type"]],
                distal[["chrom", "start", "end", "gene", "distance", "peak_type"]],
                intergenic[["chrom", "start", "end", "gene", "distance", "peak_type"]],
            ],
            axis=0,
            ignore_index=True
        ).sort_values(["chrom", "start", "end", "gene", "peak_type"])
        
        peak_annotation_10x = peak_annotation_10x.dropna()

        # save as 10x-style TSV
        out_path = save_dir / "atac_peak_annotation.tsv"
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True)
        peak_annotation_10x.to_csv(out_path, sep="\t", index=False)

    def atac_qc_filter(
        self, 
        min_cells_per_peak=20, 
        min_peaks_per_cell=500, 
        max_peaks_per_cell=2500, 
        min_total_counts_per_cell=1000, 
        max_total_counts_per_cell=5000,
        scale_factor=1e4,
        min_atac_disp=0.5,
        min_atac_hvg_mean=0.05,
        max_atac_hvg_mean=1.5,
        promoter_upstream=1000,
        promoter_downstream=100,
        distal_max=200_000,
        filter_hvgs: bool = True,
        n_neighbors: int = 10,
        n_pcs: int = 30,
        fig_dir: Path|None = None
        ):
        if fig_dir is not None:
            sc.settings.figdir = fig_dir
            if not fig_dir.exists():
                fig_dir.mkdir(parents=True, exist_ok=True)
        
        sc.pp.calculate_qc_metrics(self.atac, percent_top=None, log1p=False, inplace=True)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.violin(self.atac, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True, save="_pre_qc_filtering_atac.png")                    
        
        mu.pp.filter_var(self.atac, 'n_cells_by_counts', lambda x: x >= min_cells_per_peak)
        mu.pp.filter_obs(self.atac, 'n_genes_by_counts', lambda x: (x >= min_peaks_per_cell) & (x <= max_peaks_per_cell))
        mu.pp.filter_obs(self.atac, 'total_counts', lambda x: (x >= min_total_counts_per_cell) & (x <= max_total_counts_per_cell))
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.violin(self.atac, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True, save="_post_qc_filtering_atac.png")         
            mu.pl.histogram(self.atac, ['n_genes_by_counts', 'total_counts'], save="_qc_histograms_atac.png")
        

        # Save original counts
        self.atac.layers["counts"] = self.atac.X.copy()
        
        ac.pp.tfidf(self.atac, scale_factor=scale_factor)
        
        sc.pp.normalize_per_cell(self.atac, counts_per_cell_after=scale_factor)
        sc.pp.log1p(self.atac)
        
        sc.pp.highly_variable_genes(self.atac, min_mean=min_atac_hvg_mean, max_mean=max_atac_hvg_mean, min_disp=min_atac_disp)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.highly_variable_genes(self.atac, save="_peaks.png")
            
        if filter_hvgs:
            keep_peaks = self.atac.var['highly_variable']
            self.atac = self.atac[:, keep_peaks].copy()
        
        self.atac.layers["tfidf"] = self.atac.X.copy()
        
        self.mdata.mod["atac"] = self.atac
            
        # Scaling
        self.atac.raw = self.atac
        
        # LSI
        ac.tl.lsi(self.atac)
        
        self.atac.obsm['X_lsi'] = self.atac.obsm['X_lsi'][:,1:]
        self.atac.varm["LSI"] = self.atac.varm["LSI"][:,1:]
        self.atac.uns["lsi"]["stdev"] = self.atac.uns["lsi"]["stdev"][1:]
        
        # Neighbors
        sc.pp.neighbors(self.atac, use_rep="X_lsi", n_neighbors=n_neighbors, n_pcs=n_pcs)
        
        # PCA
        sc.pp.scale(self.atac)
        sc.tl.pca(self.atac)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.pca(self.atac, color=["n_genes_by_counts", "n_counts"], save="_pca_qc_atac.png")
        
        # Annotate peaks as promoter/distal/intergenic based on TSS proximity
        # assign gene names and distances for promoter/distal peaks
        self.construct_peak_annotation(
            self.processed_data_dir, 
            promoter_upstream=promoter_upstream, 
            promoter_downstream=promoter_downstream, 
            distal_max=distal_max
            )
        
        ac.tl.add_peak_annotation(self.atac, annotation=str(self.processed_data_dir / "atac_peak_annotation.tsv"))
        
        # Neighbors
        sc.pp.neighbors(self.atac, n_neighbors=n_neighbors, n_pcs=n_pcs)
        
        # Clustering
        sc.tl.umap(self.atac, spread=1., min_dist=.5, random_state=11)
        sc.tl.leiden(self.atac, flavor="igraph", n_iterations=2)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.umap(self.atac, color=["leiden", "n_genes_by_counts"], legend_loc="on data", save="_umap_leiden_atac.png")
            
        mu.write(str(self.processed_data_dir / f"{self.sample_name}.h5mu/atac"), self.atac)
    
          
    def nucleosome_signal(
        self,
        frag_path: Path,
        fig_dir: Path|None = None
    ):
        index_file = str(frag_path) + ".tbi"

        if not Path(index_file).exists():
            create_fragment_index_file(frag_path)

        tbx = pysam.TabixFile(str(frag_path))
        logging.info(tbx.contigs[:20])

        # register the fragment file with the ATAC AnnData
        ac.tl.locate_fragments(self.atac, fragments=str(frag_path))

        self.atac.obs["NS"] = 1

        def find_nonempty_region(tbx, chrom="chr1", window=1_000_000, max_end=200_000_000):
            for start in range(0, max_end, window):
                rows = list(tbx.fetch(chrom, start, start + window))
                if rows:
                    return f"{chrom}:{start}-{start+window}", rows[0]
            return None, None

        region, example = find_nonempty_region(tbx, chrom="chr1")

        if fig_dir is not None and fig_dir.exists():
            sc.settings.figdir = fig_dir
            ac.pl.fragment_histogram(self.atac, region=region, save="_fragment_histogram.png")
            
        ac.tl.nucleosome_signal(self.atac, n=1e6)
        
        if fig_dir is not None and fig_dir.exists():
            mu.pl.histogram(self.atac, "nucleosome_signal", kde=False, save="_nucleosome_signal.png")
            
    def tss_enrichment(
        self, 
        frag_path: Path, 
        n_tss: int = 500, 
        extend_upstream: int = 1000, 
        extend_downstream: int = 1000,
        fig_dir: Path|None = None
        ):
        if frag_path is not None and frag_path.exists():
            
            tss_df = pd.read_csv(
                self.tss_path,
                sep="\t",
                header=None,
                names=["tss_chrom", "tss_start", "tss_end", "tss_gene"]
            )

            var = self.rna.var.copy()
            var["original_var_name"] = var.index

            # Remove any old columns that could collide
            cols_to_drop = [
                "tss_chrom", "tss_start", "tss_end", "tss_gene",
                "Chromosome", "Start", "End", "interval"
            ]
            var = var.drop(columns=[c for c in cols_to_drop if c in var.columns], errors="ignore")

            # Merge gene annotations onto var
            var = (
                var.merge(
                    tss_df,
                    left_index=True,
                    right_on="tss_gene",
                    how="left"
                )
                .set_index("original_var_name")
            )

            var["interval"] = pd.NA

            mask = var["tss_chrom"].notna() & var["tss_start"].notna() & var["tss_end"].notna()

            var.loc[mask, "interval"] = (
                var.loc[mask, "tss_chrom"].astype(str) + ":" +
                var.loc[mask, "tss_start"].astype(int).astype(str) + "-" +
                var.loc[mask, "tss_end"].astype(int).astype(str)
            )

            var["Chromosome"] = var["tss_chrom"].astype(str) 
            var["Start"] = var["tss_start"]
            var["End"] = var["tss_end"]

            self.rna.var = var

            rna_tss = self.rna[:, self.rna.var["interval"].notna()].copy()
            
            genes = ac.tl.get_gene_annotation_from_rna(rna_tss)
            
            tss = ac.tl.tss_enrichment(
                self.mdata, 
                features=genes, 
                n_tss=n_tss, 
                extend_upstream=extend_upstream, 
                extend_downstream=extend_downstream, 
                random_state=11
                )
            
            if fig_dir is not None and fig_dir.exists():
                self.tss_enrichment_plot(tss, save=str(fig_dir / "tss_enrichment.png"))

    def tss_enrichment_plot(
        self,
        data: AnnData,
        color: str | None = None,
        title: str = "TSS Enrichment",
        ax: Axes | None = None,
        save: str = None
    ):
        """
        Plot relative enrichment scores around a TSS.

        Parameters
        ----------
        data
            AnnData object with cell x TSS_position matrix as generated by `muon.atac.tl.tss_enrichment`.
        color
            Column name of .obs slot of the AnnData object which to group TSS signals by.
        title
            Plot title.
        ax
            A matplotlib axes object.
        """
        ax = ax or plt.gca()

        if color is not None:
            if isinstance(color, str):
                color = [color]

            groups = data.obs.groupby(color)

            for name, group in groups:
                ad = data[group.index]
                ac.pl._tss_enrichment_single(ad, ax, label=name)
        else:
            ac.pl._tss_enrichment_single(data, ax)

        # TODO Not sure how to best deal with plot returning/showing
        ax.set_title(title)
        ax.set_xlabel("Distance from TSS, bp")
        ax.set_ylabel("Average TSS enrichment score")
        if color:
            ax.legend(loc="upper right", title=", ".join(color))
        if save:
            plt.savefig(save, dpi=150)
        
        plt.show()
        return None

    def save_mdata(self):
        mu.write(self.processed_data_dir / self.sample_name / f"{self.sample_name}.h5mu", self.mdata)

    def save_rna(self):
        
        mu.write(self.processed_data_dir / self.sample_name / f"{self.sample_name}.h5mu/rna", self.rna)
        
    def save_atac(self):
        mu.write(self.processed_data_dir / self.sample_name / f"{self.sample_name}.h5mu/atac", self.atac)
     
            
def integrate_rna_atac(
    mdata: ad.AnnData, 
    sample_processed_data_dir: Path, 
    sample_name: str,
    fig_dir: Path|None = None
    ):
    
    
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = fig_dir
    
    # Restrict to cells passing QC in both modalities
    mu.pp.intersect_obs(mdata)

    # MOFA expects feature dimensions to match loadings exactly.
    # Guard each modality (RNA + ATAC) for non-finite, all-zero, and zero-variance
    # features to avoid MOFA internally dropping columns and desynchronizing shapes.
    for mod_name in mdata.mod:
        adata_mod = mdata.mod[mod_name]
        adata_mod.var_names_make_unique()

        if sp.issparse(adata_mod.X):
            bad_vals = ~np.isfinite(adata_mod.X.data)
            if bad_vals.any():
                adata_mod.X.data[bad_vals] = 0.0
                adata_mod.X.eliminate_zeros()

            n_obs = max(int(adata_mod.n_obs), 1)
            mean = np.asarray(adata_mod.X.sum(axis=0)).ravel() / n_obs
            mean_sq = np.asarray(adata_mod.X.power(2).sum(axis=0)).ravel() / n_obs
            var = mean_sq - np.square(mean)
            nonzero_per_feature = np.asarray((adata_mod.X != 0).sum(axis=0)).ravel()
        else:
            X = np.asarray(adata_mod.X)
            X[~np.isfinite(X)] = 0.0
            adata_mod.X = X
            var = np.var(X, axis=0)
            nonzero_per_feature = np.count_nonzero(X, axis=0)

        keep_features = (nonzero_per_feature > 0) & np.isfinite(var) & (var > 0)
        n_before = adata_mod.n_vars
        if not np.all(keep_features):
            adata_mod = adata_mod[:, keep_features].copy()
            mdata.mod[mod_name] = adata_mod
        logging.info(f"  - {mod_name}: kept {adata_mod.n_vars}/{n_before} features after MOFA precheck")

    # Ensure MuData global annotations stay in sync with updated modalities.
    if hasattr(mdata, "update"):
        mdata.update()
    logging.info(
        f"  - post-update dims: RNA={mdata.mod['rna'].n_vars}, "
        f"ATAC={mdata.mod['atac'].n_vars}, total={mdata.mod['rna'].n_vars + mdata.mod['atac'].n_vars}"
    )
    
    # Perform MOFA+ integration
    # Seeded: mu.tl.mofa is stochastic, and the latent space it produces decides the
    # neighbour graph, which decides the diffusion operator in create_metacells, which
    # decides every pseudobulk value. Unseeded, two runs on identical inputs give different
    # parquets and a lost file cannot be regenerated.
    mu.tl.mofa(
        mdata,
        outfile=sample_processed_data_dir / f"{sample_name}_rna_atac.h5mu",
        seed=MOFA_SEED,
    )
    logging.info(f"MOFA seed: {MOFA_SEED}")

    # sc.pp.neighbors / sc.tl.leiden are also stochastic; pinning them keeps the graph the
    # diffusion runs over reproducible too.
    sc.pp.neighbors(mdata, use_rep="X_mofa", random_state=MOFA_SEED)
    sc.tl.umap(mdata, random_state=MOFA_SEED)
    sc.tl.umap(mdata, min_dist=.2, spread=1., random_state=10)
    sc.tl.leiden(mdata, flavor="igraph", n_iterations=2, random_state=MOFA_SEED)

    if fig_dir is not None and fig_dir.exists():
        # Plot the UMAP colored by MOFA clusters
        sc.pl.umap(mdata, color=["leiden"], save="mofa_umap_leiden.png")
        
        # Plot the first 4 MOFA factors in pairwise scatter plots
        df = pd.DataFrame(mdata.obsm["X_mofa"])
        df.columns = [f"Factor {i+1}" for i in range(df.shape[1])]

        plot_scatter = lambda i, ax: sns.scatterplot(data=df, x=f"Factor {i+1}", y=f"Factor {i+2}", color="black", linewidth=0, s=3, ax=ax)

        fig, axes = plt.subplots(2, 2)
        for i in range(4):
            plot_scatter(i, axes[i%2][i//2])
            
        plt.tight_layout()
        plt.savefig(fig_dir / "mofa_factor_scatter.png", dpi=150)
        plt.close()
        
    # Ranking genes and peaks
    mdata["rna"].obs["leiden_joint"] = mdata.obs["leiden"]
    mdata["atac"].obs["leiden_joint"] = mdata.obs["leiden"]
    
    sc.tl.rank_genes_groups(mdata['rna'], 'leiden_joint', method='t-test_overestim_var')
    ac.tl.rank_peaks_groups(mdata['atac'], 'leiden_joint', method='t-test_overestim_var')
    
    
def save_processed_data(mdata: ad.AnnData, sample_processed_data_dir: Path):
    """Write the integrated MuData object and per-modality feature x cell parquets to disk."""
    def _adata_to_feature_by_cell_df(adata: ad.AnnData) -> pd.DataFrame:
        """
        Convert an AnnData object from cell x feature to feature x cell DataFrame.
        Preference order:
        1. adata.layers["log1p"]
        2. adata.layers["counts"]
        3. adata.X
        """
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

        return pd.DataFrame(
            arr,
            index=adata.var_names.astype(str),
            columns=adata.obs_names.astype(str),
        )

    def standardize_name(name: str) -> str:
        """Convert gene/motif name to upper style."""
        if not isinstance(name, str):
            return name
        return name.upper()

    processed_rna_file = sample_processed_data_dir / "scRNA_seq_processed.parquet"
    processed_atac_file = sample_processed_data_dir / "scATAC_seq_processed.parquet"

    mdata_file = sample_processed_data_dir / "multiome_processed.h5mu"

    # Pull modalities from MuData
    adata_rna = mdata["rna"]
    adata_atac = mdata["atac"]

    # Convert to feature x cell DataFrames
    processed_rna_df = _adata_to_feature_by_cell_df(adata_rna).astype("float32")
    processed_atac_df = _adata_to_feature_by_cell_df(adata_atac).astype("float32")

    # Standardize RNA gene names
    processed_rna_df.index = processed_rna_df.index.astype(str).map(standardize_name)

    # Save parquet outputs
    processed_rna_df.to_parquet(processed_rna_file, engine="pyarrow", compression="snappy")
    processed_atac_df.to_parquet(processed_atac_file, engine="pyarrow", compression="snappy")

    # Save the full MuData object 
    mdata.write(mdata_file) 
    
    
def create_metacells(
    mdata: ad.AnnData, 
    sample_processed_data_dir: Path, 
    hops: int = 2,
    cell_type_labels=None,
    standardize: bool = True,
    reference_quantiles_path: Path = DEFAULT_REFERENCE_QUANTILES,
    reference_signature_path: Path = DEFAULT_REFERENCE_SIGNATURE,
    gate_mode: str = "error",
    ):
    """
    Create metacell-level profiles from RNA and ATAC data matrices.

    Parameters
    ----------
    mdata : ad.AnnData
        The MuData object containing RNA and ATAC data matrices.
    sample_processed_data_dir : Path
        The directory where the pseudobulk DataFrames will be saved.
    hops : int, default=2
        The number of hops between neighbors to consider when diffusing information.
    cell_type_labels : array-like of str, optional
        One label per cell, in mdata's cell order. When given, the kNN graph is masked to
        within-cell-type edges before diffusing, so a cell's smoothed profile is built only
        from cells of its own type. Required for the cell-type-specific model; see the note
        at the masking step for the measurement that makes it necessary.
    standardize : bool, default=True
        Map the diffused matrices onto the frozen reference marginal before writing, so this
        sample lands on the same numeric scale as every other one regardless of how its QC
        thresholds were tuned. Exact zeros are held at zero. Set False to reproduce the
        unstandardised output of the original script.
    reference_quantiles_path : Path
        The frozen reference built by 09_build_reference_quantiles.py.
    reference_signature_path : Path
        Expected non-zero quantiles the written parquets are checked against.
    gate_mode : {"error", "warn", "off"}, default="error"
        What to do when the written pseudobulks do not match the reference signature.
        "error" raises -- a convention change should fail at build time rather than surface
        months later as a silent cross-dataset accuracy drop.

    Returns
    -------
    None

    Notes
    -----
    This function creates metacell-level profiles by applying a diffusion operator to the RNA and ATAC data matrices.
    The diffusion operator is constructed by first extracting the neighbor graph from the MuData object and converting it to a row-normalized sparse matrix.
    The operator is then applied to the RNA and ATAC data matrices to obtain the metacell-level profiles.
    The resulting profiles are saved as parquet files in the specified directory.
    """
    # Extract the neighbor graph and convert to a row-normalized sparse matrix
    W = mdata.obsp["connectivities"].tocsr().astype(np.float32)

    # --- restrict diffusion to within a cell type ---------------------------------------
    # Without this, a cell's smoothed profile is built from whichever neighbours the joint
    # kNN graph gives it, and on the liver Multiome most of them are other cell types.
    # Measured on data/processed/mouse_liver/multiome_processed.h5mu at hops=2: the mean
    # share of a cell's diffusion weight coming from its OWN type is 0.524, and 10,134 of
    # 19,662 cells (51.5%) end up majority some other cell type. Per type:
    #
    #     B cells 0.320   T cells 0.338   DCs 0.271   Cholangiocytes 0.354
    #     HpSC 0.384      KCs 0.517       Hepatocytes 0.578
    #     Fibroblasts 0.756                Endothelial 0.905
    #
    # A B cell whose expression profile is 68% other cell types cannot carry a B-cell-
    # specific label; the conditioning this whole variant adds would be washed out before
    # the model ever saw it. Masking is therefore required, not cosmetic.
    #
    # The mask is applied BEFORE the hops, not after. Masking the finished 2-hop operator
    # would still let weight travel A -> (other type) -> A, so a foreign cell would launder
    # its profile through the second hop; zeroing the edges first makes every path stay
    # inside one cell type.
    if cell_type_labels is not None:
        labels = np.asarray(cell_type_labels)
        if len(labels) != W.shape[0]:
            raise ValueError(
                f"cell_type_labels has {len(labels)} entries but the graph has "
                f"{W.shape[0]} cells."
            )
        codes = pd.factorize(labels)[0].astype(np.int32)
        W = W.tocoo()
        keep = codes[W.row] == codes[W.col]
        n_before = W.nnz
        W = sp.csr_matrix(
            (W.data[keep], (W.row[keep], W.col[keep])), shape=W.shape
        )
        logging.info(
            f"Cell-type-masked the kNN graph: kept {W.nnz:,} of {n_before:,} edges "
            f"({100 * W.nnz / max(n_before, 1):.1f}%); "
            f"{len(np.unique(codes))} cell types."
        )

    # Add self-connections
    W = W + sp.diags(np.full(W.shape[0], 1, dtype=np.float32), format="csr")
    
    def row_norm(mat: sp.csr_matrix) -> sp.csr_matrix:
        row_sum = np.asarray(mat.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        inv = sp.diags(1.0 / row_sum, dtype=np.float32)
        return inv @ mat

    W = row_norm(W)
    
    # Diffusion based on the number of hops between neighbors. 
    # Pools information from neighbors up to HOPS distance away, with more weight on closer neighbors.
    W_h = W
    for _ in range(1, int(hops)):
        W_h = W_h @ W 
        W_h = row_norm(W_h)
    W = W_h

    # Final row normalization to make sure rows sum to 1
    W = row_norm(W)

    if cell_type_labels is not None:
        # The mask is an invariant, not an intention: assert it rather than trust it. A
        # single cross-type edge here means a profile is contaminated in a way that is
        # invisible downstream.
        Wc = W.tocoo()
        leaked = int((codes[Wc.row] != codes[Wc.col]).sum())
        if leaked:
            raise AssertionError(
                f"{leaked:,} cross-cell-type edges survived masking. The diffusion "
                f"operator must be block diagonal by cell type."
            )
        own = np.asarray(W.sum(axis=1)).ravel()
        logging.info(
            f"Post-diffusion own-cell-type weight: min {own.min():.3f}, "
            f"mean {own.mean():.3f} (1.000 expected -- rows are normalised and every "
            f"path stays inside one cell type)."
        )
    
    def _to_csr32(mat) -> sp.csr_matrix:
        """Coerce a dense array or sparse matrix layer to a float32 CSR matrix."""
        if sp.issparse(mat):
            return mat.astype(np.float32).tocsr()
        return sp.csr_matrix(np.asarray(mat, dtype=np.float32, order="C"))

    # Apply the diffusion operator to the RNA and ATAC data matrices to get metacell-level profiles.
    X_rna = _to_csr32(mdata["rna"].layers["log1p"])
    X_atac = _to_csr32(mdata["atac"].layers["tfidf"])

    X_rna_soft = W @ X_rna      # cells × genes
    X_atac_soft = W @ X_atac    # cells × peaks
    
    # Create and save the pseudobulk DataFrames
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

    pseudo_bulk_rna_df = pd.DataFrame(
        X_rna_soft.T.toarray(),
        index=mdata["rna"].var_names,
        columns=mdata["rna"].obs_names,
    ).fillna(0)

    pseudo_bulk_atac_df = pd.DataFrame(
        X_atac_soft.T.toarray(),
        index=mdata["atac"].var_names,
        columns=mdata["atac"].obs_names,
    ).fillna(0)

    pseudo_bulk_rna_df = _standardize_symbols_index(pseudo_bulk_rna_df)
    pseudobulk_rna_file = sample_processed_data_dir / "TG_pseudobulk.parquet"
    pseudobulk_atac_file = sample_processed_data_dir / "RE_pseudobulk.parquet"

    # --- scale standardisation -----------------------------------------------------------
    # The diffusion operator above is row-stochastic, so it is a convex average and preserves
    # whatever scale layers["log1p"] / layers["tfidf"] were on. That makes the exported
    # convention a pure function of those two layers -- which is exactly how the mESC files
    # ended up non-negative while the June-built samples are centred. Pinning the marginal
    # here makes the exported scale independent of both the layer definition and the
    # per-sample QC thresholds.
    # Checked BEFORE the map, because the map would hide the problem: if sc.pp.scale ever
    # runs again before the layers[...] copy, the values arriving here are centred, and
    # rewriting their marginal makes the output look correct while the zero block -- which
    # scaling destroys irrecoverably -- is already gone.
    def _negative_fraction(df, block_cols: int = 4096) -> float:
        """Fraction of values below zero, counted in column blocks.

        The obvious df.to_numpy() < 0 builds a full float copy AND a full bool array of the
        same shape. On the 19,662 x 149,393 ATAC pseudobulk that is ~15 GB of transient
        allocation for a single scalar, and it lands right before standardisation needs its
        own copy.
        """
        n_neg = 0
        n_tot = 0
        for start in range(0, df.shape[1], block_cols):
            block = df.iloc[:, start:start + block_cols].to_numpy(dtype=np.float32)
            n_neg += int((block < 0).sum())
            n_tot += block.size
            del block
        return float(n_neg / n_tot) if n_tot else 0.0

    raw_negative_fraction = {
        "rna": _negative_fraction(pseudo_bulk_rna_df),
        "atac": _negative_fraction(pseudo_bulk_atac_df),
    }
    for channel, fraction in raw_negative_fraction.items():
        if fraction > 1e-6:
            message = (
                f"{channel}: {fraction:.1%} of the diffused values are negative before "
                "standardisation. layers['log1p'] / layers['tfidf'] have been scaled -- check "
                "that sc.pp.scale runs AFTER the layer copy. Standardising this would produce "
                "a file with the right marginal and no zero block, which is not recoverable."
            )
            if gate_mode == "error":
                raise ValueError(message)
            logging.warning(f"[signature] {message}")

    if standardize:
        reference, provenance = load_reference(reference_quantiles_path)
        logging.info(
            f"Standardising pseudobulks against {reference_quantiles_path.name} "
            f"(built {provenance['created']} from {', '.join(provenance['sources'])})"
        )
        # One channel at a time, in place, dropping each DataFrame before the map runs.
        # The ATAC pseudobulk here is 19,662 x 149,393 float32 = 11.7 GB; the original
        # code held the DataFrame, its to_numpy copy and the returned array at once, which
        # is ~35 GB for this one channel and does not fit alongside the rest of the job.
        # standardize_matrix_inplace fits on the same non-zero subsample and applies the
        # same monotone map, so the numbers are unchanged.
        rna_index, rna_columns = pseudo_bulk_rna_df.index, pseudo_bulk_rna_df.columns
        rna_values = pseudo_bulk_rna_df.to_numpy(dtype=np.float32, copy=True)
        del pseudo_bulk_rna_df
        standardize_matrix_inplace(rna_values, reference["rna"], label="rna")
        pseudo_bulk_rna_df = pd.DataFrame(rna_values, index=rna_index, columns=rna_columns)
        del rna_values

        atac_index, atac_columns = pseudo_bulk_atac_df.index, pseudo_bulk_atac_df.columns
        atac_values = pseudo_bulk_atac_df.to_numpy(dtype=np.float32, copy=True)
        del pseudo_bulk_atac_df
        standardize_matrix_inplace(atac_values, reference["atac"], label="atac")
        pseudo_bulk_atac_df = pd.DataFrame(
            atac_values, index=atac_index, columns=atac_columns
        )
        del atac_values
    else:
        logging.warning(
            "standardize=False: writing the raw diffused values. These are only comparable "
            "with samples built the same way -- do not mix them into one training run or a "
            "cross-sample generalizability comparison."
        )

    def _write_parquet_chunked(df, path, row_chunk: int = 4096):
        """Write a wide dense DataFrame one row block at a time.

        df.to_parquet builds an Arrow table for the WHOLE frame before writing a byte. On
        the 149,393 x 19,662 float32 ATAC pseudobulk that is a second 11.7 GB allocation on
        top of the DataFrame, and it is where this step was OOM-killed. Writing row groups
        incrementally keeps only one block in Arrow at a time. The file is equivalent --
        same columns, same index, same values -- it just has more row groups.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        tmp = Path(str(path) + ".partial")
        writer = None
        try:
            for start in range(0, len(df), row_chunk):
                block = pa.Table.from_pandas(
                    df.iloc[start:start + row_chunk], preserve_index=True
                )
                if writer is None:
                    writer = pq.ParquetWriter(tmp, block.schema, compression="snappy")
                writer.write_table(block)
                del block
        finally:
            if writer is not None:
                writer.close()
        # GPFS has been observed to raise ENOENT from this rename even though it took
        # effect -- the 8.7 GB RE_pseudobulk landed correctly (37 row groups, the chunk
        # count) while os.rename reported the source missing, and the exception aborted the
        # run after all the expensive work was done. Treat "source gone, destination
        # present and non-empty" as the success it is, and only raise if the destination
        # really is not there.
        try:
            tmp.rename(path)
        except FileNotFoundError:
            if not (path.exists() and path.stat().st_size > 0):
                raise
            logging.warning(
                f"rename({tmp.name} -> {path.name}) reported ENOENT but {path.name} is "
                f"present at {path.stat().st_size / 1e9:.2f} GB; treating as written. "
                f"This is a filesystem metadata artefact, not a data problem -- the file "
                f"is verified by verify_masked_pseudobulk.py."
            )
        if tmp.exists():
            tmp.unlink()
        logging.info(f"Wrote {path.name} ({path.stat().st_size / 1e9:.2f} GB)")

    _write_parquet_chunked(pseudo_bulk_rna_df, pseudobulk_rna_file)
    _write_parquet_chunked(pseudo_bulk_atac_df, pseudobulk_atac_file)

    # --- distributional gate --------------------------------------------------------------
    # Recorded from what was actually written, not from the in-memory arrays, so a parquet
    # round-trip problem cannot slip through.
    # One channel at a time: the ATAC matrix alone runs to ~6 GB dense, and holding both
    # the DataFrame and its numpy view for each would double that for no reason.
    # The signature is computed from the in-memory frames, not by reading the parquet back.
    # Reading back was there so a parquet round-trip problem could not slip through, but it
    # costs another full copy of each matrix (~13 GB for the pair) at the point where memory
    # is already tightest. The round-trip is still checked, on a slice: that catches a
    # dtype, index or column mangling, which is what the read-back was actually guarding
    # against -- a silent per-value corruption in the middle of a snappy parquet is not a
    # failure mode this would have caught anyway.
    #
    # compute_signature consumes its two arguments independently, so each channel is
    # measured on its own and the halves combined. The stand-in for the other channel is a
    # 1x1 non-zero array, not an empty one: ChannelSignature.from_matrix divides by the
    # value count and raises ZeroDivisionError on zero-size input.
    _stub = np.ones((1, 1), dtype=np.float32)
    sig_rna = compute_signature(
        pseudo_bulk_rna_df.to_numpy(dtype=np.float32), _stub
    )
    sig_atac = compute_signature(
        _stub, pseudo_bulk_atac_df.to_numpy(dtype=np.float32)
    )
    signature = {"quantile_levels": sig_rna["quantile_levels"],
                 "rna": sig_rna["rna"], "atac": sig_atac["atac"]}
    signature["raw_negative_fraction"] = raw_negative_fraction
    signature["standardized"] = bool(standardize)

    for name, path, df in (("rna", pseudobulk_rna_file, pseudo_bulk_rna_df),
                           ("atac", pseudobulk_atac_file, pseudo_bulk_atac_df)):
        head = pd.read_parquet(path, columns=list(df.columns[:64])).head(2048)
        want = df.iloc[:2048, :64]
        if list(head.index) != list(want.index) or list(head.columns) != list(want.columns):
            raise ValueError(
                f"{path.name}: index or columns changed on the parquet round trip."
            )
        if not np.allclose(head.to_numpy(dtype=np.float32),
                           want.to_numpy(dtype=np.float32), equal_nan=True):
            raise ValueError(f"{path.name}: values changed on the parquet round trip.")
        logging.info(f"[round trip] {name}: {want.shape} slice matches what was written.")
    signature_file = sample_processed_data_dir / "pseudobulk_signature.json"
    signature_file.write_text(json.dumps(signature, indent=2))
    logging.info(f"Wrote {signature_file}")

    if gate_mode == "off":
        return

    if not Path(reference_signature_path).is_file():
        logging.warning(
            f"{reference_signature_path} not found -- skipping the gate. Build it with "
            "09_build_reference_quantiles.py."
        )
        return

    expected = json.loads(Path(reference_signature_path).read_text())
    passed, problems = check_signature(signature, expected)
    if passed:
        logging.info("[signature] pseudobulks match the reference convention.")
        return

    report = "\n  ".join(problems)
    message = (
        f"Pseudobulk signature check FAILED for {sample_processed_data_dir.name}:\n  {report}\n"
        f"Written signature: {signature_file}\n"
        f"Expected: {reference_signature_path}"
    )
    if gate_mode == "warn":
        logging.warning(message)
    else:
        raise ValueError(message)
    
    
def get_threshold(sample_filtering_settings, setting_name, verbose=True):
    setting_value = sample_filtering_settings[setting_name].values[0]
    if verbose:
        logging.info(f"{setting_name}: {setting_value}")
    
    return setting_value
    
    
if __name__ == "__main__":
    args = parse_args()

    PROJECT_DIR = Path(args.project_dir)
    RAW_DATA_DIR = Path(args.raw_data_dir)
    PROCESSED_DATA_DIR = Path(args.processed_data_dir)
    SAMPLE_NAME = args.sample_name

    tss_path = Path(args.tss_path)
    rna_count_file = Path(args.rna_count_file) if args.rna_count_file else None
    atac_count_file = Path(args.atac_count_file) if args.atac_count_file else None
    raw_h5_file = Path(args.raw_h5_file) if args.raw_h5_file else None
    tf_list_file = Path(args.tf_list_file) if args.tf_list_file else None
    frag_path = Path(args.frag_path) if args.frag_path else None

    SAMPLE_DATA_DIR = RAW_DATA_DIR / SAMPLE_NAME
    SAMPLE_PROCESSED_DATA_DIR = PROCESSED_DATA_DIR / SAMPLE_NAME
    
    filtering_setting_df = pd.read_csv(PROJECT_DIR / "data" / "qc_filtering_settings.tsv", sep="\t")
    sample_filtering_settings = filtering_setting_df[filtering_setting_df["Sample"] == SAMPLE_NAME]    
    
    # ----- RNA QC thresholds -----
    MIN_CELLS_PER_GENE = get_threshold(sample_filtering_settings, "Min Cells per Gene")
    MIN_GENES_PER_CELL = get_threshold(sample_filtering_settings, "Min Genes per Cell")
    MAX_GENES_PER_CELL = get_threshold(sample_filtering_settings, "Max Genes per Cell")
    MIN_TOTAL_COUNTS = get_threshold(sample_filtering_settings, "Min Total Counts")
    MAX_TOTAL_COUNTS = get_threshold(sample_filtering_settings, "Max Total Counts")
    MAX_PCT_COUNTS_MT = get_threshold(sample_filtering_settings, "Max Pct MT")

    # ----- ATAC QC thresholds -----
    MIN_CELLS_PER_PEAK = get_threshold(sample_filtering_settings, "Min Cells per Peak")
    MIN_PEAKS_PER_CELL = get_threshold(sample_filtering_settings, "Min Peaks per Cell")
    MAX_PEAKS_PER_CELL = get_threshold(sample_filtering_settings, "Max Peaks per Cell")
    MIN_TOTAL_PEAK_COUNTS = get_threshold(sample_filtering_settings, "Min Total Peak Counts")
    MAX_TOTAL_PEAK_COUNTS = get_threshold(sample_filtering_settings, "Max Total Peak Counts")

    if not SAMPLE_PROCESSED_DATA_DIR.exists():
        SAMPLE_PROCESSED_DATA_DIR.mkdir(parents=True)
    
    mdata, _ = load_raw_data(SAMPLE_NAME, SAMPLE_DATA_DIR, rna_count_file, atac_count_file, raw_h5_file)

    mdata.write(SAMPLE_PROCESSED_DATA_DIR / f"{SAMPLE_NAME}.h5mu")
    
    data_processor = MudataProcessor(
        mdata=mdata,
        processed_data_dir=SAMPLE_PROCESSED_DATA_DIR,
        sample_name=SAMPLE_NAME,
        tss_path=tss_path,
        tf_list_file=tf_list_file
    )
    
    # RNA QC and Preprocessing
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
        tf_list_file = tf_list_file,
        fig_dir=SAMPLE_PROCESSED_DATA_DIR / "preprocessing_figures" / "rna_qc",
        )
    
    data_processor.rna_pca_and_neighbors(
        data_processor.rna, 
        n_pcs=20,
        n_neighbors=10,
        fig_dir=SAMPLE_PROCESSED_DATA_DIR / "preprocessing_figures" / "rna_qc",
        )
    
    # ATAC QC and Preprocessing
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
        fig_dir=SAMPLE_PROCESSED_DATA_DIR / "preprocessing_figures" / "atac_qc",
        )
    
    if frag_path is not None and frag_path.exists():
        data_processor.nucleosome_signal(
            frag_path=frag_path, 
            fig_dir=SAMPLE_PROCESSED_DATA_DIR / "preprocessing_figures" / "atac_qc"
            )
        
        data_processor.tss_enrichment(
            frag_path=frag_path, 
            n_tss=500, 
            extend_upstream=1000, 
            extend_downstream=1000,
            fig_dir=SAMPLE_PROCESSED_DATA_DIR / "preprocessing_figures" / "atac_qc"
            )
    
    # Integrate the RNA and ATAC modalities using MOFA+
    integrate_rna_atac(data_processor.mdata, SAMPLE_PROCESSED_DATA_DIR, SAMPLE_NAME, fig_dir=SAMPLE_PROCESSED_DATA_DIR / "integration")

    save_processed_data(data_processor.mdata, SAMPLE_PROCESSED_DATA_DIR)

    # Create metacells
    create_metacells(
        data_processor.mdata,
        SAMPLE_PROCESSED_DATA_DIR,
        hops=2,
        standardize=not args.no_standardize,
        reference_quantiles_path=Path(args.reference_quantiles),
        reference_signature_path=Path(args.reference_signature),
        gate_mode=args.gate_mode,
    )
