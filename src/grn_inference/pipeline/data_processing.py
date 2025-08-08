import os

import anndata  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import scanpy as sc  # type: ignore[import-untyped]
from scipy.sparse import csr_matrix
from typing import Union
import logging
import pybedtools # type: ignore[import-untyped]
from pybiomart import Server # type: ignore[import-untyped]
import argparse

logging.basicConfig(level=logging.INFO, format='%(message)s')

from grn_inference.plotting import plot_feature_score_histogram
from grn_inference.normalization import minmax_normalize_pandas
from grn_inference.utils import (
    load_atac_dataset,
    load_rna_dataset,
    format_peaks,
    find_genes_near_peaks
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument("--atac_data_file", type=str, required=True, help="Path to the scATAC-seq dataset")
    parser.add_argument("--rna_data_file", type=str, required=True, help="Path to the scRNA-seq dataset")
    parser.add_argument("--species", type=str, required=True, help="Species genome, e.g. 'mm10' or 'hg38'")
    parser.add_argument("--tss_distance_cutoff", type=int, required=False, default=1_000_000, help="Distance (bp) from TSS to filter peaks (default: 1,000,000)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the sample")
    parser.add_argument("--fig_dir", type=str, required=True, help="Path to the figure directory for the sample")
    parser.add_argument("--overwrite_files", type=str, choices=["True", "False"], required=False, default="False", 
                        help="Overwrite existing processed datasets if they exist. True / False (default: False)")
    return parser.parse_args()

def anndata_from_dataframe(df, id_col_name):
    # 1) Validate input
    if id_col_name not in df.columns:
        raise ValueError(f"Identifier column '{id_col_name}' not found in DataFrame.")

    # Separate gene IDs vs. raw count matrix
    gene_or_peak_ids = df[id_col_name].astype(str).tolist()
    counts_df = df.drop(columns=[id_col_name]).copy()

    # Ensure all other columns are numeric; coerce non‐numeric to NaN→0
    counts_df = counts_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    # Extract cell IDs from the DataFrame columns
    cell_ids = counts_df.columns.astype(str).tolist()

    # 2) Build AnnData with shape (cells × genes)
    #    We must transpose counts so rows=cells, columns=genes
    counts_matrix = csr_matrix(counts_df.values)           # shape: (n_genes, n_cells)
    counts_matrix = counts_matrix.T                         # now (n_cells, n_genes)

    adata = anndata.AnnData(X=counts_matrix)
    adata.obs_names = cell_ids       # each row = one cell
    adata.var_names = gene_or_peak_ids       # each column = one gene or peak
    
    return adata


def convert_anndata_to_pandas(adata: anndata.AnnData, id_col_name: str) -> pd.DataFrame:
    """
    Convert an AnnData object to a Pandas DataFrame.
    
    - **var_names** = gene / peak names
    - **obs_names** = cell names / barcodes

    Args:
        adata (AnnData): AnnData object containing scATAC-seq or scRNA-seq data
        id_col_name (str): Name for the peak / gene ID column ("peak_id" or "gene_id")

    Returns:
        pd.DataFrame: DataFrame of gene x cell expression data. Header contains cell names, column 0 
        contains gene / peak names
    """
    adata = adata.copy()
    
    df = pd.DataFrame(
        data=adata.X.T.toarray(),
        index=adata.var_names,    # genes
        columns=adata.obs_names    # filtered cells
    )
    
    # Add the gene / peak names as column 0 rather than the index
    df.insert(loc=0, column=id_col_name, value=df.index.astype(str))
    df = df.reset_index(drop=True)

    return df


def write_processed_dataframe_to_parquet(df: pd.DataFrame, data_file_path: str) -> None:
    """
    Writes the processed DataFrame object to a '_processed.parquet' file.
    
    The data_file_path argument should be the same as the path to the raw data file. This function
    saves the processed parquet file to the same location as the input data, but adds '_processed.parquet'
    to the end of the file.

    Args:
        df (pd.DataFrame): DataFrame of gene x cell expression data. Header contains cell names, column 0 
        contains gene / peak names
        data_file_path (str): Path to the input data file.
    """

    if not "_processed.parquet" in data_file_path:
    
        def update_name(filename):
            base, ext = os.path.splitext(filename)
            return f"{base}_processed.parquet"

        data_file_path = update_name(data_file_path)
        logging.info(f"  - Updated file: {data_file_path}")
        
    else:
        logging.info(f"  - Save file already contains '_processed.parquet' ({os.path.basename(data_file_path)}), skipping renaming")
    
    logging.info(f'  - Writing processed dataset to {data_file_path}')
    df.to_parquet(data_file_path, engine="pyarrow", compression="snappy", index=False)
    logging.info("  Done!")


def extract_atac_peaks(atac_df, tmp_dir):
    
    if not os.path.exists(f"{tmp_dir}/peak_df.parquet"):
        logging.info(f"Extracting peak information and saving as a bed file")
        def parse_peak_str(s):
            try:
                chrom, coords = s.split(":")
                start_s, end_s = coords.split("-")
                return chrom.replace("chr", ""), int(start_s), int(end_s)
            except Exception:
                raise ValueError(f"Malformed peak_id '{s}'; expected 'chrN:start-end'.")

        # List of peak strings
        peak_pos = atac_df["peak_id"].tolist()

        # Apply parsing function to all peak strings
        parsed = [parse_peak_str(s) for s in peak_pos]

        # Construct DataFrame
        peak_df = pd.DataFrame(parsed, columns=["chr", "start", "end"])
        peak_df["peak_id"] = peak_pos
        
        # Write the peak DataFrame to a file
        peak_df.to_parquet(os.path.join(tmp_dir, "peak_df.parquet"), engine="pyarrow", index=False, compression="snappy")
        
    else:
        logging.info("ATAC-seq peak_df.parquet file exists, loading...")
        peak_df = pd.read_parquet(os.path.join(tmp_dir, "peak_df.parquet"), engine="pyarrow")
        
    return peak_df


def load_ensembl_organism_tss(organism, tmp_dir):
    if not os.path.exists(os.path.join(tmp_dir, "ensembl.parquet")):
        logging.info(f"Loading Ensembl TSS locations for {organism}")
        # Connect to the Ensembl BioMart server
        server = Server(host='http://www.ensembl.org')

        gene_ensembl_name = f'{organism}_gene_ensembl'
        
        # Select the Ensembl Mart and the human dataset
        mart = server['ENSEMBL_MART_ENSEMBL']
        try:
            dataset = mart[gene_ensembl_name]
        except KeyError:
            raise RuntimeError(f"BioMart dataset {gene_ensembl_name} not found. Check if ‘{organism}’ is correct.")

        # Query for attributes: Ensembl gene ID, gene name, strand, and transcription start site (TSS)
        ensembl_df = dataset.query(attributes=[
            'external_gene_name', 
            'strand', 
            'chromosome_name',
            'transcription_start_site'
        ])

        ensembl_df.rename(columns={
            "Chromosome/scaffold name": "chr",
            "Transcription start site (TSS)": "tss",
            "Gene name": "gene_id"
        }, inplace=True)
        
        # Make sure TSS is integer (some might be floats).
        ensembl_df["tss"] = ensembl_df["tss"].astype(int)

        # In a BED file, we’ll store TSS as [start, end) = [tss, tss+1)
        ensembl_df["start"] = ensembl_df["tss"].astype(int)
        ensembl_df["end"] = ensembl_df["tss"].astype(int) + 1

        # Re-order columns for clarity: [chr, start, end, gene]
        ensembl_df = ensembl_df[["chr", "start", "end", "gene_id"]]
        
        ensembl_df["chr"] = ensembl_df["chr"].astype(str)
        ensembl_df["gene_id"] = ensembl_df["gene_id"].astype(str)
        
        # Write the peak DataFrame to a file
        ensembl_df.to_parquet(os.path.join(tmp_dir, "ensembl.parquet"), engine="pyarrow", index=False, compression="snappy")
        
    else:
        logging.info("Ensembl gene TSS BED file exists, loading...")
        ensembl_df = pd.read_parquet(os.path.join(tmp_dir, "ensembl.parquet"), engine="pyarrow")
    
    return ensembl_df

def calculate_tss_distance_score(
    peak_bed: pybedtools.BedTool, 
    tss_bed: pybedtools.BedTool, 
    gene_names: set[str], 
    tss_distance_cutoff: Union[int, float] = 1e6
    ):
    """
    Identify genes whose transcription start sites (TSS) are near scATAC-seq peaks.
    
    This function:
        1. Calculates the absolute distances between peaks and gene TSSs within `tss_distance_cutoff`.
        2. Scales peak to gene distances using an exponential drop-off function (e^-dist/250000),
           the same method used in the LINGER cis-regulatory potential calculation.
        3. Deduplicates the data to keep the minimum (i.e., best) peak-to-gene connection.
        4. Only keeps genes that are present in the RNA-seq dataset.
        
    Args:
        peak_bed (pybedtools.BedTool):
            BedTool object representing scATAC-seq peaks.
        tss_bed (pybedtools.BedTool):
            BedTool object representing gene TSS locations.
        gene_names (set[str]):
            Set of gene names from the scRNA-seq dataset.
        tss_distance_cutoff (int): 
            The maximum distance (in bp) from a TSS to consider a peak as potentially regulatory.
        
    Returns:
        peak_tss_subset_df (pandas.DataFrame): 
            A DataFrame containing columns "peak_id", "target_id", and the scaled TSS distance "TSS_dist"
            for peak–gene pairs.
    """
    peak_tss_distance_df = find_genes_near_peaks(
        peak_bed, tss_bed, tss_distance_cutoff
        )

    # Scale the TSS distance using an exponential drop-off function
    # e^-dist/25000, same scaling function used in LINGER Cis-regulatory potential calculation
    # https://github.com/Durenlab/LINGER
    peak_tss_distance_df["TSS_dist_score"] = np.exp(-peak_tss_distance_df["TSS_dist"] / 250000)
    
    # Keep only the necessary columns.
    peak_tss_subset_df: pd.DataFrame = peak_tss_distance_df[["peak_id", "target_id", "TSS_dist_score"]]
    
    # Filter out any genes not found in the RNA-seq dataset.
    gene_names_upper = set(g.upper() for g in gene_names)
    
    mask = peak_tss_subset_df["target_id"].str.upper().isin(gene_names_upper)
    peak_tss_subset_df = peak_tss_subset_df[mask]
    logging.info(peak_tss_subset_df.head())
    logging.info(f'\t- Number of peaks: {len(peak_tss_subset_df.drop_duplicates(subset="peak_id"))}')
    
    return peak_tss_subset_df

def extract_atac_peaks_near_rna_genes(
    atac_df: pd.DataFrame, 
    gene_names: set[str], 
    organism: str, 
    tss_distance_cutoff: Union[int, float], 
    output_dir: str
    ) -> pd.DataFrame:
    """
    Identify genes whose transcription start sites (TSS) are near scATAC-seq peaks.
    
    This function:
        1. Uses BedTools to find peaks that are within peak_dist_limit bp of each gene's TSS.
        2. Converts the BedTool result to a pandas DataFrame.
        3. Computes the absolute distance between the peak end and gene start (as a proxy for TSS distance).
        4. Scales these distances using an exponential drop-off function (e^-dist/250000),
           the same method used in the LINGER cis-regulatory potential calculation.
        5. Deduplicates the data to keep the minimum (i.e., best) peak-to-gene connection.
        6. Only keeps genes that are present in the RNA-seq dataset.
        
    Parameters
    ----------
    atac_df (pd.DataFrame):
        DataFrame of scATAC-seq peak x cell counts. Must contain column "peak_id" with peaks in format chr:start-end.
    gene_names (set[str]):
        The set of gene names in the scRNA-seq datset.
    organism (str):
        The Ensembl organism name for downloading the gene TSS locations ("hsapiens" or "mmusculus")
    tss_distance_cutoff : int
        The maximum distance (in bp) from a TSS to consider a peak as potentially regulatory. Default 1e6.
    output_dir: str
        Output directory for the sample, used to save the tss_distance_score.parquet file

        
    Returns
    -------
    peaks_near_genes_df : pandas.DataFrame
        A DataFrame containing columns "peak_id", "target_id", and the scaled TSS distance "TSS_dist"
        for peak–gene pairs.
    """
    
    if not os.path.exists(os.path.join(output_dir, "tss_distance_score.parquet")):
        tmp_dir = os.path.join(output_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        
        if organism not in ("hsapiens", "mmusculus"):
            if organism == "hg38":
                organism = "hsapiens"
            elif organism == "mm10":
                organism = "mmusculus"
            else:
                raise ValueError(f"Organism not recognized: {organism} (must be 'hg38' or 'mm10').")

        # Format the ATAC peaks in BED format or load a cached file
        if not os.path.exists(f"{tmp_dir}/peak_df.parquet"):
            logging.info(f"    - Extracting peaks within {tss_distance_cutoff} bp of gene TSS")
            
            peak_df = format_peaks(atac_df["peak_id"])
        
            # Set the peak location dataframe to the correct BED format
            bedtool_df = peak_df.rename(columns={
                "chromosome": "chrom",
                "peak_id": "name"
            })[["chrom", "start", "end", "name", "strand"]]
            
            bedtool_df.to_parquet(os.path.join(tmp_dir, "peak_df.parquet"), engine="pyarrow", compression="snappy")
        
        else:
            logging.info("ATAC-seq BED file exists, loading...")
            peak_df = pd.read_parquet(os.path.join(tmp_dir, "peak_df.parquet"), engine="pyarrow")
            
            logging.info(peak_df.head())
            
            assert list(peak_df.columns) == ["chrom", "start", "end", "name", "strand"], \
                "peak_df must have columns: chrom, start, end, name, strand"
                
        peak_df = peak_df.rename(columns={
            "chrom": "chr",
            "name": "peak_id"
        })
                
        tss_df: pd.DataFrame = load_ensembl_organism_tss(organism, tmp_dir)
        
        logging.info(tss_df.head())

        pybedtools.set_tempdir(tmp_dir)
        try:
            # Ensure columns exist and are valid types
            for col in ["start", "end"]:
                peak_df[col] = pd.to_numeric(peak_df[col], errors="coerce")

            # Drop rows with invalid values
            peak_df = peak_df.dropna(subset=["chr", "start", "end", "peak_id"])

            # Force integer dtype
            peak_df["start"] = peak_df["start"].astype(int)
            peak_df["end"] = peak_df["end"].astype(int)
            
            tss_df = tss_df.dropna(subset=["chr", "start", "end", "gene_id"])
            tss_df["start"] = tss_df["start"].astype(int)
            tss_df["end"] = tss_df["end"].astype(int)
            
            def add_chr_prefix(s):
                s = str(s)
                if s.startswith("chr"):
                    return s
                elif not s.startswith("chr"):
                    return "chr" + s

            peak_df["chr"] = peak_df["chr"].astype(str)
            tss_df["chr"] = tss_df["chr"].astype(str)
            
            tss_df["chr"]  = tss_df["chr"].map(add_chr_prefix)
            peak_df["chr"] = peak_df["chr"].map(add_chr_prefix)
            
            logging.info("peak_df.head()")
            logging.info(peak_df.head())
            logging.info("\n")
            logging.info("tss_df.head()")
            logging.info(tss_df.head())
            
            peak_bed = pybedtools.BedTool.from_dataframe(peak_df[["chr", "start", "end", "peak_id"]])
            tss_bed = pybedtools.BedTool.from_dataframe(tss_df[["chr", "start", "end", "gene_id"]])
            
            if peak_bed is None or tss_bed is None:
                raise RuntimeError("Failed to create BedTool objects — check for invalid rows or types.")
            
            peaks_near_genes_df = calculate_tss_distance_score(
                peak_bed,
                tss_bed,
                gene_names,
                tss_distance_cutoff
            )
            
            peaks_near_genes_df = minmax_normalize_pandas(
                df=peaks_near_genes_df, 
                score_cols=["TSS_dist_score"], 
                )
            
            print(peaks_near_genes_df)
            
            logging.info(f"    - Saving peak to TSS distances to: {os.path.join(output_dir, 'tss_distance_score.parquet')}")
            peaks_near_genes_df.to_parquet(os.path.join(output_dir, "tss_distance_score.parquet"), index=False, engine="pyarrow", compression="snappy")
            
        finally:
            pybedtools.helpers.cleanup(verbose=False, remove_all=True)
            
        plot_feature_score_histogram(peaks_near_genes_df, "TSS_dist_score", output_dir)

        return peaks_near_genes_df
    
    else:
        logging.info('    - tss_distance_score.parquet TSS distance file exists, loading...')
        peaks_near_genes_df = pd.read_parquet(os.path.join(output_dir, "tss_distance_score.parquet"), engine="pyarrow")
    
    plot_feature_score_histogram(peaks_near_genes_df, "TSS_dist_score", output_dir)
    
    return peaks_near_genes_df


def atac_data_preprocessing(
    atac_data_path: str, 
    barcodes: list[str],
    gene_names: list[str],
    h5ad_save_path: str,
    filter_peak_min_cells: int = 30, 
    min_peaks_per_cell: int = 1000,
    target_read_depth: float = 1e6,
    tss_distance_cutoff: Union[int, float] = 1e6,
    fig_dir: str = 'figures',
    dataset_dir: str = 'mira-datasets',
    ensembl_species: str = "mmusculus",
    plot_peaks_by_counts: bool = True,
    overwrite: bool = False
    ) -> anndata.AnnData:
    """
    QC filtering and preprocessing of an ATACseq AnnData object.

    Args:
        atac_data_path (str): 
            Path to the input ATAC data (.csv, .tsv. or .parquet file).
        atac_h5ad_save_path (str): 
            Path to save the processed ATAC AnnData object as an h5ad file.
        barcodes (list[str]):
            A list of paired barcodes from the RNAseq dataset.
        gene_names (list[str]):
            A list of gene names from the RNAseq dataset.
        filter_peak_min_cells (int, optional): 
            A peak must be be expressed in greater than this number of cells. Defaults to 30.
        min_peaks_per_cell (int, optional): 
            A cell must be expressing more than this number of peaks. Defaults to 1000.
        target_read_depth (float, optional):
            Normalizes counts per cell to this value. Defaults to 1e6 (CPM normalization).
        tss_distance_cutoff (int | float, optional):
            Filter out peak to gene edges for peaks further away than this distance (in base pairs) 
            from the gene's TSS. Defaults to 1 MB.
        fig_dir (str, optional): 
            Figure for saving the `accessibility_genes_by_counts.png` figure. Defaults to 'figures'.
        dataset_dir (str, optional):
            Directory for saving the "tss_distance_score.parquet" file.
        ensembl_species (str, optional):
            Ensembl species name for loading gene TSS locations ('hsapiens' or 'mmusculus'). 
            Defaults to "mmusculus".
        plot_peaks_by_counts (bool, optional): 
            True to plot the figure, False to skip plotting. Defaults to True.
        h5ad_save_path (None | str): 
            Path to save the processed ATAC AnnData object as an h5 file.
        overwrite (bool):
            Set to True to overwrite the h5ad file if it exists. Defaults to False.

    Returns:
        atac_adata (anndata.AnnData): Filtered ATAC AnnData object
    """
    if not os.path.isfile(atac_data_path):
        raise FileNotFoundError(f"ATAC file not found: {atac_data_path}")
    
    if not barcodes:
        raise Exception("barcodes argument is None or empty, pass in the cell names / barcodes \
            from the scRNA-seq dataset")
    
    file_missing = not os.path.isfile(h5ad_save_path)
    if file_missing or overwrite:
        if "_processed" in os.path.basename(atac_data_path):
            raise Exception("Use scATAC-seq dataset with raw counts, raw counts \
                required when fitting the MIRA LITE model")
    
    
        logging.info(f"  - Reading ATACseq raw data file {atac_data_path}")
        raw_atac_df = load_atac_dataset(atac_data_path)
        
        logging.info("  - Filtering ATAC peaks by distance to TSS")
        atac_df_filtered = filter_atac_by_distance_to_tss(
            raw_atac_df, 
            gene_names,
            ensembl_species,
            tss_distance_cutoff,
            dataset_dir,
            fig_dir
            )
        
        atac_adata = anndata_from_dataframe(atac_df_filtered, "peak_id")
                
        logging.info(f"    - Number of Cells (unfiltered): {atac_adata.shape[0]}")
        logging.info(f"    - Number of Peaks (unfiltered): {atac_adata.shape[1]-1}")
        
        logging.info("    (1/5) Filtering out very rare peaks")
        sc.pp.filter_genes(atac_adata, min_cells = filter_peak_min_cells)

        valid_barcodes = [bc for bc in barcodes if bc in atac_adata.obs_names]
        
        if len(valid_barcodes) == 0:
            raise Exception("No matches between barcodes and atac_adata.obs_names")
        
        atac_adata = atac_adata[valid_barcodes]
        
        logging.info("    (2/5) Calculating QC metrics")
        sc.pp.calculate_qc_metrics(atac_adata, inplace=True, log1p=False)
        
        if plot_peaks_by_counts:
            logging.info("      - Plotting genes by counts vs total counts")
            ax: plt.Axes = sc.pl.scatter(atac_adata,
                        x = 'n_genes_by_counts',
                        y = 'total_counts',
                        show = False,
                        size = 2,
                        )

            ax.vlines(1000, 100, 1e5)
            ax.set(xscale = 'log', yscale = 'log')
            
            fig = ax.get_figure()
            
            qc_fig_path = os.path.join(fig_dir, "QC_figs")
            os.makedirs(qc_fig_path, exist_ok=True)
            
            if isinstance(fig, plt.Figure):
                fig.savefig(
                    os.path.join(qc_fig_path, "accessibility_peaks_by_counts.png"),
                    dpi=200,
                    bbox_inches="tight"
                )

        logging.info(f"    (3/5) Filtering cells by {min_peaks_per_cell} min peaks per cell")
        sc.pp.filter_cells(atac_adata, min_genes=min_peaks_per_cell)
        atac_adata.layers["counts"] = atac_adata.layers["counts"] = atac_adata.X.copy().astype(np.uint16)
        
        logging.info(f"    (4/5) Normalizing to a read depth of {target_read_depth}")
        sc.pp.normalize_total(atac_adata, target_sum=target_read_depth)

        logging.info("    (5/5) Logarithmizing the data")
        sc.pp.log1p(atac_adata)

        logging.info(f"    (5/5) Subsampling to 1e5 peaks per cell")
        # # If needed, reduce the size of the dataset by subsampling
        np.random.seed(0)
        atac_adata.var['endogenous_peaks'] = np.random.rand(atac_adata.shape[1]) <= min(1e5/atac_adata.shape[1], 1)
        
        logging.info(f"    - Number of Cells (filtered): {atac_adata.shape[0]}")
        logging.info(f"    - Number of Peaks (filtered): {atac_adata.shape[1]-1}")
        
        if h5ad_save_path:
            logging.info(f"    Writing h5ad file to {os.path.basename(h5ad_save_path)}")
            atac_adata.write_h5ad(h5ad_save_path)
    
        return atac_adata
    
    else:
        logging.info(f"  - Loading existing h5ad file found at {h5ad_save_path}")
        return anndata.read_h5ad(h5ad_save_path)


def rna_data_preprocessing(
    rna_data_path: str, 
    rna_h5ad_save_path: str,
    min_cells_per_gene: int = 15,
    target_read_depth: float = 1e6, 
    min_gene_disp: float = 0.5,
    min_genes: int = 200,
    max_genes: int = 2500,
    max_pct_mt: float = 5.0,
    overwrite: bool = False
    ) -> anndata.AnnData:
    """
    Runs QC filtering and preprocessing for scRNA-seq data.
    
    Args:
        rna_data_path (str): 
            Path to the raw, unprocessed scRNA-seq gene x cell count matrix.
        rna_h5ad_save_path (str):
            Path to save the processed and filtered scRNA-seq AnnData object.
        min_cells_per_gene (int):
            Genes must be expressed in at least this number of cells. Defaults to 15.
        target_read_depth (float, optional): 
            Normalizes the read depth of each cell. Defaults to 1e6.
        min_gene_disp (float, optional): 
            Minimum gene variability by dispersion. Defaults to 0.5.
        min_genes (int, optional):
            Cells must be expressing at least this number of genes. Defaults to 200.
        max_genes (int, optional):
            Cells cannot be expressing over this number of genes. Defaults to 2500.
        max_pct_mt (float, optional):
            Cells cannot be expressing over this percentage of mitochondrial genes. Defaults to 5%.
        h5ad_save_path (None | str): 
            Path to save the processed RNA AnnData object as an h5 file.
        overwrite (bool):
            Set to True to overwrite the h5ad file if it exists. Defaults to False.

    Returns:
        rna_adata (anndata.AnnData): Filtered RNA AnnData object
    """
    file_missing = not os.path.isfile(rna_h5ad_save_path)
    if file_missing or overwrite:
        logging.info("  - Reading RNAseq raw data parquet file")
        rna_data = load_rna_dataset(rna_data_path)
        logging.info(f"  - Number of Cells (unfiltered): {rna_data.shape[0]}")
        logging.info(f"  - Number of Genes (unfiltered): {rna_data.shape[1]}")

        logging.info("  - Converting DataFrame to AnnData object")
        rna_adata = anndata_from_dataframe(rna_data, "gene_id")
        logging.info("  - QC Filtering and Preprocessing:")
        logging.info("      (1/6) Filtering out mitochondrial, ribosomal, and hemoglobin genes")
        #    Mitochondrial: genes whose name starts with "MT-" (case-insensitive).
        #    Ribosomal:     genes whose name starts with "RPS" or "RPL".
        #    Hemoglobin:    genes whose name matches "^HB(?!P)" (so HB* but not HBP).
        var_names_lower = rna_adata.var_names.str.lower()

        rna_adata.var["mt"] = var_names_lower.str.startswith("mt-")
        rna_adata.var["ribo"] = var_names_lower.str.startswith(("rps", "rpl"))
        rna_adata.var["hb"] = var_names_lower.str.contains(r"^hb(?!p)")
        
        sc.pp.calculate_qc_metrics(
            rna_adata,
            qc_vars=["mt", "ribo", "hb"],
            percent_top=None,
            log1p=False,
            inplace=True,
        )
        
        cell_mask = (
            (rna_adata.obs["n_genes_by_counts"] >= min_genes)
            & (rna_adata.obs["n_genes_by_counts"] <= max_genes)
            & (rna_adata.obs["pct_counts_mt"] < max_pct_mt)
        )

        n_before = rna_adata.n_obs
        n_after = cell_mask.sum()
        logging.info(f"      (2/6) Filtering genes expressed in fewer than {min_cells_per_gene} cells (after cell filtering)")

        if n_after == 0:
            raise RuntimeError("No cells passed the filtering criteria. "
                            "Check `min_genes`, `max_genes`, `max_pct_mt` settings.")

        filtered_adata = rna_adata[cell_mask].copy()
        
        logging.info("      (3/6) Filtering out very rare genes")
        sc.pp.filter_genes(filtered_adata, min_cells=min_cells_per_gene)
        rawdata = filtered_adata.X.copy()
        
        logging.info(f"      (4/6) Normalizing to a read depth of {target_read_depth}")
        sc.pp.normalize_total(filtered_adata, target_sum=target_read_depth)

        logging.info("      (5/6) Log1p normalizing the data")
        sc.pp.log1p(filtered_adata)

        logging.info(f"      (6/6) Filtering for highly variable genes with dispersion > {min_gene_disp}")
        sc.pp.highly_variable_genes(filtered_adata, min_disp = min_gene_disp)
        
        logging.info(f"  - Number of Cells (filtered): {filtered_adata.shape[1]}")
        logging.info(f"  - Number of Genes (filtered): {filtered_adata.shape[0]}")

        filtered_adata.layers['counts'] = rawdata

        if rna_h5ad_save_path:
            logging.info(f"\nWriting h5ad file to {os.path.basename(rna_h5ad_save_path)}")
            filtered_adata.write_h5ad(rna_h5ad_save_path)
        
        return filtered_adata
                
    else:
        logging.info("  - RNA h5ad file found, loading")
        return anndata.read_h5ad(rna_h5ad_save_path)


def filter_atac_by_distance_to_tss(
    atac_df: pd.DataFrame, 
    gene_names: Union[list[str], set[str]],
    species: str,
    tss_distance_cutoff: Union[int, float],
    output_dir: str,
    fig_dir: str,
    ) -> anndata.AnnData:
    """
    Measures the distance between each peaks and gene transcription start sites within tss_distance_cutoff base pairs of each peak.
    
    Filters the scATAC-seq data to only include peaks within `tss_distance_cutoff` base pairs of a gene
    in the scRNA-seq datset. Saves the TSS distances and TSS distance scores to the `output_dir` directory
    as `tss_distance_score.parquet`.
    
    If the h5ad file exists at `atac_h5ad_save_path`, the function returns the ATAC AnnData object. 
    Writes the procesed data to the same location as `atac_data_path` but with the file ending in
    `_processed.parquet` if it does not exist.

    Args:
        atac_df (pd.DataFrame): 
            Processed scATAC-seq gene x cell DataFrame. Must contain the column "peak_id" with peak locations
            in the format of "chr:start-end".
        gene_names (list[str] | set[str]):
            List of gene names from the scRNA-seq dataset.
        species (str):
            Ensembl species name for loading gene TSS locations ('hsapiens' or 'mmusculus').
        tss_distance_cutoff (int | float):
            Maximum distance in base pairs to filter when associating peaks to potential target genes by distance.
        output_dir (str):
            Path to the output directory for the sample.
        fig_dir (str): 
            Path to the figure directory for the sample.

    Raises:
        Exception: barcodes argument must contain cell names / barcodes.
        FileNotFoundError: ATAC data file path must exist.

    Returns:
        pd.DataFrame: Returns atac_df filtered to exclude peaks further than 1 MB from a gene TSS
    """
    
    if not os.path.isdir(output_dir):
        raise Exception(f"Output directory {output_dir} does not exist")
        
    logging.info("    - Extracting ATAC peaks within 1 MB of a gene from the RNA dataset")
    gene_names_set = set(gene_names)

    peaks_near_genes_df = extract_atac_peaks_near_rna_genes(
        atac_df, 
        gene_names_set, 
        species, 
        tss_distance_cutoff, 
        output_dir
        )
    
    plot_feature_score_histogram(peaks_near_genes_df, "TSS_dist_score", fig_dir)
    
    logging.info("    - Filtering for peaks with 1MB of a gene's TSS")
    peak_subset = set(peaks_near_genes_df["peak_id"])
    atac_df_filtered = atac_df[atac_df["peak_id"].isin(peak_subset)]
    logging.info(f'    - Number of peaks after filtering: {len(atac_df_filtered)} / {len(atac_df)}')
    
    return atac_df_filtered

def main(args):
    
    overwrite_files = False
    if args.overwrite_files.lower() == "true":
        overwrite_files = True
        logging.info("\nOverwrite files argument set to 'True', processing parquet files")
    
    rna_h5ad_save_file = os.path.splitext(args.rna_data_file)[0] + ".h5ad"
    atac_h5ad_save_file = os.path.splitext(args.atac_data_file)[0] + ".h5ad"

    # ------ RNA and ATAC data preprocessing ------
    logging.info("\nLoading and processing the scRNA-seq data")
    rna_adata_processed = rna_data_preprocessing(
        args.rna_data_file, 
        rna_h5ad_save_file,
        min_cells_per_gene = 15,
        target_read_depth = 1e6, 
        min_gene_disp = 0.5,
        min_genes = 200,
        max_genes = 2500,
        max_pct_mt = 5.0,
        overwrite=overwrite_files
    )

    rna_df = convert_anndata_to_pandas(rna_adata_processed, "gene_id")
    write_processed_dataframe_to_parquet(rna_df, args.rna_data_file)

    barcodes = rna_adata_processed.obs_names.to_list()
    gene_names = rna_adata_processed.var_names.to_list()

    logging.info("\nRunning ATAC preprocessing")
    atac_adata_processed = atac_data_preprocessing(
        args.atac_data_file,
        barcodes,
        gene_names,
        filter_peak_min_cells=30,
        min_peaks_per_cell=1000,
        target_read_depth=1e6,
        tss_distance_cutoff=args.tss_distance_cutoff,
        fig_dir=args.fig_dir,
        dataset_dir=args.output_dir,
        ensembl_species=args.species,
        plot_peaks_by_counts=True,
        h5ad_save_path=atac_h5ad_save_file,
        overwrite=overwrite_files
    )
    logging.info(f"  - Pre-processed ATAC shape: {atac_adata_processed.shape}")

    atac_df = convert_anndata_to_pandas(atac_adata_processed, "peak_id")
    write_processed_dataframe_to_parquet(atac_df, args.atac_data_file)

    logging.info(f"  - ATAC filtered by TSS shape: {atac_adata_processed.shape}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args()
    
    main(args)
