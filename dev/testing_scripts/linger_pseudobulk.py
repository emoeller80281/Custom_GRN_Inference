import numpy as np
import pandas as pd
import scanpy as sc
import random
import os
import warnings
import sys
import argparse
import scipy.sparse as sp
import logging
from anndata import AnnData

from grn_inference import utils

def tfidf(atac_matrix: np.ndarray) -> np.ndarray:
    """
    Performs a TF-IDF-like transformation on the ATAC-seq matrix to highlight important regulatory elements.

    Parameters:
        atac_matrix (np.ndarray):
            A matrix of ATAC-seq data, where rows are regulatory elements (peaks) and columns are cells. 
            Values represent the accessibility of the peaks in each cell.

    Returns:
        transformed_matrix (np.ndarray):
            Transformed matrix where rows represent regulatory elements (peaks) and columns represent cells,
            with values weighted by TF-IDF-like scores.
    """

    # Create a binary matrix indicating presence/absence of peaks
    binary_matrix: np.ndarray = 1 * (atac_matrix > 0)
    
    # Calculate term frequency (TF) normalized by log of total accessibility per cell
    term_freq: np.ndarray = binary_matrix / (np.ones((binary_matrix.shape[0], 1)) * np.log(1 + np.sum(binary_matrix, axis=0))[np.newaxis, :])
    
    # Calculate inverse document frequency (IDF) based on peak occurrence across cells
    inverse_doc_freq: np.ndarray = np.log(1 + binary_matrix.shape[1] / (1 + np.sum(binary_matrix > 0, axis=1)))
    
    # Compute the TF-IDF-like matrix
    tfidf_matrix: np.ndarray = term_freq * (inverse_doc_freq[:, np.newaxis] * np.ones((1, binary_matrix.shape[1])))
    
    # Replace any NaN values with 0 (due to division by zero)
    tfidf_matrix[np.isnan(tfidf_matrix)] = 0
    
    # Return the transposed matrix (cells as rows, peaks as columns)
    transformed_matrix: np.ndarray = tfidf_matrix.T
    return transformed_matrix

def find_neighbors(rna_data: AnnData, atac_data: AnnData) -> tuple[AnnData, AnnData]:
    """
    Combines RNA and ATAC-seq data in a joint PCA space and identifies neighbors based on combined features.

    Parameters:
        rna_data (AnnData):
            AnnData object containing RNA expression data.
        atac_data (AnnData):
            AnnData object containing ATAC-seq data.

    Returns:
        tuple (AnnData, AnnData):
            Updated `rna_data` and `atac_data` objects with combined PCA representation.
    """
    neighbors_k: int = 20  # Number of neighbors to find
    
    ### RNA Data Preprocessing ###
    # Normalize RNA expression data and log-transform
    sc.pp.normalize_total(rna_data, target_sum=1e4)
    sc.pp.log1p(rna_data)
    
    # Identify highly variable genes
    sc.pp.highly_variable_genes(rna_data, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Save raw data and subset highly variable genes
    rna_data.raw = rna_data
    rna_data = rna_data[:, rna_data.var.highly_variable]
    
    # Scale the data and perform PCA for dimensionality reduction
    sc.pp.scale(rna_data, max_value=10)
    sc.tl.pca(rna_data, n_comps=15, svd_solver="arpack")
    
    # Store the PCA results for RNA
    pca_rna: np.ndarray = rna_data.obsm['X_pca']
    
    ### ATAC Data Preprocessing ###
    # Log-transform ATAC-seq data
    sc.pp.log1p(atac_data)
    
    # Identify highly variable peaks
    sc.pp.highly_variable_genes(atac_data, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Save raw ATAC data and subset highly variable peaks
    atac_data.raw = atac_data
    atac_data = atac_data[:, atac_data.var.highly_variable]
    
    # Scale the ATAC data and perform PCA
    sc.pp.scale(atac_data, max_value=10, zero_center=True)
    sc.tl.pca(atac_data, n_comps=15, svd_solver="arpack")
    
    # Store the PCA results for ATAC
    pca_atac: np.ndarray = atac_data.obsm['X_pca']
    
    ### Combine RNA and ATAC PCA Results ###
    combined_pca: np.ndarray = np.concatenate((pca_rna, pca_atac), axis=1)
    
    # Store the combined PCA representation in both AnnData objects
    rna_data.obsm['pca'] = combined_pca
    atac_data.obsm['pca'] = combined_pca

    return rna_data, atac_data

def pseudo_bulk(rna_data: AnnData, atac_data: AnnData, single_pseudo_bulk: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates pseudo-bulk RNA and ATAC profiles by aggregating cells with similar profiles based on neighbors.

    Parameters:
        rna_data (AnnData):
            AnnData object containing RNA expression data.
        atac_data (AnnData):
            AnnData object containing ATAC-seq data.
        single_pseudo_bulk (int):
            If set to a value greater than 0, limits each cluster to 1 sample for pseudo-bulk creation.

    Returns:
        tuple (pd.DataFrame, pd.DataFrame):
            pseudo_bulk_rna : pd.DataFrame
                Pseudo-bulk RNA expression matrix (genes x pseudo-bulk samples).
            pseudo_bulk_atac : pd.DataFrame
                Pseudo-bulk ATAC accessibility matrix (peaks x pseudo-bulk samples).
    """
    neighbors_k: int = 20  # Number of neighbors to use for aggregation

    ### RNA Data Preprocessing ###
    sc.pp.normalize_total(rna_data, target_sum=1e4)
    sc.pp.log1p(rna_data)
    sc.pp.filter_genes(rna_data, min_cells=3)
    sc.pp.highly_variable_genes(rna_data, min_mean=0.0125, max_mean=3, min_disp=0.5)
    rna_data.raw = rna_data
    rna_data = rna_data[:, rna_data.var.highly_variable]
    sc.pp.scale(rna_data, max_value=10)
    sc.tl.pca(rna_data, n_comps=15, svd_solver="arpack")
    
    ### ATAC Data Preprocessing ###
    sc.pp.log1p(atac_data)
    sc.pp.filter_genes(atac_data, min_cells=3)
    sc.pp.highly_variable_genes(atac_data, min_mean=0.0125, max_mean=3, min_disp=0.5)
    atac_data.raw = atac_data
    atac_data = atac_data[:, atac_data.var.highly_variable]
    sc.pp.scale(atac_data, max_value=10, zero_center=True)
    sc.tl.pca(atac_data, n_comps=15, svd_solver="arpack")
    
    ### Combine RNA and ATAC PCA Results ###
    pca_rna: np.ndarray = rna_data.obsm['X_pca']
    pca_atac: np.ndarray = atac_data.obsm['X_pca']
    combined_pca: np.ndarray = np.concatenate((pca_rna, pca_atac), axis=1)
    rna_data.obsm['pca'] = combined_pca
    atac_data.obsm['pca'] = combined_pca
    
    ### Neighbor Graph Construction ###
    sc.pp.neighbors(rna_data, n_neighbors=neighbors_k, n_pcs=30, use_rep='pca')
    connectivity_matrix: np.ndarray = (rna_data.obsp['distances'] > 0)
    
    ### Label Processing and Pseudo-bulk Generation ###
    cell_labels: pd.DataFrame = pd.DataFrame(rna_data.obs['label'])
    cell_labels.index = rna_data.obs_names.tolist()
    
    # Identify unique clusters of cells
    unique_clusters: list = list(set(cell_labels['label'].values))
    selected_indices: list = []
    
    np.random.seed(42)  # Set seed for reproducibility
    
    for cluster_label in unique_clusters:
        cluster_indices: pd.Index = cell_labels.index
        num_cells_in_cluster: int = len(cluster_indices)  # Total number of elements in the cluster
        
        if num_cells_in_cluster >= 10:
            sample_size: int = int(np.floor(np.sqrt(num_cells_in_cluster))) + 1  # Number of elements to sample
            
            if single_pseudo_bulk > 0:
                sample_size = 1  # If single_pseudo_bulk is greater than 0, limit to 1 sample
                
            sampled_elements = random.sample(range(num_cells_in_cluster), sample_size)
            cluster_indices = cluster_indices[sampled_elements]
            selected_indices += cluster_indices.tolist()
    
    ### Aggregating RNA and ATAC Profiles ###
    connectivity_df: pd.DataFrame = pd.DataFrame(connectivity_matrix.toarray(), index=rna_data.obs_names.tolist())
    selected_connectivity_matrix: np.ndarray = connectivity_df.loc[selected_indices].values
    
    # Aggregate RNA expression and ATAC accessibility
    aggregated_rna: np.ndarray = (selected_connectivity_matrix @ rna_data.raw.X.toarray())
    pseudo_bulk_rna: pd.DataFrame = pd.DataFrame(
        (aggregated_rna / (neighbors_k - 1)).T, 
        columns=selected_indices, 
        index=rna_data.raw.var['gene_ids'].tolist())
    
    aggregated_atac: np.ndarray = (selected_connectivity_matrix @ atac_data.raw.X.toarray())
    pseudo_bulk_atac: pd.DataFrame = pd.DataFrame(
        (aggregated_atac / (neighbors_k - 1)).T, 
        columns=selected_indices, 
        index=atac_data.raw.var['gene_ids'].tolist())
    
    return pseudo_bulk_rna, pseudo_bulk_atac

atac_data_filename = "mESC_filtered_L2_E7.5_rep1_ATAC.h5ad"
rna_data_filename = "mESC_filtered_L2_E7.5_rep1_RNA.h5ad"

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
MM10_GENOME_DIR = os.path.join(PROJECT_DIR, "data/reference_genome/mm10")
MM10_GENE_TSS_FILE = os.path.join(PROJECT_DIR, "data/genome_annotation/mm10/mm10_TSS.bed")
GROUND_TRUTH_DIR = os.path.join(PROJECT_DIR, "ground_truth_files")
SAMPLE_INPUT_DIR = os.path.join(PROJECT_DIR, "input/mESC/filtered_L2_E7.5_rep1")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output/transformer_testing_output")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Filter warnings about copying objects from AnnData
warnings.filterwarnings("ignore", message="Received a view of an AnnData. Making a copy.")
warnings.filterwarnings("ignore", message="Trying to modify attribute `.obs` of view, initializing view as actual.")

adata_RNA = sc.read_h5ad(os.path.join(SAMPLE_INPUT_DIR, rna_data_filename))
adata_ATAC = sc.read_h5ad(os.path.join(SAMPLE_INPUT_DIR, atac_data_filename))

logging.info(f'\tscRNAseq Dataset: {adata_RNA.shape[1]} genes, {adata_RNA.shape[0]} cells')
logging.info(f'\tscATACseq Dataset: {adata_ATAC.shape[1]} peaks, {adata_ATAC.shape[0]} cells')

# Remove low count cells and genes
logging.info('\nFiltering Data')
logging.info(f'\tFiltering out cells with less than 200 genes...')
sc.pp.filter_cells(adata_RNA, min_genes=200)
adata_RNA = adata_RNA.copy()
logging.info(f'\t\tShape of the RNA dataset = {adata_RNA.shape[1]} genes, {adata_RNA.shape[0]} cells')

logging.info(f'\tFiltering out genes expressed in fewer than 3 cells...')
sc.pp.filter_genes(adata_RNA, min_cells=3)
adata_RNA = adata_RNA.copy()
logging.info(f'\t\tShape of the RNA dataset = {adata_RNA.shape[1]} genes, {adata_RNA.shape[0]} cells')

logging.info(f'\tFiltering out cells with less than 200 ATAC-seq peaks...')
sc.pp.filter_cells(adata_ATAC, min_genes=200)
adata_ATAC = adata_ATAC.copy()
logging.info(f'\t\tShape of the ATAC dataset = {adata_ATAC.shape[1]} peaks, {adata_ATAC.shape[0]} cells')

logging.info(f'\tFiltering out peaks expressed in fewer than 3 cells...')
sc.pp.filter_genes(adata_ATAC, min_cells=3)
adata_ATAC = adata_ATAC.copy()
logging.info(f'\t\tShape of the ATAC dataset = {adata_ATAC.shape[1]} peaks, {adata_ATAC.shape[0]} cells')

logging.info('\nShape of the dataset after filtering')
logging.info(f'\tscRNAseq Dataset: {adata_RNA.shape[1]} genes, {adata_RNA.shape[0]} cells')
logging.info(f'\tscATACseq Dataset: {adata_ATAC.shape[1]} peaks, {adata_ATAC.shape[0]} cells')

logging.info(f'\nCombining RNA and ATAC seq barcodes')
selected_barcode = list(set(adata_RNA.obs_names) & set(adata_ATAC.obs_names))

rna_barcode_idx = pd.DataFrame(range(adata_RNA.shape[0]), index=adata_RNA.obs_names)
atac_barcode_idx = pd.DataFrame(range(adata_ATAC.shape[0]), index=adata_ATAC.obs_names)

adata_RNA = adata_RNA[rna_barcode_idx.loc[selected_barcode][0]].copy()
adata_ATAC = adata_ATAC[atac_barcode_idx.loc[selected_barcode][0]].copy()

logging.info(f'\nGenerating pseudo-bulk / metacells')
samplelist = list(set(adata_ATAC.obs['sample'].values))
tempsample = samplelist[0]

TG_pseudobulk = pd.DataFrame([])
RE_pseudobulk = pd.DataFrame([])

singlepseudobulk = (adata_RNA.obs['sample'].unique().shape[0] * adata_RNA.obs['sample'].unique().shape[0] > 100)

for tempsample in samplelist:
    adata_RNAtemp = adata_RNA[adata_RNA.obs['sample'] == tempsample].copy()
    adata_ATACtemp = adata_ATAC[adata_ATAC.obs['sample'] == tempsample].copy()

    TG_pseudobulk_temp, RE_pseudobulk_temp = pseudo_bulk(adata_RNAtemp, adata_ATACtemp, singlepseudobulk)

    TG_pseudobulk = pd.concat([TG_pseudobulk, TG_pseudobulk_temp], axis=1)
    RE_pseudobulk = pd.concat([RE_pseudobulk, RE_pseudobulk_temp], axis=1)

    RE_pseudobulk[RE_pseudobulk > 100] = 100

logging.info(f'Writing adata_ATAC.h5ad and adata_RNA.h5ad')
adata_ATAC.write_h5ad(os.path.join(OUTPUT_DIR, 'adata_ATAC.h5ad'))
adata_RNA.write_h5ad(os.path.join(OUTPUT_DIR, 'adata_RNA.h5ad'))

TG_pseudobulk = TG_pseudobulk.fillna(0)
RE_pseudobulk = RE_pseudobulk.fillna(0)

logging.info(f'Writing out peak gene ids')
pd.DataFrame(adata_ATAC.var['gene_ids']).to_csv(os.path.join(OUTPUT_DIR, 'Peaks.txt'), header=None, index=None)

logging.info(f'Writing out pseudobulk...')
TG_pseudobulk.to_csv(os.path.join(OUTPUT_DIR, 'TG_pseudobulk.tsv'), sep='\t', index=True)
RE_pseudobulk.to_csv(os.path.join(OUTPUT_DIR, 'RE_pseudobulk.tsv'), sep='\t', index=True)