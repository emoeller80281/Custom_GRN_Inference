import os
import re
import json
import glob
import pandas as pd
import scanpy as sc
import logging
from pathlib import Path
import warnings
import numpy as np
import scipy.sparse as sp
from typing import Tuple
from anndata import AnnData
from tqdm import tqdm
import pybedtools

from multiomic_transformer.data.moods_scan import run_moods_scan_batched
from multiomic_transformer.utils.standardize import standardize_name
from multiomic_transformer.utils.files import atomic_json_dump
from multiomic_transformer.utils.peaks import find_genes_near_peaks
from config.settings import *


def filter_and_qc(adata_RNA: AnnData, adata_ATAC: AnnData) -> Tuple[AnnData, AnnData]:
    
    adata_RNA = adata_RNA.copy()
    adata_ATAC = adata_ATAC.copy()
    
    logging.info(f"[START] RNA shape={adata_RNA.shape}, ATAC shape={adata_ATAC.shape}")

    
    # Synchronize barcodes
    adata_RNA.obs['barcode'] = adata_RNA.obs_names
    adata_ATAC.obs['barcode'] = adata_ATAC.obs_names

    common_barcodes = adata_RNA.obs['barcode'].isin(adata_ATAC.obs['barcode'])
    n_before = (adata_RNA.n_obs, adata_ATAC.n_obs)
    adata_RNA = adata_RNA[common_barcodes].copy()
    adata_ATAC = adata_ATAC[adata_ATAC.obs['barcode'].isin(adata_RNA.obs['barcode'])].copy()
    
    logging.info(
        f"[BARCODES] before sync RNA={n_before[0]}, ATAC={n_before[1]} → after sync RNA={adata_RNA.n_obs}, ATAC={adata_ATAC.n_obs}"
    )
    
    # --- QC and filtering ---
    
    adata_RNA.var['mt'] = adata_RNA.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata_RNA, qc_vars=["mt"], inplace=True)
    adata_RNA = adata_RNA[adata_RNA.obs.pct_counts_mt < 5].copy()
    adata_RNA.var_names_make_unique()
    adata_RNA.var['gene_ids'] = adata_RNA.var.index
        
    sc.pp.filter_cells(adata_RNA, min_genes=200)
    sc.pp.filter_genes(adata_RNA, min_cells=3)
    sc.pp.filter_cells(adata_ATAC, min_genes=200)
    sc.pp.filter_genes(adata_ATAC, min_cells=3)
    
    # --- Preprocess RNA ---
    sc.pp.normalize_total(adata_RNA, target_sum=1e4)
    sc.pp.log1p(adata_RNA)
    sc.pp.highly_variable_genes(adata_RNA, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable]
    sc.pp.scale(adata_RNA, max_value=10)
    sc.tl.pca(adata_RNA, n_comps=25, svd_solver="arpack")

    # --- Preprocess ATAC ---
    sc.pp.log1p(adata_ATAC)
    sc.pp.highly_variable_genes(adata_ATAC, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable]
    sc.pp.scale(adata_ATAC, max_value=10, zero_center=True)
    sc.tl.pca(adata_ATAC, n_comps=25, svd_solver="arpack")
    
    # --- After filtering to common barcodes ---
    common_barcodes = adata_RNA.obs_names.intersection(adata_ATAC.obs_names)
    
    adata_RNA = adata_RNA[common_barcodes].copy()
    adata_ATAC = adata_ATAC[common_barcodes].copy()
    
    return adata_RNA, adata_ATAC

def create_tf_tg_combination_files(genes):
    mouse_tf_df = pd.read_excel(DATA_DIR / "mouse_tfs" / "mouse_tf_list.xlsx", header=6, index_col=None)
    overlap_tf = mouse_tf_df[mouse_tf_df["Gene Symbol"].isin(genes)].drop_duplicates()

    tfs = overlap_tf["Gene Symbol"].to_list()
    tgs = [gene for gene in genes if gene not in tfs]

    print(f"Number of TFs: {len(tfs):,}")
    print(f"Number of TGs: {len(tgs):,}")

    mux = pd.MultiIndex.from_product([tfs, tgs], names=["TF", "TG"])
    tf_tg_df = mux.to_frame(index=False)
    
    print(f"TF-TG combinations: {len(tf_tg_df):,}")

    tf_tg_outdir = DATA_DIR / "tf_tg_combos"
    os.makedirs(tf_tg_outdir, exist_ok=True)
    
    if not os.path.isfile(tf_tg_outdir / "tg_list.csv"):
        print("Writing TF list to 'tg_list.csv'")
        with open(tf_tg_outdir / "tg_list.csv", 'w') as tg_list_file:
            tg_list_file.write("TG\n")
            for i in tgs:
                tg_list_file.write(f"{i}\n")
    
    if not os.path.isfile(tf_tg_outdir / "tf_list.csv"):
        print("Writing TF list to 'tf_list.csv'")
        with open(tf_tg_outdir / "tf_list.csv", "w") as tf_list_file:
            tf_list_file.write("TF\n")
            for i in tfs:
                tf_list_file.write(f"{i}\n")
    
    if not os.path.isfile(tf_tg_outdir / "tf_tg_combos.csv"):
        print("Writing combinations to 'tf_tg_combos.csv'")
        tf_tg_df.to_csv(tf_tg_outdir / "tf_tg_combos.csv", header=True, index=False)
        
    print("Done!")

    return tfs, tgs, tf_tg_df

def make_chrom_gene_tss_df(gene_tss_file, genome_dir):
    gene_tss_bed = pybedtools.BedTool(gene_tss_file)
    gene_tss_df = (
        gene_tss_bed
        .saveas(os.path.join(genome_dir, f"gene_tss.bed"))
        .to_dataframe()
        .sort_values(by="start", ascending=True)
        )
    return gene_tss_df

def build_global_tg_vocab(gene_tss_file, vocab_file):
    """
    Build or update a global TG vocab from all genes in the genome.
    """
    # Load existing vocab if present
    if os.path.isfile(vocab_file):
        with open(vocab_file) as f:
            vocab = json.load(f)
    else:
        vocab = {}

    # Load all gene names genome-wide
    gene_tss_bed = pybedtools.BedTool(gene_tss_file)
    gene_tss_df = (
        gene_tss_bed
        .to_dataframe()
        .sort_values(by="start", ascending=True)
        )
    all_genes = [standardize_name(n) for n in gene_tss_df["name"].unique()]

    updated = False
    for name in sorted(set(all_genes)):
        if name not in vocab:
            vocab[name] = len(vocab)
            updated = True

    if updated:
        atomic_json_dump(vocab, vocab_file)
        logging.info(f"Updated TG vocab: {len(vocab)} genes")

    return vocab

def build_peak_locs_from_index(peak_index) -> pd.DataFrame:
    rows = []
    for pid in map(str, peak_index):
        m = re.match(r"^(chr)?(\w+)[_:](\d+)[-_:](\d+)$", pid)
        if not m:
            raise ValueError(f"Cannot parse peak id: {pid!r}; expected formats like 'chr1:100-200'")
        _, chrom_core, start, end = m.groups()
        chrom = f"chr{chrom_core}" if not chrom_core.startswith("chr") else chrom_core
        start, end = int(start), int(end)
        if start > end:
            start, end = end, start
        rows.append((chrom, start, end, pid))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "peak_id"])

def calculate_peak_to_tg_distance_score(
    peak_bed_file,
    tss_bed_file,
    peak_gene_dist_file,
    mesc_atac_peak_loc_df, 
    gene_tss_df, 
    max_peak_distance=1e6, 
    distance_factor_scale=25000, 
    force_recalculate=False
    ) -> pd.DataFrame:
    if not os.path.isfile(peak_gene_dist_file) or force_recalculate:
        if not os.path.isfile(peak_bed_file) or not os.path.isfile(tss_bed_file) or force_recalculate:
        
            logging.info("Calculating peak to TG distance score")
            peak_bed = pybedtools.BedTool.from_dataframe(
                mesc_atac_peak_loc_df[["chrom", "start", "end", "peak_id"]]
                ).saveas(peak_bed_file)

            tss_bed = pybedtools.BedTool.from_dataframe(
                gene_tss_df[["chrom", "start", "end", "name"]]
                ).saveas(tss_bed_file)
            
        peak_bed = pybedtools.BedTool(peak_bed_file)
        tss_bed = pybedtools.BedTool(tss_bed_file)
    
        genes_near_peaks = find_genes_near_peaks(peak_bed, tss_bed, tss_distance_cutoff=max_peak_distance)

        # Restrict to peaks within 1 Mb of a gene TSS
        genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= max_peak_distance]

        # Scale the TSS distance score by the exponential scaling factor
        genes_near_peaks = genes_near_peaks.copy()
        genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / distance_factor_scale)

        genes_near_peaks.to_parquet(peak_gene_dist_file, compression="snappy", engine="pyarrow")
    else:
        genes_near_peaks = pd.read_parquet(peak_gene_dist_file, engine="pyarrow")
    
    return genes_near_peaks

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rna_file = RAW_DATA / "DS012" / "scRNA_seq.csv"
    atac_file = RAW_DATA / "DS012" / "peakMatrix.csv"

    rna_df = pd.read_csv(rna_file, delimiter="\t", header=0, index_col=0)
    atac_df = pd.read_csv(atac_file, delimiter="\t", header=0, index_col=0)

    # Temporarily rename columns, the barcodes in my test don't line up
    rna_df.columns = [f"cell_{i}" for i in range(1, rna_df.shape[1] + 1)]
    atac_df.columns = [f"cell_{i}" for i in range(1, atac_df.shape[1] + 1)]

    genes = rna_df.index.to_list()
    peaks = atac_df.index.to_list()

    print(genes[:5])
    print(peaks[:5])

    tfs, tgs, tf_tg_df = create_tf_tg_combination_files(genes)

    adata_rna = AnnData(rna_df.T)
    adata_atac = AnnData(atac_df.T)
    
    adata_rna_filtered, adata_atac_filtered = filter_and_qc(adata_rna, adata_atac)
    print(adata_rna_filtered.shape)

    if not os.path.isdir(GENOME_DIR):
        os.makedirs(GENOME_DIR)

    gene_tss_df = make_chrom_gene_tss_df(
        gene_tss_file=GENE_TSS_FILE,
        genome_dir=GENOME_DIR
    )

    processed_rna_df = pd.DataFrame(
        adata_rna_filtered.X,
        index=adata_rna_filtered.obs_names,
        columns=adata_rna_filtered.var_names,
    ).T
    processed_atac_df = pd.DataFrame(
        adata_atac_filtered.X,
        index=adata_atac_filtered.obs_names,
        columns=adata_atac_filtered.var_names,
    ).T

    print("Processed RNA df")
    print(processed_rna_df.head())

    print("\nProcessed ATAC df")
    print(processed_atac_df.head())

    # Build peak locations from original ATAC peak names
    peak_locs_df = build_peak_locs_from_index(atac_df.index)

    peak_bed_file = DATA_DIR / "processed" / "peaks.bed"
    peak_to_gene_dist_df = calculate_peak_to_tg_distance_score(
        peak_bed_file=peak_bed_file,
        tss_bed_file= DATA_DIR / "genome_data" / "genome_annotation" / "mm10" / "gene_tss.bed",
        peak_gene_dist_file= DATA_DIR / "processed" / "peak_to_gene_dist.parquet",
        mesc_atac_peak_loc_df=peak_locs_df,
        gene_tss_df= gene_tss_df,
    )
    print("Peak to Gene Distance DataFrame")
    print(peak_to_gene_dist_df.head())

    print("\nTF-TG Combinations")
    print(tf_tg_df.head())

    print("Running MOODS TF-peak binding calculation")
    jaspar_pfm_paths = DATA_DIR / "motif_information" / "mm10" / "JASPAR" / "pfm_files"
    motif_paths = list(jaspar_pfm_paths.glob("*.pfm"))
    moods_sites_file = DATA_DIR / "processed" / "moods_sites.tsv"
    
    peaks_bed_path = Path(peak_bed_file)
    peaks_df = pybedtools.BedTool(peaks_bed_path)
    
    run_moods_scan_batched(
        peaks_bed=peaks_df, 
        fasta_path=os.path.join(GENOME_DIR, f"mm10.fa.gz"), 
        motif_paths=motif_paths, 
        out_tsv=moods_sites_file, 
        n_cpus=4,
        pval_threshold=MOODS_PVAL_THRESHOLD, 
        bg="auto"
    )




    
