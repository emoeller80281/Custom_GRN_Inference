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
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple
from anndata import AnnData
from tqdm import tqdm
import pybedtools
import argparse

import sys
sys.path.append(Path(__file__).resolve().parent.parent.parent)

from multiomic_transformer.data.moods_scan import run_moods_scan_batched
from multiomic_transformer.utils.standardize import standardize_name
from multiomic_transformer.utils.files import atomic_json_dump
from multiomic_transformer.utils.peaks import find_genes_near_peaks, format_peaks
from multiomic_transformer.utils.downloads import *
from multiomic_transformer.data.sliding_window import run_sliding_window_scan
# from config.settings import *

from grn_inference.utils import read_ground_truth


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
    
    # QC and filtering
    
    adata_RNA.var['mt'] = adata_RNA.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata_RNA, qc_vars=["mt"], inplace=True)
    adata_RNA = adata_RNA[adata_RNA.obs.pct_counts_mt < 5].copy()
    adata_RNA.var_names_make_unique()
    adata_RNA.var['gene_ids'] = adata_RNA.var.index
        
    sc.pp.filter_cells(adata_RNA, min_genes=200)
    sc.pp.filter_genes(adata_RNA, min_cells=3)
    sc.pp.filter_cells(adata_ATAC, min_genes=200)
    sc.pp.filter_genes(adata_ATAC, min_cells=3)
    
    # Preprocess RNA
    sc.pp.normalize_total(adata_RNA, target_sum=1e4)
    sc.pp.log1p(adata_RNA)
    sc.pp.highly_variable_genes(adata_RNA, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable]
    sc.pp.scale(adata_RNA, max_value=10)
    sc.tl.pca(adata_RNA, n_comps=25, svd_solver="arpack")

    # Preprocess ATAC
    sc.pp.log1p(adata_ATAC)
    sc.pp.highly_variable_genes(adata_ATAC, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable]
    sc.pp.scale(adata_ATAC, max_value=10, zero_center=True)
    sc.tl.pca(adata_ATAC, n_comps=25, svd_solver="arpack")
    
    # After filtering to common barcodes
    common_barcodes = adata_RNA.obs_names.intersection(adata_ATAC.obs_names)
    
    adata_RNA = adata_RNA[common_barcodes].copy()
    adata_ATAC = adata_ATAC[common_barcodes].copy()
    
    return adata_RNA, adata_ATAC

def create_tf_tg_combination_files(genes, tf_list_file):
    known_tfs = pd.read_csv(tf_list_file, sep="\t", header=0, index_col=None)
    overlap_tf = known_tfs[known_tfs["TF_Name"].isin(genes)].drop_duplicates()

    tfs = overlap_tf["TF_Name"].to_list()
    tgs = [gene for gene in genes if gene not in tfs]

    logging.info("\nCreating TF-TG combination files")
    logging.info(f"  - Number of TFs: {len(tfs):,}")
    logging.info(f"  - Number of TGs: {len(tgs):,}")

    mux = pd.MultiIndex.from_product([tfs, tgs], names=["TF", "TG"])
    tf_tg_df = mux.to_frame(index=False)
    
    logging.info(f"  - TF-TG combinations: {len(tf_tg_df):,}")

    tf_tg_outdir = DATA_DIR / "tf_tg_combos"
    os.makedirs(tf_tg_outdir, exist_ok=True)
    
    
    if not os.path.isfile(tf_tg_outdir / "total_genes.csv"):
        logging.info("Writing total genes to 'total_genes.csv'")
        with open(tf_tg_outdir / "total_genes.csv", 'w') as total_genes_file:
            total_genes_file.write("Gene\n")
            for i in genes:
                total_genes_file.write(f"{i}\n")
    
    if not os.path.isfile(tf_tg_outdir / "tg_list.csv"):
        logging.info("  Writing TF list to 'tg_list.csv'")
        with open(tf_tg_outdir / "tg_list.csv", 'w') as tg_list_file:
            tg_list_file.write("TG\n")
            for i in tgs:
                tg_list_file.write(f"{i}\n")
    
    if not os.path.isfile(tf_tg_outdir / "tf_list.csv"):
        logging.info("  Writing TF list to 'tf_list.csv'")
        with open(tf_tg_outdir / "tf_list.csv", "w") as tf_list_file:
            tf_list_file.write("TF\n")
            for i in tfs:
                tf_list_file.write(f"{i}\n")
    
    if not os.path.isfile(tf_tg_outdir / "tf_tg_combos.csv"):
        logging.info("  Writing combinations to 'tf_tg_combos.csv'")
        tf_tg_df.to_csv(tf_tg_outdir / "tf_tg_combos.csv", header=True, index=False)
        
    logging.info("Done!")

    return tfs, tgs, tf_tg_df

def make_chrom_gene_tss_df(gene_tss_file, genome_dir):
    gene_tss_bed = pybedtools.BedTool(gene_tss_file)
    gene_tss_df = (
        gene_tss_bed.to_dataframe(header=None, usecols=[0, 1, 2, 3])
        .rename(columns={0: "chrom", 1: "start", 2: "end", 3: "name"})
        .sort_values(by="start", ascending=True)
    )
    bed_path = os.path.join(genome_dir, "gene_tss.bed")
    gene_tss_df.to_csv(bed_path, sep="\t", header=False, index=False)
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

def build_peak_locs_from_index(peak_index: pd.Index) -> pd.DataFrame:
    """
    Parse peak IDs like 'chr1:100-200' into BED-format dataframe.
    Returns columns: chrom, start, end, peak_id
    """
    rows = []
    for pid in map(str, peak_index):
        m = re.match(r"^(chr)?(\w+)[_:](\d+)[-_:](\d+)$", pid)
        if not m:
            logging.warning(f"Skipping malformed peak ID: {pid}")
            continue
        _, chrom_core, start, end = m.groups()
        chrom = f"chr{chrom_core}" if not chrom_core.startswith("chr") else chrom_core
        start, end = int(start), int(end)
        if start > end:
            start, end = end, start
        rows.append((chrom, start, end, pid))

    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "peak_id"])
    df = df[df["chrom"].str.match(r"^chr[\dXYM]+$")]  # keep valid chromosomes only
    df = df.astype({"start": int, "end": int})
    return df

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
    """
    Compute peak-to-gene distance features (BEDTools-based), ensuring BED compliance.
    """
    # Validate and convert peaks to BED format
    required_cols = {"chrom", "start", "end", "peak_id"}
    if not required_cols.issubset(mesc_atac_peak_loc_df.columns):
        logging.warning("Converting peak index to BED format (chr/start/end parsing)")
        mesc_atac_peak_loc_df = build_peak_locs_from_index(mesc_atac_peak_loc_df.index)

    print("\nmesc_atac_peak_loc_df")
    print(mesc_atac_peak_loc_df.head())
    
    # Ensure numeric types
    mesc_atac_peak_loc_df["start"] = mesc_atac_peak_loc_df["start"].astype(int)
    mesc_atac_peak_loc_df["end"] = mesc_atac_peak_loc_df["end"].astype(int)

    # Ensure proper columns for gene_tss_df
    if not {"chrom", "start", "end", "name"}.issubset(gene_tss_df.columns):
        gene_tss_df = gene_tss_df.rename(columns={"chromosome_name": "chrom", "gene_start": "start", "gene_end": "end"})

    print("\gene_tss_df")
    print(gene_tss_df.head())
    
    # Step 1: Write valid BED files if missing
    if not os.path.isfile(peak_bed_file) or not os.path.isfile(tss_bed_file) or force_recalculate:
        logging.info("Writing BED files for peaks and gene TSSs")
        pybedtools.BedTool.from_dataframe(mesc_atac_peak_loc_df[["chrom", "start", "end", "peak_id"]]).saveas(peak_bed_file)
        pybedtools.BedTool.from_dataframe(gene_tss_df[["chrom", "start", "end", "name"]]).saveas(tss_bed_file)

    # Step 2: Run BEDTools overlap
    logging.info(f"Locating peaks within ±{max_peak_distance:,} bp of TSSs")
    peak_bed = pybedtools.BedTool(peak_bed_file)
    tss_bed = pybedtools.BedTool(tss_bed_file)

    genes_near_peaks = find_genes_near_peaks(peak_bed, tss_bed, tss_distance_cutoff=max_peak_distance)
    
    genes_near_peaks = genes_near_peaks.rename(columns={"gene_id": "target_id"})

    # Step 3: Compute distances and scores
    genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= max_peak_distance]
    genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / distance_factor_scale)

    # Step 4: Save and return
    genes_near_peaks.to_parquet(peak_gene_dist_file, compression="snappy", engine="pyarrow")
    logging.info(f"Saved peak–gene distance table: {genes_near_peaks.shape}")
    return genes_near_peaks


def process_single_tf(tf, tf_df, peak_to_gene_dist_df):
    # Compute softmax per peak (for this TF only)
    tf_df["sliding_window_tf_softmax"] = tf_df.groupby("peak_id")["sliding_window_score"].transform(softmax)

    merged = pd.merge(
        tf_df[["peak_id", "sliding_window_tf_softmax"]],
        peak_to_gene_dist_df,
        on="peak_id",
        how="inner"
    )
    merged["tf_tg_contrib"] = merged["sliding_window_tf_softmax"] * merged["TSS_dist_score"]

    tf_tg_reg_pot = (
        merged.groupby("target_id", as_index=False)
        .agg(reg_potential=("tf_tg_contrib", "sum"),
             motif_density=("peak_id", "nunique"))
        .rename(columns={"target_id": "TG"})
    )

    tf_tg_reg_pot["TF"] = tf
    return tf_tg_reg_pot

def calculate_tf_tg_regulatory_potential(sliding_window_score_file, tf_tg_reg_pot_file, num_cpu=8):
    logging.info("Calculating TF–TG regulatory potential (per TF mode)")
    sliding_window_df = pd.read_parquet(sliding_window_score_file, engine="pyarrow")
    peak_to_gene_dist_df = pd.read_parquet(SAMPLE_PROCESSED_DATA_DIR / sample_name / "peak_to_gene_dist.parquet", engine="pyarrow")

    relevant_peaks = set(sliding_window_df["peak_id"].unique())
    peak_to_gene_dist_df = peak_to_gene_dist_df[peak_to_gene_dist_df["peak_id"].isin(relevant_peaks)]

    
    # --- Clean ---
    sliding_window_df = sliding_window_df.dropna(subset=["source_id", "peak_id", "sliding_window_score"])
    sliding_window_df["source_id"] = (
        sliding_window_df["source_id"]
        .str.replace(r"\.pfm$", "", regex=True)
        .apply(standardize_name)
    )

    # --- Group by TF ---
    tf_groups = {tf: df for tf, df in sliding_window_df.groupby("source_id", sort=False)}

    logging.info(f"Processing {len(tf_groups)} TFs using {num_cpu} CPUs")
    results = []

    with ProcessPoolExecutor(max_workers=num_cpu) as ex:
        futures = {
            ex.submit(process_single_tf, tf, df, peak_to_gene_dist_df): tf
            for tf, df in tf_groups.items()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="TF processing"):
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

    # --- Save ---
    tf_tg_reg_pot.to_parquet(tf_tg_reg_pot_file, engine="pyarrow", compression="snappy")
    logging.info(f"Saved TF–TG regulatory potential: {tf_tg_reg_pot.shape}")

def load_rna_adata(sample_raw_data_dir: str) -> sc.AnnData:
    # Look for features file
    features = [f for f in os.listdir(sample_raw_data_dir) if f.endswith("features.tsv.gz") or f.endswith("features.tsv")]
    assert len(features) == 1, f"Expected 1 features.tsv.gz, found {features}"

    prefix = features[0].replace("features.tsv.gz", "")
    prefix = prefix.replace("features.tsv", "")
    logging.info(f"Detected RNA prefix: {prefix}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Only considering the two last:")
        adata = sc.read_10x_mtx(
            sample_raw_data_dir,
            var_names="gene_symbols",
            make_unique=True,
            prefix=prefix
        )
    return adata

def process_10x_to_csv(raw_10x_rna_data_dir, raw_atac_peak_file, rna_outfile_path, atac_outfile_path):
    
    def load_rna_adata(sample_raw_data_dir: str) -> sc.AnnData:
        # Look for features file
        features = [f for f in os.listdir(sample_raw_data_dir) if f.endswith("features.tsv.gz")]
        assert len(features) == 1, \
            f"Expected 1 features.tsv.gz, found {features}. Make sure the files are gunziped for sc.read_10x_mtx."

        prefix = features[0].replace("features.tsv.gz", "")
        logging.info(f"Detected RNA prefix: {prefix}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Only considering the two last:")
            adata = sc.read_10x_mtx(
                sample_raw_data_dir,
                var_names="gene_symbols",
                make_unique=True,
                prefix=prefix
            )
        return adata
    
    def get_adata_from_peakmatrix(peak_matrix_file: Path, label: pd.DataFrame, sample_name: str) -> AnnData:
        logging.info(f"[{sample_name}] Reading ATAC peaks")
        # Read header only
        all_cols = pd.read_csv(peak_matrix_file, sep="\t", nrows=10).columns[1:]
        logging.info(f"  - First ATAC Barcode: {all_cols[0]}")
        
        # Identify barcodes shared between RNA and ATAC
        matching_barcodes = set(label["barcode_use"]) & set(all_cols)
        logging.info(f"  - Matched {len(matching_barcodes):,} barcodes with scRNA-seq file")

        # Map from original index -> normalized barcode
        col_map = {i: bc for i, bc in enumerate(all_cols)}

        # Always keep the first column (peak IDs)
        keep_indices = [0] + [i for i, bc in col_map.items() if bc in matching_barcodes]
        header = pd.read_csv(peak_matrix_file, sep="\t", nrows=0)
        first_col = header.columns[0]
        keep_cols = [first_col] + [c for c in header.columns[1:] if c in matching_barcodes]
        compression = "gzip" if str(peak_matrix_file).endswith(".gz") else None

        # Read only those columns
        logging.info("  - Reading data for matching barcodes")
        peak_matrix = pd.read_csv(
            peak_matrix_file,
            sep="\t",
            usecols=keep_cols,
            index_col=0,
            compression=compression,
            low_memory=False
        )
        logging.info("\tDone reading filtered peak matrix")

        # Replace column names with normalized barcodes
        new_cols = [col_map[i] for i in keep_indices[1:]]
        peak_matrix.columns = new_cols

        # Construct AnnData
        logging.info("  - Constructing AnnData for scATAC-seq data")
        adata_ATAC = AnnData(X=sp.csr_matrix(peak_matrix.values.T))
        adata_ATAC.obs_names = peak_matrix.columns
        adata_ATAC.var_names = peak_matrix.index
        adata_ATAC.obs["barcode"] = adata_ATAC.obs_names
        adata_ATAC.obs["sample"] = sample_name
        adata_ATAC.obs["label"] = label.set_index("barcode_use").loc[peak_matrix.columns, "label"].values
        
        logging.info("\tDone!")

        return adata_ATAC
    
    # --- load raw data ---
    adata_RNA = load_rna_adata(raw_10x_rna_data_dir)
    adata_RNA.obs_names = [(sample_name + "." + i).replace("-", ".") for i in adata_RNA.obs_names]
    # adata_RNA.obs_names = [i.replace("-", ".") for i in adata_RNA.obs_names]
    logging.info(f"[{sample_name}] Found {len(adata_RNA.obs_names)} RNA barcodes")
    logging.info(f"  - First RNA barcode: {adata_RNA.obs_names[0]}")

    label = pd.DataFrame({"barcode_use": adata_RNA.obs_names,
                            "label": ["mESC"] * len(adata_RNA.obs_names)})

    adata_ATAC = get_adata_from_peakmatrix(raw_atac_peak_file, label, sample_name)
    
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
    
    raw_sc_rna_df.to_csv(rna_outfile_path, header=True, index=True)
    raw_sc_atac_df.to_csv(atac_outfile_path, header=True, index=True)

def calculate_summed_tf_tg_score(sliding_window_with_targets: pd.DataFrame):
    # Group by TF and sum all sliding window scores
    sum_of_tf_peaks = (
        sliding_window_with_targets
        .groupby("source_id")["sliding_window_score"]
        .sum()
        .reset_index()
        .rename(columns={"sliding_window_score":"total_tf_score"})
        )
    
    # Group by TF-TG edge and sum for all peaks for that edge
    sum_of_tf_tg_peak_scores = (
        sliding_window_with_targets
        .groupby(["source_id", "target_id"])["sliding_window_score"]
        .sum()
        .reset_index()
        .rename(columns={"sliding_window_score":"tf_to_tg_peak_scores_summed"})
        )
    
    # Merge the total TF peaks and summed TF-TG edges
    sliding_window_sum_calculation_df = pd.merge(
        sum_of_tf_tg_peak_scores, 
        sum_of_tf_peaks, 
        how="left", 
        on="source_id"
        )
    
    
    sliding_window_sum_calculation_df["sliding_window_score"] = (
        sliding_window_sum_calculation_df["tf_to_tg_peak_scores_summed"] / sliding_window_sum_calculation_df["total_tf_score"]
        ) * 1e6
    # sliding_window_sum_df = sliding_window_sum_calculation_df[["source_id", "target_id", "sliding_window_score"]]
    
    return sliding_window_sum_calculation_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Convert NetworkX gpickle to Cosmograph binaries.")
    parser.add_argument("--num_cpu", required=True, help="Number of cores for parallel processing")
    args = parser.parse_args()
    num_cpu = int(args.num_cpu)
    
    DATASET_NAME = "PBMC"
    sample_name = "LINGER_PBMC_SC_DATA"
    
    RAW_DATA = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/raw") 
    GENOME_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/genome_data/genome_annotation/hg38")
    sample_input_dir = RAW_DATA / sample_name
    
    processed_rna_file = sample_input_dir / "scRNA_seq_processed.parquet"
    processed_atac_file = sample_input_dir / "scATAC_seq_processed.parquet"
    
    organism_code = "hg38"
    ensemble_dataset_name = "hsapiens_gene_ensembl"
    GENE_TSS_FILE = DATA_DIR / "databases/gene_tss/hg38.gene_tss.bed.gz"
    PROCESSED_DATA = DATA_DIR / "processed"
    SAMPLE_PROCESSED_DATA_DIR = PROCESSED_DATA / DATASET_NAME
    RAW_10X_RNA_DATA_DIR = RAW_DATA / sample_name / "PBMC_raw"
    RAW_ATAC_PEAK_MATRIX_FILE = RAW_DATA / sample_name / "Peaks.txt"
    
    tf_file = DATA_DIR / "databases" / "motif_information" / organism_code / "TF_Information_all_motifs.txt"
    
    IGNORE_PROCESSED_FILES = False
    logging.info(f"IGNORE_PROCESSED_FILES: {IGNORE_PROCESSED_FILES}")
    
    if (not os.path.isfile(processed_rna_file)) or (not os.path.isfile(processed_atac_file)) or IGNORE_PROCESSED_FILES:
        
        if (not os.path.isfile(sample_input_dir / "adata_ATAC.h5ad")) or (not os.path.isfile(sample_input_dir / "adata_RNA.h5ad")) or IGNORE_PROCESSED_FILES:
    
            logging.info("\nReading RNA and ATAC files")
            rna_file = sample_input_dir / "scRNA_seq_raw.csv"
            atac_file = sample_input_dir / "scATAC_seq_raw.csv"
            
            if (not os.path.isfile(rna_file)) or (not os.path.isfile(atac_file)):
                if RAW_10X_RNA_DATA_DIR is not None:
                    process_10x_to_csv(RAW_10X_RNA_DATA_DIR, RAW_ATAC_PEAK_MATRIX_FILE, rna_file, atac_file)
                else:
                    logging.error("ERROR: Input RNA or ATAC file not found")

            logging.info("Reading RNA and ATAC files")
            rna_df = pd.read_csv(rna_file, delimiter=",", header=0, index_col=0)
            atac_df = pd.read_csv(atac_file, delimiter=",", header=0, index_col=0)

            adata_rna = AnnData(rna_df.T)
            adata_atac = AnnData(atac_df.T)
            
            adata_rna_filtered, adata_atac_filtered = filter_and_qc(adata_rna, adata_atac)
            # logging.info(adata_rna_filtered.shape)
        else:
            logging.info("\nReading pre-filtered RNA and ATAC files")
            adata_rna_filtered = sc.read_h5ad(sample_input_dir / "adata_RNA.h5ad")
            adata_atac_filtered = sc.read_h5ad(sample_input_dir / "adata_ATAC.h5ad")
            
            if "gene_ids" in adata_atac_filtered.var.columns:
                adata_atac_filtered.var_names = adata_atac_filtered.var["gene_ids"].astype(str)

        logging.info("  - adata_rna_filtered.obs_names: " + str(adata_rna_filtered.obs_names[:5]))
        logging.info("  - adata_atac_filtered.obs_names: " + str(adata_atac_filtered.obs_names[:5]))
        
        logging.info("  - adata_rna_filtered.var_names: " + str(adata_rna_filtered.var_names[:5]))
        logging.info("  - adata_atac_filtered.var_names: " + str(adata_atac_filtered.var_names[:5]))
        
        logging.info("  - Shape of adata_rna_filtered: " + str(adata_rna_filtered.shape))
        logging.info("  - Shape of adata_atac_filtered: " + str(adata_atac_filtered.shape))
        
        logging.info("\nConverting adata to dense dataframe")
        
        def adata_to_dense_df(adata):
            X = adata.X
            if sp.issparse(X):
                X = X.toarray()
            else:
                X = np.asarray(X)
            return pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names).T
        
        logging.info("  - Converting RNA adata to dense dataframe")
        processed_rna_df  = adata_to_dense_df(adata_rna_filtered)
        logging.info("  - Converting ATAC adata to dense dataframe")
        processed_atac_df = adata_to_dense_df(adata_atac_filtered)
        
        logging.info("\nWriting processed RNA and ATAC files")
        logging.info("  - Writing pre-processed RNA file")
        processed_rna_df.to_parquet(sample_input_dir / "scRNA_seq_processed.parquet", engine="pyarrow", compression="snappy")
        logging.info("  - Writing pre-processed ATAC file")
        processed_atac_df.to_parquet(sample_input_dir / "scATAC_seq_processed.parquet", engine="pyarrow", compression="snappy")

    else:
        logging.info("Pre-processed RNA and ATAC files found, loading...")
        logging.info("  - Reading pre-processed RNA file")
        processed_rna_df = pd.read_parquet(processed_rna_file, engine="pyarrow")
        logging.info("  - Reading pre-processed ATAC file")
        processed_atac_df = pd.read_parquet(processed_atac_file, engine="pyarrow")
    
    genes = processed_rna_df.index.to_list()
    peaks = processed_atac_df.index.to_list()
    
    logging.info("\nProcessed RNA and ATAC files loaded")
    logging.info(f"  - Number of genes: {processed_rna_df.shape[0]}: {genes[:3]}")
    logging.info(f"  - Number of peaks: {processed_atac_df.shape[0]}: {peaks[:3]}")

    tfs, tgs, tf_tg_df = create_tf_tg_combination_files(genes, tf_file)
    
    if not os.path.isdir(GENOME_DIR):
        os.makedirs(GENOME_DIR)
    
    if not os.path.isfile(GENE_TSS_FILE):
        download_gene_tss_file(
            save_file=GENE_TSS_FILE,
            gene_dataset_name=ensemble_dataset_name,
        )

    gene_tss_df = make_chrom_gene_tss_df(
        gene_tss_file=GENE_TSS_FILE,
        genome_dir=GENOME_DIR
    )

    peak_bed_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "peaks.bed"
    
    # Format peaks correctly into BED-compatible dataframe
    peak_locs_df = format_peaks(pd.Series(processed_atac_df.index)).rename(columns={"chromosome": "chrom"})

    peak_bed_file.parent.mkdir(parents=True, exist_ok=True)
    pybedtools.BedTool.from_dataframe(
        peak_locs_df[["chrom", "start", "end", "peak_id"]]
    ).saveas(peak_bed_file)
    
    if not os.path.isfile(SAMPLE_PROCESSED_DATA_DIR / sample_name / "peak_to_gene_dist.parquet") or IGNORE_PROCESSED_FILES:
        peak_to_gene_dist_df = calculate_peak_to_tg_distance_score(
            peak_bed_file=peak_bed_file,
            tss_bed_file=GENE_TSS_FILE,
            peak_gene_dist_file=SAMPLE_PROCESSED_DATA_DIR / sample_name / "peak_to_gene_dist.parquet",
            mesc_atac_peak_loc_df=peak_locs_df,
            gene_tss_df=gene_tss_df,
            force_recalculate=IGNORE_PROCESSED_FILES
        )
    else:
        logging.info("Peak to gene distance file found, loading...")
        peak_to_gene_dist_df = pd.read_parquet(SAMPLE_PROCESSED_DATA_DIR / sample_name / "peak_to_gene_dist.parquet", engine="pyarrow")
    
    logging.info("\nPeak to gene distance file loaded")
    logging.info("  - Number of peaks to gene distances: " + str(peak_to_gene_dist_df.shape[0]))
    logging.info("  - Example peak to gene distances: " + str(peak_to_gene_dist_df.head()))
    
    # Calculate TF-peak binding score
    sliding_window_score_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "sliding_window.parquet"
    if not os.path.isfile(sliding_window_score_file):

        # jaspar_pfm_dir = JASPAR_PFM_DIR
        # # if not os.path.isdir(jaspar_pfm_dir):
        # download_jaspar_pfms(
        #     str(jaspar_pfm_dir),
        #     tax_id="9606",
        #     max_workers=3
        #     )
        
        genome_fasta_file = GENOME_DIR / (organism_code + ".fa.gz")
        if not os.path.isfile(genome_fasta_file):
            download_genome_fasta(
                organism_code=organism_code,
                save_dir=GENOME_DIR
            )

        # jaspar_pfm_paths = [os.path.join(jaspar_pfm_dir, f) for f in os.listdir(jaspar_pfm_dir) if f.endswith(".pfm")]
        
        peaks_bed_path = Path(peak_bed_file)
        peaks_df = pybedtools.BedTool(peaks_bed_path)
        
        tf_info_file = DATA_DIR / "databases" / "motif_information" / organism_code / "TF_Information_all_motifs.txt"
        motif_dir = DATA_DIR / "databases" / "motif_information" / organism_code / str(organism_code + "_motif_meme_files")

        # logging.info("Running MOODS TF-peak binding calculation")
        # run_moods_scan_batched(
        #     peaks_bed=peak_bed_file, 
        #     fasta_path=genome_fasta_file, 
        #     motif_paths=jaspar_pfm_paths, 
        #     out_file=moods_sites_file, 
        #     n_cpus=min(8, int(args.num_cpu)),
        #     pval_threshold=1e-4, 
        #     bg="auto",
        #     batch_size=25
        # )
        
        run_sliding_window_scan(
            tf_name_list=tfs,
            tf_info_file=str(tf_info_file),
            motif_dir=str(motif_dir),
            genome_fasta=str(genome_fasta_file),
            peak_bed_file=str(peak_bed_file),
            output_dir=str(SAMPLE_PROCESSED_DATA_DIR / sample_name),
            num_cpu=num_cpu
        )

    # Get the 0-1 MinMax normalized ATAC accessibility
    def minmax_scale_across_cells(df):
        norm_df = df.copy()
        scaler = MinMaxScaler()
        x = norm_df.values
        norm_df.loc[:, :] = scaler.fit_transform(x)

        return norm_df
    
    tf_df = processed_rna_df[processed_rna_df.index.isin(tfs)]
    tg_df = processed_rna_df[processed_rna_df.index.isin(tgs)]

    norm_atac_df = minmax_scale_across_cells(processed_atac_df)
    norm_tf_df = minmax_scale_across_cells(tf_df)
    norm_tg_df = minmax_scale_across_cells(tg_df)

    # Calculate the mean ATAC accessibility per peak
    mean_norm_atac_acc = norm_atac_df.mean(axis=1)

    # NOTE: Instead of mean, should I pass each cell's expression or pseudobulk?
    # Calculate the mean TF and TG expression
    mean_norm_tg_expr = (
        norm_tg_df
        .mean(axis=1)
        .reset_index()
        .rename(columns={"index": "TG", 0: "mean_tg_expr"})
    )

    mean_norm_tf_expr = (
        norm_tf_df
        .mean(axis=1)
        .reset_index()
        .rename(columns={"index": "TF", 0: "mean_tf_expr"})
    )

    tf_tg_reg_pot_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "tf_tg_regulatory_potential.parquet"
    if not os.path.isfile(tf_tg_reg_pot_file):
        calculate_tf_tg_regulatory_potential(sliding_window_score_file, tf_tg_reg_pot_file, num_cpu)
    
    logging.info("\nLoading TF-TG regulatory potential scores")
    tf_tg_reg_pot = pd.read_parquet(tf_tg_reg_pot_file, engine="pyarrow")
    logging.info("  - Example TF-TG regulatory potential: " + str(tf_tg_reg_pot.head()))

    logging.info("\nLoading ChIP-seq ground truth for labeling edges")
    ground_truth_file = DATA_DIR / "ground_truth_files" / "rn204_macrophage_human_chipseq.tsv"
    ground_truth_df = pd.read_csv(ground_truth_file, sep="\t", header=0, index_col=None)
    ground_truth_df = ground_truth_df.rename(columns={
        "Source":"TF",
        "Target":"TG"
    })
    ground_truth_df["TF"] = ground_truth_df["TF"].str.upper()
    ground_truth_df["TG"] = ground_truth_df["TG"].str.upper()
    tf_tg_df["TF"] = tf_tg_df["TF"].str.upper()
    tf_tg_df["TG"] = tf_tg_df["TG"].str.upper()
    
    logging.info("  - Example ground truth: \n" + str(ground_truth_df.head()))
    logging.info("  - Number of unique TFs in ground truth: " + str(ground_truth_df["TF"].nunique()))
    logging.info("  - Number of unique TGs in ground truth: " + str(ground_truth_df["TG"].nunique()))
    logging.info("  - Number of ground truth edges: " + str(ground_truth_df.shape[0]))

    logging.info("Merging TF-TG attributes with all combinations")
    logging.info("  - Merging TF-TG Regulatory Potential")
    tf_tg_df = pd.merge(
        tf_tg_df,
        tf_tg_reg_pot,
        how="left",
        on=["TF", "TG"]
    ).fillna(0)

    logging.info("  - Merging mean min-max normalized TF expression")
    
    logging.info(f"tf_tg_df columns: {list(tf_tg_df.columns)}")
    logging.info(f"mean_norm_tf_expr columns: {list(mean_norm_tf_expr.columns)}")

    tf_tg_df = pd.merge(
        tf_tg_df,
        mean_norm_tf_expr,
        how="left",
        on=["TF"]
    ).dropna(subset="mean_tf_expr")

    logging.info("  - Merging mean min-max normalized TG expression")
    tf_tg_df = pd.merge(
        tf_tg_df,
        mean_norm_tg_expr,
        how="left",
        on=["TG"]
    ).dropna(subset="mean_tg_expr")
    
    tf_tg_df["expr_product"] = tf_tg_df["mean_tf_expr"] * tf_tg_df["mean_tg_expr"]
    tf_tg_df["log_reg_pot"] = np.log1p(tf_tg_df["reg_potential"])
    tf_tg_df["motif_present"] = (tf_tg_df["motif_density"] > 0).astype(int)
    
    tf_tg_df.to_parquet(SAMPLE_PROCESSED_DATA_DIR / sample_name / "tf_tg_data_unlabeled.parquet", index=False)

    
    
    # Create a set of ground-truth pairs
    gt_pairs = set(zip(ground_truth_df["TF"], ground_truth_df["TG"]))
    logging.info("  - Number of ground truth pairs: " + str(len(gt_pairs)))
    logging.info("  - Example ground truth pairs: " + str(list(gt_pairs)[:10]))

    # Assign label = 1 if (TF, TG) pair exists in ground truth
    tf_tg_df["label"] = [
        1 if (tf, tg) in gt_pairs else 0
        for tf, tg in zip(tf_tg_df["TF"], tf_tg_df["TG"])
    ]
    logging.info("  - Number of positive labels: " + str(tf_tg_df["label"].sum()))
    logging.info("  - Number of negative labels: " + str((tf_tg_df["label"] == 0).sum()))
    
    gt_tfs = set(ground_truth_df["TF"].unique())
    gt_tgs = set(ground_truth_df["TG"].unique())
    pred_tfs = set(tf_tg_df["TF"].unique())
    pred_tgs = set(tf_tg_df["TG"].unique())

    print("TF overlap:", len(gt_tfs & pred_tfs), "/", len(gt_tfs))
    print("TG overlap:", len(gt_tgs & pred_tgs), "/", len(gt_tgs))
    print("Example missing TFs:", list(gt_tfs - pred_tfs)[:10])
    print("Example missing TGs:", list(gt_tgs - pred_tgs)[:10])

    true_df = tf_tg_df[tf_tg_df["label"] == 1]
    false_df = tf_tg_df[tf_tg_df["label"] == 0]
    
    print(tf_tg_df["TF"].nunique(), "TFs in training")
    print(tf_tg_df["TG"].nunique(), "TGs in training")
    print(tf_tg_df["label"].value_counts())
    print(tf_tg_df.describe())

    # For each TF, randomly choose one false TG for each true TG
    balanced_rows = []
    rng = np.random.default_rng(42)
    upscale_percent = 1.5

    logging.info("\nBalancing TF-TG pairs")
    for tf, group in true_df.groupby("TF"):
        logging.info(f"  - Creating balanced dataset for TF: {tf}")
        true_tgs = group["TG"].tolist()

        # candidate false TGs for same TF
        false_candidates = false_df[false_df["TF"] == tf]
        if false_candidates.empty:
            continue  # skip if no negatives for this TF

        # sample one negative TG per true TG (with replacement)
        sampled_false = false_candidates.sample(
            n=len(true_tgs), replace=True, random_state=rng.integers(int(1e9))
        )

        # randomly resample with replacement to get a % increase in True / False edges per TF
        true_upscaled = group.sample(frac=upscale_percent, replace=True)
        false_upscaled = sampled_false.sample(frac=upscale_percent, replace=True)
        logging.info(f"      - TF: {tf}, True: {len(true_upscaled)}, False: {len(false_upscaled)}")


        
        balanced_rows.append(pd.concat([true_upscaled, false_upscaled], ignore_index=True))
        logging.info(f"       - Number of unique TGs: {pd.concat(balanced_rows, ignore_index=True)['TG'].nunique()}")

    # Combine all TF groups
    tf_tg_balanced = pd.concat(balanced_rows, ignore_index=True)
    tf_tg_balanced = tf_tg_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    logging.info("  - Balanced TF-TG pairs loaded")

    logging.info("  - Example balanced TF-TG pairs: " + str(tf_tg_balanced.head()))

    tf_tg_feature_file = SAMPLE_PROCESSED_DATA_DIR / sample_name / "tf_tg_data.parquet"
    tf_tg_balanced.to_parquet(tf_tg_feature_file, engine="pyarrow", compression="snappy")
    logging.info(f"\n Wrote TF-TG features to {tf_tg_feature_file}")