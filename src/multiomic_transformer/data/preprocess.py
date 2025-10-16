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
from typing import Tuple
from anndata import AnnData
from tqdm import tqdm
import pybedtools

import sys
sys.path.append(Path(__file__).resolve().parent.parent.parent)

from multiomic_transformer.data.moods_scan import run_moods_scan_batched
from multiomic_transformer.utils.standardize import standardize_name
from multiomic_transformer.utils.files import atomic_json_dump
from multiomic_transformer.utils.peaks import find_genes_near_peaks
from multiomic_transformer.utils.downloads import *
from config.settings import *

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
    
    if not os.path.isfile(tf_tg_outdir / "total_genes.csv"):
        print("Writing total genes to 'total_genes.csv'")
        with open(tf_tg_outdir / "total_genes.csv", 'w') as total_genes_file:
            total_genes_file.write("Gene\n")
            for i in genes:
                total_genes_file.write(f"{i}\n")
    
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

def calculate_tf_tg_regulatory_potential(moods_sites_file, tf_tg_reg_pot_file):
    logging.info(f"\nLoading MOODS TF-peak binding")
    moods_hits = pd.read_parquet(moods_sites_file, engine="pyarrow")
    
    # Drop rows with missing TFs just in case
    moods_hits = moods_hits.dropna(subset=["TF", "peak_id", "logodds"])
    # Strip ".pfm" suffix from TF names if present
    moods_hits["TF"] = moods_hits["TF"].str.replace(r"\.pfm$", "", regex=True).apply(standardize_name)
    
    # Compute per-TF binding probability across peaks
    moods_hits["logodds_tf_softmax"] = moods_hits.groupby("peak_id")["logodds"].transform(softmax)

    # Merge with peak-to-gene distance scores
    merged = pd.merge(
        moods_hits[["peak_id", "TF", "logodds_tf_softmax"]],
        peak_to_gene_dist_df[["peak_id", "target_id", "TSS_dist_score"]],
        on="peak_id",
        how="inner"
    )

    # Compute TF–TG contribution per peak
    merged["tf_tg_contrib"] = merged["logodds_tf_softmax"] * merged["TSS_dist_score"]

    # Compute motif density
    # motif_density = number of unique TF motifs per TF–TG (or per TF–TG–peak normalized)
    motif_density_df = (
        merged.groupby(["TF", "target_id"], as_index=False)
        .agg({
            "peak_id": "nunique",                 # how many peaks with that TF
        })
        .rename(columns={"peak_id": "motif_density"})
    )

    # Log1p normalize the motif density
    tf_tg_reg_pot["motif_density"] = np.log1p(tf_tg_reg_pot["motif_density"])

    # Aggregate regulatory potential over peaks
    tf_tg_reg_pot = (
        merged.groupby(["TF", "target_id"], as_index=False)
        .agg({"tf_tg_contrib": "sum"})
        .rename(columns={"target_id": "TG", "tf_tg_contrib": "reg_potential"})
    )

    # Merge motif density back in
    tf_tg_reg_pot = tf_tg_reg_pot.merge(
        motif_density_df.rename(columns={"target_id": "TG"}),
        on=["TF", "TG"],
        how="left"
    )

    # Save
    tf_tg_reg_pot.to_parquet(tf_tg_reg_pot_file, engine="pyarrow", compression="snappy")
    logging.info(f"Saved TF–TG regulatory potential with motif density: {tf_tg_reg_pot.shape}")

def load_rna_adata(sample_raw_data_dir: str) -> sc.AnnData:
    # Look for features file
    features = [f for f in os.listdir(sample_raw_data_dir) if f.endswith("features.tsv.gz")]
    assert len(features) == 1, f"Expected 1 features.tsv.gz, found {features}"

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

def process_10x_to_csv(raw_10x_rna_data_dir, raw_atac_peak_file, rna_outfile_path, atac_outfile_path):
    
    def load_rna_adata(sample_raw_data_dir: str) -> sc.AnnData:
        # Look for features file
        features = [f for f in os.listdir(sample_raw_data_dir) if f.endswith("features.tsv.gz")]
        assert len(features) == 1, f"Expected 1 features.tsv.gz, found {features}"

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

        # Map from original index -> normalized barcode
        col_map = {i: bc for i, bc in enumerate(all_cols)}

        # Always keep the first column (peak IDs)
        keep_indices = [0] + [i for i, bc in col_map.items() if bc in matching_barcodes]

        # Read only those columns
        peak_matrix = pd.read_csv(
            peak_matrix_file,
            sep="\t",
            # usecols=keep_indices,
            index_col=0
        )

        # Replace column names with normalized barcodes
        new_cols = [col_map[i] for i in keep_indices[1:]]
        peak_matrix.columns = new_cols
        new_cols = peak_matrix.columns

        # Construct AnnData
        X = sp.csr_matrix(peak_matrix.values)
        adata_ATAC = AnnData(X=X.T)

        # Assign metadata
        adata_ATAC.obs_names = new_cols
        adata_ATAC.obs["barcode"] = new_cols
        adata_ATAC.obs["sample"] = sample_name
        adata_ATAC.obs["label"] = label.set_index("barcode_use").loc[new_cols, "label"].values

        adata_ATAC.var_names = peak_matrix.index
        adata_ATAC.var["gene_ids"] = peak_matrix.index

        return adata_ATAC
    
    # --- load raw data ---
    sample_raw_data_dir = os.path.join(raw_10x_rna_data_dir, sample_name)
    adata_RNA = load_rna_adata(sample_raw_data_dir)
    adata_RNA.obs_names = [(sample_name + "." + i).replace("-", ".") for i in adata_RNA.obs_names]
    # adata_RNA.obs_names = [i.replace("-", ".") for i in adata_RNA.obs_names]
    logging.info(f"[{sample_name}] Found {len(adata_RNA.obs_names)} RNA barcodes")
    logging.info(f"  - First RNA barcode: {adata_RNA.obs_names[0]}")

    label = pd.DataFrame({"barcode_use": adata_RNA.obs_names,
                            "label": ["mESC"] * len(adata_RNA.obs_names)})

    adata_ATAC = get_adata_from_peakmatrix(raw_atac_peak_file, label, sample_name)
    
    raw_sc_rna_df = pd.DataFrame(
        adata_RNA.S,
        index=adata_RNA.var_names,
        columns=adata_RNA,
    )
    raw_sc_atac_df = pd.DataFrame(
        adata_ATAC.X,
        index=adata_ATAC.var_names,
        columns=adata_ATAC,
    )
    
    os.makedirs(rna_outfile_path, exist_ok=True)
    os.makedirs(atac_outfile_path, exist_ok=True)
    
    raw_sc_rna_df.to_csv(rna_outfile_path, header=True, index=True)
    raw_sc_atac_df.to_csv(atac_outfile_path, header=True, index=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    sample_name = "E7.5_rep1"
    
    rna_file = RAW_DATA / sample_name / "scRNA_seq_raw.csv"
    atac_file = RAW_DATA / sample_name / "scATAC_seq_raw.csv"
    
    if (not os.path.isfile(rna_file)) or (not os.path.isfile(atac_file)):
        if RAW_10X_RNA_DATA_DIR is not None:
            process_10x_to_csv(RAW_10X_RNA_DATA_DIR, RAW_ATAC_PEAK_MATRIX_FILE, rna_file, atac_file)
        else:
            logging.error("ERROR: Input RNA or ATAC file not found")

    rna_df = pd.read_csv(rna_file, delimiter="\t", header=0, index_col=0)
    atac_df = pd.read_csv(atac_file, delimiter="\t", header=0, index_col=0)

    print(f"Number of peaks: {atac_df.shape[0]}")

    atac_df = atac_df.iloc[:1500, :]

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
    # print(adata_rna_filtered.shape)

    if not os.path.isdir(GENOME_DIR):
        os.makedirs(GENOME_DIR)
    
    if not os.path.isfile(GENE_TSS_FILE):
        download_gene_tss_file(
            save_dir=GENE_TSS_FILE,
            gene_dataset_name="mmusculus_gene_ensembl",
        )

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

    # print("Processed RNA df")
    # print(processed_rna_df.head())

    # print("\nProcessed ATAC df")
    # print(processed_atac_df.head())

    # Build peak locations from original ATAC peak names
    peak_locs_df = build_peak_locs_from_index(processed_atac_df.index)

    peak_bed_file = DATA_DIR / "processed" / "peaks.bed"
    peak_bed_file.parent.mkdir(parents=True, exist_ok=True)
    peak_locs_df.to_csv(peak_bed_file, sep="\t", header=False, index=False)

    peak_to_gene_dist_df = calculate_peak_to_tg_distance_score(
        peak_bed_file=peak_bed_file,
        tss_bed_file= GENE_TSS_FILE,
        peak_gene_dist_file= DATA_DIR / "processed" / "peak_to_gene_dist.parquet",
        mesc_atac_peak_loc_df=peak_locs_df,
        gene_tss_df= gene_tss_df,
    )
    print("Peak to Gene Distance DataFrame")
    print(peak_to_gene_dist_df.head())

    print("\nTF-TG Combinations")
    print(tf_tg_df.head())

    # Calculate TF-peak binding score
    moods_sites_file = DATA_DIR / "processed" / "moods_sites.parquet"
    if not os.path.isfile(moods_sites_file):

        jaspar_pfm_dir = DATA_DIR / "motif_information" / "mm10" / "JASPAR" / "pfm_files"
        if not os.path.isdir(jaspar_pfm_dir):
            download_jaspar_pfms(
                jaspar_pfm_dir,
                tax_id="10090",
                max_workers=3
                )
        
        genome_fasta_file = GENOME_DIR / "mm10.fa.gz"
        if not os.path.isfile(genome_fasta_file):
            download_genome_fasta(
                organism_code="mm10",
                save_location=genome_fasta_file
            )

        jaspar_pfm_paths = [os.path.join(jaspar_pfm_dir, f) for f in os.listdir(jaspar_pfm_dir) if f.endswith(".pfm")]
        
        peaks_bed_path = Path(peak_bed_file)
        peaks_df = pybedtools.BedTool(peaks_bed_path)

        subset_jaspar_motifs = jaspar_pfm_paths[:50]
        print("Running MOODS TF-peak binding calculation")
        run_moods_scan_batched(
            peaks_bed=peak_bed_file, 
            fasta_path=genome_fasta_file, 
            motif_paths=subset_jaspar_motifs, 
            out_file=moods_sites_file, 
            n_cpus=3,
            pval_threshold=1e-3, 
            bg="auto",
            batch_size=500
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
        .rename(columns={"Symbol": "TG", 0: "mean_tg_expr"})
    )

    mean_norm_tf_expr = (
        norm_tf_df
        .mean(axis=1)
        .reset_index()
        .rename(columns={"Symbol": "TF", 0: "mean_tf_expr"})
    )
    print("mean_norm_tf_expr")
    print(mean_norm_tf_expr)

    print("\nmean_norm_tg_expr")
    print(mean_norm_tg_expr)

    tf_tg_reg_pot_file = DATA_DIR / "processed" / "tf_tg_regulatory_potential.parquet"
    # if not os.path.isfile(tf_tg_reg_pot_file):
    calculate_tf_tg_regulatory_potential(moods_sites_file, tf_tg_reg_pot_file)
    
    print("Loading TF-TG regulatory potential scores")
    tf_tg_reg_pot = pd.read_parquet(tf_tg_reg_pot_file, engine="pyarrow")
    print(tf_tg_reg_pot.head())

    print("Loading ChIP-seq Ground Truth for Labeling Edges")
    ground_truth_file = DATA_DIR / "ground_truth" / "RN111.tsv"
    ground_truth_df = read_ground_truth(ground_truth_file)
    ground_truth_df = ground_truth_df[["source_id", "target_id"]].rename(columns={
        "source_id":"TF",
        "target_id":"TG"
    })
    ground_truth_df["TF"] = ground_truth_df["TF"].str.capitalize()
    ground_truth_df["TG"] = ground_truth_df["TG"].str.capitalize()

    print("Merging TF-TG attributes with all combinations")
    print("  - Merging TF-TG Regulatory Potential")
    tf_tg_df = pd.merge(
        tf_tg_df,
        tf_tg_reg_pot,
        how="left",
        on=["TF", "TG"]
    ).fillna(0)

    print("  - Merging mean min-max normalized TF expression")
    tf_tg_df = pd.merge(
        tf_tg_df,
        mean_norm_tf_expr,
        how="left",
        on=["TF"]
    ).dropna(subset="mean_tf_expr")

    print("  - Merging mean min-max normalized TG expression")
    tf_tg_df = pd.merge(
        tf_tg_df,
        mean_norm_tg_expr,
        how="left",
        on=["TG"]
    ).dropna(subset="mean_tg_expr")

    # Create a set of ground-truth pairs
    gt_pairs = set(zip(ground_truth_df["TF"], ground_truth_df["TG"]))

    # Assign label = 1 if (TF, TG) pair exists in ground truth
    tf_tg_df["label"] = [
        1 if (tf, tg) in gt_pairs else 0
        for tf, tg in zip(tf_tg_df["TF"], tf_tg_df["TG"])
    ]

    true_df = tf_tg_df[tf_tg_df["label"] == 1]
    false_df = tf_tg_df[tf_tg_df["label"] == 0]

    # For each TF, randomly choose one false TG for each true TG
    balanced_rows = []
    rng = np.random.default_rng(42)
    upscale_percent = 1.5

    for tf, group in true_df.groupby("TF"):
        true_tgs = group["TG"].tolist()

        # candidate false TGs for same TF
        false_candidates = false_df[false_df["TF"] == tf]
        if false_candidates.empty:
            continue  # skip if no negatives for this TF

        # sample one negative TG per true TG (with replacement)
        sampled_false = false_candidates.sample(
            n=len(true_tgs), replace=True, random_state=rng.integers(1e9)
        )

        # randomly resample with replacement to get a % increase in True / False edges per TF
        true_upscaled = group.sample(frac=upscale_percent, replace=True)
        false_upscaled = sampled_false.sample(frac=upscale_percent, replace=True)

        balanced_rows.append(pd.concat([true_upscaled, false_upscaled], ignore_index=True))

    # Combine all TF groups
    tf_tg_balanced = pd.concat(balanced_rows, ignore_index=True)
    tf_tg_balanced = tf_tg_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(tf_tg_balanced["label"].value_counts())

    print(tf_tg_balanced.head())

    tf_tg_feature_file = DATA_DIR / "processed" / "tf_tg_data.parquet"
    tf_tg_balanced.to_parquet(tf_tg_feature_file, engine="pyarrow", compression="snappy")
    print(f"\n Wrote TF-TG features to {tf_tg_feature_file}")