import os
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import pybedtools
from typing import Optional, Union
from scipy import sparse
import logging
import json
from pathlib import Path
import scipy.sparse as sps
from joblib import Parallel, delayed
import scipy.sparse as sp

from grn_inference import utils

#=================================== USER SETTINGS ===================================
# ----- User Settings -----
load_model = False
window_size = 800
num_cells = 1000
chrom_id = "chr19"
force_recalculate = True

atac_data_filename = "mESC_filtered_L2_E7.5_rep1_ATAC_processed.parquet"
rna_data_filename = "mESC_filtered_L2_E7.5_rep1_RNA_processed.parquet"

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
MM10_GENOME_DIR = os.path.join(PROJECT_DIR, "data/reference_genome/mm10")
MM10_GENE_TSS_FILE = os.path.join(PROJECT_DIR, "data/genome_annotation/mm10/mm10_TSS.bed")
GROUND_TRUTH_DIR = os.path.join(PROJECT_DIR, "ground_truth_files")
SAMPLE_INPUT_DIR = os.path.join(PROJECT_DIR, "input/mESC/filtered_L2_E7.5_rep1")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output/transformer_testing_output")
DEBUG_FILE = os.path.join(PROJECT_DIR, "LOGS/transformer_training.debug")

MM10_FASTA_FILE = os.path.join(MM10_GENOME_DIR, f"{chrom_id}.fa")
MM10_CHROM_SIZES_FILE = os.path.join(MM10_GENOME_DIR, "chrom.sizes")

time_now = datetime.now().strftime("%d%m%y%H%M%S")

TRAINING_DIR=os.path.join(PROJECT_DIR, f"training_stats_{time_now}/")

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, f"checkpoints_{time_now}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "training_stats"), exist_ok=True)

def _ensure_dir(p: Union[str,Path]):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_sparse_npz(path: Union[str,Path], mat: sps.spmatrix):
    _ensure_dir(Path(path).parent)
    sps.save_npz(str(path), mat, compressed=True)

def save_numpy(path: Union[str,Path], arr: np.ndarray):
    _ensure_dir(Path(path).parent)
    np.save(str(path), arr)

def save_json(path: Union[str,Path], obj):
    _ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f)

def save_npz(path: Union[str,Path], **arrays):
    _ensure_dir(Path(path).parent)
    np.savez_compressed(path, **arrays)

def load_or_create_gene_tss_df(chrom_id, force_recalculate=False):
    gene_tss_outfile = os.path.join(MM10_GENOME_DIR, "mm10_ch1_gene_tss.bed")
    if not os.path.isfile(gene_tss_outfile) or force_recalculate:
        mm10_gene_tss_bed = pybedtools.BedTool(MM10_GENE_TSS_FILE)
        
        gene_tss_df = (
            mm10_gene_tss_bed
            .filter(lambda x: x.chrom == chrom_id)
            .saveas(gene_tss_outfile)
            .to_dataframe()
            .sort_values(by="start", ascending=True)
            )
    else:
        gene_tss_df = pybedtools.BedTool(gene_tss_outfile).to_dataframe().sort_values(by="start", ascending=True)
        
    return gene_tss_df

def load_atac_dataset(atac_data_filename, chrom_id):
    mesc_atac_data = pd.read_parquet(os.path.join(SAMPLE_INPUT_DIR, atac_data_filename)).set_index("peak_id")
    mesc_atac_peak_loc = mesc_atac_data.index

    # format the peaks to be in bed_format
    mesc_atac_peak_loc_df = utils.format_peaks(mesc_atac_peak_loc)
    mesc_atac_peak_loc_df = mesc_atac_peak_loc_df[mesc_atac_peak_loc_df["chromosome"] == chrom_id]
    mesc_atac_peak_loc_df = mesc_atac_peak_loc_df.rename(columns={"chromosome":"chrom"})

    # TEMPORARY Restrict to one chromosome for testing
    mesc_atac_data = mesc_atac_data[mesc_atac_data.index.isin(mesc_atac_peak_loc_df.peak_id)]
    
    return mesc_atac_data, mesc_atac_peak_loc_df

def load_rna_data(rna_data_filename):
    logging.info("Reading in the scRNA-seq dataset")
    mesc_rna_data = pd.read_parquet(
        os.path.join(SAMPLE_INPUT_DIR, rna_data_filename)).set_index("gene_id")
    return mesc_rna_data

def create_or_load_genomic_windows(chrom_id, force_recalculate=False):
    genome_window_file = os.path.join(MM10_GENOME_DIR, f"mm10_{chrom_id}_windows_{window_size // 1000}kb.bed")
    if not os.path.exists(genome_window_file) or force_recalculate:
        
        logging.info("Creating genomic windows")
        mm10_genome_windows = pybedtools.bedtool.BedTool().window_maker(g=MM10_CHROM_SIZES_FILE, w=window_size)
        mm10_windows = (
            mm10_genome_windows
            .filter(lambda x: x.chrom == chrom_id)  # TEMPORARY Restrict to one chromosome for testing
            .saveas(genome_window_file)
            .to_dataframe()
        )
    else:
        
        logging.info("Loading existing genomic windows")
        mm10_windows = pybedtools.BedTool(genome_window_file).to_dataframe()
        
    return mm10_windows

def calculate_peak_to_tg_distance_score(mesc_atac_peak_loc_df, gene_tss_df, force_recalculate=False):
    if not os.path.isfile(os.path.join(OUTPUT_DIR, "genes_near_peaks.parquet")) or force_recalculate:
        if "peak_tmp.bed" not in os.listdir(OUTPUT_DIR) or "tss_tmp.bed" not in os.listdir(OUTPUT_DIR) or force_recalculate:
        
            logging.info("Calculating peak to TG distance score")
            peak_bed = pybedtools.BedTool.from_dataframe(
                mesc_atac_peak_loc_df[["chrom", "start", "end", "peak_id"]]
                ).saveas(os.path.join(OUTPUT_DIR, "peak_tmp.bed"))

            tss_bed = pybedtools.BedTool.from_dataframe(
                gene_tss_df[["chrom", "start", "end", "name"]]
                ).saveas(os.path.join(OUTPUT_DIR, "tss_tmp.bed"))
            
        peak_bed = pybedtools.BedTool(os.path.join(OUTPUT_DIR, "peak_tmp.bed"))
        tss_bed = pybedtools.BedTool(os.path.join(OUTPUT_DIR, "tss_tmp.bed"))
    

        genes_near_peaks = utils.find_genes_near_peaks(peak_bed, tss_bed)

        # Restrict to peaks within 1 Mb of a gene TSS
        genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= 1e6]

        # Scale the TSS distance score by the exponential scaling factor
        genes_near_peaks = genes_near_peaks.copy()
        genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / 250000)

        genes_near_peaks.to_parquet(os.path.join(OUTPUT_DIR, "genes_near_peaks.parquet"), compression="snappy", engine="pyarrow")
    else:
        genes_near_peaks = pd.read_parquet(os.path.join(OUTPUT_DIR, "genes_near_peaks.parquet"), engine="pyarrow")
    
    return genes_near_peaks

def create_homer_peaks_file(genes_near_peaks):
    logging.info("Building Homer peaks file")
    homer_peaks = genes_near_peaks[["peak_id", "peak_chr", "peak_start", "peak_end"]]
    homer_peaks = homer_peaks.rename(columns={
        "peak_id":"PeakID", 
        "peak_chr":"chr",
        "peak_start":"start",
        "peak_end":"end"
        })
    homer_peaks["strand"] = ["."] * len(homer_peaks)
    homer_peaks["start"] = round(homer_peaks["start"].astype(int),0)
    homer_peaks["end"] = round(homer_peaks["end"].astype(int),0)
    homer_peaks = homer_peaks.drop_duplicates(subset="PeakID")

    os.makedirs(os.path.join(OUTPUT_DIR, "tmp"), exist_ok=True)
    homer_peak_path = os.path.join(OUTPUT_DIR, "tmp/homer_peaks.txt")
    homer_peaks.to_csv(homer_peak_path, sep="\t", header=False, index=False)

def load_homer_tf_to_peak_results():
    assert os.path.exists(os.path.join(OUTPUT_DIR, "homer_tf_to_peak.parquet")), \
        "ERROR: Homer TF to peak output parquet file required"
        
    homer_results = pd.read_parquet(os.path.join(OUTPUT_DIR, "homer_tf_to_peak.parquet"), engine="pyarrow")
    homer_results = homer_results.reset_index(drop=True)
    homer_results["source_id"] = homer_results["source_id"].str.capitalize()
    
    return homer_results

def find_shared_barcodes(atac_df, rna_df, num_cells):
    atac_cell_barcodes = atac_df.columns.to_list()
    rna_cell_barcodes = rna_df.columns.to_list()
    atac_in_rna_shared_barcodes = [i for i in atac_cell_barcodes if i in rna_cell_barcodes]

    # Make sure that the cell names are in the same order and in both datasets
    shared_barcodes = sorted(set(atac_in_rna_shared_barcodes))[:num_cells]

    atac_df_shared = atac_df[shared_barcodes]
    rna_df_shared = rna_df[shared_barcodes]
    
    return shared_barcodes, atac_df_shared, rna_df_shared

def get_unique_tfs_peaks_genes(homer_results, genes_near_peaks):  
    tfs   = homer_results["source_id"].astype(str).str.capitalize().unique()
    
    peaks = np.unique(np.concatenate([
        homer_results["peak_id"].astype(str).values,
        genes_near_peaks["peak_id"].astype(str).values
    ]))
    
    genes = genes_near_peaks["target_id"].astype(str).unique()
    
    return tfs, peaks, genes

def create_tf_peak_gene_mapping_dicts(tfs, peaks, genes):
    tf_i   = {t:i for i,t in enumerate(tfs)}
    peak_i = {p:i for i,p in enumerate(peaks)}
    gene_i = {g:i for i,g in enumerate(genes)}
    
    return tf_i, peak_i, gene_i

def cast_homer_tf_to_peak_df_sparse(homer_results, tf_i, peak_i):
    homer_results = homer_results[["source_id","peak_id","homer_binding_score"]].copy()
    homer_results["source_id"] = homer_results["source_id"].astype(str).str.capitalize().map(tf_i)
    homer_results["peak_id"]   = homer_results["peak_id"].astype(str).map(peak_i)
    homer_results = homer_results.dropna(subset=["source_id","peak_id"])
    homer_tf_peak_sparse = sparse.coo_matrix(
        (homer_results["homer_binding_score"].astype(np.float32).values,
        (homer_results["source_id"].astype(int).values, homer_results["peak_id"].astype(int).values)),
        shape=(len(tf_i), len(peak_i))
    ).tocsr()

    return homer_tf_peak_sparse

def cast_peak_to_tg_distance_sparse(genes_near_peaks, peak_i, gene_i):
    genes_near_peaks = genes_near_peaks[["peak_id","target_id","TSS_dist_score"]].copy()
    genes_near_peaks["peak_id"]   = genes_near_peaks["peak_id"].astype(str).map(peak_i)
    genes_near_peaks["target_id"] = genes_near_peaks["target_id"].astype(str).map(gene_i)
    genes_near_peaks = genes_near_peaks.dropna(subset=["peak_id","target_id"])
    gene_distance_sparse = sparse.coo_matrix(
        (genes_near_peaks["TSS_dist_score"].astype(np.float32).values,
        (genes_near_peaks["peak_id"].astype(int).values, genes_near_peaks["target_id"].astype(int).values)),
        shape=(len(peak_i), len(gene_i))
    ).tocsr()
    
    return gene_distance_sparse

def assign_peaks_to_windows(mesc_atac_peak_loc_df, peaks, peak_i, windows_df, chrom_id):
    logging.info("Assigning peaks to windows")

    # Make a Series with peak start/end by peak_id
    peak_coord_df = mesc_atac_peak_loc_df.loc[mesc_atac_peak_loc_df["peak_id"].isin(peaks), ["peak_id","chrom","start","end"]].copy()
    peak_coord_df = peak_coord_df[peak_coord_df["chrom"] == chrom_id]
    coord_map = peak_coord_df.set_index("peak_id")[["start","end"]].to_dict(orient="index")

    w = int((windows_df["end"] - windows_df["start"]).mode().iloc[0])
    win_lut = {}  # window_idx -> window_id string
    for _, row in windows_df.iterrows():
        k = row["start"] // w
        win_lut[k] = f'{row["chrom"]}:{row["start"]}-{row["end"]}'
    nW = len(win_lut)

    rng = np.random.default_rng(0)
    def assign_best_window(start, end, w):
        i0 = start // w
        i1 = (end - 1) // w
        if i1 < i0: i1 = i0
        best_k = i0
        best_ov = -1
        ties = []
        for k in range(i0, i1 + 1):
            bs, be = k * w, (k + 1) * w
            ov = max(0, min(end, be) - max(start, bs))
            if ov > best_ov:
                best_ov, best_k = ov, k
                ties = [k]
            elif ov == best_ov:
                ties.append(k)
        return best_k if len(ties) == 1 else int(rng.choice(ties))

    # map each peak to a window_idx (or -1 if we lack coords)
    peak_to_window_idx = np.full(len(peaks), -1, dtype=np.int32)
    for p, idx in peak_i.items():
        info = coord_map.get(p)
        if info is None: 
            continue
        peak_to_window_idx[idx] = assign_best_window(int(info["start"]), int(info["end"]), w)

    # build list of peak indices per window
    peaks_by_window = [np.where(peak_to_window_idx == k)[0] for k in range(nW)]
    window_ids = np.array([win_lut[k] for k in range(nW)], dtype=object)
    
    return peaks_by_window, window_ids

def build_cell_features(
    barcode: str,
    cell_col_idx: dict[str,int],
    tf_expr_arr: np.ndarray,           # [TF, n_cells] float32
    peak_acc_arr: np.ndarray,          # [P,  n_cells] float32
    homer_tf_peak_sparse: sps.csr_matrix,  # [TF,P]
    gene_distance_sparse: sps.csr_matrix,  # [P,G]
    peaks_by_window: list[np.ndarray],     # list of peak idx arrays
    clamp_val: float = 3.0,
):
    col = cell_col_idx[barcode]
    tf_expr  = tf_expr_arr[:, col]            # [TF]
    peak_acc = peak_acc_arr[:, col]           # [P]

    # scale TF-peak by TF expression (CSR * diag)
    tf_peak_binding = homer_tf_peak_sparse.multiply(tf_expr[:, None]).tocsr()

    tf_windows = []
    gene_biases = []
    for peak_idx in peaks_by_window:
        if peak_idx.size == 0:
            continue
        active = peak_idx[peak_acc[peak_idx] > 0]
        if active.size == 0:
            continue

        # TF window vector
        w = peak_acc[active]
        tf_win = (tf_peak_binding[:, active] @ w).astype(np.float32)  # [TF]
        m, s = tf_win.mean(), tf_win.std() + 1e-6
        tf_win = np.clip((tf_win - m) / s, -clamp_val, clamp_val)

        # Gene×window bias (max across peaks)
        pg_sub = gene_distance_sparse[active, :]
        
        reduced  = pg_sub.max(axis=0)
        
        if sp.issparse(reduced):
            bias = reduced.toarray().ravel().astype(np.float32)
        else:
            bias = np.asarray(reduced).ravel().astype(np.float32)
            
        tf_windows.append(tf_win)
        gene_biases.append(bias)

    if len(tf_windows) == 0:
        return None  # caller will decide how to record empty-cell

    tf_windows = np.stack(tf_windows, axis=0)   # [W', TF]
    gene_biases = np.stack(gene_biases, axis=0) # [W', G]
    return tf_windows, gene_biases

def process_and_save_dataset(
    *,
    OUTPUT_DIR: str,
    chrom_id: str,
    window_size: int,
    tfs: np.ndarray,
    peaks: np.ndarray,
    genes: np.ndarray,
    tf_i: dict[str,int],
    peak_i: dict[str,int],
    gene_i: dict[str,int],
    peaks_by_window: list[np.ndarray],
    window_ids: np.ndarray,
    homer_tf_peak_sparse: sps.csr_matrix,
    gene_distance_sparse: sps.csr_matrix,
    shared_barcodes: list[str],
    tf_expr_arr: np.ndarray,
    peak_acc_arr: np.ndarray,
    cell_col_idx: dict[str,int],
    n_jobs: int = 8,
):
    base = Path(OUTPUT_DIR) / "precomputed_dataset" / f"{chrom_id}_{window_size//1000}kb"
    cells_dir = base / "cells"
    meta_dir  = base / "meta"
    _ensure_dir(cells_dir); _ensure_dir(meta_dir)

    # ---- Save global objects ----
    save_sparse_npz(meta_dir / "homer_tf_peak_sparse.npz", homer_tf_peak_sparse)
    save_sparse_npz(meta_dir / "gene_distance_sparse.npz", gene_distance_sparse)

    # windows & mappings
    # store peaks_by_window as a single .npz with one array per window
    np.savez_compressed(meta_dir / "peaks_by_window.npz", **{f"w{i}": arr for i,arr in enumerate(peaks_by_window)})
    save_numpy(meta_dir / "window_ids.npy", window_ids)
    save_json(meta_dir / "tf_i.json", tf_i)
    save_json(meta_dir / "peak_i.json", peak_i)
    save_json(meta_dir / "gene_i.json", gene_i)
    save_numpy(meta_dir / "tfs.npy", np.array(tfs, dtype=object))
    save_numpy(meta_dir / "genes.npy", np.array(genes, dtype=object))
    save_numpy(meta_dir / "peaks.npy", np.array(peaks, dtype=object))

    # metadata.json
    meta = dict(
        chrom_id=chrom_id,
        window_size=window_size,
        n_tf=len(tfs),
        n_genes=len(genes),
        n_peaks=len(peaks),
        n_windows=len(peaks_by_window),
        n_cells=len(shared_barcodes),
    )
    save_json(meta_dir / "metadata.json", meta)

    # ---- Per-cell features (parallel) ----
    def _one(barcode: str):
        out_path = cells_dir / f"{barcode}.npz"
        res = build_cell_features(
            barcode, cell_col_idx, tf_expr_arr, peak_acc_arr,
            homer_tf_peak_sparse, gene_distance_sparse, peaks_by_window
        )
        if res is None:
            save_npz(out_path, tf_windows=np.zeros((0, len(tfs)), np.float32),
                               gene_biases=np.zeros((0, len(genes)), np.float32),
                               wlen=np.array(0, dtype=np.int32),
                               barcode=np.array(barcode, dtype=object))
            return dict(barcode=barcode, path=str(out_path), wlen=0)
        tf_windows, gene_biases = res
        save_npz(out_path, tf_windows=tf_windows, gene_biases=gene_biases,
                           wlen=np.array(tf_windows.shape[0], dtype=np.int32),
                           barcode=np.array(barcode, dtype=object))
        return dict(barcode=barcode, path=str(out_path), wlen=int(tf_windows.shape[0]))

    logging.info("Precomputing per-cell features...")
    rows = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_one)(b) for b in shared_barcodes)
    manifest = pd.DataFrame(rows).sort_values("barcode").reset_index(drop=True)
    manifest.to_parquet(meta_dir / "manifest.parquet", index=False)
    logging.info(f"Saved {len(manifest)} cells to {cells_dir}")


def main():
    logging.info("Reading processed scATAC-seq dataset")
    
    
    mesc_atac_data, mesc_atac_peak_loc_df = load_atac_dataset(atac_data_filename, chrom_id)
    mesc_rna_data = load_rna_data(rna_data_filename)

    gene_tss_df = load_or_create_gene_tss_df(chrom_id, force_recalculate=force_recalculate)
    logging.info(f"Using {gene_tss_df['name'].nunique()} genes (TSS df)")
    
    # Restrict gene TSS dataframe to only use the selected chromosome
    gene_tss_df = gene_tss_df[gene_tss_df["chrom"] == chrom_id]
    
    mesc_rna_data = mesc_rna_data[mesc_rna_data.index.isin(gene_tss_df["name"])]
    logging.info(f"Using {mesc_rna_data.index.nunique()} genes (scRNA-seq data df)")
    
    mm10_windows = create_or_load_genomic_windows(chrom_id, force_recalculate=force_recalculate)

    genes_near_peaks = calculate_peak_to_tg_distance_score(mesc_atac_peak_loc_df, gene_tss_df, force_recalculate=force_recalculate)
    
    # Restrict the genes near peaks dataframe to only using TGs from genes on one chromosome
    genes_near_peaks = genes_near_peaks[(genes_near_peaks["gene_chr"] == chrom_id) & (genes_near_peaks["peak_chr"] == chrom_id)]
    logging.info(f"Using {genes_near_peaks['target_id'].nunique()} genes (genes_near_peaks_df)")

    if not os.path.isfile(os.path.join(OUTPUT_DIR, "tmp/homer_peaks.txt")):
        create_homer_peaks_file(genes_near_peaks)
        
    homer_results = load_homer_tf_to_peak_results()
    
    top_50_expressed_genes = np.load(os.path.join(OUTPUT_DIR, "top_50_expressed_chr19_genes.npy"), allow_pickle=True)
    genes_near_peaks = genes_near_peaks[genes_near_peaks["target_id"].isin(top_50_expressed_genes)]

    tfs, peaks, genes = get_unique_tfs_peaks_genes(homer_results, genes_near_peaks)
    tf_i, peak_i, gene_i = create_tf_peak_gene_mapping_dicts(tfs, peaks, genes)
    
    logging.info("Preparing sparse components (TF×Peak and Peak×Gene)")
    homer_tf_peak_sparse = cast_homer_tf_to_peak_df_sparse(homer_results, tf_i, peak_i)
    gene_distance_sparse = cast_peak_to_tg_distance_sparse(genes_near_peaks, peak_i, gene_i)
    peaks_by_window, window_ids = assign_peaks_to_windows(mesc_atac_peak_loc_df, peaks, peak_i, mm10_windows, chrom_id)
        
    shared_barcodes, mesc_atac_data_shared, mesc_rna_data_shared = find_shared_barcodes(mesc_atac_data, mesc_rna_data, num_cells)

    rna_tfs  = mesc_rna_data_shared.reindex(index=tfs).fillna(0).astype("float32")          # rows: TFs, cols: cells
    atac_peaks  = mesc_atac_data_shared.reindex(index=peaks).fillna(0).astype("float32")   # rows: peaks, cols: cells
    
    assert set(rna_tfs.columns) == set(atac_peaks.columns), "RNA/ATAC barcode sets differ"
    
    tf_expr_arr  = rna_tfs.values          # shape [TF, n_cells]
    peak_acc_arr = atac_peaks.values          # shape [P,  n_cells]
    cell_col_idx   = {c:i for i,c in enumerate(rna_tfs.columns)}
    
    process_and_save_dataset(
        OUTPUT_DIR=OUTPUT_DIR,
        chrom_id=chrom_id,
        window_size=window_size,
        tfs=tfs, peaks=peaks, genes=genes,
        tf_i=tf_i, peak_i=peak_i, gene_i=gene_i,
        peaks_by_window=peaks_by_window,
        window_ids=window_ids,
        homer_tf_peak_sparse=homer_tf_peak_sparse,
        gene_distance_sparse=gene_distance_sparse,
        shared_barcodes=shared_barcodes,
        tf_expr_arr=tf_expr_arr.astype(np.float32, copy=False),
        peak_acc_arr=peak_acc_arr.astype(np.float32, copy=False),
        cell_col_idx=cell_col_idx,
        n_jobs=8,
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    main()    