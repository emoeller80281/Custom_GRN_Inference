import os
import gzip
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
import pyranges as pr
import scipy.io
from pybiomart import Server
from scipy.io import mmread

# ──────────────────────────────────────────────────────────────────────────────
# 1) scATAC‐seq: Load, filter peaks by TSS, compute QC metrics
# ──────────────────────────────────────────────────────────────────────────────

def load_adata_from_tsv_file(tsv_path: str):
    df = pd.read_csv(tsv_path, sep="\t", header=0, index_col=0)
    counts = csr_matrix(df.values)
    gene_or_peak_names = df.index.to_list()
    cell_names = df.columns.to_list()
    
    adata = sc.AnnData(counts)
    adata.obs_names = gene_or_peak_names
    adata.var_names = cell_names
    
    return adata



def load_scATAC_10x_matrix_full(matrix_mtx_path: str,
                                barcodes_tsv_path: str,
                                orig_peaks_bed: str) -> sc.AnnData:
    """
    Load the full 10× scATAC peak×cell matrix into AnnData.
    - matrix_mtx_path: path to matrix.mtx.gz (peaks × cells)
    - barcodes_tsv_path: path to barcodes.tsv.gz
    - orig_peaks_bed: path to the *original* peaks.bed or gzipped BED
      (must match the rows of matrix.mtx)
    Returns AnnData with X = (cells × all_peaks), var_names = "chr:start-end".
    """
    # 1) Read the sparse matrix (n_peaks × n_cells)
    mtx = mmread(matrix_mtx_path).tocsc()
    
    # 2) Load barcodes
    with gzip.open(barcodes_tsv_path, "rt") as f:
        barcodes = [line.strip() for line in f]
    
    # 3) Load original peaks BED to get ordering
    compression = "gzip" if orig_peaks_bed.endswith(".gz") else None
    peaks_df = pd.read_csv(
        orig_peaks_bed,
        sep="\t",
        header=None,
        names=["chrom", "start", "end"],
        compression=compression
    )
    peak_names = (
        peaks_df["chrom"].astype(str)
        + ":"
        + peaks_df["start"].astype(str)
        + "-"
        + peaks_df["end"].astype(str)
    )
    
    # 4) Transpose to (n_cells × n_peaks)
    X = mtx.transpose().tocsr()
    
    # 5) Build AnnData with all peaks
    adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=peak_names)
    )
    adata.var["chrom"] = peaks_df["chrom"].values
    adata.var["start"] = peaks_df["start"].values
    adata.var["end"] = peaks_df["end"].values
    
    return adata


def load_ensembl_tss(organism: str, out_parquet: str) -> None:
    """
    Query Ensembl BioMart for TSS of all protein‐coding genes in 'organism'
    (e.g. "mmusculus" for mouse → dataset "mmusculus_gene_ensembl").
    Writes a Parquet file at out_parquet with columns: ['chr','start','end','gene_id'].
    """
    server = Server(host="http://www.ensembl.org")
    dataset_name = f"{organism}_gene_ensembl"
    mart = server["ENSEMBL_MART_ENSEMBL"]
    ds = mart[dataset_name]
    
    # Query external_gene_name (symbol), strand, chr, TSS
    df = ds.query(
        attributes=[
            "external_gene_name",
            "strand",
            "chromosome_name",
            "transcription_start_site"
        ]
    )
    df.rename(
        columns={
            "Chromosome/scaffold name": "chr",
            "Transcription start site (TSS)": "tss",
            "Gene name": "gene_id"
        },
        inplace=True
    )
    df["tss"] = df["tss"].astype(int)
    df["start"] = df["tss"]
    df["end"] = df["tss"] + 1
    df = df[["chr", "start", "end", "gene_id"]].copy()
    df["chr"] = df["chr"].astype(str)
    df["gene_id"] = df["gene_id"].astype(str)
    
    df.to_parquet(out_parquet, engine="pyarrow", index=False, compression="snappy")
    print(f"Written Ensembl TSS to {out_parquet}")


def filter_peaks_by_tss(peaks_bed: str,
                        tss_parquet: str,
                        output_bed: str,
                        window_size: int = 1000) -> None:
    """
    Keep only those peaks from peaks_bed that overlap ±window_size bp around any TSS.
    - peaks_bed: path to original peaks.bed or peaks.bed.gz
    - tss_parquet: Parquet from load_ensembl_tss (columns: chr,start,end,gene_id)
    - output_bed: write filtered BED (three cols: chr,start,end)
    - window_size: half‐window around TSS in bp (default 1000 for ±1kb)
    """
    # Load TSS from Parquet into PyRanges
    tss_df = pd.read_parquet(tss_parquet)
    # Add "chr" prefix so that names match peaks_bed
    tss_df["Chromosome"] = "chr" + tss_df["chr"].astype(str)
    tss_df["Start"] = tss_df["start"]
    tss_df["End"] = tss_df["end"]
    tss_pr = pr.PyRanges(tss_df[["Chromosome", "Start", "End", "gene_id"]])
    
    # Expand TSS to ± window_size
    promoter_pr = tss_pr.slack(window_size)
    
    # Load peaks as a DataFrame, then into PyRanges
    compression = "gzip" if peaks_bed.endswith(".gz") else None
    peaks_df = pd.read_csv(
        peaks_bed,
        sep="\t",
        header=None,
        names=["Chromosome", "Start", "End"],
        compression=compression
    )
    peaks_pr = pr.PyRanges(peaks_df)
    
    # Use 'overlap' to keep only peaks overlapping promoter windows
    peaks_filtered = peaks_pr.overlap(promoter_pr)
    df = peaks_filtered.df
    
    # Take first three columns (chr, start, end) by position
    df_out = df.iloc[:, :3].copy()
    df_out.columns = ["Chromosome", "Start", "End"]
    
    # Write filtered peaks
    df_out.to_csv(output_bed, sep="\t", header=False, index=False)
    print(f"Filtered peaks written to {output_bed} (kept {df_out.shape[0]} peaks)")


def compute_total_fragments_per_cell(fragments_tsv: str,
                                     valid_barcodes: set) -> Counter:
    """
    Read fragments.tsv.gz, count total fragments per barcode (if barcode ∈ valid_barcodes).
    Returns Counter({barcode: count}).
    """
    counts = Counter()
    with gzip.open(fragments_tsv, "rt") as f:
        for line in f:
            chrom, start, end, barcode, *rest = line.strip().split("\t")
            if barcode in valid_barcodes:
                counts[barcode] += 1
    return counts


def compute_scATAC_qc(adata: sc.AnnData,
                      fragments_tsv: str,
                      min_peaks: int = 2000,
                      max_peaks: int = 40000,
                      min_frip: float = 0.3) -> sc.AnnData:
    """
    Given an AnnData for scATAC (adata.X = raw counts cells×peaks),
    compute per‐cell QC metrics: n_peaks (nonzero), fragments_in_peaks, total_fragments_raw, FRiP.
    Then filter cells by thresholds and return filtered AnnData.
    - min_fragments: minimum total fragments (raw) to keep a cell.
    - min_peaks: minimum peaks detected to keep a cell.
    - min_frip: minimum FRiP to keep a cell.
    """
    # 1) # peaks per cell (nonzero columns) and fragments in peaks
    adata.obs["n_peaks"] = np.array((adata.X > 0).sum(axis=1)).ravel()
    adata.obs["fragments_in_peaks"] = np.array(adata.X.sum(axis=1)).ravel()
    
    # 2) total raw fragments per cell (from fragments.tsv.gz)
    barcodes = set(adata.obs_names)
    frag_counts = compute_total_fragments_per_cell(fragments_tsv, barcodes)
    adata.obs["total_fragments_raw"] = adata.obs_names.map(lambda bc: frag_counts.get(bc, 0))
    
    # 3) FRiP = fragments_in_peaks / total_fragments_raw
    adata.obs["FRiP"] = adata.obs["fragments_in_peaks"] / adata.obs["total_fragments_raw"]
    
    sc.pl.violin(adata, ["n_peaks", "FRiP"], jitter=0.4, multi_panel=True)
    
    # 4) Filter cells by QC thresholds
    keep_cells = (
        (adata.obs["n_peaks"] >= min_peaks) &
        (adata.obs["n_peaks"] <= max_peaks) &
        (adata.obs["FRiP"] >= min_frip)
    )
    print(f"Keeping {keep_cells.sum()} cells out of {adata.n_obs}")
    return adata[keep_cells].copy()


# ──────────────────────────────────────────────────────────────────────────────
# 2) scRNA‐seq: Load, QC, filter, normalize
# ──────────────────────────────────────────────────────────────────────────────

def process_scRNA_10x(input_dir: str,
                      prefix: str = "",
                      min_genes: int = 1000,
                      max_genes: int = 10000,
                      max_pct_mt: float = 20.0) -> sc.AnnData:
    """
    Load a 10× Genomics scRNA‐seq dataset with muon.read_10x_mtx (var_names="gene_symbols"),
    compute QC metrics (mitochondrial, ribosomal, hemoglobin), then filter:
      - cells: min_genes <= n_genes_by_counts <= max_genes, pct_counts_mt < max_pct_mt
      - genes expressed in < min_cells_per_gene cells
    Next, normalize total counts to 1e6 and log1p. Returns filtered AnnData.
    """
    # 1) Load with Muon
    rna_adata = sc.read_10x_mtx(
        path=input_dir,
        var_names="gene_symbols",
        make_unique=True,
        prefix=prefix
    )

    # 2) Annotate mitochondrial, ribosomal, hemoglobin genes
    rna_adata.var["mt"] = rna_adata.var_names.str.startswith("mt-")
    rna_adata.var["ribo"] = rna_adata.var_names.str.startswith(("Rps", "Rpl"))
    rna_adata.var["hb"] = rna_adata.var_names.str.contains(r"^Hb[^(P)]")

    # 3) Compute QC metrics
    sc.pp.calculate_qc_metrics(
        rna_adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        log1p=False
    )
    
    sc.pl.violin(rna_adata, ["n_genes_by_counts", "n_genes_by_counts", "pct_counts_mt"], jitter=0.4, multi_panel=True)

    # 4) Filter cells by mitochondrial percentage and gene counts
    cell_mask = (
        (rna_adata.obs["n_genes_by_counts"] >= min_genes)
        & (rna_adata.obs["n_genes_by_counts"] <= max_genes)
        & (rna_adata.obs["pct_counts_mt"] < max_pct_mt)
    )
    rna_adata = rna_adata[cell_mask].copy()

    return rna_adata

# ──────────────────────────────────────────────────────────────────────────────
# 2) scRNA‐seq: Read R script output (TSV), no longer raw 10× files
# ──────────────────────────────────────────────────────────────────────────────

def load_scRNA_from_tsv(tsv_file: str) -> pd.DataFrame:
    """
    Load the gene × cell matrix produced by the R script (genes as rows, cells as columns).
    Returns a DataFrame indexed by gene and with columns = cell barcodes.
    """
    df = pd.read_csv(tsv_file, sep="\t", header=0, index_col=0)
    print(f"Loaded scRNA matrix from TSV: {df.shape[0]} genes × {df.shape[1]} cells")
    return df



# ──────────────────────────────────────────────────────────────────────────────
# Main execution: specify directories, run filtering and processing
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sc.settings.set_figure_params(dpi=50, facecolor="white")
    
    # Base input directory (modify as needed)
    input_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input"
    data_input_dir = os.path.join(input_dir, "DS010_PMID34579774_MOUSE_EB")
    rna_tsv_file = os.path.join(data_input_dir,  "scRNA_Expression.EB.tsv")
    atac_tsv_file = os.path.join(data_input_dir,  "scATAC_PeakMatrix.EB.txt")
    
    rna_adata = load_adata_from_tsv_file(rna_tsv_file)
    atac_adata = load_adata_from_tsv_file(atac_tsv_file)
    
    # 1) Fetch Ensembl TSS and filter peaks ±1kb
    tss_parquet = os.path.join(input_dir, "ensembl_tss.parquet")
    load_ensembl_tss("mmusculus", tss_parquet)
    
    raw_peaks_bed = os.path.join(data_input_dir, "GSE198730_Naive_ESC_scATAC_rep1_peaks.bed.gz")
    filtered_peaks_bed = os.path.join(data_input_dir, "peaks_within_±1kb_of_TSS.bed")
    filter_peaks_by_tss(
        peaks_bed=raw_peaks_bed,
        tss_parquet=tss_parquet,
        output_bed=filtered_peaks_bed,
        window_size=1000
    )
    
    filtered_df = pd.read_csv(
        filtered_peaks_bed,
        sep="\t",
        header=None,
        names=["chrom", "start", "end"]
    )
    filtered_peak_names = (
        filtered_df["chrom"].astype(str)
        + ":"
        + filtered_df["start"].astype(str)
        + "-"
        + filtered_df["end"].astype(str)
    )

    # 2) Load the full scATAC AnnData using the ORIGINAL peaks file:
    orig_peaks_bed = os.path.join(data_input_dir, "GSE198730_Naive_ESC_scATAC_rep1_peaks.bed.gz")
    full_atac = load_scATAC_10x_matrix_full(
        matrix_mtx_path=os.path.join(data_input_dir, "GSE198730_Naive_ESC_scATAC_rep1_matrix.mtx.gz"),
        barcodes_tsv_path=os.path.join(data_input_dir, "GSE198730_Naive_ESC_scATAC_rep1_barcodes.tsv.gz"),
        orig_peaks_bed=orig_peaks_bed
    )

    # 3) Subset to only those peaks retained by filter_peaks_by_tss
    atac_adata = full_atac[:, full_atac.var_names.isin(filtered_peak_names)].copy()

    print(f"Loaded scATAC: {atac_adata.n_obs} cells × {atac_adata.n_vars} filtered peaks")
    
    # 3) Compute and apply scATAC QC
    fragments_tsv = os.path.join(data_input_dir, "GSE198730_Naive_ESC_scATAC_rep1_fragments.tsv.gz")
    atac_adata = compute_scATAC_qc(
        adata=atac_adata,
        fragments_tsv=fragments_tsv,
        min_peaks = 500,
        max_peaks = 40000,
        min_frip = 0.1
    )
    print(f"After QC, scATAC: {atac_adata.n_obs} cells × {atac_adata.n_vars} peaks")
    
    # 5) Process scRNA
    rna_adata = process_scRNA_10x(
        input_dir=data_input_dir,
        prefix="GSE198730_Naive_ESC_scRNA_rep1_",
        min_genes=1000,
        max_genes=10000,
        max_pct_mt=20.0
    )
    print(f"Processed scRNA: {rna_adata.n_obs} cells × {rna_adata.n_vars} genes")
    
    # # 1) Find the intersection of barcode lists
    # common_cells = atac_adata.obs_names.intersection(rna_adata.obs_names)

    # # 2) Subset each AnnData to only those cells
    # atac_adata = atac_adata[common_cells].copy()
    # rna_adata = rna_adata[common_cells].copy()

    # print(f"Shared cells: {len(common_cells)}")
    # print(f"  atac now: {atac_adata.n_obs} cells × {atac_adata.n_vars} peaks")
    # print(f"  rna  now: {rna_adata.n_obs} cells × {rna_adata.n_vars} genes")
    
    # 1) Export scRNA (genes × cells) to Parquet
    #    Transpose X (cells × genes) → (genes × cells)
    rna_df = pd.DataFrame(
        rna_adata.X.T.toarray(),
        index=rna_adata.var_names,
        columns=rna_adata.obs_names
    )
    rna_df.to_parquet(os.path.join(input_dir, "DS011_mESC/DS011_mESC_sample1/filtered_scRNA_genes_by_cells.parquet"))

    # 2) Export scATAC (peaks × cells) to Parquet
    #    Transpose X (cells × peaks) → (peaks × cells)
    atac_df = pd.DataFrame(
        atac_adata.X.T.toarray(),
        index=atac_adata.var_names,
        columns=atac_adata.obs_names
    )
    atac_df.to_parquet(os.path.join(input_dir, "DS011_mESC/DS011_mESC_sample1/filtered_scATAC_peaks_by_cells.parquet"))

    print("Wrote:")
    print(" - filtered_scRNA_genes_by_cells.parquet")
    print(" - filtered_scATAC_peaks_by_cells.parquet")
