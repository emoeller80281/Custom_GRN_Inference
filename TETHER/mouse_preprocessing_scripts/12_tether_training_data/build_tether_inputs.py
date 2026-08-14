"""Turn the Argelaguet et al. 2022 metacells into TETHER sample inputs.

Writes the three files ``scripts/build_tf_to_tg_train_data.py`` expects into
``data/sample_input_data/mESC/WT_timecourse_metacells/``:

    RE_pseudobulk.parquet     peak x metacell scATAC-seq   (TF-IDF, log1p, per-peak z-score)
    TG_pseudobulk.parquet     gene x metacell scRNA-seq    (CP10K, log1p, per-gene z-score)
    peak_to_gene_dist.parquet peak -> gene links within 100 kb of a TSS

Inputs
------
RNA metacells come straight from the paper's AnnData
(``results/rna/metacells/all_cells/rna_atac/anndata_metacells.h5ad``). ATAC metacells
come from ``export_atac_metacells.R``, which lifts the ArchR SummarizedExperiment out
of R and into HDF5 -- run that first.

Both matrices are keyed by the same ``<sample>#<barcode>`` metacell IDs, so the columns
are matched nuclei: the columns kept here are the intersection of the two.

Normalisation note
------------------
The existing mESC samples were built by diffusion-smoothing single cells, which leaves
each feature centred but with std well below 1 (~0.23 ATAC / ~0.39 RNA). These are real
aggregated metacells, so there is no smoothing step to shrink the variance; features are
z-scored to unit variance instead. Same centring, cleaner scale.
"""

import argparse
import logging
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pybedtools
import scanpy as sc

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
sys.path.append(str(PROJECT_DIR / "TETHER"))

DATA_DIR = PROJECT_DIR / "data"
EXTRACT = DATA_DIR / "dropbox_data" / "extracted"
SAMPLE_DIR = DATA_DIR / "sample_input_data" / "mESC" / "WT_timecourse_metacells"

RNA_METACELLS = EXTRACT / "results/rna/metacells/all_cells/rna_atac/anndata_metacells.h5ad"
ATAC_METACELLS = SAMPLE_DIR / "atac_metacell_counts.h5"
TSS_BED = DATA_DIR / "genome_data/genome_annotation/mm10/gene_tss.bed"

MAX_PEAK_DISTANCE = 100_000
TSS_DECAY = 20_000          # TSS_dist_score = exp(-dist / TSS_DECAY), matching existing samples
VALID_CHROMS = {f"chr{i}" for i in range(1, 20)} | {"chrX", "chrY"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def zscore_rows(mat: np.ndarray) -> np.ndarray:
    """Standardise each feature (row) across metacells. Constant rows are left at 0."""
    mu = mat.mean(axis=1, keepdims=True)
    sd = mat.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (mat - mu) / sd


def load_rna(metacells: list[str] | None) -> pd.DataFrame:
    """CP10K -> log1p -> per-gene z-score, returned as genes x metacells."""
    a = ad.read_h5ad(RNA_METACELLS)
    logging.info(f"RNA metacells loaded: {a.n_obs:,} x {a.n_vars:,}")

    if metacells is not None:
        a = a[metacells].copy()

    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    x = a.X
    x = np.asarray(x.todense() if hasattr(x, "todense") else x, dtype=np.float32)

    df = pd.DataFrame(x.T, index=a.var_names.astype(str), columns=a.obs_names.astype(str))

    # TETHER matches TF/TG names upper-cased everywhere; collapse duplicates that
    # only differ by case before standardising so each gene appears once.
    df.index = df.index.str.replace(r"\.\d+$", "", regex=True).str.upper()
    df = df.groupby(level=0).sum()

    df.loc[:, :] = zscore_rows(df.to_numpy(dtype=np.float32))
    logging.info(f"RNA pseudobulk: {df.shape[0]:,} genes x {df.shape[1]:,} metacells")
    return df.astype(np.float32)


def load_atac(metacells: list[str] | None) -> pd.DataFrame:
    """TF-IDF -> log1p -> per-peak z-score, returned as peaks x metacells."""
    with h5py.File(ATAC_METACELLS, "r") as f:
        peak_ids = np.array([p.decode() for p in f["peak_ids"][:]])
        mc_ids = np.array([m.decode() for m in f["metacell_ids"][:]])
        # rhdf5 writes R's peaks x metacells as metacells x peaks in HDF5 row-major order.
        counts = f["counts"][:].astype(np.float32)

    logging.info(f"ATAC metacells loaded: {counts.shape[0]:,} metacells x {counts.shape[1]:,} peaks")

    if metacells is not None:
        pos = {m: i for i, m in enumerate(mc_ids)}
        idx = [pos[m] for m in metacells]
        counts = counts[idx]
        mc_ids = mc_ids[idx]

    # TF-IDF, matching muon's ac.pp.tfidf(scale_factor=1e4): term frequency per metacell,
    # inverse document frequency per peak, then log1p of the scaled product.
    depth = counts.sum(axis=1, keepdims=True)
    depth[depth == 0] = 1.0
    tf = counts / depth

    n_cells = counts.shape[0]
    n_open = (counts > 0).sum(axis=0)
    idf = n_cells / np.maximum(n_open, 1)

    x = np.log1p(tf * idf * 1e4)
    del counts, tf

    df = pd.DataFrame(x.T, index=peak_ids, columns=mc_ids)
    del x

    # Standard chromosomes only -- ArchR keeps a few scaffolds that have no genome FASTA
    # entry downstream, and build_tf_to_tg_train_data.py filters them out anyway.
    keep = df.index.to_series().str.split(":", n=1).str[0].isin(VALID_CHROMS)
    logging.info(f"Peaks on standard chromosomes: {keep.sum():,} / {len(keep):,}")
    df = df.loc[keep.to_numpy()]

    df.loc[:, :] = zscore_rows(df.to_numpy(dtype=np.float32))
    logging.info(f"ATAC pseudobulk: {df.shape[0]:,} peaks x {df.shape[1]:,} metacells")
    return df.astype(np.float32)


def build_peak_to_gene(peak_ids: pd.Index) -> pd.DataFrame:
    """Link every peak to genes with a TSS within MAX_PEAK_DISTANCE."""
    parsed = pd.Series(peak_ids.astype(str)).str.extract(
        r"^(?P<chrom>chr[^\s:]+):(?P<start>\d+)-(?P<end>\d+)$"
    )
    if parsed.isna().any().any():
        bad = peak_ids[parsed["chrom"].isna()][:3].tolist()
        raise ValueError(f"Could not parse peak IDs, e.g. {bad}")

    peak_df = pd.DataFrame({
        "chrom": parsed["chrom"],
        "start": parsed["start"].astype(int),
        "end": parsed["end"].astype(int),
        "peak_id": peak_ids.astype(str),
    }).sort_values(["chrom", "start"])

    peak_bed = pybedtools.BedTool.from_dataframe(peak_df)
    tss_bed = pybedtools.BedTool(str(TSS_BED))

    cols = ["peak_chr", "peak_start", "peak_end", "peak_id",
            "gene_chr", "gene_start", "gene_end", "target_id"]
    hits = peak_bed.window(tss_bed, w=MAX_PEAK_DISTANCE).to_dataframe(
        names=cols, low_memory=False
    )

    for c in ["peak_start", "peak_end", "gene_start", "gene_end"]:
        hits[c] = pd.to_numeric(hits[c], errors="coerce")
    hits = hits.dropna(subset=["peak_start", "peak_end", "gene_start", "gene_end"])

    # Same distance definition as the existing samples: peak end to gene start.
    hits["TSS_dist"] = np.abs(hits["peak_end"].values - hits["gene_start"].values).astype(np.int64)
    hits = hits[hits["TSS_dist"] <= MAX_PEAK_DISTANCE]
    hits["target_id"] = hits["target_id"].astype(str).str.upper()
    hits["TSS_dist_score"] = np.exp(-hits["TSS_dist"] / TSS_DECAY)

    hits = hits.sort_values("TSS_dist").reset_index(drop=True)
    logging.info(
        f"peak-to-gene links: {len(hits):,} "
        f"({hits['peak_id'].nunique():,} peaks, {hits['target_id'].nunique():,} genes)"
    )
    return hits[cols + ["TSS_dist", "TSS_dist_score"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    re_file = SAMPLE_DIR / "RE_pseudobulk.parquet"
    tg_file = SAMPLE_DIR / "TG_pseudobulk.parquet"
    p2g_file = SAMPLE_DIR / "peak_to_gene_dist.parquet"

    if all(f.exists() for f in (re_file, tg_file, p2g_file)) and not args.force:
        logging.info("Outputs already exist. Use --force to rebuild.")
        return

    if not ATAC_METACELLS.exists():
        raise FileNotFoundError(
            f"{ATAC_METACELLS} not found -- run export_atac_metacells.R first "
            "(with the figr_env Rscript)."
        )

    # Columns are the metacells assayed in both modalities.
    with h5py.File(ATAC_METACELLS, "r") as f:
        atac_mcs = {m.decode() for m in f["metacell_ids"][:]}
    rna_mcs = set(ad.read_h5ad(RNA_METACELLS, backed="r").obs_names)

    common = sorted(atac_mcs & rna_mcs)
    logging.info(
        f"Metacells: {len(rna_mcs):,} RNA, {len(atac_mcs):,} ATAC, "
        f"{len(common):,} in both (these are the pseudobulk columns)"
    )
    if not common:
        raise ValueError("No metacells shared between the RNA and ATAC matrices.")

    rna_df = load_rna(common)
    rna_df.to_parquet(tg_file, engine="pyarrow", compression="snappy")
    logging.info(f"Wrote {tg_file}")
    del rna_df

    atac_df = load_atac(common)
    atac_df.to_parquet(re_file, engine="pyarrow", compression="snappy")
    logging.info(f"Wrote {re_file}")

    p2g = build_peak_to_gene(atac_df.index)
    del atac_df
    p2g.to_parquet(p2g_file, engine="pyarrow", compression="snappy", index=False)
    logging.info(f"Wrote {p2g_file}")


if __name__ == "__main__":
    main()
