import inspect
import os
import sys
import numpy as np
import pandas as pd
import duckdb
import pyfaidx
from pathlib import Path
import pysam
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Subset
import json

import config
import time
import requests
from Bio import Entrez, SeqIO
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
CHKPT_DIR = PROJECT_DIR / "checkpoints"
sys.path.append(str(PROJECT_DIR))

import models.tf_to_tg as tf_to_tg_module
import models.tf_to_dna as tf_to_dna_module
from scripts.train_tf_to_tg_model import TFTGEdgeBagDataset, collate_tftg_edge_bags

import logging
import warnings

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_GENOME_HANDLE = None

def parse_peak(peak):
    """
    Parse peak string like chr1:100-200.
    """
    chrom, coords = peak.split(":")
    start, end = coords.split("-")
    
    return chrom, int(start), int(end)


def onehot_dna_sequence(seq):
    """
    Fast one-hot encoding for DNA sequence.

    Returns
    -------
    np.ndarray
        Shape: (L, 4), dtype float32
    """
    
    # Fast reusable nucleotide lookup table
    _NUC_TO_IDX = np.full(256, -1, dtype=np.int16)
    _NUC_TO_IDX[ord("A")] = 0
    _NUC_TO_IDX[ord("C")] = 1
    _NUC_TO_IDX[ord("G")] = 2
    _NUC_TO_IDX[ord("T")] = 3
    _NUC_TO_IDX[ord("a")] = 0
    _NUC_TO_IDX[ord("c")] = 1
    _NUC_TO_IDX[ord("g")] = 2
    _NUC_TO_IDX[ord("t")] = 3
    
    seq_arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    idx = _NUC_TO_IDX[seq_arr]

    onehot = np.zeros((len(seq), 4), dtype=np.float32)
    valid = idx >= 0
    onehot[np.arange(len(seq))[valid], idx[valid]] = 1.0

    return onehot


def load_peak_sequence(genome_fasta, selected_peak):
    """
    Load the DNA sequence for a given peak.

    Parameters
    ----------
    genome_fasta : str | Path
        Path to the genome fasta file.
    selected_peak : str
        Peak string in the format "chrom:start-end".

    Returns
    -------
    str
        The DNA sequence for the peak.
    """
    peak_chrom, peak_start, peak_end = parse_peak(selected_peak)

    # Load peak sequence using the genome fasta file
    with pyfaidx.Fasta(genome_fasta) as genome:
        peak_sequence = genome[peak_chrom][peak_start:peak_end].seq.upper()
        
    return peak_sequence


def load_chrom_sizes(chromsizes_file):
    """
    Load chromosome sizes from a chrom.sizes file.

    Parameters
    ----------
    chromsizes_file : str | Path
        Path to the chrom.sizes file.

    Returns
    -------
    dict
        Dictionary mapping chromosome names to sizes.
    """
    chrom_sizes = {}
    
    with open(chromsizes_file, "r") as f:
        for line in f:
            chrom, size_str = line.strip().split("\t")
            chrom_sizes[chrom] = int(size_str)
    
    return chrom_sizes


def load_ground_truth(ground_truth_file: Path | str) -> pd.DataFrame:
    if isinstance(ground_truth_file, str):
        ground_truth_file = Path(ground_truth_file)

    logging.info(f"Loading ground truth file: {ground_truth_file.name}")

    if ground_truth_file.suffix == ".csv":
        sep = ","
    elif ground_truth_file.suffix == ".tsv":
        sep = "\t"

    ground_truth_df = pd.read_csv(ground_truth_file, sep=sep, on_bad_lines="skip", engine="python")

    if "chip" in ground_truth_file.name and "atlas" in ground_truth_file.name:
        ground_truth_df = ground_truth_df[["source_id", "target_id"]]

    if ground_truth_df.columns[0] != "Source" or ground_truth_df.columns[1] != "Target":
        ground_truth_df = ground_truth_df.rename(
            columns={ground_truth_df.columns[0]: "Source", ground_truth_df.columns[1]: "Target"}
        )
    ground_truth_df["Source"] = ground_truth_df["Source"].astype(str).str.capitalize()
    ground_truth_df["Target"] = ground_truth_df["Target"].astype(str).str.capitalize()

    return ground_truth_df[["Source", "Target"]].dropna()


def load_ground_truth_files(gt_path_list: list[Path]) -> pd.DataFrame:
    gt_dfs = [load_ground_truth(gt_path) for gt_path in gt_path_list]
    
    merged_gt_df = pd.concat(gt_dfs, ignore_index=True)
    
    merged_gt_df = merged_gt_df.drop_duplicates(subset=["Source", "Target"]).reset_index(drop=True)
    
    return merged_gt_df


def _centered_peak_to_onehot(
    peak_id: str,
    genome,
    chrom_sizes: dict[str, int],
    flank_size: int,
    dtype=np.uint8,
    pad_out_of_bounds: bool = True,
):
    """
    Encode one centered peak window into a one-hot DNA matrix.

    Returns
    -------
    np.ndarray
        Shape [2 * flank_size, 4]
    """
    chrom, peak_start, peak_end = parse_peak(peak_id)

    if chrom not in chrom_sizes:
        raise KeyError(
            f"Chromosome {chrom!r} not found in chrom_sizes. "
            f"Peak: {peak_id}"
        )

    chrom_size = chrom_sizes[chrom]
    seq_len = 2 * flank_size

    peak_center = (peak_start + peak_end) // 2
    seq_start = peak_center - flank_size
    seq_end = peak_center + flank_size

    fetch_start = seq_start
    fetch_end = seq_end

    left_pad = 0
    right_pad = 0

    if fetch_start < 0:
        left_pad = -fetch_start
        fetch_start = 0

    if fetch_end > chrom_size:
        right_pad = fetch_end - chrom_size
        fetch_end = chrom_size

    if fetch_end < fetch_start:
        fetch_end = fetch_start

    seq = genome[chrom][fetch_start:fetch_end].seq.upper()

    if pad_out_of_bounds:
        if left_pad:
            seq = ("N" * left_pad) + seq
        if right_pad:
            seq = seq + ("N" * right_pad)

        if len(seq) < seq_len:
            seq = seq + ("N" * (seq_len - len(seq)))
        elif len(seq) > seq_len:
            seq = seq[:seq_len]
    else:
        if len(seq) != seq_len:
            raise ValueError(
                f"Peak {peak_id} produced sequence length {len(seq)}, "
                f"but expected {seq_len}. Use pad_out_of_bounds=True "
                f"for fixed-length output."
            )

    onehot = onehot_dna_sequence(seq).astype(dtype, copy=False)

    if onehot.shape != (seq_len, 4):
        raise ValueError(
            f"Peak {peak_id} produced one-hot shape {onehot.shape}, "
            f"but expected {(seq_len, 4)}."
        )

    return onehot


def _init_genome_handle(genome_fasta: str) -> None:
    global _GENOME_HANDLE
    _GENOME_HANDLE = pyfaidx.Fasta(genome_fasta)


def _encode_peak_chunk(args):
    """
    Worker function for multiprocessing.

    Each worker opens the FASTA once per chunk, not once per peak.
    """
    (
        peak_chunk,
        genome_fasta,
        chrom_sizes,
        flank_size,
        dtype,
        pad_out_of_bounds,
    ) = args

    results = []

    genome = _GENOME_HANDLE

    for peak_id in peak_chunk:
        onehot = _centered_peak_to_onehot(
            peak_id=peak_id,
            genome=genome,
            chrom_sizes=chrom_sizes,
            flank_size=flank_size,
            dtype=dtype,
            pad_out_of_bounds=pad_out_of_bounds,
        )
        results.append((peak_id, onehot))

    return results


def _iter_chunks(items, chunk_size: int):
    """
    Yield lists of up to chunk_size items.
    """
    chunk = []

    for item in items:
        chunk.append(item)

        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk
        

def create_centered_peak_onehot_array(
    peak_ids: list[str],
    genome_fasta: str | Path,
    chrom_sizes: dict[str, int],
    peak_id_to_idx: dict[str, int],
    flank_size: int,
    dtype=np.uint8,
    pad_out_of_bounds: bool = True,
    show_progress: bool = True,
    num_workers: int = 1,
    chunk_size: int = 1000,
):
    """
    Create a stacked one-hot encoded DNA array using an existing peak_id_to_idx map.

    Parameters
    ----------
    peak_ids : list[str]
        Peak IDs to encode. These should all exist in peak_id_to_idx.
    genome_fasta : str | Path
        Path to genome FASTA.
    chrom_sizes : dict[str, int]
        Dictionary mapping chromosome names to chromosome sizes.
    peak_id_to_idx : dict[str, int]
        Existing mapping from peak_id -> row index.
    flank_size : int
        Number of bases on each side of the peak center.
        Output length is 2 * flank_size.
    dtype : numpy dtype
        Output dtype. np.uint8 is recommended for one-hot DNA.
    pad_out_of_bounds : bool
        Whether to pad with N if the requested window goes out of bounds.
    show_progress : bool
        Whether to show tqdm progress bar.
    num_workers : int
        Number of worker processes to use. Use 1 to run serially.
    chunk_size : int
        Number of peaks per worker task when num_workers > 1.

    Returns
    -------
    np.ndarray
        Array of shape [len(peak_id_to_idx), 2 * flank_size, 4].
    """
    genome_fasta = Path(genome_fasta)

    if not genome_fasta.exists():
        raise FileNotFoundError(f"Genome FASTA file not found: {genome_fasta}")

    if flank_size is None:
        raise ValueError("flank_size must be provided for a stacked array.")

    peak_ids = list(peak_ids)

    missing_peaks = [
        peak_id for peak_id in peak_ids
        if peak_id not in peak_id_to_idx
    ]

    if missing_peaks:
        raise KeyError(
            f"{len(missing_peaks)} peak_ids are missing from peak_id_to_idx. "
            f"Example: {missing_peaks[:5]}"
        )

    seq_len = 2 * flank_size
    num_output_peaks = len(peak_id_to_idx)
    num_encoded_peaks = len(peak_ids)

    peak_onehot_array = np.zeros(
        (num_output_peaks, seq_len, 4),
        dtype=dtype,
    )

    pbar_kwargs = dict(
        total=num_encoded_peaks,
        desc="One-hot peaks",
        disable=not show_progress,
        dynamic_ncols=True,
        miniters=max(num_encoded_peaks // 1000, 1),
    )

    if num_workers <= 1:
        with pyfaidx.Fasta(str(genome_fasta)) as genome:
            for peak_id in tqdm(peak_ids, **pbar_kwargs):
                peak_idx = peak_id_to_idx[peak_id]

                peak_onehot_array[peak_idx] = _centered_peak_to_onehot(
                    peak_id=peak_id,
                    genome=genome,
                    chrom_sizes=chrom_sizes,
                    flank_size=flank_size,
                    dtype=dtype,
                    pad_out_of_bounds=pad_out_of_bounds,
                )

    else:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")

        chunk_iter = _iter_chunks(peak_ids, chunk_size)

        task_iter = (
            (
                peak_chunk,
                genome_fasta,
                chrom_sizes,
                flank_size,
                dtype,
                pad_out_of_bounds,
            )
            for peak_chunk in chunk_iter
        )

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_genome_handle,
            initargs=(str(genome_fasta),),
        ) as executor:
            encoded_chunk_iter = executor.map(
                _encode_peak_chunk,
                task_iter,
                chunksize=1,
            )

            with tqdm(**pbar_kwargs) as pbar:
                for encoded_chunk in encoded_chunk_iter:
                    for peak_id, onehot in encoded_chunk:
                        peak_idx = peak_id_to_idx[peak_id]
                        peak_onehot_array[peak_idx] = onehot

                    pbar.update(len(encoded_chunk))

    return peak_onehot_array


def create_true_false_edges(
    edge_df: pd.DataFrame,
    tf_names: list,
    tf_col: str = "source_id",
    item_col: str = "peak_id",
    pct_true_edges: float | None = 1.0,
    true_false_ratio: float = 1.0,
    seed: int = 123,
    batch_size: int = 1_000_000,
    show_progress: bool = True,
):
    """
    Create sets of true and false edges for training.

    Returns
    -------
    true_edges : set[tuple[str, str]]
        Sampled observed positive edges.

    false_edges : set[tuple[str, str]]
        Sampled unobserved negative edges, excluding all known observed edges,
        not just the sampled positives.
    """

    df_all = edge_df[[tf_col, item_col]]
    df_all = df_all[df_all[tf_col].isin(tf_names)].dropna(subset=[tf_col, item_col])

    if df_all.empty:
        raise ValueError(
            f"No edges remain after filtering by tf_names using columns "
            f"{tf_col!r} and {item_col!r}."
        )

    # Integer-code both columns with pd.factorize rather than
    # sorted(unique()) -> dict -> Series.map. Series.map on an object column is an
    # element-wise Python loop, and at ChIP-Atlas scale (268M rows for hg38) that block
    # measured 15.6 min against 2.4 min here -- a 6.4x speedup for an identical edge set.
    # Deduplication rides along for free: np.unique on the int64 codes replaces
    # drop_duplicates over two string columns.
    #
    # factorize returns codes in order of first appearance rather than sorted. Nothing
    # depends on the ordering: it is internal to the code arithmetic below, and edges leave
    # this function as (name, name) tuples.
    tf_codes, candidate_tfs = pd.factorize(df_all[tf_col], sort=False)
    item_codes, candidate_items = pd.factorize(df_all[item_col], sort=False)

    candidate_tfs = np.asarray(candidate_tfs, dtype=object)
    candidate_items = np.asarray(candidate_items, dtype=object)

    n_tfs = len(candidate_tfs)
    n_items = len(candidate_items)

    # Integer-code all observed edges. These are excluded from negative sampling.
    observed_codes = np.unique(
        tf_codes.astype(np.int64) * n_items + item_codes.astype(np.int64)
    )

    # Subsample positives after defining all observed codes.
    if pct_true_edges is not None:
        if not (0 < pct_true_edges <= 1):
            raise ValueError("pct_true_edges must be in (0, 1] or None.")

        logging.info(f"Sampling {pct_true_edges:.2%} of true edges.")
        rng_pos = np.random.default_rng(seed)
        n_keep = int(round(len(observed_codes) * pct_true_edges))
        keep = rng_pos.choice(len(observed_codes), size=n_keep, replace=False)
        pos_codes = observed_codes[keep]
        logging.info(f"  - Sampled {len(pos_codes):,} of {len(observed_codes):,} true edges")
    else:
        pos_codes = observed_codes

    true_edges = set(
        zip(
            candidate_tfs[pos_codes // n_items],
            candidate_items[pos_codes % n_items],
        )
    )

    num_false_edges = round(len(true_edges) * true_false_ratio)

    false_codes = sample_unobserved_edge_codes_fast(
        n_tfs=n_tfs,
        n_items=n_items,
        observed_codes=observed_codes,
        num_edges=num_false_edges,
        batch_size=batch_size,
        seed=seed,
        show_progress=show_progress,
    )

    false_tf_idx = false_codes // n_items
    false_item_idx = false_codes % n_items

    false_edges = set(
        zip(
            candidate_tfs[false_tf_idx],
            candidate_items[false_item_idx],
        )
    )

    return true_edges, false_edges


def sample_unobserved_edge_codes_fast(
    n_tfs: int,
    n_items: int,
    observed_codes: np.ndarray,
    num_edges: int,
    batch_size: int = 1_000_000,
    seed: int = 123,
    show_progress: bool = True,
):
    """
    Sample unique integer-coded TF-item pairs not present in observed_codes.

    edge_code = tf_idx * n_items + item_idx
    """

    universe_size = n_tfs * n_items
    max_unobserved_edges = universe_size - len(observed_codes)

    if max_unobserved_edges <= 0:
        raise ValueError(
            "No unobserved TF-item pairs are available. "
            "The observed edges cover the full TF x item universe."
        )

    if num_edges > max_unobserved_edges:
        logging.warning(
            f"Requested {num_edges:,} sampled unobserved edges, but only "
            f"{max_unobserved_edges:,} are possible. Returning all possible."
        )
        num_edges = max_unobserved_edges

    rng = np.random.default_rng(seed)

    # searchsorted needs a sorted array. Callers pass np.unique output, which is already
    # sorted, so this is normally just the O(n) check.
    observed_sorted = observed_codes
    if observed_sorted.size and not np.all(observed_sorted[:-1] <= observed_sorted[1:]):
        observed_sorted = np.sort(observed_sorted)

    # Fully vectorised. The previous version ran `for code in codes` over every drawn
    # candidate and kept a Python set of every code accepted so far: ~1M interpreter
    # iterations per batch, and a set that grows to one entry per negative edge (tens of GB
    # at the scales this is now called with). Here each batch is filtered with searchsorted
    # and duplicates are removed with np.unique, so nothing leaves numpy.
    collected = np.empty(0, dtype=np.int64)

    pbar = tqdm(
        total=num_edges,
        desc="Generating sampled unobserved TF-item edges",
        ncols=125,
        disable=not show_progress,
    )

    try:
        while len(collected) < num_edges:
            need = num_edges - len(collected)
            # Overdraw: some candidates collide with each other or with observed edges. The
            # universe is astronomically larger than any request here, so the loss is tiny.
            draw = int(min(max(need * 1.3, 1_000_000), max(batch_size, 1) * 50))

            codes = (
                rng.integers(0, n_tfs, size=draw, dtype=np.int64) * n_items
                + rng.integers(0, n_items, size=draw, dtype=np.int64)
            )
            codes = np.unique(codes)

            if observed_sorted.size:
                hit = np.searchsorted(observed_sorted, codes)
                np.clip(hit, 0, observed_sorted.size - 1, out=hit)
                codes = codes[observed_sorted[hit] != codes]

            if codes.size == 0:
                continue

            before = len(collected)
            collected = np.unique(np.concatenate([collected, codes]))
            pbar.update(min(len(collected), num_edges) - min(before, num_edges))

    finally:
        pbar.close()

    # np.unique leaves `collected` sorted, so truncating would bias towards low TF indices.
    rng.shuffle(collected)
    sampled_codes = collected[:num_edges]

    return sampled_codes


def download_gene_protein_fastas(
    gene_names,
    organism,
    output_dir,
    email,
    api_key=None,
    retmax=25,
    delay=0.5,
    max_tries=3,
    sleep_between_tries=15,
):
    """
    Download one representative RefSeq protein FASTA per gene.

    Saves:
        output_dir/{gene_name}_protein.fasta

    Uses a delay between genes to avoid NCBI rate-limit issues.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    Entrez.email = email
    Entrez.max_tries = max_tries
    Entrez.sleep_between_tries = sleep_between_tries

    if api_key is not None:
        Entrez.api_key = api_key

    saved_files = {}
    
    # Check if the gene names are already downloaded to avoid unnecessary API calls
    available_files = {f.stem.replace("_protein", ""): f for f in output_dir.glob("*_protein.fasta")}
    gene_names = [gene for gene in gene_names if gene not in available_files]

    if not gene_names:
        logging.info("All gene FASTA files already exist. No downloads needed.")
        return {gene: available_files[gene] for gene in gene_names}
    
    for i, gene_name in enumerate(gene_names, start=1):
        search_term = (
            f'{gene_name}[Gene Name] '
            f'AND {organism}[Organism] '
            f'AND srcdb_refseq[PROP]'
        )

        try:
            with Entrez.esearch(
                db="protein",
                term=search_term,
                retmax=retmax,
            ) as search_handle:
                search_results = Entrez.read(search_handle)

            protein_ids = search_results.get("IdList", [])

            if not protein_ids:
                logging.info(f"[{i}/{len(gene_names)}] No records found for {gene_name}")
                saved_files[gene_name] = None
                time.sleep(delay)
                continue

            with Entrez.efetch(
                db="protein",
                id=protein_ids,
                rettype="gb",
                retmode="text",
            ) as fetch_handle:
                records = list(SeqIO.parse(fetch_handle, "genbank"))

            if not records:
                logging.info(f"[{i}/{len(gene_names)}] Could not parse records for {gene_name}")
                saved_files[gene_name] = None
                time.sleep(delay)
                continue

            def protein_rank(record):
                accession = record.id
                description = record.description.lower()
                keywords = [k.lower() for k in record.annotations.get("keywords", [])]

                is_refseq_select = (
                    "refseq select" in description
                    or "refseq select" in keywords
                )

                is_np = accession.startswith("NP_")
                is_xp = accession.startswith("XP_")
                is_low_quality = "low quality protein" in description

                return (
                    not is_refseq_select,
                    not is_np,
                    is_xp,
                    is_low_quality,
                    -len(record.seq),
                )

            best_record = sorted(records, key=protein_rank)[0]

            output_file = output_dir / f"{gene_name}_protein.fasta"

            with open(output_file, "w") as f:
                SeqIO.write(best_record, f, "fasta")

            saved_files[gene_name] = output_file

            logging.info(
                f"[{i}/{len(gene_names)}] Saved {gene_name}: "
                f"{best_record.id} ({len(best_record.seq)} aa)"
            )

        except Exception as e:
            logging.info(f"[{i}/{len(gene_names)}] Failed for {gene_name}: {e}")
            saved_files[gene_name] = None

        time.sleep(delay)

    return saved_files


def fetch_chip_atlas_tf_list_to_parquet(
    tf_list,
    genome="mm10",
    out_dir="chip_atlas_tf_parquet",
    num_workers=10,
    threshold="05",
    timeout=120,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failed_tfs = {}
    
    existing_files = {f.stem: f for f in out_dir.glob("*.parquet")}
    if len(existing_files) > 0:
        logging.info(f"Found {len(existing_files)} / {len(tf_list)} existing parquet files. Skipping these TFs.")
    
    tf_list = [tf for tf in tf_list if tf not in existing_files]

    def fetch_chip_atlas_tf(tf):
        tf_canon = tf.replace("-", "")

        url = (
            f"https://chip-atlas.dbcls.jp/data/{genome}/assembled/"
            f"Oth.ALL.{threshold}.{tf_canon}.AllCell.bed"
        )

        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()

                df = pd.read_csv(
                    r.raw,
                    sep="\t",
                    comment="t",
                    header=None,
                    usecols=[0, 1, 2],
                    names=["peak_chr", "peak_start", "peak_end"],
                    dtype={
                        "peak_chr": "category",
                        "peak_start": "int32",
                        "peak_end": "int32",
                    },
                )

            if df.empty:
                return tf, None, "empty dataframe"

            # Deduplicate before writing.
            # This is much cheaper than one giant global dedup.
            df = df.drop_duplicates()

            df["source_id"] = tf

            # Keep peak coordinates separate for now.
            # Building millions of strings is expensive.
            df = df[["source_id", "peak_chr", "peak_start", "peak_end"]]

            out_file = out_dir / f"{tf}.parquet"
            df.to_parquet(out_file, index=False)

            return tf, out_file, None

        except Exception as e:
            return tf, None, e

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(fetch_chip_atlas_tf, tf): tf
            for tf in tf_list
        }

        for future in as_completed(futures):
            tf, out_file, error = future.result()

            if error is not None:
                failed_tfs[tf] = error
                logging.info(f"TF '{tf}' not found or failed: {error}")
                continue

            logging.info(f"Wrote {tf} to {out_file}")

    return failed_tfs


def build_chip_atlas_df_from_parquet(
    parquet_dir="chip_atlas_tf_parquet",
    output_file="chip_atlas_tf_peak_edges.parquet",
):
    parquet_dir = Path(parquet_dir)

    query = f"""
    COPY (
        SELECT DISTINCT
            source_id,
            peak_chr || ':' || peak_start::VARCHAR || '-' || peak_end::VARCHAR AS peak_id
        FROM read_parquet('{parquet_dir}/*.parquet')
    )
    TO '{output_file}'
    (FORMAT PARQUET);
    """

    duckdb.sql(query)

    return output_file


def find_latest_checkpoint(
    checkpoint_dir: Path, 
    cell_type: str, 
    sample_name: str, 
    training_number: int|None =None,
    epoch_num: int|None =None,
    verbose: bool = True
    ) -> Path:
    """
    Find the latest checkpoint file for a given cell type and sample name.
    
    Optionally takes a training_number (the SLURM ID from training) to select a specific checkpoint.
    Optionally takes an epoch_num to select a specific epoch.
    
    Parameters
    ----------
    checkpoint_dir : Path
        The base directory where checkpoints are stored.
    cell_type : str
        The cell type for which to find the checkpoint.
    sample_name : str
        The sample name for which to find the checkpoint.
    training_number : int, optional
        The specific training number (SLURM ID) to select a checkpoint from.
    epoch_num : int, optional
        The specific epoch number to select a checkpoint from.
        
    Returns
    -------
    Path or None
        The path to the latest checkpoint file, or None if no checkpoint is found.
    
    """
    
    sample_chkpt_dir = checkpoint_dir / cell_type / sample_name
    
    if not sample_chkpt_dir.exists():
        logging.warning(f"No checkpoints found for {cell_type} {sample_name} in {sample_chkpt_dir}")
        return None
    
    # Find all SLURM job directories for the given sample (or specific training number if provided)
    if training_number is not None:
        slurm_job_dirs = [d for d in sample_chkpt_dir.iterdir() if d.is_dir() and d.name.startswith(f"tf_tg_train_{sample_name}_{training_number}")]
    else:
        slurm_job_dirs = [d for d in sample_chkpt_dir.iterdir() if d.is_dir() and d.name.startswith(f"tf_tg_train_{sample_name}_")]
    
    if not slurm_job_dirs:
        logging.warning(f"No checkpoint directories found for {cell_type} {sample_name} in {sample_chkpt_dir}")
        return None
    
    # Find the latest checkpoint directory based on the SLURM job ID (the last part of the directory name)
    latest_chkpt_dir = max(slurm_job_dirs, key=lambda d: int(d.name.split("_")[-1]))
    slurm_job_id = latest_chkpt_dir.name.split("_")[-1]
    
    # Find all checkpoint files in the latest checkpoint directory
    chkpt_files = list(latest_chkpt_dir.glob("epoch=*-val_auroc=*-val_loss=*.ckpt"))
    if not chkpt_files:
        logging.warning(f"No checkpoint files found for {sample_name} in {latest_chkpt_dir}")
        return None
    
    # If epoch_num is specified, find the checkpoint for that epoch. Otherwise, find the latest checkpoint.
    chkpt_nums = [int(f.stem.split("-")[0].split("=")[1]) for f in chkpt_files]
    if epoch_num is not None:
        if epoch_num in chkpt_nums:
            latest_chkpt_file = next(f for f in chkpt_files if int(f.stem.split("-")[0].split("=")[1]) == epoch_num)
        else:
            logging.warning(f"Checkpoint for epoch {epoch_num} not found for {sample_name} in {latest_chkpt_dir}. Available epochs: {chkpt_nums}")
            return None
    else:
        latest_chkpt_file = max(chkpt_files, key=lambda f: int(f.stem.split("-")[0].split("=")[1]))
    epoch = latest_chkpt_file.stem.split("-")[0].split("=")[1]
    
    if verbose:
        logging.info(f"Latest checkpoint for {cell_type} {sample_name}: Job {slurm_job_id} Epoch {epoch}")
    
    return latest_chkpt_file


def strip_compiled_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    """
    Remove torch.compile's _orig_mod prefix from state_dict keys.

    Examples:
        model._orig_mod.encoder.weight -> model.encoder.weight
        _orig_mod.encoder.weight       -> encoder.weight
    """
    cleaned = {}

    for key, value in state_dict.items():
        cleaned_key = key.replace(prefix, "")
        cleaned[cleaned_key] = value

    return cleaned


def create_index_file_for_fragments(
    frag_path: Path,
    force_reload: bool = False
):
    if not frag_path.exists():
        raise FileNotFoundError(f"Fragment file not found: {frag_path}")
    else:
        index_file = str(frag_path) + ".tbi"

        if Path(index_file).exists() and not force_reload:
            logging.info("Found ATAC fragment file index:", index_file)

        else:
            # Index the fragment file
            logging.info("Index file not found. Creating index file...")
            pysam.tabix_index(
                str(frag_path),
                preset="bed",
                force=True
            )
            index_file = str(frag_path) + ".tbi"
            logging.info(f"  - Saved to {index_file}")


def gpu_supports_torch_compile(device):
    if device.type != "cuda":
        return False

    major, minor = torch.cuda.get_device_capability(device)
    return major >= 7


def load_tf_dna_model(
    tf_dna_model_path: Path,
    tf_embeddings_tensor: torch.Tensor,
    tf_mask_tensor: torch.Tensor,
    compile_model: bool = False,
    device: torch.device | None = None,
) -> tf_to_dna_module.LitTFPeakBindingModel:

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if compile_model and not gpu_supports_torch_compile(device):
        if device.type == "cuda":
            major, minor = torch.cuda.get_device_capability(device)
            logging.warning(
                f"Skipping torch.compile because this GPU has compute capability "
                f"{major}.{minor}; Inductor/Triton requires >= 7.0."
            )
        else:
            logging.warning("Skipping torch.compile because device is not CUDA.")

        compile_model = False

    # -----------------------------
    # 1. Recreate base TF-DNA model uncompiled
    # -----------------------------
    base_model = tf_to_dna_module.TFPeakBindingModel(
        tf_embedding_dim=128,
        hidden_dim=128,
        dropout=0.3,
        num_layers=4,
        num_heads=4,
        dim_head=32,
    )

    # -----------------------------
    # 2. Load TF-DNA checkpoint
    # -----------------------------
    tf_dna_ckpt = torch.load(
        tf_dna_model_path,
        map_location="cpu",
        weights_only=False,
    )

    tf_dna_state_dict = tf_dna_ckpt["state_dict"]

    if any("._orig_mod." in key or key.startswith("_orig_mod.") for key in tf_dna_state_dict):
        logging.info("Detected compiled TF-DNA checkpoint. Stripping _orig_mod prefixes.")
        tf_dna_state_dict = strip_compiled_prefix_from_state_dict(tf_dna_state_dict)

    lit_tf_dna_model = tf_to_dna_module.LitTFPeakBindingModel(
        model=base_model,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
    )

    lit_tf_dna_model.load_state_dict(tf_dna_state_dict, strict=True)

    if compile_model:
        logging.info("Compiling loaded TF-DNA core model.")
            # No mode="reduce-overhead". That enables CUDA graphs, which re-record
            # whenever an input shape reappears after eviction. TFTGRegulationModel
            # deliberately produces several shapes (one per TF crop width x chunk count),
            # and measured against plain compile on TF-major batches the median was 1.9x
            # worse while p90 was 14.6x worse (2653 ms vs 181 ms) -- the tail, not the
            # median, is what a full run pays. Default mode: 94 ms median / 181 ms p90.
        lit_tf_dna_model.model = torch.compile(lit_tf_dna_model.model)
        
    return lit_tf_dna_model


def load_tf_tg_regulation_model(
    tf_dna_model_path: Path,
    tf_tg_model_path: Path,
    tf_embeddings_tensor: torch.Tensor,
    tf_mask_tensor: torch.Tensor,
    tf_peak_chunk_size: int = 128,
    compile_model: bool = False,
    device: torch.device | None = None,
    model_module=None,
) -> tf_to_tg_module.LitTFTGRegulationModel:
    """
    model_module lets a caller load a checkpoint trained against a different
    TFTGRegulationModel definition than the default models.tf_to_tg -- e.g.
    models.tf_to_tg_testing while it is under active architecture changes there.
    Defaults to models.tf_to_tg (unchanged behaviour for every existing checkpoint).
    """
    model_module = model_module or tf_to_tg_module

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if compile_model and not gpu_supports_torch_compile(device):
        if device.type == "cuda":
            major, minor = torch.cuda.get_device_capability(device)
            logging.warning(
                f"Skipping torch.compile because this GPU has compute capability "
                f"{major}.{minor}; Inductor/Triton requires >= 7.0."
            )
        else:
            logging.warning("Skipping torch.compile because device is not CUDA.")

        compile_model = False

    # -----------------------------
    # 1. Recreate base TF-DNA model uncompiled
    # -----------------------------
    base_model = tf_to_dna_module.TFPeakBindingModel(
        tf_embedding_dim=128,
        hidden_dim=128,
        dropout=0.3,
        num_layers=4,
        num_heads=4,
        dim_head=32,
    )

    # -----------------------------
    # 2. Load TF-DNA checkpoint
    # -----------------------------
    tf_dna_ckpt = torch.load(
        tf_dna_model_path,
        map_location="cpu",
        weights_only=False,
    )

    tf_dna_state_dict = tf_dna_ckpt["state_dict"]

    if any("._orig_mod." in key or key.startswith("_orig_mod.") for key in tf_dna_state_dict):
        logging.info("Detected compiled TF-DNA checkpoint. Stripping _orig_mod prefixes.")
        tf_dna_state_dict = strip_compiled_prefix_from_state_dict(tf_dna_state_dict)

    lit_tf_dna_model = tf_to_dna_module.LitTFPeakBindingModel(
        model=base_model,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
    )

    lit_tf_dna_model.load_state_dict(tf_dna_state_dict, strict=True)

    trained_tf_peak_model = lit_tf_dna_model.model
    trained_tf_peak_model.eval()

    for p in trained_tf_peak_model.parameters():
        p.requires_grad = False

    # -----------------------------
    # 3. Load TF-TG checkpoint
    # -----------------------------
    tf_tg_ckpt = torch.load(
        tf_tg_model_path,
        map_location="cpu",
        weights_only=False,
    )

    tf_tg_state_dict = tf_tg_ckpt["state_dict"]

    if any("._orig_mod." in key or key.startswith("_orig_mod.") for key in tf_tg_state_dict):
        logging.info("Detected compiled TF-TG checkpoint. Stripping _orig_mod prefixes.")
        tf_tg_state_dict = strip_compiled_prefix_from_state_dict(tf_tg_state_dict)

    # -----------------------------
    # 4. Recreate TF-TG model uncompiled
    # -----------------------------
    tf_tg_kwargs = dict(
        pretrained_tf_peak_model=trained_tf_peak_model,
        d_model=128,
        tf_peak_chunk_size=tf_peak_chunk_size,
    )
    # tf_binding_hidden_dim only exists on models.tf_to_tg_testing's TFTGRegulationModel
    # (see models/tf_to_tg.py vs tf_to_tg_testing.py) -- pass it only when the target
    # class actually accepts it, so this stays a no-op against the original module.
    init_params = inspect.signature(model_module.TFTGRegulationModel.__init__).parameters
    if "tf_binding_hidden_dim" in init_params:
        tf_tg_kwargs["tf_binding_hidden_dim"] = 128 // 2  # matches base_model's hidden_dim above

    tf_tg_core_model = model_module.TFTGRegulationModel(**tf_tg_kwargs)

    lit_tf_tg_model = model_module.LitTFTGRegulationModel(
        model=tf_tg_core_model,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
    )

    lit_tf_tg_model.load_state_dict(tf_tg_state_dict, strict=True)
    lit_tf_tg_model.eval()

    for p in lit_tf_tg_model.parameters():
        p.requires_grad = False

    # -----------------------------
    # 5. Optional compile after loading
    # -----------------------------
    if compile_model:
        logging.info("Compiling loaded TF-TG core model.")
            # No mode="reduce-overhead". That enables CUDA graphs, which re-record
            # whenever an input shape reappears after eviction. TFTGRegulationModel
            # deliberately produces several shapes (one per TF crop width x chunk count),
            # and measured against plain compile on TF-major batches the median was 1.9x
            # worse while p90 was 14.6x worse (2653 ms vs 181 ms) -- the tail, not the
            # median, is what a full run pays. Default mode: 94 ms median / 181 ms p90.
        lit_tf_tg_model.model = torch.compile(lit_tf_tg_model.model)

    return lit_tf_tg_model


def tf_dna_cache_dir_for(cell_type_cache_dir: Path) -> Path:
    """The species TF-DNA cache holding the TF tables, given a cell-type cache dir.

    Layout is cached_data/<species>/{tf_dna_cache, <cell_type>_cache}, so the TF-DNA cache
    is the cell-type cache's sibling. tf_embeddings.pt / tf_masks.pt / tf_name_to_idx.csv
    live there because they are species-level -- every cell type of a species previously
    held a byte-identical copy.
    """
    return Path(cell_type_cache_dir).parent / config.TF_DNA_CACHE_DIRNAME


def load_training_cache_dataset(
    sample_name: str,
    cell_type_cache_dir: Path, 
    split_type: str = "test", 
    subset_size: int = None,
    batch_size: int = 512,
    ) -> DataLoader:
    
    assert split_type in ["train", "val", "test"], \
        "split_type must be one of 'train', 'val', or 'test'"

    sample_cache_dir = Path(cell_type_cache_dir) / sample_name
    tf_dna_dir = tf_dna_cache_dir_for(cell_type_cache_dir)
    
    # Load the compact split inputs
    tftg_inputs_test = torch.load(
        sample_cache_dir / f"tftg_inputs_{split_type}.pt",
        weights_only=False,
    )

    # Load the lookup tensors
    tf_embeddings_tensor = torch.load(
        tf_dna_dir / "tf_embeddings.pt",
        weights_only=True,
    )
    tf_mask_tensor = torch.load(
        tf_dna_dir / "tf_masks.pt",
        weights_only=True,
    )
    atac_peak_tensor = torch.load(
        sample_cache_dir / "atac_peak_tensor.pt",
        weights_only=True,
    )

    # Load the metadata
    with open(sample_cache_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Load the manifest and verify tensor shapes and dtypes match expectations
    with open(sample_cache_dir / "manifest.json") as f:
        manifest = json.load(f)
    
    assert tuple(manifest["atac_peak_tensor_shape"]) == tuple(atac_peak_tensor.shape)
    assert manifest["atac_peak_tensor_dtype"] == str(atac_peak_tensor.dtype)

    dataset = TFTGEdgeBagDataset(
        tftg_inputs_test,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor
    )
    
    subset_size = min(subset_size, len(dataset)) if subset_size is not None else None
    
    if subset_size is not None:
        dataset = Subset(dataset, list(range(subset_size)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_tftg_edge_bags,
        )

    return loader, metadata, manifest, tf_embeddings_tensor, tf_mask_tensor


# ---------------------------------------------------------------------------
# Multi-rank training infrastructure shared by stability_model_training.py,
# test_simplified_model_multigpu_safe.py, and the wandb sweep entrypoints.
# ---------------------------------------------------------------------------

def env_int(keys, default=0):
    for key in keys:
        value = os.environ.get(key)
        if value not in (None, ""):
            try:
                return int(value)
            except ValueError:
                pass
    return default


def get_rank_info():
    """Return rank information for both torchrun/Lightning and Slurm-launched jobs."""
    world_size = env_int(["WORLD_SIZE", "SLURM_NTASKS"], 1)
    global_rank = env_int(["RANK", "SLURM_PROCID"], 0)
    local_rank = env_int(["LOCAL_RANK", "SLURM_LOCALID"], 0)
    node_rank = env_int(["NODE_RANK", "SLURM_NODEID"], 0)
    return global_rank, local_rank, node_rank, world_size


def is_global_rank_zero():
    return get_rank_info()[0] == 0


def configure_rank_logging():
    global_rank, local_rank, node_rank, world_size = get_rank_info()
    level = logging.INFO if global_rank == 0 else logging.WARNING
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        handler.setFormatter(logging.Formatter(
            fmt=f"%(asctime)s | rank={global_rank}/{world_size} local={local_rank} | %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    return global_rank, local_rank, node_rank, world_size


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def atomic_json_dump(obj, path: Path, indent=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp_path, path)


def required_cache_files(paths):
    return [
        paths["atac_peak_tensor"],
        paths["metadata"],
        paths["manifest"],
        paths["train"],
        paths["val"],
        paths["test"],
    ]


def cache_is_complete(paths):
    return paths["ready"].exists() and all(p.exists() and p.stat().st_size > 0 for p in required_cache_files(paths))


def wait_for_cache(paths, poll_seconds=30, timeout_seconds=None):
    start = time.time()
    while not cache_is_complete(paths):
        if paths["failed"].exists():
            try:
                msg = paths["failed"].read_text()
            except Exception:
                msg = "Rank 0 failed while constructing the TF-TG cache."
            raise RuntimeError(msg)
        if timeout_seconds is not None and time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for rank 0 to finish cache construction in {paths['cache_dir']}")
        time.sleep(poll_seconds)


def get_reference_paths_and_chroms(species: str):
    project_data_dir = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
    if species == "mm10":
        gene_ref_file = project_data_dir / "genome_data" / "genome_annotation" / "mm10" / "Mus_musculus.GRCm39.115.gtf.gz"
        train_chroms = [str(i) for i in range(1, 16)]
        val_chroms = [str(i) for i in range(16, 18)]
        test_chroms = [str(i) for i in range(18, 20)]
        valid_chroms = {f"chr{i}" for i in range(1, 20)}
    elif species == "hg38":
        gene_ref_file = project_data_dir / "genome_data" / "genome_annotation" / "hg38" / "Homo_sapiens.GRCh38.113.gtf.gz"
        train_chroms = [str(i) for i in range(1, 18)]
        val_chroms = [str(i) for i in range(18, 20)]
        test_chroms = [str(i) for i in range(20, 23)]
        valid_chroms = {f"chr{i}" for i in range(1, 23)}
    else:
        raise ValueError(f"Unsupported species: {species}")

    genome_fasta_path = project_data_dir / "genome_data" / "reference_genome" / species / f"{species}.fa"
    chrom_sizes_path = project_data_dir / "genome_data" / "reference_genome" / species / f"{species}.chrom.sizes"
    return gene_ref_file, genome_fasta_path, chrom_sizes_path, train_chroms, val_chroms, test_chroms, valid_chroms


def tf_dna_checkpoint_for_cell_type(cell_type: str):
    mm10_tf_dna_path = CHKPT_DIR / "tf_dna_mm10_3697823" / "epoch=07-val_auroc=0.9743-val_loss=0.1661.ckpt"
    hg38_tf_dna_path = CHKPT_DIR / "tf_dna_hg38_3683606" / "epoch=13-val_auroc=0.9566-val_loss=0.2042.ckpt"

    tf_dna_model_checkpoints = {
        "mESC": mm10_tf_dna_path,
        "mouse_liver": mm10_tf_dna_path,
        "mouse_hepatocytes": mm10_tf_dna_path,
        "iPSC": hg38_tf_dna_path,
        "Macrophage": hg38_tf_dna_path,
        "K562": hg38_tf_dna_path
    }

    return tf_dna_model_checkpoints[cell_type]


def validate_tf_name_to_idx(
    *,
    tf_name_to_idx,
    tf_embeddings_tensor,
    tf_mask_tensor,
    source,
):
    n_tf_embeddings = tf_embeddings_tensor.shape[0]
    n_tf_masks = tf_mask_tensor.shape[0]

    if n_tf_embeddings != n_tf_masks:
        raise ValueError(
            f"TF embedding/mask row mismatch from {source}: "
            f"tf_embeddings={tuple(tf_embeddings_tensor.shape)}, "
            f"tf_masks={tuple(tf_mask_tensor.shape)}"
        )

    if not tf_name_to_idx:
        raise ValueError(f"Empty tf_name_to_idx loaded from {source}")

    min_idx = min(tf_name_to_idx.values())
    max_idx = max(tf_name_to_idx.values())

    if min_idx < 0 or max_idx >= n_tf_embeddings:
        bad = {
            tf: idx
            for tf, idx in tf_name_to_idx.items()
            if idx < 0 or idx >= n_tf_embeddings
        }

        raise ValueError(
            f"Incompatible tf_name_to_idx and tf_embeddings_tensor from {source}. "
            f"Valid TF embedding rows are 0-{n_tf_embeddings - 1}, "
            f"but map has min={min_idx}, max={max_idx}. "
            f"Example invalid entries: {list(bad.items())[:20]}"
        )


def load_tf_embedding_resources(paths):
    """
    Load TF embeddings, masks, and the matching tf_name_to_idx map.

    Critical: tf_name_to_idx values must index rows of tf_embeddings_tensor.
    """
    # Callers may pass the TF-DNA cache explicitly; otherwise it is the cell-type cache's
    # sibling under cached_data/<species>/.
    tf_dna_dir = paths.get("tf_dna_cache_dir") or tf_dna_cache_dir_for(
        paths["cell_type_cache_dir"]
    )

    tf_embeddings_tensor = torch.load(
        tf_dna_dir / "tf_embeddings.pt",
        map_location="cpu",
        weights_only=True,
    )
    tf_mask_tensor = torch.load(
        tf_dna_dir / "tf_masks.pt",
        map_location="cpu",
        weights_only=True,
    )

    tf_idx_csv = tf_dna_dir / "tf_name_to_idx.csv"
    tf_idx_json = tf_dna_dir / "tf_name_to_idx.json"
    metadata_json = tf_dna_dir / "metadata.json"

    if tf_idx_csv.exists():
        tf_name_to_idx_df = pd.read_csv(tf_idx_csv)
        tf_name_to_idx_df["tf_name"] = tf_name_to_idx_df["tf_name"].str.upper()
        tf_name_to_idx = (
            tf_name_to_idx_df
            .set_index("tf_name")["tf_idx"]
            .astype(int)
            .to_dict()
        )

    elif tf_idx_json.exists():
        with open(tf_idx_json) as f:
            tf_name_to_idx = json.load(f)
        tf_name_to_idx = {str(k).upper(): int(v) for k, v in tf_name_to_idx.items()}

    elif metadata_json.exists():
        with open(metadata_json) as f:
            metadata = json.load(f)
        tf_name_to_idx = {
            str(k).upper(): int(v)
            for k, v in metadata["tf_name_to_idx"].items()
        }

    else:
        raise FileNotFoundError(
            f"No TF index map found in {tf_dna_dir}. "
            "Expected one of: tf_name_to_idx.csv, tf_name_to_idx.json, metadata.json. "
            "Do not use config.tf_name_to_idx_cache_path unless it was generated with "
            "this exact tf_embeddings.pt tensor."
        )

    validate_tf_name_to_idx(
        tf_name_to_idx=tf_name_to_idx,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        source=str(tf_dna_dir),
    )

    return tf_embeddings_tensor, tf_mask_tensor, tf_name_to_idx


def load_training_cache(paths):
    wait_for_cache(paths, poll_seconds=1, timeout_seconds=None)
    tftg_inputs_train = torch.load(paths["train"], map_location="cpu", weights_only=False)
    tftg_inputs_val = torch.load(paths["val"], map_location="cpu", weights_only=False)
    tftg_inputs_test = torch.load(paths["test"], map_location="cpu", weights_only=False)
    atac_peak_tensor = torch.load(paths["atac_peak_tensor"], map_location="cpu", weights_only=True)
    tf_embeddings_tensor, tf_mask_tensor, _ = load_tf_embedding_resources(paths)

    return (
        tftg_inputs_train,
        tftg_inputs_val,
        tftg_inputs_test,
        atac_peak_tensor,
        tf_embeddings_tensor,
        tf_mask_tensor,
    )


# ---------------------------------------------------------------------------
# TF-TG edge input construction shared by scripts/build_tf_to_tg_train_data.py,
# stability_model_training.py, test_simplified_model_multigpu_safe.py,
# wandb_sweep.py, and plot_auprc_all_methods.py.
# ---------------------------------------------------------------------------

def prepare_tftg_lookup_tables(
    peak_to_gene,
    atac_peak_map,
    atac_pseudobulk,
    rna_pseudobulk_norm,
    dataset_peaks,
    common_cells,
    max_precompute_peaks=None,
):
    valid_peak_set = set(atac_peak_map.keys())

    peak_to_gene_valid = peak_to_gene[
        peak_to_gene["peak_id"].isin(valid_peak_set)
    ].copy()

    peak_to_gene_valid["abs_dist"] = peak_to_gene_valid["TSS_dist"].abs()

    tg_to_peak_info = {}

    # Subset to only peaks within 100kb of the TG TSS and sort by distance
    for tg_norm, sub in peak_to_gene_valid.groupby("target_id_norm", sort=False):
        sub = sub[sub["abs_dist"] <= 100_000].sort_values("abs_dist")

        if sub.empty:
            continue

        # Optional cap to only use the closest N peaks per TG
        if max_precompute_peaks is not None:
            sub = sub.head(max_precompute_peaks)

        peak_ids = sub["peak_id"].tolist()
        peak_indices = np.asarray(
            [atac_peak_map[p] for p in peak_ids],
            dtype=np.int64,
        )
        peak_distances = sub["TSS_dist"].to_numpy(dtype=np.float32)

        tg_to_peak_info[tg_norm] = {
            "peak_ids": peak_ids,
            "peak_indices": peak_indices,
            "peak_distances": peak_distances,
        }

    cell_to_idx = {cell: i for i, cell in enumerate(common_cells)}

    atac_mat = (
        atac_pseudobulk
        .reindex(index=dataset_peaks, columns=common_cells)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    rna_mat = (
        rna_pseudobulk_norm
        .reindex(columns=common_cells)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    gene_to_rna_idx = {gene: i for i, gene in enumerate(rna_pseudobulk_norm.index)}

    return tg_to_peak_info, cell_to_idx, atac_mat, rna_mat, gene_to_rna_idx


def build_tftg_inputs(
    tf_tg_df,
    max_peaks_per_tg=None,
    max_cells_per_pair=8,
    seed=123,
    silence=False,
    *,
    tg_to_peak_info,
    cell_to_idx,
    atac_mat,
    rna_mat,
    gene_to_rna_idx,
    common_cells,
    tf_name_to_idx,
    tg_id_to_idx,
    max_peaks_real,
):
    """
    Build one compact item per TF-TG edge.

    Peak/expression data is stored compactly rather than duplicated per edge:
      - peak_indices/peak_distance/peak_mask ARE identical across every edge sharing a TG
        (every TF paired with a given TG sees the same peak set), so they're stored once
        per TG and gathered via tg_idx at read time (see TFTGEdgeBagDataset.__getitem__).
      - peak_accessibility/tf_expression/tg_expression are NOT shared across edges on the
        same TG -- each edge draws its own independent random cell subset (see the cell
        sampling below) -- so instead of materializing the [C, P] / [C] values here, only
        the sampled cell column indices are stored (cell_indices), and the actual values
        are gathered lazily from atac_mat/rna_mat at read time. Same numbers (same rng
        draws), ~90% less storage.

    Output shapes:
        label:              [E]
        tf_idx:             [E]
        tg_idx:             [E]
        cell_indices:       [E, C]   int64 column indices into atac_mat / rna_mat
        tg_peak_indices:    [G, P]   G = len(tg_id_to_idx); gather rows via tg_idx
        tg_peak_distance:   [G, P]
        tg_peak_mask:       [G, P]

    Callers must also persist atac_mat/rna_mat/gene_to_rna_idx (returned by
    prepare_tftg_lookup_tables) alongside this dict -- TFTGEdgeBagDataset needs them to
    reconstruct peak_accessibility/tf_expression/tg_expression per item.
    """

    rng = np.random.default_rng(seed)

    tf_names = []
    tg_names = []
    cell_ids_all = []
    labels = []

    tf_indices = []
    tg_indices = []
    cell_indices_all = []

    common_cells = list(common_cells)
    n_common_cells = len(common_cells)

    # rng.choice() re-runs np.asarray() on its first argument every call, so passing a
    # Python list of thousands of cell-name strings re-converted the whole list per edge:
    # measured 417 us/edge against 8.3 us when drawing positions from an int, which was
    # ~77% of the entire build. Draw positions and index once instead. This is the same
    # draw, not an approximation -- Generator.choice picks positions and then indexes, so
    # rng.choice(names, k) == names[rng.choice(len(names), k)] for a given seed (verified
    # bit-identical over 2,000 consecutive draws).
    common_cells_arr = np.asarray(common_cells)
    common_cell_rows = np.asarray(
        [cell_to_idx[c] for c in common_cells], dtype=np.int64
    )
    take_all_cells = max_cells_per_pair is None or max_cells_per_pair >= n_common_cells

    # Per-TG padded peak bags. Every TF is paired with every TG, so without this the same
    # three np.pad calls repeat once per TF for each TG -- another ~40 us/edge.
    tg_bag_cache = {}

    n_total = len(tf_tg_df)
    log_every = max(1, n_total // 50)

    build_started = time.time()

    for i, row in enumerate(tf_tg_df.itertuples(index=False), start=1):
        if silence == False:
            if i == 1 or i % log_every == 0 or i == n_total:
                # Rate and ETA
                elapsed = time.time() - build_started
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (n_total - i) / rate if rate > 0 else 0.0
                logging.info(
                    f"Building compact TF-TG edges: {100 * i / n_total:.1f}% "
                    f"({i:,}/{n_total:,}) at {rate:,.0f} edges/s, "
                    f"elapsed {elapsed / 60:.1f}m, ETA {eta / 60:.1f}m"
                )

        tf_name = row.tf_name
        tg_name = row.tg_id
        label = float(row.label)

        tf_idx = tf_name_to_idx.get(tf_name)
        tg_idx = tg_id_to_idx.get(tg_name)

        if tf_idx is None or tg_idx is None:
            continue

        bag = tg_bag_cache.get(tg_name, 0)
        if bag == 0:
            peak_info = tg_to_peak_info.get(tg_name)
            if peak_info is None:
                tg_bag_cache[tg_name] = None
                continue

            peak_indices_real = list(peak_info["peak_indices"])
            peak_dst_real = list(peak_info["peak_distances"])

            n_peaks = len(peak_indices_real)
            if n_peaks == 0:
                tg_bag_cache[tg_name] = None
                continue

            peak_indices = np.asarray(peak_indices_real, dtype=np.int64)
            peak_dst = np.asarray(peak_dst_real, dtype=np.float32)
            peak_mask = np.ones(n_peaks, dtype=bool)

            if n_peaks < max_peaks_real:
                pad_len = max_peaks_real - n_peaks
                peak_indices = np.pad(peak_indices, (0, pad_len), constant_values=0)
                peak_dst = np.pad(peak_dst, (0, pad_len), constant_values=0.0)
                peak_mask = np.pad(peak_mask, (0, pad_len), constant_values=False)

            # Shared by every edge on this TG from here on, and np.stack copies them into
            # the output, so freeze them rather than trust that nobody writes through.
            for arr in (peak_indices, peak_dst, peak_mask):
                arr.flags.writeable = False

            bag = (peak_indices, peak_dst, peak_mask, n_peaks,
                   np.asarray(peak_indices_real, dtype=np.int64))
            tg_bag_cache[tg_name] = bag
        elif bag is None:
            continue

        peak_indices, peak_dst, peak_mask, n_peaks, peak_rows = bag

        # Sample cells
        if take_all_cells:
            sampled_cells = common_cells
            sampled_cell_indices = common_cell_rows
        else:
            sampled_positions = rng.choice(
                n_common_cells,
                size=max_cells_per_pair,
                replace=False,
            )
            sampled_cells = common_cells_arr[sampled_positions].tolist()
            sampled_cell_indices = common_cell_rows[sampled_positions]

        # peak_accessibility/tf_expression/tg_expression are gathered lazily at
        # Dataset.__getitem__ time from atac_mat/rna_mat via sampled_cell_indices (see the
        # docstring) -- only the indices are kept here, not the [C, P] / [C] values
        # themselves. `atac_mat`/`n_peaks`/`peak_rows` are unused past this point for this
        # edge; kept in the loop above only to preserve the existing tg_bag_cache shape.

        # RNA expression: validate the TF/TG resolve in the RNA matrix. The actual values
        # are gathered lazily, same as accessibility above, not materialized here.
        tf_rna_idx = gene_to_rna_idx.get(tf_name)
        tg_rna_idx = gene_to_rna_idx.get(tg_name)

        if tf_rna_idx is None or tg_rna_idx is None:
            raise ValueError(
                f"TF or TG missing from RNA matrix after filtering: "
                f"tf_name={tf_name}, tg_name={tg_name}, "
                f"tf_rna_idx={tf_rna_idx}, tg_rna_idx={tg_rna_idx}"
            )

        # Append once per TF-TG edge
        tf_names.append(tf_name)
        tg_names.append(tg_name)
        cell_ids_all.append(sampled_cells)
        labels.append(label)

        tf_indices.append(tf_idx)
        tg_indices.append(tg_idx)
        cell_indices_all.append(sampled_cell_indices)

    if len(labels) == 0:
        raise ValueError(
            "No TF-TG examples were created. Check TF/TG IDs, peak-to-gene mapping, "
            "and overlap with ATAC/RNA matrices."
        )

    # TG-level peak table: one row per TG in the global tg_id_to_idx universe, built from
    # tg_bag_cache (already deduplicated per TG by the loop above -- every edge sharing a
    # TG has the identical peak set/distances/mask). Rows for TGs that never produced an
    # edge (tg_bag_cache[name] is None, or absent from tg_id_to_idx) are left zero-filled;
    # no edge's tg_idx ever addresses them, so their content doesn't matter.
    n_tg_total = len(tg_id_to_idx)
    tg_peak_indices_arr = np.zeros((n_tg_total, max_peaks_real), dtype=np.int64)
    tg_peak_distance_arr = np.zeros((n_tg_total, max_peaks_real), dtype=np.float32)
    tg_peak_mask_arr = np.zeros((n_tg_total, max_peaks_real), dtype=bool)
    for tg_name, bag in tg_bag_cache.items():
        if bag is None:
            continue
        row = tg_id_to_idx.get(tg_name)
        if row is None:
            continue
        peak_idx_row, peak_dst_row, peak_mask_row, _, _ = bag
        tg_peak_indices_arr[row] = peak_idx_row
        tg_peak_distance_arr[row] = peak_dst_row
        tg_peak_mask_arr[row] = peak_mask_row

    return {
        "tf_name": tf_names,
        "tg_name": tg_names,
        "cell_ids": cell_ids_all,

        "label": torch.tensor(labels, dtype=torch.float32),

        "tf_idx": torch.tensor(tf_indices, dtype=torch.long),
        "tg_idx": torch.tensor(tg_indices, dtype=torch.long),
        "cell_indices": torch.tensor(np.stack(cell_indices_all), dtype=torch.long),

        "tg_peak_indices": torch.tensor(tg_peak_indices_arr, dtype=torch.long),
        "tg_peak_distance": torch.tensor(tg_peak_distance_arr, dtype=torch.float32),
        "tg_peak_mask": torch.tensor(tg_peak_mask_arr, dtype=torch.bool),
    }