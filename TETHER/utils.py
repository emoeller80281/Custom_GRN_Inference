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

import models.tf_to_dna as tf_to_dna_module
from scripts.train_tf_to_dna_model import TFPeakEdgeDataset

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
    progress_miniters: int | None = None,
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
    progress_miniters : int or None
        Minimum completed peaks between progress updates. None keeps the default.

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
        miniters=(max(num_encoded_peaks // 1000, 1)
                  if progress_miniters is None else max(progress_miniters, 1)),
        maxinterval=10.0 if progress_miniters is None else float("inf"),
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


def record_gene_symbol(record):
    """Gene symbol of a GenBank protein record, from the /gene qualifier of its CDS feature."""
    for feature in record.features:
        if feature.type in ("CDS", "gene", "Protein") and "gene" in feature.qualifiers:
            return feature.qualifiers["gene"][0]
    return None


def download_gene_protein_fastas(
    gene_names,
    organism,
    output_dir,
    email,
    api_key=None,
    retmax=100,
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

            # [Gene Name] also matches other genes' aliases, and the ranking below prefers the
            # longest protein, so an alias hit could win: Elf1 got Sptbn1 (alias "Elf"), Vdr got
            # Cyp27b1, Pdx1 got Pdhx. Keep only the records of the gene itself. Fall back to an
            # alias hit only when every record belongs to one gene (a renamed symbol, e.g.
            # Arntl -> Bmal1).
            by_gene = {}
            for record in records:
                by_gene.setdefault(record_gene_symbol(record), []).append(record)
            exact = [r for g, rs in by_gene.items() if g and g.lower() == gene_name.lower() for r in rs]
            if exact:
                records = exact
            elif len(by_gene) == 1:
                logging.info(f"[{i}/{len(gene_names)}] {gene_name}: no record under that symbol; "
                             f"using {next(iter(by_gene))} (renamed or alias)")
            else:
                logging.info(f"[{i}/{len(gene_names)}] {gene_name}: no record under that symbol and hits "
                             f"from several genes ({', '.join(sorted(map(str, by_gene)))}); skipped")
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

def load_tf_dna_data(tf_dna_input_cache_dir: Path) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Loads the TF-DNA dataset from the specified cache directory.
    
    Parameters
    ----------
    tf_dna_input_cache_dir : Path
        Path to the directory containing the cached TF-DNA dataset files.
        
    Returns
    -------
    tuple of torch.Tensor
        A tuple containing the following tensors:
        - edge_tf_idx_tensor: Tensor containing the edge indices for transcription factors.
        - edge_peak_idx_tensor: Tensor containing the edge indices for peaks.
        - edge_labels_tensor: Tensor containing the labels for the edges.
        - tf_embeddings_tensor: Tensor containing the TF embeddings.
        - tf_mask_tensor: Tensor containing the masks for the TF embeddings.
        - one_hot_peaks: Tensor containing the one-hot encoded peak sequences.
    """
    # Load the edge indices for the TF-DNA dataset
    edge_tf_idx_tensor: torch.Tensor = torch.load(tf_dna_input_cache_dir / "edge_tf_idx.pt", weights_only=True)
    edge_peak_idx_tensor: torch.Tensor = torch.load(tf_dna_input_cache_dir / "edge_peak_idx.pt", weights_only=True)
    edge_labels_tensor: torch.Tensor = torch.load(tf_dna_input_cache_dir / "edge_labels.pt", weights_only=True)

    # Load the TF embeddings and masks
    tf_embeddings_tensor: torch.Tensor = torch.load(tf_dna_input_cache_dir / "tf_embeddings.pt", weights_only=True)
    tf_mask_tensor: torch.Tensor = torch.load(tf_dna_input_cache_dir / "tf_masks.pt", weights_only=True)

    # Load the one-hot encoded peak tensor
    one_hot_peaks: torch.Tensor = torch.load(tf_dna_input_cache_dir / "peak_onehot_array.pt", weights_only=True)

    return (
        edge_tf_idx_tensor,
        edge_peak_idx_tensor,
        edge_labels_tensor,
        tf_embeddings_tensor,
        tf_mask_tensor,
        one_hot_peaks,
    )
    
def load_tf_dna_training_data(
    tf_dna_input_cache_dir: Path,
    batch_size: int = 64,
):    
    
    # Name to ID dictionaries
    tf_name_to_idx_df = pd.read_csv(tf_dna_input_cache_dir / "tf_name_to_idx.csv")
    peak_id_to_idx_df = pd.read_csv(tf_dna_input_cache_dir / "peak_id_to_idx.csv")

    # Load cached data
    tf_dna_data = load_tf_dna_data(tf_dna_input_cache_dir)

    # Unpack all returned tensors
    edge_tf_idx_tensor, edge_peak_idx_tensor, edge_labels_tensor, \
        tf_embeddings_tensor, tf_mask_tensor, peak_tensor = tf_dna_data
    
    # Load train/val/test splits
    train_idx: torch.Tensor = torch.load(tf_dna_input_cache_dir / "train_idx.pt", weights_only=True)
    val_idx: torch.Tensor = torch.load(tf_dna_input_cache_dir / "val_idx.pt", weights_only=True)
    test_idx: torch.Tensor = torch.load(tf_dna_input_cache_dir / "test_idx.pt", weights_only=True)

    # PyTorch Dataset for TF-DNA edges
    edge_dataset = TFPeakEdgeDataset(
        edge_tf_idx=edge_tf_idx_tensor,
        edge_peak_idx=edge_peak_idx_tensor,
        edge_labels=edge_labels_tensor,
        peak_tensor=peak_tensor,
    )
    
    # Subset to only the train, val, and test sets
    train_dataset = Subset(edge_dataset, train_idx.tolist())
    val_dataset = Subset(edge_dataset, val_idx.tolist())
    test_dataset = Subset(edge_dataset, test_idx.tolist())

    # Create dataloaders for each split
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "tf_embeddings_tensor": tf_embeddings_tensor,
        "tf_mask_tensor": tf_mask_tensor,
        "peak_tensor": peak_tensor,
        "edge_dataset": edge_dataset,
        "tf_name_to_idx_df": tf_name_to_idx_df,
        "peak_name_to_idx_df": peak_id_to_idx_df,
    }


def create_tf_peak_index_to_name_mappings(training_data):
    #series to dict
    tf_name_to_idx_dict = training_data["tf_name_to_idx_df"].set_index("tf_name")["tf_idx"].to_dict()
    peak_name_to_idx_dict = training_data["peak_name_to_idx_df"].set_index("peak_id")["peak_idx"].to_dict()

    tf_idx_to_name = {idx: name for name, idx in tf_name_to_idx_dict.items()}
    peak_idx_to_name = {idx: name for name, idx in peak_name_to_idx_dict.items()}
    return tf_idx_to_name, peak_idx_to_name

def get_peak_sequence(peak_idx: int, peak_tensor: torch.Tensor) -> str:
    """
    Returns the DNA sequence for a given peak index from the one-hot encoded peak tensor.
    
    Parameters
    ----------
    peak_idx : int
        The index of the peak for which to retrieve the DNA sequence.
    peak_tensor : torch.Tensor
        The one-hot encoded tensor containing the DNA sequences for all peaks.
        
    Returns
    -------
    str
        The DNA sequence corresponding to the specified peak index.
    
    """
    peak_onehot = peak_tensor[peak_idx]
    nuc_ids = torch.argmax(peak_onehot, dim=-1)
    peak_nuc_key = ["A", "C", "G", "T"]
    return "".join([peak_nuc_key[nuc_id] for nuc_id in nuc_ids])

def generate_tf_dna_binding_predictions(cell_type, tf_dna_model_path):
    training_data = load_tf_dna_training_data(
        cell_type=cell_type,
        batch_size=512,
    )

    tf_embeddings_tensor = training_data["tf_embeddings_tensor"]
    tf_mask_tensor = training_data["tf_mask_tensor"]

    tf_dna_model = load_tf_dna_model(
        tf_dna_model_path=tf_dna_model_path,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        compile_model=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tf_dna_model.model
    model = model.to(device)

    model.eval()

    tf_indices_list = []
    peak_indices_list = []
    all_scores = []
    all_labels = []

    tf_idx_to_name, peak_idx_to_name = create_tf_peak_index_to_name_mappings(training_data)

    test_loader = training_data["test_loader"]

    # print(f"Evaluating on {dataset_split_type} set")
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Evaluating", ncols=100):
            
            tf_idx = batch["tf_idx"].long()
            peak_idx = batch["peak_idx"].long()
            labels = batch["label"]

            tf_embedding = tf_embeddings_tensor[tf_idx].to(device, non_blocking=True)
            tf_mask = tf_mask_tensor[tf_idx].to(device, non_blocking=True)

            peak_embedding = batch["peak_embedding"].float().to(device, non_blocking=True)

            binding_logits = model.forward(
                tf_embedding=tf_embedding,
                tf_mask=tf_mask,
                peak_embedding=peak_embedding,
            )

            scores = torch.sigmoid(binding_logits)

            all_scores.append(scores.detach().cpu().numpy().ravel())
            all_labels.append(labels.numpy().ravel())
            
            tf_indices_list.append(tf_idx.numpy().ravel())
            peak_indices_list.append(peak_idx.numpy().ravel())

    all_tf_indices = np.concatenate(tf_indices_list)
    all_peak_indices = np.concatenate(peak_indices_list)
    all_scores_flat = np.concatenate(all_scores)
    all_labels_flat = np.concatenate(all_labels)

    tf_names = [tf_idx_to_name[idx] for idx in all_tf_indices]
    peak_names = [peak_idx_to_name[idx] for idx in all_peak_indices]

    prediction_df = pd.DataFrame({
        "TF": tf_names,
        "DNA": peak_names,
        "Score": all_scores_flat,
        "Label": all_labels_flat
    })
    
    return prediction_df, all_scores_flat, all_labels_flat
