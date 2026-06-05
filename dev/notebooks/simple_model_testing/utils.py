import numpy as np
import pandas as pd
import duckdb
import pyfaidx
from pathlib import Path
from tqdm.auto import tqdm
import re
import time
import requests
from Bio import Entrez, SeqIO
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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

from itertools import repeat

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
    return pd.concat(gt_dfs, ignore_index=True)

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

_GENOME_HANDLE = None


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

    df_all = edge_df[[tf_col, item_col]].copy()

    df_all = df_all[df_all[tf_col].isin(tf_names)]
    df_all = df_all.dropna(subset=[tf_col, item_col])

    df_all[tf_col] = df_all[tf_col].astype(str)
    df_all[item_col] = df_all[item_col].astype(str)

    df_all = df_all.drop_duplicates([tf_col, item_col]).reset_index(drop=True)

    if df_all.empty:
        raise ValueError(
            f"No edges remain after filtering by tf_names using columns "
            f"{tf_col!r} and {item_col!r}."
        )

    # Candidate universe
    candidate_tfs = np.asarray(sorted(df_all[tf_col].unique()), dtype=object)
    candidate_items = np.asarray(sorted(df_all[item_col].unique()), dtype=object)

    tf_to_i = {tf: i for i, tf in enumerate(candidate_tfs)}
    item_to_i = {item: i for i, item in enumerate(candidate_items)}

    n_tfs = len(candidate_tfs)
    n_items = len(candidate_items)

    # Integer-code all observed edges.
    # These should be excluded from negative sampling.
    obs_tf_idx = df_all[tf_col].map(tf_to_i).to_numpy(dtype=np.int64)
    obs_item_idx = df_all[item_col].map(item_to_i).to_numpy(dtype=np.int64)

    observed_codes = obs_tf_idx * n_items + obs_item_idx

    # Use np.unique so membership checks are vectorized with np.isin.
    observed_codes = np.unique(observed_codes)

    # Subsample positives after defining all observed codes.
    if pct_true_edges is not None:
        if not (0 < pct_true_edges <= 1):
            raise ValueError("pct_true_edges must be in (0, 1] or None.")

        df_pos = df_all.sample(frac=pct_true_edges, random_state=seed)
    else:
        df_pos = df_all

    true_edges = set(zip(df_pos[tf_col], df_pos[item_col]))

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
    sampled_chunks = []
    sampled_count = 0

    # Keep a Python set only for sampled integer codes, not tuple objects.
    sampled_seen = set()

    pbar = tqdm(
        total=num_edges,
        desc="Generating sampled unobserved TF-item edges",
        ncols=125,
        disable=not show_progress,
    )

    try:
        while sampled_count < num_edges:
            remaining = num_edges - sampled_count
            this_batch_size = min(batch_size, max(remaining * 3, 10_000))

            tf_idx = rng.integers(0, n_tfs, size=this_batch_size, dtype=np.int64)
            item_idx = rng.integers(0, n_items, size=this_batch_size, dtype=np.int64)

            codes = tf_idx * n_items + item_idx

            # Remove duplicates within this batch.
            codes = np.unique(codes)

            # Remove known observed positives.
            is_observed = np.isin(codes, observed_codes, assume_unique=False)
            codes = codes[~is_observed]

            if len(codes) == 0:
                continue

            # Remove codes already sampled in previous batches.
            # This small Python loop is much cheaper than tuple loops.
            new_codes = []
            for code in codes:
                code_int = int(code)
                if code_int not in sampled_seen:
                    sampled_seen.add(code_int)
                    new_codes.append(code_int)

                    if sampled_count + len(new_codes) >= num_edges:
                        break

            if not new_codes:
                continue

            new_codes = np.asarray(new_codes, dtype=np.int64)
            sampled_chunks.append(new_codes)

            sampled_count += len(new_codes)
            pbar.update(len(new_codes))

    finally:
        pbar.close()

    sampled_codes = np.concatenate(sampled_chunks)

    if len(sampled_codes) > num_edges:
        sampled_codes = sampled_codes[:num_edges]

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

def fetch_chip_atlas_tf_list(tf_list, genome="mm10", num_workers=10) -> pd.DataFrame:
    
    def fetch_chip_atlas_tf(tf, genome=genome, threshold="05", timeout=120):
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
                )

            df["source_id"] = tf
            df["peak_id"] = df["peak_chr"] + ":" + df["peak_start"].astype(str) + "-" + df["peak_end"].astype(str)
            df = df[["source_id", "peak_id"]]
            
            return tf, df, None

        except requests.exceptions.HTTPError as e:
            return tf, None, e
        except Exception as e:
            return tf, None, e


    tf_dfs = []
    failed_tfs = {}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(fetch_chip_atlas_tf, tf, genome=genome): tf
            for tf in tf_list
        }

        for future in as_completed(futures):
            tf, df, error = future.result()

            if error is not None:
                failed_tfs[tf] = error
                logging.info(f"TF '{tf}' not found or failed: {error}")
                continue

            if df is None or df.empty:
                failed_tfs[tf] = "empty dataframe"
                logging.info(f"TF '{tf}' returned no peaks")
                continue

            tf_dfs.append(df)
            logging.info(f"Loaded {tf}: {len(df):,} peaks")

    if len(tf_dfs) == 0:
        logging.warning("No ChIP-Atlas TF BED files were successfully loaded.")
        chip_atlas_df = pd.DataFrame(columns=["source_id", "peak_id"])
    else:
        chip_atlas_df = pd.concat(tf_dfs, ignore_index=True).drop_duplicates()

    return chip_atlas_df

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