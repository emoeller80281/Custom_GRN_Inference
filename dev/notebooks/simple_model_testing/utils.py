import numpy as np
import pandas as pd
import pyfaidx
from pathlib import Path
from tqdm import tqdm
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

def _encode_centered_peak_onehot(args):
    peak_id, genome_fasta, chrom_sizes, flank_size, pad_out_of_bounds, dtype = args

    with pyfaidx.Fasta(str(genome_fasta)) as genome:
        chrom, peak_start, peak_end = parse_peak(peak_id)

        if chrom not in chrom_sizes:
            raise KeyError(
                f"Chromosome {chrom!r} not found in chrom_sizes. "
                f"Peak: {peak_id}"
            )

        chrom_size = chrom_sizes[chrom]

        peak_center = (peak_start + peak_end) // 2
        seq_start = peak_center - flank_size
        seq_end = peak_center + flank_size
        target_len = 2 * flank_size

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
            seq = ("N" * left_pad) + seq + ("N" * right_pad)

            if len(seq) < target_len:
                seq = seq + ("N" * (target_len - len(seq)))
            elif len(seq) > target_len:
                seq = seq[:target_len]
        else:
            if len(seq) != target_len:
                raise ValueError(
                    f"Peak {peak_id} produced sequence length {len(seq)}, "
                    f"but expected {target_len}. Use pad_out_of_bounds=True "
                    f"for fixed-length output."
                )

        onehot = onehot_dna_sequence(seq).astype(dtype)

        if onehot.shape != (target_len, 4):
            raise ValueError(
                f"Peak {peak_id} produced one-hot shape {onehot.shape}, "
                f"but expected {(target_len, 4)}."
            )

    return peak_id, onehot

def create_centered_peak_onehot_array(
    peak_ids: list[str],
    genome_fasta: str | Path,
    chrom_sizes: dict[str, int],
    peak_id_to_idx: dict[str, int],
    flank_size: int,
    dtype=np.float32,
    pad_out_of_bounds: bool = True,
    show_progress: bool = True,
    num_workers: int = 1,
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
        Output dtype.
    pad_out_of_bounds : bool
        Whether to pad with N if the requested window goes out of bounds.
    show_progress : bool
        Whether to show tqdm progress bar.
    num_workers : int
        Number of worker processes to use. Use 1 to run serially.

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

    missing_peaks = [peak_id for peak_id in peak_ids if peak_id not in peak_id_to_idx]
    if len(missing_peaks) > 0:
        raise KeyError(
            f"{len(missing_peaks)} peak_ids are missing from peak_id_to_idx. "
            f"Example: {missing_peaks[:5]}"
        )

    seq_len = 2 * flank_size
    num_peaks = len(peak_id_to_idx)

    peak_onehot_array = np.zeros(
        (num_peaks, seq_len, 4),
        dtype=dtype,
    )

    if num_workers <= 1:
        with pyfaidx.Fasta(str(genome_fasta)) as genome:
            num_peaks = len(peak_ids)
            min_iters = max(num_peaks // 100, 1)
            for peak_id in tqdm(
                peak_ids,
                desc="Creating centered peak one-hot array",
                disable=not show_progress,
                ncols=100,
            ):
                peak_idx = peak_id_to_idx[peak_id]

                chrom, peak_start, peak_end = parse_peak(peak_id)

                if chrom not in chrom_sizes:
                    raise KeyError(
                        f"Chromosome {chrom!r} not found in chrom_sizes. "
                        f"Peak: {peak_id}"
                    )

                chrom_size = chrom_sizes[chrom]

                peak_center = (peak_start + peak_end) // 2

                seq_start = peak_center - flank_size
                seq_end = peak_center + flank_size
                target_len = seq_len

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
                    seq = ("N" * left_pad) + seq + ("N" * right_pad)

                    if len(seq) < target_len:
                        seq = seq + ("N" * (target_len - len(seq)))
                    elif len(seq) > target_len:
                        seq = seq[:target_len]
                else:
                    if len(seq) != target_len:
                        raise ValueError(
                            f"Peak {peak_id} produced sequence length {len(seq)}, "
                            f"but expected {target_len}. Use pad_out_of_bounds=True "
                            f"for fixed-length output."
                        )

                onehot = onehot_dna_sequence(seq).astype(dtype)

                if onehot.shape != (seq_len, 4):
                    raise ValueError(
                        f"Peak {peak_id} produced one-hot shape {onehot.shape}, "
                        f"but expected {(seq_len, 4)}."
                    )

                peak_onehot_array[peak_idx] = onehot
    else:
        tasks = [
            (peak_id, genome_fasta, chrom_sizes, flank_size, pad_out_of_bounds, dtype)
            for peak_id in peak_ids
        ]
        num_peaks = len(peak_ids)
        min_iters = max(num_peaks // 100, 1)
        pbar = tqdm(
            total=num_peaks,
            desc="Creating centered peak one-hot array",
            disable=not show_progress,
            ncols=100,
            miniters=min_iters,
        )
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_peak = {
                    executor.submit(_encode_centered_peak_onehot, task): task[0]
                    for task in tasks
                }
                for future in as_completed(future_to_peak):
                    peak_id, onehot = future.result()
                    peak_idx = peak_id_to_idx[peak_id]
                    peak_onehot_array[peak_idx] = onehot
                    pbar.update(1)
        finally:
            pbar.close()

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
    Create observed positive and sampled-unobserved negative TF-item edges.

    This function is generic. The second entity can be a peak, target gene,
    enhancer, motif, or any other item.

    Examples
    --------
    TF-peak edges:
        tf_col="source_id"
        item_col="peak_id"

    TF-TG edges:
        tf_col="source_id"
        item_col="tg_id"

    Parameters
    ----------
    edge_df : pd.DataFrame
        DataFrame containing observed TF-item interactions.

    tf_names : list
        TF names to include.

    tf_col : str
        Column containing TF names.

    item_col : str
        Column containing the second entity in the edge.
        For TF-peak edges, this could be "peak_id".
        For TF-TG edges, this could be "tg_id".

    pct_true_edges : float or None
        Fraction of observed positive edges to include.
        If None, use all observed positive edges.

    true_false_ratio : float
        Number of sampled unobserved edges per observed positive edge.

    seed : int
        Random seed.

    batch_size : int
        Number of candidate pairs to sample per batch.

    show_progress : bool
        Whether to show a tqdm progress bar.

    Returns
    -------
    true_edges : set[tuple]
        Observed positive TF-item edges.

    false_edges : set[tuple]
        Sampled unobserved TF-item edges.
        These are not guaranteed biological false edges.
    """

    df_all = edge_df.copy()

    # Keep only requested TFs
    df_all = df_all[df_all[tf_col].isin(tf_names)].copy()

    # Drop missing values
    df_all = df_all.dropna(subset=[tf_col, item_col])

    # Normalize to string for safer set operations
    df_all[tf_col] = df_all[tf_col].astype(str)
    df_all[item_col] = df_all[item_col].astype(str)

    # Each observed edge should count once
    df_all = df_all.drop_duplicates([tf_col, item_col]).reset_index(drop=True)

    if df_all.empty:
        raise ValueError(
            f"No edges remain after filtering by tf_names using columns "
            f"{tf_col!r} and {item_col!r}."
        )

    # Build the candidate universe before optional positive subsampling
    candidate_tfs = np.asarray(sorted(df_all[tf_col].unique()), dtype=object)
    candidate_items = np.asarray(sorted(df_all[item_col].unique()), dtype=object)

    # Optionally subsample observed positives
    df_pos = df_all

    if pct_true_edges is not None:
        if not (0 < pct_true_edges <= 1):
            raise ValueError("pct_true_edges must be in (0, 1] or None.")

        df_pos = df_all.sample(frac=pct_true_edges, random_state=seed)

    true_edges = set(zip(df_pos[tf_col], df_pos[item_col]))

    num_false_edges = round(len(true_edges) * true_false_ratio)

    false_edges = sample_unobserved_edges(
        tfs=candidate_tfs,
        items=candidate_items,
        num_edges=num_false_edges,
        true_edges=true_edges,
        batch_size=batch_size,
        seed=seed,
        show_progress=show_progress,
    )

    return true_edges, false_edges


def sample_unobserved_edges(
    tfs,
    items,
    num_edges,
    true_edges,
    batch_size=1_000_000,
    seed=123,
    show_progress=True,
):
    """
    Randomly sample unique TF-item pairs that are not in true_edges.

    These sampled pairs are unobserved negatives, not guaranteed biological
    false interactions.
    """

    tfs = np.asarray(list(tfs), dtype=object)
    items = np.asarray(list(items), dtype=object)
    true_edges = set(true_edges)

    n_tfs = len(tfs)
    n_items = len(items)

    universe_size = n_tfs * n_items
    max_unobserved_edges = universe_size - len(true_edges)

    if max_unobserved_edges <= 0:
        raise ValueError(
            "No unobserved TF-item pairs are available. "
            "The true_edges set covers the full TF x item universe."
        )

    if num_edges > max_unobserved_edges:
        logging.warning(
            f"Requested {num_edges} sampled unobserved edges, but only "
            f"{max_unobserved_edges} are possible. Returning all possible."
        )
        num_edges = max_unobserved_edges

    rng = np.random.default_rng(seed)
    sampled_edges = set()

    pbar = tqdm(
        total=num_edges,
        desc="Generating sampled unobserved TF-item edges",
        ncols=125,
        disable=not show_progress,
    )

    try:
        while len(sampled_edges) < num_edges:
            remaining = num_edges - len(sampled_edges)
            this_batch_size = min(batch_size, max(remaining * 3, 10_000))

            tf_idx = rng.integers(0, n_tfs, size=this_batch_size)
            item_idx = rng.integers(0, n_items, size=this_batch_size)

            old_count = len(sampled_edges)

            for edge in zip(tfs[tf_idx], items[item_idx]):
                if edge not in true_edges:
                    sampled_edges.add(edge)

                    if len(sampled_edges) >= num_edges:
                        break

            pbar.update(len(sampled_edges) - old_count)

    finally:
        pbar.close()

    return sampled_edges

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
                print(f"[{i}/{len(gene_names)}] No records found for {gene_name}")
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
                print(f"[{i}/{len(gene_names)}] Could not parse records for {gene_name}")
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

            print(
                f"[{i}/{len(gene_names)}] Saved {gene_name}: "
                f"{best_record.id} ({len(best_record.seq)} aa)"
            )

        except Exception as e:
            print(f"[{i}/{len(gene_names)}] Failed for {gene_name}: {e}")
            saved_files[gene_name] = None

        time.sleep(delay)

    return saved_files

def fetch_chip_atlas_tf_list(tf_list, genome="mm10", num_workers=10):
    
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