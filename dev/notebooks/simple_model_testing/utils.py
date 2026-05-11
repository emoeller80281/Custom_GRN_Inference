import numpy as np
import pandas as pd
from random import sample
import pyfaidx
from pathlib import Path
from tqdm import tqdm
import re
import time
from Bio import Entrez, SeqIO

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
    peak_chrom, peak_start, peak_end = parse_peak(selected_peak)

    # Load peak sequence using the genome fasta file
    with pyfaidx.Fasta(genome_fasta) as genome:
        peak_sequence = genome[peak_chrom][peak_start:peak_end].seq.upper()
        
    return peak_sequence
        
    

def get_centered_peak_sequence(
    genome_fasta: str | Path,
    peak,
    chrom_sizes: dict[str, int],
    flank_size=None,
    pad_out_of_bounds=True,
):
    """
    Extract a fixed-width sequence centered on a peak.

    If the requested window extends beyond chromosome bounds, pad with N's.
    N's become all-zero rows during one-hot encoding.
    """
    
    assert Path(genome_fasta).exists(), \
        f"Genome fasta file not found: {genome_fasta}"
    
    chrom, coords = peak.split(":")
    peak_start, peak_end = map(int, coords.split("-"))

    peak_center = (peak_start + peak_end) // 2
    seq_start = peak_center - flank_size
    seq_end = peak_center + flank_size
    target_len = 2 * flank_size

    chrom_size = chrom_sizes.get(chrom)

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
        
    with pyfaidx.Fasta(genome_fasta) as genome:
        seq = genome[chrom][fetch_start:fetch_end].seq.upper()

    if pad_out_of_bounds:
        seq = ("N" * left_pad) + seq + ("N" * right_pad)

        if len(seq) < target_len:
            seq = seq + ("N" * (target_len - len(seq)))
        elif len(seq) > target_len:
            seq = seq[:target_len]

    return seq



def create_true_false_edges(
    chip_atlas_df: pd.DataFrame, 
    tf_names: list, 
    tf_col: str ="source_id", 
    peak_col: str ="peak_id", 
    sample_frac: float | None =0.50
    ):
    chipatlas_df = chip_atlas_df.copy()
    chipatlas_df[tf_col] = chipatlas_df[tf_col]
    chipatlas_df = chipatlas_df[chipatlas_df[tf_col].isin(tf_names)].reset_index(drop=True)
    
    if sample_frac is not None:
        chipatlas_df = chipatlas_df.sample(frac=sample_frac, random_state=123)

    chipatlas_true_interactions = set(zip(chipatlas_df[tf_col], chipatlas_df[peak_col]))

    def create_n_false_edges(
        tfs,
        peaks,
        num_false_edges,
        true_interactions,
        batch_size=1_000_000,
        seed=None,
        show_progress=True,
    ):
        """
        Randomly sample unique TF-peak pairs that are not in true_interactions.

        Parameters
        ----------
        tfs : iterable
            TF names.
        peaks : iterable
            Peak IDs.
        num_false_edges : int
            Number of false edges to generate.
        true_interactions : set[tuple]
            Known true TF-peak interactions to exclude.
        batch_size : int
            Number of candidate pairs to sample per batch.
        seed : int or None
            Random seed.
        show_progress : bool
            Whether to show tqdm progress bar.

        Returns
        -------
        set[tuple]
            Set of sampled false interactions.
        """

        tfs = np.asarray(list(tfs), dtype=object)
        peaks = np.asarray(list(peaks), dtype=object)
        true_interactions = set(true_interactions)

        n_tfs = len(tfs)
        n_peaks = len(peaks)
        universe_size = n_tfs * n_peaks
        max_false_edges = universe_size - len(true_interactions)

        if num_false_edges > max_false_edges:
            raise ValueError(
                f"Requested {num_false_edges:,} false edges, but only "
                f"{max_false_edges:,} possible false edges exist."
            )

        rng = np.random.default_rng(seed)
        false_interactions = set()

        pbar = tqdm(
            total=num_false_edges,
            desc="Generating False Interactions",
            ncols=125,
            disable=not show_progress,
        )

        try:
            while len(false_interactions) < num_false_edges:
                remaining = num_false_edges - len(false_interactions)

                this_batch_size = min(batch_size, max(remaining * 3, 10_000))

                tf_idx = rng.integers(0, n_tfs, size=this_batch_size)
                peak_idx = rng.integers(0, n_peaks, size=this_batch_size)

                candidates = zip(tfs[tf_idx], peaks[peak_idx])

                old_count = len(false_interactions)

                for edge in candidates:
                    if edge not in true_interactions:
                        false_interactions.add(edge)

                        if len(false_interactions) >= num_false_edges:
                            break

                new_count = len(false_interactions)
                pbar.update(new_count - old_count)

        finally:
            pbar.close()

        false_tfs = {tf for tf, _ in false_interactions}
        false_peaks = {peak for _, peak in false_interactions}

        logging.info(f"Total unique TFs in false interactions: {len(false_tfs):,}")
        logging.info(f"Total unique peaks in false interactions: {len(false_peaks):,}")
        logging.info(f"Total false interactions: {len(false_interactions):,}")

        return false_interactions

    false_interactions = create_n_false_edges(
        tfs=chipatlas_df["source_id"].unique(),
        peaks=chipatlas_df["peak_id"].unique(),
        num_false_edges=len(chipatlas_true_interactions) // 4,
        true_interactions=chipatlas_true_interactions,
        batch_size=1_000_000,
        seed=123,
    )
    
    return chipatlas_true_interactions, false_interactions

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