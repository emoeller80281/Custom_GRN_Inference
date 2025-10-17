import os
import gc
import numpy as np
import pandas as pd
import psutil
from pathlib import Path
from pyfaidx import Fasta
import pybedtools
import MOODS.parsers
import MOODS.scan
import MOODS.tools
import logging
import pyfaidx
from typing import List, Tuple, Any
from tqdm import tqdm
from joblib import Parallel, delayed
from glob import glob

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

DNA = "ACGT"
IDX = {c:i for i,c in enumerate(DNA)}

def load_pfms(pfm_paths, pseudocount=0.001, bg=None):
    """
    Load PFMs from .pfm files and convert them directly to log-odds matrices using MOODS.

    Returns:
        score_mats (list): List of [4 x L] log-odds matrices
        names (list): Motif names derived from file basenames
    """
    if bg is None:
        bg = MOODS.tools.flat_bg(4)

    score_mats = []
    names = []

    logging.info(f"Loading {len(pfm_paths)} PFMs using MOODS.parsers.pfm_to_log_odds")

    for path in pfm_paths:
        mat = MOODS.parsers.pfm_to_log_odds(path, bg, pseudocount)
        if not mat or len(mat) != 4 or any(len(row) == 0 for row in mat):
            logging.warning(f"Skipping malformed PFM: {path}")
            continue
        score_mats.append(mat)
        
        # Derive motif name
        base = os.path.basename(path)
        name = os.path.splitext(base)[0]
        names.append(name.capitalize())

    logging.info(f"Loaded {len(score_mats)} valid PFMs")
    return score_mats, names

def reverse_complement_pwm(pwm):
    """
    pwm: list of lists or np.array shape [L, 4] with columns [A, C, G, T]
    returns: reversed-complemented PWM
    """
    # Reverse along length, then swap A<->T and C<->G
    pwm = np.array(pwm)
    rc = pwm[::-1, :][:, [3, 2, 1, 0]]
    return rc.tolist()

def build_first_order_bg(fasta, peaks_bed, sample=200000):
    """ crude 1st-order background over A,C,G,T transitions in peak sequences """
    logging.info("Building first-order background distribution")
    counts = np.ones((4,4))  # Laplace
    bt = pybedtools.BedTool(peaks_bed)
    n = 0
    for iv in bt:
        seq = fasta[iv.chrom][iv.start:iv.end].upper()
        seq = ''.join(c for c in seq if c in DNA)
        for x,y in zip(seq, seq[1:]):
            counts[IDX[x], IDX[y]] += 1
            n += 1
            if n>=sample: break
        if n>=sample: break
    trans = counts / counts.sum(axis=1, keepdims=True)
    # convert to zero-order for MOODS needs: stationary distribution π s.t. πP=π
    vals, vecs = np.linalg.eig(trans.T)
    i = np.argmin(np.abs(vals-1))
    pi = np.real(vecs[:, i]); pi = np.clip(pi, 1e-9, None); pi /= pi.sum()
    logging.info(f"Background distribution: {pi}")
    return pi.tolist()

def extract_peak_seqs(fasta, peaks_bed):
    logging.info(f"Extracting sequences from {peaks_bed}")
    bt = pybedtools.BedTool(peaks_bed)
    seqs = []
    ids  = []

    for k, iv in enumerate(bt):
        try:
            pid = iv.name if iv.name not in (None, "", ".") else f"{iv.chrom}:{iv.start}-{iv.end}"
            ids.append(pid)
            seqs.append(fasta[iv.chrom][iv.start:iv.end].upper())
        except pyfaidx.FetchError:
            print(f"  WARNING: Skipping peak {iv.chrom}:{iv.start}-{iv.end}, end position is outside {iv.chrom}")

    logging.info(f"Extracted {len(seqs)} peak sequences")
    return ids, seqs

def scan_pwms_batch(fwd_mats, rc_mats, thresholds, names, seqs, bg):
    """
    Scan sequences using MOODS.scan.Scanner for faster repeated scanning.

    Parameters
    ----------
    fwd_mats : list
        Forward-strand log-odds matrices
    rc_mats : list
        Reverse-complement log-odds matrices
    thresholds : list
        Per-motif detection thresholds
    names : list
        Motif names
    seqs : list[str]
        DNA sequences to scan
    bg : list[float]
        Background probabilities
    """
    # Prepare scanners once per strand
    max_len = max(len(m[0]) for m in fwd_mats)
    scanner_fwd = MOODS.scan.Scanner(max_len)
    scanner_rc = MOODS.scan.Scanner(max_len)

    scanner_fwd.set_motifs(fwd_mats, bg, thresholds)
    scanner_rc.set_motifs(rc_mats, bg, thresholds)

    results = {}

    for mi, tf in enumerate(names):
        motif_hits = []
        for seq in seqs:
            fw_hits = scanner_fwd.scan(seq)[mi]
            rc_hits = scanner_rc.scan(seq)[mi]

            # Find best-scoring hit per strand
            best_fw = max(fw_hits, key=lambda h: h.score, default=None)
            best_rc = max(rc_hits, key=lambda h: h.score, default=None)

            # Record top hit across both strands
            if best_fw is None and best_rc is None:
                motif_hits.append((np.nan, np.nan, "."))
            elif best_rc is None or (best_fw and best_fw.score >= best_rc.score):
                motif_hits.append((best_fw.pos, best_fw.score, "+"))
            else:
                motif_hits.append((best_rc.pos, best_rc.score, "-"))

        results[tf] = motif_hits
        
    del scanner_fwd, scanner_rc
    gc.collect()
        
    return results


def run_moods_scan_batched(
    peaks_bed,
    fasta_path,
    motif_paths,
    out_file,
    n_cpus,
    pval_threshold=1e-4,
    bg="auto",
    batch_size=100,
):
    """
    Memory-efficient MOODS scanning by chunking peaks.

    Parameters
    ----------
    peaks_bed : pd.DataFrame or BedTool
        Peaks with columns ["chrom","start","end","peak_id"]
    fasta_path : str
        Path to genome FASTA
    motif_paths : list[str]
        List of PFM files
    out_file : str
        Output parquet path (appended per batch)
    n_cpus : int
        Number of threads for scanning
    pval_threshold : float
        MOODS P-value threshold
    bg : "auto" or list[float]
        Background probabilities
    batch_size : int
        Number of peaks to process per batch
    """
    fasta = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
    pi = build_first_order_bg(fasta, peaks_bed) if bg == "auto" else bg
    
    # Load motifs directly as log-odds matrices
    pseudocount = 0.0001
    pfms, names = load_pfms(motif_paths, pseudocount, bg=pi)
    logging.info(f"Loaded {len(pfms)} PFMs from {len(motif_paths)} motif files")
    thresholds = [MOODS.tools.threshold_from_p(m, pi, pval_threshold) for m in pfms]
    
    # Precompute reverse complements
    rc_pfms = [reverse_complement_pwm(p) for p in pfms]
    
    # Extract the DNA sequences from the genome fasta for each peak
    peak_ids, seqs = extract_peak_seqs(fasta, peaks_bed)
    # Process each peak batch
    n_total = len(peak_ids)
    
    # Split into motif batches
    motif_batch_size = 25
    motif_batches: List[Tuple[list[Any], list[Any], list[Any], list[str]]]  = [
        (
            pfms[j:j+motif_batch_size],
            rc_pfms[j:j + motif_batch_size],
            thresholds[j:j+motif_batch_size],
            names[j:j+motif_batch_size]
        )
        for j in range(0, len(pfms), motif_batch_size)
    ]

    logging.info(f"Running {len(motif_batches)} motif sub-batches in parallel across {n_cpus} cores")
    for i in tqdm(range(0, n_total, batch_size), desc="Peak Batch", position=0):
        logging.info(f"Memory available before batch: {psutil.virtual_memory().available / 1e9:.2f} GB")

        batch_ids = peak_ids[i:i + batch_size]
        batch_seqs = seqs[i:i + batch_size]

        # Parallel scan across motif sub-batches
        batch_results = Parallel(n_jobs=min(n_cpus, len(motif_batches)), backend="threads")(
            delayed(scan_pwms_batch)(
                fwd_subset,
                rc_subset,
                thresholds_subset,
                name_subset,
                batch_seqs,
                bg=pi,
            )
            for fwd_subset, rc_subset, thresholds_subset, name_subset in motif_batches
        )

        # Write each sub-batch to Parquet
        for j, (res, (fwd_subset, rc_subset, thresholds_subset, name_subset)) in enumerate(zip(batch_results, motif_batches)):
            rows = [
                (pid, tf, pos, score, strand)
                for tf in name_subset
                for pid, (pos, score, strand) in zip(batch_ids, res[tf])
            ]
            df = pd.DataFrame(rows, columns=["peak_id", "TF", "pos", "logodds", "strand"])
            tmp_path = f"{out_file}.batch{i:03d}_motifs{j:03d}.parquet"
            df.to_parquet(tmp_path, compression="snappy", index=False)
            del res, rows, df
            
        del batch_results
        gc.collect()

    # ---------------------------------------------------------------------
    # Final merge: concatenate all batch shards into a single Parquet file
    # ---------------------------------------------------------------------
    logging.info("Merging all temporary Parquet shards...")

    tmp_files = sorted(glob(f"{out_file}.batch*.parquet"))
    if tmp_files:
        # Arrow engine can efficiently concatenate in memory
        merged_df = pd.concat(
            (pd.read_parquet(f) for f in tmp_files),
            ignore_index=True
        )
        merged_df.to_parquet(out_file, compression="snappy", index=False)

        # Optional cleanup
        for f in tmp_files:
            os.remove(f)

        logging.info(f"Wrote merged Parquet file: {out_file} ({len(merged_df):,} rows)")
    else:
        logging.warning("No temporary Parquet shards found — nothing merged.")

