import os
import numpy as np
import pandas as pd
from pathlib import Path
from pyfaidx import Fasta
from Bio import motifs
import pybedtools
import MOODS.scan
import MOODS.tools
import logging
import pyfaidx
from tqdm import tqdm
from joblib import Parallel, delayed
from glob import glob
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

DNA = "ACGT"
IDX = {c:i for i,c in enumerate(DNA)}

def load_pfms(pfm_paths):
    logging.info(f"Loading PFMs from {len(pfm_paths)} files")
    pfms, names = [], []
    for p in pfm_paths:
        with open(p) as f:
            for m in motifs.parse(f, "jaspar"):
                pfm = [[m.counts[nuc][i] for nuc in "ACGT"] for i in range(m.length)]
                pfms.append(pfm)
                
                # prefer motif name, else file basename
                raw_name = m.name or os.path.basename(p)
                # strip .pfm suffix if present
                if raw_name.lower().endswith(".pfm"):
                    raw_name = raw_name[:-4]
                # capitalize (first letter uppercase, rest lowercase)
                clean_name = raw_name.capitalize()
                names.append(clean_name)
    return pfms, names

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


def build_score_matrices(pfms, bg, pseudocount=0.001):
    """Convert PFMs to MOODS score matrices with log-odds."""
    score_mats = []
    for pfm in pfms:
        pfm = np.array(pfm, dtype=float)
        pfm = pfm + pseudocount
        pfm = pfm / pfm.sum(axis=1, keepdims=True)

        # MOODS expects [4, L]
        pwm = pfm.T.tolist()                     # ensure nested list
        bg_vec = [float(x) for x in bg]          # ensure 1D python list
        pc = float(pseudocount)

        # Use the 3-argument form only
        sm = MOODS.tools.log_odds(pwm, bg_vec, pc)
        score_mats.append(sm)

    logging.debug(f"Built {len(score_mats)} score matrices")
    return score_mats



def scan_one_sequence(seq, fwd_mats, rc_mats, thresholds, names, bg):
    s = ''.join(c if c in DNA else 'N' for c in seq)
    fw_hits = MOODS.scan.scan_dna(s, fwd_mats, thresholds, bg)
    rc_hits = MOODS.scan.scan_dna(s, rc_mats, thresholds, bg)

    results = []
    for mi, (fw, rc) in enumerate(zip(fw_hits, rc_hits)):
        best = None
        for hit in fw:
            if best is None or hit.score > best[1]:
                best = (hit.pos, hit.score, "+")
        for hit in rc:
            if best is None or hit.score > best[1]:
                best = (hit.pos, hit.score, "-")
        if best is None:
            results.append((np.nan, np.nan, "."))
        else:
            results.append(best)
    return results

def scan_pwms_batch(pfms, names, seqs, pval_threshold=1e-4, bg=None, pseudocount=0.8):
    """
    Batch version using MOODS.scan.scan_dna_batch (much faster).
    Scans all sequences at once inside the C++ engine.
    """
    logging.debug(f"[MOODS] Scanning {len(pfms)} motifs across {len(seqs)} sequences (batch mode)")

    # Build forward and reverse score matrices once
    fwd_mats = build_score_matrices(pfms, bg, pseudocount)
    rc_pfms  = [reverse_complement_pwm(p) for p in pfms]
    rc_mats  = build_score_matrices(rc_pfms, bg, pseudocount)

    thresholds = [MOODS.tools.threshold_from_p(m, bg, pval_threshold) for m in fwd_mats]
    logging.debug(f"Using per-motif thresholds at p={pval_threshold}")

    fw_hits, rc_hits = [], []
    for seq in tqdm(seqs, desc="Scanning sequences:", position=1):
        fw_hits.append(MOODS.scan.scan_dna(seq, fwd_mats, thresholds, bg))
        rc_hits.append(MOODS.scan.scan_dna(seq, rc_mats, thresholds, bg))

    # Aggregate the best hit (pos, score, strand) per motif per sequence
    results = {}
    for mi, tf in enumerate(names):
        motif_hits = []
        for si in range(len(seqs)):
            fwh = fw_hits[si][mi]
            rch = rc_hits[si][mi]
            best = None
            for h in fwh:
                if best is None or h.score > best[1]:
                    best = (h.pos, h.score, "+")
            for h in rch:
                if best is None or h.score > best[1]:
                    best = (h.pos, h.score, "-")
            if best is None:
                motif_hits.append((np.nan, np.nan, "."))
            else:
                motif_hits.append(best)
        results[tf] = motif_hits

    return results


def run_moods_scan_batched(
    peaks_bed,
    fasta_path,
    motif_paths,
    out_file,
    n_cpus,
    pval_threshold=1e-4,
    bg="auto",
    batch_size=5000,
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
    if bg == "auto":
        pi = build_first_order_bg(fasta, peaks_bed)
    else:
        pi = bg

    pfms, names = load_pfms(motif_paths)
    logging.info(f"Loaded {len(pfms)} PFMs from {len(motif_paths)} motif files")

    peak_ids, seqs = extract_peak_seqs(fasta, peaks_bed)

    # Split all motifs into chunks of size motif_batch_size
    motif_batch_size=100
    motif_batches = [
        (pfms[j:j+motif_batch_size], names[j:j+motif_batch_size])
        for j in range(0, len(pfms), motif_batch_size)
    ]

    # Process each peak batch
    n_total = len(peak_ids)
    logging.info(f"Running {len(motif_batches)} motif sub-batches in parallel across {n_cpus} cores")
    for i in tqdm(range(0, n_total, batch_size), desc="Peak Batch", position=0):
        batch_ids = peak_ids[i:i+batch_size]
        batch_seqs = seqs[i:i+batch_size]


        # Parallel scan across motif sub-batches
        batch_results = Parallel(n_jobs=n_cpus, backend="threading")(
            delayed(scan_pwms_batch)(
                pfm_subset,
                name_subset,
                batch_seqs,
                pval_threshold=pval_threshold,
                bg=pi,
            )
            for pfm_subset, name_subset in motif_batches
        )

        # Write each sub-batch to a uniquely named temporary Parquet file
        for j, (res, (_, name_subset)) in enumerate(zip(batch_results, motif_batches)):
            rows = [
                (pid, tf, pos, score, strand)
                for tf in name_subset
                for pid, (pos, score, strand) in zip(batch_ids, res[tf])
            ]

            df = pd.DataFrame(rows, columns=["peak_id", "TF", "pos", "logodds", "strand"])
            tmp_path = f"{out_file}.batch{i:03d}_motifs{j:03d}.parquet"
            df.to_parquet(tmp_path, compression="snappy", index=False)
            del res, rows, df  # free memory

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

