# moods_scan.py
import os, sys, math, gzip, json
import numpy as np
import pandas as pd
from pathlib import Path
from pyfaidx import Fasta
from Bio import motifs
import pybedtools
import MOODS.scan
import MOODS.tools
import logging
from tqdm import tqdm
from joblib import Parallel, delayed


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
    logging.info(f"Loaded {len(pfms)} motifs")
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
        pid = iv.name if iv.name not in (None, "", ".") else f"{iv.chrom}:{iv.start}-{iv.end}"
        ids.append(pid)
        seqs.append(fasta[iv.chrom][iv.start:iv.end].upper())
    logging.info(f"Extracted {len(seqs)} peak sequences")
    return ids, seqs

def build_score_matrices(pfms, bg, pseudocount=0.001):
    """
    pfms : list of PFMs, each shape [L,4]
    bg   : background probs [A,C,G,T]
    """
    score_mats = []
    for pfm in pfms:
        pfm = np.array(pfm, dtype=float)
        pfm = pfm + pseudocount
        pfm = pfm / pfm.sum(axis=1, keepdims=True)

        # MOODS expects motif as columns = positions, rows = A,C,G,T
        pwm = pfm.T.tolist()  # shape [4, L]

        # Create log-odds score matrix
        sm = MOODS.tools.log_odds(pwm, bg, pseudocount)
        score_mats.append(sm)
    logging.info(f"Built {len(score_mats)} score matrices")
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

def scan_pwms(pfms, names, seqs, pval_threshold=1e-4, bg=None, pseudocount=0.8, n_jobs=4):
    logging.info(f"Scanning {len(pfms)} motifs across {len(seqs)} sequences with {n_jobs} workers")

    fwd_mats = build_score_matrices(pfms, bg, pseudocount)
    rc_pfms  = [reverse_complement_pwm(p) for p in pfms]
    rc_mats  = build_score_matrices(rc_pfms, bg, pseudocount)
    
    thresholds = [MOODS.tools.threshold_from_p(m, bg, pval_threshold) for m in fwd_mats]
    logging.info(f"Using per-motif thresholds at p={pval_threshold}")
        
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(scan_one_sequence)(seq, fwd_mats, rc_mats, thresholds, names, bg)
        for seq in tqdm(seqs, desc="Scanning sequences", unit="seq")
    )

    # Transpose: seqs x motifs → dict[motif][seq]
    results = {names[mi]: [results_list[si][mi] for si in range(len(seqs))] for mi in range(len(names))}
    logging.info("Finished scanning motifs")
    return results

def run_moods_scan(peaks_bed, fasta_path, motif_paths, out_tsv, n_cpus, pval_threshold=1e-4, bg="auto"):
    fasta = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)
    if bg == "auto":
        pi = build_first_order_bg(fasta, peaks_bed)
    else:
        pi = bg  # list of 4
    pfms, names = load_pfms(motif_paths)

    peak_ids, seqs = extract_peak_seqs(fasta, peaks_bed)
    res = scan_pwms(pfms, names, seqs, pval_threshold=pval_threshold, bg=pi, pseudocount=0.8, n_jobs=n_cpus)

    rows = []
    for mi, tf in enumerate(names):
        for si, pid in enumerate(peak_ids):
            pos, score, strand = res[tf][si]
            rows.append((pid, tf, pos, score, strand))
    df = pd.DataFrame(rows, columns=["peak_id","TF","pos","logodds","strand"])
    df.to_csv(out_tsv, sep="\t", index=False)
    logging.info(f"Wrote results: {out_tsv} ({df.shape[0]:,} rows)")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--motifs", nargs="+", required=True, help="one or more PFM files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pval_threshold", type=float, default=6.0)
    args = ap.parse_args()
    run_moods_scan(args.peaks, args.fasta, args.motifs, args.out, pval_threshold=args.pval_threshold)
