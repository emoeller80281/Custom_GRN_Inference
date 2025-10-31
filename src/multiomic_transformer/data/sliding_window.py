#!/usr/bin/env python3
import argparse
import logging
import os
import glob
import re
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Union
from pyfaidx import Fasta
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO
import pybedtools
import pyfaidx
from matplotlib.ticker import FuncFormatter
import multiprocessing as mp
import MOODS.tools
import pyarrow as pa
import pyarrow.dataset as ds

from numba import njit, prange
from pyarrow.lib import ArrowInvalid
from pyarrow.parquet import ParquetFile
from scipy import stats
from tqdm import tqdm

# from grn_inference.normalization import minmax_normalize_dask
from grn_inference.plotting import plot_feature_score_histogram
from grn_inference.normalization import (
    clip_and_normalize_log1p_pandas,
    minmax_normalize_pandas
)
from grn_inference.create_homer_peak_file import format_peaks

DNA = "ACGT"
IDX = {c:i for i,c in enumerate(DNA)}

# at module top‐level
_global_chr_pos_to_seq: Union[None, pd.DataFrame] = None
_global_tf_df: Union[None, pd.DataFrame] = None
_global_bg_freq: Union[None, pd.Series] = None
_global_plus: Union[None, np.ndarray] = None
_global_minus: Union[None, np.ndarray] = None

SLIDING_SCHEMA = pa.schema([
    pa.field("TF", pa.string()),
    pa.field("peak_id", pa.string()),
    pa.field("sliding_window_score", pa.float32()),
])

def coalesce_sliding_window_scores_pyarrow(input_dir: str, out_parquet: str, pattern: str = "*.parquet"):
    """
    Coalesce many parquet shards into one file.
    - Expands glob explicitly (pyarrow won't expand wildcards in a single string).
    - Avoids 'hive' partitioning for flat folders.
    - Streams batches and enforces a consistent schema.
    """
    # 1) Expand the glob to real files
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No parquet files matched: {os.path.join(input_dir, pattern)}")

    # 2) Build a dataset from explicit paths (flat layout → no 'partitioning=\"hive\"')
    dataset = ds.dataset(paths, format="parquet")

    # 3) Columns we need + target schema
    cols = ["TF", "peak_id", "sliding_window_score"]
    target_schema = pa.schema([
        pa.field("TF", pa.string()),
        pa.field("peak_id", pa.string()),
        pa.field("sliding_window_score", pa.float32()),
    ])

    # 4) Helper to add missing cols and cast
    def _normalize_table(tbl: pa.Table) -> pa.Table:
        # Add any missing columns as nulls with the right type
        for field in target_schema:
            if field.name not in tbl.schema.names:
                tbl = tbl.append_column(
                    field.name,
                    pa.nulls(len(tbl), type=field.type)
                )
        # Reorder and cast
        tbl = tbl.select(target_schema.names)
        return tbl.cast(target_schema, safe=False)

    # 5) Stream and write
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_parquet, target_schema, compression="snappy")

    # Scanner lets us read only required columns and iterate in batches
    scanner = ds.Scanner.from_dataset(dataset, columns=cols, use_threads=True)
    for batch in scanner.to_batches():
        tbl = pa.Table.from_batches([batch])
        tbl = _normalize_table(tbl)
        writer.write_table(tbl)

    writer.close()

def process_motif_file_and_save(motif_idx, tf_df, bg, score_mats, names, tmp_dir, peak_ids):
    pwm = np.array(score_mats[motif_idx])
    scores = score_all_peaks(_global_plus, _global_minus, pwm)

    if len(scores) != len(peak_ids):
        logging.error(f"Length mismatch for motif {names[motif_idx]}: "
                      f"{len(scores)} scores vs {len(peak_ids)} peaks")
        return False

    motif_name = names[motif_idx]
    mask = tf_df["Motif_ID"] == motif_name
    tf_names = tf_df.loc[mask, "TF_Name"].values

    if len(tf_names) == 0:
        logging.warning(f"No TFs found for motif {motif_name} (motif_idx={motif_idx})")

    # Write one shard per (TF, motif) to avoid overwrites
    pid = np.asarray(peak_ids, dtype="U")  # ensure consistent string dtype
    sc  = scores.astype("float32", copy=False)

    for tf in tf_names:
        shard = pd.DataFrame({
            "TF": tf,                       # broadcasted scalar okay; Arrow will expand
            "peak_id": pid,
            "sliding_window_score": sc,
        })
        # Cast via Arrow to enforce schema
        table = pa.Table.from_pandas(shard, preserve_index=False).cast(SLIDING_SCHEMA, safe=False)

        # unique file name
        fn = os.path.join(tmp_dir, f"{tf}__{motif_name}.parquet")
        pq.write_table(table, fn, compression="snappy")

    return True


def share_numpy_array(arr):
    shm = mp.Array('b', arr.tobytes(), lock=False)
    shape = arr.shape
    dtype = arr.dtype
    return shm, shape, dtype

def _init_worker(tf_df, bg_freq,
                 plus_shm, plus_shape, plus_dtype,
                 minus_shm, minus_shape, minus_dtype):
    """Reconstruct shared arrays inside each worker."""
    global _global_tf_df, _global_bg_freq, _global_plus, _global_minus
    _global_tf_df   = tf_df
    _global_bg_freq = bg_freq
    _global_plus  = np.frombuffer(plus_shm, dtype=plus_dtype).reshape(plus_shape)
    _global_minus = np.frombuffer(minus_shm, dtype=minus_dtype).reshape(minus_shape)

@njit(parallel=True)
def score_all_peaks(seqs_plus, seqs_minus, pwm_values):
    
    n_peaks, L = seqs_plus.shape
    wsize = pwm_values.shape[0]
    if wsize > L:
        return np.zeros(n_peaks, dtype=np.float64)
    
    out = np.zeros(n_peaks, dtype=np.float64)

    for i in prange(n_peaks):
        total = 0.0
        for strand in (seqs_plus[i], seqs_minus[i]):
            # slide the window
            for j in range(L - wsize + 1):
                s = 0.0
                for k in range(wsize):
                    idx = strand[j + k]
                    # skip ambiguous bases (idx < 0 or >=4)
                    if 0 <= idx < 4:
                        s += pwm_values[k, idx]
                total += s
        out[i] = total
    return out

def get_background_freq(species):
    if species == "human" or species == "hg38" or species == "mmusculus":
        background_freq = pd.Series({
            "A": 0.29182,
            "C": 0.20818,
            "G": 0.20818,
            "T": 0.29182
        })
    
    elif species == "mouse" or species == "mm10" or species == "hsapiens":
        background_freq = pd.Series({
        "A": 0.2917,
        "C": 0.2083,
        "G": 0.2083,
        "T": 0.2917
    })
        
    else:
        raise Exception(f"Species {species} is not 'human', 'mouse', 'hg38', or 'mm10'")

    return background_freq
    
def is_valid_parquet(file_path):
    try:
        ParquetFile(file_path)
        return True
    except (ArrowInvalid, OSError):
        return False

def get_valid_parquet_files(directory: str) -> list[str]:
    valid_files = []
    for f in os.listdir(directory):
        if f.endswith(".parquet"):
            full_path = os.path.join(directory, f)
            try:
                _ = pq.ParquetFile(full_path).metadata  # trigger metadata read
                valid_files.append(full_path)
            except Exception as e:
                logging.debug(f"Skipping corrupt Parquet file: {f}\n  Reason: {e}")
    return valid_files

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
    logging.debug("Building first-order background distribution")
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
    logging.debug(f"Background distribution: {pi}")
    return pi.tolist()

def extract_peak_seqs(fasta, peaks_bed):
    logging.debug(f"Extracting sequences from {peaks_bed}")
    valid_chroms = set(fasta.keys())
    bt = pybedtools.BedTool(peaks_bed)
    seqs, ids = [], []

    for iv in bt:
        if iv.chrom not in valid_chroms:
            continue
        try:
            seq = fasta[iv.chrom][iv.start:iv.end].upper()
            if len(seq) == 0:
                continue
            pid = iv.name if iv.name not in (None, "", ".") else f"{iv.chrom}:{iv.start}-{iv.end}"
            ids.append(pid)
            seqs.append(seq)
        except pyfaidx.FetchError:
            logging.warning(f"Skipping invalid peak {iv.chrom}:{iv.start}-{iv.end}")
            continue

    assert len(seqs) == len(ids), f"Mismatch: {len(seqs)} seqs vs {len(ids)} IDs"
    logging.debug(f"Extracted {len(seqs)} peak sequences")
    return ids, seqs

def to_log_odds(pfm, bg, pseudocount=1e-4):
    pfm = np.asarray(pfm, dtype=float)
    pfm = np.clip(pfm, pseudocount, 1.0)
    bg = np.array(bg, dtype=float)
    bg = np.clip(bg, pseudocount, 1.0)
    pwm = np.log2(pfm / bg[:, None])
    return pwm.tolist()


def _orient_pwm_position_major(log_odds_matrix: Any) -> np.ndarray:
    """
    Ensure PWM is position-major with nucleotide scores in the last dimension.
    Accepts either shape (4, L) or (L, 4) and returns an array of shape (L, 4).
    """
    arr = np.asarray(log_odds_matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"PWM must be 2D, got {arr.shape}")
    if arr.shape[1] == 4:
        return arr
    if arr.shape[0] == 4:
        return arr.T
    raise ValueError(f"PWM must include exactly four nucleotide columns, got {arr.shape}")


def load_motifs(motif_paths, pseudocount=0.001, bg=None):
    """
    Load JASPAR-style tabular PFMs (with optional header) or MEME-format .txt.
    Converts to log-odds matrices for MOODS scanning.
    """

    if bg is None:
        bg = MOODS.tools.flat_bg(4)

    score_mats, names = [], []
    malformed = 0

    for path in motif_paths:
        base = os.path.basename(path)
        name = os.path.splitext(base)[0]

        # --- JASPAR-like PFMs (either .pfm or .txt with numeric columns) ---
        if path.endswith(".pfm") or path.endswith(".txt"):
            try:
                # Read file, skip header line if it contains letters
                with open(path) as f:
                    lines = [l.strip() for l in f if l.strip()]
                if re.match(r"^[A-Za-z]", lines[0]):
                    lines = lines[1:]  # skip "Pos A C G T"

                # Parse numeric columns (skip row index if present)
                rows = []
                for line in lines:
                    parts = re.split(r"[\t\s]+", line)
                    nums = [float(x) for x in parts[-4:]]  # last 4 columns = A,C,G,T
                    rows.append(nums)
                pfm = np.array(rows, dtype=float)  # shape (L, 4)

                if pfm.shape[1] != 4:
                    raise ValueError("PFM must have 4 rows for A,C,G,T")

                log_odds = to_log_odds(pfm.T.tolist(), bg, pseudocount)
                score_mats.append(_orient_pwm_position_major(log_odds))
                names.append(name.capitalize())
            except Exception as e:
                logging.debug(f"Skipping malformed PFM: {path}\n  Reason: {e}")
                malformed += 1
            continue

        # --- MEME-format fallback ---
        with open(path, "r") as f:
            content = f.read()
        motif_blocks = re.split(r"(?=MOTIF\s+)", content)
        for block in motif_blocks:
            if "letter-probability matrix" not in block:
                continue
            match = re.search(r"MOTIF\s+([^\s\n]+)", block)
            motif_name = match.group(1) if match else name
            lines = re.findall(r"([0-9.eE+\-]+\s+[0-9.eE+\-]+\s+[0-9.eE+\-]+\s+[0-9.eE+\-]+)", block)
            if not lines:
                continue
            pfm = np.array([[float(x) for x in line.split()] for line in lines], dtype=float).T
            log_odds = MOODS.tools.log_odds(pfm.tolist(), bg, pseudocount)
            score_mats.append(_orient_pwm_position_major(log_odds))
            names.append(motif_name.capitalize())

    logging.info(f"Loaded {len(score_mats)} valid motifs, {malformed} malformed")
    return score_mats, names


def associate_tf_with_motif_pwm(tf_info_file, meme_dir, fasta_path, peaks_bed, tf_name_list, num_cpu, output_dir) -> str:
    logging.info("\nPreparing for parallel motif scoring...")
    
    fasta = Fasta(fasta_path, as_raw=True, sequence_always_upper=True)

    motif_paths = glob.glob(os.path.join(meme_dir, "**/*.pfm"), recursive=True) \
            + glob.glob(os.path.join(meme_dir, "**/*.txt"), recursive=True)
    
    # Background nucleotide frequencies
    background_freq = build_first_order_bg(fasta, peaks_bed)
    
    mats, names = load_motifs(motif_paths, pseudocount=0.0001, bg=background_freq)
    names = [n.upper() for n in names]
    
    logging.debug(f"Loaded {len(mats)} PFMs from {len(meme_dir)} motif files")
    
    # Extract the DNA sequences from the genome fasta for each peak
    peak_ids, seqs = extract_peak_seqs(fasta, peaks_bed)

    logging.debug(f"Number of peaks: {len(peak_ids)}")

    # Encode sequences into numeric format
    lookup = np.full(256, -1, dtype=np.int8)
    lookup[ord('A')] = 0
    lookup[ord('C')] = 1
    lookup[ord('G')] = 2
    lookup[ord('T')] = 3
    lookup[ord('N')] = 4

    encoded_plus = [lookup[np.frombuffer(seq.encode('ascii'), dtype=np.uint8)] for seq in seqs]
    encoded_minus = [lookup[np.frombuffer(seq[::-1].translate(str.maketrans('ACGT', 'TGCA')).encode('ascii'), dtype=np.uint8)] for seq in seqs]

    max_len = max(len(x) for x in encoded_plus)
    logging.debug(f"\tMaximum peak length: {max_len} bp")

    def pad_sequences(seqs, fixed_len, pad_val=-1):
        padded = np.full((len(seqs), fixed_len), pad_val, dtype=np.int8)
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = seq[:fixed_len]
        return padded

    seqs_plus = pad_sequences(encoded_plus, max_len)
    seqs_minus = pad_sequences(encoded_minus, max_len)

    # Directory for cached output
    tmp_dir = os.path.join(output_dir, "tmp", "sliding_window_tf_scores")
    os.makedirs(tmp_dir, exist_ok=True)
    
    tf_info_df = pd.read_csv(tf_info_file, sep="\t")
    tf_info_df["Motif_ID"] = tf_info_df["Motif_ID"].astype(str).str.upper()
    tf_info_df["TF_Name"] = tf_info_df["TF_Name"].astype(str).str.upper()
    logging.debug(f"\nLoaded TF information from {tf_info_file}")
    logging.debug(f"TFs in tf_info_file: {tf_info_df['TF_Name'].nunique()}")
    logging.debug("TF_Name examples: {tf_info_df['TF_Name'].head()}")
    logging.debug(f"Motif_ID examples: {tf_info_df['Motif_ID'].head()}")
    logging.debug(f"Motif names from JASPAR: {names[:5]}")
    
    tf_df = tf_info_df[tf_info_df["TF_Name"].isin(tf_name_list)]
    
    def is_valid_parquet(path):
        return os.path.isfile(path) and os.path.getsize(path) > 0

    # Filter motif files for motifs where NOT all associated TF files are cached
    filtered_indices = []
    for i, name in enumerate(names):
        tf_list = tf_df.loc[tf_df["Motif_ID"] == name, "TF_Name"].tolist()
        missing = [tf for tf in tf_list if not is_valid_parquet(os.path.join(tmp_dir, f"{tf}.parquet"))]
        if missing:
            filtered_indices.append(i)
            logging.debug(f"Motif {name}: {len(missing)} TFs missing parquet files → rescoring")

    # If no motifs selected for rescoring but no valid cache exists, force rescoring
    valid_parquet_files = get_valid_parquet_files(tmp_dir)

    if len(filtered_indices) == 0 and len(valid_parquet_files) == 0:
        logging.warning("No cached TF motif parquet files found. Forcing rescoring of all motifs.")
        filtered_indices = list(range(len(names)))

    logging.info(f"Filtered to {len(filtered_indices)} motifs needing scoring "
             f"out of {len(names)} total motifs.")

    if len(filtered_indices) > 0:
        logging.debug(f"\t- Number of motif files found: {len(filtered_indices)} for {len(tf_name_list)} TFs")

        logging.info(f"\nCalculating sliding window motif scores for each ATAC-seq peak")
        logging.debug(f"\tUsing {num_cpu} processors")
        logging.debug(f"\tSize of calculation: {len(filtered_indices)} motifs × {len(peak_ids)} peaks")
        
        plus_shm, plus_shape, plus_dtype = share_numpy_array(seqs_plus)
        minus_shm, minus_shape, minus_dtype = share_numpy_array(seqs_minus) 

        logging.debug(f"\nWriting output to {tmp_dir}")
        with ProcessPoolExecutor(
        max_workers=num_cpu,
        initializer=_init_worker,
        initargs=(tf_df, background_freq,
                plus_shm, plus_shape, plus_dtype,
                minus_shm, minus_shape, minus_dtype)
        ) as executor:
            futures = {
                executor.submit(process_motif_file_and_save, i, tf_df, background_freq, mats, names, tmp_dir, peak_ids): names[i]
                for i in filtered_indices
            }

            min_update = max(1, int(0.02 * len(futures)))
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring motifs", miniters=min_update):
                _ = future.result()
        logging.info("Finished scoring all motifs. Reading TF motif parquet files...")
    else:
        logging.info("\nAll TFs have pre-existing parquet files in the tmp directory, reading cached parquet files...")
    
    os.makedirs(output_dir, exist_ok=True)
    valid_parquet_files = get_valid_parquet_files(tmp_dir)
    if not valid_parquet_files:
        raise RuntimeError("No valid TF motif parquet files found after filtering.")

    # Write a single final parquet from the temporary per-TF files
    out_parquet = os.path.join(output_dir, "sliding_window.parquet")
    coalesce_sliding_window_scores_pyarrow(
        input_dir=os.path.join(tmp_dir),
        out_parquet=out_parquet,
        pattern="*.parquet"
    )
    logging.info(f"Wrote coalesced sliding window scores → {out_parquet}")
    return out_parquet

def run_sliding_window_scan(
    tf_name_list: list[str],
    tf_info_file: str,
    motif_dir: str,
    genome_fasta: str,
    peak_bed_file: str,
    output_file: Union[str, Path],
    num_cpu: int,
):
    output_file = Path(output_file)
    output_dir = output_file.parent
    os.makedirs(output_dir, exist_ok=True)
                
    # Associate the TFs from TF_Information_all_motifs.txt to the motif with the matching motifID
    final_parquet = associate_tf_with_motif_pwm(
        tf_info_file, motif_dir, genome_fasta, peak_bed_file,
        tf_name_list, num_cpu=num_cpu, output_dir=output_dir
    )
    
    if Path(final_parquet) != output_file:
        tmp = output_file.with_suffix(".tmp.parquet")
        os.replace(final_parquet, tmp)
        os.replace(tmp, output_file)

    logging.info(f"Wrote final TF–peak sliding window scores to {output_file}")

