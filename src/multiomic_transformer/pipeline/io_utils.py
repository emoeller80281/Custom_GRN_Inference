#!/usr/bin/env python3
"""
io_utils.py
------------
I/O and logging utilities for the MultiomicTransformer pipeline.

Includes:
  • Logging configuration (with SLURM and hostname awareness)
  • Atomic file operations (safe writes)
  • Stage checkpointing (.done markers)
  • Miscellaneous helpers for progress reporting
"""

import os
import sys
import json
import time
import shutil
import socket
import logging
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------
def ensure_dir(path: Path):
    """Create directory if it doesn’t exist."""
    path = Path(path)
    if not path.exists():
        logging.info(f"Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------
def write_parquet_safe(df: pd.DataFrame, filepath: Path, compression: str = "snappy"):
    """
    Safely write DataFrame to Parquet with atomic replacement.
    Uses a temporary file to avoid corruption on failed writes.
    """
    if df.empty:
        raise ValueError(f"Attempted to write empty DataFrame to {filepath}")
    
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    tmp_path = filepath.with_suffix(".tmp.parquet")

    try:
        logging.info(f"Writing {len(df):,} rows to {filepath.name}")
        table = pa.Table.from_pandas(df)
        pq.write_table(table, tmp_path, compression=compression)
        os.replace(tmp_path, filepath)
    except Exception as e:
        logging.error(f"Failed to write {filepath}: {e}")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------
def write_done_flag(path: Path):
    """
    Write a '.done' checkpoint flag next to the given path.
    If 'path' already ends with '.done', write directly to it.
    """
    if path.suffix == ".done":
        done_path = path
    else:
        done_path = path.with_suffix(path.suffix + ".done")

    with open(done_path, "w") as f:
        f.write("done")

    logging.info(f"[CHECKPOINT] Wrote done flag: {done_path}")


def checkpoint_exists(path: Path) -> bool:
    """
    Return True if the corresponding '.done' file exists and is non-empty.
    If 'path' already ends with '.done', check it directly.
    """
    if path.suffix == ".done":
        done_path = path
    else:
        done_path = path.with_suffix(path.suffix + ".done")

    return done_path.exists() and done_path.stat().st_size > 0



def clear_checkpoints(directory: Path, prefix: str = None):
    """
    Delete all `.done` checkpoint files in a directory.
    Optionally filter by prefix (e.g., 'rna', 'atac').
    """
    directory = Path(directory)
    pattern = "*.done" if prefix is None else f"{prefix}*.done"
    for flag_file in directory.glob(pattern):
        logging.info(f"Removing checkpoint: {flag_file}")
        flag_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------
def parquet_exists(path: Path) -> bool:
    """Check if a valid Parquet file exists (non-empty, readable)."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        pq.read_metadata(path)
        return True
    except Exception:
        return False


def read_parquet_safely(path: Path) -> pd.DataFrame:
    """Read a Parquet file safely and log its dimensions."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing expected Parquet file: {path}")
    df = pd.read_parquet(path)
    logging.info(f"Loaded {path.name}: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
    return df

# ============================================================
# Logging setup
# ============================================================

def setup_logging(args, log_dir=None):
    """
    Configure logging for the pipeline.

    Creates both console and file handlers, including timestamps and host info.
    """
    log_dir = Path(log_dir or Path(args.base_dir) / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"pipeline_{timestamp}.log"

    slurm_job_id = os.getenv("SLURM_JOB_ID", "N/A")
    host = socket.gethostname()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=" * 70)
    logging.info(f"Pipeline initialized at {timestamp}")
    logging.info(f"Host: {host}")
    logging.info(f"SLURM_JOB_ID: {slurm_job_id}")
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Sample: {args.sample}")
    logging.info(f"Organism: {args.organism}")
    logging.info(f"Mode: {args.mode}")
    logging.info("=" * 70)

    return log_path


# ============================================================
# Atomic file writing utilities
# ============================================================

def atomic_json_dump(obj, path, indent=2):
    """
    Safely write JSON to disk using a temporary file + atomic move.
    Prevents corruption if a write is interrupted (e.g. HPC preemption).
    """
    path = Path(path)
    tmp_path = path.with_suffix(".tmp")

    with tempfile.NamedTemporaryFile("w", dir=tmp_path.parent, delete=False) as tmp:
        json.dump(obj, tmp, indent=indent)
        tmp.flush()
        os.fsync(tmp.fileno())

    shutil.move(tmp.name, path)
    return str(path)


def unique_path(path, sep="_"):
    """
    Generate a unique file path by appending an integer suffix if needed.
    """
    path = Path(path)
    if not path.exists():
        return path

    stem, suffix = path.stem, path.suffix
    for i in range(1, 1000):
        candidate = path.with_name(f"{stem}{sep}{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique path for {path}")


# ============================================================
# Stage checkpointing utilities
# ============================================================

def checkpoint_path(stage_name, processed_dir):
    """Return the .done file path for a given stage."""
    return Path(processed_dir) / f"{stage_name}.done"


def mark_done(stage_name, processed_dir):
    """Mark a stage as completed."""
    done_path = checkpoint_path(stage_name, processed_dir)
    with open(done_path, "w") as f:
        f.write(datetime.now().isoformat())
    logging.info(f"[✓] Marked stage '{stage_name}' as completed.")
    return done_path


def is_done(stage_name, processed_dir):
    """Check if a stage has already been completed."""
    done_path = checkpoint_path(stage_name, processed_dir)
    return done_path.exists()


def skip_if_done(stage_name, processed_dir, force_recompute=False):
    """
    Skip a stage if already completed (unless --force_recompute is True).
    """
    if is_done(stage_name, processed_dir) and not force_recompute:
        logging.info(f"[↩] Skipping stage '{stage_name}' (already completed).")
        return True
    return False


# ============================================================
# Timing / progress utilities
# ============================================================

class StageTimer:
    """
    Context manager to measure and log execution time for pipeline stages.
    """

    def __init__(self, stage_name):
        self.stage_name = stage_name

    def __enter__(self):
        self.start_time = time.time()
        logging.info(f"--- Starting stage: {self.stage_name} ---")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration = time.time() - self.start_time
        logging.info(f"--- Finished stage: {self.stage_name} (took {duration:.2f} s) ---\n")


# ============================================================
# Intermediate file cleanup
# ============================================================

def cleanup_intermediate_files(stage_outputs, keep=True):
    """
    Optionally remove intermediate files after a successful run.

    Args:
        stage_outputs (list of Path): Files or directories to remove.
        keep (bool): If False, removes the files.
    """
    if keep:
        logging.info("Keeping intermediate files (default behavior).")
        return

    for path in stage_outputs:
        path = Path(path)
        if path.exists():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                logging.info(f"Removed intermediate output: {path}")
            except Exception as e:
                logging.warning(f"Failed to delete {path}: {e}")


# ============================================================
# Utility: summarize pipeline config for logs
# ============================================================

def summarize_config(args, paths):
    logging.info("Pipeline configuration summary:")
    for k, v in vars(args).items():
        logging.info(f"  - {k}: {v}")
    logging.info("\nResolved paths:")
    for k, v in paths.items():
        logging.info(f"  - {k}: {v}")
    logging.info("=" * 70)


# ============================================================
# Example usage (testing)
# ============================================================

if __name__ == "__main__":
    from .config import parse_args, get_paths
    args = parse_args()
    paths = get_paths(args)
    log_file = setup_logging(args, log_dir=paths["log_dir"])
    summarize_config(args, paths)

    # Simulate a pipeline stage
    with StageTimer("example_stage"):
        time.sleep(2.5)
        mark_done("example_stage", paths["processed_dir"])
