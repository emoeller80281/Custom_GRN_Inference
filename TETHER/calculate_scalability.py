"""
Collect runtime and memory for TETHER and the benchmark GRN methods.

Two different kinds of measurement are combined here, and they are NOT directly
comparable without saying so:

  * The external tools are wrapped in GNU `time -v`, so their numbers cover the whole
    tool run on CPU -- data loading, motif scanning, model fitting, the lot.
  * TETHER is instrumented in-process by generate_all_predictions.py, which writes a
    per-phase TSV (resource_usage_{cell_type}_{sample_name}.tsv). Those numbers are
    inference against an already-trained model on a GPU, and exclude training entirely.

Every row therefore carries `scope`, `hardware`, `cpu_percent` and `max_rss_kb` so the
asymmetry stays visible downstream instead of being silently averaged away.

Reading the TSVs rather than the SLURM logs also splits the self-trained and
cross-trained models apart, and keeps one-off data preparation (peak one-hot encoding,
edge-bag construction) out of the per-model timings. That preparation is substantial --
37% of the job's wall clock on E7.5_rep1 -- but it belongs to neither model, so it is
simply not counted here rather than being charged to one of them.
"""

import re
from pathlib import Path

import pandas as pd

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

PROJECT_DIR = Path(
    "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/"
    "2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
)

# Per-phase resource TSVs written by generate_all_predictions.py
tether_resource_dir = PROJECT_DIR / "testing_results" / "full_test_grns"

other_method_times_dir = Path(
    "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/"
    "2024.GRN_BENCHMARKING.MOELLER/"
    "multiGRNtools/LOGS/run_tools/"
    "run_tools_3747801/resource_raw"
)

output_dir = PROJECT_DIR / "testing_results" / "scalability"

OWN_MODEL_METHOD = "TF-TG Model (own test set)"
CROSS_MODEL_METHOD = "TF-TG Model (cross-trained)"

org_dict = {
    "E7.5_rep1": ("mouse", "mESC"),
    "E8.5_rep1": ("mouse", "mESC"),
    "buffer_1": ("human", "Macrophage"),
    "buffer_2": ("human", "Macrophage"),
    "hepatocytes_1": ("mouse", "mouse_hepatocytes"),
    "hepatocytes_3": ("mouse", "mouse_hepatocytes"),
    "sample_1": ("human", "K562"),
    "WT_D13_rep1": ("human", "iPSC"),
}

sample_order = [
    "E7.5_rep1",
    "E8.5_rep1",
    "buffer_1",
    "buffer_2",
    "sample_1",
    "hepatocytes_1",
    "hepatocytes_3",
]

sample_rename_map = {
    "E7.5_rep1": "mESC-1",
    "E8.5_rep1": "mESC-2",
    "buffer_1": "Macrophage-1",
    "buffer_2": "Macrophage-2",
    "sample_1": "K562",
    "hepatocytes_1": "Hepatocytes-1",
    # Matches sample_to_title_map in generate_all_predictions.py. The previous
    # "Hepatocytes-2" here disagreed with every other figure in the project.
    "hepatocytes_3": "Hepatocytes-3",
    OWN_MODEL_METHOD: "MTGRN-STM",
    CROSS_MODEL_METHOD: "MTGRN-CTM",
}

sample_color_map = {
    "E7.5_rep1": "#4195df",
    "E8.5_rep1": "#86C7E7",
    "buffer_1": "#EF767A",
    "buffer_2": "#F9C60D",
    "sample_1": "#EF9CFA",
    "hepatocytes_1": "#82EC32",
    "hepatocytes_3": "#F98637",
}

method_color_dict = {
    OWN_MODEL_METHOD: "#4195df",
    CROSS_MODEL_METHOD: "#86C7E7",
    "LINGER": "#EF767A",
    "CellOracle": "#F9C60D",
    "Pando": "#EF9CFA",
    "SCENIC+": "#82EC32",
    "FigR": "#FDA7BB",
}

# Raw method token in a .time filename -> display name
method_name_aliases = {
    "SCENIC_PLUS": "SCENIC+",
    "SCENIC": "SCENIC+",
}


# ----------------------------------------------------------------------------------
# GNU time -v parsing (external methods)
# ----------------------------------------------------------------------------------

def elapsed_time_to_seconds(elapsed_time: str) -> float:
    """
    Convert a GNU time elapsed-time string to seconds.

    Supported formats: m:ss, m:ss.ss, h:mm:ss, h:mm:ss.ss
    """
    parts = elapsed_time.strip().split(":")

    if len(parts) == 3:
        hours, minutes, seconds = (float(part) for part in parts)
    elif len(parts) == 2:
        hours = 0.0
        minutes, seconds = (float(part) for part in parts)
    else:
        raise ValueError(f"Unrecognized elapsed-time format: '{elapsed_time}'")

    return hours * 3600 + minutes * 60 + seconds


def parse_gnu_time_file(time_file: Path) -> dict:
    """
    Parse a GNU `time -v` report into a flat dict.

    Every "Key: value" line is captured, then the fields used downstream are pulled
    out and typed. Returns {} if the elapsed-time field is absent, which is what a
    run that was killed before `time` could print its summary looks like.
    """
    fields = {}

    for line in time_file.read_text(encoding="utf-8", errors="replace").splitlines():
        # Split on ": ", not ":" -- several GNU time keys contain colons themselves,
        # e.g. "Elapsed (wall clock) time (h:mm:ss or m:ss)". Colons inside a key are
        # always followed by a letter, so colon-space only ever separates key from value.
        key, separator, value = line.strip().partition(": ")
        if not separator:
            continue
        fields[key.strip()] = value.strip()

    elapsed_key = "Elapsed (wall clock) time (h:mm:ss or m:ss)"
    if elapsed_key not in fields:
        return {}

    def _as_float(key, default=None):
        try:
            return float(fields[key].rstrip("%"))
        except (KeyError, ValueError):
            return default

    return {
        "wall_seconds": elapsed_time_to_seconds(fields[elapsed_key]),
        "user_seconds": _as_float("User time (seconds)"),
        "system_seconds": _as_float("System time (seconds)"),
        "cpu_percent": _as_float("Percent of CPU this job got"),
        "max_rss_kb": _as_float("Maximum resident set size (kbytes)"),
        "major_page_faults": _as_float("Major (requiring I/O) page faults"),
        "exit_status": _as_float("Exit status"),
        "command": fields.get("Command being timed", "").strip('"'),
    }


def load_other_method_records() -> list[dict]:
    """One record per (sample, external method) from the .time files."""
    records = []

    if not other_method_times_dir.is_dir():
        logging.warning(f"Other-method timing directory not found: {other_method_times_dir}")
        return records

    for sample_name in sample_order:
        if sample_name not in org_dict:
            logging.warning(f"Sample '{sample_name}' is not in org_dict; skipping.")
            continue

        cell_type = org_dict[sample_name][1]
        file_pattern = re.compile(
            rf"^(?P<method>.+?)_{re.escape(cell_type)}_{re.escape(sample_name)}"
            rf"_task(?P<task>\d+)\.time$"
        )

        sample_files = sorted(other_method_times_dir.glob(f"*_{cell_type}_{sample_name}_task*.time"))

        if not sample_files:
            logging.warning(f"No external-method timing files for sample '{sample_name}'.")
            continue

        seen_methods = {}

        for time_file in sample_files:
            match = file_pattern.match(time_file.name)
            if match is None:
                logging.debug(f"Unrecognized timing filename: {time_file.name}")
                continue

            # Split on the full cell_type/sample suffix rather than the first
            # underscore, so multi-token names like SCENIC_PLUS survive intact.
            raw_method = match.group("method")
            method_name = method_name_aliases.get(raw_method, raw_method)

            if method_name not in method_color_dict:
                logging.warning(
                    f"Method '{method_name}' is not in method_color_dict; "
                    f"skipping '{time_file.name}'."
                )
                continue

            stats = parse_gnu_time_file(time_file)

            if not stats:
                logging.warning(
                    f"No elapsed time in '{time_file.name}' -- the run was probably "
                    f"killed before GNU time printed its summary."
                )
                records.append({
                    "sample_name": sample_name,
                    "cell_type": cell_type,
                    "method_name": method_name,
                    "status": "incomplete",
                    "scope": "full pipeline",
                    "hardware": "cpu",
                    "source_file": time_file.name,
                })
                continue

            if method_name in seen_methods:
                logging.warning(
                    f"Duplicate timing for sample '{sample_name}', method "
                    f"'{method_name}'; keeping '{seen_methods[method_name]}' and "
                    f"ignoring '{time_file.name}'."
                )
                continue

            seen_methods[method_name] = time_file.name

            records.append({
                "sample_name": sample_name,
                "cell_type": cell_type,
                "method_name": method_name,
                "status": "ok",
                # These tools run their entire pipeline; nothing is excluded.
                "scope": "full pipeline",
                "hardware": "cpu",
                "source_file": time_file.name,
                **stats,
            })

    return records


# ----------------------------------------------------------------------------------
# TETHER per-phase resource TSVs
# ----------------------------------------------------------------------------------

def load_tether_records() -> tuple[list[dict], pd.DataFrame]:
    """
    One record per (sample, model_type) from the per-phase resource TSVs.

    A phase row is either a model load or an inference pass, tagged own/cross. Wall
    time for a model is load + inference. Data preparation shared by both models is
    not recorded in the TSV and is therefore not counted at all.
    """
    records = []
    phase_frames = []

    tsv_files = sorted(tether_resource_dir.glob("resource_usage_*.tsv"))

    if not tsv_files:
        logging.warning(
            f"No resource_usage_*.tsv found in {tether_resource_dir}. TETHER rows will "
            f"be missing -- generate_all_predictions.py writes these, so re-run it."
        )
        return records, pd.DataFrame()

    for tsv_file in tsv_files:
        phases = pd.read_csv(tsv_file, sep="\t")
        phases["source_file"] = tsv_file.name
        phase_frames.append(phases)

        for model_type, model_phases in phases.groupby("model_type"):
            sample_name = model_phases["sample_name"].iloc[0]
            cell_type = model_phases["cell_type"].iloc[0]

            method_name = OWN_MODEL_METHOD if model_type == "own" else CROSS_MODEL_METHOD

            inference = model_phases[model_phases["phase"].str.startswith("inference")]
            model_load = model_phases[model_phases["phase"].str.startswith("model_load")]

            if inference.empty:
                logging.warning(f"No inference phase for {sample_name}/{model_type} in {tsv_file.name}")
                continue

            inference_row = inference.iloc[0]

            records.append({
                "sample_name": sample_name,
                "cell_type": cell_type,
                "method_name": method_name,
                "status": "ok",
                # Inference only. Training the model is not counted here, unlike the
                # external tools whose numbers cover their whole pipeline.
                "scope": "inference only",
                "hardware": inference_row.get("device", "gpu"),
                "source_file": tsv_file.name,
                "wall_seconds": float(model_phases["wall_seconds"].sum()),
                "inference_seconds": float(inference_row["wall_seconds"]),
                "model_load_seconds": float(model_load["wall_seconds"].sum()) if not model_load.empty else 0.0,
                "user_seconds": float(model_phases["tree_user_seconds"].sum()),
                "system_seconds": float(model_phases["tree_system_seconds"].sum()),
                "cpu_percent": float(inference_row.get("tree_cpu_percent", float("nan"))),
                "max_rss_kb": float(model_phases["peak_rss_tree_kb"].max()),
                "gpu_peak_allocated_mb": float(model_phases["gpu_peak_allocated_mb"].max()),
                "n_edges": int(inference_row["n_edges"]),
                "edges_per_second": float(inference_row.get("edges_per_second", float("nan"))),
            })

    return records, pd.concat(phase_frames, ignore_index=True)


# ----------------------------------------------------------------------------------
# Assemble
# ----------------------------------------------------------------------------------

other_records = load_other_method_records()
tether_records, tether_phase_df = load_tether_records()

scalability_df = pd.DataFrame(other_records + tether_records)

if scalability_df.empty:
    raise SystemExit("No timing records were loaded; nothing to summarise.")

# Flag samples that were expected but produced nothing, so a gap in a figure reads as
# "this run did not finish" rather than "this method was not measured".
for sample_name in sample_order:
    present = scalability_df.loc[scalability_df["sample_name"] == sample_name, "method_name"]
    for method_name in method_color_dict:
        if method_name not in set(present):
            scalability_df.loc[len(scalability_df)] = {
                "sample_name": sample_name,
                "cell_type": org_dict.get(sample_name, (None, None))[1],
                "method_name": method_name,
                "status": "missing",
                "scope": "inference only" if method_name in (OWN_MODEL_METHOD, CROSS_MODEL_METHOD) else "full pipeline",
                "hardware": "gpu" if method_name in (OWN_MODEL_METHOD, CROSS_MODEL_METHOD) else "cpu",
            }

scalability_df["sample_label"] = scalability_df["sample_name"].map(sample_rename_map)
scalability_df["method_label"] = scalability_df["method_name"].map(
    lambda name: sample_rename_map.get(name, name)
)

scalability_df["wall_minutes"] = scalability_df["wall_seconds"] / 60
scalability_df["wall_hours"] = scalability_df["wall_seconds"] / 3600
scalability_df["max_rss_gb"] = scalability_df["max_rss_kb"] / 1024**2

scalability_df["sample_name"] = pd.Categorical(
    scalability_df["sample_name"], categories=sample_order, ordered=True
)
scalability_df = scalability_df.sort_values(["sample_name", "method_name"]).reset_index(drop=True)

output_dir.mkdir(parents=True, exist_ok=True)
scalability_df.to_csv(output_dir / "scalability_summary.csv", index=False)

if not tether_phase_df.empty:
    tether_phase_df.to_csv(output_dir / "tether_resource_phases.csv", index=False)


# ----------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------

logging.info("\n=== Runtime (hours), by sample and method ===")
runtime_table = scalability_df.pivot_table(
    index="method_name", columns="sample_name", values="wall_hours", aggfunc="first", observed=False
)
logging.info(runtime_table.round(2).to_string())

logging.info("\n=== Peak memory (GB) ===")
memory_table = scalability_df.pivot_table(
    index="method_name", columns="sample_name", values="max_rss_gb", aggfunc="first", observed=False
)
logging.info(memory_table.round(1).to_string())

logging.info("\n=== Measurement scope and hardware (these are NOT interchangeable) ===")
scope_table = (
    scalability_df[scalability_df["status"] == "ok"]
    .groupby(["method_name", "scope", "hardware"], observed=True)
    .agg(n_runs=("wall_seconds", "size"), median_cpu_percent=("cpu_percent", "median"))
    .reset_index()
)
logging.info(scope_table.to_string(index=False))

missing = scalability_df[scalability_df["status"] != "ok"]
if not missing.empty:
    logging.info("\n=== Missing or incomplete runs ===")
    for row in missing.itertuples(index=False):
        logging.info(f"  {row.sample_name:>14s}  {row.method_name:<28s} {row.status}")

logging.info(f"\nWrote {output_dir / 'scalability_summary.csv'}")
