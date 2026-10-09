#!/usr/bin/env python3
"""Generate GRNs for the stability cell subsamples with the tissue-holdout models.

For each tissue in subsample_cell_dict.json (made in scripts/stability.ipynb), the
"no_<tissue>" model predicts every edge (train, val and test chromosomes) of the
sample's subsampled cell type. The edges and binding scores are the full-dataset
ones from the run's prepared caches. Only the cell pool of that cell type changes:
each subsample replaces it with the subsample's barcodes.

Writes <output_dir>/<Tissue>/subsample_<i>_grn.csv with the columns
Tissue, SampleID, CellType, Source, Target, Score, Label.

Usage (on a GPU node):
  python generate_stability_subsample_grns.py --tissue Liver
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_grns_from_model import (  # noqa: E402
    CHROMOSOME_SPLITS,
    PROJECT_DIR,
    BindingScorer,
    LitTFTGRegulationModel,
    load_cell_pools,
    load_multiome_data,
    load_split_edges,
    log_auroc,
    predict,
    to_grn,
)

SUBSAMPLE_FILE = Path(
    "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/subsample_cell_dict.json"
)
OUTPUT_DIR = PROJECT_DIR / "testing_results" / "stability_results" / "celltype_holdout_models"

# Tissue in subsample_cell_dict.json -> run directory of the model that held it out
MODEL_RUNS = {
    "Liver": "celltype_joint_3899206_20261002_125300_917425",   # no_liver
    "Embryo": "celltype_joint_3899207_20261002_125401_578199",  # no_mesc
    "Brain": "celltype_joint_3899208_20261002_125401_578203",   # no_brain
    "Skin": "celltype_joint_3899209_20261002_125401_578193",    # no_skin
    "Kidney": "celltype_joint_3899210_20261002_125501_478329",  # no_kidney
    "HSC": "celltype_joint_3899211_20261002_143015_907548",     # no_hsc
}

# The caches were built before data/processed/10x_E18_mouse_brain/brain was
# renamed to E18_brain, so their sample name differs from the data directory.
CACHE_SAMPLE_NAMES = {("10x_E18_mouse_brain", "E18_brain"): "brain"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--tissue", required=True, choices=sorted(MODEL_RUNS))
    parser.add_argument("--subsample_file", type=Path, default=SUBSAMPLE_FILE)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint", default="last.ckpt",
        help="Checkpoint file name in <run_dir>/checkpoints (default: last.ckpt)",
    )
    parser.add_argument("--batch_size", type=int, help="Default: the run's batch size")
    parser.add_argument("--num_workers", type=int, help="Default: the run's num_workers")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing GRNs")
    return parser.parse_args()


def load_source(run_config, dataset, data_sample, cache_sample):
    """Like generate_grns_from_model.load_source, but also returns the cell barcodes."""
    source_dir = (
        Path(run_config["cache_dir"]) / "prepared_sources" / f"{dataset}__{cache_sample}"
    )
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Prepared source cache not found: {source_dir}")

    logging.info(f"Loading {dataset}:{data_sample} with the caches in {source_dir.resolve()}")
    rna, atac = load_multiome_data(run_config, dataset, data_sample)
    atac_peak_tensor = torch.load(
        source_dir / "prepared" / cache_sample / "atac_peak_tensor.pt",
        map_location="cpu", weights_only=True,
    )
    if atac.n_vars != atac_peak_tensor.shape[0]:
        raise ValueError(
            f"{dataset}:{data_sample}: {atac.n_vars:,} ATAC peaks but "
            f"{atac_peak_tensor.shape[0]:,} cached peak sequences"
        )
    return {
        "tissue": dataset,
        "sample_name": cache_sample,
        "source_dir": source_dir,
        "obs_names": rna.obs_names,
        "rna_names": rna.var_names.to_numpy(),
        "rna_mat": rna.X,
        "atac_mat": atac.X,
        "atac_peak_tensor": atac_peak_tensor,
        "cell_pools": load_cell_pools(source_dir),
    }


def subsample_pool(source, barcodes, full_pool, name):
    """Return the RNA/ATAC row positions of the subsample barcodes."""
    positions = source["obs_names"].get_indexer(pd.Index(barcodes))
    missing = int((positions < 0).sum())
    if missing:
        raise ValueError(f"{name}: {missing:,} of {len(barcodes):,} barcodes are not in the data")
    positions = np.sort(positions).astype(np.int64)
    in_full = np.isin(positions, full_pool).mean()
    logging.info(
        f"{name}: {len(positions):,} cells "
        f"({len(positions) / len(full_pool):.1%} of the {len(full_pool):,}-cell full pool, "
        f"{in_full:.1%} of them inside it)"
    )
    return positions


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    subsample_info = json.loads(args.subsample_file.read_text())[args.tissue]
    dataset = subsample_info["dataset"]
    data_sample = subsample_info["sample_name"]
    cell_type = subsample_info["cell_type"]
    cache_sample = CACHE_SAMPLE_NAMES.get((dataset, data_sample), data_sample)
    subsamples = subsample_info["subsamples"]

    out_dir = args.output_dir / args.tissue
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = {name: out_dir / f"{name}_grn.csv" for name in subsamples}
    pending = [name for name, path in out_paths.items() if args.overwrite or not path.is_file()]
    if not pending:
        logging.info(f"All {len(out_paths)} subsample GRNs exist in {out_dir}. Use --overwrite.")
        return

    run_dir = PROJECT_DIR / "checkpoints" / "celltype_tf_tg" / MODEL_RUNS[args.tissue]
    run_config = json.loads((run_dir / "run_config.json").read_text())
    holdouts = run_config.get("holdout_sample", [])
    if f"{dataset}:{cache_sample}" not in holdouts:
        raise ValueError(f"{run_dir.name} did not hold out {dataset}:{cache_sample}: {holdouts}")
    logging.info(f"{args.tissue}: {dataset}:{data_sample} {cell_type}, model {run_dir.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA is not available; predictions run on the CPU")

    checkpoint = run_dir / "checkpoints" / args.checkpoint
    logging.info(f"Loading model from {checkpoint}")
    model = LitTFTGRegulationModel.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model.requires_grad_(False).eval().to(device)
    binding_scorer = BindingScorer(run_config, device)

    source = load_source(run_config, dataset, data_sample, cache_sample)
    pool_key = (cache_sample, cell_type)
    if pool_key not in source["cell_pools"]:
        raise KeyError(f"No cell pool {pool_key}; pools: {sorted(source['cell_pools'])}")
    full_pool = source["cell_pools"][pool_key]

    # Edges and binding scores are the same for every subsample: load them once.
    splits = {}
    for split_name in CHROMOSOME_SPLITS:
        edges, scores = load_split_edges(
            source, split_name, run_config, binding_scorer, run_dir, celltypes={cell_type},
        )
        if edges is not None:
            splits[split_name] = (edges, scores)
    if not splits:
        raise ValueError(f"{dataset}:{cache_sample} has no {cell_type} edges")
    logging.info(
        f"{cell_type} edges: "
        + ", ".join(f"{name}={len(edges):,}" for name, (edges, _) in splits.items())
    )

    for name in pending:
        source["cell_pools"][pool_key] = subsample_pool(
            source, subsamples[name], full_pool, f"{args.tissue} {name}",
        )
        split_grns = []
        for split_name, (edges, scores) in splits.items():
            probabilities, inputs = predict(
                model, edges, scores, source, run_config, args, device,
                desc=f"{args.tissue} {name} {split_name}",
            )
            split_grns.append(to_grn(inputs, probabilities, dataset))
        grn = pd.concat(split_grns, ignore_index=True)
        log_auroc(f"{args.tissue} {name}", grn)
        grn.to_csv(out_paths[name], index=False)
        logging.info(f"Wrote {out_paths[name]}")
        gc.collect()


if __name__ == "__main__":
    main()
