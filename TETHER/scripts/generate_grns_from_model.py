#!/usr/bin/env python3
"""Generate GRNs from a trained cell-type TF->TG model.

Writes the same files as celltype_model_evaluation.ipynb into the model run directory:
  test_set_grn.csv          test-chromosome edges of every training sample
  <sample>_cross_grn.csv    every edge (train, val and test chromosomes) of each
                            --holdout_sample of the run

Each GRN has the columns SampleID, CellType, Source, Target, Score, Label.

The edges and binding scores come from the run's prepared caches. This script only
reads those caches. It does not call prepare_data, because a manifest mismatch makes
prepare_data rebuild the cache, and the per-sample caches are shared between runs.
Binding scores that are not cached are computed into
<run_dir>/holdout_evaluation/<tissue>__<sample>/prepared.

Usage (on a GPU node):
  python generate_grns_from_model.py --run_dir celltype_joint_3899207_20261002_125401_578199
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import muon as mu
import mudata
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tf_to_tg_celltype_model import (  # noqa: E402
    PROJECT_DIR,
    TFTGEdgeBagDataset,
    binding_score_cache_path,
    load_or_precompute_binding_scores,
)
from build_tf_to_tg_celltype_data import BINDING_MANIFEST, load_binding_model  # noqa: E402

sys.path.insert(0, str(PROJECT_DIR))
from models.tf_to_tg_celltype import LitTFTGRegulationModel  # noqa: E402

mudata.set_options(pull_on_update=False)

GRN_COLUMNS = ["Tissue", "SampleID", "CellType", "Source", "Target", "Score", "Label"]
CHROMOSOME_SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--run_dir", required=True,
        help="Model run directory, or its name under checkpoints/celltype_tf_tg",
    )
    parser.add_argument(
        "--checkpoint", default="last.ckpt",
        help="Checkpoint file name in <run_dir>/checkpoints (default: last.ckpt)",
    )
    parser.add_argument(
        "--holdout_celltypes_only", action="store_true",
        help="Keep only the run's holdout cell types in the holdout sample GRNs",
    )
    parser.add_argument("--batch_size", type=int, help="Default: the run's batch size")
    parser.add_argument("--num_workers", type=int, help="Default: the run's num_workers")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing GRNs")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        run_dir = PROJECT_DIR / "checkpoints" / "celltype_tf_tg" / args.run_dir
    if not (run_dir / "run_config.json").is_file():
        parser.error(f"No run_config.json in {run_dir}")
    args.run_dir = run_dir
    return args


def source_specs(run_config):
    "Retrieve a list of (tissue, sample name) tuples from the run config"
    configured = run_config.get("dataset")
    if not configured:
        configured = [f"{run_config['tissue']}:{run_config['sample_name']}"]
    if isinstance(configured, str):
        configured = [configured]
    return [tuple(value.split(":", 1)) for value in configured]


def load_multiome_data(run_config, tissue, sample_name):
    """
    Recreate the RNA and ATAC matrix order used by the training script.
    """
    data_dir = Path(run_config["data_dir"])
    mdata = mu.read(
        data_dir / "processed" / tissue
        / sample_name / "multiome_processed.h5mu"
    )
    rna = mdata.mod["rna"]
    atac = mdata.mod["atac"]
    shared_cells = rna.obs_names[rna.obs_names.isin(atac.obs_names)]
    rna = rna[shared_cells].copy()
    atac = atac[shared_cells].copy()

    max_autosome = 19 if run_config["species"] == "mm10" else 22
    valid_chroms = {f"chr{i}" for i in range(1, max_autosome + 1)}
    keep_peaks = np.array([
        peak.split(":", 1)[0] in valid_chroms for peak in atac.var_names
    ])
    atac = atac[:, keep_peaks].copy()
    rna.var_names = rna.var_names.str.upper()

    # Same as the training script: if two genes differ only by case
    # (skin_late_anagen has "PISD" and "Pisd"), keep the one with the most counts.
    if not rna.var_names.is_unique:
        gene_totals = np.asarray(rna.X.sum(axis=0)).ravel()
        order = np.argsort(-gene_totals, kind="stable")
        keep_genes = np.zeros(rna.n_vars, dtype=bool)
        keep_genes[order[~rna.var_names[order].duplicated()]] = True
        rna = rna[:, keep_genes].copy()
    assert rna.var_names.is_unique

    return rna, atac


def load_cell_pools(source_dir):
    with open(source_dir / "prepared_manifest.json") as handle:
        manifest = json.load(handle)

    cell_pools = {}
    with np.load(source_dir / "cell_pools.npz", allow_pickle=False) as archive:
        for pool in manifest["outputs"]["cell_pools"]:
            cell_pools[(pool["sample_id"], pool["cell_type"])] = (
                archive[pool["array_key"]].astype(np.int64, copy=False)
            )
    return cell_pools


def load_source(run_config, tissue, sample_name):
    """Load the cached matrices, peaks and cell pools of one sample."""
    source_dir = (
        Path(run_config["cache_dir"]) / "prepared_sources" / f"{tissue}__{sample_name}"
    )
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Prepared source cache not found: {source_dir}")

    logging.info(f"Loading {tissue}:{sample_name} from {source_dir.resolve()}")
    rna, atac = load_multiome_data(run_config, tissue, sample_name)
    atac_peak_tensor = torch.load(
        source_dir / "prepared" / sample_name / "atac_peak_tensor.pt",
        map_location="cpu", weights_only=True,
    )
    if atac.n_vars != atac_peak_tensor.shape[0]:
        raise ValueError(
            f"{tissue}:{sample_name}: {atac.n_vars:,} ATAC peaks but "
            f"{atac_peak_tensor.shape[0]:,} cached peak sequences"
        )
    return {
        "tissue": tissue,
        "sample_name": sample_name,
        "source_dir": source_dir,
        "rna_names": rna.var_names.to_numpy(),
        "rna_mat": rna.X,
        "atac_mat": atac.X,
        "atac_peak_tensor": atac_peak_tensor,
        "cell_pools": load_cell_pools(source_dir),
    }


def check_gene_indices(split_df, rna_names, source=""):
    for col, name_col in (("tf_rna_col", "tf_name"), ("tg_rna_col", "tg_id")):
        found = rna_names[split_df[col].to_numpy()]
        expected = split_df[name_col].to_numpy()
        bad = found != expected
        assert not bad.any(), (
            f"{source}: {bad.sum():,} rows of {col} point to the wrong gene, "
            f"e.g. expected {expected[bad][0]}, found {found[bad][0]}"
        )


def find_cached_binding_scores(source, split_name, split_df, run_config):
    """Return the cached binding scores for a split, or None if no cache matches."""
    source_dir = source["source_dir"]
    candidates = [binding_score_cache_path(
        source_dir / "prepared",
        split_name,
        split_df,
        tf_dna_checkpoint=Path(run_config["tf_dna_checkpoint"]),
        max_peaks_per_tg=run_config["max_peaks_per_tg"],
    )]
    binding_manifest_path = source_dir / BINDING_MANIFEST
    if binding_manifest_path.is_file():
        splits = json.loads(binding_manifest_path.read_text())["splits"]
        if split_name in splits:
            candidates.append(source_dir / splits[split_name]["file"])

    expected_shape = (len(split_df), run_config["max_peaks_per_tg"])
    for path in candidates:
        if not path.is_file():
            continue
        scores = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(scores.shape) == expected_shape:
            logging.info(f"  {split_name}: binding scores from {path}")
            return scores
        logging.warning(
            f"  {path} has shape {tuple(scores.shape)}; expected {expected_shape}"
        )
    return None


class BindingScorer:
    """Compute binding scores that are not cached. Loads the TF-DNA model on first use."""

    def __init__(self, run_config, device):
        self.run_config = run_config
        self.device = device
        self.binding = None

    def __call__(self, name, frame, source, prepared_dir):
        if self.binding is None:
            self.binding = load_binding_model(SimpleNamespace(
                species=self.run_config["species"],
                tf_dna_checkpoint=Path(self.run_config["tf_dna_checkpoint"]),
            ), self.device)
        binding_model, embeddings, masks = self.binding
        peak_tensor = source["atac_peak_tensor"].to(device=self.device, dtype=torch.uint8)
        try:
            scores, path, _ = load_or_precompute_binding_scores(
                name,
                frame,
                binding_model,
                prepared_dir=prepared_dir,
                tf_dna_checkpoint=Path(self.run_config["tf_dna_checkpoint"]),
                tf_embeddings_tensor=embeddings,
                tf_mask_tensor=masks,
                atac_peak_tensor=peak_tensor,
                max_peaks_per_tg=self.run_config["max_peaks_per_tg"],
                device=self.device,
                chunk_size=self.run_config.get("binding_chunk_size", 2048),
            )
        finally:
            del peak_tensor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        logging.info(f"  {name}: computed binding scores into {path}")
        return scores


def load_split_edges(source, split_name, run_config, binding_scorer, run_dir,
                     celltypes=None):
    """Load one chromosome split of a sample with its binding scores."""
    split_path = source["source_dir"] / f"edges_{split_name}.parquet"
    if not split_path.is_file():
        return None, None
    split_df = pd.read_parquet(split_path)
    check_gene_indices(
        split_df, source["rna_names"],
        source=f"{source['tissue']}:{source['sample_name']} {split_name}",
    )

    scores = find_cached_binding_scores(source, split_name, split_df, run_config)

    if celltypes is not None:
        keep = split_df["cell_type"].isin(celltypes).to_numpy()
        split_df = split_df.loc[keep].reset_index(drop=True)
        if scores is not None:
            scores = scores[torch.from_numpy(keep)]
    if split_df.empty:
        return None, None

    if scores is None:
        source_key = f"{source['tissue']}__{source['sample_name']}"
        scores = binding_scorer(
            split_name, split_df, source,
            prepared_dir=run_dir / "holdout_evaluation" / source_key / "prepared",
        )
    return split_df, scores


@torch.no_grad()
def predict(model, edges, binding_scores, source, run_config, args, device, desc):
    """Return the edge probabilities of the model for one sample's edges."""
    dataset = TFTGEdgeBagDataset(
        edges,
        tf_embeddings_tensor=None,
        tf_mask_tensor=None,
        atac_peak_tensor=source["atac_peak_tensor"],
        atac_mat=source["atac_mat"],
        rna_mat=source["rna_mat"],
        cell_pools=source["cell_pools"],
        max_peaks_per_tg=run_config["max_peaks_per_tg"],
        resample_max_cells_per_pair=run_config["max_cells_per_pair"],
        resample_cells=False,
        binding_scores=binding_scores,
        seed=run_config["seed"],
    )
    num_workers = run_config["num_workers"] if args.num_workers is None else args.num_workers
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or run_config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    logits = []
    for batch in tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, mininterval=30):
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        edge_logits, _ = model(batch)
        logits.append(edge_logits.float().cpu())
    return torch.cat(logits).sigmoid().numpy(), dataset.inputs


def to_grn(inputs, probabilities, tissue):
    return (
        inputs[["sample_id", "cell_type", "tf_name", "tg_id", "label"]]
        .rename(columns={
            "sample_id": "SampleID",
            "cell_type": "CellType",
            "tf_name": "Source",
            "tg_id": "Target",
            "label": "Label",
        })
        .assign(
            Tissue=tissue,
            Score=probabilities,
            Source=lambda frame: frame["Source"].astype(str).str.upper(),
            Target=lambda frame: frame["Target"].astype(str).str.upper(),
        )
        .loc[:, GRN_COLUMNS]
    )


def log_auroc(name, grn):
    labels = grn["Label"].to_numpy()
    auroc = (
        roc_auc_score(labels, grn["Score"].to_numpy())
        if labels.min() != labels.max() else float("nan")
    )
    logging.info(f"{name}: {len(grn):,} edges, AUROC={auroc:.4f}")


def holdout_grn_paths(run_dir, holdout_specs):
    """Use <sample>_cross_grn.csv as in the notebook, unless two holdouts share a name."""
    sample_names = [sample for _, sample in holdout_specs]
    paths = {}
    for tissue, sample in holdout_specs:
        stem = sample if sample_names.count(sample) == 1 else f"{tissue}__{sample}"
        paths[(tissue, sample)] = run_dir / f"{stem}_cross_grn.csv"
    return paths


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    run_dir = args.run_dir
    run_config = json.loads((run_dir / "run_config.json").read_text())
    checkpoint = run_dir / "checkpoints" / args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logging.warning("CUDA is not available; predictions run on the CPU")

    specs = source_specs(run_config)
    holdout_specs = [tuple(value.split(":", 1)) for value in run_config.get("holdout_sample", [])]
    training_specs = [spec for spec in specs if spec not in holdout_specs]
    holdout_celltypes = set(run_config.get("holdout_celltype", []))
    if args.holdout_celltypes_only and not holdout_celltypes:
        raise ValueError("--holdout_celltypes_only: the run did not hold out any cell types")

    test_grn_path = run_dir / "test_set_grn.csv"
    holdout_paths = holdout_grn_paths(run_dir, holdout_specs)

    make_test_grn = args.overwrite or not test_grn_path.is_file()
    pending_holdouts = [
        spec for spec in holdout_specs if args.overwrite or not holdout_paths[spec].is_file()
    ]
    if not make_test_grn:
        logging.info(f"Test set GRN exists: {test_grn_path}")
    for spec in set(holdout_specs) - set(pending_holdouts):
        logging.info(f"Holdout GRN exists: {holdout_paths[spec]}")
    if not make_test_grn and not pending_holdouts:
        logging.info("All GRNs exist. Use --overwrite to regenerate them.")
        return

    # load_from_checkpoint restores the train-fitted input scaler from hyper_parameters.
    logging.info(f"Loading model from {checkpoint}")
    model = LitTFTGRegulationModel.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model.requires_grad_(False).eval().to(device)
    binding_scorer = BindingScorer(run_config, device)

    # Test set GRN: test-chromosome edges of each training sample.
    if make_test_grn:
        test_grns = []
        for tissue, sample_name in training_specs:
            source = load_source(run_config, tissue, sample_name)
            edges, scores = load_split_edges(
                source, "test", run_config, binding_scorer, run_dir,
            )
            if edges is not None:
                probabilities, inputs = predict(
                    model, edges, scores, source, run_config, args, device,
                    desc=f"{sample_name} test",
                )
                test_grns.append(to_grn(inputs, probabilities, tissue))
            del source
            gc.collect()

        if not test_grns:
            raise ValueError("No training sample has test edges")
        test_grn = pd.concat(test_grns, ignore_index=True)
        log_auroc("Test set", test_grn)
        test_grn.to_csv(test_grn_path, index=False)
        logging.info(f"Wrote {test_grn_path}")

    # Holdout GRNs: every chromosome split of each holdout sample.
    for tissue, sample_name in pending_holdouts:
        source = load_source(run_config, tissue, sample_name)
        celltypes = holdout_celltypes if args.holdout_celltypes_only else None
        split_grns = []
        for split_name in CHROMOSOME_SPLITS:
            edges, scores = load_split_edges(
                source, split_name, run_config, binding_scorer, run_dir, celltypes,
            )
            if edges is None:
                continue
            probabilities, inputs = predict(
                model, edges, scores, source, run_config, args, device,
                desc=f"{sample_name} {split_name}",
            )
            split_grns.append(to_grn(inputs, probabilities, tissue))
        del source
        gc.collect()

        if not split_grns:
            logging.warning(f"{tissue}:{sample_name} has no edges to predict; skipped")
            continue
        holdout_grn = pd.concat(split_grns, ignore_index=True)
        log_auroc(f"Holdout {tissue}:{sample_name}", holdout_grn)
        holdout_grn.to_csv(holdout_paths[(tissue, sample_name)], index=False)
        logging.info(f"Wrote {holdout_paths[(tissue, sample_name)]}")


if __name__ == "__main__":
    main()
