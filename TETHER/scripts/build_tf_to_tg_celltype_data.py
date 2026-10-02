#!/usr/bin/env python
"""Prepare and cache cell-type TF->TG training data, one sample at a time.

Each sample is cached on its own, keyed only by that sample's inputs and the
preprocessing settings. The training dataset list and holdouts do not change the
key, so one cache serves every training run that uses the sample. Run one GPU job
per sample (bash_scripts/03c_build_celltype_data.sh), then train from the caches
with scripts/train_cached_tf_to_tg_celltype_model.py.

Example: python scripts/build_tf_to_tg_celltype_data.py --dataset mESC:E7.5_rep1

A sample cache holds the prepare_data outputs (edges, cell pools, peak one-hots)
and TF-DNA binding scores for every chromosome split. binding_manifest.json is
written last and marks the cache as complete. An interrupted job resumes from the
files it already wrote.
"""
import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_tf_to_tg_celltype_model import (  # noqa: E402
    PROJECT_DIR,
    _atomic_write_json,
    _cache_digest,
    _file_identity,
    _optional_file_identity,
    load_or_precompute_binding_scores,
    prepare_data,
)

SAMPLE_CACHE_VERSION = 1
BINDING_MANIFEST = "binding_manifest.json"


def add_data_args(parser):
    """Arguments that select a sample cache. The build and training scripts must agree."""
    parser.add_argument("--species", choices=["mm10", "hg38"], default="mm10")
    parser.add_argument(
        "--dataset", action="append", required=True, metavar="TISSUE:SAMPLE",
        help="Repeat for several samples, for example --dataset mESC:E7.5_rep1",
    )
    parser.add_argument("--data_dir", type=Path, default=PROJECT_DIR.parent / "data")
    parser.add_argument("--gene_ref_file", type=Path,
                        help="Override the species-specific gene annotation")
    parser.add_argument("--tf_dna_checkpoint", type=Path,
                        help="Required for hg38; defaults to the notebook checkpoint for mm10")
    parser.add_argument(
        "--sample_cache_root", type=Path,
        help="Root of the per-sample caches "
             "(default: cached_data/<species>/celltype_tf_tg/samples)",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_cells_per_pair", type=int, default=64)
    parser.add_argument("--max_peaks_per_tg", type=int, default=25)
    parser.add_argument("--peak_flank_size", type=int, default=128)
    parser.add_argument("--true_false_ratio", type=float, default=2.)
    parser.add_argument("--balance_tf", action="store_true")
    parser.add_argument("--balance_tg", action="store_true")
    parser.add_argument("--min_cells_per_slice", type=int)
    parser.add_argument("--min_tfs_per_slice", type=int, default=5)
    parser.add_argument("--max_gt_density", type=float, default=.60)
    parser.add_argument("--seed", type=int, default=123)


def finalize_data_args(parser, args):
    """Validate the data arguments, fill defaults, and return the (tissue, sample) list."""
    for key in ("max_cells_per_pair", "max_peaks_per_tg", "peak_flank_size"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be nonnegative")
    if args.true_false_ratio < 0 or not 0 < args.max_gt_density <= 1:
        parser.error("true_false_ratio must be nonnegative and max_gt_density in (0, 1]")
    if args.min_tfs_per_slice <= 0 or (
            args.min_cells_per_slice is not None and args.min_cells_per_slice <= 0):
        parser.error("Slice cell and TF thresholds must be positive")

    malformed = [value for value in args.dataset
                 if value.count(":") != 1 or not all(value.split(":"))]
    if malformed:
        parser.error(f"--dataset values must be TISSUE:SAMPLE; invalid: {malformed}")
    source_specs = [tuple(value.split(":", 1)) for value in args.dataset]
    if len(set(source_specs)) != len(source_specs):
        parser.error(f"Duplicate --dataset entries: {args.dataset}")

    if args.tf_dna_checkpoint is None:
        if args.species != "mm10":
            parser.error("--tf_dna_checkpoint is required for hg38")
        args.tf_dna_checkpoint = (
            PROJECT_DIR / "checkpoints/new_tf_dna_models/mm10_3831017_epoch_12.ckpt"
        )
    if not args.tf_dna_checkpoint.is_file():
        parser.error(f"TF-DNA checkpoint not found: {args.tf_dna_checkpoint}")
    if args.sample_cache_root is None:
        args.sample_cache_root = (
            PROJECT_DIR / "cached_data" / args.species / "celltype_tf_tg" / "samples"
        )
    return source_specs


def tf_dna_cache_dir(species):
    return PROJECT_DIR / "cached_data" / species / "tf_dna_cache"


def gene_reference_path(args):
    """The same annotation prepare_data reads."""
    if args.gene_ref_file is not None:
        return args.gene_ref_file
    file_name = (
        "Mus_musculus.GRCm39.115.gtf.gz" if args.species == "mm10"
        else "Homo_sapiens.GRCh38.113.gtf.gz"
    )
    return args.data_dir / "genome_data/genome_annotation" / args.species / file_name


def build_sample_cache_manifest(args, tissue, sample_name):
    """Describe every input that can change one sample's edges or binding scores."""
    data_dir = args.data_dir
    reference_dir = data_dir / "genome_data/reference_genome" / args.species
    ground_truth_root = data_dir / "ground_truth_files/cell_type_specific"
    ground_truth_dir = ground_truth_root / args.species / tissue
    ground_truth_files = sorted(ground_truth_dir.glob("*_ground_truth.parquet"))
    if not ground_truth_files:
        raise FileNotFoundError(f"No ground-truth parquet files found in {ground_truth_dir}")
    tf_cache = tf_dna_cache_dir(args.species)

    return {
        "cache_version": SAMPLE_CACHE_VERSION,
        "species": args.species,
        "dataset": f"{tissue}:{sample_name}",
        "settings": {
            "seed": args.seed,
            # prepare_data uses max_cells_per_pair only for this default, so the cell
            # bag size can change between training runs without a rebuild.
            "min_cells_per_slice": args.min_cells_per_slice or 2 * args.max_cells_per_pair,
            "max_peaks_per_tg": args.max_peaks_per_tg,
            "peak_flank_size": args.peak_flank_size,
            "true_false_ratio": args.true_false_ratio,
            "balance_tf": args.balance_tf,
            "balance_tg": args.balance_tg,
            "min_tfs_per_slice": args.min_tfs_per_slice,
            "max_gt_density": args.max_gt_density,
        },
        "references": {
            "gene_reference": _file_identity(gene_reference_path(args)),
            "genome_fasta": _file_identity(reference_dir / f"{args.species}.fa"),
            "chromosome_sizes": _file_identity(reference_dir / f"{args.species}.chrom.sizes"),
            "tf_name_to_idx": _file_identity(tf_cache / "tf_name_to_idx.csv"),
            "tf_embeddings": _file_identity(tf_cache / "tf_embeddings.pt"),
            "tf_masks": _file_identity(tf_cache / "tf_masks.pt"),
            "tf_dna_checkpoint": _file_identity(args.tf_dna_checkpoint),
        },
        "source": {
            "multiome": _file_identity(
                data_dir / "processed" / tissue / sample_name / "multiome_processed.h5mu"
            ),
            "label_map": _optional_file_identity(
                ground_truth_root / f"{tissue}_label_map.tsv"
            ),
            "ground_truth": [_file_identity(path) for path in ground_truth_files],
        },
    }


def sample_cache_location(args, tissue, sample_name):
    """Return (cache directory, manifest) for one sample."""
    manifest = build_sample_cache_manifest(args, tissue, sample_name)
    directory = (
        args.sample_cache_root / f"{tissue}__{sample_name}" / _cache_digest(manifest)
    )
    return directory, manifest


def is_sample_cache_complete(directory):
    marker = Path(directory) / BINDING_MANIFEST
    if not marker.is_file():
        return False
    binding_manifest = json.loads(marker.read_text())
    return binding_manifest.get("cache_key") == Path(directory).name and all(
        (Path(directory) / split["file"]).is_file()
        for split in binding_manifest["splits"].values()
    )


def prepare_source(args, tissue, sample_name, directory, manifest):
    """Run prepare_data for one sample, reading the cache when it matches."""
    source_args = copy.copy(args)
    source_args.tissue = tissue
    source_args.sample_name = sample_name
    source_args.output_dir = Path(directory)
    source_args.prepared_cache_manifest = manifest
    source_args.output_dir.mkdir(parents=True, exist_ok=True)
    splits, pools, atac, rna, peaks, embedding_path, mask_path = prepare_data(source_args)
    return {
        "tissue": tissue,
        "sample_name": sample_name,
        "output_dir": source_args.output_dir,
        "splits": splits,
        "pools": pools,
        "atac": atac,
        "rna": rna,
        "peaks": peaks,
        "embedding_path": embedding_path,
        "mask_path": mask_path,
    }


def load_cached_source(args, tissue, sample_name, directory, manifest):
    """Load a complete sample cache with its binding scores for every split."""
    source = prepare_source(args, tissue, sample_name, directory, manifest)
    binding_manifest = json.loads((Path(directory) / BINDING_MANIFEST).read_text())
    source["binding_scores"] = {}
    for split_name, frame in source["splits"].items():
        path = Path(directory) / binding_manifest["splits"][split_name]["file"]
        scores = torch.load(path, map_location="cpu", weights_only=True)
        expected_shape = (len(frame), args.max_peaks_per_tg)
        if tuple(scores.shape) != expected_shape:
            raise ValueError(
                f"{path} has shape {tuple(scores.shape)}; expected {expected_shape}"
            )
        source["binding_scores"][split_name] = (scores, path)
    return source


def load_binding_model(args, device):
    """Load the frozen TF-DNA model plus the TF embeddings and masks it reads."""
    from models.tf_to_dna import TFPeakBindingModel, LitTFPeakBindingModel

    tf_cache = tf_dna_cache_dir(args.species)
    embeddings = torch.load(tf_cache / "tf_embeddings.pt", map_location="cpu", weights_only=True)
    masks = torch.load(tf_cache / "tf_masks.pt", map_location="cpu", weights_only=True)

    base = TFPeakBindingModel(tf_embedding_dim=128, hidden_dim=128, dropout=.3,
                              num_layers=4, num_heads=4, dim_head=32)
    binding_model = LitTFPeakBindingModel.load_from_checkpoint(
        str(args.tf_dna_checkpoint),
        map_location="cpu",
        model=base,
        tf_embeddings_tensor=embeddings,
        tf_mask_tensor=masks,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
    ).model
    binding_model.requires_grad_(False).eval().to(device)
    return (
        binding_model,
        embeddings.to(device=device, dtype=torch.float32),
        masks.to(device=device, dtype=torch.bool),
    )


def build_sample_cache(args, tissue, sample_name, directory, manifest, binding, device):
    """Prepare one sample and compute binding scores for each of its splits."""
    binding_model, embeddings, masks = binding
    source = prepare_source(args, tissue, sample_name, directory, manifest)
    peak_tensor = source["peaks"].to(device=device, dtype=torch.uint8)

    splits = {}
    for split_name, frame in source["splits"].items():
        scores, path, _ = load_or_precompute_binding_scores(
            split_name,
            frame,
            binding_model,
            prepared_dir=Path(directory) / "prepared",
            tf_dna_checkpoint=args.tf_dna_checkpoint,
            tf_embeddings_tensor=embeddings,
            tf_mask_tensor=masks,
            atac_peak_tensor=peak_tensor,
            max_peaks_per_tg=args.max_peaks_per_tg,
            device=device,
            chunk_size=args.binding_chunk_size,
        )
        splits[split_name] = {
            "file": str(path.relative_to(directory)),
            "edges": len(frame),
            "positive_fraction": float(frame.label.mean()),
            "celltypes": sorted(frame.cell_type.unique().tolist()),
        }
        del scores

    del peak_tensor
    torch.cuda.empty_cache()

    _atomic_write_json(Path(directory) / BINDING_MANIFEST, {
        "cache_key": Path(directory).name,
        "dataset": f"{tissue}:{sample_name}",
        "n_cells": int(source["rna"].shape[0]),
        "n_peaks": int(source["atac"].shape[1]),
        "splits": splits,
    })
    logging.info("%s:%s cache complete: %s", tissue, sample_name, directory)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_data_args(parser)
    parser.add_argument(
        "--binding_chunk_size", type=int, default=256,
        help="(TF, peak) pairs per binding-model forward pass",
    )
    args = parser.parse_args()
    source_specs = finalize_data_args(parser, args)
    if args.binding_chunk_size <= 0:
        parser.error("--binding_chunk_size must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # precompute_binding_scores runs CUDA autocast and memory statistics.
    if not torch.cuda.is_available():
        raise RuntimeError("Binding scores need a CUDA GPU")
    device = torch.device("cuda")

    binding = None
    for tissue, sample_name in source_specs:
        directory, manifest = sample_cache_location(args, tissue, sample_name)
        if is_sample_cache_complete(directory):
            logging.info("%s:%s cache is already complete: %s", tissue, sample_name, directory)
            continue
        logging.info("%s:%s building cache in %s", tissue, sample_name, directory)
        if binding is None:
            binding = load_binding_model(args, device)
        build_sample_cache(args, tissue, sample_name, directory, manifest, binding, device)


if __name__ == "__main__":
    main()
