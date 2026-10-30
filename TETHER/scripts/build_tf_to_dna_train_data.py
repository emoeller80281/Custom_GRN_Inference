import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse

import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch.nn.utils.rnn import pad_sequence

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
sys.path.append(str(PROJECT_DIR))

import utils
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def sample_items_excluding(n_items, excluded, k, rng):
    """k distinct item codes drawn uniformly from range(n_items), none of them in `excluded`."""
    if k > n_items - len(excluded):
        raise ValueError(f"Cannot draw {k:,} items: only {n_items - len(excluded):,} are not excluded.")
    picked = np.empty(0, dtype=np.int64)
    while len(picked) < k:
        draw = rng.integers(0, n_items, size=int((k - len(picked)) * 1.1) + 16)
        draw = draw[~np.isin(draw, excluded)]
        picked = np.unique(np.concatenate([picked, draw]))
    return rng.permutation(picked)[:k]


def in_sorted(values, sorted_arr):
    """Boolean mask: which of `values` occur in the sorted array."""
    if len(sorted_arr) == 0:
        return np.zeros(len(values), dtype=bool)
    pos = np.minimum(np.searchsorted(sorted_arr, values), len(sorted_arr) - 1)
    return sorted_arr[pos] == values


def sample_items_by_source_tf(target, k, observed_item, block, rng, pool_item=None, pool_block=None, reject=None):
    """k distinct item codes spread evenly over the observed items of every TF except `target`.

    The k draws are split as evenly as possible over the source TFs (the remainder goes to
    random ones), and each source TF's share is drawn uniformly from its own observed items.
    Items observed for `target` are never drawn. A source TF with too few eligible items gives
    all it has, and the shortfall is split again over the remaining source TFs.

    pool_item/pool_block replace each source TF's observed items as the pool to draw from (e.g.
    one item per merged binding site); `observed_item`/`block` still give the exclusions.
    reject(cand) -> bool mask drops more candidates (e.g. peaks overlapping the target's own peaks).
    """
    if pool_item is None:
        pool_item, pool_block = observed_item, block
    excluded = observed_item[block[target]:block[target + 1]]       # sorted
    sources = np.array([s for s in range(len(pool_block) - 1) if s != target and pool_block[s + 1] > pool_block[s]])
    picked = np.empty(0, dtype=np.int64)
    exhausted = set()
    while len(picked) < k:
        live = sources[~np.isin(sources, list(exhausted))]
        if len(live) == 0:
            raise ValueError(f"Cannot draw {k:,} items: only {len(picked):,} are eligible.")
        need = k - len(picked)
        quota = np.full(len(live), need // len(live))
        quota[rng.choice(len(live), need % len(live), replace=False)] += 1
        draws = []
        for s, q in zip(live, quota):
            if q == 0:
                continue
            items = pool_item[pool_block[s]:pool_block[s + 1]]
            if reject is None:
                cand = items[rng.integers(0, len(items), size=2 * q + 8)]
                cand = pd.unique(cand[~in_sorted(cand, excluded)])
            else:
                # Most of a source TF's sites can overlap the target's peaks (e.g. Ctcf), so draw
                # a few rounds before falling back to scanning the whole pool
                cand = np.empty(0, dtype=np.int64)
                for _ in range(4):
                    draw = items[rng.integers(0, len(items), size=4 * (q - len(cand)) + 16)]
                    draw = draw[~in_sorted(draw, excluded)]
                    cand = pd.unique(np.concatenate([cand, draw[~reject(draw)]]))
                    if len(cand) >= q:
                        break
            if len(cand) < q:                                          # small source TF
                cand = items[~in_sorted(items, excluded)]
                if reject is not None:
                    cand = cand[~reject(cand)]
                cand = rng.permutation(cand)
                if len(cand) <= q:
                    exhausted.add(s)
            draws.append(cand[:q])
        new = pd.unique(np.concatenate(draws))                         # a peak can have several TFs
        picked = np.concatenate([picked, new[~np.isin(new, picked)]])
    return rng.permutation(picked)


def peak_genome_coords(peak_ids):
    """Start and end of each "chrN:start-end" peak ID on one genome-wide axis.

    Chromosomes are laid end to end with 1 Mb gaps (and 1 Mb before the first), so a window
    around a peak never reaches a neighbouring chromosome and coordinates compare with < and >.
    """
    # large_string: the mm10 peak IDs pass 2 GB, where a plain string array splits into chunks
    ids = pa.array(peak_ids, type=pa.large_string())
    if isinstance(ids, pa.ChunkedArray):
        ids = ids.combine_chunks()
    parts = pc.extract_regex(ids, r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
    if parts.null_count:
        raise ValueError(f"{parts.null_count:,} peak IDs are not chrom:start-end")
    chrom = pc.dictionary_encode(parts.field("chrom")).indices.to_numpy().astype(np.int64)
    start = pc.cast(parts.field("start"), pa.int64()).to_numpy()
    end = pc.cast(parts.field("end"), pa.int64()).to_numpy()
    chrom_len = np.zeros(chrom.max() + 1, dtype=np.int64)
    np.maximum.at(chrom_len, chrom, end)
    offset = 1_000_000 + np.concatenate([[0], np.cumsum(chrom_len + 1_000_000)[:-1]])
    return offset[chrom] + start, offset[chrom] + end


def merge_tf_intervals(tf, start, end, tf_stride):
    """Union of each TF's peaks, as sorted disjoint intervals on a TF-offset axis (tf * tf_stride + pos)."""
    s = tf * tf_stride + start
    e = tf * tf_stride + end
    order = np.argsort(s, kind="stable")
    s, e = s[order], e[order]
    reach = np.maximum.accumulate(e)
    first = np.flatnonzero(np.concatenate([[True], s[1:] > reach[:-1]]))
    return s[first], np.maximum.reduceat(e, first)


def cluster_tf_sites(tf, center, max_gap, tf_stride):
    """Group each TF's peaks into binding sites: sorted centres closer than max_gap bp chain together.

    ChIP-Atlas lists every experiment's peaks separately, so one site bound in many experiments
    appears many times with slightly different coordinates. Returns the index of one peak per site
    (the middle one by centre), the site sizes, and the bp span of each site.
    """
    key = tf * tf_stride + center
    order = np.argsort(key, kind="stable")
    key = key[order]
    first = np.flatnonzero(np.concatenate([[True], np.diff(key) > max_gap]))
    sizes = np.diff(np.concatenate([first, [len(key)]]))
    spans = key[first + sizes - 1] - key[first]
    return order[first + sizes // 2], sizes, spans


def create_per_tf_balanced_edges(
    edge_df: pd.DataFrame,
    tf_names: list,
    tf_col: str = "source_id",
    item_col: str = "peak_id",
    min_true_edges_per_tf: int = 500,
    max_true_edges_per_tf: int | None = 20_000,
    true_false_ratio: float = 1.0,
    false_edge_sampling: str = "by_source_tf",
    tf_lengths: dict[str, int] | None = None,
    max_tf_length: int | None = None,
    site_merge_distance: int | None = None,
    false_overlap_flank: int | None = None,
    seed: int = 123,
):
    """True and false TF-peak edges with the same true:false ratio for every TF.

    A TF's true edges are its observed peaks, or a random max_true_edges_per_tf of them when it
    has more. Its false edges are round(n_true * true_false_ratio) peaks of other TFs, never a
    peak observed for that TF. TFs with fewer than min_true_edges_per_tf observed peaks, or a
    protein longer than max_tf_length (from tf_lengths), get no edges; their peaks can still be
    drawn as false edges of other TFs. false_edge_sampling picks how the false peaks are drawn:

      "by_source_tf"  evenly over the other TFs, then uniformly within each TF's peaks
      "uniform"       uniformly over all observed peaks of all TFs

    utils.create_true_false_edges samples the false edges over all TF x peak pairs at once, so
    every TF gets about the same number of them. On mm10 that left the bound fraction per TF
    between 0.0004 and 0.97, and the model learned it from the TF embedding without reading
    the DNA. A fixed ratio per TF removes that shortcut.

    With "uniform", Ctcf (41M of 155M mm10 peaks) and Spi1 gave ~26% and ~10% of every TF's
    false peaks, and the model learned "CTCF or PU.1 motif means unbound" for nearly every TF
    (one inserted CTCF consensus lowered other TFs by a median 4.6 log-odds). Drawing evenly
    over source TFs keeps any one TF's peaks from dominating the negatives.

    ChIP-Atlas peak IDs are per-experiment coordinates, so the exact-ID exclusion above misses
    other TFs' peaks at sites the TF also binds: on the 2026-10-22 mm10 cache 34% of val false
    edges had their window centre inside a peak of the same TF, and 28% of true edges had
    another true edge of the same TF within 50 bp. Two options fix this (by_source_tf only):

      site_merge_distance  merge each TF's peaks into binding sites (centres chained within this
                           many bp) and use one peak per site, as true edges and as the pool other
                           TFs draw false edges from
      false_overlap_flank  never draw a false peak whose centre +/- this many bp overlaps any
                           peak of the TF (use the model's window half-width)

    Returns (true_edges, false_edges, tf_counts): sets of (tf, item) tuples, and one row per
    TF with its length, observed, true and false edge counts, and why it was excluded.
    """
    if max_tf_length is not None and tf_lengths is None:
        raise ValueError("max_tf_length needs tf_lengths")
    if (site_merge_distance or false_overlap_flank) and false_edge_sampling != "by_source_tf":
        raise ValueError("site_merge_distance and false_overlap_flank need false_edge_sampling='by_source_tf'")
    df_all = edge_df[[tf_col, item_col]]
    df_all = df_all[df_all[tf_col].isin(tf_names)].dropna(subset=[tf_col, item_col])

    # Same integer coding as utils.create_true_false_edges: code = tf * n_items + item.
    tf_codes, candidate_tfs = pd.factorize(df_all[tf_col], sort=False)
    item_codes, candidate_items = pd.factorize(df_all[item_col], sort=False)
    candidate_tfs = np.asarray(candidate_tfs, dtype=object)
    candidate_items = np.asarray(candidate_items, dtype=object)
    n_items = len(candidate_items)

    # Sorted unique codes keep each TF's items in one sorted block.
    observed_codes = np.unique(tf_codes.astype(np.int64) * n_items + item_codes.astype(np.int64))
    observed_tf = observed_codes // n_items
    observed_item = observed_codes % n_items
    block = np.searchsorted(observed_tf, np.arange(len(candidate_tfs) + 1))

    # Optional genome coordinates for site merging and overlap-aware false edges
    pool_item, pool_block, reject_for = None, None, None
    if site_merge_distance or false_overlap_flank:
        item_start, item_end = peak_genome_coords(candidate_items)
        item_center = (item_start + item_end) // 2
        tf_stride = int(item_end.max()) + 2_000_000                   # keeps each TF's axis separate
    if site_merge_distance:
        rep, site_sizes, site_spans = cluster_tf_sites(observed_tf, item_center[observed_item], site_merge_distance, tf_stride)
        site_order = np.lexsort((observed_item[rep], observed_tf[rep]))  # by TF, then item code
        pool_item = observed_item[rep][site_order]
        pool_block = np.searchsorted(observed_tf[rep][site_order], np.arange(len(candidate_tfs) + 1))
        logging.info(
            f"    Merged {len(observed_item):,} TF-peak pairs into {len(rep):,} sites "
            f"(centres within {site_merge_distance} bp); site size median {np.median(site_sizes):.0f}, "
            f"p99 {np.quantile(site_sizes, 0.99):.0f}; site span p50/p90/p99 "
            f"{np.quantile(site_spans, 0.5):.0f}/{np.quantile(site_spans, 0.9):.0f}/{np.quantile(site_spans, 0.99):.0f} bp"
        )
    if false_overlap_flank:
        union_start, union_end = merge_tf_intervals(observed_tf, item_start[observed_item], item_end[observed_item], tf_stride)

        def reject_for(t):
            def reject(cand):
                lo = t * tf_stride + item_center[cand] - false_overlap_flank
                hi = t * tf_stride + item_center[cand] + false_overlap_flank
                i = np.searchsorted(union_start, hi, side="left") - 1
                return (i >= 0) & (union_end[np.maximum(i, 0)] > lo)
            return reject

    rng = np.random.default_rng(seed)
    true_edges, false_edges, rows = set(), set(), []
    for t, tf in enumerate(candidate_tfs):
        items = observed_item[block[t]:block[t + 1]]
        sites = items if pool_item is None else pool_item[pool_block[t]:pool_block[t + 1]]
        length = tf_lengths.get(tf) if tf_lengths is not None else None
        n_true = n_false = 0
        excluded = ""
        if len(sites) < min_true_edges_per_tf:
            excluded = f"fewer than {min_true_edges_per_tf} peaks" if pool_item is None else f"fewer than {min_true_edges_per_tf} sites"
        elif max_tf_length is not None and (length or 0) > max_tf_length:
            excluded = f"longer than {max_tf_length} residues"
        if not excluded:
            true_items = sites
            if max_true_edges_per_tf is not None and len(sites) > max_true_edges_per_tf:
                true_items = rng.choice(sites, size=max_true_edges_per_tf, replace=False)
            n_false_wanted = int(round(len(true_items) * true_false_ratio))
            if false_edge_sampling == "by_source_tf":
                false_items = sample_items_by_source_tf(
                    t, n_false_wanted, observed_item, block, rng, pool_item=pool_item, pool_block=pool_block,
                    reject=reject_for(t) if reject_for is not None else None,
                )
            elif false_edge_sampling == "uniform":
                false_items = sample_items_excluding(n_items, items, n_false_wanted, rng)
            else:
                raise ValueError(f"Unknown false_edge_sampling: {false_edge_sampling!r}")
            true_edges.update(zip([tf] * len(true_items), candidate_items[true_items]))
            false_edges.update(zip([tf] * len(false_items), candidate_items[false_items]))
            n_true, n_false = len(true_items), len(false_items)
        rows.append({"tf_name": tf, "tf_length": length, "n_observed": len(items), "n_sites": len(sites),
                     "n_true": n_true, "n_false": n_false, "excluded": excluded})

    tf_counts = pd.DataFrame(rows).sort_values("n_observed", ascending=False, ignore_index=True)
    kept = tf_counts["excluded"] == ""
    logging.info(
        f"    {kept.sum()} of {len(tf_counts)} TFs get edges; "
        f"{(tf_counts.loc[kept, 'n_sites'] > (max_true_edges_per_tf or np.inf)).sum()} capped at "
        f"{max_true_edges_per_tf} true edges; false edges drawn {false_edge_sampling}"
        + (f", none within {false_overlap_flank} bp of the TF's own peaks" if false_overlap_flank else "")
    )
    for reason, group in tf_counts[~kept].groupby("excluded"):
        logging.info(f"    Excluded, {reason}: {', '.join(group['tf_name'])}")
    return true_edges, false_edges, tf_counts


def create_labeled_tf_peak_dataset(
    true_interactions: set[tuple[str, str]],
    false_interactions: set[tuple[str, str]],
    tf_name_to_idx: dict[str, int],
    peak_id_to_idx: dict[str, int],
    drop_missing: bool = True,
) -> pd.DataFrame:
    
    rows = []
    for tf, peak in true_interactions:
        rows.append((tf, peak, 1))
    for tf, peak in false_interactions:
        rows.append((tf, peak, 0))

    df = pd.DataFrame(rows, columns=["tf_name", "peak_id", "label"])
    df["tf_idx"] = df["tf_name"].map(tf_name_to_idx)
    df["peak_idx"] = df["peak_id"].map(peak_id_to_idx)

    missing_mask = df["tf_idx"].isna() | df["peak_idx"].isna()
    if missing_mask.any():
        n_missing = missing_mask.sum()
        if drop_missing:
            logging.info(f"Dropping {n_missing} interactions with missing TF or peak indices.")
            df = df.loc[~missing_mask].copy()
        else:
            missing_examples = df.loc[missing_mask].head()
            raise ValueError(
                f"{n_missing} interactions are missing TF or peak indices.\n"
                f"Examples:\n{missing_examples}"
            )

    df["tf_idx"] = df["tf_idx"].astype(np.int64)
    df["peak_idx"] = df["peak_idx"].astype(np.int64)
    df["label"] = df["label"].astype(np.float32)

    return df.sample(frac=1.0, random_state=123).reset_index(drop=True)


def load_ordered_tf_embeddings(
    embedding_dir,
    tf_name_to_idx,
    suffix="_protein_embedding.pt",
    weights_only=True,
):
    embedding_dir = Path(embedding_dir)
    available_files = {}

    for path in embedding_dir.glob(f"*{suffix}"):
        tf_name = path.name.replace(suffix, "")
        available_files[tf_name] = path

    n_tfs = len(tf_name_to_idx)
    ordered_tf_names = [None] * n_tfs
    ordered_embeddings = [None] * n_tfs
    ordered_lengths = [0] * n_tfs

    missing_tfs = []
    for tf_name, tf_idx in tf_name_to_idx.items():
        ordered_tf_names[tf_idx] = tf_name

        if tf_name not in available_files:
            missing_tfs.append(tf_name)
            continue

        emb = torch.load(
            available_files[tf_name],
            weights_only=weights_only,
            map_location="cpu",
        )
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
        emb = emb.float()

        ordered_embeddings[tf_idx] = emb
        ordered_lengths[tf_idx] = emb.shape[0]

    if missing_tfs:
        raise FileNotFoundError(
            f"Missing embeddings for {len(missing_tfs)} TFs. "
            f"Examples: {missing_tfs[:20]}"
        )

    lengths = torch.tensor(ordered_lengths, dtype=torch.long)
    embeddings_padded = pad_sequence(
        ordered_embeddings,
        batch_first=True,
        padding_value=0.0,
    )

    max_len = embeddings_padded.shape[1]
    mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "tf_names": ordered_tf_names,
        "embeddings": embeddings_padded,
        "lengths": lengths,
        "mask": mask,
    }


def main():
    argparser = argparse.ArgumentParser(description="Build TF-to-DNA training data")
    argparser.add_argument("--min_true_edges_per_tf", type=int, default=500,
                           help="TFs with fewer observed ChIP-Atlas peaks get no training edges.")
    argparser.add_argument("--max_true_edges_per_tf", type=int, default=20_000,
                           help="Cap on true edges per TF (e.g. Ctcf, Spi1). 0 means no cap.")
    argparser.add_argument("--true_false_ratio", type=float, default=1.0,
                           help="False edges per true edge, the same for every TF.")
    argparser.add_argument("--false_edge_sampling", choices=["by_source_tf", "uniform"], default="by_source_tf",
                           help="Draw false peaks evenly over the other TFs, or uniformly over all peaks "
                                "(the 2026-10-21 cache; Ctcf and Spi1 peaks then dominate the negatives).")
    argparser.add_argument("--max_tf_length", type=int, default=2048,
                           help="TFs with longer protein sequences get no training edges; their peaks can "
                                "still be false edges of other TFs. Matches the top TF_CROP_LADDER rung in "
                                "models/tf_to_dna.py; longer TFs pad batches to the full table width. 0 means no limit.")
    argparser.add_argument("--site_merge_distance", type=int, default=0,
                           help="Merge each TF's peaks whose centres chain within this many bp into one binding "
                                "site (one true edge per site). 0 keeps every peak, as in the 2026-10-22 cache.")
    argparser.add_argument("--false_overlap_flank", type=int, default=0,
                           help="Never draw a false peak whose centre +/- this many bp overlaps any peak of the "
                                "same TF (use --peak_flank_size). 0 only excludes exact peak-ID matches.")
    argparser.add_argument("--cache_dir", type=Path, required=True,
                           help="Write the cache here, under cached_data/<species>/. The TF-TG cache build reads "
                                "its TF table and TF embeddings from cached_data/<species>/tf_dna_cache.")
    argparser.add_argument("--force_reload", action="store_true")
    argparser.add_argument("--peak_flank_size", type=int, default=128)
    args = argparser.parse_args()

    species = config.species

    genome_fasta_path = config.genome_fasta_path
    chrom_sizes_path = config.chrom_sizes_path
    embedding_dir = config.embedding_dir

    cache_dir = args.cache_dir
    config.check_cache_dir_species(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    in_cache = lambda path: cache_dir / path.name

    # Shared cache files for both TF-to-TG and TF-to-DNA training
    tf_name_to_idx_cache_path = in_cache(config.tf_name_to_idx_cache_path)
    tf_embedding_cache_path = in_cache(config.tf_embedding_cache_path)
    tf_mask_cache_path = in_cache(config.tf_mask_cache_path)

    # TF-DNA training specific cache files
    tf_dna_peak_id_to_idx_cache_path = in_cache(config.tf_dna_peak_id_to_idx_cache_path)
    tf_dna_edge_tf_idx_cache_path = in_cache(config.tf_dna_edge_tf_idx_cache_path)
    tf_dna_edge_peak_idx_cache_path = in_cache(config.tf_dna_edge_peak_idx_cache_path)
    tf_dna_edge_labels_cache_path = in_cache(config.tf_dna_edge_labels_cache_path)
    tf_dna_tf_lengths_cache_path = in_cache(config.tf_dna_tf_lengths_cache_path)
    tf_dna_peak_onehot_cache_path = in_cache(config.tf_dna_peak_onehot_cache_path)
    tf_dna_train_idx_cache_path = in_cache(config.tf_dna_train_idx_cache_path)
    tf_dna_val_idx_cache_path = in_cache(config.tf_dna_val_idx_cache_path)
    tf_dna_test_idx_cache_path = in_cache(config.tf_dna_test_idx_cache_path)
    
    # Check if all TF-to-DNA training cache files already exist before doing any expensive processing, unless --force_reload is specified
    all_required_files = [
        tf_name_to_idx_cache_path,
        tf_embedding_cache_path,
        tf_mask_cache_path,
        tf_dna_peak_id_to_idx_cache_path,
        tf_dna_edge_tf_idx_cache_path,
        tf_dna_edge_peak_idx_cache_path,
        tf_dna_edge_labels_cache_path,
        tf_dna_tf_lengths_cache_path,
        tf_dna_peak_onehot_cache_path,
        tf_dna_train_idx_cache_path,
        tf_dna_val_idx_cache_path,
        tf_dna_test_idx_cache_path,
    ]
    
    if all(f.exists() for f in all_required_files) and not args.force_reload:
        logging.info("All TF-to-DNA training cache files already exist. Use --force_reload to rebuild.")
        return

    tf_embedding_files = list(embedding_dir.glob("*_protein_embedding.pt"))
    embedded_tf_names = [f.stem.split("_protein_embedding")[0] for f in tf_embedding_files]
    logging.info(f"TFs with embeddings: {len(embedded_tf_names)}")
    logging.info(f"Example TFs with embeddings: {embedded_tf_names[:10]}")
    
    map_cache_files = [tf_name_to_idx_cache_path]
    edge_cache_files = [
        tf_dna_edge_tf_idx_cache_path, 
        tf_dna_edge_peak_idx_cache_path, 
        tf_dna_edge_labels_cache_path
        ]
    
    # Can skip reading in the true edges if edge_cache_files and map_cache_files exist
    if not all(f.exists() for f in map_cache_files + edge_cache_files) or args.force_reload:
        # Loads the ChIP-Atlas data for the species to filter the TFs with embeddings to those in the ChIP-Atlas data
        logging.info("Loading true edges...")
        true_edge_file = DATA_DIR / "ground_truth_files" / f"chip_atlas_{species}_all.parquet"
        true_edge_df = pd.read_parquet(true_edge_file)
        
        true_edge_df = true_edge_df[true_edge_df["source_id"].isin(embedded_tf_names)]
        logging.info(f"    Done. Loaded {len(true_edge_df):,} matching ChIP-Atlas edges for the {len(set(embedded_tf_names))} TFs with available embeddings.")
    
    # Creates or loads the TF and peak name to index mapping files
    logging.info("Creating or loading TF name to index mapping...")
    if tf_name_to_idx_cache_path.exists() and not args.force_reload:
        logging.info("    Loading TF name to index mapping from cache...")
        tf_name_to_idx = (
            pd.read_csv(tf_name_to_idx_cache_path)
            .set_index("tf_name")["tf_idx"]
            .to_dict()
        )
    else:
        logging.info("    Creating TF name to index mapping...")
        tf_names = true_edge_df["source_id"].unique().tolist()
        tf_name_to_idx = {tf: idx for idx, tf in enumerate(tf_names)}

        pd.DataFrame({
            "tf_name": list(tf_name_to_idx.keys()),
            "tf_idx": list(tf_name_to_idx.values()),
        }).to_csv(tf_name_to_idx_cache_path, index=False)

    tf_names = list(tf_name_to_idx.keys())

    # Before the edges, which need the TF lengths for --max_tf_length.
    logging.info("Creating or loading ordered TF embeddings...")
    tf_cache_files = [tf_embedding_cache_path, tf_mask_cache_path, tf_dna_tf_lengths_cache_path]
    if all(f.exists() for f in tf_cache_files) and not args.force_reload:
        logging.info(f"    Loading ordered TF embeddings and masks from cache...")
        tf_embeddings_tensor = torch.load(tf_embedding_cache_path, weights_only=True)
        tf_mask_tensor = torch.load(tf_mask_cache_path, weights_only=True)
        tf_lengths_tensor = torch.load(tf_dna_tf_lengths_cache_path, weights_only=True)
    else:
        logging.info("    Ordering TF embeddings to match the TF name to index map...")
        tf_data = load_ordered_tf_embeddings(
            embedding_dir=embedding_dir,
            tf_name_to_idx=tf_name_to_idx,
        )

        tf_embeddings_tensor = tf_data["embeddings"]
        tf_mask_tensor = tf_data["mask"]
        tf_lengths_tensor = tf_data["lengths"]

        logging.info("    Saving TF embeddings and masks to cache...")
        torch.save(tf_embeddings_tensor, tf_embedding_cache_path)
        torch.save(tf_mask_tensor, tf_mask_cache_path)
        torch.save(tf_lengths_tensor, tf_dna_tf_lengths_cache_path)
    logging.info(f"    Done. Created/Loaded TF embeddings with shape {tf_embeddings_tensor.shape} and mask with shape {tf_mask_tensor.shape}.")
    tf_lengths = {tf: int(tf_lengths_tensor[idx]) for tf, idx in tf_name_to_idx.items()}

    # ------------------------------------------------------------
    # Load or create labeled TF-peak edge dataset + compact peak map
    # ------------------------------------------------------------
    logging.info("Creating or loading labeled dataset of TF-peak interactions")

    if all(f.exists() for f in edge_cache_files + [tf_dna_peak_id_to_idx_cache_path]) and not args.force_reload:
        logging.info("    Loading edge indices, labels, and TF-DNA peak map from cache...")

        edge_tf_idx_tensor = torch.load(tf_dna_edge_tf_idx_cache_path, weights_only=True)
        edge_peak_idx_tensor = torch.load(tf_dna_edge_peak_idx_cache_path, weights_only=True)
        edge_labels_tensor = torch.load(tf_dna_edge_labels_cache_path, weights_only=True)

        tf_dna_peak_id_to_idx = (
            pd.read_csv(tf_dna_peak_id_to_idx_cache_path)
            .set_index("peak_id")["peak_idx"]
            .to_dict()
        )

        # Make sure values are normal Python ints
        tf_dna_peak_id_to_idx = {
            peak_id: int(idx) for peak_id, idx in tf_dna_peak_id_to_idx.items()
        }

        # Reconstruct ordered peak_ids from the cached compact mapping
        used_peak_ids = [
            peak_id
            for peak_id, idx in sorted(
                tf_dna_peak_id_to_idx.items(),
                key=lambda x: x[1],
            )
        ]

    else:
        logging.info("    Creating True/False edges")
        full_peak_ids = true_edge_df["peak_id"].unique().tolist()
        full_peak_id_to_idx = {
            peak_id: idx for idx, peak_id in enumerate(full_peak_ids)
        }

        # Every TF stays in tf_name_to_idx (the TF-TG model shares that table); TFs below
        # min_true_edges_per_tf or above max_tf_length only lose their training edges.
        true_interactions, false_interactions, tf_edge_counts = create_per_tf_balanced_edges(
            edge_df=true_edge_df,
            tf_names=tf_names,
            tf_col="source_id",
            item_col="peak_id",
            min_true_edges_per_tf=args.min_true_edges_per_tf,
            max_true_edges_per_tf=args.max_true_edges_per_tf or None,
            true_false_ratio=args.true_false_ratio,
            false_edge_sampling=args.false_edge_sampling,
            tf_lengths=tf_lengths,
            max_tf_length=args.max_tf_length or None,
            site_merge_distance=args.site_merge_distance or None,
            false_overlap_flank=args.false_overlap_flank or None,
        )
        tf_edge_counts.to_csv(cache_dir / "tf_edge_counts.csv", index=False)

        logging.info(
            f"    Creating labeled dataset with "
            f"{len(true_interactions)} true interactions and "
            f"{len(false_interactions)} false interactions."
        )

        # This can use the full peak_id_to_idx temporarily just to validate
        # that peaks exist in the broader universe.
        tf_peak_labeled_df = create_labeled_tf_peak_dataset(
            true_interactions=true_interactions,
            false_interactions=false_interactions,
            tf_name_to_idx=tf_name_to_idx,
            peak_id_to_idx=full_peak_id_to_idx,
            drop_missing=False,
        )

        # Define compact TF-DNA peak universe
        used_peak_ids = (
            tf_peak_labeled_df["peak_id"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        tf_dna_peak_id_to_idx = {
            peak_id: idx for idx, peak_id in enumerate(used_peak_ids)
        }

        # Remap edge peak_idx to compact TF-DNA peak tensor rows
        tf_peak_labeled_df["peak_idx"] = (
            tf_peak_labeled_df["peak_id"]
            .map(tf_dna_peak_id_to_idx)
            .astype(np.int64)
        )

        # Save compact peak map
        pd.DataFrame({
            "peak_id": list(tf_dna_peak_id_to_idx.keys()),
            "peak_idx": list(tf_dna_peak_id_to_idx.values()),
        }).to_csv(tf_dna_peak_id_to_idx_cache_path, index=False)

        edge_tf_idx = tf_peak_labeled_df["tf_idx"].to_numpy(dtype=np.int64)
        edge_peak_idx = tf_peak_labeled_df["peak_idx"].to_numpy(dtype=np.int64)
        edge_labels = tf_peak_labeled_df["label"].to_numpy(dtype=np.float32)

        edge_tf_idx_tensor = torch.as_tensor(edge_tf_idx, dtype=torch.long)
        edge_peak_idx_tensor = torch.as_tensor(edge_peak_idx, dtype=torch.long)
        edge_labels_tensor = torch.as_tensor(edge_labels, dtype=torch.float32)

        torch.save(edge_tf_idx_tensor, tf_dna_edge_tf_idx_cache_path)
        torch.save(edge_peak_idx_tensor, tf_dna_edge_peak_idx_cache_path)
        torch.save(edge_labels_tensor, tf_dna_edge_labels_cache_path)

    logging.info(
        f"    Done. Created/Loaded dataset with {len(edge_labels_tensor):,} edges "
        f"and {len(used_peak_ids):,} compact peaks."
    )

    logging.info("Creating or loading peak one-hot encodings...")
    chrom_sizes = utils.load_chrom_sizes(chrom_sizes_path)

    if os.path.exists(tf_dna_peak_onehot_cache_path) and not args.force_reload:
        logging.info(f"    Loading peak one-hot encodings from cache...")
        peak_tensor = torch.load(tf_dna_peak_onehot_cache_path, weights_only=True)
    else:
        logging.info(f"    Creating centered peak one-hot encodings for {len(used_peak_ids):,} peaks...")
        peak_onehot_array = utils.create_centered_peak_onehot_array(
            peak_ids=used_peak_ids,
            genome_fasta=genome_fasta_path,
            chrom_sizes=chrom_sizes,
            peak_id_to_idx=tf_dna_peak_id_to_idx,
            flank_size=args.peak_flank_size,
            dtype=np.uint8,
            pad_out_of_bounds=True,
            num_workers=64,
            show_progress=True,
            chunk_size=100_000,
        )
        # Kept as uint8. A one-hot has no information beyond 0/1, so float32 on disk is a
        # 4x tax for nothing -- 31 GB vs 7.8 GB for hg38. Every consumer casts on use:
        # LitTFPeakBindingModel._shared_step calls .float() per batch, and
        # TFTGRegulationModel.forward has an explicit is_floating_point() check.
        peak_tensor = torch.as_tensor(peak_onehot_array, dtype=torch.uint8)
        torch.save(peak_tensor, tf_dna_peak_onehot_cache_path)
        logging.info("    Done. Created/Loaded peak one-hot encodings.")

    logging.info("Creating or loading train/val/test splits...")

    split_cache_files = [
        tf_dna_train_idx_cache_path,
        tf_dna_val_idx_cache_path,
        tf_dna_test_idx_cache_path,
    ]

    if all(p.exists() for p in split_cache_files) and not args.force_reload:
        logging.info("    Loading train/val/test splits from cache...")
        train_idx = torch.load(tf_dna_train_idx_cache_path, weights_only=True)
        val_idx = torch.load(tf_dna_val_idx_cache_path, weights_only=True)
        test_idx = torch.load(tf_dna_test_idx_cache_path, weights_only=True)

    else:
        logging.info("    Creating train/val/test splits based on chromosome...")

        idx_to_tf_dna_peak_id = {
            idx: peak_id for peak_id, idx in tf_dna_peak_id_to_idx.items()
        }

        edge_peak_ids = np.array([
            idx_to_tf_dna_peak_id[int(idx)]
            for idx in edge_peak_idx_tensor.cpu().numpy()
        ])

        def _get_chrom(peak_id: str) -> str:
            return peak_id.split(":", 1)[0]

        edge_chroms = np.array([_get_chrom(pid) for pid in edge_peak_ids])

        if config.species == "mm10":
            train_chroms = [f"chr{i}" for i in range(1, 16)]
            val_chroms = [f"chr{i}" for i in range(16, 18)]
            test_chroms = [f"chr{i}" for i in range(18, 20)]
        elif config.species == "hg38":
            train_chroms = [f"chr{i}" for i in range(1, 18)]
            val_chroms = [f"chr{i}" for i in range(18, 20)]
            test_chroms = [f"chr{i}" for i in range(20, 23)]

        train_mask = np.isin(edge_chroms, list(train_chroms))
        val_mask = np.isin(edge_chroms, list(val_chroms))
        test_mask = np.isin(edge_chroms, list(test_chroms))

        train_idx = torch.as_tensor(np.where(train_mask)[0], dtype=torch.long)
        val_idx = torch.as_tensor(np.where(val_mask)[0], dtype=torch.long)
        test_idx = torch.as_tensor(np.where(test_mask)[0], dtype=torch.long)

        torch.save(train_idx, tf_dna_train_idx_cache_path)
        torch.save(val_idx, tf_dna_val_idx_cache_path)
        torch.save(test_idx, tf_dna_test_idx_cache_path)
        
    logging.info(f"    Done. Created/Loaded train/val/test splits with {len(train_idx)}, {len(val_idx)}, and {len(test_idx)} edges, respectively.")

    logging.info("\nFinished building TF-to-DNA training data")
    logging.info(f"  - Cached TF embeddings: {tf_embeddings_tensor.shape}")
    logging.info(f"  - Cached peak tensor: {peak_tensor.shape}")
    logging.info(f"  - Train/Val/Test edges: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")


if __name__ == "__main__":
    main()
