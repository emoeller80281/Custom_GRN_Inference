"""Pairwise top-k Jaccard stability between subsample GRNs.

Edges are encoded as int64 codes so set operations run in numpy instead of
on Python tuples. Results match the tuple/MultiIndex implementation exactly,
including score tie-breaking and the seeded random baseline.
"""
import os
import zlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

GRN_COLUMNS = ["subsample", "Source", "Target", "Score"]


def encode_subsamples(tissue_grn):
    """Map each subsample to (canonical codes, canonical scores, codes by rank).

    Canonical order is lexicographic (Source, Target). Rank order is score
    descending with ties broken by canonical order.
    """
    src = tissue_grn["Source"].astype(str).str.upper()
    tgt = tissue_grn["Target"].astype(str).str.upper()

    # sort=True makes code order equal to lexicographic (Source, Target) order.
    src_codes, _ = pd.factorize(src, sort=True)
    tgt_codes, tgt_uniques = pd.factorize(tgt, sort=True)
    codes = src_codes.astype(np.int64) * len(tgt_uniques) + tgt_codes

    enc = pd.DataFrame({
        "subsample": tissue_grn["subsample"].to_numpy(),
        "code": codes,
        "score": tissue_grn["Score"].to_numpy(dtype=float),
    })
    # Keep the first occurrence, as drop_duplicates on the raw edges did.
    enc = enc.drop_duplicates(["subsample", "code"])

    subsamples = {}
    for label, group in enc.groupby("subsample", sort=True, observed=True):
        code = group["code"].to_numpy()
        score = group["score"].to_numpy()

        canon = np.argsort(code)
        canon_codes = code[canon]
        canon_scores = score[canon]

        rank = np.lexsort((canon_codes, -canon_scores))
        subsamples[label] = (canon_codes, canon_scores, canon_codes[rank])

    return subsamples


def array_jaccard(a, b):
    inter = len(np.intersect1d(a, b, assume_unique=True))
    union = len(a) + len(b) - inter
    return inter / union if union else np.nan


def randomized_top_edges(canon_codes, canon_scores, common, k, seed_text):
    mask = np.isin(canon_codes, common, assume_unique=True)
    restricted_codes = canon_codes[mask]

    rng = np.random.default_rng(zlib.crc32(seed_text.encode()))
    shuffled_scores = rng.permutation(canon_scores[mask])

    # Stable sorting breaks score ties by the canonical edge order.
    positions = np.argsort(-shuffled_scores, kind="stable")[:k]
    return restricted_codes[positions]


def tissue_stability(method_name, tissue_type, tissue_grn, top_pct):
    subsamples = encode_subsamples(tissue_grn)

    rows = []
    for label_x, label_y in combinations(subsamples, 2):
        canon_x, scores_x, ranked_x = subsamples[label_x]
        canon_y, scores_y, ranked_y = subsamples[label_y]

        common = np.intersect1d(canon_x, canon_y, assume_unique=True)
        k = int(top_pct * len(common))

        if k == 0:
            jaccard_index = np.nan
            random_jaccard_index = np.nan
        else:
            top_x = ranked_x[np.isin(ranked_x, common, assume_unique=True)][:k]
            top_y = ranked_y[np.isin(ranked_y, common, assume_unique=True)][:k]
            jaccard_index = array_jaccard(top_x, top_y)

            seed_prefix = f"{method_name}_{tissue_type}_{label_x}_{label_y}"
            random_top_x = randomized_top_edges(
                canon_x, scores_x, common, k, f"{seed_prefix}_x"
            )
            random_top_y = randomized_top_edges(
                canon_y, scores_y, common, k, f"{seed_prefix}_y"
            )
            random_jaccard_index = array_jaccard(random_top_x, random_top_y)

        rows.append({
            "method_name": method_name,
            "tissue_name": tissue_type,
            "subsample_pair": f"{label_x}-{label_y}",
            "jaccard_index": jaccard_index,
            "random_jaccard_index": random_jaccard_index,
            "universe_jaccard": array_jaccard(canon_x, canon_y),
        })

    return pd.DataFrame(rows)


def grn_fingerprint(tissue_grn):
    """Cheap signature so a cache entry is recomputed when its input GRNs change."""
    counts = tissue_grn["subsample"].value_counts().sort_index()
    count_text = ",".join(f"{k}:{v}" for k, v in counts.items())
    return f"{count_text}|{tissue_grn['Score'].sum():.10g}"


def _run_and_cache(method_name, tissue_type, tissue_grn, top_pct, cache_path, fingerprint):
    result = tissue_stability(method_name, tissue_type, tissue_grn, top_pct)
    result["input_fingerprint"] = fingerprint
    result.to_parquet(cache_path, index=False)
    return result


def compute_stability(full_method_dict, cache_dir, top_pct=0.1, n_jobs=None, force=False):
    """Run tissue_stability for every (method, tissue), in parallel, with a parquet cache.

    One cache file per (method, tissue, top_pct). An entry is reused only when
    its stored fingerprint matches the current input GRN.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if n_jobs is None:
        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))

    results = {}
    tasks = []
    for method_name, tissue_dict in full_method_dict.items():
        for tissue_type, tissue_grn in tissue_dict.items():
            key = (method_name, tissue_type)
            cache_path = cache_dir / f"{method_name}__{tissue_type}__top{top_pct:g}.parquet"
            fingerprint = grn_fingerprint(tissue_grn)

            if cache_path.exists() and not force:
                cached = pd.read_parquet(cache_path)
                if (cached["input_fingerprint"] == fingerprint).all():
                    results[key] = cached
                    continue

            tasks.append((key, tissue_grn[GRN_COLUMNS], cache_path, fingerprint))

    print(f"{len(results)} cached, {len(tasks)} to compute with n_jobs={n_jobs}")

    computed = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_and_cache)(*key, grn, top_pct, path, fp)
        for key, grn, path, fp in tasks
    )
    results.update({task[0]: res for task, res in zip(tasks, computed)})

    ordered = [
        results[(m, t)]
        for m, tissue_dict in full_method_dict.items()
        for t in tissue_dict
    ]
    return pd.concat(ordered, ignore_index=True).drop(columns="input_fingerprint")
