"""Check that TF embeddings are comparable to each other before anything trains on them.

Three checks, cheapest first. All of them operate on the mean-pooled per-TF embedding,
which is what any cross-TF comparison ultimately reduces to.

  1. BASIS      Are all TFs in the same coordinate system at all?
                If each TF were projected through its own random matrix, the pairwise
                cosines would look like independent random vectors -- SD ~ 1/sqrt(d) --
                and each TF's top principal residue direction would be unrelated to every
                other's (mean |cos| ~ sqrt(2/pi)/sqrt(d)). Both were true of the
                embeddings produced before extract_tf_embeddings.py shared its projection:
                0.0883 vs 0.0884 predicted, and 0.0707 vs 0.071 predicted.

  2. ORTHOLOGS  Mouse Sox2 and human SOX2 are near-identical proteins, so their embeddings
                should be far closer to each other than to a random TF of the other
                species. Needs both species' embedding directories.

  3. FAMILY     Within a species, TFs sharing a DBD family (CIS-BP Family_Name) should be
                more similar than TFs that do not. Reported as the AUROC of separating
                same-family from different-family pairs. Chance = 0.5; the broken
                embeddings scored 0.497.

Usage
-----
    python scripts/validate_tf_embeddings.py \
        --embedding_dir mm10=<path>/tf_data/mm10/tf_embeddings \
                        hg38=<path>/tf_data/hg38/tf_embeddings

Add --strict to exit non-zero when a check fails, e.g. from a batch script before the
cache rebuild starts.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DATA_DIR = Path(
    "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data"
)
EMBEDDING_SUFFIX = "_protein_embedding.pt"

# A cosine AUROC this far above chance is the bar for "carries protein identity".
FAMILY_AUROC_MIN = 0.60
ORTHOLOG_AUROC_MIN = 0.90


def load_pooled_embeddings(embedding_dir):
    """{TF name (upper) -> mean-pooled embedding} plus each TF's top principal direction.

    Files are unpadded [L, d], one per TF, so the pool is a plain mean over residues.
    """
    embedding_dir = Path(embedding_dir)
    files = sorted(embedding_dir.glob(f"*{EMBEDDING_SUFFIX}"))
    if not files:
        raise FileNotFoundError(f"No {EMBEDDING_SUFFIX} files in {embedding_dir}")

    # A directory holding two generations at once is the quietest way for all of this to
    # go wrong: the checks still run, on a mixture, and report a number that is neither
    # generation's. It happened -- a changed output suffix meant new files sat beside old
    # ones instead of replacing them, and the mixture scored ortholog AUROC 0.73 where the
    # new embeddings alone scored 0.99. Timestamps are the cheap tell.
    mtimes = [p.stat().st_mtime for p in files]
    span_hours = (max(mtimes) - min(mtimes)) / 3600
    if span_hours > 24:
        oldest = datetime.fromtimestamp(min(mtimes)).strftime("%Y-%m-%d %H:%M")
        newest = datetime.fromtimestamp(max(mtimes)).strftime("%Y-%m-%d %H:%M")
        print(
            f"\n  WARNING  {embedding_dir} holds files written {span_hours / 24:.1f} days apart\n"
            f"           ({oldest} .. {newest}). If these are two generations, every check\n"
            f"           below is measuring a mixture. Delete the directory and re-project."
        )

    pooled, top_directions = {}, {}
    for path in files:
        name = path.name.replace(EMBEDDING_SUFFIX, "").upper()
        emb = torch.load(path, map_location="cpu", weights_only=True).float().numpy()
        if emb.ndim == 3 and emb.shape[0] == 1:
            emb = emb[0]

        pooled[name] = emb.mean(axis=0)

        centered = emb - emb.mean(axis=0)
        top_directions[name] = np.linalg.svd(centered, full_matrices=False)[2][0]

    return pooled, top_directions


def cosine_matrix(vectors):
    V = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = V / norms
    return V @ V.T


def auroc(scores, labels):
    """Rank-based AUROC, so this file needs no sklearn import."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within ties, so exact ties score 0.5 rather than an arbitrary order
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# 1. Basis
# ---------------------------------------------------------------------------

def check_basis(species, pooled, top_directions):
    names = sorted(pooled)
    d = len(pooled[names[0]])
    n = len(names)
    iu = np.triu_indices(n, 1)

    cos = cosine_matrix([pooled[t] for t in names])[iu]
    top = np.abs(cosine_matrix([top_directions[t] for t in names]))[iu]

    random_cos_sd = 1.0 / np.sqrt(d)
    random_top_mean = np.sqrt(2.0 / np.pi) / np.sqrt(d)

    # Within a tenth of the random-basis prediction on BOTH statistics is the signature
    # of per-TF projections; one alone could happen by coincidence.
    looks_random = (
        abs(cos.std() - random_cos_sd) < 0.1 * random_cos_sd
        and abs(top.mean() - random_top_mean) < 0.1 * random_top_mean
    )

    print(f"\n[1] BASIS -- {species} ({n} TFs, d={d})")
    print(f"      SD of pairwise cosine          {cos.std():.4f}   (random bases predict {random_cos_sd:.4f})")
    print(f"      mean |cos| of top PC directions {top.mean():.4f}   (random bases predict {random_top_mean:.4f})")
    print(f"      mean pairwise cosine           {cos.mean():+.4f}")

    if looks_random:
        print("      FAIL  matches the independent-random-basis prediction on both statistics.")
        print("            Every TF is in its own coordinate system -- no cross-TF comparison is valid.")
        return False

    print("      PASS  embeddings do not look like independent random bases.")
    return True


# ---------------------------------------------------------------------------
# 2. Orthologs
# ---------------------------------------------------------------------------

def check_orthologs(pooled_by_species):
    if len(pooled_by_species) < 2:
        print("\n[2] ORTHOLOGS -- skipped (needs two species' embedding directories)")
        return None

    (sp_a, pooled_a), (sp_b, pooled_b) = list(pooled_by_species.items())[:2]
    shared = sorted(set(pooled_a) & set(pooled_b))

    print(f"\n[2] ORTHOLOGS -- {sp_a} vs {sp_b} ({len(shared)} shared TF names)")
    if len(shared) < 10:
        print("      SKIP  too few shared names to judge.")
        return None

    A = np.array([pooled_a[t] for t in shared])
    B = np.array([pooled_b[t] for t in shared])
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    B = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)

    cross = A @ B.T
    true_pairs = np.diag(cross)                       # same TF name across species
    off_diagonal = cross[~np.eye(len(shared), dtype=bool)]   # mismatched cross-species pairs

    scores = np.concatenate([true_pairs, off_diagonal])
    labels = np.concatenate([np.ones(len(true_pairs)), np.zeros(len(off_diagonal))])
    a = auroc(scores, labels)

    print(f"      median cosine, true ortholog pairs   {np.median(true_pairs):+.4f}")
    print(f"      median cosine, mismatched pairs      {np.median(off_diagonal):+.4f}")
    print(f"      AUROC separating them                {a:.4f}   (need >= {ORTHOLOG_AUROC_MIN})")

    if not np.isfinite(a) or a < ORTHOLOG_AUROC_MIN:
        print("      FAIL  orthologous proteins are not landing near each other.")
        return False

    print("      PASS")
    return True


# ---------------------------------------------------------------------------
# 3. DBD family
# ---------------------------------------------------------------------------

def load_tf_families(species):
    """{TF name (upper) -> CIS-BP Family_Name} for one species."""
    info_path = (
        PROJECT_DATA_DIR / "databases" / "motif_information" / species
        / "TF_Information_all_motifs.txt"
    )
    if not info_path.exists():
        return None

    info = pd.read_csv(info_path, sep="\t", header=0, low_memory=False)
    info["TF_Name"] = info["TF_Name"].astype(str).str.upper()
    info = info.dropna(subset=["Family_Name"])
    return info.drop_duplicates("TF_Name").set_index("TF_Name")["Family_Name"].to_dict()


def check_family(species, pooled):
    families = load_tf_families(species)
    print(f"\n[3] DBD FAMILY -- {species}")

    if families is None:
        print("      SKIP  no CIS-BP TF_Information_all_motifs.txt for this species.")
        return None

    names = sorted(set(pooled) & set(families))
    if len(names) < 20:
        print(f"      SKIP  only {len(names)} TFs have a family annotation.")
        return None

    fam = np.array([families[t] for t in names])
    cos = cosine_matrix([pooled[t] for t in names])
    iu = np.triu_indices(len(names), 1)

    same = fam[iu[0]] == fam[iu[1]]
    scores = cos[iu]
    a = auroc(scores, same)

    print(f"      {len(names)} TFs, {len(set(fam))} families, {same.sum():,} same-family pairs")
    print(f"      mean cosine  same-family {scores[same].mean():+.4f}  vs  different {scores[~same].mean():+.4f}")
    print(f"      AUROC                    {a:.4f}   (chance 0.5, need >= {FAMILY_AUROC_MIN})")

    if not np.isfinite(a) or a < FAMILY_AUROC_MIN:
        print("      FAIL  embeddings do not recover DBD family structure.")
        return False

    print("      PASS")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--embedding_dir",
        nargs="+",
        required=True,
        metavar="SPECIES=PATH",
        help="One or more species=path pairs, e.g. mm10=/.../mm10/tf_embeddings",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any check fails",
    )
    args = parser.parse_args()

    pooled_by_species, results = {}, []

    for entry in args.embedding_dir:
        if "=" not in entry:
            parser.error(f"--embedding_dir entries must be SPECIES=PATH, got: {entry}")
        species, path = entry.split("=", 1)

        pooled, top_directions = load_pooled_embeddings(path)
        pooled_by_species[species] = pooled
        results.append(check_basis(species, pooled, top_directions))

    results.append(check_orthologs(pooled_by_species))

    for species, pooled in pooled_by_species.items():
        results.append(check_family(species, pooled))

    failed = [r for r in results if r is False]
    print("\n" + "=" * 72)
    if failed:
        print(f"{len(failed)} check(s) FAILED -- do not rebuild the caches on these embeddings.")
    else:
        print("All checks passed.")
    print("=" * 72)

    if failed and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
