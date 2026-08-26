"""Score a trained TF-DNA model against another species' test edges.

This is the test the shared-PCA embedding basis was built to make possible: before it,
every TF was projected through its own random matrix, so a model trained on mouse TFs
had no way to interpret a human TF's vector. Now both species share one basis, and a
checkpoint can be pointed at the other species' cache without retraining.

Nothing about the architecture is species-specific. The TF embedding/mask tables are
registered as ``persistent=False`` buffers, so they are supplied at load time rather than
stored in the checkpoint, and both caches use 128-d embeddings over 256 bp peaks.

**Read the stratified numbers, not the pooled one.** Test-set positive rate depends
heavily on whether a TF is shared between the two species: for hg38 it is 33.6% on TFs
that also appear in mm10 versus 3.3% on TFs that do not, because well-studied TFs have
more ChIP-Atlas experiments behind them. A pooled AUROC over the whole test set therefore
rewards a model for telling well-studied TFs from obscure ones, which is not the question.
Two things fix that, and the script reports both:

  * split the edges by whether the TF is in the training species' vocabulary. Orthologs
    have embedding cosine ~0.99, so a shared TF is close to "same TF, new genome"; only
    the novel-TF column tests generalisation to unseen proteins.
  * macro per-TF AUROC, which ranks peaks within each TF separately and so is blind to
    differences in prevalence between TFs.

``--tf_embedding_mode shuffled`` is the control that decides whether any of it means
anything: it permutes which embedding each TF receives, leaving peaks untouched. If the
scores barely move, the model is reading the DNA alone and ignoring TF identity, and no
transfer result is interpretable.

Usage:
    python3 scripts/eval_tf_dna_cross_species.py \
        --model_ckpt checkpoints/tf_dna_mm10_3831017/last.ckpt \
        --model_species mm10 --eval_species hg38
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
sys.path.append(str(PROJECT_DIR))

import config  # noqa: E402
import utils  # noqa: E402
from scripts.train_tf_to_dna_model import TFPeakEdgeDataset  # noqa: E402

MIN_PER_TF = 25  # positives and negatives needed before a TF gets its own AUROC


def load_species_cache(species):
    cache_dir = config.tf_dna_cache_dir(species)
    names = pd.read_csv(cache_dir / "tf_name_to_idx.csv").sort_values("tf_idx")
    return dict(
        cache_dir=cache_dir,
        tf_embeddings=torch.load(cache_dir / "tf_embeddings.pt", map_location="cpu", weights_only=False),
        tf_masks=torch.load(cache_dir / "tf_masks.pt", map_location="cpu", weights_only=False),
        tf_names=names["tf_name"].str.upper().to_numpy(),
        edge_tf_idx=torch.load(cache_dir / "edge_tf_idx.pt", map_location="cpu", weights_only=False),
        edge_peak_idx=torch.load(cache_dir / "edge_peak_idx.pt", map_location="cpu", weights_only=False),
        edge_labels=torch.load(cache_dir / "edge_labels.pt", map_location="cpu", weights_only=False),
        test_idx=torch.load(cache_dir / "test_idx.pt", map_location="cpu", weights_only=False),
        peak_tensor=torch.load(cache_dir / "peak_onehot_array.pt", map_location="cpu", weights_only=False, mmap=True),
    )


def choose_eval_edges(cache, n_edges, seed):
    """Subsample test edges, keeping the label balance of the full split."""
    test_idx = cache["test_idx"]
    if n_edges is None or n_edges >= len(test_idx):
        return test_idx

    labels = cache["edge_labels"][test_idx]
    rng = np.random.default_rng(seed)
    keep = []
    for value in (1, 0):
        rows = np.flatnonzero((labels == value).numpy())
        share = int(round(n_edges * len(rows) / len(labels)))
        keep.append(rng.choice(rows, size=min(share, len(rows)), replace=False))
    return test_idx[np.sort(np.concatenate(keep))]


@torch.no_grad()
def score_edges(lit_model, cache, eval_idx, batch_size, device):
    """Binding probability per edge. Returns (scores, labels, tf_idx) aligned."""
    dataset = TFPeakEdgeDataset(
        cache["edge_tf_idx"], cache["edge_peak_idx"], cache["edge_labels"], cache["peak_tensor"]
    )

    # Group by protein length so each batch crops to a similar width -- the same trick
    # the training sampler uses, worth roughly a 5x speedup at eval too.
    lengths = cache["tf_masks"].sum(1)[cache["edge_tf_idx"][eval_idx]]
    order = torch.argsort(lengths)
    ordered_idx = eval_idx[order]

    loader = DataLoader(
        Subset(dataset, ordered_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    scores, labels, tf_rows = [], [], []
    started = time.time()
    for i, batch in enumerate(loader):
        tf_idx = batch["tf_idx"].long()
        # The lookup tables live on the model's device, so index them there.
        tf_idx_device = tf_idx.to(device)
        tf_mask = lit_model.tf_mask_tensor[tf_idx_device]
        crop = lit_model._crop_width(tf_mask)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = lit_model(
                tf_embedding=lit_model.tf_embeddings_tensor[:, :crop][tf_idx_device],
                tf_mask=tf_mask[:, :crop],
                peak_embedding=batch["peak_embedding"].float().to(device),
            )

        scores.append(torch.sigmoid(logits.float()).cpu())
        labels.append(batch["label"])
        tf_rows.append(tf_idx)

        if i % 200 == 0:
            done = (i + 1) * batch_size
            logging.info("  %d/%d edges (%.0f edges/s)", done, len(ordered_idx), done / (time.time() - started))

    return torch.cat(scores).numpy(), torch.cat(labels).numpy(), torch.cat(tf_rows).numpy()


def macro_tf_auroc(scores, labels, tf_idx, min_per_tf=MIN_PER_TF):
    """Mean of per-TF AUROCs -- ranks peaks within a TF, so TF prevalence cancels."""
    per_tf = []
    for tf in np.unique(tf_idx):
        rows = tf_idx == tf
        y = labels[rows]
        if y.sum() >= min_per_tf and (y == 0).sum() >= min_per_tf:
            per_tf.append(roc_auc_score(y, scores[rows]))
    return (float(np.mean(per_tf)) if per_tf else float("nan")), len(per_tf)


def report(scores, labels, tf_idx, tf_names, train_vocabulary, label):
    is_shared = np.isin(tf_names[tf_idx], list(train_vocabulary))

    logging.info("\n=== %s ===", label)
    header = f"  {'subset':16s} {'edges':>10s} {'pos%':>7s} {'AUROC':>8s} {'AUPRC':>8s} {'macro-TF':>9s} {'nTF':>5s}"
    logging.info(header)

    rows = []
    for name, mask in (("all", np.ones_like(is_shared)), ("shared TFs", is_shared), ("novel TFs", ~is_shared)):
        y, s, t = labels[mask], scores[mask], tf_idx[mask]
        if len(y) == 0 or y.sum() == 0 or (y == 0).sum() == 0:
            logging.info("  %-16s %10d  (not scoreable)", name, len(y))
            continue
        macro, n_tf = macro_tf_auroc(s, y, t)
        logging.info(
            "  %-16s %10d %6.1f%% %8.4f %8.4f %9.4f %5d",
            name, len(y), 100 * y.mean(), roc_auc_score(y, s), average_precision_score(y, s), macro, n_tf,
        )
        rows.append(dict(subset=name, edges=len(y), pos_rate=y.mean(),
                         auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s),
                         macro_tf_auroc=macro, n_tf=n_tf))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--model_species", required=True, choices=("mm10", "hg38"))
    parser.add_argument("--eval_species", required=True, choices=("mm10", "hg38"))
    parser.add_argument("--n_edges", type=int, default=200_000, help="test edges to score (0 = all)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tf_embedding_mode", default="real", choices=("real", "shuffled"))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", default=None, help="parquet of per-edge scores")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Model trained on %s, evaluating on %s (%s embeddings) on %s",
                 args.model_species, args.eval_species, args.tf_embedding_mode, device)

    cache = load_species_cache(args.eval_species)
    train_vocabulary = set(pd.read_csv(config.tf_dna_cache_dir(args.model_species) / "tf_name_to_idx.csv")["tf_name"].str.upper())

    tf_embeddings, tf_masks = cache["tf_embeddings"], cache["tf_masks"]
    if args.tf_embedding_mode == "shuffled":
        # Permute embedding AND mask together, so every TF gets some other TF's protein
        # intact rather than a chimera of one TF's residues under another's length.
        perm = torch.from_numpy(np.random.default_rng(args.seed).permutation(len(tf_embeddings)))
        tf_embeddings, tf_masks = tf_embeddings[perm], tf_masks[perm]

    lit_model = utils.load_tf_dna_model(
        tf_dna_model_path=Path(args.model_ckpt),
        tf_embeddings_tensor=tf_embeddings,
        tf_mask_tensor=tf_masks,
        compile_model=False,
        device=device,
    ).to(device).eval()

    eval_idx = choose_eval_edges(cache, args.n_edges or None, args.seed)
    logging.info("Scoring %d of %d test edges", len(eval_idx), len(cache["test_idx"]))

    scores, labels, tf_idx = score_edges(lit_model, cache, eval_idx, args.batch_size, device)

    label = f"{args.model_species} model -> {args.eval_species} test ({args.tf_embedding_mode} embeddings)"
    report(scores, labels, tf_idx, cache["tf_names"], train_vocabulary, label)

    if args.out:
        pd.DataFrame(
            dict(tf_name=cache["tf_names"][tf_idx], label=labels, score=scores,
                 shared_tf=np.isin(cache["tf_names"][tf_idx], list(train_vocabulary)))
        ).to_parquet(args.out, index=False)
        logging.info("\nPer-edge scores -> %s", args.out)


if __name__ == "__main__":
    main()
