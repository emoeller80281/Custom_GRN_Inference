#!/usr/bin/env python3
"""Per-TF share of True and False edges in the training split of a TF-DNA cache.

For each cache, the TFs are sorted by their True:False ratio in the training edges. The
10 TFs with the highest ratio, the 10 in the middle and the 10 with the lowest ratio are
drawn as stacked bars of the percent of each TF's training edges that are True and False.
Drawing the cache without per-TF balancing next to the cache with it shows how balancing
evens out the per-TF ratio.

Writes <output_dir>/tf_dna_true_false_per_tf_<cache>.png and a TSV with the counts of all TFs.

Usage (on a compute node):
  python plot_tf_dna_true_false_balance.py
  python plot_tf_dna_true_false_balance.py --caches tf_dna_cache_tf_balance_uniform_false_sampling
"""

import argparse
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import PercentFormatter

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
CACHE_ROOT = PROJECT_DIR / "cached_data" / "mm10"
OUTPUT_DIR = PROJECT_DIR / "new_plots" / "tf_dna_binding_evaluation"

# Cache folder -> chart title
CACHE_TITLES = {
    "tf_dna_cache_no_tf_balance_uniform_false_sampling": "Without Per-TF Balancing",
    "tf_dna_cache_tf_balance_uniform_false_sampling": "With Per-TF Balancing",
}

TRUE_COLOR = "#4195df"
FALSE_COLOR = "#bcbcbf"
GROUP_GAP = 1.5  # empty bar widths between the top, middle and bottom groups


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--caches", nargs="+", default=list(CACHE_TITLES),
                        help="Cache folder names in --cache_root (default: both uniform-sampling caches)")
    parser.add_argument("--cache_root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n_per_group", type=int, default=10,
                        help="TFs shown from the top, middle and bottom of the ratio ranking")
    return parser.parse_args()


def setup_font():
    font_path = PROJECT_DIR / "fonts" / "Arial.ttf"
    fm.fontManager.addfont(font_path)
    arial_font = fm.FontProperties(fname=font_path).get_name()
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [arial_font, "Helvetica", "Liberation Sans", "DejaVu Sans"]


def training_edge_counts(cache_dir):
    """True and False training edges per TF, sorted from the highest to the lowest True:False ratio."""
    edge_tf_idx = torch.load(cache_dir / "edge_tf_idx.pt", weights_only=True)
    edge_labels = torch.load(cache_dir / "edge_labels.pt", weights_only=True)
    train_idx = torch.load(cache_dir / "train_idx.pt", weights_only=True).long()

    tf_idx = edge_tf_idx[train_idx].long()
    is_true = edge_labels[train_idx] == 1
    n_tfs = int(edge_tf_idx.max()) + 1
    n_true = torch.bincount(tf_idx[is_true], minlength=n_tfs).numpy()
    n_false = torch.bincount(tf_idx[~is_true], minlength=n_tfs).numpy()

    tf_names = pd.read_csv(cache_dir / "tf_name_to_idx.csv").set_index("tf_idx")["tf_name"]
    counts = pd.DataFrame({
        "tf_name": tf_names.reindex(range(n_tfs)).to_numpy(),
        "n_true": n_true,
        "n_false": n_false,
    })
    counts = counts[counts["n_true"] + counts["n_false"] > 0].copy()

    n_total = counts["n_true"] + counts["n_false"]
    counts["pct_true"] = 100 * counts["n_true"] / n_total
    counts["pct_false"] = 100 * counts["n_false"] / n_total
    with np.errstate(divide="ignore"):
        counts["true_false_ratio"] = counts["n_true"] / counts["n_false"]

    return counts.sort_values(["true_false_ratio", "tf_name"], ascending=[False, True]).reset_index(drop=True)


def top_middle_bottom(counts, n):
    """The n TFs with the highest ratio, the n around the median rank and the n with the lowest."""
    middle_start = (len(counts) - n) // 2
    return {
        f"Highest {n}": counts.iloc[:n],
        f"Middle {n}": counts.iloc[middle_start:middle_start + n],
        f"Lowest {n}": counts.iloc[-n:],
    }


def plot_true_false_bars(counts, groups, title, cache_name):
    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")

    x_positions, tick_labels = [], []
    for g, (group_name, group) in enumerate(groups.items()):
        x = g * (len(group) + GROUP_GAP) + np.arange(len(group))
        ax.bar(x, group["pct_true"], width=0.8, color=TRUE_COLOR,
               label="True edges" if g == 0 else None)
        ax.bar(x, group["pct_false"], bottom=group["pct_true"], width=0.8, color=FALSE_COLOR,
               label="False edges" if g == 0 else None)
        ax.text(x.mean(), 102, group_name, ha="center", va="bottom", fontsize=13)
        x_positions.extend(x)
        tick_labels.extend(group["tf_name"])

    ax.axhline(50, color="0.25", linestyle=(0, (4, 3)), linewidth=1.2, label="1:1 (50%)")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=60, ha="right", rotation_mode="anchor", fontsize=11)
    ax.set_xlim(min(x_positions) - 0.7, max(x_positions) + 0.7)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("Percent of the TF's Training Edges", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    n_shown = sum(len(group) for group in groups.values())
    ax.set_title(
        f"True vs False TF-DNA Training Edges per TF: {title}\n"
        f"{n_shown} of {len(counts)} TFs, sorted by True:False ratio ({cache_name})",
        fontsize=15, pad=30,
    )
    # Matplotlib lists lines before bars; show True, False, then the 50% line
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(label) for label in ("True edges", "False edges", "1:1 (50%)")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=12)
    return fig


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    setup_font()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for cache_name in args.caches:
        cache_dir = args.cache_root / cache_name
        counts = training_edge_counts(cache_dir)
        groups = top_middle_bottom(counts, args.n_per_group)

        ratio = counts["true_false_ratio"]
        logging.info(
            f"{cache_name}: {len(counts)} TFs, {counts['n_true'].sum():,} True and "
            f"{counts['n_false'].sum():,} False training edges; True:False ratio median "
            f"{ratio.median():.2f}, range {ratio.min():.2f}-{ratio.max():.2f}; percent True "
            f"{counts['pct_true'].min():.1f}-{counts['pct_true'].max():.1f}%"
        )

        stem = f"tf_dna_true_false_per_tf_{cache_name.removeprefix('tf_dna_cache_')}"
        counts.to_csv(args.output_dir / f"{stem}.tsv", sep="\t", index=False)

        fig = plot_true_false_bars(counts, groups, CACHE_TITLES.get(cache_name, cache_name), cache_name)
        fig.savefig(args.output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        logging.info(f"Wrote {args.output_dir / stem}.png and .tsv")


if __name__ == "__main__":
    main()
