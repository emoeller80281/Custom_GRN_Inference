"""Interactive 3D Plotly landscape of TF-DNA binding predictions for inserted motifs.

The TF-DNA model takes a TF protein embedding and a one-hot DNA window, so it has no
numeric inputs to put on a grid. This script makes numeric controls by building synthetic
windows: random background DNA with a set GC fraction, plus copies of the TF's CIS-BP
consensus motif. Five controls make a 5D grid for each TF:

    copies       number of motif copies (0 = background only)
    mismatches   consensus bases replaced by the least likely base, in every copy
    position     center of the block of copies, in bp from the window center
    gc           GC fraction of the random background
    orientation  motif on the forward or reverse strand

Each grid point is the mean binding probability over --n-backgrounds random backgrounds.
The same backgrounds are used at every grid point and for every TF, so two surfaces
differ only by the motif. The backgrounds draw each base independently, so CpG and other
dinucleotide frequencies are not like genomic DNA.

The page reuses the slicing code of create_plotly_prediction_landscape.py: check two
controls to plot as a 3D surface, and set the other three with sliders. A dropdown picks
the TF. Two flat planes show the TF's mean probability on real bound and unbound peaks.

By default only the TFs whose motif clearly moves the model are predicted: one_copy_effect
(one consensus copy minus background) of at least --min-effect in the summary of an earlier
--tfs all run. --tfs picks other TFs, and --tfs all predicts every TF with a CIS-BP motif and
writes that summary (plotly_tf_dna_landscape_<species>_all_tfs_summary.tsv).

Run on a dense GPU node (about 4 s per TF on an A100; all mm10 TFs take about 25 min):
    srun --partition=dense --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 \
        python TETHER/scripts/model_evaluation/create_plotly_tf_dna_landscape.py --species mm10
"""
import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from plotly.offline import get_plotlyjs

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR))

import config
import utils
from create_plotly_prediction_landscape import JS_CORE

DEFAULT_CHECKPOINTS = {"mm10": config.mm10_tf_dna_path, "hg38": config.hg38_tf_dna_path}
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "new_plots" / "prediction_landscapes"
MOTIF_ROOT = config.DATA_DIR / "databases" / "motif_information"

# Axis order of the prediction grid: (name, display label, decimals, tick labels).
FEATURES = [
    ("copies", "Motif copies", 0, None),
    ("mismatches", "Mismatches per copy", 0, None),
    ("position", "Motif center (bp from window center)", 0, None),
    ("gc", "Background GC fraction", 2, None),
    ("orientation", "Motif strand", 0, ["forward", "reverse"]),
]
FEATURE_NAMES = [name for name, _, _, _ in FEATURES]
DEFAULT_PLOTTED = ["copies", "mismatches"]
DEFAULT_GC = 0.4            # near the mouse and human genome average
QUANT_SCALE = 65535         # probabilities are stored as uint16 to keep the page small

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--species", choices=sorted(DEFAULT_CHECKPOINTS), default="mm10")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Default: the species TF-DNA checkpoint in config.py.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: new_plots/prediction_landscapes/"
                             "plotly_tf_dna_landscape_<species>[_all_tfs].html")
    parser.add_argument("--tfs", nargs="+", default=None,
                        help="TF names to include, or 'all' for every TF with a CIS-BP motif. "
                             "Default: TFs with one_copy_effect >= --min-effect in the all-TF summary.")
    parser.add_argument("--min-effect", type=float, default=0.2,
                        help="Smallest one_copy_effect a TF needs to be in the default TF set.")
    parser.add_argument("--include-inferred-motifs", action="store_true",
                        help="Give TFs with no direct CIS-BP motif an inferred one (TF_Status 'I', "
                             "copied from a TF with a similar DNA-binding domain).")
    parser.add_argument("--n-backgrounds", type=int, default=8,
                        help="Random backgrounds averaged at each grid point.")
    parser.add_argument("--max-copies", type=int, default=4)
    parser.add_argument("--max-mismatches", type=int, default=4)
    parser.add_argument("--n-positions", type=int, default=17,
                        help="Position grid points. 17 gives 14 bp steps, which is not a multiple "
                             "of the peak encoder's 16 bp pooling stride.")
    parser.add_argument("--edge-margin", type=int, default=16,
                        help="Closest the motif center gets to either end of the window (bp).")
    parser.add_argument("--gc-values", type=float, nargs="+",
                        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--copy-gap", type=int, default=5, help="bp between motif copies.")
    parser.add_argument("--n-reference-peaks", type=int, default=200,
                        help="Real bound and unbound peaks per TF for the reference planes; 0 skips them.")
    parser.add_argument("--token-budget", type=int, default=1_000_000,
                        help="Batch size x TF length per forward pass. Lower it if the GPU runs out of memory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.max_copies < 1:
        parser.error("--max-copies must be at least 1")
    if args.n_positions < 2:
        parser.error("--n-positions must be at least 2")
    if not all(0.0 <= gc <= 1.0 for gc in args.gc_values):
        parser.error("--gc-values must be between 0 and 1")
    default_output = DEFAULT_OUTPUT_DIR / f"plotly_tf_dna_landscape_{args.species}.html"
    all_tfs_output = default_output.with_name(default_output.stem + "_all_tfs.html")
    if args.tfs is None:
        summary_path = all_tfs_output.with_name(all_tfs_output.stem + "_summary.tsv")
        if not summary_path.exists():
            parser.error(f"{summary_path} not found. Run with --tfs all first, or pass --tfs.")
        summary = pd.read_csv(summary_path, sep="\t")
        args.tfs = summary.loc[summary["one_copy_effect"] >= args.min_effect, "tf_name"].tolist()
        if not args.tfs:
            parser.error(f"No TF in {summary_path.name} has one_copy_effect >= {args.min_effect}")
        logging.info(f"{len(args.tfs)} TFs with one_copy_effect >= {args.min_effect} in {summary_path.name}")
    elif [name.lower() for name in args.tfs] == ["all"]:
        args.tfs = None
        default_output = all_tfs_output
    args.checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[args.species]
    args.output = args.output or default_output
    return args


# ----------------------------------------------------------------------------- motifs

def information_content(pwm):
    """Per-column information content in bits (0 = uniform, 2 = a fixed base)."""
    p = np.clip(pwm, 1e-9, None)
    return 2.0 + (p * np.log2(p)).sum(axis=1)


def trim_pwm(pwm, min_ic):
    """Drop uninformative leading and trailing columns."""
    keep = np.flatnonzero(information_content(pwm) >= min_ic)
    return pwm if len(keep) == 0 else pwm[keep[0]:keep[-1] + 1]


def load_cisbp_motifs(motif_dir, tf_names, include_inferred, min_ic=0.25, min_len=4):
    """One PWM per TF: the motif with the highest total information content after trimming.

    Same selection as load_cisbp_motifs in analyze_tf_embeddings.ipynb. Direct motifs
    (TF_Status "D") come first; inferred motifs only fill TFs that have no direct one.
    Returns {upper-case TF name: dict(pwm, motif_id, motif_type, family, status)}.
    """
    info = pd.read_csv(motif_dir / "TF_Information_all_motifs.txt", sep="\t", low_memory=False)
    info["TF_Name"] = info["TF_Name"].str.upper()
    info = info[info["Motif_ID"].notna() & (info["Motif_ID"] != ".")]
    info = info[info["TF_Name"].isin({name.upper() for name in tf_names})]

    motifs = {}
    for status in ("D", "I") if include_inferred else ("D",):
        for tf_name, rows in info[info["TF_Status"] == status].groupby("TF_Name"):
            if tf_name in motifs:
                continue
            best = None
            for row in rows.drop_duplicates("Motif_ID").itertuples():
                path = motif_dir / "pwms_all_motifs" / f"{row.Motif_ID}.txt"
                if not path.exists():
                    continue
                table = pd.read_csv(path, sep="\t", index_col=0)
                if table.columns.tolist() != list("ACGT") or len(table) < min_len:
                    continue
                pwm = table.to_numpy(dtype=float)
                pwm = pwm / np.clip(pwm.sum(axis=1, keepdims=True), 1e-9, None)
                pwm = trim_pwm(pwm, min_ic)
                total_ic = information_content(pwm).sum()
                if len(pwm) >= min_len and (best is None or total_ic > best["total_ic"]):
                    best = {
                        "pwm": pwm, "total_ic": total_ic, "motif_id": row.Motif_ID,
                        "motif_type": str(row.Motif_Type), "family": str(row.Family_Name),
                        "status": status,
                    }
            if best is not None:
                motifs[tf_name] = best
    return motifs


# ----------------------------------------------------------------------------- sequences

def make_grids(args, window):
    max_offset = window // 2 - args.edge_margin
    return {
        "copies": np.arange(args.max_copies + 1),
        "mismatches": np.arange(args.max_mismatches + 1),
        "position": np.rint(np.linspace(-max_offset, max_offset, args.n_positions)).astype(int),
        "gc": np.asarray(sorted(args.gc_values), dtype=float),
        "orientation": np.array([0, 1]),
    }


def make_backgrounds(gc_values, n_backgrounds, window, rng):
    """Random background bases [n_gc, n_backgrounds, window], one-hot order A, C, G, T = 0-3."""
    backgrounds = np.empty((len(gc_values), n_backgrounds, window), dtype=np.uint8)
    for gi, gc in enumerate(gc_values):
        p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
        backgrounds[gi] = rng.choice(4, size=(n_backgrounds, window), p=p)
    return backgrounds


def motif_starts(n_copies, center, motif_len, gap, window):
    """Start of each copy in a block of copies centered `center` bp from the window center.

    The block moves inward when it would run past either end of the window.
    """
    if n_copies == 0:
        return []
    block_len = n_copies * motif_len + (n_copies - 1) * gap
    start = int(round(window / 2 + center - block_len / 2))
    start = min(max(start, 0), window - block_len)
    return [start + k * (motif_len + gap) for k in range(n_copies)]


def build_sequences(pwm, backgrounds, grids, gap, rng):
    """Base indices of every synthetic window for one TF.

    Returns uint8 [copies, mismatches, position, gc, orientation, n_backgrounds, window].
    """
    n_gc, n_bg, window = backgrounds.shape
    consensus = pwm.argmax(axis=1)
    worst = pwm.argmin(axis=1)
    motif_len = len(consensus)

    # Motif variants [mismatches, orientation, background, motif_len]. Each background gets
    # its own random order of columns to mutate, so the mean over backgrounds also averages
    # over which columns carry the mismatches.
    mutate_order = np.argsort(rng.random((n_bg, motif_len)), axis=1)
    rows = np.arange(n_bg)[:, None]
    variants = np.empty((len(grids["mismatches"]), 2, n_bg, motif_len), dtype=np.uint8)
    for mi, n_mismatches in enumerate(grids["mismatches"]):
        forward = np.tile(consensus, (n_bg, 1))
        cols = mutate_order[:, :min(n_mismatches, motif_len)]
        forward[rows, cols] = worst[cols]
        variants[mi, 0] = forward
        variants[mi, 1] = 3 - forward[:, ::-1]      # reverse complement

    shape = (len(grids["copies"]), len(grids["mismatches"]), len(grids["position"]),
             n_gc, len(grids["orientation"]), n_bg, window)
    seqs = np.empty(shape, dtype=np.uint8)
    seqs[...] = backgrounds[None, None, None, :, None, :, :]
    for ci, n_copies in enumerate(grids["copies"]):
        for pi, center in enumerate(grids["position"]):
            for start in motif_starts(n_copies, center, motif_len, gap, window):
                # [mismatches, orientation, bg, L] -> [mismatches, gc, orientation, bg, L]
                seqs[ci, :, pi, :, :, :, start:start + motif_len] = variants[:, None]
    return seqs


# ----------------------------------------------------------------------------- model

def binding_probs(model, tf_emb, peaks, device, token_budget):
    """Binding probability of one TF for every peak.

    tf_emb: [tf_len, dim] with the padding removed.
    peaks:  [N, window] base indices (A, C, G, T = 0-3) or [N, window, 4] one-hot.
    """
    tf_len = tf_emb.shape[0]
    chunk = int(np.clip(token_budget // tf_len, 16, 8192))
    tf_emb = tf_emb.to(device)[None]
    tf_mask = torch.ones(1, tf_len, dtype=torch.bool, device=device)
    probs = np.empty(len(peaks), dtype=np.float32)

    # Training used 16-bit mixed precision, so float16 autocast matches it on the GPU.
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16,
                                                enabled=device.type == "cuda"):
        for start in range(0, len(peaks), chunk):
            x = torch.as_tensor(peaks[start:start + chunk]).to(device)
            if x.ndim == 2:
                x = torch.nn.functional.one_hot(x.long(), 4)
            n = x.shape[0]
            logits = model(
                tf_embedding=tf_emb.expand(n, -1, -1),
                tf_mask=tf_mask.expand(n, -1),
                peak_embedding=x.float(),
            )
            probs[start:start + n] = logits.float().sigmoid().cpu().numpy()
    return probs


def sample_reference_peaks(cache_dir, tf_indices, n_peaks, rng):
    """Up to n_peaks bound and n_peaks unbound peak rows per TF index.

    Sampled from all labeled TF-peak edges. The split files were rebuilt after the
    checkpoints were trained, so no split is safe from training peaks.
    """
    edge_tf = torch.load(cache_dir / config.tf_dna_edge_tf_idx_cache_path.name, weights_only=True).numpy()
    edge_peak = torch.load(cache_dir / config.tf_dna_edge_peak_idx_cache_path.name, weights_only=True).numpy()
    edge_bound = torch.load(cache_dir / config.tf_dna_edge_labels_cache_path.name, weights_only=True).numpy() > 0.5

    order = np.argsort(edge_tf, kind="stable")
    sorted_tf = edge_tf[order]
    reference = {}
    for tf_idx in tf_indices:
        lo, hi = np.searchsorted(sorted_tf, [tf_idx, tf_idx + 1])
        rows = order[lo:hi]
        picks = {}
        for key, rows_of_label in (("bound", rows[edge_bound[rows]]), ("unbound", rows[~edge_bound[rows]])):
            if len(rows_of_label) > n_peaks:
                rows_of_label = rng.choice(rows_of_label, size=n_peaks, replace=False)
            picks[key] = np.sort(edge_peak[rows_of_label])  # sorted for mmap locality
        reference[tf_idx] = picks
    return reference


def compute_landscapes(args):
    """Predict the 5D grid for every TF. Returns (results, grids, window)."""
    device = torch.device(args.device)
    cache_dir = config.tf_dna_cache_dir(args.species)
    rng = np.random.default_rng(args.seed)

    tf_name_to_idx = (pd.read_csv(cache_dir / config.tf_name_to_idx_cache_path.name)
                      .set_index("tf_name")["tf_idx"].to_dict())
    # Memory-mapped: only the rows of the selected TFs and peaks are read from disk.
    tf_embeddings = torch.load(cache_dir / config.tf_embedding_cache_path.name, mmap=True, weights_only=True)
    tf_masks = torch.load(cache_dir / config.tf_mask_cache_path.name, mmap=True, weights_only=True)
    peak_onehot = torch.load(cache_dir / config.tf_dna_peak_onehot_cache_path.name, mmap=True, weights_only=True)
    window = peak_onehot.shape[1]

    tf_names = list(tf_name_to_idx)
    if args.tfs is not None:
        wanted = {name.upper() for name in args.tfs}
        tf_names = [name for name in tf_names if name.upper() in wanted]
        missing = wanted - {name.upper() for name in tf_names}
        if missing:
            logging.warning(f"Not in the {args.species} TF table: {sorted(missing)}")

    motifs = load_cisbp_motifs(MOTIF_ROOT / args.species, tf_names, args.include_inferred_motifs)
    grids = make_grids(args, window)
    max_block = lambda motif_len: args.max_copies * motif_len + (args.max_copies - 1) * args.copy_gap
    tfs = []
    for name in sorted(tf_names, key=str.upper):
        motif = motifs.get(name.upper())
        if motif is None:
            continue
        if max_block(len(motif["pwm"])) > window:
            logging.warning(f"Skipping {name}: {args.max_copies} copies of its "
                            f"{len(motif['pwm'])} bp motif do not fit in {window} bp")
            continue
        tfs.append((name, tf_name_to_idx[name], motif))
    logging.info(f"{len(tfs)} of {len(tf_names)} TFs have a usable CIS-BP motif")
    if not tfs:
        raise SystemExit("No TFs to plot.")

    backgrounds = make_backgrounds(grids["gc"], args.n_backgrounds, window, rng)
    reference = {}
    if args.n_reference_peaks > 0:
        logging.info("Sampling real bound and unbound peaks")
        reference = sample_reference_peaks(cache_dir, [idx for _, idx, _ in tfs], args.n_reference_peaks, rng)

    logging.info(f"Loading checkpoint {args.checkpoint}")
    # The Lightning wrapper only uses its TF tables in _shared_step. The inner model is called
    # directly here, so placeholders keep the 1+ GB tables off the GPU.
    lit_model = utils.load_tf_dna_model(
        args.checkpoint,
        tf_embeddings_tensor=torch.zeros(1, 1, tf_embeddings.shape[-1]),
        tf_mask_tensor=torch.zeros(1, 1, dtype=torch.bool),
        device=device,
    )
    model = lit_model.model.eval().to(device)

    n_points = int(np.prod([len(grids[name]) for name in FEATURE_NAMES]))
    logging.info(f"Predicting {n_points:,} grid points x {args.n_backgrounds} backgrounds "
                 f"for {len(tfs)} TFs on {device}")
    results = []
    start_time = time.perf_counter()
    for k, (name, tf_idx, motif) in enumerate(tfs):
        tf_emb = tf_embeddings[tf_idx][tf_masks[tf_idx].bool()].float()
        tf_rng = np.random.default_rng([args.seed, tf_idx])
        seqs = build_sequences(motif["pwm"], backgrounds, grids, args.copy_gap, tf_rng)

        # copies = 0 is the bare background at every mismatch, position and strand, so it
        # is predicted once per background and broadcast.
        probs = np.empty(seqs.shape[:-1], dtype=np.float32)
        bare = binding_probs(model, tf_emb, backgrounds.reshape(-1, window), device, args.token_budget)
        probs[0] = bare.reshape(backgrounds.shape[:2])[None, None, :, None, :]
        with_motif = seqs[1:]
        probs[1:] = binding_probs(model, tf_emb, with_motif.reshape(-1, window), device,
                                  args.token_budget).reshape(with_motif.shape[:-1])
        grid = probs.mean(axis=-1)    # mean over backgrounds

        ref = {}
        for key, peak_rows in reference.get(tf_idx, {}).items():
            if len(peak_rows):
                rows = torch.as_tensor(peak_rows, dtype=torch.long)
                ref[key] = float(binding_probs(model, tf_emb, peak_onehot[rows], device, args.token_budget).mean())
            ref[f"n_{key}"] = len(peak_rows)

        results.append({"name": name, "motif": motif, "grid": grid, "ref": ref})
        if (k + 1) % 25 == 0 or k + 1 == len(tfs):
            logging.info(f"  {k + 1}/{len(tfs)} TFs, {time.perf_counter() - start_time:.0f} s")
    return results, grids, window


# ----------------------------------------------------------------------------- summary

def default_indices(grids):
    """Starting slider positions: one consensus copy at the center, forward, GC near DEFAULT_GC."""
    return {
        "copies": 1,
        "mismatches": 0,
        "position": int(np.abs(grids["position"]).argmin()),
        "gc": int(np.abs(grids["gc"] - DEFAULT_GC).argmin()),
        "orientation": 0,
    }


def summarize(results, grids):
    """One row per TF: background, one-copy and all-copies probability at the default GC."""
    gi = default_indices(grids)["gc"]
    rows = []
    for result in results:
        grid, motif, ref = result["grid"], result["motif"], result["ref"]
        background = float(grid[0, 0, 0, gi, 0])        # copies = 0 is the same at every position
        one_copy = float(grid[1, 0, :, gi, :].mean())   # mean over positions and strands
        all_copies = float(grid[-1, 0, :, gi, :].mean())
        rows.append({
            "tf_name": result["name"],
            "family": motif["family"],
            "motif_id": motif["motif_id"],
            "motif_status": motif["status"],
            "motif_len": len(motif["pwm"]),
            "consensus": "".join("ACGT"[i] for i in motif["pwm"].argmax(axis=1)),
            "p_background": background,
            "p_one_copy": one_copy,
            "p_max_copies": all_copies,
            "one_copy_effect": one_copy - background,
            "ref_bound_mean": ref.get("bound", np.nan),
            "ref_unbound_mean": ref.get("unbound", np.nan),
            "n_bound": ref.get("n_bound", 0),
            "n_unbound": ref.get("n_unbound", 0),
        })
    return pd.DataFrame(rows).sort_values("one_copy_effect", ascending=False, ignore_index=True)


# ----------------------------------------------------------------------------- page

# Page code for this landscape. sliceGrid, updateOrder and gridStats come from JS_CORE.
JS_TF_DNA_CORE = r"""
function decodeUint16(b64, scale) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const q = new Uint16Array(bytes.buffer);
    const out = new Float32Array(q.length);
    for (let i = 0; i < q.length; i++) out[i] = q[i] / scale;
    return out;
}
"""

JS_UI = r"""
const F = CONFIG.features;
const TFS = CONFIG.tfs;
const nf = F.length;
const shape = CONFIG.shape;
const strides = CONFIG.strides;
// Strides that ignore the copies axis, so every lookup lands on copies = 0 (grid index 0).
const noMotifStrides = strides.map((s, k) => (k === CONFIG.copies_axis ? 0 : s));
const sliderIdx = CONFIG.default_idx.slice();
const startIdx = CONFIG.default_idx.slice();
let order = CONFIG.default_plotted.slice();
let currentTf = CONFIG.default_tf;
const predCache = new Map();

const plotDiv = document.getElementById("plot");
const statusDiv = document.getElementById("status");
const tfInfoDiv = document.getElementById("tf-info");
const tfFilter = document.getElementById("tf-filter");
const tfSelect = document.getElementById("tf-select");
const checksDiv = document.getElementById("feature-checks");
const slidersDiv = document.getElementById("sliders");
const swapBox = document.getElementById("swap");
const contourBox = document.getElementById("contours");
const refBox = document.getElementById("ref-planes");
const colorSelect = document.getElementById("color-mode");

const fmtP = v => Number(v).toFixed(3);
const signed = v => (v >= 0 ? "+" : "") + v.toFixed(3);
const fmtValue = (k, i) => (F[k].ticks ? F[k].ticks[i] : Number(F[k].grid[i]).toFixed(F[k].decimals));
const refText = (v, n) => (v === null ? "none" : fmtP(v) + " (n = " + n + ")");

function tfPreds(t) {
    if (!predCache.has(t)) predCache.set(t, decodeUint16(PRED_B64[t], CONFIG.quant_scale));
    return predCache.get(t);
}

// TF dropdown, sorted by name. The filter matches the TF name or the motif family.
const tfOrder = TFS.map((tf, t) => t).sort((a, b) => TFS[a].name.localeCompare(TFS[b].name));
function fillTfOptions() {
    const query = tfFilter.value.trim().toUpperCase();
    tfSelect.innerHTML = "";
    for (const t of tfOrder) {
        const tf = TFS[t];
        if (query && !tf.name.toUpperCase().includes(query) && !tf.family.toUpperCase().includes(query)) continue;
        const opt = document.createElement("option");
        opt.value = t;
        opt.textContent = tf.name + " (" + tf.family + ")";
        tfSelect.append(opt);
    }
    if ([...tfSelect.options].some(o => o.value === String(currentTf))) tfSelect.value = String(currentTf);
}
fillTfOptions();
tfFilter.addEventListener("input", () => {
    fillTfOptions();
    if (tfSelect.options.length && tfSelect.value !== String(currentTf)) {
        currentTf = Number(tfSelect.value);
        render();
    }
});
tfSelect.addEventListener("change", () => { currentTf = Number(tfSelect.value); render(); });

const checkboxes = F.map((f, k) => {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = order.includes(k);
    box.addEventListener("change", () => {
        const checked = checkboxes.map((b, i) => (b.checked ? i : -1)).filter(i => i >= 0);
        order = updateOrder(order, checked);
        checkboxes.forEach((b, i) => { b.checked = order.includes(i); });
        render();
    });
    label.append(box, " " + f.label);
    checksDiv.append(label);
    return box;
});

const sliders = F.map((f, k) => {
    const wrap = document.createElement("div");
    wrap.className = "slider";
    const head = document.createElement("div");
    head.className = "slider-head";
    const name = document.createElement("span");
    name.textContent = f.label;
    const value = document.createElement("span");
    value.className = "slider-value";
    value.textContent = fmtValue(k, sliderIdx[k]);
    head.append(name, value);
    const input = document.createElement("input");
    input.type = "range";
    input.min = 0;
    input.max = f.grid.length - 1;
    input.step = 1;
    input.value = sliderIdx[k];
    input.addEventListener("input", () => {
        sliderIdx[k] = Number(input.value);
        value.textContent = fmtValue(k, sliderIdx[k]);
        render();
    });
    wrap.append(head, input);
    slidersDiv.append(wrap);
    return {wrap: wrap, input: input};
});

[swapBox, contourBox, refBox, colorSelect].forEach(el => el.addEventListener("change", render));

function tfSummary(tf) {
    let html = "<b>" + tf.name + "</b> &middot; " + tf.family + "<br>" +
        "Motif " + tf.motif_id + " (" + tf.motif_type + (tf.status === "I" ? ", inferred" : "") +
        "), " + tf.consensus.length + " bp<br><code>" + tf.consensus + "</code>";
    if (tf.ref_bound !== null || tf.ref_unbound !== null) {
        html += "<br>Mean probability on real peaks: bound " + refText(tf.ref_bound, tf.n_bound) +
            ", unbound " + refText(tf.ref_unbound, tf.n_unbound);
    }
    return html;
}

// Flat plane at a fixed probability, spanning the plotted x and y range.
function levelPlane(level, name, color, gx, gy) {
    return {
        type: "surface",
        x: [gx[0], gx[gx.length - 1]],
        y: [gy[0], gy[gy.length - 1]],
        z: [[level, level], [level, level]],
        colorscale: [[0, color], [1, color]],
        showscale: false,
        opacity: 0.3,
        hovertemplate: name + ": " + fmtP(level) + "<extra></extra>",
    };
}

function axisLayout(k, grid) {
    const axis = {title: {text: F[k].label}, range: [grid[0], grid[grid.length - 1]]};
    if (F[k].ticks) {
        axis.tickvals = grid;
        axis.ticktext = F[k].ticks;
    }
    return axis;
}

function render() {
    const tf = TFS[currentTf];
    const preds = tfPreds(currentTf);
    tfInfoDiv.innerHTML = tfSummary(tf);

    if (order.length < 2) {
        sliders.forEach(s => { s.input.disabled = false; s.wrap.classList.remove("plotted"); });
        statusDiv.innerHTML = "<b>Check two controls to plot.</b>";
        return;
    }
    let [xf, yf] = order.slice().sort((a, b) => a - b);
    if (swapBox.checked) [xf, yf] = [yf, xf];
    sliders.forEach((s, k) => {
        const plotted = k === xf || k === yf;
        s.input.disabled = plotted;
        s.wrap.classList.toggle("plotted", plotted);
    });

    const z = sliceGrid(preds, shape, strides, sliderIdx, xf, yf);
    const stats = gridStats(z);
    const gx = F[xf].grid;
    const gy = F[yf].grid;

    const mode = colorSelect.value;
    let color = z, cmin = 0, cmax = 1, colorscale = "Viridis", reversescale = false;
    let colorTitle = "Binding<br>probability";
    let diff = null, diffLabel = "";
    if (mode === "fit" && stats.max > stats.min) {
        cmin = stats.min;
        cmax = stats.max;
    } else if (mode === "effect" || mode === "change") {
        const ref = mode === "effect"
            ? sliceGrid(preds, shape, noMotifStrides, sliderIdx, xf, yf)
            : sliceGrid(preds, shape, strides, startIdx, xf, yf);
        diff = z.map((row, j) => row.map((v, i) => v - ref[j][i]));
        let maxAbs = 0.01;
        for (const row of diff) for (const v of row) maxAbs = Math.max(maxAbs, Math.abs(v));
        color = diff;
        cmin = -maxAbs;
        cmax = maxAbs;
        colorscale = "RdBu";
        reversescale = true;          // red = higher than the reference
        colorTitle = mode === "effect" ? "Motif effect<br>(vs 0 copies)" : "Change from<br>start values";
        diffLabel = mode === "effect" ? "Motif effect" : "Change";
    }

    const hover = z.map((row, j) => row.map((v, i) =>
        F[xf].label + ": " + fmtValue(xf, i) + "<br>" +
        F[yf].label + ": " + fmtValue(yf, j) + "<br>" +
        "Binding probability: " + fmtP(v) +
        (diff ? "<br>" + diffLabel + ": " + signed(diff[j][i]) : "")));

    // No hovertemplate: on 3D surfaces Plotly fills %{text} from text[x][y], the transpose of
    // z[y][x], which mirrored the labels and showed "-" where the slice is not square. Plain
    // hovertext is read as hovertext[y][x].
    const traces = [{
        type: "surface",
        x: gx,
        y: gy,
        z: z,
        surfacecolor: color,
        cmin: cmin,
        cmax: cmax,
        colorscale: colorscale,
        reversescale: reversescale,
        colorbar: {title: {text: colorTitle}},
        contours: {z: {show: contourBox.checked, usecolormap: true, project: {z: true}}},
        hovertext: hover,
        hoverinfo: "text",
    }];
    if (refBox.checked) {
        if (tf.ref_bound !== null) traces.push(levelPlane(tf.ref_bound, "Real bound peaks", "#d62728", gx, gy));
        if (tf.ref_unbound !== null) traces.push(levelPlane(tf.ref_unbound, "Real unbound peaks", "#7f7f7f", gx, gy));
    }

    const layout = {
        uirevision: "keep",           // keep the camera angle between redraws
        title: {text: tf.name + ": " + F[yf].label + " vs " + F[xf].label},
        margin: {l: 0, r: 0, t: 50, b: 0},
        scene: {
            xaxis: axisLayout(xf, gx),
            yaxis: axisLayout(yf, gy),
            zaxis: {title: {text: "Binding probability"}, range: [0, 1]},
            aspectmode: "manual",
            aspectratio: {x: 1.2, y: 1.0, z: 0.7},
        },
    };
    Plotly.react(plotDiv, traces, layout, {responsive: true});

    const fixed = [];
    for (let k = 0; k < nf; k++) {
        if (k !== xf && k !== yf) fixed.push(F[k].label + " = " + fmtValue(k, sliderIdx[k]));
    }
    statusDiv.innerHTML =
        "<b>Fixed:</b> " + fixed.join(" &nbsp;|&nbsp; ") +
        "<br><b>This slice:</b> min " + fmtP(stats.min) +
        ", mean " + fmtP(stats.mean) + ", max " + fmtP(stats.max);
}

render();
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TF-DNA Binding Landscape</title>
<style>
  body { margin: 0; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         color: #222; background: #fff; }
  #app { display: flex; gap: 16px; padding: 16px; box-sizing: border-box; }
  #controls { flex: 0 0 320px; }
  #controls h2 { font-size: 18px; margin: 0 0 12px; }
  #controls h3 { font-size: 14px; margin: 16px 0 6px; }
  #controls label { display: block; margin: 3px 0; font-size: 14px; }
  #controls select, #controls input[type=search] { width: 100%; font-size: 14px; box-sizing: border-box; }
  #tf-select { margin-top: 4px; }
  #tf-info { font-size: 13px; margin-top: 8px; line-height: 1.4; }
  #tf-info code { word-break: break-all; }
  .slider { margin: 8px 0; }
  .slider-head { display: flex; justify-content: space-between; font-size: 13px; }
  .slider-value { font-variant-numeric: tabular-nums; font-weight: 600; }
  .slider input { width: 100%; }
  .slider.plotted { opacity: 0.4; }
  #main { flex: 1 1 auto; min-width: 0; }
  #plot { width: 100%; height: 78vh; }
  #status { font-size: 14px; margin-top: 8px; }
  #notes { font-size: 12px; color: #555; margin-top: 16px; line-height: 1.4; }
</style>
<script>__PLOTLY_JS__</script>
</head>
<body>
<div id="app">
  <div id="controls">
    <h2>TF-DNA binding landscape</h2>
    <h3>TF</h3>
    <input type="search" id="tf-filter" placeholder="Filter by TF or family">
    <select id="tf-select"></select>
    <div id="tf-info"></div>
    <h3>Plotted controls <small>(check two)</small></h3>
    <div id="feature-checks"></div>
    <label><input type="checkbox" id="swap"> Swap axes</label>
    <h3>Fixed controls</h3>
    <div id="sliders"></div>
    <h3>Surface color</h3>
    <select id="color-mode">
      <option value="fixed">Binding probability (0–1)</option>
      <option value="fit">Binding probability (fit to slice)</option>
      <option value="effect">Motif effect (minus 0 copies)</option>
      <option value="change">Change from starting slider values</option>
    </select>
    <label><input type="checkbox" id="contours"> Contour lines</label>
    <label><input type="checkbox" id="ref-planes" checked> Real-peak levels (red = bound, grey = unbound)</label>
    <div id="notes">__NOTES__</div>
  </div>
  <div id="main">
    <div id="plot"></div>
    <div id="status"></div>
  </div>
</div>
<script>
const CONFIG = __CONFIG__;
const PRED_B64 = __PRED_B64__;
</script>
<script id="landscape-core">__JS_CORE__</script>
<script>__JS_TF_DNA_CORE__</script>
<script>__JS_UI__</script>
</body>
</html>
"""


def build_page(results, grids, window, summary, args):
    shape = list(results[0]["grid"].shape)
    defaults = default_indices(grids)
    features = [
        {"name": name, "label": label, "decimals": decimals, "ticks": ticks,
         "grid": [round(float(v), 6) for v in grids[name]]}
        for name, label, decimals, ticks in FEATURES
    ]

    def ref_value(ref, key):
        return round(ref[key], 6) if key in ref else None

    tfs, pred_b64 = [], []
    for result in results:
        motif, ref = result["motif"], result["ref"]
        tfs.append({
            "name": result["name"],
            "family": motif["family"],
            "motif_id": motif["motif_id"],
            "motif_type": motif["motif_type"],
            "status": motif["status"],
            "consensus": "".join("ACGT"[i] for i in motif["pwm"].argmax(axis=1)),
            "ref_bound": ref_value(ref, "bound"),
            "ref_unbound": ref_value(ref, "unbound"),
            "n_bound": ref.get("n_bound", 0),
            "n_unbound": ref.get("n_unbound", 0),
        })
        quantized = np.rint(result["grid"] * QUANT_SCALE).astype("<u2")
        pred_b64.append(base64.b64encode(quantized.ravel().tobytes()).decode("ascii"))

    # Open on the TF whose single consensus copy moves the prediction most.
    names = [tf["name"] for tf in tfs]
    config_json = {
        "features": features,
        "shape": shape,
        "strides": [int(np.prod(shape[k + 1:])) for k in range(len(shape))],
        "default_idx": [defaults[name] for name in FEATURE_NAMES],
        "default_plotted": [FEATURE_NAMES.index(name) for name in DEFAULT_PLOTTED],
        "copies_axis": FEATURE_NAMES.index("copies"),
        "quant_scale": QUANT_SCALE,
        "tfs": tfs,
        "default_tf": names.index(summary["tf_name"].iloc[0]),
    }

    notes = (
        f"<b>Windows:</b> {window} bp of random background (bases drawn independently at the set GC), "
        f"plus copies of the TF's CIS-BP consensus, {args.copy_gap} bp apart. Each point is the mean "
        f"over {args.n_backgrounds} backgrounds, the same for every point and every TF.<br>"
        "<b>Mismatches:</b> that many consensus columns, chosen at random per background, take the "
        "least likely base, in every copy.<br>"
        "<b>Position:</b> center of the block of copies. A block that would run past the window "
        "end moves inward.<br>"
        "<b>Real-peak levels:</b> mean probability on sampled ChIP-Atlas bound and unbound peaks "
        "of this TF. Some were training peaks.<br>"
        f"<b>Checkpoint:</b> {args.checkpoint.name}"
    )

    # Plotly's library goes in last, so the other placeholders are never searched for inside it.
    replacements = {
        "__NOTES__": notes,
        "__CONFIG__": json.dumps(config_json),
        "__PRED_B64__": json.dumps(pred_b64),
        "__JS_CORE__": JS_CORE,
        "__JS_TF_DNA_CORE__": JS_TF_DNA_CORE,
        "__JS_UI__": JS_UI,
        "__PLOTLY_JS__": get_plotlyjs(),
    }
    html = PAGE
    for key, value in replacements.items():
        html = html.replace(key, value)
    return html


def main():
    args = parse_args()
    results, grids, window = compute_landscapes(args)
    summary = summarize(results, grids)

    shown = ["tf_name", "family", "p_background", "p_one_copy", "one_copy_effect", "ref_bound_mean"]
    logging.info("Largest one-copy motif effects:\n" + summary[shown].head(10).to_string(index=False))
    if len(summary) > 10:
        logging.info("Smallest one-copy motif effects:\n" + summary[shown].tail(10).to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_page(results, grids, window, summary, args))
    logging.info(f"Saved {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")

    summary_path = args.output.with_name(args.output.stem + "_summary.tsv")
    summary.to_csv(summary_path, sep="\t", index=False, float_format="%.4f")
    logging.info(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
