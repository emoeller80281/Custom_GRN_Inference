"""Interactive 3D Plotly landscape of TF-TG edge predictions over the five model inputs.

The script runs the trained model on a 5D grid of
(binding_score, peak_accessibility, peak_distance, tf_expression, tg_expression).
The page shows a 3D surface for two checked features (height = edge probability);
sliders set the other three. Plotly's built-in sliders cannot combine several
independent sliders, so the controls are plain HTML and a short script slices the
embedded 5D array and redraws the surface with Plotly.react.

Every grid point sets all five inputs for every cell and every peak of an edge.
All peak tokens are then identical, so attention returns the same context for
any number of peaks, and every cell gets the same logit. The pooled edge logit
T * (logsumexp(l / T) - log n) then equals that cell logit. The prediction is
therefore the same for every edge, so the grid uses one cell and one peak per
point and needs no data.

Run on a dense GPU node (a few seconds on a CPU compute node also works):
    srun --partition=dense --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
        python TETHER/scripts/create_plotly_prediction_landscape.py
"""
import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from plotly.offline import get_plotlyjs

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_DIR))

import models.tf_to_tg_celltype as tf_to_tg_module

DEFAULT_CHECKPOINT = (
    PROJECT_DIR / "checkpoints" / "celltype_tf_tg"
    / "celltype_joint_3887934_20260925_174112_485147" / "checkpoints" / "last.ckpt"
)
DEFAULT_OUTPUT = PROJECT_DIR / "new_plots" / "prediction_landscapes" / "plotly_landscape.html"

# Model input name, display label, factor from model units to display units.
FEATURES = [
    ("binding_score", "Binding score", 1.0),
    ("peak_accessibility", "Peak accessibility (raw)", 1.0),
    ("peak_distance", "|TSS distance| (kb)", 1e-3),
    ("tf_expression", "TF expression (raw log1p)", 1.0),
    ("tg_expression", "TG expression (raw log1p)", 1.0),
]
FEATURE_NAMES = [name for name, _, _ in FEATURES]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-grid", type=int, default=15,
                        help="Grid points per feature (total points = n_grid ** 5).")
    parser.add_argument("--expr-sd", type=float, default=4.0,
                        help="Upper grid limit for expression and accessibility, "
                             "as scaler mean + expr_sd * scaler std.")
    parser.add_argument("--max-distance-kb", type=float, default=250.0,
                        help="Upper distance limit; the model's scaled distance saturates at 250 kb.")
    parser.add_argument("--range", action="append", default=[], metavar="NAME=LO,HI",
                        help="Override one grid range in model units (distance in bp), "
                             "e.g. --range peak_accessibility=0,1.5. Can be repeated.")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.n_grid < 2:
        parser.error("--n-grid must be at least 2")
    return args


def default_ranges(hparams, expr_sd, max_distance_kb):
    """Grid limits in model units. Expression and accessibility come from the checkpoint scaler."""
    def scaler_hi(prefix):
        return hparams[f"{prefix}_mean"] + expr_sd * hparams[f"{prefix}_std"]

    return {
        "binding_score": (0.0, 1.0),                         # sigmoid output of the TF-DNA model
        "peak_accessibility": (0.0, scaler_hi("peak_accessibility")),
        "peak_distance": (0.0, max_distance_kb * 1e3),       # forward() uses |distance|
        "tf_expression": (0.0, scaler_hi("tf_expression")),
        "tg_expression": (0.0, scaler_hi("tg_expression")),
    }


def apply_range_overrides(ranges, overrides):
    for item in overrides:
        name, _, values = item.partition("=")
        if name not in ranges:
            raise ValueError(f"Unknown feature in --range: {name!r}; choose from {FEATURE_NAMES}")
        lo, hi = (float(v) for v in values.split(","))
        if hi <= lo:
            raise ValueError(f"--range {item}: HI must be larger than LO")
        ranges[name] = (lo, hi)
    return ranges


def slider_defaults(hparams):
    """Starting slider positions in model units: scaler means for the scaled inputs."""
    return {
        "binding_score": 0.5,
        "peak_accessibility": hparams["peak_accessibility_mean"],
        "peak_distance": 50_000.0,     # the model's distance-weight decay length
        "tf_expression": hparams["tf_expression_mean"],
        "tg_expression": hparams["tg_expression_mean"],
    }


def predict_grid(lit_model, grids, device, chunk_size):
    """Edge probabilities for every combination of grid values. Returns shape [n] * 5."""
    mesh = np.meshgrid(*(grids[name] for name in FEATURE_NAMES), indexing="ij")
    flat = {
        name: torch.as_tensor(values.ravel(), dtype=torch.float32)
        for name, values in zip(FEATURE_NAMES, mesh)
    }
    n_points = mesh[0].size
    preds = np.empty(n_points, dtype=np.float32)

    lit_model.eval().to(device)
    with torch.inference_mode():
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            n = end - start

            # One edge per grid point, with one cell (C=1) and one peak (P=1).
            batch = {
                "binding_score": flat["binding_score"][start:end, None],                # [n, P]
                "peak_distance": flat["peak_distance"][start:end, None],                # [n, P]
                "peak_accessibility": flat["peak_accessibility"][start:end, None, None],  # [n, C, P]
                "tf_expression": flat["tf_expression"][start:end, None],                # [n, C]
                "tg_expression": flat["tg_expression"][start:end, None],                # [n, C]
                "cell_mask": torch.ones(n, 1, dtype=torch.bool),
                "peak_mask": torch.ones(n, 1, dtype=torch.bool),
            }
            batch = {k: v.to(device) for k, v in batch.items()}

            edge_logits, _ = lit_model(batch)   # full model, including the saved scaler
            preds[start:end] = edge_logits.sigmoid().float().cpu().numpy()

    return preds.reshape(mesh[0].shape)


def compute_landscape(args):
    """Load the checkpoint and predict the 5D grid. Returns (preds, grids, ranges, hparams)."""
    logging.info(f"Loading checkpoint {args.checkpoint}")
    lit_model = tf_to_tg_module.LitTFTGRegulationModel.load_from_checkpoint(
        str(args.checkpoint), map_location="cpu",
    )
    hparams = dict(lit_model.hparams)

    ranges = default_ranges(hparams, args.expr_sd, args.max_distance_kb)
    ranges = apply_range_overrides(ranges, args.range)
    grids = {name: np.linspace(lo, hi, args.n_grid) for name, (lo, hi) in ranges.items()}
    for name, (lo, hi) in ranges.items():
        logging.info(f"  {name}: {lo:.4g} to {hi:.4g}")

    n_points = args.n_grid ** len(FEATURES)
    logging.info(f"Predicting {n_points:,} grid points on {args.device}")
    start = time.perf_counter()
    preds = predict_grid(lit_model, grids, args.device, args.chunk_size)
    logging.info(f"  done in {time.perf_counter() - start:.1f} s; "
                 f"probability range {preds.min():.3f} to {preds.max():.3f}")
    return preds, grids, ranges, hparams

# Pure functions, kept apart from the page wiring so they can be tested with Node.
JS_CORE = r"""
function decodeFloat32(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
}

// Order of the checked features, oldest first. A third check drops the oldest.
function updateOrder(previous, checked) {
    const kept = previous.filter(k => checked.includes(k));
    const added = checked.filter(k => !kept.includes(k));
    return kept.concat(added).slice(-2);
}

// 2D slice of the flattened 5D grid: z[j][i] = value at (feature xf = i, feature yf = j),
// with the other features at fixedIdx.
function sliceGrid(preds, shape, strides, fixedIdx, xf, yf) {
    const cur = fixedIdx.slice();
    const z = [];
    for (let j = 0; j < shape[yf]; j++) {
        cur[yf] = j;
        const row = new Array(shape[xf]);
        for (let i = 0; i < shape[xf]; i++) {
            cur[xf] = i;
            let flat = 0;
            for (let k = 0; k < shape.length; k++) flat += cur[k] * strides[k];
            row[i] = preds[flat];
        }
        z.push(row);
    }
    return z;
}

function gridStats(z) {
    let min = Infinity, max = -Infinity, sum = 0, n = 0;
    for (const row of z) {
        for (const v of row) {
            min = Math.min(min, v);
            max = Math.max(max, v);
            sum += v;
            n += 1;
        }
    }
    return {min: min, max: max, mean: sum / n};
}
"""

JS_UI = r"""
const preds = decodeFloat32(PRED_B64);
const F = CONFIG.features;
const nf = F.length;
const sliderIdx = CONFIG.default_idx.slice();
const startIdx = CONFIG.default_idx.slice();
let order = [0, 1];

const plotDiv = document.getElementById("plot");
const statusDiv = document.getElementById("status");
const checksDiv = document.getElementById("feature-checks");
const slidersDiv = document.getElementById("sliders");
const swapBox = document.getElementById("swap");
const contourBox = document.getElementById("contours");
const colorSelect = document.getElementById("color-mode");
const fmt = v => Number(v).toFixed(3);

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
    value.textContent = fmt(f.grid[sliderIdx[k]]);
    head.append(name, value);
    const input = document.createElement("input");
    input.type = "range";
    input.min = 0;
    input.max = f.grid.length - 1;
    input.step = 1;
    input.value = sliderIdx[k];
    input.addEventListener("input", () => {
        sliderIdx[k] = Number(input.value);
        value.textContent = fmt(f.grid[sliderIdx[k]]);
        render();
    });
    wrap.append(head, input);
    slidersDiv.append(wrap);
    return {wrap: wrap, input: input};
});

[swapBox, contourBox, colorSelect].forEach(el => el.addEventListener("change", render));

function render() {
    if (order.length < 2) {
        sliders.forEach(s => { s.input.disabled = false; s.wrap.classList.remove("plotted"); });
        statusDiv.innerHTML = "<b>Check two features to plot.</b>";
        return;
    }
    let [xf, yf] = order.slice().sort((a, b) => a - b);
    if (swapBox.checked) [xf, yf] = [yf, xf];
    sliders.forEach((s, k) => {
        const plotted = k === xf || k === yf;
        s.input.disabled = plotted;
        s.wrap.classList.toggle("plotted", plotted);
    });

    const shape = CONFIG.shape;
    const z = sliceGrid(preds, shape, CONFIG.strides, sliderIdx, xf, yf);
    const stats = gridStats(z);
    const gx = F[xf].grid;
    const gy = F[yf].grid;

    const mode = colorSelect.value;
    let color = z, cmin = 0, cmax = 1, colorscale = "Viridis", reversescale = false;
    let colorTitle = "Edge<br>probability";
    let text = null;
    if (mode === "fit" && stats.max > stats.min) {
        cmin = stats.min;
        cmax = stats.max;
    } else if (mode === "change") {
        const ref = sliceGrid(preds, shape, CONFIG.strides, startIdx, xf, yf);
        color = z.map((row, j) => row.map((v, i) => v - ref[j][i]));
        let maxAbs = 0.01;
        for (const row of color) for (const v of row) maxAbs = Math.max(maxAbs, Math.abs(v));
        cmin = -maxAbs;
        cmax = maxAbs;
        colorscale = "RdBu";
        reversescale = true;          // red = higher than at the starting slider values
        colorTitle = "Change from<br>start values";
        text = color.map(row => row.map(v => (v >= 0 ? "+" : "") + v.toFixed(3)));
    }

    const trace = {
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
        hovertemplate:
            F[xf].label + ": %{x:.3f}<br>" +
            F[yf].label + ": %{y:.3f}<br>" +
            "Edge probability: %{z:.3f}" +
            (text ? "<br>Change: %{text}" : "") +
            "<extra></extra>",
    };
    if (text) trace.text = text;

    const layout = {
        uirevision: "keep",           // keep the camera angle between redraws
        title: {text: F[yf].label + " vs " + F[xf].label},
        margin: {l: 0, r: 0, t: 50, b: 0},
        scene: {
            xaxis: {title: {text: F[xf].label}, range: [gx[0], gx[gx.length - 1]]},
            yaxis: {title: {text: F[yf].label}, range: [gy[0], gy[gy.length - 1]]},
            zaxis: {title: {text: "Edge probability"}, range: [0, 1]},
            aspectmode: "manual",
            aspectratio: {x: 1.2, y: 1.0, z: 0.7},
        },
    };
    Plotly.react(plotDiv, [trace], layout, {responsive: true});

    const fixed = [];
    for (let k = 0; k < nf; k++) {
        if (k !== xf && k !== yf) fixed.push(F[k].label + " = " + fmt(F[k].grid[sliderIdx[k]]));
    }
    statusDiv.innerHTML =
        "<b>Fixed:</b> " + fixed.join(" &nbsp;|&nbsp; ") +
        "<br><b>This slice:</b> min " + fmt(stats.min) +
        ", mean " + fmt(stats.mean) + ", max " + fmt(stats.max);
}

render();
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TF-TG Prediction Landscape (3D)</title>
<style>
  body { margin: 0; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
         color: #222; background: #fff; }
  #app { display: flex; gap: 16px; padding: 16px; box-sizing: border-box; }
  #controls { flex: 0 0 300px; }
  #controls h2 { font-size: 18px; margin: 0 0 12px; }
  #controls h3 { font-size: 14px; margin: 16px 0 6px; }
  #controls label { display: block; margin: 3px 0; font-size: 14px; }
  #controls select { width: 100%; font-size: 14px; }
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
    <h2>Prediction landscape</h2>
    <h3>Plotted features <small>(check two)</small></h3>
    <div id="feature-checks"></div>
    <label><input type="checkbox" id="swap"> Swap axes</label>
    <h3>Fixed features</h3>
    <div id="sliders"></div>
    <h3>Surface color</h3>
    <select id="color-mode">
      <option value="fixed">Edge probability (0–1)</option>
      <option value="fit">Edge probability (fit to slice)</option>
      <option value="change">Change from starting slider values</option>
    </select>
    <label><input type="checkbox" id="contours"> Contour lines</label>
    <div id="notes">__NOTES__</div>
  </div>
  <div id="main">
    <div id="plot"></div>
    <div id="status"></div>
  </div>
</div>
<script>
const CONFIG = __CONFIG__;
const PRED_B64 = "__PRED_B64__";
</script>
<script id="landscape-core">__JS_CORE__</script>
<script>__JS_UI__</script>
</body>
</html>
"""


def build_page(preds, grids, defaults, ranges, checkpoint):
    shape = list(preds.shape)
    features, default_idx = [], []
    for name, label, scale in FEATURES:
        grid = np.asarray(grids[name]) * scale
        features.append({"name": name, "label": label, "grid": [round(float(v), 6) for v in grid]})
        default_idx.append(int(np.abs(grid - defaults[name] * scale).argmin()))

    config = {
        "features": features,
        "shape": shape,
        "strides": [int(np.prod(shape[k + 1:])) for k in range(len(shape))],
        "default_idx": default_idx,
    }
    pred_b64 = base64.b64encode(preds.astype("<f4").ravel().tobytes()).decode("ascii")

    range_text = "<br>".join(
        f"{label}: {ranges[name][0] * scale:.3f} – {ranges[name][1] * scale:.3f}"
        for name, label, scale in FEATURES
    )
    notes = (
        f"<b>Grid:</b> {shape[0]} points per feature<br>{range_text}<br>"
        f"<b>Checkpoint:</b> {checkpoint.parent.parent.name}<br>"
        "Every cell and peak of the edge gets these values, so the prediction "
        "is the same for any edge."
    )

    # Plotly's library goes in last, so the other placeholders are never searched for inside it.
    replacements = {
        "__NOTES__": notes,
        "__CONFIG__": json.dumps(config),
        "__PRED_B64__": pred_b64,
        "__JS_CORE__": JS_CORE,
        "__JS_UI__": JS_UI,
        "__PLOTLY_JS__": get_plotlyjs(),
    }
    html = PAGE
    for key, value in replacements.items():
        html = html.replace(key, value)
    return html


def main():
    args = parse_args()
    preds, grids, ranges, hparams = compute_landscape(args)
    html = build_page(preds, grids, slider_defaults(hparams), ranges, args.checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html)
    logging.info(f"Saved {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
