# transformer_testing.py
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import FuncFormatter, MultipleLocator
from sklearn.metrics import r2_score
import argparse

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
SRC_DIR = str(Path(PROJECT_DIR) / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from multiomic_transformer.datasets.dataset import SimpleScaler
from multiomic_transformer.models.model import MultiomicTransformer

logging.basicConfig(level=logging.INFO, format='%(message)s')

def format_gpu_usage_file(gpu_log_file):
    gpu = pd.read_csv(gpu_log_file)
    gpu.columns = gpu.columns.str.strip()
    gpu["timestamp"] = pd.to_datetime(gpu["timestamp"], errors="coerce")
    gpu["tsec"] = gpu["timestamp"].dt.floor("s")

    gpu["memory_used_gib"]  = gpu["memory.used [MiB]"].astype(str).str.extract(r"(\d+)").astype(float) / 1024
    gpu["memory_total_gib"] = gpu["memory.total [MiB]"].astype(str).str.extract(r"(\d+)").astype(float) / 1024

    t0 = gpu["tsec"].min()
    gpu["elapsed_s"] = (gpu["tsec"] - t0).dt.total_seconds().astype(int)
    gpu["elapsed_min"] = gpu["elapsed_s"] / 60.0
    gpu["elapsed_hr"] = gpu["elapsed_s"] / 3600.0
    

    # mean per second, then carry minutes as a column
    mean_per_sec = (
        gpu.groupby("elapsed_s", as_index=False)["memory_used_gib"]
           .mean()
           .sort_values("elapsed_s")
    )
    mean_per_sec["elapsed_min"] = mean_per_sec["elapsed_s"] / 60.0
    mean_per_sec["elapsed_hr"] = mean_per_sec["elapsed_s"] / 3600.0

    total_gib = float(gpu["memory_total_gib"].iloc[0])
    return gpu, mean_per_sec, total_gib

def plot_gpu_usage(gpu_log_dict, align_to_common_duration=False, smooth=None):
    """
    align_to_common_duration: if True, truncate each run to the shortest duration so curves end together.
    smooth: optional int window (in seconds) for a centered rolling mean on memory (e.g., smooth=5).
    """
    fig, ax = plt.subplots(figsize=(7,4))

    totals = []
    max_elapsed_by_run = {}
    for label, (_, mean_per_min, total_gib) in gpu_log_dict.items():
        totals.append(total_gib)
        max_elapsed_by_run[label] = mean_per_min["elapsed_min"].max()

    # shortest duration across runs (so lines end at the same x)
    common_end = min(max_elapsed_by_run.values()) if align_to_common_duration else None

    for label, (_, mean_per_min, total_gib) in gpu_log_dict.items():
        m = mean_per_min.copy()
        if align_to_common_duration and common_end is not None:
            m = m[m["elapsed_min"] <= common_end]

        if smooth and smooth > 1:
            m["memory_used_gib"] = m["memory_used_gib"].rolling(smooth, center=True, min_periods=1).mean()

        ax.plot(m["elapsed_hr"], m["memory_used_gib"], label=f"{label}", linewidth=3)

    max_total = max(totals)
    ax.axhline(max_total, linestyle="--", label=f"Max RAM")
    ax.set_ylabel("GiB")
    ax.set_xlabel("Minutes since start")
    ax.set_ylim(0, max_total + 1)
    ax.xaxis.set_major_locator(MultipleLocator(1))  # tick every 1 hour
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
    ax.set_xlabel("Hours since start")

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
    ax.set_title(
        f"Average GPU Memory vs. Elapsed Time\n"
    )
    plt.tight_layout()
    plt.show()
    return fig

def plot_gpu_usage_boxplots(gpu_log_dict, use_mean_per_sec: bool = True, title_suffix: str = ""):
    """
    gpu_log_dict: {label: (gpu_df, mean_per_sec_df, total_gib)}
        - same structure returned/consumed by format_gpu_usage_file / plot_gpu_usage.
    use_mean_per_sec:
        - True: boxplots over per-second mean memory (smoother).
        - False: boxplots over all raw samples.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    data = []
    labels = []
    capacities = []

    for label, (gpu_df, mean_per_sec_df, total_gib) in gpu_log_dict.items():
        if use_mean_per_sec:
            series = mean_per_sec_df["memory_used_gib"]
        else:
            series = gpu_df["memory_used_gib"]

        # drop NaNs just in case
        series = series.dropna()
        if series.empty:
            continue

        data.append(series.values)
        labels.append(label)
        capacities.append(total_gib)

    if not data:
        raise ValueError("No GPU usage data available to plot.")

    # Boxplot: one box per run
    bp = ax.boxplot(
        data,
        labels=labels,
        showmeans=True,
        meanline=True,
        patch_artist=True,
    )

    # Optional: horizontal line for (max) GPU capacity across runs
    max_cap = max(capacities)
    ax.axhline(max_cap, linestyle="--", linewidth=1.0,
               label=f"Max capacity")

    ax.set_ylabel("GPU memory used (GiB)")
    base_title = "GPU Memory Usage Distribution per Run"
    if title_suffix:
        base_title += f" ({title_suffix})"
    ax.set_title(base_title)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )

    plt.tight_layout()
    return fig

def plot_R2_across_epochs(df):
    fig = plt.figure(figsize=(6, 5))
    plt.plot(df.index, df["R2_u"], linewidth=2, label=f"Best R2 (unscaled) = {df['R2_u'].max():.2f}")
    plt.plot(df.index, df["R2_s"], linewidth=2, label=f"Best R2 (scaled)     = {df['R2_s'].max():.2f}")

    plt.title(f"TG Expression R2 Across Training", fontsize=17)
    plt.ylim((0,1))
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlabel("Epoch", fontsize=17)
    plt.ylabel("R2", fontsize=17)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    return fig

def plot_train_val_loss(df):
    fig = plt.figure(figsize=(6, 5))
    df = df.copy().iloc[5:, :]
    plt.plot(df["Epoch"], df["Train MSE"], label="Train MSE", linewidth=2)
    plt.plot(df["Epoch"], df["Val MSE"], label="Val MSE", linewidth=2)
    # plt.plot(df["Epoch"], df["Train Total Loss"], label="Train Total Loss", linestyle="--", alpha=0.7)

    plt.title(f"Train Val Loss Curves", fontsize=17)
    plt.xlabel("Epoch", fontsize=17)
    plt.ylabel("Loss", fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.ylim([0, 1])
    # plt.xlim(left=2)
    plt.legend(fontsize=15)
    plt.tight_layout()
    return fig

def nanaware_per_gene_stats(y_true, y_pred, eps=1e-8):
    """Compute per-gene metrics accounting for NaN entries."""
    N, G = y_true.shape
    r2 = np.full(G, np.nan, dtype=np.float64)
    pearson = np.full(G, np.nan, dtype=np.float64)
    mae = np.full(G, np.nan, dtype=np.float64)
    rmse = np.full(G, np.nan, dtype=np.float64)
    n_obs = np.zeros(G, dtype=np.int32)

    for j in range(G):
        mask = np.isfinite(y_true[:, j]) & np.isfinite(y_pred[:, j])
        m = mask.sum()
        n_obs[j] = m
        
        if m < 2:
            continue
            
        yt = y_true[mask, j].astype(np.float64)
        yp = y_pred[mask, j].astype(np.float64)

        diff = yt - yp
        mae[j] = np.mean(np.abs(diff))
        rmse[j] = np.sqrt(np.mean(diff**2))

        yt_c = yt - yt.mean()
        yp_c = yp - yp.mean()
        yt_std = np.sqrt((yt_c**2).sum())
        yp_std = np.sqrt((yp_c**2).sum())
        if yt_std > 0 and yp_std > 0:
            pearson[j] = (yt_c @ yp_c) / (yt_std * yp_std)

        sst = ((yt - yt.mean())**2).sum()
        sse = (diff**2).sum()
        r2[j] = 1.0 - sse / (sst + eps)

    return {"r2": r2, "pearson": pearson, "mae": mae, "rmse": rmse, "n_obs": n_obs}

def load_model(selected_experiment_dir, checkpoint_file, device):
    params_path = selected_experiment_dir / "run_parameters.json"
    with open(params_path, "r") as f:
        params = json.load(f)

    # Pull out architecture hyperparameters
    d_model   = params.get("d_model")
    num_heads = params.get("num_heads")
    num_layers = params.get("num_layers")
    d_ff      = params.get("d_ff")
    dropout   = params.get("dropout", 0.0)
    use_shortcut   = params.get("use_shortcut", False)
    use_dist_bias  = params.get("use_dist_bias", False)
    use_motif_mask = params.get("use_motif_mask", False)

    
    # 1) Load test loader and checkpoint
    test_loader = torch.load(selected_experiment_dir / "test_loader.pt", weights_only=False)

    ckpt_path = os.path.join(selected_experiment_dir, checkpoint_file)
    state = torch.load(ckpt_path, map_location="cpu")
    
    # 2) Recreate model EXACTLY as in training
    model = MultiomicTransformer(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        tf_vocab_size=len(state["tf_scaler_mean"]),
        tg_vocab_size=len(state["tg_scaler_mean"]),
        use_bias=use_dist_bias,
        use_shortcut=use_shortcut,
        use_motif_mask=use_motif_mask,
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        missing, unexpected = model.load_state_dict(
            state["model_state_dict"], strict=False
        )
        if len(missing) > 0:
            logging.info("Missing keys:", missing)
        if len(unexpected) > 0:
            logging.info("Unexpected keys:", unexpected)
    elif isinstance(state, dict) and "model_state_dict" not in state:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0:
            logging.info("Missing keys:", missing)
        if len(unexpected) > 0:
            logging.info("Unexpected keys:", unexpected)
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0:
            logging.info("Missing keys:", missing)
        if len(unexpected) > 0:
            logging.info("Unexpected keys:", unexpected)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 3) Rebuild scalers on the SAME DEVICE as inputs
    tg_scaler = SimpleScaler(
        mean=torch.as_tensor(state["tg_scaler_mean"], device=device, dtype=torch.float32),
        std=torch.as_tensor(state["tg_scaler_std"],  device=device, dtype=torch.float32),
    )
    tf_scaler = SimpleScaler(
        mean=torch.as_tensor(state["tf_scaler_mean"], device=device, dtype=torch.float32),
        std=torch.as_tensor(state["tf_scaler_std"],  device=device, dtype=torch.float32),
    )

    return model, test_loader, tg_scaler, tf_scaler, state

def plot_model_tg_predictions(device, model, test_loader, tg_scaler, tf_scaler):
        
    G_total = tg_scaler.mean.shape[0]  # total number of genes

    # ---- global per-gene accumulators (unscaled space) ----
    sse_g   = torch.zeros(G_total, dtype=torch.float64)
    sumy_g  = torch.zeros(G_total, dtype=torch.float64)
    sumy2_g = torch.zeros(G_total, dtype=torch.float64)
    cnt_g   = torch.zeros(G_total, dtype=torch.float64)
    sumpred_g = torch.zeros(G_total, dtype=torch.float64)

    ### For overall R² and scatter:
    all_preds_for_plot = []
    all_tgts_for_plot  = []

    with torch.no_grad():
        for batch in test_loader:
            atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
            atac_wins  = atac_wins.to(device)
            tf_tensor  = tf_tensor.to(device)
            targets    = targets.to(device)
            bias       = bias.to(device)
            tf_ids     = tf_ids.to(device)
            tg_ids     = tg_ids.to(device)
            motif_mask = motif_mask.to(device)

            # scale / predict exactly like in validation
            if tf_scaler is not None:
                tf_tensor = tf_scaler.transform(tf_tensor, tf_ids)
            if tg_scaler is not None:
                targets_s = tg_scaler.transform(targets, tg_ids)
            else:
                targets_s = targets

            preds_s, _, _, _ = model(
                atac_wins, tf_tensor,
                tf_ids=tf_ids, tg_ids=tg_ids,
                bias=bias, motif_mask=motif_mask,
                return_edge_logits=True, return_shortcut_contrib=False,
                edge_extra_features=None,
            )

            preds_s   = torch.nan_to_num(preds_s.float(),   nan=0.0, posinf=1e6, neginf=-1e6)
            targets_s = torch.nan_to_num(targets_s.float(), nan=0.0, posinf=1e6, neginf=-1e6)

            # unscale + clamp
            if tg_scaler is not None:
                targets_u = tg_scaler.inverse_transform(targets_s, tg_ids)
                preds_u   = tg_scaler.inverse_transform(preds_s,   tg_ids)
            else:
                targets_u, preds_u = targets_s, preds_s

            targets_u = torch.nan_to_num(targets_u.float(), nan=0.0, posinf=1e6, neginf=-1e6)
            preds_u   = torch.nan_to_num(preds_u.float(),   nan=0.0, posinf=1e6, neginf=-1e6)
            preds_u   = preds_u.clamp_min(0.0)
            
            sumpred_batch = preds_u.sum(dim=0)


            # ---- store for overall R² / scatter ----
            all_tgts_for_plot.append(targets_u.detach().cpu().numpy())
            all_preds_for_plot.append(preds_u.detach().cpu().numpy())

            # ---- per-gene accumulators (unscaled) ----
            # shapes: [B, G_eval]
            err2   = (targets_u - preds_u) ** 2
            B      = targets_u.shape[0]

            # reduce over batch
            sse_batch   = err2.sum(dim=0)              # [G_eval]
            sumy_batch  = targets_u.sum(dim=0)
            sumy2_batch = (targets_u ** 2).sum(dim=0)
            cnt_batch   = torch.full_like(sse_batch, B, dtype=torch.float64)

            # move ids to CPU, accumulate into global vectors
            ids_cpu = tg_ids.cpu()
            sse_g.index_add_(0, ids_cpu, sse_batch.cpu().to(torch.float64))
            sumy_g.index_add_(0, ids_cpu, sumy_batch.cpu().to(torch.float64))
            sumy2_g.index_add_(0, ids_cpu, sumy2_batch.cpu().to(torch.float64))
            cnt_g.index_add_(0, ids_cpu, cnt_batch.cpu().to(torch.float64))
            sumpred_g.index_add_(0, ids_cpu, sumpred_batch.cpu().to(torch.float64))

    # ============================
    # 4) Per-gene R² (global)
    # ============================
    eps = 1e-12
    mask = cnt_g > 0  # genes that appeared in the test set

    mean_g = sumy_g[mask] / cnt_g[mask]
    sst_g  = sumy2_g[mask] - cnt_g[mask] * (mean_g ** 2)

    valid = sst_g > eps  # genes with non-trivial variance

    r2_g = torch.full_like(sse_g, float("nan"), dtype=torch.float64)

    idx_all  = mask.nonzero(as_tuple=True)[0]   # indices of genes with any data
    idx_keep = idx_all[valid]                   # subset with non-zero variance

    r2_g_values = 1.0 - (sse_g[idx_keep] / torch.clamp(sst_g[valid], min=eps))
    r2_g[idx_keep] = r2_g_values

    r2_g_cpu = r2_g.cpu().numpy()
    median_r2_gene = np.nanmedian(r2_g_cpu)

    # ============================
    # 5) Global R² + scatter plot
    # ============================
    preds_flat = np.concatenate([p.reshape(-1) for p in all_preds_for_plot])
    tgts_flat  = np.concatenate([t.reshape(-1) for t in all_tgts_for_plot])

    # Remove NaNs / infs
    valid = np.isfinite(preds_flat) & np.isfinite(tgts_flat)
    preds_clean = preds_flat[valid]
    tgts_clean  = tgts_flat[valid]
    
    mean_true_g = torch.full_like(sumy_g, float("nan"))
    mean_pred_g = torch.full_like(sumpred_g, float("nan"))

    mask = cnt_g > 0
    mean_true_g[mask] = sumy_g[mask] / cnt_g[mask]
    mean_pred_g[mask] = sumpred_g[mask] / cnt_g[mask]

    mean_true = mean_true_g.cpu().numpy()
    mean_pred = mean_pred_g.cpu().numpy()
    
    var_g = (sumy2_g[mask] / cnt_g[mask]) - (mean_true_g[mask] ** 2)
    keep = var_g.cpu().numpy() > 1e-6

    mean_true = mean_true[keep]
    mean_pred = mean_pred[keep]

    # Overall R² across all points
    r2_overall = r2_score(tgts_clean, preds_clean)

    logging.info("\n" + "="*60)
    logging.info("SCATTER PLOT STATISTICS")
    logging.info("="*60)
    logging.info(f"Overall R² (from all points): {r2_overall:.4f}")
    logging.info(f"N samples (valid points): {len(preds_clean):,}")
    logging.info(f"Median per-gene R²: {median_r2_gene:.4f}")
    
    # ---- per-gene mean expression plot ----
    per_gene_mean_fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(mean_true, mean_pred, alpha=0.6, s=20)

    lims = [
        min(mean_true.min(), mean_pred.min()),
        max(mean_true.max(), mean_pred.max()),
    ]
    ax.plot(lims, lims, "k--", lw=2)

    r2_tg = r2_score(mean_true, mean_pred)

    ax.set_xlabel("Mean actual TG expression")
    ax.set_ylabel("Mean predicted TG expression")
    ax.set_title(
        "Gene-level agreement between predicted \nand observed target gene expression\n"
        f"Mean $R^2$ = {r2_tg:.4f}"
    )

    ax.grid(alpha=0.3)
    plt.tight_layout()

    # ---- scatter plot of all points ----
    all_point_fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(tgts_clean, preds_clean, alpha=0.3, s=10, color='steelblue')

    lims = [
        min(np.min(tgts_clean), np.min(preds_clean)),
        max(np.max(tgts_clean), np.max(preds_clean))
    ]
    ax.plot(lims, lims, 'k--', lw=2, label='Perfect prediction')

    ax.set_xlabel('Actual Expression', fontsize=12)
    ax.set_ylabel('Predicted Expression', fontsize=12)
    ax.set_title(
        "Predicted vs Actual TG Expression\n"
        f"$R^2 = {r2_overall:.4f}$",
        fontsize=14,
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    return per_gene_mean_fig, all_point_fig, tgts_clean, preds_clean, mean_true, mean_pred

def locate_last_checkpoint(experiment_dir):
    checkpoint_files = sorted(experiment_dir.glob("checkpoint_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {experiment_dir}")
    last_checkpoint = checkpoint_files[-1]
    return last_checkpoint.name

if __name__ == "__main__":    
    arg_parser = argparse.ArgumentParser(description="Run AUROC and AUPR testing for trained models")

    arg_parser.add_argument("--experiment", type=str, required=True, help="Name of the experiment to test")
    arg_parser.add_argument("--training_num", type=str, required=False, default="model_training_001", help="Training number folder to test")
    arg_parser.add_argument("--experiment_dir", type=Path, required=True, help="Full path to the experiment directory to test")
    arg_parser.add_argument("--model_file", type=str, required=False, default="trained_model.pt", help="Name of the trained model file (default: trained_model.pt)")

    args = arg_parser.parse_args()

    experiment = args.experiment
    experiment_dir = Path(args.experiment_dir)
    training_num = args.training_num if args.training_num else "model_training_001"
    
    FIG_DIR = Path("/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/FIGURES")
    FIG_DATA = Path("/gpfs/Labs/Uzun/RESULTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/FIGURE_DATA")

    if "chr19" in [p.name for p in Path(experiment_dir / experiment).iterdir()] and experiment != "mESC_no_scale_linear":
        EXPERIMENT_DIR = experiment_dir / experiment / "chr19" / training_num
        
        exp_fig_dir = FIG_DIR / experiment / training_num
        exp_fig_data_dir = FIG_DATA / experiment / training_num
    else:
        EXPERIMENT_DIR = experiment_dir / experiment / training_num
        
        exp_fig_dir = FIG_DIR / experiment / training_num
        exp_fig_data_dir = FIG_DATA / experiment / training_num

    logging.info(f"Selected experiment directory: {EXPERIMENT_DIR}")

    if not os.path.exists(exp_fig_data_dir):
        os.makedirs(exp_fig_data_dir)
        
    if not os.path.exists(exp_fig_dir):
        os.makedirs(exp_fig_dir)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH  = 32
    TG_CHUNK = 64
    
    # Plot GPU usage
    for file in EXPERIMENT_DIR.glob("gpu_usage*.csv"):
        print(f"Processing GPU log file: {file.name}")
        gpu_file = format_gpu_usage_file(file)
        gpu_fig = plot_gpu_usage({"GPU Usage": gpu_file})

        logging.info("Plotting GPU memory usage...")
        gpu_fig.savefig(EXPERIMENT_DIR / "gpu_memory_requirements.svg")
        gpu_fig.savefig(exp_fig_dir / "gpu_memory_requirements.svg")
    else:
        logging.info("No GPU usage log found, skipping GPU memory plotting.")
    
    if os.path.exists(EXPERIMENT_DIR / "training_log.csv"):
        # Load training log
        training_log_df = pd.read_csv(os.path.join(EXPERIMENT_DIR, "training_log.csv"), header=0)
        training_log_df.dropna(inplace=True)
        
        # Plot Training R2 across epochs
        logging.info("Plotting training R2 across epochs...")
        train_r2_fig = plot_R2_across_epochs(training_log_df)
        train_r2_fig.savefig(os.path.join(EXPERIMENT_DIR, "eval_results_pearson_corr.svg"))
        train_r2_fig.savefig(os.path.join(EXPERIMENT_DIR, "eval_results_pearson_corr.png"), dpi=200)
        train_r2_fig.savefig(exp_fig_dir / "eval_results_pearson_corr.svg")
        training_log_df.to_csv(exp_fig_data_dir / "training_log_df.csv")
        
        # Plot Train Val Loss curves
        logging.info("Plotting training and validation loss curves...")
        train_val_loss_fig = plot_train_val_loss(training_log_df)
        train_val_loss_fig.savefig(os.path.join(EXPERIMENT_DIR, "train_val_loss_curves.svg"))
        train_val_loss_fig.savefig(os.path.join(EXPERIMENT_DIR, "train_val_loss_curves.png"), dpi=200)
        train_val_loss_fig.savefig(exp_fig_dir / "train_val_loss_curves.svg")
    else:
        logging.info("No training log found, skipping training R2 and loss plotting.")
    
    if not os.path.exists(EXPERIMENT_DIR / args.model_file):
        # Try to load the last checkpoint if the specified model file does not exist
        logging.info(f"Model file {args.model_file} not found, attempting to load last checkpoint...")
        last_checkpoint_file = locate_last_checkpoint(EXPERIMENT_DIR)
        args.model_file = last_checkpoint_file
        logging.info(f"Using last checkpoint file: {args.model_file}")

    # Load model and test data
    logging.info("Loading model and test data...")
    model, test_loader, tg_scaler, tf_scaler, state = load_model(
        selected_experiment_dir=EXPERIMENT_DIR,
        checkpoint_file=args.model_file,
        device=DEVICE
    )
    
    # Plot TG predictions
    logging.info("Plotting TG predictions...")
    per_gene_mean_fig, all_point_fig, tgts_clean, preds_clean, mean_true, mean_pred = plot_model_tg_predictions(
        device=DEVICE,
        model=model,
        test_loader=test_loader,
        tg_scaler=tg_scaler,
        tf_scaler=tf_scaler,
    )
    
    per_gene_mean_fig.savefig(EXPERIMENT_DIR / "per_gene_mean_expression.png", dpi=200)
    per_gene_mean_fig.savefig(exp_fig_dir / "per_gene_mean_expression.svg")
    
    all_point_fig.savefig(EXPERIMENT_DIR / "test_set_r2_distribution.png", dpi=200)
    all_point_fig.savefig(exp_fig_dir / "test_set_r2_distribution.svg")
    
    # Save R2 prediction data
    logging.info("Saving R2 prediction data...")
    r2_acc_dir = exp_fig_data_dir / "tg_pred_acc_data"
    if not os.path.isdir(r2_acc_dir):
        os.makedirs(r2_acc_dir, exist_ok=True)
    np.save(r2_acc_dir / "tgts_clean.npy", tgts_clean)
    np.save(r2_acc_dir / "preds_clean.npy", preds_clean)
    
    # Saving per-gene mean expression data
    np.save(r2_acc_dir / "mean_true.npy", mean_true)
    np.save(r2_acc_dir / "mean_pred.npy", mean_pred)

    logging.info("All done!")
    
    
    
