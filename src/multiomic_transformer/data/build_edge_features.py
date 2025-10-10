#!/usr/bin/env python
import os, json, torch, logging, argparse
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from multiomic_transformer.datasets.dataset import MultiomicTransformerDataset
from multiomic_transformer.models.model import MultiomicTransformer
from torch.utils.data import DataLoader
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format="%(message)s")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def load_model_from_run_params(run_dir, tf_vocab_size, tg_vocab_size, device="cpu"):
    with open(os.path.join(run_dir, "run_parameters.json")) as f:
        run_params = json.load(f)
    model = MultiomicTransformer(
        d_model=run_params.get("d_model", 384),
        num_heads=run_params.get("num_heads", 6),
        num_layers=run_params.get("num_layers", 3),
        d_ff=run_params.get("d_feedforward", 768),
        dropout=run_params.get("dropout", 0.1),
        tf_vocab_size=tf_vocab_size,
        tg_vocab_size=tg_vocab_size,
        bias_scale=run_params.get("attn_bias_scale", 1.0),
        use_shortcut=run_params.get("use_shortcut", True),
        use_motif_mask=run_params.get("use_motif_mask", False),
        lambda_l1=run_params.get("shortcut_l1", 0.0),
        lambda_l2=run_params.get("shortcut_l2", 0.0),
        topk=run_params.get("shortcut_topk", None),
        shortcut_dropout=run_params.get("shortcut_dropout", 0.0),
    ).to(device)
    return model, run_params

# ---------------------------------------------------------------------
# Helpers for normalization / gradients
# ---------------------------------------------------------------------
@torch.no_grad()
def _global_minmax_(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)

def zscore_per_cell(x: torch.Tensor, eps=1e-6):
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True).clamp_min(eps)
    return (x - mu) / sd


def gradient_attribution_matrix(model, dataset, loader, device=DEVICE, normalize="global"):
    """
    Compute gradient attribution per TF–TG pair efficiently by vectorizing
    across TGs within a single backward pass.
    """
    TF, TG = len(dataset.tf_names), len(dataset.tg_names)
    acc = torch.zeros(TF, TG, dtype=torch.float32, device="cpu")

    for batch in tqdm(loader, desc="Fast gradient attribution"):
        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]

        # enable gradient on TFs
        tf_tensor = tf_tensor.detach().clone().requires_grad_(True)
        tf_norm = zscore_per_cell(tf_tensor)

        # Forward full TG slice (all TGs at once)
        preds = model(
            atac_wins,
            tf_norm,
            tf_ids=tf_ids,
            tg_ids=torch.arange(TG, device=device),
            bias=bias,
            motif_mask=motif_mask,
        )
        preds = preds[0] if isinstance(preds, tuple) else preds  # [B, TG]

        # ---- Compute mean output per TG ----
        # shape: [TG]; scalar loss aggregates all TGs equally
        mean_per_tg = preds.mean(dim=0)  # [TG]
        total = mean_per_tg.sum()

        # ---- One backward pass (for all TGs) ----
        grads = torch.autograd.grad(total, tf_norm, retain_graph=False, create_graph=False)[0]  # [B, TF]

        # collapse batch → TF
        grad_vec = grads.abs().sum(dim=0).cpu()  # [TF]
        acc += grad_vec.unsqueeze(1).expand(-1, TG) / len(loader)

        torch.cuda.empty_cache()

    if normalize == "global":
        acc = _global_minmax_(acc)

    return pd.DataFrame(acc.numpy(), index=dataset.tf_names, columns=dataset.tg_names)

def extract_edge_features(
    model, dataloader, tf_names, tg_names, tg_id_map,
    chip_edges=None, gradient_attrib_df=None,
    device="cuda", tg_chunk=64
):
    """
    Extract TF–TG edge-level features from a trained MultiomicTransformer.

    Returns
    -------
    DataFrame with columns:
        ['TF', 'TG', 'attn', 'motif_mask', 'grad_attr',
         'pred_mean', 'pred_std', 'bias_mean', 'label' (optional)]
    """
    model.eval()
    TF, TG = len(tf_names), len(tg_names)
    all_records = []

    tf_names_np = np.array(tf_names)
    tg_names_np = np.array(tg_names)

    for batch in tqdm(dataloader, desc="Extracting edge features"):
        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]

        if not torch.is_tensor(tg_ids):
            tg_ids = torch.as_tensor(tg_ids, dtype=torch.long, device=device)

        tg_global = tg_id_map[tg_ids.cpu().numpy()]
        tg_batch_names = np.array(tg_names)[tg_global]
        B = len(tg_batch_names)

        with torch.no_grad():
            out = model(
                atac_wins, tf_tensor,
                tf_ids=tf_ids, tg_ids=tg_ids,
                bias=bias, motif_mask=motif_mask
            )
            preds, attn = out if isinstance(out, tuple) else (out, None)

        preds_cpu = preds.detach().cpu()
        attn_cpu  = attn.detach().cpu() if attn is not None else None
        bias_cpu  = bias.mean(dim=(0, 2)).detach().cpu()
        motif_cpu = motif_mask.float().cpu() if motif_mask is not None else None

        # --------------------------
        # Ensure all arrays same len
        # --------------------------
        min_len = min(
            preds_cpu.shape[0],
            len(tg_batch_names),
            bias_cpu.shape[0] if bias_cpu.ndim else len(bias_cpu)
        )
        tg_batch_names = tg_batch_names[:min_len]

        # Compute statistics
        if preds_cpu.dim() > 1:
            pred_mean = preds_cpu.mean(dim=1).numpy()[:min_len]
            pred_std  = preds_cpu.std(dim=1).numpy()[:min_len]
        else:
            pred_mean = preds_cpu.numpy()[:min_len]
            pred_std  = np.zeros_like(pred_mean)

        bias_mean = bias_cpu.numpy()[:min_len]

        tg_stats = pd.DataFrame({
            "TG": tg_batch_names,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "bias_mean": bias_mean,
        })

        # Long-form attention and motif tables
        if attn_cpu is not None:
            attn_cpu_np = attn_cpu.numpy()[:min_len]
            attn_df = pd.DataFrame(attn_cpu_np, columns=tf_names_np, index=tg_batch_names)
            attn_long = attn_df.stack().reset_index()
            attn_long.columns = ["TG", "TF", "attn"]
        else:
            attn_long = pd.DataFrame(columns=["TG", "TF", "attn"])

        if motif_cpu is not None:
            motif_cpu_np = motif_cpu.numpy()[:min_len]
            motif_df = pd.DataFrame(motif_cpu_np, columns=tf_names_np, index=tg_batch_names)
            motif_long = motif_df.stack().reset_index()
            motif_long.columns = ["TG", "TF", "motif_mask"]
        else:
            motif_long = pd.DataFrame(columns=["TG", "TF", "motif_mask"])

        # Combine everything
        edge_batch_df = (
            attn_long
            .merge(motif_long, on=["TG", "TF"], how="outer")
            .merge(tg_stats, on="TG", how="left")
        )
        
        if gradient_attrib_df is not None:
            edge_batch_df = edge_batch_df.merge(
                gradient_attrib_df.stack().rename("grad_attr").reset_index().rename(columns={"level_0": "TF", "level_1": "TG"}),
                on=["TF", "TG"], how="left"
            )
            edge_batch_df["grad_attr"] = edge_batch_df["grad_attr"].fillna(0.0)
        else:
            edge_batch_df["grad_attr"] = 0.0
            
        all_records.append(edge_batch_df)

        del preds_cpu, attn_cpu, bias_cpu, motif_cpu
        torch.cuda.empty_cache()

    df = pd.concat(all_records, ignore_index=True)
    del all_records

    if chip_edges is not None:
        chip_set = set((t.upper(), g.upper()) for t, g in chip_edges)
        df["label"] = [(tf.upper(), tg.upper()) in chip_set for tf, tg in zip(df["TF"], df["TG"])]
        df["label"] = df["label"].astype(int)

    return df


# ---------------------------------------------------------------------
# 1. Aggregate TF–TG motif/distance/expression features
# ---------------------------------------------------------------------

def build_global_tf_tg_features(cache_dir: str, chrom_list: list[str], n_jobs: int = -1) -> pd.DataFrame:
    """
    Aggregate TF–TG motif/distance/expression features across all chromosomes.
    Uses vectorized correlation and multi-core parallelism.

    Parameters
    ----------
    cache_dir : str
        Directory containing per-chromosome cached data (moods_sites_*.tsv, tg_tensor_all_*.pt, etc.)
    n_jobs : int, default=-1
        Number of CPU cores to use in parallel (joblib). Use -1 for all available cores.

    Returns
    -------
    all_df : pd.DataFrame
        Aggregated TF–TG feature dataframe with motif, distance, and correlation features.
    """

    chr_dirs = sorted([d for d in os.listdir(cache_dir) if d.startswith("chr") and d in chrom_list])
    logging.info(f"Found {len(chr_dirs)} chromosomes under {cache_dir}")

    # --- Load shared TF tensor once ---
    tf_tensor_path = os.path.join(cache_dir, "tf_tensor_all.pt")
    tf_names_path = os.path.join(cache_dir, "tf_names.json")

    if not (os.path.exists(tf_tensor_path) and os.path.exists(tf_names_path)):
        raise FileNotFoundError(f"Missing TF tensor or vocab in {cache_dir}")

    tf_tensor = torch.load(tf_tensor_path, map_location="cpu")
    with open(tf_names_path) as f:
        tf_names = json.load(f)

    tf_expr = tf_tensor.numpy()
    tf_centered = tf_expr - tf_expr.mean(axis=1, keepdims=True)
    tf_std = tf_centered.std(axis=1, keepdims=True)

    # -------------------------------
    # Function to process one chromosome
    # -------------------------------
    def process_chrom(chr_dir):
        chr_path = os.path.join(cache_dir, chr_dir)
        moods_path = os.path.join(chr_path, f"moods_sites_{chr_dir}.tsv")
        dist_path = os.path.join(chr_path, f"genes_near_peaks_{chr_dir}.parquet")
        tg_tensor_path = os.path.join(chr_path, f"tg_tensor_all_{chr_dir}.pt")
        tg_names_path = os.path.join(chr_path, f"tg_names_{chr_dir}.json")

        if not (os.path.exists(moods_path) and os.path.exists(dist_path)
                and os.path.exists(tg_tensor_path) and os.path.exists(tg_names_path)):
            logging.warning(f"Skipping {chr_dir}: missing required files.")
            return None

        # --- load minimal columns from genes_near_peaks ---
        dist_df = pd.read_parquet(dist_path, columns=["peak_id", "target_id", "TSS_dist", "TSS_dist_score"])
        dist_df = dist_df.rename(columns={"target_id": "TG_name"})

        # --- expression correlation (same as before) ---
        tg_tensor = torch.load(tg_tensor_path, map_location="cpu")
        with open(tg_names_path) as f:
            tg_names = json.load(f)

        tg_expr = tg_tensor.numpy()
        tg_centered = tg_expr - tg_expr.mean(axis=1, keepdims=True)
        tg_std = tg_centered.std(axis=1, keepdims=True)
        tf_std_safe = np.clip(tf_std, 1e-8, None)
        tg_std_safe = np.clip(tg_std, 1e-8, None)

        corr_matrix = (tf_centered @ tg_centered.T) / (tf_std_safe * tg_std_safe.T * tf_expr.shape[1])
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        corr_df = pd.DataFrame({
            "TF_name": np.repeat(tf_names, len(tg_names)),
            "TG_name": np.tile(tg_names, len(tf_names)),
            "TF_TG_expr_corr": corr_matrix.ravel(),
        })

        # --- chunked merge / aggregation ---
        chunksize = 2_000_000   # adjust for your memory; ~1–2 M rows per chunk
        agg_chunks = []

        usecols = ["peak_id", "TF", "logodds"]  # minimal columns for merge
        for moods_chunk in pd.read_csv(moods_path, sep="\t", usecols=usecols, chunksize=chunksize):
            moods_chunk = moods_chunk.rename(columns={"TF": "TF_name"})
            merged = moods_chunk.merge(dist_df, on="peak_id", how="inner")

            # aggregate immediately to free memory
            agg_part = (
                merged.groupby(["TF_name", "TG_name"])
                .agg(
                    n_peaks_linking=("peak_id", "nunique"),
                    n_motifs_linking=("logodds", "count"),
                    mean_motif_score=("logodds", "mean"),
                    min_tss_dist=("TSS_dist", "min"),
                    mean_tss_score=("TSS_dist_score", "mean"),
                )
                .reset_index()
            )
            agg_chunks.append(agg_part)
            del merged, moods_chunk, agg_part
            torch.cuda.empty_cache()

        if not agg_chunks:
            logging.warning(f"No merged rows for {chr_dir}")
            return None

        # concatenate all chunk aggregates
        agg = pd.concat(agg_chunks, ignore_index=True)
        del agg_chunks

        # add chromosome ID and merge correlations
        agg["chrom"] = chr_dir
        agg = agg.merge(corr_df, on=["TF_name", "TG_name"], how="left")

        return agg


    # -------------------------------
    # Run in parallel
    # -------------------------------
    logging.info(f"Aggregating TF–TG features using {n_jobs if n_jobs > 0 else os.cpu_count()} cores...")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_chrom)(chr_dir) for chr_dir in chr_dirs
    )

    # Filter out None results
    results = [r for r in results if r is not None]

    if not results:
        raise RuntimeError("No valid chromosome data found — check your input paths.")

    # -------------------------------
    # Combine and finalize
    # -------------------------------
    all_df = pd.concat(results, ignore_index=True)
    all_df["motif_density"] = all_df["n_motifs_linking"] / (all_df["n_peaks_linking"] + 1e-6)
    all_df["log_mean_score"] = np.log1p(all_df["mean_motif_score"])
    all_df["neg_log_tss"] = -np.log1p(all_df["min_tss_dist"])

    logging.info(f"Aggregated features across {len(results)} chromosomes: {all_df.shape}")
    return all_df


# ---------------------------------------------------------------------
# 2. Extract Transformer-based deep features (attention, gradient, etc.)
# ---------------------------------------------------------------------
def build_edge_feature_table(
    model_ckpt, cache_dir, chip_file, out_csv, chrom_id="chr1", batch_size=32
):
    chrom_dir = Path(cache_dir) / chrom_id
    global_vocab_dir = Path(
        "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/training_data_cache/common"
    )

    # ----------------------------------------------------------
    # 1. Load global vocabularies
    # ----------------------------------------------------------
    with open(global_vocab_dir / "tf_vocab.json") as f:
        tf_vocab = json.load(f)
    with open(global_vocab_dir / "tg_vocab.json") as f:
        global_tg_vocab = json.load(f)

    # Per-chromosome TG list/dict
    tg_vocab_path = chrom_dir / f"tg_names_{chrom_id}.json"
    with open(tg_vocab_path) as f:
        tg_vocab_raw = json.load(f)
    if isinstance(tg_vocab_raw, list):
        tg_vocab_dict = {name: i for i, name in enumerate(tg_vocab_raw)}
        tmp_tg_vocab_path = chrom_dir / f"tg_vocab_{chrom_id}.json"
        with open(tmp_tg_vocab_path, "w") as f:
            json.dump(tg_vocab_dict, f)
        tg_vocab_path = tmp_tg_vocab_path
        logging.info(f"Converted TG list to dict: {tg_vocab_path.name}")

    # ----------------------------------------------------------
    # 2. Initialize dataset (chromosome-specific TGs)
    # ----------------------------------------------------------
    dataset = MultiomicTransformerDataset(
        data_dir=Path(cache_dir),
        chrom_id=chrom_id,
        tf_vocab_path=os.path.join(global_vocab_dir, "tf_vocab.json"),
        tg_vocab_path=tg_vocab_path,
        fine_tuner=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MultiomicTransformerDataset.collate_fn,
    )

    # ----------------------------------------------------------
    # 3. Load full trained model (global TG vocab)
    # ----------------------------------------------------------
    model = MultiomicTransformer(
        d_model=384,
        num_heads=6,
        num_layers=3,
        d_ff=768,
        dropout=0.1,
        tf_vocab_size=len(tf_vocab),
        tg_vocab_size=len(global_tg_vocab),  # match checkpoint vocab size
    ).to(DEVICE)

    state_dict = torch.load(model_ckpt, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logging.info(f"Loaded model with {len(global_tg_vocab)} TG embeddings")

    # ----------------------------------------------------------
    # 4. Filter to TGs actually present in this chromosome
    # ----------------------------------------------------------
    tg_names_filtered = [tg for tg in dataset.tg_names if tg in global_tg_vocab]
    if len(tg_names_filtered) < len(dataset.tg_names):
        logging.warning(
            f"Filtered out {len(dataset.tg_names) - len(tg_names_filtered)} unseen TGs for {chrom_id}"
        )

    dataset.tg_names = tg_names_filtered
    dataset.tg_ids = [dataset.tg_name2id[tg] for tg in tg_names_filtered]
    logging.info(f"Using {len(tg_names_filtered)} TGs for {chrom_id} (seen during training)")

    # ----------------------------------------------------------
    # 5. Optional gradient attribution
    # ----------------------------------------------------------
    grad_path = os.path.join(os.path.dirname(out_csv), f"gradient_attribution_{chrom_id}.csv")
    if not os.path.exists(grad_path):
        grad_df = gradient_attribution_matrix(model, dataset, loader, device=DEVICE)
        grad_df.to_csv(grad_path)
    else:
        grad_df = pd.read_csv(grad_path, index_col=0)

    # ----------------------------------------------------------
    # 6. ChIP-seq label edges and extract features
    # ----------------------------------------------------------
    chip = pd.read_csv(chip_file).dropna()
    chip_edges = {(t.upper(), g.upper()) for t, g in zip(chip.iloc[:, 0], chip.iloc[:, 1])}
    tg_id_map = np.arange(len(dataset.tg_names))

    edge_df = extract_edge_features(
        model,
        loader,
        dataset.tf_names,
        dataset.tg_names,
        tg_id_map,
        gradient_attrib_df=grad_df,
        device=DEVICE,
    )

    edge_df.to_csv(out_csv, index=False)
    logging.info(f"Saved combined edge feature table: {out_csv}  shape={edge_df.shape}")
    return edge_df


# ---------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate and extract TF–TG edge features")
    parser.add_argument("--cache_dir", required=True, help="Training data cache directory (per-chromosome files)")
    parser.add_argument("--model_ckpt", required=True, help="Path to trained MultiomicTransformer .pt file")
    parser.add_argument("--chip_file", required=True, help="Path to ChIP-seq CSV ground truth file")
    parser.add_argument("--out_csv", required=True, help="Output CSV path for combined edge features")
    parser.add_argument(
        "--chrom_list",
        default="chr1",
        help="Comma-separated list of chromosomes, e.g. 'chr1,chr2,chr3'"
    )
    parser.add_argument("--num_cpu", required=True, help="Number of cores for processing (one per chromosome)")
    args = parser.parse_args()

    # Parse comma-separated chromosome IDs
    chrom_list = [c.strip() for c in args.chrom_list.split(",") if c.strip()]
    logging.info(f"Processing chromosomes: {chrom_list}")

    # Build the global TF–TG aggregated features once (shared across all chromosomes)
    agg_df = build_global_tf_tg_features(args.cache_dir, chrom_list, int(args.num_cpu))

    all_chrom_results = []

    # Loop over chromosomes
    for chrom_id in chrom_list:
        logging.info(f"--- Processing {chrom_id} ---")

        # 2. Extract model-based features for this chromosome
        edge_df = build_edge_feature_table(
            args.model_ckpt, args.cache_dir, args.chip_file, args.out_csv, chrom_id=chrom_id
        )

        # 3. Merge aggregated & deep features (by TF/TG)
        full_df = edge_df.merge(
            agg_df, left_on=["TF", "TG"], right_on=["TF_name", "TG_name"], how="left"
        )
        full_df["chrom"] = chrom_id
        all_chrom_results.append(full_df)

    # Combine all chromosomes into a single output table
    merged_all = pd.concat(all_chrom_results, ignore_index=True)
    merged_all.to_csv(args.out_csv.replace(".csv", "_full.csv"), index=False)
    logging.info(f"Saved final merged TF–TG feature table across {len(chrom_list)} chromosomes: "
                 f"{args.out_csv.replace('.csv', '_full.csv')}")