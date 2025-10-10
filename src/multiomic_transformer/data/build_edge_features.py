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
    device="cuda", tg_chunk=None
):
    """
    Vectorized edge feature extraction for the MultiomicTransformer.
    Produces a long-format DataFrame with one row per (TF, TG).

    tg_id_map : array mapping raw tg_ids (from dataloader) -> [0 … TG-1]
    """
    model.eval()
    TF, TG = len(tf_names), len(tg_names)

    # --- Accumulators on CPU (to avoid GPU OOM) ---
    attn_acc   = torch.zeros(TG, TF, dtype=torch.float32)   # [TG, TF]
    motif_acc  = torch.zeros(TG, TF, dtype=torch.float32)   # [TG, TF]
    pred_mu    = torch.zeros(TG, dtype=torch.float32)       # [TG]
    pred_sd    = torch.zeros(TG, dtype=torch.float32)       # [TG]
    bias_mu    = torch.zeros(TG, dtype=torch.float32)       # [TG]
    counts     = torch.zeros(TG, dtype=torch.float32)       # [TG]

    for batch in tqdm(dataloader, desc="Extracting edge features"):
        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]

        # --- map raw tg_ids -> global indices ---
        if isinstance(tg_ids, torch.Tensor):
            tg_idx = tg_ids.cpu().numpy()
        else:
            tg_idx = np.array(tg_ids)

        tg_batch_global = torch.as_tensor(
            tg_id_map[tg_idx], device="cpu"
        )

        if tg_batch_global.max() >= TG or tg_batch_global.min() < 0:
            raise ValueError(
                f"tg_batch_global out of bounds! "
                f"max={tg_batch_global.max().item()}, TG={TG}"
            )

        # --- forward pass ---
        with torch.no_grad():
            # --- ensure tg_ids is a torch.LongTensor ---
            if not torch.is_tensor(tg_ids):
                tg_ids = torch.tensor(tg_ids, dtype=torch.long, device=device)
            
            if tg_chunk is None:
                out = model(atac_wins, tf_tensor,
                            tf_ids=tf_ids, tg_ids=tg_ids,
                            bias=bias, motif_mask=motif_mask)
                preds, attn = out if isinstance(out, tuple) else (out, None)

                preds_cpu = preds.detach().cpu()
                attn_cpu  = attn.detach().cpu() if attn is not None else None
                bias_cpu  = bias.mean(dim=(0,2)).detach().cpu()

                # accumulate
                pred_mu[tg_batch_global] += preds_cpu.mean(dim=0)
                pred_sd[tg_batch_global] += preds_cpu.std(dim=0)
                bias_mu[tg_batch_global] += bias_cpu
                counts[tg_batch_global]  += 1

                if attn_cpu is not None:
                    attn_acc[tg_batch_global] += attn_cpu
                if motif_mask is not None:
                    motif_acc[tg_batch_global] += motif_mask.float().cpu()

            else:
                # --- chunk over TGs to reduce memory ---
                for j0 in range(0, tg_ids.shape[0], tg_chunk):
                    j1 = min(j0 + tg_chunk, tg_ids.shape[0])
                    tg_subset = tg_ids[j0:j1]
                    tg_global_subset = tg_batch_global[j0:j1]

                    out = model(atac_wins, tf_tensor,
                                tf_ids=tf_ids, tg_ids=tg_subset,
                                bias=bias, motif_mask=motif_mask)
                    preds, attn = out if isinstance(out, tuple) else (out, None)

                    preds_cpu = preds.detach().cpu()
                    attn_cpu  = attn.detach().cpu() if attn is not None else None
                    bias_cpu  = bias[:, j0:j1, :].mean(dim=(0,2)).detach().cpu()

                    pred_mu[tg_global_subset] += preds_cpu.mean(dim=0)
                    pred_sd[tg_global_subset] += preds_cpu.std(dim=0)
                    bias_mu[tg_global_subset] += bias_cpu
                    counts[tg_global_subset]  += 1

                    if attn_cpu is not None:
                        attn_acc[tg_global_subset] += attn_cpu
                    if motif_mask is not None:
                        motif_acc[tg_global_subset] += motif_mask[:, j0:j1].float().cpu()

        # free GPU memory
        del preds, attn, bias
        torch.cuda.empty_cache()

    # --- Average by counts ---
    denom_TG   = counts.clamp_min(1)
    denom_TG2D = denom_TG.view(-1, 1)

    pred_mu    = (pred_mu / denom_TG)
    pred_sd    = (pred_sd / denom_TG)
    bias_mu    = (bias_mu / denom_TG)
    attn_mean  = (attn_acc / denom_TG2D)
    motif_mean = (motif_acc / denom_TG2D)

    # --- gradient attribution (optional) ---
    if gradient_attrib_df is not None:
        gradient_attrib_df = gradient_attrib_df.reindex(index=tf_names, columns=tg_names, fill_value=0.0)
        grad_attr = torch.tensor(gradient_attrib_df.values.T, dtype=torch.float32)
    else:
        grad_attr = torch.zeros_like(attn_mean)

    # --- build edge DataFrame ---
    tg_index = pd.Index(tg_names, name="TG")
    tf_index = pd.Index(tf_names, name="TF")

    df_attn  = pd.DataFrame(attn_mean.numpy(),  index=tg_index, columns=tf_index)
    df_motif = pd.DataFrame(motif_mean.numpy(), index=tg_index, columns=tf_index)
    df_grad  = pd.DataFrame(grad_attr.numpy(),  index=tg_index, columns=tf_index)

    long_attn  = df_attn.stack().rename("attn").reset_index()
    long_motif = df_motif.stack().rename("motif_mask").reset_index()
    long_grad  = df_grad.stack().rename("grad_attr").reset_index()

    df_tg = pd.DataFrame({
        "TG": tg_names,
        "pred_mean": pred_mu.numpy(),
        "pred_std":  pred_sd.numpy(),
        "bias_mean": bias_mu.numpy(),
    })

    edges = long_attn.merge(long_motif, on=["TG", "TF"], how="left") \
                     .merge(long_grad,  on=["TG", "TF"], how="left") \
                     .merge(df_tg,      on="TG",         how="left")

    # if chip_edges is not None:
    #     chip_set = set((t.upper(), g.upper()) for t, g in chip_edges)
    #     edges["label"] = [(tf.upper(), tg.upper()) in chip_set
    #                       for tf, tg in zip(edges["TF"], edges["TG"])]
    #     edges["label"] = edges["label"].astype(int)

    cols = ["TF", "TG", "attn", "pred_mean", "pred_std", "bias_mean", "grad_attr", "motif_mask"]
    if "label" in edges:
        cols.append("label")
    edges = edges[cols]

    return edges

# ---------------------------------------------------------------------
# 1. Aggregate TF–TG motif/distance/expression features
# ---------------------------------------------------------------------

def build_global_tf_tg_features(cache_dir: str, n_jobs: int = -1) -> pd.DataFrame:
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

    chr_dirs = sorted([d for d in os.listdir(cache_dir) if d.startswith("chr")])
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

        moods_df = pd.read_csv(moods_path, sep="\t").rename(columns={"TF": "TF_name"})
        dist_df = pd.read_parquet(dist_path).rename(columns={"target_id": "TG_name"})
        tg_tensor = torch.load(tg_tensor_path, map_location="cpu")
        with open(tg_names_path) as f:
            tg_names = json.load(f)

        # --- Compute TF–TG correlations (vectorized) ---
        tg_expr = tg_tensor.numpy()
        tg_centered = tg_expr - tg_expr.mean(axis=1, keepdims=True)
        tg_std = tg_centered.std(axis=1, keepdims=True)

        # --- Avoid zero division and NaNs ---
        tf_std_safe = np.clip(tf_std, 1e-8, None)
        tg_std_safe = np.clip(tg_std, 1e-8, None)

        corr_matrix = (tf_centered @ tg_centered.T) / (tf_std_safe * tg_std_safe.T * tf_expr.shape[1])
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        corr_df = pd.DataFrame({
            "TF_name": np.repeat(tf_names, len(tg_names)),
            "TG_name": np.tile(tg_names, len(tf_names)),
            "TF_TG_expr_corr": corr_matrix.ravel(),
        })

        # --- Aggregate motif/distance info ---
        merged = moods_df.merge(dist_df, on="peak_id", how="inner")
        agg = (
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
    agg_df = build_global_tf_tg_features(args.cache_dir, int(args.num_cpu))

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