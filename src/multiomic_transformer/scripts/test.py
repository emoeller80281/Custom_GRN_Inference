import os, csv, torch, json, logging
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

from config.settings import *
from multiomic_transformer.datasets.dataset import MultiomicTransformerDataset
from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.utils import plotting

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Helpers
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

@torch.no_grad()
def extract_shortcut_matrix(model, dataset, device="cuda:0", normalize="global"):
    """
    Runs a dummy forward to pull out the TF→TG attention (shortcut) matrix.
    Returns DataFrame [TF × TG].
    """
    model.eval()
    
    B, W = 1, 1
    atac_windows = torch.zeros(B, W, 1, device=device)
    tf_expr = torch.zeros(B, len(dataset.tf_names), device=device)
    tf_ids = torch.arange(len(dataset.tf_names), device=device)
    tg_ids = torch.arange(len(dataset.tg_names), device=device)
    
    logits, attn = model(
        atac_windows, tf_expr, tf_ids=tf_ids, tg_ids=tg_ids,
        bias=None, motif_mask=None
    )
    
    mat = attn.detach().cpu().T  # [T, G]
    
    # Apply scale if present
    if hasattr(model, "shortcut") and hasattr(model.shortcut, "scale"):
        mat = model.shortcut.scale.item() * mat
    
    if normalize == "global":
        mat = (mat - mat.min()) / (mat.max() - mat.min() + 1e-8)
    elif normalize == "per_tg":
        mn = mat.min(dim=0, keepdim=True).values
        mx = mat.max(dim=0, keepdim=True).values
        mat = (mat - mn) / (mx - mn + 1e-8)

    return pd.DataFrame(
        mat.numpy(),
        index=dataset.tf_names,
        columns=dataset.tg_names
    )


def evaluate_chip_aucs(tf_imp_df, chip_csv):
    chip = pd.read_csv(chip_csv)
    chip_edges = {(t.upper(), g.upper()) for t, g in zip(chip["Gene1"], chip["Gene2"])}
    tf_imp = tf_imp_df.copy()
    tf_imp.index = [x.upper() for x in tf_imp.index]
    tf_imp.columns = [x.upper() for x in tf_imp.columns]
    scores, labels = [], []
    for tg in tf_imp.columns:
        for tf, score in tf_imp[tg].items():
            scores.append(score)
            labels.append(1 if (tf, tg) in chip_edges else 0)
    if len(set(labels)) < 2:
        return {"AUROC": np.nan, "PR-AUC": np.nan}
    return {
        "AUROC": roc_auc_score(labels, scores),
        "PR-AUC": average_precision_score(labels, scores),
    }

@torch.no_grad()
def _global_minmax_(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)

def zscore_per_cell(x: torch.Tensor, eps=1e-6):
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True).clamp_min(eps)
    return (x - mu) / sd

def gradient_attribution_matrix(model, dataset, loader, tg_chunk=16, device=DEVICE,
                                normalize="global"):
    """
    Returns TF×TG DataFrame of mean |∂TG_j / ∂TF_i| over cells.
    """
    TF = len(dataset.tf_names)
    TG = len(dataset.tg_names)
    acc = torch.zeros(TF, TG, device=device)

    for (atac_wins, tf_tensor, tg_true, bias, tf_ids, tg_ids, motif_mask) in loader:
        atac_wins  = atac_wins.to(device)
        bias       = bias.to(device)
        tf_ids     = tf_ids.to(device)
        tg_ids     = tg_ids.to(device)
        motif_mask = motif_mask.to(device)

        # inputs we want gradients for
        tf_tensor = tf_tensor.to(device).detach().clone().requires_grad_(True)
        tf_norm   = zscore_per_cell(tf_tensor)

        # forward pass
        out = model(atac_wins, tf_norm, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias, motif_mask=motif_mask)
        preds = out[0] if isinstance(out, tuple) else out  # [B, TG_eval]

        B, G_eval = preds.shape
        assert G_eval == TG, "TG dimension mismatch."

        for j0 in range(0, TG, tg_chunk):
            j1 = min(j0 + tg_chunk, TG)
            for j in range(j0, j1):
                out = preds[:, j].mean()
                grad = torch.autograd.grad(out, tf_norm, retain_graph=True)[0]  # [B, TF]
                acc[:, j] += grad.abs().sum(dim=0)

        del preds
        tf_tensor.grad = None

    acc = acc / max(1, len(loader))  # average

    # normalization
    if normalize == "global":
        acc = _global_minmax_(acc)
    elif normalize == "per_tg":
        mn = acc.min(dim=0, keepdim=True).values
        mx = acc.max(dim=0, keepdim=True).values
        acc = (acc - mn) / (mx - mn + 1e-8)

    df = pd.DataFrame(acc.detach().cpu().numpy(),
                      index=dataset.tf_names, columns=dataset.tg_names)
    return df

# ---------------------------------------------------------------------
# Main Test
# ---------------------------------------------------------------------
def run_test(checkpoint_path, out_dir, batch_size=BATCH_SIZE, gpu_id=0, chip_file=None):
    ckpt_dir = os.path.dirname(checkpoint_path)
    dataset = MultiomicTransformerDataset(
        data_dir=SAMPLE_DATA_CACHE_DIR,
        chrom_id=CHROM_ID,
        tf_vocab_path=os.path.join(COMMON_DATA, "tf_vocab.json"),
        tg_vocab_path=os.path.join(COMMON_DATA, "tg_vocab.json"),
    )

    model, run_params = load_model_from_run_params(ckpt_dir, len(dataset.tf_name2id), len(dataset.tg_name2id), DEVICE)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=MultiomicTransformerDataset.collate_fn)

    preds_list, tgts_list = [], []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = [x.to(DEVICE) for x in batch]
            preds, _ = model(atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias, motif_mask=motif_mask)
            total_loss += torch.nn.functional.mse_loss(preds, targets).item(); n_batches += 1
            preds_list.append(preds.cpu().numpy()); tgts_list.append(targets.cpu().numpy())
    val_loss = total_loss / max(1, n_batches)

    all_preds, all_tgts = np.concatenate(preds_list), np.concatenate(tgts_list)
    pearson_corr, _ = pearsonr(all_preds.ravel(), all_tgts.ravel())
    spearman_corr, _ = spearmanr(all_preds.ravel(), all_tgts.ravel())

    logging.info(f"[Test] Loss={val_loss:.4f}, Pearson={pearson_corr:.3f}, Spearman={spearman_corr:.3f}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test_results.csv"), "w") as f:
        csv.writer(f).writerows([["Val Loss","Pearson","Spearman"], [val_loss, pearson_corr, spearman_corr]])

    scatter_fig = plotting.plot_per_gene_correlation_scatterplot(model, loader, use_mask=False, gpu_id=gpu_id)
    scatter_fig.savefig(os.path.join(out_dir, "test_scatter.png"), dpi=300)

    # --- TF-TG shortcut importance ---
    tf_importance_df = extract_shortcut_matrix(model, dataset)
    tf_importance_df.to_csv(os.path.join(out_dir, "shortcut_matrix.csv"))
    
    logging.info("\nGenerating gradient attribution matrix")
    gradient_attrib_df = gradient_attribution_matrix(
        model, dataset, loader, tg_chunk=16, device=DEVICE, normalize="global"
    )
    grad_attrib_out_csv = os.path.join(OUTPUT_DIR, "gradient_attribution.csv")
    gradient_attrib_df.to_csv(grad_attrib_out_csv)
    logging.info(f"\tSaved gradient attribution matrix: {grad_attrib_out_csv}  shape={gradient_attrib_df.shape}")
    
    importance_out_csv = os.path.join(OUTPUT_DIR, "shortcut_matrix.csv")
    tf_importance_df.to_csv(importance_out_csv)
    logging.info(f"\tSaved shortcut matrix: {importance_out_csv}  shape={tf_importance_df.shape}")

    if chip_file:
        results = evaluate_chip_aucs(tf_importance_df, chip_file)
        logging.info(f"CHIP eval: AUROC={results['AUROC']:.3f}, AUPRC={results['PR-AUC']:.3f}")

    return val_loss, pearson_corr, spearman_corr


if __name__ == "__main__":
    ckpt_path = OUTPUT_DIR / "model_training_014" / "trained_model.pt"
    run_test(ckpt_path, OUTPUT_DIR / "model_training_014/test_results", gpu_id=0,
             chip_file="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/ground_truth_files/mESC_beeline_ChIP-seq.csv")
