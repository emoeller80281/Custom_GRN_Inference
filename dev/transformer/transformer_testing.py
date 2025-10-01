# transformer_testing.py
import os, sys, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------
PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
DEV_DIR     = os.path.join(PROJECT_DIR, "dev/transformer")
sys.path.append(DEV_DIR)

from transformer import MultiomicTransformer
from transformer_dataset import MultiomicTransformerDataset
from transformer_training import prepare_dataloader
import eval

SAMPLE_NAME = "mESC"
CHROM_ID    = "chr19"
TRAINING_RUN_NAME = "model_training_01_10_15_15_36"

TRANSFORMER_DATA_DIR = os.path.join(DEV_DIR, f"transformer_data/{SAMPLE_NAME}")
COMMON_DIR           = os.path.join(DEV_DIR, "transformer_data/common")
OUTPUT_DIR           = os.path.join(PROJECT_DIR, "output/transformer_testing_output")
TEST_DIR             = os.path.join(OUTPUT_DIR, f"{SAMPLE_NAME}/{CHROM_ID}/{TRAINING_RUN_NAME}")  # where checkpoint lives

GROUND_TRUTH_FILE = os.path.join(PROJECT_DIR, "ground_truth_files/ORTI_rank1_ground_truth_TF_TG.csv")
OUT_DIR = os.path.join(TEST_DIR, "tf_gradient_attributions")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH  = 32
TG_CHUNK = 64  # how many TGs to do at once when computing grads

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def zscore_per_cell(x: torch.Tensor, eps=1e-6):
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True).clamp_min(eps)
    return (x - mu) / sd

def load_model_and_data():
    # --- run params ---
    with open(os.path.join(TEST_DIR, "run_parameters.json")) as f:
        run_params = json.load(f)

    d_model   = run_params["d_model"]
    n_heads   = run_params["Attention Heads"]
    n_layers  = run_params["Model Layers"]
    d_ff      = run_params["d_feedforward"]
    dropout   = run_params["Dropout"]
    model_pt  = os.path.join(TEST_DIR, "checkpoint.pt")

    # --- dataset ---
    dataset = MultiomicTransformerDataset(
        data_dir=TRANSFORMER_DATA_DIR,
        chrom_id=CHROM_ID,
        tf_vocab_path=os.path.join(COMMON_DIR, "tf_vocab.json"),
        tg_vocab_path=os.path.join(COMMON_DIR, "tg_vocab.json"),
    )
    train_loader, _, test_loader = prepare_dataloader(dataset, batch_size=BATCH, world_size=1, rank=0)

    # --- model ---
    # infer whether shortcut existed in checkpoint
    sd = torch.load(model_pt, map_location=DEVICE)
    use_shortcut = run_params.get("Shortcut L1", None) is not None

    model = MultiomicTransformer(
        d_model=d_model, num_heads=n_heads, num_layers=n_layers,
        d_ff=d_ff, dropout=dropout,
        tf_vocab_size=len(dataset.tf_name2id),
        tg_vocab_size=len(dataset.tg_name2id),
        use_shortcut=use_shortcut,
        use_motif_mask=run_params.get("Motif Mask", "No") == "Yes",
        lambda_l1=run_params.get("Shortcut L1", 0.0),
        lambda_l2=run_params.get("Shortcut L2", 0.0),
        topk=run_params.get("Shortcut Top K", None),
        shortcut_dropout=run_params.get("Shortcut Dropout", 0.0)
    ).to(DEVICE)
    model.load_state_dict(sd, strict=True)
    model.eval()

    return model, dataset, train_loader, test_loader

@torch.no_grad()
def _global_minmax_(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)

@torch.no_grad()
def shortcut_matrix(model, dataset, normalize="global"):
    """
    Extracts the learned TF→TG shortcut matrix directly from the model.

    Returns
    -------
    DataFrame [TF × TG] with importance scores.
    """
    if not hasattr(model, "shortcut") or model.shortcut is None:
        raise ValueError("Model does not have a shortcut connection enabled.")

    # Typically the shortcut weight is [TG, TF]
    W = model.shortcut.weight.detach().cpu().float()  # [TG, TF]

    # Transpose to TF×TG
    mat = W.T

    # Normalize
    if normalize == "global":
        mn, mx = mat.min(), mat.max()
        mat = (mat - mn) / (mx - mn + 1e-8)
    elif normalize == "per_tg":
        mn = mat.min(dim=0, keepdim=True).values
        mx = mat.max(dim=0, keepdim=True).values
        mat = (mat - mn) / (mx - mn + 1e-8)

    df = pd.DataFrame(mat.numpy(),
                      index=dataset.tf_names,
                      columns=dataset.tg_names)
    return df


def gradient_attribution_matrix(model, dataset, loader, tg_chunk=TG_CHUNK, device=DEVICE,
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


def evaluate_chip_aucs(tf_importance_df, chip_csv, k_list=(100, 500, 1000, 5000)):
    """
    Evaluate AUROC / PR-AUC / Precision K against CHIP edges.
    We uppercase both the CHIP and the DF indexing to align.
    """
    chip = pd.read_csv(chip_csv)
    chip_edges = {(t.upper(), g.upper()) for t, g in zip(chip["Gene1"], chip["Gene2"])}

    # Uppercase DF indexing for matching
    tf_imp = tf_importance_df.copy()
    tf_imp.index   = [x.upper() for x in tf_imp.index]
    tf_imp.columns = [x.upper() for x in tf_imp.columns]

    rn111_tfs = {t for t, _ in chip_edges}
    rn111_tgs = {g for _, g in chip_edges}

    tf_imp = tf_imp.loc[tf_imp.index.intersection(rn111_tfs),
                        tf_imp.columns.intersection(rn111_tgs)]
    if tf_imp.empty:
        raise ValueError("No overlap between TF/TG names and CHIP set.")

    scores, labels, edges = [], [], []
    # Flatten
    for tg in tf_imp.columns:
        col = tf_imp[tg]
        for tf, score in col.items():
            scores.append(float(score))
            labels.append(1 if (tf, tg) in chip_edges else 0)
            edges.append((tf, tg))

    if len(set(labels)) < 2:
        raise ValueError("Only one class present after overlap; AUROC/PR-AUC undefined.")

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    # Precision K
    df_scored = pd.DataFrame(edges, columns=["tf", "tg"])
    df_scored["score"] = scores
    df_scored["label"] = labels
    df_scored = df_scored.sort_values("score", ascending=False).reset_index(drop=True)

    results = {"AUROC": auroc, "PR-AUC": auprc, "positives": int(sum(labels)), "edges": int(len(labels))}
    for k in k_list:
        k = int(k)
        if k <= len(df_scored):
            prec_k = df_scored.head(k)["label"].mean()
            results[f"Precision@{k}"] = float(prec_k)

    return results, df_scored

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.info("\nLoading model and dataset")
    model, dataset, train_loader, test_loader = load_model_and_data()
    
    logging.info(f"\nPlotting gene correlation scatterplot")
    fig = eval.plot_per_gene_correlation_scatterplot(
        model, 
        train_loader, 
        DEVICE, 
        outpath=os.path.join(TEST_DIR, "eval_results_scatter.png")
        )

    if getattr(model, "use_shortcut", False):
        logging.info("\nExtracting shortcut TF×TG matrix (no gradients needed)")
        tf_importance_df = shortcut_matrix(model, dataset, normalize="global")
    else:
        logging.info("\nGenerating gradient attribution matrix")
        tf_importance_df = gradient_attribution_matrix(
            model, dataset, test_loader, tg_chunk=TG_CHUNK, device=DEVICE, normalize="global"
        )
    out_csv = os.path.join(OUT_DIR, "tf_importance_matrix.csv")
    tf_importance_df.to_csv(out_csv)
    logging.info(f"\tSaved TF×TG importance matrix: {out_csv}  shape={tf_importance_df.shape}")

    # --- Evaluate vs CHIP edges ---
    logging.info("\nEvaluating AUROC and AUPRC")
    results, scored_edges = evaluate_chip_aucs(tf_importance_df, GROUND_TRUTH_FILE, k_list=(100, 500, 1000, 5000))
    logging.info(f"AUROC = {results['AUROC']:.4f}  |  PR-AUC = {results['PR-AUC']:.4f}  "
          f"| positives={results['positives']} / {results['edges']} edges")
    for k in (100, 500, 1000, 5000):
        key = f"Precision@{k}"
        if key in results:
            logging.info(f"{key}: {results[key]:.3f}")

    # Save scored edges for inspection
    scored_edges_path = os.path.join(OUT_DIR, "scored_edges.tsv")
    scored_edges.to_csv(scored_edges_path, sep="\t", index=False)
    logging.info(f"Wrote ranked edges: {scored_edges_path}")
