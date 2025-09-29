import os, sys, torch, joblib, pandas as pd
import numpy as np
import json

sys.path.append("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/transformer")

from scipy.stats import skew, kurtosis
from transformer import MultiomicTransformer
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import roc_auc_score, average_precision_score
from transformer_dataset import MultiomicTransformerDataset
from eval import (
    per_gene_correlation,
    plot_per_gene_correlation_scatterplot,
    plot_gene_correlation_distribution,
    train_classifier
)
from transformer_training import (
    prepare_dataloader,
    
)

SAMPLE_NAME="DS011"
CHROM_ID="chr19"

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
TRANSFORMER_DATA_DIR = os.path.join(PROJECT_DIR, F"dev/transformer/transformer_data/{SAMPLE_NAME}_{CHROM_ID}")
OUTPUT_DIR = os.path.join(PROJECT_DIR, f"output/transformer_testing_output")
TEST_DIR=os.path.join(OUTPUT_DIR, f"best_model_0.82_corr")


def build_gene_features(tf_tg_weights, dataset):
    """
    Create extended per-gene feature set.
    
    Args:
        tf_tg_weights : np.ndarray [num_tfs, num_genes]
        dataset       : MultiomicTransformerDataset (with tg_tensor_all etc.)
    
    Returns:
        DataFrame [num_genes x features]
    """
    tg_expr = dataset.tg_tensor_all.numpy()   # [genes x cells]
    
    # 1. TF->TG weight stats
    tf_sum = tf_tg_weights.sum(axis=0)
    tf_max = tf_tg_weights.max(axis=0)
    tf_mean = tf_tg_weights.mean(axis=0)
    tf_std = tf_tg_weights.std(axis=0)
    tf_median = np.median(tf_tg_weights, axis=0)
    tf_nonzero = (tf_tg_weights != 0).sum(axis=0)
    tf_skew = skew(tf_tg_weights, axis=0, bias=False)
    tf_kurt = kurtosis(tf_tg_weights, axis=0, bias=False)
    
    # 2. Expression stats
    expr_mean = tg_expr.mean(axis=1)
    expr_std = tg_expr.std(axis=1)
    expr_cv = np.divide(expr_std, expr_mean + 1e-8)
    
    # Optionally: ATAC stats if dataset.atac_tensor_all exists
    atac_mean = None
    if hasattr(dataset, "atac_tensor_all"):
        atac_arr = dataset.atac_tensor_all.numpy()
        atac_mean = atac_arr.mean(axis=1)[:len(dataset.tg_names)]  # adapt indexing
    
    # Assemble into DataFrame
    features = {
        "tf_weight_sum": tf_sum,
        "tf_weight_max": tf_max,
        "tf_weight_mean": tf_mean,
        "tf_weight_std": tf_std,
        "tf_weight_median": tf_median,
        "tf_weight_nonzero": tf_nonzero,
        "tf_weight_skew": tf_skew,
        "tf_weight_kurt": tf_kurt,
        "tg_expr_mean": expr_mean,
        "tg_expr_std": expr_std,
        "tg_expr_cv": expr_cv,
    }
    if atac_mean is not None:
        features["atac_mean"] = atac_mean
    
    return pd.DataFrame(features, index=dataset.tg_names)

with open(os.path.join(TEST_DIR, "run_parameters.json"), 'r') as f:
    run_params = json.loads(f.read())

for key, value in run_params.items():
    print(key, value)

TOTAL_EPOCHS=run_params["Epochs"]
BATCH_SIZE=run_params["Batch Size"]

D_MODEL = run_params["d_model"]
NUM_HEADS = run_params["Attention Heads"]
NUM_LAYERS = run_params["Model Layers"]
D_FF = run_params["d_feedforward"]
DROPOUT = run_params["Dropout"]

# Paths
model_path = os.path.join(TEST_DIR, "checkpoint.pt")

# Load the dataset to evaluate
dataset = MultiomicTransformerDataset(
    data_dir=TRANSFORMER_DATA_DIR, 
    chrom_id=CHROM_ID
)

# Load the trained model
model = MultiomicTransformer(
    D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT,
    dataset.num_tf, dataset.num_tg
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

def gradient_attribution_matrix(model, dataset, num_batches=5, device="cuda:0"):
    """
    Compute gradient attribution scores for all TF–TG pairs.
    Returns a TF x TG DataFrame with global min-max scaled scores.
    """
    model.eval()
    _, _, test_loader = prepare_dataloader(dataset, batch_size=32, world_size=1, rank=0)
    
    # Initialize importance accumulator per TG
    importance_dict = {tg: torch.zeros(len(dataset.tf_names), device=device) 
                       for tg in dataset.tg_names}

    for i, (atac_wins, tf_tensor, targets, bias) in enumerate(test_loader):
        if i >= num_batches:
            break

        atac_wins = atac_wins.to(device)
        tf_tensor = tf_tensor.to(device).detach().clone().requires_grad_(True)
        bias = bias.to(device)

        preds = model(atac_wins, tf_tensor, bias=bias)  # [batch, n_genes]

        for tg in dataset.tg_names:
            tg_idx = dataset.tg_names.index(tg)
            tg_pred = preds[:, tg_idx].mean()

            model.zero_grad()
            tg_pred.backward(retain_graph=True)

            importance_dict[tg] += tf_tensor.grad.abs().sum(dim=0)

    # Convert to DataFrame
    tf_importance_df = pd.DataFrame(
        {tg: (importance_dict[tg] / num_batches).detach().cpu().numpy() 
         for tg in dataset.tg_names},
        index=dataset.tf_names
    )

    # Global min-max normalization (not per column)
    min_val, max_val = tf_importance_df.values.min(), tf_importance_df.values.max()
    tf_importance_df = (tf_importance_df - min_val) / (max_val - min_val + 1e-8)

    return tf_importance_df

def evaluate_auc_and_topk(tf_importance_df, chip_edges, k_list=[100, 500, 1000]):
    """
    Compute AUROC, PR-AUC, and top-k precision for TF–TG importance scores vs ground truth.
    
    Args:
        tf_importance_df : pd.DataFrame [TFs x TGs] (importance scores, normalized 0–1)
        chip_edges       : set of (TF, TG) ground truth edges (uppercase)
        k_list           : list of top-k cutoffs to evaluate precision
    
    Returns:
        dict with AUROC, PR-AUC, and precision@k values
    """
    # Restrict TFs and TGs to RN111
    rn111_tfs = set(g1 for g1, _ in chip_edges)
    rn111_tgs = set(g2 for _, g2 in chip_edges)
    tf_importance_df = tf_importance_df.loc[
        tf_importance_df.index.intersection(rn111_tfs),
        tf_importance_df.columns.intersection(rn111_tgs)
    ]

    scores, labels, edges = [], [], []
    for tg in tf_importance_df.columns:
        for tf in tf_importance_df.index:
            score = tf_importance_df.loc[tf, tg]
            label = 1 if (tf.upper(), tg.upper()) in chip_edges else 0
            scores.append(score)
            labels.append(label)
            edges.append((tf, tg))

    positives = []
    for tf in tf_importance_df.index:
        for tg in tf_importance_df.columns:
            if (tf.upper(), tg.upper()) in chip_edges:
                positives.append((tf, tg))
                
    all_tgs_in_chip = set(g2 for _, g2 in chip_edges)
    overlap = all_tgs_in_chip.intersection(set(tf_importance_df.columns))

    print(f"Total TGs in RN111: {len(all_tgs_in_chip)}")
    print(f"TGs overlap on {CHROM_ID} (dataset): {len(tf_importance_df.columns)}")
    print(f"TGs overlap between RN111 and {CHROM_ID} dataset: {len(overlap)}")
    print("Example overlap TGs:", list(overlap)[:20])

    print(f"Found {len(positives)} positive TF–TG edges in evaluation set.")
    print("Example positives:", positives[:10])
    
    print("Overlap TFs:", len(tf_importance_df.index.intersection(rn111_tfs)))
    print("Overlap TGs:", len(tf_importance_df.columns.intersection(rn111_tgs)))

    if len(set(labels)) < 2:
        raise ValueError("Ground truth labels have only one class; AUROC/PR-AUC undefined.")

    # --- AUROC and PR-AUC ---
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    # --- Top-k precision ---
    results = {"AUROC": auroc, "PR-AUC": auprc}
    scored_edges = pd.DataFrame({"tf": [e[0] for e in edges],
                                 "tg": [e[1] for e in edges],
                                 "score": scores,
                                 "label": labels})
    scored_edges = scored_edges.sort_values("score", ascending=False).reset_index(drop=True)

    total_pos = sum(labels)
    total = len(labels)
    print(f"Evaluated on {total} edges ({total_pos} positives, {total_pos/total:.3%})")

    for k in k_list:
        topk = scored_edges.head(k)
        precision_at_k = topk["label"].sum() / len(topk)
        results[f"Precision@{k}"] = precision_at_k

    return results

tf_imp_dir = os.path.join(TEST_DIR, "tf_gradient_attributions")
os.makedirs(tf_imp_dir, exist_ok=True)

# --- Run for your top-50 genes ---
tf_importance_df = gradient_attribution_matrix(model, dataset, num_batches=10, device=device)

# Save results
tf_importance_df.to_csv(os.path.join(tf_imp_dir, "tf_importance_matrix_exp.csv"))
print(tf_importance_df.shape)
print(tf_importance_df.head())

ground_truth_file = os.path.join(PROJECT_DIR, "ground_truth_files/mESC_beeline_ChIP-seq.csv")
chip_df = pd.read_csv(ground_truth_file)
chip_edges = set((g1.capitalize(), g2.capitalize()) for g1, g2 in zip(chip_df["Gene1"], chip_df["Gene2"]))

# --- Run evaluation ---
results = evaluate_auc_and_topk(tf_importance_df, chip_edges, k_list=[100, 500, 1000, 5000])

print(f"AUROC: {results['AUROC']:.4f}")
print(f"PR-AUC: {results['PR-AUC']:.4f}")
for k in [100, 500, 1000, 5000]:
    if f"Precision@{k}" in results:
        print(f"Precision@{k}: {results[f'Precision@{k}']:.3f}")

# model.eval()

# # Prepare DataLoader for test split
# _, _, test_loader = prepare_dataloader(dataset, BATCH_SIZE, world_size=1, rank=0)

# # --- Evaluation ---
# out_prefix = os.path.join(TEST_DIR, "eval_results")

# plot_per_gene_correlation_scatterplot(
#     model, test_loader, gpu_id=0,
#     outpath=out_prefix + "_scatter.png"
# )

# corr_df = per_gene_correlation(model, test_loader, gpu_id=0, gene_names=dataset.tg_names)
# corr_df.to_csv(out_prefix + ".csv", index=False)

# plot_gene_correlation_distribution(corr_df, out_prefix)

# --- Classifier on "learnable genes" ---
# tf_tg_weights = pd.read_csv(os.path.join(TEST_DIR, "tf_tg_weights.csv"), index_col=0)

# gene_features = build_gene_features(tf_tg_weights, dataset)

# # Align labels
# labels = corr_df.set_index("gene").loc[gene_features.index, "label"].to_numpy()

# # Train + evaluate classifier
# train_classifier(gene_features.values, labels, out_prefix)

