import os, sys, torch, joblib, pandas as pd
import numpy as np
import json

sys.path.append("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/transformer")

from scipy.stats import skew, kurtosis
from transformer import MultiomicTransformer
from sklearn.preprocessing import minmax_scale
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

selected_date = "26_09_11_56_06"
PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
TRANSFORMER_DATA_DIR = os.path.join(PROJECT_DIR, "dev/transformer/transformer_data")
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

# Rebuild dataset + model
dataset = MultiomicTransformerDataset(
    data_dir=TRANSFORMER_DATA_DIR, 
    chrom_id="chr19"
)
model = MultiomicTransformer(
    D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT,
    dataset.num_tf, dataset.num_tg
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

top_50_genes = np.load(os.path.join(OUTPUT_DIR, "top_50_expressed_chr19_genes.npy"), allow_pickle=True)

def gradient_attribution_matrix(model, dataset, tg_names, num_batches=5, device="cuda:0"):
    """
    Compute gradient attribution scores for all TF–TG pairs.

    Args:
        model       : trained MultiomicTransformer
        dataset     : MultiomicTransformerDataset
        tg_names    : list of TG names to evaluate
        num_batches : number of batches to average over
        device      : GPU/CPU

    Returns:
        pd.DataFrame [TFs x TGs] with column-wise min-max scaled attribution scores
    """
    model.eval()
    _, _, test_loader = prepare_dataloader(dataset, batch_size=32, world_size=1, rank=0)

    tg_names = [i for i in tg_names if tg_names in dataset.tg_names]
    
    # Initialize importance accumulator per TG
    importance_dict = {tg: torch.zeros(len(dataset.tf_names), device=device) for tg in tg_names}

    for i, (atac_wins, tf_tensor, targets, bias) in enumerate(test_loader):
        if i >= num_batches:
            break

        atac_wins = atac_wins.to(device)
        tf_tensor = tf_tensor.to(device).detach().clone().requires_grad_(True)
        bias = bias.to(device)

        preds = model(atac_wins, tf_tensor, bias=bias)  # [batch, n_genes]

        for tg in tg_names:
            tg_idx = dataset.tg_names.index(tg)
            tg_pred = preds[:, tg_idx].mean()

            model.zero_grad()
            tg_pred.backward(retain_graph=True)

            importance_dict[tg] += tf_tensor.grad.abs().sum(dim=0)

    # Convert to DataFrame
    tf_importance_df = pd.DataFrame(
        {tg: (importance_dict[tg] / num_batches).detach().cpu().numpy() for tg in tg_names},
        index=dataset.tf_names
    )

    # Column-wise min-max normalization
    tf_importance_df = tf_importance_df.apply(lambda col: minmax_scale(col), axis=0, result_type="broadcast")

    return tf_importance_df

tf_imp_dir = os.path.join(TEST_DIR, "tf_gradient_attributions")
os.makedirs(tf_imp_dir, exist_ok=True)

# --- Run for your top-50 genes ---
top_50_genes = np.load(os.path.join(OUTPUT_DIR, "top_50_expressed_chr19_genes.npy"), allow_pickle=True).tolist()
tf_importance_df = gradient_attribution_matrix(model, dataset, tg_names=top_50_genes, num_batches=10, device=device)

# Save results
tf_importance_df.to_csv(os.path.join(tf_imp_dir, "tf_importance_matrix.csv"))
print(tf_importance_df.shape)
print(tf_importance_df.head())


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

