import os
import pandas as pd
import json
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from transformer import MultiomicTransformer
from transformer_dataset import MultiomicTransformerDataset
from eval import (
    per_gene_correlation
)
from transformer_training import (
    prepare_dataloader
)

def train_classifier(feature_matrix, labels):
    
    X = feature_matrix
    y = labels
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    clf = RandomForestClassifier(
        n_estimators=200, 
        random_state=42,
        class_weight="balanced"   # upweight minority class
    )
    clf.fit(X_train_res, y_train_res)

    probs = clf.predict_proba(X_test)[:, 1]
    
    predicted_subset = gene_features.index[probs > 0.5]
    
    predicted_subset.to_series().to_csv(os.path.join(TEST_DIR, "learnable_genes_iter1.csv"))

    roc_score = roc_auc_score(y_test, probs)
    
    return roc_score

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

selected_date = "26_09_11_56_06"
PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
TRANSFORMER_DATA_DIR = os.path.join(PROJECT_DIR, "dev/transformer/transformer_data")

TEST_DIR=os.path.join(PROJECT_DIR, f"output/transformer_testing_output/model_training_{selected_date}")

with open(os.path.join(TEST_DIR, "run_parameters.json"), 'r') as f:
    run_params = json.loads(f.read())

TOTAL_EPOCHS=run_params["Epochs"]
BATCH_SIZE=run_params["Batch Size"]

D_MODEL = run_params["d_model"]
NUM_HEADS = run_params["Attention Heads"]
NUM_LAYERS = run_params["Model Layers"]
D_FF = run_params["d_feedforward"]
DROPOUT = run_params["Dropout"]

# Paths
model_path = os.path.join(TEST_DIR, "final_model.pt")
scaler = joblib.load(os.path.join(TEST_DIR, "tg_scaler.pkl"))

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
model.eval()

# Prepare DataLoader for test split
_, _, test_loader = prepare_dataloader(dataset, BATCH_SIZE, world_size=1, rank=0)

tf_tg_weights = pd.read_csv(os.path.join(TEST_DIR, "tf_tg_weights.csv"), index_col=0)

gene_features = build_gene_features(tf_tg_weights, dataset)

corr_df = per_gene_correlation(model, test_loader, scaler, gpu_id=0, gene_names=dataset.tg_names)

# Align labels
labels = corr_df.set_index("gene").loc[gene_features.index, "label"].to_numpy()

# Train + evaluate classifier
roc_score = train_classifier(gene_features.values, labels)

