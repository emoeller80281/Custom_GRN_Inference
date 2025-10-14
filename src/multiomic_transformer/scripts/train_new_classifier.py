import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from multiomic_transformer.models.tf_tg_classifier import GRN_GAT_Bidirectional

from config.settings import *

# Load Data
df = pd.read_parquet(DATA_DIR / "processed" / "tf_tg_data.parquet")

# Encode TFs and TGs into integer node IDs
tf_encoder = LabelEncoder()
tg_encoder = LabelEncoder()

tf_ids = tf_encoder.fit_transform(df["TF"])
tg_ids = tg_encoder.fit_transform(df["TG"]) + len(tf_encoder.classes_)  # offset TG indices

n_tfs = len(tf_encoder.classes_)
n_tgs = len(tg_encoder.classes_)
n_nodes = n_tfs + n_tgs

print(f"Total nodes: {n_nodes} ({n_tfs} TFs, {n_tgs} TGs)")

# Build edge index
edge_index = torch.tensor(
    [tf_ids.flatten(), tg_ids.flatten()],
    dtype=torch.long
)

# Edge attributes
edge_attr = torch.tensor(df[["reg_potential"]].values, dtype=torch.float32)

# Node features (mean expression)
node_features = torch.zeros((n_nodes, 1), dtype=torch.float32)

tf_expr = df.groupby("TF")["mean_tf_expr"].mean()
tg_expr = df.groupby("TG")["mean_tg_expr"].mean()

for tf, idx in zip(tf_encoder.classes_, range(n_tfs)):
    node_features[idx] = tf_expr[tf]

for tg, idx in zip(tg_encoder.classes_, range(n_tgs)):
    node_features[n_tfs + idx] = tg_expr[tg]


# Labels (per-edge)
labels = torch.tensor(df["label"].values, dtype=torch.float32)

# Get the TF-TG pairs
pairs = torch.stack([torch.tensor(tf_ids), torch.tensor(tg_ids)], dim=1)

# ----- Train / Test / Validate Splits -----
# Convert to numpy for sklearn
pairs_np = pairs.cpu().numpy()
labels_np = labels.cpu().numpy()

# 70% train, 15% val, 15% test — stratified by label
pairs_train, pairs_temp, y_train, y_temp = train_test_split(
    pairs_np, labels_np, test_size=0.3, stratify=labels_np, random_state=42
)
pairs_val, pairs_test, y_val, y_test = train_test_split(
    pairs_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Convert back to tensors
pairs_train = torch.tensor(pairs_train, dtype=torch.long)
pairs_val   = torch.tensor(pairs_val, dtype=torch.long)
pairs_test  = torch.tensor(pairs_test, dtype=torch.long)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val   = torch.tensor(y_val, dtype=torch.float32)
y_test  = torch.tensor(y_test, dtype=torch.float32)

print(f"Train: {len(pairs_train)} | Val: {len(pairs_val)} | Test: {len(pairs_test)}")

# ----- Construct Model, Data, and Objectives -----

data = Data(
    x=node_features,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=labels,
    pairs=pairs
)

# ----- Training Loop -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = GRN_GAT_Bidirectional(
    in_node_feats=1,    # e.g., mean expression
    in_edge_feats=1,    # e.g., reg_potential
    hidden_dim=64,
    heads=4,
    dropout=0.2,
    edge_dropout_p=0.2
).to(device)

# ----- Training loop -----
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()

    data = data.to(device)
    pairs_train, pairs_val = pairs_train.to(device), pairs_val.to(device)
    y_train, y_val = y_train.to(device), y_val.to(device)
    
    logits_train = model(data.x, data.edge_index, data.edge_attr, pairs_train)
    loss = criterion(logits_train, y_train)
    loss.backward()
    optimizer.step()

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        logits_val = model(data.x, data.edge_index, data.edge_attr, pairs_val)
        preds_val = torch.sigmoid(logits_val)
        val_loss = criterion(logits_val, y_val)
        auc_val = roc_auc_score(y_val.cpu(), preds_val.cpu())
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | TrainLoss={loss.item():.4f} | ValLoss={val_loss.item():.4f} | Val AUROC={auc_val:.3f}")

model.eval()
with torch.no_grad():
    logits_test = model(data.x, data.edge_index, data.edge_attr, pairs_test.to(device))
    preds_test = torch.sigmoid(logits_test)
    auc_test = roc_auc_score(y_test.cpu(), preds_test.cpu())
    aupr_test = average_precision_score(y_test.cpu(), preds_test.cpu())
print(f"\nFinal Test AUROC={auc_test:.3f} | AUPR={aupr_test:.3f}")
