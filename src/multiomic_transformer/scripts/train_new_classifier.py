#!/usr/bin/env python3
"""
Two-phase GAT training:
  1. Self-supervised Deep Graph Infomax (DGI)
  2. Semi-supervised fine-tuning on known PKN edges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from collections import defaultdict
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from multiomic_transformer.models.tf_tg_classifier import GRN_GAT_Encoder, EdgeClassifier
from multiomic_transformer.utils.gene_canonicalizer import GeneCanonicalizer
import logging, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from config.settings import *

logging.basicConfig(level=logging.INFO, format="%(message)s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# =========================
# Config
# =========================
SEED = 42
PLOT_DPI = 180
PAIR_BATCH = 131072        # batch size for pair scoring

torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

if ORGANISM_CODE == "mm10":
    species_taxid = "10090"
elif ORGANISM_CODE == "hg38":
    species_taxid = "9606"

canon = GeneCanonicalizer()
canon.load_gtf(str(GTF_FILE_DIR / "Mus_musculus.GRCm39.115.gtf.gz"))
canon.load_ncbi_gene_info(str(NCBI_FILE_DIR / "Mus_musculus.gene_info.gz"), species_taxid=species_taxid)
logging.info(f"Map sizes: {canon.coverage_report()}")
# ============================================================
# Load data
# ============================================================

sample_dfs = []
for sample in SAMPLE_NAMES:
    if os.path.exists(SAMPLE_PROCESSED_DATA_DIR / sample / "tf_tg_data.parquet"):
        logging.info(f"Loading data for {sample}")
        
        sample_df = pd.read_parquet(SAMPLE_PROCESSED_DATA_DIR / sample / "tf_tg_data.parquet")
        logging.info(f"  - Loaded {sample}: {len(sample_df)} edges | {sample_df['TF'].nunique()} TFs | {sample_df['TG'].nunique()} TGs")

        sample_dfs.append(sample_df)
    
df = pd.concat(sample_dfs, ignore_index=True)
logging.info(f"Total: {len(df)} edges | {df['TF'].nunique()} TFs | {df['TG'].nunique()} TGs")

# ---- Edge feature selection ----
base_feats = [
    "reg_potential", "motif_density", "mean_tf_expr", 
    "mean_tg_expr", "expr_product", "motif_present"
    ]

# Include log_reg_pot if present
if "log_reg_pot" in df.columns:
    base_feats.append("log_reg_pot")

# case-insensitive prefix match for extras
extra_numeric = [
    c for c in df.columns
    if (c.lower().startswith("string_") or c.lower().startswith("trrust_") or c.lower().startswith("kegg_"))
    and pd.api.types.is_numeric_dtype(df[c])
]

edge_features = base_feats + extra_numeric
edge_features = list(dict.fromkeys(edge_features))  # dedupe, preserve order
if not edge_features:
    raise ValueError("No edge features found.")

# Ensure we have labels
if "label" not in df.columns:
    if "in_pkn" in df.columns:
        df["label"] = (df["in_pkn"] == 1).astype(int)
        logging.info("Created 'label' from 'in_pkn'.")
    else:
        logging.warning("No 'label' or 'in_pkn' found — Phase 2 will be skipped.")

# Keep only what we need going forward
keep_cols = ["TF", "TG"] + (["label"] if "label" in df.columns else [])
df = df[edge_features + keep_cols]

# ============================================================
# Coerce features to numeric & scale (safe)
# ============================================================

def _flatten_features(feats):
    flat = []
    for x in feats:
        if isinstance(x, (list, tuple, np.ndarray, pd.Index)):
            flat.extend(list(x))
        else:
            flat.append(x)
    out, seen = [], set()
    for f in map(str, flat):
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

edge_features = _flatten_features(edge_features)

missing = [c for c in edge_features if c not in df.columns]
if missing:
    raise ValueError(f"Missing edge feature columns: {missing}")

# Make binary columns float for scaler
if "motif_present" in df.columns:
    df["motif_present"] = pd.to_numeric(df["motif_present"], errors="coerce").fillna(0.0).astype(float)

coerced_info = []
for c in edge_features:
    s = pd.to_numeric(df[c], errors="coerce")
    n_null = int(s.isna().sum())
    df[c] = s.fillna(0.0)
    if n_null:
        coerced_info.append(f"{c}({n_null} NaN→0)")
if coerced_info:
    logging.info("Coerced edge features: " + "; ".join(coerced_info))

# ---- Build pair key and do a group-aware split so identical pairs can't leak
df = df.copy()
df["pair"] = list(zip(df["TF"].astype(str), df["TG"].astype(str)))

from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_idx, test_idx = next(gss.split(df, df["label"], groups=df["pair"]))
df_train = df.iloc[train_idx].reset_index(drop=True)
df_test  = df.iloc[test_idx].reset_index(drop=True)

# ---- Fit scaler on TRAIN ONLY, then transform ALL edges to keep a single graph
scaler = StandardScaler()
scaler.fit(df_train[edge_features].to_numpy())

edge_attr = torch.tensor(
    scaler.transform(df[edge_features].to_numpy()),
    dtype=torch.float32
)
assert edge_attr.shape[0] == len(df)


# ============================================================
# Load & align pretrained TF/TG embeddings by name
# (exported from the transformer run)
# ============================================================
import torch, os

# ---- Load the saved vectors + id→name lists
emb_bundle = torch.load(PRETRAINED_EMB_DIR / "tf_tg_embeddings.pt", map_location="cpu")
lab_bundle = torch.load(PRETRAINED_EMB_DIR / "tf_tg_vocab_id2name.pt", map_location="cpu")
tf_vecs = emb_bundle["tf_emb"]          # [T_src, D]
tg_vecs = emb_bundle["tg_emb"]          # [G_src, D]
D = tf_vecs.shape[1]                    # embedding dim from the source model

src_tf_id2name = lab_bundle["tf_id2name"]  # list[str], index = old id
src_tg_id2name = lab_bundle["tg_id2name"]

# Build name→old_index maps (case-insensitive to be safe)
name_to_old_tf = { (n.upper() if n is not None else None): i
                   for i, n in enumerate(src_tf_id2name) if n is not None }
name_to_old_tg = { (n.upper() if n is not None else None): i
                   for i, n in enumerate(src_tg_id2name) if n is not None }

def _align(encoder_classes, name_to_old, src_vecs, D, init="zeros"):
    out = torch.zeros(len(encoder_classes), D)
    if init == "normal":
        torch.nn.init.normal_(out, mean=0.0, std=0.02)
    hits = misses = 0
    for j, name in enumerate(encoder_classes):
        i = name_to_old.get(str(name).upper(), None)
        if i is None:
            misses += 1
            # keep init for OOV (zeros or normal)
        else:
            out[j] = src_vecs[i]
            hits += 1
    return out, hits, misses

# ---- Build id tensors (reusing encoders already fit on full df, that's fine for a single graph)
tf_encoder = LabelEncoder().fit(df["TF"].astype(str))
tg_encoder = LabelEncoder().fit(df["TG"].astype(str))

n_tfs, n_tgs = len(tf_encoder.classes_), len(tg_encoder.classes_)
n_nodes = n_tfs + n_tgs
logging.info(f"Total nodes: {n_nodes} ({n_tfs} TFs, {n_tgs} TGs)")

tf_ids_all = tf_encoder.transform(df["TF"].astype(str)).astype(np.int64)
tg_ids_all = tg_encoder.transform(df["TG"].astype(str)).astype(np.int64) + n_tfs

pairs_all = torch.tensor(np.stack([tf_ids_all, tg_ids_all], axis=1), dtype=torch.long)
labels_all = torch.tensor(df["label"].astype(int).values, dtype=torch.float32)

pairs_train = pairs_all[train_idx].to(device)
pairs_test  = pairs_all[test_idx].to(device)
y_train     = labels_all[train_idx].float().to(device)
y_test      = labels_all[test_idx].float().to(device)

tf_emb_aligned, tf_hits, tf_miss = _align(tf_encoder.classes_, name_to_old_tf, tf_vecs, D, init="zeros")
tg_emb_aligned, tg_hits, tg_miss = _align(tg_encoder.classes_, name_to_old_tg, tg_vecs, D, init="zeros")

os.makedirs("outputs", exist_ok=True)

print(f"[Emb remap] TF matched {tf_hits}/{len(tf_encoder.classes_)}; "
      f"TG matched {tg_hits}/{len(tg_encoder.classes_)}")

pos_by_tf_ids = defaultdict(list)
neg_by_tf_ids = defaultdict(list)

train_np = pairs_train.cpu().numpy()
y_train_np = y_train.cpu().numpy()

for (tf_id, tg_id), y in zip(train_np, y_train_np):
    if y == 1:
        pos_by_tf_ids[int(tf_id)].append(int(tg_id))
    else:
        neg_by_tf_ids[int(tf_id)].append(int(tg_id))

# Filter to TFs that actually have both classes
valid_tfs = [tf for tf in pos_by_tf_ids if len(neg_by_tf_ids[tf]) > 0]
pos_by_tf_ids = {tf: pos_by_tf_ids[tf] for tf in valid_tfs}
neg_by_tf_ids = {tf: neg_by_tf_ids[tf] for tf in valid_tfs}

# ============================================================
# Build graph
# ============================================================
# Directed TF->TG edges; the encoder will mirror them internally
edge_index_np = np.array([tf_ids_all, tg_ids_all], dtype=np.int64)
edge_index = torch.from_numpy(edge_index_np)

# Node priors: simple expression priors (shape: n_nodes x 1)
tf_expr = df.groupby("TF")["mean_tf_expr"].mean()
tg_expr = df.groupby("TG")["mean_tg_expr"].mean()

node_priors = torch.tensor(
    np.concatenate([
        tf_expr.reindex(tf_encoder.classes_).fillna(0).to_numpy().reshape(-1, 1),
        tg_expr.reindex(tg_encoder.classes_).fillna(0).to_numpy().reshape(-1, 1),
    ]),
    dtype=torch.float32
)
node_features = torch.cat([torch.cat([tf_emb_aligned, tg_emb_aligned], dim=0), node_priors], dim=1)
in_node_feats = D + 1

# ============================================================
# Model & device setup
# ============================================================
# Move to device
node_features = node_features.to(device)

data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr).to(device)

model = GRN_GAT_Encoder(
    in_node_feats=in_node_feats,
    in_edge_feats=len(edge_features),
    hidden_dim=D_MODEL,
    heads=GAT_HEADS,
    dropout=GAT_DROPOUT,
    edge_dropout_p=EDGE_DROPOUT
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Sanity-check devices
logging.info(
    f"Device check → x:{data.x.device}, edge_index:{data.edge_index.device}, "
    f"edge_attr:{data.edge_attr.device}, model:{next(model.parameters()).device}"
)

# ============================================================
# Utility functions for self-supervised DGI training
# ============================================================

def corruption(x: torch.Tensor) -> torch.Tensor:
    """
    Corrupt node features by shuffling rows.
    Used in Deep Graph Infomax to create negative samples.
    """
    idx = torch.randperm(x.size(0))
    return x[idx]


def infomax_loss(h_real: torch.Tensor,
                 h_fake: torch.Tensor,
                 g_real: torch.Tensor,
                 EPS: float = 1e-8) -> torch.Tensor:
    """
    Deep Graph Infomax mutual-information objective.
    Encourages agreement between node embeddings (h_real)
    and the global summary vector (g_real) while penalizing
    agreement with corrupted features (h_fake).
    """
    # Global discriminator scores
    score_real = torch.sum(h_real * g_real, dim=1)
    score_fake = torch.sum(h_fake * g_real, dim=1)

    # Binary cross-entropy losses
    loss_real = -torch.log(torch.sigmoid(score_real) + EPS).mean()
    loss_fake = -torch.log(1 - torch.sigmoid(score_fake) + EPS).mean()

    return loss_real + loss_fake


# ============================================================
# Phase 1 — Self-supervised Deep Graph Infomax (DGI)
# ============================================================
phase1_losses = []

for epoch in range(1, DGI_EPOCHS+1):
    model.train()
    optimizer.zero_grad()
    h_real, g_real = model(data.x, data.edge_index, data.edge_attr)
    h_fake, _ = model(corruption(data.x), data.edge_index, data.edge_attr)
    loss = infomax_loss(h_real, h_fake, g_real)
    loss.backward()
    optimizer.step()
    phase1_losses.append(loss.item())
    
    if epoch % 10 == 0:
        logging.info(f"[DGI] Epoch {epoch:03d} | Loss={loss.item():.6f}")


torch.save(model.state_dict(), "outputs/self_supervised_gat.pt")
logging.info("Saved pretrained weights to outputs/self_supervised_gat.pt")

# ============================================================
# Phase 2 — Semi-supervised fine-tuning using label edges
# ============================================================
class PerTFBalancedSampler:
    def __init__(self, pos_by_tf, neg_by_tf, batch_tfs=64, k_per_tf=1, seed=SEED):
        self.pos_by_tf = {tf: v for tf, v in pos_by_tf.items() if len(neg_by_tf.get(tf, [])) > 0}
        self.neg_by_tf = neg_by_tf
        self.tfs = list(self.pos_by_tf.keys())
        self.batch_tfs = batch_tfs
        self.k = k_per_tf
        self.rng = random.Random(seed)

    def __iterseqs__(self):
        self.rng.shuffle(self.tfs)
        for i in range(0, len(self.tfs), self.batch_tfs):
            chunk = self.tfs[i:i+self.batch_tfs]
            batch = []
            for tf in chunk:
                pos_tgs = self.pos_by_tf[tf]
                neg_tgs = self.neg_by_tf[tf]
                for _ in range(self.k):
                    if not pos_tgs or not neg_tgs: continue
                    batch.append((tf, self.rng.choice(pos_tgs), 1))
                    batch.append((tf, self.rng.choice(neg_tgs), 0))
            yield batch

    # PyTorch-style iterator:
    def __iter__(self):
        yield from self.__iterseqs__()


if "label" not in df.columns:
    logging.warning("No 'label' column found — skipping fine-tuning phase.")
    exit()

logging.info("=== Phase 2: Semi-supervised fine-tuning on PKN edges ===")

embed_dim = D_MODEL
finetune_model = EdgeClassifier(model, embed_dim).to(device)
# Unfreeze only the last GAT + projection head
for n, p in finetune_model.encoder.named_parameters():
    p.requires_grad = ("gat2" in n) or ("proj" in n)

# Different LRs: small for encoder, larger for head
param_groups = [
    {"params": [p for n, p in finetune_model.encoder.named_parameters() if p.requires_grad], "lr": LR_ENCODER},
    {"params": finetune_model.classifier.parameters(), "lr": LR_HEAD},
]
optimizer_finetune = torch.optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)
sampler = PerTFBalancedSampler(pos_by_tf_ids, neg_by_tf_ids, batch_tfs=64, k_per_tf=4, seed=SEED)
criterion = nn.BCEWithLogitsLoss()
phase2_train_losses, phase2_val_auroc, phase2_val_aupr = [], [], []

pretrained_state = {k: v.detach().clone() for k, v in finetune_model.encoder.state_dict().items()}

def l2sp_loss(module, ref_state):
    d = next(module.parameters()).device
    reg = torch.zeros((), device=d)
    for k, p in module.state_dict().items():
        if p.dtype.is_floating_point and k in ref_state:
            reg = reg + (p - ref_state[k].to(d)).pow(2).sum()
    return reg

for epoch in range(1, FINETUNE_EPOCHS + 1):
    finetune_model.train()
    running, nb = 0.0, 0
    for batch in sampler:
        if len(batch) < 4: # require the batch to have at least 2 pos/neg pairs total
            continue
        # batch is a list[(tf_id, tg_id, y)]
        arr = np.array(batch, dtype=np.int64)
        pairs_b = torch.tensor(arr[:, :2], dtype=torch.long, device=device)
        y_b     = torch.tensor(arr[:, 2],  dtype=torch.float32, device=device)

        optimizer_finetune.zero_grad()
        logits = finetune_model(data.x, data.edge_index, data.edge_attr, pairs_b)
        loss = criterion(logits, y_b) + L2SP_LAMBDA * l2sp_loss(finetune_model.encoder, pretrained_state)
        loss.backward(); optimizer_finetune.step()
        running += float(loss.item()); nb += 1

    if nb:
        phase2_train_losses.append(running / nb)

    if epoch % 10 == 0:
        finetune_model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(finetune_model(data.x, data.edge_index, data.edge_attr, pairs_test))
            auc = roc_auc_score(y_test.cpu(), preds.cpu())
            aupr = average_precision_score(y_test.cpu(), preds.cpu())
            phase2_val_auroc.append(auc); phase2_val_aupr.append(aupr)
        logging.info(f"[FineTune] Epoch {epoch:03d} | Loss={phase2_train_losses[-1]:.4f} | AUROC={auc:.3f} | AUPR={aupr:.3f}")
torch.save(finetune_model.state_dict(), "outputs/fine_tuned_gat_classifier.pt")
logging.info("Saved fine-tuned model to outputs/fine_tuned_gat_classifier.pt")

# --- use the SAME encoders used for the graph ---
# make dicts from the fitted LabelEncoders
tf_to_local = {name: i for i, name in enumerate(tf_encoder.classes_)}
tg_to_local = {name: i for i, name in enumerate(tg_encoder.classes_)}

df_scores = df[["TF", "TG"]].copy()
df_scores["TF"] = df_scores["TF"].astype(str).str.upper().str.strip()
df_scores["TG"] = df_scores["TG"].astype(str).str.upper().str.strip()

# map to local node ids; TGs must be offset by +n_tfs
tf_local = df_scores["TF"].map(tf_to_local)
tg_local = df_scores["TG"].map(tg_to_local)

valid = tf_local.notna() & tg_local.notna()
pairs_local = np.stack(
    [tf_local[valid].astype(int).to_numpy(),
     tg_local[valid].astype(int).to_numpy() + n_tfs],  # <-- offset
    axis=1
)
pairs_local = torch.as_tensor(pairs_local, dtype=torch.long, device=device)

# sanity checks
assert pairs_local[:,0].min().item() >= 0 and pairs_local[:,0].max().item() < n_tfs
assert pairs_local[:,1].min().item() >= n_tfs and pairs_local[:,1].max().item() < (n_tfs + n_tgs)

# run batched inference; align back to rows
probs = np.full(len(df_scores), np.nan, dtype=np.float32)
finetune_model.eval()
with torch.no_grad():
    B = PAIR_BATCH
    outs = []
    for i in range(0, pairs_local.shape[0], B):
        sl = pairs_local[i:i+B]
        logits = finetune_model(data.x, data.edge_index, data.edge_attr, sl)
        outs.append(logits.detach().cpu().numpy())
    logits = np.concatenate(outs, axis=0)
    probs_valid = 1.0 / (1.0 + np.exp(-logits))
probs[valid.values] = probs_valid.ravel()

df_scores["Score"] = probs

# all TFxTG combinations (using the same local-id scheme)
tf_ids = np.arange(n_tfs, dtype=int)
tg_ids = np.arange(n_tgs, dtype=int) + n_tfs  # offset TG node ids

all_pairs = np.stack(
    [np.repeat(tf_ids, n_tgs), np.tile(tg_ids, n_tfs)],
    axis=1
)
all_pairs_t = torch.as_tensor(all_pairs, dtype=torch.long, device=device)

# run batched inference on ALL pairs
finetune_model.eval()
outs = []
with torch.no_grad():
    B = PAIR_BATCH
    for i in range(0, all_pairs_t.shape[0], B):
        sl = all_pairs_t[i:i+B]
        logits = finetune_model(data.x, data.edge_index, data.edge_attr, sl)
        outs.append(logits.detach().cpu().numpy())
all_logits = np.concatenate(outs, axis=0)
all_probs = 1.0 / (1.0 + np.exp(-all_logits)).ravel()

# make a nice table with names
tf_names = tf_encoder.classes_.astype(str)
tg_names = tg_encoder.classes_.astype(str)
df_all = pd.DataFrame({
    "TF": np.repeat(tf_names, n_tgs),
    "TG": np.tile(tg_names, n_tfs),
    "Score": all_probs.astype(np.float32),
})
df_all.to_csv("outputs/inferred_grn_all_pairs.csv", index=False)

# ============================================================
# ===  ChIP-seq comparison (TF–TG gold edges)              ===
# ============================================================
# Expect a file with first two columns = TF, TG (case-insensitive).

def _load_chip_gold(gt_path: str, sep: str) -> pd.DataFrame:
    gt = pd.read_csv(gt_path, sep=sep)
    # assume first two columns are TF, TG
    gt = gt.rename(columns={gt.columns[0]: "TF", gt.columns[1]: "TG"})
    # If no label column, treat all as positives
    if "label" not in gt.columns:
        logging.warning("No 'label' column in ChIP file — assuming all listed edges are positives.")
        gt["label"] = 1.0
    gt["label"] = pd.to_numeric(gt["label"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    # strip/upper only (true canonicalization happens later once)
    for c in ("TF","TG"):
        gt[c] = gt[c].astype(str).str.upper().str.strip()
    gt = gt.dropna(subset=["TF","TG"])
    gt = gt[(gt["TF"]!="") & (gt["TG"]!="")]
    return gt[["TF", "TG", "label"]]


def _precision_at_k_over_tfs(score_df: pd.DataFrame, gold_map: dict, ks=PRECISION_AT_K):
    rows = []
    for k in ks:
        vals = []
        for tf, pos_set in gold_map.items():
            sub = score_df[score_df["TF"] == tf]
            if sub.empty:
                continue
            s = sub.sort_values("Score", ascending=False).head(k)
            vals.append(s["TG"].isin(pos_set).mean())
        rows.append((k, float(np.mean(vals)) if vals else np.nan))
    return pd.DataFrame(rows, columns=["k","Precision"])

scored = pd.DataFrame()  # <--- guard for later sections

try:
    if CHIP_GT_PATH and os.path.exists(CHIP_GT_PATH):
        chip_dir = os.path.join("outputs", "chip_eval")
        os.makedirs(chip_dir, exist_ok=True)
        logging.info(f"\n=== ChIP-seq comparison using: {CHIP_GT_PATH} ===")
        chip = _load_chip_gold(CHIP_GT_PATH, CHIP_GT_SEP)

        # --- BEFORE standardization (true pre-std diagnostics)
        scores_before = df_scores[["TF","TG"]].copy()
        chip_before   = chip[["TF","TG"]].copy()
        tf_chip_pre = set(chip_before["TF"].astype(str).str.upper().str.strip().unique())
        tg_chip_pre = set(chip_before["TG"].astype(str).str.upper().str.strip().unique())
        tf_mod_pre  = set(scores_before["TF"].astype(str).str.upper().str.strip().unique())
        tg_mod_pre  = set(scores_before["TG"].astype(str).str.upper().str.strip().unique())
        logging.info("[Pre-Std] TF overlap=%d  TG overlap=%d",
                     len(tf_chip_pre & tf_mod_pre), len(tg_chip_pre & tg_mod_pre))

        # --- canonicalize BOTH tables exactly once
        df_scores = canon.standardize_df(df_scores, tf_col="TF", tg_col="TG")
        chip      = canon.standardize_df(chip,       tf_col="TF", tg_col="TG")

        # change accounting
        def changed_count(before_s, after_s):
            b = before_s.astype(str).str.upper().str.strip()
            a = after_s.astype(str).str.upper().str.strip()
            return int((b != a).sum()), int(len(b))
        tf_chg_scores, tf_tot_scores = changed_count(scores_before["TF"], df_scores["TF"])
        tg_chg_scores, tg_tot_scores = changed_count(scores_before["TG"], df_scores["TG"])
        logging.info("Canonicalizer changed %d/%d TFs and %d/%d TGs in model scores",
                     tf_chg_scores, tf_tot_scores, tg_chg_scores, tg_tot_scores)
        tf_chg_chip, tf_tot_chip = changed_count(chip_before["TF"], chip["TF"])
        tg_chg_chip, tg_tot_chip = changed_count(chip_before["TG"], chip["TG"])
        logging.info("Canonicalizer changed %d/%d TFs and %d/%d TGs in ChIP table",
                     tf_chg_chip, tf_tot_chip, tg_chg_chip, tg_tot_chip)

        # dedupe after std
        df_scores = df_scores.drop_duplicates(subset=["TF","TG"], keep="first").reset_index(drop=True)
        chip      = chip.drop_duplicates(subset=["TF","TG"], keep="first").reset_index(drop=True)

        # model universe (encoder vocab) — optionally canonicalize these too
        tf_vocab = set(map(str, tf_encoder.classes_))
        tg_vocab = set(map(str, tg_encoder.classes_))
        # If your encoders were not trained on canonicalized symbols, do:
        # tf_vocab = {canon.to_symbol(x) for x in tf_encoder.classes_}
        # tg_vocab = {canon.to_symbol(x) for x in tg_encoder.classes_}
        chip_in_universe = chip[chip["TF"].isin(tf_vocab) & chip["TG"].isin(tg_vocab)].copy()

        # AFTER-std overlap logs (the ones you wanted to keep)
        tf_chip = set(chip["TF"].unique()); tg_chip = set(chip["TG"].unique())
        tf_mod  = set(df_scores["TF"].unique()); tg_mod  = set(df_scores["TG"].unique())
        logging.info("[Post-Std] TF overlap=%d  TG overlap=%d",
                     len(tf_chip & tf_mod), len(tg_chip & tg_mod))

        unmapped_tf = sorted(list(tf_chip - tf_mod))[:20]
        unmapped_tg = sorted(list(tg_chip - tg_mod))[:20]
        if unmapped_tf:
            logging.info("Example TFs present in ChIP but not in model after std: %s",
                        ", ".join(unmapped_tf[:10]))
        if unmapped_tg:
            logging.info("Example TGs present in ChIP but not in model after std: %s",
                        ", ".join(unmapped_tg[:10]))

        # merge restricted to the model’s universe
        scored = df_scores.merge(chip_in_universe, on=["TF","TG"], how="left")
        scored["label"] = scored["label"].fillna(0).astype(int)
                    
        # === Coverage diagnostics ===
        tf_chip = set(chip["TF"].unique())
        tg_chip = set(chip["TG"].unique())
        tf_model = set(df_scores["TF"].unique())
        tg_model = set(df_scores["TG"].unique())

        logging.info(f"[ChIP] TFs in file: {len(tf_chip)}, TGs in file: {len(tg_chip)}")
        logging.info(f"[Model] TFs in graph: {len(tf_model)}, TGs in graph: {len(tg_model)}")
        logging.info(f"[Overlap] TFs: {len(tf_chip & tf_model)} | TGs: {len(tg_chip & tg_model)}")

        pos_pairs_total = (chip["label"] == 1).sum()
        pos_pairs_in_model = chip.merge(df_scores[["TF","TG"]], on=["TF","TG"], how="inner")
        pos_pairs_in_model = pos_pairs_in_model[pos_pairs_in_model["label"] == 1]
        logging.info(f"[Positives] Total in ChIP file: {pos_pairs_total} | Present in model edges: {len(pos_pairs_in_model)}")

        # Per-TF positive coverage
        tf_pos_counts = chip[chip["label"]==1].groupby("TF").size().rename("n_pos_chip")
        tf_pos_in_model = pos_pairs_in_model.groupby("TF").size().rename("n_pos_in_model")
        tf_cov = pd.concat([tf_pos_counts, tf_pos_in_model], axis=1).fillna(0).astype(int)
        tf_cov["coverage_pct"] = 100 * tf_cov["n_pos_in_model"] / tf_cov["n_pos_chip"].replace(0, np.nan)
        tf_cov.sort_values("coverage_pct", ascending=True).to_csv(os.path.join(chip_dir, "tf_positive_coverage.csv"))
        logging.info(f"Saved TF positive coverage → {os.path.join(chip_dir, 'tf_positive_coverage.csv')}")

        # Overall metrics (if both classes present)
        chip_dir = os.path.join("outputs", "chip_eval")
        os.makedirs(chip_dir, exist_ok=True)
        scored.to_csv(os.path.join(chip_dir, "scored_edges_with_chip.csv"), index=False)

        y_true = scored["label"].values
        y_pred = scored["Score"].values
        if np.unique(y_true).size > 1:
            auroc = roc_auc_score(y_true, y_pred)
            aupr  = average_precision_score(y_true, y_pred)
            logging.info(f"[ChIP Overall] AUROC={auroc:.3f} | AUPR={aupr:.3f}")

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            prec, rec, _ = precision_recall_curve(y_true, y_pred)

            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--',c='gray')
            plt.title(f"ChIP ROC (AUROC={auroc:.3f})"); plt.xlabel("FPR"); plt.ylabel("TPR")
            plt.subplot(1,2,2); plt.plot(rec,prec)
            plt.title(f"ChIP PR (AUPR={aupr:.3f})"); plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.tight_layout(); plt.savefig(os.path.join(chip_dir, "overall_roc_pr.png"), dpi=PLOT_DPI); plt.close()
        else:
            logging.warning("ChIP labels contain a single class — skipping overall AUROC/AUPR.")

        # Per-TF metrics and P@k
        gold_map = (
            scored.loc[scored["label"] == 1, ["TF","TG"]]
            .groupby("TF")["TG"].apply(lambda s: set(s.astype(str))).to_dict()
        )

        per_rows = []
        for tf, pos_set in gold_map.items():
            sub = scored[scored["TF"] == tf]
            if sub.empty: 
                continue
            y = sub["TG"].isin(pos_set).astype(int).values
            s = sub["Score"].values
            if y.sum() == 0 or y.sum() == len(y):
                auc_tf, aupr_tf = np.nan, np.nan
            else:
                auc_tf = roc_auc_score(y, s)
                aupr_tf = average_precision_score(y, s)
            order = np.argsort(-s)
            p50  = y[order[:50]].mean()  if len(y) >= 50  else y[order].mean()
            p100 = y[order[:100]].mean() if len(y) >= 100 else y[order].mean()
            per_rows.append({"TF": tf, "n": int(len(y)), "n_pos": int(y.sum()),
                             "AUROC": auc_tf, "AUPR": aupr_tf, "P@50": p50, "P@100": p100})

        per_tf = pd.DataFrame(per_rows).sort_values(["AUPR","AUROC"], ascending=False)
        per_tf.to_csv(os.path.join(chip_dir, "per_TF_metrics.csv"), index=False)
        logging.info(f"Saved per-TF ChIP metrics for {len(per_tf)} TFs → {os.path.join(chip_dir,'per_TF_metrics.csv')}")

        # Precision@k curve across TFs
        p_at_k = _precision_at_k_over_tfs(scored.rename(columns={"prob":"Score"}) if "prob" in scored.columns else scored,
                                          gold_map)
        p_at_k.to_csv(os.path.join(chip_dir, "precision_at_k.csv"), index=False)
        plt.figure(figsize=(6,4))
        plt.plot(p_at_k["k"], p_at_k["Precision"], marker="o")
        plt.xscale("log"); plt.xlabel("k"); plt.ylabel("Precision"); plt.title("Precision@k (avg over TFs)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(chip_dir, "precision_at_k.png"), dpi=PLOT_DPI); plt.close()
        
        def lift_at_k(y, s, k):
            order = np.argsort(-s)
            top = y[order[:k]]
            base = y.mean() + 1e-12
            return float(top.mean() / base)

        def recall_at_k(y, s, k):
            order = np.argsort(-s)
            top_pos = (y[order[:k]] == 1).sum()
            total_pos = int((y == 1).sum())
            return float(top_pos / max(total_pos, 1))

        rows = []
        for k in (10, 20, 50, 100, 200, 500):
            lifts, recalls = [], []
            for tf, sub in scored.groupby("TF"):
                if sub["label"].sum() == 0:
                    continue
                y = sub["label"].to_numpy()
                s = sub["Score"].to_numpy()
                lifts.append(lift_at_k(y, s, min(k, len(sub))))
                recalls.append(recall_at_k(y, s, min(k, len(sub))))
            rows.append({"k": k, "Lift@k": np.mean(lifts) if lifts else np.nan,
                            "Recall@k": np.mean(recalls) if recalls else np.nan})
        pd.DataFrame(rows).to_csv(os.path.join(chip_dir, "lift_recall_at_k.csv"), index=False)
    else:
        logging.warning("CHIP_GT_PATH not set or file not found — skipping ChIP comparison.")
except Exception as e:
    logging.exception(f"ChIP-seq comparison failed: {e}")
    
# ============================================================
# === Balanced scoring: macro per-TF and balanced micro     ===
# ============================================================

# 1) Macro per-TF (each TF contributes equally via simple mean)
valid_rows = []
for tf, sub in scored.groupby("TF"):
    # Need both classes to compute AUCs
    if sub["label"].nunique() < 2:
        continue
    y = sub["label"].to_numpy()
    s = sub["Score"].to_numpy()
    try:
        auc_tf  = roc_auc_score(y, s)
        aupr_tf = average_precision_score(y, s)
    except Exception:
        auc_tf, aupr_tf = np.nan, np.nan
    valid_rows.append({"TF": tf, "AUROC": auc_tf, "AUPR": aupr_tf,
                       "n": int(len(sub)), "n_pos": int(sub["label"].sum())})

macro_df = pd.DataFrame(valid_rows)
macro_auroc = float(np.nanmean(macro_df["AUROC"])) if not macro_df.empty else np.nan
macro_aupr  = float(np.nanmean(macro_df["AUPR"]))  if not macro_df.empty else np.nan
logging.info(f"[ChIP Macro] Mean per-TF AUROC={macro_auroc:.3f} | AUPR={macro_aupr:.3f}")
macro_df.to_csv(os.path.join(chip_dir, "per_TF_macro_metrics.csv"), index=False)

# 2) Balanced micro using per-TF sample weights
#    Each TF contributes the same total weight; within a TF, positives and negatives
#    split weight 50/50 (and are spread equally over their counts).
def per_tf_balanced_weights(df_tf):
    # Get the length of the TF positives and negatives
    n = len(df_tf)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    
    # Get the number of positive and negatives
    n_pos = int(df_tf["label"].sum())
    n_neg = n - n_pos
    
    # Set up a number of weight for each 
    w = np.zeros(n, dtype=np.float64)
    if n_pos == 0 and n_neg == 0:
        return w
    
    # Within-TF: allocate 0.5 weight mass to each class (if present)
    if n_pos > 0:
        w[df_tf["label"].values == 1] = 0.5 / n_pos
        
    if n_neg > 0:
        w[df_tf["label"].values == 0] = 0.5 / n_neg
        
    # Will rescale across TFs so each TF sums to 1/N_tf later
    return w

parts, weights = [], []
for tf, sub in scored.groupby("TF"):
    if sub.empty:
        continue
    w_tf = per_tf_balanced_weights(sub)
    if w_tf.sum() == 0:
        continue
    parts.append(sub[["label","Score"]].reset_index(drop=True))
    weights.append(w_tf)

if parts:
    blended = pd.concat(parts, axis=0, ignore_index=True)
    w = np.concatenate(weights, axis=0)
    
    # Give each TF the same total weight: divide by #TFs that contributed
    n_tfs_contributing = len(weights)
    w = w / n_tfs_contributing
    y_all = blended["label"].to_numpy()
    s_all = blended["Score"].to_numpy()
    try:
        bal_micro_auroc = roc_auc_score(y_all, s_all, sample_weight=w)
        bal_micro_aupr  = average_precision_score(y_all, s_all, sample_weight=w)
        logging.info(f"[ChIP Balanced-Micro] AUROC={bal_micro_auroc:.3f} | AUPR={bal_micro_aupr:.3f}  "
                     f"(TF-equal + class-balanced within TF)")
        with open(os.path.join(chip_dir, "balanced_micro_summary.txt"), "w") as f:
            f.write(f"Balanced-Micro AUROC={bal_micro_auroc:.6f}\n")
            f.write(f"Balanced-Micro AUPR={bal_micro_aupr:.6f}\n")
            f.write(f"TFs contributing: {n_tfs_contributing}\n")
            
        # y_all, s_all, w already defined above in the balanced-micro section
        fpr_w, tpr_w, _ = roc_curve(y_all, s_all, sample_weight=w)
        prec_w, rec_w, _ = precision_recall_curve(y_all, s_all, sample_weight=w)

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(fpr_w, tpr_w)
        plt.plot([0,1],[0,1],'--',c='gray',lw=1)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"Balanced-Micro ROC (AUROC={bal_micro_auroc:.3f})")

        plt.subplot(1,2,2)
        plt.plot(rec_w, prec_w)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"Balanced-Micro PR (AUPR={bal_micro_aupr:.3f})")

        plt.tight_layout()
        plt.savefig(os.path.join(chip_dir, "balanced_micro_roc_pr.png"), dpi=PLOT_DPI)
        plt.close()

    except Exception as e:
        logging.warning(f"Balanced-micro scoring failed: {e}")
else:
    logging.warning("No TF groups available for balanced-micro scoring.")

# 3) (Optional) Undersampled micro (equal #pos=#neg per TF) – sanity check
#    This avoids weighting but keeps strict balance per TF by subsampling.
try:
    us_parts = []
    rng = np.random.default_rng(42)
    for tf, sub in scored.groupby("TF"):
        pos = sub[sub["label"] == 1]
        neg = sub[sub["label"] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        k = min(len(pos), len(neg))
        pos_s = pos.sample(n=k, random_state=42) if len(pos) > k else pos
        neg_s = neg.sample(n=k, random_state=42) if len(neg) > k else neg
        us_parts.append(pd.concat([pos_s, neg_s], axis=0))
    if us_parts:
        us = pd.concat(us_parts, axis=0, ignore_index=True)
        y_us = us["label"].to_numpy()
        s_us = us["Score"].to_numpy()
        auroc_us = roc_auc_score(y_us, s_us) if np.unique(y_us).size > 1 else np.nan
        aupr_us  = average_precision_score(y_us, s_us) if np.unique(y_us).size > 1 else np.nan
        logging.info(f"[ChIP Undersampled-Micro] AUROC={auroc_us:.3f} | AUPR={aupr_us:.3f}  "
                     f"(per-TF pos=neg subsample)")
        us.loc[:, ["TF","TG","Score","label"]].to_csv(os.path.join(chip_dir, "undersampled_micro_eval_table.csv"), index=False)
    else:
        logging.warning("No TFs had both classes for undersampled-micro scoring.")
except Exception as e:
    logging.warning(f"Undersampled-micro scoring failed: {e}")

# Fair-scope overall: only TFs with ≥1 positive, and only edges for those TFs
sc_tfs = [tf for tf, grp in scored.groupby("TF") if (grp["label"]==1).any()]
scoped = scored[scored["TF"].isin(sc_tfs)].copy()
if scoped["label"].nunique() > 1:
    auroc_sc = roc_auc_score(scoped["label"].values, scoped["Score"].values)
    aupr_sc  = average_precision_score(scoped["label"].values, scoped["Score"].values)
    logging.info(f"[ChIP FairScope Overall] AUROC={auroc_sc:.3f} | AUPR={aupr_sc:.3f}")

# =========================
# Optuna sweep (small, fast)
# =========================
if RUN_OPTUNA:
    import json, optuna
    from optuna.pruners import MedianPruner

    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)


    # ---------- Lightweight ChIP eval prep used inside Optuna ----------
    def _prepare_chip_universe_for_optuna():
        """Load & canonicalize ChIP once; build a deduped TF/TG table in model vocab."""
        if not (CHIP_GT_PATH and os.path.exists(CHIP_GT_PATH)):
            logging.warning("CHIP_GT_PATH missing; Optuna balanced-micro will be skipped.")
            return None, None

        # 1) minimal load & canonicalize (no plotting, no CSVs)
        chip = _load_chip_gold(CHIP_GT_PATH, CHIP_GT_SEP)
        chip = canon.standardize_df(chip, tf_col="TF", tg_col="TG").dropna()
        chip = chip.drop_duplicates(subset=["TF","TG"], keep="first").reset_index(drop=True)

        # 2) model vocab (encoders you already fit)
        tf_vocab = set(map(str, tf_encoder.classes_))
        tg_vocab = set(map(str, tg_encoder.classes_))
        chip_in_universe = chip[chip["TF"].isin(tf_vocab) & chip["TG"].isin(tg_vocab)].copy()

        # 3) a deduped TF/TG “template” of model edges (no Score yet)
        df_scores_base = df[["TF","TG"]].copy()
        df_scores_base = canon.standardize_df(df_scores_base, tf_col="TF", tg_col="TG").dropna()
        df_scores_base = df_scores_base.drop_duplicates(subset=["TF","TG"], keep="first").reset_index(drop=True)

        return chip_in_universe, df_scores_base

    CHIP_U, DF_SCORES_BASE = _prepare_chip_universe_for_optuna()


    def _balanced_micro_auroc_from_scored(scored_df: pd.DataFrame) -> float:
        """
        Compute balanced-micro AUROC (equal TF weight, 50/50 within TF),
        robust to NaNs and degenerate label distributions.
        Expects columns: TF, Score, label.
        """
        # 1) Keep only finite scores
        s = scored_df.copy()
        s = s[np.isfinite(s["Score"].to_numpy())]

        # 2) Build per-TF parts with 50/50 class weights; skip TFs with <2 classes
        def _per_tf_weights(df_tf):
            lbl = df_tf["label"].to_numpy()
            n_pos = int(lbl.sum())
            n_neg = len(lbl) - n_pos
            w = np.zeros(len(lbl), dtype=np.float64)
            if n_pos > 0: w[lbl == 1] = 0.5 / n_pos
            if n_neg > 0: w[lbl == 0] = 0.5 / n_neg
            return w

        parts, weights = [], []
        for _, sub in s.groupby("TF", sort=False):
            # need both classes *after* removing NaNs
            if sub["label"].nunique() < 2:
                continue
            w = _per_tf_weights(sub)
            if not np.isfinite(w).all() or w.sum() == 0:
                continue
            parts.append(sub[["label","Score"]].reset_index(drop=True))
            weights.append(w)

        if not parts:
            return float("nan")

        blended = pd.concat(parts, axis=0, ignore_index=True)
        y_all = blended["label"].to_numpy()
        s_all = blended["Score"].to_numpy()
        w_all = (np.concatenate(weights, axis=0) / len(weights)).astype(np.float64)

        # final safety checks
        if not (np.isfinite(s_all).all() and np.isfinite(w_all).all()):
            return float("nan")
        if len(np.unique(y_all)) < 2:
            return float("nan")

        return float(roc_auc_score(y_all, s_all, sample_weight=w_all))



    def _score_all_pairs_fast(model_ft) -> pd.Series:
        """Return a pandas Series of scores aligned to DF_SCORES_BASE rows (named 'Score')."""
        # map TF/TG to local ids once (encoders already fit)
        tf_to_local = {name: i for i, name in enumerate(tf_encoder.classes_)}
        tg_to_local = {name: i for i, name in enumerate(tg_encoder.classes_)}

        sub = DF_SCORES_BASE
        tf_local = sub["TF"].map(tf_to_local)
        tg_local = sub["TG"].map(tg_to_local)
        valid = tf_local.notna() & tg_local.notna()
        pairs_local = np.stack(
            [tf_local[valid].astype(int).to_numpy(),
            tg_local[valid].astype(int).to_numpy() + n_tfs],
            axis=1
        )
        pairs_local = torch.as_tensor(pairs_local, dtype=torch.long, device=device)

        probs = np.full(len(sub), np.nan, dtype=np.float32)
        model_ft.eval()
        with torch.no_grad():
            B = PAIR_BATCH
            outs = []
            for i in range(0, pairs_local.shape[0], B):
                sl = pairs_local[i:i+B]
                logits = model_ft(data.x, data.edge_index, data.edge_attr, sl)
                outs.append(torch.sigmoid(logits).detach().cpu().numpy())
            probs_valid = np.concatenate(outs, axis=0)
        probs[valid.values] = probs_valid.ravel()
        return pd.Series(probs, name="Score")

    def build_encoder(hp):
        return GRN_GAT_Encoder(
            in_node_feats=1,
            in_edge_feats=len(edge_features),
            hidden_dim=hp["hidden_dim"],
            heads=hp["heads"],
            dropout=hp["dropout"],
            edge_dropout_p=hp["edge_dropout"]
        ).to(device)

    def pretrain_dgi_once(base_hp):
        """
        Pretrains an encoder with DGI ONCE and returns its state_dict on CPU.
        All trials will reuse this snapshot to keep things fair and fast.
        """
        enc = build_encoder(base_hp)
        opt = torch.optim.Adam(enc.parameters(), lr=1e-4, weight_decay=1e-5)

        for epoch in range(1, DGI_EPOCHS + 1):
            enc.train(); opt.zero_grad()
            h_real, g_real = enc(data.x, data.edge_index, data.edge_attr)
            h_fake, _ = enc(corruption(data.x), data.edge_index, data.edge_attr)
            loss = infomax_loss(h_real, h_fake, g_real)
            loss.backward(); opt.step()
            if epoch % 25 == 0:
                logging.info(f"[DGI/Pretrain] {epoch:03d}/{DGI_EPOCHS} | Loss={loss.item():.4f}")
        return {k: v.detach().cpu() for k, v in enc.state_dict().items()}

    # Cache a baseline DGI snapshot using your top-level hyperparams
    BASE_HP = dict(hidden_dim=HIDDEN_DIM, heads=GAT_HEADS, dropout=DROPOUT, edge_dropout=EDGE_DROPOUT)
    DGI_SNAPSHOT = pretrain_dgi_once(BASE_HP)  # one-time

    def finetune_validate(pretrained_state, hp, report_every=5):
        """
        Builds a fresh encoder, loads DGI weights, adds head, fine-tunes with L2-SP,
        and returns BEST validation AUPR across fine-tune epochs.
        """
        # Fresh encoder per trial
        enc = build_encoder(hp)
        enc.load_state_dict(pretrained_state, strict=True)

        class EdgeClassifier(nn.Module):
            def __init__(self, base_model, dim):
                super().__init__()
                self.encoder = base_model
                self.classifier = nn.Sequential(
                    nn.Linear(dim * 2, dim),
                    nn.ReLU(),
                    nn.Linear(dim, 1),
                )
            def forward(self, x, edge_index, edge_attr, pairs):
                h, _ = self.encoder(x, edge_index, edge_attr)
                tf_emb = h[pairs[:,0]]; tg_emb = h[pairs[:,1]]
                return self.classifier(torch.cat([tf_emb, tg_emb], dim=1)).squeeze(-1)

        model_ft = EdgeClassifier(enc, hp["hidden_dim"]).to(device)

        # Unfreeze policy: last GAT/proj only (matches your main script)
        for n, p in model_ft.encoder.named_parameters():
            p.requires_grad = ("gat2" in n) or ("proj" in n)

        # Param groups
        param_groups = [
            {"params": [p for n,p in model_ft.encoder.named_parameters() if p.requires_grad], "lr": hp["lr_encoder"]},
            {"params": model_ft.classifier.parameters(), "lr": hp["lr_head"]},
        ]
        opt = torch.optim.Adam(param_groups, weight_decay=hp["weight_decay"])
        crit = nn.BCEWithLogitsLoss()

        # L2-SP reference on device
        ref = {k: v.to(device) for k, v in pretrained_state.items()}
        def l2sp_loss(module):
            reg = 0.0
            for (k, p) in module.state_dict().items():
                if p.dtype.is_floating_point and k in ref:
                    reg = reg + (p - ref[k]).pow(2).sum()
            return reg

        best_val_aupr = -float("inf")
        for epoch in range(1, FINETUNE_EPOCHS + 1):
            model_ft.train(); opt.zero_grad()
            logits = model_ft(data.x, data.edge_index, data.edge_attr, pairs_train)
            loss = crit(logits, y_train) + hp["l2sp_lambda"] * l2sp_loss(model_ft.encoder)
            loss.backward(); opt.step()

            if (epoch % report_every == 0) or (epoch == FINETUNE_EPOCHS):
                model_ft.eval()
                with torch.no_grad():
                    preds = torch.sigmoid(model_ft(data.x, data.edge_index, data.edge_attr, pairs_test))
                    val_aupr = average_precision_score(y_test.cpu(), preds.cpu())
                    if val_aupr > best_val_aupr:
                        best_val_aupr = val_aupr
        return model_ft, float(best_val_aupr)

    def objective(trial: optuna.Trial):
        if CHIP_U is None or DF_SCORES_BASE is None:
            raise RuntimeError("ChIP universe was not prepared. Check CHIP_GT_PATH and canonicalizer inputs.")

        hp = {
            "hidden_dim":   trial.suggest_categorical("hidden_dim", [96, 128, 192]),
            "heads":        trial.suggest_categorical("heads", [2, 4, 6]),
            "dropout":      trial.suggest_float("dropout", 0.2, 0.5, step=0.1),
            "edge_dropout": trial.suggest_float("edge_dropout", 0.2, 0.5, step=0.1),
            "lr_head":      trial.suggest_float("lr_head", 1e-4, 3e-4, log=True),
            "lr_encoder":   trial.suggest_float("lr_encoder", 1e-5, 1e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
            "l2sp_lambda":  trial.suggest_float("l2sp_lambda", 1e-4, 3e-3, log=True),
        }

        # train (reusing global DGI snapshot when possible)
        try:
            model_ft, val_aupr = finetune_validate(DGI_SNAPSHOT, hp)
        except RuntimeError:
            trial_snapshot = pretrain_dgi_once({
                "hidden_dim": hp["hidden_dim"],
                "heads": hp["heads"],
                "dropout": hp["dropout"],
                "edge_dropout": hp["edge_dropout"],
            })
            model_ft, val_aupr = finetune_validate(trial_snapshot, hp)

        # lightweight scoring
        scores = _score_all_pairs_fast(model_ft)

        # merge with ChIP_in_universe and compute balanced-micro AUROC
        # (copy to avoid mutating the cached template)
        scored_tmp = DF_SCORES_BASE.copy()
        scored_tmp["Score"] = scores.values
        scored_tmp = scored_tmp.merge(CHIP_U, on=["TF","TG"], how="left")
        scored_tmp["label"] = scored_tmp["label"].fillna(0).astype(int)

        bm_auroc = _balanced_micro_auroc_from_scored(scored_tmp)
        
        if not np.isfinite(bm_auroc):
            # annotate why we returned a bad score, then give a low value
            trial.set_user_attr("bm_auroc_reason", "nan_or_single_class_after_filtering")
            return -1.0

        # log the secondary metric for later inspection
        trial.set_user_attr("val_aupr", float(val_aupr))

        # prune/report on the true objective
        trial.report(bm_auroc, step=FINETUNE_EPOCHS)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return bm_auroc

    N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "30"))   # set to 0 to skip, or export a smaller number for quick tests
    PRUNER   = MedianPruner(n_warmup_steps=max(1, FINETUNE_EPOCHS // 5))

    study = optuna.create_study(direction="maximize", pruner=PRUNER, study_name="gat_opt_balanced_micro")
    logging.info(f"Starting Optuna (objective = balanced-micro AUROC) with {N_TRIALS} trials…")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    # Save results
    best = {"best_value_bm_auroc": study.best_value, "best_params": study.best_trial.params,
            "best_val_aupr_attr": study.best_trial.user_attrs.get("val_aupr")}
    with open(os.path.join(OUT_DIR, "optuna_best.json"), "w") as f:
        json.dump(best, f, indent=2)
    df_trials = study.trials_dataframe(attrs=("number","value","state","params","user_attrs","datetime_start","datetime_complete"))
    df_trials.to_csv(os.path.join(OUT_DIR, "optuna_trials.csv"), index=False)

    logging.info(f"Best balanced-micro AUROC: {study.best_value:.4f}")
    logging.info(f"Best params: {study.best_trial.params}")




# ============================================================
# ===  Plot Phase 1 and Phase 2 Loss Curves  ===
# ============================================================
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

# ---- Phase 1 ----
plt.subplot(1,2,1)
plt.plot(range(1, len(phase1_losses)+1), phase1_losses, color='steelblue')
plt.title("Phase 1: Self-supervised (DGI) Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)

# ---- Phase 2 ----
plt.subplot(1,2,2)
plt.plot(range(1, len(phase2_train_losses)+1), phase2_train_losses, color='darkorange', label="Train Loss")
if len(phase2_val_auroc) > 0:
    # Plot validation AUROC scaled for visual comparison
    val_epochs = [i*10 for i in range(1, len(phase2_val_auroc)+1)]
    plt.twinx()
    plt.plot(val_epochs, phase2_val_auroc, color='green', label="Val AUROC")
    plt.ylabel("Validation AUROC", color='green')
plt.title("Phase 2: Fine-tuning Loss / AUROC")
plt.xlabel("Epoch")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("GAT_training_outputs/training_loss_curves.png", dpi=300)
plt.close()
logging.info("Saved training loss curves to GAT_training_outputs/training_loss_curves.png")


# ============================================================
# ===  Evaluation & Feature Importance Analysis  ===
# ============================================================


finetune_model.eval()
with torch.no_grad():
    logits = finetune_model(data.x, data.edge_index, data.edge_attr, pairs_test)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = y_test.cpu().numpy()

# ---------- Core Metrics ----------
fpr, tpr, _ = roc_curve(y_true, probs)
precision, recall, _ = precision_recall_curve(y_true, probs)
auc_val = roc_auc_score(y_true, probs)
aupr_val = average_precision_score(y_true, probs)
logging.info(f"\n=== Evaluation Metrics ===\nAUROC = {auc_val:.3f} | AUPR = {aupr_val:.3f}")

# ---------- Confusion Matrix ----------
pred_binary = (probs >= 0.5).astype(int)
cm = confusion_matrix(y_true, pred_binary)
tn, fp, fn, tp = cm.ravel()
logging.info(f"Confusion Matrix:\n{cm}")
logging.info(f"Precision={tp/(tp+fp+1e-9):.3f}, Recall={tp/(tp+fn+1e-9):.3f}, F1={(2*tp)/(2*tp+fp+fn+1e-9):.3f}")

# ---------- ROC & PR Curves ----------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, label=f"AUROC={auc_val:.3f}")
plt.plot([0,1],[0,1],'--',color='gray')
plt.title("ROC Curve"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend()

plt.subplot(1,2,2)
plt.plot(recall, precision, label=f"AUPR={aupr_val:.3f}")
plt.title("Precision-Recall Curve"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend()
plt.tight_layout()
plt.savefig("GAT_training_outputs/fine_tune_eval_curves.png", dpi=200)
plt.close()

# ============================================================
# ===  Self-Supervised Embedding Quality (Phase 1) ===
# ============================================================
logging.info("\n=== Embedding Quality Check (self-supervised phase) ===")
with torch.no_grad():
    h_final, _ = model(data.x, data.edge_index, data.edge_attr)

# Cosine similarity between known positive vs. random pairs
def cosine_sim(a,b): return F.cosine_similarity(a,b).mean().item()
pos_pairs = pairs_test[y_true == 1]
neg_pairs = pairs_test[y_true == 0]
pos_sim = cosine_sim(h_final[pos_pairs[:,0]], h_final[pos_pairs[:,1]])
neg_sim = cosine_sim(h_final[neg_pairs[:,0]], h_final[neg_pairs[:,1]])
logging.info(f"Mean cosine similarity — Positives: {pos_sim:.4f}, Negatives: {neg_sim:.4f}")

# ============================================================
# ===  Feature Importances via Gradients  ===
# ============================================================
logging.info("\n=== Computing feature importances via gradients ===")

# Clone edge attributes and enable gradients on GPU
edge_attr = data.edge_attr.clone().detach().to(device).requires_grad_(True)
finetune_model.zero_grad()

criterion = nn.BCEWithLogitsLoss()
logits = finetune_model(
    data.x.to(device),
    data.edge_index.to(device),
    edge_attr,
    pairs_test.to(device)
)
loss = criterion(logits, y_test.to(device))
loss.backward()

# Compute mean absolute gradients per feature
importance = edge_attr.grad.abs().mean(dim=0).cpu().numpy()
feature_importance = pd.Series(importance, index=edge_features).sort_values(ascending=False)
logging.info("\nFeature importance (mean |∂L/∂feature|):\n" + str(feature_importance))

# Plot importance
plt.figure(figsize=(7,4))
sns.barplot(x=feature_importance.values, y=feature_importance.index, orient="h")
plt.title("Edge Feature Importance (Gradient Magnitude)")
plt.tight_layout()
plt.savefig("GAT_training_outputs/feature_importance_barplot.png", dpi=200)
plt.close()

# ============================================================
# ===  Sensitivity / Ablation Analysis  ===
# ============================================================
logging.info("\n=== Performing sensitivity test per feature ===")
delta = 0.1
results = {}
for feat in edge_features:
    idx = edge_features.index(feat)
    edge_attr_mod = edge_attr.clone()
    edge_attr_mod[:, idx] += delta
    with torch.no_grad():
        preds = torch.sigmoid(finetune_model(data.x, data.edge_index, edge_attr_mod, pairs_test))
        results[feat] = roc_auc_score(y_test.cpu(), preds.cpu())
sens_df = pd.DataFrame.from_dict(results, orient='index', columns=['AUC_after_perturb'])
sens_df['ΔAUC'] = sens_df['AUC_after_perturb'] - auc_val
logging.info("\nFeature sensitivity (ΔAUROC when feature +0.1):\n" + str(sens_df.sort_values('ΔAUC', ascending=False)))
sens_df.to_csv("GAT_training_outputs/feature_sensitivity.csv")

# ============================================================
# ===  Per-TF, Per-TG, and Per-Pathway Evaluation  ===
# ============================================================

logging.info("\n=== Evaluating per-TF, per-TG, and KEGG pathway performance ===")

# ----- Per-TF AUROC / AUPR -----
tf_scores = []
for tf_name, tf_id in zip(tf_encoder.classes_, range(n_tfs)):
    mask = pairs_test[:,0].cpu().numpy() == tf_id
    if mask.sum() < 5:  # skip if too few edges
        continue
    y_true_tf = y_true[mask]
    y_pred_tf = probs[mask]
    if len(np.unique(y_true_tf)) < 2:
        continue
    auc_tf = roc_auc_score(y_true_tf, y_pred_tf)
    aupr_tf = average_precision_score(y_true_tf, y_pred_tf)
    tf_scores.append((tf_name, auc_tf, aupr_tf))
tf_df = pd.DataFrame(tf_scores, columns=["TF", "AUROC", "AUPR"]).sort_values("AUROC", ascending=False)
tf_df.to_csv("GAT_training_outputs/per_TF_metrics.csv", index=False)
logging.info(f"Saved per-TF metrics for {len(tf_df)} TFs to GAT_training_outputs/per_TF_metrics.csv")

# ----- Per-TG AUROC -----
tg_scores = []
for tg_name, tg_id in zip(tg_encoder.classes_, range(n_tgs)):
    mask = pairs_test[:,1].cpu().numpy() == (tg_id + n_tfs)
    if mask.sum() < 5:
        continue
    y_true_tg = y_true[mask]
    y_pred_tg = probs[mask]
    if len(np.unique(y_true_tg)) < 2:
        continue
    auc_tg = roc_auc_score(y_true_tg, y_pred_tg)
    tg_scores.append((tg_name, auc_tg))
tg_df = pd.DataFrame(tg_scores, columns=["TG", "AUROC"]).sort_values("AUROC", ascending=False)
tg_df.to_csv("GAT_training_outputs/per_TG_metrics.csv", index=False)
logging.info(f"Saved per-TG metrics for {len(tg_df)} TGs to GAT_training_outputs/per_TG_metrics.csv")

# ----- Per-Pathway (KEGG) Evaluation -----
if "kegg_pathways" in df.columns:
    logging.info("Computing per-KEGG-pathway AUROC")
    df_path = df.loc[df["kegg_pathways"].notna()].copy()
    df_path["kegg_pathways"] = df_path["kegg_pathways"].astype(str)

    pathway_metrics = []
    for pathway, subset in df_path.groupby("kegg_pathways"):
        idx = df_path.index.isin(subset.index)
        # Align with test mask
        mask = torch.tensor(idx[df.index[pairs_test.cpu().numpy()[:,0]]], dtype=torch.bool)
        y_true_pw = y_true[mask.cpu().numpy()] if mask.sum() > 0 else []
        y_pred_pw = probs[mask.cpu().numpy()] if mask.sum() > 0 else []
        if len(y_true_pw) < 5 or len(np.unique(y_true_pw)) < 2:
            continue
        auc_pw = roc_auc_score(y_true_pw, y_pred_pw)
        aupr_pw = average_precision_score(y_true_pw, y_pred_pw)
        pathway_metrics.append((pathway, auc_pw, aupr_pw))
    if pathway_metrics:
        kegg_df = pd.DataFrame(pathway_metrics, columns=["KEGG_Pathway","AUROC","AUPR"]).sort_values("AUROC", ascending=False)
        kegg_df.to_csv("GAT_training_outputs/per_KEGG_metrics.csv", index=False)
        logging.info(f"Saved per-pathway metrics for {len(kegg_df)} KEGG pathways to GAT_training_outputs/per_KEGG_metrics.csv")
    else:
        logging.warning("No valid KEGG pathway entries found for evaluation.")
else:
    logging.warning("No 'kegg_pathways' column present — skipping pathway evaluation.")

# ----- Visualization (optional) -----
import matplotlib.pyplot as plt
import seaborn as sns

# Top 50 TFs by AUROC
plt.figure(figsize=(8,4))
sns.barplot(x="AUROC", y="TF", hue="TF", data=tf_df.head(50), legend=False, palette="viridis")
plt.title("Top 50 TFs by AUROC (Fine-tuned GAT)")
plt.tight_layout()
plt.savefig("GAT_training_outputs/top_TF_AUROC.png", dpi=200)
plt.close()

# Top 25 KEGG pathways
if "kegg_df" in locals() and not kegg_df.empty:
    plt.figure(figsize=(8,4))
    sns.barplot(x="AUROC", y="KEGG_Pathway", data=kegg_df.head(25), hue="AUROC", palette="viridis")
    plt.title("Top 25 KEGG Pathways by AUROC")
    plt.tight_layout()
    plt.savefig("GAT_training_outputs/top_KEGG_AUROC.png", dpi=200)
    plt.close()

# ============================================================
# ===  Integrated Gradients (IG) Feature Attribution  ===
# ============================================================
logging.info("\n=== Computing feature importances via Integrated Gradients (IG) ===")

def integrated_gradients(
    model, data, pairs_test, y_test, edge_features, baseline=None, steps=50
):
    """
    Integrated gradients for edge features.
    Integrates gradients between a baseline and the actual input.
    """
    device = next(model.parameters()).device
    model.eval()

    # Baseline: zeros or provided
    if baseline is None:
        baseline = torch.zeros_like(data.edge_attr).to(device)

    # Scale inputs between baseline and actual edge attributes
    scaled_inputs = [
        baseline + (float(i) / steps) * (data.edge_attr - baseline)
        for i in range(steps + 1)
    ]

    total_gradients = torch.zeros_like(data.edge_attr).to(device)
    criterion = nn.BCEWithLogitsLoss()

    for scaled_attr in scaled_inputs:
        scaled_attr.requires_grad_(True)
        model.zero_grad()

        logits = finetune_model(
            data.x.to(device),
            data.edge_index.to(device),
            scaled_attr,
            pairs_test.to(device)
        )
        loss = criterion(logits, y_test.to(device))
        loss.backward()

        total_gradients += scaled_attr.grad.detach()

    avg_gradients = total_gradients / (steps + 1)
    ig_attributions = (data.edge_attr - baseline) * avg_gradients
    return ig_attributions.mean(dim=0).cpu().numpy()

# Run IG attribution
ig_attr = integrated_gradients(
    finetune_model, data, pairs_test, y_test, edge_features, steps=50
)
ig_series = pd.Series(ig_attr, index=edge_features).sort_values(ascending=False)
logging.info("\nIntegrated Gradients feature importances:\n" + str(ig_series))

# Plot IG feature importances
plt.figure(figsize=(7, 4))
sns.barplot(x=ig_series.values, y=ig_series.index, orient="h")
plt.title("Integrated Gradients Feature Importance (Mean Attribution)")
plt.tight_layout()
plt.savefig("GAT_training_outputs/feature_importance_IG.png", dpi=200)
plt.close()

# ============================================================
# ===  TF-Specific Integrated Gradients Analysis  ===
# ============================================================
logging.info("\n=== Computing TF-specific Integrated Gradients ===")

tf_ig_records = []

for tf_name, tf_id in zip(tf_encoder.classes_, range(n_tfs)):
    mask = pairs_test[:,0].cpu().numpy() == tf_id
    if mask.sum() < 10:  # need enough edges
        continue

    sub_pairs = pairs_test[mask]
    sub_labels = y_test[mask]
    if len(np.unique(sub_labels.cpu().numpy())) < 2:
        continue

    ig_attr = integrated_gradients(
        finetune_model,
        data,
        sub_pairs,
        sub_labels,
        edge_features,
        steps=30
    )

    tf_ig_records.append({
        "TF": tf_name,
        **{feat: val for feat, val in zip(edge_features, ig_attr)}
    })

tf_ig_df = pd.DataFrame(tf_ig_records)
tf_ig_df.to_csv("GAT_training_outputs/tf_specific_integrated_gradients.csv", index=False)
logging.info(f"Saved TF-specific IG feature attributions for {len(tf_ig_df)} TFs")

# ------------------------------------------------------------
# Optional: visualize top features for top 5 TFs (safe version)
# ------------------------------------------------------------
top_tfs = tf_df.head(5)["TF"].tolist()
plt.figure(figsize=(10, 4))

plotted = 0
for tf in top_tfs:
    match = tf_ig_df[tf_ig_df["TF"] == tf]
    if match.empty:
        logging.warning(f"Skipping {tf} — no IG attribution data found.")
        continue

    vals = match.iloc[0, 1:]  # feature columns
    plt.subplot(1, min(5, len(tf_ig_df)), plotted + 1)
    sns.barplot(x=vals.values, y=vals.index, orient="h")
    plt.title(tf)
    plotted += 1

if plotted == 0:
    logging.warning("No TFs had IG attribution data for plotting.")
else:
    plt.tight_layout()
    plt.savefig("GAT_training_outputs/tf_specific_IG_barplots.png", dpi=200)
    plt.close()
    logging.info(f"Saved IG feature barplots for {plotted} TFs.")


# ============================================================
# ===  TF Embedding Visualization (UMAP / t-SNE) ===
# ============================================================
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logging.info("\n=== Visualizing TF embeddings ===")

# Extract final node embeddings from self-supervised encoder
with torch.no_grad():
    h_final, _ = model(data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device))
h_final = h_final.cpu().numpy()

# Extract TF-only embeddings (first n_tfs entries)
tf_embeddings = h_final[:n_tfs]
tf_names = tf_encoder.classes_

# ------------------------------------------------------------
# Load per-TF metrics if available
# ------------------------------------------------------------
try:
    tf_metrics = pd.read_csv("GAT_training_outputs/per_TF_metrics.csv")
    tf_metrics = tf_metrics.set_index("TF")
    auroc_map = [tf_metrics.loc[tf, "AUROC"] if tf in tf_metrics.index else np.nan for tf in tf_names]
except FileNotFoundError:
    logging.warning("per_TF_metrics.csv not found; skipping AUROC coloring.")
    auroc_map = [np.nan] * len(tf_names)

# ------------------------------------------------------------
# UMAP projection
# ------------------------------------------------------------
scaler = StandardScaler()
tf_scaled = scaler.fit_transform(tf_embeddings)

umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
tf_umap = umap_model.fit_transform(tf_scaled)

# Optional: alternative t-SNE view (comment out if large)
# tsne_model = TSNE(n_components=2, perplexity=20, random_state=42, learning_rate="auto")
# tf_tsne = tsne_model.fit_transform(tf_scaled)

# ------------------------------------------------------------
# Combine into DataFrame for plotting
# ------------------------------------------------------------
tf_vis_df = pd.DataFrame({
    "TF": tf_names,
    "UMAP1": tf_umap[:,0],
    "UMAP2": tf_umap[:,1],
    "AUROC": auroc_map
})
tf_vis_df.to_csv("GAT_training_outputs/tf_embedding_umap.csv", index=False)
logging.info(f"Saved TF UMAP embeddings to GAT_training_outputs/tf_embedding_umap.csv")

# ------------------------------------------------------------
# Plot UMAP colored by AUROC
# ------------------------------------------------------------
plt.figure(figsize=(7,6))
sns.scatterplot(
    data=tf_vis_df,
    x="UMAP1", y="UMAP2",
    hue="AUROC", palette="viridis", s=60, edgecolor="none"
)
plt.title("TF Embedding Map (colored by AUROC)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("GAT_training_outputs/tf_embedding_umap_by_AUROC.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Optional: Cluster labeling
# ------------------------------------------------------------
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, random_state=42).fit(tf_scaled)
tf_vis_df["Cluster"] = kmeans.labels_

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=tf_vis_df,
    x="UMAP1", y="UMAP2",
    hue="Cluster", palette="tab10", s=60, edgecolor="none"
)
plt.title("TF Embedding Clusters (KMeans)")
plt.tight_layout()
plt.savefig("GAT_training_outputs/tf_embedding_clusters.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Optional: Link clusters to top features (if IG results exist)
# ------------------------------------------------------------
try:
    ig_df = pd.read_csv("GAT_training_outputs/tf_specific_integrated_gradients.csv")
    ig_summary = ig_df.groupby("TF")[edge_features].mean()
    ig_summary["Cluster"] = tf_vis_df.set_index("TF")["Cluster"]
    cluster_summary = ig_summary.groupby("Cluster").mean()
    cluster_summary.to_csv("GAT_training_outputs/tf_cluster_feature_summary.csv")
    logging.info("Saved cluster-wise mean feature attributions to GAT_training_outputs/tf_cluster_feature_summary.csv")
except FileNotFoundError:
    logging.warning("tf_specific_integrated_gradients.csv not found; skipping feature summary by cluster.")
