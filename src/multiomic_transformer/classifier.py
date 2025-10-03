import torch
import pandas as pd

@torch.no_grad()
def extract_edge_features(model, dataloader, tf_names, tg_names, chip_edges=None, device="cuda"):
    model.eval()
    edge_features = []
    
    for batch in dataloader:
        atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = [
            x.to(device) if torch.is_tensor(x) else x for x in batch
        ]

        # forward pass
        preds_out = model(atac_wins, tf_tensor, tf_ids=tf_ids, tg_ids=tg_ids,
                          bias=bias, motif_mask=motif_mask)
        preds = preds_out[0] if isinstance(preds_out, tuple) else preds_out

        # shortcut attention [G, T]
        if hasattr(model.shortcut_layer, "attn"):
            attn = model.shortcut_layer.attn.detach().cpu()

            for tg_idx, tg in enumerate(tg_names):
                for tf_idx, tf in enumerate(tf_names):
                    feat = {
                        "TF": tf,
                        "TG": tg,
                        "attn": float(attn[tg_idx, tf_idx]),
                        "pred_mean": float(preds[:, tg_idx].mean().cpu()),
                        "pred_std": float(preds[:, tg_idx].std().cpu()),
                        "bias_mean": float(bias[:, tg_idx].mean().cpu()),
                        "motif_mask": float(motif_mask[tg_idx, tf_idx].cpu()) if motif_mask is not None else 0.0,
                    }
                    edge_features.append(feat)
    
    df = pd.DataFrame(edge_features)

    # Add binary labels if ground truth is provided
    if chip_edges is not None:
        chip_set = set(chip_edges)
        df["label"] = [(tf, tg) in chip_set for tf, tg in zip(df["TF"], df["TG"])]
        df["label"] = df["label"].astype(int)
    
    return df

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

def train_edge_classifier(df):
    features = ["attn", "pred_mean", "pred_std", "bias_mean", "motif_mask"]
    X = df[features].values
    y = df["label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()  # balance
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, y_pred)
    auprc = average_precision_score(y_val, y_pred)

    print(f"Validation AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}")
    return clf

df = extract_edge_features(trainer.model.module, val_loader,
                           tf_names=dataset.tf_names,
                           tg_names=dataset.tg_names,
                           chip_edges=chip_edges,
                           device=f"cuda:{rank}")

df.to_csv("edge_features.csv", index=False)

clf = train_edge_classifier(df)
import joblib
joblib.dump(clf, "edge_classifier.pkl")

