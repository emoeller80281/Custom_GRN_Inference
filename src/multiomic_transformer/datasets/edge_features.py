# edge_features.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

@dataclass
class EdgeFeatureConfig:
    base_feats: Iterable[str] = field(default_factory=lambda: [
        "reg_potential", "motif_density", "mean_tf_expr",
        "mean_tg_expr", "expr_product", "motif_present"
    ])
    extra_prefixes: Tuple[str, ...] = ("string_", "trrust_", "kegg_")
    edge_pca_dim: int = 16
    test_size: float = 0.2
    seed: int = 42
    # unlabeled positives (GAE) heuristics
    motif_col: str = "motif_present"
    regpot_col: str = "reg_potential"
    regpot_percentile: float = 0.75

class EdgeFeatureBuilder:
    """
    Build a reusable feature bundle for TF–TG edge models.
    Produces:
      - encoders, pairs (train/test), edge_attr (PCA-compressed)
      - helper to build unlabeled positives for GAE
    """
    def __init__(self, cfg: EdgeFeatureConfig):
        self.cfg = cfg
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.edge_features: List[str] = []
        self.tf_encoder: Optional[LabelEncoder] = None
        self.tg_encoder: Optional[LabelEncoder] = None
        self._n_tfs: int = 0
        self._n_tgs: int = 0

    # ---------- public API ----------

    def build_all(self, df: pd.DataFrame, device: torch.device = torch.device("cpu")) -> Dict[str, Any]:
        """End-to-end build."""
        df = df.copy()
        # Select features
        self.edge_features = self._select_features(df)
        # Coerce numerics (safe), especially binary columns as float
        df = self._coerce_numeric(df, self.edge_features)

        # Build encoders for TF/TG names
        self.tf_encoder, self.tg_encoder = self._fit_encoders(df)
        self._n_tfs = len(self.tf_encoder.classes_)
        self._n_tgs = len(self.tg_encoder.classes_)

        # Build group-aware train/test split
        train_idx, test_idx = self._group_split(df)

        # Build edge_attr (scale+PCA on ALL edges, scaler fit on train only)
        edge_attr = self._build_edge_attr(df, self.edge_features, train_idx, device=device)

        # Build 0-based integer pairs: (tf_id, tg_id_offsetted)
        pairs_all = self._build_pairs(df)

        # Labels if present (else zeros to keep shape)
        if "label" in df.columns:
            labels_all = torch.as_tensor(df["label"].astype(int).to_numpy(), dtype=torch.float32, device=device)
        else:
            labels_all = torch.zeros((len(df),), dtype=torch.float32, device=device)

        # Slice to train/test tensors
        train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
        test_idx_t  = torch.as_tensor(test_idx, dtype=torch.long, device=device)
        pairs_train = pairs_all.index_select(0, train_idx_t)
        pairs_test  = pairs_all.index_select(0, test_idx_t)
        y_train     = labels_all.index_select(0, train_idx_t)
        y_test      = labels_all.index_select(0, test_idx_t)

        bundle = {
            "edge_attr": edge_attr,                # torch [E, d_edge]
            "pairs_all": pairs_all,                # torch [E, 2] (TG already offset by n_tfs)
            "pairs_train": pairs_train,
            "pairs_test": pairs_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "tf_encoder": self.tf_encoder,
            "tg_encoder": self.tg_encoder,
            "n_tfs": self._n_tfs,
            "n_tgs": self._n_tgs,
            "tf_classes": self.tf_encoder.classes_,
            "tg_classes": self.tg_encoder.classes_,
            "edge_features": list(self.edge_features),
            # store transformers for reuse if needed later
            "scaler": self.scaler,
            "pca": self.pca,
        }
        return bundle

    def build_unlabeled_positives(self, df_edges: pd.DataFrame) -> np.ndarray:
        """
        Construct unlabeled 'positive' edges from unsupervised signals only:
          • motif_present > 0 OR reg_potential >= percentile.
        Returns np.int64 array shape [E_pos, 2] with local (tf_id, tg_id) (TG NOT offset).
        """
        if self.tf_encoder is None or self.tg_encoder is None:
            raise RuntimeError("Encoders are not initialized. Call build_all() first.")

        df = df_edges.copy()
        mcol = self.cfg.motif_col
        rcol = self.cfg.regpot_col

        if mcol in df.columns:
            df[mcol] = pd.to_numeric(df[mcol], errors="coerce").fillna(0.0)
        if rcol in df.columns:
            df[rcol] = pd.to_numeric(df[rcol], errors="coerce").fillna(0.0)

        crit = pd.Series(False, index=df.index)
        if mcol in df.columns:
            crit = crit | (df[mcol] > 0)
        if rcol in df.columns and df[rcol].notna().any():
            thr = float(df[rcol].quantile(self.cfg.regpot_percentile))
            crit = crit | (df[rcol] >= thr)

        df_pos = df.loc[crit].copy()
        if df_pos.empty:
            raise ValueError("No unlabeled positives selected; relax thresholds or provide motif/reg_potential.")

        tf_ids = self.tf_encoder.transform(df_pos["TF"].astype(str))
        tg_ids = self.tg_encoder.transform(df_pos["TG"].astype(str))
        pos_edges = np.stack([tf_ids, tg_ids], axis=1).astype(np.int64)
        pos_edges = np.unique(pos_edges, axis=0)
        return pos_edges

    # ---------- internals ----------

    def _select_features(self, df: pd.DataFrame) -> List[str]:
        base = list(dict.fromkeys([c for c in self.cfg.base_feats if c in df.columns]))
        # include extras by prefix and numeric dtype
        extras = []
        for c in df.columns:
            cl = str(c).lower()
            if any(cl.startswith(p) for p in self.cfg.extra_prefixes) and pd.api.types.is_numeric_dtype(df[c]):
                extras.append(c)
        feats = base + extras
        feats = list(dict.fromkeys(feats))
        if not feats:
            raise ValueError("No edge features found (after selection).")
        return feats

    def _coerce_numeric(self, df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        # special binary → float
        if "motif_present" in out.columns:
            out["motif_present"] = pd.to_numeric(out["motif_present"], errors="coerce").fillna(0.0).astype(float)
        # general numeric coercion
        for c in feat_cols:
            s = pd.to_numeric(out[c], errors="coerce")
            out[c] = s.fillna(0.0)
        return out

    def _fit_encoders(self, df: pd.DataFrame) -> Tuple[LabelEncoder, LabelEncoder]:
        tf_enc = LabelEncoder().fit(df["TF"].astype(str))
        tg_enc = LabelEncoder().fit(df["TG"].astype(str))
        return tf_enc, tg_enc

    def _group_split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # group by exact TF–TG pair so the same pair can’t leak
        pairs = list(zip(df["TF"].astype(str), df["TG"].astype(str)))
        gss = GroupShuffleSplit(n_splits=1, test_size=self.cfg.test_size, random_state=self.cfg.seed)
        train_idx, test_idx = next(gss.split(df, df.get("label", None), groups=pairs))
        return train_idx, test_idx

    def _build_edge_attr(
        self,
        df: pd.DataFrame,
        feat_cols: List[str],
        train_idx: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        X = df[feat_cols].to_numpy()
        # missingness mask
        mask = np.isnan(X).astype(np.float32)
        X_filled = np.nan_to_num(X, nan=0.0)

        # fit scaler on train only
        self.scaler = StandardScaler()
        self.scaler.fit(np.nan_to_num(df.iloc[train_idx][feat_cols].to_numpy(), nan=0.0))
        X_scaled = self.scaler.transform(X_filled).astype(np.float32)

        X_comb = np.concatenate([X_scaled, mask], axis=1)  # [E, 2F]
        # PCA
        d_edge = int(self.cfg.edge_pca_dim)
        d_edge = max(1, min(d_edge, X_comb.shape[1]))
        self.pca = PCA(n_components=d_edge, random_state=self.cfg.seed).fit(X_comb)
        X_edge = self.pca.transform(X_comb).astype(np.float32)
        return torch.from_numpy(X_edge).to(device)

    def _build_pairs(self, df: pd.DataFrame) -> torch.Tensor:
        tf_ids = self.tf_encoder.transform(df["TF"].astype(str)).astype(np.int64)
        tg_ids = self.tg_encoder.transform(df["TG"].astype(str)).astype(np.int64)
        # Offset TG by #TFs so pairs refer to absolute node indices in a concatenated TF+TG node list.
        tg_ids_offset = tg_ids + self._n_tfs
        pairs = np.stack([tf_ids, tg_ids_offset], axis=1).astype(np.int64)
        return torch.as_tensor(pairs, dtype=torch.long)
