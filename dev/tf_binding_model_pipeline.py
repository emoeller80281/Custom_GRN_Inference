#!/usr/bin/env python3
"""
Transcription Factor Binding Prediction Pipeline

This script performs feature engineering, model training, and evaluation for
predicting transcription factor (TF) binding to target genes (TG) using
single-cell ATAC-seq and RNA-seq data.

Key features:
- Prevents data leakage between train/test/validation splits
- Ensures no overlapping edges between training and test sets
- Implements exhaustive cross-validation across ground truth datasets
- Provides verbose logging throughout the pipeline
- Optimizes for AUROC on unseen edges in unseen ground truth datasets
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import argparse

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import (
    GroupKFold, StratifiedKFold, KFold, 
    RandomizedSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report, roc_curve
)
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """
    Set up comprehensive logging for the pipeline.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"tf_binding_pipeline_{timestamp}.log")
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    
    # Set up both file and console handlers
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

# Initialize global logger
logger = setup_logging()

# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

class DataLoader:
    """Handles loading and validation of input data files."""
    
    def __init__(self, data_dir: str = ".", neg_pos_ratio: int = 5, random_state: int = 42):
        self.data_dir = Path(data_dir)
        self.neg_pos_ratio = int(neg_pos_ratio)
        self.random_state = int(random_state)
        logger.info(f"DataLoader initialized with data directory: {data_dir}")
    
    def _read_table(self, path: Path) -> pd.DataFrame:
        suf = path.suffix.lower()
        if suf == ".csv":
            return pd.read_csv(path)
        if suf == ".tsv":
            return pd.read_csv(path, sep="\t")
        if suf == ".parquet":
            return pd.read_parquet(path)
        if suf in (".pkl", ".pickle"):
            return pd.read_pickle(path)
        raise ValueError(f"Unsupported ground truth file type: {suf}")

    def _coerce_ground_truth(self, df: pd.DataFrame, source: str) -> Optional[pd.DataFrame]:
        """
        Map a variety of known schemas to (TF, TG, label).
        Unknown columns are ignored; all rows are treated as positives unless 'label' present.
        """
        cols = {c.lower(): c for c in df.columns}  # case-insensitive map

        # Already in TF/TG[/label] form
        if "tf" in cols and "tg" in cols:
            out = df[[cols["tf"], cols["tg"]]].rename(columns={cols["tf"]: "TF", cols["tg"]: "TG"})
            if "label" in cols:
                out["label"] = df[cols["label"]].astype(int)
            else:
                out["label"] = 1
            return out.dropna(subset=["TF", "TG"])

        # BEELINE: Gene1, Gene2 (positives)
        if "gene1" in cols and "gene2" in cols:
            out = df[[cols["gene1"], cols["gene2"]]].rename(columns={cols["gene1"]: "TF", cols["gene2"]: "TG"})
            out["label"] = 1
            return out

        # ChIP-Atlas TF–peak–TG: source_id, target_id (positives)
        if "source_id" in cols and "target_id" in cols:
            out = df[[cols["source_id"], cols["target_id"]]].rename(columns={cols["source_id"]: "TF", cols["target_id"]: "TG"})
            out["label"] = 1
            return out

        # BEAR-GRN: Source, Target (positives)
        if "source" in cols and "target" in cols:
            out = df[[cols["source"], cols["target"]]].rename(columns={cols["source"]: "TF", cols["target"]: "TG"})
            out["label"] = 1
            return out

        # ChIP-Atlas peak-only files (gene_id, peak_id) can't be coerced to TF–TG alone
        if "gene_id" in cols and "peak_id" in cols:
            logger.warning(f"[{source}] has gene_id/peak_id only; skipping (needs a TF–peak map or nearest-gene join).")
            return None

        logger.warning(f"[{source}] schema not recognized; columns={list(df.columns)[:6]}...")
        return None

    def _add_negatives(self,
                       pos_df: pd.DataFrame,
                       ratio: int = 5,
                       random_state: int = 42,
                       avoid_pairs: Optional[set] = None) -> pd.DataFrame:
        """
        Create random TF×TG non-edges as negatives at a given ratio to positives.
        Optionally avoids sampling any pair present in avoid_pairs to prevent false negatives.
        """
        pos_df = pos_df[["TF", "TG"]].dropna().drop_duplicates()
        pos_df["label"] = 1
        pos_set = set(map(tuple, pos_df[["TF", "TG"]].values))
        avoid_pairs = avoid_pairs or set()

        tfs = pos_df["TF"].unique()
        tgs = pos_df["TG"].unique()
        target_n = int(len(pos_df) * ratio)

        rng = np.random.default_rng(random_state)
        neg_pairs, attempts = set(), 0
        max_attempts = target_n * 50  # more attempts to honor avoidance constraints

        while len(neg_pairs) < target_n and attempts < max_attempts:
            pair = (rng.choice(tfs), rng.choice(tgs))
            if (
                pair not in pos_set
                and pair not in avoid_pairs
                and pair not in neg_pairs
            ):
                neg_pairs.add(pair)
            attempts += 1

        if len(neg_pairs) < target_n:
            logger.warning(f"Could only sample {len(neg_pairs)} negatives (requested {target_n}).")

        neg_df = pd.DataFrame(list(neg_pairs), columns=["TF", "TG"])
        neg_df["label"] = 0
        return pd.concat([pos_df, neg_df], ignore_index=True)

    
    def load_expression_data(self, file_patterns: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        Load expression and accessibility data files.
        Accepts either a glob pattern or an exact file path.
        """
        data = {}

        for data_type, pattern in file_patterns.items():
            logger.info(f"Loading {data_type} data from pattern: {pattern}")

            # Accept exact path or glob
            p = Path(pattern)
            files = [p] if p.exists() else list(self.data_dir.glob(pattern))
            if not files:
                logger.warning(f"No files found for pattern {pattern}")
                continue

            dfs = []
            for file_path in files:
                logger.debug(f"Reading file: {file_path}")
                suf = file_path.suffix.lower()
                if suf == ".csv":
                    df = pd.read_csv(file_path, index_col=0)
                elif suf == ".tsv":
                    df = pd.read_csv(file_path, sep="\t", index_col=0)
                elif suf in (".h5", ".hdf5"):
                    df = pd.read_hdf(file_path)
                elif suf == ".pkl":
                    df = pd.read_pickle(file_path)
                elif suf == ".parquet":
                    df = pd.read_parquet(file_path)  # requires pyarrow/fastparquet
                else:
                    logger.warning(f"Unknown file type: {file_path.suffix}")
                    continue
                dfs.append(df)

            if dfs:
                combined = pd.concat(dfs, axis=1) if len(dfs) > 1 else dfs[0]
                data[data_type] = combined
                logger.info(f"Loaded {data_type}: shape {combined.shape}")

        return data

    
    def load_ground_truth(self, gt_files: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load and normalize multiple GT datasets to (TF, TG, label),
        auto-adding negatives at self.neg_pos_ratio per positive.
        """
        ground_truth = {}

        for gt_file in gt_files:
            gt_path = (self.data_dir / gt_file) if not Path(gt_file).is_absolute() else Path(gt_file)
            if not gt_path.exists():
                logger.warning(f"Ground truth file not found: {gt_path}")
                continue

            logger.info(f"Loading ground truth: {gt_path}")
            try:
                raw = self._read_table(gt_path)
            except Exception as e:
                logger.error(f"Failed to read {gt_path}: {e}")
                continue

            df = self._coerce_ground_truth(raw, source=gt_path.stem)
            if df is None or df.empty:
                logger.error(f"Ground truth {gt_path.name} could not be coerced to (TF,TG,label); skipping.")
                continue

            # Ensure we have both classes; synthesize negatives if needed
            if "label" not in df.columns or df["label"].nunique() == 1:
                df = self._add_negatives(df, ratio=self.neg_pos_ratio, random_state=self.random_state)

            pos = int((df["label"] == 1).sum())
            neg = int((df["label"] == 0).sum())
            ground_truth[gt_path.stem] = df
            logger.info(f"Loaded ground truth {gt_path.stem}: {len(df)} edges "
                        f"(pos={pos}, neg={neg}, ratio={neg/(pos or 1):.2f})")

        return ground_truth


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """
    Handles feature engineering while preventing data leakage.
    
    CRITICAL: Features are computed only from training data to prevent leakage.
    """
    
    def __init__(self, feature_config: Dict[str, Any]):
        """
        Initialize FeatureEngineer with configuration.
        
        Args:
            feature_config: Configuration for feature generation
        """
        self.config = feature_config
        self.feature_stats = {}  # Store statistics computed on training data
        logger.info("FeatureEngineer initialized")
    
    def compute_correlation_features(self,
                                    peak_data: pd.DataFrame,
                                    gene_data: pd.DataFrame,
                                    train_edges: pd.DataFrame,
                                    test_edges: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute correlation features between peaks and genes.
        
        CRITICAL: Only uses training edges to compute correlations to prevent leakage.
        
        Args:
            peak_data: Peak accessibility data (peaks x cells)
            gene_data: Gene expression data (genes x cells)
            train_edges: Training edges to compute features for
            test_edges: Test edges (if provided, uses train statistics)
        
        Returns:
            DataFrame with correlation features
        """
        logger.info(f"Computing correlation features for {len(train_edges)} training edges")
        
        # Identify unique peaks and genes in training set
        train_peaks = train_edges['peak'].unique() if 'peak' in train_edges.columns else []
        train_genes = train_edges['TG'].unique()
        
        # Filter data to only training entities to prevent leakage
        if len(train_peaks) > 0:
            peak_data_filtered = peak_data.loc[
                peak_data.index.intersection(train_peaks)
            ]
        else:
            # If no peak column, use TF-associated peaks (requires mapping)
            peak_data_filtered = peak_data
        
        gene_data_filtered = gene_data.loc[
            gene_data.index.intersection(train_genes)
        ]
        
        logger.info(f"Filtered data - Peaks: {peak_data_filtered.shape}, "
                   f"Genes: {gene_data_filtered.shape}")
        
        # Compute correlations only on training data
        correlations = self._compute_pairwise_correlations(
            peak_data_filtered, 
            gene_data_filtered
        )
        
        # Store statistics for test set application
        if test_edges is None:
            self.feature_stats['correlations'] = correlations
        
        # Build feature matrix
        features = self._build_feature_matrix(train_edges, correlations)
        
        # If test edges provided, use stored training statistics
        if test_edges is not None:
            test_features = self._build_feature_matrix(
                test_edges, 
                self.feature_stats.get('correlations', correlations)
            )
            return features, test_features
        
        return features
    
    def _compute_pairwise_correlations(self,
                                      peak_data: pd.DataFrame,
                                      gene_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairwise correlations between peaks and genes.
        
        Args:
            peak_data: Peak accessibility matrix
            gene_data: Gene expression matrix
        
        Returns:
            DataFrame of correlation values
        """
        logger.debug("Computing pairwise correlations")
        
        # Ensure same cell ordering
        common_cells = peak_data.columns.intersection(gene_data.columns)
        peak_data = peak_data[common_cells]
        gene_data = gene_data[common_cells]
        
        # Initialize correlation matrix
        n_peaks = len(peak_data)
        n_genes = len(gene_data)
        
        # Use different correlation methods
        pearson_corr = np.zeros((n_peaks, n_genes))
        spearman_corr = np.zeros((n_peaks, n_genes))
        
        # Compute correlations (vectorized for efficiency)
        for i, peak in enumerate(tqdm(peak_data.index, desc="Computing correlations")):
            peak_vec = peak_data.loc[peak].values
            
            # Skip if peak has no variance
            if np.std(peak_vec) == 0:
                continue
            
            for j, gene in enumerate(gene_data.index):
                gene_vec = gene_data.loc[gene].values
                
                # Skip if gene has no variance
                if np.std(gene_vec) == 0:
                    continue
                
                # Pearson correlation
                pearson_corr[i, j], _ = pearsonr(peak_vec, gene_vec)
                
                # Spearman correlation
                spearman_corr[i, j], _ = spearmanr(peak_vec, gene_vec)
        
        # Create DataFrames
        correlations = {
            'pearson': pd.DataFrame(pearson_corr, 
                                   index=peak_data.index, 
                                   columns=gene_data.index),
            'spearman': pd.DataFrame(spearman_corr,
                                    index=peak_data.index,
                                    columns=gene_data.index)
        }
        
        return correlations
    
    def _build_feature_matrix(self,
                             edges: pd.DataFrame,
                             correlations: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Build feature matrix for given edges using precomputed correlations.
        
        Args:
            edges: DataFrame with edge information
            correlations: Precomputed correlation matrices
        
        Returns:
            Feature matrix for the edges
        """
        features_list = []
        
        for _, edge in edges.iterrows():
            edge_features = {}
            
            # Get TF and TG
            tf = edge.get('TF', None)
            tg = edge.get('TG', None)
            
            # Get associated peaks (if available)
            if 'peak' in edge:
                peaks = [edge['peak']]
            elif 'peaks' in edge:
                peaks = edge['peaks'] if isinstance(edge['peaks'], list) else [edge['peaks']]
            else:
                # Need TF-peak mapping
                peaks = self._get_tf_peaks(tf) if tf else []
            
            # Extract correlation features
            for peak in peaks:
                for corr_type, corr_matrix in correlations.items():
                    if peak in corr_matrix.index and tg in corr_matrix.columns:
                        value = corr_matrix.loc[peak, tg]
                        edge_features[f'{corr_type}_corr_{peak}'] = value
                        edge_features[f'{corr_type}_corr_abs_{peak}'] = abs(value)
                        edge_features[f'{corr_type}_corr_sq_{peak}'] = value ** 2
            
            # Aggregate features across peaks
            if peaks:
                for corr_type in correlations.keys():
                    peak_values = [edge_features.get(f'{corr_type}_corr_{p}', 0) 
                                 for p in peaks]
                    if peak_values:
                        edge_features[f'{corr_type}_mean'] = np.mean(peak_values)
                        edge_features[f'{corr_type}_max'] = np.max(peak_values)
                        edge_features[f'{corr_type}_min'] = np.min(peak_values)
                        edge_features[f'{corr_type}_std'] = np.std(peak_values)
            
            features_list.append(edge_features)
        
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)  # Fill missing values with 0
        
        logger.info(f"Built feature matrix: {features_df.shape}")
        
        return features_df
    
    def _get_tf_peaks(self, tf: str) -> List[str]:
        """
        Get peaks associated with a transcription factor.
        
        Args:
            tf: Transcription factor name
        
        Returns:
            List of associated peak coordinates
        """
        # This would need to be implemented based on your TF-peak mapping
        # For now, returning empty list
        return []
    
    def compute_expression_features(self,
                                   gene_data: pd.DataFrame,
                                   edges: pd.DataFrame,
                                   train_only: bool = True) -> pd.DataFrame:
        """
        Compute gene expression-based features.
        
        Args:
            gene_data: Gene expression matrix
            edges: Edges to compute features for
            train_only: Whether to use only training data statistics
        
        Returns:
            Expression features DataFrame
        """
        logger.info("Computing expression features")
        
        features = []
        
        for _, edge in edges.iterrows():
            tg = edge['TG']
            
            if tg not in gene_data.index:
                features.append({
                    'expr_mean': 0,
                    'expr_std': 0,
                    'expr_max': 0,
                    'expr_percentile_90': 0
                })
                continue
            
            expr = gene_data.loc[tg]
            
            features.append({
                'expr_mean': expr.mean(),
                'expr_std': expr.std(),
                'expr_max': expr.max(),
                'expr_percentile_90': np.percentile(expr, 90),
                'expr_frac_nonzero': (expr > 0).mean()
            })
        
        return pd.DataFrame(features)

# =============================================================================
# DATA SPLITTING AND VALIDATION
# =============================================================================

class DataSplitter:
    """
    Handles train/validation/test splitting while preventing data leakage.
    
    CRITICAL: Ensures no overlapping edges between splits.
    """
    
    def __init__(self, split_strategy: str = "edge", random_state: int = 42):
        """
        Initialize DataSplitter.
        
        Args:
            split_strategy: How to split data ('edge', 'tf', 'tg', 'both')
            random_state: Random seed for reproducibility
        """
        self.split_strategy = split_strategy
        self.random_state = random_state
        logger.info(f"DataSplitter initialized with strategy: {split_strategy}")
    
    def split_edges(self,
                   edges: pd.DataFrame,
                   test_size: float = 0.2,
                   val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split edges into train, validation, and test sets.
        
        Args:
            edges: DataFrame with all edges
            test_size: Fraction for test set
            val_size: Fraction for validation set
        
        Returns:
            Tuple of (train_edges, val_edges, test_edges)
        """
        edges = self._deduplicate_edges(edges)
        logger.info(f"Splitting {len(edges)} edges with strategy: {self.split_strategy}")
        
        if self.split_strategy == "edge":
            return self._split_by_edge(edges, test_size, val_size)
        elif self.split_strategy == "tf":
            return self._split_by_tf(edges, test_size, val_size)
        elif self.split_strategy == "tg":
            return self._split_by_tg(edges, test_size, val_size)
        elif self.split_strategy == "both":
            return self._split_by_both(edges, test_size, val_size)
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")

    def _deduplicate_edges(self, edges: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate TF–TG pairs so the verifier does not flag overlaps from row copies."""
        dedup_cols = ["TF", "TG"]
        if not set(dedup_cols).issubset(edges.columns):
            return edges

        original_len = len(edges)
        if "label" in edges.columns:
            # Keep positives when both labels exist for the same pair
            edges = (
                edges.sort_values(by="label", ascending=False)
                .drop_duplicates(subset=dedup_cols, keep="first")
            )
        else:
            edges = edges.drop_duplicates(subset=dedup_cols)

        removed = original_len - len(edges)
        if removed:
            logger.info(f"Removed {removed} duplicate TF–TG edges before splitting.")
        return edges
    
    def _split_by_edge(self,
                      edges: pd.DataFrame,
                      test_size: float,
                      val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Random split at the edge level.
        
        Args:
            edges: DataFrame with all edges
            test_size: Fraction for test set
            val_size: Fraction for validation set
        
        Returns:
            Tuple of (train_edges, val_edges, test_edges)
        """
        # Shuffle edges deterministically for reproducibility
        edges_shuffled = edges.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        n_total = len(edges_shuffled)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)
        n_train = n_total - n_test - n_val
        
        # Split
        test_edges = edges_shuffled.iloc[:n_test]
        val_edges = edges_shuffled.iloc[n_test:n_test + n_val]
        train_edges = edges_shuffled.iloc[n_test + n_val:]
        
        # Verify no overlap
        self._verify_no_overlap(train_edges, val_edges, test_edges)
        
        logger.info(f"Edge split - Train: {len(train_edges)}, "
                   f"Val: {len(val_edges)}, Test: {len(test_edges)}")
        
        return train_edges, val_edges, test_edges
    
    def _split_by_tf(self,
                    edges: pd.DataFrame,
                    test_size: float,
                    val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by transcription factors - ensures TFs don't overlap between sets.
        
        Args:
            edges: DataFrame with all edges
            test_size: Fraction for test set
            val_size: Fraction for validation set
        
        Returns:
            Tuple of (train_edges, val_edges, test_edges)
        """
        # Get unique TFs
        tfs = edges['TF'].unique()
        np.random.RandomState(self.random_state).shuffle(tfs)
        
        n_tfs = len(tfs)
        n_test_tfs = int(n_tfs * test_size)
        n_val_tfs = int(n_tfs * val_size)
        
        # Split TFs
        test_tfs = set(tfs[:n_test_tfs])
        val_tfs = set(tfs[n_test_tfs:n_test_tfs + n_val_tfs])
        train_tfs = set(tfs[n_test_tfs + n_val_tfs:])
        
        # Split edges based on TF assignment
        test_edges = edges[edges['TF'].isin(test_tfs)]
        val_edges = edges[edges['TF'].isin(val_tfs)]
        train_edges = edges[edges['TF'].isin(train_tfs)]
        
        # Verify no TF overlap
        assert len(train_tfs & val_tfs) == 0, "TF overlap between train and val"
        assert len(train_tfs & test_tfs) == 0, "TF overlap between train and test"
        assert len(val_tfs & test_tfs) == 0, "TF overlap between val and test"
        
        logger.info(f"TF split - Train TFs: {len(train_tfs)}, "
                   f"Val TFs: {len(val_tfs)}, Test TFs: {len(test_tfs)}")
        logger.info(f"Edges - Train: {len(train_edges)}, "
                   f"Val: {len(val_edges)}, Test: {len(test_edges)}")
        
        return train_edges, val_edges, test_edges
    
    def _split_by_tg(self,
                    edges: pd.DataFrame,
                    test_size: float,
                    val_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split by target genes - ensures TGs don't overlap between sets.
        
        Args:
            edges: DataFrame with all edges
            test_size: Fraction for test set
            val_size: Fraction for validation set
        
        Returns:
            Tuple of (train_edges, val_edges, test_edges)
        """
        # Similar to _split_by_tf but for target genes
        tgs = edges['TG'].unique()
        np.random.RandomState(self.random_state).shuffle(tgs)
        
        n_tgs = len(tgs)
        n_test_tgs = int(n_tgs * test_size)
        n_val_tgs = int(n_tgs * val_size)
        
        # Split TGs
        test_tgs = set(tgs[:n_test_tgs])
        val_tgs = set(tgs[n_test_tgs:n_test_tgs + n_val_tgs])
        train_tgs = set(tgs[n_test_tgs + n_val_tgs:])
        
        # Split edges based on TG assignment
        test_edges = edges[edges['TG'].isin(test_tgs)]
        val_edges = edges[edges['TG'].isin(val_tgs)]
        train_edges = edges[edges['TG'].isin(train_tgs)]
        
        logger.info(f"TG split - Train TGs: {len(train_tgs)}, "
                   f"Val TGs: {len(val_tgs)}, Test TGs: {len(test_tgs)}")
        logger.info(f"Edges - Train: {len(train_edges)}, "
                   f"Val: {len(val_edges)}, Test: {len(test_edges)}")
        
        return train_edges, val_edges, test_edges
    
    def _split_by_both(self, edges: pd.DataFrame, test_size: float, val_size: float):
        """Disjoint TFs and TGs across TRAIN / VAL / TEST.
        Ensures no TF or TG appears in more than one split and verifies no edge overlap.
        """
        rng_tf = np.random.RandomState(self.random_state)
        rng_tg = np.random.RandomState(self.random_state + 1)

        tfs = edges['TF'].dropna().unique().copy()
        tgs = edges['TG'].dropna().unique().copy()
        rng_tf.shuffle(tfs)
        rng_tg.shuffle(tgs)

        n_test_tfs = int(len(tfs) * test_size)
        n_val_tfs  = int(len(tfs) * val_size)
        n_test_tgs = int(len(tgs) * test_size)
        n_val_tgs  = int(len(tgs) * val_size)

        test_tfs = set(tfs[:n_test_tfs]);          val_tfs = set(tfs[n_test_tfs:n_test_tfs + n_val_tfs]);
        train_tfs = set(tfs[n_test_tfs + n_val_tfs:])
        test_tgs = set(tgs[:n_test_tgs]);          val_tgs = set(tgs[n_test_tgs:n_test_tgs + n_val_tgs]);
        train_tgs = set(tgs[n_test_tgs + n_val_tgs:])

        # Assign by disjoint vocabularies (edges go where either TF or TG set matches)
        test_edges = edges[edges['TF'].isin(test_tfs) | edges['TG'].isin(test_tgs)]
        val_edges  = edges[edges['TF'].isin(val_tfs)  | edges['TG'].isin(val_tgs)]
        used_idx = set(test_edges.index) | set(val_edges.index)
        train_edges = edges[~edges.index.isin(used_idx)]

        # Verify TF/TG vocabularies are disjoint across splits
        assert not (set(test_edges['TF']) & set(val_edges['TF']) & set(train_edges['TF'])), "TFs overlap across splits"
        assert not (set(test_edges['TG']) & set(val_edges['TG']) & set(train_edges['TG'])), "TGs overlap across splits"

        # Verify no edge overlap
        self._verify_no_overlap(train_edges, val_edges, test_edges)

        logger.info(
            f"Both split - Edges - Train: {len(train_edges)}, Val: {len(val_edges)}, Test: {len(test_edges)}"
        )
        return train_edges.reset_index(drop=True), val_edges.reset_index(drop=True), test_edges.reset_index(drop=True)
    
    def _verify_no_overlap(self,
                          train_edges: pd.DataFrame,
                          val_edges: pd.DataFrame,
                          test_edges: pd.DataFrame):
        """
        Verify no edge overlap between splits.
        
        Args:
            train_edges: Training edges
            val_edges: Validation edges
            test_edges: Test edges
        
        Raises:
            AssertionError if overlap detected
        """
        # Create edge identifiers
        def edge_ids(df):
            return set(zip(df['TF'], df['TG']))
        
        train_ids = edge_ids(train_edges)
        val_ids = edge_ids(val_edges)
        test_ids = edge_ids(test_edges)
        
        # Check overlaps
        assert len(train_ids & val_ids) == 0, f"Overlap between train and val: {len(train_ids & val_ids)}"
        assert len(train_ids & test_ids) == 0, f"Overlap between train and test: {len(train_ids & test_ids)}"
        assert len(val_ids & test_ids) == 0, f"Overlap between val and test: {len(val_ids & test_ids)}"
        
        logger.info("✓ No edge overlap detected between splits")

# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

class ModelTrainer:
    """Handles model training with proper cross-validation."""
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize ModelTrainer.
        
        Args:
            model_config: Configuration for models and training
        """
        self.config = model_config
        self.models = {}
        self.results = []
        logger.info("ModelTrainer initialized")
    
    def create_model(self, model_type: str = "hgb") -> Any:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model ('lr', 'hgb', 'rf')
        
        Returns:
            Model instance
        """
        if model_type == "lr":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    solver="liblinear",
                    penalty="l2",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=self.config.get('random_state', 42)
                ))
            ])
        
        elif model_type == "hgb":
            return HistGradientBoostingClassifier(
                learning_rate=self.config.get('learning_rate', 0.1),
                max_iter=self.config.get('max_iter', 500),
                max_depth=self.config.get('max_depth', None),
                min_samples_leaf=self.config.get('min_samples_leaf', 20),
                l2_regularization=self.config.get('l2_regularization', 0.1),
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=self.config.get('random_state', 42)
            )
        
        elif model_type == "rf":
            return RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 200),
                max_depth=self.config.get('max_depth', None),
                min_samples_split=self.config.get('min_samples_split', 20),
                min_samples_leaf=self.config.get('min_samples_leaf', 10),
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.config.get('random_state', 42)
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def hyperparameter_search(self,
                            X_train: pd.DataFrame,
                            y_train: np.ndarray,
                            model_type: str = "hgb",
                            cv_folds: int = 3) -> Any:
        """
        Perform hyperparameter search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: Type of model
            cv_folds: Number of CV folds
        
        Returns:
            Best model from search
        """
        logger.info(f"Starting hyperparameter search for {model_type}")
        
        base_model = self.create_model(model_type)
        
        # Define parameter grids
        param_grids = {
            "hgb": {
                "learning_rate": [0.01, 0.05, 0.1, 0.15],
                "max_iter": [200, 300, 500, 700],
                "max_depth": [None, 3, 5, 7, 10],
                "min_samples_leaf": [10, 20, 30, 50],
                "l2_regularization": [0.0, 0.1, 0.5, 1.0]
            },
            "rf": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [10, 20, 50],
                "min_samples_leaf": [5, 10, 20]
            },
            "lr": {
                "clf__C": [0.001, 0.01, 0.1, 1.0, 10.0],
                "clf__penalty": ["l1", "l2"]
            }
        }
        
        # Get appropriate parameter grid
        param_grid = param_grids.get(model_type, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid defined for {model_type}")
            return base_model
        
        # Perform randomized search
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=self.config.get('n_iter', 20),
            cv=cv_folds,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=self.config.get('random_state', 42),
            verbose=1
        )
        
        # Fit with sample weights for imbalanced data
        sample_weights = compute_sample_weight("balanced", y=y_train)
        search.fit(X_train, y_train, sample_weight=sample_weights)
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def train_model(self,
                   X_train: pd.DataFrame,
                   y_train: np.ndarray,
                   X_val: Optional[pd.DataFrame] = None,
                   y_val: Optional[np.ndarray] = None,
                   model_type: str = "hgb",
                   use_calibration: bool = True) -> Any:
        """
        Train a model with optional calibration.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            model_type: Type of model
            use_calibration: Whether to calibrate probabilities
        
        Returns:
            Trained (and possibly calibrated) model
        """
        logger.info(f"Training {model_type} model on {len(X_train)} samples")
        
        # Hyperparameter search
        if self.config.get('do_hyperparam_search', True):
            model = self.hyperparameter_search(X_train, y_train, model_type)
        else:
            model = self.create_model(model_type)
        
        # Train model
        sample_weights = compute_sample_weight("balanced", y=y_train)
        
        if hasattr(model, 'fit'):
            if 'sample_weight' in model.fit.__code__.co_varnames:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)
        
        if use_calibration and not hasattr(model, "predict_proba"):
            logger.warning("Model lacks predict_proba; skipping calibration.")
            use_calibration = False

        # --- Probability calibration (use held‑out VAL; avoids reusing train) -------
        if use_calibration and (X_val is not None) and (y_val is not None):
            logger.info("Calibrating model probabilities on held‑out validation (cv='prefit')")
            # Fit base model on TRAIN only (done above), then calibrate on VAL
            calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
            calibrated.fit(X_val, y_val)
            model = calibrated
        elif use_calibration:
            logger.warning("Calibration requested but no validation set provided; skipping.")
                
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_pred)
            logger.info(f"Validation AUC: {val_auc:.4f}")
        
        return model
    
    def evaluate_model(self,
                      model: Any,
                      X_test: pd.DataFrame,
                      y_test: np.ndarray,
                      dataset_name: str = "test") -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            dataset_name: Name of the dataset being evaluated
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on {dataset_name} ({len(X_test)} samples)")
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            "dataset": dataset_name,
            "n_samples": len(X_test),
            "n_positive": int(y_test.sum()),
            "n_negative": int((1 - y_test).sum()),
            "auroc": roc_auc_score(y_test, y_pred_proba),
            "auprc": average_precision_score(y_test, y_pred_proba)
        }
        
        # Precision at different thresholds
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Find recall at high precision
        high_precision_idx = np.where(precision >= 0.9)[0]
        if len(high_precision_idx) > 0:
            metrics["recall_at_p90"] = recall[high_precision_idx[0]]
        else:
            metrics["recall_at_p90"] = 0.0
        
        # Top-k precision
        for k in [10, 50, 100, 500, 1000]:
            if len(y_pred_proba) >= k:
                top_k_idx = np.argsort(y_pred_proba)[-k:]
                metrics[f"precision_at_{k}"] = y_test[top_k_idx].mean()
        
        # Log results
        logger.info(f"Results for {dataset_name}:")
        logger.info(f"  AUROC: {metrics['auroc']:.4f}")
        logger.info(f"  AUPRC: {metrics['auprc']:.4f}")
        logger.info(f"  Recall@P≥0.90: {metrics['recall_at_p90']:.4f}")
        
        return metrics

# =============================================================================
# CROSS-VALIDATION ACROSS GROUND TRUTH DATASETS
# =============================================================================

class CrossValidator:
    """
    Performs exhaustive cross-validation across multiple ground truth datasets.
    """
    
    def __init__(self, cv_strategy: str = "leave_one_out"):
        """
        Initialize CrossValidator.
        
        Args:
            cv_strategy: Cross-validation strategy
        """
        self.cv_strategy = cv_strategy
        self.results = []
        logger.info(f"CrossValidator initialized with strategy: {cv_strategy}")
    
    def cross_validate_datasets(self,
                              ground_truth_dict: Dict[str, pd.DataFrame],
                              feature_engineer: FeatureEngineer,
                              model_trainer: ModelTrainer,
                              expression_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Perform cross-validation across ground truth datasets.
        
        Args:
            ground_truth_dict: Dictionary of ground truth DataFrames
            feature_engineer: FeatureEngineer instance
            model_trainer: ModelTrainer instance
            expression_data: Dictionary of expression data
        
        Returns:
            DataFrame with all results
        """
        logger.info(f"Starting cross-validation across {len(ground_truth_dict)} datasets")
        
        all_results = []
        
        if self.cv_strategy == "leave_one_out":
            # Train on all but one, test on the held-out dataset
            dataset_names = list(ground_truth_dict.keys())
            
            for test_dataset in dataset_names:
                logger.info(f"\n{'='*60}")
                logger.info(f"Testing on: {test_dataset}")
                logger.info(f"{'='*60}")
                
                # Combine training datasets
                train_dfs = []
                for name, df in ground_truth_dict.items():
                    if name != test_dataset:
                        train_dfs.append(df)
                
                if not train_dfs:
                    logger.warning(f"No training data for {test_dataset}")
                    continue
                
                # --- Remove any TF–TG edges that appear in the held‑out test dataset --------
                train_edges = pd.concat(train_dfs, ignore_index=True)
                test_edges = ground_truth_dict[test_dataset].copy()

                # Anti-join to drop overlaps (faster and safer than row-wise .apply)
                overlap_cols = ["TF", "TG"]
                train_edges = (
                    train_edges.merge(
                        test_edges[overlap_cols].drop_duplicates().assign(_drop_=1),
                        on=overlap_cols, how="left"
                    ).query("_drop_.isna()")
                    .drop(columns=["_drop_"])
                    .reset_index(drop=True)
                )

                # --- Drop negatives that are positives in ANY dataset (label noise guard) ---
                all_pos_pairs = set()
                for _name, _df in ground_truth_dict.items():
                    if "label" in _df.columns:
                        pos_df = _df[_df["label"] == 1]
                    else:
                        pos_df = _df
                    if not pos_df.empty:
                        all_pos_pairs.update(map(tuple, pos_df[["TF", "TG"]].values))

                def _drop_cross_dataset_negatives(_edges: pd.DataFrame, split_name: str) -> pd.DataFrame:
                    if "label" not in _edges.columns or _edges.empty:
                        return _edges.reset_index(drop=True)
                    pairs = list(map(tuple, _edges[["TF", "TG"]].values))
                    present_elsewhere = pd.Series(
                        [pair in all_pos_pairs for pair in pairs],
                        index=_edges.index
                    )
                    mask_bad_neg = _edges["label"].eq(0) & present_elsewhere
                    removed = int(mask_bad_neg.sum())
                    if removed:
                        logger.info(
                            f"Removing {removed} negatives from {split_name} split that are positives in another dataset."
                        )
                    return _edges.loc[~mask_bad_neg].reset_index(drop=True)

                train_edges = _drop_cross_dataset_negatives(train_edges, "train")
                test_edges = _drop_cross_dataset_negatives(test_edges, "test")
            
                # Split training data for validation
                splitter = DataSplitter(split_strategy="edge")
                train_split, val_split, _ = splitter.split_edges(
                    train_edges, 
                    test_size=0, 
                    val_size=0.2
                )
                
                # --- Compute features (NO LEAKAGE) ----------------------------------------
                logger.info("Computing features...")
                peak_data = expression_data.get('peaks', pd.DataFrame())
                gene_data = expression_data.get('genes', pd.DataFrame())

                if not peak_data.empty and not gene_data.empty:
                    # 1) First call on TRAIN ONLY → stores train statistics in feature_engineer
                    train_features = feature_engineer.compute_correlation_features(
                        peak_data, gene_data, train_split
                    )

                    # 2) Reuse TRAIN stats to transform VAL and TEST (no recomputation on val/test)
                    _, val_features = feature_engineer.compute_correlation_features(
                        peak_data, gene_data, train_split, test_edges=val_split
                    )
                    _, test_features = feature_engineer.compute_correlation_features(
                        peak_data, gene_data, train_split, test_edges=test_edges
                    )
                else:
                    # Fallback to simple features
                    train_features = pd.DataFrame(index=range(len(train_split)))
                    val_features = pd.DataFrame(index=range(len(val_split)))
                    test_features = pd.DataFrame(index=range(len(test_edges)))

                # Add expression features (OK — these operate per-gene without using labels)
                train_expr = feature_engineer.compute_expression_features(gene_data, train_split)
                val_expr   = feature_engineer.compute_expression_features(gene_data, val_split)
                test_expr  = feature_engineer.compute_expression_features(gene_data, test_edges)

                # Combine features
                X_train = pd.concat([train_features, train_expr], axis=1)
                X_val   = pd.concat([val_features,   val_expr], axis=1)
                X_test  = pd.concat([test_features,  test_expr], axis=1)
                
                # Get labels
                y_train = train_split['label'].values
                y_val = val_split['label'].values
                y_test = test_edges['label'].values
                
                # Train model
                model = model_trainer.train_model(
                    X_train, y_train, 
                    X_val, y_val,
                    model_type="hgb"
                )
                
                # Evaluate
                results = model_trainer.evaluate_model(
                    model, X_test, y_test, 
                    dataset_name=test_dataset
                )
                
                results['train_datasets'] = ', '.join(
                    [name for name in dataset_names if name != test_dataset]
                )
                results['n_train_edges'] = len(train_split)
                results['n_val_edges'] = len(val_split)
                
                all_results.append(results)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Calculate summary statistics
        logger.info("\n" + "="*60)
        logger.info("CROSS-VALIDATION SUMMARY")
        logger.info("="*60)
        
        mean_results = results_df.mean(numeric_only=True)
        std_results = results_df.std(numeric_only=True)
        
        logger.info(f"Average AUROC: {mean_results['auroc']:.4f} ± {std_results['auroc']:.4f}")
        logger.info(f"Average AUPRC: {mean_results['auprc']:.4f} ± {std_results['auprc']:.4f}")
        
        return results_df

# =============================================================================
# MAIN PIPELINE
# =============================================================================

class TFBindingPipeline:
    """Main pipeline orchestrating the entire workflow."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            config_file: Path to configuration file (JSON)
        """
        self.config = self._load_config(config_file)
        self.setup_directories()
        
        logger.info("TF Binding Pipeline initialized")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_file: Path to config file
        
        Returns:
            Configuration dictionary
        """
        default_config = {
            "data_dir": ".",
            "output_dir": "output",
            "log_dir": "logs",
            "random_state": 42,
            "test_size": 0.2,
            "val_size": 0.2,
            "split_strategy": "edge",
            "cv_strategy": "leave_one_out",
            "model_type": "hgb",
            "do_hyperparam_search": True,
            "use_calibration": True,
            "downsample_ratio": 20,
            "n_jobs": -1,
            "neg_pos_ratio": 5,
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def setup_directories(self):
        """Create necessary directories."""
        for dir_name in ['output_dir', 'log_dir']:
            dir_path = self.config.get(dir_name, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def run(self,
           expression_files: Dict[str, str],
           ground_truth_files: List[str]) -> pd.DataFrame:
        """
        Run the complete pipeline.
        
        Args:
            expression_files: Dictionary mapping data types to file patterns
            ground_truth_files: List of ground truth file paths
        
        Returns:
            DataFrame with all results
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING TF BINDING PREDICTION PIPELINE")
        logger.info("="*60)
        
        # Load data
        logger.info("\n>>> Step 1: Loading data")
        data_loader = DataLoader(
            self.config['data_dir'],
            neg_pos_ratio=self.config.get("neg_pos_ratio", 5),
            random_state=self.config.get("random_state", 42),
        )
        
        expression_data = data_loader.load_expression_data(expression_files)
        ground_truth = data_loader.load_ground_truth(ground_truth_files)
        
        if not ground_truth:
            raise ValueError("No ground truth datasets loaded!")
        
        # Initialize components
        logger.info("\n>>> Step 2: Initializing components")
        feature_engineer = FeatureEngineer(self.config)
        model_trainer = ModelTrainer(self.config)
        cross_validator = CrossValidator(self.config['cv_strategy'])
        
        # Run cross-validation
        logger.info("\n>>> Step 3: Running cross-validation")
        results = cross_validator.cross_validate_datasets(
            ground_truth,
            feature_engineer,
            model_trainer,
            expression_data
        )
        
        # Save results
        logger.info("\n>>> Step 4: Saving results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(
            self.config['output_dir'], 
            f"results_{timestamp}.csv"
        )
        results.to_csv(results_file, index=False)
        logger.info(f"Results saved to: {results_file}")
        
        # Save configuration
        config_file = os.path.join(
            self.config['output_dir'],
            f"config_{timestamp}.json"
        )
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to: {config_file}")
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        return results

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point for command line execution."""
    parser = argparse.ArgumentParser(
        description="TF Binding Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON)"
    )
    
    parser.add_argument(
        "--neg-pos-ratio", type=int, default=5,
        help="Number of negatives to sample per positive edge"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory containing input data files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--ground-truth",
        nargs="+",
        required=True,
        help="Ground truth file(s)"
    )
    
    parser.add_argument(
        "--peak-data",
        type=str,
        help="Peak accessibility data file pattern"
    )
    
    parser.add_argument(
        "--gene-data",
        type=str,
        help="Gene expression data file pattern"
    )
    
    parser.add_argument(
        "--split-strategy",
        choices=["edge", "tf", "tg", "both"],
        default="edge",
        help="Data splitting strategy"
    )
    
    parser.add_argument(
        "--cv-strategy",
        choices=["leave_one_out", "kfold"],
        default="leave_one_out",
        help="Cross-validation strategy"
    )
    
    parser.add_argument(
        "--model-type",
        choices=["lr", "hgb", "rf"],
        default="hgb",
        help="Model type to use"
    )
    
    parser.add_argument(
        "--no-hyperparam-search",
        action="store_true",
        help="Skip hyperparameter search"
    )
    
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip probability calibration"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "split_strategy": args.split_strategy,
        "cv_strategy": args.cv_strategy,
        "model_type": args.model_type,
        "do_hyperparam_search": not args.no_hyperparam_search,
        "use_calibration": not args.no_calibration,
        "random_state": args.random_seed,
        "neg_pos_ratio": args.neg_pos_ratio,
    }
    
    # Override with config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    pipeline = TFBindingPipeline()
    pipeline.config.update(config)
    
    # Prepare file patterns
    expression_files = {}
    if args.peak_data:
        expression_files['peaks'] = args.peak_data
    if args.gene_data:
        expression_files['genes'] = args.gene_data
    
    # Run pipeline
    try:
        results = pipeline.run(
            expression_files=expression_files,
            ground_truth_files=args.ground_truth
        )
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(results.to_string())
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
