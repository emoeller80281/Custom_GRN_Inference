import os
import logging
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client

from dask_ml.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from typing import Union

import xgboost as xgb

from dask.distributed import Client

def train_xgboost_dask(X_train_dd, y_train_dd, feature_names, client=None):
    """
    Train an XGBoost model using DaskDMatrix (distributed).
    
    Args:
        X_train_dd (dask.dataframe.DataFrame): Training features (Dask)
        y_train_dd (dask.dataframe.Series): Training labels (Dask)
        feature_names (list): List of feature column names
        client (dask.distributed.Client or None): Optional Dask client

    Returns:
        xgboost.Booster: Trained model
    """
    logging.info("Training XGBoost model with Dask")

    client_created = False
    if client is None:
        logging.info("No Dask client provided — starting a local threaded client")
        client = Client(processes=False)
        client_created = True

    dtrain = xgb.dask.DaskDMatrix(
        client=client,
        data=X_train_dd,
        label=y_train_dd,
        feature_names=feature_names
    )

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 1,
        'reg_alpha': 0.5,
        'reg_lambda': 1,
        'random_state': 42,
        'verbosity': 0
    }

    output = xgb.dask.train(
        client=client,
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[],
    )

    booster = output['booster']
    booster.set_attr(feature_names=",".join(feature_names))

    if client_created:
        logging.info("Closing auto-started Dask client")
        client.close()

    return booster


def parameter_grid_search(X_train_dd, y_train_dd, features, cpu_count, fig_dir):
    logging.info("Starting XGBoost hyperparameter grid search (auto-balanced)")

    # Convert Dask → Pandas
    X_train = X_train_dd.compute()
    y_train = y_train_dd.compute()

    # Log class distribution
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    logging.info(f"Training samples: {len(y_train)} — Pos: {n_pos}, Neg: {n_neg}")

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Training set must contain both positive and negative samples.")

    # Compute scale_pos_weight for imbalance handling
    scale_pos_weight = n_neg / n_pos

    # Define parameter grid
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [4, 6, 8],
        "gamma": [0, 1, 5],
        "reg_alpha": [0.0, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        tree_method="hist",
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=cpu_count,
        verbose=2
    )

    logging.info("Running GridSearchCV on imbalanced training set")
    grid.fit(X_train, y_train)

    logging.info(f"  Best CV score:  {grid.best_score_:.4f}")
    logging.info(f"  Best parameters: {grid.best_params_}")

    return grid

    

def xgb_classifier_from_booster(booster: xgb.Booster, feature_names: Union[list, np.ndarray, pd.Index]) -> xgb.XGBClassifier:
    """
    Converts a trained XGBoost Booster (e.g., from Dask) to a scikit-learn XGBClassifier.
    This allows use of sklearn-compatible APIs like predict_proba, permutation_importance, etc.

    Parameters:
    -----------
    booster : xgb.Booster
        Trained Booster object (e.g. from xgb.dask.train)

    feature_names : list or array
        List of feature names used during training

    Returns:
    --------
    xgb.XGBClassifier
        Fully compatible sklearn-style classifier loaded from booster
    """
    clf = xgb.XGBClassifier()
    clf._Booster = booster
    clf.classes_ = np.array([0, 1])
    clf._le = xgb.compat._create_le(clf.classes_)
    clf._feature_names = list(feature_names)
    return clf