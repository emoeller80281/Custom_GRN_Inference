import os
import logging
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da

from dask_ml.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.inspection import permutation_importance
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

import xgboost as xgb


def train_xgboost_dask(X_train_dd, y_train_dd, feature_names):
    """
    Train an XGBoost model using DaskDMatrix (distributed).
    
    Args:
        X_train_dd (dask.dataframe.DataFrame): Training features (Dask)
        y_train_dd (dask.dataframe.Series): Training labels (Dask)
        feature_names (list): List of feature column names
        
    Returns:
        booster (xgboost.Booster): Trained XGBoost Booster object
    """
    logging.info("Training XGBoost model with Dask")

    # Create a DaskDMatrix
    dtrain = xgb.dask.DaskDMatrix(
        client=None,  # If using local CPU, otherwise pass Dask client
        data=X_train_dd,
        label=y_train_dd,
        feature_names=feature_names
    )

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',    # highly recommended for large data
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 1,
        'reg_alpha': 0.5,
        'reg_lambda': 1,
        'random_state': 42
    }

    output = xgb.dask.train(
        client=None,       # None uses threads; you could pass a Dask client
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train')],
    )

    booster = output['booster']  # This is the trained model
    booster.set_attr(feature_names=",".join(feature_names))  # Save feature names

    return booster

def parameter_grid_search(X_train_dd, y_train_dd, features, cpu_count, fig_dir):
    logging.info("⚙️ Starting XGBoost hyperparameter grid search")

    # Convert Dask → pandas (required for GridSearchCV)
    X_train = X_train_dd.compute()
    y_train = y_train_dd.compute()

    # Combine into single DataFrame for balancing
    train_data = X_train.copy()
    train_data["label"] = y_train

    # Balance classes
    pos_train = train_data[train_data["label"] == 1]
    neg_train = train_data[train_data["label"] == 0]
    neg_train_sampled = neg_train.sample(n=len(pos_train), random_state=42)
    train_data_balanced = pd.concat([pos_train, neg_train_sampled])

    X_bal = train_data_balanced[features]
    y_bal = train_data_balanced["label"]

    # Define parameter grid
    param_grid = {
        "n_estimators":      [50, 100, 200],
        "max_depth":         [4, 6, 8],
        "gamma":             [0, 1, 5],
        "reg_alpha":         [0.0, 0.5, 1.0],
        "reg_lambda":        [1.0, 2.0, 5.0],
        "subsample":         [0.8, 1.0],
        "colsample_bytree":  [0.8, 1.0],
    }

    # Initialize classifier (use hist method for speed)
    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        tree_method="hist",
        use_label_encoder=False
    )

    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform grid search
    grid = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=cpu_count,
        verbose=2
    )

    logging.info("Running GridSearchCV on balanced training set")
    grid.fit(X_bal, y_bal)

    logging.info(f"  Best CV score:  {grid.best_score_:.4f}")
    logging.info(f"  Best parameters: {grid.best_params_}")

    best_model = grid.best_estimator_

    # Save plot
    output_dir = os.path.join(fig_dir, "parameter_search")
    os.makedirs(output_dir, exist_ok=True)

    plot_feature_importance(
        features=features,
        model=best_model,
        fig_dir=output_dir
    )

def xgb_classifier_from_booster(booster: xgb.Booster, feature_names: list | np.ndarray | pd.Index) -> xgb.XGBClassifier:
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
    clf.n_features_in_ = len(feature_names)
    clf.feature_names_in_ = np.array(feature_names)
    clf.classes_ = np.array([0, 1])
    return clf
