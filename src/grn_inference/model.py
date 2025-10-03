import os
import logging
import numpy as np
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client

from dask_ml.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid, ParameterSampler
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.inspection import permutation_importance
from scipy.stats import uniform, randint
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from typing import Union
from tqdm import tqdm

import xgboost as xgb

from dask.distributed import Client

def undersample_training_set(X_dd, y_dd, seed=42):
    """Undersample the majority class to balance positives and negatives.

    Parameters
    ----------
    X_dd : dask.dataframe.DataFrame
        Feature matrix where rows correspond to observations.
    y_dd : dask.dataframe.Series
        Binary labels associated with ``X_dd``.
    seed : int, optional
        Random seed used when shuffling and sampling.  Defaults to ``42``.

    Returns
    -------
    tuple (dask.dataframe.DataFrame, dask.dataframe.Series)
        Balanced feature matrix and label vector.  The number of negative
        examples is randomly reduced to match the count of positive examples.

    Notes
    -----
    This function logs the number of positive/negative samples and may trigger
    Dask computations when determining sample counts.
    """
    logging.info("Scaling the number of positive and negatives values for training")
    
    # Combine X and y
    y_dd = y_dd.rename("label")
    df = X_dd.assign(label=y_dd)

    # Split into positive and negative
    pos_df = df[df.label == 1]
    neg_df = df[df.label == 0]
    
    # Compute number of positives
    n_pos = pos_df.shape[0].compute()
    n_neg = neg_df.shape[0].compute()
    frac = n_pos / n_neg
    
    logging.info(f'\tPositive values: {n_pos}')
    logging.info(f'\tNegative values: {n_neg}')
    
    # Randomly sample negatives
    neg_df_sampled = neg_df.sample(frac=frac, random_state=seed)
    
    logging.info(f"\tUnderscaling so positive and negative classes both have {n_pos} values")

    # Combine
    balanced_df = dd.concat([pos_df, neg_df_sampled])
    balanced_df = balanced_df.shuffle(on="label", seed=seed)

    # Separate X and y again
    y_balanced_dd = balanced_df["label"]
    X_balanced_dd = balanced_df.drop(columns=["label"])

    return X_balanced_dd, y_balanced_dd

def train_xgboost_dask(train_test_dir, feature_names, client=None):
    """Train an XGBoost model on data stored in ``train_test_dir`` using Dask.

    Parameters
    ----------
    train_test_dir : str
        Directory containing ``X_train.parquet`` and ``y_train.parquet`` files
        produced by :func:`dask_ml.model_selection.train_test_split`.
    feature_names : list of str
        Names of the feature columns in the training data.
    client : dask.distributed.Client, optional
        Existing Dask client.  If ``None`` a local threaded client is created
        for the duration of the call.

    Returns
    -------
    xgboost.Booster
        The trained booster object.

    Notes
    -----
    The function reads training data from disk, optionally starts a local Dask
    client and logs progress messages.  Use ``booster.save_model`` to persist
    the trained model.
    """
    logging.info("Training XGBoost model with Dask")
    logging.info("Reading X_train.parquet")
    X_train_dd = dd.read_parquet(os.path.join(train_test_dir, "X_train.parquet"))
    logging.info("\tDONE!")
    
    logging.info("Reading y_train.parquet")
    y_train_dd = dd.read_parquet(os.path.join(train_test_dir, "y_train.parquet"))["label"]
    logging.info("\tDONE!")

    client_created = False
    if client is None:
        logging.info("No Dask client provided — starting a local threaded client")
        client = Client(processes=False)
        client_created = True
        
    X_train_dd_bal, y_train_dd_bal = undersample_training_set(X_train_dd, y_train_dd)

    dtrain = xgb.dask.DaskDMatrix(
        client=client,
        data=X_train_dd_bal,
        label=y_train_dd_bal,
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

def parameter_grid_search(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    out_dir: str,
    fig_dir: str,
    cpu_count: int,
):
    """Perform a random parameter search for an XGBoost classifier.

    Parameters
    ----------
    X_tr, y_tr : pandas.DataFrame, pandas.Series
        Training features and labels.
    X_val, y_val : pandas.DataFrame, pandas.Series
        Validation features and labels used to score each parameter set.
    out_dir : str
        Directory where the search results will be written as a parquet file.
    fig_dir : str
        Directory for saving the summary scatter plot of AUROC vs. feature
        importance entropy.
    cpu_count : int
        Number of parallel jobs to run.

    Returns
    -------
    xgb.XGBClassifier
        Model retrained using the best set of parameters found.

    Notes
    -----
    This function performs the search using :class:`joblib.Parallel` and logs
    progress.  A ``grid_search_results.parquet`` file and a PNG figure showing
    performance are saved in ``out_dir`` and ``fig_dir`` respectively.
    """
    param_dist = {
        'n_estimators':      randint(100, 500),
        'max_depth':         randint(4, 12),
        'gamma':             uniform(0, 20),
        'reg_alpha':         uniform(0, 5),
        'reg_lambda':        uniform(0.5, 10),
        'subsample':         uniform(0.6, 0.4),           # 0.6 to 1.0
        'colsample_bytree':  uniform(0.6, 0.4),           # 0.6 to 1.0
        'learning_rate':     uniform(0.01, 0.3)           # 0.01 to 0.31
    }
    n_iter = 100  # Try 100 random combos
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))


    # # parameter grid
    # param_grid = {
    #     'n_estimators':      [50, 100, 200],
    #     'max_depth':         [4, 6, 8],
    #     'gamma':             [0, 1, 5],
    #     'reg_alpha':         [0.0, 0.5, 1.0],
    #     'reg_lambda':        [1.0, 2.0, 5.0],
    #     'subsample':         [0.8, 1.0],
    #     'colsample_bytree':  [0.8, 1.0],
    #     'learning_rate':     [0.01, 0.1]
    # }
    # grid_list = list(ParameterGrid(param_grid))
    # total = len(grid_list)
    # logging.info(f"Parameter grid has {total} combinations")

    def eval_params(params: dict) -> dict:
        model = xgb.XGBClassifier(
            **params,
            n_jobs=1,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        y_pred = model.predict_proba(X_val)[:, 1]
        val_ap = average_precision_score(y_val, y_pred)
        val_auc = roc_auc_score(y_val, y_pred)

        imp = model.feature_importances_
        p = imp / np.sum(imp + 1e-12)
        entropy = -np.sum(p * np.log(p + 1e-12))
        cv = np.std(p) / (np.mean(p) + 1e-12)

        return {**params,
                'val_ap': val_ap,
                'val_auc': val_auc,
                'imp_entropy': entropy,
                'imp_cv': cv}

    # # parallel evaluation
    # logging.info(f"Starting parallel grid search on {cpu_count} cores...")
    # results = Parallel(n_jobs=cpu_count)(
    #     delayed(eval_params)(p) for p in tqdm(
    #         grid_list,
    #         total=total,
    #         smoothing=0.9,
    #         desc="Grid search",
    #         miniters=max(1, total // 20)  # update every 5%
    #     )
    # )
    
        # parallel evaluation
    logging.info(f"Starting parallel grid search on {cpu_count} cores...")
    results = Parallel(n_jobs=cpu_count)(
        delayed(eval_params)(p) for p in tqdm(
            param_list,
            total=n_iter,
            smoothing=0.9,
            desc="Grid search",
            miniters=max(1, n_iter // 20)  # update every 5%
        )
    )

    df_res = pd.DataFrame(results)
    results_out_path = os.path.join(out_dir, 'grid_search_results.parquet')
    os.makedirs(out_dir, exist_ok=True)
    df_res.to_parquet(results_out_path, index=False, compression='snappy', engine='pyarrow')
    logging.info(f"Saved parameter search results to {results_out_path}")

    # identify best
    best_auc = df_res.loc[df_res.val_auc.idxmax()]
    best_flat = df_res.loc[df_res.imp_entropy.idxmax()]
    logging.info(f"Best AUROC={best_auc.val_auc:.4f} at {best_auc.to_dict()}")
    logging.info(f"Highest entropy={best_flat.imp_entropy:.4f} at {best_flat.to_dict()}")
    
    # Compute ranks (lower rank = better)
    df_res["auc_rank"] = df_res["val_auc"].rank(ascending=False, method="min")
    df_res["entropy_rank"] = df_res["imp_entropy"].rank(ascending=False, method="min")

    # Sum ranks
    df_res["combined_rank"] = df_res["auc_rank"] + df_res["entropy_rank"]

    # Select best overall
    best_overall = df_res.loc[df_res["combined_rank"].idxmin()]
    logging.info(f"Best overall (AUROC + entropy rank) = {best_overall.to_dict()}")

    # retrain best by AP on full X/y
    # cast best parameters to correct types
    best_params = {
        'n_estimators': int(best_overall['n_estimators']),
        'max_depth': int(best_overall['max_depth']),
        'gamma': int(best_overall['gamma']),
        'reg_alpha': float(best_overall['reg_alpha']),
        'reg_lambda': float(best_overall['reg_lambda']),
        'subsample': float(best_overall['subsample']),
        'colsample_bytree': float(best_overall['colsample_bytree']),
        'learning_rate': float(best_overall['learning_rate'])
    }
    
    best_model = xgb.XGBClassifier(
        **best_params,
        random_state=42,
        use_label_encoder=False
    )
    best_model.fit(X_tr, y_tr)
    
    json_model_path = os.path.join(out_dir, 'best_model_combined_rank.json')
    best_model.get_booster().save_model(json_model_path)
    logging.info(f"Saved best model by combined rank to {json_model_path}")

    # scatter plot AP vs entropy
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,6))
    plt.scatter(df_res.imp_entropy, df_res.val_auc, alpha=0.7)
    plt.xlabel('Feature Importance Entropy', fontsize=14)
    plt.ylabel('AUROC Score', fontsize=14)
    plt.title('AUROC Performance vs Feature Importance Entropy', fontsize=13)
    plt.tight_layout()
    plot_out = os.path.join(fig_dir, 'param_grid_search_auroc_vs_feature_importance.png')
    plt.savefig(plot_out, dpi=200)
    plt.close()
    logging.info(f"Saved flatness vs AUROC plot to {plot_out}")
    
    return best_model

def xgb_classifier_from_booster(booster: xgb.Booster, feature_names: Union[list, np.ndarray, pd.Index]) -> xgb.XGBClassifier:
    """Wrap a trained :class:`xgboost.Booster` in an ``XGBClassifier`` instance.

    Parameters
    ----------
    booster : xgboost.Booster
        Trained booster object, typically obtained from ``xgb.dask.train``.
    feature_names : sequence of str
        Names of the features that were used to train ``booster``.

    Returns
    -------
    xgboost.XGBClassifier
        Classifier instance with the booster attached so that scikit-learn
        utilities such as ``predict_proba`` and ``permutation_importance`` can
        be used.

    Notes
    -----
    The returned classifier does not need to be fitted again.  Only the booster
    attributes are set; therefore calls that rely on training data (e.g.
    ``fit``) should be avoided.
    """
    clf = xgb.XGBClassifier()
    clf._Booster = booster
    clf.classes_ = np.array([0, 1])
    clf._le = xgb.compat._create_le(clf.classes_)
    clf._feature_names = list(feature_names)
    return clf