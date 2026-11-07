"""Shared data/metric utilities for the Classification-or-Regression project (regression track).

Reuses bike_preprocessing.py (the EDA's fitted pipeline) for data loading and
the train/test split; adds only what the modelling project needs on top:
an internal train/validation split (carved out of the training set, never
touching test) and the regression metrics used throughout steps 1 and 5.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from bike_preprocessing import build_preprocessor, load_data, split  # noqa: E402

RANDOM_STATE = 42
VAL_FOLDS = 6  # 1/6 ~= 16.7% of the training set held out for validation


def load_train_val_test():
    """(X_train2, X_val, X_test, y_train2, y_val, y_test): the EDA's 80/20 test
    split, with the 80% training side further split (same grouped-by-date,
    stratified-by-season recipe) into an internal train/validation pair.
    Test is returned untouched and is not to be looked at before Step 5."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)

    sgkf = StratifiedGroupKFold(n_splits=VAL_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(sgkf.split(X_train, X_train["seasons"], groups=X_train["date"]))
    X_train2, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train2, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    return X_train2, X_val, X_test, y_train2, y_val, y_test, X_train, y_train


def fit_pipeline(X_train2):
    """The EDA's preprocessing Pipeline, fit on the inner training fold only."""
    return build_preprocessor().fit(X_train2)


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R2 in the original bikes/hour scale, plus MAPE (the target's
    minimum is 2, not 0, so MAPE is computable, but <0.5% of rows have
    y < 10 and can swing it - report it, flag it as the least robust of the four)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape,
    }


def sqrt_transform(y):
    return np.sqrt(np.asarray(y, dtype=float))


def sqrt_inverse(y_sqrt):
    """Inverse of sqrt_transform; squares are always >= 0 so no clipping is needed."""
    return np.square(np.asarray(y_sqrt, dtype=float))
