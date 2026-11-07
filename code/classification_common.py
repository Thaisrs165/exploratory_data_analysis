"""Shared data/metric utilities for the Classification-or-Regression project (classification track).

Reuses bank_preprocessing.py (the EDA's fitted pipeline) for data loading and
the train/test split; adds only what the modelling project needs on top: an
internal train/validation split (carved out of the training set, never
touching test) and the classification metrics used throughout steps 1 and 5.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from bank_preprocessing import build_preprocessor, load_data, split  # noqa: E402

RANDOM_STATE = 42
VAL_SIZE = 0.15  # of the training set


def load_train_val_test():
    """(X_train2, X_val, X_test, y_train2, y_val, y_test): the EDA's 80/20 test
    split, with the 80% training side further split (stratified on y) into an
    internal train/validation pair. Test is returned untouched."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)

    X_train2, X_val, y_train2, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=RANDOM_STATE
    )
    return X_train2, X_val, X_test, y_train2, y_val, y_test, X_train, y_train


def fit_pipeline(X_train2):
    """The EDA's preprocessing Pipeline, fit on the inner training fold only."""
    return build_preprocessor().fit(X_train2)


def best_threshold_by_f1(y_true, y_proba) -> float:
    """Validation-only: the probability threshold that maximizes F1 on y_true/y_proba."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    best_idx = int(np.argmax(f1[:-1]))
    return float(thresholds[best_idx])


def classification_metrics(y_true, y_proba, threshold: float) -> dict:
    """Threshold-dependent (confusion matrix, precision/recall/F1 per class, macro F1)
    and threshold-independent (ROC-AUC, PR-AUC) metrics, all from predicted probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision_0": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_0": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1_0": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }
