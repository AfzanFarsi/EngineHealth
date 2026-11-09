"""Metrics for regression (RUL) and classification (fault risk)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score,
                             mean_squared_error)

import config as cfg


def nasa_score(y_true, y_pred) -> float:
    """NASA PHM08 asymmetric score (lower is better; late predictions cost more)."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / cfg.NASA_A1) - 1, np.exp(d / cfg.NASA_A2) - 1)))


def regression_metrics(y_true, y_pred, with_score: bool = False) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    out = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }
    if with_score:
        out["NASA_score"] = nasa_score(y_true, y_pred)
    return out


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_score)) if y_score is not None and len(set(y_true)) > 1 else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    return out


def health_category(rul: float) -> str:
    """Map a predicted RUL (cycles) to a health category."""
    if rul <= cfg.FAULT_THRESHOLD:
        return "Critical"
    if rul <= cfg.WARNING_THRESHOLD:
        return "Warning"
    return "Healthy"


def maintenance_advice(rul: float) -> str:
    cat = health_category(rul)
    return {
        "Critical": "Schedule maintenance immediately - engine is close to the end of its useful life.",
        "Warning": "Plan maintenance in the near future and increase monitoring frequency.",
        "Healthy": "No action required. Continue routine condition monitoring.",
    }[cat]
