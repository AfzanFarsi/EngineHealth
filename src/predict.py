"""Load the saved models and make predictions (used by the website and the tests)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

import config as cfg
from src.evaluate import health_category, maintenance_advice


def bundle_path(subset: str, kind: str = "rul", models_dir=None):
    return (models_dir or cfg.MODELS_DIR) / f"{subset}_{kind}_model.joblib"


def load_bundle(subset: str, kind: str = "rul", models_dir=None) -> dict:
    """kind = 'rul' (regression) or 'risk' (fault-risk classifier)."""
    path = bundle_path(subset, kind, models_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\nTrain it first with:  python -m src.train --subset {subset}")
    return joblib.load(path)


def make_features(preprocessor, raw_df: pd.DataFrame, window: int = cfg.WINDOW) -> pd.DataFrame:
    """raw sensor table -> model input table (index = row order of raw_df)."""
    from src.feature_engineering import engineer_features
    norm = preprocessor.transform(raw_df)
    return engineer_features(norm, preprocessor.sensors_, window)


def validate_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check a user-supplied table (26 NASA columns, with or without header)."""
    if list(df.columns) != cfg.RAW_COLUMNS:
        if df.shape[1] == len(cfg.RAW_COLUMNS):
            df = df.copy()
            df.columns = cfg.RAW_COLUMNS
        else:
            raise ValueError(f"Expected {len(cfg.RAW_COLUMNS)} columns "
                             f"({', '.join(cfg.RAW_COLUMNS[:7])}, ...), got {df.shape[1]}.")
    if df.isna().any().any():
        raise ValueError("The file contains missing values.")
    df = df.apply(pd.to_numeric, errors="raise")
    return df.sort_values([cfg.ID_COL, cfg.CYCLE_COL]).reset_index(drop=True)


def predict_rul(bundle: dict, raw_df: pd.DataFrame) -> pd.DataFrame:
    """RUL prediction for every row of raw_df (rolling features use only past cycles)."""
    X = make_features(bundle["preprocessor"], raw_df, bundle["window"])
    pred = np.clip(bundle["model"].predict(X[bundle["feature_names"]]), 0, cfg.RUL_CAP)
    out = raw_df[[cfg.ID_COL, cfg.CYCLE_COL]].copy()
    out["rul_pred"] = pred
    out["rul_low"] = np.clip(pred + bundle["residual_q"]["low"], 0, None)
    out["rul_high"] = np.clip(pred + bundle["residual_q"]["high"], 0, cfg.RUL_CAP)
    out["category"] = [health_category(p) for p in pred]
    out["advice"] = [maintenance_advice(p) for p in pred]
    return out


def predict_risk(bundle: dict, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Probability-like score that RUL <= FAULT_THRESHOLD for every row."""
    X = make_features(bundle["preprocessor"], raw_df, bundle["window"])[bundle["feature_names"]]
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X)[:, 1]
    else:
        score = 1 / (1 + np.exp(-model.decision_function(X)))
    out = raw_df[[cfg.ID_COL, cfg.CYCLE_COL]].copy()
    out["risk_score"] = score
    out["risk_flag"] = model.predict(X).astype(int)
    return out
