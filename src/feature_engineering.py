"""Temporal feature engineering for engine sensor time-series.

All features are *causal*: the value at cycle t uses only cycles <= t of the same
engine, so a prediction made at cycle t never "sees the future".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


def engineer_features(norm_df: pd.DataFrame, sensors: list[str], window: int = cfg.WINDOW) -> pd.DataFrame:
    """Build the model input table (rows keep the order of `norm_df`).

    Per sensor: current value, rolling mean, rolling std, and trend
    (average change per cycle over the window).  Plus the cycle number.
    """
    df = norm_df.sort_values([cfg.ID_COL, cfg.CYCLE_COL])
    g = df.groupby(cfg.ID_COL, sort=False)[sensors]
    roll = g.rolling(window, min_periods=1)
    mean = roll.mean().reset_index(level=0, drop=True)
    std = g.rolling(window, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
    lagged = g.shift(window - 1)
    first = g.transform("first")
    lagged = lagged.fillna(first)
    steps = np.minimum(df[cfg.CYCLE_COL].to_numpy() - 1, window - 1).clip(min=1)[:, None]
    trend = (df[sensors] - lagged) / steps

    feats = {cfg.CYCLE_COL: df[cfg.CYCLE_COL].astype(float)}
    for s in sensors:
        feats[s] = df[s]
        feats[f"{s}_mean{window}"] = mean[s]
        feats[f"{s}_std{window}"] = std[s]
        feats[f"{s}_trend{window}"] = trend[s]
    out = pd.DataFrame(feats, index=df.index)
    return out.loc[norm_df.index]          # restore original row order


def feature_groups(sensors: list[str], window: int = cfg.WINDOW) -> dict[str, list[str]]:
    """Names of the feature families (used for polynomial features / reporting)."""
    return {
        "base": list(sensors),
        "rolling_mean": [f"{s}_mean{window}" for s in sensors],
        "rolling_std": [f"{s}_std{window}" for s in sensors],
        "trend": [f"{s}_trend{window}" for s in sensors],
    }
