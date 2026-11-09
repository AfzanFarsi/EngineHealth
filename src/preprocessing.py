"""Data-quality analysis and the fitted preprocessing step.

The Preprocessor learns everything from *training engines only* (operating
condition clusters, scaling statistics, which sensors to keep) and can then be
applied to validation / test / new data without leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import config as cfg


# ------------------------------------------------------------------ EDA ----
def data_quality_report(df: pd.DataFrame) -> dict:
    """Missing values, duplicates, constant columns and IQR outlier counts."""
    cols = cfg.SETTING_COLS + cfg.SENSOR_COLS
    q1, q3 = df[cols].quantile(0.25), df[cols].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[cols] < q1 - 3 * iqr) | (df[cols] > q3 + 3 * iqr)).sum()
    lengths = df.groupby(cfg.ID_COL)[cfg.CYCLE_COL].max()
    return {
        "rows": int(len(df)),
        "engines": int(df[cfg.ID_COL].nunique()),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [c for c in cols if df[c].nunique() <= 1],
        "engine_life_min": int(lengths.min()),
        "engine_life_mean": round(float(lengths.mean()), 1),
        "engine_life_max": int(lengths.max()),
        "outliers_3xIQR_per_column": {c: int(v) for c, v in outliers.items() if v > 0},
        "cycles_are_consecutive": bool(
            (df.groupby(cfg.ID_COL)[cfg.CYCLE_COL].apply(lambda s: (s.diff().dropna() == 1).all())).all()
        ),
    }


# --------------------------------------------------------- preprocessor ----
class Preprocessor:
    """Operating-condition aware sensor normalisation + feature selection.

    Steps (all learned on training engines only):
      1. Operating condition id: K-Means on the 3 operating settings when the
         subset has several conditions (FD002/FD004); a single group otherwise.
      2. Constant-sensor removal: a sensor that has <= 2 distinct values inside
         every operating condition carries no information.
      3. Per-condition standardisation (z-score) of every remaining sensor, so
         that sensor changes caused by *altitude/speed settings* are not mistaken
         for *degradation*.
      4. Correlation based feature selection: sensors whose |Spearman| with the
         (capped) RUL is below `min_abs_corr` are dropped.
    """

    def __init__(self, n_conditions: int = 1, min_abs_corr: float = cfg.MIN_ABS_SPEARMAN,
                 random_state: int = cfg.RANDOM_STATE):
        self.n_conditions = n_conditions
        self.min_abs_corr = min_abs_corr
        self.random_state = random_state

    # -- helpers
    def _condition(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_conditions <= 1:
            return np.zeros(len(df), dtype=int)
        z = self.setting_scaler_.transform(df[cfg.SETTING_COLS].to_numpy())
        return self.kmeans_.predict(z)

    # -- learn
    def fit(self, df: pd.DataFrame, rul_capped) -> "Preprocessor":
        if self.n_conditions > 1:
            self.setting_scaler_ = StandardScaler().fit(df[cfg.SETTING_COLS].to_numpy())
            self.kmeans_ = KMeans(self.n_conditions, n_init=10, random_state=self.random_state).fit(
                self.setting_scaler_.transform(df[cfg.SETTING_COLS].to_numpy()))
        cond = pd.Series(self._condition(df), index=df.index, name="cond")
        grouped = df[cfg.SENSOR_COLS].groupby(cond)
        nunique = grouped.nunique().max()                      # worst case over conditions
        self.constant_sensors_ = [s for s in cfg.SENSOR_COLS if nunique[s] <= 2]
        candidates = [s for s in cfg.SENSOR_COLS if s not in self.constant_sensors_]
        self.cond_mean_ = grouped.mean()[candidates]
        self.cond_std_ = grouped.std()[candidates].replace(0, 1.0)

        z = self._normalise(df, cond, candidates)
        rho = z.corrwith(pd.Series(np.asarray(rul_capped), index=df.index), method="spearman")
        self.low_corr_sensors_ = [s for s in candidates if abs(rho[s]) < self.min_abs_corr]
        self.sensors_ = [s for s in candidates if s not in self.low_corr_sensors_]
        self.selection_report_ = pd.DataFrame({
            "sensor": cfg.SENSOR_COLS,
            "description": [cfg.SENSOR_DESCRIPTIONS[s] for s in cfg.SENSOR_COLS],
            "spearman_with_rul": [rho.get(s, np.nan) for s in cfg.SENSOR_COLS],
            "status": ["dropped: constant" if s in self.constant_sensors_ else
                       "dropped: weak correlation" if s in self.low_corr_sensors_ else "kept"
                       for s in cfg.SENSOR_COLS],
        })
        return self

    def _normalise(self, df, cond, sensors) -> pd.DataFrame:
        mean = self.cond_mean_.loc[cond.to_numpy(), sensors].to_numpy()
        std = self.cond_std_.loc[cond.to_numpy(), sensors].to_numpy()
        return pd.DataFrame((df[sensors].to_numpy() - mean) / std, columns=sensors, index=df.index)

    # -- apply
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return unit, cycle, condition id and the normalised *selected* sensors."""
        cond = pd.Series(self._condition(df), index=df.index, name="cond")
        z = self._normalise(df, cond, self.sensors_)
        out = pd.concat([df[[cfg.ID_COL, cfg.CYCLE_COL]], cond, z], axis=1)
        return out


def pca_analysis(norm_df: pd.DataFrame, sensors: list[str]):
    """Unit I - dimensionality reduction. Returns the fitted PCA and the 2-D projection."""
    from sklearn.decomposition import PCA
    pca = PCA(random_state=cfg.RANDOM_STATE).fit(norm_df[sensors])
    return pca, pca.transform(norm_df[sensors])[:, :2]
