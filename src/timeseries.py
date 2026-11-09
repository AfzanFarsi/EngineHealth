"""Unit V - classical time-series analysis of C-MAPSS sensor data.

1. Time-series characteristics: stationarity (ADF test), ACF/PACF of one sensor of one engine.
2. Univariate forecasting of that sensor with Naive, Moving Average, AR, MA, ARIMA and SARIMA
   using a *chronological* split (first 80 % of the engine's life -> fit, last 20 % -> forecast).
3. Classical RUL baseline: a multivariate "health indicator" (average of the normalised
   degrading sensors) is extrapolated with ARIMA until it reaches the failure level seen in
   the training engines.  Compared with the supervised ML model on the official test set.

    python -m src.timeseries --all
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, pacf

import config as cfg
from src import data_loader as dl
from src.evaluate import nasa_score, regression_metrics
from src.preprocessing import Preprocessor

warnings.filterwarnings("ignore")


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def univariate_analysis(subset: str, metrics_dir=None) -> dict:
    metrics_dir = metrics_dir or cfg.METRICS_DIR
    train = dl.add_train_rul(dl.load_train(subset))
    pre = Preprocessor(cfg.SUBSETS[subset]["conditions"]).fit(train, dl.cap_rul(train["rul"]))
    norm = pre.transform(train)
    # sensor most correlated with RUL, engine with median life
    sensor = pre.selection_report_.query("status == 'kept'").assign(
        a=lambda d: d.spearman_with_rul.abs()).sort_values("a", ascending=False).iloc[0]["sensor"]
    life = train.groupby(cfg.ID_COL)[cfg.CYCLE_COL].max()
    unit = int((life - life.median()).abs().sort_values().index[0])
    series_raw = train.loc[train[cfg.ID_COL] == unit, sensor].to_numpy()
    series_norm = norm.loc[norm[cfg.ID_COL] == unit, sensor].to_numpy()

    out = {"subset": subset, "engine": unit, "sensor": sensor,
           "sensor_description": cfg.SENSOR_DESCRIPTIONS[sensor], "n_cycles": int(len(series_raw)),
           "interval": "regular: exactly one observation per operating cycle (no gaps)",
           "variables": f"univariate view = 1 sensor; multivariate view = {len(pre.sensors_)} sensors + 3 settings"}
    for label, s in (("raw", series_raw), ("condition_normalised", series_norm)):
        p0 = adfuller(s, autolag="AIC")[1]
        p1 = adfuller(np.diff(s), autolag="AIC")[1]
        out[f"adf_pvalue_{label}"] = float(p0)
        out[f"adf_pvalue_{label}_first_difference"] = float(p1)
        out[f"stationary_{label}"] = bool(p0 < 0.05)

    s = series_norm
    n_fit = int(0.8 * len(s))
    fit, hold = s[:n_fit], s[n_fit:]
    h = len(hold)
    fc, aic = {}, {}
    fc["Naive (last value)"] = np.repeat(fit[-1], h)
    fc["Moving average (10)"] = np.repeat(fit[-10:].mean(), h)
    ar = AutoReg(fit, lags=2, trend="ct").fit()
    fc["AR(2) + trend"] = ar.predict(start=n_fit, end=len(s) - 1)
    aic["AR(2) + trend"] = float(ar.aic)
    ma = ARIMA(fit, order=(0, 0, 2), trend="ct").fit()
    fc["MA(2) + trend"] = ma.forecast(h); aic["MA(2) + trend"] = float(ma.aic)
    arima = ARIMA(fit, order=(1, 1, 1), trend="t").fit()
    fc["ARIMA(1,1,1) + drift"] = arima.forecast(h); aic["ARIMA(1,1,1) + drift"] = float(arima.aic)

    d = np.diff(fit)
    ac = acf(d, nlags=30)
    season = int(np.argmax(np.abs(ac[2:])) + 2)
    band = 1.96 / np.sqrt(len(d))
    sarima = SARIMAX(fit, order=(1, 1, 1), seasonal_order=(1, 0, 1, season), trend="t").fit(disp=False)
    name = f"SARIMA(1,1,1)(1,0,1,{season})"
    fc[name] = sarima.forecast(h); aic[name] = float(sarima.aic)
    out["seasonality_check"] = {"strongest_lag_in_differenced_series": season,
                                "abs_acf_at_that_lag": float(abs(ac[season])), "95pct_band": float(band),
                                "seasonal_pattern_present": bool(abs(ac[season]) > 3 * band)}
    out["forecast_rmse_normalised_units"] = {k: _rmse(hold, v) for k, v in fc.items()}
    out["aic"] = aic
    out["holdout_cycles"] = int(h)

    # forecast csv for plotting
    df = pd.DataFrame({"cycle": np.arange(1, len(s) + 1), "actual": s,
                       "split": ["fit"] * n_fit + ["holdout"] * h})
    for k, v in fc.items():
        df[k] = np.r_[np.full(n_fit, np.nan), np.asarray(v)]
    df.to_csv(metrics_dir / f"{subset}_ts_forecast.csv", index=False)
    pd.DataFrame({"lag": np.arange(0, 31), "acf": acf(s, nlags=30), "pacf": pacf(s, nlags=30)}
                 ).to_csv(metrics_dir / f"{subset}_ts_acf.csv", index=False)
    return out


def health_indicator(norm: pd.DataFrame, sensors, signs: pd.Series) -> pd.Series:
    """Average of sign-corrected normalised sensors -> one degradation curve per engine."""
    return (norm[sensors] * signs[sensors]).mean(axis=1)


def rul_baseline(subset: str, metrics_dir=None, pred_dir=None) -> dict:
    """ARIMA extrapolation of the health indicator vs the supervised model (official test RMSE)."""
    metrics_dir, pred_dir = metrics_dir or cfg.METRICS_DIR, pred_dir or cfg.PREDICTIONS_DIR
    train = dl.add_train_rul(dl.load_train(subset))
    test = dl.load_test(subset)
    truth = dl.load_rul_truth(subset)
    pre = Preprocessor(cfg.SUBSETS[subset]["conditions"]).fit(train, dl.cap_rul(train["rul"]))
    sensors = pre.sensors_
    signs = np.sign(pre.selection_report_.set_index("sensor")["spearman_with_rul"]).fillna(0) * -1  # HI rises
    ntr, nte = pre.transform(train), pre.transform(test)
    ntr["hi"], nte["hi"] = health_indicator(ntr, sensors, signs), health_indicator(nte, sensors, signs)

    def smooth(x, w=10):
        return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()

    fail_level = float(np.median([smooth(g["hi"].to_numpy())[-1] for _, g in ntr.groupby(cfg.ID_COL)]))
    rows = []
    for unit, g in nte.groupby(cfg.ID_COL):
        y = smooth(g["hi"].to_numpy())
        last = y[-1]
        recent = y[-60:]
        try:
            m = ARIMA(recent, order=(0, 1, 1), trend="t").fit()
            path = m.forecast(cfg.RUL_CAP + 1)
        except Exception:                                     # fall back to a straight line
            slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
            path = last + slope * np.arange(1, cfg.RUL_CAP + 2)
        cross = np.where(np.asarray(path) >= fail_level)[0]
        rul = float(cross[0]) if len(cross) else float(cfg.RUL_CAP)
        rows.append({"unit": int(unit), "rul_true": float(truth.loc[unit]), "rul_pred_arima_hi": min(rul, cfg.RUL_CAP)})
    res = pd.DataFrame(rows)

    ml = pd.read_csv(pred_dir / f"{subset}_test_predictions.csv.gz")
    summ = json.loads((metrics_dir / f"{subset}_summary.json").read_text())
    best = f"pred::{summ['best_model']}"
    ml = ml[ml["is_last_cycle"]].set_index("unit")
    res["rul_pred_ml"] = res["unit"].map(ml[best])
    res.to_csv(metrics_dir / f"{subset}_ts_rul_baseline.csv", index=False)
    a = regression_metrics(res["rul_true"], res["rul_pred_arima_hi"], with_score=True)
    b = regression_metrics(res["rul_true"], res["rul_pred_ml"], with_score=True)
    return {"failure_level_health_indicator": fail_level, "arima_health_indicator": a,
            "supervised_best_model": {"name": summ["best_model"], **b}}


def run(subset: str) -> dict:
    out = univariate_analysis(subset)
    out["rul_baseline_comparison"] = rul_baseline(subset)
    (cfg.METRICS_DIR / f"{subset}_timeseries.json").write_text(json.dumps(out, indent=2))
    r = out["rul_baseline_comparison"]
    print(f"[{subset}] ADF raw p={out['adf_pvalue_raw']:.3f}, normalised p={out['adf_pvalue_condition_normalised']:.3f} | "
          f"HI-ARIMA RMSE={r['arima_health_indicator']['RMSE']:.1f} vs ML RMSE={r['supervised_best_model']['RMSE']:.1f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=list(cfg.SUBSETS), default=cfg.DEFAULT_SUBSET)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for s in (list(cfg.SUBSETS) if a.all else [a.subset]):
        run(s)
