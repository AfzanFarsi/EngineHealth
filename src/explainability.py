"""Explainable AI: global importance, SHAP (global + local) and LIME.

* Global  = "which features matter for the model on average?"
* Local   = "why did the model predict THIS RUL for THIS engine at THIS cycle?"
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config as cfg
from src import data_loader as dl
from src.predict import load_bundle, make_features


def sensor_of(feature: str) -> str:
    """'s11_mean10' -> 's11'; 'cycle' -> 'cycle'."""
    return feature.split("_")[0]


def make_explainer(model):
    import shap
    return shap.TreeExplainer(model)


def shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(explainer.shap_values(X))


def local_explanation(explainer, x_row: pd.DataFrame) -> pd.DataFrame:
    """SHAP contribution of every feature for one row (positive = pushes RUL up)."""
    sv = shap_values(explainer, x_row)[0]
    out = pd.DataFrame({"feature": x_row.columns, "value": x_row.iloc[0].to_numpy(), "shap": sv})
    out["sensor"] = out["feature"].map(sensor_of)
    base = float(np.ravel(explainer.expected_value)[0])
    out.attrs["base_value"] = base
    return out.sort_values("shap", key=np.abs, ascending=False).reset_index(drop=True)


def sensor_contributions(local: pd.DataFrame) -> pd.DataFrame:
    """Sum feature-level SHAP values per physical sensor."""
    g = local.groupby("sensor")["shap"].sum().reset_index()
    g["description"] = g["sensor"].map(lambda s: cfg.SENSOR_DESCRIPTIONS.get(s, "Operating cycle count"))
    return g.sort_values("shap", key=np.abs, ascending=False).reset_index(drop=True)


def lime_explanation(model, X_background: pd.DataFrame, x_row: pd.DataFrame, num_features: int = 10,
                     num_samples: int = 1500) -> pd.DataFrame:
    """LIME: fit a small linear model around one prediction."""
    from lime.lime_tabular import LimeTabularExplainer
    exp = LimeTabularExplainer(X_background.to_numpy(), feature_names=list(X_background.columns),
                               mode="regression", discretize_continuous=False, random_state=cfg.RANDOM_STATE)
    res = exp.explain_instance(x_row.iloc[0].to_numpy(), model.predict, num_features=num_features,
                               num_samples=num_samples)
    return pd.DataFrame(res.as_list(), columns=["feature", "weight"])


def run(subset: str, n_sample: int = 1200) -> None:
    """Pre-compute global explanations used by the website and the saved figures."""
    from sklearn.inspection import permutation_importance
    b = load_bundle(subset)
    model = b["explain_model"]
    test = dl.add_test_rul(dl.load_test(subset), dl.load_rul_truth(subset))
    X = make_features(b["preprocessor"], test, b["window"])[b["feature_names"]]
    rng = np.random.RandomState(cfg.RANDOM_STATE)
    idx = np.sort(rng.choice(len(X), min(n_sample, len(X)), replace=False))
    Xs = X.iloc[idx]
    ex = make_explainer(model)
    sv = shap_values(ex, Xs)
    glob = pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
    glob["sensor"] = glob["feature"].map(sensor_of)
    glob.sort_values("mean_abs_shap", ascending=False).to_csv(cfg.METRICS_DIR / f"{subset}_shap_global.csv", index=False)
    sample = Xs.copy().add_prefix("val::")
    sample = pd.concat([sample, pd.DataFrame(sv, columns=[f"shap::{c}" for c in Xs.columns], index=Xs.index)], axis=1)
    sample.to_csv(cfg.METRICS_DIR / f"{subset}_shap_sample.csv.gz", index=False)

    y = dl.cap_rul(test["rul"]).to_numpy()[idx]
    pi = permutation_importance(model, Xs, y, n_repeats=5, random_state=cfg.RANDOM_STATE,
                                scoring="neg_root_mean_squared_error", n_jobs=1)
    pd.DataFrame({"feature": Xs.columns, "importance": pi.importances_mean, "std": pi.importances_std}
                 ).sort_values("importance", ascending=False).to_csv(
        cfg.METRICS_DIR / f"{subset}_permutation_importance.csv", index=False)
    print(f"[{subset}] SHAP + permutation importance saved ({b['explain_model_name']}); "
          f"top feature: {glob.sort_values('mean_abs_shap').iloc[-1]['feature']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=list(cfg.SUBSETS), default=cfg.DEFAULT_SUBSET)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for s in (list(cfg.SUBSETS) if a.all else [a.subset]):
        run(s)
