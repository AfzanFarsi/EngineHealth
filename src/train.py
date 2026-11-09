"""Train, tune, evaluate and save everything for one (or all) C-MAPSS subsets.

Usage (from the project folder):
    python -m src.train --subset FD001
    python -m src.train --all
    python -m src.train --subset FD001 --quick      # small & fast (used by the tests)

Protocol (no data leakage):
  A) VALIDATION: split the *training engines* 80/20 by engine id.  The preprocessor
     is fitted on the 80 % only.  All models are compared on the held-out 20 %.
     Hyper-parameters are tuned with GroupKFold (folds never split an engine).
  B) FINAL: the preprocessor and every model are re-fitted on all training
     engines and evaluated ONCE on the official test set (using RUL_FDxxx.txt).
The test set is never used to choose a model or a hyper-parameter.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, RandomizedSearchCV

import config as cfg
from src import data_loader as dl
from src import models as zoo
from src.evaluate import classification_metrics, regression_metrics
from src.feature_engineering import feature_groups
from src.predict import make_features
from src.preprocessing import Preprocessor, data_quality_report

warnings.filterwarnings("ignore")


def _paths(out_root=None):
    if out_root is None:
        return cfg.MODELS_DIR, cfg.METRICS_DIR, cfg.PREDICTIONS_DIR
    from pathlib import Path
    out_root = Path(out_root)
    return out_root / "models", out_root / "metrics", out_root / "predictions"


def _log(subset, msg):
    print(f"[{subset}] {time.strftime('%H:%M:%S')}  {msg}", flush=True)


def _fit_timed(est, X, y):
    t = time.time()
    est = clone(est).fit(X, y)
    return est, time.time() - t


def run_subset(subset: str, quick: bool = False, out_root=None) -> dict:
    models_dir, metrics_dir, pred_dir = _paths(out_root)
    for d in (models_dir, metrics_dir, pred_dir):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    n_cond = cfg.SUBSETS[subset]["conditions"]

    # ------------------------------------------------------------ data ------
    train = dl.add_train_rul(dl.load_train(subset))
    test_raw = dl.load_test(subset)
    truth = dl.load_rul_truth(subset)
    test = dl.add_test_rul(test_raw, truth)
    _log(subset, f"loaded train {train.shape}, test {test.shape}, truth {len(truth)}")
    (metrics_dir / f"{subset}_data_quality.json").write_text(json.dumps(
        {"train": data_quality_report(train), "test": data_quality_report(test_raw)}, indent=2))

    # ---------------------------------------------- A) validation protocol --
    gss = GroupShuffleSplit(1, test_size=cfg.VALIDATION_FRACTION, random_state=cfg.RANDOM_STATE)
    tr_idx, va_idx = next(gss.split(train, groups=train[cfg.ID_COL]))
    tr_part, va_part = train.iloc[tr_idx], train.iloc[va_idx]
    assert set(tr_part[cfg.ID_COL]).isdisjoint(va_part[cfg.ID_COL])       # leakage guard
    pre_v = Preprocessor(n_cond).fit(tr_part, dl.cap_rul(tr_part["rul"]))
    Xtr, Xva = make_features(pre_v, tr_part), make_features(pre_v, va_part)
    ytr, yva = dl.cap_rul(tr_part["rul"]).to_numpy(), dl.cap_rul(va_part["rul"]).to_numpy()
    groups = tr_part[cfg.ID_COL].to_numpy()
    base_cols = list(pre_v.sensors_)
    pre_v.selection_report_.to_csv(metrics_dir / f"{subset}_feature_selection.csv", index=False)
    _log(subset, f"features {Xtr.shape[1]} ({len(pre_v.sensors_)} sensors kept); "
                 f"train engines {tr_part[cfg.ID_COL].nunique()}, val engines {va_part[cfg.ID_COL].nunique()}")

    registry = zoo.regression_models(base_cols)
    if quick:
        registry = {k: registry[k] for k in ["Baseline (mean RUL)", "Ridge", "Decision Tree", "HistGradientBoosting"]}
    families = {k: v[1] for k, v in registry.items()}
    registry = {k: v[0] for k, v in registry.items()}

    val_rows, val_pred = [], {}
    for name, est in registry.items():
        fitted, secs = _fit_timed(est, Xtr, ytr)
        p = np.clip(fitted.predict(Xva), 0, cfg.RUL_CAP)
        val_pred[name] = p
        val_rows.append({"model": name, "family": families[name], "tuned": False,
                         **regression_metrics(yva, p), "fit_seconds": round(secs, 1)})
        _log(subset, f"val  {name:32s} RMSE={val_rows[-1]['RMSE']:.2f}  ({secs:.0f}s)")

    # ---- hyper-parameter tuning (GroupKFold = folds never split an engine)
    cv = GroupKFold(3)
    tuned_estimators, tuning_rows = {}, []
    searches = [] if quick else \
        [("GridSearchCV", k, v) for k, v in zoo.GRID_SEARCHES.items()] + \
        [("RandomizedSearchCV", k, v) for k, v in zoo.RANDOM_SEARCHES.items()]
    if quick:
        searches = [("GridSearchCV", "Decision Tree", zoo.GRID_SEARCHES["Decision Tree"])]
    for method, name, spec in searches:
        grid, base_name = spec[0], spec[1]
        Xs, ys, gs = (Xtr.iloc[::3], ytr[::3], groups[::3]) if base_name == "Random Forest" else (Xtr, ytr, groups)
        if method == "GridSearchCV":
            search = GridSearchCV(clone(registry[base_name]), grid, cv=cv, n_jobs=1,
                                  scoring="neg_root_mean_squared_error")
        else:
            search = RandomizedSearchCV(clone(registry[base_name]), grid, n_iter=spec[2], cv=cv, n_jobs=1,
                                        scoring="neg_root_mean_squared_error", random_state=cfg.RANDOM_STATE)
        t = time.time()
        search.fit(Xs, ys, groups=gs)
        tuned_name = f"{name} (tuned)"
        fitted, secs = _fit_timed(search.best_estimator_, Xtr, ytr)
        p = np.clip(fitted.predict(Xva), 0, cfg.RUL_CAP)
        val_pred[tuned_name] = p
        tuned_estimators[tuned_name] = (search.best_estimator_, name)
        families[tuned_name] = families[base_name]
        val_rows.append({"model": tuned_name, "family": families[base_name], "tuned": True,
                         **regression_metrics(yva, p), "fit_seconds": round(secs, 1)})
        default_rmse = next(r["RMSE"] for r in val_rows if r["model"] == base_name)
        tuning_rows.append({"model": base_name, "method": method,
                            "best_params": json.dumps(search.best_params_, default=str),
                            "cv_rmse_train_engines": round(-search.best_score_, 3),
                            "val_rmse_default": round(default_rmse, 3),
                            "val_rmse_tuned": round(val_rows[-1]["RMSE"], 3),
                            "search_seconds": round(time.time() - t, 1)})
        _log(subset, f"tune {method} {name}: val RMSE {default_rmse:.2f} -> {val_rows[-1]['RMSE']:.2f}")

    val_table = pd.DataFrame(val_rows).sort_values("RMSE").reset_index(drop=True)
    val_table.to_csv(metrics_dir / f"{subset}_regression_validation.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(metrics_dir / f"{subset}_tuning.csv", index=False)
    contenders = val_table[val_table["family"] != "baseline"]
    best_name = contenders.iloc[0]["model"]
    tree_name = next(n for n in contenders["model"] if zoo.is_tree_model(
        (tuned_estimators[n][0] if n in tuned_estimators else registry[n])))
    _log(subset, f"best validation model: {best_name} | best tree model (for SHAP): {tree_name}")

    # empirical prediction interval from the held-out validation residuals of the best model
    res = yva - val_pred[best_name]
    lo, hi = (1 - cfg.INTERVAL_COVERAGE) / 2, 1 - (1 - cfg.INTERVAL_COVERAGE) / 2
    residual_q = {"low": float(np.quantile(res, lo)), "high": float(np.quantile(res, hi)),
                  "coverage": cfg.INTERVAL_COVERAGE}

    # validation predictions file
    vp = va_part[[cfg.ID_COL, cfg.CYCLE_COL, "rul"]].rename(columns={"rul": "rul_true"}).copy()
    vp["rul_true_capped"] = yva
    for n, p in val_pred.items():
        vp[f"pred::{n}"] = p
    vp.to_csv(pred_dir / f"{subset}_val_predictions.csv.gz", index=False)

    # --------------------------------------------------- B) final refit ----
    pre = Preprocessor(n_cond).fit(train, dl.cap_rul(train["rul"]))
    Xall, Xte = make_features(pre, train), make_features(pre, test)
    yall = dl.cap_rul(train["rul"]).to_numpy()
    feature_names = list(Xall.columns)
    replaced = {b for _, b in tuned_estimators.values() if b in ("Ridge", "Decision Tree", "SVR (RBF)")}
    final_registry = {n: e for n, e in registry.items() if n not in replaced}
    final_registry.update({n: e for n, (e, _) in tuned_estimators.items()})

    is_last = (test[cfg.CYCLE_COL] == test.groupby(cfg.ID_COL)[cfg.CYCLE_COL].transform("max")).to_numpy()
    yte = test["rul"].to_numpy()
    yte_cap = dl.cap_rul(yte)
    tp = test[[cfg.ID_COL, cfg.CYCLE_COL, "rul"]].rename(columns={"rul": "rul_true"}).copy()
    tp["is_last_cycle"] = is_last
    final_models, test_rows = {}, []
    for name, est in final_registry.items():
        fitted, secs = _fit_timed(est, Xall, yall)
        final_models[name] = fitted
        p = np.clip(fitted.predict(Xte), 0, cfg.RUL_CAP)
        tp[f"pred::{name}"] = p
        off = regression_metrics(yte[is_last], p[is_last], with_score=True)
        test_rows.append({
            "model": name, "family": families[name], "tuned": name in tuned_estimators,
            "official_MAE": off["MAE"], "official_RMSE": off["RMSE"], "official_R2": off["R2"],
            "official_NASA_score": off["NASA_score"],
            "last_cycle_RMSE_vs_capped_RUL": regression_metrics(yte_cap[is_last], p[is_last])["RMSE"],
            **{f"all_cycles_{k}": v for k, v in regression_metrics(yte_cap, p).items()},
        })
        _log(subset, f"test {name:32s} official RMSE={off['RMSE']:.2f} NASA={off['NASA_score']:.0f} ({secs:.0f}s)")
    test_table = pd.DataFrame(test_rows).sort_values("official_RMSE").reset_index(drop=True)
    test_table.to_csv(metrics_dir / f"{subset}_regression_test.csv", index=False)

    best_model = final_models[best_name]
    train_pred = train[[cfg.ID_COL, cfg.CYCLE_COL, "rul"]].rename(columns={"rul": "rul_true"}).copy()
    train_pred["pred"] = np.clip(best_model.predict(Xall), 0, cfg.RUL_CAP)
    train_pred.to_csv(pred_dir / f"{subset}_train_predictions.csv.gz", index=False)

    rul_bundle = {
        "subset": subset, "preprocessor": pre, "model": best_model, "model_name": best_name,
        "feature_names": feature_names, "window": cfg.WINDOW, "rul_cap": cfg.RUL_CAP,
        "residual_q": residual_q,
        "explain_model": final_models[tree_name], "explain_model_name": tree_name,
        "trained_on_engines": int(train[cfg.ID_COL].nunique()),
    }
    joblib.dump(rul_bundle, models_dir / f"{subset}_rul_model.joblib", compress=3)
    _log(subset, "saved RUL model")

    # ------------------------------------------ C) fault-risk classifier ----
    clf_summary = run_classification(subset, quick, models_dir, metrics_dir, pred_dir, tr_part, va_part,
                                     Xtr, Xva, Xall, Xte, train, test, is_last, pre, feature_names, tp)
    tp.to_csv(pred_dir / f"{subset}_test_predictions.csv.gz", index=False)

    # ------------------------------------------------------------ summary ---
    best_test = test_table[test_table["model"] == best_name].iloc[0].to_dict()
    summary = {
        "subset": subset, "conditions": n_cond, "fault_modes": cfg.SUBSETS[subset]["fault_modes"],
        "train_engines": int(train[cfg.ID_COL].nunique()), "test_engines": int(test[cfg.ID_COL].nunique()),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "n_features": len(feature_names), "sensors_kept": pre.sensors_,
        "sensors_dropped_constant": pre.constant_sensors_, "sensors_dropped_weak": pre.low_corr_sensors_,
        "best_model": best_name, "best_tree_model_for_shap": tree_name,
        "validation_best": val_table[val_table["model"] == best_name].iloc[0].to_dict(),
        "test_best": best_test, "residual_interval": residual_q,
        "classification": clf_summary,
        "rul_cap": cfg.RUL_CAP, "fault_threshold": cfg.FAULT_THRESHOLD,
        "total_minutes": round((time.time() - t_start) / 60, 1),
    }
    (metrics_dir / f"{subset}_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _log(subset, f"DONE in {summary['total_minutes']} min | best={best_name} | official test RMSE={best_test['official_RMSE']:.2f}")
    return summary


def run_classification(subset, quick, models_dir, metrics_dir, pred_dir, tr_part, va_part,
                       Xtr, Xva, Xall, Xte, train, test, is_last, pre, feature_names, tp) -> dict:
    """Imminent-fault classifier: positive class = RUL <= FAULT_THRESHOLD cycles."""
    thr = cfg.FAULT_THRESHOLD
    ytr, yva = (tr_part["rul"] <= thr).astype(int).to_numpy(), (va_part["rul"] <= thr).astype(int).to_numpy()
    yall, yte = (train["rul"] <= thr).astype(int).to_numpy(), (test["rul"] <= thr).astype(int).to_numpy()
    registry = zoo.classification_models()
    if quick:
        registry = {k: registry[k] for k in ["Logistic Regression", "Decision Tree"]}

    def score_of(m, X):
        return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)

    val_rows = []
    for name, est in registry.items():
        fitted, secs = _fit_timed(est, Xtr, ytr)
        val_rows.append({"model": name, **classification_metrics(yva, fitted.predict(Xva), score_of(fitted, Xva)),
                         "fit_seconds": round(secs, 1)})
        _log(subset, f"clf val {name:22s} F1={val_rows[-1]['F1']:.3f} AUC={val_rows[-1]['ROC_AUC']:.3f}")
    val_table = pd.DataFrame(val_rows).sort_values("F1", ascending=False).reset_index(drop=True)
    val_table.to_csv(metrics_dir / f"{subset}_classification_validation.csv", index=False)
    best = val_table.iloc[0]["model"]

    test_rows, finals = [], {}
    for name, est in registry.items():
        fitted, _ = _fit_timed(est, Xall, yall)
        finals[name] = fitted
        pred, sc = fitted.predict(Xte), score_of(fitted, Xte)
        test_rows.append({"model": name, "evaluation": "all test cycles", "n": len(yte), "positives": int(yte.sum()),
                          **classification_metrics(yte, pred, sc)})
        test_rows.append({"model": name, "evaluation": "last cycle (official 1 row/engine)", "n": int(is_last.sum()),
                          "positives": int(yte[is_last].sum()),
                          **classification_metrics(yte[is_last], pred[is_last], sc[is_last])})
    test_table = pd.DataFrame(test_rows)
    test_table.to_csv(metrics_dir / f"{subset}_classification_test.csv", index=False)

    tp["risk_score"] = score_of(finals[best], Xte) if hasattr(finals[best], "predict_proba") else \
        1 / (1 + np.exp(-finals[best].decision_function(Xte)))
    tp["risk_true"] = yte
    tp["risk_flag"] = finals[best].predict(Xte)
    joblib.dump({"subset": subset, "preprocessor": pre, "model": finals[best], "model_name": best,
                 "feature_names": feature_names, "window": cfg.WINDOW, "threshold": thr},
                models_dir / f"{subset}_risk_model.joblib", compress=3)
    sel = test_table[(test_table["model"] == best) & (test_table["evaluation"] == "all test cycles")].iloc[0]
    return {"best_model": best, "threshold_cycles": thr,
            "positive_rate_train": float(yall.mean()),
            "validation": val_table[val_table["model"] == best].iloc[0].to_dict(),
            "test_all_cycles": sel.to_dict()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train EngineHealth models")
    ap.add_argument("--subset", choices=list(cfg.SUBSETS), default=cfg.DEFAULT_SUBSET)
    ap.add_argument("--all", action="store_true", help="train FD001..FD004")
    ap.add_argument("--quick", action="store_true", help="few models, no tuning (smoke test)")
    ap.add_argument("--out-root", default=None, help="write outputs somewhere else (tests)")
    a = ap.parse_args()
    for s in (list(cfg.SUBSETS) if a.all else [a.subset]):
        run_subset(s, a.quick, a.out_root)
