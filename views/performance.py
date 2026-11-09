import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve

import config as cfg
from src import app_data, ui
from src import visualization as viz
from src.evaluate import regression_metrics

st.title("Model Performance")
subset = app_data.current_subset()
if subset not in app_data.available_subsets():
    st.warning(app_data.missing_artifacts_message(subset)); st.stop()

summ = app_data.summary(subset)
best = summ["best_model"]
val_t = app_data.metrics_csv(subset, "regression_validation")
test_t = app_data.metrics_csv(subset, "regression_test")
te = app_data.predictions(subset, "test")
va = app_data.predictions(subset, "val")
tr = app_data.predictions(subset, "train")

ui.kpi_row([
    ("Best model (by validation RMSE)", best, "selected without looking at the test set", "#22D3EE"),
    ("Test RMSE", f"{summ['test_best']['official_RMSE']:.2f}", "official last-cycle RUL", "#3B82F6"),
    ("Test MAE", f"{summ['test_best']['official_MAE']:.2f}", f"R² {summ['test_best']['official_R2']:.3f}", "#A78BFA"),
    ("NASA score", f"{summ['test_best']['official_NASA_score']:.0f}", "lower is better; late predictions cost more", "#FBBF24"),
])
ui.note("Regression is judged with MAE / RMSE / R² / NASA score (not accuracy). The 85-95 % target applies naturally to the "
        "<b>classification</b> tab; for RUL, R² is the closest percentage-style measure. Validation = held-out training engines; "
        "test = official NASA test engines, evaluated once.")

tabs = st.tabs(["Model comparison", "RUL prediction curves", "Actual vs predicted", "Residuals", "Tuning",
                "Fault-risk classification", "All subsets"])

with tabs[0]:
    c1, c2 = st.columns(2)
    v = val_t[val_t["family"] != "baseline"]
    c1.plotly_chart(viz.plotly_model_bars(v, "RMSE", "Validation RMSE (held-out engines)", best), use_container_width=True, config={"displayModeBar": False})
    t = test_t[test_t["family"] != "baseline"].rename(columns={"official_RMSE": "RMSE_test"})
    c2.plotly_chart(viz.plotly_model_bars(t, "RMSE_test", "Test RMSE (official)", best), use_container_width=True, config={"displayModeBar": False})
    st.markdown("**Validation table** (target: RUL capped at 125)")
    st.dataframe(val_t.round(3), hide_index=True, use_container_width=True)
    st.markdown("**Test table** (official = last cycle of each engine vs RUL_FDxxx.txt; all-cycles = every test row vs derived, capped RUL)")
    st.dataframe(test_t.round(3), hide_index=True, use_container_width=True)
    fam = val_t[val_t["family"] != "baseline"].groupby("family")["RMSE"].min().sort_values().reset_index()
    st.markdown("**Best validation RMSE by model family** (ensembles vs single models)")
    st.dataframe(fam.rename(columns={"family": "Family", "RMSE": "Best RMSE"}).round(2), hide_index=True, use_container_width=True)

with tabs[1]:
    split = st.radio("Data split", ["Test (official engines)", "Validation (held-out training engines)"], horizontal=True, key="curve_split")
    d = te if split.startswith("Test") else va
    truth_col = "rul_true"
    units = sorted(d["unit"].unique())
    last = te[te["is_last_cycle"]].set_index("unit")["rul_true"]
    default_units = [u for u in viz.pick_engines(last, 4) if u in units] if split.startswith("Test") else units[:2]
    c1, c2 = st.columns([1, 3])
    eng = c1.selectbox("Engine", units, index=units.index(default_units[0]) if default_units else 0, key="curve_engine")
    models = [c.replace("pred::", "") for c in d.columns if c.startswith("pred::") and "Baseline" not in c]
    top = [m for m in test_t["model"] if m in models and "Baseline" not in m][:3]
    pick = c1.multiselect("Models", models, default=[m for m in [best] + top if m in models][:3], key="curve_models")
    g = d[d["unit"] == eng].sort_values("cycle")
    fig = viz.plotly_rul_curve(g, {m: f"pred::{m}" for m in pick}, f"{split.split(' (')[0]} engine {eng}: predicted RUL vs ground truth")
    c2.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Ground truth: validation = true run-to-failure RUL; test = RUL derived from the official RUL_FDxxx.txt file. "
               f"Both are capped at {cfg.RUL_CAP} because models are trained on the piece-wise linear target.")

with tabs[2]:
    split = st.radio("Split", ["Training (in-sample)", "Validation", "Test (last cycle, official)", "Test (all cycles)"], horizontal=True, key="avp_split")
    pc = f"pred::{best}"
    if split.startswith("Training"):
        yt, yp = np.minimum(tr["rul_true"], cfg.RUL_CAP), tr["pred"]
    elif split == "Validation":
        yt, yp = va["rul_true_capped"], va[pc]
    elif "last" in split:
        l = te[te["is_last_cycle"]]; yt, yp = l["rul_true"], l[pc]
    else:
        yt, yp = np.minimum(te["rul_true"], cfg.RUL_CAP), te[pc]
    if len(yt) > 6000:
        i = np.random.RandomState(0).choice(len(yt), 6000, replace=False); yt, yp = np.asarray(yt)[i], np.asarray(yp)[i]
    m = regression_metrics(yt, yp)
    st.plotly_chart(viz.plotly_scatter_actual_pred(yt, yp, f"{split}: RMSE {m['RMSE']:.2f} · MAE {m['MAE']:.2f} · R² {m['R2']:.3f}"),
                    use_container_width=True, config={"displayModeBar": False})
    if split.startswith("Training"):
        st.caption("In-sample fit: it is expected to look better than validation/test and must not be used to judge the model.")

with tabs[3]:
    l = te[te["is_last_cycle"]]
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.plotly_residuals(va["rul_true_capped"], va[f"pred::{best}"], "Validation residuals"), use_container_width=True, config={"displayModeBar": False})
    c2.plotly_chart(viz.plotly_residuals(l["rul_true"], l[f"pred::{best}"], "Test residuals (last cycle)"), use_container_width=True, config={"displayModeBar": False})
    res = va[f"pred::{best}"] - va["rul_true_capped"]
    bins = pd.cut(va["rul_true_capped"], [-1, 30, 70, 110, 126])
    by = (va.assign(abs_err=res.abs()).groupby(bins, observed=True)["abs_err"].mean().round(2).reset_index())
    by.columns = ["True RUL range (cycles)", "Mean absolute error"]
    by["True RUL range (cycles)"] = by["True RUL range (cycles)"].astype(str)
    st.markdown("**Error by remaining-life range (validation)** - the model is usually most accurate close to failure, where it matters most.")
    st.dataframe(by, hide_index=True, use_container_width=True)

with tabs[4]:
    tune = app_data.metrics_csv(subset, "tuning")
    st.markdown("GridSearchCV and RandomizedSearchCV with **GroupKFold (3 folds, never splitting an engine)**.")
    st.dataframe(tune, hide_index=True, use_container_width=True)

with tabs[5]:
    cv = app_data.metrics_csv(subset, "classification_validation")
    ct = app_data.metrics_csv(subset, "classification_test")
    pos = summ["classification"]["positive_rate_train"]
    st.markdown(f"Target: **imminent fault = RUL ≤ {cfg.FAULT_THRESHOLD} cycles** (positive class is only {pos*100:.1f} % of training rows, "
                "so imbalance is handled with `class_weight='balanced'` and accuracy is reported alongside precision/recall/F1/AUC).")
    cols = ["model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    st.markdown("**Validation (held-out engines)**")
    st.dataframe(cv[cols].round(3), hide_index=True, use_container_width=True)
    which = st.radio("Test evaluation", ["all test cycles", "last cycle (official 1 row/engine)"], horizontal=True, key="clf_eval")
    sel = ct[ct["evaluation"] == which]
    st.markdown(f"**Test - {which}** (n = {int(sel['n'].iloc[0]):,}, positives = {int(sel['positives'].iloc[0]):,})")
    st.dataframe(sel[cols].round(3), hide_index=True, use_container_width=True)
    name = st.selectbox("Confusion matrix for", list(sel["model"]), index=list(sel["model"]).index(summ["classification"]["best_model"]), key="clf_cm")
    r = sel[sel["model"] == name].iloc[0]
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.plotly_confusion(np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]]).astype(int), f"{name} - {which}"),
                    use_container_width=True, config={"displayModeBar": False})
    if "risk_score" in te.columns:
        fpr, tpr, _ = roc_curve(te["risk_true"], te["risk_score"])
        import plotly.graph_objects as go
        f = go.Figure(go.Scatter(x=fpr, y=tpr, line={"color": viz.PALETTE["cyan"], "width": 2.5}, name="ROC"))
        f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line={"color": viz.PALETTE["muted"], "dash": "dash"}, name="Chance"))
        c2.plotly_chart(viz._layout(f, f"ROC curve - {summ['classification']['best_model']} (all test cycles)", 360, "False-positive rate", "True-positive rate"),
                        use_container_width=True, config={"displayModeBar": False})
    ui.note("Accuracy over <i>all</i> cycles is flattered by the many healthy rows. Recall (how many near-failure cycles are caught) and F1 "
            "are the honest indicators for a maintenance alert.")

with tabs[6]:
    rows = []
    for s in app_data.available_subsets():
        j = app_data.summary(s)
        rows.append({"Subset": s, "Conditions": j["conditions"], "Fault modes": j["fault_modes"], "Best model": j["best_model"],
                     "Test RMSE": j["test_best"]["official_RMSE"], "Test MAE": j["test_best"]["official_MAE"],
                     "Test R²": j["test_best"]["official_R2"], "NASA score": j["test_best"]["official_NASA_score"],
                     "Risk F1 (all cycles)": j["classification"]["test_all_cycles"]["F1"],
                     "Risk AUC": j["classification"]["test_all_cycles"]["ROC_AUC"]})
    d = pd.DataFrame(rows)
    st.dataframe(d.round(3), hide_index=True, use_container_width=True)
    if len(d) > 1:
        st.plotly_chart(viz.plotly_model_bars(d.rename(columns={"Subset": "model"}), "Test RMSE", "Official test RMSE by subset"),
                        use_container_width=True, config={"displayModeBar": False})
    ui.note("FD002/FD004 (six operating conditions) and FD003/FD004 (two fault modes) are harder: more variability and overlapping "
            "degradation signatures. Each subset is trained and evaluated separately; subsets are never merged.")
