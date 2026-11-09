import pandas as pd
import streamlit as st

import config as cfg
from src import app_data, explainability as xai, ui
from src import visualization as viz

st.title("Explainable AI")
subset = app_data.current_subset()
if subset not in app_data.available_subsets():
    st.warning(app_data.missing_artifacts_message(subset)); st.stop()

bundle = app_data.rul_bundle(subset)
name = bundle["explain_model_name"]
ui.note(f"Explanations are computed for the best tree-based model, <b>{name}</b>. "
        "<b>Global</b> interpretability answers “which sensors matter overall?”; <b>local</b> interpretability answers "
        "“why did the model predict <i>this</i> RUL for <i>this</i> engine now?”. SHAP values are in cycles: positive pushes the predicted RUL up (healthier), negative pushes it down.")

tab_g, tab_l, tab_lime = st.tabs(["Global (SHAP)", "Local explanation", "LIME"])

with tab_g:
    gs = app_data.metrics_csv(subset, "shap_global")
    sample = pd.read_csv(cfg.METRICS_DIR / f"{subset}_shap_sample.csv.gz")
    st.plotly_chart(viz.plotly_beeswarm(sample), use_container_width=True, config={"displayModeBar": False})
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.plotly_bar_h(gs.head(12), "mean_abs_shap", "feature", "Mean |SHAP| per feature"), use_container_width=True, config={"displayModeBar": False})
    sens = gs.groupby("sensor")["mean_abs_shap"].sum().sort_values(ascending=False).head(10).reset_index()
    sens["label"] = sens["sensor"].map(lambda s: f"{s} · {cfg.SENSOR_DESCRIPTIONS.get(s, 'cycle count')[:28]}")
    c2.plotly_chart(viz.plotly_bar_h(sens, "mean_abs_shap", "label", "Per physical sensor (sum over its features)", viz.PALETTE["blue"]),
                    use_container_width=True, config={"displayModeBar": False})
    st.caption("Computed on 1,200 random test rows. Each dot = one row; colour = the feature's value (blue low, red high).")


def pick_instance(prefix: str):
    test = app_data.raw_test(subset)
    c1, c2 = st.columns(2)
    unit = c1.selectbox("Engine", sorted(test["unit"].unique()), key=f"{prefix}_unit")
    g = test[test["unit"] == unit]
    mx = int(g["cycle"].max())
    cycle = c2.slider("Cycle", 1, mx, mx, key=f"{prefix}_cycle_{subset}_{unit}")
    X = app_data.test_features(subset)
    idx = g.index[g["cycle"] == cycle][0]
    return unit, cycle, X.loc[[idx]], float(g.loc[idx, "rul"])


with tab_l:
    unit, cycle, x, truth = pick_instance("loc")
    if st.button("Explain this prediction", type="primary", key="btn_local"):
        st.session_state["local"] = (subset, unit, cycle)
    if st.session_state.get("local") == (subset, unit, cycle):
        ex = app_data.shap_explainer(subset)
        loc = xai.local_explanation(ex, x)
        base = loc.attrs["base_value"]
        model_pred = float(bundle["explain_model"].predict(x)[0])
        ui.kpi_row([("Average model output", f"{base:.1f}", "SHAP base value", "#3B82F6"),
                    (f"Prediction for engine {unit}", f"{model_pred:.1f}", f"cycle {cycle}", "#22D3EE"),
                    ("Ground truth (capped)", f"{min(truth, cfg.RUL_CAP):.0f}", f"true RUL {truth:.0f}", "#34D399")])
        c1, c2 = st.columns(2)
        top = loc.head(12).copy()
        c1.plotly_chart(viz.plotly_bar_h(top, "shap", "feature", "Top feature contributions (cycles)", signed=True), use_container_width=True, config={"displayModeBar": False})
        sc = xai.sensor_contributions(loc).head(10)
        sc["label"] = sc["sensor"] + " · " + sc["description"].str.slice(0, 26)
        c2.plotly_chart(viz.plotly_bar_h(sc, "shap", "label", "Per-sensor contribution (cycles)", signed=True), use_container_width=True, config={"displayModeBar": False})
        st.caption(f"base value {base:.1f} + sum of all contributions {loc['shap'].sum():+.1f} = model output {model_pred:.1f}")
        st.dataframe(top[["feature", "value", "shap"]].rename(columns={"value": "Feature value", "shap": "SHAP (cycles)"}).round(3),
                     hide_index=True, use_container_width=True)
    else:
        st.info("Select an engine and cycle, then press **Explain this prediction**.")

with tab_lime:
    st.markdown("LIME fits a small linear model around one prediction by perturbing the inputs - an independent, model-agnostic cross-check of SHAP.")
    unit, cycle, x, truth = pick_instance("lime")
    if st.button("Run LIME", type="primary", key="btn_lime"):
        with st.spinner("Fitting local surrogate model..."):
            bg = app_data.train_features(subset).sample(1500, random_state=0)
            res = xai.lime_explanation(bundle["explain_model"], bg, x)
        st.session_state["lime"] = (subset, unit, cycle, res)
    r = st.session_state.get("lime")
    if r and r[:3] == (subset, unit, cycle):
        st.plotly_chart(viz.plotly_bar_h(r[3], "weight", "feature", "LIME local weights (effect on predicted RUL)", signed=True),
                        use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("Press **Run LIME** for the selected engine and cycle.")
