import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from src import app_data, ui
from src import visualization as viz
from src.evaluate import health_category

st.title("Fleet Dashboard")
subset = app_data.current_subset()
if subset not in app_data.available_subsets():
    st.warning(app_data.missing_artifacts_message(subset)); st.stop()

summ = app_data.summary(subset)
pred = app_data.predictions(subset, "test")
best = f"pred::{summ['best_model']}"
last = pred[pred["is_last_cycle"]].copy()
last["category"] = last[best].map(health_category)
counts = last["category"].value_counts().to_dict()
attention = int((last[best] <= cfg.WARNING_THRESHOLD).sum())

st.caption(f"{subset} · latest recorded cycle of every test engine · predictions from the saved **{summ['best_model']}** model")
ui.kpi_row([
    ("Engines analysed", f"{len(last)}", f"{summ['test_rows']:,} test cycles", "#22D3EE"),
    ("Average predicted RUL", f"{last[best].mean():.1f}", f"cycles (cap {cfg.RUL_CAP}) · true avg {np.minimum(last['rul_true'], cfg.RUL_CAP).mean():.1f}", "#3B82F6"),
    ("Need maintenance attention", f"{attention}", f"{counts.get('Critical', 0)} critical · {counts.get('Warning', 0)} warning", "#F87171"),
    ("Model RMSE (official test)", f"{summ['test_best']['official_RMSE']:.1f}", f"MAE {summ['test_best']['official_MAE']:.1f} · R² {summ['test_best']['official_R2']:.2f}", "#A78BFA"),
])

left, right = st.columns([1, 2])
with left:
    ui.section("Engine health overview", f"Healthy > {cfg.WARNING_THRESHOLD} · Warning {cfg.FAULT_THRESHOLD + 1}-{cfg.WARNING_THRESHOLD} · Critical ≤ {cfg.FAULT_THRESHOLD} cycles")
    st.plotly_chart(viz.plotly_health_donut(counts), use_container_width=True, config={"displayModeBar": False})
with right:
    ui.section("Predicted vs actual RUL", "Each dot is one test engine at its last recorded cycle (official ground truth)")
    st.plotly_chart(viz.plotly_scatter_actual_pred(last["rul_true"], last[best], ""), use_container_width=True,
                    config={"displayModeBar": False})

ui.section("Maintenance priority list", "Engines with the lowest predicted RUL first")
view = last.sort_values(best).head(15)[["unit", "cycle", best, "rul_true", "category"]].rename(columns={
    "unit": "Engine", "cycle": "Last cycle", best: "Predicted RUL", "rul_true": "True RUL", "category": "Health"})
view["Predicted RUL"] = view["Predicted RUL"].round(1)
st.dataframe(view, hide_index=True, use_container_width=True)

ui.section("Sensor monitoring", "Condition-normalised sensor readings (0 = typical healthy value) - drifting away signals degradation")
bundle = app_data.rul_bundle(subset)
test = app_data.raw_test(subset)
norm = bundle["preprocessor"].transform(test)
c1, c2 = st.columns([1, 3])
engine = c1.selectbox("Engine", sorted(test["unit"].unique()), index=0, key="dash_engine")
sensors = c1.multiselect("Sensors", bundle["preprocessor"].sensors_, default=bundle["preprocessor"].sensors_[:4],
                         format_func=lambda s: f"{s} · {cfg.SENSOR_DESCRIPTIONS[s].split(' - ')[1]}", key="dash_sensors")
if sensors:
    g = norm[norm["unit"] == engine]
    c2.plotly_chart(viz.plotly_lines(g, "cycle", sensors, f"Engine {engine}: normalised sensor trends", "z-score"),
                    use_container_width=True, config={"displayModeBar": False})
else:
    c2.info("Pick at least one sensor.")

ui.section("Model performance snapshot")
vt = app_data.metrics_csv(subset, "regression_test")
st.dataframe(vt[vt["family"] != "baseline"].head(6)[["model", "official_RMSE", "official_MAE", "official_R2", "official_NASA_score"]]
             .rename(columns={"official_RMSE": "RMSE", "official_MAE": "MAE", "official_R2": "R²", "official_NASA_score": "NASA score"})
             .round(2), hide_index=True, use_container_width=True)
