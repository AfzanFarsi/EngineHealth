import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as cfg
from src import app_data, ui
from src import visualization as viz
from src.preprocessing import pca_analysis

st.title("Sensor Analytics")
subset = app_data.current_subset()
if subset not in app_data.available_subsets():
    st.warning(app_data.missing_artifacts_message(subset)); st.stop()

train = app_data.raw_train(subset)
bundle = app_data.rul_bundle(subset)
pre = bundle["preprocessor"]
summ = app_data.summary(subset)
sensors = pre.sensors_
fmt = lambda s: f"{s} · {cfg.SENSOR_DESCRIPTIONS[s]}"

tabs = st.tabs(["Sensor trends", "Correlation", "Distributions", "Degradation curves", "Feature importance",
                "Operating conditions", "PCA", "Time-series (Unit V)", "Data quality"])

with tabs[0]:
    c1, c2 = st.columns([1, 3])
    sensor = c1.selectbox("Sensor", sensors, format_func=fmt, key="an_sensor")
    mode = c1.radio("Values", ["Raw", "Condition-normalised"], key="an_mode")
    units = sorted(train["unit"].unique())
    pick = c1.multiselect("Engines", units, default=units[:4], key="an_units")
    data = train if mode == "Raw" else pre.transform(train)
    fig = go.Figure()
    for i, u in enumerate(pick):
        g = data[data["unit"] == u]
        fig.add_trace(go.Scatter(x=g["cycle"], y=g[sensor], name=f"Engine {u}", mode="lines",
                                 line={"color": viz.SERIES[i % 6], "width": 1.6}))
    c2.plotly_chart(viz._layout(fig, f"{fmt(sensor)} ({mode.lower()})", 420, "Operating cycle", sensor),
                    use_container_width=True, config={"displayModeBar": False})
    if mode == "Raw" and cfg.SUBSETS[subset]["conditions"] > 1:
        c2.caption("With six operating conditions the raw signal jumps between condition levels, hiding the slow "
                   "degradation. Switch to *Condition-normalised* to see the trend.")

with tabs[1]:
    method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True, key="an_corr")
    cols = sensors + ["rul"]
    corr = train[cols].corr(method=method)
    st.plotly_chart(viz.plotly_heatmap(corr, f"{subset} - {method.title()} correlation (raw sensors vs RUL)"),
                    use_container_width=True, config={"displayModeBar": False})
    top = corr["rul"].drop("rul").sort_values(key=abs, ascending=False).head(8).reset_index()
    top.columns = ["Sensor", f"{method} r with RUL"]
    top["Meaning"] = top["Sensor"].map(cfg.SENSOR_DESCRIPTIONS)
    st.dataframe(top.round(3), hide_index=True, use_container_width=True)
    if cfg.SUBSETS[subset]["conditions"] > 1:
        st.caption("Correlations of raw values are weaker in multi-condition subsets because operating conditions add large variation.")

with tabs[2]:
    sensor = st.selectbox("Sensor", sensors, format_func=fmt, key="an_dist_sensor")
    d = pre.transform(train).assign(rul=train["rul"].to_numpy())
    fig = go.Figure()
    for name, mask, col in [("Early life (RUL > 100)", d["rul"] > 100, viz.PALETTE["blue"]),
                            ("Near failure (RUL ≤ 30)", d["rul"] <= cfg.FAULT_THRESHOLD, viz.PALETTE["red"])]:
        fig.add_trace(go.Histogram(x=d.loc[mask, sensor], name=name, opacity=0.65, marker_color=col, nbinsx=50))
    fig.update_layout(barmode="overlay")
    st.plotly_chart(viz._layout(fig, f"{sensor}: distribution (condition-normalised)", 380, "z-score", "Count"),
                    use_container_width=True, config={"displayModeBar": False})
    st.caption("A good degradation sensor shows two clearly shifted distributions. The overlap tells you how hard the problem is.")

with tabs[3]:
    d = pre.transform(train).assign(rul=train["rul"].to_numpy())
    d = d[d["rul"] <= 200]
    pick = st.multiselect("Sensors", sensors, default=sensors[:4], format_func=fmt, key="an_deg")
    fig = go.Figure()
    for i, s in enumerate(pick):
        g = d.groupby("rul")[s].agg(["mean", "std"]).reset_index()
        c = viz.SERIES[i % 6]
        fig.add_trace(go.Scatter(x=g["rul"], y=g["mean"] + g["std"], line={"width": 0}, showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=g["rul"], y=g["mean"] - g["std"], fill="tonexty", line={"width": 0}, showlegend=False,
                                 fillcolor="rgba(120,160,255,0.10)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=g["rul"], y=g["mean"], name=s, line={"color": c, "width": 2}))
    fig.update_xaxes(autorange="reversed")
    st.plotly_chart(viz._layout(fig, "Average degradation curve (all training engines aligned at failure)", 420,
                                "Remaining useful life (cycles)  →  failure at the right", "Normalised sensor (z-score)"),
                    use_container_width=True, config={"displayModeBar": False})

with tabs[4]:
    gs = app_data.metrics_csv(subset, "shap_global")
    pi = app_data.metrics_csv(subset, "permutation_importance")
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.plotly_bar_h(gs.head(12), "mean_abs_shap", "feature", f"SHAP importance ({bundle['explain_model_name']})"),
                    use_container_width=True, config={"displayModeBar": False})
    c2.plotly_chart(viz.plotly_bar_h(pi.head(12), "importance", "feature", "Permutation importance (RMSE increase)", viz.PALETTE["blue"]),
                    use_container_width=True, config={"displayModeBar": False})
    sens = gs.groupby("sensor")["mean_abs_shap"].sum().sort_values(ascending=False).head(10).reset_index()
    sens["meaning"] = sens["sensor"].map(lambda s: cfg.SENSOR_DESCRIPTIONS.get(s, "Operating cycle count"))
    st.markdown("**Importance aggregated per physical sensor**")
    st.dataframe(sens.rename(columns={"sensor": "Sensor", "mean_abs_shap": "Σ mean |SHAP|", "meaning": "Meaning"}).round(3),
                 hide_index=True, use_container_width=True)

with tabs[5]:
    n = cfg.SUBSETS[subset]["conditions"]
    st.markdown(f"**{subset}** has **{n} operating condition(s)** (altitude, Mach number, throttle resolver angle).")
    s = train.sample(min(8000, len(train)), random_state=1)
    cond = pre._condition(s)
    fig = go.Figure()
    for k in sorted(set(cond)):
        m = cond == k
        fig.add_trace(go.Scatter(x=s.loc[m, "op1"], y=s.loc[m, "op3"] if n == 1 else s.loc[m, "op2"], mode="markers",
                                 name=f"Condition {k + 1}", marker={"size": 4, "color": viz.SERIES[k % 6], "opacity": 0.6}))
    st.plotly_chart(viz._layout(fig, "Operating settings (clusters found by K-Means)" if n > 1 else "Operating settings (single condition: tiny noise only)",
                                380, "op1 (altitude)", "op2 (Mach)" if n > 1 else "op3"),
                    use_container_width=True, config={"displayModeBar": False})
    if n > 1:
        sensor = st.selectbox("Sensor to compare", sensors, format_func=fmt, key="an_cond_sensor")
        g = train[train["unit"] == train["unit"].iloc[0]]
        norm = pre.transform(train)
        gn = norm[norm["unit"] == train["unit"].iloc[0]]
        c1, c2 = st.columns(2)
        c1.plotly_chart(viz.plotly_lines(g, "cycle", [sensor], "Raw (condition-dominated)"), use_container_width=True, config={"displayModeBar": False})
        c2.plotly_chart(viz.plotly_lines(gn, "cycle", [sensor], "After per-condition normalisation"), use_container_width=True, config={"displayModeBar": False})
        ui.note("Per-condition normalisation is the key preprocessing step for FD002/FD004; without it the models would mistake "
                "operating-condition changes for degradation.")
    else:
        ui.note("FD001/FD003 have a single operating condition, so global (single-group) standardisation is enough.")

with tabs[6]:
    norm = pre.transform(train)
    pca, xy = pca_analysis(norm, sensors)
    c1, c2 = st.columns(2)
    ev = np.cumsum(pca.explained_variance_ratio_)
    fig = go.Figure(go.Scatter(x=list(range(1, len(ev) + 1)), y=ev, mode="lines+markers", line={"color": viz.PALETTE["cyan"]}))
    fig.add_hline(y=0.95, line={"dash": "dash", "color": viz.PALETTE["amber"]}, annotation_text="95 %")
    c1.plotly_chart(viz._layout(fig, "Cumulative explained variance", 380, "Principal components", "Explained variance"),
                    use_container_width=True, config={"displayModeBar": False})
    idx = np.random.RandomState(0).choice(len(xy), min(6000, len(xy)), replace=False)
    fig2 = go.Figure(go.Scattergl(x=xy[idx, 0], y=xy[idx, 1], mode="markers",
                                  marker={"size": 4, "color": np.minimum(train["rul"].to_numpy()[idx], cfg.RUL_CAP),
                                          "colorscale": "Turbo", "showscale": True, "colorbar": {"title": "RUL"}}))
    c2.plotly_chart(viz._layout(fig2, "PC1 vs PC2 coloured by RUL", 380, "PC1", "PC2"), use_container_width=True, config={"displayModeBar": False})
    k95 = int(np.argmax(ev >= 0.95) + 1)
    st.caption(f"{k95} of {len(sensors)} components explain 95 % of the variance. PCA is shown as an analysis (Unit I); "
               "the final models use the original, interpretable sensor features.")

with tabs[7]:
    try:
        ts = app_data.metrics_json(subset, "timeseries")
        fc = app_data.metrics_csv(subset, "ts_forecast")
        ac = app_data.metrics_csv(subset, "ts_acf")
    except FileNotFoundError:
        st.info("Run `python -m src.timeseries --all` to create the time-series results."); st.stop()
    st.markdown(f"**Engine {ts['engine']} · sensor {ts['sensor']} ({ts['sensor_description']}) · {ts['n_cycles']} cycles**")
    ui.kpi_row([
        ("ADF p-value (raw)", f"{ts['adf_pvalue_raw']:.3f}", "stationary" if ts["stationary_raw"] else "non-stationary (p ≥ 0.05)", "#F87171" if not ts["stationary_raw"] else "#34D399"),
        ("ADF p (normalised)", f"{ts['adf_pvalue_condition_normalised']:.3f}", "stationary" if ts["stationary_condition_normalised"] else "non-stationary", "#3B82F6"),
        ("ADF p (1st difference)", f"{ts['adf_pvalue_condition_normalised_first_difference']:.3g}", "differencing removes the trend", "#22D3EE"),
        ("Seasonality", "none" if not ts["seasonality_check"]["seasonal_pattern_present"] else "present",
         f"strongest lag {ts['seasonality_check']['strongest_lag_in_differenced_series']}, |ACF| {ts['seasonality_check']['abs_acf_at_that_lag']:.2f}", "#A78BFA"),
    ])
    st.markdown(f"Observations are **{ts['interval']}**. {ts['variables']}.")
    models = [c for c in fc.columns if c not in ("cycle", "actual", "split")]
    pickm = st.multiselect("Forecast models", models, default=models, key="ts_models")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["cycle"], y=fc["actual"], name="Actual (normalised sensor)", line={"color": viz.PALETTE["text"], "width": 1.6}))
    cut = int(fc.loc[fc["split"] == "holdout", "cycle"].min())
    fig.add_vrect(x0=cut, x1=fc["cycle"].max(), fillcolor="rgba(251,191,36,0.07)", line_width=0, annotation_text="chronological hold-out")
    for i, m in enumerate(pickm):
        fig.add_trace(go.Scatter(x=fc["cycle"], y=fc[m], name=m, line={"color": viz.SERIES[(i + 1) % 6], "width": 2}))
    st.plotly_chart(viz._layout(fig, "Univariate forecasting on a chronological split (first 80 % fit, last 20 % forecast)", 420,
                                "Operating cycle", "z-score"), use_container_width=True, config={"displayModeBar": False})
    c1, c2 = st.columns(2)
    rm = pd.DataFrame({"Model": list(ts["forecast_rmse_normalised_units"]), "Hold-out RMSE": list(ts["forecast_rmse_normalised_units"].values())})
    rm["AIC"] = rm["Model"].map(ts["aic"])
    c1.dataframe(rm.round(3), hide_index=True, use_container_width=True)
    fa = go.Figure()
    fa.add_trace(go.Bar(x=ac["lag"], y=ac["acf"], name="ACF", marker_color=viz.PALETTE["cyan"]))
    fa.add_trace(go.Bar(x=ac["lag"], y=ac["pacf"], name="PACF", marker_color=viz.PALETTE["violet"]))
    fa.update_layout(barmode="group")
    c2.plotly_chart(viz._layout(fa, "ACF / PACF of the sensor", 300, "Lag (cycles)", "Correlation"), use_container_width=True, config={"displayModeBar": False})
    rb = ts["rul_baseline_comparison"]
    st.markdown("**Classical baseline vs supervised ML (official test RUL)**")
    cmp = pd.DataFrame([{"Approach": "ARIMA(0,1,1)+drift on a health indicator", **rb["arima_health_indicator"]},
                        {"Approach": f"Supervised ML ({rb['supervised_best_model']['name']})",
                         **{k: v for k, v in rb["supervised_best_model"].items() if k != "name"}}])
    st.dataframe(cmp.round(2), hide_index=True, use_container_width=True)
    ui.note("<b>Applicability:</b> ARIMA/SARIMA model <i>one</i> series. C-MAPSS has many engines, each with its own start state and "
            "fault progression, and (FD002/FD004) switching operating conditions. Classical models therefore forecast a single sensor or a "
            "health indicator per engine but cannot learn across the fleet. SARIMA needs a repeating seasonal cycle; degradation here is a "
            "trend with no seasonality, so SARIMA brings no benefit (see the seasonality check and AIC).")

with tabs[8]:
    q = app_data.metrics_json(subset, "data_quality")
    t = q["train"]
    ui.kpi_row([("Training rows", f"{t['rows']:,}", f"{t['engines']} engines", "#22D3EE"),
                ("Missing values", str(t["missing_values"]), "nothing to impute", "#34D399"),
                ("Duplicate rows", str(t["duplicate_rows"]), "", "#34D399"),
                ("Engine life", f"{t['engine_life_min']}-{t['engine_life_max']}", f"mean {t['engine_life_mean']} cycles", "#3B82F6")])
    st.markdown("**Sensor selection (feature selection step)**")
    st.dataframe(app_data.metrics_csv(subset, "feature_selection").round(3), hide_index=True, use_container_width=True)
    st.markdown(f"**Constant columns in raw data:** {', '.join(t['constant_columns']) or 'none'}  ·  "
                f"**Cycles consecutive for every engine:** {t['cycles_are_consecutive']}")
    oc = pd.Series(t["outliers_3xIQR_per_column"]).sort_values(ascending=False).head(10)
    if len(oc):
        st.markdown("**Extreme values (> 3×IQR beyond quartiles)** are kept: in degradation data they are the signal, not errors.")
        st.dataframe(oc.rename("count").reset_index().rename(columns={"index": "Column"}), hide_index=True, use_container_width=True)
