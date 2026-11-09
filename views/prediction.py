import io

import numpy as np
import pandas as pd
import streamlit as st

import config as cfg
from src import app_data, ui
from src import visualization as viz
from src.evaluate import health_category, maintenance_advice
from src.predict import predict_rul, predict_risk, validate_raw_frame

st.title("Engine Prediction")
subset = app_data.current_subset()
if subset not in app_data.available_subsets():
    st.warning(app_data.missing_artifacts_message(subset)); st.stop()

bundle = app_data.rul_bundle(subset)
try:
    risk = app_data.risk_bundle(subset)
except FileNotFoundError:
    risk = None

st.caption(f"{subset} · model: **{bundle['model_name']}** (saved model, trained on {bundle['trained_on_engines']} engines). "
           "Rolling features use only cycles up to the selected one, so no future data is used.")

source = st.radio("Data source", ["NASA test engine", "Upload my own sensor file"], horizontal=True, key="pred_source")
traj, truth_available = None, False

if source == "NASA test engine":
    test = app_data.raw_test(subset)
    c1, c2 = st.columns(2)
    units = sorted(test["unit"].unique())
    unit = c1.selectbox("Engine ID", units, key="pred_unit")
    g = test[test["unit"] == unit].reset_index(drop=True)
    max_c = int(g["cycle"].max())
    cycle = c2.slider("Operating cycle", 1, max_c, max_c, key=f"pred_cycle_{subset}_{unit}")
    traj, truth_available = g, True
    st.markdown(f"**Sensor readings at cycle {cycle}** (raw values)")
    row = g[g["cycle"] == cycle].iloc[0]
    show = pd.DataFrame({"Signal": cfg.SETTING_COLS + bundle["preprocessor"].sensors_,
                         "Meaning": ["Operating setting"] * 3 + [cfg.SENSOR_DESCRIPTIONS[s] for s in bundle["preprocessor"].sensors_],
                         "Value": [row[c] for c in cfg.SETTING_COLS + bundle["preprocessor"].sensors_]})
    with st.expander("Show sensor table", expanded=False):
        st.dataframe(show, hide_index=True, use_container_width=True, height=300)
else:
    st.markdown("Upload a text/CSV file with the **26 NASA columns** "
                "(`unit, cycle, op1-op3, s1-s21`), with or without a header row. Cycles must start at 1 and be consecutive for each engine.")
    sample = app_data.raw_test(subset)
    sample = sample[sample["unit"] == sample["unit"].min()].drop(columns="rul")
    st.download_button("Download a sample file", sample.to_csv(index=False).encode(), f"sample_{subset}_engine.csv", "text/csv")
    up = st.file_uploader("Sensor file", type=["csv", "txt"])
    if up is not None:
        try:
            raw = up.read().decode("utf-8", errors="ignore")
            has_header = raw.lstrip().lower().startswith("unit")
            df = pd.read_csv(io.StringIO(raw), sep=None, engine="python", header=0 if has_header else None)
            df = validate_raw_frame(df)
            units = sorted(df["unit"].unique())
            unit = st.selectbox("Engine ID in file", units, key="up_unit")
            traj = df[df["unit"] == unit].reset_index(drop=True)
            bad = (traj["cycle"].diff().dropna() != 1).any() or traj["cycle"].iloc[0] != 1
            if bad:
                st.error("Cycles must start at 1 and increase by exactly 1."); traj = None
            else:
                cycle = st.slider("Operating cycle", 1, int(traj["cycle"].max()), int(traj["cycle"].max()), key="up_cycle")
        except Exception as e:
            st.error(f"Could not read the file: {e}"); traj = None

if traj is not None:
    key = (subset, source, int(unit), int(cycle))
    if st.button("Run prediction", type="primary", icon=":material/play_arrow:"):
        hist = traj[traj["cycle"] <= cycle].reset_index(drop=True)
        r = predict_rul(bundle, hist)
        rk = predict_risk(risk, hist) if risk else None
        st.session_state["pred_result"] = {"key": key, "rul": r, "risk": rk}

    res = st.session_state.get("pred_result")
    if res and res["key"] == key:
        r, rk = res["rul"], res["risk"]
        now = r.iloc[-1]
        cat = health_category(now["rul_pred"])
        kind = cat
        truth = None
        if truth_available:
            truth = float(traj.loc[traj["cycle"] == cycle, "rul"].iloc[0])
        ui.kpi_row([
            ("Predicted RUL", f"{now['rul_pred']:.0f} cycles", f"{int(cfg.INTERVAL_COVERAGE*100)}% interval: {now['rul_low']:.0f} - {now['rul_high']:.0f}", "#22D3EE"),
            ("Health category", cat, f"thresholds {cfg.FAULT_THRESHOLD} / {cfg.WARNING_THRESHOLD} cycles", {"Healthy": "#34D399", "Warning": "#FBBF24", "Critical": "#F87171"}[cat]),
            ("Imminent-fault risk", f"{rk.iloc[-1]['risk_score']*100:.0f}%" if rk is not None else "n/a",
             f"classifier score for RUL ≤ {cfg.FAULT_THRESHOLD}" if rk is not None else "risk model not found", "#A78BFA"),
            ("Ground truth", f"{min(truth, cfg.RUL_CAP):.0f} cycles" if truth is not None else "unknown",
             (f"abs. error {abs(now['rul_pred'] - min(truth, cfg.RUL_CAP)):.1f} · true RUL {truth:.0f}" if truth is not None else "uploaded data has no label"), "#3B82F6"),
        ])
        st.markdown(f"<div class='eh-note'><b>Recommendation:</b> {maintenance_advice(now['rul_pred'])}</div>", unsafe_allow_html=True)
        d = traj[traj["cycle"] <= cycle][["cycle"] + (["rul"] if truth_available else [])].reset_index(drop=True)
        d["pred"] = r["rul_pred"].to_numpy()
        d["low"], d["high"] = r["rul_low"].to_numpy(), r["rul_high"].to_numpy()
        if not truth_available:
            d["rul_true"] = np.nan
        else:
            d = d.rename(columns={"rul": "rul_true"})
        fig = viz.plotly_rul_curve(d, {f"Predicted RUL ({bundle['model_name']})": "pred"},
                                   f"Engine {unit}: predicted RUL up to cycle {cycle}", d["low"], d["high"], cycle)
        if not truth_available:
            fig.data = tuple(t for t in fig.data if not str(t.name).startswith("Ground truth"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("The interval is the empirical 80 % range of residuals on held-out validation engines (a global estimate, "
                   "not engine-specific). The ground-truth curve for NASA test engines is derived from the official RUL_FDxxx.txt file.")
    else:
        st.info("Choose an engine and cycle, then press **Run prediction**.")
