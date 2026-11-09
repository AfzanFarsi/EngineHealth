import streamlit as st

import config as cfg
from src import app_data, ui

subsets = app_data.available_subsets()

st.markdown("""
<div class='eh-hero'>
  <div class='eh-eyebrow'>Condition-based maintenance · SIH26054</div>
  <h1>Know your engine's <span class='eh-grad'>remaining life</span> before it matters.</h1>
  <p class='eh-lead'>EngineHealth turns engine sensor time-series into a predicted Remaining Useful Life (RUL),
  a health category and a maintenance recommendation, with every prediction explained, so maintenance can move
  from fixed schedules to real engine condition.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, _ = st.columns([1.2, 1.2, 4])
if c1.button("Open dashboard", type="primary", icon=":material/dashboard:"):
    st.switch_page("views/dashboard.py")
if c2.button("Try a prediction", icon=":material/query_stats:"):
    st.switch_page("views/prediction.py")

if subsets:
    s = app_data.current_subset()
    j = app_data.summary(s)
    ui.kpi_row([
        ("Subsets trained", str(len(subsets)), ", ".join(subsets), "#22D3EE"),
        (f"{s} test RMSE", f"{j['test_best']['official_RMSE']:.1f} cycles", f"best model: {j['best_model']}", "#3B82F6"),
        (f"{s} test R²", f"{j['test_best']['official_R2']:.2f}", "official last-cycle RUL", "#A78BFA"),
        ("Fault-risk F1", f"{j['classification']['test_all_cycles']['F1']:.2f}",
         f"RUL ≤ {cfg.FAULT_THRESHOLD} cycles · {j['classification']['best_model']}", "#34D399"),
    ])
else:
    st.warning(app_data.missing_artifacts_message(cfg.DEFAULT_SUBSET))

ui.note("<b>Proxy-data notice:</b> the assigned problem concerns MALE UAV <b>piston</b> engines. NASA C-MAPSS simulates "
        "<b>turbofan</b> degradation and is used as an accessible open proxy; results describe turbofan data, not real piston-engine telemetry.")

ui.section("Key features")
feats = [
    ("timer", "RUL prediction", "Regression models estimate how many operating cycles remain before failure."),
    ("warning", "Fault-risk alert", f"A separate classifier flags engines within {cfg.FAULT_THRESHOLD} cycles of failure."),
    ("psychology", "Explainable AI", "SHAP and LIME show which sensors drive each individual prediction."),
    ("flight", "Four NASA subsets", "FD001-FD004: one or six operating conditions, one or two fault modes."),
    ("insights", "Honest evaluation", "Engine-wise validation, official test RUL, NASA score and uncertainty intervals."),
    ("science", "Classical baselines", "AR / MA / ARIMA / SARIMA compared with supervised ML."),
]
for row in (feats[:3], feats[3:]):
    cols = st.columns(3)
    for c, f in zip(cols, row):
        c.markdown(ui.card(*f), unsafe_allow_html=True)
    st.write("")

ui.section("How the AI system works")
steps = [("Sensor data", "21 sensors + 3 operating settings per flight cycle."),
         ("Preprocessing", "Operating-condition normalisation, constant-sensor removal, feature selection."),
         ("Feature engineering", "Rolling mean, rolling std and trend over the last 10 cycles."),
         ("Models", "Linear, tree, kernel and ensemble regressors; tuned with GroupKFold."),
         ("Decision support", "Predicted RUL → Healthy / Warning / Critical → recommendation.")]
cols = st.columns(len(steps))
for i, (c, (t, d)) in enumerate(zip(cols, steps), 1):
    c.markdown(f"<div class='eh-card'><div class='eh-step'><div class='n'>{i}</div><div><h4>{t}</h4><p>{d}</p></div></div></div>",
               unsafe_allow_html=True)

ui.section("Technology stack")
st.markdown(ui.pills(["Python 3.12", "Pandas", "NumPy", "scikit-learn", "statsmodels", "SHAP", "LIME", "Plotly",
                      "Matplotlib", "Streamlit", "Joblib", "Pytest"]), unsafe_allow_html=True)
