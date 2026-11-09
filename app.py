"""EngineHealth - AI-powered engine health monitoring & predictive maintenance.

Run with:   streamlit run app.py
"""
import streamlit as st

import config as cfg
from src import app_data, ui

st.set_page_config(page_title="EngineHealth", page_icon=":material/flight:", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()
st.logo("assets/logo.svg", size="large")

subsets = app_data.available_subsets()

with st.sidebar:
    st.markdown("### Dataset")
    if subsets:
        st.selectbox("NASA C-MAPSS subset", subsets, key="subset",
                     help="Each subset has its own preprocessing, trained models, metrics and curves.")
        info = cfg.SUBSETS[st.session_state["subset"]]
        st.caption(f"{info['conditions']} operating condition(s) · {info['fault_modes']}")
    else:
        st.warning("No trained models found yet.")
    st.divider()
    st.caption("Proxy dataset: NASA C-MAPSS simulates **turbofan** degradation and is used here as an open "
               "stand-in for piston-engine telemetry.")

pages = {
    "": [st.Page("views/home.py", title="Home", icon=":material/home:", default=True)],
    "Monitor": [
        st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:"),
        st.Page("views/prediction.py", title="Engine Prediction", icon=":material/query_stats:"),
    ],
    "Analyse": [
        st.Page("views/analytics.py", title="Sensor Analytics", icon=":material/monitoring:"),
        st.Page("views/performance.py", title="Model Performance", icon=":material/speed:"),
        st.Page("views/explainability.py", title="Explainable AI", icon=":material/psychology:"),
    ],
    "Project": [st.Page("views/about.py", title="About Project", icon=":material/info:")],
}
st.navigation(pages).run()
