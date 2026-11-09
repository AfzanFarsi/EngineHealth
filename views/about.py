import pandas as pd
import streamlit as st

import config as cfg
from src import app_data, ui
from src.syllabus import coverage_percent, coverage_table

st.title("About the Project")

ui.section("Official problem statement")
st.markdown(
    "**SIH26054 · EngineHealth - Aero Piston Engine Fault Prediction** (Smart India Hackathon 2026, adapted for INT395 Supervised Learning CA1)\n\n"
    "> MALE UAV piston engines currently rely on scheduled maintenance rather than condition-based prediction, risking in-flight "
    "failures and unnecessary servicing. Build a supervised regression model that predicts remaining useful life (RUL) or imminent-fault "
    "probability from engine sensor time-series. Deliverable: trained model + RUL prediction curves vs. ground truth.")
ui.note("<b>Proxy dataset.</b> The NASA C-MAPSS Turbofan Engine Degradation Simulation data set is used as an accessible open proxy for piston-engine "
        "telemetry. It is simulated <b>turbofan</b> data, not real piston-engine telemetry, so the numbers here demonstrate the method and "
        "must not be read as piston-engine accuracy.")

ui.section("Objectives")
st.markdown("- Predict **Remaining Useful Life** (regression) from multivariate sensor time-series.\n"
            f"- Flag **imminent faults** (classification: RUL ≤ {cfg.FAULT_THRESHOLD} cycles) as a separate module.\n"
            "- Compare many supervised algorithms fairly, without data leakage, on all four C-MAPSS subsets.\n"
            "- Explain predictions (SHAP, LIME) and deliver a deployable monitoring website.")

ui.section("Dataset information")
st.markdown("Each subset is a fleet of engines of the same type simulated with the C-MAPSS model (Saxena et al., PHM08). Each row = one flight cycle: "
            "engine id, cycle, 3 operating settings, 21 sensors. Training engines run to failure; test engines stop before failure and "
            "`RUL_FDxxx.txt` gives the true remaining cycles after the last recorded cycle.")
rows = []
for s, info in cfg.SUBSETS.items():
    try:
        tr, te = app_data.raw_train(s), app_data.raw_test(s)
        rows.append({"Subset": s, "Operating conditions": info["conditions"], "Fault modes": info["fault_modes"],
                     "Train engines": tr["unit"].nunique(), "Train cycles": len(tr), "Test engines": te["unit"].nunique(), "Test cycles": len(te)})
    except FileNotFoundError:
        rows.append({"Subset": s, "Operating conditions": info["conditions"], "Fault modes": info["fault_modes"]})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

ui.section("Methodology")
st.markdown(
    "1. **Load & check** - 26 columns, no header; verify missing values, duplicates, constant sensors, cycle continuity.\n"
    f"2. **Target** - training RUL = last cycle − current cycle; a piece-wise linear cap at {cfg.RUL_CAP} cycles treats early life as equally healthy.\n"
    "3. **Preprocessing (fitted on training engines only)** - K-Means operating-condition id (6-condition subsets), constant-sensor removal, per-condition z-score, "
    f"correlation-based selection (|Spearman| ≥ {cfg.MIN_ABS_SPEARMAN}).\n"
    f"4. **Feature engineering** - per sensor: current value, rolling mean / std (window {cfg.WINDOW}), trend; plus cycle count. All causal (no future data).\n"
    "5. **Splitting** - 80/20 by *engine id*, GroupKFold for tuning; the preprocessor is fitted on the 80 % only during validation.\n"
    "6. **Model comparison & tuning** - 13 regression algorithms (+ mean baseline and tuned variants), GridSearchCV / RandomizedSearchCV; best model chosen by validation RMSE.\n"
    "7. **Final test** - refit on all training engines, evaluate once on the official test set.\n"
    "8. **Explain** - SHAP + LIME; **Time series** - AR/MA/ARIMA/SARIMA baselines.")

ui.section("Models")
rows = []
for s in app_data.available_subsets():
    j = app_data.summary(s)
    rows.append({"Subset": s, "Best RUL model": j["best_model"], "Explained model (SHAP)": j["best_tree_model_for_shap"],
                 "Risk classifier": j["classification"]["best_model"], "Features": j["n_features"], "Sensors kept": len(j["sensors_kept"])})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.markdown("**Algorithms:** Linear, Polynomial (deg 2), Ridge, Lasso, RANSAC, Decision Tree, SVR, Bagging, Random Forest, AdaBoost, Gradient Boosting, "
            "HistGradientBoosting, Stacking. **Classifiers:** Perceptron, Logistic Regression, KNN, Decision Tree, SVM, Naive Bayes, Random Forest, HistGradientBoosting.")

ui.section("Limitations")
st.markdown(
    "- C-MAPSS is **simulated turbofan** data - a proxy, not piston-engine telemetry; no real-flight validation.\n"
    "- Models predict a **capped** RUL (early life treated as ≥ 125 cycles); the official test RMSE is measured against the uncapped truth.\n"
    "- The prediction interval is a global empirical residual range, not an engine-specific guarantee.\n"
    "- SVR/SVM are trained on 6,000-row samples for speed; Bayesian optimisation is not implemented.\n"
    "- Only the last cycle of 100-260 test engines is scored officially, so test metrics have sampling noise.\n"
    "- ARIMA/SARIMA are single-series models and are used only as baselines.")

ui.section("Technology stack")
st.markdown(ui.pills(["Python 3.12", "pandas", "NumPy", "scikit-learn", "statsmodels", "SHAP", "LIME", "Plotly", "Matplotlib",
                      "Seaborn", "Streamlit", "Joblib", "Pytest"]), unsafe_allow_html=True)

ui.section("INT395 syllabus coverage (Units I-V)")
t = coverage_table()
counted = t[t["Status"] != "Not applicable"]
ui.kpi_row([("Syllabus topics mapped", str(len(t)), "Units I-V", "#22D3EE"),
            ("Implemented with evidence", str((t["Status"] == "Implemented").sum()), f"{coverage_percent():.0f} % of applicable topics", "#34D399"),
            ("Not implemented", str((t["Status"] == "Not implemented").sum()), "Bayesian optimisation", "#FBBF24"),
            ("Not applicable", str((t["Status"] == "Not applicable").sum()), "no categorical variables", "#3B82F6")])
unit = st.multiselect("Filter by unit", ["I", "II", "III", "IV", "V"], default=["I", "II", "III", "IV", "V"], key="syl_units")
st.dataframe(t[t["Unit"].isin(unit)], hide_index=True, use_container_width=True, height=420)
