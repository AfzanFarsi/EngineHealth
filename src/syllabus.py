"""INT395 syllabus coverage map.  Status values are honest:
Implemented = code runs and produces evidence; Not implemented = only discussed / absent;
Not applicable = the dataset has no such issue (explained)."""
from __future__ import annotations

import pandas as pd

IMPL, NOT, NA = "Implemented", "Not implemented", "Not applicable"

_ROWS = [
    # unit, concept, where, evidence, status
    ("I", "Overview & types of supervised learning (regression vs classification)", "views/about.py, README", "Both a regression (RUL) and a classification (fault risk) task", IMPL),
    ("I", "Python / scikit-learn environment", "requirements.txt, README", "Virtual-environment setup instructions", IMPL),
    ("I", "Types of data, loading datasets", "src/data_loader.py", "Multivariate time-series, 26 columns, 4 subsets loaded", IMPL),
    ("I", "Data exploration (Pandas, Matplotlib/Seaborn)", "views/analytics.py, src/visualization.py", "outputs/figures/<subset>/sensor_*.png, correlation_heatmap.png", IMPL),
    ("I", "Common data issues & missing values", "src/preprocessing.py::data_quality_report", "outputs/metrics/<subset>_data_quality.json (0 missing values found)", IMPL),
    ("I", "Outliers", "src/preprocessing.py::data_quality_report", "IQR outlier counts reported; kept because extremes carry degradation signal", IMPL),
    ("I", "Handling class imbalance", "src/models.py (class_weight='balanced')", "Positive class (RUL<=30) is a minority; classification tables", IMPL),
    ("I", "Scaling features", "src/preprocessing.py::Preprocessor", "Per-operating-condition z-score", IMPL),
    ("I", "Encoding categorical variables", "-", "All 26 columns are numeric; operating condition id is derived numerically", NA),
    ("I", "Feature engineering", "src/feature_engineering.py", "Rolling mean/std, trend per sensor, cycle number", IMPL),
    ("I", "Feature selection", "src/preprocessing.py::Preprocessor", "outputs/metrics/<subset>_feature_selection.csv (constant + weak sensors dropped)", IMPL),
    ("I", "Data splitting", "src/train.py", "80/20 split by engine id (GroupShuffleSplit) + GroupKFold", IMPL),
    ("I", "Dimensionality reduction (PCA)", "src/preprocessing.py::pca_analysis, views/analytics.py", "PCA tab: explained variance + PC1/PC2 map (analysis only, not used by the final model)", IMPL),
    ("II", "Overview of classification", "src/train.py::run_classification", "Imminent-fault classifier (RUL <= 30 cycles)", IMPL),
    ("II", "Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix", "src/evaluate.py", "outputs/metrics/<subset>_classification_*.csv", IMPL),
    ("II", "Perceptron", "src/models.py", "Row in classification tables", IMPL),
    ("II", "Logistic Regression", "src/models.py", "Row in classification tables", IMPL),
    ("II", "k-Nearest Neighbors", "src/models.py", "Row in classification tables", IMPL),
    ("II", "Decision Tree", "src/models.py", "Row in classification tables", IMPL),
    ("II", "Support Vector Machine", "src/models.py", "Row in classification tables (trained on a 6,000-row sample)", IMPL),
    ("II", "Naive Bayes", "src/models.py", "Row in classification tables", IMPL),
    ("III", "Overview of ensembles; how ensembles reduce error", "views/about.py, views/performance.py", "Single tree vs bagging/boosting/stacking comparison", IMPL),
    ("III", "Bagging", "src/models.py", "Bagging (Decision Trees), Random Forest", IMPL),
    ("III", "Boosting", "src/models.py", "AdaBoost, Gradient Boosting, HistGradientBoosting", IMPL),
    ("III", "Stacking", "src/models.py", "Stacking (RF+HGB+Ridge)", IMPL),
    ("III", "Need for hyper-parameter tuning; Grid search", "src/train.py", "GridSearchCV: Ridge, Decision Tree, SVR -> <subset>_tuning.csv", IMPL),
    ("III", "Random search", "src/train.py", "RandomizedSearchCV: HistGB, Random Forest -> <subset>_tuning.csv", IMPL),
    ("III", "Bayesian optimization", "-", "Not used (would need an extra library such as scikit-optimize)", NOT),
    ("III", "Need for XAI; global vs local interpretability", "views/explainability.py", "Explainability page", IMPL),
    ("III", "Feature importance", "src/explainability.py", "Permutation importance + SHAP importance CSVs", IMPL),
    ("III", "LIME", "src/explainability.py::lime_explanation", "Local explanation on the Explainability page", IMPL),
    ("III", "SHAP", "src/explainability.py", "Summary plot + local waterfall-style chart", IMPL),
    ("III", "Model transparency & trustworthiness", "views/explainability.py", "Sensor-level contributions + empirical prediction interval", IMPL),
    ("IV", "Regression workflow & EDA", "src/train.py, views/analytics.py", "End-to-end RUL regression pipeline", IMPL),
    ("IV", "Evaluation metrics (MAE, RMSE, R2, NASA score)", "src/evaluate.py", "<subset>_regression_*.csv", IMPL),
    ("IV", "Linear regression", "src/models.py", "Row in regression tables", IMPL),
    ("IV", "RANSAC", "src/models.py", "Row in regression tables", IMPL),
    ("IV", "Polynomial regression", "src/models.py", "Degree-2 polynomial on current sensor values + Ridge", IMPL),
    ("IV", "Regularized regression (Ridge, Lasso)", "src/models.py", "Rows in regression tables", IMPL),
    ("IV", "Support vector regression", "src/models.py", "SVR (RBF), trained on a 6,000-row sample", IMPL),
    ("IV", "Decision tree regression", "src/models.py", "Row in regression tables", IMPL),
    ("IV", "Random forest regression", "src/models.py", "Row in regression tables", IMPL),
    ("V", "Time-series characteristics (trend, stationarity)", "src/timeseries.py", "ADF test + ACF/PACF -> <subset>_timeseries.json", IMPL),
    ("V", "Univariate vs multivariate; regular vs irregular intervals", "src/timeseries.py, views/analytics.py", "Documented in the Time-series tab", IMPL),
    ("V", "Preparing time-series data & splitting strategies", "src/feature_engineering.py, src/train.py", "Causal rolling features; engine-wise and chronological splits", IMPL),
    ("V", "Autoregressive (AR) models", "src/timeseries.py", "AR(2)+trend forecast of one sensor", IMPL),
    ("V", "Moving average models", "src/timeseries.py", "Moving-average forecast + MA(2) model", IMPL),
    ("V", "ARIMA", "src/timeseries.py", "ARIMA(1,1,1)+drift forecast + ARIMA health-indicator RUL baseline", IMPL),
    ("V", "SARIMA", "src/timeseries.py", "Fitted and compared; seasonality check shows it is not appropriate here", IMPL),
]


def coverage_table() -> pd.DataFrame:
    return pd.DataFrame(_ROWS, columns=["Unit", "Concept", "Where implemented", "Evidence / output", "Status"])


def coverage_percent() -> float:
    t = coverage_table()
    t = t[t["Status"] != NA]
    return 100.0 * (t["Status"] == IMPL).mean()
