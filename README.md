# EngineHealth: AI-Powered Engine Health Monitoring & Predictive Maintenance

**INT395 Supervised Learning · CA1 project · SIH26054 "EngineHealth - Aero Piston Engine Fault Prediction"**

EngineHealth predicts the **Remaining Useful Life (RUL)** of an engine from its sensor time-series, flags engines that are
close to failure, explains every prediction, and presents everything in a Streamlit website.

> **Proxy-data notice.** The assigned problem concerns MALE UAV *piston* engines. The NASA **C-MAPSS** data set simulates
> **turbofan** engine degradation and is used as an accessible open proxy. It is *not* real piston-engine telemetry, so the
> results demonstrate the method and must not be read as piston-engine accuracy.

## 1. Problem statement and objectives
MALE UAV engines are serviced on a fixed schedule instead of by condition, risking in-flight failures and unnecessary servicing.
Objectives: (1) build a supervised **regression** model for RUL; (2) add a separate **fault-risk classifier** (RUL ≤ 30 cycles);
(3) compare many algorithms on all four C-MAPSS subsets without data leakage; (4) explain predictions (SHAP/LIME);
(5) deliver a deployable website with RUL prediction curves vs ground truth.

## 2. Dataset (NASA C-MAPSS, kept unchanged in `dataset/raw/`)
Each row = one flight cycle: `unit, cycle, op1-op3` (operating settings) and `s1-s21` (sensors) - 26 space-separated columns, no header.
Training engines run to failure; test engines stop earlier and `RUL_FDxxx.txt` gives the true remaining cycles.

| Subset | Operating conditions | Fault modes | Train engines | Test engines |
|---|---|---|---|---|
| FD001 | 1 | HPC degradation | 100 | 100 |
| FD002 | 6 | HPC degradation | 260 | 259 |
| FD003 | 1 | HPC + Fan degradation | 100 | 100 |
| FD004 | 6 | HPC + Fan degradation | 249 | 248 |

Checks (`outputs/metrics/<subset>_data_quality.json`): no missing values, no duplicates, 5-6 constant sensors in every subset, cycles consecutive.
Source: A. Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08 (NASA Prognostics Data Repository, public).

## 3. Architecture
```
sensor files -> data_loader -> Preprocessor (condition clusters, constant-sensor removal, per-condition z-score, feature selection)
            -> feature_engineering (rolling mean/std, trend, cycle) -> models (13 regressors, 8 classifiers)
            -> tuning (GridSearchCV / RandomizedSearchCV + GroupKFold) -> saved model (.joblib)
            -> predict.py -> Streamlit website (7 pages) ; explainability.py (SHAP/LIME) ; timeseries.py (AR/MA/ARIMA/SARIMA)
```

## 4. Preprocessing and why
| Step | Why |
|---|---|
| Missing-value / duplicate / constant checks | confirm data quality; constant sensors carry no information |
| RUL target = last cycle − cycle, **capped at 125** | early life is "healthy"; the piece-wise linear target is the standard in the literature |
| Operating-condition id (K-Means, FD002/FD004) + **per-condition z-score** | otherwise the model confuses altitude/speed changes with degradation |
| Feature selection (constant + weak-correlation sensors dropped) | fewer noisy inputs |
| Rolling mean / std / trend (10 cycles), **causal** | sensors are noisy; trends reveal degradation; never uses the future |
| **Split by engine id** (80/20) and GroupKFold | rows of one engine are near-identical; random row splits leak |
| Preprocessor fitted on training engines only | no leakage of validation/test statistics |
| Outliers kept | extreme values near failure are the signal |

## 5. Algorithms and evaluation
* **Regression (13 + baseline):** Linear, Polynomial (deg 2), Ridge, Lasso, RANSAC, Decision Tree, SVR (RBF), Bagging, Random Forest, AdaBoost,
  Gradient Boosting, HistGradientBoosting, Stacking; GridSearchCV (Ridge, Decision Tree, SVR) and RandomizedSearchCV (HistGB, Random Forest).
* **Classification (target RUL ≤ 30):** Perceptron, Logistic Regression, KNN, Decision Tree, SVM, Naive Bayes, Random Forest, HistGradientBoosting (`class_weight='balanced'`).
* **Metrics:** regression - MAE, RMSE, R², NASA score (lower is better, penalises late predictions); classification - Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix.
  Accuracy is **not** used for regression.
* **Protocol:** models are compared on held-out *training engines* (validation); the best one is re-fitted on all training engines and scored **once** on the official test set.
  "Official" test metrics use the last cycle of each test engine against `RUL_FDxxx.txt` (uncapped truth).
* **Explainability:** SHAP (global + local), permutation importance, LIME. **Time series (Unit V):** ADF test, ACF/PACF, AR, MA, ARIMA, SARIMA,
  and an ARIMA health-indicator RUL baseline compared with the ML model.

## 6. Results (all numbers are produced by the code; see `outputs/metrics/`)
**Regression - best model per subset** (selected on validation RMSE; test = official NASA last-cycle RUL, evaluated once):

| Subset | Best model | Val RMSE | Test RMSE | Test MAE | Test R² | NASA score |
|---|---|---|---|---|---|---|
| FD001 | Random Forest (tuned) | 14.51 | 18.63 | 13.53 | 0.799 | 668 |
| FD002 | Stacking (RF+HGB+Ridge) | 15.87 | 27.29 | 19.15 | 0.743 | 9306 |
| FD003 | Random Forest | 12.70 | 19.44 | 14.88 | 0.779 | 922 |
| FD004 | Stacking (RF+HGB+Ridge) | 19.03 | 28.44 | 20.98 | 0.728 | 5784 |

**Fault-risk classification** (RUL ≤ 30; best model by validation F1; test = all test cycles):

| Subset | Best classifier | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| FD001 | SVM (RBF) | 0.985 | 0.695 | 0.741 | 0.717 | 0.985 |
| FD002 | Random Forest | 0.979 | 0.621 | 0.874 | 0.726 | 0.989 |
| FD003 | K-Nearest Neighbors | 0.993 | 0.849 | 0.698 | 0.766 | 0.970 |
| FD004 | HistGradientBoosting | 0.987 | 0.660 | 0.793 | 0.720 | 0.994 |

**Classical time-series baseline** (ARIMA health-indicator extrapolation) vs supervised ML, official test RMSE:

| Subset | ARIMA baseline | Supervised ML |
|---|---|---|
| FD001 | 35.1 | 18.6 |
| FD002 | 40.0 | 27.3 |
| FD003 | 39.9 | 19.4 |
| FD004 | 46.7 | 28.4 |

**How to read these honestly**
* The mean-RUL baseline has test RMSE ≈ 43 on FD001, so the models add real skill, but the official RMSE (18-28 cycles) is far from perfect. R² is 0.73-0.80.
* The 85-95 % "accuracy" guideline is met only by **classification accuracy (98-99 %)**, which is inflated because near-failure rows are rare. Precision/recall/F1 (0.72-0.77 F1) are the more honest figures. For RUL regression use RMSE/MAE/R².
* FD002/FD004 (six operating conditions) and FD003/FD004 (two fault modes) are harder; validation RMSE is lower than test RMSE because validation uses all cycles of held-out engines and the test score uses only each engine's last cycle against the uncapped truth.
* Full tables: `outputs/metrics/<subset>_regression_validation.csv`, `_regression_test.csv`, `_tuning.csv`, `_classification_*.csv`; figures: `outputs/figures/<subset>/`; cross-subset: `outputs/figures/cross_subset_comparison.png`.


## 7. Website pages
Home · Dashboard (fleet KPIs, health donut, priority list, sensor monitor) · Engine Prediction (pick an engine and cycle or upload a file; RUL, category,
recommendation, 80 % interval, fault-risk score) · Sensor Analytics (trends, correlation, distributions, degradation curves, importance, operating conditions,
PCA, time-series, data quality) · Model Performance (comparison, RUL curves, actual vs predicted, residuals, tuning, classification, all subsets) ·
Explainable AI · About (problem, method, limits, INT395 syllabus coverage). The **subset selector** (FD001-FD004) is in the sidebar.

## 8. Folder structure
```
EngineHealth/
├── app.py                 Streamlit entry point (navigation + dataset selector)
├── run_pipeline.py        one command: train -> explain -> time-series -> figures
├── config.py              all paths and constants
├── requirements.txt       exact dependency versions (requirements-dev.txt adds pytest/ipykernel)
├── .streamlit/config.toml theme + server settings
├── assets/logo.svg
├── dataset/raw/           the 12 NASA files (+ NASA_readme.txt)  - never modified
├── src/                   data_loader, preprocessing, feature_engineering, models, train, evaluate,
│                          predict, explainability, timeseries, visualization, syllabus, ui, app_data
├── views/                 the 7 website pages
├── models/                trained .joblib files (<subset>_rul_model, <subset>_risk_model)
├── outputs/               figures/<subset>/*.png, metrics/ (csv/json), predictions/ (csv.gz)
├── notebooks/             EngineHealth_FD001_Walkthrough.ipynb (explained step by step)
└── tests/                 pytest suite (pipeline + website)
```

## 9. Install and run (Windows, Python 3.12)
```powershell
cd D:\Python_Projects\EngineHealth
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
streamlit run app.py
```
Run `python check_setup.py` first to confirm all files are present (if you downloaded the project in several ZIP parts, extract them all into the same `EngineHealth` folder).
Open <http://localhost:8501>. Stop with `Ctrl+C`. Restart later: open the folder in VS Code, run `.venv\Scripts\activate`, then `streamlit run app.py`.
The trained models are already included, so no training is needed. To retrain: `python run_pipeline.py --subset FD001` (≈ 8 min) or `--all` (≈ 1 hour on 2 CPU cores).
Tests: `pip install -r requirements-dev.txt` then `python -m pytest -q`.

### Common errors
| Error | Fix |
|---|---|
| `python` / `pip` not recognized | reinstall Python from python.org and tick **Add python.exe to PATH**, or use `py -3.12 -m venv .venv` |
| Activation blocked ("running scripts is disabled") | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Command Prompt: `.venv\Scripts\activate.bat` |
| `ModuleNotFoundError` | the venv is not active, or run `pip install -r requirements.txt` again |
| `Dataset file not found` | the 12 files `train_/test_/RUL_FD00x.txt` must be in `dataset\raw\` |
| `Model file not found` | run `python run_pipeline.py --subset FD001` |
| Port already in use | `streamlit run app.py --server.port 8502` |
| C: drive is full during install | `pip install --no-cache-dir ...` and keep the `.venv` inside the project on D: |

## 10. Deployment (Streamlit Community Cloud)
1. Create a GitHub repository and push this folder (the `models/` files are < 100 MB each; check with `dir models`).
2. Go to <https://share.streamlit.io>, sign in with GitHub, **Create app**, choose the repo, branch `main`, main file `app.py`.
3. In *Advanced settings* choose **Python 3.12**. Deploy.
`requirements.txt` pins exact versions, so the cloud build matches local. *Deployment to the cloud has not been performed by the author of this package; it has been tested locally only.*

## 11. Limitations
Simulated turbofan proxy data; capped RUL; global (not engine-specific) prediction interval; SVR/SVM trained on 6,000-row samples;
Bayesian optimisation not implemented; only one row per engine is officially scored (noisy test metrics); ARIMA/SARIMA are single-series baselines.

## 12. Future improvements
Sequence models (LSTM/1-D CNN/Transformer) for comparison; Bayesian optimisation; conformal prediction intervals per engine;
real piston-engine telemetry as a primary dataset; live sensor streaming; model monitoring and drift detection.
