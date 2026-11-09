"""Cached data access for the website (nothing here trains or fabricates results)."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import config as cfg
from src import data_loader as dl
from src.predict import load_bundle, make_features


def available_subsets() -> list[str]:
    """Subsets whose trained model + metrics exist."""
    return [s for s in cfg.SUBSETS
            if (cfg.MODELS_DIR / f"{s}_rul_model.joblib").exists() and (cfg.METRICS_DIR / f"{s}_summary.json").exists()]


def current_subset() -> str:
    return st.session_state.get("subset", cfg.DEFAULT_SUBSET)


@st.cache_data(show_spinner=False)
def summary(subset: str) -> dict:
    return json.loads((cfg.METRICS_DIR / f"{subset}_summary.json").read_text())


@st.cache_data(show_spinner=False)
def metrics_csv(subset: str, name: str) -> pd.DataFrame:
    return pd.read_csv(cfg.METRICS_DIR / f"{subset}_{name}.csv")


@st.cache_data(show_spinner=False)
def metrics_json(subset: str, name: str) -> dict:
    return json.loads((cfg.METRICS_DIR / f"{subset}_{name}.json").read_text())


@st.cache_data(show_spinner=False)
def predictions(subset: str, kind: str) -> pd.DataFrame:
    return pd.read_csv(cfg.PREDICTIONS_DIR / f"{subset}_{kind}_predictions.csv.gz")


@st.cache_data(show_spinner=False)
def raw_train(subset: str) -> pd.DataFrame:
    return dl.add_train_rul(dl.load_train(subset))


@st.cache_data(show_spinner=False)
def raw_test(subset: str) -> pd.DataFrame:
    """Test trajectories with ground-truth RUL for every row (derived from RUL_FDxxx.txt)."""
    return dl.add_test_rul(dl.load_test(subset), dl.load_rul_truth(subset))


@st.cache_resource(show_spinner=False)
def rul_bundle(subset: str) -> dict:
    return load_bundle(subset, "rul")


@st.cache_resource(show_spinner=False)
def risk_bundle(subset: str) -> dict:
    return load_bundle(subset, "risk")


@st.cache_data(show_spinner=False)
def train_features(subset: str) -> pd.DataFrame:
    b = rul_bundle(subset)
    return make_features(b["preprocessor"], raw_train(subset), b["window"])[b["feature_names"]]


@st.cache_data(show_spinner=False)
def test_features(subset: str) -> pd.DataFrame:
    b = rul_bundle(subset)
    return make_features(b["preprocessor"], raw_test(subset), b["window"])[b["feature_names"]]


@st.cache_resource(show_spinner=False)
def shap_explainer(subset: str):
    from src.explainability import make_explainer
    return make_explainer(rul_bundle(subset)["explain_model"])


def missing_artifacts_message(subset: str) -> str:
    return (f"No trained model found for **{subset}**. Train it first:\n\n"
            f"`python run_pipeline.py --subset {subset}`")
