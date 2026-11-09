"""Pipeline tests: loading, preprocessing, features, training, saving, loading, prediction."""
import numpy as np
import pandas as pd
import pytest

import config as cfg
from src import data_loader as dl
from src.evaluate import classification_metrics, health_category, nasa_score, regression_metrics
from src.feature_engineering import engineer_features
from src.predict import load_bundle, make_features, predict_risk, predict_rul, validate_raw_frame
from src.preprocessing import Preprocessor, data_quality_report

pytestmark = pytest.mark.skipif(not dl.raw_file("train", "FD001").exists(), reason="NASA files not in dataset/raw")


@pytest.fixture(scope="module")
def fd001():
    return dl.add_train_rul(dl.load_train("FD001"))


@pytest.mark.parametrize("subset", list(cfg.SUBSETS))
def test_loading_shapes_and_truth(subset):
    tr, te, truth = dl.load_train(subset), dl.load_test(subset), dl.load_rul_truth(subset)
    assert list(tr.columns) == cfg.RAW_COLUMNS and tr.shape[1] == 26
    assert te[cfg.ID_COL].nunique() == len(truth)            # one truth value per test engine
    assert tr.isna().sum().sum() == 0


def test_expected_fd001_size(fd001):
    assert fd001.shape[0] == 20631 and fd001[cfg.ID_COL].nunique() == 100


def test_rul_targets(fd001):
    last = fd001.groupby(cfg.ID_COL)["rul"].min()
    assert (last == 0).all()                                  # RUL is 0 at the last cycle
    assert dl.cap_rul(fd001["rul"]).max() == cfg.RUL_CAP
    te = dl.add_test_rul(dl.load_test("FD001"), dl.load_rul_truth("FD001"))
    l = dl.last_cycle_rows(te)
    assert np.allclose(l.set_index(cfg.ID_COL)["rul"].to_numpy(), dl.load_rul_truth("FD001").to_numpy())


def test_data_quality(fd001):
    q = data_quality_report(fd001)
    assert q["missing_values"] == 0 and q["duplicate_rows"] == 0 and q["cycles_are_consecutive"]
    assert {"s1", "s5", "s10", "s16", "s18", "s19"} <= set(q["constant_columns"])


def test_preprocessor_drops_constants_and_is_fitted_on_train_only(fd001):
    pre = Preprocessor(1).fit(fd001, dl.cap_rul(fd001["rul"]))
    assert "s1" in pre.constant_sensors_ and "s2" in pre.sensors_
    z = pre.transform(fd001)
    assert np.allclose(z[pre.sensors_].mean(), 0, atol=1e-6)
    assert not z.isna().any().any()


def test_multicondition_normalisation():
    tr = dl.add_train_rul(dl.load_train("FD002"))
    pre = Preprocessor(6).fit(tr, dl.cap_rul(tr["rul"]))
    z = pre.transform(tr)
    assert z["cond"].nunique() == 6
    assert np.allclose(z.groupby("cond")[pre.sensors_].mean().abs().max().max(), 0, atol=1e-6)


def test_features_are_causal(fd001):
    """Features at cycle t must not change if all later cycles are deleted."""
    pre = Preprocessor(1).fit(fd001, dl.cap_rul(fd001["rul"]))
    eng = fd001[fd001[cfg.ID_COL] == 1]
    full = make_features(pre, eng)
    cut = make_features(pre, eng[eng[cfg.CYCLE_COL] <= 60])
    pd.testing.assert_frame_equal(full.iloc[:60].reset_index(drop=True), cut.reset_index(drop=True))
    assert not full.isna().any().any()


def test_metrics():
    assert nasa_score([10], [10]) == 0
    assert nasa_score([10], [20]) > nasa_score([10], [0])      # late prediction penalised more
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["RMSE"] == 0 and m["R2"] == 1
    c = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])
    assert c["TP"] == 1 and c["FN"] == 1 and c["Precision"] == 1
    assert [health_category(x) for x in (10, 50, 100)] == ["Critical", "Warning", "Healthy"]


def test_validate_raw_frame_rejects_bad_input():
    with pytest.raises(ValueError):
        validate_raw_frame(pd.DataFrame(np.zeros((3, 5))))


def test_quick_training_save_load_predict(tmp_path):
    """Train a small model into a temporary folder, load it back and predict."""
    from src import train
    summary = train.run_subset("FD001", quick=True, out_root=tmp_path)
    assert summary["test_best"]["official_RMSE"] < 30          # far better than the 43 of the mean baseline
    for f in ("FD001_rul_model.joblib", "FD001_risk_model.joblib"):
        assert (tmp_path / "models" / f).exists()
    b = load_bundle("FD001", "rul", tmp_path / "models")
    raw = dl.load_test("FD001")
    eng = raw[raw[cfg.ID_COL] == 1]
    out = predict_rul(b, eng)
    assert len(out) == len(eng) and out["rul_pred"].between(0, cfg.RUL_CAP).all()
    r = predict_risk(load_bundle("FD001", "risk", tmp_path / "models"), eng)
    assert r["risk_score"].between(0, 1).all()


def test_missing_model_message():
    with pytest.raises(FileNotFoundError, match="Train it first"):
        load_bundle("FD001", "rul", cfg.ROOT / "nonexistent")


@pytest.mark.parametrize("subset", list(cfg.SUBSETS))
def test_saved_model_reproduces_stored_predictions(subset):
    """The model file on disk must give exactly the predictions stored in outputs/."""
    if not (cfg.MODELS_DIR / f"{subset}_rul_model.joblib").exists():
        pytest.skip("model not trained")
    b = load_bundle(subset, "rul")
    te = dl.load_test(subset)
    units = te[cfg.ID_COL].unique()[:3]
    sub = te[te[cfg.ID_COL].isin(units)]
    out = predict_rul(b, sub)
    stored = pd.read_csv(cfg.PREDICTIONS_DIR / f"{subset}_test_predictions.csv.gz")
    stored = stored[stored[cfg.ID_COL].isin(units)][f"pred::{b['model_name']}"].to_numpy()
    assert np.allclose(out["rul_pred"].to_numpy(), stored, atol=1e-6)
