"""Website tests with Streamlit's headless AppTest: every page renders without errors."""
import pytest
from streamlit.testing.v1 import AppTest

import config as cfg
from src import app_data

SUBSETS = app_data.available_subsets()
pytestmark = pytest.mark.skipif(not SUBSETS, reason="no trained models")

PAGES = ["views/home.py", "views/dashboard.py", "views/prediction.py", "views/analytics.py",
         "views/performance.py", "views/explainability.py", "views/about.py"]


def _app(subset=None):
    at = AppTest.from_file(str(cfg.ROOT / "app.py"), default_timeout=180)
    if subset:
        at.session_state["subset"] = subset
    return at.run()


def test_app_starts():
    at = _app()
    assert not at.exception


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = _app()
    at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("subset", SUBSETS)
def test_dataset_selector_all_subsets(subset):
    at = _app(subset)
    for page in ("views/dashboard.py", "views/performance.py"):
        at.switch_page(page).run()
        assert not at.exception, (subset, page, [e.value for e in at.exception])


def test_prediction_button_uses_saved_model():
    at = _app()
    at.switch_page("views/prediction.py").run()
    btn = [b for b in at.button if "Run prediction" in b.label][0]
    btn.click().run()
    assert not at.exception
    labels = [m.label if hasattr(m, "label") else "" for m in at.markdown]
    assert any("Predicted RUL" in str(m.value) for m in at.markdown)


def test_explainability_local_button():
    at = _app()
    at.switch_page("views/explainability.py").run()
    btn = [b for b in at.button if "Explain this prediction" in b.label][0]
    btn.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Average model output" in str(m.value) for m in at.markdown)
