"""Model zoo: every supervised algorithm used in the project (regression + classification)."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor, StackingRegressor)
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression, Perceptron,
                                  RANSACRegressor, Ridge)
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import config as cfg

RS = cfg.RANDOM_STATE


class Subsampled(BaseEstimator):
    """Wrap a slow O(n^2) model (SVR / SVC) so it trains on a random subset of rows.

    Kernel SVMs become very slow beyond ~10,000 rows, so we fit them on a random
    sample (documented in the README) and predict for all rows."""

    def __init__(self, estimator=None, max_rows: int = 6000, random_state: int = RS):
        self.estimator = estimator
        self.max_rows = max_rows
        self.random_state = random_state

    def fit(self, X, y):
        n = len(X)
        idx = np.arange(n)
        if n > self.max_rows:
            idx = np.sort(np.random.RandomState(self.random_state).choice(n, self.max_rows, replace=False))
        Xs = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        self.estimator_ = clone(self.estimator).fit(Xs, np.asarray(y)[idx])
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def decision_function(self, X):
        return self.estimator_.decision_function(X)


def _scaled(model):
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def regression_models(base_cols: list[str]) -> dict:
    """name -> (estimator, family).  `base_cols` = current-value sensor columns."""
    poly = Pipeline([
        ("poly", ColumnTransformer([("p", PolynomialFeatures(2, include_bias=False), base_cols)],
                                   remainder="drop")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ])
    stack = StackingRegressor(
        estimators=[("rf", RandomForestRegressor(60, max_depth=14, min_samples_leaf=5,
                                                 max_features=0.5, n_jobs=-1, random_state=RS)),
                    ("hgb", HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=RS)),
                    ("ridge", _scaled(Ridge(alpha=10.0)))],
        final_estimator=LinearRegression(),
        # un-shuffled folds on data sorted by engine = (almost) engine-wise folds
        cv=KFold(3, shuffle=False), n_jobs=1)
    return {
        "Baseline (mean RUL)": (DummyRegressor(strategy="mean"), "baseline"),
        "Linear Regression": (_scaled(LinearRegression()), "linear"),
        "Polynomial Regression (deg 2)": (poly, "linear"),
        "Ridge": (_scaled(Ridge(alpha=10.0)), "linear"),
        "Lasso": (_scaled(Lasso(alpha=0.01, max_iter=5000)), "linear"),
        "RANSAC": (_scaled(RANSACRegressor(LinearRegression(), min_samples=0.5, residual_threshold=25.0,
                                                 max_trials=50, random_state=RS)), "linear"),
        "Decision Tree": (DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=RS), "tree"),
        "SVR (RBF)": (Subsampled(_scaled(SVR(C=50.0, epsilon=2.0, gamma="scale")), 6000), "kernel"),
        "Bagging (Decision Trees)": (BaggingRegressor(DecisionTreeRegressor(max_depth=12, min_samples_leaf=5),
                                                      n_estimators=40, max_samples=0.5, n_jobs=-1,
                                                      random_state=RS), "bagging"),
        "Random Forest": (RandomForestRegressor(100, max_depth=14, min_samples_leaf=5, max_features=0.5,
                                                n_jobs=-1, random_state=RS), "bagging"),
        "AdaBoost": (AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=60,
                                       learning_rate=0.5, random_state=RS), "boosting"),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=3,
                                                        subsample=0.7, random_state=RS), "boosting"),
        "HistGradientBoosting": (HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08,
                                                               random_state=RS), "boosting"),
        "Stacking (RF+HGB+Ridge)": (stack, "stacking"),
    }


# Hyper-parameter search spaces (see src/train.py)
GRID_SEARCHES = {   # exhaustive GridSearchCV
    "Ridge": ({"model__alpha": [0.1, 1, 10, 100, 1000]}, "Ridge"),
    "Decision Tree": ({"max_depth": [4, 6, 8, 12], "min_samples_leaf": [5, 20, 50]}, "Decision Tree"),
    "SVR (RBF)": ({"estimator__model__C": [10, 50, 200], "estimator__model__gamma": ["scale", 0.01]},
                  "SVR (RBF)"),
}
RANDOM_SEARCHES = {  # RandomizedSearchCV
    "HistGradientBoosting": ({"learning_rate": [0.03, 0.05, 0.08, 0.12], "max_iter": [150, 250, 400],
                              "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [20, 50, 100],
                              "l2_regularization": [0.0, 1.0, 10.0]}, "HistGradientBoosting", 10),
    "Random Forest": ({"max_depth": [10, 14, 18], "min_samples_leaf": [3, 5, 10, 20],
                       "max_features": [0.3, 0.5, 0.8]}, "Random Forest", 5),
}


def classification_models() -> dict:
    return {
        "Perceptron": _scaled(Perceptron(class_weight="balanced", random_state=RS)),
        "Logistic Regression": _scaled(LogisticRegression(class_weight="balanced", max_iter=2000)),
        "K-Nearest Neighbors": _scaled(KNeighborsClassifier(15, n_jobs=-1)),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                                class_weight="balanced", random_state=RS),
        "SVM (RBF)": Subsampled(_scaled(SVC(C=5.0, class_weight="balanced")), 6000),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(100, max_depth=14, min_samples_leaf=5, max_features=0.5,
                                                class_weight="balanced", n_jobs=-1, random_state=RS),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                                               class_weight="balanced", random_state=RS),
    }


def is_tree_model(est) -> bool:
    """True if SHAP's fast TreeExplainer supports the estimator."""
    return isinstance(est, (DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor,
                            HistGradientBoostingRegressor))
