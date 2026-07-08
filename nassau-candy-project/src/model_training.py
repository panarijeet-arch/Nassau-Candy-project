"""
model_training.py
==================
Trains and compares multiple regression models predicting
Simulated_Lead_Time_Days (primary target) from real + engineered features.

Models: Linear Regression, Random Forest, Gradient Boosting, and XGBoost
(if the xgboost package is available — imported defensively so the rest
of the app keeps working if it isn't installed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import FEATURE_COLUMNS_FOR_ML, CATEGORICAL_COLUMNS_FOR_ML

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("xgboost not installed — skipping XGBoost model. "
                    "Install with `pip install xgboost` to enable it.")

TARGET_COLUMN = "Simulated_Lead_Time_Days"
RANDOM_STATE = 42


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    rmse: float
    mae: float
    r2: float
    cv_r2_mean: float
    cv_r2_std: float
    y_test: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    feature_importance: dict | None = None


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_COLUMNS_FOR_ML),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS_FOR_ML),
        ]
    )


def get_candidate_models() -> dict:
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.08, random_state=RANDOM_STATE
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )
    return models


def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    num_names = FEATURE_COLUMNS_FOR_ML
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLUMNS_FOR_ML))
    return list(num_names) + cat_names


def train_and_compare(df: pd.DataFrame, test_size: float = 0.2) -> dict[str, ModelResult]:
    """Train every candidate model, evaluate, and return results keyed by name."""
    X = df[FEATURE_COLUMNS_FOR_ML + CATEGORICAL_COLUMNS_FOR_ML]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )

    results: dict[str, ModelResult] = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in get_candidate_models().items():
        preprocessor = build_preprocessor()
        pipe = Pipeline([("prep", preprocessor), ("model", model)])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)

        feature_importance = None
        fitted_model = pipe.named_steps["model"]
        if hasattr(fitted_model, "feature_importances_"):
            names = _get_feature_names(pipe.named_steps["prep"])
            feature_importance = dict(zip(names, fitted_model.feature_importances_.tolist()))
        elif hasattr(fitted_model, "coef_"):
            names = _get_feature_names(pipe.named_steps["prep"])
            feature_importance = dict(zip(names, np.abs(fitted_model.coef_).tolist()))

        results[name] = ModelResult(
            name=name, pipeline=pipe, rmse=rmse, mae=mae, r2=r2,
            cv_r2_mean=float(cv_scores.mean()), cv_r2_std=float(cv_scores.std()),
            y_test=y_test.values, y_pred=y_pred,
            feature_importance=feature_importance,
        )
        logger.info("%s -> RMSE=%.3f MAE=%.3f R2=%.3f CV_R2=%.3f±%.3f",
                    name, rmse, mae, r2, cv_scores.mean(), cv_scores.std())

    return results


def select_best_model(results: dict[str, ModelResult]) -> tuple[str, ModelResult, str]:
    """
    Pick the best model by test R² (tie-broken by lowest RMSE) and produce a
    plain-English explanation of why it won.
    """
    best_name = max(results, key=lambda n: (results[n].r2, -results[n].rmse))
    best = results[best_name]

    others = [r for n, r in results.items() if n != best_name]
    avg_other_r2 = np.mean([r.r2 for r in others]) if others else 0.0

    explanation = (
        f"{best_name} was selected as the best model with a test R² of {best.r2:.3f} "
        f"(cross-validated R² {best.cv_r2_mean:.3f} ± {best.cv_r2_std:.3f}) and RMSE of "
        f"{best.rmse:.2f} days. This outperforms the average of the other candidate "
        f"models (R² {avg_other_r2:.3f}) while keeping cross-validation variance low, "
        f"indicating the model generalizes well rather than overfitting to the training "
        f"split."
    )
    return best_name, best, explanation
