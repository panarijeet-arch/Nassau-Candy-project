"""
utils.py
========
Shared utilities: model persistence (save/load trained pipeline so the
Streamlit app doesn't retrain on every run), logging setup, and small
formatting helpers used across dashboard pages.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def save_pipeline(pipeline, name: str = "best_model") -> Path:
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Saved pipeline to %s", path)
    return path


def load_pipeline(name: str = "best_model"):
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}%"


def format_days(value: float) -> str:
    return f"{value:.1f} days"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def apply_filters(
    df: pd.DataFrame,
    region: list[str] | None = None,
    division: list[str] | None = None,
    product: list[str] | None = None,
    factory: list[str] | None = None,
    ship_mode: list[str] | None = None,
) -> pd.DataFrame:
    """Apply the sidebar filter selections. Any None/empty filter is skipped."""
    out = df
    if region:
        out = out[out["Region"].isin(region)]
    if division:
        out = out[out["Division"].isin(division)]
    if product:
        out = out[out["Product Name"].isin(product)]
    if factory:
        out = out[out["Assigned_Factory_Name"].isin(factory)]
    if ship_mode:
        out = out[out["Ship Mode"].isin(ship_mode)]
    return out
