"""
optimization_engine.py
=======================
Simulates reassigning each order/product to every alternative factory in
the network, predicts the resulting lead time using the trained model,
and scores each alternative on a weighted multi-objective function:

    minimize lead time, maximize profit, minimize risk, maximize efficiency

This produces ranked recommendations comparing the current factory
assignment against every alternative.

PERFORMANCE NOTE: the simulation is fully vectorized (no per-row Python
loops or per-row model.predict calls). Every order is exploded against
every candidate factory into one large batch, scored with a single
model.predict() call, then regrouped — this scales to the full dataset
in well under a second instead of minutes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from factory_mapping import ALL_FACTORY_IDS, FACTORY_NETWORK, factory_coords
from feature_engineering import haversine_km, FEATURE_COLUMNS_FOR_ML, CATEGORICAL_COLUMNS_FOR_ML

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "lead_time": 0.30,    # minimize
    "profit": 0.30,       # maximize
    "risk": 0.20,         # minimize
    "efficiency": 0.20,   # maximize
}

DISTANCE_COST_PER_KM = 0.0008  # illustrative logistics-cost proxy, $/km


def _normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    rng = series.max() - series.min()
    if rng == 0 or pd.isna(rng):
        out = pd.Series(50.0, index=series.index)
    else:
        out = (series - series.min()) / rng * 100
    return 100 - out if invert else out


def _build_candidate_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode every order row into one row per candidate factory.
    Vectorized — no Python-level row loop.
    """
    base_cols = list(set(
        FEATURE_COLUMNS_FOR_ML + CATEGORICAL_COLUMNS_FOR_ML
        + ["Order ID", "Product Name", "Region", "Customer_Lat", "Customer_Lon",
           "Gross Profit", "Sales", "Assigned_Factory_ID"]
    ))
    base = df[base_cols].reset_index(drop=True)
    base["_order_row"] = base.index

    candidates = pd.concat([base.assign(Candidate_Factory_ID=fid) for fid in ALL_FACTORY_IDS],
                            ignore_index=True)

    fac_lat = candidates["Candidate_Factory_ID"].map(lambda f: factory_coords(f)[0])
    fac_lon = candidates["Candidate_Factory_ID"].map(lambda f: factory_coords(f)[1])
    candidates["Shipping_Distance_KM"] = haversine_km(
        candidates["Customer_Lat"], candidates["Customer_Lon"], fac_lat, fac_lon
    )
    # The model was trained using "Assigned_Factory_ID" as the categorical
    # factory feature — overwrite it with the CANDIDATE factory for this
    # simulated row so the model scores the hypothetical assignment.
    candidates["Assigned_Factory_ID_actual"] = candidates["Assigned_Factory_ID"]
    candidates["Assigned_Factory_ID"] = candidates["Candidate_Factory_ID"]

    candidates["Is_Current_Factory"] = (
        candidates["Candidate_Factory_ID"] == candidates["Assigned_Factory_ID_actual"]
    )
    candidates["Factory_Name"] = candidates["Candidate_Factory_ID"].map(
        lambda f: FACTORY_NETWORK[f]["name"]
    )
    return candidates


def simulate_all_factories_batch(df: pd.DataFrame, model_pipeline) -> pd.DataFrame:
    """
    Vectorized simulation: for every order in df, predict lead time under
    every candidate factory assignment. Returns a long-format DataFrame
    with one row per (order, candidate factory).
    """
    candidates = _build_candidate_matrix(df)

    X = candidates[FEATURE_COLUMNS_FOR_ML + CATEGORICAL_COLUMNS_FOR_ML]
    candidates["Predicted_Lead_Time_Days"] = model_pipeline.predict(X).round(2)

    distance_cost = candidates["Shipping_Distance_KM"] * DISTANCE_COST_PER_KM
    candidates["Adjusted_Profit"] = (candidates["Gross Profit"] - distance_cost).round(2)
    candidates["Adjusted_Margin"] = (candidates["Adjusted_Profit"] / candidates["Sales"]).round(4)

    # Scores computed PER ORDER (group-wise normalization across that
    # order's 3 candidate factories), fully vectorized via groupby-transform.
    grp = candidates.groupby("_order_row")

    def _norm_grp(s: pd.Series, invert: bool) -> pd.Series:
        gmin = grp[s.name].transform("min")
        gmax = grp[s.name].transform("max")
        rng = (gmax - gmin).replace(0, np.nan)
        out = (s - gmin) / rng * 100
        out = out.fillna(50.0)
        return 100 - out if invert else out

    candidates["Efficiency_Score"] = _norm_grp(candidates["Predicted_Lead_Time_Days"], invert=True)
    risk_raw = candidates["Predicted_Lead_Time_Days"] * 0.5 + candidates["Shipping_Distance_KM"] * 0.5
    candidates["_risk_raw"] = risk_raw
    candidates["Risk_Score"] = _norm_grp(candidates["_risk_raw"], invert=False)
    candidates["Profit_Score"] = _norm_grp(candidates["Adjusted_Profit"], invert=False)

    w = DEFAULT_WEIGHTS
    candidates["Recommendation_Score"] = (
        w["lead_time"] * candidates["Efficiency_Score"]
        + w["profit"] * candidates["Profit_Score"]
        + w["risk"] * (100 - candidates["Risk_Score"])
        + w["efficiency"] * candidates["Efficiency_Score"]
    ).round(2)

    return candidates


def simulate_all_factories(row: pd.Series, model_pipeline) -> pd.DataFrame:
    """
    Single-order convenience wrapper around the batch engine, used by the
    interactive Streamlit simulator page (one order at a time).
    """
    single_df = pd.DataFrame([row])
    batch = simulate_all_factories_batch(single_df, model_pipeline)
    out = batch.rename(columns={"Candidate_Factory_ID": "Factory_ID"})[
        ["Factory_ID", "Factory_Name", "Predicted_Lead_Time_Days", "Shipping_Distance_KM",
         "Adjusted_Profit", "Adjusted_Margin", "Is_Current_Factory",
         "Efficiency_Score", "Risk_Score", "Profit_Score", "Recommendation_Score"]
    ]
    return out.sort_values("Recommendation_Score", ascending=False).reset_index(drop=True)


def generate_top_recommendations(
    df: pd.DataFrame, model_pipeline, sample_size: int | None = 2000, top_n: int = 5
) -> pd.DataFrame:
    """
    Run the vectorized simulation across many orders (sampled for very large
    datasets, though the full dataset runs in well under a second) and
    return the top N highest-value reallocation recommendations — i.e.
    cases where switching factories materially improves the score.
    """
    work_df = df if sample_size is None else df.sample(
        n=min(sample_size, len(df)), random_state=42
    ).reset_index(drop=True)

    batch = simulate_all_factories_batch(work_df, model_pipeline)

    current = batch[batch["Is_Current_Factory"]].set_index("_order_row")
    best_idx = batch.groupby("_order_row")["Recommendation_Score"].idxmax()
    best = batch.loc[best_idx].set_index("_order_row")

    joined = current.join(best, lsuffix="_cur", rsuffix="_best")
    changed = joined[joined["Candidate_Factory_ID_cur"] != joined["Candidate_Factory_ID_best"]].copy()

    if changed.empty:
        return pd.DataFrame()

    changed["Lead_Time_Improvement_Days"] = (
        changed["Predicted_Lead_Time_Days_cur"] - changed["Predicted_Lead_Time_Days_best"]
    ).round(2)
    changed["Score_Improvement_Points"] = (
        changed["Recommendation_Score_best"] - changed["Recommendation_Score_cur"]
    ).round(1)
    changed["Expected_Profit_Impact"] = (
        changed["Adjusted_Profit_best"] - changed["Adjusted_Profit_cur"]
    ).round(2)
    changed["Risk_Level"] = pd.cut(
        changed["Risk_Score_best"], bins=[-1, 35, 65, 101], labels=["Low", "Medium", "High"]
    )
    changed["Business_Explanation"] = changed.apply(
        lambda r: (
            f"Reassigning to {r['Factory_Name_best']} reduces predicted lead time by "
            f"{r['Lead_Time_Improvement_Days']:.1f} days and changes profit by "
            f"${r['Expected_Profit_Impact']:+.2f}, raising the overall recommendation "
            f"score by {r['Score_Improvement_Points']:.1f} points (0-100 scale)."
        ),
        axis=1,
    )

    out = changed.rename(columns={
        "Order ID_cur": "Order ID",
        "Product Name_cur": "Product Name",
        "Region_cur": "Region",
        "Factory_Name_cur": "Current_Factory",
        "Factory_Name_best": "Recommended_Factory",
        "Predicted_Lead_Time_Days_best": "Expected_Lead_Time_Days",
        "Recommendation_Score_best": "Confidence_Score",
    })[[
        "Order ID", "Product Name", "Region", "Current_Factory", "Recommended_Factory",
        "Expected_Lead_Time_Days", "Lead_Time_Improvement_Days", "Score_Improvement_Points",
        "Expected_Profit_Impact", "Confidence_Score", "Risk_Level", "Business_Explanation",
    ]].reset_index(drop=True)

    out = out.sort_values("Score_Improvement_Points", ascending=False)
    return out.head(top_n) if top_n else out
