"""
recommender.py
===============
High-level recommendation summaries and the executive KPI system. This
module sits on top of optimization_engine.py and feature_engineering.py
to produce the numbers shown on the Executive Dashboard and
Recommendation Dashboard pages.
"""

from __future__ import annotations

import pandas as pd

from optimization_engine import simulate_all_factories_batch


def compute_executive_kpis(df: pd.DataFrame) -> dict:
    """All KPIs are computed directly from real or clearly-labeled-simulated
    columns already present in the feature table — no new assumptions here."""
    kpis = {
        "Total Sales": float(df["Sales"].sum()),
        "Total Profit": float(df["Gross Profit"].sum()),
        "Total Cost": float(df["Cost"].sum()),
        "Total Orders": int(df["Order ID"].nunique()),
        "Total Units": int(df["Units"].sum()),
        "Average Lead Time (days, simulated)": float(df["Simulated_Lead_Time_Days"].mean()),
        "Average Shipping Distance (km)": float(df["Shipping_Distance_KM"].mean()),
        "Average Profit Margin": float(df["Profit_Margin"].mean()),
        "Factory Efficiency (avg score)": float(df["Shipping_Efficiency_Score"].mean()),
        "Risk Index (avg score)": float(df["Operational_Risk_Score"].mean()),
        "Recommendation Confidence (avg)": float(df["Recommendation_Confidence"].mean()),
    }
    return kpis


def compute_factory_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-factory aggregate performance table for the Executive Dashboard."""
    summary = df.groupby("Assigned_Factory_Name").agg(
        Orders=("Order ID", "nunique"),
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Gross Profit", "sum"),
        Avg_Lead_Time=("Simulated_Lead_Time_Days", "mean"),
        Avg_Distance_KM=("Shipping_Distance_KM", "mean"),
        Avg_Efficiency_Score=("Shipping_Efficiency_Score", "mean"),
        Avg_Risk_Score=("Operational_Risk_Score", "mean"),
        Utilization_Pct=("Factory_Utilization_Score", "first"),
    ).round(2).reset_index()
    return summary.sort_values("Avg_Efficiency_Score", ascending=False)


def compute_regional_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("Region").agg(
        Orders=("Order ID", "nunique"),
        Total_Sales=("Sales", "sum"),
        Total_Profit=("Gross Profit", "sum"),
        Avg_Lead_Time=("Simulated_Lead_Time_Days", "mean"),
        Avg_Risk_Score=("Operational_Risk_Score", "mean"),
    ).round(2).reset_index()
    return summary.sort_values("Total_Sales", ascending=False)


def recommendation_coverage(df: pd.DataFrame, model_pipeline, sample_size: int | None = None) -> dict:
    """
    What share of orders are already on their optimal factory vs. would
    benefit from reassignment? Computed via the same vectorized batch engine.
    """
    work_df = df if sample_size is None else df.sample(n=min(sample_size, len(df)), random_state=42)
    batch = simulate_all_factories_batch(work_df, model_pipeline)
    best_idx = batch.groupby("_order_row")["Recommendation_Score"].idxmax()
    best = batch.loc[best_idx]
    current = batch[batch["Is_Current_Factory"]]

    merged = current.merge(
        best[["_order_row", "Candidate_Factory_ID"]], on="_order_row", suffixes=("_cur", "_best")
    )
    already_optimal = (merged["Candidate_Factory_ID_cur"] == merged["Candidate_Factory_ID_best"]).mean()

    return {
        "orders_analyzed": len(work_df),
        "pct_already_optimal": round(already_optimal * 100, 1),
        "pct_recommend_change": round((1 - already_optimal) * 100, 1),
    }
