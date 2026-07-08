"""
feature_engineering.py
=======================
Builds all engineered features used by the ML models, clustering, and
the optimization/recommendation engine.

Feature provenance is intentionally explicit in this file: every feature
is tagged as REAL (derived purely from real columns) or SIMULATED
(derived from an assumption — documented inline and surfaced in the app).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from factory_mapping import FACTORY_NETWORK, default_factory_for_division, factory_coords
from geo_lookup import get_coords

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# Ship-Mode -> realistic base lead time in days (SIMULATED — Ship Date column
# is corrupted per the data-quality report, so we simulate a believable lead
# time from Ship Mode, which is a real and trustworthy column).
SHIP_MODE_BASE_DAYS = {
    "Same Day": 0.5,
    "First Class": 1.8,
    "Second Class": 3.4,
    "Standard Class": 5.6,
}

RNG_SEED = 42


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach customer lat/lon (state centroid) and assigned factory lat/lon."""
    df = df.copy()
    coords = df["State/Province"].apply(get_coords)
    df["Customer_Lat"] = coords.apply(lambda t: t[0])
    df["Customer_Lon"] = coords.apply(lambda t: t[1])

    df["Assigned_Factory_ID"] = df["Division"].apply(default_factory_for_division)
    df["Assigned_Factory_Name"] = df["Assigned_Factory_ID"].map(
        lambda fid: FACTORY_NETWORK[fid]["name"]
    )
    fac_lat = df["Assigned_Factory_ID"].map(lambda fid: factory_coords(fid)[0])
    fac_lon = df["Assigned_Factory_ID"].map(lambda fid: factory_coords(fid)[1])
    df["Factory_Lat"] = fac_lat
    df["Factory_Lon"] = fac_lon

    # Distance feature -- REAL geometry (haversine) applied to approximate
    # (state-centroid) coordinates. Falls back to NaN if geocoding failed.
    valid = df["Customer_Lat"].notna() & df["Factory_Lat"].notna()
    df["Shipping_Distance_KM"] = np.nan
    df.loc[valid, "Shipping_Distance_KM"] = haversine_km(
        df.loc[valid, "Customer_Lat"], df.loc[valid, "Customer_Lon"],
        df.loc[valid, "Factory_Lat"], df.loc[valid, "Factory_Lon"],
    )
    # Median-impute any remaining gaps so downstream models never see NaN
    df["Shipping_Distance_KM"] = df["Shipping_Distance_KM"].fillna(
        df["Shipping_Distance_KM"].median()
    )
    return df


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Sales_per_Unit"] = df["Sales"] / df["Units"]
    df["Profit_Margin"] = df["Gross Profit"] / df["Sales"]
    df["Cost_Ratio"] = df["Cost"] / df["Sales"]
    df["Profit_per_Unit"] = df["Gross Profit"] / df["Units"]
    return df


def add_simulated_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    SIMULATED FEATURE — clearly labeled.
    Because Ship Date is corrupted (see data-quality report), we simulate a
    plausible lead time driven by the real Ship Mode column, plus a small
    distance-driven and reproducibly random component, rather than fabricate
    false precision. The seed is fixed so the app is fully reproducible.
    """
    df = df.copy()
    rng = np.random.default_rng(RNG_SEED)

    base = df["Ship Mode"].map(SHIP_MODE_BASE_DAYS).fillna(4.0)
    # Distance adds realistic friction: +1 day per ~1500km, capped
    distance_component = np.clip(df["Shipping_Distance_KM"] / 1500.0, 0, 4)
    noise = rng.normal(loc=0, scale=0.6, size=len(df))

    sim_lead = base + distance_component + noise
    df["Simulated_Lead_Time_Days"] = np.clip(sim_lead, 0.25, None).round(2)
    return df


def add_demand_and_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Regional demand: total units sold per region, broadcast back to rows
    region_units = df.groupby("Region")["Units"].transform("sum")
    df["Regional_Demand"] = region_units

    # Product popularity: total units sold per product
    product_units = df.groupby("Product ID")["Units"].transform("sum")
    df["Product_Popularity"] = product_units

    # Average factory performance proxy: mean simulated lead time per factory
    fac_perf = df.groupby("Assigned_Factory_ID")["Simulated_Lead_Time_Days"].transform("mean")
    df["Avg_Factory_Lead_Time"] = fac_perf

    # Average region performance: mean simulated lead time per region
    region_perf = df.groupby("Region")["Simulated_Lead_Time_Days"].transform("mean")
    df["Avg_Region_Lead_Time"] = region_perf

    # Average ship-mode delay: mean lead time per ship mode
    mode_perf = df.groupby("Ship Mode")["Simulated_Lead_Time_Days"].transform("mean")
    df["Avg_ShipMode_Lead_Time"] = mode_perf

    # Factory utilization score: this factory's order volume share vs network total
    fac_orders = df.groupby("Assigned_Factory_ID")["Units"].transform("sum")
    total_orders = df["Units"].sum()
    df["Factory_Utilization_Score"] = (fac_orders / total_orders * 100).round(2)

    return df


def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Business composite scores (0-100 scale where useful) combining multiple
    normalized signals. These are interpretable, documented formulas — not
    black-box scores — so they can be explained to a business stakeholder.
    """
    df = df.copy()

    def norm(s: pd.Series, invert: bool = False) -> pd.Series:
        rng = s.max() - s.min()
        if rng == 0:
            out = pd.Series(50.0, index=s.index)
        else:
            out = (s - s.min()) / rng * 100
        return 100 - out if invert else out

    # Shipping efficiency: fast + short distance = high score
    df["Shipping_Efficiency_Score"] = (
        0.6 * norm(df["Simulated_Lead_Time_Days"], invert=True)
        + 0.4 * norm(df["Shipping_Distance_KM"], invert=True)
    ).round(2)

    # Profitability score: higher margin & profit-per-unit = higher score
    df["Profitability_Score"] = (
        0.5 * norm(df["Profit_Margin"])
        + 0.5 * norm(df["Profit_per_Unit"])
    ).round(2)

    # Route congestion score: proxy from regional demand intensity (higher demand
    # relative to factory capacity = more congestion risk)
    fac_capacity = df["Assigned_Factory_ID"].map(
        lambda fid: FACTORY_NETWORK[fid]["base_capacity_units_per_day"]
    )
    df["Route_Congestion_Score"] = norm(df["Regional_Demand"] / fac_capacity).round(2)

    # Operational risk score: blends congestion + lead time variability + distance
    df["Operational_Risk_Score"] = (
        0.4 * df["Route_Congestion_Score"]
        + 0.35 * norm(df["Simulated_Lead_Time_Days"])
        + 0.25 * norm(df["Shipping_Distance_KM"])
    ).round(2)

    # Recommendation confidence: inverse of risk, blended with data completeness
    # (always 100 here since this dataset has no missing values, but kept as a
    # real formula so it generalizes to messier data)
    completeness = 100 - df[["Sales", "Units", "Gross Profit", "Cost"]].isna().mean(axis=1) * 100
    df["Recommendation_Confidence"] = (
        0.7 * (100 - df["Operational_Risk_Score"]) + 0.3 * completeness
    ).round(2)

    return df


def build_feature_table(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline in the correct order."""
    df = df_clean.copy()
    df = add_geo_features(df)
    df = add_financial_features(df)
    df = add_simulated_lead_time(df)
    df = add_demand_and_performance_features(df)
    df = add_composite_scores(df)
    logger.info("Feature engineering complete: %s columns", df.shape[1])
    return df


FEATURE_COLUMNS_FOR_ML = [
    "Sales", "Units", "Cost", "Sales_per_Unit", "Profit_Margin", "Cost_Ratio",
    "Shipping_Distance_KM", "Regional_Demand", "Product_Popularity",
    "Factory_Utilization_Score", "Route_Congestion_Score",
]

CATEGORICAL_COLUMNS_FOR_ML = ["Division", "Region", "Ship Mode", "Assigned_Factory_ID"]
