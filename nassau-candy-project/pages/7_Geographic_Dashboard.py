"""Page 7 — Geographical Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from theme import inject_global_css, hero, COLORS, rgba  # noqa: E402
from factory_mapping import FACTORY_NETWORK  # noqa: E402

st.set_page_config(page_title="Geographic Dashboard | Nassau Candy", page_icon="🗺️", layout="wide")
inject_global_css()

if "features_df" not in st.session_state:
    st.error("Please open the app from the main page (app.py) first so data can load.")
    st.stop()

df: pd.DataFrame = st.session_state["features_df"]

hero(
    "Geographic Dashboard",
    "Factories, customer demand concentration, and shipping routes.",
    badges=["State-Level Geocoding", "Route Map"],
)

st.markdown(
    """
    <div class="nc-assumption-box">
    <b>Geocoding note:</b> Customer locations are approximated at the
    State/Province level using public geographic centroids (616 distinct
    cities map to 59 states/provinces). This is appropriate for regional
    visualization but is not precise city-level geocoding.
    </div>
    """,
    unsafe_allow_html=True,
)

state_summary = df.groupby(["State/Province", "Customer_Lat", "Customer_Lon"]).agg(
    Orders=("Order ID", "nunique"), Sales=("Sales", "sum"), Avg_Risk=("Operational_Risk_Score", "mean"),
).reset_index()

fig = go.Figure()

# Customer demand bubbles
fig.add_trace(go.Scattergeo(
    lon=state_summary["Customer_Lon"], lat=state_summary["Customer_Lat"],
    text=state_summary["State/Province"] + "<br>Orders: " + state_summary["Orders"].astype(str)
         + "<br>Sales: $" + state_summary["Sales"].round(0).astype(str),
    marker=dict(
        size=state_summary["Sales"] / state_summary["Sales"].max() * 40 + 6,
        color=state_summary["Avg_Risk"], colorscale=[[0, COLORS["mint"]], [0.5, COLORS["caramel"]], [1, COLORS["cherry"]]],
        line=dict(width=0.5, color="white"), showscale=True, colorbar=dict(title="Avg Risk"),
    ),
    name="Customer Demand",
))

# Factories
fac_lats = [info["lat"] for info in FACTORY_NETWORK.values()]
fac_lons = [info["lon"] for info in FACTORY_NETWORK.values()]
fac_names = [info["name"] for info in FACTORY_NETWORK.values()]
fig.add_trace(go.Scattergeo(
    lon=fac_lons, lat=fac_lats, text=fac_names,
    marker=dict(size=22, color=COLORS["chocolate"], symbol="square", line=dict(width=2, color="white")),
    name="Factories",
))

# Routes (sampled for clarity)
route_sample = df.groupby(["Customer_Lat", "Customer_Lon", "Factory_Lat", "Factory_Lon"]).size().reset_index(name="n")
route_sample = route_sample.nlargest(40, "n")
for _, r in route_sample.iterrows():
    fig.add_trace(go.Scattergeo(
        lon=[r["Customer_Lon"], r["Factory_Lon"]], lat=[r["Customer_Lat"], r["Factory_Lat"]],
        mode="lines", line=dict(width=max(0.5, min(r["n"] / 30, 3)), color=rgba(COLORS['caramel'], 0.47)),
        showlegend=False, hoverinfo="skip",
    ))

fig.update_geos(scope="north america", showland=True, landcolor="#F5F0E8", showocean=True,
                 oceancolor="#EAF2F0", showlakes=True, lakecolor="#EAF2F0", showcountries=True,
                 countrycolor="#D8CCB8")
fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0),
                   legend=dict(orientation="h", y=0.02, x=0.02))
st.plotly_chart(fig, width="stretch")

st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("##### Top States by Order Volume")
    st.dataframe(
        state_summary.sort_values("Orders", ascending=False).head(10)[["State/Province", "Orders", "Sales"]],
        width="stretch", hide_index=True,
    )
with col_b:
    st.markdown("##### Factory Coverage")
    fac_cov = df.groupby("Assigned_Factory_Name").agg(
        States_Served=("State/Province", "nunique"), Orders=("Order ID", "nunique"),
        Avg_Distance_KM=("Shipping_Distance_KM", "mean"),
    ).round(1).reset_index()
    st.dataframe(fac_cov, width="stretch", hide_index=True)
