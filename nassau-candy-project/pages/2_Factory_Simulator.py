"""Page 2 — Factory Optimization Simulator."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from theme import inject_global_css, hero, kpi_card, COLORS  # noqa: E402
from optimization_engine import simulate_all_factories  # noqa: E402

st.set_page_config(page_title="Factory Simulator | Nassau Candy", page_icon="🏭", layout="wide")
inject_global_css()

if "features_df" not in st.session_state:
    st.error("Please open the app from the main page (app.py) first so data can load.")
    st.stop()

df: pd.DataFrame = st.session_state["features_df"]
pipeline = st.session_state["model_pipeline"]
model_name = st.session_state["model_name"]

hero(
    "Factory Optimization Simulator",
    "Pick a product, region, and ship mode to see predicted lead time, profit, "
    "and risk under every candidate factory — powered by the trained "
    f"{model_name} model.",
    badges=["Interactive", "What-If"],
)

c1, c2, c3 = st.columns(3)
with c1:
    product = st.selectbox("Product", sorted(df["Product Name"].unique()))
with c2:
    region = st.selectbox("Region", sorted(df["Region"].unique()))
with c3:
    ship_mode = st.selectbox("Ship Mode", sorted(df["Ship Mode"].unique()))

subset = df[(df["Product Name"] == product) & (df["Region"] == region) & (df["Ship Mode"] == ship_mode)]

if subset.empty:
    st.warning(
        "No historical orders match this exact combination. Using the closest available "
        "order for this product to seed the simulation (other fields averaged)."
    )
    subset = df[df["Product Name"] == product]
    if subset.empty:
        st.error("No data available for this product.")
        st.stop()

base_row = subset.iloc[0].copy()
base_row["Region"] = region
base_row["Ship Mode"] = ship_mode

st.markdown("---")
sim_result = simulate_all_factories(base_row, pipeline)

best_row = sim_result.iloc[0]
current_row = sim_result[sim_result["Is_Current_Factory"]]
current_row = current_row.iloc[0] if not current_row.empty else best_row

st.markdown("#### Recommendation")
rcol1, rcol2, rcol3, rcol4 = st.columns(4)
kpi_card("Recommended Factory", best_row["Factory_Name"], rcol1)
kpi_card("Predicted Lead Time", f"{best_row['Predicted_Lead_Time_Days']:.1f} days", rcol2)
kpi_card("Predicted Profit", f"${best_row['Adjusted_Profit']:.2f}", rcol3)
kpi_card("Recommendation Score", f"{best_row['Recommendation_Score']:.0f}/100", rcol4)

st.markdown("---")
st.markdown("#### All Factory Candidates Compared")

display_cols = {
    "Factory_Name": "Factory",
    "Predicted_Lead_Time_Days": "Lead Time (days)",
    "Shipping_Distance_KM": "Distance (km)",
    "Adjusted_Profit": "Profit ($)",
    "Risk_Score": "Risk Score",
    "Recommendation_Score": "Rec. Score",
}
show_df = sim_result.rename(columns=display_cols)[list(display_cols.values()) + ["Is_Current_Factory"]]
show_df["Status"] = show_df["Is_Current_Factory"].map({True: "📍 Current", False: ""})
st.dataframe(
    show_df.drop(columns=["Is_Current_Factory"]).style.format({
        "Lead Time (days)": "{:.2f}", "Distance (km)": "{:.0f}",
        "Profit ($)": "${:.2f}", "Risk Score": "{:.0f}", "Rec. Score": "{:.0f}",
    }),
    width="stretch", hide_index=True,
)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=sim_result["Factory_Name"], y=sim_result["Recommendation_Score"],
    marker_color=[COLORS["caramel"] if not c else COLORS["cherry"] for c in sim_result["Is_Current_Factory"]],
    text=sim_result["Recommendation_Score"], texttemplate="%{text:.0f}", textposition="outside",
))
fig.update_layout(
    title="Recommendation Score by Factory (red = current assignment)",
    height=360, margin=dict(l=10, r=10, t=50, b=10),
    plot_bgcolor="white", paper_bgcolor="white", yaxis_title="Score (0-100)",
)
st.plotly_chart(fig, width="stretch")

st.markdown(
    """
    <div class="nc-assumption-box">
    <b>How this works:</b> The simulator recalculates shipping distance for each
    candidate factory using haversine geometry against the customer's
    state-centroid location, feeds the resulting feature set through the trained
    ML pipeline to predict lead time, and applies a transparent distance-based
    logistics-cost proxy to estimate profit impact. The Recommendation Score
    blends lead time, profit, and risk (see Optimization Engine in the README
    for the exact formula and weights).
    </div>
    """,
    unsafe_allow_html=True,
)
