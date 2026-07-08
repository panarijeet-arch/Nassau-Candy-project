"""Page 5 — Risk Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from theme import inject_global_css, hero, kpi_card, COLORS  # noqa: E402

st.set_page_config(page_title="Risk Dashboard | Nassau Candy", page_icon="⚠️", layout="wide")
inject_global_css()

if "features_df" not in st.session_state:
    st.error("Please open the app from the main page (app.py) first so data can load.")
    st.stop()

df: pd.DataFrame = st.session_state["features_df"]

hero(
    "Risk Dashboard",
    "Operational risk concentration across products, regions, and factories.",
    badges=["Monitoring", "Early Warning"],
)

avg_risk = df["Operational_Risk_Score"].mean()
avg_conf = df["Recommendation_Confidence"].mean()
high_risk_pct = (df["Operational_Risk_Score"] > 65).mean() * 100

c1, c2, c3 = st.columns(3)
with c1:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=avg_risk,
        title={"text": "Average Operational Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLORS["chocolate"]},
            "steps": [
                {"range": [0, 35], "color": COLORS["mint"]},
                {"range": [35, 65], "color": COLORS["caramel_light"]},
                {"range": [65, 100], "color": COLORS["cherry"]},
            ],
        },
    ))
    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    st.plotly_chart(fig_gauge, width="stretch")
kpi_card("High-Risk Order Share", f"{high_risk_pct:.1f}%", c2)
kpi_card("Avg. Recommendation Confidence", f"{avg_conf:.0f}/100", c3)

st.markdown("---")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("##### High-Risk Products")
    prod_risk = df.groupby("Product Name")["Operational_Risk_Score"].mean().sort_values(ascending=False).head(8).reset_index()
    fig1 = px.bar(prod_risk, x="Operational_Risk_Score", y="Product Name", orientation="h",
                  color_discrete_sequence=[COLORS["cherry"]])
    fig1.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white", paper_bgcolor="white",
                        yaxis=dict(autorange="reversed"), xaxis_title="Avg Risk Score")
    st.plotly_chart(fig1, width="stretch")

with col_b:
    st.markdown("##### High-Risk Regions")
    reg_risk = df.groupby("Region")["Operational_Risk_Score"].mean().sort_values(ascending=False).reset_index()
    fig2 = px.bar(reg_risk, x="Region", y="Operational_Risk_Score", color="Region",
                  color_discrete_sequence=[COLORS["cherry"], COLORS["caramel"], COLORS["caramel_light"], COLORS["mint"]])
    fig2.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
                        plot_bgcolor="white", paper_bgcolor="white", xaxis_title="", yaxis_title="Avg Risk Score")
    st.plotly_chart(fig2, width="stretch")

st.markdown("---")
st.markdown("##### Factory Bottlenecks (Utilization vs. Risk)")
fac_view = df.groupby("Assigned_Factory_Name").agg(
    Utilization=("Factory_Utilization_Score", "first"),
    Avg_Risk=("Operational_Risk_Score", "mean"),
    Orders=("Order ID", "nunique"),
).reset_index()
fig3 = px.scatter(
    fac_view, x="Utilization", y="Avg_Risk", size="Orders", color="Assigned_Factory_Name",
    text="Assigned_Factory_Name", color_discrete_sequence=[COLORS["caramel"], COLORS["cherry"], COLORS["mint"]],
    size_max=60,
)
fig3.update_traces(textposition="top center")
fig3.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis_title="Utilization (%)", yaxis_title="Avg Operational Risk Score")
st.plotly_chart(fig3, width="stretch")

st.markdown("---")
st.markdown("##### Profit Alerts")
low_margin = df[df["Profit_Margin"] < df["Profit_Margin"].quantile(0.10)]
if low_margin.empty:
    st.success("No orders currently fall in the bottom 10% profit margin band.")
else:
    st.warning(
        f"**{len(low_margin):,} orders** ({len(low_margin)/len(df)*100:.1f}% of filtered data) "
        f"fall in the bottom 10% profit margin band (below {df['Profit_Margin'].quantile(0.10)*100:.1f}% margin)."
    )
    st.dataframe(
        low_margin[["Order ID", "Product Name", "Region", "Sales", "Gross Profit", "Profit_Margin"]]
        .sort_values("Profit_Margin").head(20),
        width="stretch", hide_index=True,
    )
