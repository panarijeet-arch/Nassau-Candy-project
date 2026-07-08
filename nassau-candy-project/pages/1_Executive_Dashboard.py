"""Page 1 — Executive Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from theme import inject_global_css, hero, kpi_card, COLORS, PLOTLY_TEMPLATE_COLORWAY, rgba  # noqa: E402
from utils import apply_filters  # noqa: E402
from recommender import compute_executive_kpis, compute_factory_summary, compute_regional_summary  # noqa: E402

st.set_page_config(page_title="Executive Dashboard | Nassau Candy", page_icon="📊", layout="wide")
inject_global_css()

if "features_df" not in st.session_state:
    st.error("Please open the app from the main page (app.py) first so data can load.")
    st.stop()

df_all: pd.DataFrame = st.session_state["features_df"]

hero(
    "Executive Dashboard",
    "Sales, profit, lead time, and factory performance at a glance.",
    badges=["Live Filters", "KPI System"],
)

# ---- Sidebar filters ----
with st.sidebar:
    st.markdown("#### Filters")
    region_f = st.multiselect("Region", sorted(df_all["Region"].unique()))
    division_f = st.multiselect("Division", sorted(df_all["Division"].unique()))
    shipmode_f = st.multiselect("Ship Mode", sorted(df_all["Ship Mode"].unique()))
    factory_f = st.multiselect("Factory", sorted(df_all["Assigned_Factory_Name"].unique()))

df = apply_filters(df_all, region=region_f, division=division_f, factory=factory_f, ship_mode=shipmode_f)
if df.empty:
    st.warning("No orders match the selected filters.")
    st.stop()

kpis = compute_executive_kpis(df)

st.markdown("#### Key Performance Indicators")
c1, c2, c3, c4, c5 = st.columns(5)
kpi_card("Total Sales", f"${kpis['Total Sales']:,.0f}", c1)
kpi_card("Total Profit", f"${kpis['Total Profit']:,.0f}", c2)
kpi_card("Total Orders", f"{kpis['Total Orders']:,}", c3)
kpi_card("Avg Lead Time*", f"{kpis['Average Lead Time (days, simulated)']:.1f} d", c4)
kpi_card("Avg Margin", f"{kpis['Average Profit Margin']*100:.1f}%", c5)

c6, c7, c8, c9, c10 = st.columns(5)
kpi_card("Total Cost", f"${kpis['Total Cost']:,.0f}", c6)
kpi_card("Total Units", f"{kpis['Total Units']:,}", c7)
kpi_card("Avg Distance*", f"{kpis['Average Shipping Distance (km)']:.0f} km", c8)
kpi_card("Efficiency Score", f"{kpis['Factory Efficiency (avg score)']:.0f}/100", c9)
kpi_card("Risk Index", f"{kpis['Risk Index (avg score)']:.0f}/100", c10)

st.caption("*Lead time and distance are derived metrics — see sidebar note on the home page for details.")

st.markdown("---")

# ---- Trends ----
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("##### Sales & Profit Trend (by Order Date)")
    trend = df.groupby(df["Order Date"].dt.to_period("M")).agg(
        Sales=("Sales", "sum"), Profit=("Gross Profit", "sum")
    ).reset_index()
    trend["Order Date"] = trend["Order Date"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["Order Date"], y=trend["Sales"], name="Sales",
                              line=dict(color=COLORS["caramel"], width=3), fill="tozeroy",
                              fillcolor=rgba(COLORS['caramel'], 0.13)))
    fig.add_trace(go.Scatter(x=trend["Order Date"], y=trend["Profit"], name="Profit",
                              line=dict(color=COLORS["mint"], width=3)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                       plot_bgcolor="white", paper_bgcolor="white",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

with col_b:
    st.markdown("##### Simulated Lead Time Trend")
    lt_trend = df.groupby(df["Order Date"].dt.to_period("M"))["Simulated_Lead_Time_Days"].mean().reset_index()
    lt_trend["Order Date"] = lt_trend["Order Date"].astype(str)
    fig2 = px.line(lt_trend, x="Order Date", y="Simulated_Lead_Time_Days",
                   color_discrete_sequence=[COLORS["cherry"]])
    fig2.update_traces(line_width=3)
    fig2.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                        plot_bgcolor="white", paper_bgcolor="white", yaxis_title="Days")
    st.plotly_chart(fig2, width="stretch")

col_c, col_d = st.columns(2)

with col_c:
    st.markdown("##### Factory Performance")
    fac_summary = compute_factory_summary(df)
    fig3 = px.bar(fac_summary, x="Assigned_Factory_Name", y="Avg_Efficiency_Score",
                   color="Assigned_Factory_Name", color_discrete_sequence=PLOTLY_TEMPLATE_COLORWAY,
                   text="Avg_Efficiency_Score")
    fig3.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig3.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
                        plot_bgcolor="white", paper_bgcolor="white", xaxis_title="", yaxis_title="Efficiency Score")
    st.plotly_chart(fig3, width="stretch")

with col_d:
    st.markdown("##### Regional Performance")
    reg_summary = compute_regional_summary(df)
    fig4 = px.bar(reg_summary, x="Region", y="Total_Profit", color="Region",
                   color_discrete_sequence=PLOTLY_TEMPLATE_COLORWAY, text="Total_Profit")
    fig4.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig4.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
                        plot_bgcolor="white", paper_bgcolor="white", xaxis_title="", yaxis_title="Total Profit")
    st.plotly_chart(fig4, width="stretch")

st.markdown("---")
st.markdown("##### Division → Factory → Region Flow")
st.caption("Sankey view of how order volume flows from product Division through the assumed factory network to delivery Region.")

sankey_df = df.groupby(["Division", "Assigned_Factory_Name", "Region"]).size().reset_index(name="count")
divisions = sankey_df["Division"].unique().tolist()
factories = sankey_df["Assigned_Factory_Name"].unique().tolist()
regions = sankey_df["Region"].unique().tolist()
nodes = divisions + factories + regions
node_idx = {n: i for i, n in enumerate(nodes)}

links_source, links_target, links_value = [], [], []
for _, r in sankey_df.iterrows():
    links_source.append(node_idx[r["Division"]])
    links_target.append(node_idx[r["Assigned_Factory_Name"]])
    links_value.append(r["count"])

fac_region = df.groupby(["Assigned_Factory_Name", "Region"]).size().reset_index(name="count")
for _, r in fac_region.iterrows():
    links_source.append(node_idx[r["Assigned_Factory_Name"]])
    links_target.append(node_idx[r["Region"]])
    links_value.append(r["count"])

node_colors = (
    [COLORS["cherry"]] * len(divisions)
    + [COLORS["caramel"]] * len(factories)
    + [COLORS["mint"]] * len(regions)
)

fig_sankey = go.Figure(go.Sankey(
    node=dict(label=nodes, color=node_colors, pad=18, thickness=18,
              line=dict(color="white", width=1)),
    link=dict(source=links_source, target=links_target, value=links_value,
              color=rgba(COLORS['chocolate'], 0.13)),
))
fig_sankey.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), font_size=13)
st.plotly_chart(fig_sankey, width="stretch")

st.markdown("---")
st.markdown("##### Business Summary")
top_region = reg_summary.iloc[0]
top_factory = fac_summary.iloc[0]
st.write(
    f"In the current filtered view, **{top_region['Region']}** generates the highest profit "
    f"(${top_region['Total_Profit']:,.0f} across {top_region['Orders']:,} orders), while "
    f"**{top_factory['Assigned_Factory_Name']}** leads on shipping efficiency "
    f"(score {top_factory['Avg_Efficiency_Score']:.0f}/100). Average simulated lead time across "
    f"all filtered orders is **{kpis['Average Lead Time (days, simulated)']:.1f} days** at an "
    f"average shipping distance of **{kpis['Average Shipping Distance (km)']:.0f} km**."
)
