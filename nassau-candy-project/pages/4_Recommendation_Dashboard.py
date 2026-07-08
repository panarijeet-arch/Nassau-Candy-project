"""Page 4 — Recommendation Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from theme import inject_global_css, hero, kpi_card, COLORS, PLOTLY_TEMPLATE_COLORWAY  # noqa: E402
from optimization_engine import generate_top_recommendations  # noqa: E402
from recommender import recommendation_coverage  # noqa: E402

st.set_page_config(page_title="Recommendations | Nassau Candy", page_icon="🏆", layout="wide")
inject_global_css()

if "features_df" not in st.session_state:
    st.error("Please open the app from the main page (app.py) first so data can load.")
    st.stop()

df: pd.DataFrame = st.session_state["features_df"]
pipeline = st.session_state["model_pipeline"]

hero(
    "Recommendation Dashboard",
    "Ranked factory-reallocation opportunities across the entire order book, "
    "with downloadable results.",
    badges=["Ranked", "Exportable"],
)

with st.spinner("Running optimization across all orders..."):
    coverage = recommendation_coverage(df, pipeline)
    all_recs = generate_top_recommendations(df, pipeline, sample_size=None, top_n=None)

c1, c2, c3 = st.columns(3)
kpi_card("Orders Analyzed", f"{coverage['orders_analyzed']:,}", c1)
kpi_card("Already Optimal", f"{coverage['pct_already_optimal']:.1f}%", c2)
kpi_card("Recommend Reallocation", f"{coverage['pct_recommend_change']:.1f}%", c3)

st.markdown("---")

if all_recs.empty:
    st.success("Every order is already on its optimal factory assignment — no reallocations recommended.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🏆 Top Recommendations", "📦 Products to Reallocate", "🏭 Factory Gains"])

with tab1:
    st.markdown("##### Top 5 Highest-Impact Recommendations")
    top5 = all_recs.head(5)
    for _, r in top5.iterrows():
        with st.container(border=True):
            cc1, cc2, cc3 = st.columns([3, 2, 2])
            with cc1:
                st.markdown(f"**{r['Product Name']}**  ·  Order `{r['Order ID']}`  ·  {r['Region']}")
                st.caption(r["Business_Explanation"])
            with cc2:
                st.markdown(f"{r['Current_Factory']} → **{r['Recommended_Factory']}**")
                st.caption(f"Risk: {r['Risk_Level']}")
            with cc3:
                st.metric("Score Gain", f"+{r['Score_Improvement_Points']:.0f} pts")

    st.markdown("##### Full Ranked Table")
    st.dataframe(all_recs, width="stretch", hide_index=True)
    csv = all_recs.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download All Recommendations (CSV)", csv, "nassau_candy_recommendations.csv", "text/csv")

with tab2:
    st.markdown("##### Most-Recommended Products")
    prod_counts = all_recs["Product Name"].value_counts().reset_index()
    prod_counts.columns = ["Product Name", "Recommendation Count"]
    fig = px.bar(prod_counts.head(10), x="Recommendation Count", y="Product Name", orientation="h",
                 color_discrete_sequence=[COLORS["caramel"]])
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white", paper_bgcolor="white",
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")

with tab3:
    st.markdown("##### Net Operational Gains by Recommended Factory")
    fac_gains = all_recs.groupby("Recommended_Factory").agg(
        Recommendations=("Order ID", "count"),
        Avg_Lead_Time_Improvement=("Lead_Time_Improvement_Days", "mean"),
        Total_Profit_Impact=("Expected_Profit_Impact", "sum"),
    ).round(2).reset_index()
    st.dataframe(fac_gains, width="stretch", hide_index=True)
    fig2 = px.bar(fac_gains, x="Recommended_Factory", y="Total_Profit_Impact",
                  color="Recommended_Factory", color_discrete_sequence=PLOTLY_TEMPLATE_COLORWAY, text="Total_Profit_Impact")
    fig2.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig2.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
                        plot_bgcolor="white", paper_bgcolor="white", xaxis_title="", yaxis_title="Total Profit Impact ($)")
    st.plotly_chart(fig2, width="stretch")
