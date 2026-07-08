"""Page 3 — What-If Analysis: Current vs Recommended Factory."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from theme import inject_global_css, hero, kpi_card, COLORS  # noqa: E402
from optimization_engine import simulate_all_factories  # noqa: E402

st.set_page_config(page_title="What-If Analysis | Nassau Candy", page_icon="🔀", layout="wide")
inject_global_css()

if "features_df" not in st.session_state:
    st.error("Please open the app from the main page (app.py) first so data can load.")
    st.stop()

df: pd.DataFrame = st.session_state["features_df"]
pipeline = st.session_state["model_pipeline"]

hero(
    "What-If Analysis",
    "Compare an order's current factory assignment against the AI-recommended "
    "alternative, side by side.",
    badges=["Before / After", "Confidence Scored"],
)

order_options = df["Order ID"].drop_duplicates().sample(min(500, df["Order ID"].nunique()), random_state=1).tolist()
selected_order = st.selectbox("Select an Order ID to analyze (sampled for performance)", sorted(order_options))

row = df[df["Order ID"] == selected_order].iloc[0]
sim = simulate_all_factories(row, pipeline)
current = sim[sim["Is_Current_Factory"]].iloc[0]
best = sim.iloc[0]

is_already_optimal = current["Factory_ID"] == best["Factory_ID"]

st.markdown("---")
st.markdown(f"#### {row['Product Name']}  ·  {row['Region']} Region  ·  {row['Ship Mode']}")

col_before, col_arrow, col_after = st.columns([5, 1, 5])

with col_before:
    st.markdown("##### 📍 Current")
    st.markdown(f"**{current['Factory_Name']}**")
    kpi_card("Lead Time", f"{current['Predicted_Lead_Time_Days']:.1f} d", st.container())
    kpi_card("Profit", f"${current['Adjusted_Profit']:.2f}", st.container())
    kpi_card("Risk Score", f"{current['Risk_Score']:.0f}/100", st.container())
    kpi_card("Rec. Score", f"{current['Recommendation_Score']:.0f}/100", st.container())

with col_arrow:
    st.markdown("<div style='text-align:center; font-size:2.4rem; margin-top:3rem;'>→</div>", unsafe_allow_html=True)

with col_after:
    label = "✅ Recommended (same)" if is_already_optimal else "⭐ Recommended"
    st.markdown(f"##### {label}")
    st.markdown(f"**{best['Factory_Name']}**")
    kpi_card("Lead Time", f"{best['Predicted_Lead_Time_Days']:.1f} d", st.container())
    kpi_card("Profit", f"${best['Adjusted_Profit']:.2f}", st.container())
    kpi_card("Risk Score", f"{best['Risk_Score']:.0f}/100", st.container())
    kpi_card("Rec. Score", f"{best['Recommendation_Score']:.0f}/100", st.container())

st.markdown("---")

if is_already_optimal:
    st.success("This order is already assigned to the optimal factory under current conditions. No change recommended.")
else:
    lead_delta = current["Predicted_Lead_Time_Days"] - best["Predicted_Lead_Time_Days"]
    profit_delta = best["Adjusted_Profit"] - current["Adjusted_Profit"]
    score_delta = best["Recommendation_Score"] - current["Recommendation_Score"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Lead Time Change", f"{lead_delta:+.1f} days", delta=f"{lead_delta:+.1f}", delta_color="inverse")
    m2.metric("Profit Change", f"${profit_delta:+.2f}", delta=f"{profit_delta:+.2f}")
    m3.metric("Score Improvement", f"{score_delta:+.0f} pts", delta=f"{score_delta:+.0f}")

    st.markdown("##### Side-by-Side Comparison")
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Lead Time (days)", "Profit ($)", "Risk Score"))
    pairs = [
        ("Predicted_Lead_Time_Days", 1), ("Adjusted_Profit", 2), ("Risk_Score", 3),
    ]
    for metric, col_pos in pairs:
        fig.add_trace(go.Bar(
            x=["Current", "Recommended"], y=[current[metric], best[metric]],
            marker_color=[COLORS["chocolate_light"], COLORS["mint"]],
            text=[f"{current[metric]:.1f}", f"{best[metric]:.1f}"], textposition="outside",
            showlegend=False,
        ), row=1, col=col_pos)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, width="stretch")

    st.info(
        f"**Business explanation:** Reassigning this order from {current['Factory_Name']} to "
        f"{best['Factory_Name']} is predicted to {'reduce' if lead_delta > 0 else 'increase'} lead time "
        f"by {abs(lead_delta):.1f} days and {'improve' if profit_delta > 0 else 'reduce'} profit by "
        f"${abs(profit_delta):.2f}, for a net recommendation score improvement of {score_delta:.0f} points "
        f"(confidence: {best['Recommendation_Score']:.0f}/100)."
    )
