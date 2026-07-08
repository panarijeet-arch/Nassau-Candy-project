"""Page 6 — Machine Learning Dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from theme import inject_global_css, hero, COLORS  # noqa: E402

st.set_page_config(page_title="ML Dashboard | Nassau Candy", page_icon="🤖", layout="wide")
inject_global_css()

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"

hero(
    "Machine Learning Dashboard",
    "Model comparison, feature importance, and prediction accuracy for the "
    "lead-time prediction pipeline.",
    badges=["Explainable AI", "Model Comparison"],
)

comparison_path = REPORTS_DIR / "model_comparison.json"
if not comparison_path.exists():
    st.error("Model comparison report not found. Run `python src/run_pipeline.py` first.")
    st.stop()

with open(comparison_path) as f:
    comparison = json.load(f)

models = comparison["models"]
best_model_name = comparison["best_model"]

st.markdown("#### Model Comparison")
comp_df = pd.DataFrame(models).T.reset_index().rename(columns={"index": "Model"})
comp_df = comp_df.round(4)

c1, c2 = st.columns([2, 1])
with c1:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comp_df["Model"], y=comp_df["r2"], name="Test R²",
                          marker_color=[COLORS["cherry"] if m == best_model_name else COLORS["caramel"] for m in comp_df["Model"]],
                          text=comp_df["r2"], texttemplate="%{text:.3f}", textposition="outside"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white",
                       yaxis_title="R² Score", title="Test R² by Model (red = selected)")
    st.plotly_chart(fig, width="stretch")
with c2:
    st.dataframe(comp_df.set_index("Model")[["rmse", "mae", "r2", "cv_r2_mean"]], width="stretch")

st.success(f"**Best model: {best_model_name}** — {comparison['explanation']}")

st.markdown("---")
st.markdown("#### RMSE & MAE Comparison")
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=comp_df["Model"], y=comp_df["rmse"], name="RMSE", marker_color=COLORS["chocolate_light"]))
fig2.add_trace(go.Bar(x=comp_df["Model"], y=comp_df["mae"], name="MAE", marker_color=COLORS["mint"]))
fig2.update_layout(barmode="group", height=340, margin=dict(l=10, r=10, t=20, b=10),
                    plot_bgcolor="white", paper_bgcolor="white", yaxis_title="Days",
                    legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig2, width="stretch")

st.markdown("---")
st.markdown("#### Feature Importance & Residuals (Selected Model)")

if "features_df" in st.session_state and "model_pipeline" in st.session_state:
    from sklearn.model_selection import train_test_split
    from feature_engineering import FEATURE_COLUMNS_FOR_ML, CATEGORICAL_COLUMNS_FOR_ML
    from model_training import TARGET_COLUMN, _get_feature_names

    df = st.session_state["features_df"]
    pipeline = st.session_state["model_pipeline"]

    X = df[FEATURE_COLUMNS_FOR_ML + CATEGORICAL_COLUMNS_FOR_ML]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = pipeline.predict(X_test)

    col_imp, col_res = st.columns(2)

    with col_imp:
        fitted_model = pipeline.named_steps["model"]
        if hasattr(fitted_model, "feature_importances_"):
            names = _get_feature_names(pipeline.named_steps["prep"])
            imp = pd.Series(fitted_model.feature_importances_, index=names).sort_values(ascending=False).head(12)
            fig3 = px.bar(imp[::-1], orientation="h", color_discrete_sequence=[COLORS["caramel"]])
            fig3.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                                plot_bgcolor="white", paper_bgcolor="white", title="Top Feature Importances",
                                xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig3, width="stretch")
        elif hasattr(fitted_model, "coef_"):
            names = _get_feature_names(pipeline.named_steps["prep"])
            imp = pd.Series(abs(fitted_model.coef_), index=names).sort_values(ascending=False).head(12)
            fig3 = px.bar(imp[::-1], orientation="h", color_discrete_sequence=[COLORS["caramel"]])
            fig3.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                                plot_bgcolor="white", paper_bgcolor="white", title="Top Coefficient Magnitudes",
                                xaxis_title="|Coefficient|", yaxis_title="")
            st.plotly_chart(fig3, width="stretch")

    with col_res:
        residuals = y_test.values - y_pred
        fig4 = px.scatter(x=y_pred, y=residuals, color_discrete_sequence=[COLORS["cherry"]],
                          labels={"x": "Predicted Lead Time (days)", "y": "Residual"})
        fig4.add_hline(y=0, line_dash="dash", line_color=COLORS["chocolate"])
        fig4.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                            plot_bgcolor="white", paper_bgcolor="white", title="Residual Plot")
        st.plotly_chart(fig4, width="stretch")

    st.markdown("##### Prediction Accuracy")
    fig5 = px.scatter(x=y_test, y=y_pred, opacity=0.4, color_discrete_sequence=[COLORS["caramel"]],
                      labels={"x": "Actual Lead Time (days)", "y": "Predicted Lead Time (days)"})
    min_v, max_v = float(y_test.min()), float(y_test.max())
    fig5.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                              line=dict(color=COLORS["chocolate"], dash="dash"), name="Perfect Prediction"))
    fig5.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                        plot_bgcolor="white", paper_bgcolor="white", title="Predicted vs Actual")
    st.plotly_chart(fig5, width="stretch")
else:
    st.info("Open the app from the main page first to enable live feature-importance and residual analysis.")
