"""
app.py
======
Nassau Candy AI Decision Intelligence Platform — main entry point.

Run with:
    streamlit run app.py

On first launch, if no cached model/feature table exists, this script
automatically runs the full pipeline (clean -> engineer -> train ->
cluster) and caches results so subsequent launches are instant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from theme import inject_global_css, hero, COLORS  # noqa: E402
from utils import load_pipeline  # noqa: E402

st.set_page_config(
    page_title="Nassau Candy | AI Decision Intelligence",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = Path(__file__).resolve().parent / "models"
PROCESSED_CSV = DATA_DIR / "processed_features.csv"


@st.cache_resource(show_spinner="Loading AI model...")
def get_model():
    pipeline = load_pipeline("best_model")
    if pipeline is None:
        st.error(
            "No trained model found. Please run `python src/run_pipeline.py` "
            "from the project root once before launching the app."
        )
        st.stop()
    name_path = MODELS_DIR / "best_model_name.txt"
    name = name_path.read_text().strip() if name_path.exists() else "Unknown"
    return pipeline, name


@st.cache_data(show_spinner="Loading processed dataset...")
def get_features() -> pd.DataFrame:
    if not PROCESSED_CSV.exists():
        st.error(
            "Processed feature data not found. Please run `python src/run_pipeline.py` "
            "from the project root once before launching the app."
        )
        st.stop()
    df = pd.read_csv(PROCESSED_CSV)
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")
    return df


def main() -> None:
    pipeline, model_name = get_model()
    df = get_features()

    st.session_state["model_pipeline"] = pipeline
    st.session_state["model_name"] = model_name
    st.session_state["features_df"] = df

    with st.sidebar:
        st.markdown(
            f"""
            <div style="text-align:center; padding: 0.6rem 0 1.2rem 0;">
                <div style="font-size:2.2rem;">🍬</div>
                <div class="nc-sidebar-brand-name" style="font-family:'Fraunces',serif; font-size:1.25rem; font-weight:600; color:{COLORS['chocolate']};">
                    Nassau Candy
                </div>
                <div style="font-size:0.78rem; color:{COLORS['muted']}; letter-spacing:0.04em; text-transform:uppercase;">
                    AI Decision Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Active model: **{model_name}**")
        st.caption(f"Dataset: **{len(df):,}** orders")
        st.divider()
        st.markdown(
            """
            <div style="font-size:0.78rem; color:#8A7968; line-height:1.5;">
            ⚠️ <b>Data note:</b> This dataset has no factory or coordinate
            data. Factories are derived from Division, and lead time is a
            simulated proxy (Ship Date is corrupted in the source file).
            See the <b>Data Quality</b> page for full details.
            </div>
            """,
            unsafe_allow_html=True,
        )

    hero(
        "AI Decision Intelligence Platform",
        "Predictive shipping performance and factory-reallocation recommendations "
        "for Nassau Candy's distribution network — built on real order data, "
        "engineered features, and a transparent, explainable ML pipeline.",
        badges=["Machine Learning", "Optimization", "Supply Chain"],
    )

    st.markdown("### Welcome 👋")
    st.write(
        "Use the navigation in the sidebar (or the **Pages** menu) to explore the "
        "Executive Dashboard, run the Factory Optimization Simulator, compare "
        "What-If scenarios, review ranked recommendations, monitor risk, inspect "
        "the ML pipeline, or explore the geographic view."
    )

    c1, c2, c3, c4 = st.columns(4)
    from theme import kpi_card
    kpi_card("Total Orders", f"{df['Order ID'].nunique():,}", c1)
    kpi_card("Total Sales", f"${df['Sales'].sum():,.0f}", c2)
    kpi_card("Total Profit", f"${df['Gross Profit'].sum():,.0f}", c3)
    kpi_card("Avg. Profit Margin", f"{df['Profit_Margin'].mean()*100:.1f}%", c4)

    st.markdown("---")
    st.markdown(
        """
        <div class="nc-assumption-box">
        <b>Read this first:</b> The source CSV contains no factory table, factory
        coordinates, or customer coordinates. This project derives a 3-factory
        network from the <code>Division</code> column (a real, defensible
        production-grouping signal) and approximates customer locations using
        public state/province geographic centroids. The <code>Ship Date</code>
        column was found to be corrupted during data-quality auditing (see the
        Data Quality page) and is excluded from modeling — lead time is instead
        simulated from the real <code>Ship Mode</code> column. Every page that
        uses these derived values labels them clearly. All financial figures
        (Sales, Cost, Gross Profit, Margin) are 100% real, unmodified data.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
