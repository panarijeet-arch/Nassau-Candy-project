"""Page 8 — Data Quality Report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from theme import inject_global_css, hero, kpi_card, COLORS  # noqa: E402

st.set_page_config(page_title="Data Quality | Nassau Candy", page_icon="🔍", layout="wide")
inject_global_css()

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"

hero(
    "Data Quality Report",
    "A complete, honest audit of the source dataset — including the critical "
    "Ship Date anomaly that shapes every modeling decision in this project.",
    badges=["Auditable", "Transparent"],
)

report_path = REPORTS_DIR / "data_quality_report.json"
if not report_path.exists():
    st.error("Data quality report not found. Run `python src/run_pipeline.py` first.")
    st.stop()

with open(report_path) as f:
    report = json.load(f)

c1, c2, c3, c4 = st.columns(4)
kpi_card("Total Rows", f"{report['n_rows']:,}", c1)
kpi_card("Total Columns", f"{report['n_cols']}", c2)
kpi_card("Duplicate Rows", f"{report['duplicate_rows']}", c3)
kpi_card("Financial Inconsistencies", f"{report.get('financial_inconsistent_rows', 0)}", c4)

st.markdown("---")
st.markdown("### 🚨 Critical Finding: Ship Date Anomaly")

anomaly = report.get("ship_date_anomaly", {})
st.markdown(
    f"""
    <div class="nc-assumption-box">
    {anomaly.get('verdict', '')}
    </div>
    """,
    unsafe_allow_html=True,
)

ac1, ac2, ac3 = st.columns(3)
ac1.metric("Order Date Range", f"{anomaly.get('order_date_range', ['', ''])[0]} → {anomaly.get('order_date_range', ['', ''])[1]}")
ac2.metric("Ship Date Range", f"{anomaly.get('ship_date_range', ['', ''])[0]} → {anomaly.get('ship_date_range', ['', ''])[1]}")
ac3.metric("Raw Lead Time Range", f"{anomaly.get('raw_lead_days_min', 0):.0f} – {anomaly.get('raw_lead_days_max', 0):.0f} days")

st.caption(
    "For reference: a real-world candy distributor would expect lead times measured in "
    "single-digit days, not hundreds or thousands. This confirms the column cannot be used "
    "as-is for lead-time modeling."
)

st.markdown("---")
st.markdown("### Missing Values")
missing = pd.Series(report["missing_values"]).reset_index()
missing.columns = ["Column", "Missing Count"]
missing = missing[missing["Missing Count"] > 0]
if missing.empty:
    st.success("No missing values found in any column — the dataset is complete.")
else:
    fig = px.bar(missing, x="Missing Count", y="Column", orientation="h", color_discrete_sequence=[COLORS["cherry"]])
    st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.markdown("### Outlier Scan (IQR Method)")
outliers = pd.Series(report["outlier_counts_iqr"]).reset_index()
outliers.columns = ["Column", "Outlier Count"]
fig2 = px.bar(outliers, x="Column", y="Outlier Count", color_discrete_sequence=[COLORS["caramel"]], text="Outlier Count")
fig2.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig2, width="stretch")
st.caption(
    "Outliers detected here are large/small but legitimate orders (e.g. bulk purchases), "
    "not data errors — confirmed during manual audit. They are retained in modeling."
)

st.markdown("---")
st.markdown("### Categorical Value Inventory")
cc1, cc2, cc3, cc4 = st.columns(4)
with cc1:
    st.markdown("**Division**")
    st.write(report.get("Division_unique_values", []))
with cc2:
    st.markdown("**Region**")
    st.write(report.get("Region_unique_values", []))
with cc3:
    st.markdown("**Ship Mode**")
    st.write(report.get("Ship Mode_unique_values", []))
with cc4:
    st.markdown("**Country/Region**")
    st.write(report.get("Country/Region_unique_values", []))

st.markdown("---")
st.markdown("### Summary of Data Decisions Made in This Project")
st.markdown(
    """
| Issue Found | Decision | Where Applied |
|---|---|---|
| No factory table exists | Derived 3 factories from `Division` (clearly labeled assumption) | All pages |
| No customer/factory coordinates | Used public US/Canada state-centroid lookup for customers; placed factories at real logistics-hub coordinates | Geographic Dashboard, Optimization Engine |
| `Ship Date` column corrupted (900–1600 "day" gaps) | Excluded from modeling; simulated a Ship-Mode-driven lead time instead, clearly labeled "Simulated" everywhere it appears | Feature Engineering, ML Dashboard |
| Financial columns (Sales/Cost/Profit) | Verified fully consistent (Sales = Cost + Profit, no negatives) — used as-is, no transformation needed | All pages |
| No missing values or duplicates | Confirmed via audit — minimal cleaning required | Preprocessing |
"""
)
