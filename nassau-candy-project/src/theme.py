"""
theme.py — SAFE MODE
=====================
Zero custom CSS or raw HTML injection. Every element here is a native,
built-in Streamlit component (st.title, st.metric, st.info, st.caption).

Why this version exists: every visual bug hit during deployment so far
traced back to custom CSS/HTML fighting with a Streamlit or Plotly
version upgrade on the hosting platform. Native Streamlit components are
maintained by Streamlit itself and are guaranteed to render correctly on
whatever version Streamlit Cloud installs -- there is no custom styling
left that can drift out of sync with a platform update. This trades away
the candy-shop visual branding (chocolate/caramel/cherry palette, custom
fonts) for maximum reliability. Once the deployment is stable, the
branded version (src/theme.py in the main project) can be swapped back
in with a one-line change to the import in app.py and each page.
"""

from __future__ import annotations

import streamlit as st

COLORS = {
    "chocolate": "#2D1B12",
    "chocolate_light": "#4A2F20",
    "caramel": "#C17817",
    "caramel_light": "#E5A445",
    "cherry": "#D62839",
    "cream": "#FAF6F0",
    "cream_dark": "#F0E6D8",
    "mint": "#3FA796",
    "ink": "#1A1410",
    "muted": "#8A7968",
}

# Plain, safe hex colors only -- no alpha suffixes, no rgba needed, since
# these are only ever used as plain marker/line colors, never as fill
# colors with transparency (the thing that broke under a Plotly upgrade).
PLOTLY_TEMPLATE_COLORWAY = [
    COLORS["caramel"], COLORS["cherry"], COLORS["mint"],
    COLORS["chocolate_light"], COLORS["caramel_light"], COLORS["muted"],
]


def rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' + 0-1 alpha into 'rgba(r,g,b,a)' -- accepted by
    every Plotly version, unlike 8-digit hex-with-alpha strings."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def inject_global_css() -> None:
    """No-op in safe mode. Kept as a function so app.py and every page can
    call it unchanged -- it simply does nothing, leaving Streamlit's own
    default (guaranteed-to-work) theme in place."""
    return


def hero(title: str, subtitle: str, badges: list[str] | None = None) -> None:
    st.title(title)
    st.caption(subtitle)
    if badges:
        st.write(" · ".join(f"**{b}**" for b in badges))
    st.divider()


def kpi_card(label: str, value: str, col) -> None:
    with col:
        st.metric(label, value)


def assumption_box(text: str) -> None:
    st.info(f"**Assumption:** {text}")
