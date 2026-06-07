"""Streamlit GUI entry point for the Yb-Doped Fiber MOPA Simulator.

Launch from the project root:

    streamlit run app.py

The GUI is a thin front-end over the existing engine — `SystemConfig` /
`Simulator` (framework.py), the component registry (components.py), and the
plotting helpers (utils/plotting.py). No engine code is modified.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when launched via `streamlit run`.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from gui import builder, home, results  # noqa: E402

st.set_page_config(
    page_title="Yb MOPA Simulator",
    page_icon="⚡",
    layout="wide",
)

# Explicit url_path on each page — all three callables are named `render`, so
# without this Streamlit would infer the same pathname for all of them.
home_page = st.Page(home.render, title="Home", icon="🏠",
                    url_path="home", default=True)
builder_page = st.Page(builder.render, title="Builder", icon="🛠️",
                       url_path="builder")
results_page = st.Page(results.render, title="Run & Results", icon="📊",
                       url_path="results")

# Stash the page objects so button handlers can navigate via st.switch_page.
st.session_state["_pages"] = {
    "home": home_page,
    "builder": builder_page,
    "results": results_page,
}

st.navigation([home_page, builder_page, results_page]).run()
