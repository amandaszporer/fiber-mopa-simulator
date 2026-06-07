"""Run & Results page — execute the simulation and show report + plots."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import streamlit as st

from framework import Simulator, SystemConfig

from gui.state import (
    REPORT_DIR, get_config, goto, has_config, strip_uids, validate_config,
)

_MODE_HELP = (
    "time-dependent — physically accurate auto-dispatch (default). "
    "steady — force the fast steady-state BVP. "
    "full — force the Level-5 B1+B2 pulse cycle (slow; pulse-shape diagnostics)."
)


def _run(cfg: dict, mode: str) -> None:
    try:
        validate_config(cfg)
    except ValueError as exc:
        st.error(f"Invalid configuration — {exc}")
        return

    clean = strip_uids(cfg)
    if not clean["components"]:
        st.error("Add at least one component on the Builder page first.")
        return

    with st.spinner(f"Running simulation in '{mode}' mode — this may take a while…"):
        sim = Simulator.from_config(SystemConfig.from_dict(clean))
        sim.run(mode=mode)
        report_text = sim.report(mode_label=mode)
        rows = [
            {"Parameter": p, "Actual": a, "Criterion": c,
             "Result": "✅ PASS" if ok else "❌ FAIL"}
            for p, a, c, ok in sim.check_requirements()
        ]
        all_pass = all(r["Result"].startswith("✅") for r in rows) if rows else None

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = REPORT_DIR / f"{ts}_gui"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.txt").write_text(report_text)
        from utils import plotting
        plot_paths = [str(p) for p in plotting.plot_all(sim, out_dir)]

    st.session_state.results = {
        "report": report_text,
        "rows": rows,
        "all_pass": all_pass,
        "plots": plot_paths,
        "out_dir": str(out_dir),
        "config_json": json.dumps(clean, indent=2),
        "mode": mode,
    }


def render() -> None:
    if not has_config():
        st.title("📊 Run & Results")
        st.warning("No system loaded. Start from the Home page.")
        if st.button("Go to Home"):
            goto("home")
        return

    cfg = get_config()
    st.title("📊 Run & Results")
    st.write(
        f"System: **{cfg.get('name') or '(unnamed)'}** — "
        f"{len(cfg['components'])} components"
    )

    mode = st.radio("Solver mode", ["time-dependent", "steady", "full"],
                    horizontal=True, help=_MODE_HELP)
    if st.button("▶ Run simulation", type="primary"):
        _run(cfg, mode)

    results = st.session_state.get("results")
    if not results:
        st.info("Press **Run simulation** to compute the report and plots.")
        return

    st.divider()
    rows = results["rows"]
    if rows:
        if results["all_pass"]:
            st.success("V&V compliance — ALL REQUIREMENTS PASS")
        else:
            st.error("V&V compliance — ONE OR MORE REQUIREMENTS FAIL")
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No V&V requirements defined — showing raw output only.")

    st.subheader("Report")
    st.code(results["report"], language="text")

    st.subheader("Plots")
    any_plot = False
    for p in results["plots"]:
        path = Path(p)
        if path.exists() and path.stat().st_size > 0:
            any_plot = True
            st.image(p, caption=path.stem.replace("_", " ").title(),
                     width="stretch")
    if not any_plot:
        st.info("No plots were produced for this run.")

    st.subheader("Downloads")
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Report (summary.txt)", results["report"],
                       file_name="summary.txt", width="stretch")
    d2.download_button("⬇ Architecture (JSON)", results["config_json"],
                       file_name=f"{cfg.get('name') or 'system'}.json",
                       mime="application/json", width="stretch")
    st.caption(f"Run output written to `{results['out_dir']}`")
