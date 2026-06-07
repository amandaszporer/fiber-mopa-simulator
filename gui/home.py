"""Home page — choose an existing architecture or create a new one."""
from __future__ import annotations

import streamlit as st

from framework import SystemConfig

from gui.state import EXAMPLES_DIR, goto, init_config_from_file, init_config_new


def render() -> None:
    st.title("⚡ Yb-Doped Fiber MOPA Simulator")
    st.write(
        "Build, save, and simulate a multi-stage Ytterbium-doped fiber "
        "amplifier chain for the BGU p-2026-158 project. Start a new design "
        "below, or open a saved one."
    )

    st.subheader("Create")
    if st.button("➕ Create new system", type="primary"):
        init_config_new()
        goto("builder")

    st.divider()
    st.subheader("Open an existing architecture")
    files = sorted(EXAMPLES_DIR.glob("*.json"))
    if not files:
        st.info(f"No saved architectures found in `{EXAMPLES_DIR}`.")
        return

    for path in files:
        try:
            cfg = SystemConfig.load(path)
        except Exception as exc:  # noqa: BLE001 — surface bad files, keep going
            st.warning(f"Skipped `{path.name}` — {exc}")
            continue
        with st.container(border=True):
            st.markdown(f"### {cfg.name or path.stem}")
            st.caption(f"`{path.name}`")
            if cfg.description:
                st.write(cfg.description)
            n_amp = sum(1 for c in cfg.components if c.get("type") == "Amplifier")
            st.write(
                f"**{len(cfg.components)}** components · "
                f"{n_amp} amplifier stage{'' if n_amp == 1 else 's'}"
            )
            if st.button("Open in builder", key=f"open:{path.name}"):
                init_config_from_file(path)
                goto("builder")
