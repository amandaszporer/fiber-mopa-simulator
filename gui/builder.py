"""Builder page — assemble, edit, reorder, and save a MOPA system."""
from __future__ import annotations

import re
from pathlib import Path

import streamlit as st
from streamlit_sortables import sort_items

from components import COMPONENT_REGISTRY, Amplifier
from framework import SystemConfig

from gui.state import (
    EXAMPLES_DIR, get_config, goto, has_config, new_uid, strip_uids,
    validate_config,
)
from gui.widgets import (
    render_component_editor, render_metadata_form, render_requirements_form,
    render_seed_form,
)

_AMP_PRESETS = ["Custom", "yb_5_130", "yb_10_125", "yb_30_250"]


def _slug(text: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", (text or "").strip().lower()).strip("_")
    return s or "system"


def _add_component(cfg, ctype, preset) -> None:
    """Append a new component dict, parameters seeded from defaults / preset."""
    if ctype == "Amplifier" and preset and preset != "Custom":
        comp = getattr(Amplifier, preset)(name="").to_dict()
    else:
        cls = COMPONENT_REGISTRY[ctype]
        comp = {"type": ctype, "name": ""}
        for pname, pmeta in cls.parameters().items():
            comp[pname] = pmeta.default
    comp["_uid"] = new_uid()
    cfg["components"].append(comp)


def render() -> None:
    if not has_config():
        st.title("🛠️ Builder")
        st.warning("No system loaded. Start from the Home page.")
        if st.button("Go to Home"):
            goto("home")
        return

    cfg = get_config()
    st.title("🛠️ Build & Edit System")

    cfg["name"] = st.text_input("System name", value=cfg.get("name", ""))
    cfg["description"] = st.text_area(
        "Description", value=cfg.get("description", ""), height=80,
    )

    with st.expander("🌱 Seed signal", expanded=not cfg["components"]):
        render_seed_form(cfg["seed"])
    with st.expander("✅ V&V requirements"):
        render_requirements_form(cfg["requirements"])
    with st.expander("📋 Metadata"):
        render_metadata_form(cfg["metadata"])

    st.divider()
    st.subheader("Add a component")
    c1, c2, c3 = st.columns([3, 3, 2])
    ctype = c1.selectbox("Component type", list(COMPONENT_REGISTRY.keys()))
    preset = None
    if ctype == "Amplifier":
        preset = c2.selectbox(
            "Fiber preset", _AMP_PRESETS,
            help="Presets pre-fill fiber geometry from the Nufern inventory; "
                 "tune pump power / length after adding.",
        )
    if c3.button("➕ Add component", width="stretch"):
        _add_component(cfg, ctype, preset)
        st.rerun()

    st.divider()
    comps = cfg["components"]
    st.subheader(f"Component chain ({len(comps)})")
    if not comps:
        st.info("No components yet — add some from the palette above.")
    else:
        st.caption("Drag to reorder — the signal flows top → bottom.")
        labels = [
            f"{i + 1}. {c['type']}" + (f" — {c['name']}" if c.get("name") else "")
            for i, c in enumerate(comps)
        ]
        sorted_labels = sort_items(labels, key="chain_sortable")
        if sorted_labels and list(sorted_labels) != labels:
            order = [labels.index(lbl) for lbl in sorted_labels]
            cfg["components"] = [comps[i] for i in order]
            st.rerun()

        st.markdown("##### Component parameters")
        for comp in list(cfg["components"]):
            if render_component_editor(comp):
                cfg["components"] = [
                    c for c in cfg["components"] if c["_uid"] != comp["_uid"]
                ]
                st.rerun()

    st.divider()
    st.subheader("💾 Save & run")
    src = st.session_state.get("source_path")
    default_fname = Path(src).name if src else f"{_slug(cfg.get('name'))}.json"
    fname = st.text_input("Filename (saved into examples/)", value=default_fname)

    s1, s2 = st.columns(2)
    if s1.button("💾 Save to examples/", width="stretch"):
        try:
            validate_config(cfg)
        except ValueError as exc:
            st.error(f"Cannot save — {exc}")
        else:
            name = fname if fname.endswith(".json") else f"{fname}.json"
            path = EXAMPLES_DIR / name
            SystemConfig.from_dict(strip_uids(cfg)).save(path)
            st.session_state.source_path = str(path)
            st.success(f"Saved to {path}")
    if s2.button("▶ Go to Run & Results", width="stretch"):
        goto("results")
