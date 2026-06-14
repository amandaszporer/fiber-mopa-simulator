"""Streamlit widget builders driven by component `Param` metadata.

`render_param` turns a single `Param` into the right input widget and returns
the value in SI units. The form builders (`render_component_editor`,
`render_seed_form`, `render_metadata_form`, `render_requirements_form`) mutate
the working-config dicts in place each rerun.
"""
from __future__ import annotations

import streamlit as st

from components import COMPONENT_REGISTRY, SHAPE_FACTOR

from gui.units import display_spec


def render_param(name, param, current, key):
    """Render one `Param` as a widget. Returns the value in SI units."""
    disp_unit, scale = display_spec(name, param.unit)
    label = f"{name} [{disp_unit}]" if disp_unit else name
    help_txt = param.description or None

    # Enumerated choice -> selectbox (e.g. pump_direction, m_pol).
    if param.choices is not None:
        opts = list(param.choices)
        idx = opts.index(current) if current in opts else 0
        return st.selectbox(label, opts, index=idx, key=key, help=help_txt)

    # Integer -> stepped number_input.
    if param.type is int:
        kw = {}
        if param.min is not None:
            kw["min_value"] = int(param.min)
        if param.max is not None:
            kw["max_value"] = int(param.max)
        cur = int(current) if current is not None else int(param.default or 0)
        return int(st.number_input(label, value=cur, step=1, key=key,
                                   help=help_txt, **kw))

    # Float -> number_input in friendly display units, converted back to SI.
    if param.type is float:
        kw = {}
        if param.min is not None:
            kw["min_value"] = float(param.min) * scale
        if param.max is not None:
            kw["max_value"] = float(param.max) * scale
        cur = float(current) if current is not None else float(param.default or 0.0)
        val = st.number_input(label, value=cur * scale, format="%g",
                              key=key, help=help_txt, **kw)
        return float(val) / scale

    # String without choices (e.g. dopant).
    cur = "" if current is None else str(current)
    return st.text_input(label, value=cur, key=key, help=help_txt)


def render_component_editor(comp) -> bool:
    """Render a component's full parameter editor inside an expander.

    Mutates `comp` in place. Returns True if the delete button was pressed.
    """
    uid = comp["_uid"]
    ctype = comp["type"]
    cls = COMPONENT_REGISTRY[ctype]
    title = f"{ctype}  —  {comp.get('name') or '(unnamed)'}"
    deleted = False
    with st.expander(title):
        comp["name"] = st.text_input("name", value=comp.get("name", ""),
                                     key=f"{uid}:name")
        cols = st.columns(2)
        for i, (pname, pmeta) in enumerate(cls.parameters().items()):
            with cols[i % 2]:
                comp[pname] = render_param(
                    pname, pmeta, comp.get(pname, pmeta.default),
                    key=f"{uid}:{pname}",
                )
        if st.button("🗑 Delete component", key=f"{uid}:del"):
            deleted = True
    return deleted


def render_seed_form(seed) -> None:
    """Render the 6 editable seed-signal fields; mutates `seed` in place."""
    fields = [
        ("average_power", "Average power", "W"),
        ("rep_rate", "Rep rate", "Hz"),
        ("pulse_duration", "Pulse duration", "s"),
        ("linewidth", "Linewidth", "Hz"),
        ("wavelength", "Centre wavelength", "m"),
        ("mfd", "Seed MFD", "m"),
    ]
    cols = st.columns(3)
    for i, (fname, label, si_unit) in enumerate(fields):
        disp_unit, scale = display_spec(fname, si_unit)
        cur = float(seed.get(fname, 0.0))
        with cols[i % 3]:
            val = st.number_input(f"{label} [{disp_unit}]", value=cur * scale,
                                  min_value=0.0, format="%g", key=f"seed:{fname}")
        seed[fname] = val / scale

    avg = seed.get("average_power", 0.0)
    rep = seed.get("rep_rate", 0.0)
    dur = seed.get("pulse_duration", 0.0)
    energy = avg / rep if rep > 0 else 0.0
    peak = energy / (dur * SHAPE_FACTOR) if dur > 0 else 0.0
    st.caption(
        f"Derived → pulse energy ≈ {energy * 1e9:.3f} nJ,  "
        f"peak power ≈ {peak / 1e3:.2f} kW   "
        f"(peak = energy ÷ (duration × {SHAPE_FACTOR}))"
    )


def render_metadata_form(meta) -> None:
    """Render the metadata fields; mutates `meta` in place."""
    c = st.columns(2)
    meta["project"] = c[0].text_input("Project", value=meta.get("project", ""),
                                      key="meta:project")
    meta["institution"] = c[1].text_input("Institution",
                                          value=meta.get("institution", ""),
                                          key="meta:institution")
    meta["created"] = c[0].text_input("Created", value=meta.get("created", ""),
                                      key="meta:created")
    meta["version"] = c[1].text_input("Version", value=meta.get("version", ""),
                                      key="meta:version")
    meta["notes"] = st.text_area("Notes", value=meta.get("notes", ""),
                                 key="meta:notes")


def _req_minmax(req, key, label, unit, scale, defaults) -> None:
    """Render an optional ``{min, max}`` requirement block."""
    on = st.checkbox(f"{label} range", value=key in req, key=f"req:{key}:on")
    if on:
        cur = req.get(key) or {"min": defaults[0], "max": defaults[1]}
        c = st.columns(2)
        lo = c[0].number_input(f"Min {label.lower()} [{unit}]",
                               value=float(cur.get("min", defaults[0])) * scale,
                               format="%g", key=f"req:{key}:min")
        hi = c[1].number_input(f"Max {label.lower()} [{unit}]",
                               value=float(cur.get("max", defaults[1])) * scale,
                               format="%g", key=f"req:{key}:max")
        req[key] = {"min": lo / scale, "max": hi / scale}
    else:
        req.pop(key, None)


def render_requirements_form(req) -> None:
    """Render the V&V acceptance-criteria editor; mutates `req` in place."""
    st.caption("Enable acceptance criteria to get a pass/fail compliance "
               "table in the report.")

    on = st.checkbox("Wavelength target", value="wavelength" in req,
                     key="req:wavelength:on")
    if on:
        cur = req.get("wavelength") or {"target": 1.064e-6, "tolerance": 0.2e-9}
        c = st.columns(2)
        tgt = c[0].number_input("Target [nm]",
                                value=float(cur.get("target", 1.064e-6)) * 1e9,
                                format="%g", key="req:wavelength:target")
        tol = c[1].number_input("Tolerance [nm]",
                                value=float(cur.get("tolerance", 0.2e-9)) * 1e9,
                                min_value=0.0, format="%g",
                                key="req:wavelength:tol")
        req["wavelength"] = {"target": tgt / 1e9, "tolerance": tol / 1e9}
    else:
        req.pop("wavelength", None)

    on = st.checkbox("Max spectral width", value="spectral_width" in req,
                     key="req:spectral_width:on")
    if on:
        cur = req.get("spectral_width") or {"max": 0.05e-9}
        m = st.number_input("Max spectral width [nm]",
                            value=float(cur.get("max", 0.05e-9)) * 1e9,
                            min_value=0.0, format="%g",
                            key="req:spectral_width:max")
        req["spectral_width"] = {"max": m / 1e9}
    else:
        req.pop("spectral_width", None)

    _req_minmax(req, "rep_rate", "Rep rate", "Hz", 1.0, (10.0, 100000.0))
    _req_minmax(req, "pulse_duration", "Pulse duration", "ns", 1e9, (4e-9, 8e-9))
    _req_minmax(req, "avg_power", "Avg power", "W", 1.0, (22.0, 70.0))
    _req_minmax(req, "peak_power", "Peak power", "kW", 1e-3, (15000.0, 50000.0))

    on = st.checkbox("Per-amplifier gates", value="amplifier" in req,
                     key="req:amplifier:on")
    if on:
        cur = req.get("amplifier") or {}
        c = st.columns(3)
        ase = c[0].number_input("Max ASE ratio [dB]",
                                value=float(cur.get("ase_ratio_dB_max", -20.0)),
                                format="%g", key="req:amplifier:ase")
        sbs = c[1].number_input("Max SBS ratio",
                                value=float(cur.get("sbs_ratio_max", 1.0)),
                                format="%g", key="req:amplifier:sbs")
        srs = c[2].number_input("Max SRS ratio",
                                value=float(cur.get("srs_ratio_max", 1.0)),
                                format="%g", key="req:amplifier:srs")
        stable = st.checkbox("Require stable solver convergence",
                             value=bool(cur.get("solver_stable", True)),
                             key="req:amplifier:solver_stable")
        req["amplifier"] = {
            "ase_ratio_dB_max": ase,
            "sbs_ratio_max": sbs,
            "srs_ratio_max": srs,
            "solver_stable": stable,
        }
    else:
        req.pop("amplifier", None)
