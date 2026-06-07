"""Session-state helpers.

The whole working system lives in ``st.session_state.config`` as a plain dict
in ``SystemConfig.to_dict()`` shape (``name, description, seed, components,
metadata, requirements``). Each component dict also carries a synthetic
``_uid`` used as a stable widget / reorder key; ``_uid`` is stripped before the
config is saved, run, or exported.
"""
from __future__ import annotations

import copy
import uuid
from pathlib import Path

import streamlit as st

from components import component_from_dict, make_seed
from framework import SystemConfig, signal_from_dict, signal_to_dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
REPORT_DIR = PROJECT_ROOT / "report"

_DEFAULT_METADATA = {
    "project": "", "institution": "", "created": "", "version": "", "notes": "",
}


def new_uid() -> str:
    """Short, collision-free id for a component dict."""
    return uuid.uuid4().hex[:8]


def attach_uids(cfg: dict) -> dict:
    """Give every component dict a stable ``_uid`` (in place)."""
    for comp in cfg.get("components", []):
        comp.setdefault("_uid", new_uid())
    return cfg


def strip_uids(cfg: dict) -> dict:
    """Deep copy of ``cfg`` with all ``_uid`` keys removed (for save / run)."""
    clean = copy.deepcopy(cfg)
    for comp in clean.get("components", []):
        comp.pop("_uid", None)
    return clean


def _clear_form_widget_state() -> None:
    """Drop seed / metadata / requirements widget keys.

    Those widgets use config-independent keys, so without this a freshly
    loaded config would be masked by stale widget values from the previous
    one. Component editors are keyed by ``_uid`` (regenerated per load) and so
    need no clearing.
    """
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(("seed:", "meta:", "req:")):
            del st.session_state[key]


def init_config_new() -> None:
    """Start a blank system with a default seed signal."""
    _clear_form_widget_state()
    st.session_state.config = {
        "name": "",
        "description": "",
        "seed": signal_to_dict(make_seed()),
        "components": [],
        "metadata": dict(_DEFAULT_METADATA),
        "requirements": {},
    }
    st.session_state.source_path = None
    st.session_state.pop("results", None)


def init_config_from_file(path) -> None:
    """Load a saved architecture from a JSON file into the working state."""
    _clear_form_widget_state()
    cfg = SystemConfig.load(path).to_dict()
    cfg.setdefault("seed", signal_to_dict(make_seed()))
    metadata = dict(_DEFAULT_METADATA)
    metadata.update(cfg.get("metadata") or {})
    cfg["metadata"] = metadata
    cfg.setdefault("requirements", {})
    attach_uids(cfg)
    st.session_state.config = cfg
    st.session_state.source_path = str(path)
    st.session_state.pop("results", None)


def has_config() -> bool:
    return "config" in st.session_state


def get_config() -> dict:
    return st.session_state.config


def validate_config(cfg: dict) -> None:
    """Raise ``ValueError`` if the config cannot be built into an engine run.

    Reuses the engine's own validation: ``signal_from_dict`` for the seed and
    ``component_from_dict`` for every component (type + range + choices).
    """
    signal_from_dict(cfg.get("seed", {}))
    for comp in strip_uids(cfg)["components"]:
        component_from_dict(comp)


def goto(page_key: str) -> None:
    """Programmatically switch pages (keys: home / builder / results)."""
    st.switch_page(st.session_state["_pages"][page_key])
