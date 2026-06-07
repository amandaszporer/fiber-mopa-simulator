"""
Yb-doped silica fiber cross-section loader.

Reads ``ase/data/cross_sections_yb.csv`` (lambda_nm, sigma_a_m2, sigma_e_m2)
and exposes interpolation onto an arbitrary wavelength grid.

The CSV is the **measured** aluminosilicate (AS) cross-section table of
Melkumov et al., FORC Preprint No. 5 (2004), arXiv:1502.02885, Appendix 2 —
the closest published host-glass match for our Coherent/Nufern
SM-YDF-5/130-VIII fiber. It is stored on the measured grid as published
(~4 nm spacing, 1 nm across the 968-986 nm peak, range 848-1180 nm),
converted from the published pm² units by ``sigma_m2 = sigma_pm2 * 1e-24``.
It is the measured source of truth, *not* a generated artefact: there are no
hand-traced anchors to regenerate it from. The earlier 11-anchor Paschotta-1997
table was hand-read off a figure and had the 1000-1030 nm emission ordering
inverted; this dataset fixes that.

``regenerate_csv`` only performs the pm²→m² unit conversion of the raw
provenance file (``melkumov_AS_raw.csv``); re-run with
``python -m ase.cross_sections`` to reproduce ``cross_sections_yb.csv``.

Interpolation is piecewise **log-linear** (linear in log σ between measured
nodes). See :func:`_log_linear_interp` for the rationale.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

_DATA_PATH = Path(__file__).parent / "data" / "cross_sections_yb.csv"
_RAW_PATH = Path(__file__).parent / "data" / "melkumov_AS_raw.csv"

# Published pm² → m² conversion (1 pm² = 1e-24 m²).
_PM2_TO_M2 = 1e-24

# Cross-sections span ~4 decades and the long-λ McCumber tail can in principle
# reach zero in other datasets; clamp to this positive floor before log() so the
# interpolation never sees log(0). The shipped Melkumov AS data has no zeros
# (min σ_a ≈ 2.2e-30 m²), so this is a defensive guard, not an active rescale.
_SIGMA_FLOOR_M2 = 1e-30


def _read_csv(path: Path) -> np.ndarray:
    """Read a `#`-commented numeric CSV into a float array, skipping the header row."""
    rows: list[list[float]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append([float(x) for x in line.split(",")])
            except ValueError:
                # Column-header row (non-numeric) — skip
                continue
    return np.asarray(rows)


def load_yb_cross_sections() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lambdas_m, sigma_a_m2, sigma_e_m2) read from the CSV."""
    raw = _read_csv(_DATA_PATH)
    lambdas_m = raw[:, 0] * 1e-9
    return lambdas_m, raw[:, 1], raw[:, 2]


def _log_linear_interp(
    targets_m: np.ndarray,
    nodes_m: np.ndarray,
    sigma_a: np.ndarray,
    sigma_e: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise log-linear interpolation (linear in log σ between nodes).

    Why log-linear rather than log-PCHIP? The old PCHIP-on-log choice existed to
    reconstruct a smooth curve from 11 sparse, widely spaced hand-read anchors
    without spline overshoot. With the dense measured Melkumov grid (1-4 nm) we
    are only mildly upsampling onto the ~1 nm simulation grid, so smoothness buys
    nothing. Log-linear is monotone between nodes by construction → zero
    overshoot near the narrow 976 nm peak (FWHM 7.7 nm) and the 1030 nm emission
    shoulder, robust across the ~4-decade absorption tail, and predictable.

    σ is clamped to a small positive floor before log() so any zero/sub-floor
    values (e.g. a McCumber-derived long-λ tail in some other dataset) cannot
    break log(); the result is then exp()'d back to linear space.
    """
    log_nodes = nodes_m
    log_a = np.interp(targets_m, log_nodes, np.log(np.maximum(sigma_a, _SIGMA_FLOOR_M2)))
    log_e = np.interp(targets_m, log_nodes, np.log(np.maximum(sigma_e, _SIGMA_FLOOR_M2)))
    return np.exp(log_a), np.exp(log_e)


def interpolate_to(target_lambdas_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate σ_a and σ_e onto the given wavelength grid (in metres).

    Interpolation is piecewise log-linear (see :func:`_log_linear_interp`).
    Raises ValueError if any target wavelength is outside the table (no
    extrapolation — see docs/CHANGELOG.md #1, the "no gain" disaster).
    """
    lambdas_m, sigma_a, sigma_e = load_yb_cross_sections()
    targets = np.atleast_1d(np.asarray(target_lambdas_m, dtype=float))
    lo, hi = lambdas_m[0], lambdas_m[-1]
    if targets.min() < lo or targets.max() > hi:
        bad = targets[(targets < lo) | (targets > hi)]
        raise ValueError(
            f"Wavelength(s) {bad * 1e9} nm outside Yb cross-section table range "
            f"{lo * 1e9:.1f}-{hi * 1e9:.1f} nm"
        )
    return _log_linear_interp(targets, lambdas_m, sigma_a, sigma_e)


def at(lambda_m: float) -> tuple[float, float]:
    """Cross-sections at a single wavelength. Returns (sigma_a, sigma_e) in m²."""
    sa, se = interpolate_to(np.array([lambda_m]))
    return float(sa[0]), float(se[0])


def regenerate_csv(
    path: str | Path = _DATA_PATH, raw_path: str | Path = _RAW_PATH
) -> Path:
    """Convert the published pm² provenance table to the m² CSV.

    Reads ``melkumov_AS_raw.csv`` (published σ in pm²), multiplies by
    ``1e-24`` to get m², and writes ``cross_sections_yb.csv`` on the measured
    grid AS-IS — no interpolation, no renormalisation. Deterministic.
    """
    raw = _read_csv(Path(raw_path))
    lam_nm = raw[:, 0]
    sa = raw[:, 1] * _PM2_TO_M2
    se = raw[:, 2] * _PM2_TO_M2

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(
            "# Yb-doped ALUMINOSILICATE silica fiber cross-sections (host match for\n"
            "# Coherent/Nufern SM-YDF-5/130-VIII). Measured data, NOT hand-traced.\n"
            "#\n"
            "# PRIMARY DATA SOURCE (full numeric table; NOT peer-reviewed preprint):\n"
            "#   M.A. Melkumov, I.A. Bufetov, K.S. Kravtsov, A.V. Shubin, E.M. Dianov,\n"
            "#   'Absorption and emission cross section of Yb3+ ions in Al2O3 and P2O5\n"
            "#   doped fibers', FORC Preprint No. 5, Moscow, 2004; arXiv:1502.02885,\n"
            "#   Appendix 2, aluminosilicate ('AC'/AS) columns.\n"
            "# PEER-REVIEWED VERSION (shorter, does NOT contain this table):\n"
            "#   Mel'kumov et al., 'Lasing parameters of ytterbium-doped fibres doped\n"
            "#   with P2O5 and Al2O3', Quantum Electronics 34(9), 843-848 (2004),\n"
            "#   DOI 10.1070/QE2004v034n09ABEH002688.\n"
            "# Host: Al2O3-silica (1-2 wt% Al, 1-3 wt% Yb, small GeO2), MCVD.\n"
            "# tau = 0.83e-3 s. T = 293 K. Absorption/emission peak (zero line) = 976.0 nm.\n"
            "# Central abs FWHM 7.7 nm. Peak sigma_a(976)=2.69e-24 m^2, sigma_e/sigma_a|peak=1.10.\n"
            "# Long-wavelength (>~1000 nm) absorption is McCumber-derived in the source.\n"
            "# Grid: ~4 nm spacing, 1 nm near the peak (968-986 nm). Range 848-1180 nm.\n"
            "# Converted from published pm^2 (melkumov_AS_raw.csv) by sigma_m2 = sigma_pm2 * 1e-24.\n"
            "# Regenerate with `python -m ase.cross_sections`.\n"
            "# Columns: lambda_nm, sigma_a_m2, sigma_e_m2\n"
        )
        f.write("lambda_nm,sigma_a_m2,sigma_e_m2\n")
        for lam, a, e in zip(lam_nm, sa, se):
            f.write(f"{lam:.1f},{a:.6e},{e:.6e}\n")
    return out


if __name__ == "__main__":  # pragma: no cover
    path = regenerate_csv()
    print(f"Wrote {path}")
