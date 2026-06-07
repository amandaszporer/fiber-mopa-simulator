"""Cross-section CSV + interpolation tests.

These validate the *physical shape* of the measured Melkumov AS dataset
(arXiv:1502.02885, Appendix 2), not any prior numeric output. The previous
hand-traced Paschotta anchors had the 1000-1030 nm emission ordering inverted;
these tests pin the corrected shape.
"""

import numpy as np
import pytest

from ase import cross_sections
from ase.dopants import get_dopant


def _e(lam_nm):
    return cross_sections.at(lam_nm * 1e-9)[1]


def _a(lam_nm):
    return cross_sections.at(lam_nm * 1e-9)[0]


def test_csv_loads():
    lambdas, sa, se = cross_sections.load_yb_cross_sections()
    # Measured Melkumov AS grid: 848-1180 nm, 98 rows (4 nm spacing, 1 nm at peak).
    assert lambdas.shape == (98,)
    assert sa.shape == (98,)
    assert se.shape == (98,)
    assert lambdas[0] == pytest.approx(848e-9)
    assert lambdas[-1] == pytest.approx(1180e-9)


def test_measured_nodes_round_trip():
    """Interpolating at a tabulated wavelength returns that row's value exactly
    (log-linear interpolation passes through every measured node)."""
    lambdas, sa, se = cross_sections.load_yb_cross_sections()
    got_a, got_e = cross_sections.interpolate_to(lambdas)
    assert np.allclose(got_a, sa, rtol=1e-12)
    assert np.allclose(got_e, se, rtol=1e-12)


def test_emission_shoulder_ordering():
    """The bug being fixed: σ_e must INCREASE 1000 → 1010 → 1030 nm, the real
    Yb:silica emission shoulder peaking past 1020 nm."""
    assert _e(1030) > _e(1010) > _e(1000)


def test_emission_shoulder_ratios():
    assert _e(1030) / _e(1000) == pytest.approx(1.7, abs=0.3)
    assert _e(1030) / _e(1010) == pytest.approx(1.24, abs=0.2)


def test_zero_line_at_976():
    """At the zero line σ_a ≈ σ_e, with σ_e/σ_a ≈ 1.10 (Melkumov AS)."""
    sa, se = cross_sections.at(976e-9)
    assert se / sa == pytest.approx(1.10, abs=0.1)


def test_absorption_drop_and_four_level_at_1064():
    """σ_a falls steeply 1000 → 1064 nm (≳20×); at 1064 nm σ_e ≫ σ_a (≳50×),
    i.e. Yb is nearly four-level there."""
    assert _a(1000) / _a(1064) >= 20.0
    sa, se = cross_sections.at(1064e-9)
    assert se / sa >= 50.0


def test_no_negative_or_nan_on_sim_grid():
    """No negative / NaN σ anywhere on the simulation grid."""
    grid = np.arange(970.5, 1130.0, 1.0) * 1e-9
    sa, se = cross_sections.interpolate_to(grid)
    assert np.all(sa > 0) and np.all(se > 0)
    assert not np.any(np.isnan(sa)) and not np.any(np.isnan(se))


def test_absorption_tail_monotone():
    """The far absorption tail (>~1080 nm) decays monotonically."""
    lams = np.linspace(1100e-9, 1180e-9, 41)
    sa, _ = cross_sections.interpolate_to(lams)
    assert np.all(np.diff(sa) <= 0)


def test_dopant_and_module_interpolation_agree():
    """ase.cross_sections and ase.dopants must use the identical policy."""
    yb = get_dopant("Yb")
    lams = np.linspace(976e-9, 1064e-9, 50)
    m_a, m_e = cross_sections.interpolate_to(lams)
    d_a, d_e = yb.interpolate_to(lams)
    assert np.allclose(m_a, d_a) and np.allclose(m_e, d_e)


def test_out_of_range_raises():
    """Out-of-table lookups must raise, not silently extrapolate."""
    yb = get_dopant("Yb")
    with pytest.raises(ValueError, match="outside .* dopant data range"):
        yb.sigma_a_at(800e-9)
    with pytest.raises(ValueError, match="outside .* dopant data range"):
        yb.sigma_e_at(1300e-9)
    with pytest.raises(ValueError, match="outside .* dopant data range"):
        yb.interpolate_to(np.array([1000e-9, 1500e-9]))
    with pytest.raises(ValueError, match="outside Yb cross-section table range"):
        cross_sections.at(800e-9)
