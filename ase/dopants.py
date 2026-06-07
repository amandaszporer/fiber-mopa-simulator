"""
Rare-earth dopant data — cross-section spectra and upper-state lifetime.

`DopantData` carries everything the BVP solver needs to know about the active
ion species: the absorption and emission cross-sections as a function of
wavelength, the upper-state lifetime τ, and a name. Any new dopant can be
added by registering a `DopantData` instance in `DOPANT_REGISTRY`.

The simulator ships with measured Yb aluminosilicate data from Melkumov et al.,
arXiv:1502.02885 (FORC Preprint No. 5, 2004), Appendix 2 — see
``ase/data/cross_sections_yb.csv``. Future Er, Tm, Nd, Ho support is just a
matter of adding the right CSV and registering it here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cross_sections


@dataclass(frozen=True)
class DopantData:
    """Cross-section spectra and lifetime for a rare-earth dopant.

    The wavelength array must be strictly increasing. The interpolation methods
    use piecewise log-linear interpolation (linear in log σ between nodes), the
    same policy as :mod:`ase.cross_sections` — see
    :func:`ase.cross_sections._log_linear_interp` for the rationale. Both
    cross-sections span several orders of magnitude across the Yb band, so
    log-space avoids negative artefacts; linear-in-log is monotone between the
    dense measured nodes and cannot overshoot.
    """

    name: str
    tau: float                          # upper-state lifetime [s]
    wavelengths: np.ndarray             # [m], shape (N,)
    sigma_absorption: np.ndarray        # [m²], shape (N,)
    sigma_emission: np.ndarray          # [m²], shape (N,)

    def __post_init__(self) -> None:
        if self.wavelengths.shape != self.sigma_absorption.shape:
            raise ValueError("σ_a array shape must match wavelengths")
        if self.wavelengths.shape != self.sigma_emission.shape:
            raise ValueError("σ_e array shape must match wavelengths")
        if not np.all(np.diff(self.wavelengths) > 0):
            raise ValueError("wavelengths must be strictly increasing")
        if self.tau <= 0:
            raise ValueError(f"tau must be positive, got {self.tau!r}")

    def sigma_a_at(self, lam: float) -> float:
        """Interpolated absorption cross-section at wavelength λ [m].

        Raises ValueError if λ is outside the dopant's tabulated range —
        log-space extrapolation produces unphysical values, so we fail loudly
        rather than letting nonsense propagate downstream.
        """
        return float(self.interpolate_to(np.array([lam]))[0][0])

    def sigma_e_at(self, lam: float) -> float:
        """Interpolated emission cross-section at wavelength λ [m].

        Raises ValueError if λ is outside the dopant's tabulated range.
        """
        return float(self.interpolate_to(np.array([lam]))[1][0])

    def interpolate_to(
        self, target_lambdas_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bulk interpolation onto an arbitrary wavelength grid.

        Uses the same piecewise log-linear policy as
        :func:`ase.cross_sections._log_linear_interp`. Raises ValueError if any
        target wavelength is outside the dopant's tabulated range.
        """
        self._check_in_range(target_lambdas_m)
        targets = np.atleast_1d(np.asarray(target_lambdas_m, dtype=float))
        return cross_sections._log_linear_interp(
            targets, self.wavelengths, self.sigma_absorption, self.sigma_emission
        )

    def _check_in_range(self, lam) -> None:
        lam_arr = np.atleast_1d(np.asarray(lam, dtype=float))
        lo = float(self.wavelengths[0])
        hi = float(self.wavelengths[-1])
        if lam_arr.min() < lo or lam_arr.max() > hi:
            bad = lam_arr[(lam_arr < lo) | (lam_arr > hi)]
            raise ValueError(
                f"Wavelength(s) {bad * 1e9} nm outside {self.name} dopant data "
                f"range {lo * 1e9:.1f}-{hi * 1e9:.1f} nm. "
                f"Extend ase/data/cross_sections_{self.name.lower()}.csv "
                f"or pick a pump/signal wavelength inside the table."
            )

    @classmethod
    def from_csv(
        cls, name: str, tau: float, csv_path: str | Path
    ) -> "DopantData":
        """Load a dopant from a CSV file.

        The CSV must have columns ``lambda_nm,sigma_a_m2,sigma_e_m2`` (the same
        format as ``ase/data/cross_sections_yb.csv``). Comment lines starting
        with ``#`` and any non-numeric header row are skipped.
        """
        path = Path(csv_path)
        rows: list[list[float]] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rows.append([float(x) for x in line.split(",")])
                except ValueError:
                    continue
        raw = np.asarray(rows)
        return cls(
            name=name,
            tau=tau,
            wavelengths=raw[:, 0] * 1e-9,
            sigma_absorption=raw[:, 1],
            sigma_emission=raw[:, 2],
        )


def _build_yb() -> DopantData:
    lambdas, sa, se = cross_sections.load_yb_cross_sections()
    return DopantData(
        name="Yb",
        # Melkumov AS (aluminosilicate) upper-state lifetime, arXiv:1502.02885.
        # (Was 0.84e-3 with the old Paschotta germanosilicate anchors.)
        tau=0.83e-3,
        wavelengths=lambdas,
        sigma_absorption=sa,
        sigma_emission=se,
    )


# Built-in dopants. To add a new one, drop a CSV under ase/data/ and append to
# this dict. Do NOT mutate after import; tests rely on the Yb entry's identity.
DOPANT_REGISTRY: dict[str, DopantData] = {
    "Yb": _build_yb(),
}


def get_dopant(name: str) -> DopantData:
    """Look up a dopant by name. Raises KeyError if unknown."""
    if name not in DOPANT_REGISTRY:
        raise KeyError(
            f"Unknown dopant {name!r}. Available: {sorted(DOPANT_REGISTRY)}"
        )
    return DOPANT_REGISTRY[name]
