"""
Plotting helpers — render the standard 4 PNG report figures from a `Simulator`.

These functions are pulled out of the simulation logic so the framework itself
has zero matplotlib dependency. The example script and the `simulate.py`
CLI call into these only when `--plots` is requested.

Each function takes a `Simulator` (already `run()`-ed) and an output directory.
Use `plot_all(sim, dir)` to write all four at once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_power_evolution(sim, out_dir: str | Path) -> Path:
    """Avg power, peak power, and integrated forward ASE at every component."""
    plt = _import_plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = ["Seed"] + [r.component.name for r in sim.results]
    avg = [sim.seed.average_power] + [r.state_out.signal.average_power for r in sim.results]
    peak = [sim.seed.peak_power] + [r.state_out.signal.peak_power for r in sim.results]
    ase = [0.0] + [
        (r.state_out.ase.total_fwd() if r.state_out.ase is not None else 0.0)
        for r in sim.results
    ]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = range(len(names))

    color_a = "tab:blue"
    ax1.semilogy(x, [max(v, 1e-12) for v in avg], "o-", color=color_a,
                 label="Average power")
    ax1.set_ylabel("Average / ASE Power [W]", color=color_a)
    ax1.set_xlabel("Component")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax1.tick_params(axis="y", labelcolor=color_a)
    color_c = "tab:orange"
    ax1.semilogy(x, [max(v, 1e-12) for v in ase], "^:", color=color_c,
                 label="ASE forward (integrated)")

    ax2 = ax1.twinx()
    color_b = "tab:red"
    ax2.semilogy(x, [max(v, 1e-12) for v in peak], "s--", color=color_b,
                 label="Peak power")
    ax2.set_ylabel("Peak Power [W]", color=color_b)
    ax2.tick_params(axis="y", labelcolor=color_b)

    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    fig.suptitle("Power Evolution Through MOPA Chain")
    fig.tight_layout()
    out_path = out_dir / "power_evolution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_amplifier_details(sim, out_dir: str | Path) -> Path:
    """Per-amplifier signal/pump/ASE evolution along z."""
    plt = _import_plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    amps = sim.amplifiers
    n = len(amps)
    if n == 0:
        return out_dir / "amplifier_details.png"

    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)

    for i, amp in enumerate(amps):
        ax = axes[i, 0]
        info = amp.info
        z_cm = np.asarray(info["z"]) * 100

        ax.semilogy(z_cm, np.maximum(info["P_signal_z"], 1e-30) * 1e3,
                    label="P_signal [mW]", color="tab:blue")
        ax.semilogy(z_cm, np.maximum(info["P_pump_z"], 1e-30) * 1e3,
                    label="P_pump [mW]", color="tab:green")
        ax.semilogy(z_cm, np.maximum(info["P_ase_z"], 1e-30) * 1e3,
                    label="P_ASE_fwd (integrated) [mW]",
                    color="tab:orange", linestyle="--")
        bwd_total = info["ase_bwd_spectrum_z"].sum(axis=1)
        ax.semilogy(z_cm, np.maximum(bwd_total, 1e-30) * 1e3,
                    label="P_ASE_bwd (integrated) [mW]",
                    color="tab:red", linestyle=":")

        ax.set_xlabel("Position [cm]")
        ax.set_ylabel("Power [mW]")
        ax.set_title(
            f"{amp.name} — {amp.core_diameter*1e6:.0f}/"
            f"{amp.clad_diameter*1e6:.0f} um, "
            f"L={amp.length:.1f} m, "
            f"Pump={amp.pump_power:.1f} W"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "amplifier_details.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_nonlinear_margins(sim, out_dir: str | Path) -> Path:
    """SBS / SRS ratios per amplifier with the threshold line at 1.0."""
    plt = _import_plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    amps = sim.amplifiers
    if not amps:
        return out_dir / "nonlinear_margins.png"

    names = [a.name for a in amps]
    sbs = [a.info["sbs_ratio"] for a in amps]
    srs = [a.info["srs_ratio"] for a in amps]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], sbs, width, label="SBS ratio",
           color="tab:blue")
    ax.bar([i + width / 2 for i in x], srs, width, label="SRS ratio",
           color="tab:orange")
    ax.axhline(y=1.0, color="red", linestyle="--", label="Threshold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Peak / Threshold ratio")
    ax.set_title("SBS & SRS Nonlinear Margins")
    ax.legend()

    fig.tight_layout()
    out_path = out_dir / "nonlinear_margins.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ase_spectra(sim, out_dir: str | Path) -> Path:
    """Forward and backward ASE spectra at the endpoints of each amplifier."""
    plt = _import_plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    amps = sim.amplifiers
    n = len(amps)
    if n == 0:
        return out_dir / "ase_spectra.png"

    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)

    for i, amp in enumerate(amps):
        ax = axes[i, 0]
        info = amp.info
        wl = info["wavelengths_nm"]
        fwd = info["ase_fwd_spectrum_W"]
        bwd = info["ase_bwd_spectrum_W"]

        d_lambda_nm = wl[1] - wl[0]
        fwd_dBm_per_nm = 10 * np.log10(np.maximum(fwd / d_lambda_nm * 1e3, 1e-15))
        bwd_dBm_per_nm = 10 * np.log10(np.maximum(bwd / d_lambda_nm * 1e3, 1e-15))

        ax.plot(wl, fwd_dBm_per_nm, color="tab:orange",
                label=f"Forward ASE @ z=L (Σ={info['ase_power_out']*1e6:.1f} uW)")
        ax.plot(wl, bwd_dBm_per_nm, color="tab:red", linestyle="--",
                label=f"Backward ASE @ z=0 (Σ={info['ase_power_bwd']*1e6:.1f} uW)")
        ax.axvline(1064.0, color="grey", linestyle=":", alpha=0.6,
                   label="Signal (1064 nm)")
        ax.axvline(1030.0, color="green", linestyle=":", alpha=0.4,
                   label="Yb gain peak (1030 nm)")
        ax.axvline(976.0, color="blue", linestyle=":", alpha=0.4,
                   label="Pump (976 nm)")

        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Spectral power density [dBm/nm]")
        ax.set_title(f"{amp.name} ASE Spectrum")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=max(-100,
                                min(fwd_dBm_per_nm.min(), bwd_dBm_per_nm.min()) - 5))

    fig.tight_layout()
    out_path = out_dir / "ase_spectra.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_bpf_spectra(sim, out_dir: str | Path) -> Path:
    """Forward ASE spectrum before and after each BandpassFilter in the chain."""
    plt = _import_plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from components import BandpassFilter
    bpf_stages = [
        (i, r) for i, r in enumerate(sim.results)
        if isinstance(r.component, BandpassFilter)
        and r.state_in.ase is not None and r.state_out.ase is not None
    ]
    if not bpf_stages:
        return out_dir / "bpf_spectra.png"

    n = len(bpf_stages)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), squeeze=False)

    for ax_row, (_, r) in enumerate(bpf_stages):
        ax = axes[ax_row, 0]
        bpf = r.component
        wl = r.state_in.ase.spectral_grid.wavelengths * 1e9
        fwd_in = r.state_in.ase.fwd_spectrum
        fwd_out = r.state_out.ase.fwd_spectrum

        d_lambda_nm = wl[1] - wl[0]
        in_dBm = 10 * np.log10(np.maximum(fwd_in / d_lambda_nm * 1e3, 1e-15))
        out_dBm = 10 * np.log10(np.maximum(fwd_out / d_lambda_nm * 1e3, 1e-15))

        ax.plot(wl, in_dBm, color="tab:orange", alpha=0.7,
                label=f"Before BPF (Σ={fwd_in.sum()*1e6:.2f} uW)")
        ax.plot(wl, out_dBm, color="tab:blue",
                label=f"After BPF  (Σ={fwd_out.sum()*1e6:.2f} uW)")
        ax.axvline(bpf.center_wavelength * 1e9, color="grey", linestyle=":",
                   alpha=0.6, label=f"BPF center ({bpf.center_wavelength*1e9:.1f} nm)")
        ax.axvline(1030.0, color="green", linestyle=":", alpha=0.4,
                   label="Yb gain peak (1030 nm)")

        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Spectral power density [dBm/nm]")
        ax.set_title(
            f"{bpf.name} — center {bpf.center_wavelength*1e9:.1f} nm, "
            f"FWHM {bpf.fwhm*1e9:.1f} nm, "
            f"peak loss {bpf.insertion_loss_dB:.2f} dB"
        )
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "bpf_spectra.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_pulse_shape(sim, out_dir: str | Path) -> Path:
    """Per-amplifier input vs output pulse temporal profile (B2 only).

    Only renders for amplifiers whose `info` carries `P_signal_out_t_W` —
    i.e., those that actually ran the time-dependent B2 path. Returns the
    output path even when no amp ran B2 (empty file), mirroring the
    no-op behaviour of `plot_bpf_spectra` when no BPFs are present.
    """
    plt = _import_plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    amps = sim.amplifiers
    b2_amps = [a for a in amps if a.info.get("P_signal_out_t_W") is not None]
    out_path = out_dir / "pulse_shape.png"
    if not b2_amps:
        return out_path

    n = len(b2_amps)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.5 * n), squeeze=False)

    for i, amp in enumerate(b2_amps):
        ax = axes[i, 0]
        t_ns = amp.info["t_ns"]
        p_in = amp.info["P_signal_in_t_W"]
        p_out = amp.info["P_signal_out_t_W"]

        ax.plot(t_ns, p_in, color="tab:blue", label=f"Input @ z=0 (peak {p_in.max():.2f} W)")
        ax.plot(t_ns, p_out, color="tab:red", label=f"Output @ z=L (peak {p_out.max():.0f} W)")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Signal power [W]")
        ax.set_title(f"{amp.name} pulse profile (Level 5 B2)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_all(sim, out_dir: str | Path) -> list[Path]:
    """Render every standard figure into `out_dir`. Returns the file paths."""
    return [
        plot_power_evolution(sim, out_dir),
        plot_amplifier_details(sim, out_dir),
        plot_nonlinear_margins(sim, out_dir),
        plot_ase_spectra(sim, out_dir),
        plot_bpf_spectra(sim, out_dir),
        plot_pulse_shape(sim, out_dir),
    ]
