"""Synthetic-event generators for the web app's Synthetic tab.

Two surfaces:

* :func:`generate` — interactive, parameterized single-event generator
  using :func:`qp.signal.synthetic.simulate_multi_component`.
* :func:`benchmark` — small canned ensembles (low / med / high SNR) per
  band, run through :func:`qp.events.detector.detect_round8` and tagged
  TP / miss for review.
"""

from __future__ import annotations

import datetime
from functools import lru_cache
from typing import Any

import numpy as np

from qp.events.bands import QP_BAND_NAMES, QP_BANDS
from qp.events.catalog import WaveTemplate
from qp.events.detector import detect_round8
from qp.signal.fft import welch_psd
from qp.signal.synthetic import simulate_multi_component


_DEFAULT_PERIOD_MIN: dict[str, float] = {
    name: band.period_centroid_minutes for name, band in QP_BANDS.items()
}


def _epoch() -> datetime.datetime:
    return datetime.datetime(2000, 1, 1)


def _spectrum(fields: np.ndarray, dt: float = 60.0) -> dict[str, list[float]]:
    nperseg = min(fields.shape[0], 12 * 60)
    freq, psd_par = welch_psd(fields[:, 0], dt=dt, nperseg=nperseg)
    _, psd_p1 = welch_psd(fields[:, 1], dt=dt, nperseg=nperseg)
    _, psd_p2 = welch_psd(fields[:, 2], dt=dt, nperseg=nperseg)
    keep = freq > 0
    f = freq[keep]
    period_min = 1.0 / f / 60.0
    order = np.argsort(period_min)
    return {
        "freq_hz": f[order].tolist(),
        "period_min": period_min[order].tolist(),
        "psd_par": psd_par[keep][order].tolist(),
        "psd_perp1": psd_p1[keep][order].tolist(),
        "psd_perp2": psd_p2[keep][order].tolist(),
        "qp_periods_min": [_DEFAULT_PERIOD_MIN[b] for b in QP_BAND_NAMES],
    }


def generate(
    band: str = "QP60",
    amplitude: float = 2.0,
    period_min: float | None = None,
    decay_h: float = 4.0,
    noise_sigma: float = 0.3,
    seed: int = 0,
    n_hours: float = 36.0,
) -> dict[str, Any]:
    """Build a single synthetic 3-component event + spectrum."""
    if period_min is None:
        period_min = _DEFAULT_PERIOD_MIN.get(band, 60.0)
    n_samples = int(round(n_hours * 60))
    template = WaveTemplate(
        period=period_min * 60.0,
        amplitude=amplitude,
        phase=0.0,
        waveform="sine",
        decay_width=decay_h * 3600.0,
        shift=n_samples * 60.0 / 2.0,
    )
    t, fields = simulate_multi_component(
        n_samples=n_samples,
        dt=60.0,
        waves=[template],
        noise_sigma=noise_sigma,
        seed=seed,
    )
    epoch = _epoch()
    times = [epoch + datetime.timedelta(seconds=float(s)) for s in t]
    return {
        "band": band,
        "amplitude_nT": amplitude,
        "period_min": period_min,
        "decay_h": decay_h,
        "noise_sigma_nT": noise_sigma,
        "seed": seed,
        "times": [tt.isoformat() for tt in times],
        "epoch_s": t.tolist(),
        "b_par": fields[:, 0].tolist(),
        "b_perp1": fields[:, 1].tolist(),
        "b_perp2": fields[:, 2].tolist(),
        "b_tot": fields[:, 3].tolist(),
        "spectrum": _spectrum(fields),
    }


# --------------------------------------------------------------------- #
# Benchmark presets                                                     #
# --------------------------------------------------------------------- #


_PRESETS: dict[str, dict[str, Any]] = {
    "low_snr": {"amps": [0.4, 0.6, 0.8, 1.0], "noise": 0.6},
    "med_snr": {"amps": [0.8, 1.2, 1.6, 2.0], "noise": 0.5},
    "high_snr": {"amps": [1.5, 2.5, 3.5, 5.0], "noise": 0.4},
}
_PRESET_BANDS: tuple[str, ...] = QP_BAND_NAMES
_PRESET_SEEDS: tuple[int, ...] = (0, 1, 2)


@lru_cache(maxsize=4)
def benchmark(preset: str = "med_snr") -> dict[str, Any]:
    """Run the detector across a fixed (band x amp x seed) grid.

    Each row is one synthetic event; ``detected`` flags whether
    `detect_round8` found it (i.e. true positive). Cached per preset.
    """
    if preset not in _PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    cfg = _PRESETS[preset]
    rows: list[dict[str, Any]] = []
    n_samples = 36 * 60
    epoch = _epoch()
    t = np.arange(n_samples, dtype=float) * 60.0
    for band in _PRESET_BANDS:
        period_min = _DEFAULT_PERIOD_MIN[band]
        for amp in cfg["amps"]:
            for seed in _PRESET_SEEDS:
                template = WaveTemplate(
                    period=period_min * 60.0,
                    amplitude=amp,
                    decay_width=4 * 3600.0,
                    shift=n_samples * 60.0 / 2.0,
                )
                _, fields = simulate_multi_component(
                    n_samples=n_samples,
                    dt=60.0,
                    waves=[template],
                    noise_sigma=cfg["noise"],
                    seed=seed,
                )
                detected = False
                detected_band: str | None = None
                period_detected_min: float | None = None
                try:
                    events = detect_round8(t, fields[:, :3], dt=60.0, epoch=epoch)
                    target = QP_BANDS[band]
                    for ev in events:
                        peak = ev.peak
                        ev_band_name = peak.band
                        period_sec = peak.period_sec
                        ev_period_min = float(period_sec) / 60.0 if period_sec else None
                        in_target = (
                            ev_period_min is not None
                            and target.period_min_minutes
                            <= ev_period_min
                            <= target.period_max_minutes
                        )
                        if ev_band_name == band or in_target:
                            detected = True
                            detected_band = ev_band_name or band
                            period_detected_min = ev_period_min
                            break
                except Exception as exc:  # noqa: BLE001
                    rows.append(
                        {
                            "band": band,
                            "amplitude_nT": amp,
                            "seed": seed,
                            "noise_sigma_nT": cfg["noise"],
                            "detected": False,
                            "error": str(exc),
                        }
                    )
                    continue
                rows.append(
                    {
                        "band": band,
                        "amplitude_nT": amp,
                        "seed": seed,
                        "noise_sigma_nT": cfg["noise"],
                        "detected": detected,
                        "detected_band": detected_band,
                        "detected_period_min": period_detected_min,
                    }
                )
    by_band = {b: {"n": 0, "tp": 0} for b in _PRESET_BANDS}
    for r in rows:
        b = r["band"]
        by_band[b]["n"] += 1
        if r.get("detected"):
            by_band[b]["tp"] += 1
    summary = {
        b: {
            **stats,
            "recall": (stats["tp"] / stats["n"]) if stats["n"] else 0.0,
        }
        for b, stats in by_band.items()
    }
    return {
        "preset": preset,
        "noise_sigma_nT": cfg["noise"],
        "amplitudes_nT": cfg["amps"],
        "bands": list(_PRESET_BANDS),
        "seeds": list(_PRESET_SEEDS),
        "n_total": len(rows),
        "summary": summary,
        "rows": rows,
    }


def benchmark_event(
    preset: str,
    band: str,
    amplitude: float,
    seed: int,
) -> dict[str, Any]:
    """Re-render a specific benchmark event for plotting."""
    if preset not in _PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    cfg = _PRESETS[preset]
    out = generate(
        band=band,
        amplitude=amplitude,
        period_min=_DEFAULT_PERIOD_MIN[band],
        decay_h=4.0,
        noise_sigma=cfg["noise"],
        seed=seed,
    )
    out["preset"] = preset
    return out
