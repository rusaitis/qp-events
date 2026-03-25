"""Power ratio computation (Equation 4 from the paper).

    r_i = P(b_i) / P(<B_T>_f)

where r_i is the ratio of power in the i-th MFA component to the power
in the smoothed background total field, at each frequency f.

This is the key metric for identifying QP30/QP60/QP120 events:
ratios >> 1 indicate wave power above the background.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from qp.signal.fft import welch_psd, estimate_background


def compute_power_ratios(
    b_par: ArrayLike,
    b_perp1: ArrayLike,
    b_perp2: ArrayLike,
    b_total: ArrayLike,
    dt: float = 60.0,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    nfft: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute power ratios r_i for MFA field components.

    Parameters
    ----------
    b_par, b_perp1, b_perp2 : array_like
        MFA magnetic field perturbation components.
    b_total : array_like
        Total field magnitude perturbations.
    dt : float
        Sampling interval in seconds.
    nperseg, noverlap, window, nfft
        Welch PSD parameters (passed through).

    Returns
    -------
    dict with keys:
        'freq' : frequency array (Hz)
        'psd_par', 'psd_perp1', 'psd_perp2', 'psd_total' : raw PSDs
        'background' : smoothed background estimate of total field PSD
        'r_par', 'r_perp1', 'r_perp2', 'r_total' : power ratios
    """
    welch_kw = dict(dt=dt, nperseg=nperseg, noverlap=noverlap, window=window, nfft=nfft)

    freq, psd_par = welch_psd(b_par, **welch_kw)
    _, psd_perp1 = welch_psd(b_perp1, **welch_kw)
    _, psd_perp2 = welch_psd(b_perp2, **welch_kw)
    _, psd_total = welch_psd(b_total, **welch_kw)

    # Background estimate of total field (smoothed in log-log space)
    background = estimate_background(psd_total, freq)

    # Power ratios (Eq 4)
    # Avoid division by zero
    bg_safe = np.where(background > 0, background, 1e-30)
    r_par = psd_par / bg_safe
    r_perp1 = psd_perp1 / bg_safe
    r_perp2 = psd_perp2 / bg_safe
    r_total = psd_total / bg_safe

    return {
        "freq": freq,
        "psd_par": psd_par,
        "psd_perp1": psd_perp1,
        "psd_perp2": psd_perp2,
        "psd_total": psd_total,
        "background": background,
        "r_par": r_par,
        "r_perp1": r_perp1,
        "r_perp2": r_perp2,
        "r_total": r_total,
    }


def freq_to_period_minutes(freq: ArrayLike) -> np.ndarray:
    """Convert frequency (Hz) to period (minutes)."""
    freq = np.asarray(freq)
    return np.where(freq > 0, 1.0 / (freq * 60), np.inf)
