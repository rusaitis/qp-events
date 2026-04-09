r"""Phase 7.4 — Composite quality score per event.

Replaces the binary accept/reject gate with a continuous quality metric
$q \in [0, 1]$ that combines multiple independent detection indicators.
Events can then be filtered post-hoc by ``quality > threshold`` without
re-running the sweep.

Metrics combined (all normalized to [0, 1]):
- ``wavelet_sigma``: how many σ above background at the CWT ridge peak
- ``fft_screen_ratio``: power-law FFT ratio at the event frequency
- ``mf_snr``: matched-filter SNR
- ``coherence``: wavelet coherence between b_perp1 and b_perp2
- ``n_oscillations``: number of periods in the event window
- ``transverse_ratio``: (perp1² + perp2²) / par² — should be ≫ 1
- ``polarization_fraction``: from Stokes parameters
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class QualityConfig:
    r"""Normalization percentile anchors for quality scoring.

    Each metric is mapped to [0, 1] using a sigmoid:
    $q_i = 1 / (1 + \exp(-k (x - x_{50})))$

    where $x_{50}$ and $k$ are derived from the ``p10`` and ``p90``
    values: the 10th percentile maps to ~0.1, the 90th to ~0.9.
    """
    # (p10, p90) anchors — calibrated on v2 mission catalog (1636 events)
    wavelet_sigma: tuple[float, float] = (0.5, 9.0)
    fft_ratio: tuple[float, float] = (0.1, 2.0)
    mf_snr: tuple[float, float] = (1.7, 16.0)
    coherence: tuple[float, float] = (0.12, 0.66)
    n_oscillations: tuple[float, float] = (3.0, 10.0)
    transverse_ratio: tuple[float, float] = (0.001, 0.65)
    polarization_fraction: tuple[float, float] = (0.3, 0.9)


DEFAULT_QUALITY_CONFIG = QualityConfig()


def _sigmoid_normalize(x: float, p10: float, p90: float) -> float:
    r"""Map a raw metric to [0, 1] via a logistic sigmoid.

    Anchored so that p10 → 0.1, p90 → 0.9.
    """
    if p90 <= p10:
        return 0.5
    # Solve: 1/(1+exp(-k*(p10 - x50))) = 0.1 → k*(p10-x50) = -ln(9)
    # and:  1/(1+exp(-k*(p90 - x50))) = 0.9 → k*(p90-x50) = ln(9)
    # ⇒ k = 2*ln(9) / (p90-p10), x50 = (p10+p90)/2
    k = 2 * np.log(9.0) / (p90 - p10)
    x50 = (p10 + p90) / 2.0
    return float(1.0 / (1.0 + np.exp(-k * (x - x50))))


def compute_quality(
    wavelet_sigma: float | None = None,
    fft_ratio: float | None = None,
    mf_snr: float | None = None,
    coherence: float | None = None,
    n_oscillations: float | None = None,
    transverse_ratio: float | None = None,
    polarization_fraction: float | None = None,
    config: QualityConfig | None = None,
) -> float:
    r"""Compute the composite quality score for one event.

    Returns the geometric mean of all available normalized metrics.
    Missing metrics (None) are excluded from the product — the score
    degrades gracefully when not all metrics are available.
    """
    if config is None:
        config = DEFAULT_QUALITY_CONFIG

    parts: list[float] = []
    if wavelet_sigma is not None:
        parts.append(_sigmoid_normalize(wavelet_sigma, *config.wavelet_sigma))
    if fft_ratio is not None:
        parts.append(_sigmoid_normalize(fft_ratio, *config.fft_ratio))
    if mf_snr is not None:
        parts.append(_sigmoid_normalize(mf_snr, *config.mf_snr))
    if coherence is not None:
        parts.append(_sigmoid_normalize(coherence, *config.coherence))
    if n_oscillations is not None:
        parts.append(_sigmoid_normalize(n_oscillations,
                                         *config.n_oscillations))
    if transverse_ratio is not None:
        parts.append(_sigmoid_normalize(transverse_ratio,
                                         *config.transverse_ratio))
    if polarization_fraction is not None:
        parts.append(_sigmoid_normalize(polarization_fraction,
                                         *config.polarization_fraction))

    if not parts:
        return 0.0
    return float(np.exp(np.mean(np.log(np.clip(parts, 1e-10, 1.0)))))


def quality_scores_array(
    catalog_df,
    config: QualityConfig | None = None,
) -> NDArray[np.floating]:
    r"""Compute quality scores for an entire catalog DataFrame.

    Parameters
    ----------
    catalog_df : pandas.DataFrame
        Event catalog with Phase 7 columns.
    config : QualityConfig, optional

    Returns
    -------
    scores : ndarray, shape (n_events,)
    """
    n = len(catalog_df)
    scores = np.zeros(n, dtype=float)

    def _col(name):
        if name in catalog_df.columns:
            return catalog_df[name].values
        return np.full(n, np.nan)

    ws = _col("wavelet_sigma")
    fr = _col("fft_screen_ratio")
    ms = _col("mf_snr")
    co = _col("coherence")
    no = _col("n_oscillations")
    tr = _col("transverse_ratio")
    pf = _col("polarization_fraction")

    for i in range(n):
        scores[i] = compute_quality(
            wavelet_sigma=ws[i] if np.isfinite(ws[i]) else None,
            fft_ratio=fr[i] if np.isfinite(fr[i]) else None,
            mf_snr=ms[i] if np.isfinite(ms[i]) else None,
            coherence=co[i] if np.isfinite(co[i]) else None,
            n_oscillations=no[i] if np.isfinite(no[i]) else None,
            transverse_ratio=tr[i] if np.isfinite(tr[i]) else None,
            polarization_fraction=pf[i] if np.isfinite(pf[i]) else None,
            config=config,
        )
    return scores
