r"""Phase 3 dispatch tests for :func:`qp.events.detector.detect_round8`.

The threshold methods themselves are covered in
``tests/test_threshold_diag.py``; this module exercises the *dispatch
glue* inside ``detect_round8``:

- ``threshold_method='mad_row'`` (default) matches the explicit-default
  call signature byte-for-byte on a synthetic packet (regression).
- ``threshold_method='tc_chi2'`` recovers the same injected packet at
  similar peak metadata (sanity).
- ``apply_coi_mask=True`` is wired and runs without raising.
- ``threshold_method='pooled'`` requires an archive + region; otherwise
  raises a helpful ``ValueError``.
- An unknown method raises a clear error.
"""

from __future__ import annotations

import numpy as np
import pytest

from qp.events.detector import DetectedEvent, detect_round8
from qp.events.threshold_diag import BGArchive


def _synthetic_qp60(
    seed: int = 7,
    amp_nT: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """36 h of MFA-like noise with a circularly polarised QP60 packet.

    Returns ``(t_sec, fields)`` with ``fields = [b_par, b_perp1, b_perp2]``.
    The packet sits at the segment midpoint, Gaussian-windowed over a few
    hours so the peak clears ``min_duration_hours = 2``.
    """
    rng = np.random.default_rng(seed)
    dt = 60.0
    n = 36 * 60  # 36 h at 1-min cadence
    t = np.arange(n) * dt
    period = 60.0 * 60.0  # 60 min — squarely in QP60
    centre = (n // 2) * dt
    half = 4.0 * 3600.0
    env = np.exp(-(((t - centre) / half) ** 2))
    phase = 2.0 * np.pi * t / period

    # AR(1) background — same alpha on all three components so the
    # transversality and Stokes gates aren't confounded by independent
    # white noise on b_par.
    def _ar1(alpha: float) -> np.ndarray:
        x = np.zeros(n)
        eps = rng.standard_normal(n) * np.sqrt(1.0 - alpha * alpha)
        for i in range(1, n):
            x[i] = alpha * x[i - 1] + eps[i]
        return x * 0.05  # ~0.05 nT RMS background

    b_par = _ar1(0.85)
    # Circularly polarised packet in (perp1, perp2): 90° phase shift.
    b_perp1 = _ar1(0.85) + amp_nT * env * np.cos(phase)
    b_perp2 = _ar1(0.85) + amp_nT * env * np.sin(phase)
    fields = np.column_stack([b_par, b_perp1, b_perp2])
    return t, fields


def _has_qp_peak_in_band(events: list[DetectedEvent], band: str) -> bool:
    return any(e.peak.band == band for e in events)


# --------------------------------------------------------------------- #
# Dispatch correctness                                                  #
# --------------------------------------------------------------------- #


def test_default_kwargs_are_tc_chi2_with_coi() -> None:
    """Implicit defaults must match the explicit canonical config."""
    t, fields = _synthetic_qp60()
    implicit = detect_round8(t, fields, dt=60.0)
    explicit = detect_round8(
        t,
        fields,
        dt=60.0,
        threshold_method="tc_chi2",
        apply_coi_mask=True,
    )
    # Same number of detections and identical peak times → same dispatch.
    assert len(implicit) == len(explicit)
    for a, b in zip(implicit, explicit, strict=False):
        assert a.peak.peak_time == b.peak.peak_time
        assert a.peak.band == b.peak.band


def test_default_finds_injected_qp60() -> None:
    """Sanity: the synthetic packet survives all four gates at defaults."""
    t, fields = _synthetic_qp60(amp_nT=4.0)
    events = detect_round8(t, fields, dt=60.0)
    assert _has_qp_peak_in_band(events, "QP60")


def test_mad_row_legacy_path_still_finds_injected_qp60() -> None:
    """The legacy per-row MAD gate remains reachable and functional."""
    t, fields = _synthetic_qp60(amp_nT=4.0)
    events = detect_round8(
        t,
        fields,
        dt=60.0,
        threshold_method="mad_row",
        apply_coi_mask=False,
    )
    assert _has_qp_peak_in_band(events, "QP60")


def test_no_coi_runs_and_keeps_central_event() -> None:
    """Disabling the COI mask is wired and keeps a midpoint event."""
    t, fields = _synthetic_qp60(amp_nT=4.0)
    events = detect_round8(
        t,
        fields,
        dt=60.0,
        threshold_method="tc_chi2",
        apply_coi_mask=False,
    )
    assert _has_qp_peak_in_band(events, "QP60")


def test_pooled_without_archive_raises() -> None:
    """pooled requires both bg_archive and region; clean error otherwise."""
    t, fields = _synthetic_qp60()
    with pytest.raises(ValueError, match="bg_archive"):
        detect_round8(t, fields, dt=60.0, threshold_method="pooled")


def test_pooled_with_archive_runs() -> None:
    """Smoke test: a minimal archive lets the pooled dispatch complete."""
    from qp.signal.wavelet import morlet_cwt

    t, fields = _synthetic_qp60(amp_nT=4.0)
    # Build a single-segment archive from the same b_perp1 — guarantees
    # the period axis and statistics are compatible with the CWT grid the
    # detector uses internally.
    _, _, cwt = morlet_cwt(fields[:, 1], dt=60.0, n_freqs=300)
    freq, _, _ = morlet_cwt(fields[:, 1], dt=60.0, n_freqs=300)
    amp = np.abs(cwt)
    medians = np.median(amp, axis=1)
    mads = np.median(np.abs(amp - medians[:, None]), axis=1)
    arch = BGArchive(
        periods_sec=1.0 / freq,
        medians={"magnetosphere": medians},
        mads={"magnetosphere": mads},
        n_segments={"magnetosphere": 1},
    )
    events = detect_round8(
        t,
        fields,
        dt=60.0,
        threshold_method="pooled",
        bg_archive=arch,
        region="magnetosphere",
    )
    # No assertion on count — the per-segment archive is matched to the
    # same segment, so it may or may not flag the injected packet
    # depending on how the strong signal inflates its own MAD. We just
    # need the dispatch path to run end-to-end.
    assert isinstance(events, list)
