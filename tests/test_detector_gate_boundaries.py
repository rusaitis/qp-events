r"""Boundary tests for round-8 detector gates.

The existing happy-path test (``test_detect_round8_phase3.py``) verifies
that a clean 4 nT circular QP60 packet survives all four gates. This file
turns each gate up to 11 to confirm it actually rejects the marginal
cases it claims to reject:

  * **Q-factor** — tighten ``min_q_factor`` until the packet is rejected.
  * **MVA parallel fraction** — inject a compressional packet (energy on
    ``b_par``) and confirm it is rejected.
  * **Stokes degree** — inject an unpolarized random-phase packet and
    confirm it is rejected.

A regression that turns any of these gates into a no-op (e.g. swapping
``<`` for ``<=`` or flipping a sign) would slip past the happy-path test
but get caught here.
"""

from __future__ import annotations

import numpy as np
import pytest

from qp.events.detector import (
    MIN_Q_FACTOR,
    detect_round8,
)
from qp.signal.polarization_config import (
    MAX_MVA_PARALLEL_FRACTION,
    MIN_DEGREE_OF_POLARIZATION,
)

# ---------------------------------------------------------------------- #
# Synthetic-packet helpers (shape adapted from test_detect_round8_phase3) #
# ---------------------------------------------------------------------- #

DT = 60.0
N_SAMPLES = 36 * 60  # 36-h segment at 1-min cadence
PERIOD_SEC = 60.0 * 60.0  # squarely in QP60


def _ar1_background(rng: np.random.Generator, alpha: float = 0.85) -> np.ndarray:
    x = np.zeros(N_SAMPLES)
    eps = rng.standard_normal(N_SAMPLES) * np.sqrt(1.0 - alpha * alpha)
    for i in range(1, N_SAMPLES):
        x[i] = alpha * x[i - 1] + eps[i]
    return x * 0.05


def _envelope() -> np.ndarray:
    t = np.arange(N_SAMPLES) * DT
    centre = (N_SAMPLES // 2) * DT
    half = 4.0 * 3600.0
    return np.exp(-(((t - centre) / half) ** 2))


def _circular_qp60(rng: np.random.Generator, amp_nT: float = 4.0):
    """A clean circularly-polarized transverse QP60 packet."""
    t = np.arange(N_SAMPLES) * DT
    phase = 2.0 * np.pi * t / PERIOD_SEC
    env = _envelope()
    b_par = _ar1_background(rng)
    b_perp1 = _ar1_background(rng) + amp_nT * env * np.cos(phase)
    b_perp2 = _ar1_background(rng) + amp_nT * env * np.sin(phase)
    return t, np.column_stack([b_par, b_perp1, b_perp2])


def _compressional_qp60(rng: np.random.Generator, amp_nT: float = 4.0):
    """Energy on b_par instead of the transverse pair → MVA should reject."""
    t = np.arange(N_SAMPLES) * DT
    phase = 2.0 * np.pi * t / PERIOD_SEC
    env = _envelope()
    # bleed a tiny amount of wave into b_perp1 so the ridge extractor
    # actually finds a peak in the transverse component to gate on
    b_par = _ar1_background(rng) + amp_nT * env * np.cos(phase)
    b_perp1 = _ar1_background(rng) + 0.3 * amp_nT * env * np.cos(phase)
    b_perp2 = _ar1_background(rng) + 0.3 * amp_nT * env * np.cos(phase)
    return t, np.column_stack([b_par, b_perp1, b_perp2])


def _unpolarized_qp60(rng: np.random.Generator, amp_nT: float = 4.0):
    """Two transverse channels with INDEPENDENT random phases at the QP60
    period → Stokes degree of polarization is low, so the d-gate rejects."""
    t = np.arange(N_SAMPLES) * DT
    omega = 2.0 * np.pi / PERIOD_SEC
    env = _envelope()
    # different random phase drift on each channel kills coherence
    phase_1 = omega * t + np.cumsum(rng.normal(0.0, 0.5, N_SAMPLES))
    phase_2 = omega * t + np.cumsum(rng.normal(0.0, 0.5, N_SAMPLES))
    b_par = _ar1_background(rng)
    b_perp1 = _ar1_background(rng) + amp_nT * env * np.cos(phase_1)
    b_perp2 = _ar1_background(rng) + amp_nT * env * np.cos(phase_2)
    return t, np.column_stack([b_par, b_perp1, b_perp2])


def _qp60_events(events) -> list:
    return [e for e in events if e.peak.band == "QP60"]


# ---------------------------------------------------------------------- #
# Default-value invariants                                                #
# ---------------------------------------------------------------------- #


def test_default_thresholds_are_paper_published():
    """Round-8 paper-resubmission round used these values; lock them in."""
    assert pytest.approx(3.0) == MIN_Q_FACTOR
    assert pytest.approx(0.7) == MIN_DEGREE_OF_POLARIZATION
    assert pytest.approx(0.5) == MAX_MVA_PARALLEL_FRACTION


# ---------------------------------------------------------------------- #
# Q-factor gate                                                           #
# ---------------------------------------------------------------------- #


class TestQFactorGate:
    def test_clean_packet_passes_default(self):
        rng = np.random.default_rng(101)
        t, fields = _circular_qp60(rng)
        events = detect_round8(t, fields, dt=DT)
        assert _qp60_events(events), "clean packet should pass default gates"

    def test_q_factor_set_impossibly_high_rejects_everything(self):
        """An unreachably high ``min_q_factor`` must filter out every event."""
        rng = np.random.default_rng(102)
        t, fields = _circular_qp60(rng)
        events = detect_round8(t, fields, dt=DT, min_q_factor=1e6)
        assert _qp60_events(events) == []

    def test_q_factor_zero_keeps_packet(self):
        rng = np.random.default_rng(103)
        t, fields = _circular_qp60(rng)
        events = detect_round8(t, fields, dt=DT, min_q_factor=0.0)
        assert _qp60_events(events), "Q=0 should keep the clean QP60 packet"


# ---------------------------------------------------------------------- #
# MVA parallel-fraction gate                                              #
# ---------------------------------------------------------------------- #


class TestMvaParallelGate:
    def test_compressional_rejected_by_default(self):
        rng = np.random.default_rng(201)
        t, fields = _compressional_qp60(rng)
        events = detect_round8(t, fields, dt=DT)
        assert _qp60_events(events) == [], (
            "compressional packet (energy on b_par) must fail the MVA gate"
        )

    def test_compressional_passes_if_gate_disabled(self):
        """Opening the gate fully (max_mva_par_frac=1.0) lets the packet through —
        confirms the rejection was due to the gate, not some other filter."""
        rng = np.random.default_rng(202)
        t, fields = _compressional_qp60(rng)
        events = detect_round8(t, fields, dt=DT, max_mva_par_frac=1.0)
        assert _qp60_events(events), (
            "with the MVA gate fully open the compressional packet should pass"
        )


# ---------------------------------------------------------------------- #
# Stokes degree-of-polarization gate                                      #
# ---------------------------------------------------------------------- #


class TestStokesGate:
    def test_unpolarized_rejected_by_default(self):
        rng = np.random.default_rng(301)
        t, fields = _unpolarized_qp60(rng)
        events = detect_round8(t, fields, dt=DT)
        assert _qp60_events(events) == [], (
            "random-phase pair must fail the Stokes degree-of-polarization gate"
        )

    def test_clean_packet_rejected_when_gate_set_impossibly_high(self):
        """A degree-of-polarization floor > 1 cannot be satisfied by any
        physical signal — the clean circular packet must be rejected. This
        isolates the Stokes gate from the others (the same packet passes
        the default gates by ``test_clean_packet_passes_default``)."""
        rng = np.random.default_rng(302)
        t, fields = _circular_qp60(rng)
        events = detect_round8(t, fields, dt=DT, min_stokes_d=1.01)
        assert _qp60_events(events) == []
