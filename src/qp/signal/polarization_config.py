r"""Named constants for polarization analysis and event gating.

Single source of truth for thresholds previously scattered across
``qp.signal.cross_correlation`` and ``qp.events.detector``. Empirical
justification for ``MIN_DEGREE_OF_POLARIZATION = 0.7`` lives next to
its consumer in ``qp.events.detector``.
"""

from __future__ import annotations

#: Stokes degree-of-polarization gate. Coherent waves have ``d → 1``;
#: incoherent broadband transients have ``d → 0``. See the detector
#: module for the full empirical justification (round-8 catalogue
#: shoulder at d ≈ 0.7).
MIN_DEGREE_OF_POLARIZATION: float = 0.7

#: MVA major-axis maximum parallel fraction. Transverse waves have
#: ``(e1·b_par)^2 → 0``; the gate rejects compressional perturbations
#: where the major axis aligns with B_0.
MAX_MVA_PARALLEL_FRACTION: float = 0.5

#: Half-width (degrees) of the band around 90°/270° that
#: ``classify_polarization`` calls "circular", and around 0°/180°/360°
#: that it calls "linear". A 30° tolerance straddles the natural
#: bimodal distribution of QP-event phase shifts.
CIRCULAR_LINEAR_TOL_DEG: float = 30.0

#: Cosine-taper fraction for Hilbert-transformed Stokes computation.
#: ``alpha=0.25`` tapers the outer 12.5% of each window, suppressing
#: edge artifacts in events shorter than ~5 oscillations.
TUKEY_TAPER_ALPHA: float = 0.25
