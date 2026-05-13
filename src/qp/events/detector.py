"""Wave event detection from CWT power.

Two layers:

1. :func:`detect_wave_packets_multi` — band-agnostic ridge extraction
   over :data:`qp.events.bands.QP_SEARCH_BAND` (10–160 min). Returns
   :class:`WavePacketPeak` instances with bands assigned post-hoc by
   :func:`qp.events.bands.classify_period`. This is what the
   round-8 mission sweep calls for every transverse component.

2. :func:`detect_round8` — wraps the multi-detector with the
   round-8 amplitude / Q-factor / MVA / Stokes gates and the
   per-segment dedup. Public entry point for the production sweep
   (``scripts/sweep_events_round8.py``).
"""

from __future__ import annotations

import datetime
import math
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qp.events.bands import QP_SEARCH_BAND, classify_period
from qp.events.catalog import WavePacketPeak
from qp.events.ridge import Ridge, extract_ridges
from qp.signal.wavelet import morlet_cwt

if TYPE_CHECKING:
    from qp.events.threshold_diag import BGArchive

#: Round-8 threshold-method dispatch labels.
#:
#: ``tc_chi2`` (default) — Torrence & Compo (1998) §4 AR(1)+χ²(2)
#: per-pixel test at the round-8 Bonferroni FWER. Recovers ~3× more
#: events than ``mad_row`` at the 0.1–0.3 nT detection edge at matched
#: bg-rate; methods converge above ~1 nT.
#:
#: ``mad_row`` — per-row MAD on background period rows, log-period
#: interpolated across QP bands. Legacy gate; kept for parity with the
#: pre-Phase-3 catalogue.
#:
#: ``fdr_chi2``, ``pooled`` — diagnostic-only alternatives reachable
#: from the sweep script for sensitivity studies; not recommended as a
#: production default.
ThresholdMethod = Literal["mad_row", "tc_chi2", "fdr_chi2", "pooled"]

# Imported lazily inside functions to avoid an import cycle:
#   qp.events.threshold imports qp.signal.pipeline.SpectralResult
#   qp.signal.pipeline currently does not import qp.events, so we're
#   safe to import threshold at module top, but we keep it local for
#   the functions that use it for clarity.


def detect_wave_packets_multi(
    data: ArrayLike,
    times: list[datetime.datetime] | NDArray[np.floating],
    dt: float = 60.0,
    *,
    cwt_freq: ArrayLike | None = None,
    cwt_power: ArrayLike | None = None,
    threshold_mask: ArrayLike | None = None,
    min_duration_hours: float = 2.0,
    min_pixels: int = 50,
    coi_factor: float = 1.0,
    n_freqs: int = 300,
) -> list[WavePacketPeak]:
    r"""Band-agnostic wave-packet detection from a CWT scalogram.

    Pipeline
    --------
    1. Compute the CWT once (or accept a precomputed scalogram so the
       caller can reuse it across components).
    2. Run :func:`qp.events.ridge.extract_ridges` once over
       :data:`qp.events.bands.QP_SEARCH_BAND` (10–160 min) so a ridge
       straddling an octave boundary (20, 40, 80 min) is one packet,
       not two amputated halves.
    3. Convert each ridge to a :class:`WavePacketPeak` and assign the
       canonical QP band post-hoc via :func:`classify_period` on the
       interpolated peak period. Bands are a *labelling* concept here,
       not a detection axis.

    Parameters
    ----------
    data : array_like
        Time-series of one MFA component (typically ``b_perp1``).
    times : list[datetime] or float ndarray
        Sample timestamps. If a numpy array is passed it is assumed to
        be POSIX seconds.
    dt : float
        Sampling interval in seconds (default 60 s).
    cwt_freq, cwt_power : array_like, optional
        Precomputed CWT frequency axis and ``|cwt|`` power matrix. If
        either is None, the CWT is computed internally with
        ``omega0=10`` and ``n_freqs`` rows.
    threshold_mask : array_like, optional
        Boolean mask the same shape as ``cwt_power``. ``True`` means
        the cell is above the σ-threshold. If None, a fall-back
        ``cwt_power.max() / 4`` mask is used (suitable for tests, but
        the production sweep should always pass an explicit mask from
        :func:`qp.events.threshold.wavelet_sigma_mask`).
    min_duration_hours : float
        Reject ridges shorter than this in time.
    min_pixels : int
        Reject blobs with fewer than this many pixels (denoising).
    coi_factor : float
        Cone-of-influence factor passed to the ridge extractor.
    n_freqs : int
        Number of CWT rows when computing internally.

    Returns
    -------
    list[WavePacketPeak]
        Sorted by peak time. Each carries ``band`` and ``period_sec``;
        ``band`` is the post-hoc classification of ``period_sec`` and
        is ``None`` if the peak lies outside every QP band.
    """
    data = np.asarray(data, dtype=float)
    n_time = data.size

    # Build a datetime list — accept POSIX float arrays for convenience
    if isinstance(times, np.ndarray) and times.dtype.kind in ("f", "i"):
        _epoch = datetime.datetime(1970, 1, 1)
        times_list: list[datetime.datetime] = [
            _epoch + datetime.timedelta(seconds=float(ts)) for ts in times
        ]
    else:
        times_list = list(times)
    if len(times_list) != n_time:
        raise ValueError(f"len(times)={len(times_list)} != len(data)={n_time}")

    if cwt_freq is None or cwt_power is None:
        freq, _, cwt_matrix = morlet_cwt(data, dt=dt, n_freqs=n_freqs)
        cwt_power = np.abs(cwt_matrix)
    else:
        freq = np.asarray(cwt_freq, dtype=float)
        cwt_power = np.asarray(cwt_power, dtype=float)
        if cwt_power.shape != (freq.size, n_time):
            raise ValueError(
                f"cwt_power shape {cwt_power.shape} does not match "
                f"(n_freq={freq.size}, n_time={n_time})"
            )

    min_duration_sec = min_duration_hours * 3600.0

    ridges = extract_ridges(
        cwt_power,
        freq,
        band=QP_SEARCH_BAND,
        threshold_mask=threshold_mask,
        dt=dt,
        min_duration_sec=min_duration_sec,
        min_pixels=min_pixels,
        coi_factor=coi_factor,
    )
    packets = [_ridge_to_packet(ridge, times_list, n_time) for ridge in ridges]

    # Post-hoc band labelling. Ridges from QP_SEARCH_BAND carry no
    # canonical band; classify_period maps each peak's interpolated
    # period to QP15/30/60/120 (or None outside the search range).
    for p in packets:
        if p.period_sec is not None and p.period_sec > 0:
            p.band = classify_period(p.period_sec)

    packets.sort(key=lambda p: p.peak_time)
    return packets


def _ridge_to_packet(
    ridge: Ridge,
    times: list[datetime.datetime],
    n_time: int,
) -> WavePacketPeak:
    """Convert a :class:`Ridge` to a :class:`WavePacketPeak`."""
    t_start = times[max(0, ridge.t_start_idx)]
    t_end = times[min(n_time - 1, ridge.t_end_idx)]
    t_peak = times[ridge.peak_time_idx]
    return WavePacketPeak(
        peak_time=t_peak,
        prominence=float(ridge.peak_power),
        date_from=t_start,
        date_to=t_end,
        period_sec=float(ridge.peak_period_sec),
        period_fwhm_sec=float(ridge.period_fwhm_sec),
    )


def dedup_peaks_by_period(
    peaks: list[WavePacketPeak],
    dt_sec: float = 7200.0,
    *,
    period_log2_tol: float = 0.5,
) -> list[WavePacketPeak]:
    r"""Suppress peaks too close in time *and* period to a prior peak.

    The transverse pair (``b_perp1``, ``b_perp2``) frequently picks up
    the same physical wave train twice. We collapse those near-duplicates
    while preserving genuinely close peaks at different periods (e.g.
    co-existing QP30 + QP60 harmonics, or a sequence of distinct wave
    packets in the same band).

    Two peaks are duplicates when

    - :math:`|\Delta t| < \mathtt{dt\_sec}` (default 2 h), AND
    - :math:`|\log_2(P_1 / P_2)| < \mathtt{period\_log2\_tol}` (default
      a half-octave, matching the width of one QP band).

    The earlier peak is kept; the later one is dropped. This is the
    detector-level dedup applied *inside* a 36-h segment, on the
    union of (b_perp1, b_perp2) peaks; the parquet-level cross-segment
    pass in :mod:`qp.events.dedup` repeats the same idea on the
    assembled catalogue.

    Parameters
    ----------
    peaks : list[WavePacketPeak]
        Must be sorted by ``peak_time``. Not modified.
    dt_sec : float, default 7200
        Time-separation cutoff in seconds. Peaks farther apart than
        this are never duplicates.
    period_log2_tol : float, default 0.5
        Log-period tolerance in octaves. Peaks farther apart in
        :math:`\log_2 P` than this are never duplicates. The default
        half-octave matches the width of one QP band, so this rule
        coincides with the legacy "same band" rule when both peaks
        sit inside the same canonical octave.

    Returns
    -------
    list[WavePacketPeak]
    """
    merged: list[WavePacketPeak] = []
    for peak in peaks:
        p_cur = peak.period_sec
        if p_cur is None or p_cur <= 0:
            # Round-8 detections always populate period_sec; a missing
            # period means a malformed peak — keep it as-is rather than
            # silently dropping or merging on a band string.
            merged.append(peak)
            continue
        is_dup = False
        for prev in reversed(merged):
            sep = abs((peak.peak_time - prev.peak_time).total_seconds())
            if sep >= dt_sec:
                break  # merged is time-sorted; earlier entries are farther
            p_prev = prev.period_sec
            if p_prev is None or p_prev <= 0:
                continue
            if abs(math.log2(p_cur / p_prev)) < period_log2_tol:
                is_dup = True
                break
        if not is_dup:
            merged.append(peak)
    return merged


# ---------------------------------------------------------------------
# Round-8 detector — public entry point
# ---------------------------------------------------------------------
#
# Four physically distinct gates, each with a canonical (or
# search-volume-derived) threshold:
#
#   amplitude       whitened sigma-mask            Bonferroni FWER 1%
#   narrowness      spectral Q-factor              >= 3
#   transversality  MVA major-axis projection      |e_max . b_par|^2 <= 0.5
#   polarization    Stokes degree d                >= 0.7
#
# Plus three housekeeping rules: duration >= 2 h, dedup 2 h same-band,
# min_pixels 10. See the round-7/round-8 retrospectives in
# planner notes for the full lineage.

#: Family-wise error rate for the whitened CWT sigma-mask. The actual
#: sigma threshold is derived from the effective number of independent
#: time-frequency cells of the Morlet scalogram (see
#: :func:`bonferroni_n_sigma_for_cwt`).
SEGMENT_FWER_ALPHA: float = 0.01

#: Minimum spectral Q = period / FWHM. Floor for any band-limited peak.
MIN_Q_FACTOR: float = 3.0

#: Re-exports from :mod:`qp.signal.polarization_config` — the canonical
#: home for polarization thresholds. The justification block below is
#: kept here next to its consumer; the constants themselves live in the
#: config module so :mod:`qp.signal.cross_correlation` and the rest of
#: the polarization surface share a single source of truth.
#:
#: ``d = sqrt(Q^2 + U^2 + V^2) / I`` is computed from the cross-Stokes
#: parameters of the b_perp1 / b_perp2 Morlet-CWT analytic-signal
#: coefficients, averaged over the band's frequency rows and the
#: event's time window (see
#: :func:`qp.signal.polarization.degree_of_polarization`). ``d = 1``
#: is a fully polarized wave — circular, linear, or elliptical alike;
#: ``d = 0`` is unpolarized broadband noise, where the cross-terms
#: :math:`\langle z_1 z_2^* \rangle` average to zero.
#:
#: Community convention. The 0.6–0.8 range is standard for "polarized
#: event" identification in magnetospheric ULF-wave catalogues (Samson
#: 1973, *Geophys. J. R. Astr. Soc.* 34, 403; Anderson et al. 1990,
#: *JGR* 95 A6; Bortnik et al. 2007, *JGR* 112 A11; Engebretson et al.
#: 1986, *JGR* 91 A7); 0.7 is the most frequently adopted single value.
#:
#: Empirical justification for 0.7 in this catalogue —
#: ``scripts/diag_stokes_distribution.py`` re-runs the detector on 111
#: representative segments with the Stokes gate disabled and records
#: ``d`` for every candidate that survives the σ-mask + Q-factor + MVA
#: gates (83 candidates total). The distribution is bimodal-ish: a long
#: "mixed/ambiguous" tail centred near 0.55 and a sharp "clean
#: transverse Alfvén" peak near 0.9
#: (see ``Output/figures/diag_p4_stokes_distribution.png``). The 0.7
#: cut sits at the natural shoulder between the two populations — the
#: catalogue would grow by ~30 % at 0.6 (mixing in linear / partly
#: compressional candidates) and shrink by ~30 % at 0.8 (excluding
#: moderately polarized real waves). 0.7 is therefore both the most
#: defensible single value here and aligned with community convention.
from qp.signal.polarization_config import (  # noqa: E402
    MAX_MVA_PARALLEL_FRACTION,
    MIN_DEGREE_OF_POLARIZATION,
)


def _v_indep_for_cwt(
    n_time: int,
    dt: float,
    freq: np.ndarray,
    morlet_omega0: float = 10.0,
) -> float:
    r"""Effective number of independent time-frequency cells in the Morlet CWT.

    The Morlet wavelet has temporal correlation length ~1 period at
    every frequency and frequency-bandwidth :math:`\Delta f / f \approx
    1/\omega_0`. The number of *independent* time-frequency cells is

    .. math::

        V_{\mathrm{indep}} = \frac{\omega_0}{2\pi}
            \ln\!\left(\frac{f_{\max}}{f_{\min}}\right)
        \cdot
        n_t\,dt\,\bar f.
    """
    freq = np.asarray(freq, dtype=float)
    f_min = float(freq[freq > 0].min())
    f_max = float(freq.max())
    n_freq_indep = (morlet_omega0 / (2.0 * np.pi)) * np.log(f_max / f_min)
    n_time_indep = n_time * dt * float(freq.mean())
    return max(n_freq_indep * n_time_indep, 1.0)


def bonferroni_n_sigma_for_cwt(
    n_time: int,
    dt: float,
    freq: np.ndarray,
    morlet_omega0: float = 10.0,
    alpha: float = SEGMENT_FWER_ALPHA,
) -> float:
    r"""Sigma threshold for FWER control over the Morlet-CWT search volume.

    Bonferroni sets the per-pixel FP probability to
    :math:`\alpha / V_{\mathrm{indep}}` (see :func:`_v_indep_for_cwt`)
    and the threshold is the corresponding Gaussian quantile (~4.6σ at
    α = 0.01).
    """
    from scipy.stats import norm

    v_indep = _v_indep_for_cwt(n_time, dt, freq, morlet_omega0)
    return float(norm.isf(alpha / v_indep))


from dataclasses import dataclass as _dataclass


@_dataclass(frozen=True, slots=True)
class DetectedEvent:
    """A WavePacketPeak that passed all round-8 gates, with diagnostics.

    The four gate values plus the bandpass amplitudes per component are
    retained so callers can persist them in tabular form without
    rerunning the detector. The Stokes vector and derived ellipticity /
    inclination / polarized fraction are computed once from the same
    in-band CWT slice that feeds the ``stokes_d`` gate — persisting
    them costs nothing and lets downstream consumers (Fig 10, fig11,
    sensitivity analysis) read the polarization geometry directly from
    the catalogue without rerunning the sweep.
    """

    peak: WavePacketPeak
    q_factor: float
    mva_par_frac: float
    stokes_d: float
    b_perp1_amp: float
    b_perp2_amp: float
    b_par_amp: float
    stokes_i: float
    stokes_q: float
    stokes_u: float
    stokes_v: float
    ellipticity: float
    inclination_deg: float
    polarized_fraction: float


def detect_round8(
    t: NDArray[np.floating],
    fields: NDArray[np.floating],
    dt: float = 60.0,
    *,
    n_freqs: int = 300,
    fwer_alpha: float = SEGMENT_FWER_ALPHA,
    min_q_factor: float = MIN_Q_FACTOR,
    min_stokes_d: float = MIN_DEGREE_OF_POLARIZATION,
    max_mva_par_frac: float = MAX_MVA_PARALLEL_FRACTION,
    min_duration_hours: float = 2.0,
    min_pixels: int = 10,
    epoch: datetime.datetime | None = None,
    threshold_method: ThresholdMethod = "tc_chi2",
    apply_coi_mask: bool = True,
    bg_archive: "BGArchive | None" = None,
    region: str | None = None,
) -> list[DetectedEvent]:
    """Round-8 simplified wave-event detector.

    Parameters
    ----------
    t : array_like, shape (N,)
        Time samples in seconds since ``epoch``.
    fields : array_like, shape (N, 3)
        ``[b_par, b_perp1, b_perp2]`` in MFA frame.
    dt : float, default 60
        Sampling interval, seconds.
    epoch : datetime, optional
        Reference epoch for ``t``. Default 2000-01-01 (J2000-ish).
    threshold_method : {"mad_row", "tc_chi2", "fdr_chi2", "pooled"}
        CWT amplitude gate. ``tc_chi2`` (default) — Torrence & Compo
        1998 AR(1)+χ²(2) per-pixel test at the same Bonferroni FWER
        ``fwer_alpha``. ``mad_row`` keeps the legacy per-row MAD on
        background period rows. ``fdr_chi2`` and ``pooled`` are
        diagnostic-only; ``pooled`` additionally requires ``bg_archive``
        and ``region``.
    apply_coi_mask : bool, default True
        Drop wavelet cells inside the Morlet cone of influence before
        ridge extraction. The ``min_duration_hours = 2`` cut discards
        most COI-tainted events implicitly; this is the rigorous
        version and pairs naturally with the χ²-based gate.
    bg_archive : BGArchive, optional
        Pre-computed pooled background-row statistics (see
        :mod:`qp.events.threshold_diag`). Only used when
        ``threshold_method='pooled'``.
    region : str, optional
        Plasma-region key into ``bg_archive``. Only used when
        ``threshold_method='pooled'``.

    Returns
    -------
    list of DetectedEvent
        One entry per peak that passed all four gates. The gate values
        are populated; the underlying ``WavePacketPeak`` is unchanged.
    """
    from qp.events.threshold import wavelet_sigma_mask
    from qp.events.threshold_diag import (
        coi_mask as _coi_mask,
        fdr_chi2_mask,
        pooled_archive_mask,
        torrence_compo_chi2_mask,
    )
    from qp.signal.polarization import (
        ellipticity_inclination_from_stokes,
        mva_major_axis_parallel_fraction,
        stokes_parameters,
    )

    if epoch is None:
        epoch = datetime.datetime(2000, 1, 1)

    b_par = np.asarray(fields[:, 0], dtype=float)
    b_perp1 = np.asarray(fields[:, 1], dtype=float)
    b_perp2 = np.asarray(fields[:, 2], dtype=float)

    # Complex CWTs of all three field components. The transverse pair
    # feeds ridge extraction and Stokes; b_par enters MVA.
    freq, _, cwt_par = morlet_cwt(b_par, dt=dt, n_freqs=n_freqs)
    _, _, cwt1 = morlet_cwt(b_perp1, dt=dt, n_freqs=n_freqs)
    _, _, cwt2 = morlet_cwt(b_perp2, dt=dt, n_freqs=n_freqs)
    power1 = np.abs(cwt1)
    power2 = np.abs(cwt2)

    n_time = power1.shape[1]

    if threshold_method == "mad_row":
        n_sigma = bonferroni_n_sigma_for_cwt(n_time, dt, freq, alpha=fwer_alpha)
        mask1 = wavelet_sigma_mask(power1, freq, n_sigma=n_sigma)
        mask2 = wavelet_sigma_mask(power2, freq, n_sigma=n_sigma)
    elif threshold_method == "tc_chi2":
        # Bonferroni-corrected per-pixel α under the same FWER budget.
        v_indep = _v_indep_for_cwt(n_time, dt, freq)
        alpha_pixel = fwer_alpha / v_indep
        mask1 = torrence_compo_chi2_mask(
            power1,
            freq,
            b_perp1,
            dt=dt,
            alpha=alpha_pixel,
        )
        mask2 = torrence_compo_chi2_mask(
            power2,
            freq,
            b_perp2,
            dt=dt,
            alpha=alpha_pixel,
        )
    elif threshold_method == "fdr_chi2":
        # FDR target is the FWER budget — apples-to-apples conservatism.
        mask1 = fdr_chi2_mask(power1, freq, b_perp1, dt=dt, q=fwer_alpha)
        mask2 = fdr_chi2_mask(power2, freq, b_perp2, dt=dt, q=fwer_alpha)
    elif threshold_method == "pooled":
        if bg_archive is None or region is None:
            raise ValueError(
                "threshold_method='pooled' requires both bg_archive and region; "
                "build the archive with scripts/build_bg_archive.py and pass "
                "the segment's plasma-region label"
            )
        n_sigma = bonferroni_n_sigma_for_cwt(n_time, dt, freq, alpha=fwer_alpha)
        mask1 = pooled_archive_mask(power1, freq, region, bg_archive, n_sigma=n_sigma)
        mask2 = pooled_archive_mask(power2, freq, region, bg_archive, n_sigma=n_sigma)
    else:  # pragma: no cover — Literal exhausts the cases
        raise ValueError(f"unknown threshold_method: {threshold_method!r}")

    if apply_coi_mask:
        coi = _coi_mask(freq, n_time, dt=dt)
        mask1 = mask1 & coi
        mask2 = mask2 & coi

    times = [epoch + datetime.timedelta(seconds=float(s)) for s in t]

    all_peaks: list[WavePacketPeak] = []
    for component, power, mask in (
        (b_perp1, power1, mask1),
        (b_perp2, power2, mask2),
    ):
        peaks = detect_wave_packets_multi(
            data=component,
            times=times,
            dt=dt,
            cwt_freq=freq,
            cwt_power=power,
            threshold_mask=mask,
            min_duration_hours=min_duration_hours,
            min_pixels=min_pixels,
        )
        all_peaks.extend(peaks)

    all_peaks.sort(key=lambda p: p.peak_time)
    merged = dedup_peaks_by_period(all_peaks, dt_sec=7200.0)

    # Per-detection physical gates.
    n_time = cwt1.shape[1]
    kept: list[DetectedEvent] = []
    for peak in merged:
        if peak.period_sec is None or peak.period_sec <= 0 or peak.band is None:
            continue
        q = peak.q_factor
        if q is None or q < min_q_factor:
            continue
        i_start = max(
            0,
            int(np.floor((peak.date_from - epoch).total_seconds() / dt)),
        )
        i_end = min(
            n_time - 1,
            int(np.ceil((peak.date_to - epoch).total_seconds() / dt)),
        )
        if i_end <= i_start:
            continue
        # Transversality: MVA on bandpass-filtered 3-component field.
        i_freq_peak = int(np.argmin(np.abs(freq - 1.0 / peak.period_sec)))
        sl = slice(i_start, i_end + 1)
        field_bp = np.column_stack(
            [
                np.real(cwt_par[i_freq_peak, sl]),
                np.real(cwt1[i_freq_peak, sl]),
                np.real(cwt2[i_freq_peak, sl]),
            ]
        )
        par_frac = mva_major_axis_parallel_fraction(field_bp, par_axis=0)
        if par_frac > max_mva_par_frac:
            continue
        # Polarization purity over a peak-centred half-octave window
        # (same width as one QP band, but anchored on the actual peak
        # period rather than a fixed octave grid — so the gate doesn't
        # depend on which side of a band edge the ridge happens to peak).
        f_peak = 1.0 / peak.period_sec
        in_band = (freq >= f_peak / math.sqrt(2.0)) & (freq < f_peak * math.sqrt(2.0))
        if not in_band.any():
            continue
        c1_window = cwt1[in_band, sl]
        c2_window = cwt2[in_band, sl]
        s_i, s_q, s_u, s_v = stokes_parameters(
            c1_window.ravel(),
            c2_window.ravel(),
        )
        d = (np.sqrt(s_q * s_q + s_u * s_u + s_v * s_v) / s_i) if s_i > 0 else 0.0
        if d < min_stokes_d:
            continue
        ell, incl_deg, pol_frac = ellipticity_inclination_from_stokes(
            s_i,
            s_q,
            s_u,
            s_v,
        )
        # Bandpass amplitudes (RMS of the real CWT slice at peak f).
        b_perp1_amp = float(np.sqrt(np.mean(field_bp[:, 1] ** 2)))
        b_perp2_amp = float(np.sqrt(np.mean(field_bp[:, 2] ** 2)))
        b_par_amp = float(np.sqrt(np.mean(field_bp[:, 0] ** 2)))
        kept.append(
            DetectedEvent(
                peak=peak,
                q_factor=float(q),
                mva_par_frac=float(par_frac),
                stokes_d=float(d),
                b_perp1_amp=b_perp1_amp,
                b_perp2_amp=b_perp2_amp,
                b_par_amp=b_par_amp,
                stokes_i=float(s_i),
                stokes_q=float(s_q),
                stokes_u=float(s_u),
                stokes_v=float(s_v),
                ellipticity=float(ell),
                inclination_deg=float(incl_deg),
                polarized_fraction=float(pol_frac),
            )
        )
    return kept


def dedup_peaks_by_band(
    peaks: list[WavePacketPeak],
    dt_sec: float = 7200.0,
    *,
    period_log2_tol: float = 0.5,
) -> list[WavePacketPeak]:
    """Deprecated alias for :func:`dedup_peaks_by_period`.

    .. deprecated:: post-round-8
       The dedup is band-agnostic and keys on log-period proximity, so
       the name no longer reflects the behaviour. Call
       :func:`dedup_peaks_by_period` directly.
    """
    import warnings

    warnings.warn(
        "dedup_peaks_by_band is deprecated; use dedup_peaks_by_period instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return dedup_peaks_by_period(peaks, dt_sec=dt_sec, period_log2_tol=period_log2_tol)
