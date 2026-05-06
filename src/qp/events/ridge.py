r"""Band-aware ridge extraction from a CWT power matrix.

A "ridge" is a contiguous (time, period) blob of CWT power above a
threshold inside one of the canonical QP bands. This module turns a
boolean mask over the (period x time) plane into a list of
:class:`Ridge` objects, each describing the time/period extent of one
wave packet candidate.

The ridge extractor is intentionally band-local — it only inspects
period rows that fall inside the requested band, so a strong QP120
ridge cannot bleed into a QP60 detection. The cone of influence (COI)
is enforced by masking samples within ``coi_factor * period`` of either
edge of the segment.

Fused ridges (two wave packets that overlap in time, so their CWT
blobs are connected) are optionally split at intra-ridge envelope
minima using the canonical **half-power** criterion: the on-ridge
amplitude must drop to :math:`1/\sqrt{2}` of the local peak between
two sub-peaks for the split to be accepted. This is the textbook
FWHM definition; in amplitude space the prominence threshold is
:math:`1 - 1/\sqrt{2} \approx 0.293`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage
from scipy.signal import find_peaks

from qp.events.bands import Band, freq_to_period, get_band

#: Half-power prominence threshold in amplitude space.
#:
#: A coherent wave packet splits cleanly from its neighbour when the
#: on-ridge amplitude drops to half-power, i.e. amplitude factor
#: :math:`1/\sqrt{2}` of peak (power factor 1/2). The required
#: prominence is therefore :math:`1 - 1/\sqrt{2} \approx 0.293`.
HALF_POWER_PROMINENCE: float = 1.0 - 1.0 / math.sqrt(2.0)


# Default cone-of-influence factor for the Morlet wavelet with omega0=10.
# A point at scale s is "inside the COI" if it is at least sqrt(2) * s
# from either edge; we use a slightly more conservative 1.0 * period.
DEFAULT_COI_FACTOR: float = 1.0


@dataclass(frozen=True, slots=True)
class Ridge:
    r"""A connected (time, period) region of elevated CWT power.

    Attributes
    ----------
    band : str
        The QP band this ridge sits in (e.g. ``"QP60"``).
    t_start_idx, t_end_idx : int
        Inclusive sample indices marking the time extent.
    p_min_idx, p_max_idx : int
        Inclusive period-row indices into the CWT matrix.
    peak_time_idx : int
        Time index of the maximum power within the ridge.
    peak_period_idx : int
        Period-row index of the maximum power within the ridge.
    peak_period_sec : float
        Period (in seconds) at the ridge peak.
    peak_power : float
        CWT power at the peak.
    period_fwhm_sec : float
        Full-width-half-maximum of the period marginal at peak time.
    n_pixels : int
        Number of pixels in the connected blob.
    """

    band: str
    t_start_idx: int
    t_end_idx: int
    p_min_idx: int
    p_max_idx: int
    peak_time_idx: int
    peak_period_idx: int
    peak_period_sec: float
    peak_power: float
    period_fwhm_sec: float
    n_pixels: int

    @property
    def duration_samples(self) -> int:
        return self.t_end_idx - self.t_start_idx + 1

    def duration_seconds(self, dt: float) -> float:
        return self.duration_samples * dt


def _coi_mask(
    cwt_freq: NDArray[np.floating],
    n_time: int,
    dt: float,
    coi_factor: float = DEFAULT_COI_FACTOR,
) -> NDArray[np.bool_]:
    r"""Boolean mask of CWT cells *outside* the cone of influence.

    Cells inside the COI are ``True``; cells too close to the segment
    edge for the corresponding period are ``False``. Shape matches
    ``(n_freq, n_time)``.
    """
    periods_sec = freq_to_period(cwt_freq)
    edge_samples = (coi_factor * periods_sec / dt).astype(int)
    mask = np.ones((len(cwt_freq), n_time), dtype=bool)
    for i, edge in enumerate(edge_samples):
        e = min(edge, n_time // 2)
        if e > 0:
            mask[i, :e] = False
            mask[i, n_time - e :] = False
    return mask


def _band_row_indices(
    cwt_freq: NDArray[np.floating],
    band: Band,
) -> NDArray[np.intp]:
    r"""Return the indices of CWT rows whose period lies inside ``band``."""
    periods_sec = freq_to_period(cwt_freq)
    in_band = (
        (periods_sec >= band.period_min_sec)
        & (periods_sec < band.period_max_sec)
    )
    return np.flatnonzero(in_band)


def extract_ridges(
    cwt_power: ArrayLike,
    cwt_freq: ArrayLike,
    band: str | Band,
    threshold_mask: ArrayLike | None = None,
    *,
    dt: float = 60.0,
    min_duration_sec: float = 2 * 3600,
    min_pixels: int = 50,
    coi_factor: float = DEFAULT_COI_FACTOR,
    split_fused_ridges: bool = True,
) -> list[Ridge]:
    r"""Extract connected ridges in one period band.

    Parameters
    ----------
    cwt_power : array_like, shape (n_freq, n_time)
        CWT power (``|cwt|^2`` or ``|cwt|``).
    cwt_freq : array_like, shape (n_freq,)
        Wavelet frequency axis in Hz.
    band : str or Band
        Which QP band to scan.
    threshold_mask : array_like, optional
        Boolean mask the same shape as ``cwt_power``. ``True`` means
        the cell is above the detection threshold. If ``None`` the
        global ``cwt_power.max() / 4`` is used as a fall-back threshold
        (suitable for tests; production callers should pass an explicit
        sigma mask from :mod:`qp.events.threshold`).
    dt : float
        Time-axis sampling interval in seconds.
    min_duration_sec : float
        Reject ridges shorter than this in time.
    min_pixels : int
        Reject blobs with fewer than this many pixels (denoising).
    coi_factor : float
        How many wavelet periods of edge to mask out per scale.

    Returns
    -------
    list[Ridge]
        Sorted by ``peak_time_idx``.
    """
    cwt_power = np.asarray(cwt_power, dtype=float)
    cwt_freq = np.asarray(cwt_freq, dtype=float)
    _n_freq, n_time = cwt_power.shape

    band_obj = get_band(band)
    row_idx = _band_row_indices(cwt_freq, band_obj)
    if row_idx.size == 0:
        return []

    if threshold_mask is None:
        thr_global = float(cwt_power[row_idx].max()) / 4.0
        mask_input = cwt_power > thr_global
    else:
        mask_input = np.asarray(threshold_mask, dtype=bool)
        if mask_input.shape != cwt_power.shape:
            raise ValueError(
                f"threshold_mask shape {mask_input.shape} does not match "
                f"cwt_power shape {cwt_power.shape}"
            )

    coi = _coi_mask(cwt_freq, n_time, dt=dt, coi_factor=coi_factor)

    # Restrict to the band rows + COI + caller threshold
    band_mask = np.zeros_like(mask_input)
    band_mask[row_idx, :] = True

    full_mask = band_mask & coi & mask_input

    if not full_mask.any():
        return []

    # Connected-component labelling, 8-connectivity
    structure = ndimage.generate_binary_structure(2, 2)
    labels, n_components = ndimage.label(full_mask, structure=structure)
    if n_components == 0:
        return []

    # Periods on the band rows for FWHM bookkeeping
    periods_sec = 1.0 / cwt_freq

    ridges: list[Ridge] = []
    min_duration_samples = int(math.ceil(min_duration_sec / dt))

    objects = ndimage.find_objects(labels)
    for blob_idx, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        f_slice, t_slice = sl
        sub_power = cwt_power[f_slice, t_slice]
        sub_label = labels[f_slice, t_slice] == blob_idx
        masked_power = np.where(sub_label, sub_power, 0.0)

        n_pixels = int(sub_label.sum())
        if n_pixels < min_pixels:
            continue

        t_start_idx = int(t_slice.start)
        t_end_idx = int(t_slice.stop) - 1
        duration_samples = t_end_idx - t_start_idx + 1
        if duration_samples < min_duration_samples:
            continue

        p_min_idx = int(f_slice.start)
        p_max_idx = int(f_slice.stop) - 1

        # Locate the peak
        flat = masked_power.argmax()
        peak_local = np.unravel_index(flat, masked_power.shape)
        peak_period_idx = p_min_idx + int(peak_local[0])
        peak_time_idx = t_start_idx + int(peak_local[1])
        peak_power = float(cwt_power[peak_period_idx, peak_time_idx])
        peak_period_sec = float(periods_sec[peak_period_idx])

        # FWHM of the period marginal at peak time
        col_full = cwt_power[row_idx, peak_time_idx]
        col_periods = periods_sec[row_idx]
        if col_full.max() <= 0:
            period_fwhm_sec = 0.0
        else:
            half = col_full.max() / 2.0
            above = col_full >= half
            if above.any():
                period_fwhm_sec = float(
                    col_periods[above].max() - col_periods[above].min()
                )
            else:
                period_fwhm_sec = 0.0

        ridges.append(
            Ridge(
                band=band_obj.name,
                t_start_idx=t_start_idx,
                t_end_idx=t_end_idx,
                p_min_idx=p_min_idx,
                p_max_idx=p_max_idx,
                peak_time_idx=peak_time_idx,
                peak_period_idx=peak_period_idx,
                peak_period_sec=peak_period_sec,
                peak_power=peak_power,
                period_fwhm_sec=period_fwhm_sec,
                n_pixels=n_pixels,
            )
        )

    if split_fused_ridges:
        cwt_power_arr = np.asarray(cwt_power, dtype=float)
        split: list[Ridge] = []
        for r in ridges:
            split.extend(
                split_ridge_at_half_power(r, cwt_power_arr, dt, min_duration_sec)
            )
        ridges = split

    ridges.sort(key=lambda r: r.peak_time_idx)
    return ridges


def split_ridge_at_half_power(
    ridge: Ridge,
    cwt_power: NDArray[np.floating],
    dt: float,
    min_duration_sec: float,
) -> list[Ridge]:
    r"""Split a fused ridge at on-ridge envelope half-power minima.

    The on-ridge amplitude is :math:`A(t) = |\mathrm{CWT}|` at the
    ridge's peak period row over the ridge's time extent. We search
    for sub-peaks in :math:`A(t)` whose prominence exceeds the
    half-power threshold (:math:`1 - 1/\sqrt{2} \approx 0.293` of the
    ridge maximum), then split the ridge at the time of minimum
    amplitude between consecutive sub-peaks. Sub-ridges shorter than
    ``min_duration_sec`` are discarded.

    Returns ``[ridge]`` if no split passes the criterion, so the
    function is a safe no-op for clean single-packet ridges.
    """
    envelope = np.asarray(
        cwt_power[ridge.peak_period_idx, ridge.t_start_idx : ridge.t_end_idx + 1],
        dtype=float,
    )
    if envelope.size < 3 or envelope.max() <= 0:
        return [ridge]

    threshold = HALF_POWER_PROMINENCE * float(envelope.max())
    sub_peaks, _ = find_peaks(envelope, prominence=threshold)
    if sub_peaks.size <= 1:
        return [ridge]

    # Boundaries: ridge start, valleys between sub-peaks, ridge end.
    boundaries = [0]
    for i in range(len(sub_peaks) - 1):
        seg = envelope[sub_peaks[i] : sub_peaks[i + 1] + 1]
        boundaries.append(int(sub_peaks[i] + np.argmin(seg)))
    boundaries.append(envelope.size - 1)

    min_samples = int(math.ceil(min_duration_sec / dt))
    sub_ridges: list[Ridge] = []
    for j in range(len(boundaries) - 1):
        local_start = boundaries[j]
        local_end = boundaries[j + 1]
        if local_end - local_start + 1 < min_samples:
            continue
        sub_envelope = envelope[local_start : local_end + 1]
        peak_local = int(np.argmax(sub_envelope))
        sub_ridges.append(replace(
            ridge,
            t_start_idx=ridge.t_start_idx + local_start,
            t_end_idx=ridge.t_start_idx + local_end,
            peak_time_idx=ridge.t_start_idx + local_start + peak_local,
            peak_power=float(sub_envelope[peak_local]),
        ))

    return sub_ridges if sub_ridges else [ridge]
