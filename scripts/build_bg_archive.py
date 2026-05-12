r"""Build the pooled-archive bg power statistics for the alternative gate.

For every valid 36-h MFA segment in the Cassini archive, compute the
per-period median and MAD of the transverse CWT amplitude
``sqrt(|W_⊥1| · |W_⊥2|)`` over the segment's time axis. Bin those
per-segment per-row statistics by plasma region at segment midpoint
(``magnetosphere`` / ``magnetosheath`` / ``solar_wind`` / ``unknown``),
then write the cross-segment 16/50/84 percentiles to
``Output/bg_archive.zarr``.

The pooled gate
(:func:`qp.events.threshold_diag.pooled_archive_mask`) reads the
``median[p50]`` and ``mad[p50]`` arrays for the segment's region and
applies them as the per-row noise floor. Replacing the per-segment
MAD with a cross-segment median of MADs trades sensitivity to local
contamination (a long sheath excursion in an otherwise-MS segment
will no longer inflate the threshold for the MS portion) for a small
loss of segment-specific tuning.

Runtime
-------
Single-threaded: ~15-30 min on the full 4743-segment archive at
n_freqs=300. Multiprocessing is straightforward (each segment is
independent) but kept out of the v1 builder to keep the failure modes
visible. Use ``--limit N`` for smoke tests.

Usage
-----
``uv run python scripts/build_bg_archive.py
     --output Output/bg_archive.zarr``
"""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import numpy as np

import qp
from qp.events.sweep_loader import (
    load_segments,
    region_at_peak_from_info,
    segment_to_payload,
)
from qp.io import legacy_pickle
from qp.signal.wavelet import morlet_cwt

log = logging.getLogger("build_bg_archive")

#: Schema constants mirror :mod:`qp.events.detector` so the archive is
#: directly usable as a drop-in replacement for ``wavelet_sigma_mask``.
_DT_SEC: float = 60.0
_N_FREQS: int = 300
_OMEGA0: float = 10.0
_REGIONS: tuple[str, ...] = (
    "magnetosphere",
    "magnetosheath",
    "solar_wind",
    "unknown",
)


def _per_segment_row_stats(
    b_perp1: np.ndarray,
    b_perp2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row median and MAD of the transverse-amplitude geometric mean.

    Returns ``(freq, median_per_row, mad_per_row)`` where each is a
    1-D array of length ``n_freqs``.
    """
    freq, _, cwt1 = morlet_cwt(
        b_perp1,
        dt=_DT_SEC,
        omega0=_OMEGA0,
        n_freqs=_N_FREQS,
    )
    _, _, cwt2 = morlet_cwt(
        b_perp2,
        dt=_DT_SEC,
        omega0=_OMEGA0,
        n_freqs=_N_FREQS,
    )
    amp = np.sqrt(np.abs(cwt1) * np.abs(cwt2) + 1e-30)
    med = np.median(amp, axis=1)
    mad = np.median(np.abs(amp - med[:, None]), axis=1)
    return freq, med, mad


def _aggregate_percentiles(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-segment 16/50/84 percentiles, ignoring NaN.

    ``samples`` has shape ``(n_segments_in_region, n_freqs)``.
    """
    if samples.shape[0] == 0:
        # No segments in this region — emit NaN sentinels.
        nan_row = np.full(samples.shape[1], np.nan)
        return nan_row, nan_row.copy(), nan_row.copy()
    p16 = np.nanpercentile(samples, 16, axis=0)
    p50 = np.nanpercentile(samples, 50, axis=0)
    p84 = np.nanpercentile(samples, 84, axis=0)
    return p16, p50, p84


# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(qp.OUTPUT_DIR / "bg_archive.zarr"),
        help="output zarr store path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N valid segments (smoke testing)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="restrict to segments whose first sample is in this year",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    legacy_pickle.register_stubs()
    arr, keep_idx = load_segments(year=args.year)
    log.info("loaded %d segments (year filter: %s)", len(keep_idx), args.year)

    if args.limit is not None:
        keep_idx = keep_idx[: args.limit]
        log.info("limiting to first %d segments", len(keep_idx))

    # Per-region accumulator lists of (median_per_row, mad_per_row).
    medians_by_region: dict[str, list[np.ndarray]] = {r: [] for r in _REGIONS}
    mads_by_region: dict[str, list[np.ndarray]] = {r: [] for r in _REGIONS}
    freq_axis: np.ndarray | None = None

    n_processed = 0
    n_skipped = 0
    t_start = datetime.datetime.now()
    for k, seg_idx in enumerate(keep_idx):
        seg = arr[seg_idx]
        payload = segment_to_payload(seg_idx, seg)
        if payload is None:
            n_skipped += 1
            continue
        midpoint = payload.times[len(payload.times) // 2]
        region = region_at_peak_from_info(payload.info, midpoint)
        if region not in medians_by_region:
            region = "unknown"
        try:
            freq, med, mad = _per_segment_row_stats(
                payload.b_perp1,
                payload.b_perp2,
            )
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("segment %d failed (%s); skipping", seg_idx, exc)
            n_skipped += 1
            continue
        if freq_axis is None:
            freq_axis = freq
        elif freq_axis.shape != freq.shape:
            log.warning(
                "segment %d freq axis mismatch; skipping",
                seg_idx,
            )
            n_skipped += 1
            continue
        medians_by_region[region].append(med)
        mads_by_region[region].append(mad)
        n_processed += 1
        if (k + 1) % 100 == 0:
            elapsed = (datetime.datetime.now() - t_start).total_seconds()
            rate = (k + 1) / elapsed if elapsed > 0 else 0.0
            log.info(
                "%d / %d segments, %.1f s elapsed, %.2f seg/s, skipped %d so far",
                k + 1,
                len(keep_idx),
                elapsed,
                rate,
                n_skipped,
            )

    log.info(
        "processed %d segments, skipped %d, by region: %s",
        n_processed,
        n_skipped,
        {r: len(medians_by_region[r]) for r in _REGIONS},
    )

    if freq_axis is None:
        log.error("no segments processed — nothing to write")
        return

    periods_sec = np.where(freq_axis > 0, 1.0 / freq_axis, np.inf)

    # Aggregate per region.
    aggregated: dict[str, dict[str, np.ndarray]] = {}
    for r in _REGIONS:
        med_arr = (
            np.array(medians_by_region[r])
            if medians_by_region[r]
            else (np.zeros((0, len(freq_axis))))
        )
        mad_arr = (
            np.array(mads_by_region[r])
            if mads_by_region[r]
            else (np.zeros((0, len(freq_axis))))
        )
        med_p16, med_p50, med_p84 = _aggregate_percentiles(med_arr)
        mad_p16, mad_p50, mad_p84 = _aggregate_percentiles(mad_arr)
        aggregated[r] = {
            "median_p16": med_p16,
            "median_p50": med_p50,
            "median_p84": med_p84,
            "mad_p16": mad_p16,
            "mad_p50": mad_p50,
            "mad_p84": mad_p84,
            "n_segments": np.array(med_arr.shape[0], dtype=np.int64),
        }

    # Write zarr.
    import zarr

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(out_path), mode="w")
    store.create_array(name="periods_sec", data=periods_sec.astype(np.float64))
    store.create_array(name="freq_hz", data=freq_axis.astype(np.float64))
    # One subgroup per percentile-family for ergonomic indexing in the
    # gate code (``medians/<region>``, ``mads/<region>``, etc.).
    for family in (
        "medians",
        "medians_p16",
        "medians_p84",
        "mads",
        "mads_p16",
        "mads_p84",
        "n_segments",
    ):
        store.require_group(family)
    for r in _REGIONS:
        agg = aggregated[r]
        store["medians"].create_array(name=r, data=agg["median_p50"])
        store["medians_p16"].create_array(name=r, data=agg["median_p16"])
        store["medians_p84"].create_array(name=r, data=agg["median_p84"])
        store["mads"].create_array(name=r, data=agg["mad_p50"])
        store["mads_p16"].create_array(name=r, data=agg["mad_p16"])
        store["mads_p84"].create_array(name=r, data=agg["mad_p84"])
        store["n_segments"].create_array(name=r, data=agg["n_segments"])

    store.attrs["dt_sec"] = _DT_SEC
    store.attrs["n_freqs"] = _N_FREQS
    store.attrs["omega0"] = _OMEGA0
    store.attrs["regions"] = list(_REGIONS)
    store.attrs["created"] = datetime.datetime.now(datetime.UTC).isoformat()
    store.attrs["source"] = "Cassini_MAG_MFA_36H.npy"
    store.attrs["n_processed"] = int(n_processed)
    store.attrs["n_skipped"] = int(n_skipped)

    log.info("wrote %s (%d segments)", out_path, n_processed)


if __name__ == "__main__":
    main()
