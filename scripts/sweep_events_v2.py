"""Phase 7.9 — v2 mission-wide QP event sweep with all Phase 7 metrics.

Extends the Phase 3 sweep with:
- Power-law FFT background + restored FFT screen ratio
- Matched-filter SNR per event
- Wavelet coherence per event
- Tapered Stokes / per-oscillation polarization
- Transverse ratio (perp/par)
- Composite quality score
- Dipole invariant latitude

Output: ``Output/events_qp_v2.parquet``

Usage::

    uv run python scripts/sweep_events_v2.py                # full mission
    uv run python scripts/sweep_events_v2.py --year 2007    # one year
    uv run python scripts/sweep_events_v2.py --serial        # debug mode
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
import types
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


# Stub legacy classes for unpickling SignalSnapshot objects
def _register_pickle_stubs() -> None:
    stub_classes = [
        "SignalSnapshot", "NewSignal", "Interval", "FFT_list",
        "WaveSignal", "Wave",
    ]
    stub_modules = [
        "__main__", "data_sweeper", "mag_fft_sweeper",
        "cassinilib", "cassinilib.NewSignal",
    ]
    for mod_path in stub_modules:
        if mod_path not in sys.modules:
            sys.modules[mod_path] = types.ModuleType(mod_path)
        for cls in stub_classes:
            setattr(sys.modules[mod_path], cls, type(cls, (), {}))


_register_pickle_stubs()

import qp  # noqa: E402
from qp.events.bands import QP_BAND_NAMES, get_band  # noqa: E402
from qp.events.catalog import WaveEvent  # noqa: E402
from qp.events.detector import detect_with_gate  # noqa: E402
from qp.events.quality import compute_quality  # noqa: E402
from qp.events.threshold import GateConfig  # noqa: E402
from qp.signal.coherence import ridge_coherence, wavelet_coherence  # noqa: E402
from qp.signal.cross_correlation import (  # noqa: E402
    classify_polarization,
    ellipticity_inclination_tapered,
    per_oscillation_ellipticity,
    phase_shift,
)
from qp.signal.fft import estimate_background_powerlaw, welch_psd  # noqa: E402
from qp.signal.matched_filter import matched_filter_peak_snr  # noqa: E402

# Re-use helpers from v1 sweep
from sweep_events import (  # noqa: E402
    SegmentPayload,
    _ppo_at_peak_from_info,
    _region_at_peak_from_info,
    _segment_central_window,
    event_to_dict,
    load_segments,
    segment_to_payload,
    smoke_check,
    write_catalog,
)


# --- Gate (same as v1 production) ---
V2_GATE: GateConfig = GateConfig(
    n_sigma=5.0,
    min_pixels=300,
    min_duration_hours=2.5,
    min_oscillations=3.0,
)


def process_segment_v2(
    args: tuple[SegmentPayload, GateConfig],
) -> list[WaveEvent]:
    """Worker: detect events + compute all Phase 7 metrics."""
    payload, gate = args
    seg_idx = payload.seg_idx
    info = payload.info or {}
    n_samples = len(payload.times)
    n_nan = info.get("NaN_count", 0)
    if n_nan and n_nan > 18 * 60:
        return []
    if n_samples < 18 * 60:
        return []

    b_par = payload.b_par
    b_perp1 = payload.b_perp1
    b_perp2 = payload.b_perp2
    times = payload.times
    dt = 60.0

    # --- Stage 1: detect events (same as v1) ---
    try:
        packets = detect_with_gate(
            b_perp1, b_perp2, times, dt=dt,
            bands=QP_BAND_NAMES, gate=gate,
        )
    except Exception:
        return []

    if not packets:
        return []

    # Central 24h restriction
    central_start, central_end = _segment_central_window(times)
    central = [
        p for p in packets
        if central_start <= p.peak_time < central_end
    ]
    if not central:
        return []

    # --- Stage 2: compute segment-level spectral features ---
    # Power-law FFT background on b_perp1
    try:
        freq_psd, psd1 = welch_psd(b_perp1, dt=dt, nperseg=12 * 60,
                                     noverlap=6 * 60)
        bg1 = estimate_background_powerlaw(psd1, freq_psd)
        ratio_psd1 = psd1 / np.maximum(bg1, 1e-30)

        freq_psd2, psd2 = welch_psd(b_perp2, dt=dt, nperseg=12 * 60,
                                      noverlap=6 * 60)
        bg2 = estimate_background_powerlaw(psd2, freq_psd2)
    except Exception:
        freq_psd, psd1, bg1, ratio_psd1 = None, None, None, None
        freq_psd2, psd2, bg2 = None, None, None

    # Wavelet coherence (once per segment)
    try:
        coh_freq, coh_matrix, coh_phase, _ = wavelet_coherence(
            b_perp1, b_perp2, dt=dt, n_freqs=300,
        )
    except Exception:
        coh_freq, coh_matrix, coh_phase = None, None, None

    # --- Stage 3: enrich each packet ---
    epoch = datetime.datetime(1970, 1, 1)
    time_unix = np.array(
        [(t - epoch).total_seconds() for t in times], dtype=float
    )

    coords = {
        "r": payload.coord_r,
        "th": payload.coord_th,
        "phi": payload.coord_phi,
    }
    median_lt = info.get("median_LT")

    events: list[WaveEvent] = []
    for pkt_idx, pkt in enumerate(central):
        peak_unix = (pkt.peak_time - epoch).total_seconds()
        peak_idx = int(np.argmin(np.abs(time_unix - peak_unix)))
        i_from = int(np.argmin(
            np.abs(time_unix - (pkt.date_from - epoch).total_seconds())
        ))
        i_to = int(np.argmin(
            np.abs(time_unix - (pkt.date_to - epoch).total_seconds())
        ))
        sl = slice(i_from, i_to + 1)
        if i_to <= i_from:
            continue

        # --- Spacecraft coordinates ---
        r_peak = (
            float(coords["r"][peak_idx])
            if coords["r"] is not None else None
        )
        th_peak = (
            float(coords["th"][peak_idx])
            if coords["th"] is not None else None
        )
        phi_peak = (
            float(coords["phi"][peak_idx])
            if coords["phi"] is not None else None
        )
        mag_lat = float(np.degrees(th_peak)) if th_peak is not None else None

        if median_lt is not None:
            lt = float(median_lt)
        elif phi_peak is not None:
            lt = ((np.degrees(phi_peak) / 15.0) + 12.0) % 24.0
        else:
            lt = None

        # --- Amplitudes ---
        b_perp1_amp = float(np.nanmax(np.abs(b_perp1[sl])))
        b_perp2_amp = float(np.nanmax(np.abs(b_perp2[sl])))
        b_par_amp = float(np.nanmax(np.abs(b_par[sl])))
        rms_perp = float(np.sqrt(
            np.nanmean(b_perp1[sl] ** 2 + b_perp2[sl] ** 2)
        ))
        amplitude = max(b_perp1_amp, b_perp2_amp)

        # --- 7.1: FFT screen ratio at event frequency ---
        fft_ratio = None
        if freq_psd is not None and ratio_psd1 is not None and pkt.period_sec:
            f_event = 1.0 / pkt.period_sec
            f_idx = int(np.argmin(np.abs(freq_psd - f_event)))
            fft_ratio = float(ratio_psd1[f_idx])

        # --- 7.2: matched-filter SNR ---
        mf_snr_val = None
        if pkt.period_sec and pkt.period_sec > 0:
            try:
                mf_snr_val = matched_filter_peak_snr(
                    b_perp1, dt=dt, period=pkt.period_sec,
                    t_peak_idx=peak_idx,
                    background=bg1, freq=freq_psd,
                )
            except Exception:
                pass

        # --- 7.3: wavelet coherence over ridge ---
        coh_val, coh_phase_val = None, None
        if (coh_freq is not None and coh_matrix is not None
                and coh_phase is not None and pkt.band):
            try:
                band_obj = get_band(pkt.band)
                coh_val, coh_phase_val = ridge_coherence(
                    coh_matrix, coh_phase, coh_freq,
                    band_obj.freq_min_hz, band_obj.freq_max_hz,
                    i_from, i_to,
                )
            except Exception:
                pass

        # --- 7.8: tapered polarization ---
        try:
            _, phase_deg = phase_shift(
                b_perp1[sl], b_perp2[sl], dt=dt,
                period=pkt.period_sec or 3600.0,
            )
            polarization = classify_polarization(phase_deg)
        except Exception:
            phase_deg = None
            polarization = None

        try:
            ellipticity, inclination_deg, pol_frac = (
                ellipticity_inclination_tapered(b_perp1[sl], b_perp2[sl])
            )
        except Exception:
            ellipticity, inclination_deg, pol_frac = None, None, None

        # --- Wavelet sigma: peak CWT power / noise threshold ---
        # (snr = pkt.prominence is already the peak CWT |W| value,
        #  which in the ridge extractor is the max CWT amplitude on the ridge)
        wavelet_sigma_val = pkt.prominence if pkt.prominence else None

        # --- Transverse ratio ---
        perp_power = np.nanmean(b_perp1[sl] ** 2 + b_perp2[sl] ** 2)
        par_power = np.nanmean(b_par[sl] ** 2)
        tr_ratio = (
            float(perp_power / par_power) if par_power > 1e-20 else None
        )

        # --- Duration / oscillations ---
        duration_sec = (pkt.date_to - pkt.date_from).total_seconds()
        n_osc = (
            duration_sec / pkt.period_sec if pkt.period_sec else None
        )

        # --- 7.6: Dipole invariant latitude ---
        inv_lat = None
        if r_peak is not None and th_peak is not None and phi_peak is not None:
            from qp.coords.ksm import dipole_invariant_latitude
            # Convert KRTP (r, th_lat_rad, phi) → approximate KSM (x, y, z)
            cos_th = np.cos(th_peak)
            x_ksm = r_peak * cos_th * np.cos(phi_peak)
            y_ksm = r_peak * cos_th * np.sin(phi_peak)
            z_ksm = r_peak * np.sin(th_peak)
            try:
                inv_lat = float(dipole_invariant_latitude(x_ksm, y_ksm, z_ksm))
            except Exception:
                pass

        # --- 7.4: quality score ---
        quality = compute_quality(
            wavelet_sigma=wavelet_sigma_val,
            fft_ratio=fft_ratio,
            mf_snr=mf_snr_val,
            coherence=coh_val,
            n_oscillations=float(n_osc) if n_osc else None,
            transverse_ratio=tr_ratio,
            polarization_fraction=pol_frac,
        )

        # --- PPO phase ---
        ppo = _ppo_at_peak_from_info(info, pkt.peak_time, times[0], n_samples)
        region = _region_at_peak_from_info(info, pkt.peak_time)

        event = WaveEvent(
            date_from=pkt.date_from,
            date_to=pkt.date_to,
            period=pkt.period_sec,
            amplitude=amplitude,
            snr=pkt.prominence,
            local_time=lt,
            mag_lat=mag_lat,
            r_distance=r_peak,
            coord_krtp=(
                (r_peak, th_peak, phi_peak)
                if all(v is not None for v in (r_peak, th_peak, phi_peak))
                else None
            ),
            band=pkt.band,
            period_peak_min=(pkt.period_sec / 60.0) if pkt.period_sec else None,
            rms_amplitude_perp=rms_perp,
            b_perp1_amp=b_perp1_amp,
            b_perp2_amp=b_perp2_amp,
            b_par_amp=b_par_amp,
            region=region,
            polarization=polarization,
            phase_deg=phase_deg,
            ppo_phase_n_deg=ppo.get("sls5n"),
            ppo_phase_s_deg=ppo.get("sls5s"),
            event_id=f"seg{seg_idx:05d}-{pkt.band}-{pkt_idx:02d}",
            segment_id=seg_idx,
            n_oscillations=int(n_osc) if n_osc is not None else None,
            ellipticity=ellipticity,
            inclination_deg=inclination_deg,
            polarization_fraction=pol_frac,
            # Phase 7 fields
            fft_screen_ratio=fft_ratio,
            mf_snr=mf_snr_val,
            coherence=coh_val,
            coherence_phase_deg=coh_phase_val,
            wavelet_sigma=wavelet_sigma_val,
            transverse_ratio=tr_ratio,
            quality=quality,
            dipole_inv_lat=inv_lat,
        )
        events.append(event)

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--output", type=Path,
                         default=_PROJECT_ROOT / "Output" / "events_qp_v2.parquet")
    parser.add_argument("--summary", type=Path,
                         default=_PROJECT_ROOT / "Output" / "diagnostics" /
                                 "event_catalog_v2_summary.txt")
    parser.add_argument("--serial", action="store_true")
    args = parser.parse_args()

    n_workers = args.n_workers or max(1, (os.cpu_count() or 4) - 2)
    print("Loading segments...")
    segments, keep_idx = load_segments(args.year)
    if args.max_segments is not None:
        keep_idx = keep_idx[:args.max_segments]
    print(f"  total in archive: {len(segments)}")
    print(f"  to process     : {len(keep_idx)}")

    print("Extracting payloads...")
    payloads: list[SegmentPayload] = []
    for i in keep_idx:
        p = segment_to_payload(i, segments[i])
        if p is not None:
            payloads.append(p)
    print(f"  payloads ready : {len(payloads)}")
    tasks = [(p, V2_GATE) for p in payloads]

    t0 = time.time()
    all_events: list[WaveEvent] = []
    if args.serial:
        for task in tasks:
            all_events.extend(process_segment_v2(task))
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for events in pool.imap_unordered(
                process_segment_v2, tasks, chunksize=4,
            ):
                all_events.extend(events)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.1f} s")
    print(f"  events found    : {len(all_events)}")

    # Quality-score summary
    if all_events:
        quals = [e.quality for e in all_events if e.quality is not None]
        if quals:
            q = np.array(quals)
            print(f"  quality: median={np.median(q):.3f}, "
                  f"p10={np.percentile(q, 10):.3f}, "
                  f"p90={np.percentile(q, 90):.3f}")
            print(f"  quality>0.3: {(q > 0.3).sum()} events "
                  f"({(q > 0.3).sum()/len(q)*100:.1f}%)")
            print(f"  quality>0.5: {(q > 0.5).sum()} events "
                  f"({(q > 0.5).sum()/len(q)*100:.1f}%)")

    out_path = write_catalog(all_events, args.output)
    smoke_check(all_events, args.summary)
    print(f"  catalog at      : {out_path}")
    print(f"  summary at      : {args.summary}")


if __name__ == "__main__":
    main()
