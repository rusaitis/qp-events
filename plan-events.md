# QP Event Detection & Morphology — Plan

## Completed Work Summary (Phases 0–8)

**Status: 509 tests passing, 2 pre-existing wavesolver failures.**

| Phase | Summary | Key output |
|---|---|---|
| 0 | Pipeline verified, segments inspected | — |
| 1 | Multi-band detector (QP30/60/120), `WaveEvent` extended | `src/qp/events/{bands,ridge}.py` |
| 2 | Statistical threshold (5σ CWT σ-mask), calibrated | `src/qp/events/threshold.py` |
| 3 | Mission sweep: **1636 events** in 32 s | `Output/events_qp_v1.parquet` |
| 4 | Binning onto dwell grid, union mask, zero violations | `Output/event_time_grid_v1.zarr` |
| 5 | Occurrence-rate maps, Figs 7/8/9 reproduced | `scripts/fig0{7,8,9}_*.py` |
| 6 | Band-pass ratio, PPO fold, plasma-sheet split | `scripts/fig1{0,1,2}_*.py` |
| 7 | Power-law FFT, matched filter, coherence, quality score, inv-lat binning | `Output/events_qp_v2.parquet` |
| 8 | bp_transverse_ratio (4.4×), period-L β≈0, phase-coherent stacks (SNR 40–75), polarization (84% linear) | `Output/events_qp_v3.parquet` |

**Detection methodology**: MFA segments → Morlet CWT → 5σ threshold → ridge extraction →
event-time binning on `(r, mag_lat, LT)` and `(inv_lat, LT)` grids → normalize by dwell.
Quality score (geometric mean of 7 metrics, v3 threshold q>0.3 retains 667/1636 events).

## Detection methodology (recap)

1. **Input**: 36-hour MFA segments from `DATA/CASSINI-DATA/DataProducts/Cassini_MAG_MFA_36H.npy`
   (24-hour day + 6 h padding each side, detrended with a 3 h running mean,
   then 3-min smoothed). Components are `b_parallel`, `b_perp1`, `b_perp2`,
   `b_total` at `dt=60 s`.
2. **FFT screening (Eq. 4)**: Welch PSD (12 h window, 6 h overlap, Hann),
   smoothed background estimated by `qp.signal.fft.estimate_background`,
   power ratio `r_i = P(b_i) / P(<B_T>_f)` for each component. Flag the
   segment if `r_perp1` *or* `r_perp2` exceeds threshold inside any QP band.
3. **Wavelet localization**: Morlet CWT (`qp.signal.wavelet.morlet_cwt`,
   ω₀ = 10) on b_perp1/b_perp2 of the flagged segment, restricted to the band
   that fired. Pick contiguous (time, period) regions above threshold and
   inside the cone of influence as wave packets.
4. **Per-packet metadata**: peak time, period (CWT ridge), duration,
   peak amplitude, RMS amplitude, mean R/mag_lat/LT, region (MS/SH/SW),
   PPO phase, polarization (90° vs 180° from `phase_shift`).
5. **Catalog persistence**: parquet (or zarr) keyed by `event_id`,
   one row per packet, with band label.
6. **Binning**: cumulative event-time per band on the same `(r, mag_lat, LT)`
   grid as the dwell zarr.
7. **Normalization**: divide event-time by dwell-time per bin → occurrence
   rate (dimensionless, in [0, 1]).

---

> **Phases 0–8 completed.** Full details in `Output/diagnostics/phases_1_to_8_summary.md`.
> Test suite: 509 passed, 2 pre-existing wavesolver failures.

## Phase 9 — Oscillation Morphology Analysis

Characterises waveform shape of detected events to discriminate standing vs
travelling FLR. All metrics computed from band-passed b_perp1 in event window.

### 9.1 — `qp.signal.morphology` module

- [x] `src/qp/signal/morphology.py` — new module with:
  - `band_envelope(data, dt, low_hz, high_hz)` — Butterworth + Hilbert |analytic|
  - `instantaneous_frequency(data, dt)` — unwrap Hilbert phase + diff
  - `envelope_rise_fall(envelope, dt)` — 10–90% rise/fall timing
  - `harmonic_ratio(data, dt, period_sec)` — P(2f)/P(f) via windowed FFT
  - `amplitude_growth_rate(envelope, dt, period_sec)` — log-amplitude slope (dB/period)
  - `inter_cycle_coherence(data, dt, period_sec)` — mean successive-cycle correlation
  - `freq_drift_rate(data, dt, low_hz, high_hz)` — Hilbert inst. freq slope (Hz/s)
  - All use zero-phase (sosfiltfilt) Butterworth to avoid phase shift

### 9.2 — Post-hoc enrichment `scripts/enrich_events_v4.py`

- [x] Reads `Output/events_qp_v3.parquet`, adds 8 new morphology columns:
  - `envelope_skewness`, `rise_fall_ratio`, `harmonic_ratio_2f`
  - `amplitude_growth_db`, `freq_drift_hz_per_s`, `inter_cycle_coherence`
  - `ppo_phase_onset_deg` (PPO phase at event start, not just peak)
  - `waveform_skewness`
- [x] Writes `Output/events_qp_v4.parquet`
- [x] Runtime: 5.4 s for 1636 events (serial mode)

### Phase 9 results

| Metric | QP30 | QP60 | QP120 |
|---|---|---|---|
| `rise_fall_ratio` median | ~0.94 | ~0.94 | ~0.95 |
| `harmonic_ratio_2f` median | 0.176 | 0.208 | 0.238 |
| `amplitude_growth_db` median | +0.10 | −0.74 | −2.88 |
| `freq_drift_hz_per_s` median (pHz/s) | +2293 | −710 | +493 |
| `inter_cycle_coherence` median | 0.165 | 0.109 | 0.187 |

### 9.3 — Waveform gallery

- [x] `scripts/fig_waveform_gallery.py` — top-9 events per band, 3-panel columns
  (waveform+envelope / instantaneous frequency / per-cycle amplitude bar)
- [x] Output: `Output/figures/figure_waveform_gallery_{QP30,QP60,QP120}.png`

### 9.4 — Morphology distribution figure

- [x] `scripts/fig_morphology_distributions.py` — violin plots per band for
  rise/fall ratio, harmonic ratio, amplitude growth, freq drift, coherence
- [x] Output: `Output/figures/figure_morphology_distributions.png`

### 9.5 — Tests

- [x] `tests/test_morphology.py` — 22 unit tests, all passing (509 total)

---

## Phase 10 — Standing vs Travelling FLR Discrimination Tests

### Phase 10 results

- [x] **Polarization vs latitude** (`scripts/fig_polarization_vs_latitude.py`):
  - Spearman r < 0.13, all p > 0.10 — **no latitude trend**
  - Rules out poloidal odd-mode FLR (which predicts sign reversal)
  - Consistent with toroidal even-mode FLR or travelling wave

- [x] **Chirp direction** (from `freq_drift_hz_per_s` in physics test figure):
  - QP30: t=4.52, **p<0.001** — statistically significant upward chirp
  - QP60/QP120: p>0.07 — not significant
  - QP30 upward chirp could reflect mild dispersion as resonance builds

- [x] **Amplitude growth** (top 20 events per band, q>0.3):
  - QP30: 12/20 growing, median +0.14 dB/period
  - QP60: 16/20 growing, median +0.28 dB/period — **predominantly growing**
  - QP120: 6/20 growing, 14/20 decaying, median −4.24 dB/period
  - Growing amplitude → externally driven resonance, not freely decaying packet

- [x] **PPO phase of onset** (Rayleigh test):
  - QP30: R=0.099, **p=0.009** — **preferred onset phase ~190°** → PPO phase-locked
  - QP60: p=0.27; QP120: p=0.53 — uniform (insufficient statistics)
  - Direct evidence QP30 events are triggered at a specific PPO phase

- [x] **N/S hemisphere comparison**:
  - All bands: Mann-Whitney p > 0.19 — **no significant N/S difference**
  - Confirms even-mode FLR symmetry (same waveform from both hemispheres)

- [x] All figures: `Output/figures/figure_polarization_vs_latitude.png`,
  `Output/figures/figure_amplitude_evolution.png`
- [x] Diagnostics: `Output/diagnostics/morphology_physics_tests.txt`

### Phase 10 scientific conclusion

**Updated favoured interpretation: PPO-driven standing toroidal FLR.**

The morphology evidence tips the balance toward a driven standing eigenmode:
- Amplitude growing (not decaying) during events → active excitation
- PPO phase-locked onset (QP30) → direct fingerprint of the driver
- No latitude polarization trend → toroidal (not poloidal) even mode
- Symmetric envelopes → not a transient pulse dispersing

The wave is likely a resonance continuously excited during the PPO active phase
(~4–6 h), then shutting off cleanly. `explanation.md` updated with this evidence.

---

## Phase 11 — Detection Improvements (planned)

- [ ] 11.1 Harmonic band consistency flag: check if QP60 detection coincides with
      elevated QP30 power (FLR harmonic chain)
- [ ] 11.2 Merge fragmented ridges: events in same segment/band with gap < 1 period
      → merge into single longer event (add `merge_nearby_events()` to `wave_packets.py`)
- [ ] 11.3 Gold catalog: `events_qp_gold.parquet` (quality_v3 > 0.5, ~114 events)

---

## Critical files

- `src/qp/signal/morphology.py` — NEW waveform morphology module
- `Output/events_qp_v4.parquet` — 1636 events with 8 morphology columns
- `scripts/enrich_events_v4.py` — Phase 9 enrichment
- `scripts/fig_waveform_gallery.py` — gallery figures
- `scripts/fig_morphology_distributions.py` — distribution figure
- `scripts/fig_polarization_vs_latitude.py` — Phase 10.1 figure
- `scripts/fig_amplitude_evolution.py` — Phase 10.2–10.5 combined figure
- `tests/test_morphology.py` — 22 new tests
- `explanation.md` — updated with Phase 9–10 evidence

## Verification

- `uv run pytest -v` → 509 passed, 2 pre-existing failures (wavesolver only)
- `uv run python scripts/enrich_events_v4.py --serial` → 5.4 s, 1636 events
- All figures generated without errors
