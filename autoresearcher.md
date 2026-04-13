# QP Wave Detection Auto-Researcher

Autonomous research loop for improving the QP wave event detection pipeline.

## Goal

**Maximize the composite detection score** in `scoreboard.tsv` by modifying the detection algorithm. The score (weighted harmonic mean of F1, band accuracy, period accuracy, decoy rejection, and IoU) starts at **0.26** — there is enormous room to improve.

## Setup

1. **Read the baseline**: `cat scoreboard.tsv` to see the current best score.
2. **Read the detection pipeline files** to understand what you're working with:
   - `src/qp/benchmark/runner.py` — `_detect_events_in_dataset()`: entry point, hardcoded params
   - `src/qp/events/detector.py` — `detect_wave_packets_multi()`: main detector
   - `src/qp/events/ridge.py` — `extract_ridges()`: CWT blob finding
   - `src/qp/events/threshold.py` — `wavelet_sigma_mask()`, `GateConfig`
   - `src/qp/signal/wavelet.py` — `morlet_cwt()`: CWT computation
   - `src/qp/events/quality.py` — `compute_quality()`: post-detection scoring
3. **Read the scoring**: `src/qp/benchmark/scoring.py` — understand what the score measures.
4. **Verify benchmark data exists**: `ls Output/benchmark/*.zarr | wc -l` should show 40.
5. **Create branch**: `git checkout -b autoresearch/<tag>` from current main.

## What you CAN modify

These files contain the detection pipeline — everything is fair game:

| File | What to tune |
|---|---|
| `src/qp/events/detector.py` | Detection logic, `min_duration_hours`, `min_pixels`, `coi_factor` |
| `src/qp/events/ridge.py` | Blob filtering, peak finding, duration/pixel thresholds |
| `src/qp/events/threshold.py` | `n_sigma`, MAD background estimation, `GateConfig`, FFT screen |
| `src/qp/signal/wavelet.py` | `omega0`, `n_freqs`, frequency range, wavelet family |
| `src/qp/events/quality.py` | Quality scoring weights, normalization anchors |
| `src/qp/benchmark/runner.py` | `_detect_events_in_dataset()` only — detection params and dedup logic |

## What you CANNOT modify

- **Benchmark data generation**: `generator.py`, `scenarios.py`, `noise.py`, `synthetic.py`
- **Scoring framework**: `scoring.py`, `manifest.py`
- **Band definitions**: `bands.py` (physical constraints)
- **Canonical datasets**: `Output/benchmark/*.zarr` — these are fixed
- Do not install new packages beyond what's in `pyproject.toml`

## Running a benchmark

```bash
uv run python scripts/score_pipeline.py --notes "description of what changed"
```

This loads the canonical datasets from `Output/benchmark/`, runs detection, scores, and appends a row to `scoreboard.tsv`. Takes ~15–20 seconds. **Timeout: 5 minutes max** — if it exceeds this, kill it and treat as a crash.

## The scoreboard

`scoreboard.tsv` is tab-separated with these columns:

```
commit  date  score  f1  precision  recall  band_acc  period_err_pct  mean_iou  decoy_rejection  f1@0.5  tier1_recall  tier2_recall  tier3_recall  tier4_recall  loc  runtime_sec  status  notes
```

- `score`: the composite metric to maximize (0–1)
- `loc`: Python lines of code in `src/` (from `cloc`)
- `runtime_sec`: wall-clock time for benchmark run
- `status`: `keep`, `discard`, or `crash`

## The experiment loop

**Max steps: 10** (for testing — increase later for longer runs).

For each step:

### 1. Research

Look at the current scoreboard to identify the weakest metric. The baseline shows:
- Precision 14.6% → too many false detections (biggest opportunity)
- Decoy rejection 8.8% → detector doesn't filter by wave mode
- Period error 24% → frequency estimates are rough
- Gap handling → detector crashes on NaN (0% recall on gap scenarios)

Think about what change would give the biggest score improvement. Read relevant code. Consider:
- Threshold tuning (raise `n_sigma` or `min_pixels` to cut false positives)
- Transverse ratio filtering (reject compressional detections)
- NaN-aware CWT (handle gaps gracefully)
- Better deduplication (merge overlapping same-band detections)
- Quality gating (use `compute_quality()` to filter low-quality detections)
- Enabling the FFT pre-screen
- Tuning `omega0` for better frequency resolution

### 2. Implement

Make targeted changes. Prefer small, focused modifications over large rewrites.

### 3. Run

```bash
uv run python scripts/score_pipeline.py --notes "brief description"
```

If it crashes, read the error, fix it, and re-run. If you can't fix after 2 attempts, revert and try something else.

### 4. Evaluate

Read `scoreboard.tsv` to compare against the previous best.

**Keep if:**
- Score improved by ≥ 0.005 (0.5 percentage points), OR
- Score is equal but LOC decreased (simplification win), OR
- Score improved AND LOC increase is ≤ 10%

**Discard if:**
- Score decreased or stayed the same with more code
- Score improved < 0.005 but LOC increased > 10%

### 5. Commit or revert

**Keep:** `git add -A && git commit -m "feat: <what changed>"` — advance the branch.

**Discard:** `git checkout -- .` to revert all changes. Log the attempt in the scoreboard with `status=discard` anyway (so we know what was tried).

**Crash:** Log with `status=crash` and `score=0.0000`. Revert.

### 6. Repeat

Go back to step 1 with the updated codebase. After 5 steps, stop.

## Key insights for improvement

The baseline (score=0.26) reveals a detector that finds everything but can't say no:

1. **Precision is 14.6%** — for every real event found, there are ~6 false detections. The `max/4` fallback threshold in `extract_ridges()` is extremely permissive. Using `wavelet_sigma_mask()` with `n_sigma=3–5` would dramatically cut false positives.

2. **Decoy rejection is 8.8%** — the detector finds "events" in compressional oscillations, single pulses, broadband bursts, PPO, and step functions. A transverse-ratio filter or quality gate would reject most of these.

3. **Gap scenarios produce 0% recall** — the CWT computation likely fails or produces garbage on NaN-containing data. NaN interpolation before CWT would help.

4. **Period error is 24%** — the CWT ridge peak period could be refined by fitting a parabola to the power spectrum around the peak.

## Code size watchdog

Current baseline: ~10,629 Python LOC. Each run records LOC. If code grows beyond ~11,700 (10% increase) without a meaningful score improvement (≥ 0.01), the change should be discarded. Simpler code that achieves the same score is always preferred.

A good detection algorithm should be tight, not sprawling. The goal is a better algorithm, not more code.
