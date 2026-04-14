# QP Wave Detection Auto-Researcher

Autonomous research loop for improving the QP wave event detection pipeline.

## Goal

**Improve the composite detection score on a realistic 64-scenario benchmark** while keeping the detection method physically motivated, general, and defensible in a paper. The benchmark (v3 hardened) includes multi-band co-occurrence, realistic noise (alpha=1.2, PPO modulation), broadband decoys, log-uniform-jittered periods within each QP band, 2–4.5-cycle short packets that stress `min_oscillations`, continuous-spectrum population tests, and out-of-band decoy waves (15–19, 80–90, 155–180 min) that stress the edge veto.

**Current score: 0.791** (composite, default weights) on the post-round-4 hardened
benchmark — see "Post-hardening baseline" below. The pre-hardening score on the
older v3 benchmark was ~0.937; the drop reflects a strictly harder benchmark
(real-data-calibrated FGM decoys, regime-switching / heavy-tailed noise, OOD
holdout family, phase-locked sawtooth harmonics), not a detector regression.

The main bottlenecks now are:
- **Decoy rejection (0.828)** — the FGM-artifact decoys (impulsive spikes, range-change steps, SCAS plateaus) and the broadband-burst / PPO-only scenarios still leak
- **OOD holdout precision** — `holdout_band_edges` F1 = 0.67, `holdout_unseen_regime` F1 = 0.76 (recall is 1.00 in both — extra detections, not misses)
- **tier2 F1 = 0.895** — co-occurrence recall (0.820) still the weakest QP tier
- **Macro band accuracy = 0.699** — one band drags the macro-mean below the micro-mean (0.902)

### Post-hardening baseline (2026-04-14, commit `ae8dd3b`)

After four rounds of synthetic hardening (`b6e1417 → 80dd385 → 70bc35e → e3742bd → ae8dd3b`):

| metric | value |
|---|---|
| Composite (default) | **0.791** |
| F1 | 0.907 |
| Precision / Recall | 0.908 / 0.906 |
| Band accuracy (micro / macro) | 0.902 / 0.699 |
| Decoy rejection | 0.828 |
| Wall time (cached datasets) | ~60 s |

| tier | recall | F1 |
|---|---|---|
| tier1 | 0.915 | 0.925 |
| tier2 | 0.820 | 0.895 |
| tier3 | 0.958 | 0.935 |
| tier4 | 0.887 | 0.893 |

This is the new floor. Further detector work should be measured against this
number, not the historical 0.937 / 0.889 baselines that were on the older
synthetic suite.

## Philosophy

Prefer **well-justified general techniques** over hand-tuned parameters. A method that works because of a physical or statistical principle is worth more than one that works because a threshold was tuned to the benchmark.

**Good approaches:**
- Standard signal processing techniques (matched filtering, coherence analysis, adaptive thresholding)
- Physics-based discrimination (wave mode identification, polarization analysis, dispersion relations)
- Robust statistical methods (information-theoretic criteria, Bayesian detection, non-parametric tests)
- Established methods from the space physics / geophysics literature

**Bad approaches:**
- Band-specific thresholds (different magic numbers for QP30 vs QP60 vs QP120)
- Scenario-specific heuristics that only help one benchmark case
- Complex multi-stage pipelines where each stage has 3+ tuned parameters
- Overfitting to the benchmark's specific noise seeds or event placements

**Every new filter or threshold must answer: "What is the physical or statistical principle behind this, and would a referee accept it in two sentences?"**

## Setup

1. **Read the baseline**: `cat scoreboard.tsv | tail -5` to see recent scores.
2. **Read the detection pipeline**: focus on `src/qp/benchmark/runner.py` (`_detect_events_in_dataset()`) and `src/qp/events/detector.py` (`detect_with_gate()`, `filter_detections()`).
3. **Read the scoring**: `src/qp/benchmark/scoring.py` — understand what the composite score rewards.
4. **Read the scenarios**: `src/qp/benchmark/scenarios.py` — understand the 54 scenarios, especially the co-occurrence and decoy ones.
5. **Create branch**: `git checkout -b autoresearch/<tag>` from current branch.

## Current pipeline anatomy

The detection in `_detect_events_in_dataset()` (benchmark path):

| Stage | Physical basis |
|---|---|
| NaN interpolation | Data gaps in Cassini MAG |
| CWT (2 transverse components) | Standard time-frequency analysis |
| Joint power = (|cwt1| + |cwt2|) / 2 | Combined transverse Alfven wave power |
| Single sigma-mask at 3.5 sigma | Robust MAD-based outlier detection |
| Ridge extraction per QP band | Connected-component labeling in (time, period) |
| `filter_detections()` post-processing | Min oscillations, transverse ratio, spectral concentration (disabled), dedup |

Production path (`detect_with_gate()`) is similar but also computes parallel CWT for transverse ratio filtering.

### Known weaknesses

1. **Co-occurrence blindness**: When QP60 and QP120 are simultaneous, the ridge extractor finds the stronger one. The weaker signal's ridge gets swallowed or doesn't form a separate connected component.
2. **Spectral concentration disabled**: Was killing real events, now disabled — but this means broadband decoys leak through. Need a replacement discriminator.
3. **No coherence test**: The pipeline doesn't check whether the detected signal is a coherent wave (consistent phase across cycles) vs. incoherent noise.
4. **Single threshold**: The 3.5-sigma mask is a single global threshold. Adaptive or multi-scale thresholding might recover weak signals near strong ones.

## What you CAN modify

| File | What to do |
|---|---|
| `src/qp/benchmark/runner.py` | Modify `_detect_events_in_dataset()` — the benchmark detection path |
| `src/qp/events/detector.py` | Modify `detect_with_gate()`, `filter_detections()`, or add new detection functions |
| `src/qp/events/ridge.py` | Improve ridge extraction (e.g., multi-scale, adaptive thresholds) |
| `src/qp/events/threshold.py` | Improve thresholding (e.g., adaptive sigma, scale-dependent masks) |
| `src/qp/signal/wavelet.py` | Modify CWT parameters or add new transforms |
| `src/qp/events/quality.py` | Add quality metrics that discriminate waves from noise |

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

This loads the canonical datasets from `Output/benchmark/`, runs detection, scores, and appends a row to `scoreboard.tsv`. Takes ~25-30 seconds. **Timeout: 5 minutes max.**

## The experiment loop

**Max iterations: 20.**

For each iteration:

### 1. Research

Before changing anything, ask:

- **What is the biggest score bottleneck right now?** Check per-tier recall, decoy rejection, and F1 separately.
- **Is there a known signal processing or space physics technique that addresses this?** Use web search if you're stuck — look for methods in the literature (e.g., wavelet coherence, Hilbert-Huang transform, matched filtering for dispersive waves, polarization filtering).
- **Will this approach generalize to real Cassini data?** If it only helps one scenario, it's overfitting.

### 2. Implement

Priorities (in order):

1. **Fix the co-occurrence problem** — this is the single biggest score drag. Ideas: separate ridge extraction per component before combining, iterative detection (find strongest, subtract, re-detect), multi-scale thresholding.
2. **Replace spectral concentration** with a better broadband discriminator — e.g., wavelet coherence (coherent waves have consistent phase, noise doesn't), or temporal regularity (waves are quasi-periodic, noise isn't).
3. **Improve weak-signal detection** — adaptive thresholds that lower the sigma near strong detections, or matched filtering against known wave templates.
4. **Reduce false positives** — only if decoy rejection is the bottleneck.

**Hard rules:**
- Every threshold must have a comment explaining the physical or statistical reasoning
- No band-specific special cases unless physically justified (e.g., QP120 has different coherence properties because of CWT resolution)
- Prefer fewer, more general techniques over many specific filters
- If you can't explain a method in one sentence to a referee, simplify it
- New code is fine as long as it improves the score — this is not a LOC reduction exercise

### 3. Run

```bash
uv run python scripts/score_pipeline.py --notes "brief description"
```

### 4. Evaluate

**Keep if:**
- Composite score improved by >= 0.005
- Score stayed the same but a specific bottleneck improved (e.g., co-occurrence recall up, decoy rejection up) without regressing others
- A principled method was added that improves one metric without hurting others, even if composite gain is < 0.005

**Discard if:**
- Score dropped
- New code adds scenario-specific heuristics
- Method is not explainable in 1-2 sentences

### 5. Commit or revert

**Keep:** `git add -A && git commit -m "feat: <what improved>"` or `refactor:` if restructuring.

**Discard:** `git checkout -- .` to revert.

### 6. When stuck — search the literature

If you've tried 5+ iterations without improvement, **use web search** to find new approaches:

- Search for: "wavelet coherence wave detection", "quasi-periodic signal detection magnetosphere", "matched filter dispersive waves", "time-frequency ridge extraction multi-component"
- Look at recent papers on ULF wave detection, FLR identification, or magnetospheric wave catalogs
- Check if scipy, pywt, or other installed packages have relevant functions you haven't tried

Then implement the most promising approach and test it.

### 7. Repeat

## Score breakdown reference

The composite score is a weighted harmonic mean:
- 0.35 x F1 (detection rate — precision x recall balance)
- 0.20 x band_accuracy (correct band identification)
- 0.15 x period_accuracy (period error, clamped)
- 0.15 x decoy_rejection_rate (false positive suppression)
- 0.15 x mean_iou (temporal localization)

Current bottlenecks (post-hardening 0.791 baseline, 2026-04-14):
- F1 = 0.907 (tier2 recall = 0.820 still co-occurrence-limited)
- Band accuracy micro = 0.902, macro = 0.699 (one band underperforms)
- Decoy rejection = 0.828 (FGM-artifact + broadband-burst leakage)
- OOD holdout precision drag — `holdout_band_edges` F1 = 0.67, `holdout_unseen_regime` F1 = 0.76 (over-detection, not misses)

**Biggest gains available: decoy rejection (0.83 → 0.95+) and OOD precision (cut spurious detections in `holdout_*`).**
