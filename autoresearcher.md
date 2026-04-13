# QP Wave Detection Auto-Researcher

Autonomous research loop for improving the QP wave event detection pipeline.

## Goal

**Build a detection algorithm that is simple, physically motivated, and defensible in a paper.** The composite detection score in `scoreboard.tsv` tracks progress, but the real goal is a method you could describe in 2–3 sentences in a Methods section with clear physical reasoning for every filter and threshold.

**Current score: ~0.905.** A slight reduction (down to ~0.88) is acceptable if it comes with significant simplification or better physical motivation. A score increase is welcome only if the method stays clean.

## Philosophy

The current pipeline (`_detect_events_in_dataset()` in `runner.py`) grew organically through 40 iterations of score-chasing on synthetic data. It now has several heuristics that are hard to justify physically:

- Magic thresholds tuned to specific scenarios (60% spectral concentration, 0.5 transverse ratio)
- A single-component fallback that only fires when AND mask is empty
- Ridge splitting via peak-finding with arbitrary prominence thresholds
- Duplicate detection logic that belongs in the core detector, not the benchmark wrapper

**Every filter must answer: "Why would a physicist apply this to real Cassini data?"**

Good reasons: "Alfvén waves are transverse, so compressional detections are rejected" (transverse ratio). "The MAD-based σ-mask is a standard robust outlier detection method" (sigma mask).

Bad reasons: "60% threshold gave the best score on the benchmark" (overfitting). "We split ridges because QP120 produces one giant blob" (implementation artifact).

## Setup

1. **Read the baseline**: `cat scoreboard.tsv | tail -5` to see recent scores.
2. **Read the detection pipeline**: focus on `src/qp/benchmark/runner.py` (`_detect_events_in_dataset()`) — this is where all the benchmark-specific logic accumulated.
3. **Read the core detector**: `src/qp/events/detector.py` — `detect_with_gate()` already has a cleaner, more principled pipeline. Consider using it instead of the ad-hoc logic in runner.py.
4. **Read the scoring**: `src/qp/benchmark/scoring.py` — understand what the score measures.
5. **Create branch**: `git checkout -b autoresearch/<tag>` from current branch.

## What you CAN modify

| File | What to do |
|---|---|
| `src/qp/benchmark/runner.py` | **Primary target.** Simplify `_detect_events_in_dataset()`. Move logic into core modules or delete it. |
| `src/qp/events/detector.py` | Move physically-motivated filters here (they belong in the detector, not the benchmark wrapper) |
| `src/qp/events/ridge.py` | Simplify if possible |
| `src/qp/events/threshold.py` | Adjust `GateConfig` defaults if they better reflect the physics |
| `src/qp/signal/wavelet.py` | Only if changing CWT parameters for physical reasons |
| `src/qp/events/quality.py` | Only if integrating quality gating into the pipeline |

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

This loads the canonical datasets from `Output/benchmark/`, runs detection, scores, and appends a row to `scoreboard.tsv`. Takes ~15–20 seconds. **Timeout: 5 minutes max.**

## The experiment loop

**Max steps: 10.**

For each step:

### 1. Research

Before changing anything, ask:

- **What is the simplest detection pipeline that is physically defensible?** The core idea: CWT → σ-mask → ridge extraction → post-filters. Each step has a clear role.
- **What logic in `_detect_events_in_dataset()` should live in the core detector modules instead?** The benchmark wrapper should be thin — ideally just calling `detect_with_gate()` or similar.
- **What filters have clear physical motivation vs. what is benchmark tuning?**
- **Can I delete code and keep the score above ~0.88?**

### 2. Implement

Priorities (in order):

1. **Delete or simplify** heuristics that lack physical motivation
2. **Move** physically-motivated logic from `runner.py` into `detector.py` or `threshold.py` where it belongs
3. **Consolidate** — use `detect_with_gate()` or extend it rather than reimplementing in the benchmark wrapper
4. **Improve** — only add new logic if it has a clear physical basis and the paper sentence is obvious

**Hard rules:**
- Every threshold must have a comment explaining the physical reasoning, not just what value was chosen
- No band-specific special cases (e.g., different thresholds for QP120 vs QP60) unless physically justified
- Prefer fewer, more general filters over many specific ones
- If you can't explain a filter in one sentence to a referee, delete it

### 3. Run

```bash
uv run python scripts/score_pipeline.py --notes "brief description"
```

### 4. Evaluate

**Keep if:**
- LOC decreased AND score ≥ 0.88 (simplification win — this is the primary goal)
- Score improved by ≥ 0.005 with no LOC increase
- Score stayed the same but code is measurably simpler (fewer magic numbers, logic moved to proper modules)

**Discard if:**
- LOC increased without a clear physical justification for the new code
- Score dropped below 0.88 (too much regression)
- New code adds benchmark-specific heuristics (overfitting)

### 5. Commit or revert

**Keep:** `git add -A && git commit -m "refactor: <what simplified>"` — prefer `refactor:` over `feat:` when simplifying.

**Discard:** `git checkout -- .` to revert. Log with `status=discard`.

### 6. Repeat

## Current pipeline anatomy

The detection in `_detect_events_in_dataset()` currently has these stages:

| Stage | LOC | Physical basis | Concern |
|---|---|---|---|
| NaN interpolation | 8 | Data gaps in Cassini MAG | Fine — necessary preprocessing |
| CWT (3 components) | 12 | Standard time-frequency analysis | Fine, but 3 CWTs is expensive |
| Coincidence AND mask | 3 | Alfvén waves are transverse in both perp components | **Good** — physically motivated |
| Single-component fallback | 12 | Linearly polarized waves exist | OK but the trigger logic is ad-hoc |
| Ridge splitting at minima | 40 | Long CWT ridges at QP120 merge separate packets | **Suspect** — artifact of CWT resolution, not physics |
| Band row mask setup | 8 | — | Bookkeeping |
| Min oscillations filter | 4 | Wave packet needs ≥N cycles to be identifiable | **Good** |
| CWT transverse ratio | 4 | Alfvén waves have perp > par power | **Good** — uses in-band CWT power |
| Spectral concentration | 14 | Reject broadband transients | **OK** but the 60% threshold is tuned |
| Same-band dedup | 8 | Avoid double-counting | Fine |
| **Total** | ~115 | | Should be ~40–50 |

**Target: reduce `_detect_events_in_dataset()` to ≤50 lines** by moving defensible filters into the core detector and deleting the rest.

## What the Methods section should say

> "We detect quasi-periodic wave packets using a continuous wavelet transform (Morlet, ω₀=10) of both transverse MFA components. Statistically significant features are identified using a robust σ-mask (median + n·MAD of background period rows, n=3). We require coincidence between both transverse components to confirm the Alfvén wave polarization signature, with a single-component fallback for linearly polarized events. Connected ridges in the (time, period) plane exceeding 2 hours are classified into QP30/QP60/QP120 bands. Post-detection filters reject compressional contamination (in-band transverse-to-parallel CWT power ratio > 0.5) and broadband transients (spectral concentration test)."

That's ~80 words. Every word should map to code, and every piece of code should map to a word.

## Code size watchdog

Current: ~10,784 LOC. **Target: ≤ 10,700 LOC** (net reduction). Each run records LOC.

The `_detect_events_in_dataset()` function is currently ~225 lines. It should be ≤60.
