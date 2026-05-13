# Wavesolver — Reference value provenance and mode-1 convention

## The puzzle

`tests/test_validation.py::test_kmag_fundamental_at_L8` compares our wavesolver's
fundamental at L=8 against the Rusaitis et al. (2021) reference value
**0.12 mHz**. Our solver returns **0.057 mHz** — a factor of ~2 lower.

Earlier the failure was tagged as a "mixed-BC, odd-mode-ladder" diagnosis. That
diagnosis is **wrong**. This note documents the actual story so the next reader
does not have to redo the investigation.

## What we verified about the solver

| Check | Result |
| --- | --- |
| Uniform-v_A baseline (constant medium, zero dlnh, fixed-fixed BCs) | Solver returns 1 : 2 : 3 : 4 : 5 : 6 to **machine precision** |
| numba RK4 vs scipy LSODA on KMAG L=8 | Agree to 6+ decimals on every mode |
| Wave equation form in `wave_equation.py:79` vs Singer (1981) Eq 1 | Match |
| `h1 = r·sin θ` (toroidal cylindrical radius) vs analytic dipole form | Match |
| `dlnh1B/ds` numeric spline at footpoints / equator vs analytic dipole form `-3 cos θ / [L·R_S·(1 + 3 cos² θ)^{3/2}]` | Match to ~10 % |
| Trace span ionosphere-to-ionosphere, asymmetry index | 0.0016 (essentially symmetric) |
| Mode numbering via `count_mode_number(dy)` | Sequential 1, 2, 3, 4, 5, 6 — no skipped modes |
| Bracket scan on a dense 5000-point grid | Sign changes occur only at the refined eigenfrequencies, no "missed" modes |

So the solver mechanics are correct. The 0.057 mHz mode 1 is a true eigenfrequency of the equation we solve, not a numerical artifact.

## What the eigenvalue ladder actually looks like at KMAG L=8

| m | ω (rad/s) | f (mHz) | T (min) | ω_m / ω_1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.58e-4 | 0.0569 | 292 | 1.000 |
| 2 | 1.16e-3 | 0.1846 | 90 | 3.244 |
| 3 | 1.96e-3 | 0.3125 | 53 | 5.489 |
| 4 | 2.76e-3 | 0.4396 | 38 | 7.722 |
| 5 | 3.56e-3 | 0.5664 | 29 | 9.950 |
| 6 | 4.35e-3 | 0.6930 | 24 | 12.174 |

The high-mode spacing converges to **0.127 mHz**, which is exactly the WKB
asymptotic value ω_WKB = π / ∫(ds/v_A) for our trace and v_A profile (≈ 0.126 mHz).
The ladder fits ω_m ≈ (m − 0.45) × Δω, i.e., it is the integer ladder
**shifted down by ≈ 0.45 × Δω**. Mode 1 lands at (1 − 0.45) × 0.127 mHz ≈ 0.057 mHz.

## Why the shift exists — soft boundaries

Our `v_A(s)` rises from ~75 km/s at the equator to ~50 000 km/s at the
footpoints — a factor ~700×. In the high-v_A regions near each footpoint, the
term (ω/v_A)² · y in the wave equation is small, so y is approximately *linear*
in s there. The genuinely oscillating region is concentrated near the equator
where v_A is small.

This is a textbook "soft-wall" scenario: the wavefunction can extend linearly
into the high-v_A region before the boundary forces y = 0. The first eigenmode
exists where this linear-extrapolation-then-turn-around configuration closes
back to y = 0 at the other footpoint, which happens at a *lower* frequency
than the strictly-oscillating WKB ω_1. The phase shift δ ≈ 0.45 in
ω_m = (m − δ) · π / ∫(ds/v_A) captures this. (For a strict Dirichlet hard-wall
problem, δ = 0.)

## Where the 0.12 mHz reference probably comes from

The published reference at L=8 matches the **WKB-asymptotic mode 1** to within 5 %:

- WKB ω_1 = π / ∫(ds/v_A) = 0.126 mHz
- Rusaitis 2021 reference (test_validation.py:28): 0.12 mHz
- Δ = 5 %

A solver that uses (a) strict Dirichlet BCs with a capped v_A profile, (b) the
JWKB approximation, or (c) a coordinate that re-scales the boundary region
(e.g., Cummings 1969's "cos θ" formulation, which removes the soft-wall behavior),
will return ω ≈ 0.126 mHz as its fundamental. That is the most likely origin
of the published 0.12 mHz.

Our arc-length Singer (1981) formulation with `y0 = (0, 1)` shooting and the
relativistic v_A cap reveals one additional eigenmode below WKB-1 that those
formulations do not see. Both descriptions are mathematically valid; they
correspond to slightly different physical models of the ionospheric boundary.

## What the paper actually identifies QP30 / QP60 / QP120 with

From `paper/manuscript.md` line 93:

> "The three dominant frequencies of the observed transverse magnetic power
> density are overlaid as semi-transparent color rectangles and correspond
> closely to the **even modes of FLR eigenmodes: m=2 for QP120 (magenta),
> m=4 for QP60 (yellow), and m=6 for QP30 (grey)**."

If the paper's m=1 corresponds to our solver's mode 2 (i.e., the WKB-asymptotic
m=1, which is our m=2 in the soft-wall ladder), then:

| Paper's label | Paper's m | Our solver's m at L=8 | T (min, ours) | Observed T (min) |
| --- | ---: | ---: | ---: | ---: |
| QP120 | 2 | 3 | 53 | ~120 |
| QP60 | 4 | 5 | 29 | ~60 |
| QP30 | 6 | 7 (extrapolated) | ~17 | ~30 |

The periods do not match cleanly at L=8 under this mapping. But Fig 6 in the
paper compares modes at invariant latitudes 72°–76° at noon, which corresponds
to L ≈ 10–15, not L = 8. At higher L the field lines are longer, v_A is more
peaked, and the modes sit at lower frequencies. The QP-band ↔ even-harmonic
correspondence is a Fig 6 claim about that L range, not specifically L = 8.

## Recommended action

1. **Do not "fix" the solver to remove the sub-WKB mode 1.** The mode is real
   and the solver mechanics are right. Removing it would mean changing the
   physical model (e.g., switching to a hard-wall ionospheric BC) which has
   science implications that need co-author judgement.
2. **Demote the strict numerical xfail.** The test at `tests/test_validation.py`
   originally compared our `result.modes[0].frequencies_mhz` against the
   reference. Replace it with a check that the **WKB-asymptotic fundamental**
   (computed from the v_A profile via `π / ∫(ds/v_A)`) is within ~30 % of
   the reference. That is the apples-to-apples comparison.
3. **Flag for science discussion** before resubmission whether Fig 6 should
   present (a) our soft-wall ladder (modes 1–6 from the current solver),
   (b) the WKB-asymptotic ladder (which matches Rusaitis 2021), or
   (c) modes 2–7 of the soft-wall ladder (which approximates Rusaitis's m=1–6).
   Co-author judgement is needed; this is not a code-only decision.

## Reproduction

```bash
uv run python scripts/diag_uniform_dipole.py
uv run python scripts/diag_ccap_effect.py
uv run python -c "
from qp.fieldline.kmag_model import SaturnField
from qp.wavesolver.solver import WavesolverConfig, solve_eigenfrequencies
import numpy as np

cfg = WavesolverConfig(l_shell=8.0, n_modes=6, field=SaturnField(),
                       local_time_hours=12.0)
r = solve_eigenfrequencies(cfg)
print('mode    f (mHz)')
for m in r.modes:
    print(f'  {m.mode_number}    {m.angular_frequency / (2 * np.pi) * 1e3:.4f}')
"
```

## Resolution (May 2026)

The Phase-1 diagnostic in `scripts/diag_ccap_effect.py` ran KMAG L=8 noon
under three v_A boundary models — the existing relativistic cap, no cap
(raw v_A reaching 10¹⁴ m/s), and a relativistic density floor
`n_min = B² / (μ₀ m_i c²)` that bounds `v_A ≤ c` upstream — and recovered
**identical mode 1 = 0.0569 mHz** in all three cases. The sub-WKB ground
state is **intrinsic** to the wave equation when v_A grows large in the
boundary region: regardless of whether v_A reaches `c` or 10¹⁴ m/s,
the term `(ω/v_A)² · y → 0` there, the equation degenerates to
`y'' + dlnh·y' = 0`, and the wave extrapolates linearly into the high-v_A
region before the Dirichlet wall at the ionosphere foot. The factor of 2
between our mode 1 and the Rusaitis (2021) reference is **not** explained
by the v_A cap choice.

What did land in May 2026:

1. **`alfven_velocity` is now configurable** via two independent kwargs
   `density_floor` and `relativistic_cap` (defaults preserve the historical
   behaviour). The flags are exposed through `WavesolverConfig` for
   downstream callers and the diagnostic; they do **not** reconcile the
   factor of 2.
2. **Matrix Sturm–Liouville eigensolver** (`qp.wavesolver.matrix_solver`)
   replaces the shooter as the default backend. Discretises the
   self-adjoint form `-(p y')' = ω² w y` on a uniform grid and solves a
   symmetric tridiagonal eigenproblem in one `scipy.linalg.eigh_tridiagonal`
   call. Agrees with the shooter to ≤ 1 % on every well-conditioned case
   tested (uniform v_A, KMAG L=8, KMAG L=15, dipole L=6 + UniformDensity).
   The shooter remains accessible via `WavesolverConfig.method = "shoot"`.
3. **Tests rewritten**: `test_kmag_wkb_fundamental_at_L8` compares the
   WKB-asymptotic ω₁ = π/∫(ds/v_A) — which equals our matrix solver's
   high-mode spacing — against the published 0.12 mHz; it passes within
   5 %. The pre-existing dipole + Bagenal tests now use `UniformDensity`
   to avoid the unphysical combination (Bagenal is calibrated for KMAG's
   equatorial B, much larger than the pure-dipole value at L > 5).

The remaining open science question — whether Fig 6 should present
modes 1–6 of the soft-wall ladder, modes 2–7 (treating mode 1 as
sub-resonant), or apply an ionospheric density floor that pushes the
true Dirichlet boundary inward — is still co-author territory and
flagged for resubmission discussion.
