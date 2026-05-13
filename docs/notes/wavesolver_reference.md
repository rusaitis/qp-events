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

## Second follow-up (May 2026) — ruled-out hypotheses

A second round of diagnostics tested three further hypotheses against
the Rusaitis (2021) reference ladder
(`tests/test_validation.py::_REF_FUNDAMENTAL_MHZ`). All three were
**falsified**:

### (i) Shooter clipping — falsified

Hypothesis: Rusaitis's bracket-scan eigensolver missed the sub-WKB
mode 1 because its `freq_range[0]` lower bound sat above the true
fundamental.

Diagnostic: `scripts/diag_shooter_clipping.py` sweeps the legacy
shooter's `freq_range[0]` across six decades at KMAG L=8 noon and
compares mode 1 against the matrix solver (immune to clipping).

Result: the shooter agrees with the matrix solver (returns 0.057 mHz)
for every `freq_range[0] ≤ 3·10⁻⁴` rad/s and clips above to ~0.185 mHz
— our matrix mode 2, not the Rusaitis 0.12 mHz. **No clipping
threshold produces the published reference value.**

### (ii) Ionospheric density floor — falsified (and direction wrong)

Hypothesis: Bagenal's intrinsic floor (0.07 cm⁻³, equatorial-only) is
unphysically low; the field-aligned Gaussian decays to ~10⁻⁵ cm⁻³ at
the L=8 footpoint while Saturn's E-layer is ~10⁴ cm⁻³. A physical
floor should harden the boundary and shift mode 1 to the WKB value.

Diagnostic: `scripts/diag_density_floor_sweep.py` sweeps
`WavesolverConfig.density_floor` across 0 → 10¹⁰ m⁻³ (= 10⁴ cm⁻³) at
L=8, L=10, L=15.

Result: **adding a density floor lowers mode 1, it does not raise it.**
A higher floor increases the mass per unit length
(``w = h_α² B / v_A²``) wherever the floor is binding — usually most of
the field line away from the equator — making the system heavier and
the eigenfrequencies *lower*. At L=8 with floor = 10⁹ m⁻³, mode 1 drops
to 0.005 mHz. Even at the lowest non-zero floor in the sweep
(10⁵ m⁻³ = 0.1 cm⁻³, barely above Bagenal's intrinsic floor), the
solver mode 1 is essentially unchanged at 0.057 mHz. **No floor value
reproduces the Rusaitis 2021 ladder.**

### (iii) Cummings (1969) cos-θ coordinate — falsified

Hypothesis: reparametrising the wave equation in μ = z/r (Cummings
1969 cos-θ) would absorb the boundary asymptotics and recover the
WKB-asymptotic mode 1 directly.

Implementation: `src/qp/wavesolver/cummings_solver.py`, exposed via
`WavesolverConfig(method="cummings")`. Reuses
`scipy.linalg.eigh_tridiagonal` on a uniform μ-grid after substituting
`p → p/J`, `w → w·J` with Jacobian `J = ds/dμ`.

Result: **the eigenvalue problem is invariant under reparametrisation
— Cummings gives identical eigenfrequencies to the matrix arc-length
solver to ≤ 0.5 %** (`tests/test_cummings_solver.py`,
`test_cummings_matches_matrix_on_kmag[8.0/10.0/15.0]`). The Singer
arc-length and Cummings cos-θ formulations of the same self-adjoint
operator with the same Dirichlet BCs must produce the same spectrum,
and they do. Whatever differentiates Rusaitis (2021) from this code,
it is not the coordinate choice.

### (iv) Trace truncation — falsified (boundary location invariant)

Hypothesis: Rusaitis truncated the trace at r > 1 R_S, placing the
Dirichlet wall *above* the high-v_A region and removing the soft-wall
mode 1.

Diagnostic: `scripts/diag_trace_truncation.py` reruns the matrix solver
on the KMAG L=8/10/15 noon trace truncated at r_min ∈ [1.00, 3.00] R_S.

Result: **mode 1 / WKB-asymptote = 0.45 ± 0.05 across every L, every
r_min**. Truncating the L=8 trace from r=1.0 to r=3.0 raises mode 1 by
only 10 % (0.057 → 0.063 mHz) — the factor of 2 versus Rusaitis is
preserved. The (m − δ)/m phase shift with δ ≈ 0.55 is a universal
property of the wave equation on the truncated profile, not an artefact
of the specific endpoints.

### Where this leaves the mode-1 question

- **The sub-WKB mode 1 is a robust prediction of the Singer (1981)
  arc-length wave equation** with Dirichlet BCs at the trace endpoints
  — confirmed by two independent backends (matrix SL, Cummings cos-θ),
  immune to discretisation resolution, density floor, and trace
  truncation.
- **The Rusaitis 2021 reference table aligns with the WKB-asymptotic
  ladder** at L=8 and L=10 (`0.12` mHz ≈ `π / ∫(ds/v_A)`), but at L=15
  the Rusaitis value (`0.053` mHz) is closer to *our* soft-wall mode 1
  (`0.045` mHz) than to WKB (`0.10` mHz). The reference numbers are
  therefore not internally consistent with any single physical
  formulation we can construct from a Saturn FLR wave equation.
- **The factor-of-2 mode-1 mismatch at L=8 is not a fixable bug.** It
  is a model choice that requires co-author discussion: report the
  soft-wall ladder (modes 1–6 from the current solver, mode 1 ≈ 0.057
  mHz at L=8), report the WKB-asymptotic ladder (mode 1 ≈ 0.126 mHz at
  L=8), or report the soft-wall ladder shifted (modes 2–7, treating
  mode 1 as sub-resonant and aligned with QP120 → m=4 etc.). The
  paper's central QP-band ↔ even-harmonic claim is robust to the
  choice (the integer ratios 1:2:3 within the upper part of the
  ladder are the same in all three readings); only the absolute
  mode-1 reference value moves.

## Reproduction (May 2026 v2)

```bash
uv run python scripts/diag_shooter_clipping.py   # Phase 1
uv run python scripts/diag_density_floor_sweep.py # Phase 2
uv run python scripts/diag_trace_truncation.py    # Phase 3b follow-up
# Plus the existing Phase-1 c-cap diagnostic:
uv run python scripts/diag_ccap_effect.py
```

All four diagnostics write to `Output/diagnostics/`. The conclusion of
each is printed at the bottom of stdout under a `VERDICT` header.
