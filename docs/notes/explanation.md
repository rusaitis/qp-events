# Wave Interpretation: PPO-Driven Standing Toroidal FLR

## Background

The paper argues that QP30/QP60/QP120 waves are signatures of **field line resonances (FLR)** — Alfvén waves standing between northern and southern ionospheres. The three period bands match even harmonics of the FLR eigenfrequencies computed from the KMAG field model with Bagenal & Delamere density at L ~ 18–22 Rs.

The proposed driver is **periodic vertical flapping of the magnetotail** at the PPO period (~10.7 h), which excites Alfvénic perturbations at resonant field lines. Phases 1–10 now provide a comprehensive set of observational constraints on whether these are standing eigenmodes or transient resonant wave packets.

---

## Full Evidence Table

| Observation | Standing eigenmode | Transient resonant packet | Measured result |
|---|---|---|---|
| β ≈ 0 (no T–L trend within band) | Consistent: PPO driver fixes period, resonance selects L-shell | Consistent: same reasoning | **β ≈ 0 confirmed (Phases 3–5)** |
| Events cluster at L ≈ 18–22 Rs | Consistent: only L where eigenfreq = driver freq is selected | Same | **L ≈ 18–22 Rs confirmed** |
| Periods match KMAG even harmonics | Consistent | Consistent | **Exact match (Phase 1, Fig 6)** |
| Median duration 4–6 h (3–10 cycles) | Borderline: few bounces, but low-Q driven resonance can build in 3–5 cycles | Consistent | **3–10 cycles measured** |
| Band-pass transverse ratio 4.4× | Consistent: toroidal FLR is transverse | Consistent | **4.4× (Phase 8)** |
| Linear polarization (84%, Stokes) | Consistent: toroidal mode is linearly polarized | Consistent: single-pass pulse | **84% linear (Phase 8)** |
| PPO-modulated separation times (~10.7 h) | Consistent: PPO drives both | Same | **Confirmed (Phase 5, Fig 9)** |
| Phase-coherent stacks (SNR 16–75) | Consistent: repeatable coherent signal | Consistent: repeatable driven oscillation | **SNR 16–75 (Phase 8)** |
| Highest occurrence post-dusk, mid-high lat | Consistent: PPO tail flapping geometry | Same | **Confirmed (Figs 7, 8)** |
| Waveform symmetry (rise/fall ratio) | ≈ 1 (symmetric standing mode) | < 1 (sharp onset, slow decay) | **Median 0.94 — nearly symmetric (Phase 9)** |
| Harmonic content P(2f)/P(f) | Low (sine-like) | Low | **Median 0.20 — moderate non-sinusoidal (Phase 9)** |
| Amplitude growth per cycle (top-20 by SNR) | Grow then plateau (driven resonance building) | Monotonic decay (dispersing packet) | **QP30/60 mostly growing: 12/20, 16/20 (Phase 10)** |
| QP120 amplitude trend (top-20 by SNR) | Weaker driving at longer period | Same | **QP120 mostly decaying: 14/20 (Phase 10)** |
| Frequency drift (chirp) | ≈ 0 (symmetric standing mode) | Nonzero if dispersive | **QP30 significant upward chirp (t=4.5, p<0.001); QP60/QP120 not significant (Phase 10)** |
| Polarization vs latitude | Toroidal even-mode: no trend; Poloidal: sign reversal | No trend | **No trend (Spearman r<0.13, all p>0.1) — rules out poloidal (Phase 10)** |
| N/S hemisphere ellipticity | Even mode: identical | No difference | **Identical (Mann-Whitney p>0.19) — confirms even-mode (Phase 10)** |
| PPO phase of onset | Peaked if PPO-driven; uniform if freely ringing | Peaked if PPO-driven | **QP30 non-uniform (Rayleigh R=0.099, p=0.009, mean 189.8°) — PPO phase-locked (Phase 10)** |
| QP60/QP120 PPO phase of onset | — | — | **Uniform (p>0.27) — insufficient statistics** |

---

## Favoured Interpretation: PPO-Driven Standing Toroidal FLR

The morphology evidence decisively shifts the interpretation from "ambiguous" toward **driven standing toroidal FLR**:

1. **Amplitude growing in QP30 and QP60** (12/20 and 16/20 highest-SNR events, medians +0.14 and +0.28 dB/period) — signature of an externally *driven* resonance building up, not a freely decaying packet. QP120 mostly decaying (14/20, median −4.2 dB/period) — possibly less efficiently coupled at longer period.

2. **QP30 PPO phase-locked onset** (Rayleigh p=0.009, preferred phase ~190°) — direct evidence that the PPO driver triggers each event. QP60/QP120 show no preference (p>0.27), likely insufficient statistics (154 and 32 events vs 481).

3. **Toroidal even-mode confirmed** — no latitude polarization trend (rules out poloidal odd-mode sign reversal) and identical N/S hemisphere ellipticity (confirms even-mode symmetry).

4. **Symmetric waveforms** (rise/fall ratio ≈ 0.94) — inconsistent with a sharp transient pulse; consistent with continuous driving during the PPO active phase.

5. **QP30 upward frequency chirp** (t=4.5, p<0.001) — expected for a driven resonance still building toward steady state; distinguishes this from a pure freely-ringing eigenmode.

---

## Alternative Theories

| Mechanism | Evidence for | Evidence against |
|---|---|---|
| **PPO-driven standing toroidal FLR** (favoured) | β ≈ 0 (PPO fixes period), amplitude growing (driven resonance), PPO phase-locked onset (QP30), no latitude polarization trend, N/S symmetry, period bands match KMAG even harmonics, post-dusk occurrence, PPO-period separations | QP30 upward chirp unexpected for pure eigenmode; too few oscillations to establish ideal standing pattern; QP120 mostly decaying |
| **Transient resonant Alfvén packet** (disfavoured) | β ≈ 0, linear polarization, short duration (3–10 cycles), spatial clustering at single L-shell | Amplitude growing (not decaying); PPO phase-locked onset (direct driver fingerprint); symmetric waveforms inconsistent with sharp transient; N/S symmetry inconsistent with single-pass packet |
| **Standing FLR eigenmode (freely ringing)** | Period bands match KMAG even harmonics exactly, phase-coherent stacking, spatial distribution at resonant L | β ≈ 0 (no T–L trend expected in free eigenmode), too few oscillations to build standing pattern, uniform PPO onset phase expected but not observed |
| **PPO-modulated compressional waves** | PPO period in separation times, compressional modes exist in outer magnetosphere | Band-pass transverse ratio 4.4× (transverse not compressional), not confined to equatorial plane |
| **Kelvin-Helmholtz instability at magnetopause** | KHI produces Alfvénic surface waves, expected post-dusk (velocity shear), can excite FLR | Periods do not match KHI growth rates; no correlation with solar wind speed; KHI waves are broadband |
| **Magnetotail reconnection-driven Alfvén waves** | Reconnection launches Alfvén wave fronts, consistent with sporadic events | Reconnection is impulsive and irregular; frequency of events matches PPO not substorm rate |
| **Mirror mode / drift-mirror instability** | Mirror modes common in Saturn's magnetosheath | Mirror modes are compressional; transverse ratio 4.4× inconsistent; periods not explained |
| **Waveguide modes (cavity resonance)** | Magnetospheric cavity can support eigenfrequencies | Volume-filling modes would appear at all LT equally; post-dusk preference inconsistent |
| **Ion cyclotron or drift-Alfvén waves** | Ion cyclotron waves at Saturn exist | Periods >> ion cyclotron period at L=18–22 Rs; wave properties inconsistent |

---

## Summary of Physical Picture

PPO-period magnetotail flapping (~10.7 h) launches Alfvénic perturbations each cycle. At L ~ 18–22 Rs, where the Alfvén travel time matches the QP period (ω_driver = ω_FLR for even harmonics), the wave is **resonantly amplified** — a toroidal, even-mode, linearly polarized standing oscillation that grows over 3–10 cycles during the PPO active phase (~4–6 h) and shuts off cleanly when the driver moves away.

This refines the original paper's interpretation: the eigenfrequency match (KMAG even harmonics), post-dusk occurrence, and PPO-modulated separations all remain robust. The morphology evidence now adds active excitation (amplitude growth), phase-locking (QP30 onset at ~190° PPO phase), and mode symmetry (toroidal even-mode) as confirmations.
