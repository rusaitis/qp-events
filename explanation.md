# Wave Interpretation: Standing vs Transient Resonant FLR

## Background

The paper argues that QP30/QP60/QP120 waves are signatures of **field line resonances (FLR)** — Alfvén waves standing between northern and southern ionospheres. The three period bands match even harmonics of the FLR eigenfrequencies computed from the KMAG field model with Bagenal & Delamere density at L ~ 18–22 Rs.

The driver proposed is **periodic vertical flapping of the magnetotail** at the PPO period (~10.7 h), which excites Alfvénic perturbations at resonant field lines. The question is whether these are *true standing eigenmodes* or *transient resonant wave packets* — and the distinction matters for what the polarization, coherence, and period statistics tell us.

---

## Standing Eigenmode vs Transient Resonant FLR

A **standing FLR eigenmode** requires:
- Sufficient time for Alfvén waves to reflect many times between ionospheres and build up a standing pattern
- Narrow, L-dependent eigenfrequency selection (T ∝ L^α)
- High coherence across the wave train
- Circular polarization from superposition of inward + outward propagating helical Alfvén waves

A **transient resonant Alfvén packet** occurs when:
- The driver (PPO tail flapping) excites a short burst at a field line whose eigenfrequency matches the driving frequency
- The wave persists for a few bounce times (~4–6 h, 3–10 oscillations) before the driver moves on or the packet disperses
- The period is set by the *driver* frequency, not the local eigenfrequency — hence no T–L trend within a narrow band
- Polarization may be linear (single propagating pulse) or elliptical depending on how many reflections occur

| Observation | Standing eigenmode | Transient resonant packet |
|---|---|---|
| β ≈ 0 (no T–L trend within band) | Tension: each L should have a different eigenfrequency | Consistent: driver sets period, resonance selects L |
| Events cluster at L ≈ 18–22 Rs | Unexpected: all L should be eligible | Natural: only L-shell where eigenfreq ≈ driver freq is selected |
| Median duration 4–6 h (3–10 cycles) | Tension: too few bounces to build eigenmode | Expected: a few bounce times is typical for driven transient |
| Moderate coherence (quality ~ 0.27) | Below what a pure eigenmode produces | Expected: packet disperses before fully standing |
| Linear polarization (84%, Stokes) | Mixed: eigenmodes predict circular | Consistent: linear for a single-pass propagating pulse |
| PPO-modulated separation times (median ~ 10.7 h) | Consistent: PPO drives both | Directly expected: each PPO cycle triggers a new burst |
| Period bands match even FLR harmonics (KMAG) | Consistent | Consistent: resonance condition still met |
| Phase-coherent stack SNR 16–75 | Consistent: coherent signal exists | Consistent: repeatable driven oscillation |
| Highest occurrence post-dusk, mid-high lat | Predicted by PPO tail flapping geometry | Same |

## Phase 9–10 Morphology Evidence (new)

Computed from `events_qp_v4.parquet` (1636 events, 8 new morphology fields).

| Observation | Standing eigenmode | Transient resonant packet | **Measured** |
|---|---|---|---|
| Waveform symmetry (rise/fall ratio) | ≈ 1 (symmetric) | < 1 (sharp onset) | **median 0.94** — nearly symmetric |
| Harmonic content (P(2f)/P(f)) | Low (sine-like) | Low | **median 0.20** — notable harmonic content |
| Amplitude growth per cycle | Grow then plateau | Monotonic decay | **QP30/60 mostly growing** (+0.14, +0.28 dB/period) |
| Frequency drift (chirp) | ≈ 0 | Could drift if dispersive | **QP30 significant upward chirp** (t=4.5, p<0.001) |
| Polarization vs latitude | Poloidal: sign reversal | No trend | **No trend** (Spearman r<0.13, all p>0.1) |
| N/S hemisphere ellipticity | Even mode: identical | No difference | **Identical** (Mann-Whitney p>0.19) |
| PPO phase of onset | Uniform if freely ringing | Peaked if PPO-driven | **QP30 non-uniform** (R=0.10, p=0.009) |

### Key new findings:

1. **QP30 events are predominantly growing in amplitude** (12/20 top events growing, median +0.14 dB/period). **QP60 even more clearly growing** (16/20, median +0.28 dB/period). This is the signature of an **externally driven** resonance building up — consistent with PPO-driven FLR excitation. QP120 shows opposite behaviour (decaying, median -4.2 dB/period), possibly because these longer-period modes are less efficiently driven.

2. **No polarization trend with latitude** (all Spearman r < 0.13). This rules out poloidal odd-mode FLR (which predicts sign reversal between hemispheres). Consistent with **toroidal even-mode FLR** or travelling wave.

3. **QP30 PPO phase of onset is non-uniform** (Rayleigh p=0.009, preferred phase ~190°). This is the clearest evidence yet that **QP30 events are triggered at a specific PPO phase** — direct fingerprint of the PPO driver. QP60 and QP120 show no significant preference (p>0.27), possibly because their lower statistics dilute the signal.

4. **QP30 shows statistically significant upward frequency chirp** (positive drift). This is unexpected for a pure standing mode (which should have zero drift) and could indicate a mild dispersive effect as the wave packet builds up before the resonance is fully established.

5. **Harmonic ratio ~0.20** for all bands — moderate non-sinusoidal content. This is consistent with a **driven resonance with slight waveform distortion** (not a perfect sine), but not as extreme as a sawtooth.

6. **N/S hemisphere ellipticity identical** — confirms even-mode FLR symmetry (same waveform seen from both hemispheres).

### Updated interpretation:

The morphology evidence supports a **driven standing FLR** picture more than a transient dispersing packet:
- Amplitude *growing* during events → resonance is building (externally driven, not freely decaying)
- PPO phase-locked onset (QP30) → direct PPO driving
- No latitude trend in polarization → toroidal (not poloidal) even mode
- Symmetric envelopes → not a sharp transient pulse

The wave is likely a **resonance being continuously excited by the PPO driver** during its active phase (~4–6 h), then shutting off cleanly when the PPO flapping moves away. This is closer to a driven standing eigenmode than a freely propagating packet, but the distinction from "transient resonant" remains subtle since the observation window (3–10 cycles) is too short to fully distinguish.

---

**Favoured interpretation**: **PPO-driven standing toroidal FLR**, excited when the PPO-period tail flapping aligns with the resonant L-shell eigenfrequency. The wave **grows in amplitude** during the excitation phase, has **linear polarization** (toroidal mode), shows **no latitude trend**, and is triggered at a **preferred PPO phase**. The ~10.7h recurrence is directly set by the PPO driver period.

This does NOT invalidate the original paper's interpretation — it confirms and refines it. The eigenfrequency match (KMAG even harmonics), the post-dusk occurrence, and the PPO-modulated separations all remain robust.

---

---

## Alternative Theories

| Mechanism | Evidence for | Evidence against |
|---|---|---|
| **Transient driven FLR** (favoured) | β ≈ 0 (PPO fixes period), linear polarization, short duration (3–10 cycles), spatial clustering at single L-shell, phase-coherent stacks, PPO-period separations | Fewer bounces than ideal standing mode; does not fully explain why QP120 is rarer |
| **Standing FLR eigenmode** (original paper) | Period bands match KMAG even harmonics exactly, phase-coherent stacking shows real wave, spatial distribution peaks at resonant L | β ≈ 0 (no T–L trend), too few oscillations to establish standing mode, linear polarization inconsistent with circular prediction |
| **PPO-modulated compressional waves** | PPO period seen in separation times, compressional modes exist in outer magnetosphere | Band-pass transverse ratio >> 1 (waves are transverse, not compressional), QP waves not confined to equatorial plane |
| **Kelvin-Helmholtz instability at magnetopause** | KHI produces Alfvénic surface waves, expected post-dusk (velocity shear), can excite FLR | Periods do not match KHI growth rates at Saturn's magnetopause; no clear correlation with solar wind speed; KHI waves are broadband, not narrow-period |
| **Magnetotail reconnection-driven Alfvén waves** | Reconnection launches Alfvén wave fronts along field lines, consistent with sporadic events | Reconnection is impulsive and irregular, inconsistent with repeatable PPO-period recurrence; frequency of events matches PPO not substorm rate |
| **Mirror mode / drift-mirror instability** | Mirror modes are common in Saturn's magnetosheath; can excite field-line-aligned oscillations | Mirror modes are compressional, transverse ratio << 1 expected; our band-pass transverse ratio >> 1; mirror modes do not produce the observed 30/60/120 min periodicity |
| **Waveguide modes (cavity resonance)** | Magnetospheric cavity can support eigenfrequencies; global resonances at Saturn have been proposed | Cavity modes are volume-filling and would appear at all LT sectors equally; observed post-dusk preference is inconsistent; periods depend on magnetospheric size not consistent with three fixed bands |
| **Ion cyclotron or drift-Alfvén waves** | Ion cyclotron waves exist at Saturn, produce narrow-band oscillations | Periods >> ion cyclotron period at these distances; wave properties (transverse, long-period) inconsistent with ion cyclotron |

---

## Summary of Physical Picture

The most coherent interpretation is:

1. Saturn's magnetotail undergoes **periodic vertical flapping at the PPO period** (~10.7 h), driven by the rotating internal plasma asymmetry.
2. Each flapping cycle launches an Alfvénic perturbation into the inner magnetosphere.
3. At field lines where the Alfvén travel time matches the QP period (i.e., ω_driver = ω_FLR), the wave is **resonantly amplified**.
4. The resulting wave packet propagates 3–10 bounce times, appearing as a coherent QP oscillation in the Cassini MAG data.
5. The wave disperses before a true standing eigenmode is established, explaining the linear polarization and moderate coherence.
6. The specific L-shell selection (L ~ 18–22 Rs for all three bands) reflects where the KMAG even-harmonic eigenfrequencies match the three PPO sub-harmonics.

This picture is consistent with all Phase 1–8 observational findings.
