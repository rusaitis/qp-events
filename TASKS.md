# QP Events — Task List

## Phase 1: Strip Dead Code

- [x] **1. Delete dead top-level scripts (12 files)**
  - `pandasdemo.py` — pandas learning demo
  - `slider_demo.py` — matplotlib widget demo
  - `test_fieldcoords.py` — ad-hoc test
  - `test_read.py` — one-off FFT archive test
  - `test_wavelet.py` — scipy CWT demo
  - `mission_test_MAG.py` — debugging artifact, mostly commented out
  - `mission_testing.py` — 60-line array test
  - `mission_investigate_Moons.py` — one-off Titan analysis
  - `KMAG.py` — KMAG viz demo, superseded by `kmag_traces_plot.py`
  - `fft_sweep_automator.py` — batch subprocess wrapper
  - `fft_slopes.py` — one-off seaborn plot, no paper figure
  - `event_meta_searcher.py` — exploratory scatter plots

- [x] **2. Delete dead cassinilib modules (6 files)**
  - `FileCheck.py` — 8 lines, use `pathlib`
  - `Tic.py` — 12 lines, use `time.perf_counter()`
  - `Transformations.py` — 16 lines, duplicates ToMagneticCoords, has syntax error
  - `Equations.py` — 20 lines, just comments
  - `Fields.py` — 18 lines, incomplete mayavi experiment
  - `io.py` — 7 lines, custom arg parser

- [x] **3. Delete stale artifacts**
  - `setup.py` — references nonexistent file, unused
  - `KMAGhelper/KMAG` — x86_64 binary, won't run on ARM64
  - `KMAGhelper/KMAGtracer_experimental.py` — experimental, unused

---

## Phase 2: Restructure into Package

- [x] **4. Create `pyproject.toml` and `src/qp/` skeleton**
  - pixi-compatible build
  - Dependencies: numpy, scipy, matplotlib
  - `DATA_ROOT` env var for data path resolution

- [x] **5. Extract `src/qp/io/`**
  - `pds.py` ← DataPaths + SelectData (PDS file reading)
  - `products.py` ← load DataProducts/*.npy
  - `crossings.py` ← boundary_crossings.py

- [x] **6. Extract `src/qp/coords/`**
  - `mfa.py` ← ToMagneticCoords (consolidate 3 redundant functions into 1)
  - `transforms.py` ← Vector + linalg coordinate conversions
  - `ksm.py` — KSM coordinate definitions, offset dipole

- [x] **7. Extract `src/qp/signal/`**
  - `timeseries.py` ← NewSignal core (resampling, detrending, smoothing)
  - `fft.py` ← NewSignal FFT + Welch
  - `wavelet.py` ← NewSignal Morlet CWT
  - `power_ratio.py` ← Eq 4: r_i = P(b_i) / P(<B_T>_f)
  - `cross_correlation.py` ← Eq 6-7: polarization analysis

- [x] **8. Extract `src/qp/plotting/`**
  - `style.py` ← paper.mplstyle + color palettes
  - `timeseries.py` ← PlotTimeseries
  - `spectra.py` ← PlotFFT (split monolithic function)
  - `maps.py` ← dwell time heatmaps
  - `orbits.py` ← Plot.py 3D visualization
  - `eigenmodes.py` ← wavesolver/plot.py

- [x] **9. Extract `src/qp/events/`**
  - `detector.py` ← wave_finder from NewSignal + data_sweeper
  - `catalog.py` ← Event dataclass + metadata
  - `wave_packets.py` ← separation times, train properties

- [x] **10. Extract `src/qp/fieldline/`**
  - `kmag.py` ← KMAGhelper/KmagFunctions
  - `tracer.py` ← mission_trace tracing logic
  - `dwell_time.py` ← mission_trace + mission_trace_reader binning

- [x] **11. Extract `src/qp/wavesolver/`**
  - `simulation.py` ← sim.py (60-arg init → dataclass)
  - `eigensolver.py` ← shoot.py
  - `fieldline.py` ← fieldline.py
  - `density.py` ← model.py density models
  - `field.py` ← model.py field models
  - `pde.py` ← model.py wave equation
  - `configurations.py` ← cleaned presets

- [x] **12. Move Fortran to `fortran/`, recompile for ARM64**
  - Move KMAG2012.f, KMAG_pipe.f, KMAGtracer.f
  - Move kmag_params.txt, kmag_input.txt, discontinuities.txt
  - `gfortran -O2 -o KMAG KMAGtracer.f KMAG2012.f KMAG_pipe.f`

- [x] **13. Move reference data to `ref_data/`**
  - `Master_BS_MP_Crossing_List_04_16_incCMJcorr_revised.txt`
  - `metadata_latitude.txt`

---

## Phase 3: Modernize

- [x] **14. Fix critical bugs**
  - `NewSignal.py:155` — `power=power` undefined variable
  - `NewSignal.py:617` — `s.dt` → `self.dt`
  - `Event.py:58` — `TIME_MARGIN` undefined
  - `SelectData.py:73` — `dayEnd` undefined
  - Remove all `from pylab import *`

- [x] **15. Break circular dependencies**
  - `ToMagneticCoords → Plot → ToMagneticCoords` → `coords/mfa.py` depends only on numpy
  - `ToMagneticCoords → NewSignal` → plotting depends on coords, not vice versa

- [x] **16. Modernize patterns throughout**
  - `from pylab import *` → explicit imports
  - Mutable default args → `None` sentinel
  - `os.path` → `pathlib.Path`
  - `os.system()` → `subprocess.run(..., check=True)`
  - Hardcoded paths → `DATA_ROOT`
  - Type hints on public functions
  - Numpy-style docstrings on public API

---

## Phase 4: Reproduce Figures

### Batch 1 — High-trust (validate pipeline)

- [x] **17. Fig 1 — QP60 example timeseries**
  - Load day from `Cassini_MAG_KSM_36H.npy`, transform to MFA
  - Plot KSM (panel a) + MFA (panel b), mark wave trains
  - Referee: black background, ephemeris in caption

- [x] **18. Fig 4 — Power spectra + ratios (Jan 2, 2007)**
  - Single-day FFT/Welch, plot perturbations + power density + ratios
  - Referee: black background, ephemeris ticks, visible vertical lines

- [x] **19. Fig 5 — Median power ratios by LT quadrant (CENTRAL)**
  - Group 4278 days by LT, compute median + quartile r_i
  - 4-panel with QP30/QP60/QP120 labeled
  - Referee: white dashed line at 50-min

- [x] **20. Fig 9 — Wave train separation distribution**
  - Wavelet peak separations → histogram + PDF
  - Mark median at 10.73h

- [x] **21. Fig 10 — Polarization cross-correlation**
  - Two examples: circular (90 deg) + linear (180 deg)
  - Referee: full 360 deg, no phase-jump artifacts

### Batch 2 — Medium-trust (may need reanalysis)

- [x] **22. Fig 2 — Orbit + dwell time map**
  - Referee: ±3h LT bins, add 6±3h/18±3h panels, remove spurious line, fix green line

- [x] **23. Fig 3 — Dwell time along field lines (KMAG)**
  - Referee: black background

- [x] **24. Fig 6 — Eigenfrequencies overlay**
  - Run wavesolver noon + midnight, overlay QP rectangles with meaningful widths

- [ ] **25. Fig 7 — Event/dwell ratio vs latitude**
  - Referee: "magnetic latitude (KSM, offset dipole)" label

- [ ] **26. Fig 8 — QP60 occurrence heatmap**
  - Normalized event time / dwell time in LT-conjugate lat bins

- [ ] **27. SI Fig 1 — Closed flux tube dwell time**
  - N+S hemisphere panels

- [ ] **28. SI Fig 2 — Plasma sheet dwell time pre-equinox**
  - Weak B (<2 nT) filter, mid-2004 to mid-2009

- [ ] **29. Movie S1 — Animated power ratios**
  - Sweep through LT ranges with supplementary panels

---

## Phase 5: Finalize

- [ ] **30. Write tests**
  - `test_mfa_transform.py`
  - `test_power_ratio.py`
  - `test_fft.py`
  - `test_fieldline_tracing.py`
  - `test_eigensolver.py`

- [ ] **31. Delete old flat files from root**
  - Remove original scripts after they're absorbed into `src/qp/` + `scripts/`
  - Remove old `cassinilib/`, `wavesolver/`, `KMAGhelper/`

- [ ] **32. Final verification**
  - `pixi run pytest` — all tests pass
  - `pixi run python -c "import qp"` — clean import
  - All 13 figures reproduced and compared against originals
  - No circular imports, no star imports, no hardcoded paths
