/* QP Event Review — vanilla JS, uPlot. */

"use strict";

const FIELD_COLORS = {
  par:   "#DC267F",
  perp1: "#FFB000",
  perp2: "#FE6100",
  tot:   "#648FFF",
};
// Same colors at low alpha for the per-sample dots in the field plot.
const FIELD_DOT_FILL = {
  par:   "rgba(220, 38, 127, 0.45)",
  perp1: "rgba(255, 176, 0, 0.45)",
  perp2: "rgba(254, 97, 0, 0.45)",
  tot:   "rgba(100, 143, 255, 0.45)",
};
const REGION_COLORS = {
  magnetosphere: "#12d5ae",
  magnetosheath: "#f29539",
  solar_wind:    "#f26b59",
  unknown:       "#555555",
};
const BAND_COLORS = { QP15: "#4ecdc4", QP30: "#80c0ff", QP60: "#ffb000", QP120: "#f06090" };
// Indexed by uPlot series index (0 = x). Used by cursor.points.fill — uPlot
// wraps series.stroke into a function internally, so we cannot read it back
// at runtime; we keep the literal hex strings here instead.
const WAVE_CURSOR_COLORS = [
  null, FIELD_COLORS.par, FIELD_COLORS.perp1, FIELD_COLORS.perp2, FIELD_COLORS.tot,
];

// Rolling-mean window per band (minutes ≈ 4× central period). Removing
// this trend makes the small QP perturbations visible against the
// dominant background field, especially in |B|.
const DETREND_WINDOW_MIN = { QP15: 60, QP30: 120, QP60: 240, QP120: 480 };
const DEFAULT_DETREND_MIN = 240;

// Each uPlot instance carries its own mutable plugin context at
// `u._qpCtx`. Plugins read from there on redraw. This lets us reuse
// uPlot instances across event navigation (just `setData`) without
// the events/synthetic/benchmark plots stomping on each other's state.
function _qpCtx(u) {
  return u._qpCtx || (u._qpCtx = {
    spans: [],
    peakEpoch: null,
    eventWindow: null,
    xRange: null,
    yRange: null,
    periods: [15, 30, 60, 120],
  });
}

const ALL_BANDS = ["QP15", "QP30", "QP60", "QP120"];
const ALL_REGIONS = ["magnetosphere", "magnetosheath", "solar_wind"];

const state = {
  allEvents: [],       // every event, sorted by current sort key (slim: id/uid/peak/band/region)
  events: [],          // filter-applied subset visible to the user
  timelineAll: [],     // unfiltered, for top strip
  timelineAllById: null, // Map<event_id, event> built once after fetch
  regionIntervals: [], // mission-wide MS/SH/SW intervals
  timelineCache: null, // pre-rendered timeline strip (canvas)
  detailCache: new Map(), // event_id → full /api/events/{id} payload (heavy stat fields)
  filteredIds: [],     // event_id list in current filter order
  pos: 0,              // index in filteredIds
  bandFilter:   new Set(ALL_BANDS),
  regionFilter: new Set(ALL_REGIONS),
  // Range filters: 4 numeric/date axes, each independently active.
  // fmin/fmax = full data range (set once after fetch). min/max = current
  // user selection. For dates the value space is unix epoch seconds.
  // `field` (optional) decouples the filter key from the data column it
  // pulls from — lets us have multiple filters on the same field (e.g.
  // generic "Amp" + explicit per-component sliders in the right column).
  rangeFilters: {
    peak_time:   { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "Date",   unit: "counts",         isDate: true,  step: null },
    b_perp1_amp: { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "Amp",    unit: "nT",             isDate: false, step: null },
    q_factor:    { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "Q",      unit: "quality factor", isDate: false, step: null },
    period_min:  { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "Period", unit: "min",            isDate: false, step: 1 },
    r_distance:  { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "R",      unit: "Rs",             isDate: false, step: null },
    local_time:  { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "LT",     unit: "h",              isDate: false, step: null },
    mag_lat:     { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "Mag lat", unit: "°",  isDate: false, step: null },
    l_shell:     { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "L",       unit: "shell", isDate: false, step: null },
    bperp1:      { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "B⊥₁",   unit: "nT", isDate: false, step: null,
                   field: "b_perp1_amp" },
    bperp2:      { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "B⊥₂",   unit: "nT", isDate: false, step: null,
                   field: "b_perp2_amp" },
    bpar:        { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "B∥",    unit: "nT", isDate: false, step: null,
                   field: "b_par_amp" },
    stokes_d:    { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "Stokes d", unit: "",  isDate: false, step: null,
                   // Stokes d has a hard theoretical domain — pin the slider
                   // to it instead of fitting to observed data so the linear
                   // (0) baseline and the L/R-circular extremes are visible.
                   fminOverride: -1, fmaxOverride: 1 },
  },
  sort: "peak_time",
  sortReverse: false,
  hoverEventId: null,  // event hovered on the timeline canvas
  hoverEvent: null,    // full hit object: {x, id, band, peak_time}
  zoom: true,          // focus on event window ± 1h (default on)
  detrend: false,      // subtract band-aware rolling mean
  bandpass: false,     // brick-wall FFT bandpass around detected period
  bandpassRf: {        // rf-shaped object so we can reuse buildSliderUI
    fmin: 1, fmax: 180,    // slider domain (period in minutes)
    min: null, max: null,  // current band; seeded from event.period_min ± 100%
    step: 1,
  },
  bandpassSlider: null,        // {el, syncFromState} returned by buildSliderUI
  bandpassMark:   null,        // vertical marker showing the event's P₀
  bandpassSeedEventId: null,   // id of event currently reflected in the band
  showSpan: true,      // draw event-window highlight
  showPeak: true,      // draw peak-time vertical line
  lastWf: null,        // cached waveform JSON for re-render on toggle
  lastSpec: null,      // cached spectrum JSON
  waveformPlot: null,
  spectrumPlot: null,
  synWavePlot: null,
  synSpecPlot: null,
  benchWavePlot: null,
  benchSpecPlot: null,
  benchData: null,
  benchSelected: null,
  inflight: null,
};

/* ----------------------------- DOM helpers ----------------------------- */

const $ = (sel) => document.querySelector(sel);

function setStatus(msg) { $("#status").textContent = msg; }

function toEpoch(iso) { return Date.parse(iso) / 1000; }

function fmt(v, digits) {
  return (v == null || !Number.isFinite(v)) ? "—" : v.toFixed(digits);
}

function qFactorClass(q) {
  if (q == null || !Number.isFinite(q)) return "q-unknown";
  if (q >= 4.0) return "q-high";
  if (q >= 2.5) return "q-mid";
  return "q-low";
}

// Short explanations surfaced via a "?" hint. Plain text — rendered
// through the native browser title-attribute tooltip.
const HELP_TEXT = {
  q_factor: "Spectral quality factor — peak frequency divided by FWHM. Higher Q means a sharper, more coherent oscillation.",
  bperp1:   "First transverse component (perpendicular to mean B) in mean-field-aligned (MFA) coordinates. Carries Alfvénic / shear wave power.",
  bperp2:   "Second transverse component (perpendicular to both B and B⊥₁) in MFA coordinates. Together with B⊥₁ describes the polarization plane.",
  bpar:     "Component parallel to the mean field. Compressional / fast-mode wave power lives here.",
  stokes_d: "Degree of circular polarization. +1 = right-handed circular, −1 = left-handed circular, 0 = linear. Computed from the cross-correlation of B⊥₁ and B⊥₂.",
  l_shell:  "Dipole L-shell parameter L = R / cos²(λ_mag). Equatorial radius of the field line through the spacecraft, in Rs. Derived from r_distance and mag_lat.",
  mva_par:  "MVA major-axis parallel fraction: (ê_max · b̂∥)². The minimum-variance principal axis of the bandpass-filtered field. Lower = more transverse. Detector requires ≤ 0.5.",
  sigma_pk: "Robust σ above the per-row CWT background. Each frequency row's noise model is the median + MAD × 1.4826 (Gaussian-equivalent σ) computed on out-of-band period rows only, then interpolated in log-period to the in-band rows so QP signal cannot inflate its own noise estimate. The threshold itself is set by Bonferroni FWER control over the effective number of independent time-frequency cells in the search volume (α = 0.01, n_σ ≈ 4.6 for a 36-h segment). Not a fixed 3σ or 5σ — derived from the search volume.",
  co_bands: "Other QP bands whose detection window overlaps this event's [date_from, date_to] inside the same 36-h MFA segment. Multi-harmonic events (e.g., simultaneous QP60 + QP120) flag the FLR's even-mode comb directly.",
};

// Gate-summary classifier: returns CSS class for a value compared to its
// threshold. "Marginal" = within 10 % of the gate. Direction encodes
// which side of the threshold "passes" — ">" means larger passes, "<"
// means smaller passes.
function gateClass(value, threshold, direction) {
  if (value == null || !Number.isFinite(value)) return "gate-unknown";
  const margin = 0.10 * Math.abs(threshold);
  if (direction === ">") {
    if (value < threshold) return "gate-fail";
    if (value < threshold + margin) return "gate-marginal";
    return "gate-pass";
  }
  // direction "<"
  if (value > threshold) return "gate-fail";
  if (value > threshold - margin) return "gate-marginal";
  return "gate-pass";
}

function gateChip(value, threshold, direction, digits = 2) {
  const cls = gateClass(value, threshold, direction);
  const v = (value == null || !Number.isFinite(value)) ? "—" : value.toFixed(digits);
  return `<span class="gate-chip ${cls}">${v}</span>`;
}

function helpHintHTML(text) {
  return ` <span class="help-hint" title="${escapeHtml(text)}">?</span>`;
}

function helpHintEl(text) {
  const el = document.createElement("span");
  el.className = "help-hint";
  el.title = text;
  el.textContent = "?";
  // Cell titles are <label>s — clicking the help icon would otherwise
  // toggle the cell's checkbox. Stop both click and mousedown so neither
  // the focus nor the checkbox change fires.
  el.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); });
  el.addEventListener("mousedown", (e) => { e.preventDefault(); e.stopPropagation(); });
  return el;
}

function statTile(label, val, unit, extraClass, help) {
  const u = unit ? `<em>${unit}</em>` : "";
  const cls = "stat-val" + (extraClass ? " " + extraClass : "");
  const hint = help ? helpHintHTML(help) : "";
  return `<div class="stat"><span class="stat-label">${label}${hint}</span>` +
         `<span class="${cls}">${val}${u}</span></div>`;
}

function fmtIsoCompact(iso) {
  // "2007-12-19T22:14:00" → "2007-12-19 22:14"
  if (!iso) return "—";
  return iso.replace("T", " ").slice(0, 16);
}

// Defensive HTML escape — region/band/uid are server-validated against
// tight regexes, but we still inject them into innerHTML, so escaping
// removes the implicit-trust assumption at near-zero perf cost.
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

const debounce = (fn, ms) => {
  let h;
  return (...a) => {
    clearTimeout(h);
    h = setTimeout(() => fn(...a), ms);
  };
};

function renderEventStats(summary, wf, detail, wavelet) {
  // Heavy stat fields live in the lazily-fetched `detail`; fall back to
  // `summary` if it hasn't arrived yet (rare — fetched in parallel with
  // the waveform). `wf`/`summary` always cover the band/region pills.
  // `wavelet` (also lazily fetched) carries σ-at-peak and the canonical
  // gate thresholds for the gate-summary chips.
  const s = detail || {};
  const q = s.q_factor;
  const qPill = `<span class="q-chip ${qFactorClass(q)}">${fmt(q, 2)}</span>`;
  const w = wavelet || {};
  const thr = w.thresholds || {};
  // Co-occurring sibling bands — string field on the parquet row.
  const coRaw = (detail && detail.co_bands) || summary.co_bands || "";
  const coBands = (coRaw && coRaw !== "" )
    ? String(coRaw).split(",").map(b => b.trim()).filter(Boolean)
    : [];
  const coBandsHtml = coBands.length
    ? coBands.map(b => `<span class="pill" data-band="${escapeHtml(b)}">${escapeHtml(b)}</span>`).join(" ")
    : '—';
  // Fall back to the canonical detector defaults so the gate chips
  // colour correctly even before the wavelet fetch has resolved.
  const Q_MIN     = thr.q_factor_min     ?? 3.0;
  const MVA_MAX   = thr.mva_par_frac_max ?? 0.5;
  const STOKES_MIN= thr.stokes_d_min     ?? 0.7;
  const FWER_THR  = w.n_sigma_threshold  ?? null;
  const SIGMA_PK  = w.sigma_at_peak      ?? null;
  const region = escapeHtml(wf?.region ?? summary.region ?? "unknown");
  const regionPill = `<span class="pill" data-region="${region}">${region}</span>`;
  const band = escapeHtml(wf?.band ?? summary.band ?? "?");
  const bandPill = `<span class="pill" data-band="${band}">${band}</span>`;
  const tFrom = fmtIsoCompact(wf?.date_from);
  const tPeak = fmtIsoCompact(wf?.peak_time);
  const tTo   = fmtIsoCompact(wf?.date_to);

  $("#event-stats").innerHTML = `
    <div class="stat-group narrow">
      <div class="stat-group-title">Location</div>
      <div class="stat-grid stack">
        ${statTile("R",        fmt(s.r_distance, 1), "R<sub>S</sub>")}
        ${statTile("Mag lat",  fmt(s.mag_lat, 1) + "°", "")}
        ${statTile("LT",       fmt(s.local_time, 1), "h")}
        <div class="stat"><span class="stat-label">Region</span>
          <span class="stat-val">${regionPill}</span></div>
      </div>
    </div>
    <div class="stat-group narrow">
      <div class="stat-group-title">Time</div>
      <div class="stat-grid stack">
        ${statTile("Start",    tFrom, "", "time")}
        ${statTile("Peak",     tPeak, "", "time")}
        ${statTile("End",      tTo,   "", "time")}
        ${statTile("Duration", fmt(s.duration_minutes, 0), "min")}
      </div>
    </div>
    <div class="stat-group wide">
      <div class="stat-group-title">Wave Period</div>
      <canvas id="wave-period-hist" class="wave-period-hist"></canvas>
    </div>
    <div class="stat-group">
      <div class="stat-group-title">Polarization</div>
      <div class="pol-section">
        <div class="pol-left">${polarizationTiles(s)}</div>
        <div class="pol-right">
          <div class="stat"><span class="stat-label">Q factor${helpHintHTML(HELP_TEXT.q_factor)}</span>
            <span class="stat-val">${qPill}</span></div>
          ${statTile("Stokes d", fmt(s.stokes_d, 2), "", null, HELP_TEXT.stokes_d)}
        </div>
      </div>
    </div>
    <div class="stat-group">
      <div class="stat-group-title">Gates</div>
      <div class="stat-grid stack">
        <div class="stat"><span class="stat-label">Q ≥ ${Q_MIN}${helpHintHTML(HELP_TEXT.q_factor)}</span>
          <span class="stat-val">${gateChip(q, Q_MIN, ">")}</span></div>
        <div class="stat"><span class="stat-label">MVA∥ ≤ ${MVA_MAX}${helpHintHTML(HELP_TEXT.mva_par)}</span>
          <span class="stat-val">${gateChip(s.mva_par_frac, MVA_MAX, "<")}</span></div>
        <div class="stat"><span class="stat-label">Stokes d ≥ ${STOKES_MIN}${helpHintHTML(HELP_TEXT.stokes_d)}</span>
          <span class="stat-val">${gateChip(s.stokes_d, STOKES_MIN, ">")}</span></div>
        <div class="stat"><span class="stat-label">σ-peak ≥ ${FWER_THR != null ? FWER_THR.toFixed(2) : "—"}${helpHintHTML(HELP_TEXT.sigma_pk)}</span>
          <span class="stat-val">${FWER_THR != null ? gateChip(SIGMA_PK, FWER_THR, ">") : '<span class="gate-chip gate-unknown">…</span>'}</span></div>
        <div class="stat"><span class="stat-label">Co-occurs with${helpHintHTML(HELP_TEXT.co_bands)}</span>
          <span class="stat-val">${coBandsHtml}</span></div>
      </div>
    </div>
  `;
  renderWavePeriodGraph(+s.period_min, band);
}

/* ---- Wave-group period histogram ---- */
// Log-spaced bin edges in minutes — period bands separate cleanly in log P,
// and this matches the paper's log-frequency axes (Figs 4-5).
const WAVE_HIST_MIN = 10;       // log10(10)  = 1.000
const WAVE_HIST_MAX = 180;      // log10(180) = 2.255
const WAVE_HIST_NBINS = 60;
const WAVE_BAND_RANGES = [      // (band, lo_min, hi_min, css-var)
  ["QP15",  10, 20,  "--b-qp15"],
  ["QP30",  20, 40,  "--b-qp30"],
  ["QP60",  40, 80,  "--b-qp60"],
  ["QP120", 80, 160, "--b-qp120"],
];
const WAVE_HIST_TICKS = [10, 20, 30, 60, 120, 180];

function logScale(v) { return Math.log10(v); }

function buildWavePeriodHistogram() {
  // Recomputed every render — cheap (≤1881 events, 60 bins) and avoids
  // stale results when band/region/range filters change. Uses the
  // filter-applied subset so the histogram tracks "where this event sits
  // in the distribution actually being navigated."
  const lo = logScale(WAVE_HIST_MIN), hi = logScale(WAVE_HIST_MAX);
  const counts = new Uint32Array(WAVE_HIST_NBINS);
  // Use the filter-applied subset directly — an explicit empty set must
  // yield an empty histogram (the user has deselected everything). Fall
  // back to allEvents only during the boot window where state.events
  // hasn't been populated yet.
  const source = state.events ?? state.allEvents;
  for (const e of source) {
    const p = +e.period_min;
    if (!Number.isFinite(p) || p <= 0) continue;
    const lp = logScale(p);
    if (lp < lo || lp > hi) continue;
    let i = Math.floor(((lp - lo) / (hi - lo)) * WAVE_HIST_NBINS);
    if (i < 0) i = 0; else if (i >= WAVE_HIST_NBINS) i = WAVE_HIST_NBINS - 1;
    counts[i]++;
  }
  let mx = 0;
  for (let i = 0; i < counts.length; i++) if (counts[i] > mx) mx = counts[i];
  return { counts, max: mx };
}

function renderWavePeriodGraph(periodMin, eventBand) {
  const canvas = $("#wave-period-hist");
  if (!canvas) return;
  const cssW = canvas.clientWidth || canvas.parentElement?.clientWidth || 320;
  const cssH = 178;
  const dpr = window.devicePixelRatio || 1;
  canvas.style.height = cssH + "px";
  canvas.width  = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  // Gutters: left for y-axis tick labels + rotated "counts" label, bottom
  // for x-axis ticks + "period" label.
  const padL = 30, padR = 6, padT = 14, padB = 28;
  const plotW = cssW - padL - padR;
  const plotH = cssH - padT - padB;
  const lo = logScale(WAVE_HIST_MIN), hi = logScale(WAVE_HIST_MAX);
  const xOf = (pmin) => padL + ((logScale(pmin) - lo) / (hi - lo)) * plotW;
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const AXIS_GREY = "rgba(180, 180, 180, 0.75)";
  const GRID_GREY = "rgba(255, 255, 255, 0.12)";

  // Band tints — subtle for non-active bands, brighter for the event's band.
  for (const [name, bLo, bHi, varName] of WAVE_BAND_RANGES) {
    const col = cssVar(varName) || "#888";
    const x0 = xOf(bLo), x1 = xOf(bHi);
    ctx.fillStyle = col + (name === eventBand ? "66" : "18");
    ctx.fillRect(x0, padT, x1 - x0, plotH);
    ctx.fillStyle = col + (name === eventBand ? "ee" : "9c");
    ctx.font = `${name === eventBand ? 700 : 500} 10px system-ui, sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(name, (x0 + x1) / 2, padT - 12);
  }

  // Histogram bars.
  const hist = buildWavePeriodHistogram();
  const mx = hist.max || 1;
  const binW = plotW / WAVE_HIST_NBINS;
  ctx.fillStyle = "rgba(220, 220, 220, 0.55)";
  for (let i = 0; i < WAVE_HIST_NBINS; i++) {
    const h = (hist.counts[i] / mx) * plotH;
    if (h <= 0) continue;
    ctx.fillRect(padL + i * binW + 0.5, padT + plotH - h, Math.max(1, binW - 1), h);
  }

  // Y-axis: baseline, top gridline at max, numeric labels.
  ctx.fillStyle = GRID_GREY;
  ctx.fillRect(padL, padT + plotH, plotW, 1);
  if (mx > 0) {
    ctx.fillRect(padL, padT, plotW, 1);
    ctx.fillStyle = AXIS_GREY;
    ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(String(mx), padL - 3, padT - 1);
    ctx.textBaseline = "alphabetic";
    ctx.fillText("0", padL - 3, padT + plotH + 1);
  }

  // X-axis ticks.
  ctx.fillStyle = AXIS_GREY;
  ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of WAVE_HIST_TICKS) {
    const x = xOf(t);
    ctx.fillStyle = GRID_GREY;
    ctx.fillRect(x, padT + plotH, 1, 3);
    ctx.fillStyle = AXIS_GREY;
    ctx.fillText(String(t), x, padT + plotH + 4);
  }

  // Axis titles — same uppercase muted styling as the section headers,
  // just smaller. Non-italic now so they match the rest of the chrome.
  ctx.fillStyle = "rgba(200, 200, 200, 0.85)";
  ctx.font = "500 11px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("period (min)", padL + plotW / 2, cssH - 3);
  ctx.save();
  ctx.translate(10, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("event counts", 0, 0);
  ctx.restore();

  // Vertical line at this event's period, with the numeric value sitting
  // mid-line (replaces the redundant Period tile in the wave-side column).
  if (Number.isFinite(periodMin) && periodMin > 0) {
    const x = xOf(Math.max(WAVE_HIST_MIN, Math.min(WAVE_HIST_MAX, periodMin)));
    ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, padT);
    ctx.lineTo(x + 0.5, padT + plotH);
    ctx.stroke();

    // Pill-style period label centered on the line, in the middle of the
    // plot vertically. Black stroke gives it contrast over the bars.
    const label = `${periodMin.toFixed(0)} min`;
    const midY = padT + plotH / 2;
    ctx.font = "700 13px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(15, 15, 15, 0.85)";
    ctx.strokeText(label, x, midY);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, x, midY);
  }
}

function polarizationTiles(s) {
  // Show each component's RMS amplitude with its share of the joint
  // (perp1² + perp2² + par²) power budget, and mark the dominant one.
  // Across the catalogue, ~98% of events are dominated by perp1 or perp2;
  // the rare par-dominant cases are interesting outliers worth flagging.
  const a1 = Number.isFinite(+s.b_perp1_amp) ? +s.b_perp1_amp : 0;
  const a2 = Number.isFinite(+s.b_perp2_amp) ? +s.b_perp2_amp : 0;
  const ap = Number.isFinite(+s.b_par_amp)   ? +s.b_par_amp   : 0;
  const amps = [a1, a2, ap];
  const tot  = a1*a1 + a2*a2 + ap*ap;
  let dom = -1;
  if (tot > 0) {
    dom = 0;
    for (let i = 1; i < 3; i++) if (amps[i] > amps[dom]) dom = i;
  }
  const COMPONENTS = ["perp1", "perp2", "par"];
  const tile = (i, label, val, help) => {
    const pct = tot > 0 ? Math.round(100 * amps[i] * amps[i] / tot) : 0;
    const bar = tot > 0
      ? `<div class="amp-bar" title="${pct}% of joint perturbation power">` +
        `<div class="amp-bar-fill" style="width:${pct}%"></div></div>`
      : "";
    const cls = `stat amp amp-${COMPONENTS[i]}` + (i === dom ? " amp-dominant" : "");
    const hint = help ? helpHintHTML(help) : "";
    return `<div class="${cls}"><span class="stat-label">${label}${hint}</span>` +
           `<span class="stat-val">${val}<em>nT</em></span>${bar}</div>`;
  };
  return [
    tile(0, "|B<sub>⊥1</sub>|", fmt(s.b_perp1_amp, 2), HELP_TEXT.bperp1),
    tile(1, "|B<sub>⊥2</sub>|", fmt(s.b_perp2_amp, 2), HELP_TEXT.bperp2),
    tile(2, "|B<sub>∥</sub>|",  fmt(s.b_par_amp,   2), HELP_TEXT.bpar),
  ].join("");
}

/* ----------------------- Rolling-mean detrend ------------------------- */

function rollingMean(arr, window) {
  // Centered moving average; ignores null / non-finite samples in the
  // accumulator. Returns an array of the same length, with edge windows
  // truncated to whatever data is available.
  const n = arr.length;
  const out = new Array(n);
  const half = Math.floor(window / 2);
  const cum = new Float64Array(n + 1);
  const cnt = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) {
    const v = arr[i];
    const ok = v !== null && Number.isFinite(v);
    cum[i + 1] = cum[i] + (ok ? v : 0);
    cnt[i + 1] = cnt[i] + (ok ? 1 : 0);
  }
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - half);
    const hi = Math.min(n, i + half + 1);
    const c = cnt[hi] - cnt[lo];
    out[i] = c > 0 ? (cum[hi] - cum[lo]) / c : null;
  }
  return out;
}

function subtract(a, b) {
  const n = a.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const av = a[i], bv = b[i];
    out[i] = (av === null || bv === null || !Number.isFinite(av) || !Number.isFinite(bv))
      ? null : av - bv;
  }
  return out;
}

function detrendComponents(wf) {
  const w = DETREND_WINDOW_MIN[wf.band] ?? DEFAULT_DETREND_MIN;
  return {
    b_par:   subtract(wf.b_par,   rollingMean(wf.b_par,   w)),
    b_perp1: subtract(wf.b_perp1, rollingMean(wf.b_perp1, w)),
    b_perp2: subtract(wf.b_perp2, rollingMean(wf.b_perp2, w)),
    b_tot:   subtract(wf.b_tot,   rollingMean(wf.b_tot,   w)),
    windowMin: w,
  };
}

/* --------------------------- FFT bandpass ---------------------------- */
/* In-place iterative Cooley-Tukey radix-2 FFT on (re, im); length must
   be a power of two. Used by bandpassSeries below — keeps the webapp
   dependency-free (no scipy round-trip per slider drag).               */

function fftRadix2(re, im, inverse) {
  const n = re.length;
  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const ang = (inverse ? 2 : -2) * Math.PI / size;
    const wRe0 = Math.cos(ang), wIm0 = Math.sin(ang);
    for (let i = 0; i < n; i += size) {
      let wr = 1, wi = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const tr = wr * re[b] - wi * im[b];
        const ti = wr * im[b] + wi * re[b];
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr;        im[a] += ti;
        const nr = wr * wRe0 - wi * wIm0;
        wi = wr * wIm0 + wi * wRe0;
        wr = nr;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

function bandpassSeries(values, dtSec, periodLoMin, periodHiMin) {
  // Brick-wall bandpass via DFT: keep frequencies inside
  // [1/(periodHi*60), 1/(periodLo*60)] Hz, zero everything else (including
  // the conjugate negative-frequency bins). Brick-wall is fine here — the
  // 12 h window is wide relative to the kept band, and the default zoom
  // (±1 h around peak) hides the segment edges where Gibbs ringing lives.
  const n = values.length;
  if (n < 4 || !(dtSec > 0)) return values.slice();
  if (!(periodLoMin > 0) || !(periodHiMin > periodLoMin)) return values.slice();

  // Valid-sample mask — restored in the output so the plot keeps its gaps.
  const valid = new Uint8Array(n);
  let nValid = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    if (v != null && Number.isFinite(v)) { valid[i] = 1; nValid++; }
  }
  if (nValid < 4) return values.slice();

  // Linear-interpolate gaps; constant-extend the leading/trailing tails.
  let firstIdx = 0; while (!valid[firstIdx]) firstIdx++;
  let lastIdx  = n - 1; while (!valid[lastIdx]) lastIdx--;
  const firstVal = +values[firstIdx], lastVal = +values[lastIdx];
  const work = new Float64Array(n);
  for (let i = 0; i < firstIdx; i++)     work[i] = firstVal;
  for (let i = lastIdx + 1; i < n; i++)  work[i] = lastVal;
  work[firstIdx] = firstVal;
  let prev = firstIdx;
  for (let i = firstIdx + 1; i <= lastIdx; i++) {
    if (valid[i]) {
      const v = +values[i];
      if (i > prev + 1) {
        const a = work[prev], stride = (v - a) / (i - prev);
        for (let k = prev + 1; k < i; k++) work[k] = a + stride * (k - prev);
      }
      work[i] = v;
      prev = i;
    }
  }

  // Mean removal — keeps DC at 0 (the bandpass would zero it anyway, but
  // explicit subtraction tightens spectral leakage at the FFT edges).
  let mean = 0;
  for (let i = 0; i < n; i++) mean += work[i];
  mean /= n;
  for (let i = 0; i < n; i++) work[i] -= mean;

  // Zero-pad to next power of two.
  let nfft = 1;
  while (nfft < n) nfft <<= 1;
  const re = new Float64Array(nfft);
  const im = new Float64Array(nfft);
  for (let i = 0; i < n; i++) re[i] = work[i];

  fftRadix2(re, im, false);

  const fLo = 1 / (periodHiMin * 60);
  const fHi = 1 / (periodLoMin * 60);
  const dfHz = 1 / (nfft * dtSec);
  const half = nfft >> 1;
  for (let k = 0; k <= half; k++) {
    const f = k * dfHz;
    if (f < fLo || f > fHi) {
      re[k] = 0; im[k] = 0;
      const m = nfft - k;
      if (m !== k && m < nfft) { re[m] = 0; im[m] = 0; }
    }
  }

  fftRadix2(re, im, true);

  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = valid[i] ? re[i] : null;
  return out;
}

function bandpassComponents(wf, periodLoMin, periodHiMin) {
  const xs = wf.epoch_s;
  let dt = 60;
  if (xs && xs.length >= 2) {
    const guess = xs[1] - xs[0];
    if (Number.isFinite(guess) && guess > 0) dt = guess;
  }
  return {
    b_par:   bandpassSeries(wf.b_par,   dt, periodLoMin, periodHiMin),
    b_perp1: bandpassSeries(wf.b_perp1, dt, periodLoMin, periodHiMin),
    b_perp2: bandpassSeries(wf.b_perp2, dt, periodLoMin, periodHiMin),
    b_tot:   bandpassSeries(wf.b_tot,   dt, periodLoMin, periodHiMin),
    bandLo: periodLoMin,
    bandHi: periodHiMin,
  };
}

/* ============================== Timeline =============================== */

// Layout constants (CSS pixels).
const TL_REGION_Y = 2,  TL_REGION_H = 7;
const TL_EVENT_Y  = 13, TL_EVENT_H  = 32;
const TL_HEIGHT   = 64;
const TL_T0 = toEpoch("2004-01-01T00:00:00");
const TL_T1 = toEpoch("2018-01-01T00:00:00");

function buildTimelineCache(cssW, dpr) {
  // Pre-render region strip + event ticks + year markers to an offscreen
  // canvas. Recomputed only when canvas size or DPR changes — not per
  // event navigation. Cuts ~5300 fillRect calls per nav down to a single
  // drawImage blit.
  const off = document.createElement("canvas");
  off.width = cssW * dpr;
  off.height = TL_HEIGHT * dpr;
  const c = off.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  const xOf = (t) => 4 + (cssW - 8) * (t - TL_T0) / (TL_T1 - TL_T0);

  // year tick lines through event area. Edge years (2004 / 2018) anchor
  // to "left" / "right" alignment so the labels don't clip past the
  // canvas edges; middle years stay centered on their tick.
  c.font = "10px ui-monospace, monospace";
  for (let y = 2004; y <= 2018; y++) {
    const x = xOf(toEpoch(`${y}-01-01T00:00:00`));
    c.fillStyle = "#333";
    c.fillRect(x, TL_EVENT_Y, 1, TL_EVENT_H);
    if (y % 2 === 0) {
      c.fillStyle = "#9b9b9b";
      c.textAlign = y === 2004 ? "left" : y === 2018 ? "right" : "center";
      c.fillText(String(y), x, TL_HEIGHT - 3);
    }
  }
  c.textAlign = "center";

  // region strip — full mission context (not filtered, so users can see
  // where Cassini was even when those regions are excluded from the
  // event view).
  for (const reg of state.regionIntervals) {
    const x0 = xOf(reg.epoch_start);
    const x1 = xOf(reg.epoch_end);
    if (x1 <= x0) continue;
    c.fillStyle = REGION_COLORS[reg.region] || REGION_COLORS.unknown;
    c.globalAlpha = 0.85;
    c.fillRect(x0, TL_REGION_Y, Math.max(1, x1 - x0), TL_REGION_H);
  }
  c.globalAlpha = 1;
  c.strokeStyle = "#2c2c2c";
  c.lineWidth = 1;
  c.strokeRect(4, TL_REGION_Y - 0.5, cssW - 8, TL_REGION_H + 1);

  // event ticks — only events that pass the current band+region filter.
  // Also remember each filtered event's x position for hit-testing on
  // hover/click.
  const bf = state.bandFilter, rf = state.regionFilter;
  const eventsXs = [];
  c.globalAlpha = 0.6;
  for (const ev of state.timelineAll) {
    if (!bf.has(ev.band) || !rf.has(ev.region)) continue;
    const x = xOf(toEpoch(ev.peak_time));
    c.fillStyle = BAND_COLORS[ev.band] || "#666";
    c.fillRect(x, TL_EVENT_Y, 1, TL_EVENT_H);
    eventsXs.push({ x, id: ev.event_id, band: ev.band, peak_time: ev.peak_time });
  }
  c.globalAlpha = 1;
  // Sort by x for fast nearest-x lookup; also expose an id→entry Map so
  // drawTimeline's hover overlay is O(1) instead of an Array.find scan.
  eventsXs.sort((a, b) => a.x - b.x);
  const eventsXsById = new Map(eventsXs.map((e) => [e.id, e]));
  return { canvas: off, cssW, dpr, xOf, eventsXs, eventsXsById };
}

function nearestTimelineEvent(xCss, tolPx = 12) {
  const tc = state.timelineCache;
  if (!tc || !tc.eventsXs.length) return null;
  const arr = tc.eventsXs;
  // Binary search for first x >= xCss
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid].x < xCss) lo = mid + 1;
    else hi = mid;
  }
  const candidates = [];
  if (lo > 0)            candidates.push(arr[lo - 1]);
  if (lo < arr.length)   candidates.push(arr[lo]);
  let best = null, bestD = Infinity;
  for (const c of candidates) {
    const d = Math.abs(c.x - xCss);
    if (d < bestD) { bestD = d; best = c; }
  }
  return (best && bestD <= tolPx) ? best : null;
}

function drawTimeline(currentId) {
  const cv = $("#timeline");
  const ctx = cv.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssW = cv.clientWidth || cv.parentElement.clientWidth;
  if (!cssW || state.timelineAll.length === 0) return;

  cv.width = cssW * dpr;
  cv.height = TL_HEIGHT * dpr;
  cv.style.height = TL_HEIGHT + "px";

  // Refresh cache only on size / DPR change or first call.
  const cache = state.timelineCache;
  if (!cache || cache.cssW !== cssW || cache.dpr !== dpr) {
    state.timelineCache = buildTimelineCache(cssW, dpr);
  }
  const tc = state.timelineCache;

  // Blit pre-rendered strip (1 GPU op) — no per-nav per-region work.
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.drawImage(tc.canvas, 0, 0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Hover highlight: emphasize the hovered event tick. The mousemove
  // handler stashes the full hit object on state.hoverEvent, but the
  // tick's x position lives on the cache (which rebuilds on resize/
  // filter), so we look it up on the cache by id.
  if (state.hoverEventId != null) {
    const hov = tc.eventsXsById.get(state.hoverEventId);
    if (hov) {
      ctx.fillStyle = BAND_COLORS[hov.band] || "#666";
      ctx.globalAlpha = 1;
      ctx.fillRect(hov.x - 1.5, TL_EVENT_Y - 2, 3, TL_EVENT_H + 4);
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.lineWidth = 1;
      ctx.strokeRect(hov.x - 1.5, TL_EVENT_Y - 2, 3, TL_EVENT_H + 4);
    }
  }

  // Current event marker spans both strips (cheap, ~5 ops).
  if (currentId != null) {
    const cur = state.timelineAllById?.get(currentId);
    if (cur) {
      const x = tc.xOf(toEpoch(cur.peak_time));
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, TL_REGION_Y);
      ctx.lineTo(x, TL_EVENT_Y + TL_EVENT_H);
      ctx.stroke();
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(x, TL_EVENT_Y + TL_EVENT_H / 2, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

// Debounced so a single drag fires drawTimeline once when the user lets
// go, instead of rebuilding the offscreen cache every pixel of resize.
window.addEventListener("resize", debounce(() => {
  drawTimeline(currentEventId());
}, 100));

function bindTimelineInteraction() {
  const cv = $("#timeline");
  cv.style.cursor = "crosshair";
  cv.addEventListener("mousemove", (e) => {
    const rect = cv.getBoundingClientRect();
    const xCss = e.clientX - rect.left;
    const hit = nearestTimelineEvent(xCss);
    const newId = hit?.id ?? null;
    if (newId !== state.hoverEventId) {
      state.hoverEventId = newId;
      state.hoverEvent = hit;
      cv.style.cursor = newId != null ? "pointer" : "crosshair";
      cv.title = hit ? `event ${hit.id} · ${hit.band} · ${hit.peak_time.replace("T", " ")}` : "";
      drawTimeline(currentEventId());
    }
  });
  cv.addEventListener("mouseleave", () => {
    if (state.hoverEventId != null) {
      state.hoverEventId = null;
      state.hoverEvent = null;
      cv.style.cursor = "crosshair";
      cv.title = "";
      drawTimeline(currentEventId());
    }
  });
  cv.addEventListener("click", (e) => {
    const rect = cv.getBoundingClientRect();
    const xCss = e.clientX - rect.left;
    const hit = nearestTimelineEvent(xCss);
    if (!hit) return;
    const pos = state.filteredIds.indexOf(hit.id);
    if (pos >= 0) loadEventAtPos(pos);
  });
}

/* ====================== uPlot factories / region bands ================== */

// Visual geometry of the region strip in the wave plot's top padding.
const REGION_STRIP_H_DPR = 8;   // height in CSS px (DPR-multiplied below)
const REGION_STRIP_GAP_DPR = 4; // gap between strip and data area, CSS px

const regionBandsPlugin = {
  hooks: {
    draw: [(u) => {
      const spans = _qpCtx(u).spans;
      if (!spans || !spans.length) return;
      const ctx = u.ctx;
      const dpr = devicePixelRatio || 1;
      const stripH = REGION_STRIP_H_DPR * dpr;
      const gap    = REGION_STRIP_GAP_DPR * dpr;
      const yTop   = u.bbox.top - gap - stripH;
      if (yTop < 0) return;
      ctx.save();
      for (const span of spans) {
        const x0 = u.valToPos(toEpoch(span.t0), "x", true);
        const x1 = u.valToPos(toEpoch(span.t1), "x", true);
        if (x1 <= x0) continue;
        ctx.fillStyle = REGION_COLORS[span.region] || REGION_COLORS.unknown;
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x0, yTop, Math.max(1, x1 - x0), stripH);
      }
      ctx.globalAlpha = 1;
      ctx.strokeStyle = "#2c2c2c";
      ctx.lineWidth = 1;
      ctx.strokeRect(u.bbox.left, yTop - 0.5, u.bbox.width, stripH + 1);
      ctx.restore();
    }],
  },
};

const peakLinePlugin = {
  hooks: {
    draw: [(u) => {
      const c = _qpCtx(u);
      if (!c.showPeak) return;
      const peakEpoch = c.peakEpoch;
      if (peakEpoch == null) return;
      const x = u.valToPos(peakEpoch, "x", true);
      u.ctx.save();
      u.ctx.strokeStyle = "#ffffff";
      u.ctx.setLineDash([4, 3]);
      u.ctx.lineWidth = 1;
      u.ctx.beginPath();
      u.ctx.moveTo(x, u.bbox.top);
      u.ctx.lineTo(x, u.bbox.top + u.bbox.height);
      u.ctx.stroke();
      u.ctx.restore();
    }],
  },
};

const eventWindowPlugin = {
  hooks: {
    draw: [(u) => {
      const c = _qpCtx(u);
      if (!c.showSpan || !c.eventWindow) return;
      const ew = c.eventWindow;
      const x0 = u.valToPos(ew.from, "x", true);
      const x1 = u.valToPos(ew.to,   "x", true);
      if (!Number.isFinite(x0) || !Number.isFinite(x1) || x1 <= x0) return;
      const ctx = u.ctx;
      ctx.save();
      ctx.fillStyle = "#FFD24A";
      ctx.globalAlpha = 0.12;
      ctx.fillRect(x0, u.bbox.top, x1 - x0, u.bbox.height);
      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = "#FFD24A";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(x0, u.bbox.top); ctx.lineTo(x0, u.bbox.top + u.bbox.height);
      ctx.moveTo(x1, u.bbox.top); ctx.lineTo(x1, u.bbox.top + u.bbox.height);
      ctx.stroke();
      ctx.restore();
    }],
  },
};

const zeroLinePlugin = {
  hooks: {
    draw: [(u) => {
      const y0 = u.valToPos(0, "y", true);
      if (!Number.isFinite(y0)) return;
      const top = u.bbox.top, bot = top + u.bbox.height;
      if (y0 < top || y0 > bot) return;
      const ctx = u.ctx;
      ctx.save();
      ctx.strokeStyle = "rgba(245, 245, 245, 0.55)";
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(u.bbox.left, y0);
      ctx.lineTo(u.bbox.left + u.bbox.width, y0);
      ctx.stroke();
      ctx.restore();
    }],
  },
};

const periodMarkerPlugin = {
  hooks: {
    draw: [(u) => {
      const periods = _qpCtx(u).periods;
      const ctx = u.ctx;
      const top = u.bbox.top, bot = top + u.bbox.height;
      ctx.save();
      // Dashed vertical line per QP period.
      ctx.strokeStyle = "rgba(245, 245, 245, 0.45)";
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 1.5;
      for (const p of periods) {
        const x = u.valToPos(p, "x", true);
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, bot);
        ctx.stroke();
      }
      // Labels — large bold, no pill; thin dark stroke for legibility.
      // Centered horizontally on the dashed line so the "30 min" reads as
      // a label *of* that line rather than the next one.
      ctx.setLineDash([]);
      const fontPx = 26;
      ctx.font = `700 ${fontPx}px ui-monospace, monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.lineJoin = "round";
      for (const p of periods) {
        const x = u.valToPos(p, "x", true);
        const label = `${p} min`;
        const ly = top + 8;
        ctx.lineWidth = 4;
        ctx.strokeStyle = "rgba(15, 15, 15, 0.85)";
        ctx.strokeText(label, x, ly);
        ctx.fillStyle = "#f5f5f5";
        ctx.fillText(label, x, ly);
      }
      // Dominant-component peak: solid line in the component's plot color,
      // drawn last so it sits on top of the QP reference lines.
      const dom = _qpCtx(u).dominantMark;
      if (dom && Number.isFinite(dom.period)) {
        const dx = u.valToPos(dom.period, "x", true);
        if (dx >= u.bbox.left && dx <= u.bbox.left + u.bbox.width) {
          ctx.strokeStyle = dom.color;
          ctx.lineWidth = 2;
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.moveTo(dx, top);
          ctx.lineTo(dx, bot);
          ctx.stroke();
          // Period readout near the bottom of the line — dark stroke for
          // legibility over the PSD curves.
          const label = `${dom.period.toFixed(0)} min`;
          ctx.font = `700 14px ui-monospace, monospace`;
          ctx.textAlign = "center";
          ctx.textBaseline = "bottom";
          const ly = bot - 8;
          ctx.lineWidth = 4;
          ctx.lineJoin = "round";
          ctx.strokeStyle = "rgba(15, 15, 15, 0.85)";
          ctx.strokeText(label, dx, ly);
          ctx.fillStyle = dom.color;
          ctx.fillText(label, dx, ly);
        }
      }
      ctx.restore();
    }],
  },
};

function yRangeForVisible(data, xRange) {
  // data = [xs, s1, s2, ...] — returns [yMin, yMax] across all series
  // restricted to xs values within xRange. Adds 5% headroom.
  const xs = data[0];
  let lo = 0, hi = xs.length;
  if (xRange) {
    while (lo < hi && xs[lo] < xRange[0]) lo++;
    while (hi > lo && xs[hi - 1] > xRange[1]) hi--;
  }
  let mn = +Infinity, mx = -Infinity;
  for (let s = 1; s < data.length; s++) {
    const arr = data[s];
    for (let i = lo; i < hi; i++) {
      const v = arr[i];
      if (v !== null && Number.isFinite(v)) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
  }
  if (!Number.isFinite(mn) || !Number.isFinite(mx)) return null;
  const pad = (mx - mn) * 0.05 || 0.5;
  return [mn - pad, mx + pad];
}

const PLOT_CANVAS_HEIGHT = 280;  // canvas height; envelope auto-sizes around it

function ensureWavePlot(target) {
  target = target || $("#plot-wave");
  if (target._uPlot) return target._uPlot;
  const opts = {
    width: target.clientWidth,
    height: PLOT_CANVAS_HEIGHT,
    title: "MFA components",
    cursor: {
      focus: { prox: 24 },
      // Solid disk, no border, in the curve's own color. uPlot wraps
      // series.stroke into a function internally, so we look up the literal
      // color from WAVE_CURSOR_COLORS rather than u.series[sIdx].stroke.
      points: {
        size: 10,
        fill: (_u, sIdx) => WAVE_CURSOR_COLORS[sIdx],
      },
    },
    focus: { alpha: 0.20 },
    legend: { live: true },
    padding: [REGION_STRIP_H_DPR + REGION_STRIP_GAP_DPR + 4, null, null, null],
    scales: {
      x: { range: (u, mn, mx) => _qpCtx(u).xRange ?? [mn, mx] },
      y: { range: (u, mn, mx) => _qpCtx(u).yRange ?? [mn, mx] },
    },
    series: [
      { label: "time" },
      { label: "B∥",   stroke: FIELD_COLORS.par,   width: 1, spanGaps: true,
        value: (u, v) => v == null ? "—" : `${v.toFixed(2)} nT`,
        points: { show: true, size: 3.5, fill: FIELD_DOT_FILL.par,   stroke: FIELD_DOT_FILL.par } },
      { label: "B⊥₁",  stroke: FIELD_COLORS.perp1, width: 1, spanGaps: true,
        value: (u, v) => v == null ? "—" : `${v.toFixed(2)} nT`,
        points: { show: true, size: 3.5, fill: FIELD_DOT_FILL.perp1, stroke: FIELD_DOT_FILL.perp1 } },
      { label: "B⊥₂",  stroke: FIELD_COLORS.perp2, width: 1, spanGaps: true,
        value: (u, v) => v == null ? "—" : `${v.toFixed(2)} nT`,
        points: { show: true, size: 3.5, fill: FIELD_DOT_FILL.perp2, stroke: FIELD_DOT_FILL.perp2 } },
      { label: "|B|",  stroke: FIELD_COLORS.tot,   width: 1, spanGaps: true,
        value: (u, v) => v == null ? "—" : `${v.toFixed(2)} nT`,
        points: { show: true, size: 3.5, fill: FIELD_DOT_FILL.tot,   stroke: FIELD_DOT_FILL.tot } },
    ],
    axes: [
      { stroke: "#9b9b9b",
        grid:  { stroke: "rgba(235,235,235,0.05)", width: 1 },
        ticks: { stroke: "rgba(235,235,235,0.10)", width: 1 } },
      { stroke: "#9b9b9b",
        label: "B [nT]",
        labelSize: 28,
        labelFont: "12px ui-monospace, monospace",
        grid:  { stroke: "rgba(235,235,235,0.05)", width: 1 },
        ticks: { stroke: "rgba(235,235,235,0.10)", width: 1 } },
    ],
    plugins: [regionBandsPlugin, eventWindowPlugin, zeroLinePlugin, peakLinePlugin],
  };
  const p = new uPlot(opts, [[0], [null], [null], [null], [null]], target);
  target._uPlot = p;
  return p;
}

function updateWavePlot(target, data, peakEpoch, spans, eventWindow, xRange, title) {
  const p = ensureWavePlot(target);
  const ctx = _qpCtx(p);
  ctx.peakEpoch = peakEpoch;
  ctx.spans = spans || [];
  ctx.eventWindow = eventWindow;
  ctx.xRange = xRange;
  ctx.yRange = xRange ? yRangeForVisible(data, xRange) : null;
  ctx.showSpan = state.showSpan;
  ctx.showPeak = state.showPeak;
  p.setData(data);
  const titleEl = p.root.querySelector(".u-title");
  if (titleEl) titleEl.textContent = title || "MFA components";
  return p;
}

function ensureSpectrumPlot(target) {
  target = target || $("#plot-spec");
  if (target._uPlot) return target._uPlot;
  const opts = {
    width: target.clientWidth,
    height: PLOT_CANVAS_HEIGHT,
    title: "Welch PSD",
    cursor: {
      focus: { prox: 24 },
      points: { size: 10 },
    },
    focus: { alpha: 0.20 },
    legend: { live: true },
    scales: {
      // Period range [10 min, 240 min]:
      //   - left  = REJECT_BAND_HF cutoff (Khurana 2020, qp.events.bands):
      //     below 10 min the 1-min cadence is < 5× Nyquist and the spectrum
      //     is dominated by quantization / aliasing, not real wave power.
      //   - right = 4 h: above this the Welch 12-h window holds <3 cycles,
      //     so estimates are unreliable.
      // time:false — the x axis is period in minutes, not Unix epoch, so
      // disable uPlot's default time-mode formatting (which would otherwise
      // label this column "Time" and render values as 1970-01-01 ticks).
      x: { time: false, distr: 3, log: 10, range: () => [10, 240] },
      y: { distr: 3, log: 10 },
    },
    series: [
      { label: "period",
        value: (u, v) => v == null ? "—" : `${v.toFixed(1)} min` },
      { label: "PSD∥",  stroke: FIELD_COLORS.par,   width: 1, points: { size: 3 },
        value: (u, v) => v == null ? "—" : `${v.toExponential(2)} nT²/Hz` },
      { label: "PSD⊥₁", stroke: FIELD_COLORS.perp1, width: 1, points: { size: 3 },
        value: (u, v) => v == null ? "—" : `${v.toExponential(2)} nT²/Hz` },
      { label: "PSD⊥₂", stroke: FIELD_COLORS.perp2, width: 1, points: { size: 3 },
        value: (u, v) => v == null ? "—" : `${v.toExponential(2)} nT²/Hz` },
    ],
    axes: [
      {
        stroke: "#cfcfcf",
        label: "period [min]",
        labelSize: 22,
        size: 56,
        font: "12px ui-monospace, monospace",
        labelFont: "12px ui-monospace, monospace",
        grid:  { stroke: "rgba(235,235,235,0.05)", width: 1 },
        ticks: { stroke: "rgba(235,235,235,0.18)", width: 1, size: 5 },
        // Force a 1-2-5 sequence per decade — uPlot's log distr otherwise
        // filters out sub-decade ticks at small canvas widths, leaving
        // only "10" and "100".
        splits: (u, axIdx, mn, mx) =>
          [10, 15, 20, 30, 50, 70, 100, 150, 200].filter((v) => v >= mn && v <= mx),
        filter: (u, splits) => splits,  // bypass auto-suppression
        values: (u, splits) => splits.map((v) =>
          v == null ? "" : v.toFixed(0)
        ),
        space: 20,
      },
      {
        stroke: "#cfcfcf",
        label: "PSD [nT²/Hz]",
        labelSize: 28,
        size: 60,
        font: "12px ui-monospace, monospace",
        labelFont: "12px ui-monospace, monospace",
        grid:  { stroke: "rgba(235,235,235,0.05)", width: 1 },
        ticks: { stroke: "rgba(235,235,235,0.18)", width: 1, size: 5 },
        values: (u, splits) => splits.map((v) => v == null ? "" : v.toExponential(0)),
      },
    ],
    plugins: [periodMarkerPlugin],
  };
  const p = new uPlot(opts, [[1], [1], [1], [1]], target);
  target._uPlot = p;
  return p;
}

function updateSpectrumPlot(target, period_min, psd_par, psd_p1, psd_p2, periods, detail) {
  const p = ensureSpectrumPlot(target);
  const ctx = _qpCtx(p);
  ctx.periods = periods || [30, 60, 120];
  // Mark the event's detected period with a vertical line, colored by the
  // dominant component (same ranking as the Polarization tiles: max of the
  // time-domain RMS amplitudes). The position is the canonical
  // `period_min` from the detector, not a per-component PSD peak.
  ctx.dominantMark = null;
  if (detail && Number.isFinite(+detail.period_min) && +detail.period_min > 0) {
    const cands = [
      [+detail.b_perp1_amp || 0, "perp1", FIELD_COLORS.perp1],
      [+detail.b_perp2_amp || 0, "perp2", FIELD_COLORS.perp2],
      [+detail.b_par_amp   || 0, "par",   FIELD_COLORS.par],
    ];
    let best = cands[0];
    for (let i = 1; i < cands.length; i++) if (cands[i][0] > best[0]) best = cands[i];
    ctx.dominantMark = {
      period: +detail.period_min,
      color: best[2],
      label: best[1],
    };
  }
  p.setData([period_min, psd_par, psd_p1, psd_p2]);
  return p;
}

/* =============================== Events =============================== */

window.addEventListener("error", (e) => {
  setStatus(`JS error: ${e.message}`);
  console.error("[QP] uncaught:", e.error || e);
});
window.addEventListener("unhandledrejection", (e) => {
  setStatus(`promise rejected: ${e.reason?.message ?? e.reason}`);
  console.error("[QP] unhandled rejection:", e.reason);
});

async function fetchEvents() {
  setStatus("loading events…");
  const params = new URLSearchParams();
  if (state.sort) params.set("sort", state.sort);
  // Parallel-fan-out the three independent boot fetches: previously the
  // events response had to land before timeline/regions even started.
  const needTimeline = !state.timelineAll.length;
  const reqs = [fetch("/api/events?" + params.toString())];
  if (needTimeline) {
    reqs.push(fetch("/api/timeline"), fetch("/api/regions"));
  }
  const [evR, tR, rR] = await Promise.all(reqs);
  state.allEvents = await evR.json();
  if (needTimeline) {
    state.timelineAll = await tR.json();
    state.regionIntervals = await rR.json();
    // O(1) id→event lookup for drawTimeline's white "current" indicator.
    state.timelineAllById = new Map(
      state.timelineAll.map((e) => [e.event_id, e]),
    );
  }
  // Establish each range filter's full bounds — this is the slider's
  // "no narrowing" default position. Resort doesn't change bounds, so
  // we only need to do this once per allEvents fetch.
  computeRangeBounds();
  reflectFilterButton();
  setStatus("");
  applyFilter({ keepPos: false });
}

function rfValue(e, key, rf) {
  // Pull the data value for filter `key`. `rf.field` lets a filter map
  // onto a different data column (e.g. the bperp1/bperp2/bpar filters
  // share field names with b_perp1_amp/b_perp2_amp/b_par_amp).
  if (rf.isDate) return toEpoch(e.peak_time);
  return e[rf.field || key];
}

function computeRangeBounds() {
  for (const [key, rf] of Object.entries(state.rangeFilters)) {
    let mn = +Infinity, mx = -Infinity;
    for (const e of state.allEvents) {
      const v = rfValue(e, key, rf);
      if (v == null || !Number.isFinite(v)) continue;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    if (Number.isFinite(mn) && Number.isFinite(mx)) {
      // Filters with hard theoretical bounds (e.g. Stokes d ∈ [-1, +1])
      // pin the slider domain regardless of where the observed data
      // happens to sit — so the user can see "all events cluster at the
      // top of the full polarization scale".
      rf.fmin = rf.fminOverride ?? mn;
      rf.fmax = rf.fmaxOverride ?? mx;
      rf.min = rf.fmin;
      rf.max = rf.fmax;
    }
  }
}

function applyFilter({ keepPos = true } = {}) {
  const bf = state.bandFilter, rf = state.regionFilter;
  const ranges = state.rangeFilters;
  // Pre-extract just the active range filters so the inner loop avoids
  // re-iterating all four every event.
  const activeRanges = [];
  for (const key in ranges) {
    if (ranges[key].active) activeRanges.push([key, ranges[key]]);
  }
  state.events = state.allEvents.filter((e) => {
    if (!bf.has(e.band) || !rf.has(e.region)) return false;
    for (const [key, r] of activeRanges) {
      const v = rfValue(e, key, r);
      if (v == null || !Number.isFinite(v)) return false;
      if (v < r.min || v > r.max) return false;
    }
    return true;
  });
  state.filteredIds = state.events.map((e) => e.event_id);
  state.timelineCache = null;  // events on timeline depend on current filter
  $("#event-position-total").textContent = state.events.length;
  if (!state.filteredIds.length) {
    $("#event-position-num").textContent = "—";
    $("#event-uid").textContent = "";
    $("#event-stats").innerHTML =
      '<div class="no-events">No events match the current selection. ' +
      'Re-enable a band/region or clear the advanced filter.</div>';
    if (state.waveformPlot) state.waveformPlot.setData([[0], [null], [null], [null], [null]]);
    if (state.spectrumPlot) state.spectrumPlot.setData([[1], [1], [1], [1]]);
    state.lastWf = null;
    state.lastSpec = null;
    drawTimeline(null);
    return;
  }
  state.pos = keepPos ? Math.min(state.pos, state.filteredIds.length - 1) : 0;
  loadEventAtPos(state.pos);
}

function currentEventId() {
  return state.filteredIds[state.pos];
}

/** Jump to an event by its YYMMDDHHMMX UID. If the event is filtered
 *  out, clear band+region filters so it becomes visible. Returns true
 *  on success. Reflects pill state in the UI on filter reset. */
function gotoEventByUid(uid) {
  const target = (uid || "").trim().toUpperCase();
  if (!target) return false;
  const ev = state.allEvents.find((e) => (e.event_uid || "").toUpperCase() === target);
  if (!ev) return false;
  let pos = state.events.indexOf(ev);
  if (pos < 0) {
    // Not in current filter — reset bands, regions, and range filters,
    // then reapply (no narrowing left).
    state.bandFilter   = new Set(ALL_BANDS);
    state.regionFilter = new Set(ALL_REGIONS);
    reflectFilterPills();
    clearAllRangeFilters();
    state.events = state.allEvents.slice();
    state.filteredIds = state.events.map((e) => e.event_id);
    state.timelineCache = null;
    pos = state.events.indexOf(ev);
  }
  if (pos < 0) return false;
  loadEventAtPos(pos);
  return true;
}

async function loadEventAtPos(pos) {
  if (!state.filteredIds.length) { setStatus("no events"); return; }
  state.pos = (pos + state.filteredIds.length) % state.filteredIds.length;
  const id = state.filteredIds[state.pos];
  setStatus(`loading event ${id}…`);
  const ctrl = new AbortController();
  if (state.inflight) state.inflight.abort();
  state.inflight = ctrl;
  // Detail (heavy stat fields) is cached client-side after first fetch;
  // serve from cache when available, otherwise fetch in parallel with wf+spec.
  const cachedDetail = state.detailCache.get(id);
  try {
    const requests = [
      fetch(`/api/events/${id}/waveform`, { signal: ctrl.signal }),
      fetch(`/api/events/${id}/spectrum`, { signal: ctrl.signal }),
      fetch(`/api/events/${id}/wavelet`,  { signal: ctrl.signal }),
    ];
    if (!cachedDetail) {
      requests.push(fetch(`/api/events/${id}`, { signal: ctrl.signal }));
    }
    const responses = await Promise.all(requests);
    const [wfR, specR, wvltR, detailR] = responses;
    setStatus(`fetched event ${id}, parsing…`);
    if (!wfR.ok) throw new Error(`waveform HTTP ${wfR.status}`);
    if (!specR.ok) throw new Error(`spectrum HTTP ${specR.status}`);
    // Wavelet gate JSON is a nice-to-have — a failure shouldn't block
    // the wf/spec render. Fall back to null so the gate chips show "—".
    const wf = await wfR.json();
    const spec = await specR.json();
    const wavelet = wvltR.ok ? await wvltR.json() : null;
    let detail = cachedDetail;
    if (detailR) {
      if (!detailR.ok) throw new Error(`detail HTTP ${detailR.status}`);
      detail = await detailR.json();
      state.detailCache.set(id, detail);
    }
    setStatus(`rendering event ${id}…`);
    renderEvent(wf, spec, detail, wavelet);
    drawTimeline(id);
    setStatus(`event ${id}`);
  } catch (e) {
    if (e.name !== "AbortError") {
      setStatus(`error loading ${id}: ${e.message}`);
      console.error("[QP]", e);
    }
  }
}

function renderEvent(wf, spec, detail, wavelet) {
  const summary = state.events[state.pos] || {};
  const total = state.events.length;
  $("#event-position-num").textContent = state.pos + 1;
  $("#event-position-total").textContent = total;
  $("#event-uid").textContent = summary.event_uid || "";
  renderEventStats(summary, wf, detail, wavelet);

  state.lastWf = wf;
  state.lastSpec = spec;

  maybeReseedBandpass();

  const xs = wf.epoch_s;
  let par = wf.b_par, p1 = wf.b_perp1, p2 = wf.b_perp2, tot = wf.b_tot;
  let title = "MFA components";
  if (state.detrend) {
    const dt = detrendComponents(wf);
    par = dt.b_par; p1 = dt.b_perp1; p2 = dt.b_perp2; tot = dt.b_tot;
    title = `MFA components, detrended (${dt.windowMin}-min rolling mean)`;
  }
  if (state.bandpass) {
    const rf = state.bandpassRf;
    const bp = bandpassComponents(
      { ...wf, b_par: par, b_perp1: p1, b_perp2: p2, b_tot: tot },
      rf.min, rf.max,
    );
    par = bp.b_par; p1 = bp.b_perp1; p2 = bp.b_perp2; tot = bp.b_tot;
    const lo = rf.min.toFixed(0), hi = rf.max.toFixed(0);
    title = state.detrend
      ? `MFA components, detrended + bandpass [${lo}–${hi} min]`
      : `MFA components, bandpass [${lo}–${hi} min]`;
  }
  const data = [xs, par, p1, p2, tot];
  const f = toEpoch(wf.date_from);
  const t = toEpoch(wf.date_to);
  const xRange = state.zoom ? [f - 3600, t + 3600] : null;
  state.waveformPlot = updateWavePlot(
    $("#plot-wave"), data, toEpoch(wf.peak_time), wf.region_spans,
    { from: f, to: t }, xRange, title,
  );
  state.spectrumPlot = updateSpectrumPlot(
    $("#plot-spec"),
    spec.period_min, spec.psd_par, spec.psd_perp1, spec.psd_perp2,
    spec.qp_periods_min,
    detail,
  );
  updateWaveletPanel(summary.event_id);
}

function updateWaveletPanel(eventId) {
  // Server-rendered CWT scalogram + σ-mask. Setting `src` triggers the
  // browser to fetch; HTTP cache + endpoint Cache-Control gives instant
  // navigation on revisit. The wrapping div shows a "loading" tint until
  // onload fires.
  const wrap = $("#plot-wavelet");
  const img  = $("#wavelet-img");
  if (!wrap || !img || eventId == null) return;
  wrap.classList.add("loading");
  img.onload = () => wrap.classList.remove("loading");
  img.onerror = () => wrap.classList.remove("loading");
  img.src = `/api/events/${eventId}/wavelet.png`;
}

/* =============================== Synthetic ============================== */

const SYN_PRESETS = {
  low:  { amp: 0.8, period_min: 60, decay_h: 4, noise: 0.6 },
  med:  { amp: 2.0, period_min: 60, decay_h: 4, noise: 0.4 },
  high: { amp: 4.0, period_min: 60, decay_h: 4, noise: 0.3 },
};

let synAbort = null;

async function loadSynthetic() {
  const params = new URLSearchParams({
    band: $("#syn-band").value,
    amp: $("#syn-amp").value,
    period_min: $("#syn-period").value,
    decay_h: $("#syn-decay").value,
    noise: $("#syn-noise").value,
    seed: $("#syn-seed").value,
  });
  setStatus("generating…");
  // Cancel any in-flight generate so a stale response can't overwrite the latest plot.
  if (synAbort) synAbort.abort();
  synAbort = new AbortController();
  try {
    const r = await fetch(
      "/api/synthetic/generate?" + params.toString(),
      { signal: synAbort.signal },
    );
    const g = await r.json();
    setStatus("");

    const xs = g.epoch_s;
    const data = [xs, g.b_par, g.b_perp1, g.b_perp2, g.b_tot];
    state.synWavePlot = updateWavePlot(
      $("#plot-syn-wave"), data, null, [], null, null,
      "Synthetic MFA components",
    );
    state.synSpecPlot = updateSpectrumPlot(
      $("#plot-syn-spec"),
      g.spectrum.period_min,
      g.spectrum.psd_par, g.spectrum.psd_perp1, g.spectrum.psd_perp2,
      g.spectrum.qp_periods_min,
    );
  } catch (e) {
    if (e.name !== "AbortError") {
      setStatus(`error generating: ${e.message}`);
      console.error("[QP]", e);
    }
  }
}

const debouncedSyn = debounce(loadSynthetic, 250);

let benchAbort = null;

async function loadBenchmark() {
  const preset = $("#bench-preset").value;
  setStatus("running benchmark…");
  if (benchAbort) benchAbort.abort();
  benchAbort = new AbortController();
  let data;
  try {
    const r = await fetch(
      "/api/synthetic/benchmark?preset=" + preset,
      { signal: benchAbort.signal },
    );
    data = await r.json();
  } catch (e) {
    if (e.name === "AbortError") return;
    setStatus(`error running benchmark: ${e.message}`);
    console.error("[QP]", e);
    return;
  }
  state.benchData = data;
  setStatus("");

  const grid = $("#bench-grid");
  grid.innerHTML = "";
  const summary = data.summary;
  const summaryParts = Object.entries(summary).map(
    ([b, s]) => `${b}: ${s.tp}/${s.n} (${(s.recall * 100).toFixed(0)}%)`,
  );
  $("#bench-summary").textContent =
    `noise=${data.noise_sigma_nT} nT · ` + summaryParts.join("  ·  ");

  for (const row of data.rows) {
    const card = document.createElement("div");
    card.className = "bench-card " + (row.detected ? "tp" : "miss");
    card.innerHTML =
      `<div class="row"><span>${row.band} · amp=${row.amplitude_nT.toFixed(1)} · seed=${row.seed}</span>` +
      `<span class="bench-tag ${row.detected ? "tp" : "miss"}">${row.detected ? "TP" : "miss"}</span></div>` +
      (row.detected_period_min
        ? `<div class="row"><span>detected period</span><span>${row.detected_period_min.toFixed(1)} min</span></div>`
        : `<div class="row"><span>—</span><span></span></div>`);
    card.dataset.band = row.band;
    card.dataset.amp = row.amplitude_nT;
    card.dataset.seed = row.seed;
    card.addEventListener("click", () => selectBenchEvent(card, row, preset));
    grid.appendChild(card);
  }
  if (data.rows.length) {
    grid.firstElementChild.click();
  }
}

async function selectBenchEvent(card, row, preset) {
  document.querySelectorAll(".bench-card.selected").forEach(
    (c) => c.classList.remove("selected"),
  );
  card.classList.add("selected");
  state.benchSelected = row;
  const params = new URLSearchParams({
    preset, band: row.band,
    amp: row.amplitude_nT, seed: row.seed,
  });
  const r = await fetch("/api/synthetic/benchmark/event?" + params.toString());
  const g = await r.json();
  const xs = g.epoch_s;
  const data = [xs, g.b_par, g.b_perp1, g.b_perp2, g.b_tot];
  state.benchWavePlot = updateWavePlot(
    $("#plot-bench-wave"), data, null, [], null, null,
    "Benchmark MFA components",
  );
  state.benchSpecPlot = updateSpectrumPlot(
    $("#plot-bench-spec"),
    g.spectrum.period_min,
    g.spectrum.psd_par, g.spectrum.psd_perp1, g.spectrum.psd_perp2,
    g.spectrum.qp_periods_min,
  );
}

/* ============================== Tabs / keys ============================== */

function showTab(name) {
  $("#tab-events").classList.toggle("hidden", name !== "events");
  $("#tab-synthetic").classList.toggle("hidden", name !== "synthetic");
  document.querySelectorAll(".tab").forEach(
    (t) => t.classList.toggle("active", t.dataset.tab === name),
  );
  if (name === "synthetic" && !$("#plot-syn-wave")._uPlot) {
    loadSynthetic();
  }
}

function showSubtab(name) {
  $("#sub-generator").classList.toggle("hidden", name !== "generator");
  $("#sub-benchmark").classList.toggle("hidden", name !== "benchmark");
  document.querySelectorAll(".subtab").forEach(
    (t) => t.classList.toggle("active", t.dataset.subtab === name),
  );
  if (name === "benchmark" && !state.benchData) loadBenchmark();
}

// Hoisted by bindUI(); used by bindKeys to avoid two DOM lookups + two
// classList reads on every keystroke.
let tabEventsEl = null;

function bindKeys() {
  document.addEventListener("keydown", (e) => {
    if (
      e.target.tagName === "INPUT" || e.target.tagName === "SELECT"
      || e.target.tagName === "TEXTAREA"
      || e.target.isContentEditable
    ) return;
    const eventsTabActive = !tabEventsEl.classList.contains("hidden");
    if (eventsTabActive) {
      const step = e.shiftKey ? 10 : 1;
      if (e.key === "ArrowLeft")  { e.preventDefault(); loadEventAtPos(state.pos - step); }
      if (e.key === "ArrowRight") { e.preventDefault(); loadEventAtPos(state.pos + step); }
      if (e.key === "1") { toggleBand("QP15"); }
      if (e.key === "2") { toggleBand("QP30"); }
      if (e.key === "3") { toggleBand("QP60"); }
      if (e.key === "4") { toggleBand("QP120"); }
      if (e.key === "m") { toggleRegion("magnetosphere"); }
      if (e.key === "s") { toggleRegion("magnetosheath"); }
      if (e.key === "w") { toggleRegion("solar_wind"); }
      if (e.key === "g") { e.preventDefault(); $("#event-position-num").focus(); }
      if (e.key === "z") { e.preventDefault(); toggleZoom(); }
      if (e.key === ".") { e.preventDefault(); toggleDetrend(); }
      if (e.key === "b") { e.preventDefault(); toggleBandpass(); }
    }
    if (e.key === "t") {
      showTab(eventsTabActive ? "synthetic" : "events");
    }
  });
}

function reflectFilterPills() {
  document.querySelectorAll(".filter-pill[data-band]").forEach((b) => {
    b.classList.toggle("active", state.bandFilter.has(b.dataset.band));
  });
  document.querySelectorAll(".filter-pill[data-region]").forEach((b) => {
    b.classList.toggle("active", state.regionFilter.has(b.dataset.region));
  });
}

function reflectSortButtons() {
  const sel = $("#sort-select");
  if (sel && sel.value !== state.sort) sel.value = state.sort;
  $("#reverse-btn").classList.toggle("active", state.sortReverse);
}

function sortAllEvents() {
  const key = state.sort;
  state.allEvents.sort((a, b) => {
    const av = a[key], bv = b[key];
    if (av == null && bv == null) return 0;
    if (av == null) return  1;
    if (bv == null) return -1;
    if (typeof av === "string") return av.localeCompare(bv);
    return av - bv;
  });
  if (state.sortReverse) state.allEvents.reverse();
}

function setSort(key) {
  if (state.sort === key) return;
  state.sort = key;
  reflectSortButtons();
  sortAllEvents();
  applyFilter({ keepPos: false });
}

function toggleReverse() {
  state.sortReverse = !state.sortReverse;
  reflectSortButtons();
  sortAllEvents();
  applyFilter({ keepPos: false });
}
function toggleBand(b) {
  if (!ALL_BANDS.includes(b)) return;
  if (state.bandFilter.has(b)) state.bandFilter.delete(b);
  else state.bandFilter.add(b);
  reflectFilterPills();
  applyFilter();
}
function toggleRegion(r) {
  if (!ALL_REGIONS.includes(r)) return;
  if (state.regionFilter.has(r)) state.regionFilter.delete(r);
  else state.regionFilter.add(r);
  reflectFilterPills();
  applyFilter();
}

/* ============== Filter popover (single button → 2×2 grid) ============== */
//
// One Filter button opens a popover with a 2×2 grid of cells (Date /
// Amp / Q / Period). Each cell pairs a small distribution histogram
// with a dual-handle slider that brushes a range over it. Histograms
// cross-filter: brushing axis X reshapes the histograms of the OTHER
// axes to reflect the joint event population that passes the active
// constraints — but axis X's own histogram stays fixed while you drag
// it, giving a stable scaffold for brushing.

const POPOVER_HOVER_OPEN_MS  = 200;
const POPOVER_HOVER_CLOSE_MS = 300;
const HIST_BINS = { peak_time: 60, b_perp1_amp: 40, q_factor: 40, period_min: 40,
                    r_distance: 40, local_time: 48,
                    mag_lat: 40, l_shell: 40,
                    bperp1: 40, bperp2: 40, bpar: 40, stokes_d: 40 };
const HIST_BAR_GREY = "#3a3a3a";
const HIST_BAR_ACTIVE = "#6ee0b1";  // matches --filter-accent (Zoom-LED green-teal)
const HIST_AXIS_COLOR = "#6b6b6b";
const HIST_GRID_COLOR = "rgba(155, 155, 155, 0.12)";
// Gutters reserved inside the canvas for axis labels. Mirror in style.css
// (.filter-cell .range-slider-wrap margin-left/right) so the slider track's
// 0%–100% extent maps onto the same pixel range as the histogram bars.
const HIST_PAD_LEFT   = 28;
const HIST_PAD_RIGHT  = 6;
const HIST_PAD_TOP    = 4;
const HIST_PAD_BOTTOM = 16;
// X-axis tick positions per filter axis, in *value space*. Period gets
// the QP band centres (10 lives below the data minimum so it's clipped).
// Date is computed at runtime in xTicksFor() since it depends on the
// observed mission span.
const STATIC_X_TICKS = {
  b_perp1_amp: [2, 4, 6, 8],
  q_factor:    [5, 10, 15],
  period_min:  [10, 30, 60, 120],
  r_distance:  [5, 10, 30, 60, 100],   // Rs — Cassini orbit spans ~3–150
  local_time:  [0, 6, 12, 18, 24],     // h — cardinal LT (midnight/dawn/noon/dusk)
  mag_lat:     [-60, -30, 0, 30, 60],  // ° — symmetric about magnetic equator
  l_shell:     [10, 20, 50, 100],      // dipole L = R / cos²(λ); long tail
  bperp1:      [2, 4, 6, 8],
  bperp2:      [2, 4, 6, 8],
  bpar:        [2, 4, 6, 8],
  stokes_d:    [-1, -0.5, 0, 0.5, 1],  // full theoretical [−1,+1] domain
};

let popOpen = false;
let popPinned = false;
let popOpenT = null;
let popCloseT = null;
const cellRenderers = new Map();  // key → { repaintHist, syncSlider }

function formatRangeValue(rf, v) {
  if (rf.isDate) return new Date(v * 1000).toISOString().slice(0, 10);
  if (rf.step === 1) return Math.round(v).toString();
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 10)  return v.toFixed(1);
  return v.toFixed(2);
}

function epochToDateInput(epoch) {
  return new Date(epoch * 1000).toISOString().slice(0, 10);
}
function dateInputToEpoch(s) {
  const t = Date.parse(s + "T00:00:00Z");
  return Number.isFinite(t) ? t / 1000 : null;
}

function buildSliderUI(rf, onChange) {
  const wrap = document.createElement("div");
  wrap.className = "range-slider-wrap";
  const track = document.createElement("div");
  track.className = "range-track";
  const fill = document.createElement("div");
  fill.className = "range-fill";
  const lo = document.createElement("input");
  const hi = document.createElement("input");
  lo.type = hi.type = "range";
  lo.min  = hi.min  = String(rf.fmin);
  lo.max  = hi.max  = String(rf.fmax);
  const span = rf.fmax - rf.fmin;
  const step = rf.step ?? (span > 0 ? span / 200 : 1);
  lo.step = hi.step = String(step);
  lo.value = String(rf.min);
  hi.value = String(rf.max);
  lo.className = "rs-low";
  hi.className = "rs-high";
  wrap.append(track, fill, lo, hi);
  function pct(v) { return ((v - rf.fmin) / span) * 100; }
  function paint() {
    fill.style.setProperty("--lo-pct", pct(parseFloat(lo.value)) + "%");
    fill.style.setProperty("--hi-pct", pct(parseFloat(hi.value)) + "%");
  }
  paint();
  lo.addEventListener("input", () => {
    if (parseFloat(lo.value) > parseFloat(hi.value)) lo.value = hi.value;
    paint(); onChange(parseFloat(lo.value), parseFloat(hi.value));
  });
  hi.addEventListener("input", () => {
    if (parseFloat(hi.value) < parseFloat(lo.value)) hi.value = lo.value;
    paint(); onChange(parseFloat(lo.value), parseFloat(hi.value));
  });
  return { el: wrap, syncFromState: () => {
    lo.value = String(rf.min); hi.value = String(rf.max); paint();
  }};
}

/* ---- histogram compute + render ---- */

function computeHistogram(key, events) {
  const rf = state.rangeFilters[key];
  const nb = HIST_BINS[key];
  const fmin = rf.fmin, fmax = rf.fmax;
  const span = fmax - fmin;
  const counts = new Uint32Array(nb);
  if (span <= 0) return { fmin, fmax, counts };
  for (const e of events) {
    const v = rfValue(e, key, rf);
    if (v == null || !Number.isFinite(v)) continue;
    let i = Math.floor(((v - fmin) / span) * nb);
    if (i < 0) i = 0; else if (i >= nb) i = nb - 1;
    counts[i]++;
  }
  return { fmin, fmax, counts };
}

function eventsForHistogram(excludeKey) {
  // Cross-filter: events that pass band, region, and every active range
  // filter EXCEPT the one we're computing the histogram for. That's why
  // the dragged axis's own histogram stays put — we exclude it from the
  // constraint set so its bars never shrink under their own brush.
  const bf = state.bandFilter, rfs = state.regionFilter;
  const active = [];
  for (const k in state.rangeFilters) {
    if (k === excludeKey) continue;
    const r = state.rangeFilters[k];
    if (r.active) active.push([k, r]);
  }
  return state.allEvents.filter((e) => {
    if (!bf.has(e.band) || !rfs.has(e.region)) return false;
    for (const [k, r] of active) {
      const v = rfValue(e, k, r);
      if (v == null || !Number.isFinite(v)) return false;
      if (v < r.min || v > r.max) return false;
    }
    return true;
  });
}

function xTicksFor(key, fmin, fmax) {
  if (key === "peak_time") {
    // Year ticks across the observed span. fmin/fmax are unix epoch s.
    const y0 = new Date(fmin * 1000).getUTCFullYear();
    const y1 = new Date(fmax * 1000).getUTCFullYear();
    const stride = y1 - y0 > 8 ? 4 : 2;  // 4-year ticks for >8-year span
    const out = [];
    const start = Math.ceil((y0 + 1) / stride) * stride;
    for (let y = start; y < y1; y += stride) {
      const e = Date.UTC(y, 0, 1) / 1000;
      out.push({ value: e, label: String(y) });
    }
    return out;
  }
  return (STATIC_X_TICKS[key] || []).map((v) => ({ value: v, label: String(v) }));
}

function renderHistogram(canvas, hist, rf, key) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
  if (!cssW || !cssH) return;
  if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
    canvas.width  = cssW * dpr;
    canvas.height = cssH * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  // Bar area inside the gutters reserved for axis labels.
  const barX = HIST_PAD_LEFT;
  const barY = HIST_PAD_TOP;
  const barW = cssW - HIST_PAD_LEFT - HIST_PAD_RIGHT;
  const barH = cssH - HIST_PAD_TOP - HIST_PAD_BOTTOM;
  if (barW <= 0 || barH <= 0) return;

  const nb = hist.counts.length;
  let mx = 0;
  for (let i = 0; i < nb; i++) if (hist.counts[i] > mx) mx = hist.counts[i];

  ctx.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillStyle = HIST_AXIS_COLOR;

  // Y-axis: subtle baseline + max-count label at top.
  ctx.fillStyle = HIST_GRID_COLOR;
  ctx.fillRect(barX, barY + barH, barW, 1);  // baseline
  if (mx > 0) {
    ctx.fillStyle = HIST_GRID_COLOR;
    ctx.fillRect(barX, barY, barW, 1);       // top gridline at max
    ctx.fillStyle = HIST_AXIS_COLOR;
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillText(String(mx), barX - 3, barY - 1);
    ctx.textBaseline = "alphabetic";
    ctx.fillText("0", barX - 3, barY + barH + 1);
  }

  // Bars. When the filter is disabled, every bar stays grey — the
  // checkbox is the explicit gate, so the histogram only "lights up"
  // (cyan in-range bars) once the user opts in.
  if (mx > 0) {
    const span = hist.fmax - hist.fmin;
    const inMin = (rf.min - hist.fmin) / span;
    const inMax = (rf.max - hist.fmin) / span;
    const colW = barW / nb;
    for (let i = 0; i < nb; i++) {
      const h = (hist.counts[i] / mx) * barH;
      const x = barX + i * colW;
      let color = HIST_BAR_GREY;
      if (rf.active) {
        const binCenter = (i + 0.5) / nb;
        if (binCenter >= inMin && binCenter <= inMax) color = HIST_BAR_ACTIVE;
      }
      ctx.fillStyle = color;
      ctx.fillRect(x + 0.5, barY + barH - h, Math.max(1, colW - 1), h);
    }
  }

  // X-axis ticks (clipped to the slider's value domain so labels never
  // sit outside the brushable area).
  const ticks = xTicksFor(key, hist.fmin, hist.fmax);
  const span = hist.fmax - hist.fmin;
  ctx.fillStyle = HIST_AXIS_COLOR;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of ticks) {
    if (t.value < hist.fmin || t.value > hist.fmax) continue;
    const px = barX + ((t.value - hist.fmin) / span) * barW;
    ctx.fillStyle = HIST_GRID_COLOR;
    ctx.fillRect(px, barY + barH, 1, 3);     // tick mark
    ctx.fillStyle = HIST_AXIS_COLOR;
    ctx.fillText(t.label, px, barY + barH + 4);
  }
}

/* ---- popover content (2×2 grid) ---- */

function buildFilterCell(key) {
  const rf = state.rangeFilters[key];
  const cell = document.createElement("div");
  cell.className = "filter-cell";
  cell.dataset.range = key;

  // Title row with a custom-styled checkbox. The checkbox mirrors
  // `rf.active` and gives the user explicit on/off control without
  // needing a separate per-cell reset button.
  const title = document.createElement("label");
  title.className = "cell-title cell-toggle";
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = !!rf.active;
  const box = document.createElement("span");
  box.className = "check-box";
  const titleText = document.createElement("span");
  titleText.className = "title-text";
  titleText.textContent = rf.unit ? `${rf.label} (${rf.unit})` : rf.label;
  title.append(cb, box, titleText);
  // Tooltip for the four scientific concepts the user might not recognise.
  if (HELP_TEXT[key]) title.appendChild(helpHintEl(HELP_TEXT[key]));
  cell.appendChild(title);

  const canvas = document.createElement("canvas");
  canvas.className = "histogram-canvas";
  cell.appendChild(canvas);

  let inLo, inHi;
  const slider = buildSliderUI(rf, (lo, hi) => {
    if (rf.isDate) {
      inLo.value = epochToDateInput(lo);
      inHi.value = epochToDateInput(hi);
    } else {
      inLo.value = formatRangeValue(rf, lo);
      inHi.value = formatRangeValue(rf, hi);
    }
    // Dragging away from the default range auto-enables the filter so
    // the brush has an immediate effect. Reflect the new state in the
    // checkbox so the visible toggle stays in sync.
    onCellSliderChange(key, lo, hi);
    cb.checked = !!rf.active;
  });
  cell.appendChild(slider.el);

  const numRow = document.createElement("div");
  numRow.className = "pop-row";
  inLo = document.createElement("input");
  inHi = document.createElement("input");
  if (rf.isDate) {
    inLo.type = inHi.type = "date";
    inLo.value = epochToDateInput(rf.min);
    inHi.value = epochToDateInput(rf.max);
  } else {
    inLo.type = inHi.type = "number";
    inLo.step = inHi.step = "any";
    inLo.value = formatRangeValue(rf, rf.min);
    inHi.value = formatRangeValue(rf, rf.max);
  }
  const dash = document.createElement("span");
  dash.className = "dash";
  dash.textContent = rf.isDate ? "→" : "–";
  numRow.append(inLo, dash, inHi);
  cell.appendChild(numRow);

  // Numeric / date input → state, then sync slider + redraw.
  const onNum = () => {
    let lo, hi;
    if (rf.isDate) {
      lo = dateInputToEpoch(inLo.value); hi = dateInputToEpoch(inHi.value);
    } else {
      lo = parseFloat(inLo.value); hi = parseFloat(inHi.value);
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi < lo) return;
    lo = Math.max(rf.fmin, Math.min(rf.fmax, lo));
    hi = Math.max(rf.fmin, Math.min(rf.fmax, hi));
    onCellSliderChange(key, lo, hi);
    slider.syncFromState();
    cb.checked = !!rf.active;
  };
  inLo.addEventListener("change", onNum);
  inHi.addEventListener("change", onNum);

  // Checkbox: explicit toggle that preserves the slider's current
  // [min, max]. Lets the user disable a filter without losing the
  // configured range, so re-enabling restores it instantly.
  cb.addEventListener("change", () => {
    rf.active = cb.checked;
    reflectFilterButton();
    cellRenderers.get(key)?.repaintHist();
    scheduleCrossFilterUpdate(key);
  });

  cellRenderers.set(key, {
    repaintHist: () => renderHistogram(canvas, computeHistogram(key, eventsForHistogram(key)), rf, key),
    syncSlider:  () => { slider.syncFromState(); cb.checked = !!rf.active; },
  });
  return cell;
}

// Two-grid split: the main grid keeps the original 2-col layout
// unchanged; the right column hosts the explicit B-component sliders
// so all three components are visible side-by-side for comparison.
const FILTER_GRID_KEYS_MAIN = [
  "peak_time",  "b_perp1_amp",
  "q_factor",   "period_min",
  "r_distance", "local_time",
  "mag_lat",    "l_shell",
];
const FILTER_GRID_KEYS_COMPONENTS = ["bperp1", "bperp2", "bpar", "stokes_d"];

function buildFilterGridContent() {
  const root = $("#range-popover");
  root.innerHTML = "";
  cellRenderers.clear();

  const body = document.createElement("div");
  body.className = "popover-body";

  const mainGrid = document.createElement("div");
  mainGrid.className = "filter-grid";
  for (const key of FILTER_GRID_KEYS_MAIN) {
    mainGrid.appendChild(buildFilterCell(key));
  }
  body.appendChild(mainGrid);

  const divider = document.createElement("div");
  divider.className = "vertical-divider";
  body.appendChild(divider);

  const compGrid = document.createElement("div");
  compGrid.className = "filter-grid filter-grid-components";
  for (const key of FILTER_GRID_KEYS_COMPONENTS) {
    compGrid.appendChild(buildFilterCell(key));
  }
  body.appendChild(compGrid);

  root.appendChild(body);

  const footer = document.createElement("div");
  footer.className = "popover-footer";
  const count = document.createElement("span");
  count.className = "match-count";
  count.id = "popover-match-count";
  const resetAll = document.createElement("button");
  resetAll.type = "button";
  resetAll.className = "reset-btn";
  resetAll.textContent = "reset all";
  resetAll.addEventListener("click", () => {
    clearAllRangeFilters();
    syncAllSliders();
    refreshAllHistograms();
    applyFilter();
    updateMatchCount();
  });
  footer.append(count, resetAll);
  root.appendChild(footer);
}

/* ---- cross-filter update flow ---- */

function refreshAllHistograms() {
  for (const [, r] of cellRenderers) r.repaintHist();
}
function refreshOtherHistograms(key) {
  for (const [k, r] of cellRenderers) if (k !== key) r.repaintHist();
}
function syncAllSliders() {
  for (const [, r] of cellRenderers) r.syncSlider();
}
function updateMatchCount() {
  const el = document.getElementById("popover-match-count");
  if (el) el.textContent = `${state.events.length} of ${state.allEvents.length} match`;
}

let crossUpdatePending = false;
let crossUpdateExcludeKey = null;
function scheduleCrossFilterUpdate(excludeKey) {
  crossUpdateExcludeKey = excludeKey;
  if (crossUpdatePending) return;
  crossUpdatePending = true;
  requestAnimationFrame(() => {
    crossUpdatePending = false;
    applyFilter();
    refreshOtherHistograms(crossUpdateExcludeKey);
    updateMatchCount();
  });
}

function onCellSliderChange(key, lo, hi) {
  const rf = state.rangeFilters[key];
  rf.min = lo; rf.max = hi;
  const eps = (rf.fmax - rf.fmin) * 1e-6;
  rf.active = (lo > rf.fmin + eps) || (hi < rf.fmax - eps);
  reflectFilterButton();
  // The dragged axis repaints synchronously so the cyan brush-overlay
  // tracks the handle without lag. Other cells + applyFilter happen in
  // requestAnimationFrame to coalesce rapid `input` events.
  cellRenderers.get(key)?.repaintHist();
  scheduleCrossFilterUpdate(key);
}

/* ---- filter-button display + popover lifecycle ---- */

function reflectFilterButton() {
  const btn = $("#filter-btn");
  if (!btn) return;
  let n = 0;
  for (const k in state.rangeFilters) if (state.rangeFilters[k].active) n++;
  btn.classList.toggle("active", n > 0);
  const badge = $("#filter-active-count");
  if (n > 0) { badge.hidden = false; badge.textContent = String(n); }
  else       { badge.hidden = true;  badge.textContent = ""; }
}

function positionPopover(anchor) {
  const pop = $("#range-popover");
  const ar = anchor.getBoundingClientRect();
  // Align the popover under the anchor, but keep it inside the viewport
  // horizontally — for a 540 px popover the right edge would otherwise
  // clip on narrower windows.
  const popW = pop.offsetWidth || 540;
  const margin = 8;
  let left = window.scrollX + ar.left;
  const maxLeft = window.scrollX + document.documentElement.clientWidth - popW - margin;
  if (left > maxLeft) left = Math.max(margin + window.scrollX, maxLeft);
  pop.style.left = left + "px";
  pop.style.top  = (window.scrollY + ar.bottom + 10) + "px";
  // Arrow points up at the anchor's centre, clamped to the popover bounds.
  const anchorCenterPage = window.scrollX + ar.left + ar.width / 2;
  const arrowX = Math.max(12, Math.min(popW - 24, anchorCenterPage - left - 6));
  pop.style.setProperty("--arrow-x", arrowX + "px");
}

function openFilterPopover(pinned) {
  const btn = $("#filter-btn");
  if (!popOpen) buildFilterGridContent();
  popOpen = true;
  popPinned = popPinned || pinned;
  const pop = $("#range-popover");
  pop.classList.remove("hidden");
  positionPopover(btn);
  btn.setAttribute("aria-expanded", "true");
  // Histograms need the canvas to have a non-zero clientWidth — paint
  // after the popover is visible & laid out.
  requestAnimationFrame(() => {
    refreshAllHistograms();
    updateMatchCount();
  });
}

function closeFilterPopover() {
  popOpen = false;
  popPinned = false;
  if (popOpenT)  { clearTimeout(popOpenT);  popOpenT = null; }
  if (popCloseT) { clearTimeout(popCloseT); popCloseT = null; }
  $("#range-popover").classList.add("hidden");
  const btn = $("#filter-btn");
  if (btn) btn.setAttribute("aria-expanded", "false");
}

function bindFilterButton() {
  const btn = $("#filter-btn");
  const pop = $("#range-popover");
  if (!btn || !pop) return;

  btn.addEventListener("mouseenter", () => {
    if (popCloseT) { clearTimeout(popCloseT); popCloseT = null; }
    if (popOpen) return;
    if (popOpenT) clearTimeout(popOpenT);
    popOpenT = setTimeout(() => openFilterPopover(false), POPOVER_HOVER_OPEN_MS);
  });
  btn.addEventListener("mouseleave", () => {
    if (popOpenT) { clearTimeout(popOpenT); popOpenT = null; }
    if (popPinned) return;
    popCloseT = setTimeout(closeFilterPopover, POPOVER_HOVER_CLOSE_MS);
  });
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    if (popOpen && popPinned) { closeFilterPopover(); return; }
    openFilterPopover(true);
  });

  pop.addEventListener("mouseenter", () => {
    if (popCloseT) { clearTimeout(popCloseT); popCloseT = null; }
  });
  pop.addEventListener("mouseleave", () => {
    if (popPinned) return;
    popCloseT = setTimeout(closeFilterPopover, POPOVER_HOVER_CLOSE_MS);
  });

  document.addEventListener("click", (e) => {
    if (!popPinned) return;
    if (pop.contains(e.target)) return;
    if (btn.contains(e.target)) return;
    closeFilterPopover();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && popOpen) {
      e.preventDefault();
      closeFilterPopover();
    }
  });
  window.addEventListener("resize", debounce(() => {
    if (popOpen) positionPopover(btn);
  }, 100));
}

function clearAllRangeFilters() {
  for (const [, rf] of Object.entries(state.rangeFilters)) {
    rf.min = rf.fmin; rf.max = rf.fmax; rf.active = false;
  }
  reflectFilterButton();
}

function reflectToggles() {
  $("#zoom-btn").classList.toggle("active", state.zoom);
  $("#detrend-btn").classList.toggle("active", state.detrend);
  $("#bandpass-btn").classList.toggle("active", state.bandpass);
  $("#bandpass-slider-wrap").hidden = !state.bandpass;
  $("#span-btn").classList.toggle("active", state.showSpan);
  $("#peak-btn").classList.toggle("active", state.showPeak);
}

function syncBandpassUI() {
  const rf = state.bandpassRf;
  if (state.bandpassSlider) state.bandpassSlider.syncFromState();
  $("#bp-lo-label").textContent = rf.min == null ? "—" : `${rf.min.toFixed(0)}m`;
  $("#bp-hi-label").textContent = rf.max == null ? "—" : `${rf.max.toFixed(0)}m`;
  // Position the event-period marker (white vertical line inside the slider).
  const mark = state.bandpassMark;
  if (mark) {
    const e = state.events[state.pos];
    const P0 = e?.period_min;
    if (P0 != null && Number.isFinite(P0)) {
      const pct = Math.max(0, Math.min(100,
        (P0 - rf.fmin) / (rf.fmax - rf.fmin) * 100));
      mark.style.left = pct + "%";
      mark.title = `event period: ${P0.toFixed(0)} min`;
      mark.hidden = false;
    } else {
      mark.hidden = true;
    }
  }
}

function reseedBandpassFromEvent() {
  // Default band: low cutoff fixed at 10 min (below the Nyquist guard
  // for 1-min MAG data), high cutoff at P₀ + 50% (covers the wave's
  // natural spread on the slow side). For the rare event with P₀ ≤ 10 min
  // the low handle slides under P₀ so the wave itself isn't excluded.
  const rf = state.bandpassRf;
  const e = state.events[state.pos];
  const P0 = e?.period_min;
  if (P0 != null && Number.isFinite(P0)) {
    const loDefault = Math.min(10, Math.max(rf.fmin, Math.round(P0 - 5)));
    rf.min = Math.max(rf.fmin, loDefault);
    rf.max = Math.min(rf.fmax, Math.max(rf.min + 1, Math.round(P0 * 1.5)));
  } else {
    rf.min = rf.fmin;
    rf.max = rf.fmax;
  }
  syncBandpassUI();
}

function maybeReseedBandpass() {
  // Reseed only on event change — toggling sibling controls (zoom/detrend/
  // span/peak) calls renderEvent again with the same event, and we don't
  // want to clobber any manual slider edits the user made on this event.
  const e = state.events[state.pos];
  const id = e?.event_id ?? null;
  if (id === state.bandpassSeedEventId) return;
  state.bandpassSeedEventId = id;
  reseedBandpassFromEvent();
}

function rerenderCurrent() {
  if (state.lastWf && state.lastSpec) {
    const id = state.filteredIds[state.pos];
    renderEvent(state.lastWf, state.lastSpec, state.detailCache.get(id));
  }
}

function toggleZoom() {
  state.zoom = !state.zoom;
  reflectToggles();
  rerenderCurrent();
}

function toggleDetrend() {
  state.detrend = !state.detrend;
  reflectToggles();
  rerenderCurrent();
}

function toggleBandpass() {
  state.bandpass = !state.bandpass;
  reflectToggles();
  rerenderCurrent();
}

function bindBandpassSlider() {
  const wrap = $("#bp-slider");
  if (!wrap) return;
  const rf = state.bandpassRf;
  // Seed from full domain so the slider has valid initial values until the
  // first event loads and reseeds it to P₀ ± 50%.
  if (rf.min == null) rf.min = rf.fmin;
  if (rf.max == null) rf.max = rf.fmax;
  let pending = false;
  const slider = buildSliderUI(rf, (lo, hi) => {
    rf.min = lo; rf.max = hi;
    $("#bp-lo-label").textContent = `${lo.toFixed(0)}m`;
    $("#bp-hi-label").textContent = `${hi.toFixed(0)}m`;
    if (!state.bandpass) return;
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; rerenderCurrent(); });
  });
  wrap.appendChild(slider.el);
  // Insert a non-interactive marker showing where the event's P₀ sits
  // within the slider's domain. Positioned inside the slider wrap so it
  // moves with the track, behind the thumbs.
  const mark = document.createElement("div");
  mark.className = "bp-event-mark";
  mark.hidden = true;
  slider.el.appendChild(mark);
  state.bandpassSlider = slider;
  state.bandpassMark = mark;
  syncBandpassUI();
}

function toggleSpan() {
  state.showSpan = !state.showSpan;
  reflectToggles();
  rerenderCurrent();
}

function togglePeak() {
  state.showPeak = !state.showPeak;
  reflectToggles();
  rerenderCurrent();
}

/* ================================ Init ================================ */

function bindUI() {
  tabEventsEl = $("#tab-events");
  document.querySelectorAll(".tab").forEach(
    (t) => t.addEventListener("click", () => showTab(t.dataset.tab)),
  );
  document.querySelectorAll(".subtab").forEach(
    (t) => t.addEventListener("click", () => showSubtab(t.dataset.subtab)),
  );

  $("#prev-btn").addEventListener("click", () => loadEventAtPos(state.pos - 1));
  $("#next-btn").addEventListener("click", () => loadEventAtPos(state.pos + 1));

  const posEl = $("#event-position-num");
  posEl.addEventListener("focus", () => {
    // Select existing text so typing replaces it.
    const sel = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(posEl);
    sel.removeAllRanges();
    sel.addRange(range);
  });
  posEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); posEl.blur(); }
    else if (e.key === "Escape") {
      e.preventDefault();
      posEl.textContent = state.pos + 1;
      posEl.blur();
    }
  });
  posEl.addEventListener("blur", () => {
    const n = parseInt((posEl.textContent || "").replace(/\D/g, ""), 10);
    if (!state.filteredIds.length) { posEl.textContent = "—"; return; }
    if (!Number.isFinite(n)) { posEl.textContent = state.pos + 1; return; }
    const total = state.filteredIds.length;
    const clamped = Math.min(Math.max(1, n), total);
    if (clamped - 1 !== state.pos) loadEventAtPos(clamped - 1);
    else posEl.textContent = state.pos + 1;
  });

  const uidEl = $("#event-uid");
  if (uidEl) {
    uidEl.addEventListener("focus", () => {
      const sel = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(uidEl);
      sel.removeAllRanges();
      sel.addRange(range);
    });
    uidEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); uidEl.blur(); }
      else if (e.key === "Escape") {
        e.preventDefault();
        const cur = state.events[state.pos];
        uidEl.textContent = cur ? cur.event_uid || "" : "";
        uidEl.blur();
      }
    });
    uidEl.addEventListener("blur", () => {
      const typed = (uidEl.textContent || "").trim();
      const cur = state.events[state.pos];
      const curUid = cur ? cur.event_uid || "" : "";
      if (!typed || typed.toUpperCase() === curUid.toUpperCase()) {
        uidEl.textContent = curUid;
        return;
      }
      if (!gotoEventByUid(typed)) {
        setStatus(`uid ${typed} not found`);
        uidEl.textContent = curUid;
      }
    });
  } else {
    console.warn("[QP] #event-uid missing — stale HTML, hard-reload required.");
  }

  document.querySelectorAll(".filter-pill[data-band]").forEach((b) => {
    b.addEventListener("click", () => toggleBand(b.dataset.band));
  });
  document.querySelectorAll(".filter-pill[data-region]").forEach((b) => {
    b.addEventListener("click", () => toggleRegion(b.dataset.region));
  });
  $("#sort-select").addEventListener("change", (e) => setSort(e.target.value));
  $("#reverse-btn").addEventListener("click", toggleReverse);
  reflectFilterPills();
  reflectSortButtons();
  bindFilterButton();

  $("#zoom-btn").addEventListener("click", toggleZoom);
  $("#detrend-btn").addEventListener("click", toggleDetrend);
  $("#bandpass-btn").addEventListener("click", toggleBandpass);
  bindBandpassSlider();
  $("#span-btn").addEventListener("click", toggleSpan);
  $("#peak-btn").addEventListener("click", togglePeak);
  reflectToggles();

  ["syn-band", "syn-amp", "syn-period", "syn-decay", "syn-noise", "syn-seed"]
    .forEach((id) => $("#" + id).addEventListener("input", debouncedSyn));
  $("#syn-randomize").addEventListener("click", () => {
    $("#syn-seed").value = Math.floor(Math.random() * 100000);
    loadSynthetic();
  });
  document.querySelectorAll(".presets button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const cfg = SYN_PRESETS[btn.dataset.preset];
      $("#syn-amp").value = cfg.amp;
      $("#syn-period").value = cfg.period_min;
      $("#syn-decay").value = cfg.decay_h;
      $("#syn-noise").value = cfg.noise;
      loadSynthetic();
    });
  });
  $("#bench-preset").addEventListener("change", () => {
    state.benchData = null;
    loadBenchmark();
  });
}

(function init() {
  bindUI();
  bindKeys();
  bindTimelineInteraction();
  fetchEvents();
})();
