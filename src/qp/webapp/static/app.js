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
const BAND_COLORS = { QP30: "#80c0ff", QP60: "#ffb000", QP120: "#f06090" };
// Indexed by uPlot series index (0 = x). Used by cursor.points.fill — uPlot
// wraps series.stroke into a function internally, so we cannot read it back
// at runtime; we keep the literal hex strings here instead.
const WAVE_CURSOR_COLORS = [
  null, FIELD_COLORS.par, FIELD_COLORS.perp1, FIELD_COLORS.perp2, FIELD_COLORS.tot,
];

// Rolling-mean window per band (minutes ≈ 4× central period). Removing
// this trend makes the small QP perturbations visible against the
// dominant background field, especially in |B|.
const DETREND_WINDOW_MIN = { QP30: 120, QP60: 240, QP120: 480 };
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
    periods: [30, 60, 120],
  });
}

const ALL_BANDS = ["QP30", "QP60", "QP120"];
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
  rangeFilters: {
    peak_time:   { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "Date",   unit: "",    isDate: true,  step: null },
    b_perp1_amp: { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "Amp",    unit: "nT",  isDate: false, step: null },
    q_factor:    { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "Q",      unit: "",    isDate: false, step: null },
    period_min:  { active: false, min: null, max: null, fmin: null, fmax: null,
                   label: "Period", unit: "min", isDate: false, step: 1 },
  },
  sort: "peak_time",
  sortReverse: false,
  hoverEventId: null,  // event hovered on the timeline canvas
  hoverEvent: null,    // full hit object: {x, id, band, peak_time}
  zoom: true,          // focus on event window ± 1h (default on)
  detrend: false,      // subtract band-aware rolling mean
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

function statTile(label, val, unit, extraClass) {
  const u = unit ? `<em>${unit}</em>` : "";
  const cls = "stat-val" + (extraClass ? " " + extraClass : "");
  return `<div class="stat"><span class="stat-label">${label}</span>` +
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

function renderEventStats(summary, wf, detail) {
  // Heavy stat fields live in the lazily-fetched `detail`; fall back to
  // `summary` if it hasn't arrived yet (rare — fetched in parallel with
  // the waveform). `wf`/`summary` always cover the band/region pills.
  const s = detail || {};
  const q = s.q_factor;
  const qPill = `<span class="q-chip ${qFactorClass(q)}">${fmt(q, 2)}</span>`;
  const region = escapeHtml(wf?.region ?? summary.region ?? "unknown");
  const regionPill = `<span class="pill" data-region="${region}">${region}</span>`;
  const band = escapeHtml(wf?.band ?? summary.band ?? "?");
  const bandPill = `<span class="pill" data-band="${band}">${band}</span>`;
  const tFrom = fmtIsoCompact(wf?.date_from);
  const tPeak = fmtIsoCompact(wf?.peak_time);
  const tTo   = fmtIsoCompact(wf?.date_to);

  $("#event-stats").innerHTML = `
    <div class="stat-group">
      <div class="stat-group-title">Location</div>
      <div class="stat-grid">
        ${statTile("R",        fmt(s.r_distance, 1), "R<sub>S</sub>")}
        ${statTile("Mag lat",  fmt(s.mag_lat, 1) + "°", "")}
        ${statTile("LT",       fmt(s.local_time, 1), "h")}
        <div class="stat span-all"><span class="stat-label">Region</span>
          <span class="stat-val">${regionPill}</span></div>
      </div>
    </div>
    <div class="stat-group">
      <div class="stat-group-title">Time</div>
      <div class="stat-grid stack">
        ${statTile("Start",   tFrom, "", "time")}
        ${statTile("Peak",    tPeak, "", "time")}
        ${statTile("End",     tTo,   "", "time")}
      </div>
    </div>
    <div class="stat-group">
      <div class="stat-group-title">Wave</div>
      <div class="stat-grid">
        ${statTile("Period",    fmt(s.period_min, 1), "min")}
        ${statTile("Duration",  fmt(s.duration_minutes, 0), "min")}
        <div class="stat"><span class="stat-label">Q factor</span>
          <span class="stat-val">${qPill}</span></div>
        <div class="stat span-all"><span class="stat-label">Band</span>
          <span class="stat-val">${bandPill}</span></div>
      </div>
    </div>
    <div class="stat-group">
      <div class="stat-group-title">Polarization</div>
      <div class="stat-grid cols-2">
        ${statTile("|B<sub>⊥1</sub>|", fmt(s.b_perp1_amp, 2), "nT")}
        ${statTile("|B<sub>⊥2</sub>|", fmt(s.b_perp2_amp, 2), "nT")}
        ${statTile("|B<sub>∥</sub>|",  fmt(s.b_par_amp, 2),   "nT")}
        ${statTile("Stokes d",         fmt(s.stokes_d, 2),    "")}
      </div>
    </div>
  `;
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

  // year tick lines through event area
  c.font = "10px ui-monospace, monospace";
  c.textAlign = "center";
  for (let y = 2004; y <= 2018; y++) {
    const x = xOf(toEpoch(`${y}-01-01T00:00:00`));
    c.fillStyle = "#333";
    c.fillRect(x, TL_EVENT_Y, 1, TL_EVENT_H);
    if (y % 2 === 0) {
      c.fillStyle = "#9b9b9b";
      c.fillText(String(y), x, TL_HEIGHT - 3);
    }
  }

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
      ctx.setLineDash([]);
      const fontPx = 26;
      ctx.font = `700 ${fontPx}px ui-monospace, monospace`;
      ctx.textBaseline = "top";
      ctx.lineJoin = "round";
      for (const p of periods) {
        const x = u.valToPos(p, "x", true);
        const label = `${p} min`;
        const lx = x + 8;
        const ly = top + 8;
        ctx.lineWidth = 4;
        ctx.strokeStyle = "rgba(15, 15, 15, 0.85)";
        ctx.strokeText(label, lx, ly);
        ctx.fillStyle = "#f5f5f5";
        ctx.fillText(label, lx, ly);
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
        points: { show: true, size: 3.5, fill: FIELD_DOT_FILL.par,   stroke: FIELD_DOT_FILL.par } },
      { label: "B⊥₁",  stroke: FIELD_COLORS.perp1, width: 1, spanGaps: true,
        points: { show: true, size: 3.5, fill: FIELD_DOT_FILL.perp1, stroke: FIELD_DOT_FILL.perp1 } },
      { label: "B⊥₂",  stroke: FIELD_COLORS.perp2, width: 1, spanGaps: true,
        points: { show: true, size: 3.5, fill: FIELD_DOT_FILL.perp2, stroke: FIELD_DOT_FILL.perp2 } },
      { label: "|B|",  stroke: FIELD_COLORS.tot,   width: 1, spanGaps: true,
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
      x: { distr: 3, log: 10, range: () => [10, 240] },
      y: { distr: 3, log: 10 },
    },
    series: [
      { label: "period [min]" },
      { label: "PSD∥",  stroke: FIELD_COLORS.par,   width: 1, points: { size: 3 } },
      { label: "PSD⊥₁", stroke: FIELD_COLORS.perp1, width: 1, points: { size: 3 } },
      { label: "PSD⊥₂", stroke: FIELD_COLORS.perp2, width: 1, points: { size: 3 } },
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

function updateSpectrumPlot(target, period_min, psd_par, psd_p1, psd_p2, periods) {
  const p = ensureSpectrumPlot(target);
  _qpCtx(p).periods = periods || [30, 60, 120];
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

function computeRangeBounds() {
  for (const [key, rf] of Object.entries(state.rangeFilters)) {
    let mn = +Infinity, mx = -Infinity;
    for (const e of state.allEvents) {
      const v = rf.isDate ? toEpoch(e.peak_time) : e[key];
      if (v == null || !Number.isFinite(v)) continue;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    if (Number.isFinite(mn) && Number.isFinite(mx)) {
      rf.fmin = mn; rf.fmax = mx;
      rf.min = mn;  rf.max = mx;
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
      const v = r.isDate ? toEpoch(e.peak_time) : e[key];
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
    ];
    if (!cachedDetail) {
      requests.push(fetch(`/api/events/${id}`, { signal: ctrl.signal }));
    }
    const responses = await Promise.all(requests);
    const [wfR, specR, detailR] = responses;
    setStatus(`fetched event ${id}, parsing…`);
    if (!wfR.ok) throw new Error(`waveform HTTP ${wfR.status}`);
    if (!specR.ok) throw new Error(`spectrum HTTP ${specR.status}`);
    const wf = await wfR.json();
    const spec = await specR.json();
    let detail = cachedDetail;
    if (detailR) {
      if (!detailR.ok) throw new Error(`detail HTTP ${detailR.status}`);
      detail = await detailR.json();
      state.detailCache.set(id, detail);
    }
    setStatus(`rendering event ${id}…`);
    renderEvent(wf, spec, detail);
    drawTimeline(id);
    setStatus(`event ${id}`);
  } catch (e) {
    if (e.name !== "AbortError") {
      setStatus(`error loading ${id}: ${e.message}`);
      console.error("[QP]", e);
    }
  }
}

function renderEvent(wf, spec, detail) {
  const summary = state.events[state.pos] || {};
  const total = state.events.length;
  $("#event-position-num").textContent = state.pos + 1;
  $("#event-position-total").textContent = total;
  $("#event-uid").textContent = summary.event_uid || "";
  renderEventStats(summary, wf, detail);

  state.lastWf = wf;
  state.lastSpec = spec;

  const xs = wf.epoch_s;
  let par = wf.b_par, p1 = wf.b_perp1, p2 = wf.b_perp2, tot = wf.b_tot;
  let title = "MFA components";
  if (state.detrend) {
    const dt = detrendComponents(wf);
    par = dt.b_par; p1 = dt.b_perp1; p2 = dt.b_perp2; tot = dt.b_tot;
    title = `MFA components, detrended (${dt.windowMin}-min rolling mean)`;
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
  );
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
      if (e.key === "1") { toggleBand("QP30"); }
      if (e.key === "2") { toggleBand("QP60"); }
      if (e.key === "3") { toggleBand("QP120"); }
      if (e.key === "m") { toggleRegion("magnetosphere"); }
      if (e.key === "s") { toggleRegion("magnetosheath"); }
      if (e.key === "w") { toggleRegion("solar_wind"); }
      if (e.key === "g") { e.preventDefault(); $("#event-position-num").focus(); }
      if (e.key === "z") { e.preventDefault(); toggleZoom(); }
      if (e.key === ".") { e.preventDefault(); toggleDetrend(); }
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
const HIST_BINS = { peak_time: 60, b_perp1_amp: 40, q_factor: 40, period_min: 40 };
const HIST_BAR_GREY = "#3a3a3a";
const HIST_BAR_ACTIVE = "#4dd2ff";  // matches --accent

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
    const v = rf.isDate ? toEpoch(e.peak_time) : e[key];
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
      const v = r.isDate ? toEpoch(e.peak_time) : e[k];
      if (v == null || !Number.isFinite(v)) return false;
      if (v < r.min || v > r.max) return false;
    }
    return true;
  });
}

function renderHistogram(canvas, hist, rf) {
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
  const nb = hist.counts.length;
  let mx = 0;
  for (let i = 0; i < nb; i++) if (hist.counts[i] > mx) mx = hist.counts[i];
  if (mx === 0) return;
  const barW = cssW / nb;
  const span = hist.fmax - hist.fmin;
  const inMin = (rf.min - hist.fmin) / span;
  const inMax = (rf.max - hist.fmin) / span;
  for (let i = 0; i < nb; i++) {
    const h = (hist.counts[i] / mx) * (cssH - 2);
    const x = i * barW;
    const binCenter = (i + 0.5) / nb;
    const inRange = binCenter >= inMin && binCenter <= inMax;
    ctx.fillStyle = inRange ? HIST_BAR_ACTIVE : HIST_BAR_GREY;
    ctx.fillRect(x + 0.5, cssH - h, Math.max(1, barW - 1), h);
  }
}

/* ---- popover content (2×2 grid) ---- */

function buildFilterCell(key) {
  const rf = state.rangeFilters[key];
  const cell = document.createElement("div");
  cell.className = "filter-cell";
  cell.dataset.range = key;

  const title = document.createElement("div");
  title.className = "cell-title";
  title.textContent = rf.unit ? `${rf.label} (${rf.unit})` : rf.label;
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
    onCellSliderChange(key, lo, hi);
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

  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "cell-reset";
  reset.textContent = "reset";
  reset.addEventListener("click", () => {
    rf.min = rf.fmin; rf.max = rf.fmax; rf.active = false;
    if (rf.isDate) {
      inLo.value = epochToDateInput(rf.fmin);
      inHi.value = epochToDateInput(rf.fmax);
    } else {
      inLo.value = formatRangeValue(rf, rf.fmin);
      inHi.value = formatRangeValue(rf, rf.fmax);
    }
    slider.syncFromState();
    reflectFilterButton();
    refreshAllHistograms();
    applyFilter();
    updateMatchCount();
  });
  cell.appendChild(reset);

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
  };
  inLo.addEventListener("change", onNum);
  inHi.addEventListener("change", onNum);

  cellRenderers.set(key, {
    repaintHist: () => renderHistogram(canvas, computeHistogram(key, eventsForHistogram(key)), rf),
    syncSlider:  () => slider.syncFromState(),
  });
  return cell;
}

function buildFilterGridContent() {
  const root = $("#range-popover");
  root.innerHTML = "";
  cellRenderers.clear();
  const grid = document.createElement("div");
  grid.className = "filter-grid";
  for (const key of Object.keys(state.rangeFilters)) {
    grid.appendChild(buildFilterCell(key));
  }
  root.appendChild(grid);

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
  $("#span-btn").classList.toggle("active", state.showSpan);
  $("#peak-btn").classList.toggle("active", state.showPeak);
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
