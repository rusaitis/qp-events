"""FastAPI server for the QP event review web app.

Run with ``uv run python -m qp.webapp`` or
``uv run uvicorn qp.webapp.server:app --reload``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from . import loaders, synthetic

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="QP Event Review", docs_url="/api/docs", redoc_url=None)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def disable_static_cache(request: Request, call_next):
    """Force the browser to revalidate every reload. ETag still gives 304s."""
    response = await call_next(request)
    if request.url.path.startswith("/static/") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.get("/api/events")
def list_events(
    band: str | None = Query(None, pattern="^(QP30|QP60|QP120)$"),
    region: str | None = Query(
        None, pattern="^(magnetosphere|magnetosheath|solar_wind|unknown)$",
    ),
    sort: str = Query("peak_time"),
) -> list[dict]:
    return loaders.event_summaries(band=band, region=region, sort=sort)


@app.get("/api/events/{event_id}")
def event_detail(event_id: int) -> dict:
    detail = loaders.event_detail(event_id)
    if detail is None:
        raise HTTPException(404, f"event {event_id} not found")
    return detail


@app.get("/api/events/{event_id}/waveform")
def event_waveform(
    event_id: int,
    hours: float = Query(12.0, gt=0.5, le=18.0),
) -> dict:
    wf = loaders.event_waveform(event_id, hours_pad=hours)
    if wf is None:
        raise HTTPException(404, f"waveform unavailable for event {event_id}")
    return wf


@app.get("/api/events/{event_id}/spectrum")
def event_spectrum(
    event_id: int,
    hours: float = Query(12.0, gt=0.5, le=18.0),
) -> dict:
    spec = loaders.event_spectrum(event_id, hours_pad=hours)
    if spec is None:
        raise HTTPException(404, f"spectrum unavailable for event {event_id}")
    return spec


@app.get("/api/timeline")
def timeline() -> list[dict]:
    return loaders.timeline_summary()


@app.get("/api/regions")
def regions() -> list[dict]:
    return loaders.region_intervals()


@app.get("/api/synthetic/generate")
def synthetic_generate(
    band: str = Query("QP60", pattern="^(QP30|QP60|QP120)$"),
    amp: float = Query(2.0, gt=0.0, le=20.0),
    period_min: float | None = Query(None, gt=1.0, le=600.0),
    decay_h: float = Query(4.0, gt=0.1, le=24.0),
    noise: float = Query(0.3, ge=0.0, le=5.0),
    seed: int = Query(0, ge=0),
) -> dict:
    return synthetic.generate(
        band=band,
        amplitude=amp,
        period_min=period_min,
        decay_h=decay_h,
        noise_sigma=noise,
        seed=seed,
    )


@app.get("/api/synthetic/benchmark")
def synthetic_benchmark(
    preset: str = Query("med_snr", pattern="^(low_snr|med_snr|high_snr)$"),
) -> dict:
    return synthetic.benchmark(preset=preset)


@app.get("/api/synthetic/benchmark/event")
def synthetic_benchmark_event(
    preset: str = Query("med_snr", pattern="^(low_snr|med_snr|high_snr)$"),
    band: str = Query(..., pattern="^(QP30|QP60|QP120)$"),
    amp: float = Query(..., gt=0.0, le=20.0),
    seed: int = Query(0, ge=0),
) -> dict:
    return synthetic.benchmark_event(
        preset=preset, band=band, amplitude=amp, seed=seed,
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
