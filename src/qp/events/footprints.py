r"""Per-event sparse spatial footprints for interactive filtering.

Each round-8 wave event has a detection window
``[date_from, date_to]`` of typically 2-6 hours. A "footprint" is the
sparse contribution of that single event to the canonical dwell-grid
denominator: which ``(r, mag_lat, LT)`` cells the spacecraft visited
during the event, and how many minutes it spent in each. The same
sparse representation extends to KMAG-traced axes
``(kmag_inv_lat, LT)`` (closed lines only) and
``(kmag_eq_r, LT)`` (closed lines only, i.e. equatorial apex).

Storage is CSR-style:

- ``offsets`` (N+1,) — per-event slice boundaries.
- ``bin_indices`` (M,) int32 — flat ravel indices into the grid.
- ``weights_min`` (M,) float32 — minutes contributed.

Reconstructing the unfiltered map is::

    np.bincount(bin_indices, weights=weights_min,
                minlength=grid.size).reshape(grid_shape)

Filtering by an arbitrary boolean event mask is::

    event_idx = np.repeat(np.arange(N), np.diff(offsets))
    sel = mask[event_idx]
    np.bincount(bin_indices[sel], weights=weights_min[sel],
                minlength=grid.size).reshape(grid_shape)

Three grids are supported, all sharing bin edges with
:class:`qp.dwell.grid.DwellGridConfig` so the per-event sum can be
divided directly by the canonical dwell denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from qp.dwell.grid import DwellGridConfig

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


GRID_NAMES: tuple[str, ...] = ("g3d", "g_kmag_inv_lat", "g_l_eq")
SCHEMA_VERSION: str = "footprints.1"


def grid_shape(name: str, config: DwellGridConfig) -> tuple[int, ...]:
    """Return the per-grid bin shape used by the footprint layout."""
    if name == "g3d":
        return config.shape
    if name == "g_kmag_inv_lat":
        return (config.n_lat, config.n_lt)
    if name == "g_l_eq":
        return (config.n_r, config.n_lt)
    raise ValueError(f"unknown grid name: {name!r}")


@dataclass(frozen=True, slots=True)
class SparseGrid:
    """CSR view of one grid's per-event contributions."""

    offsets: NDArray[np.int64]  # shape (N + 1,)
    bin_indices: NDArray[np.int32]  # shape (M,)
    weights_min: NDArray[np.float32]  # shape (M,)
    shape: tuple[int, ...]  # grid bin shape


@dataclass(frozen=True, slots=True)
class EventFootprints:
    """Footprints for every event in the source parquet, all grids."""

    event_ids: NDArray[np.int64]
    grids: dict[str, SparseGrid] = field(default_factory=dict)
    config: DwellGridConfig = field(default_factory=DwellGridConfig)
    schema_version: str = SCHEMA_VERSION

    def total(
        self,
        grid_name: str,
        mask: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float64]:
        """Sum per-event contributions; optionally subset to a boolean event mask.

        Returns minutes, shape matches the named grid. ``mask`` has
        length ``N == len(event_ids)``.
        """
        g = self.grids[grid_name]
        if mask is None:
            bins = g.bin_indices
            weights = g.weights_min
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.size != self.event_ids.size:
                raise ValueError(
                    f"mask length {mask.size} != n_events {self.event_ids.size}",
                )
            event_idx = np.repeat(
                np.arange(self.event_ids.size, dtype=np.int64),
                np.diff(g.offsets),
            )
            sel = mask[event_idx]
            bins = g.bin_indices[sel]
            weights = g.weights_min[sel]
        n_bins = int(np.prod(g.shape))
        out = np.bincount(bins, weights=weights, minlength=n_bins)
        return out.reshape(g.shape).astype(np.float64)


def build_sparse_grid(
    per_event_bins: list[NDArray[np.int32]],
    per_event_weights: list[NDArray[np.float32]],
    shape: tuple[int, ...],
) -> SparseGrid:
    """Pack per-event (unique-bin, weight) lists into CSR arrays."""
    if len(per_event_bins) != len(per_event_weights):
        raise ValueError("per_event_bins and per_event_weights length mismatch")
    n_events = len(per_event_bins)
    sizes = np.fromiter(
        (b.size for b in per_event_bins),
        dtype=np.int64,
        count=n_events,
    )
    offsets = np.empty(n_events + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])
    total = int(offsets[-1])
    bins = np.empty(total, dtype=np.int32)
    weights = np.empty(total, dtype=np.float32)
    for i, (b, w) in enumerate(zip(per_event_bins, per_event_weights, strict=False)):
        start, stop = offsets[i], offsets[i + 1]
        bins[start:stop] = b
        weights[start:stop] = w
    return SparseGrid(
        offsets=offsets,
        bin_indices=bins,
        weights_min=weights,
        shape=shape,
    )


def apply_filter(df: pd.DataFrame, expr: str | None) -> NDArray[np.bool_]:
    """Convert a pandas-eval expression into a boolean event mask.

    ``None``/empty expr returns an all-True mask. The expression is
    evaluated with ``DataFrame.eval`` so columns from the parquet are
    available as bare names (``q_factor``, ``band``, ``region``,
    ``l_eq_peak``, ...).
    """
    n = len(df)
    if not expr or expr.strip() == "":
        return np.ones(n, dtype=bool)
    out = df.eval(expr)
    arr = np.asarray(out)
    if arr.dtype != np.bool_:
        raise ValueError(
            f"filter expression {expr!r} must evaluate to bool, got dtype {arr.dtype}",
        )
    if arr.size != n:
        raise ValueError(
            f"filter result has size {arr.size} != n_events {n}",
        )
    return arr


def write_zarr(
    fp: EventFootprints,
    path: str,
    *,
    extra_attrs: dict | None = None,
) -> None:
    """Persist an :class:`EventFootprints` to a zarr store.

    Layout::

        <path>/
          event_ids                  (N,)
          {grid}/
            offsets                  (N+1,)
            bin_indices              (M_grid,)
            weights_min              (M_grid,)
            shape                    attrs
        attrs:
          schema_version, grid_config, ...
    """
    from typing import Any, cast

    import zarr

    root = cast(Any, zarr.open(path, mode="w"))

    def _store(group: Any, name: str, data: np.ndarray) -> None:
        arr = group.create_array(name, shape=data.shape, dtype=data.dtype)
        arr[...] = data

    _store(root, "event_ids", fp.event_ids)
    for name in GRID_NAMES:
        if name not in fp.grids:
            continue
        g = fp.grids[name]
        grp = root.create_group(name)
        _store(grp, "offsets", g.offsets)
        _store(grp, "bin_indices", g.bin_indices)
        _store(grp, "weights_min", g.weights_min)
        grp.attrs["shape"] = list(g.shape)
    root.attrs["schema_version"] = fp.schema_version
    root.attrs["n_events"] = int(fp.event_ids.size)
    root.attrs["grid_config"] = {
        "n_r": fp.config.n_r,
        "n_lat": fp.config.n_lat,
        "n_lt": fp.config.n_lt,
        "r_range": list(fp.config.r_range),
        "lat_range": list(fp.config.lat_range),
        "lt_range": list(fp.config.lt_range),
    }
    for k, v in (extra_attrs or {}).items():
        root.attrs[k] = v


def read_zarr(path: str) -> EventFootprints:
    """Inverse of :func:`write_zarr`."""
    from typing import Any, cast

    import zarr

    root = cast(Any, zarr.open(path, mode="r"))
    event_ids = np.asarray(root["event_ids"], dtype=np.int64)
    schema_version = str(root.attrs.get("schema_version", "unknown"))
    gc = dict(root.attrs.get("grid_config", {}))
    config = DwellGridConfig(
        n_r=int(gc.get("n_r", 100)),
        n_lat=int(gc.get("n_lat", 180)),
        n_lt=int(gc.get("n_lt", 96)),
        r_range=tuple(gc.get("r_range", (0.0, 100.0))),  # type: ignore[arg-type]
        lat_range=tuple(gc.get("lat_range", (-90.0, 90.0))),  # type: ignore[arg-type]
        lt_range=tuple(gc.get("lt_range", (0.0, 24.0))),  # type: ignore[arg-type]
    )
    grids: dict[str, SparseGrid] = {}
    for name in GRID_NAMES:
        if name not in root:
            continue
        grp = root[name]
        shape = tuple(int(s) for s in grp.attrs["shape"])
        grids[name] = SparseGrid(
            offsets=np.asarray(grp["offsets"], dtype=np.int64),
            bin_indices=np.asarray(grp["bin_indices"], dtype=np.int32),
            weights_min=np.asarray(grp["weights_min"], dtype=np.float32),
            shape=shape,
        )
    return EventFootprints(
        event_ids=event_ids,
        grids=grids,
        config=config,
        schema_version=schema_version,
    )
