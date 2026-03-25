"""Generic filtering and binning operations for event/segment collections.

Extracts pure data operations from ``cassinilib/PlotFFT.py`` (lines 816-975).
These are generic — they work on any sequence of objects with attribute access.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


def filter_by_property(
    items: Sequence[Any],
    key: str | Callable[[Any], float],
    min_val: float,
    max_val: float,
) -> list[Any]:
    r"""Filter items where a property falls within [min_val, max_val).

    Replaces ``cassinilib/PlotFFT.py:filterByProperties()``.

    Parameters
    ----------
    items : sequence
        Objects to filter.
    key : str or callable
        If a string, ``getattr(item, key)`` is used. If callable, called
        on each item to extract the value.
    min_val, max_val : float
        Inclusive lower bound, exclusive upper bound.

    Returns
    -------
    list
        Filtered items.
    """
    getter = _make_getter(key)
    return [item for item in items if min_val <= getter(item) < max_val]


def filter_by_datetime(
    items: Sequence[Any],
    date_from: datetime.datetime,
    date_to: datetime.datetime,
    key: str | Callable[[Any], datetime.datetime] = "date_from",
) -> list[Any]:
    r"""Filter items by datetime range.

    Replaces ``cassinilib/PlotFFT.py:selectByDatetime()``.

    Parameters
    ----------
    items : sequence
        Objects to filter.
    date_from, date_to : datetime
        Inclusive bounds.
    key : str or callable
        Attribute name or callable to extract the datetime from each item.

    Returns
    -------
    list
        Items within the datetime range.
    """
    getter = _make_getter(key)
    return [item for item in items if date_from <= getter(item) <= date_to]


def value_to_bin(
    value: float | np.ndarray,
    min_val: float,
    max_val: float,
    n_bins: int,
) -> int | np.ndarray:
    r"""Map a continuous value to a bin index.

    Replaces ``cassinilib/PlotFFT.py:value2bin()``.

    Parameters
    ----------
    value : float or ndarray
        Value(s) to bin.
    min_val, max_val : float
        Bin range.
    n_bins : int
        Number of bins.

    Returns
    -------
    int or ndarray
        Bin index/indices, clipped to [0, n_bins - 1].
    """
    scalar = np.isscalar(value)
    value = np.atleast_1d(np.asarray(value, dtype=float))
    bin_width = (max_val - min_val) / n_bins
    indices = np.floor((value - min_val) / bin_width).astype(int)
    indices = np.clip(indices, 0, n_bins - 1)
    return int(indices[0]) if scalar else indices


def bin_to_value(
    index: int | np.ndarray,
    min_val: float,
    max_val: float,
    n_bins: int,
) -> float | np.ndarray:
    r"""Map a bin index to the bin center value.

    Replaces ``cassinilib/PlotFFT.py:bin2value()``.

    Parameters
    ----------
    index : int or ndarray
        Bin index/indices.
    min_val, max_val : float
        Bin range.
    n_bins : int
        Number of bins.

    Returns
    -------
    float or ndarray
        Bin center value(s).
    """
    scalar = np.isscalar(index)
    index = np.atleast_1d(np.asarray(index, dtype=float))
    bin_width = (max_val - min_val) / n_bins
    values = index * bin_width + min_val + 0.5 * bin_width
    return float(values[0]) if scalar else values


def group_by_bins(
    items: Sequence[Any],
    key: str | Callable[[Any], float],
    min_val: float,
    max_val: float,
    n_bins: int,
) -> tuple[list[list[Any]], np.ndarray]:
    r"""Group items into bins based on a numeric property.

    Replaces ``cassinilib/PlotFFT.py:sortDataByBins()``.

    Parameters
    ----------
    items : sequence
        Objects to bin.
    key : str or callable
        Property to bin by.
    min_val, max_val : float
        Bin range.
    n_bins : int
        Number of bins.

    Returns
    -------
    bins : list[list]
        List of ``n_bins`` lists, each containing the items in that bin.
    bin_centers : ndarray
        Center values of each bin.
    """
    getter = _make_getter(key)
    bins: list[list[Any]] = [[] for _ in range(n_bins)]
    bin_width = (max_val - min_val) / n_bins
    bin_centers = np.linspace(
        min_val + 0.5 * bin_width,
        max_val - 0.5 * bin_width,
        n_bins,
    )

    for item in items:
        val = getter(item)
        idx = int(np.floor((val - min_val) / bin_width))
        if 0 <= idx < n_bins:
            bins[idx].append(item)

    return bins, bin_centers


# --- Internal helpers ---


def _make_getter(key: str | Callable) -> Callable:
    """Build an attribute-getter from a string or pass through a callable."""
    if callable(key):
        return key
    return lambda item: getattr(item, key)
