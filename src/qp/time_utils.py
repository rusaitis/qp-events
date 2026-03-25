"""Datetime/timestamp conversions for Cassini mission data.

Replaces ``cassinilib/DatetimeFunctions.py`` with timezone-aware functions
that avoid the deprecated ``datetime.utcfromtimestamp()`` API.
"""

from __future__ import annotations

import datetime

_UTC = datetime.timezone.utc


def to_timestamp(dt: datetime.datetime) -> float:
    r"""Convert a datetime to a POSIX timestamp (seconds since 1970-01-01 UTC).

    Parameters
    ----------
    dt : datetime.datetime
        Input datetime. If naive (no tzinfo), it is assumed to be UTC.

    Returns
    -------
    float
        Seconds since the Unix epoch.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_UTC)
    return dt.timestamp()


def from_timestamp(ts: float) -> datetime.datetime:
    r"""Convert a POSIX timestamp to a UTC datetime.

    Parameters
    ----------
    ts : float
        Seconds since the Unix epoch.

    Returns
    -------
    datetime.datetime
        Timezone-aware UTC datetime.
    """
    return datetime.datetime.fromtimestamp(ts, tz=_UTC)


def parse_datetime(
    s: str,
    fmt: str = "%Y-%m-%dT%H:%M:%S",
) -> datetime.datetime:
    r"""Parse a datetime string.

    Parameters
    ----------
    s : str
        Datetime string to parse.
    fmt : str
        ``strptime`` format string.

    Returns
    -------
    datetime.datetime
        Parsed datetime (naive, no timezone).
    """
    return datetime.datetime.strptime(s, fmt)
