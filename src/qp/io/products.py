"""Load pre-computed DataProducts .npy files.

These are large numpy arrays produced by data_sweeper.py and
boundary_crossings.py from raw PDS data. For most figure reproduction,
start here rather than reprocessing from raw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import qp


def _load(name: str, products_dir: Path | None = None) -> np.ndarray:
    """Load a .npy file from DataProducts with allow_pickle=True."""
    d = products_dir or qp.DATA_PRODUCTS
    return np.load(d / name, allow_pickle=True)


def load_crossings(products_dir: Path | None = None) -> np.ndarray:
    """Load magnetosphere/magnetosheath/solar wind boundary crossings.

    Returns shape (3, N) object array:
        row 0: datetime objects (hourly timestamps)
        row 1: location codes (0=MS, 1=SH, 2=SW, 9=unknown)
        row 2: crossing type strings ('BSI','BSO','MPI','MPO') or None
    """
    return _load("CROSSINGS.npy", products_dir)


def load_spacecraft_position(products_dir: Path | None = None) -> np.ndarray:
    """Load Cassini position time series in KSM coordinates.

    Returns shape (N, 11) object array. Columns:
        0: datetime
        1-4: Bx, By, Bz, Btot (nT)  [or field components]
        5-7: x, y, z (R_S) in KSM
        8-10: location flags
    """
    return _load("CassiniLocation_KSM.npy", products_dir)


def load_mag_segments(
    coord_system: str = "KSM",
    segment_hours: int = 36,
    products_dir: Path | None = None,
) -> np.ndarray:
    """Load pre-computed MAG data segments.

    Parameters
    ----------
    coord_system : str
        'KSM', 'KRTP', or 'MFA'.
    segment_hours : int
        Segment length: 24 or 36.

    Returns the full numpy array (typically 800+ MB).
    """
    names = {
        ("KSM", 36): "Cassini_MAG_KSM_36H.npy",
        ("KRTP", 36): "Cassini_MAG_KRTP_36H.npy",
        ("KRTP", 24): "Cassini_MAG_KRTP_24H.npy",
        ("MFA", 36): "Cassini_MAG_MFA_36H.npy",
    }
    key = (coord_system.upper(), segment_hours)
    if key not in names:
        raise ValueError(
            f"No pre-computed segments for {coord_system} {segment_hours}H"
        )
    return _load(names[key], products_dir)


def load_mag_metadata(
    coord_system: str = "KRTP",
    segment_hours: int = 36,
    products_dir: Path | None = None,
) -> np.ndarray:
    """Load metadata for MAG segments (positions, local times, etc.)."""
    names = {
        ("KRTP", 36): "Cassini_MAG_META_KRTP_36H.npy",
    }
    key = (coord_system.upper(), segment_hours)
    if key not in names:
        raise ValueError(f"No metadata for {coord_system} {segment_hours}H")
    return _load(names[key], products_dir)


def load_sls5(products_dir: Path | None = None) -> np.ndarray:
    """Load Saturn Longitude System 5 (PPO phase reference)."""
    return _load("SLS5_2004-2018.npy", products_dir)


def load_scas_times(products_dir: Path | None = None) -> np.ndarray:
    """Load spacecraft calibration roll times.

    Note: this file contains pickled custom objects (Interval class).
    May fail on Python versions where the class isn't defined.
    """
    return _load("SCAS_TIMES.npy", products_dir)
