r"""Physical constants for space physics analysis.

Replaces ``cassinilib/constants.py`` with ``scipy.constants`` values
per project convention.

Saturn-specific constants are kept here since they have no scipy source.
"""

from __future__ import annotations

from scipy.constants import electron_mass as ELECTRON_MASS  # noqa: F401
from scipy.constants import elementary_charge as ELEMENTARY_CHARGE  # noqa: F401
from scipy.constants import proton_mass as PROTON_MASS  # noqa: F401

# Time conversion factors (exact values)
SEC_PER_MIN: float = 60.0
SEC_PER_HOUR: float = 3_600.0
SEC_PER_DAY: float = 86_400.0
SEC_PER_YEAR: float = 365.0 * SEC_PER_DAY  # Julian year approximation

# Saturn
SATURN_RADIUS: float = 60_268e3  # equatorial radius (m)
