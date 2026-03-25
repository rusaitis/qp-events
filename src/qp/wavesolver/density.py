r"""Plasma density models for Saturn's magnetosphere.

Each model provides:
- Equatorial number density $n_{eq}(L)$ in m⁻³
- Field-aligned density profile $n(s)$ as a function of arc length
- Scale height $H(L)$ in Saturn radii

Models
------
- **BagenalDelamere**: Bagenal & Delamere (2011), water-group ions from
  Cassini CAPS. Primary model for the QP paper.
- **PersoonEtAl**: Persoon et al. (2013), inner magnetosphere electron density.
- **PowerLawDensity**: Cummings & Coleman (1969), $n = n_0 (L/r)^m$.
  Used for Earth and analytical tests.
- **UniformDensity**: Constant density, for testing.

All models conform to the ``DensityModel`` protocol and can be passed
to the wavesolver interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.constants import atomic_mass as AMU
from scipy.constants import mu_0 as MU0
from scipy.constants import speed_of_light as SPEED_OF_LIGHT
from scipy.interpolate import CubicSpline

SATURN_RADIUS = 60268e3  # Saturn equatorial radius (m) — no scipy constant

# ============================================================================
# Digitized observational data
# ============================================================================

# Bagenal & Delamere (2011), Figure 2: water-group (W+) densities
# First array: radial distance (R_S), second: density (cm⁻³)
_BAGENAL_N_DATA = np.array(
    [
        [
            3.01,
            3.13,
            3.24,
            3.35,
            3.45,
            3.52,
            3.58,
            3.67,
            3.78,
            3.96,
            4.14,
            4.34,
            4.54,
            4.74,
            4.92,
            5.10,
            5.32,
            5.49,
            5.69,
            5.90,
            6.09,
            6.28,
            6.48,
            6.64,
            6.91,
            7.06,
            7.26,
            7.41,
            7.62,
            7.81,
            7.97,
            8.26,
            8.54,
            8.81,
            9.19,
            9.56,
            9.88,
            10.22,
            10.57,
            10.90,
            11.24,
            11.50,
            11.69,
            11.95,
            12.29,
            12.61,
            12.78,
            13.15,
            13.56,
            13.90,
            14.34,
            14.62,
            15.00,
            15.51,
            16.04,
            16.82,
            17.30,
            17.89,
            18.66,
            19.19,
            19.68,
        ],
        [
            17.56,
            21.54,
            26.05,
            31.97,
            39.81,
            52.56,
            63.56,
            77.99,
            86.40,
            86.40,
            85.14,
            82.69,
            81.49,
            79.14,
            76.86,
            71.44,
            65.44,
            59.08,
            53.33,
            48.14,
            42.83,
            38.10,
            34.90,
            31.97,
            30.16,
            25.68,
            20.92,
            18.08,
            15.17,
            12.73,
            10.53,
            9.23,
            8.21,
            6.50,
            5.61,
            4.71,
            3.84,
            3.08,
            2.48,
            1.96,
            1.62,
            1.34,
            1.09,
            0.93,
            0.75,
            0.62,
            0.51,
            0.43,
            0.39,
            0.33,
            0.28,
            0.24,
            0.20,
            0.17,
            0.13,
            0.12,
            0.11,
            0.10,
            0.09,
            0.08,
            0.07,
        ],
    ]
)

# Bagenal & Delamere (2011), Figure 5: W+ scale height (R_S)
_BAGENAL_H_DATA = np.array(
    [
        [
            2.99,
            3.16,
            3.33,
            3.55,
            3.78,
            3.98,
            4.31,
            4.52,
            4.75,
            5.05,
            5.34,
            5.70,
            6.12,
            6.47,
            7.00,
            7.46,
            8.03,
            8.86,
            9.59,
            10.22,
            11.00,
            11.72,
            12.49,
            13.09,
            13.84,
            14.55,
            15.18,
            15.96,
            16.52,
            17.28,
            18.07,
            18.80,
            19.77,
        ],
        [
            0.22,
            0.24,
            0.26,
            0.27,
            0.31,
            0.31,
            0.39,
            0.48,
            0.56,
            0.64,
            0.74,
            0.87,
            1.04,
            1.24,
            1.47,
            1.62,
            1.81,
            1.97,
            2.08,
            2.23,
            2.40,
            2.55,
            2.82,
            2.99,
            3.30,
            3.62,
            3.84,
            4.11,
            4.40,
            4.64,
            4.78,
            4.78,
            4.76,
        ],
    ]
)

# Persoon et al. (2013), Figure 3(b): equatorial electron density (cm⁻³)
_PERSOON_N_DATA = np.array(
    [
        [
            2.50,
            2.70,
            2.90,
            3.10,
            3.30,
            3.50,
            3.70,
            3.89,
            4.09,
            4.30,
            4.50,
            4.70,
            4.90,
            5.09,
            5.28,
            5.48,
            5.68,
            5.89,
            6.10,
            6.29,
            6.49,
            6.69,
            6.88,
            7.09,
            7.29,
            7.50,
            7.69,
            7.87,
            8.07,
            8.28,
            8.49,
            8.71,
            8.95,
            9.20,
            9.37,
            9.48,
            9.67,
            9.90,
            10.07,
        ],
        [
            17.27,
            14.99,
            21.85,
            31.83,
            36.09,
            45.25,
            50.88,
            60.66,
            65.98,
            71.16,
            73.00,
            72.43,
            68.35,
            62.38,
            55.06,
            47.01,
            42.19,
            36.93,
            32.87,
            29.26,
            25.82,
            22.41,
            19.62,
            17.03,
            15.03,
            13.27,
            12.00,
            10.68,
            9.59,
            8.60,
            7.85,
            7.02,
            6.56,
            6.18,
            6.12,
            5.22,
            4.49,
            4.35,
            3.97,
        ],
    ]
)

# Persoon et al. (2013), Figure 2: plasma scale height (R_S)
_PERSOON_H_DATA = np.array(
    [
        [
            2.710,
            2.777,
            2.844,
            2.900,
            2.927,
            2.962,
            2.988,
            3.014,
            3.041,
            3.068,
            3.104,
            3.181,
            3.261,
            3.319,
            3.356,
            3.393,
            3.423,
            3.453,
            3.488,
            3.534,
            3.577,
            3.619,
            3.660,
            3.700,
            3.782,
            3.867,
            3.917,
            3.952,
            3.987,
            4.022,
            4.061,
            4.107,
            4.164,
            4.212,
            4.267,
            4.323,
            4.376,
            4.420,
            4.479,
            4.534,
            4.604,
            4.685,
            4.756,
            4.857,
            4.942,
            5.023,
            5.089,
            5.128,
            5.167,
            5.210,
            5.258,
            5.312,
            5.404,
            5.508,
            5.637,
            5.729,
            5.801,
            5.887,
            6.044,
            6.152,
            6.221,
            6.284,
            6.352,
            6.410,
            6.465,
            6.510,
            6.552,
            6.586,
            6.640,
            6.700,
            6.838,
            6.905,
            7.065,
            7.237,
            7.414,
            7.602,
            7.722,
            7.799,
            7.851,
            7.868,
            7.970,
            7.993,
            8.021,
            8.093,
            8.209,
            8.320,
            8.431,
            8.491,
            8.549,
            8.628,
            8.703,
            8.850,
            8.899,
            8.945,
            8.985,
            9.027,
            9.061,
            9.100,
            9.204,
            9.256,
            9.272,
            9.374,
            9.422,
            9.445,
            9.470,
        ],
        [
            0.353,
            0.345,
            0.338,
            0.331,
            0.323,
            0.314,
            0.306,
            0.299,
            0.292,
            0.285,
            0.278,
            0.277,
            0.275,
            0.277,
            0.286,
            0.295,
            0.302,
            0.310,
            0.319,
            0.328,
            0.337,
            0.346,
            0.354,
            0.362,
            0.355,
            0.350,
            0.362,
            0.372,
            0.383,
            0.393,
            0.405,
            0.417,
            0.427,
            0.438,
            0.449,
            0.462,
            0.475,
            0.487,
            0.501,
            0.514,
            0.528,
            0.545,
            0.560,
            0.575,
            0.590,
            0.607,
            0.623,
            0.640,
            0.657,
            0.676,
            0.699,
            0.718,
            0.738,
            0.756,
            0.773,
            0.791,
            0.812,
            0.836,
            0.853,
            0.877,
            0.905,
            0.930,
            0.958,
            0.988,
            1.019,
            1.046,
            1.072,
            1.098,
            1.127,
            1.154,
            1.132,
            1.129,
            1.153,
            1.176,
            1.203,
            1.212,
            1.201,
            1.167,
            1.140,
            1.117,
            1.186,
            1.152,
            1.225,
            1.274,
            1.316,
            1.351,
            1.385,
            1.423,
            1.476,
            1.526,
            1.556,
            1.517,
            1.466,
            1.417,
            1.370,
            1.328,
            1.293,
            1.256,
            1.211,
            1.173,
            1.153,
            1.246,
            1.293,
            1.209,
            1.333,
        ],
    ]
)

# Precompute interpolants (module-level, built once)
_fn_bagenal = CubicSpline(_BAGENAL_N_DATA[0], _BAGENAL_N_DATA[1], extrapolate=True)
_fH_bagenal = CubicSpline(_BAGENAL_H_DATA[0], _BAGENAL_H_DATA[1], extrapolate=True)
_fn_persoon = CubicSpline(_PERSOON_N_DATA[0], _PERSOON_N_DATA[1], extrapolate=True)
_fH_persoon = CubicSpline(_PERSOON_H_DATA[0], _PERSOON_H_DATA[1], extrapolate=True)


# ============================================================================
# Protocol
# ============================================================================


class DensityModel(Protocol):
    """Protocol for plasma density models."""

    ion_mass_amu: float

    @property
    def ion_mass_kg(self) -> float: ...

    def equatorial_density(self, l_shell: float) -> float:
        """Number density at the magnetic equator (m⁻³)."""
        ...

    def scale_height(self, l_shell: float) -> float:
        """Plasma scale height (R_S)."""
        ...

    def field_aligned_density(
        self,
        l_shell: float,
        s: np.ndarray,
        s_equator: float,
    ) -> np.ndarray:
        """Density along a field line as function of arc length.

        Parameters
        ----------
        l_shell : float
            L-shell of the field line.
        s : ndarray
            Arc length along the field line (R_S).
        s_equator : float
            Arc length at the magnetic equator (R_S).
        """
        ...


# ============================================================================
# Density models
# ============================================================================


@dataclass(frozen=True, slots=True)
class BagenalDelamere:
    r"""Bagenal & Delamere (2011) water-group ion density.

    Equatorial density and scale height interpolated from digitized
    CAPS observations (Sittler et al. 2008, Thomsen et al. 2010).

    $$n(s) = n_{eq}(L) \cdot \exp\!\left(-\frac{(s - s_{eq})^2}{H(L)^2}\right)$$

    where $s$ is arc length along the field line in R_S.

    References
    ----------
    Bagenal, F. & Delamere, P.A. (2011), J. Geophys. Res., 116, A05209.
    """

    ion_mass_amu: float = 18.0

    @property
    def ion_mass_kg(self) -> float:
        return self.ion_mass_amu * AMU

    def equatorial_density(self, l_shell: float) -> float:
        """Equatorial W+ density in m⁻³."""
        n_cm3 = float(_fn_bagenal(np.clip(l_shell, 3.01, 19.68)))
        return max(n_cm3, 0.07) * 1e6  # cm⁻³ → m⁻³, floor at 0.07 cm⁻³

    def scale_height(self, l_shell: float) -> float:
        """Plasma scale height in R_S."""
        return float(_fH_bagenal(l_shell))

    def field_aligned_density(
        self,
        l_shell: float,
        s: np.ndarray,
        s_equator: float,
    ) -> np.ndarray:
        neq = self.equatorial_density(l_shell)
        H = self.scale_height(l_shell) * SATURN_RADIUS  # R_S → meters
        s_m = s * SATURN_RADIUS  # R_S → meters
        s_eq_m = s_equator * SATURN_RADIUS
        return neq * np.exp(-(((s_m - s_eq_m) / H) ** 2))


@dataclass(frozen=True, slots=True)
class PersoonEtAl:
    r"""Persoon et al. (2013) inner magnetosphere electron density.

    $$n(L, \lambda) = n_{eq}(L) \cdot \exp\!\left(
        -\frac{L^2 (1 - \cos^6\lambda)}{3 H(L)^2}\right)$$

    where $\lambda$ is magnetic latitude.

    For arc-length-based computation, uses the same Gaussian form as
    Bagenal but with Persoon's equatorial density and scale height.

    References
    ----------
    Persoon, A.M. et al. (2013), J. Geophys. Res., 118, 2970-2974.
    """

    ion_mass_amu: float = 18.0

    @property
    def ion_mass_kg(self) -> float:
        return self.ion_mass_amu * AMU

    def equatorial_density(self, l_shell: float) -> float:
        """Equatorial electron density in m⁻³."""
        return float(_fn_persoon(l_shell)) * 1e6

    def scale_height(self, l_shell: float) -> float:
        """Plasma scale height in R_S."""
        return float(_fH_persoon(l_shell))

    def field_aligned_density(
        self,
        l_shell: float,
        s: np.ndarray,
        s_equator: float,
    ) -> np.ndarray:
        neq = self.equatorial_density(l_shell)
        H = self.scale_height(l_shell) * SATURN_RADIUS
        s_m = s * SATURN_RADIUS
        s_eq_m = s_equator * SATURN_RADIUS
        return neq * np.exp(-(((s_m - s_eq_m) / H) ** 2))


@dataclass(frozen=True, slots=True)
class PowerLawDensity:
    r"""Power-law density: $n = n_0 (L/r)^m$.

    Cummings & Coleman (1969) model for Earth's plasmasphere.
    Used for analytical comparisons.

    References
    ----------
    Cummings, W.D., O'Sullivan, R.J. & Coleman, P.J. (1969),
    J. Geophys. Res., 74, 778.
    """

    n0: float = 1e6  # equatorial density (m⁻³)
    m_index: float = 6.0  # power-law exponent
    ion_mass_amu: float = 1.0  # proton for Earth

    @property
    def ion_mass_kg(self) -> float:
        return self.ion_mass_amu * AMU

    def equatorial_density(self, l_shell: float) -> float:
        return self.n0

    def scale_height(self, l_shell: float) -> float:
        return float("inf")  # not applicable for power law

    def field_aligned_density(
        self,
        l_shell: float,
        s: np.ndarray,
        s_equator: float,
    ) -> np.ndarray:
        # For power-law, density depends on r not s.
        # This is a simplified version that returns equatorial density.
        # The full computation needs r(s), handled in field_profile.py.
        return np.full_like(s, self.n0)


@dataclass(frozen=True, slots=True)
class UniformDensity:
    """Constant density everywhere. For testing."""

    n0: float = 1e6  # m⁻³
    ion_mass_amu: float = 18.0  # water group

    @property
    def ion_mass_kg(self) -> float:
        return self.ion_mass_amu * AMU

    def equatorial_density(self, l_shell: float) -> float:
        return self.n0

    def scale_height(self, l_shell: float) -> float:
        return float("inf")

    def field_aligned_density(
        self,
        l_shell: float,
        s: np.ndarray,
        s_equator: float,
    ) -> np.ndarray:
        return np.full_like(s, self.n0)


# ============================================================================
# Alfvén velocity
# ============================================================================


def alfven_velocity(
    B: np.ndarray,
    n: np.ndarray,
    ion_mass_kg: float,
) -> np.ndarray:
    r"""Compute Alfvén speed with relativistic correction.

    $$v_A = \frac{B}{\sqrt{\mu_0 \, n \, m_i}} \cdot
            \frac{c}{\sqrt{v_A^2 + c^2}}$$

    Parameters
    ----------
    B : ndarray
        Magnetic field magnitude (T).
    n : ndarray
        Number density (m⁻³).
    ion_mass_kg : float
        Ion mass (kg).

    Returns
    -------
    ndarray
        Alfvén speed (m/s), capped at speed of light.
    """
    n_safe = np.maximum(n, 1e-10)  # avoid division by zero
    va_raw = B / np.sqrt(MU0 * n_safe * ion_mass_kg)
    # Relativistic correction
    va = np.sqrt(va_raw**2 * SPEED_OF_LIGHT**2 / (va_raw**2 + SPEED_OF_LIGHT**2))
    return np.nan_to_num(va, nan=SPEED_OF_LIGHT, posinf=SPEED_OF_LIGHT)
