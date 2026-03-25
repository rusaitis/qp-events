"""Result types for eigenfrequency computations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EigenMode:
    """A single standing wave eigenmode.

    Attributes
    ----------
    angular_frequency : float
        Eigenfrequency in rad/s.
    mode_number : int
        Harmonic mode number (1 = fundamental, 2 = second harmonic, ...).
    eigenfunction : ndarray or None
        Displacement y(s) along the field line.
    eigenfunction_derivative : ndarray or None
        dy/ds along the field line.
    arc_length : ndarray or None
        Arc-length coordinate s for the eigenfunction.
    """

    angular_frequency: float
    mode_number: int
    eigenfunction: np.ndarray | None = None
    eigenfunction_derivative: np.ndarray | None = None
    arc_length: np.ndarray | None = None

    @property
    def frequency_mhz(self) -> float:
        """Frequency in millihertz."""
        return self.angular_frequency / (2.0 * np.pi) * 1e3

    @property
    def period_minutes(self) -> float:
        """Period in minutes."""
        if self.angular_frequency <= 0:
            return np.inf
        return 2.0 * np.pi / self.angular_frequency / 60.0


@dataclass(frozen=True, slots=True)
class EigenResult:
    """Complete result of an eigenfrequency computation for one field line.

    Attributes
    ----------
    modes : list[EigenMode]
        Found eigenmodes, sorted by frequency.
    l_shell : float
        Equatorial crossing distance in R_S.
    conjugate_latitude : float
        Invariant latitude at the planet surface (degrees).
    component : str
        Wave component ('toroidal' or 'poloidal').
    arc_length : ndarray or None
        Arc-length grid along the field line (R_S).
    alfven_velocity : ndarray or None
        v_A profile (m/s).
    field_magnitude : ndarray or None
        |B| profile (nT).
    density : ndarray or None
        Number density profile (m⁻³).
    """

    modes: list[EigenMode]
    l_shell: float
    conjugate_latitude: float
    component: str
    arc_length: np.ndarray | None = None
    alfven_velocity: np.ndarray | None = None
    field_magnitude: np.ndarray | None = None
    density: np.ndarray | None = None

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    @property
    def angular_frequencies(self) -> np.ndarray:
        """All eigenfrequencies as an array (rad/s)."""
        return np.array([m.angular_frequency for m in self.modes])

    @property
    def frequencies_mhz(self) -> np.ndarray:
        """All frequencies in millihertz."""
        return np.array([m.frequency_mhz for m in self.modes])

    @property
    def periods_minutes(self) -> np.ndarray:
        """All periods in minutes."""
        return np.array([m.period_minutes for m in self.modes])

    @property
    def mode_numbers(self) -> list[int]:
        return [m.mode_number for m in self.modes]
