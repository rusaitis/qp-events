"""Pure Python implementation of the KMAG Saturn magnetic field model.

Replaces the Fortran 77 KMAG2012.f with a direct Python/NumPy translation,
accelerated by numba JIT compilation (~164,000 field evaluations/sec).

The model computes Saturn's magnetospheric B-field as:

    B_total = B_internal + B_shielded_dipole + B_current_sheet + B_imf_penetration

References
----------
- Khurana, K.K. (2020), KMAG Saturn magnetospheric field model.
- Cao, H. et al. (2012), Saturn's high degree magnetic moments, Icarus, 221(1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qp.fieldline.kmag_coefficients import (
    CS_DHD,
    DP_NOMINAL,
    INTERNAL_G,
    INTERNAL_H,
    INTERNAL_NM,
    MP_A1,
    MP_A2,
)
from qp.fieldline.saturn_coords import (
    epoch_to_j2000,
    rotate_vector,
    sph2car_field,
)
from qp.fieldline._numba_kernels import (
    _build_rotation_matrices,
    field_s3c_kernel,
)


@dataclass(frozen=True, slots=True)
class SaturnFieldConfig:
    """Configuration for the KMAG Saturn magnetic field model."""

    dp: float = 0.017  # solar wind dynamic pressure (nPa)
    by_imf: float = -0.2  # IMF By component (nT, KSM)
    bz_imf: float = 0.1  # IMF Bz component (nT, KSM)
    epoch: str = "j2000"  # time epoch


class SaturnField:
    """Pure Python KMAG Saturn magnetic field model (Khurana 2020).

    All field computation is JIT-compiled via numba. The first call
    incurs a ~2s compilation cost; subsequent calls run at ~164k evals/sec.

    Parameters
    ----------
    config : SaturnFieldConfig, optional
        Model configuration. Uses defaults if not provided.

    Examples
    --------
    >>> field = SaturnField()
    >>> br, bt, bp = field.field_s3c(r=10.0, theta=1.5708, phi=0.0, time=0.0)
    """

    def __init__(self, config: SaturnFieldConfig | None = None) -> None:
        self.config = config or SaturnFieldConfig()
        self._init_internal_field()
        self._init_pressure_scaling()

    def _init_pressure_scaling(self) -> None:
        """Precompute pressure-dependent quantities."""
        r0_nominal = MP_A1 * DP_NOMINAL ** (-MP_A2)
        r0_actual = MP_A1 * self.config.dp ** (-MP_A2)
        self._r0or = r0_nominal / r0_actual
        self._cs_half_thickness = CS_DHD / self._r0or

    def _init_internal_field(self) -> None:
        """Precompute normalized spherical harmonic coefficients.

        Schmidt semi-normalization of the Gauss coefficients, matching
        the one-time initialization in Fortran KRONIAN_HIGHER.
        """
        g = INTERNAL_G.copy()
        h = INTERNAL_H.copy()
        rec = np.ones(92)

        for n in range(1, 14):
            n2 = 2 * n - 1
            n2 = n2 * (n2 - 2)
            for m in range(1, n + 1):
                mn = n * (n - 1) // 2 + m
                if mn < 92:
                    rec[mn] = (
                        float((n - m) * (n + m - 2)) / float(n2) if n2 != 0 else 0.0
                    )

        s = 1.0
        for n in range(2, 14):
            mn = n * (n - 1) // 2 + 1
            if mn >= 92:
                break
            s *= float(2 * n - 3) / float(n - 1)
            g[mn] *= s
            h[mn] *= s
            p = s
            for m in range(2, n + 1):
                aa = 2.0 if m == 2 else 1.0
                p *= math.sqrt(aa * float(n - m + 1) / float(n + m - 2))
                mnn = mn + m - 1
                if mnn < 92:
                    g[mnn] *= p
                    h[mnn] *= p

        self._g = g
        self._h = h
        self._rec = rec

    def field_s3c(
        self,
        r: float,
        theta: float,
        phi: float,
        time: float,
    ) -> tuple[float, float, float]:
        r"""Compute B-field in S3C spherical coordinates.

        Parameters
        ----------
        r : float
            Radial distance in Saturn radii (R_S).
        theta : float
            Colatitude in radians (0 at north pole).
        phi : float
            System III longitude in radians.
        time : float
            Time in the configured epoch.

        Returns
        -------
        br, btheta, bphi : float
            Magnetic field components in nT.
        """
        j2000_time = epoch_to_j2000(time, self.config.epoch)

        dis_to_s3c, s3c_to_dis, ksm_to_s3c, s3c_to_ksm, si = _build_rotation_matrices(
            j2000_time
        )

        br, bt, bp = field_s3c_kernel(
            r,
            theta,
            phi,
            self._g,
            self._h,
            self._rec,
            INTERNAL_NM,
            self._r0or,
            self._cs_half_thickness,
            dis_to_s3c,
            s3c_to_dis,
            ksm_to_s3c,
            s3c_to_ksm,
            si,
            self.config.by_imf,
            self.config.bz_imf,
            self.config.dp,
        )
        return br, bt, bp

    def field_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        time: float,
        coord: str = "KSM",
    ) -> tuple[float, float, float]:
        """Compute B-field in Cartesian coordinates.

        Parameters
        ----------
        x, y, z : float
            Position in R_S in the given coordinate system.
        time : float
            Time in the configured epoch.
        coord : str
            Coordinate system of input/output ('KSM', 'S3C', 'DIS', etc.).

        Returns
        -------
        bx, by, bz : float
            Field components in nT in the requested coord system.
        """
        j2000_time = epoch_to_j2000(time, self.config.epoch)

        pos_s3c = rotate_vector(np.array([x, y, z]), coord, "S3C", j2000_time)
        r = math.sqrt(pos_s3c[0] ** 2 + pos_s3c[1] ** 2 + pos_s3c[2] ** 2)
        rho = math.sqrt(pos_s3c[0] ** 2 + pos_s3c[1] ** 2)
        theta = math.atan2(rho, pos_s3c[2])
        phi = math.atan2(pos_s3c[1], pos_s3c[0])

        br, bt, bp = self.field_s3c(r, theta, phi, time)

        bx_s3c, by_s3c, bz_s3c = sph2car_field(br, bt, bp, theta, phi)

        b_out = rotate_vector(
            np.array([bx_s3c, by_s3c, bz_s3c]),
            "S3C",
            coord,
            j2000_time,
        )
        return float(b_out[0]), float(b_out[1]), float(b_out[2])


def make_kmag_field_func(
    field: SaturnField,
    time: float,
    coord: str = "KSM",
):
    r"""Return a closure ``f(pos) -> ndarray(3,)`` that evaluates KMAG at a fixed time.

    Within a single field-line trace, ``time`` is constant. The rotation
    matrices and field-model coefficients can therefore be precomputed
    once and reused for every step. This eliminates the per-call cost of
    :func:`field_cartesian` (rebuilding rotation matrices via the Python
    ``rotate_vector`` chain on every evaluation).

    The returned closure is a drop-in replacement for the callable produced
    by :func:`qp.fieldline.tracer.saturn_field_wrapper` and is ~3-5x faster
    per call. Field values agree with :meth:`SaturnField.field_cartesian`
    to numerical noise (verified by unit test).

    Parameters
    ----------
    field : SaturnField
        Configured field model.
    time : float
        Time in the model's epoch (J2000 seconds by default).
    coord : str, optional
        Coordinate system for input position and output field. Only
        ``"KSM"`` is fast-pathed; other systems fall back to
        :func:`saturn_field_wrapper`.

    Returns
    -------
    callable
        ``f(pos) -> ndarray(3,)`` that returns the field at ``pos`` (KSM,
        $R_S$) in nT.
    """
    from qp.fieldline.tracer import saturn_field_wrapper

    if coord != "KSM":
        return saturn_field_wrapper(field, time, coord)

    j2000_time = epoch_to_j2000(time, field.config.epoch)
    dis_to_s3c, s3c_to_dis, ksm_to_s3c, s3c_to_ksm, si = _build_rotation_matrices(
        j2000_time
    )

    # Bind everything as closure-locals (fastest variable access in Python)
    g = field._g
    h = field._h
    rec = field._rec
    r0or = field._r0or
    cs_thick = field._cs_half_thickness
    by_imf = field.config.by_imf
    bz_imf = field.config.bz_imf
    dp = field.config.dp

    # Pre-extract rotation matrix elements as scalar locals
    a00 = float(ksm_to_s3c[0, 0]); a01 = float(ksm_to_s3c[0, 1]); a02 = float(ksm_to_s3c[0, 2])
    a10 = float(ksm_to_s3c[1, 0]); a11 = float(ksm_to_s3c[1, 1]); a12 = float(ksm_to_s3c[1, 2])
    a20 = float(ksm_to_s3c[2, 0]); a21 = float(ksm_to_s3c[2, 1]); a22 = float(ksm_to_s3c[2, 2])
    b00 = float(s3c_to_ksm[0, 0]); b01 = float(s3c_to_ksm[0, 1]); b02 = float(s3c_to_ksm[0, 2])
    b10 = float(s3c_to_ksm[1, 0]); b11 = float(s3c_to_ksm[1, 1]); b12 = float(s3c_to_ksm[1, 2])
    b20 = float(s3c_to_ksm[2, 0]); b21 = float(s3c_to_ksm[2, 1]); b22 = float(s3c_to_ksm[2, 2])

    def f(position: np.ndarray) -> np.ndarray:
        x = position[0]
        y = position[1]
        z = position[2]
        # KSM -> S3C (manual matrix multiply, no allocation)
        x_s3c = a00 * x + a01 * y + a02 * z
        y_s3c = a10 * x + a11 * y + a12 * z
        z_s3c = a20 * x + a21 * y + a22 * z
        # Spherical conversion
        rho_sq = x_s3c * x_s3c + y_s3c * y_s3c
        r = math.sqrt(rho_sq + z_s3c * z_s3c)
        theta = math.atan2(math.sqrt(rho_sq), z_s3c)
        phi = math.atan2(y_s3c, x_s3c)
        # Field evaluation (jitted) with cached rotation matrices
        br, bt, bp = field_s3c_kernel(
            r, theta, phi, g, h, rec, INTERNAL_NM, r0or, cs_thick,
            dis_to_s3c, s3c_to_dis, ksm_to_s3c, s3c_to_ksm, si,
            by_imf, bz_imf, dp,
        )
        # Spherical -> S3C cartesian (inlined sph2car_field)
        st = math.sin(theta); ct = math.cos(theta)
        sp = math.sin(phi); cp = math.cos(phi)
        bx_s3c = br * st * cp + bt * ct * cp - bp * sp
        by_s3c = br * st * sp + bt * ct * sp + bp * cp
        bz_s3c = br * ct - bt * st
        # S3C -> KSM (manual matrix multiply)
        out = np.empty(3)
        out[0] = b00 * bx_s3c + b01 * by_s3c + b02 * bz_s3c
        out[1] = b10 * bx_s3c + b11 * by_s3c + b12 * bz_s3c
        out[2] = b20 * bx_s3c + b21 * by_s3c + b22 * bz_s3c
        return out

    return f
