"""Numba JIT-compiled kernels for the KMAG Saturn field model.

All functions are decorated with @njit for ahead-of-time compilation.
They operate on plain floats and numpy arrays — no Python objects.

The calling convention: SaturnField precomputes rotation matrices once
per time step and passes them into field_s3c_kernel, which dispatches
to all sub-kernels without returning to Python.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from qp.fieldline.kmag_coefficients import (
    CS_ABCD,
    CS_AMPLITUDES,
    CS_BETA,
    CS_DQL,
    CS_F,
    CS_PQ,
    CS_RH0,
    DIPOLE_B0X,
    DIPOLE_B0Y,
    DIPOLE_B0Z,
    IMF_E1,
    IMF_E2,
    JTIME1,
    KSUN_AA,
    KSUN_BB,
    KSUN_CC,
    MP_A1,
    MP_A2,
    MP_DK,
    OMEGA_ROT,
    OMEGA_Y,
    OMEGA_Z,
    SHIELD_A,
    SHIELD_B,
    YEAR_SECONDS,
    YR_SAT,
    ZTHETD,
)

_DEG2RAD = math.pi / 180.0

# ======================================================================
# Bessel functions J0, J1 (rational approximations from KMAG2012.f)
# ======================================================================


@njit(cache=True)
def _bessj0(x: float) -> float:
    """Bessel function J0 via rational approximation."""
    if abs(x) < 8.0:
        y = x * x
        num = 57568490574.0 + y * (
            -13362590354.0
            + y
            * (
                651619640.7
                + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))
            )
        )
        den = 57568490411.0 + y * (
            1029532985.0
            + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0)))
        )
        return num / den
    else:
        ax = abs(x)
        z = 8.0 / ax
        y = z * z
        xx = ax - 0.785398164
        p = 1.0 + y * (
            -0.1098628627e-2
            + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6))
        )
        q = -0.1562499995e-1 + y * (
            0.1430488765e-3
            + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * (-0.934945152e-7)))
        )
        return math.sqrt(0.636619772 / ax) * (math.cos(xx) * p - z * math.sin(xx) * q)


@njit(cache=True)
def _bessj1(x: float) -> float:
    """Bessel function J1 via rational approximation."""
    if abs(x) < 8.0:
        y = x * x
        num = x * (
            72362614232.0
            + y
            * (
                -7895059235.0
                + y
                * (
                    242396853.1
                    + y * (-2972611.439 + y * (15704.4826 + y * (-30.16036606)))
                )
            )
        )
        den = 144725228442.0 + y * (
            2300535178.0
            + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0)))
        )
        return num / den
    else:
        ax = abs(x)
        z = 8.0 / ax
        y = z * z
        xx = ax - 2.356194491
        p = 1.0 + y * (
            0.183105e-2
            + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6)))
        )
        q = 0.04687499995 + y * (
            -0.2002690873e-3
            + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6))
        )
        result = math.sqrt(0.636619772 / ax) * (math.cos(xx) * p - z * math.sin(xx) * q)
        return result if x > 0 else -result


# ======================================================================
# Coordinate transforms
# ======================================================================


@njit(cache=True)
def _sun_position(j2000_time: float) -> tuple[float, float, float, float]:
    """KSun ephemeris — sun and Zkso directions in S3C."""
    t = j2000_time - JTIME1
    omyt = OMEGA_Y * t
    t_over_yr = t / YEAR_SECONDS

    # Build basis vector and compute dot products inline
    stheta_deg = 0.0
    fphi_deg = 0.0
    zfphi_deg = 0.0

    cos_vals = np.empty(4)
    sin_vals = np.empty(4)
    for i in range(4):
        cos_vals[i] = math.cos((i + 1) * omyt)
        sin_vals[i] = math.sin((i + 1) * omyt)

    for i in range(4):
        stheta_deg += KSUN_AA[2 * i] * cos_vals[i] + KSUN_AA[2 * i + 1] * sin_vals[i]
        fphi_deg += KSUN_BB[2 * i] * cos_vals[i] + KSUN_BB[2 * i + 1] * sin_vals[i]
        zfphi_deg += KSUN_CC[2 * i] * cos_vals[i] + KSUN_CC[2 * i + 1] * sin_vals[i]

    stheta_deg += KSUN_AA[8] * t_over_yr**2 + KSUN_AA[9] * t_over_yr + KSUN_AA[10]
    fphi_deg += KSUN_BB[8] * t_over_yr**2 + KSUN_BB[9] * t_over_yr + KSUN_BB[10]
    zfphi_deg += KSUN_CC[8] * t_over_yr**2 + KSUN_CC[9] * t_over_yr + KSUN_CC[10]

    fphi_deg = (fphi_deg + t / YR_SAT * 360.0) % 360.0
    sphi_deg = (fphi_deg - t * OMEGA_ROT) % 360.0
    if sphi_deg < 0.0:
        sphi_deg += 360.0

    zfphi_deg = (zfphi_deg + t / YR_SAT * 360.0 + 180.0) % 360.0
    zphi_deg = (zfphi_deg - t * OMEGA_Z) % 360.0
    if zphi_deg < 0.0:
        zphi_deg += 360.0

    return (
        stheta_deg * _DEG2RAD,
        sphi_deg * _DEG2RAD,
        ZTHETD * _DEG2RAD,
        zphi_deg * _DEG2RAD,
    )


@njit(cache=True)
def _build_rotation_matrices(j2000_time: float):
    """Precompute all rotation matrices needed for one field evaluation.

    Returns (dis_to_s3c, s3c_to_dis, ksm_to_s3c, s3c_to_ksm, si)
    where si is the dipole tilt angle used by mapit.
    """
    stheta, sphi, _, _ = _sun_position(j2000_time)
    cs, ss = math.cos(stheta), math.sin(stheta)
    cp, sp = math.cos(sphi), math.sin(sphi)

    # Sun direction in S3C
    x_sun = np.array([cs * cp, cs * sp, ss])

    # DIS system: Z_DIS = [0,0,1] (dipole), Y = Z×X_sun (normalized), X = Y×Z
    y_dis = np.array([-x_sun[1], x_sun[0], 0.0])  # cross([0,0,1], x_sun)
    y_norm = math.sqrt(y_dis[0] ** 2 + y_dis[1] ** 2)
    y_dis[0] /= y_norm
    y_dis[1] /= y_norm
    x_dis = np.array([y_dis[1], -y_dis[0], 0.0])  # cross(y_dis, [0,0,1])

    # DIS axes as rows → this matrix takes S3C→DIS
    s3c_to_dis = np.empty((3, 3))
    s3c_to_dis[0, 0] = x_dis[0]
    s3c_to_dis[0, 1] = x_dis[1]
    s3c_to_dis[0, 2] = x_dis[2]
    s3c_to_dis[1, 0] = y_dis[0]
    s3c_to_dis[1, 1] = y_dis[1]
    s3c_to_dis[1, 2] = y_dis[2]
    s3c_to_dis[2, 0] = 0.0
    s3c_to_dis[2, 1] = 0.0
    s3c_to_dis[2, 2] = 1.0
    dis_to_s3c = s3c_to_dis.T.copy()

    # KSM system: X = sun, Y = dip×X (normalized), Z = X×Y
    y_ksm = np.array([-x_sun[1], x_sun[0], 0.0])  # cross([0,0,1], x_sun)
    y_norm_ksm = math.sqrt(y_ksm[0] ** 2 + y_ksm[1] ** 2)
    y_ksm[0] /= y_norm_ksm
    y_ksm[1] /= y_norm_ksm
    # Z_KSM = X × Y
    z_ksm = np.array(
        [
            x_sun[1] * y_ksm[2] - x_sun[2] * y_ksm[1],
            x_sun[2] * y_ksm[0] - x_sun[0] * y_ksm[2],
            x_sun[0] * y_ksm[1] - x_sun[1] * y_ksm[0],
        ]
    )

    s3c_to_ksm = np.empty((3, 3))
    s3c_to_ksm[0, 0] = x_sun[0]
    s3c_to_ksm[0, 1] = x_sun[1]
    s3c_to_ksm[0, 2] = x_sun[2]
    s3c_to_ksm[1, 0] = y_ksm[0]
    s3c_to_ksm[1, 1] = y_ksm[1]
    s3c_to_ksm[1, 2] = y_ksm[2]
    s3c_to_ksm[2, 0] = z_ksm[0]
    s3c_to_ksm[2, 1] = z_ksm[1]
    s3c_to_ksm[2, 2] = z_ksm[2]
    ksm_to_s3c = s3c_to_ksm.T.copy()

    # Dipole tilt: z-hat in KSM
    z_hat_ksm_2 = s3c_to_ksm[2, 2]  # dot(z_ksm_row, [0,0,1]) = z_ksm[2]
    z_hat_ksm_0 = s3c_to_ksm[0, 2]  # dot(x_sun_row, [0,0,1]) = x_sun[2]
    si = math.acos(max(-1.0, min(1.0, z_hat_ksm_2)))
    if z_hat_ksm_0 < 0.0:
        si = -si

    return dis_to_s3c, s3c_to_dis, ksm_to_s3c, s3c_to_ksm, si


@njit(cache=True)
def _mat_vec(mat, vec):
    """3x3 matrix × 3-vector."""
    return np.array(
        [
            mat[0, 0] * vec[0] + mat[0, 1] * vec[1] + mat[0, 2] * vec[2],
            mat[1, 0] * vec[0] + mat[1, 1] * vec[1] + mat[1, 2] * vec[2],
            mat[2, 0] * vec[0] + mat[2, 1] * vec[1] + mat[2, 2] * vec[2],
        ]
    )


@njit(cache=True)
def _car2sph_field(bx, by, bz, theta, phi):
    """Cartesian → spherical field components."""
    st, ct = math.sin(theta), math.cos(theta)
    sp, cp = math.sin(phi), math.cos(phi)
    br = bx * st * cp + by * st * sp + bz * ct
    bt = bx * ct * cp + by * ct * sp - bz * st
    bp = -bx * sp + by * cp
    return br, bt, bp


# ======================================================================
# Field components
# ======================================================================


@njit(cache=True)
def _internal_field(r, theta, phi, g, h, rec, nm):
    """Spherical harmonic internal field (KRONIAN_HIGHER)."""
    k = nm + 1
    pp = 1.0 / r
    a = np.empty(14)
    b = np.empty(14)
    p_val = pp
    for n in range(1, k + 1):
        p_val *= pp
        a[n] = p_val
        b[n] = p_val * n

    p = 1.0
    d = 0.0
    bbr = bbt = bbf = 0.0
    cf = math.cos(phi)
    sf = math.sin(phi)
    c = math.cos(theta)
    s = math.sin(theta)
    bk = s < 1.0e-5
    x_val = 0.0
    y_val = 1.0

    for m in range(1, k + 1):
        bm = m == 1
        if not bm:
            w = x_val
            x_val = w * cf + y_val * sf
            y_val = y_val * cf - w * sf

        q = p
        z_val = d
        bi = 0.0
        p2 = 0.0
        d2 = 0.0

        for n in range(m, k + 1):
            an = a[n]
            mn = n * (n - 1) // 2 + m
            if mn >= 92:
                break
            e = g[mn]
            hh = h[mn]
            w = e * y_val + hh * x_val
            if abs(p2) < 1.0e-38:
                p2 = 0.0
            if abs(q) < 1.0e-38:
                q = 0.0
            bbr += b[n] * w * q
            bbt -= an * w * z_val
            if not bm:
                qq = z_val if bk else q
                bi += an * (e * x_val - hh * y_val) * qq
            xk = rec[mn]
            dp_val = c * z_val - s * q - xk * d2
            pm = c * q - xk * p2
            d2 = z_val
            p2 = q
            z_val = dp_val
            q = pm

        d = s * d + c * p
        p = s * p
        if not bm:
            bi *= m - 1
            bbf += bi

    br = bbr
    bt = bbt
    if bk:
        bf = -bbf if c < 0.0 else bbf
    else:
        bf = bbf / s
    return br, bt, bf


@njit(cache=True)
def _dipole_field_tensor(b0x, b0y, b0z, x, y, z):
    """Dipole field tensor."""
    r2 = x * x + y * y + z * z
    r = math.sqrt(r2)
    r5 = r2 * r2 * r
    a11 = (3.0 * x * x - r2) / r5
    a12 = 3.0 * x * y / r5
    a13 = 3.0 * x * z / r5
    a22 = (3.0 * y * y - r2) / r5
    a23 = 3.0 * y * z / r5
    a33 = (3.0 * z * z - r2) / r5
    bx = a11 * b0x + a12 * b0y + a13 * b0z
    by = a12 * b0x + a22 * b0y + a23 * b0z
    bz = a13 * b0x + a23 * b0y + a33 * b0z
    return bx, by, bz


@njit(cache=True)
def _bessel_shielding(rho, phi, x, r0or):
    """8-mode Bessel magnetopause shielding."""
    bperpr = bperpf = bperpx = 0.0
    for k in range(8):
        b_scaled = SHIELD_B[k] / r0or
        rho_scaled = rho / b_scaled
        x_scaled = x / b_scaled
        exp_term = math.exp(x_scaled)
        j0_val = _bessj0(rho_scaled)
        j1_val = _bessj1(rho_scaled)
        j1_over_rho = j1_val / rho_scaled if abs(rho_scaled) > 1e-30 else 0.5
        coeff = SHIELD_A[k] * r0or * r0or * r0or
        bperpr += coeff * math.sin(phi) * exp_term * (j1_over_rho - j0_val)
        bperpf += coeff * (-math.cos(phi)) * exp_term * j1_over_rho
        bperpx += coeff * (-math.sin(phi)) * exp_term * j1_val
    return bperpr, bperpf, bperpx


@njit(cache=True)
def _shielded_dipole(x, y, z, r0or):
    """Dipole + Chapman-Ferraro shielding."""
    bxd, byd, bzd = _dipole_field_tensor(DIPOLE_B0X, DIPOLE_B0Y, DIPOLE_B0Z, x, y, z)
    if z == 0.0 and y == 0.0:
        phi_cyl = 0.0
    else:
        phi_cyl = math.atan2(z, y)
    rho = y * math.cos(phi_cyl) + z * math.sin(phi_cyl)
    brho1, bphi1, bx1 = _bessel_shielding(rho, phi_cyl, x, r0or)
    by1 = brho1 * math.cos(phi_cyl) - bphi1 * math.sin(phi_cyl)
    bz1 = brho1 * math.sin(phi_cyl) + bphi1 * math.cos(phi_cyl)
    return bxd - bx1, byd - by1, bzd - bz1


@njit(cache=True)
def _mapit_core(xdis, ydis, zdis, rh, si):
    """Current sheet mapping (core, without rotation matrix computation)."""
    rho_dis = math.sqrt(xdis * xdis + ydis * ydis)
    phi_lt = math.atan2(ydis, xdis)
    z_cs = (rho_dis - rh * math.tanh(rho_dis / rh)) * math.tan(-si)
    th_cs = math.atan2(z_cs, rho_dis)
    if xdis > 0.0:
        th_cs = -th_cs
    ct, st_cs = math.cos(th_cs), math.sin(th_cs)
    rhomap = rho_dis * ct + zdis * st_cs
    zmap = -rho_dis * st_cs + zdis * ct
    xmap = rhomap * math.cos(phi_lt)
    ymap = rhomap * math.sin(phi_lt)
    return xmap, ymap, zmap


@njit(cache=True)
def _mapped_field(x, y, z, bx, by, bz, rh, si):
    """Jacobian transform via finite differences of mapit."""
    dx = dy = dz_ = 0.01
    xpp, ypp, zpp = _mapit_core(x + dx, y, z, rh, si)
    xpm, ypm, zpm = _mapit_core(x - dx, y, z, rh, si)
    dxpdx = (xpp - xpm) / (2.0 * dx)
    dypdx = (ypp - ypm) / (2.0 * dx)
    dzpdx = (zpp - zpm) / (2.0 * dx)

    xpp, ypp, zpp = _mapit_core(x, y + dy, z, rh, si)
    xpm, ypm, zpm = _mapit_core(x, y - dy, z, rh, si)
    dxpdy = (xpp - xpm) / (2.0 * dy)
    dypdy = (ypp - ypm) / (2.0 * dy)
    dzpdy = (zpp - zpm) / (2.0 * dy)

    xpp, ypp, zpp = _mapit_core(x, y, z + dz_, rh, si)
    xpm, ypm, zpm = _mapit_core(x, y, z - dz_, rh, si)
    dxpdz = (xpp - xpm) / (2.0 * dz_)
    dypdz = (ypp - ypm) / (2.0 * dz_)
    dzpdz = (zpp - zpm) / (2.0 * dz_)

    txx = dypdy * dzpdz - dypdz * dzpdy
    txy = dxpdz * dzpdy - dxpdy * dzpdz
    txz = dxpdy * dypdz - dxpdz * dypdy
    tyx = dypdz * dzpdx - dypdx * dzpdz
    tyy = dxpdx * dzpdz - dxpdz * dzpdx
    tyz = dxpdz * dypdx - dxpdx * dypdz
    tzx = dypdx * dzpdy - dypdy * dzpdx
    tzy = dxpdy * dzpdx - dxpdx * dzpdy
    tzz = dxpdx * dypdy - dxpdy * dypdx

    bxmap = txx * bx + txy * by + txz * bz
    bymap = tyx * bx + tyy * by + tyz * bz
    bzmap = tzx * bx + tzy * by + tzz * bz
    return bxmap, bymap, bzmap


@njit(cache=True)
def _gradU(x, y, z, abcd, pq, r0or):
    """Gradient of shielding potential for one mode."""
    M = 5
    M2 = 10
    a_mat = abcd[:25].reshape(M, M).copy()
    p_arr = np.empty(M2)
    for i in range(M):
        p_arr[i] = pq[i] / r0or
        p_arr[i + M] = pq[i + M] / r0or

    bxm = bym = bzm = 0.0
    for i in range(M):
        for k in range(M, M2):
            j = k - M
            t1 = math.sqrt(1.0 / (p_arr[i] * p_arr[i]) + 1.0 / (p_arr[k] * p_arr[k]))
            t2 = math.exp(t1 * x)
            cos_y = math.cos(y / p_arr[i])
            sin_y = math.sin(y / p_arr[i])
            sin_z = math.sin(z / p_arr[k])
            cos_z = math.cos(z / p_arr[k])
            aij = a_mat[i, j]
            bxm += t1 * aij * t2 * cos_y * sin_z
            bym -= aij / p_arr[i] * t2 * sin_y * sin_z
            bzm += aij / p_arr[k] * t2 * cos_y * cos_z
    return -bxm, -bym, -bzm


@njit(cache=True)
def _imf_penetration(by_imf, bz_imf):
    """IMF penetration through magnetopause."""
    if by_imf == 0.0 and bz_imf == 0.0:
        return 0.0, 0.0
    theta = math.atan2(by_imf, bz_imf)
    if theta < 0.0:
        theta += 2.0 * math.pi
    pen = IMF_E1 + IMF_E2 * math.cos(theta / 2.0) ** 2
    return pen * by_imf, pen * bz_imf


@njit(cache=True)
def _is_inside_mp(r, x, dp):
    """Magnetopause check."""
    if r == 0.0:
        return True
    r0 = MP_A1 * dp ** (-MP_A2)
    cos_theta = x / r
    r_mp = r0 * (2.0 / (1.0 + cos_theta)) ** MP_DK
    return r_mp >= r


# ======================================================================
# Main kernel: complete field evaluation
# ======================================================================


@njit(cache=True)
def field_s3c_kernel(
    r,
    theta,
    phi,
    g,
    h,
    rec,
    nm,
    r0or,
    cs_half_thickness,
    dis_to_s3c,
    s3c_to_dis,
    ksm_to_s3c,
    s3c_to_ksm,
    si,
    by_imf,
    bz_imf,
    dp,
):
    """Complete KMAG field evaluation — fully JIT compiled.

    All rotation matrices and tilt angle are precomputed by the caller.
    """
    rh = CS_RH0 / r0or
    D = cs_half_thickness
    drho = 0.05
    dz = 0.05

    # 1. Internal field
    bri, bti, bpi = _internal_field(r, theta, phi, g, h, rec, nm)

    # 2. S3C Cartesian → DIS
    st, ct = math.sin(theta), math.cos(theta)
    sp, cp = math.sin(phi), math.cos(phi)
    pos_s3c = np.array([r * st * cp, r * st * sp, r * ct])
    pos_dis = _mat_vec(s3c_to_dis, pos_s3c)
    xdis, ydis, zdis = pos_dis[0], pos_dis[1], pos_dis[2]

    # 3. Map to current sheet coords
    xmap, ymap, zmap = _mapit_core(xdis, ydis, zdis, rh, si)
    rmap = math.sqrt(xmap * xmap + ymap * ymap + zmap * zmap)

    # 4. Magnetopause check
    is_inside = _is_inside_mp(rmap, xmap, dp)

    # 5. Shielded dipole
    bxd, byd, bzd = _shielded_dipole(xmap, ymap, zmap, r0or)

    # 6. Jacobian transform
    bxdmap, bydmap, bzdmap = _mapped_field(xmap, ymap, zmap, bxd, byd, bzd, rh, si)

    # 7. Rotate DIS → S3C → spherical
    b_dis = np.array([bxdmap, bydmap, bzdmap])
    b_s3c = _mat_vec(dis_to_s3c, b_dis)
    brd, btd, bpd = _car2sph_field(b_s3c[0], b_s3c[1], b_s3c[2], theta, phi)

    # 8. Current sheet field (5 modes)
    # Recompute DIS position and mapping for current sheet (same as steps 2-3)
    z_cs = zmap
    rhomag = math.sqrt(xmap * xmap + ymap * ymap)

    # Brho computation (finite difference in z)
    xrho = np.empty(5)
    for ll in range(5):
        zm_val = abs(z_cs - dz)
        if zm_val < D:
            zm_val = 0.5 * (zm_val * zm_val / D + D)
        zp_val = abs(z_cs + dz)
        if zp_val < D:
            zp_val = 0.5 * (zp_val * zp_val / D + D)
        xlpp = xlpm = 0.0
        for i in range(6):
            beta_s = CS_BETA[ll, i] / r0or
            s1p = math.sqrt((beta_s + zp_val) ** 2 + (rhomag + beta_s) ** 2)
            s2p = math.sqrt((beta_s + zp_val) ** 2 + (rhomag - beta_s) ** 2)
            tp = 2.0 * beta_s / (s1p + s2p)
            aap = tp * math.sqrt(1.0 - tp * tp) / (s1p * s2p)
            s1m = math.sqrt((beta_s + zm_val) ** 2 + (rhomag + beta_s) ** 2)
            s2m = math.sqrt((beta_s + zm_val) ** 2 + (rhomag - beta_s) ** 2)
            tm = 2.0 * beta_s / (s1m + s2m)
            aam = tm * math.sqrt(1.0 - tm * tm) / (s1m * s2m)
            xlpp += CS_F[ll, i] * aap * rhomag
            xlpm += CS_F[ll, i] * aam * rhomag
        xrho[ll] = -(xlpp - xlpm) / (2.0 * dz)

    # Bz computation (finite difference in rho)
    xz = np.empty(5)
    for ll in range(5):
        rhom = rhomag - drho
        rhop = rhomag + drho
        xi = abs(z_cs)
        if xi <= D:
            xi = 0.5 * (z_cs * z_cs / D + D)
        xlpp = xlpm = 0.0
        for i in range(6):
            beta_s = CS_BETA[ll, i] / r0or
            s1p = math.sqrt((beta_s + xi) ** 2 + (rhop + beta_s) ** 2)
            s2p = math.sqrt((beta_s + xi) ** 2 + (rhop - beta_s) ** 2)
            s1m = math.sqrt((beta_s + xi) ** 2 + (rhom + beta_s) ** 2)
            s2m = math.sqrt((beta_s + xi) ** 2 + (rhom - beta_s) ** 2)
            tp = 2.0 * beta_s / (s1p + s2p)
            tm = 2.0 * beta_s / (s1m + s2m)
            aap = tp * math.sqrt(1.0 - tp * tp) / (s1p * s2p)
            aam = tm * math.sqrt(1.0 - tm * tm) / (s1m * s2m)
            xlpp += CS_F[ll, i] * aap * rhop
            xlpm += CS_F[ll, i] * aam * rhom
        xz[ll] = (
            (rhop * xlpp - rhom * xlpm) / (2.0 * drho) / rhomag if rhomag > 0 else 0.0
        )

    # Combine modes with shielding
    bx_total = by_total = bz_total = 0.0
    phimap = math.atan2(ymap, xmap) if (ymap != 0.0 or xmap != 0.0) else 0.0

    for ll in range(5):
        bx1 = xrho[ll] * math.cos(phimap)
        by1 = xrho[ll] * math.sin(phimap)
        bz1 = xz[ll]
        bx2, by2, bz2 = _gradU(xmap, ymap, zmap, CS_ABCD[ll], CS_PQ[ll], r0or)
        scale = CS_AMPLITUDES[ll] * r0or**CS_DQL
        bx_total += scale * (bx1 / (r0or * r0or) - bx2 / r0or)
        by_total += scale * (by1 / (r0or * r0or) - by2 / r0or)
        bz_total += scale * (bz1 / (r0or * r0or) - bz2 / r0or)

    # Jacobian + rotate for current sheet
    bxdis_cs, bydis_cs, bzdis_cs = _mapped_field(
        xmap,
        ymap,
        zmap,
        bx_total,
        by_total,
        bz_total,
        rh,
        si,
    )
    b_cs_dis = np.array([bxdis_cs, bydis_cs, bzdis_cs])
    b_cs_s3c = _mat_vec(dis_to_s3c, b_cs_dis)
    brscs, btscs, bpscs = _car2sph_field(
        b_cs_s3c[0], b_cs_s3c[1], b_cs_s3c[2], theta, phi
    )

    # 9. IMF penetration
    by_p, bz_p = _imf_penetration(by_imf, bz_imf)
    b_imf_ksm = np.array([0.0, by_p, bz_p])
    b_imf_s3c = _mat_vec(ksm_to_s3c, b_imf_ksm)
    br_p, bt_p, bp_p = _car2sph_field(
        b_imf_s3c[0], b_imf_s3c[1], b_imf_s3c[2], theta, phi
    )

    # 10. Sum
    if is_inside:
        br = bri + brd + brscs + br_p
        bt = bti + btd + btscs + bt_p
        bp = bpi + bpd + bpscs + bp_p
    else:
        br = br_p
        bt = bt_p
        bp = bp_p

    return br, bt, bp
