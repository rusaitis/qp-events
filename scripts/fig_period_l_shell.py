"""Phase 8.4 — Period vs L-shell validation of FLR interpretation.

For each event with quality > 0.3, computes L-shell from the spacecraft
KSM position (dipole formula: L = r / cos²(λ)) and plots period vs L
coloured by quality. FLR theory predicts:

- QP120 events cluster at larger L than QP30.
- Within each band, period increases with L.

Fits a power law T ∝ L^β per band and compares to wavesolver Fig 6.

Output: ``Output/figures/figure_period_l_shell.png``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from qp.plotting.style import use_paper_style  # noqa: E402

# ── Expected period centroids (min) per L-shell from wavesolver Fig 6 ─────────
# Very approximate: QP30 ↔ L~15-20, QP60 ↔ L~20-25, QP120 ↔ L~25-30
# (even harmonics m=6, 4, 2 of FLR)


def _l_shell(r: np.ndarray, coord_th_rad: np.ndarray) -> np.ndarray:
    """Dipole L-shell from r (R_S) and magnetic latitude (radians)."""
    cos2 = np.cos(coord_th_rad) ** 2
    cos2 = np.where(cos2 > 0.01, cos2, np.nan)
    return r / cos2


def main() -> None:
    cat_path = _PROJECT_ROOT / "Output" / "events_qp_v3.parquet"
    if not cat_path.exists():
        cat_path = _PROJECT_ROOT / "Output" / "events_qp_v2.parquet"
        print(f"  using v2 catalog: {cat_path}")
    df = pd.read_parquet(cat_path)
    quality_col = "quality_v3" if "quality_v3" in df.columns else "quality"

    # Compute L-shell
    r = df["coord_r"].values.astype(float)
    th = df["coord_th"].values.astype(float)
    df["l_shell_computed"] = _l_shell(r, th)

    # Filter to quality > 0.3 and valid L, period
    mask = (
        (df[quality_col].fillna(0) > 0.3)
        & (df["l_shell_computed"].between(1, 100))
        & (df["period_peak_min"].notna())
    )
    dq = df[mask].copy()
    print(f"Events with q>0.3 and valid L: {len(dq)}")
    for band in ["QP30", "QP60", "QP120"]:
        b = dq[dq.band == band]
        if len(b) < 2:
            continue
        print(f"  {band}: n={len(b)}, L median={b['l_shell_computed'].median():.1f}, "
              f"period median={b['period_peak_min'].median():.1f} min")

    use_paper_style()
    colors = {"QP30": "#4ecdc4", "QP60": "#ff6b6b", "QP120": "#ffd93d"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False,
                              constrained_layout=True)

    for ax, band in zip(axes, ["QP30", "QP60", "QP120"]):
        b = dq[dq.band == band]
        if len(b) < 2:
            ax.set_title(band)
            continue

        L = b["l_shell_computed"].values
        T = b["period_peak_min"].values
        q = b[quality_col].values

        sc = ax.scatter(L, T, c=q, cmap="viridis", s=8, alpha=0.7,
                         vmin=0.3, vmax=0.8, rasterized=True)
        plt.colorbar(sc, ax=ax, label="quality")

        # Power-law fit in log-log space
        finite = np.isfinite(L) & np.isfinite(T) & (L > 1) & (T > 0)
        if finite.sum() >= 5:
            logL = np.log10(L[finite])
            logT = np.log10(T[finite])
            coeffs = np.polyfit(logL, logT, deg=1)
            beta = coeffs[0]
            L_fit = np.linspace(L[finite].min(), L[finite].max(), 100)
            T_fit = 10 ** np.polyval(coeffs, np.log10(L_fit))
            ax.plot(L_fit, T_fit, color=colors[band], lw=2,
                     label=f"T ∝ L^{beta:.2f}")
            ax.legend(fontsize=9, frameon=False)
            print(f"  {band}: T ∝ L^{beta:.2f} (n={finite.sum()})")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("L-shell (dipole)")
        ax.set_ylabel("Period (min)")
        ax.set_title(band, fontsize=13)
        ax.set_xlim(5, 100)
        ax.grid(alpha=0.2, which="both")

    fig.suptitle(
        "Phase 8.4 — Period vs L-shell (q > 0.3, dipole formula)",
        fontsize=12,
    )
    out_dir = _PROJECT_ROOT / "Output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "figure_period_l_shell.png"
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
