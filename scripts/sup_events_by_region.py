"""Supplementary figure + table: round-8 event characteristics by region.

For each of five region categories — inner magnetosphere (r <= 12 R_S), outer
magnetosphere (r > 12), plasma sheet (outer-MS, |mag.lat| < 15 deg),
magnetosheath, and solar wind — summarize the round-8 detection statistics so
readers can see whether the detector finds qualitatively similar or
qualitatively different waves outside the closed-field region.

Outputs (overwriting):
    Output/figures/sup_events_by_region.png
    Output/figures/sup_events_by_region.csv
    Output/figures/sup_events_by_region.tex

Usage::

    uv run python scripts/sup_events_by_region.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

import matplotlib.pyplot as plt  # noqa: E402

import qp  # noqa: E402
from qp.plotting.style import use_paper_style  # noqa: E402

log = logging.getLogger(__name__)


# Group → color: blues for closed-field MS regions, warm colors for boundary/SW.
GROUP_COLORS = {
    "MS inner (r$\\leq$12)": "#7FC8FF",
    "MS outer (r$>$12)": "#3D9BD8",
    "plasma sheet": "#12d5ae",
    "magnetosheath": "#f29539",
    "solar wind": "#f26b59",
}


def build_groups(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split events into the five comparison categories (mutually exclusive
    except for the plasma-sheet subset, which is a slice of MS-outer).

    ``mag_lat`` in the parquet is signed magnetic latitude in degrees
    (range ~ +/-60 deg for Cassini), positive = north.
    """
    df = df.copy()
    df["abs_mlat"] = df["mag_lat"].abs()

    is_ms = df["region"] == "magnetosphere"
    return {
        "MS inner (r$\\leq$12)": df[is_ms & (df.r_distance <= 12)],
        "MS outer (r$>$12)": df[is_ms & (df.r_distance > 12)],
        "plasma sheet": df[is_ms & (df.r_distance > 12) & (df.abs_mlat < 15)],
        "magnetosheath": df[df.region == "magnetosheath"],
        "solar wind": df[df.region == "solar_wind"],
    }


def summary_table(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Median characteristics per group, plus KS distance vs the inner-MS
    reference for the most discriminating fields."""
    ref = groups["MS inner (r$\\leq$12)"]
    a_perp_ref = np.hypot(ref.b_perp1_amp, ref.b_perp2_amp)
    ratio_ref = ref.b_par_amp / np.maximum(a_perp_ref, 1e-6)

    rows: list[dict[str, float | str | int]] = []
    for name, g in groups.items():
        a_perp = np.hypot(g.b_perp1_amp, g.b_perp2_amp).values
        a_par = g.b_par_amp.values
        ratio = a_par / np.maximum(a_perp, 1e-6)

        if name == "MS inner (r$\\leq$12)":
            ks_period = ks_mva = ks_ratio = (np.nan, np.nan)
        else:
            ks_period = stats.ks_2samp(ref.period_min.values, g.period_min.values)
            ks_mva = stats.ks_2samp(ref.mva_par_frac.values, g.mva_par_frac.values)
            ks_ratio = stats.ks_2samp(ratio_ref.values, ratio)

        rows.append(
            {
                "region": name,
                "n": len(g),
                "QP30 [%]": 100 * (g.band == "QP30").mean(),
                "QP60 [%]": 100 * (g.band == "QP60").mean(),
                "QP120 [%]": 100 * (g.band == "QP120").mean(),
                "period [min]": float(np.median(g.period_min)),
                "Q-factor": float(np.median(g.q_factor)),
                "MVA |e·b̂‖|²": float(np.median(g.mva_par_frac)),
                "Stokes d": float(np.median(g.stokes_d)),
                "duration [h]": float(np.median(g.duration_minutes)) / 60,
                "A⊥ [nT]": float(np.median(a_perp)),
                "A‖ [nT]": float(np.median(a_par)),
                "A‖/A⊥": float(np.median(ratio)),
                "|mag.lat|": float(np.median(g.abs_mlat)),
                "LT [h]": float(np.median(g.local_time)),
                "KS(period)": ks_period[0],
                "KS(MVA)": ks_mva[0],
                "KS(A‖/A⊥)": ks_ratio[0],
            }
        )
    return pd.DataFrame(rows).set_index("region")


def make_figure(groups: dict[str, pd.DataFrame], out_path: Path) -> None:
    use_paper_style()
    plt.rcParams["font.size"] = 12
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    def step_hist(ax, x_per_group, *, bins, xlim, xlabel, title, log10=False):
        for name, g in groups.items():
            x = x_per_group(g)
            if log10:
                x = np.log10(np.maximum(x, 1e-3))
            ax.hist(
                x,
                bins=bins,
                density=True,
                histtype="step",
                lw=1.7,
                color=GROUP_COLORS[name],
                label=f"{name} (n={len(g)})",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_xlim(xlim)
        ax.set_title(title)

    # 1. Period
    step_hist(
        axes[0, 0],
        lambda g: g.period_min.values,
        bins=np.linspace(15, 180, 50),
        xlim=(15, 180),
        xlabel="period  [min]",
        title="period distribution",
    )
    for p, c in zip((30, 60, 120), ("grey", "#FFB000", "#DC267F"), strict=True):
        axes[0, 0].axvline(p, color=c, ls=":", alpha=0.7)
    axes[0, 0].legend(fontsize=8, loc="upper right", framealpha=0.5)

    # 2. MVA major-axis transversality
    step_hist(
        axes[0, 1],
        lambda g: g.mva_par_frac.values,
        bins=np.linspace(0, 0.5, 50),
        xlim=(0, 0.5),
        xlabel=r"$|\hat e_\mathrm{max}\cdot\hat b_\parallel|^2$",
        title="MVA major-axis transversality",
    )
    axes[0, 1].axvline(0.5, color="white", ls="--", alpha=0.5, lw=0.8)

    # 3. A_par / A_perp
    step_hist(
        axes[0, 2],
        lambda g: g.b_par_amp.values / np.maximum(np.hypot(g.b_perp1_amp, g.b_perp2_amp).values, 1e-6),
        bins=np.linspace(-3, 1, 50),
        xlim=(-3, 1),
        xlabel=r"$\log_{10}(A_\parallel / A_\perp)$",
        title="compressional / transverse ratio",
        log10=True,
    )
    axes[0, 2].axvline(0, color="white", ls=":", alpha=0.5, lw=0.8)

    # 4. |mag. latitude|
    step_hist(
        axes[1, 0],
        lambda g: g.mag_lat.abs().values,
        bins=np.linspace(0, 70, 40),
        xlim=(0, 70),
        xlabel="|magnetic latitude|  [deg]",
        title="latitude distribution",
    )

    # 5. Stokes d
    step_hist(
        axes[1, 1],
        lambda g: g.stokes_d.values,
        bins=np.linspace(0.6, 1.0, 40),
        xlim=(0.6, 1.0),
        xlabel=r"Stokes degree of polarization $d$",
        title="polarization purity",
    )
    axes[1, 1].axvline(0.7, color="white", ls="--", alpha=0.5, lw=0.8)

    # 6. Q-factor
    step_hist(
        axes[1, 2],
        lambda g: g.q_factor.values,
        bins=np.linspace(2, 12, 40),
        xlim=(2, 12),
        xlabel=r"spectral $Q = f/\Delta f_{\mathrm{FWHM}}$",
        title="spectral narrowness",
    )
    axes[1, 2].axvline(3, color="white", ls="--", alpha=0.5, lw=0.8)

    fig.suptitle(
        "Round-8 events by region — distinguishing the closed-field "
        "magnetosphere from boundary regions",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    log.info("wrote %s", out_path)


def write_csv_and_latex(table: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    table.to_csv(csv_path, float_format="%.3g")
    log.info("wrote %s", csv_path)

    # LaTeX table — pick the most informative subset of columns
    cols = [
        "n",
        "QP60 [%]",
        "QP120 [%]",
        "period [min]",
        "MVA |e·b̂‖|²",
        "Stokes d",
        "A⊥ [nT]",
        "A‖/A⊥",
        "|mag.lat|",
        "KS(MVA)",
        "KS(A‖/A⊥)",
    ]
    relabel = {
        "MS inner (r$\\leq$12)": r"MS inner ($r\leq 12\,R_S$)",
        "MS outer (r$>$12)": r"MS outer ($r>12\,R_S$)",
        "plasma sheet": r"plasma sheet",
        "magnetosheath": r"magnetosheath",
        "solar wind": r"solar wind",
    }
    header_relabel = {
        "MVA |e·b̂‖|²": r"$|\hat e_\mathrm{max}\!\cdot\!\hat b_\parallel|^2$",
        "Stokes d": r"Stokes $d$",
        "A⊥ [nT]": r"$A_\perp$ [nT]",
        "A‖/A⊥": r"$A_\parallel/A_\perp$",
        "|mag.lat|": r"$|\lambda_m|$",
        "QP60 [%]": r"QP60 [\%]",
        "QP120 [%]": r"QP120 [\%]",
        "KS(MVA)": r"$D_\mathrm{KS}(\mathrm{MVA})$",
        "KS(A‖/A⊥)": r"$D_\mathrm{KS}(A_\parallel/A_\perp)$",
    }

    int_cols = {"n"}

    def fmt(v: object, col: str) -> str:
        if col in int_cols:
            return f"{int(v):d}"  # type: ignore[arg-type]
        if isinstance(v, float):
            if not np.isfinite(v):
                return "--"
            return f"{v:.3g}"
        return str(v)

    header_cells = [header_relabel.get(c, c) for c in cols]
    rows_tex = [
        " & ".join(["Region"] + header_cells) + r" \\",
        r"\hline",
    ]
    for idx, row in table[cols].iterrows():
        cells = [relabel.get(idx, str(idx))] + [fmt(row[c], c) for c in cols]
        rows_tex.append(" & ".join(cells) + r" \\")

    caption = (
        "Round-8 event characteristics by region. Columns are medians, except "
        "$n$ and band fractions. KS columns give the two-sample "
        "Kolmogorov--Smirnov distance against the inner-magnetosphere reference "
        "($r\\leq 12\\,R_S$); values $\\gtrsim 0.4$ indicate qualitatively "
        "different distributions ($p\\ll 10^{-20}$). The compressional fraction "
        "$A_\\parallel/A_\\perp$ and MVA transversality "
        "$|\\hat e_\\mathrm{max}\\!\\cdot\\!\\hat b_\\parallel|^2$ separate inner-MS "
        "waves (FLR-like, transverse) from magnetosheath waves (compressional)."
    )
    full = "\n".join(
        [
            r"\begin{table}",
            r"\centering",
            r"\small",
            r"\caption{" + caption + "}",
            r"\label{tab:sup_events_by_region}",
            r"\begin{tabular}{l" + "r" * len(cols) + "}",
            *rows_tex,
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    tex_path.write_text(full)
    log.info("wrote %s", tex_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parquet = qp.OUTPUT_DIR / "events_round8.parquet"
    df = pd.read_parquet(parquet)
    log.info("loaded %d events from %s", len(df), parquet)

    # Cut the proximal-orbit pathology (events at r < 3 R_S have unphysical
    # >10 kT amplitudes from the rotating ambient dipole field).
    n_drop = int((df.r_distance <= 3.0).sum())
    df = df[df.r_distance > 3.0]
    log.info("dropped %d events at r <= 3 R_S; %d remain", n_drop, len(df))

    groups = build_groups(df)
    table = summary_table(groups)
    print(table.to_string(float_format=lambda x: f"{x:.3g}"))

    fig_dir = qp.OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    make_figure(groups, fig_dir / "sup_events_by_region.png")
    write_csv_and_latex(
        table,
        fig_dir / "sup_events_by_region.csv",
        fig_dir / "sup_events_by_region.tex",
    )


if __name__ == "__main__":
    main()
