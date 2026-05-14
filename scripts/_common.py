"""Shared CLI boilerplate for ``scripts/`` entry points.

Not part of the ``qp`` package — this module just keeps logging,
project-root resolution, and output-directory creation from being
copy-pasted across two dozen scripts. Import directly with
``from _common import …``; Python adds the script's directory to
``sys.path`` automatically when a script is executed by path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "Output"
FIGURES_DIR: Final[Path] = OUTPUT_DIR / "figures"
DIAGNOSTICS_DIR: Final[Path] = OUTPUT_DIR / "diagnostics"

DEFAULT_LOG_FMT: Final[str] = "%(message)s"
TIMESTAMPED_LOG_FMT: Final[str] = "%(asctime)s %(message)s"


def setup_logging(
    verbose: bool = True,
    *,
    fmt: str = DEFAULT_LOG_FMT,
) -> logging.Logger:
    """Configure root logging and return the root logger.

    Parameters
    ----------
    verbose
        ``True`` → INFO; ``False`` → WARNING. Maps directly onto the
        ``--verbose`` flag most scripts already accept.
    fmt
        Override for the formatter (use ``TIMESTAMPED_LOG_FMT`` for
        long-running diagnostics).
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format=fmt,
    )
    return logging.getLogger()


#: FLR-purity cut value used to produce the published round-8 figures.
#: Readers of the event-time zarr (Fig 7 & Fig 8) warn if the zarr was
#: produced with a looser cut, so default-pipeline outputs don't end up
#: silently rendered as the publication figure.
PUBLISHED_MAX_MVA_PAR_FRAC: Final[float] = 0.2


def _check_mva_par_frac(ev) -> None:  # noqa: ANN001 — xarray.Dataset
    """Log the event zarr's FLR-purity cut; warn if it differs from 0.2.

    CLAUDE.md pins Figs 7 & 8 to ``--max-mva-par-frac 0.2``. The
    ``bin_events_round8.py`` default is looser, so a freshly-built zarr
    can otherwise be plotted without the user noticing.
    """
    log = logging.getLogger("qp")
    cut = ev.attrs.get("max_mva_par_frac")
    if cut is None:
        log.warning(
            "event zarr has no 'max_mva_par_frac' attribute — published "
            "Figs 7 & 8 used %.2f; this zarr may render different numbers.",
            PUBLISHED_MAX_MVA_PAR_FRAC,
        )
        return
    log.info("mva_par_frac cut (from zarr attrs): <= %s", cut)
    if float(cut) > PUBLISHED_MAX_MVA_PAR_FRAC + 1e-12:
        log.warning(
            "event zarr cut (%.3f) is looser than the published value (%.2f); "
            "the figure will include more compressional-boundary events than "
            "the version in the paper.",
            float(cut),
            PUBLISHED_MAX_MVA_PAR_FRAC,
        )


def ensure_figures_dir() -> Path:
    """Create (if missing) and return ``Output/figures/``."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def ensure_diagnostics_dir() -> Path:
    """Create (if missing) and return ``Output/diagnostics/``."""
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    return DIAGNOSTICS_DIR
