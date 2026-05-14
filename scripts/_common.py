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


def ensure_figures_dir() -> Path:
    """Create (if missing) and return ``Output/figures/``."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def ensure_diagnostics_dir() -> Path:
    """Create (if missing) and return ``Output/diagnostics/``."""
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    return DIAGNOSTICS_DIR
