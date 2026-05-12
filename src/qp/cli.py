r"""Shared argparse helpers for QP scripts.

Each helper mutates an ``argparse`` action container (a parser or
``add_argument_group`` result) in place. The defaults match the
mission-wide canonical values; callers that want a different default
pass it explicitly.

The helpers stay tiny and stateless — they exist only to remove
duplicated ``add_argument`` calls across five sweep / bin /
footprint / dwell-grid scripts.
"""

from __future__ import annotations

import argparse
import os

__all__ = [
    "add_year_range_args",
    "add_workers_arg",
    "add_verbosity_arg",
    "add_field_model_args",
    "add_tracing_args",
]


def add_year_range_args(
    group: argparse._ActionsContainer,
    *,
    default_from: int = 2004,
    default_to: int = 2017,
) -> None:
    """Add ``--year-from`` and ``--year-to`` (Cassini mission span)."""
    group.add_argument(
        "--year-from",
        type=int,
        default=default_from,
        help="First year to process",
    )
    group.add_argument(
        "--year-to",
        type=int,
        default=default_to,
        help="Last year to process",
    )


def add_workers_arg(
    group: argparse._ActionsContainer,
    *,
    default: int | None = None,
    flag: str = "--workers",
) -> None:
    """Add a worker-count flag.

    ``default=None`` picks ``max(1, cpu_count - 1)`` at parse time.
    Pass ``flag="--trace-workers"`` for legacy scripts that use a
    different name.
    """
    if default is None:
        default = max(1, (os.cpu_count() or 2) - 1)
    group.add_argument(
        flag,
        type=int,
        default=default,
        help="Multiprocessing workers (1 = serial)",
    )


def add_verbosity_arg(group: argparse._ActionsContainer) -> None:
    """Add ``-v`` / ``--verbose``. Caller wires the logging level."""
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (INFO level)",
    )


def add_field_model_args(group: argparse._ActionsContainer) -> None:
    """Add KMAG field-model parameters: ``--dp``, ``--by-imf``, ``--bz-imf``.

    Defaults are the Khurana 2020 nominal Cassini values; the field is
    insensitive to IMF inside ``L = 20`` and ``dp = 0.017`` nPa is
    within 1% of the bias-minimising value at ``L = 10-20``.
    """
    group.add_argument(
        "--dp",
        type=float,
        default=0.017,
        help="Solar wind dynamic pressure (nPa)",
    )
    group.add_argument(
        "--by-imf",
        type=float,
        default=-0.2,
        help="IMF By component (nT, KSM)",
    )
    group.add_argument(
        "--bz-imf",
        type=float,
        default=0.1,
        help="IMF Bz component (nT, KSM)",
    )


def add_tracing_args(group: argparse._ActionsContainer) -> None:
    """Add KMAG tracing parameters: ``--trace-step``, ``--trace-max-radius``.

    Defaults match the canonical dwell grid (``step=0.15`` R_S, outer
    boundary at 60 R_S).
    """
    group.add_argument(
        "--trace-step",
        type=float,
        default=0.15,
        help="RK4 step size (R_S)",
    )
    group.add_argument(
        "--trace-max-radius",
        type=float,
        default=60.0,
        help="Outer tracing boundary (R_S)",
    )
