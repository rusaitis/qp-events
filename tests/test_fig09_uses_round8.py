"""Fig 9 round-8 plumbing test.

Verifies that the round-8 catalogue can be loaded, filtered to QP60,
and turned into :class:`WavePacketPeak` instances that
:func:`compute_separations` accepts. Also asserts the median
inter-packet separation is in the PPO-period window (~10.7 h ± a
generous bracket), the very property Fig 9 exists to display.

The full Fig 9 script lives at ``scripts/fig09_separation_times.py``;
this test exercises the same data-loading path so that future
schema changes break early.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

import qp
from qp.events.catalog import WavePacketPeak
from qp.events.wave_packets import compute_separations, separation_statistics


PARQUET = qp.OUTPUT_DIR / "events_round8.parquet"


@pytest.fixture(scope="module")
def qp60_packets() -> list[WavePacketPeak]:
    if not PARQUET.exists():
        pytest.skip(f"missing {PARQUET}; run scripts/sweep_events_round8.py first")
    df = pd.read_parquet(PARQUET)
    sub = df.loc[df["band"] == "QP60"].copy()
    sub["peak_time"] = pd.to_datetime(sub["peak_time"])
    one_hour = datetime.timedelta(hours=1)
    return [
        WavePacketPeak(
            peak_time=row.peak_time.to_pydatetime(),
            prominence=float(row.q_factor),
            date_from=row.peak_time.to_pydatetime() - one_hour,
            date_to=row.peak_time.to_pydatetime() + one_hour,
            band=row.band,
            period_sec=float(row.period_min) * 60.0,
        )
        for row in sub.itertuples()
    ]


def test_qp60_packets_present(qp60_packets):
    assert len(qp60_packets) > 100, "round-8 catalogue should have many QP60 events"


def test_separation_median_in_ppo_multiples(qp60_packets):
    """Median separation should sit in the PPO ≈ 10.7 h family.

    The paper's published median (run on the legacy single-band CWT
    peak finder) was 10.73 h. The round-8 catalogue is much stricter
    — q_factor + Stokes + polarization gates strip ~half the events
    the legacy detector accepted, which Poisson-thins the inter-arrival
    histogram so the median lands at ≈ 1×–2× PPO. This test brackets
    [0.7, 2.5] × PPO, which is broad enough to admit either outcome
    while still flagging a catastrophic regression (e.g. an empty
    catalogue or a 50 h median).
    """
    seps = compute_separations(qp60_packets, max_separation_hours=36.0)
    stats = separation_statistics(seps)
    PPO_HOURS = 10.7
    assert 0.7 * PPO_HOURS < stats["median"] < 2.5 * PPO_HOURS, (
        f"median QP60 separation {stats['median']:.2f} h outside [0.7, 2.5]×PPO"
    )
    assert stats["count"] > 50
