"""Integration tests for the benchmark suite."""

import numpy as np
import pytest

from qp.benchmark.generator import EventSpec, ScenarioConfig, generate_benchmark_dataset
from qp.benchmark.manifest import (
    InjectedEvent,
    DatasetManifest,
    events_to_csv,
    events_from_csv,
    manifest_to_json,
    manifest_from_json,
)
from qp.benchmark.scenarios import ALL_SCENARIOS, tier1_clean_qp60
from qp.benchmark.scoring import score_dataset, score_suite
from qp.events.catalog import WavePacketPeak


class TestGeneratorBasics:
    """Verify the dataset generator produces valid output."""

    def test_output_shapes(self):
        scenario = ScenarioConfig(
            dataset_id="test_shapes",
            duration_days=1.0, dt=60.0,
            noise_alpha=0.0, noise_sigma=0.01,
            background_trend=False,
            event_specs=[
                EventSpec(band="QP60", amplitude=1.0, center_hours=12.0),
            ],
        )
        t, fields, manifest = generate_benchmark_dataset(scenario, seed=42)
        assert t.shape == (1440,)
        assert fields.shape == (1440, 4)
        assert len(manifest.events) == 1

    def test_seed_reproducibility(self):
        scenario = tier1_clean_qp60()
        _, a, _ = generate_benchmark_dataset(scenario, seed=100)
        _, b, _ = generate_benchmark_dataset(scenario, seed=100)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self):
        scenario = tier1_clean_qp60()
        _, a, _ = generate_benchmark_dataset(scenario, seed=1)
        _, b, _ = generate_benchmark_dataset(scenario, seed=2)
        assert not np.allclose(a, b)

    def test_manifest_metadata(self):
        scenario = ScenarioConfig(
            dataset_id="test_meta",
            duration_days=2.0, dt=60.0,
            noise_alpha=1.0, noise_sigma=0.05,
            background_trend=False,
            event_specs=[
                EventSpec(band="QP60", amplitude=1.0, center_hours=12.0),
                EventSpec(band="QP30", amplitude=0.5, center_hours=36.0,
                          should_detect=False, event_type="decoy_test"),
            ],
        )
        _, _, manifest = generate_benchmark_dataset(scenario, seed=42)
        assert manifest.n_events == 2
        assert manifest.n_detectable == 1
        assert manifest.n_decoys == 1
        assert manifest.events[0].band == "QP60"
        assert manifest.events[1].should_detect is False


class TestManifestIO:
    """Verify JSON and CSV round-trip serialization."""

    def _make_event(self) -> InjectedEvent:
        return InjectedEvent(
            event_id="test-000", dataset_id="test",
            event_type="qp_wave", should_detect=True,
            band="QP60", period_sec=3600.0, amplitude_nT=1.0,
            start_sec=0.0, end_sec=14400.0, center_sec=7200.0,
            duration_sec=14400.0, n_oscillations=4.0,
            polarization="circular", ellipticity=1.0,
            wave_mode="alfvenic", propagation="standing",
            chirp_rate=0.0, waveform="sine", sawtooth_width=0.8,
            envelope_asymmetry=0.5, amplitude_jitter=0.0,
            harmonic_content=0.0, snr_injected=20.0, difficulty="easy",
        )

    def test_json_roundtrip(self, tmp_path):
        event = self._make_event()
        manifest = DatasetManifest(
            dataset_id="test", description="test dataset",
            duration_days=1.0, dt=60.0, n_samples=1440,
            seed=42, noise_alpha=1.0, noise_sigma=0.05,
            difficulty_tier="tier1", events=[event],
        )
        path = tmp_path / "manifest.json"
        manifest_to_json(manifest, path)
        loaded = manifest_from_json(path)
        assert loaded.dataset_id == "test"
        assert len(loaded.events) == 1
        assert loaded.events[0].band == "QP60"
        assert loaded.events[0].period_sec == 3600.0

    def test_csv_roundtrip(self, tmp_path):
        events = [self._make_event()]
        path = tmp_path / "events.csv"
        events_to_csv(events, path)
        loaded = events_from_csv(path)
        assert len(loaded) == 1
        assert loaded[0].event_id == "test-000"
        assert loaded[0].amplitude_nT == 1.0
        assert loaded[0].should_detect is True


class TestScoringBasics:
    """Verify scoring logic on trivial cases."""

    def test_perfect_detection(self):
        """All GT events matched → recall=1, precision=1."""
        from datetime import datetime, timedelta

        events = [
            InjectedEvent(
                event_id=f"e{i}", dataset_id="test",
                event_type="qp_wave", should_detect=True,
                band="QP60", period_sec=3600.0, amplitude_nT=1.0,
                start_sec=i * 50000, end_sec=i * 50000 + 20000,
                center_sec=i * 50000 + 10000, duration_sec=20000,
                n_oscillations=5.5, polarization="circular",
                ellipticity=1.0, wave_mode="alfvenic", propagation="standing",
                chirp_rate=0.0, waveform="sine", sawtooth_width=0.8,
                envelope_asymmetry=0.5, amplitude_jitter=0.0,
                harmonic_content=0.0, snr_injected=20.0, difficulty="easy",
            )
            for i in range(3)
        ]
        manifest = DatasetManifest(
            dataset_id="test_perfect", description="", duration_days=5,
            dt=60.0, n_samples=7200, seed=1, noise_alpha=0, noise_sigma=0,
            difficulty_tier="tier1", events=events,
        )

        epoch = datetime(2000, 1, 1)
        detections = [
            WavePacketPeak(
                peak_time=epoch + timedelta(seconds=e.center_sec),
                prominence=1.0,
                date_from=epoch + timedelta(seconds=e.start_sec),
                date_to=epoch + timedelta(seconds=e.end_sec),
                band="QP60", period_sec=3600.0,
            )
            for e in events
        ]

        score = score_dataset(manifest, detections, t0_sec=epoch.timestamp())
        assert score.recall == pytest.approx(1.0)
        assert score.precision == pytest.approx(1.0)
        assert score.f1 == pytest.approx(1.0)

    def test_no_detections(self):
        """No detections → recall=0, precision=1 (vacuous)."""
        events = [
            InjectedEvent(
                event_id="e0", dataset_id="test",
                event_type="qp_wave", should_detect=True,
                band="QP60", period_sec=3600.0, amplitude_nT=1.0,
                start_sec=0, end_sec=10000, center_sec=5000,
                duration_sec=10000, n_oscillations=2.8,
                polarization="circular", ellipticity=1.0,
                wave_mode="alfvenic", propagation="standing",
                chirp_rate=0.0, waveform="sine", sawtooth_width=0.8,
                envelope_asymmetry=0.5, amplitude_jitter=0.0,
                harmonic_content=0.0, snr_injected=20.0, difficulty="easy",
            )
        ]
        manifest = DatasetManifest(
            dataset_id="test_empty", description="", duration_days=1,
            dt=60.0, n_samples=1440, seed=1, noise_alpha=0, noise_sigma=0,
            difficulty_tier="tier1", events=events,
        )
        score = score_dataset(manifest, [])
        assert score.recall == 0.0
        assert score.n_false_negatives == 1

    def test_suite_aggregation(self):
        """SuiteScore aggregates correctly."""
        from qp.benchmark.scoring import BenchmarkScore

        scores = [
            BenchmarkScore(
                dataset_id="a", n_ground_truth=10, n_detectable=10,
                n_detected=8, n_true_positives=7, n_false_positives=1,
                n_false_negatives=3, precision=7 / 8, recall=7 / 10, f1=0.0,
                band_accuracy=1.0, mean_period_error_pct=5.0, mean_iou=0.5,
                recall_by_difficulty={}, n_decoy_events=0, n_decoy_detected=0,
                decoy_rejection_rate=1.0,
            ),
        ]
        suite = score_suite(scores)
        assert suite.overall_recall == 0.7
        assert suite.overall_precision == pytest.approx(7 / 8)


class TestAllScenariosGenerate:
    """Smoke test: every registered scenario generates without error."""

    @pytest.mark.parametrize("scenario_id", list(ALL_SCENARIOS.keys())[:5])
    def test_scenario_generates(self, scenario_id):
        """First 5 scenarios generate valid datasets."""
        factory = ALL_SCENARIOS[scenario_id]
        scenario = factory()
        t, fields, manifest = generate_benchmark_dataset(scenario, seed=42)
        assert len(t) > 0
        assert fields.shape[1] == 4
        assert manifest.dataset_id == scenario_id
        assert manifest.n_events > 0
