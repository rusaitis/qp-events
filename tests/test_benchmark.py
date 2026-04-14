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
            harmonic_content=0.0, snr_injected=20.0, snr_in_band=20.0,
            start_2sigma_sec=0.0, end_2sigma_sec=14400.0, difficulty="easy",
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
                harmonic_content=0.0, snr_injected=20.0, snr_in_band=20.0,
            start_2sigma_sec=0.0, end_2sigma_sec=14400.0, difficulty="easy",
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
                harmonic_content=0.0, snr_injected=20.0, snr_in_band=20.0,
            start_2sigma_sec=0.0, end_2sigma_sec=14400.0, difficulty="easy",
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
                mean_detection_latency_sec=0.0,
                median_period_error_pct=5.0,
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


class TestRound3Scoring:
    """Round-3 scoring additions: per-tier F1, weight ablation, FP/day rate."""

    def test_per_tier_f1_present(self):
        from qp.benchmark.runner import run_benchmark
        # Pick one cheap scenario per tier with no zarr churn — generate
        # on the fly to keep this fast.
        suite = run_benchmark(
            scenario_ids=["tier1_clean_qp60"],
            data_dir=__import__("pathlib").Path("/tmp/qp_test_per_tier"),
            regenerate=True,
        )
        assert "tier1" in suite.per_tier_f1
        assert 0.0 <= suite.per_tier_f1["tier1"] <= 1.0

    def test_composite_weight_ablation(self):
        from qp.benchmark.scoring import (
            COMPOSITE_WEIGHTS,
            composite_score_ablation,
            SuiteScore,
            BenchmarkScore,
        )
        # Build a synthetic SuiteScore with minimal fields filled
        s = SuiteScore(
            overall_precision=0.9, overall_recall=0.9, overall_f1=0.9,
            per_tier_recall={}, per_band_recall={},
            band_accuracy=0.95, decoy_rejection_rate=0.85,
            summary_score=0.9, band_accuracy_macro=0.92,
            dataset_scores=[],
        )
        scores = composite_score_ablation(s)
        assert set(scores) == set(COMPOSITE_WEIGHTS)
        # All scores are in [0, 1].
        for name, v in scores.items():
            assert 0.0 <= v <= 1.0, f"{name}: {v}"

    def test_empty_null_uses_fp_per_day(self):
        """decoy_rejection on empty-null scenarios scales with duration."""
        import datetime as _dt
        from qp.benchmark.scoring import score_dataset, FP_PER_DAY_TOLERANCE
        from qp.benchmark.manifest import DatasetManifest
        from qp.events.catalog import WavePacketPeak

        # 10-day pure-null manifest with 0 events.
        m = DatasetManifest(
            dataset_id="decoy_test",
            description="", duration_days=10.0, dt=60.0,
            n_samples=14400, seed=0, noise_alpha=1.2, noise_sigma=0.05,
            difficulty_tier="decoy",
        )
        # 1 detection over 10 days = 0.1 FP/day; tolerance 0.2 → score = 1 - 0.1/0.4 = 0.75
        epoch = _dt.datetime(2000, 1, 1)
        det = WavePacketPeak(
            peak_time=epoch + _dt.timedelta(hours=10),
            prominence=1.0,
            date_from=epoch + _dt.timedelta(hours=8),
            date_to=epoch + _dt.timedelta(hours=12),
            band="QP60", period_sec=3600.0,
        )
        score = score_dataset(m, [det])
        expected = 1.0 - (0.1 / (2.0 * FP_PER_DAY_TOLERANCE))
        assert abs(score.decoy_rejection_rate - expected) < 1e-9
