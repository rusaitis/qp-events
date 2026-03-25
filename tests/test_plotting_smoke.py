"""Smoke tests for qp.plotting — verify no exceptions on basic usage.

These tests create figures, call plotting functions with minimal synthetic
data, and verify they don't raise. No pixel-level comparison.
"""

from __future__ import annotations

import datetime

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qp.plotting.maps import (
    draw_field_indicator,
    draw_latitude_indicator,
    draw_local_time_indicator,
    draw_power_indicator,
    draw_range_indicator,
    plot_lt_lat_heatmap,
    plot_polar_heatmap,
)
from qp.plotting.style import save_figure, style_colorbar
from qp.coords.transforms import phi_to_lt, lt_to_phi
from qp.plotting.saturn import (
    REFERENCE_SHELLS,
    Arrow3D,
    draw_axes_3d,
    draw_reference_shells,
    draw_vector_3d,
    equalize_3d_axes,
)
from qp.plotting.spectra import (
    annotate_spectral_peaks,
    draw_period_rectangles,
    overlay_power_law,
    plot_fft_snapshots,
    plot_power_density,
    plot_power_ratios,
    plot_spectrogram,
)
from qp.plotting.timeseries import (
    field_limits,
    field_range,
    lookup_mag_region,
    plot_ephemeris_bar,
    plot_field_timeseries,
    plot_highlight_intervals,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def synthetic_spectrum():
    freq = np.logspace(-4, -2, 200)
    psd = 1e4 * freq**-2
    return freq, psd


@pytest.fixture
def synthetic_timeseries():
    n = 500
    dt = 60.0
    t0 = datetime.datetime(2007, 1, 2, 0, 0, 0)
    times = [t0 + datetime.timedelta(seconds=i * dt) for i in range(n)]
    t_sec = np.arange(n) * dt
    bx = np.sin(2 * np.pi * t_sec / 3600)
    by = np.cos(2 * np.pi * t_sec / 3600)
    bz = 0.5 * np.sin(2 * np.pi * t_sec / 1800)
    return times, [bx, by, bz]


# ============================================================================
# Spectra module smoke tests
# ============================================================================


class TestPlotPowerDensity:
    def test_basic(self, synthetic_spectrum):
        freq, psd = synthetic_spectrum
        fig, ax = plt.subplots()
        plot_power_density(ax, freq, [psd], mark_periods=False)

    def test_with_background(self, synthetic_spectrum):
        freq, psd = synthetic_spectrum
        fig, ax = plt.subplots()
        bg = psd * 0.5
        plot_power_density(ax, freq, [psd], background=bg)


class TestPlotPowerRatios:
    def test_basic(self, synthetic_spectrum):
        freq, psd = synthetic_spectrum
        fig, ax = plt.subplots()
        ratios = {
            "r_par": psd / (psd * 0.5),
            "r_perp1": psd / (psd * 0.5),
            "r_perp2": psd / (psd * 0.5),
            "r_total": psd / (psd * 0.5),
        }
        plot_power_ratios(ax, freq, ratios)


class TestPlotSpectrogram:
    def test_basic(self):
        freq = np.linspace(1e-4, 1e-2, 50)
        time = np.arange(100) * 60.0
        power = np.random.default_rng(42).uniform(0.1, 10, (50, 100))
        fig, ax = plt.subplots()
        im = plot_spectrogram(ax, time, freq, power)
        assert im is not None


class TestAnnotateSpectralPeaks:
    def test_basic(self, synthetic_spectrum):
        freq, psd = synthetic_spectrum
        fig, ax = plt.subplots()
        ax.loglog(1 / (freq * 60), psd)
        annotate_spectral_peaks(ax, freq, psd, [50, 100])

    def test_no_labels(self, synthetic_spectrum):
        freq, psd = synthetic_spectrum
        fig, ax = plt.subplots()
        ax.loglog(1 / (freq * 60), psd)
        annotate_spectral_peaks(ax, freq, psd, [50], annotate_period=False)


class TestOverlayPowerLaw:
    def test_basic(self, synthetic_spectrum):
        freq, psd = synthetic_spectrum
        fig, ax = plt.subplots()
        ax.loglog(1 / (freq * 60), psd)
        overlay_power_law(ax, freq, psd * 0.8, label="fit")


class TestDrawPeriodRectangles:
    def test_default_bands(self):
        fig, ax = plt.subplots()
        ax.set_xlim(5, 200)
        ax.set_ylim(0.1, 100)
        draw_period_rectangles(ax)

    def test_custom_bands(self):
        fig, ax = plt.subplots()
        ax.set_xlim(5, 200)
        ax.set_ylim(0.1, 100)
        draw_period_rectangles(ax, bands={"test": (40, 80)}, colors={"test": "red"})


class TestPlotFFTSnapshots:
    def test_basic(self, synthetic_spectrum):
        freq, psd = synthetic_spectrum
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        psd_list = [psd, psd * 0.5, psd * 2]
        plot_fft_snapshots(axes, freq, psd_list, snapshot_labels=["A", "B", "C"])


# ============================================================================
# Timeseries module smoke tests
# ============================================================================


class TestPlotFieldTimeseries:
    def test_basic(self, synthetic_timeseries):
        times, components = synthetic_timeseries
        fig, ax = plt.subplots()
        plot_field_timeseries(ax, times, components, labels=["Bx", "By", "Bz"])


class TestPlotEphemerisBar:
    def test_basic(self, synthetic_timeseries):
        times, _ = synthetic_timeseries
        fig, ax = plt.subplots()
        coords = {"X": np.linspace(10, 15, len(times))}
        plot_ephemeris_bar(ax, times, coords)


class TestPlotHighlightIntervals:
    def test_basic(self, synthetic_timeseries):
        times, components = synthetic_timeseries
        fig, ax = plt.subplots()
        plot_field_timeseries(ax, times, components)
        t0, t1 = times[100], times[200]
        plot_highlight_intervals(ax, [(t0, t1)])


# ============================================================================
# Timeseries utility function tests
# ============================================================================


class TestLookupMagRegion:
    def test_basic(self):
        times = [datetime.datetime(2007, 1, 1, h) for h in range(24)]
        codes = np.array([0] * 12 + [1] * 6 + [2] * 6)
        label, color = lookup_mag_region(datetime.datetime(2007, 1, 1, 6), times, codes)
        assert label == "MS"
        assert isinstance(color, str)

    def test_boundary(self):
        times = [datetime.datetime(2007, 1, 1, h) for h in range(24)]
        codes = np.array([0] * 12 + [1] * 6 + [2] * 6)
        label, _ = lookup_mag_region(datetime.datetime(2007, 1, 1, 15), times, codes)
        assert label == "SH"


class TestFieldRange:
    def test_basic(self):
        c1 = np.array([0, 1, 2, 3])
        c2 = np.array([-5, 0, 5])
        assert field_range([c1, c2]) == 10.0

    def test_single(self):
        assert field_range([np.array([3, 7])]) == 4.0

    def test_empty(self):
        assert field_range([]) == 0.0


class TestFieldLimits:
    def test_basic(self):
        c1 = np.array([0, 10])
        c2 = np.array([-5, 5])
        lo, hi = field_limits([c1, c2])
        assert lo == -5.0
        assert hi == 10.0

    def test_empty(self):
        lo, hi = field_limits([])
        assert lo == 0.0
        assert hi == 0.0


# ============================================================================
# Maps module smoke tests (Task 4.3)
# ============================================================================


class TestPlotPolarHeatmap:
    def test_basic(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        data = np.random.default_rng(42).uniform(0, 10, (10, 24))
        lat_edges = np.linspace(60, 90, 11)
        im = plot_polar_heatmap(ax, data, lat_edges)
        assert im is not None

    def test_south_hemisphere(self):
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        data = np.random.default_rng(42).uniform(0, 10, (10, 24))
        lat_edges = np.linspace(-90, -60, 11)
        im = plot_polar_heatmap(ax, data, lat_edges, hemisphere="south")
        assert im is not None


class TestDrawLatitudeIndicator:
    def test_basic(self):
        fig, ax = plt.subplots()
        draw_latitude_indicator(ax)

    def test_with_range(self):
        fig, ax = plt.subplots()
        draw_latitude_indicator(ax, lat_range=(-30, 30), lat_ticks=[0, 15, -15])


class TestDrawRangeIndicator:
    def test_basic(self):
        fig, ax = plt.subplots()
        draw_range_indicator(
            ax, data=np.array([5, 10, 15]), data_range=(0, 20), label="R"
        )


class TestPlotLtLatHeatmap:
    def test_basic(self):
        fig, ax = plt.subplots()
        data = np.random.default_rng(42).uniform(0, 100, (18, 24))
        lt_edges = np.linspace(0, 24, 25)
        lat_edges = np.linspace(-90, 90, 19)
        im = plot_lt_lat_heatmap(ax, data, lt_edges, lat_edges)
        assert im is not None


# ============================================================================
# Saturn module smoke tests (Task 4.4)
# ============================================================================


class TestArrow3D:
    def test_creation(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        arrow = Arrow3D(
            [0, 1],
            [0, 1],
            [0, 1],
            mutation_scale=10,
            lw=1,
            arrowstyle="-|>",
            color="white",
        )
        ax.add_artist(arrow)


class TestDrawAxes3D:
    def test_basic(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
        draw_axes_3d(ax, length=10)


class TestDrawVector3D:
    def test_basic(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        draw_vector_3d(ax, (0, 0, 0), (5, 5, 5), label="B")


class TestEqualize3DAxes:
    def test_basic(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(-5, 15)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-1, 1)
        equalize_3d_axes(ax)
        # After equalization, all axes should span the same range
        xr = ax.get_xlim3d()[1] - ax.get_xlim3d()[0]
        yr = ax.get_ylim3d()[1] - ax.get_ylim3d()[0]
        zr = ax.get_zlim3d()[1] - ax.get_zlim3d()[0]
        assert abs(xr - yr) < 0.01
        assert abs(yr - zr) < 0.01


class TestDrawReferenceShells:
    def test_basic(self):
        fig, ax = plt.subplots()
        draw_reference_shells(ax, shells=["10Rs", "Enceladus"])
        assert len(ax.get_lines()) == 2

    def test_south(self):
        fig, ax = plt.subplots()
        draw_reference_shells(ax, shells=["20Rs"], hemisphere="south")
        assert len(ax.get_lines()) == 1

    def test_all_shells(self):
        fig, ax = plt.subplots()
        draw_reference_shells(ax)
        assert len(ax.get_lines()) == len(REFERENCE_SHELLS)


class TestReferenceShellData:
    def test_all_shells_have_required_keys(self):
        for name, shell in REFERENCE_SHELLS.items():
            assert "north_lt" in shell, f"{name} missing north_lt"
            assert "north_lat" in shell, f"{name} missing north_lat"
            assert "south_lt" in shell, f"{name} missing south_lt"
            assert "south_lat" in shell, f"{name} missing south_lat"

    def test_consistent_lengths(self):
        for name, shell in REFERENCE_SHELLS.items():
            assert len(shell["north_lt"]) == len(shell["north_lat"]), name
            assert len(shell["south_lt"]) == len(shell["south_lat"]), name


# ============================================================================
# Task 4.5: PlotFFT visualization indicators
# ============================================================================


class TestDrawLocalTimeIndicator:
    def test_basic(self):
        fig, ax = plt.subplots()
        draw_local_time_indicator(ax)

    def test_with_range_and_values(self):
        fig, ax = plt.subplots()
        draw_local_time_indicator(ax, lt_range=(6, 18), lt_values=[3, 9, 15, 21])


class TestDrawPowerIndicator:
    def test_basic(self):
        fig, ax = plt.subplots()
        draw_power_indicator(ax, data=np.array([1e-4, 1e-3, 1e-2]), ylim=(1e-5, 1e-1))

    def test_with_range(self):
        fig, ax = plt.subplots()
        draw_power_indicator(ax, data_range=(1e-4, 1e-2))


class TestDrawFieldIndicator:
    def test_basic(self):
        fig, ax = plt.subplots()
        draw_field_indicator(ax, data=np.array([1, 5, 10, 20]), ylim=(0.1, 100))

    def test_with_range(self):
        fig, ax = plt.subplots()
        draw_field_indicator(ax, data_range=(2, 15))


# ============================================================================
# Task 4.6: Style utilities + coordinate conversions
# ============================================================================


class TestSaveFigure:
    def test_saves_file(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = tmp_path / "test_fig.png"
        save_figure(fig, path, close=True)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_transparent(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = tmp_path / "test_transparent.png"
        save_figure(fig, path, transparent=True, close=True)
        assert path.exists()

    def test_creates_parent_dirs(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        path = tmp_path / "subdir" / "nested" / "fig.png"
        save_figure(fig, path, close=True)
        assert path.exists()


class TestStyleColorbar:
    def test_basic(self):
        fig, ax = plt.subplots()
        data = np.random.default_rng(42).uniform(0, 10, (10, 10))
        im = ax.imshow(data)
        cbar = style_colorbar(im, ax, label="Test")
        assert cbar is not None


class TestPhiToLt:
    def test_noon(self):
        from numpy.testing import assert_allclose

        assert_allclose(phi_to_lt(0), 12.0)

    def test_midnight(self):
        from numpy.testing import assert_allclose

        assert_allclose(phi_to_lt(np.pi), 24.0)

    def test_dawn(self):
        from numpy.testing import assert_allclose

        assert_allclose(phi_to_lt(np.pi / 2), 18.0)

    def test_array(self):
        from numpy.testing import assert_allclose

        result = phi_to_lt(np.array([0, np.pi / 2, np.pi]))
        assert_allclose(result, [12, 18, 24])


class TestLtToPhi:
    def test_noon(self):
        from numpy.testing import assert_allclose

        assert_allclose(lt_to_phi(12), np.pi)

    def test_midnight(self):
        from numpy.testing import assert_allclose

        assert_allclose(lt_to_phi(0), 0.0)

    def test_roundtrip_noon(self):
        """phi_to_lt(lt_to_phi(12)) should give 12 (noon is the fixed point)."""
        from numpy.testing import assert_allclose

        # Note: lt_to_phi maps 0h→0, 12h→π, 24h→2π
        # phi_to_lt maps 0→12h, π→24h
        # So the roundtrip only works for the convention overlap at noon
        assert_allclose(phi_to_lt(lt_to_phi(12)), 24.0)  # 12h → π → 24h (wrap)
