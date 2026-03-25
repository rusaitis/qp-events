"""Tests for the generic field line tracer."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from qp.fieldline.tracer import (
    FieldLineTrace,
    dipole_field,
    saturn_field_wrapper,
    trace_fieldline,
    trace_fieldline_bidirectional,
)
from qp.fieldline.kmag_model import SaturnField


class TestTraceFieldline:
    """Tests for the generic trace_fieldline function."""

    def test_dipole_trace_reaches_surface(self):
        """Tracing from equator at 10 Rs inward should reach ~1 Rs."""
        # Start at equator, trace along B (which points south at equator)
        trace = trace_fieldline(dipole_field, [10.0, 0.0, 0.0], step=-0.05)
        # Should terminate near the surface
        assert trace.r[-1] < 1.5

    def test_dipole_trace_returns_fieldlinetrace(self):
        """Should return a FieldLineTrace with correct shapes."""
        trace = trace_fieldline(dipole_field, [5.0, 0.0, 0.0], step=-0.05)
        assert isinstance(trace, FieldLineTrace)
        n = len(trace.positions)
        assert trace.field.shape == (n, 3)
        assert trace.field_magnitude.shape == (n,)
        assert trace.positions.shape == (n, 3)

    def test_dipole_field_magnitude_positive(self):
        """Field magnitude should always be positive."""
        trace = trace_fieldline(dipole_field, [5.0, 0.0, 3.0], step=0.05, max_steps=200)
        assert np.all(trace.field_magnitude > 0)

    def test_dipole_bidirectional_symmetric(self):
        """Bidirectional trace from equator should be roughly symmetric."""
        trace = trace_fieldline_bidirectional(
            dipole_field,
            [10.0, 0.0, 0.0],
            step=0.1,
            max_steps=5000,
        )
        # Both ends should be near the surface
        assert trace.r[0] < 1.5
        assert trace.r[-1] < 1.5
        # The trace should pass through the starting point
        # (maximum r should be near 10 Rs)
        assert trace.r.max() > 9.0

    def test_dipole_bidirectional_latitudes(self):
        """Endpoints should be at conjugate latitudes (symmetric for dipole)."""
        trace = trace_fieldline_bidirectional(
            dipole_field,
            [8.0, 0.0, 0.0],
            step=0.05,
            max_steps=10000,
        )
        rtp = trace.spherical
        # Endpoints should be near surface with conjugate colatitudes
        if trace.r[0] < 1.5 and trace.r[-1] < 1.5:
            theta_south = rtp[0, 1]
            theta_north = rtp[-1, 1]
            # For a dipole, footpoints are symmetric: theta + (pi-theta) = pi
            assert_allclose(theta_south + theta_north, math.pi, atol=0.1)

    def test_max_steps_respected(self):
        """Trace should not exceed max_steps."""
        trace = trace_fieldline(dipole_field, [5.0, 0.0, 0.0], step=0.05, max_steps=10)
        assert len(trace.positions) <= 11  # initial point + max_steps

    def test_max_radius_boundary(self):
        """Trace should stop at max_radius."""
        trace = trace_fieldline(
            dipole_field,
            [0.0, 0.0, 5.0],
            step=0.1,
            max_radius=20.0,
        )
        assert trace.r[-1] <= 20.5  # small overshoot from step size

    def test_flattening_changes_surface(self):
        """With flattening, polar traces should reach smaller r."""
        # Trace toward the pole (polar radius = 1 - 0.098 = 0.902)
        trace_sphere = trace_fieldline(
            dipole_field,
            [0.0, 0.0, 3.0],
            step=-0.02,
            min_radius=1.0,
            flattening=0.0,
        )
        trace_oblate = trace_fieldline(
            dipole_field,
            [0.0, 0.0, 3.0],
            step=-0.02,
            min_radius=1.0,
            flattening=0.09796,
        )
        # Oblate surface at pole is at r = 0.902, so trace goes deeper
        assert trace_oblate.r[-1] < trace_sphere.r[-1]


class TestSaturnFieldWrapper:
    """Tests for the saturn_field_wrapper adapter."""

    @pytest.fixture
    def field_func(self):
        field = SaturnField()
        return saturn_field_wrapper(field, time=284040000.0, coord="KSM")

    def test_returns_3_vector(self, field_func):
        """Wrapper should return shape (3,) ndarray."""
        B = field_func(np.array([10.0, 0.0, 0.0]))
        assert B.shape == (3,)
        assert np.all(np.isfinite(B))

    def test_trace_with_kmag(self, field_func):
        """Should be able to trace a field line with KMAG."""
        trace = trace_fieldline(field_func, [10.0, 0.0, 0.0], step=0.1)
        assert len(trace.positions) > 10
        assert trace.r[-1] < 2.0  # should reach surface

    def test_kmag_trace_closed_fieldline(self, field_func):
        """Dayside field line at moderate distance should close."""
        trace = trace_fieldline_bidirectional(
            field_func,
            [8.0, 0.0, 0.0],
            step=0.1,
            max_steps=5000,
        )
        # Both endpoints should be near the surface for a closed field line
        assert trace.r[0] < 2.0
        assert trace.r[-1] < 2.0


class TestFieldLineTrace:
    """Tests for the FieldLineTrace dataclass."""

    def test_spherical_property(self):
        """spherical property should give (r, theta, phi)."""
        positions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        field = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        trace = FieldLineTrace(
            positions=positions, field=field, field_magnitude=np.array([1.0, 1.0])
        )
        rtp = trace.spherical
        assert_allclose(rtp[0, 0], 1.0)  # r
        assert_allclose(rtp[0, 1], math.pi / 2, atol=1e-10)  # theta = pi/2 at equator
        assert_allclose(rtp[1, 0], 2.0)  # r
        assert_allclose(rtp[1, 1], 0.0, atol=1e-10)  # theta = 0 at pole

    def test_r_property(self):
        """r property should give radial distance."""
        positions = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        trace = FieldLineTrace(
            positions=positions, field=positions, field_magnitude=np.array([5.0, 5.0])
        )
        assert_allclose(trace.r, [5.0, 5.0])

    def test_conjugate_latitude_at_surface(self):
        """conjugate_latitude should find surface intersection."""
        # Create a trace that passes through r=1
        r_vals = np.linspace(1.0, 10.0, 100)
        theta = 0.5  # fixed colatitude
        positions = np.column_stack(
            [
                r_vals * np.sin(theta),
                np.zeros(100),
                r_vals * np.cos(theta),
            ]
        )
        trace = FieldLineTrace(
            positions=positions,
            field=np.ones_like(positions),
            field_magnitude=np.ones(100),
        )
        lat = trace.conjugate_latitude()
        expected = 90.0 - np.degrees(theta)
        assert lat is not None
        assert_allclose(lat, expected, atol=1.0)
