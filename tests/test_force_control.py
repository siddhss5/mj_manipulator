# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Tests for ForceThresholds and SpeedProfile."""

import numpy as np
import pytest

from mj_manipulator.force_control import ForceThresholds, SpeedProfile


class TestForceThresholds:
    def test_below_threshold(self):
        ft = ForceThresholds(force_n=10.0, torque_nm=2.0)
        wrench = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        assert not ft.exceeded(wrench)

    def test_force_exceeded(self):
        ft = ForceThresholds(force_n=10.0, torque_nm=2.0)
        wrench = np.array([8.0, 8.0, 0.0, 0.0, 0.0, 0.0])  # ||f|| ≈ 11.3
        assert ft.exceeded(wrench)

    def test_torque_exceeded(self):
        ft = ForceThresholds(force_n=10.0, torque_nm=2.0)
        wrench = np.array([0.0, 0.0, 0.0, 1.5, 1.5, 0.0])  # ||t|| ≈ 2.1
        assert ft.exceeded(wrench)

    def test_both_exceeded(self):
        ft = ForceThresholds(force_n=1.0, torque_nm=1.0)
        wrench = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        assert ft.exceeded(wrench)

    def test_check_returns_magnitudes(self):
        ft = ForceThresholds(force_n=10.0, torque_nm=2.0)
        wrench = np.array([3.0, 4.0, 0.0, 1.0, 0.0, 0.0])
        exceeded, force_mag, torque_mag = ft.check(wrench)
        assert not exceeded
        assert abs(force_mag - 5.0) < 1e-6
        assert abs(torque_mag - 1.0) < 1e-6

    def test_zero_wrench(self):
        ft = ForceThresholds(force_n=1.0, torque_nm=1.0)
        assert not ft.exceeded(np.zeros(6))

    def test_exact_threshold_not_exceeded(self):
        """Threshold is strict >."""
        ft = ForceThresholds(force_n=5.0, torque_nm=2.0)
        wrench = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 0.0])  # ||f|| = 5.0 exactly
        assert not ft.exceeded(wrench)

    def test_frozen(self):
        ft = ForceThresholds(force_n=10.0, torque_nm=2.0)
        with pytest.raises(AttributeError):
            ft.force_n = 20.0


class TestSpeedProfile:
    def test_max_speed_beyond_ramp(self):
        sp = SpeedProfile(max_linear=0.15, min_linear=0.06, max_angular=0.15, min_angular=0.075, ramp_distance=0.3)
        assert sp.linear_speed(0.5) == 0.15
        assert sp.angular_speed(0.5) == 0.15

    def test_min_speed_at_target(self):
        sp = SpeedProfile(max_linear=0.15, min_linear=0.06, max_angular=0.15, min_angular=0.075, ramp_distance=0.3)
        assert sp.linear_speed(0.0) == 0.06
        assert sp.angular_speed(0.0) == 0.075

    def test_midpoint_interpolation(self):
        sp = SpeedProfile(max_linear=0.15, min_linear=0.06, max_angular=0.15, min_angular=0.075, ramp_distance=0.3)
        # At ramp_distance/2 = 0.15m, t = 0.5
        # linear: 0.06 + 0.5 * (0.15 - 0.06) = 0.105
        assert abs(sp.linear_speed(0.15) - 0.105) < 1e-6
        # angular: 0.075 + 0.5 * (0.15 - 0.075) = 0.1125
        assert abs(sp.angular_speed(0.15) - 0.1125) < 1e-6

    def test_at_ramp_boundary(self):
        sp = SpeedProfile(max_linear=0.15, min_linear=0.06, max_angular=0.15, min_angular=0.075, ramp_distance=0.3)
        assert sp.linear_speed(0.3) == 0.15
        assert sp.angular_speed(0.3) == 0.15

    def test_constant_profile(self):
        sp = SpeedProfile.constant(linear=0.1, angular=0.2)
        assert sp.linear_speed(0.0) == 0.1
        assert sp.linear_speed(100.0) == 0.1
        assert sp.angular_speed(0.0) == 0.2
        assert sp.angular_speed(100.0) == 0.2

    def test_zero_ramp_distance(self):
        """Zero ramp → always min speed (no room to accelerate)."""
        sp = SpeedProfile(max_linear=0.15, min_linear=0.06, max_angular=0.15, min_angular=0.075, ramp_distance=0.0)
        assert sp.linear_speed(0.0) == 0.06
        assert sp.linear_speed(0.5) == 0.06  # no ramp = stuck at min

    def test_frozen(self):
        sp = SpeedProfile.constant(0.1, 0.2)
        with pytest.raises(AttributeError):
            sp.max_linear = 0.5

    def test_ada_mouth_approach_profile(self):
        """Reproduces the original ada_feeding mouth approach parameters."""
        sp = SpeedProfile(
            max_linear=0.15,
            min_linear=0.06,
            max_angular=0.15,
            min_angular=0.075,
            ramp_distance=0.3,
        )
        # Far away: full speed
        assert sp.linear_speed(0.4) == 0.15
        # At 10cm: decelerating
        speed_10cm = sp.linear_speed(0.1)
        assert 0.06 < speed_10cm < 0.15
        # At 1cm: nearly stopped
        speed_1cm = sp.linear_speed(0.01)
        assert abs(speed_1cm - 0.063) < 0.005
