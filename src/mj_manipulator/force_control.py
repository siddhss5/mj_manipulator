# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Force/torque monitoring types for contact-rich manipulation.

Provides ``ForceThresholds`` for abort conditions and ``SpeedProfile``
for distance-based deceleration. Used by ``servo_to_pose`` and
``ft_guarded_move`` in the servo module.

These are general types — not feeding-specific. Any contact-rich task
(guarded placement, surface following, insertion) uses the same patterns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForceThresholds:
    """Force/torque thresholds for motion abort.

    When the measured wrench exceeds these thresholds, the motion
    aborts with ``FailureKind.SAFETY_ABORTED``.

    Args:
        force_n: Maximum allowed force magnitude (N).
        torque_nm: Maximum allowed torque magnitude (N·m).
    """

    force_n: float
    torque_nm: float

    def exceeded(self, wrench: np.ndarray) -> bool:
        """Check if a 6D wrench [fx, fy, fz, tx, ty, tz] exceeds thresholds."""
        force_mag = float(np.linalg.norm(wrench[:3]))
        torque_mag = float(np.linalg.norm(wrench[3:]))
        return force_mag > self.force_n or torque_mag > self.torque_nm

    def check(self, wrench: np.ndarray) -> tuple[bool, float, float]:
        """Check thresholds and return (exceeded, force_mag, torque_mag)."""
        force_mag = float(np.linalg.norm(wrench[:3]))
        torque_mag = float(np.linalg.norm(wrench[3:]))
        return (
            force_mag > self.force_n or torque_mag > self.torque_nm,
            force_mag,
            torque_mag,
        )


@dataclass(frozen=True)
class SpeedProfile:
    """Distance-based speed profile for Cartesian servo.

    Linearly interpolates between max and min speeds based on distance
    to target. At distances >= ``ramp_distance``, uses max speed. At
    distance 0, uses min speed.

    Used by ``servo_to_pose`` for smooth deceleration near targets
    (e.g., approaching a person's mouth, precision placement).

    Args:
        max_linear: Maximum linear speed (m/s) at full distance.
        min_linear: Minimum linear speed (m/s) near target.
        max_angular: Maximum angular speed (rad/s) at full distance.
        min_angular: Minimum angular speed (rad/s) near target.
        ramp_distance: Distance (m) at which deceleration starts.
    """

    max_linear: float
    min_linear: float
    max_angular: float
    min_angular: float
    ramp_distance: float

    def linear_speed(self, distance: float) -> float:
        """Compute linear speed for a given distance to target."""
        if self.ramp_distance <= 0:
            return self.min_linear
        if distance >= self.ramp_distance:
            return self.max_linear
        t = distance / self.ramp_distance
        return self.min_linear + t * (self.max_linear - self.min_linear)

    def angular_speed(self, distance: float) -> float:
        """Compute angular speed for a given distance to target."""
        if self.ramp_distance <= 0:
            return self.min_angular
        if distance >= self.ramp_distance:
            return self.max_angular
        t = distance / self.ramp_distance
        return self.min_angular + t * (self.max_angular - self.min_angular)

    @classmethod
    def constant(cls, linear: float, angular: float) -> SpeedProfile:
        """Create a constant-speed profile (no deceleration)."""
        return cls(
            max_linear=linear,
            min_linear=linear,
            max_angular=angular,
            min_angular=angular,
            ramp_distance=0.0,
        )
