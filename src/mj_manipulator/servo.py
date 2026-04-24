# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Force-aware Cartesian servo primitives.

General-purpose primitives for contact-rich manipulation:

- ``servo_to_pose``: Move to a target pose with distance-based
  deceleration and F/T monitoring (e.g., mouth approach, precision
  placement near fragile objects).

- ``ft_guarded_move``: Move in a fixed direction until F/T exceeds
  a threshold or duration elapses (e.g., food stabbing, surface
  contact detection, bite detection).

Both return ``Outcome`` with structured failure information and use
the existing ``CartesianController`` + ``ctx.step_cartesian``
infrastructure internally.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from mj_manipulator.force_control import ForceThresholds, SpeedProfile
from mj_manipulator.outcome import FailureKind, Outcome, failure, success

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.protocols import ExecutionContext

logger = logging.getLogger(__name__)


def _rotation_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Compute rotation error as axis-angle vector."""
    R_err = R_target @ R_current.T
    cos_angle = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    ) / (2 * np.sin(angle))
    return axis * angle


def _check_ft(arm: Arm, threshold: ForceThresholds | None) -> tuple[bool, float, float]:
    """Check F/T against threshold. Returns (exceeded, force_mag, torque_mag)."""
    if threshold is None:
        return False, 0.0, 0.0
    if not arm.has_ft_sensor or not arm.ft_valid:
        return False, 0.0, 0.0
    wrench = arm.get_ft_wrench()
    if np.any(np.isnan(wrench)):
        return False, 0.0, 0.0
    return threshold.check(wrench)


def servo_to_pose(
    target: np.ndarray,
    arm: Arm,
    ctx: ExecutionContext,
    *,
    speed_profile: SpeedProfile | None = None,
    ft_threshold: ForceThresholds | None = None,
    position_tol: float = 0.005,
    rotation_tol: float = 0.05,
    timeout: float = 10.0,
) -> Outcome:
    """Cartesian servo to a target pose with deceleration and F/T monitoring.

    Computes a proportional twist from current EE to target each control
    cycle, with speed scaled by distance to target (via ``speed_profile``).
    Aborts if F/T exceeds threshold.

    This is the general primitive for any approach-to-pose motion where
    the robot must decelerate near the target and react to contact:
    mouth approach, precision placement, guarded insertion.

    Args:
        target: 4x4 target pose (world frame).
        arm: Arm instance (for EE pose, F/T, joint state).
        ctx: Execution context (for step_cartesian).
        speed_profile: Distance-based speed ramp. If None, uses
            constant 0.05 m/s.
        ft_threshold: Force/torque abort threshold. If None, no F/T
            monitoring.
        position_tol: Stop when position error < this (meters).
        rotation_tol: Stop when rotation error < this (radians).
        timeout: Maximum duration (seconds).

    Returns:
        Outcome with success=True if target reached, or failure with
        SAFETY_ABORTED (F/T exceeded) or TIMEOUT.
    """
    if speed_profile is None:
        speed_profile = SpeedProfile.constant(linear=0.05, angular=0.3)

    dt = ctx.control_dt
    arm_name = arm.config.name
    t0 = time.monotonic()

    while time.monotonic() - t0 < timeout:
        # Current EE pose
        ee_pose = arm.get_ee_pose()
        pos_err = target[:3, 3] - ee_pose[:3, 3]
        rot_err = _rotation_error(target[:3, :3], ee_pose[:3, :3])

        pos_err_norm = float(np.linalg.norm(pos_err))
        rot_err_norm = float(np.linalg.norm(rot_err))

        # Convergence check
        if pos_err_norm < position_tol and rot_err_norm < rotation_tol:
            return success(
                position_error_m=pos_err_norm,
                rotation_error_rad=rot_err_norm,
            )

        # F/T check
        exceeded, force_mag, torque_mag = _check_ft(arm, ft_threshold)
        if exceeded:
            return failure(
                FailureKind.SAFETY_ABORTED,
                "servo_to_pose:ft_exceeded",
                force_n=force_mag,
                torque_nm=torque_mag,
                threshold_force_n=ft_threshold.force_n,
                threshold_torque_nm=ft_threshold.torque_nm,
                position_error_m=pos_err_norm,
            )

        # Compute twist with speed profile
        lin_speed = speed_profile.linear_speed(pos_err_norm)
        ang_speed = speed_profile.angular_speed(pos_err_norm)

        v = pos_err * min(1.0, lin_speed / (pos_err_norm + 1e-8))
        w = rot_err * min(1.0, ang_speed / (rot_err_norm + 1e-8))

        # Convert twist to joint targets via Jacobian
        q_current = arm.get_joint_positions()
        J = arm.get_ee_jacobian()
        twist = np.concatenate([v, w])
        qd = np.linalg.lstsq(J, twist, rcond=None)[0]

        # Clamp to velocity limits
        limits = getattr(arm.config, "kinematic_limits", None)
        if limits is not None:
            scale = np.max(np.abs(qd) / limits.velocity)
            if scale > 1.0:
                qd /= scale

        q_target = q_current + qd * dt
        ctx.step_cartesian(arm_name, q_target, qd)

    return failure(
        FailureKind.TIMEOUT,
        "servo_to_pose:timeout",
        timeout_s=timeout,
        position_error_m=float(np.linalg.norm(target[:3, 3] - arm.get_ee_pose()[:3, 3])),
    )


def ft_guarded_move(
    twist: np.ndarray,
    arm: Arm,
    ctx: ExecutionContext,
    *,
    ft_threshold: ForceThresholds,
    duration: float = 2.0,
    timeout: float | None = None,
) -> Outcome:
    """Move in a fixed direction until F/T exceeds threshold.

    Applies a constant 6D twist each control cycle while monitoring
    the F/T sensor. Returns success when:
    - F/T threshold is exceeded (contact detected — ``details["contact"]=True``)
    - Duration elapses (motion complete — ``details["contact"]=False``)

    This is the general primitive for any guarded motion: food stabbing,
    surface contact detection, bite detection, insertion.

    Args:
        twist: 6D twist [vx, vy, vz, wx, wy, wz] in world frame.
        arm: Arm instance (for F/T, joint state).
        ctx: Execution context (for step_cartesian).
        ft_threshold: Force/torque threshold for contact detection.
        duration: Maximum motion duration (seconds). Motion completes
            successfully if this elapses without contact.
        timeout: Hard timeout (seconds). If None, uses ``duration * 1.5``.

    Returns:
        Outcome with success=True. Check ``details["contact"]`` to
        distinguish between contact-terminated and duration-elapsed.
        Returns SAFETY_ABORTED only for unexpected F/T spikes beyond
        the threshold (reserved for future graduated thresholds).
    """
    if timeout is None:
        timeout = duration * 1.5

    dt = ctx.control_dt
    arm_name = arm.config.name
    t0 = time.monotonic()
    elapsed = 0.0

    while elapsed < timeout:
        # F/T check
        exceeded, force_mag, torque_mag = _check_ft(arm, ft_threshold)
        if exceeded:
            return success(
                contact=True,
                force_n=force_mag,
                torque_nm=torque_mag,
                elapsed_s=elapsed,
            )

        # Duration check (motion complete without contact)
        if elapsed >= duration:
            return success(
                contact=False,
                elapsed_s=elapsed,
            )

        # Apply twist as joint velocities
        q_current = arm.get_joint_positions()
        J = arm.get_ee_jacobian()
        qd = np.linalg.lstsq(J, twist, rcond=None)[0]

        # Clamp to velocity limits
        limits = getattr(arm.config, "kinematic_limits", None)
        if limits is not None:
            scale = np.max(np.abs(qd) / limits.velocity)
            if scale > 1.0:
                qd /= scale

        q_target = q_current + qd * dt
        ctx.step_cartesian(arm_name, q_target, qd)

        elapsed = time.monotonic() - t0

    return failure(
        FailureKind.TIMEOUT,
        "ft_guarded_move:timeout",
        timeout_s=timeout,
        duration_s=duration,
    )
