# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Force-aware Cartesian servo primitives.

General-purpose primitives for contact-rich manipulation:

- ``servo_to_pose``: Move to a target pose with distance-based
  deceleration and F/T monitoring.

- ``ft_guarded_move``: Move in a fixed direction until F/T exceeds
  a threshold or duration elapses.

Both are built on ``TeleopController`` — the same collision checking,
velocity clamping, and IK/Jacobian infrastructure that drives interactive
teleop. The servo primitives are teleop with a programmatic target
instead of a gizmo.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from mj_manipulator.force_control import ForceThresholds, SpeedProfile
from mj_manipulator.outcome import FailureKind, Outcome, failure, success
from mj_manipulator.teleop import SafetyMode, TeleopConfig, TeleopController, TeleopState

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


def _make_teleop(arm: Arm, ctx: ExecutionContext) -> TeleopController:
    """Create a TeleopController configured for programmatic servo."""
    config = TeleopConfig(safety_mode=SafetyMode.REJECT)
    ctrl = TeleopController(arm, ctx, config=config)
    ctrl.activate()
    return ctrl


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
    """Cartesian servo to a 6D target pose with deceleration and F/T monitoring.

    Servos to the full SE(3) pose — both position and orientation.
    Built on TeleopController — gets collision checking, velocity
    clamping, and IK/Jacobian infrastructure for free.

    The speed profile controls the approach speed based on distance
    to target (decelerates near the target). F/T monitoring aborts
    on contact. Progress detection aborts if the arm stalls.

    Args:
        target: 4x4 target pose (world frame). Both position and
            orientation are tracked.
        arm: Arm instance.
        ctx: Execution context.
        speed_profile: Distance-based speed ramp. If None, constant 0.05 m/s.
        ft_threshold: F/T abort threshold. If None, no F/T monitoring.
        position_tol: Stop when position error < this (meters).
        rotation_tol: Stop when rotation error < this (radians).
        timeout: Maximum duration (seconds).

    Returns:
        Outcome with success=True if target reached, or failure with
        SAFETY_ABORTED (F/T), EXECUTION_FAILED (collision/no_progress),
        or TIMEOUT.
    """
    if speed_profile is None:
        speed_profile = SpeedProfile.constant(linear=0.05, angular=0.3)

    ctrl = _make_teleop(arm, ctx)
    dt = ctx.control_dt
    t0 = time.monotonic()

    # Progress tracking: check distance over rolling windows
    progress_check_interval = 1.0
    last_progress_check = t0
    last_progress_pos = arm.get_ee_pose()[:3, 3].copy()
    min_progress_m = 0.0005

    try:
        while time.monotonic() - t0 < timeout:
            ee_pose = arm.get_ee_pose()
            pos_err = target[:3, 3] - ee_pose[:3, 3]
            pos_err_norm = float(np.linalg.norm(pos_err))
            rot_err = _rotation_error(target[:3, :3], ee_pose[:3, :3])
            rot_err_norm = float(np.linalg.norm(rot_err))

            # Convergence — both position and orientation
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
                    position_error_m=pos_err_norm,
                )

            # Progress check
            now = time.monotonic()
            if now - last_progress_check >= progress_check_interval:
                progress = float(np.linalg.norm(ee_pose[:3, 3] - last_progress_pos))
                if progress < min_progress_m and pos_err_norm > position_tol:
                    return failure(
                        FailureKind.EXECUTION_FAILED,
                        "servo_to_pose:no_progress",
                        position_error_m=pos_err_norm,
                        progress_m=progress,
                    )
                last_progress_pos = ee_pose[:3, 3].copy()
                last_progress_check = now

            # Compute intermediate target with speed profile.
            # Position: step toward target at profiled speed.
            # Orientation: use the final target orientation — TeleopController
            # handles the interpolation via velocity clamping.
            lin_speed = speed_profile.linear_speed(pos_err_norm)
            step_distance = lin_speed * dt
            if pos_err_norm > 1e-8:
                direction = pos_err / pos_err_norm
                step_pos = ee_pose[:3, 3] + direction * min(step_distance, pos_err_norm)
            else:
                step_pos = target[:3, 3]

            step_pose = target.copy()
            step_pose[:3, 3] = step_pos

            # Drive TeleopController — collision check + velocity clamp
            ctrl.set_target_pose(step_pose)
            state = ctrl.step()

            if state == TeleopState.UNREACHABLE:
                return failure(
                    FailureKind.EXECUTION_FAILED,
                    "servo_to_pose:collision",
                    position_error_m=pos_err_norm,
                )

        return failure(
            FailureKind.TIMEOUT,
            "servo_to_pose:timeout",
            timeout_s=timeout,
            position_error_m=float(np.linalg.norm(target[:3, 3] - arm.get_ee_pose()[:3, 3])),
        )
    finally:
        ctrl.deactivate()


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

    Built on TeleopController — gets collision checking and velocity
    clamping for free. Converts the constant twist to a sequence of
    target poses and drives TeleopController each cycle.

    Args:
        twist: 6D twist [vx, vy, vz, wx, wy, wz] in world frame.
        arm: Arm instance.
        ctx: Execution context.
        ft_threshold: F/T threshold for contact detection.
        duration: Maximum motion duration (seconds).
        timeout: Hard timeout. If None, uses duration * 1.5.

    Returns:
        Outcome with success=True. ``details["contact"]`` distinguishes
        contact-terminated from duration-elapsed.
    """
    if timeout is None:
        timeout = duration * 1.5

    ctrl = _make_teleop(arm, ctx)
    dt = ctx.control_dt
    t0 = time.monotonic()

    # Progress tracking
    has_linear = float(np.linalg.norm(twist[:3])) > 1e-4
    progress_check_interval = 0.5
    last_progress_check = t0
    last_progress_pos = arm.get_ee_pose()[:3, 3].copy()
    min_progress_m = 0.0003

    try:
        while True:
            elapsed = time.monotonic() - t0

            if elapsed >= timeout:
                return failure(
                    FailureKind.TIMEOUT,
                    "ft_guarded_move:timeout",
                    elapsed_s=elapsed,
                )

            # F/T check
            exceeded, force_mag, torque_mag = _check_ft(arm, ft_threshold)
            if exceeded:
                return success(
                    contact=True,
                    force_n=force_mag,
                    torque_nm=torque_mag,
                    elapsed_s=elapsed,
                )

            # Duration complete without contact
            if elapsed >= duration:
                return success(contact=False, elapsed_s=elapsed)

            # Compute next target pose by integrating twist
            ee_pose = arm.get_ee_pose()
            step_pose = ee_pose.copy()
            step_pose[:3, 3] += twist[:3] * dt
            # TODO: integrate angular twist for rotation

            # Drive TeleopController
            ctrl.set_target_pose(step_pose)
            state = ctrl.step()

            if state == TeleopState.UNREACHABLE:
                return failure(
                    FailureKind.EXECUTION_FAILED,
                    "ft_guarded_move:collision",
                    elapsed_s=elapsed,
                )

            # Progress check
            now = time.monotonic()
            if has_linear and now - last_progress_check >= progress_check_interval:
                ee_pos = arm.get_ee_pose()[:3, 3]
                progress = float(np.linalg.norm(ee_pos - last_progress_pos))
                if progress < min_progress_m:
                    return failure(
                        FailureKind.EXECUTION_FAILED,
                        "ft_guarded_move:no_progress",
                        elapsed_s=elapsed,
                        progress_m=progress,
                    )
                last_progress_pos = ee_pos.copy()
                last_progress_check = now
    finally:
        ctrl.deactivate()
