# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Force-aware Cartesian servo primitives.

General-purpose primitives for contact-rich manipulation:

- ``servo_to_pose``: Move to a target pose with distance-based
  deceleration and F/T monitoring.

- ``ft_guarded_move``: Move in a fixed direction until F/T exceeds
  a threshold or duration elapses.

Both are built on ``TeleopController`` — the same collision checking,
velocity clamping, and IK/Jacobian infrastructure that drives
interactive teleop. The servo primitives are teleop with a
programmatic target instead of a gizmo.
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rotation_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Compute rotation error as axis-angle vector.

    Returns a 3D vector whose direction is the rotation axis and
    whose magnitude is the rotation angle (radians).
    """
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
    """Check F/T against threshold.

    Returns (exceeded, force_mag, torque_mag). Returns (False, 0, 0)
    when no threshold is set, no sensor is available, or sensor data
    is invalid (kinematic mode).
    """
    if threshold is None:
        return False, 0.0, 0.0
    if not arm.has_ft_sensor or not arm.ft_valid:
        return False, 0.0, 0.0
    wrench = arm.get_ft_wrench()
    if np.any(np.isnan(wrench)):
        return False, 0.0, 0.0
    return threshold.check(wrench)


def _make_teleop(
    arm: Arm,
    ctx: ExecutionContext,
    safety_mode: SafetyMode = SafetyMode.REJECT,
) -> TeleopController:
    """Create a TeleopController configured for programmatic servo."""
    config = TeleopConfig(safety_mode=safety_mode)
    ctrl = TeleopController(arm, ctx, config=config)
    ctrl.activate()
    return ctrl


def _rodrigues_step(R: np.ndarray, angular_velocity: np.ndarray, dt: float) -> np.ndarray:
    """Integrate orientation by one step using Rodrigues' rotation formula.

    Args:
        R: 3x3 current rotation matrix.
        angular_velocity: 3D angular velocity in world frame (rad/s).
        dt: Time step (seconds).

    Returns:
        Updated 3x3 rotation matrix.
    """
    ang = angular_velocity * dt
    ang_norm = float(np.linalg.norm(ang))
    if ang_norm < 1e-8:
        return R
    axis = ang / ang_norm
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_step = np.eye(3) + np.sin(ang_norm) * K + (1 - np.cos(ang_norm)) * K @ K
    return R_step @ R


# ---------------------------------------------------------------------------
# Public primitives
# ---------------------------------------------------------------------------


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
    safety_mode: SafetyMode = SafetyMode.REJECT,
) -> Outcome:
    """Cartesian servo to a 6D target pose with deceleration and F/T monitoring.

    Servos to the full SE(3) pose — both position and orientation.
    Built on TeleopController — gets collision checking, velocity
    clamping, and IK/Jacobian infrastructure for free.

    The speed profile controls the Cartesian speed limit based on
    distance to target (decelerates near the target). F/T monitoring
    aborts on contact. Progress detection aborts if the arm stalls.

    Args:
        target: 4x4 target pose (world frame).
        arm: Arm instance.
        ctx: Execution context.
        speed_profile: Distance-based speed ramp. If None, constant
            0.05 m/s linear, 0.3 rad/s angular.
        ft_threshold: F/T abort threshold. If None, no F/T monitoring.
        position_tol: Convergence threshold for position (meters).
        rotation_tol: Convergence threshold for rotation (radians).
        timeout: Maximum duration (seconds).
        safety_mode: Collision handling — REJECT (abort on collision,
            default) or ALLOW (continue through collision).

    Returns:
        Outcome with:
        - success=True if target reached within tolerance
        - SAFETY_ABORTED if F/T exceeded
        - EXECUTION_FAILED if collision (REJECT mode) or no progress
        - TIMEOUT if not converged within timeout
    """
    if speed_profile is None:
        speed_profile = SpeedProfile.constant(linear=0.05, angular=0.3)

    ctrl = _make_teleop(arm, ctx, safety_mode)
    t0 = time.monotonic()

    # Progress tracking: measure EE movement over rolling 1-second windows.
    # Physics PD dynamics produce tiny per-step motion — per-tick checking
    # triggers false positives.
    progress_check_interval = 1.0
    last_progress_check = t0
    last_progress_pos = arm.get_ee_pose()[:3, 3].copy()
    min_progress_m = 0.0005  # 0.5mm per second minimum

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

            # Set Cartesian speed limit from the speed profile.
            # TeleopController computes the twist internally from the
            # full 6D pose error — both position and orientation are
            # corrected proportionally. max_cartesian_speed limits
            # how fast the EE moves.
            ctrl._config.max_cartesian_speed = speed_profile.linear_speed(pos_err_norm)

            ctrl.set_target_pose(target)
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
    safety_mode: SafetyMode = SafetyMode.REJECT,
) -> Outcome:
    """Move in a fixed direction until F/T exceeds threshold.

    Integrates a running target pose at the twist velocity and drives
    TeleopController to track it. Gets collision checking and velocity
    clamping for free.

    The running target advances independently of the arm's actual
    position — in physics mode it leads ahead of the PD controller,
    which is correct (same principle as trajectory feedforward).

    Args:
        twist: 6D twist [vx, vy, vz, wx, wy, wz] in world frame.
        arm: Arm instance.
        ctx: Execution context.
        ft_threshold: F/T threshold for contact detection.
        duration: Maximum motion duration (seconds).
        timeout: Hard timeout. If None, uses ``duration * 1.5``.
        safety_mode: Collision handling — REJECT (abort, default)
            or ALLOW (continue through collision, useful when contact
            is expected, e.g., food stabbing).

    Returns:
        Outcome with success=True. Check ``details["contact"]`` to
        distinguish contact-terminated (True) from duration-elapsed
        (False).
    """
    if timeout is None:
        timeout = duration * 1.5

    ctrl = _make_teleop(arm, ctx, safety_mode)
    t0 = time.monotonic()

    # Running target: starts at current pose, advances at twist velocity.
    running_target = arm.get_ee_pose().copy()
    lin_speed = float(np.linalg.norm(twist[:3]))
    dt_step = ctx.control_dt

    # Set Cartesian speed limit to the twist magnitude
    ctrl._config.max_cartesian_speed = lin_speed if lin_speed > 1e-6 else None

    # Progress tracking
    has_linear = lin_speed > 1e-4
    progress_check_interval = 0.5
    last_progress_check = t0
    last_progress_pos = arm.get_ee_pose()[:3, 3].copy()
    min_progress_m = 0.0003  # 0.3mm per 0.5s

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

            # Advance the running target
            running_target[:3, 3] += twist[:3] * dt_step
            running_target[:3, :3] = _rodrigues_step(running_target[:3, :3], twist[3:], dt_step)

            # Drive TeleopController with the running target
            ctrl.set_target_pose(running_target)
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
