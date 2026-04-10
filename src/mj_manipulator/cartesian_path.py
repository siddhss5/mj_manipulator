# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Plan a joint-space trajectory that follows an SE(3) waypoint sequence.

:func:`plan_cartesian_path` is the building block for **scripted** Cartesian
motion — motions where the caller knows the desired end-effector path ahead
of time and wants the arm to follow it as a retimed trajectory. Examples:

- Post-grasp lift: straight-line +Z waypoints from the grasp pose
- Pre-grasp approach: straight-line toward the grasp pose along the approach axis
- Post-place retreat: straight-line backout after release
- Tool-frame probes: small increments along tool Z until contact

The returned :class:`~mj_manipulator.trajectory.Trajectory` is intended to be
executed via :meth:`SimContext.execute`, which enforces joint velocity and
acceleration limits via TOPP-RA retiming. Because the execution path is
identical to the one used by the CBiRRT planner, scripted Cartesian motion
inherits the same smooth PD tracking that planned motion already has —
without the per-segment jerkiness that a reactive Cartesian controller can
introduce in physics mode.

This is the **scripted** Cartesian primitive. For operator-driven reactive
Cartesian control (teleop, force-guided motion), use
:class:`~mj_manipulator.cartesian.CartesianController` instead.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.trajectory import Trajectory

logger = logging.getLogger(__name__)


def plan_cartesian_path(
    arm: Arm,
    waypoints: list[np.ndarray],
    *,
    q_start: np.ndarray | None = None,
    control_dt: float = 0.008,
    max_branch_jump: float | None = None,
    partial_ok: bool = False,
) -> Trajectory:
    """Plan a joint-space trajectory that follows an SE(3) waypoint sequence.

    Solves analytical IK at each waypoint via the arm's ``ik_solver``,
    selects the solution closest in joint space to the previous
    configuration (greedy nearest branch), then retimes the resulting
    joint-space path with TOPP-RA using the arm's kinematic limits.

    The returned trajectory can be executed via :meth:`SimContext.execute`.
    TOPP-RA produces a smoothly varying position-velocity-acceleration
    profile, so the PD controller receives continuous targets (unlike a
    per-segment reactive loop, which introduces discrete jumps at segment
    boundaries).

    The IK selection is **local greedy**: at each waypoint we pick the
    valid solution nearest to the previous commanded configuration. This
    is robust for short monotonic paths (lift, approach, retreat) where
    the IK branch does not change along the path. For long paths that
    cross kinematic singularities or branch boundaries, this can fail
    without backtracking — use ``max_branch_jump`` to catch the failure
    early rather than silently producing a discontinuous path.

    Args:
        arm: Arm to plan for. Must have an IK solver attached.
        waypoints: List of 4x4 SE(3) pose matrices in world frame. The
            trajectory starts at the arm's current configuration (or
            ``q_start``) and visits each waypoint in order.
        q_start: Optional joint configuration to use as the starting
            state. Defaults to ``arm.get_joint_positions()``. Useful
            when chaining multiple paths without executing in between.
        control_dt: Control timestep for trajectory sampling (seconds).
            Must match the execution context's control rate.
        max_branch_jump: Optional per-step joint-space distance threshold
            (radians). If the nearest IK solution to the previous
            configuration is farther than this, raise ``ValueError``
            instead of silently jumping branches. ``None`` disables the
            check. A reasonable value for manipulation-scale waypoint
            spacing (5-10 mm) is ~0.5 rad (~30°).
        partial_ok: If ``True``, return a trajectory for the longest
            feasible prefix instead of raising when IK fails partway
            through. Useful for collision-aware primitives like
            :func:`safe_retract` that want "move as far as you can" semantics
            when the requested distance exits the reachable workspace.
            Still raises if the very first waypoint has no IK solution
            (no feasible prefix exists). ``max_branch_jump`` failures
            are treated the same as IK failures under ``partial_ok``.

    Returns:
        Time-parameterized :class:`~mj_manipulator.trajectory.Trajectory`
        ready for :meth:`SimContext.execute`. Under ``partial_ok``, the
        trajectory may cover fewer waypoints than were requested — check
        the number of waypoints or the final EE pose if needed.

    Raises:
        RuntimeError: If the arm has no IK solver attached.
        ValueError: If the waypoint list is empty, a waypoint has a bad
            shape, no IK solution exists for a waypoint (unless
            ``partial_ok=True``, in which case only the first waypoint
            being infeasible raises), or the nearest-branch selection
            exceeds ``max_branch_jump`` (same ``partial_ok`` semantics).
    """
    if arm.ik_solver is None:
        raise RuntimeError(
            f"plan_cartesian_path requires an arm with an IK solver; arm '{arm.config.name}' was created without one."
        )

    if not waypoints:
        raise ValueError("plan_cartesian_path: waypoints must be non-empty")

    q_current = arm.get_joint_positions().copy() if q_start is None else np.asarray(q_start, dtype=float).copy()

    joint_path: list[np.ndarray] = [q_current]

    for i, pose in enumerate(waypoints):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError(f"plan_cartesian_path: waypoint {i} has shape {pose.shape}, expected (4, 4)")

        solutions = arm.ik_solver.solve_valid(pose, q_init=q_current)
        if not solutions:
            if partial_ok and i > 0:
                logger.info(
                    "plan_cartesian_path: IK infeasible at waypoint %d/%d; "
                    "returning partial trajectory for %d feasible waypoints",
                    i,
                    len(waypoints),
                    i,
                )
                break
            raise ValueError(
                f"plan_cartesian_path: no valid IK solution at waypoint {i} (pose translation = {pose[:3, 3]})"
            )

        q_next = min(solutions, key=lambda q: float(np.linalg.norm(q - q_current)))

        if max_branch_jump is not None:
            jump = float(np.linalg.norm(q_next - q_current))
            if jump > max_branch_jump:
                if partial_ok and i > 0:
                    logger.info(
                        "plan_cartesian_path: IK branch jump at waypoint %d/%d "
                        "(%.3f rad > %.3f); returning partial trajectory for %d "
                        "feasible waypoints",
                        i,
                        len(waypoints),
                        jump,
                        max_branch_jump,
                        i,
                    )
                    break
                raise ValueError(
                    f"plan_cartesian_path: IK branch jump at waypoint {i} "
                    f"({jump:.3f} rad > max_branch_jump={max_branch_jump:.3f}). "
                    "The nearest IK solution is far from the previous "
                    "configuration — likely a singularity or branch boundary. "
                    "Consider denser waypoints, a different path, or increasing "
                    "max_branch_jump if the jump is intentional."
                )

        joint_path.append(q_next)
        q_current = q_next

    return arm.retime(joint_path, control_dt=control_dt)


def translational_waypoints(
    start_pose: np.ndarray,
    direction: np.ndarray,
    distance: float,
    *,
    segment_length: float = 0.005,
) -> list[np.ndarray]:
    """Generate SE(3) waypoints along a straight-line translation.

    Produces waypoints at ``segment_length`` increments from ``start_pose``
    in the given world-frame ``direction``, preserving orientation. The
    final waypoint is exactly at ``start + direction * distance`` (possibly
    closer than ``segment_length`` to the penultimate waypoint).

    Useful for :func:`plan_cartesian_path` callers who want a simple
    straight-line Cartesian motion (post-grasp lift, pre-grasp approach,
    post-place retreat).

    Args:
        start_pose: 4x4 SE(3) starting pose in world frame.
        direction: 3D direction vector. Will be normalized internally.
        distance: Total translation distance along ``direction`` (meters).
        segment_length: Spacing between consecutive waypoints (meters).
            Smaller = more IK solves but smoother path reconstruction.
            5 mm is a good default for manipulation-scale motions.

    Returns:
        List of 4x4 SE(3) waypoint poses.
    """
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return []
    unit = direction / norm

    start_rot = np.asarray(start_pose[:3, :3], dtype=float)
    start_trans = np.asarray(start_pose[:3, 3], dtype=float)

    n_segments = max(1, int(np.ceil(distance / segment_length)))
    waypoints: list[np.ndarray] = []
    for i in range(1, n_segments + 1):
        step = min(i * segment_length, distance)
        pose = np.eye(4)
        pose[:3, :3] = start_rot
        pose[:3, 3] = start_trans + unit * step
        waypoints.append(pose)
    return waypoints
