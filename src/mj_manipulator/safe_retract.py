# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Planned Cartesian retract along a straight-line direction.

``safe_retract()`` moves the arm's end-effector along a twist direction
up to a maximum distance. The trajectory is planned collision-free
(IK + collision check at every waypoint via ``plan_cartesian_path``).

Runtime collision detection is delegated to the hardware safety layer
(e.g. UR protective stop) rather than software contact monitoring.
The only software abort is via ``stop_condition`` (e-stop, ownership).

Use for:

- Post-grasp lift
- Recovery retract after failed grasp
- Any directional motion away from a surface
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from mj_manipulator.cartesian_path import plan_cartesian_path, translational_waypoints

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.protocols import ExecutionContext

logger = logging.getLogger(__name__)


def safe_retract(
    arm: Arm,
    ctx: ExecutionContext,
    twist: np.ndarray,
    max_distance: float,
    *,
    segment_length: float = 0.005,
    max_branch_jump: float | None = None,
    stop_condition: Callable[[], bool] | None = None,
) -> float:
    """Move the EE along ``twist`` up to ``max_distance``.

    Plans a collision-free Cartesian trajectory along the twist direction
    and executes it via ``ctx.execute``. The trajectory is planned with
    collision checking at every waypoint; runtime collision protection
    is handled by the hardware safety layer (e.g. UR protective stop).

    Currently handles translational twists only (angular components
    ``twist[3:]`` must be zero).

    Args:
        arm: Arm to move. Must have an IK solver attached.
        ctx: Execution context (SimContext or HardwareContext).
        twist: 6D twist [vx, vy, vz, wx, wy, wz]. Only the translational
            part is used; angular components must be zero.
        max_distance: Maximum distance to travel along the twist (meters).
        segment_length: Cartesian spacing between IK waypoints (meters).
        max_branch_jump: Maximum per-waypoint joint-space step (radians,
            vector norm across all joints). Caps the greedy-nearest IK
            branch selection so a small Cartesian step cannot produce a
            huge joint motion (lefty↔righty elbow flips). Under
            ``partial_ok``, an offending step truncates the retract
            instead of raising. ``None`` (default) disables the check.
            Reasonable opt-in values are 0.3-1.0 rad for typical 5-10 mm
            segment spacing; tune for the robot. NOTE: tight values can
            expose upstream IK/FK discrepancies (see the arm's IK solver
            — if IK at the current pose does not return the current
            config, the first waypoint will always look like a jump).
        stop_condition: Optional early-termination predicate (e-stop,
            ownership abort). Checked each control cycle.

    Returns:
        Signed projected distance traveled along the twist direction
        (meters). Equal to ``max_distance`` on clean completion, less
        on early stop or partial IK feasibility.
    """
    if not np.allclose(twist[3:], 0.0):
        raise NotImplementedError(
            f"safe_retract currently handles translational twists only; got angular components {twist[3:]}"
        )

    linear = np.asarray(twist[:3], dtype=float)
    linear_norm = float(np.linalg.norm(linear))
    if linear_norm < 1e-9:
        logger.warning("safe_retract: twist has zero magnitude; nothing to do")
        return 0.0

    direction = linear / linear_norm

    start_pose = arm.get_ee_pose()
    start_pos = start_pose[:3, 3].copy()

    waypoints = translational_waypoints(
        start_pose,
        direction,
        max_distance,
        segment_length=segment_length,
    )
    try:
        trajectory = plan_cartesian_path(
            arm,
            waypoints,
            partial_ok=True,
            max_branch_jump=max_branch_jump,
        )
    except ValueError as exc:
        logger.warning("safe_retract: no feasible prefix (%s); not moving", exc)
        return 0.0

    ctx.execute(trajectory, abort_fn=stop_condition)

    data = arm.env.data
    end_pos = data.site_xpos[arm.ee_site_id].copy()
    distance_traveled = float(np.dot(end_pos - start_pos, direction))
    logger.info("safe_retract: moved %.3fm along twist", distance_traveled)
    return distance_traveled
