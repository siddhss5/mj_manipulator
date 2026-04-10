# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Collision-aware directional motion via a planned Cartesian path.

``safe_retract()`` moves the arm's end-effector along a straight-line
direction (currently translation only) until new collisions appear or the
target distance is reached. It tracks the starting contact state as a
**baseline** — contacts present at the start are allowed to persist
(e.g. a held object touching its source surface), but new contacts cause
the motion to stop immediately.

Internally this builds a Cartesian SE(3) waypoint sequence via
:func:`~mj_manipulator.cartesian_path.translational_waypoints`, plans a
joint-space trajectory with
:func:`~mj_manipulator.cartesian_path.plan_cartesian_path` (analytical IK
per waypoint + TOPP-RA retiming respecting the arm's velocity and
acceleration limits), and executes it via :meth:`SimContext.execute` with
a baseline-contact abort predicate. Because TOPP-RA produces a smoothly
varying target profile and the execution path is the standard trajectory
runner, the gripper experiences no per-segment jerks and the held object
stays pinched throughout the lift.

Use for:

- Post-grasp lift (start has held object touching source surface)
- Recovery after failed grasp (start has arm touching bumped object)
- Any directional motion that might hit something

See also :func:`~mj_manipulator.cartesian_path.plan_cartesian_path` for
the general-purpose scripted-Cartesian primitive; this function is a
collision-aware wrapper around it for post-grasp motion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import mujoco
import numpy as np

from mj_manipulator.cartesian_path import plan_cartesian_path, translational_waypoints
from mj_manipulator.contacts import iter_contacts

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.protocols import ExecutionContext

logger = logging.getLogger(__name__)


def _get_contact_pairs(model: mujoco.MjModel, data: mujoco.MjData) -> set[tuple[int, int]]:
    """Get the set of unordered (body1, body2) pairs currently in contact."""
    pairs = set()
    for body1, body2, _ in iter_contacts(model, data):
        if body1 == body2:
            continue
        pair = (min(body1, body2), max(body1, body2))
        pairs.add(pair)
    return pairs


def safe_retract(
    arm: Arm,
    ctx: ExecutionContext,
    twist: np.ndarray,
    max_distance: float,
    *,
    segment_length: float = 0.005,
    stop_condition: Callable[[], bool] | None = None,
) -> float:
    """Move the EE along ``twist`` until new collisions appear.

    Plans a Cartesian trajectory along the twist direction and executes
    it via ``ctx.execute`` with a baseline-contact abort predicate. If a
    new contact pair appears during execution (one not present at the
    start pose), the trajectory runner halts at the next control cycle
    and the arm holds its current position.

    Currently handles translational twists only (angular components
    ``twist[3:]`` must be zero). Rotational lifts are unusual for
    post-grasp retraction and can be added when needed.

    Args:
        arm: Arm to move. Must have an IK solver attached.
        ctx: Execution context (e.g. ``SimContext`` or a hardware
            context implementing ``ExecutionContext``).
        twist: 6D twist [vx, vy, vz, wx, wy, wz]. Only the translational
            part is used; angular components must be zero.
        max_distance: Maximum distance to travel along the twist (meters).
        segment_length: Cartesian spacing between IK waypoints (meters).
            Smaller = more IK solves, denser waypoint reconstruction,
            slower planning. 5 mm is a good default for manipulation.
        stop_condition: Optional additional early-termination predicate.
            Checked alongside the baseline-contact check each control
            cycle.

    Returns:
        Signed projected distance traveled along the twist direction
        (meters). Equal to ``max_distance`` on clean completion, less
        on collision stop, possibly less on IK failure during planning.
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

    model = arm.env.model
    data = arm.env.data

    # Baseline contacts at the start pose — these are allowed to persist.
    # Must be captured AFTER mj_forward to reflect the current kinematic
    # state, not any stale contact data.
    mujoco.mj_forward(model, data)
    baseline = _get_contact_pairs(model, data)
    logger.info("safe_retract: baseline has %d contacts", len(baseline))

    start_pose = arm.get_ee_pose()
    start_pos = start_pose[:3, 3].copy()

    # Build the Cartesian path and plan a joint-space trajectory for the
    # longest feasible prefix. If the requested distance exits the
    # reachable workspace partway through, ``partial_ok=True`` gives us
    # back whatever prefix IS feasible rather than refusing to move.
    waypoints = translational_waypoints(
        start_pose,
        direction,
        max_distance,
        segment_length=segment_length,
    )
    try:
        trajectory = plan_cartesian_path(arm, waypoints, partial_ok=True)
    except ValueError as exc:
        # Even the first waypoint is infeasible — nothing reachable
        # from the current pose along the twist direction.
        logger.warning("safe_retract: no feasible prefix (%s); not moving", exc)
        return 0.0

    # Abort predicate: stop the trajectory the instant we see a contact
    # pair that was not present at the baseline. Runs once per control
    # cycle (every 8 ms at 125 Hz).
    stopped_due_to_contact = {"flag": False}

    def _abort() -> bool:
        if stop_condition is not None and stop_condition():
            return True
        current = _get_contact_pairs(model, data)
        new_contacts = current - baseline
        if new_contacts:
            if not stopped_due_to_contact["flag"]:
                stopped_due_to_contact["flag"] = True
                names = []
                for b1, b2 in new_contacts:
                    n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body_{b1}"
                    n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body_{b2}"
                    names.append(f"{n1}<->{n2}")
                logger.info(
                    "safe_retract: new contact detected: %s",
                    ", ".join(names),
                )
            return True
        return False

    ctx.execute(trajectory, abort_fn=_abort)

    end_pos = data.site_xpos[arm.ee_site_id].copy()
    distance_traveled = float(np.dot(end_pos - start_pos, direction))
    logger.info("safe_retract: moved %.3fm along twist", distance_traveled)
    return distance_traveled
