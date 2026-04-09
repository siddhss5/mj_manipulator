# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Collision-aware directional motion.

safe_retract() moves the arm along a twist until new collisions are
introduced. Tracks the starting contact state as a baseline — any
contacts present at the start are allowed to persist (e.g. held object
touching source surface), but new contacts cause the motion to stop.

Use for:
- Post-grasp lift (start has held object touching source surface)
- Recovery after failed grasp (start has arm touching bumped object)
- Any directional motion that might hit something
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import mujoco
import numpy as np

from mj_manipulator.cartesian import CartesianController
from mj_manipulator.contacts import iter_contacts

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm


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
    step_fn: Callable[[np.ndarray, np.ndarray], None],
    twist: np.ndarray,
    max_distance: float,
    *,
    dt: float = 0.008,
    stop_condition: Callable[[], bool] | None = None,
) -> float:
    """Move along twist until NEW collisions appear. Returns distance moved.

    Records the set of contacts at the start pose as a baseline. Moves
    incrementally. If a new contact pair appears (not in the baseline),
    stops immediately and holds position.

    This handles the common cases where the start pose is in collision:
    - Post-grasp lift: held object touching the source surface
    - Recovery after failed grasp: arm bumped into an object
    The baseline contacts are ignored; only new contacts stop the motion.

    Args:
        arm: Arm to move.
        step_fn: Callback to apply joint targets (e.g. ctx.step_cartesian).
        twist: 6D twist direction [vx, vy, vz, wx, wy, wz].
        max_distance: Maximum distance to travel (meters).
        dt: Control timestep for the cartesian controller.
        stop_condition: Optional early-termination check.

    Returns:
        Distance moved before stopping (meters).
    """
    model = arm.env.model
    data = arm.env.data

    # Record baseline contacts — these are allowed to persist
    mujoco.mj_forward(model, data)
    baseline = _get_contact_pairs(model, data)

    ctrl = CartesianController.from_arm(arm, step_fn=step_fn)
    ctrl.reset()

    total_distance = 0.0
    last_pos = data.site_xpos[arm.ee_site_id].copy()

    while total_distance < max_distance:
        if stop_condition is not None and stop_condition():
            break

        result = ctrl.step(twist, dt)

        current_pos = data.site_xpos[arm.ee_site_id].copy()
        total_distance += float(np.linalg.norm(current_pos - last_pos))
        last_pos = current_pos

        if result.achieved_fraction < 0.1:
            break

        # Check for new collisions (not in baseline)
        current_contacts = _get_contact_pairs(model, data)
        new_contacts = current_contacts - baseline
        if new_contacts:
            # New collision — stop here and hold position
            hold_q = arm.get_joint_positions()
            step_fn(hold_q, np.zeros_like(hold_q))
            break

    return total_distance
