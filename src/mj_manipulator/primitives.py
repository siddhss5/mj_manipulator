"""High-level manipulation primitives.

Simple reference implementations of pickup and place that compose
the lower-level planning, execution, and grasping APIs. These work
with any arm, gripper, and execution context (sim or hardware).

For more complex strategies (multi-arm fallback, base repositioning,
contact-based approach), build custom logic using the same APIs.

Usage::

    from tsr import load_package_template

    template = load_package_template("grasps", "mug_handle_grasp.yaml")
    grasp_tsrs = [template.instantiate(mug_pose)]

    success = pickup(ctx, arm, "mug", grasp_tsrs)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.protocols import ExecutionContext

logger = logging.getLogger(__name__)


def pickup(
    ctx: ExecutionContext,
    arm: Arm,
    object_name: str,
    grasp_tsrs: list,
    *,
    constraint_tsrs: list | None = None,
    lift_height: float = 0.05,
    timeout: float = 30.0,
) -> bool:
    """Pick up an object using TSR-defined grasps.

    Plans a path to the grasp region, executes the approach trajectory,
    closes the gripper, and lifts the object.

    Args:
        ctx: Execution context (sim or hardware).
        arm: Arm to use for grasping.
        object_name: Name of the object to grasp.
        grasp_tsrs: TSR goals defining valid grasp poses.
        constraint_tsrs: Optional path constraints (e.g., keep upright).
        lift_height: Height to lift after grasping (meters).
        timeout: Planning timeout in seconds.

    Returns:
        True if the object was successfully grasped.
    """
    # 1. Plan approach to grasp region
    path = arm.plan_to_tsrs(
        grasp_tsrs,
        constraint_tsrs=constraint_tsrs,
        timeout=timeout,
    )
    if path is None:
        logger.warning("pickup: planning failed for '%s'", object_name)
        return False

    # 2. Execute approach trajectory
    traj = arm.retime(path)
    if not ctx.execute(traj):
        logger.warning("pickup: execution failed for '%s'", object_name)
        return False

    # 3. Grasp
    arm_name = arm.config.name
    grasped = ctx.arm(arm_name).grasp(object_name)
    if not grasped:
        logger.warning("pickup: grasp failed for '%s'", object_name)
        return False

    # 4. Lift (best-effort — grasp already succeeded)
    if lift_height > 0:
        _lift(ctx, arm, lift_height)

    return True


def place(
    ctx: ExecutionContext,
    arm: Arm,
    place_tsrs: list,
    *,
    object_name: str | None = None,
    constraint_tsrs: list | None = None,
    retract_height: float = 0.05,
    timeout: float = 30.0,
) -> bool:
    """Place a held object at a TSR-defined destination.

    Plans a path to the placement region, executes the trajectory,
    releases the object, and retracts upward.

    Args:
        ctx: Execution context (sim or hardware).
        arm: Arm holding the object.
        place_tsrs: TSR goals defining valid placement poses.
        object_name: Object to release (None releases whatever is held).
        constraint_tsrs: Optional path constraints (e.g., keep upright).
        retract_height: Height to retract after releasing (meters).
        timeout: Planning timeout in seconds.

    Returns:
        True if placement succeeded.
    """
    # 1. Plan approach to placement region
    path = arm.plan_to_tsrs(
        place_tsrs,
        constraint_tsrs=constraint_tsrs,
        timeout=timeout,
    )
    if path is None:
        logger.warning("place: planning failed")
        return False

    # 2. Execute placement trajectory
    traj = arm.retime(path)
    if not ctx.execute(traj):
        logger.warning("place: execution failed")
        return False

    # 3. Release
    arm_name = arm.config.name
    ctx.arm(arm_name).release(object_name)

    # 4. Retract (best-effort — release already succeeded)
    if retract_height > 0:
        _lift(ctx, arm, retract_height)

    return True


def _lift(ctx: ExecutionContext, arm: Arm, height: float) -> bool:
    """Lift end-effector by *height* meters in world Z. Best-effort."""
    ee_pose = arm.get_ee_pose()
    lift_pose = ee_pose.copy()
    lift_pose[2, 3] += height

    lift_path = arm.plan_to_pose(lift_pose, timeout=5.0)
    if lift_path is None:
        logger.info("_lift: planning failed, skipping")
        return False

    return ctx.execute(arm.retime(lift_path))
