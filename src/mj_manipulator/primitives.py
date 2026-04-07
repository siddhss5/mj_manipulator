# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Generic manipulation primitives: pickup, place, go_home.

These functions work with any robot that implements the ManipulationRobot
protocol and provides a GraspSource for TSR generation.

Usage::

    from mj_manipulator.primitives import pickup, place, go_home

    robot = MyRobot(objects={"mug": 1})
    with robot.sim() as ctx:
        pickup(robot, "mug_0")
        place(robot, "table_0")
        go_home(robot)
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import py_trees
from py_trees.common import Access, Status

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tick_tree(root: py_trees.behaviour.Behaviour, verbose: bool = False) -> bool:
    """Reset and tick a tree to completion. Returns True if SUCCESS."""
    for node in root.iterate():
        node.status = Status.INVALID
    tree = py_trees.trees.BehaviourTree(root=root)
    tree.tick()

    if verbose:
        print(py_trees.display.ascii_tree(root, show_status=True))

    return root.status == Status.SUCCESS


def _deactivate_teleop_for_arms(robot) -> None:
    """Deactivate teleop on all arms before a user-initiated primitive.

    Called at the entry of pickup/place/go_home — the user explicitly
    issued a command, so teleop should yield. Internal retries and
    recovery within the primitive do NOT call this.
    """
    ctx = getattr(robot, "_active_context", None)
    if ctx is None or not hasattr(ctx, "ownership") or ctx.ownership is None:
        return
    from mj_manipulator.ownership import OwnerKind

    for arm_name in ctx.ownership.arm_names:
        kind, _ = ctx.ownership.owner_of(arm_name)
        if kind == OwnerKind.TELEOP:
            ctx._deactivate_teleop_for(arm_name)


def _setup_blackboard(robot, ctx, arm_name: str, arm, ns: str) -> None:
    """Set up blackboard for a manipulation BT run."""
    bb = py_trees.blackboard.Client(name="primitives")
    bb.register_key(key="/context", access=Access.WRITE)
    bb.register_key(key="/abort_fn", access=Access.WRITE)
    bb.register_key(key=f"{ns}/arm", access=Access.WRITE)
    bb.register_key(key=f"{ns}/arm_name", access=Access.WRITE)
    bb.register_key(key=f"{ns}/timeout", access=Access.WRITE)
    bb.register_key(key=f"{ns}/goal_config", access=Access.WRITE)
    bb.register_key(key=f"{ns}/object_name", access=Access.WRITE)
    bb.register_key(key=f"{ns}/destination", access=Access.WRITE)
    bb.register_key(key=f"{ns}/grasp_source", access=Access.WRITE)
    bb.register_key(key=f"{ns}/hand_type", access=Access.WRITE)
    bb.register_key(key=f"{ns}/robot", access=Access.WRITE)

    bb.set("/context", ctx)
    bb.set("/abort_fn", robot.is_abort_requested)
    bb.set(f"{ns}/arm", arm)
    bb.set(f"{ns}/arm_name", arm_name)
    bb.set(f"{ns}/grasp_source", robot.grasp_source)
    bb.set(f"{ns}/robot", robot)

    # Hand type from gripper if available
    gripper = arm.gripper
    hand_type = getattr(gripper, "hand_type", "parallel_jaw") if gripper else "parallel_jaw"
    bb.set(f"{ns}/hand_type", hand_type)

    # Planning timeout
    timeout = getattr(getattr(robot, "config", None), "planning", None)
    if timeout is not None:
        timeout = getattr(timeout, "timeout", 5.0)
    else:
        timeout = 5.0
    bb.set(f"{ns}/timeout", timeout)

    # Ready pose for this arm (for recovery)
    ready = robot.named_poses.get("ready", {}).get(arm_name)
    if ready is not None:
        bb.set(f"{ns}/goal_config", np.array(ready))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pickup(
    robot,
    target: str | None = None,
    *,
    arm: str | None = None,
    verbose: bool = False,
) -> bool:
    """Pick up an object.

    Args:
        robot: ManipulationRobot instance.
        target: Object name/type to pick up, or None for nearest.
        arm: Arm name to use, or None to try all arms.
        verbose: Show BT tree status.

    Returns:
        True if pickup succeeded.
    """

    ctx = getattr(robot, "_active_context", None)
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    robot.clear_abort()
    _deactivate_teleop_for_arms(robot)
    try:
        return _pickup_inner(robot, ctx, target, arm=arm, verbose=verbose)
    except KeyboardInterrupt:
        robot.request_abort()
        logger.warning("Pickup interrupted by user")
        return False
    finally:
        robot.clear_abort()


def _pickup_inner(robot, ctx, target, *, arm, verbose) -> bool:
    from mj_manipulator.bt.subtrees import full_pickup

    # Determine which arms to try
    if arm is not None:
        sides = [arm]
    else:
        sides = list(robot.arms.keys())
        random.shuffle(sides)

    for side in sides:
        arm_obj = robot.arms[side]
        ns = f"/{side}"
        _setup_blackboard(robot, ctx, side, arm_obj, ns)

        # Set target
        bb = py_trees.blackboard.Client(name="pickup_target")
        bb.register_key(key=f"{ns}/object_name", access=Access.WRITE)
        bb.set(f"{ns}/object_name", target)

        tree = full_pickup(ns)
        if _tick_tree(tree, verbose=verbose):
            return True

    if target:
        logger.warning("Pickup failed for target '%s'", target)
    else:
        logger.warning("Pickup failed: no reachable objects")
    return False


def place(
    robot,
    destination: str | None = None,
    *,
    arm: str | None = None,
    verbose: bool = False,
) -> bool:
    """Place the held object at a destination.

    Args:
        robot: ManipulationRobot instance.
        destination: Where to place, or None for auto-detect.
        arm: Arm name holding the object, or None to auto-detect.
        verbose: Show BT tree status.

    Returns:
        True if placement succeeded.
    """

    ctx = getattr(robot, "_active_context", None)
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    # Auto-detect holding arm
    if arm is None:
        for name, arm_obj in robot.arms.items():
            if arm_obj.gripper and arm_obj.gripper.is_holding:
                arm = name
                break
        if arm is None:
            logger.warning("Place failed: no arm is holding an object")
            return False

    robot.clear_abort()
    _deactivate_teleop_for_arms(robot)
    try:
        return _place_inner(robot, ctx, destination, arm=arm, verbose=verbose)
    except KeyboardInterrupt:
        robot.request_abort()
        logger.warning("Place interrupted by user")
        return False
    finally:
        robot.clear_abort()


def _place_inner(robot, ctx, destination, *, arm, verbose) -> bool:
    from mj_manipulator.bt.subtrees import full_place

    arm_obj = robot.arms[arm]
    ns = f"/{arm}"
    _setup_blackboard(robot, ctx, arm, arm_obj, ns)

    bb = py_trees.blackboard.Client(name="place_target")
    bb.register_key(key=f"{ns}/destination", access=Access.WRITE)
    bb.register_key(key=f"{ns}/object_name", access=Access.WRITE)
    bb.set(f"{ns}/destination", destination)

    # Set held object name for placement TSR generation
    held_name = None
    if arm_obj.gripper:
        held_name = arm_obj.gripper.held_object
    bb.set(f"{ns}/object_name", held_name)

    tree = full_place(ns)
    if _tick_tree(tree, verbose=verbose):
        return True

    logger.warning("Place failed for destination '%s'", destination)
    return False


def go_home(
    robot,
    *,
    arm: str | None = None,
    verbose: bool = False,
) -> bool:
    """Return arm(s) to the ready configuration.

    Args:
        robot: ManipulationRobot instance.
        arm: Specific arm, or None for all arms.
        verbose: Show BT tree status.

    Returns:
        True if all specified arms returned to ready.
    """
    ctx = getattr(robot, "_active_context", None)
    if ctx is None:
        raise RuntimeError("No active execution context. Use 'with robot.sim() as ctx:'")

    robot.clear_abort()
    try:
        return _go_home_inner(robot, ctx, arm=arm, verbose=verbose)
    except KeyboardInterrupt:
        robot.request_abort()
        logger.warning("go_home interrupted by user")
        return False
    finally:
        robot.clear_abort()


def _go_home_inner(robot, ctx, *, arm, verbose) -> bool:
    ready_poses = robot.named_poses.get("ready", {})
    if not ready_poses:
        logger.warning("No 'ready' pose defined")
        return False

    if arm is not None:
        arms_to_home = [arm]
    else:
        arms_to_home = list(robot.arms.keys())

    all_ok = True
    for side in arms_to_home:
        if side not in ready_poses:
            continue
        arm_obj = robot.arms[side]
        goal = np.array(ready_poses[side])

        abort_fn = robot.is_abort_requested
        path = arm_obj.plan_to_configuration(goal, abort_fn=abort_fn)
        if path is None:
            logger.warning("go_home: planning failed for %s", side)
            all_ok = False
            continue

        traj = arm_obj.retime(path)
        if not ctx.execute(traj):
            logger.warning("go_home: execution failed for %s", side)
            all_ok = False

    ctx.sync()
    return all_ok
