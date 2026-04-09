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

    if root.status != Status.SUCCESS:
        tip = root.tip()
        if tip is not None and tip.feedback_message and tip.feedback_message != "Aborted":
            logger.warning("%s: %s", tip.name, tip.feedback_message)

    return root.status == Status.SUCCESS


def _maybe_hide_in_container(robot, ns: str, destination: str | None, held_name: str) -> None:
    """Hide object if it was placed into a container (bin, tote)."""
    import re

    resolved = destination
    if resolved is None:
        try:
            _bb = py_trees.blackboard.Client(name=f"place_resolve{ns}")
            _bb.register_key(key=f"{ns}/tsr_to_destination", access=Access.READ)
            _bb.register_key(key=f"{ns}/goal_tsr_index", access=Access.READ)
            mapping = _bb.get(f"{ns}/tsr_to_destination")
            idx = _bb.get(f"{ns}/goal_tsr_index")
            if mapping and idx is not None and idx < len(mapping):
                resolved = mapping[idx]
        except (KeyError, RuntimeError):
            pass

    if resolved is None or resolved == "worktop":
        return

    try:
        from asset_manager import AssetManager
        from prl_assets import OBJECTS_DIR

        m = re.match(r"^(.+?)_(\d+)$", resolved)
        dest_type = m.group(1) if m else resolved
        assets = AssetManager(str(OBJECTS_DIR))
        gp = assets.get(dest_type)["geometric_properties"]
        if gp.get("type") not in ("open_box", "tote"):
            return
    except (KeyError, TypeError, ImportError):
        return

    env = getattr(robot, "_env", None)
    if env is not None and hasattr(env, "registry") and env.registry is not None:
        if env.registry.is_active(held_name):
            env.registry.hide(held_name)
            import mujoco

            mujoco.mj_forward(robot.model, robot.data)


def _set_hud_action(robot, arm_name: str, text: str) -> None:
    """Update HUD status for an arm (no-op if no HUD)."""
    hud = getattr(robot, "_status_hud", None)
    if hud is not None:
        hud.set_action(arm_name, text)


def _sync_viewer(robot) -> None:
    """Force a viewer sync so HUD updates are visible immediately."""
    ctx = getattr(robot, "_active_context", None)
    if ctx is not None and hasattr(ctx, "sync"):
        try:
            ctx.sync()
        except Exception:
            pass


def _arm_preempted(robot, arm_name: str) -> bool:
    """Check if an arm was taken by another controller (e.g. teleop)."""
    ctx = getattr(robot, "_active_context", None)
    if ctx is None or not hasattr(ctx, "ownership") or ctx.ownership is None:
        return False
    from mj_manipulator.ownership import OwnerKind

    kind, _ = ctx.ownership.owner_of(arm_name)
    return kind not in (OwnerKind.IDLE, OwnerKind.TRAJECTORY)


def _deactivate_teleop_for_arms(robot, arms: list[str] | None = None) -> None:
    """Deactivate teleop on specified arms (or all) before a primitive."""
    ctx = getattr(robot, "_active_context", None)
    if ctx is None or not hasattr(ctx, "ownership") or ctx.ownership is None:
        return
    from mj_manipulator.ownership import OwnerKind

    arm_names = arms if arms is not None else ctx.ownership.arm_names
    for arm_name in arm_names:
        kind, _ = ctx.ownership.owner_of(arm_name)
        if kind == OwnerKind.TELEOP:
            ctx._deactivate_teleop_for(arm_name)


def _pickup_details(ns: str) -> tuple[list[str], str | None, bool, bool, str | None]:
    """Read pickup results from blackboard after a failed attempt.

    Returns:
        (attempted_objects, reached_object, plan_succeeded, grasp_succeeded,
         plan_failure_reason)
    """
    attempted: list[str] = []
    reached: str | None = None
    plan_failed = False
    grasped = False
    plan_reason: str | None = None
    try:
        bb = py_trees.blackboard.Client(name=f"pickup_report{ns}")
        bb.register_key(key=f"{ns}/tsr_to_object", access=Access.READ)
        bb.register_key(key=f"{ns}/object_name", access=Access.READ)
        bb.register_key(key=f"{ns}/grasped", access=Access.READ)
        bb.register_key(key=f"{ns}/plan_failure_reason", access=Access.READ)
        mapping = bb.get(f"{ns}/tsr_to_object")
        if mapping:
            attempted = sorted(set(mapping))
        obj = bb.get(f"{ns}/object_name")
        if obj:
            reached = obj
        plan_reason = bb.get(f"{ns}/plan_failure_reason")
        plan_failed = plan_reason is not None
        grasped = bool(bb.get(f"{ns}/grasped"))
    except (KeyError, RuntimeError):
        pass
    return attempted, reached, not plan_failed, grasped, plan_reason


def _report_pickup_failure(robot, sides_tried: list[str], target: str | None) -> None:
    """Log detailed failure information after all arms have been tried."""
    all_attempted: set[str] = set()
    plan_failures: list[str] = []
    grasp_failures: list[str] = []

    for side in sides_tried:
        ns = f"/{side}"
        attempted, reached, planned, grasped, plan_reason = _pickup_details(ns)
        all_attempted.update(attempted)
        if reached and planned and not grasped:
            grasp_failures.append(f"{reached} ({side} arm)")
            _set_hud_action(robot, side, "✗ pickup: grasp failed")
        elif reached and not planned:
            detail = f"{reached} ({side} arm)"
            short = plan_reason.split(":")[0] if plan_reason else "plan failed"
            if plan_reason:
                detail += f": {plan_reason}"
            plan_failures.append(detail)
            _set_hud_action(robot, side, f"✗ pickup: {short}")

    if grasp_failures:
        logger.warning("Pickup failed: reached %s but grasp failed", ", ".join(grasp_failures))
    elif plan_failures:
        logger.warning("Pickup failed: could not plan to %s", "; ".join(plan_failures))
    elif all_attempted:
        logger.warning("Pickup failed: could not plan to %s", ", ".join(sorted(all_attempted)))
    else:
        desc = f"'{target}'" if target else "any object"
        logger.warning("Pickup failed: no graspable %s found", desc)


def _setup_blackboard(robot, ctx, arm_name: str, arm, ns: str) -> None:
    """Set up blackboard for a manipulation BT run."""
    bb = py_trees.blackboard.Client(name="primitives")
    keys = [
        "/context",
        "/abort_fn",
        f"{ns}/arm",
        f"{ns}/arm_name",
        f"{ns}/timeout",
        f"{ns}/goal_config",
        f"{ns}/object_name",
        f"{ns}/destination",
        f"{ns}/grasp_source",
        f"{ns}/hand_type",
        f"{ns}/robot",
        f"{ns}/grasp_tsrs",
        f"{ns}/place_tsrs",
        f"{ns}/tsr_to_object",
        f"{ns}/tsr_to_destination",
        f"{ns}/goal_tsr_index",
        f"{ns}/plan_failure_reason",
        f"{ns}/grasped",
        f"{ns}/path",
        f"{ns}/trajectory",
        f"{ns}/twist",
        f"{ns}/distance",
    ]
    for key in keys:
        try:
            bb.register_key(key=key, access=Access.WRITE)
        except KeyError:
            pass

    bb.set("/context", ctx)
    bb.set("/abort_fn", lambda: robot.is_abort_requested() or _arm_preempted(robot, arm_name))
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

    # Clear stale results from previous runs
    for stale_key in [
        "path",
        "trajectory",
        "grasped",
        "grasp_tsrs",
        "place_tsrs",
        "tsr_to_object",
        "tsr_to_destination",
        "goal_tsr_index",
        "plan_failure_reason",
    ]:
        bb.set(f"{ns}/{stale_key}", None)


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
        for side in robot.arms:
            _set_hud_action(robot, side, "⊘ interrupted")
        _sync_viewer(robot)
        return False
    finally:
        robot.clear_abort()


def _pickup_inner(robot, ctx, target, *, arm, verbose) -> bool:
    from mj_manipulator.bt.subtrees import full_pickup

    # Quick check: are there any matching objects?
    if not robot.find_objects(target):
        desc = f"'{target}'" if target else "any object"
        logger.warning("Pickup failed: no graspable objects found for %s", desc)
        return False

    # Determine which arms to try
    if arm is not None:
        sides = [arm]
    else:
        sides = list(robot.arms.keys())
        random.shuffle(sides)

    sides_tried = []
    for i, side in enumerate(sides):
        arm_obj = robot.arms[side]
        ns = f"/{side}"
        _setup_blackboard(robot, ctx, side, arm_obj, ns)

        bb = py_trees.blackboard.Client(name="pickup_target")
        bb.register_key(key=f"{ns}/object_name", access=Access.WRITE)
        bb.set(f"{ns}/object_name", target)

        desc = target or "any"
        _set_hud_action(robot, side, f"⟳ pickup({desc})")
        tree = full_pickup(ns)
        if _tick_tree(tree, verbose=verbose):
            _set_hud_action(robot, side, f"✓ pickup({desc})")
            _sync_viewer(robot)
            return True

        _set_hud_action(robot, side, f"✗ pickup({desc})")
        sides_tried.append(side)

        # Stop if abort or this arm was preempted (e.g. teleop)
        if robot.is_abort_requested() or _arm_preempted(robot, side):
            _sync_viewer(robot)
            return False

        # Before trying the next arm, send this arm home
        # so it doesn't block the workspace (skip if preempted)
        if i < len(sides) - 1 and not _arm_preempted(robot, side):
            go_home(robot, arm=side)

    _report_pickup_failure(robot, sides_tried, target)
    _sync_viewer(robot)
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
        _set_hud_action(robot, arm, "⊘ interrupted")
        _sync_viewer(robot)
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

    held_name = None
    if arm_obj.gripper:
        held_name = arm_obj.gripper.held_object
    bb.set(f"{ns}/object_name", held_name)

    desc = destination or "auto"
    _set_hud_action(robot, arm, f"⟳ place({desc})")
    tree = full_place(ns)
    ok = _tick_tree(tree, verbose=verbose)

    if ok:
        _set_hud_action(robot, arm, f"✓ place({desc})")
        if held_name:
            _maybe_hide_in_container(robot, ns, destination, held_name)
    else:
        _set_hud_action(robot, arm, f"✗ place({desc})")
        logger.warning("Place failed for destination '%s'", destination)

    _sync_viewer(robot)
    return ok


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
    arms_to_home = [arm] if arm is not None else list(robot.arms.keys())
    _deactivate_teleop_for_arms(robot, arms_to_home)
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

        def abort_fn(s=side):
            return robot.is_abort_requested() or _arm_preempted(robot, s)

        goal = np.array(ready_poses[side])

        try:
            path = arm_obj.plan_to_configuration(goal, abort_fn=abort_fn)
        except Exception as e:
            logger.warning("go_home %s: plan failed: %s", side, e)
            path = None

        if path is None:
            # Retract up first, then retry
            logger.warning("go_home %s: retract up and retry", side)
            from mj_manipulator.cartesian import CartesianController

            arm_name = arm_obj.config.name

            def _step_fn(q, qd):
                ctx.step_cartesian(arm_name, q, qd)

            ctrl = CartesianController.from_arm(arm_obj, step_fn=_step_fn)
            ctrl.move(
                np.array([0.0, 0.0, 0.10, 0.0, 0.0, 0.0]),
                dt=ctx.control_dt,
                max_distance=0.10,
                stop_condition=abort_fn,
            )
            try:
                path = arm_obj.plan_to_configuration(goal, abort_fn=abort_fn)
            except Exception as e:
                logger.warning("go_home %s: retry failed: %s", side, e)
                path = None

        if path is not None:
            traj = arm_obj.retime(path)
            if not ctx.execute(traj):
                logger.warning("go_home: execution failed for %s", side)
                all_ok = False
        else:
            logger.warning("go_home: could not plan %s to ready", side)
            all_ok = False

    ctx.sync()
    return all_ok
