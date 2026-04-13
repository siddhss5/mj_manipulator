# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Behavior tree leaf nodes for manipulation.

Each node reads inputs from the py_trees blackboard, calls one
mj_manipulator operation, and writes results back. All nodes take a
``ns`` (namespace) parameter so multiple arms can share a tree.

These are **Layer 1 building blocks** — pure, composable, no opinions
about recovery or task sequencing. See ``docs/behavior-trees.md`` for
the composition guide and blackboard schema.

Blackboard keys (complete reference)
-------------------------------------

Shared (no namespace):

- ``/context`` — ExecutionContext (SimContext or HardwareContext)
- ``/abort_fn`` — optional abort predicate, checked by planners

Per-arm (namespaced with ``{ns}/``):

=====================  ==================  =================  =================
Key                    Type                Written by         Read by
=====================  ==================  =================  =================
arm                    Arm                 setup              most nodes
arm_name               str                 setup              Grasp, Release
timeout                float               setup              PlanTo* nodes
object_name            str | None          setup / Grasp      Grasp, Generate*
destination            str | None          setup              GeneratePlaceTSRs
grasp_source           GraspSource         setup              Generate* nodes
hand_type              str                 setup              GenerateGrasps
goal_config            ndarray             setup              PlanToConfig
grasp_tsrs             list[TSR]           GenerateGrasps     PlanToTSRs
place_tsrs             list[TSR]           GeneratePlaceTSRs  PlanToTSRs
tsr_to_object          list[str]           GenerateGrasps     Grasp
tsr_to_destination     list[str]           GeneratePlaceTSRs  (diagnostics)
goal_tsr_index         int                 PlanToTSRs         Grasp
plan_failure_reason    str | None          PlanToTSRs         (diagnostics)
path                   list[ndarray]       PlanTo* nodes      Retime
trajectory             Trajectory          Retime             Execute
grasped                str | None          Grasp              (diagnostics)
twist                  ndarray (6,)        setup / subtree    SafeRetract
distance               float               setup / subtree    SafeRetract
=====================  ==================  =================  =================
"""

from __future__ import annotations

import numpy as np
import py_trees
from py_trees.common import Access, Status


class _ManipulationNode(py_trees.behaviour.Behaviour):
    """Base class for manipulation leaf nodes.

    Provides namespace support so multiple arms can share a tree.
    Each node attaches a blackboard client scoped to its ``ns``
    prefix. Use :meth:`_key` to build namespaced keys.
    """

    def __init__(self, name: str, ns: str = ""):
        super().__init__(name)
        self.ns = ns
        self.bb = self.attach_blackboard_client(name=name)

    def _key(self, name: str) -> str:
        """Namespaced blackboard key."""
        return f"{self.ns}/{name}" if self.ns else name


# ---------------------------------------------------------------------------
# Planning nodes
# ---------------------------------------------------------------------------


class PlanToTSRs(_ManipulationNode):
    """Plan a collision-free path to a set of TSR goals.

    Reads: ``{ns}/arm``, ``{ns}/{tsrs_key}``, ``{ns}/timeout``, ``/abort_fn``
    Writes: ``{ns}/path``, ``{ns}/goal_tsr_index``, ``{ns}/plan_failure_reason``

    Returns SUCCESS when: a collision-free path is found to at least one TSR.
    Returns FAILURE when: no feasible path exists, or planning was aborted.
    """

    def __init__(self, ns: str = "", tsrs_key: str = "tsrs", name: str = "PlanToTSRs"):
        super().__init__(name, ns)
        self._tsrs_key = tsrs_key
        self.bb.register_key(key=self._key("arm"), access=Access.READ)
        self.bb.register_key(key=self._key(tsrs_key), access=Access.READ)
        self.bb.register_key(key=self._key("timeout"), access=Access.READ)
        self.bb.register_key(key=self._key("path"), access=Access.WRITE)
        self.bb.register_key(key=self._key("goal_tsr_index"), access=Access.WRITE)
        self.bb.register_key(key=self._key("plan_failure_reason"), access=Access.WRITE)
        try:
            self.bb.register_key(key="/abort_fn", access=Access.READ)
        except KeyError:
            pass

    def update(self) -> Status:
        arm = self.bb.get(self._key("arm"))
        tsrs = self.bb.get(self._key(self._tsrs_key))
        timeout = self.bb.get(self._key("timeout"))
        try:
            abort_fn = self.bb.get("/abort_fn")
        except KeyError:
            abort_fn = None
        try:
            result = arm.plan_to_tsrs(tsrs, timeout=timeout, return_details=True, abort_fn=abort_fn)
        except Exception as e:
            reason = str(e)
            self.feedback_message = reason
            self.bb.set(self._key("plan_failure_reason"), reason)
            return Status.FAILURE
        if result is None or not result.success:
            reason = getattr(result, "failure_reason", None) if result else None
            self.feedback_message = reason or "Planning failed"
            self.bb.set(self._key("plan_failure_reason"), reason or "Planning failed")
            return Status.FAILURE
        self.bb.set(self._key("plan_failure_reason"), None)
        self.bb.set(self._key("path"), result.path)
        self.bb.set(self._key("goal_tsr_index"), result.goal_index)
        return Status.SUCCESS


class PlanToConfig(_ManipulationNode):
    """Plan a collision-free path to a joint configuration.

    Not used by the default subtrees — available as a building block
    for custom recovery or go-home sequences.

    Reads: ``{ns}/arm``, ``{ns}/goal_config``, ``{ns}/timeout``, ``/abort_fn``
    Writes: ``{ns}/path``

    Returns SUCCESS when: a collision-free path is found.
    Returns FAILURE when: planning fails or is aborted.
    """

    def __init__(self, ns: str = "", name: str = "PlanToConfig"):
        super().__init__(name, ns)
        self.bb.register_key(key=self._key("arm"), access=Access.READ)
        self.bb.register_key(key=self._key("goal_config"), access=Access.READ)
        self.bb.register_key(key=self._key("timeout"), access=Access.READ)
        self.bb.register_key(key=self._key("path"), access=Access.WRITE)
        try:
            self.bb.register_key(key="/abort_fn", access=Access.READ)
        except KeyError:
            pass

    def update(self) -> Status:
        arm = self.bb.get(self._key("arm"))
        goal = self.bb.get(self._key("goal_config"))
        timeout = self.bb.get(self._key("timeout"))
        try:
            abort_fn = self.bb.get("/abort_fn")
        except KeyError:
            abort_fn = None
        try:
            path = arm.plan_to_configuration(goal, timeout=timeout, abort_fn=abort_fn)
        except Exception as e:
            self.feedback_message = str(e)
            return Status.FAILURE
        if path is None:
            if abort_fn is not None and abort_fn():
                self.feedback_message = "Aborted"
            else:
                self.feedback_message = "Planning to configuration failed"
            return Status.FAILURE
        self.bb.set(self._key("path"), path)
        return Status.SUCCESS


# ---------------------------------------------------------------------------
# Trajectory execution
# ---------------------------------------------------------------------------


class Retime(_ManipulationNode):
    """Time-parameterize a path into a trajectory via TOPP-RA.

    Reads: ``{ns}/arm``, ``{ns}/path``
    Writes: ``{ns}/trajectory``

    Returns SUCCESS always (TOPP-RA doesn't fail on valid paths).
    """

    def __init__(self, ns: str = "", name: str = "Retime"):
        super().__init__(name, ns)
        self.bb.register_key(key=self._key("arm"), access=Access.READ)
        self.bb.register_key(key=self._key("path"), access=Access.READ)
        self.bb.register_key(key=self._key("trajectory"), access=Access.WRITE)

    def update(self) -> Status:
        arm = self.bb.get(self._key("arm"))
        path = self.bb.get(self._key("path"))
        traj = arm.retime(path)
        self.bb.set(self._key("trajectory"), traj)
        return Status.SUCCESS


class Execute(_ManipulationNode):
    """Execute a trajectory via the execution context.

    Reads: ``/context``, ``{ns}/trajectory``

    Returns SUCCESS when: trajectory completes without abort.
    Returns FAILURE when: execution is aborted (by abort_fn, ownership
        preemption, or GraspVerifier drop detection).
    """

    def __init__(self, ns: str = "", name: str = "Execute"):
        super().__init__(name, ns)
        self.bb.register_key(key="/context", access=Access.READ)
        self.bb.register_key(key=self._key("trajectory"), access=Access.READ)

    def update(self) -> Status:
        ctx = self.bb.get("/context")
        traj = self.bb.get(self._key("trajectory"))
        ok = ctx.execute(traj)
        if not ok:
            self.feedback_message = "Trajectory execution failed"
        return Status.SUCCESS if ok else Status.FAILURE


# ---------------------------------------------------------------------------
# Grasp / Release
# ---------------------------------------------------------------------------


class Grasp(_ManipulationNode):
    """Close gripper and grasp an object.

    If ``{ns}/tsr_to_object`` and ``{ns}/goal_tsr_index`` are set
    (written by GenerateGrasps + PlanToTSRs), resolves the actual
    object name from the planner's goal index. This enables "pick
    any object" workflows where GenerateGrasps combines TSRs from
    multiple objects.

    Reads: ``/context``, ``{ns}/arm_name``, ``{ns}/object_name``,
        ``{ns}/tsr_to_object``, ``{ns}/goal_tsr_index``
    Writes: ``{ns}/object_name`` (resolved), ``{ns}/grasped``

    Returns SUCCESS when: ``ctx.arm(name).grasp(obj)`` returns a name.
    Returns FAILURE when: grasp returns None (no contact detected).
    """

    def __init__(self, ns: str = "", name: str = "Grasp"):
        super().__init__(name, ns)
        self.bb.register_key(key="/context", access=Access.READ)
        self.bb.register_key(key=self._key("arm_name"), access=Access.READ)
        self.bb.register_key(key=self._key("object_name"), access=Access.WRITE)
        self.bb.register_key(key=self._key("grasped"), access=Access.WRITE)
        self.bb.register_key(key=self._key("goal_tsr_index"), access=Access.READ)
        self.bb.register_key(key=self._key("tsr_to_object"), access=Access.READ)

    def update(self) -> Status:
        ctx = self.bb.get("/context")
        arm_name = self.bb.get(self._key("arm_name"))
        obj = self.bb.get(self._key("object_name"))

        # Resolve actual object from planner's goal index if available
        try:
            tsr_to_obj = self.bb.get(self._key("tsr_to_object"))
            goal_idx = self.bb.get(self._key("goal_tsr_index"))
            if tsr_to_obj is not None and goal_idx is not None and goal_idx < len(tsr_to_obj):
                obj = tsr_to_obj[goal_idx]
        except KeyError:
            pass

        # Always write the resolved object name so diagnostics know what was attempted
        self.bb.set(self._key("object_name"), obj)
        grasped = ctx.arm(arm_name).grasp(obj)
        self.bb.set(self._key("grasped"), grasped)
        if not grasped:
            self.feedback_message = f"Grasp failed: no contact with {obj or 'any object'}"
        return Status.SUCCESS if grasped else Status.FAILURE


class Release(_ManipulationNode):
    """Open gripper and release held object(s).

    Reads: ``/context``, ``{ns}/arm_name``

    Returns SUCCESS always.
    """

    def __init__(self, ns: str = "", name: str = "Release"):
        super().__init__(name, ns)
        self.bb.register_key(key="/context", access=Access.READ)
        self.bb.register_key(key=self._key("arm_name"), access=Access.READ)

    def update(self) -> Status:
        ctx = self.bb.get("/context")
        arm_name = self.bb.get(self._key("arm_name"))
        ctx.arm(arm_name).release()
        return Status.SUCCESS


# ---------------------------------------------------------------------------
# Cartesian motion
# ---------------------------------------------------------------------------


class SafeRetract(_ManipulationNode):
    """Retract along a twist, aborting on new collisions.

    Plans a Cartesian path along the twist direction and executes it
    with a baseline-contact abort predicate. Contacts present at the
    start (e.g. held object touching the table) are tolerated; only
    *new* contacts stop the motion. Used for post-grasp lifts on
    fixed-base arms (Franka).

    Reads: ``{ns}/arm``, ``{ns}/twist``, ``{ns}/distance``,
        ``/context``, ``/abort_fn``

    Returns SUCCESS always (motion runs to completion or aborts
        gracefully; the caller checks the arm's actual position to
        determine how far it moved).
    """

    def __init__(
        self,
        ns: str = "",
        name: str = "SafeRetract",
        *,
        twist: np.ndarray | None = None,
        distance: float | None = None,
    ):
        super().__init__(name, ns)
        self._twist = twist
        self._distance = distance
        self.bb.register_key(key=self._key("arm"), access=Access.READ)
        if twist is None:
            self.bb.register_key(key=self._key("twist"), access=Access.READ)
        if distance is None:
            self.bb.register_key(key=self._key("distance"), access=Access.READ)
        self.bb.register_key(key="/context", access=Access.READ)
        try:
            self.bb.register_key(key="/abort_fn", access=Access.READ)
        except KeyError:
            pass

    def update(self) -> Status:
        from mj_manipulator.safe_retract import safe_retract

        arm = self.bb.get(self._key("arm"))
        twist = self._twist if self._twist is not None else self.bb.get(self._key("twist"))
        distance = self._distance if self._distance is not None else self.bb.get(self._key("distance"))
        ctx = self.bb.get("/context")

        try:
            abort_fn = self.bb.get("/abort_fn")
        except KeyError:
            abort_fn = None

        safe_retract(
            arm,
            ctx,
            twist,
            max_distance=distance,
            stop_condition=abort_fn,
        )
        return Status.SUCCESS


class CartesianMove(_ManipulationNode):
    """Move end-effector along a twist using Cartesian velocity control.

    Not used by the default subtrees — available as a building block
    for custom motion primitives (e.g. guarded moves, compliant
    insertion). Differs from SafeRetract: no collision baseline,
    no path planning, pure reactive velocity control.

    Reads: ``{ns}/arm``, ``{ns}/arm_name``, ``{ns}/twist``,
        ``{ns}/distance``, ``/context``, ``/abort_fn``

    Returns SUCCESS always (motion runs to completion or abort).
    """

    def __init__(self, ns: str = "", name: str = "CartesianMove"):
        super().__init__(name, ns)
        self.bb.register_key(key=self._key("arm"), access=Access.READ)
        self.bb.register_key(key=self._key("arm_name"), access=Access.READ)
        self.bb.register_key(key=self._key("twist"), access=Access.READ)
        self.bb.register_key(key=self._key("distance"), access=Access.READ)
        self.bb.register_key(key="/context", access=Access.READ)
        try:
            self.bb.register_key(key="/abort_fn", access=Access.READ)
        except KeyError:
            pass

    def update(self) -> Status:
        from mj_manipulator.cartesian import CartesianController

        arm = self.bb.get(self._key("arm"))
        arm_name = self.bb.get(self._key("arm_name"))
        twist = self.bb.get(self._key("twist"))
        distance = self.bb.get(self._key("distance"))
        ctx = self.bb.get("/context")

        try:
            abort_fn = self.bb.get("/abort_fn")
        except KeyError:
            abort_fn = None

        def step_fn(q, qd):
            ctx.step_cartesian(arm_name, q, qd)

        ctrl = CartesianController.from_arm(arm, step_fn=step_fn)
        ctrl.move(twist, dt=ctx.control_dt, max_distance=distance, stop_condition=abort_fn)
        return Status.SUCCESS


# ---------------------------------------------------------------------------
# Guards and utilities
# ---------------------------------------------------------------------------


class CheckNotNearConfig(_ManipulationNode):
    """Succeed if arm is NOT near a goal configuration.

    Not used by the default subtrees — available as a building block
    for conditional execution in custom trees (e.g. "only retract if
    arm has moved away from home").

    Reads: ``{ns}/arm``, ``{ns}/goal_config``

    Returns SUCCESS when: max joint distance to goal > tolerance.
    Returns FAILURE when: arm is already near goal (within tolerance).
    """

    def __init__(self, ns: str = "", name: str = "CheckNotNearConfig", tolerance: float = 0.1):
        super().__init__(name, ns)
        self._tolerance = tolerance
        self.bb.register_key(key=self._key("arm"), access=Access.READ)
        self.bb.register_key(key=self._key("goal_config"), access=Access.READ)

    def update(self) -> Status:
        arm = self.bb.get(self._key("arm"))
        goal = self.bb.get(self._key("goal_config"))
        q = arm.get_joint_positions()
        if np.max(np.abs(q - goal)) < self._tolerance:
            return Status.FAILURE  # already near goal
        return Status.SUCCESS


class Sync(_ManipulationNode):
    """Synchronize execution context (mj_forward + viewer sync).

    Reads: ``/context``

    Returns SUCCESS always.
    """

    def __init__(self, ns: str = "", name: str = "Sync"):
        super().__init__(name, ns)
        self.bb.register_key(key="/context", access=Access.READ)

    def update(self) -> Status:
        ctx = self.bb.get("/context")
        ctx.sync()
        return Status.SUCCESS


# ---------------------------------------------------------------------------
# TSR generation
# ---------------------------------------------------------------------------


class GenerateGrasps(_ManipulationNode):
    """Generate grasp TSRs from the robot's GraspSource.

    Queries ``grasp_source.get_grasps()`` for each target object and
    combines all TSRs with a mapping (``tsr_to_object``) so the
    Grasp node can resolve which object the planner actually reached.

    When ``{ns}/object_name`` is None, generates TSRs for *all*
    graspable objects in the scene (the "pick anything" workflow).

    Reads: ``{ns}/object_name``, ``{ns}/grasp_source``, ``{ns}/hand_type``
    Writes: ``{ns}/grasp_tsrs``, ``{ns}/tsr_to_object``

    Returns SUCCESS when: at least one TSR was generated.
    Returns FAILURE when: no graspable objects found, or no TSRs
        generated for any object.
    """

    def __init__(self, ns: str = "", name: str = "GenerateGrasps"):
        super().__init__(name, ns)
        self.bb.register_key(key=self._key("object_name"), access=Access.READ)
        self.bb.register_key(key=self._key("grasp_source"), access=Access.READ)
        self.bb.register_key(key=self._key("hand_type"), access=Access.READ)
        self.bb.register_key(key=self._key("grasp_tsrs"), access=Access.WRITE)
        self.bb.register_key(key=self._key("tsr_to_object"), access=Access.WRITE)

    def update(self) -> Status:
        grasp_source = self.bb.get(self._key("grasp_source"))
        target = self.bb.get(self._key("object_name"))
        hand_type = self.bb.get(self._key("hand_type"))

        # Resolve objects to grasp
        if target is not None:
            objects = [target]
        else:
            objects = grasp_source.get_graspable_objects()

        if not objects:
            self.feedback_message = f"No graspable objects found for '{target}'"
            return Status.FAILURE

        all_tsrs = []
        tsr_to_object = []
        for obj_name in objects:
            tsrs = grasp_source.get_grasps(obj_name, hand_type)
            for _ in tsrs:
                tsr_to_object.append(obj_name)
            all_tsrs.extend(tsrs)

        if not all_tsrs:
            self.feedback_message = f"No grasp TSRs generated for {objects}"
            return Status.FAILURE

        self.bb.set(self._key("grasp_tsrs"), all_tsrs)
        self.bb.set(self._key("tsr_to_object"), tsr_to_object)
        return Status.SUCCESS


class GeneratePlaceTSRs(_ManipulationNode):
    """Generate placement TSRs from the robot's GraspSource.

    Queries ``grasp_source.get_placements()`` for each destination.
    Handles both instance names (``"recycle_bin_0"``) and type names
    (``"recycle_bin"`` → all instances of that type).

    Reads: ``{ns}/destination``, ``{ns}/grasp_source``, ``{ns}/object_name``
    Writes: ``{ns}/place_tsrs``, ``{ns}/tsr_to_destination``

    Returns SUCCESS when: at least one placement TSR was generated.
    Returns FAILURE when: no valid destinations found, or no TSRs
        generated for any destination.
    """

    def __init__(self, ns: str = "", name: str = "GeneratePlaceTSRs"):
        super().__init__(name, ns)
        self.bb.register_key(key=self._key("destination"), access=Access.READ)
        self.bb.register_key(key=self._key("grasp_source"), access=Access.READ)
        self.bb.register_key(key=self._key("object_name"), access=Access.READ)
        self.bb.register_key(key=self._key("place_tsrs"), access=Access.WRITE)
        self.bb.register_key(key=self._key("tsr_to_destination"), access=Access.WRITE)

    def update(self) -> Status:
        grasp_source = self.bb.get(self._key("grasp_source"))
        destination = self.bb.get(self._key("destination"))
        object_name = self.bb.get(self._key("object_name"))

        # Resolve destinations — handle type names (e.g. "recycle_bin")
        # by matching against all available destinations
        if destination is not None:
            all_dests = grasp_source.get_place_destinations(object_name)
            # Check if it's an exact match (instance name like "recycle_bin_0")
            if destination in all_dests:
                destinations = [destination]
            else:
                # Type name — find all instances matching this prefix
                destinations = [d for d in all_dests if d.startswith(destination)]
                if not destinations:
                    destinations = [destination]  # pass through, let get_placements handle it
        else:
            destinations = grasp_source.get_place_destinations(object_name)

        if not destinations:
            self.feedback_message = f"No placement destinations for '{destination}'"
            return Status.FAILURE

        all_tsrs = []
        tsr_to_dest = []
        for dest in destinations:
            tsrs = grasp_source.get_placements(dest, object_name)
            for _ in tsrs:
                tsr_to_dest.append(dest)
            all_tsrs.extend(tsrs)

        if not all_tsrs:
            self.feedback_message = f"No placement TSRs for destinations {destinations}"
            return Status.FAILURE

        self.bb.set(self._key("place_tsrs"), all_tsrs)
        self.bb.set(self._key("tsr_to_destination"), tsr_to_dest)
        return Status.SUCCESS
