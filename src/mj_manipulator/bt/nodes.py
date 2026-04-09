# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Behavior tree leaf nodes wrapping mj_manipulator operations.

Each node reads inputs from the blackboard, calls one mj_manipulator
operation, and writes results back. All nodes take a ``ns`` (namespace)
parameter so multiple arms can share a tree.

Blackboard layout::

    /context          — SimContext (shared)
    {ns}/arm          — Arm instance
    {ns}/arm_name     — arm name string (for ctx.arm(name))
    {ns}/tsrs         — list of TSR goals
    {ns}/goal_config  — numpy array goal configuration
    {ns}/timeout      — planning timeout (float, seconds)
    {ns}/path         — planned path (list of numpy arrays)
    {ns}/trajectory   — retimed Trajectory
    {ns}/object_name  — object to grasp/release
    {ns}/grasped      — name of grasped object (after Grasp)
    {ns}/twist        — 6D twist for CartesianMove
    {ns}/distance     — max distance for CartesianMove
"""

from __future__ import annotations

import numpy as np
import py_trees
from py_trees.common import Access, Status


class _ManipulationNode(py_trees.behaviour.Behaviour):
    """Base class for manipulation leaf nodes with namespace support."""

    def __init__(self, name: str, ns: str = ""):
        super().__init__(name)
        self.ns = ns
        self.bb = self.attach_blackboard_client(name=name)

    def _key(self, name: str) -> str:
        """Namespaced blackboard key."""
        return f"{self.ns}/{name}" if self.ns else name


class PlanToTSRs(_ManipulationNode):
    """Plan a collision-free path to a set of TSR goals.

    Reads: ``{ns}/arm``, ``{ns}/{tsrs_key}``, ``{ns}/timeout``
    Writes: ``{ns}/path``, ``{ns}/goal_tsr_index``
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

    Reads: ``{ns}/arm``, ``{ns}/goal_config``, ``{ns}/timeout``
    Writes: ``{ns}/path``
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


class Retime(_ManipulationNode):
    """Time-parameterize a path into a trajectory via TOPP-RA.

    Reads: ``{ns}/arm``, ``{ns}/path``
    Writes: ``{ns}/trajectory``
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


class Grasp(_ManipulationNode):
    """Close gripper and grasp an object.

    Reads: ``/context``, ``{ns}/arm_name``, ``{ns}/object_name``
    Writes: ``{ns}/grasped``

    If ``{ns}/tsr_to_object`` (list mapping TSR index → object name) and
    ``{ns}/goal_tsr_index`` are set, resolves the actual object name from
    the planner's goal index. This enables "pick any object" workflows.
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


class SafeRetract(_ManipulationNode):
    """Move along a twist until NEW collisions appear (post-grasp lift).

    Unlike CartesianMove, this tracks the baseline contact state and stops
    if new contacts appear. Start-state collisions (e.g. held object
    touching source surface) are tolerated.

    Reads: ``{ns}/arm``, ``{ns}/twist``, ``{ns}/distance``, ``/context``
    """

    def __init__(self, ns: str = "", name: str = "SafeRetract"):
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
        from mj_manipulator.safe_retract import safe_retract

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

        safe_retract(
            arm,
            step_fn,
            twist,
            max_distance=distance,
            dt=ctx.control_dt,
            stop_condition=abort_fn,
        )
        return Status.SUCCESS


class CartesianMove(_ManipulationNode):
    """Move end-effector along a twist using Cartesian velocity control.

    Reads: ``{ns}/arm``, ``{ns}/twist``, ``{ns}/distance``, ``{ns}/step_fn``
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


class CheckNotNearConfig(_ManipulationNode):
    """Succeed if arm is NOT near goal_config (i.e. needs recovery motion).

    Returns FAILURE if arm is already near home — used as a guard to skip
    unnecessary retract moves during recovery.

    Reads: ``{ns}/arm``, ``{ns}/goal_config``
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
            return Status.FAILURE  # already near home, skip retract
        return Status.SUCCESS


class Sync(_ManipulationNode):
    """Synchronize execution context (mj_forward + viewer sync).

    Reads: ``/context``
    """

    def __init__(self, ns: str = "", name: str = "Sync"):
        super().__init__(name, ns)
        self.bb.register_key(key="/context", access=Access.READ)

    def update(self) -> Status:
        ctx = self.bb.get("/context")
        ctx.sync()
        return Status.SUCCESS


class GenerateGrasps(_ManipulationNode):
    """Generate grasp TSRs using the robot's GraspSource.

    Queries ``grasp_source.get_grasps()`` for the target object(s) and
    combines all TSRs with a mapping so the Grasp node knows which
    object was reached.

    Reads: ``{ns}/object_name``, ``{ns}/grasp_source``, ``{ns}/hand_type``
    Writes: ``{ns}/grasp_tsrs``, ``{ns}/tsr_to_object``
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
    """Generate placement TSRs using the robot's GraspSource.

    Queries ``grasp_source.get_placements()`` for the target destination
    and the currently held object.

    Reads: ``{ns}/destination``, ``{ns}/grasp_source``, ``{ns}/object_name``
    Writes: ``{ns}/place_tsrs``, ``{ns}/tsr_to_destination``
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
