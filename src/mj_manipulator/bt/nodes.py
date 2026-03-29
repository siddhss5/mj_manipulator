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

    def update(self) -> Status:
        arm = self.bb.get(self._key("arm"))
        tsrs = self.bb.get(self._key(self._tsrs_key))
        timeout = self.bb.get(self._key("timeout"))
        try:
            result = arm.plan_to_tsrs(tsrs, timeout=timeout, return_details=True)
        except Exception as e:
            self.feedback_message = str(e)
            return Status.FAILURE
        if result is None or not result.success:
            reason = getattr(result, "failure_reason", None) if result else None
            self.feedback_message = reason or "Planning failed"
            return Status.FAILURE
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

    def update(self) -> Status:
        arm = self.bb.get(self._key("arm"))
        goal = self.bb.get(self._key("goal_config"))
        timeout = self.bb.get(self._key("timeout"))
        try:
            path = arm.plan_to_configuration(goal, timeout=timeout)
        except Exception as e:
            self.feedback_message = str(e)
            return Status.FAILURE
        if path is None:
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

    def update(self) -> Status:
        from mj_manipulator.cartesian import CartesianController

        arm = self.bb.get(self._key("arm"))
        arm_name = self.bb.get(self._key("arm_name"))
        twist = self.bb.get(self._key("twist"))
        distance = self.bb.get(self._key("distance"))
        ctx = self.bb.get("/context")

        def step_fn(q, qd):
            ctx.step_cartesian(arm_name, q, qd)

        ctrl = CartesianController.from_arm(arm, step_fn=step_fn)
        ctrl.move(twist, dt=0.004, max_distance=distance)
        return Status.SUCCESS


class CheckNotNearConfig(_ManipulationNode):
    """Succeed if arm is NOT near goal_config (i.e. needs recovery motion).

    Returns FAILURE if arm is already near home — used as a guard to skip
    unnecessary retract moves during recovery.

    Reads: ``{ns}/arm``, ``{ns}/goal_config``
    """

    def __init__(self, ns: str = "", name: str = "CheckNotNearConfig",
                 tolerance: float = 0.1):
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
