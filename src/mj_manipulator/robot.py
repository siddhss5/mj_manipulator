# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Robot protocol and base class for generic manipulation.

ManipulationRobot defines the minimal interface. RobotBase provides
convenience methods (pickup, place, go_home, etc.) so concrete robots
only need to set up arms, grasp_source, and named_poses.

Usage::

    from mj_manipulator.robot import RobotBase

    class MyRobot(RobotBase):
        def __init__(self):
            env = Environment(model_path)
            arm = create_ur5e_arm(env, "ur5e", with_gripper=True)
            gm = GraspManager(env.model, env.data)
            super().__init__(
                model=env.model, data=env.data,
                arms={"ur5e": arm}, grasp_manager=gm,
                named_poses={"ready": {"ur5e": [...]}},
            )
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mj_manipulator.arm import Arm
    from mj_manipulator.grasp_manager import GraspManager
    from mj_manipulator.protocols import GraspSource


@runtime_checkable
class ManipulationRobot(Protocol):
    """Minimal interface for a robot that supports generic manipulation.

    The generic console, primitives, and teleop panels call only
    these methods and properties. Everything else is robot-specific.
    """

    @property
    def model(self) -> mujoco.MjModel: ...

    @property
    def data(self) -> mujoco.MjData: ...

    @property
    def arms(self) -> dict[str, Arm]:
        """All arms, keyed by name (e.g. {"left": arm, "right": arm})."""
        ...

    @property
    def grasp_source(self) -> GraspSource:
        """Provides grasp and placement TSRs for objects."""
        ...

    @property
    def grasp_manager(self) -> GraspManager:
        """Tracks grasped objects and kinematic attachments."""
        ...

    @property
    def named_poses(self) -> dict[str, dict[str, list[float]]]:
        """Named joint configurations."""
        ...

    def sim(
        self,
        *,
        physics: bool = True,
        headless: bool = False,
        viewer: object = None,
        event_loop: object = None,
    ) -> object:
        """Create simulation execution context (context manager)."""
        ...

    def request_abort(self) -> None: ...
    def clear_abort(self) -> None: ...
    def is_abort_requested(self) -> bool: ...


class RobotBase:
    """Base class providing manipulation convenience methods.

    Subclass and pass arms, grasp_manager, etc. to __init__. Gets
    pickup/place/go_home/find_objects/holding/get_object_pose for free.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        arms: dict[str, Arm],
        grasp_manager: GraspManager,
        named_poses: dict[str, dict[str, list[float]]] | None = None,
        grasp_source: GraspSource | None = None,
    ):
        self.model = model
        self.data = data
        self.arms = arms
        self.grasp_manager = grasp_manager
        self.named_poses = named_poses or {}
        self._grasp_source = grasp_source
        self._context = None
        self._abort_event = threading.Event()

    @property
    def grasp_source(self) -> GraspSource:
        if self._grasp_source is None:
            self._grasp_source = _NullGraspSource()
        return self._grasp_source

    @grasp_source.setter
    def grasp_source(self, value: GraspSource) -> None:
        self._grasp_source = value

    @property
    def _active_context(self):
        return self._context

    @_active_context.setter
    def _active_context(self, ctx):
        self._context = ctx

    # -- Execution context -----------------------------------------------------

    def sim(self, *, physics=True, headless=False, viewer=None, event_loop=None):
        """Create simulation execution context."""
        from mj_manipulator.sim_context import SimContext

        inner = SimContext(
            self.model,
            self.data,
            self.arms,
            physics=physics,
            headless=headless,
            viewer=viewer,
            event_loop=event_loop,
            abort_fn=self.is_abort_requested,
        )
        return _SimContextWrapper(inner, self)

    # -- Abort -----------------------------------------------------------------

    def request_abort(self):
        if self._context is not None and hasattr(self._context, "ownership") and self._context.ownership is not None:
            self._context.ownership.abort_all()
        self._abort_event.set()

    def clear_abort(self):
        if self._context is not None and hasattr(self._context, "ownership") and self._context.ownership is not None:
            self._context.ownership.clear_all()
        self._abort_event.clear()

    def is_abort_requested(self):
        return self._abort_event.is_set()

    # -- Manipulation primitives -----------------------------------------------

    def pickup(self, target=None, **kwargs):
        """Pick up an object."""
        from mj_manipulator.primitives import pickup

        return pickup(self, target, **kwargs)

    def place(self, destination=None, **kwargs):
        """Place the held object."""
        from mj_manipulator.primitives import place

        return place(self, destination, **kwargs)

    def go_home(self, **kwargs):
        """Return to ready configuration."""
        from mj_manipulator.primitives import go_home

        return go_home(self, **kwargs)

    # -- Scene queries ---------------------------------------------------------

    def find_objects(self, target=None):
        """List graspable objects in the scene."""
        objects = self.grasp_source.get_graspable_objects()
        if target:
            return [o for o in objects if target in o]
        return objects

    def holding(self):
        """Get (arm_name, object_name) if any arm is holding, else None."""
        for arm_name, arm in self.arms.items():
            if arm.gripper and arm.gripper.is_holding and arm.gripper.held_object:
                return (arm_name, arm.gripper.held_object)
        return None

    def get_object_pose(self, body_name):
        """Get 4x4 world-frame pose of a MuJoCo body."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body not found: {body_name}")
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[bid].reshape(3, 3)
        pose[:3, 3] = self.data.xpos[bid]
        return pose


class _SimContextWrapper:
    """Sets robot._active_context on enter/exit."""

    def __init__(self, inner, robot):
        self._inner = inner
        self._robot = robot

    def __enter__(self):
        ctx = self._inner.__enter__()
        self._robot._active_context = ctx
        return ctx

    def __exit__(self, *args):
        self._robot._active_context = None
        return self._inner.__exit__(*args)


class _NullGraspSource:
    """GraspSource that returns empty results."""

    def get_grasps(self, object_name, hand_type):
        return []

    def get_placements(self, destination, object_name):
        return []

    def get_graspable_objects(self):
        return []

    def get_place_destinations(self, object_name):
        return []
